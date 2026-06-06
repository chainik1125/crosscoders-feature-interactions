from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import yaml
import os
import sys
import argparse
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions')
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.analysis.analysis_utils import feature_interactions_mlp, get_activations

torch.set_grad_enabled(False)



from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from einops import rearrange

# ---- shapiq imports (STII permutation sampler + InteractionValues helper) ----
from shapiq.approximator.permutation.stii import PermutationSamplingSTII  # STII sampler (perm)  # noqa: E402
from shapiq.interaction_values import InteractionValues   


# st2_sae_pipeline.py
# Minimal, clean scaffold for Shapley–Taylor (order≤2) on SAE latents inside TransformerLens.
# Assumes: PyTorch, einops, shapiq, TransformerLens, your SAE (crosscoder).


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup device based on config."""
    device_setting = config.get('performance', {}).get('device', 'auto')
    if device_setting == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_setting == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device




# --------------------------------------------------------------------------------
# 0) Utility: safe device / dtype
# --------------------------------------------------------------------------------
def _torch_device_of(model) -> torch.device:
    try:
        return model.cfg.device  # TransformerLens
    except Exception:
        return next(model.parameters()).device

# --------------------------------------------------------------------------------
# 1) You already have get_activations(); we add a tiny helper to also return tokens
# --------------------------------------------------------------------------------
def tokenize_to_device(input_text: str, model, max_len: int = 128) -> torch.Tensor:
    tokens = torch.tensor(model.tokenizer.encode(input_text)[:max_len], device=_torch_device_of(model))
    return tokens.unsqueeze(0)  # [1, S]

# --------------------------------------------------------------------------------
# 2) Hooked forward: inject at hook_resid_pre[L] and capture hook_resid_post[L]
#    Vectorized over batch of coalition reconstructions.
# --------------------------------------------------------------------------------
def validate_hook_points(start_hook: str, end_hook: str, available_hooks: List[str]) -> None:
    """Validate that the specified hook points are in the SAE training set."""
    if start_hook not in available_hooks:
        raise ValueError(f"Start hook '{start_hook}' not in available hooks: {available_hooks}")
    if end_hook not in available_hooks:
        raise ValueError(f"End hook '{end_hook}' not in available hooks: {available_hooks}")

    # Check that end hook comes after start hook
    start_idx = available_hooks.index(start_hook)
    end_idx = available_hooks.index(end_hook)
    if end_idx < start_idx:
        raise ValueError(f"End hook '{end_hook}' must come after start hook '{start_hook}'")


def build_forward_from_hook_to_hook(
    model,
    tokens_1S: torch.Tensor,
    start_hook: str,
    end_hook: str,
    pos_idx: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a function f: z_BD -> y_BD that:
      - clones the input tokens across batch B,
      - sets start_hook[:, pos_idx, :] = z_BD[i]
      - runs the model one full forward,
      - returns end_hook at pos_idx (shape [B, d_model]).
    """
    device = _torch_device_of(model)
    tokens_1S = tokens_1S.to(device)

    def run(z_BD: torch.Tensor) -> torch.Tensor:
        B, D = z_BD.shape
        toks = tokens_1S.expand(B, -1)  # duplicate the same sequence for each coalition
        out_holder: Dict[str, torch.Tensor] = {}

        def edit_hook(value, hook):
            # value: [B, S, D]
            value = value.clone()
            value[:, pos_idx, :] = z_BD.to(value.device)
            return value

        def capture_hook(value, hook):
            out_holder["value"] = value.detach()
            return value

        hook_in = (start_hook, edit_hook)
        hook_out = (end_hook, capture_hook)

        # We don't need logits; we read the output at end_hook.
        model.run_with_hooks(toks, return_type=None, fwd_hooks=[hook_in, hook_out])  # cleans up hooks automatically
        return out_holder["value"][:, pos_idx, :]  # [B, D]

    return run

# --------------------------------------------------------------------------------
# 3) Baselines for missing SAE latents
# --------------------------------------------------------------------------------
@dataclass
class BaselinePool:
    """
    Baselines are SAE latent vectors on the *active* coordinates (n players).
    We'll average v(S) over this pool (observational semantics).
    """
    baselines_Bn: torch.Tensor  # [B, n] on the same device as model/SAE

    @staticmethod
    def zeros(n: int, device: torch.device, B: int = 1, dtype: torch.dtype = torch.float32) -> "BaselinePool":
        return BaselinePool(torch.zeros(B, n, device=device, dtype=dtype))

# --------------------------------------------------------------------------------
# 4) SAE decode helper (adapt to your crosscoder)
# --------------------------------------------------------------------------------
def decode_latents_to_resid_pre(
    crosscoder,
    latents_BH: torch.Tensor,  # full H-dim vectors
) -> torch.Tensor:
    """
    Returns z_BD at the 'resid_pre' space (i.e., same shape as hook_resid_pre).
    Adjust this to your SAE (bias terms etc). Below are common conventions.
    """
    # The crosscoder has W_dec_HXD weights that decode from hidden_dim to output
    # W_dec shape is [H, X1, X2, D] where X1, X2 are the crosscoding dimensions
    # We need to decode the latents back to the residual stream

    # W_dec has shape [H, 1, 17, D], so we use the full einsum notation
    output_BX1X2D = torch.einsum("hijD,Bh->BijD", crosscoder.W_dec_HXD, latents_BH)

    if hasattr(crosscoder, 'b_dec_XD') and crosscoder.b_dec_XD is not None:
        output_BX1X2D = output_BX1X2D + crosscoder.b_dec_XD

    # For the residual stream at a single position, we need [B, D]
    # Since crosscoding_dims are (1, 17), we're looking at position 0 of the first dim
    # and need to select which of the 17 positions we're at
    # For now, let's take position 0 of both dimensions to get the residual stream
    output_BD = output_BX1X2D[:, 0, 0, :]  # Take first position

    return output_BD

# --------------------------------------------------------------------------------
# 5) Vector-valued coalition cache: computes & caches y(S) \in R^{D} for each coalition S
# --------------------------------------------------------------------------------
class SAELatentVectorGameCache:
    """
    Caches the *vector* output y(S) for coalition masks S (binary rows).
    We later wrap this with scalar projection games per output coordinate.
    """

    def __init__(
        self,
        *,
        model,
        crosscoder,
        tokens_1S: torch.Tensor,
        start_hook: str,
        end_hook: str,
        pos_idx: int,
        H: int,                          # SAE latent dimension
        active_idx: torch.Tensor,        # [n] indices of active players within H
        phi_active_n: torch.Tensor,      # [n] the active latent values for THIS token
        baseline_pool: BaselinePool,     # [B, n] baselines on active coords
        batch_size: int = 256,           # coalitions per forward batch (times #baselines)
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.crosscoder = crosscoder
        self.forward_from_hook = build_forward_from_hook_to_hook(model, tokens_1S, start_hook, end_hook, pos_idx)
        self.device = _torch_device_of(model)
        self.start_hook = start_hook
        self.end_hook = end_hook
        self.pos_idx = pos_idx
        self.H = int(H)
        self.active_idx = active_idx.to(self.device)
        self.n = int(active_idx.numel())
        self.phi_active_n = phi_active_n.to(self.device, dtype=dtype)
        self.baselines_Bn = baseline_pool.baselines_Bn.to(self.device, dtype=dtype)
        self.B = int(self.baselines_Bn.shape[0])
        self.batch_size = int(batch_size)
        self.dtype = dtype

        # cache: mask_int -> y_D (float32 on CPU to save VRAM)
        self._cache: Dict[int, np.ndarray] = {}

        # Precompute baseline and full coalition for convenience
        self._baseline_vec_D = None   # y(empty)
        self._full_vec_D = None       # y(all)

    @staticmethod
    def masks_to_int(masks_np: np.ndarray) -> np.ndarray:
        """Pack binary mask rows into integers for dict keys."""
        # Handle 1D case (single mask)
        if masks_np.ndim == 1:
            masks_np = masks_np.reshape(1, -1)
        # masks_np: [m, n] with n <= 64ish; if larger, pack to bytes with .tobytes()
        if masks_np.shape[1] <= 62:
            # little-endian bit-pack
            pows = (1 << np.arange(masks_np.shape[1], dtype=np.uint64))
            return (masks_np.astype(np.uint64) * pows).sum(axis=1)
        else:
            # fallback: use bytes as key
            return np.array([hash(row.tobytes()) for row in masks_np], dtype=np.int64)

    @torch.inference_mode()
    def _compute_vectors_for_masks(self, masks_np: np.ndarray) -> np.ndarray:
        """
        Compute y(S) for each row in masks_np, averaging over baselines.
        Returns array [m, D] on CPU (float32). Uses batches over coalitions.
        """
        m = masks_np.shape[0]
        out_chunks: List[np.ndarray] = []
        ptr = 0
        while ptr < m:
            chunk = masks_np[ptr : ptr + self.batch_size]  # [m_c, n]
            m_c = chunk.shape[0]

            # Build all (baseline, coalition) combinations: shape [B*m_c, n]
            masks = torch.from_numpy(chunk).to(self.device, dtype=self.phi_active_n.dtype)  # [m_c, n]
            # broadcast: [B, 1, n], [1, m_c, n], [1, 1, n]
            base = self.baselines_Bn[:, None, :]                               # [B, 1, n]
            mask = masks[None, :, :]                                           # [1, m_c, n]
            phi  = self.phi_active_n[None, None, :]                             # [1, 1, n]
            latents_Bmc_n = base + mask * (phi - base)                          # [B, m_c, n]
            latents_Bmc_n = latents_Bmc_n.reshape(self.B * m_c, self.n)         # [(B*m_c), n]

            # Expand to full H-dim code (non-players set to zero baseline)
            latents_Bmc_H = torch.zeros(self.B * m_c, self.H, device=self.device, dtype=self.dtype)
            latents_Bmc_H[:, self.active_idx] = latents_Bmc_n

            # Decode to resid_pre and run forward to end_hook
            z_Bmc_D = decode_latents_to_resid_pre(self.crosscoder, latents_Bmc_H)     # [(B*m_c), D]
            y_Bmc_D = self.forward_from_hook(z_Bmc_D)                                 # [(B*m_c), D]

            # Average over baselines
            y_B_m_c_D = y_Bmc_D.reshape(self.B, m_c, -1).mean(dim=0)                  # [m_c, D]
            out_chunks.append(y_B_m_c_D.detach().to("cpu", non_blocking=True).float().numpy())
            ptr += m_c

        return np.concatenate(out_chunks, axis=0)  # [m, D]

    def get_vectors(self, masks_np: np.ndarray) -> np.ndarray:
        """
        Return y(S) for each S in masks_np, using cache.
        """
        # Handle 1D case (single mask)
        if masks_np.ndim == 1:
            masks_np = masks_np.reshape(1, -1)
        keys = self.masks_to_int(masks_np)
        to_compute_rows = [i for i, k in enumerate(keys) if k not in self._cache]
        if to_compute_rows:
            computed = self._compute_vectors_for_masks(masks_np[to_compute_rows])
            for i, row_idx in enumerate(to_compute_rows):
                self._cache[int(keys[row_idx])] = computed[i]
        # Assemble in same order
        return np.stack([self._cache[int(k)] for k in keys], axis=0)

    def baseline_vector(self) -> np.ndarray:
        if self._baseline_vec_D is None:
            empty = np.zeros((1, self.n), dtype=np.int64)
            self._baseline_vec_D = self.get_vectors(empty)[0]
        return self._baseline_vec_D

    def full_vector(self) -> np.ndarray:
        if self._full_vec_D is None:
            full = np.ones((1, self.n), dtype=np.int64)
            self._full_vec_D = self.get_vectors(full)[0]
        return self._full_vec_D

# --------------------------------------------------------------------------------
# 6) Step 2: Prepare inputs from harvested activations
# --------------------------------------------------------------------------------
@dataclass
class PrepConfig:
    token_pos: int
    start_hook: str
    end_hook: str
    top_k_players: int = 30
    num_data_baselines: int = 0       # if >0, sample baselines from other positions
    add_zero_baseline: bool = True
    batch_size_coalitions: int = 256  # coalitions per forward *before* multiplying by #baselines
    dtype: torch.dtype = torch.float32
    seed: int = 0

@dataclass
class PreparedTS2:
    game_cache: SAELatentVectorGameCache
    active_idx: torch.Tensor               # [n]
    baseline_vec_D: np.ndarray             # y(empty)
    full_vec_D: np.ndarray                 # y(all)
    # bookkeeping
    H: int
    n: int
    D: int

@torch.inference_mode()
def prepare_ts2_inputs(
    *,
    model,
    crosscoder,
    input_text: str,
    feature_activations_SH: torch.Tensor,   # [S, H] from your get_activations
    activations_SMLD: torch.Tensor,         # [(B·S), M, L, D] (unused here)
    cfg: PrepConfig,
) -> PreparedTS2:
    device = _torch_device_of(model)
    tokens_1S = tokenize_to_device(input_text, model)
    S, H = feature_activations_SH.shape
    D = model.cfg.d_model

    # pick this token's latents, choose players = top-k by |value|
    phi_H = feature_activations_SH[cfg.token_pos].to(device)
    topk = min(cfg.top_k_players, int((phi_H != 0).sum().item()) or cfg.top_k_players)
    top_vals, top_idx = torch.topk(phi_H.abs(), k=topk)
    active_idx = top_idx.sort().values  # keep them sorted
    phi_active_n = phi_H[active_idx]    # [n]

    # baselines on active coords: 0 plus (optional) sampled others
    baselines = []
    if cfg.add_zero_baseline:
        baselines.append(torch.zeros_like(phi_active_n, device=device))
    if cfg.num_data_baselines > 0:
        rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
        # sample other tokens' codes (avoid current position if desired)
        idx = torch.randint(low=0, high=S, size=(cfg.num_data_baselines,), generator=rng)
        sampled = feature_activations_SH[idx.cpu()][:, active_idx.cpu()].to(device)  # [B, n]
        baselines.append(sampled)
    baselines_Bn = torch.vstack(baselines) if len(baselines) > 0 else torch.zeros(1, len(active_idx), device=device)

    # build coalition game cache
    game_cache = SAELatentVectorGameCache(
        model=model,
        crosscoder=crosscoder,
        tokens_1S=tokens_1S,
        start_hook=cfg.start_hook,
        end_hook=cfg.end_hook,
        pos_idx=cfg.token_pos,
        H=H,
        active_idx=active_idx,
        phi_active_n=phi_active_n,
        baseline_pool=BaselinePool(baselines_Bn),
        batch_size=cfg.batch_size_coalitions,
        dtype=cfg.dtype,
    )

    baseline_vec_D = game_cache.baseline_vector()
    full_vec_D = game_cache.full_vector()
    return PreparedTS2(
        game_cache=game_cache,
        active_idx=active_idx,
        baseline_vec_D=baseline_vec_D,
        full_vec_D=full_vec_D,
        H=H,
        n=len(active_idx),
        D=D,
    )

# --------------------------------------------------------------------------------
# 7) Step 3: Compute ST (order ≤ 2) for ALL output coords with shared cache
# --------------------------------------------------------------------------------
@dataclass
class ST2Config:
    budget: int = 8192
    batch_size: int = 128
    random_state: int = 0  # fix so we reuse the *same* coalitions across coords

@dataclass
class ST2Result:
    # tensors are CPU numpy for easy saving; shapes:
    singles_OD: np.ndarray     # [D, n]  order-1 for each output dim
    pairs_Oij: np.ndarray      # [D, n, n] symmetric, zero diag
    baseline_O: np.ndarray     # [D] = y(empty) per output
    full_O: np.ndarray         # [D] = y(all) per output

def _scalar_game_from_vector_cache(game_cache: SAELatentVectorGameCache, out_dim_idx: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function mapping coalitions -> scalar (the given coordinate),
    while reusing the vector cache internally to avoid re-running forwards.
    """
    def game_fn(coalitions: np.ndarray) -> np.ndarray:
        # coalitions: [m, n] binary
        vectors = game_cache.get_vectors(coalitions)        # [m, D]
        result = vectors[:, out_dim_idx]                    # [m]
        # Ensure we return a numpy array with the right dtype
        return result.astype(np.float64)
    return game_fn

def compute_st2_all_outputs(
    prepared: PreparedTS2,
    st_cfg: ST2Config,
) -> ST2Result:
    n = prepared.n
    D = prepared.D

    singles_OD = np.zeros((D, n), dtype=np.float32)
    pairs_Oij  = np.zeros((D, n, n), dtype=np.float32)
    baseline_O = prepared.baseline_vec_D.astype(np.float32).copy()
    full_O     = prepared.full_vec_D.astype(np.float32).copy()

    # We reuse the same sampler seed so the *same coalitions* are queried for every output dim.
    approximator = PermutationSamplingSTII(n=n, max_order=2, random_state=st_cfg.random_state)

    for out_idx in range(D):
        scalar_game = _scalar_game_from_vector_cache(prepared.game_cache, out_idx)
        # IMPORTANT: pass batch_size to let shapiq call our game in batches (we vectorize inside).
        ivals: InteractionValues = approximator.approximate(
            budget=st_cfg.budget,
            game=scalar_game,
            batch_size=st_cfg.batch_size,
        )
        # Extract order-1 and order-2
        # get_n_order_values(1) -> vector length n
        # get_n_order_values(2) -> matrix [n, n] (off-diag are interactions, diag often main-effect convention 0 for STII)
        order1 = ivals.get_n_order_values(1)  # shape [n]
        order2 = ivals.get_n_order_values(2)  # shape [n, n]
        singles_OD[out_idx] = order1.astype(np.float32)
        pairs_Oij[out_idx]  = order2.astype(np.float32)

    return ST2Result(singles_OD=singles_OD, pairs_Oij=pairs_Oij, baseline_O=baseline_O, full_O=full_O)

# --------------------------------------------------------------------------------
# 8) Step 4: Reconstruct and compare
# --------------------------------------------------------------------------------
@dataclass
class ReconReport:
    recon_O: np.ndarray      # reconstructed y(all) from ST2 (baseline + 1st + 2nd)
    true_O: np.ndarray       # actual y(all) from forward on all latents
    abs_err_O: np.ndarray
    rel_err_O: np.ndarray
    l2_abs: float
    l2_rel: float

def compute_ground_truth_directly(
    model,
    crosscoder,
    tokens_1S: torch.Tensor,
    feature_activations_SH: torch.Tensor,
    token_pos: int,
    start_hook: str,
    end_hook: str,
    active_idx: torch.Tensor,
    shapley_ground_truth: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Directly compute the ground truth by:
    1. Taking the actual SAE latents at the specified position
    2. Decoding them back to residual stream
    3. Injecting at hook_resid_pre
    4. Capturing output at hook_resid_post

    This is a simple, readable version for verification.

    Args:
        shapley_ground_truth: If provided, compare with this ground truth from Shapley computation
        verbose: Whether to print verification information
    """
    if verbose:
        print("\n" + "="*60)
        print("VERIFICATION: Computing ground truth directly")
        print("="*60)

    device = _torch_device_of(model)

    # Step 1: Get the actual SAE latents for this token position
    # Note: We should only use the active features to match the Shapley computation
    full_latents_H = torch.zeros(feature_activations_SH.shape[1], device=device, dtype=torch.float32)
    actual_latents_for_active = feature_activations_SH[token_pos][active_idx.cpu()].to(device)
    full_latents_H[active_idx] = actual_latents_for_active

    # Step 2: Decode SAE latents back to residual stream
    # Note: We need to add batch dimension for the decode function
    latents_1H = full_latents_H.unsqueeze(0)  # [1, H]
    decoded_resid_1D = decode_latents_to_resid_pre(crosscoder, latents_1H)  # [1, D]

    # Step 3: Run model with hooks to inject decoded residual and capture output
    output_holder = {}

    def inject_decoded_resid(value, hook):
        # value shape: [batch=1, seq_len, d_model]
        value = value.clone()
        value[:, token_pos, :] = decoded_resid_1D
        return value

    def capture_output(value, hook):
        # Capture the residual stream after this layer
        output_holder["result"] = value[:, token_pos, :].detach().cpu().numpy()
        return value

    # Step 4: Run model with these hooks
    model.run_with_hooks(
        tokens_1S,
        return_type=None,
        fwd_hooks=[
            (start_hook, inject_decoded_resid),
            (end_hook, capture_output),
        ]
    )

    direct_ground_truth = output_holder["result"][0]  # [D] numpy array

    # Step 5: Compare with Shapley ground truth if provided
    if shapley_ground_truth is not None and verbose:
        print(f"\nDirect ground truth shape: {direct_ground_truth.shape}")
        print(f"Shapley ground truth shape: {shapley_ground_truth.shape}")

        # Compute difference
        diff = np.abs(direct_ground_truth - shapley_ground_truth)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\nComparison between direct and Shapley ground truth:")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        print(f"  L2 norm of difference: {np.linalg.norm(diff):.2e}")

        # Show a few sample values for inspection
        print(f"\nFirst 5 values comparison:")
        print(f"  Direct:  {direct_ground_truth[:5]}")
        print(f"  Shapley: {shapley_ground_truth[:5]}")

        if max_diff < 1e-5:
            print("\n✓ VERIFICATION PASSED: Ground truth calculations match (within numerical precision)!")
        else:
            print(f"\n⚠ WARNING: Ground truth calculations differ by {max_diff:.2e}")

    return direct_ground_truth


def evaluate_relative_errors_across_tokens(
    model,
    crosscoder,
    input_text: str,
    num_tokens: int = 20,
    start_hook: str = "blocks.0.hook_resid_pre",
    end_hook: str = "blocks.0.hook_resid_post",
    top_k_players: int = 10,
    budget: int = 512,
    save_path: str = "results/relative_errors.png",
    verbose: bool = True,
) -> Tuple[List[float], List[int], List[str]]:
    """
    Evaluate relative reconstruction errors across the first num_tokens tokens.

    Returns:
        - List of relative errors (as percentages)
        - List of number of active features per token
        - List of token strings
    """
    import os
    import matplotlib.pyplot as plt

    if verbose:
        print(f"\nEvaluating relative errors across first {num_tokens} tokens...")
        print("="*60)

    # Tokenize to get token strings
    tokens = model.tokenizer.encode(input_text)
    token_strings = [model.tokenizer.decode([tok]) for tok in tokens]

    # Get activations for the input
    feature_activations_SH, activations_SMLD = get_activations(input_text, model, crosscoder)

    # Limit to requested number of tokens
    actual_tokens = min(num_tokens, len(tokens), feature_activations_SH.shape[0])
    token_strings = token_strings[:actual_tokens]

    relative_errors = []
    num_active_features = []

    for token_idx in range(actual_tokens):
        if verbose:
            print(f"\nProcessing token {token_idx+1}/{actual_tokens}...", end=" ")

        # Prepare inputs for this token
        prep = prepare_ts2_inputs(
            model=model,
            crosscoder=crosscoder,
            input_text=input_text,
            feature_activations_SH=feature_activations_SH,
            activations_SMLD=activations_SMLD,
            cfg=PrepConfig(
                token_pos=token_idx,
                start_hook=start_hook,
                end_hook=end_hook,
                top_k_players=top_k_players,
                num_data_baselines=2,
                add_zero_baseline=True,
                batch_size_coalitions=32,
                seed=0,
            ),
        )

        # Compute Shapley-Taylor interactions
        st2 = compute_st2_all_outputs(
            prepared=prep,
            st_cfg=ST2Config(budget=budget, batch_size=16, random_state=0),
        )

        # Reconstruct and compute error
        report = reconstruct_and_compare(prep, st2)

        # Convert to percentage
        rel_error_percent = report.l2_rel * 100
        relative_errors.append(rel_error_percent)
        num_active_features.append(prep.n)

        if verbose:
            print(f"Token: '{token_strings[token_idx]}', Active features: {prep.n}, Relative error: {rel_error_percent:.4f}%")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # X positions for plotting
    x_positions = range(actual_tokens)

    # Plot relative errors
    ax1.plot(x_positions, relative_errors, 'b-', marker='o', markersize=6, linewidth=2)
    ax1.set_ylabel('Relative Error (%)', fontsize=12)
    ax1.set_title(f'Shapley Reconstruction Error Across First {actual_tokens} Tokens', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)

    # Add average line
    avg_error = np.mean(relative_errors)
    ax1.axhline(y=avg_error, color='r', linestyle='--', alpha=0.5, label=f'Average: {avg_error:.4f}%')
    ax1.legend(loc='upper right')

    # Plot number of active features
    ax2.bar(x_positions, num_active_features, color='green', alpha=0.6)
    ax2.set_xlabel('Token', fontsize=12)
    ax2.set_ylabel('Number of Active Features', fontsize=12)
    ax2.set_title('Active Features per Token', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Set x-axis to show token text
    ax2.set_xticks(x_positions)
    # Escape special characters and limit length for display
    display_tokens = []
    for tok in token_strings:
        # Replace special characters for display
        tok_display = tok.replace('\n', '\\n').replace('\t', '\\t')
        # Limit length if needed
        if len(tok_display) > 10:
            tok_display = tok_display[:7] + '...'
        display_tokens.append(tok_display)

    ax2.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=9)

    plt.suptitle(f'{start_hook} → {end_hook}, Top-{top_k_players} Features, Budget={budget}',
                 fontsize=10, y=1.02)
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"\nPlot saved to: {save_path}")

    plt.close()

    # Print summary statistics
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Average relative error: {avg_error:.6f}%")
        print(f"Min relative error: {min(relative_errors):.6f}%")
        print(f"Max relative error: {max(relative_errors):.6f}%")
        print(f"Std dev of errors: {np.std(relative_errors):.6f}%")
        print(f"Average active features: {np.mean(num_active_features):.1f}")
        print(f"Total unique active features: {sum(num_active_features)}")

    return relative_errors, num_active_features, token_strings


@dataclass
class ShapleyAccumulator:
    """Accumulates Shapley statistics across multiple tokens/stories."""
    hidden_dim: int
    feature_counts: np.ndarray           # [H] - counts when feature is non-zero
    pairwise_counts: np.ndarray          # [H, H] - counts when both features are non-zero
    pairwise_signed_sum: np.ndarray      # [H, H] - sum of signed pairwise ST indices
    pairwise_abs_sum: np.ndarray         # [H, H] - sum of absolute pairwise ST indices
    single_indices_sum: np.ndarray       # [H] - sum of rank-1 ST indices
    total_samples: int                   # total number of tokens processed

    @staticmethod
    def create(hidden_dim: int) -> "ShapleyAccumulator":
        """Create an empty accumulator."""
        return ShapleyAccumulator(
            hidden_dim=hidden_dim,
            feature_counts=np.zeros(hidden_dim, dtype=np.int64),
            pairwise_counts=np.zeros((hidden_dim, hidden_dim), dtype=np.int64),
            pairwise_signed_sum=np.zeros((hidden_dim, hidden_dim), dtype=np.float64),
            pairwise_abs_sum=np.zeros((hidden_dim, hidden_dim), dtype=np.float64),
            single_indices_sum=np.zeros(hidden_dim, dtype=np.float64),
            total_samples=0,
        )

    def add_sample(
        self,
        active_features: torch.Tensor,  # [n] indices of active features
        st2_result: ST2Result,           # Shapley-Taylor results
        output_dim: Optional[int] = None,  # which output dimension to use (None = average)
    ):
        """Add results from one token to the accumulator."""
        # Convert to numpy if needed
        if isinstance(active_features, torch.Tensor):
            active_features = active_features.cpu().numpy()

        n = len(active_features)

        # Update feature counts (which features were non-zero)
        self.feature_counts[active_features] += 1

        # Update pairwise counts (which pairs were both non-zero)
        for i in range(n):
            for j in range(i + 1, n):
                feat_i = active_features[i]
                feat_j = active_features[j]
                self.pairwise_counts[feat_i, feat_j] += 1
                self.pairwise_counts[feat_j, feat_i] += 1  # symmetric

        # Get ST indices for specified output dim or average across all
        if output_dim is not None:
            singles = st2_result.singles_OD[output_dim]  # [n]
            pairs = st2_result.pairs_Oij[output_dim]     # [n, n]
        else:
            # Average across all output dimensions
            singles = st2_result.singles_OD.mean(axis=0)  # [n]
            pairs = st2_result.pairs_Oij.mean(axis=0)     # [n, n]

        # Accumulate single (rank-1) indices
        self.single_indices_sum[active_features] += singles

        # Accumulate pairwise indices
        for i in range(n):
            for j in range(i + 1, n):
                feat_i = active_features[i]
                feat_j = active_features[j]
                pairwise_value = pairs[i, j]

                # Signed sum
                self.pairwise_signed_sum[feat_i, feat_j] += pairwise_value
                self.pairwise_signed_sum[feat_j, feat_i] += pairwise_value  # symmetric

                # Absolute sum
                self.pairwise_abs_sum[feat_i, feat_j] += abs(pairwise_value)
                self.pairwise_abs_sum[feat_j, feat_i] += abs(pairwise_value)  # symmetric

        self.total_samples += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from accumulated data."""
        active_features = self.feature_counts > 0
        num_active = active_features.sum()

        # Find most frequent pairs
        pair_mask = self.pairwise_counts > 0
        num_pairs = pair_mask.sum() // 2  # divide by 2 for symmetry

        # Average pairwise interaction strength (where both features were active)
        avg_pairwise_interaction = np.zeros_like(self.pairwise_signed_sum)
        nonzero_pairs = self.pairwise_counts > 0
        avg_pairwise_interaction[nonzero_pairs] = (
            self.pairwise_signed_sum[nonzero_pairs] / self.pairwise_counts[nonzero_pairs]
        )

        # Average single indices (where feature was active)
        avg_single_indices = np.zeros_like(self.single_indices_sum)
        nonzero_features = self.feature_counts > 0
        avg_single_indices[nonzero_features] = (
            self.single_indices_sum[nonzero_features] / self.feature_counts[nonzero_features]
        )

        return {
            "num_active_features": num_active,
            "num_active_pairs": num_pairs,
            "total_samples": self.total_samples,
            "avg_pairwise_interaction": avg_pairwise_interaction,
            "avg_single_indices": avg_single_indices,
            "most_frequent_features": np.argsort(self.feature_counts)[-10:][::-1],
            "strongest_positive_pairs": self._get_top_pairs(self.pairwise_signed_sum, k=10, positive=True),
            "strongest_negative_pairs": self._get_top_pairs(self.pairwise_signed_sum, k=10, positive=False),
        }

    def _get_top_pairs(self, matrix: np.ndarray, k: int = 10, positive: bool = True) -> List[Tuple[int, int, float]]:
        """Get top-k pairs by value from a symmetric matrix."""
        # Only look at upper triangle to avoid duplicates
        upper_tri = np.triu(matrix, k=1)

        if positive:
            # Flatten and sort for positive values
            flat_idx = np.argsort(upper_tri.ravel())[-k:][::-1]
        else:
            # For negative values
            flat_idx = np.argsort(upper_tri.ravel())[:k]

        pairs = []
        for idx in flat_idx:
            i = idx // self.hidden_dim
            j = idx % self.hidden_dim
            value = matrix[i, j]
            if (positive and value > 0) or (not positive and value < 0):
                pairs.append((i, j, value))

        return pairs


def accumulate_shapley_over_dataset(
    model,
    crosscoder,
    dataset,
    config,
    num_stories: int = 10,
    tokens_per_story: int = 20,
    top_k_players: int = 10,
    budget: int = 512,
    save_results: bool = True,
    save_path: str = "results/tensors/shapley_accumulator.npz",
    verbose: bool = True,
) -> ShapleyAccumulator:
    """
    Accumulate Shapley statistics over multiple stories from a dataset.

    Args:
        model: The language model
        crosscoder: The SAE crosscoder
        dataset: The dataset to sample from
        num_stories: Number of stories to process
        tokens_per_story: Number of tokens to process per story
        config: Configuration dict with hook points
        top_k_players: Maximum features to consider per token
        budget: Shapley approximation budget
        save_results: Whether to save results to file (default: True)
        save_path: Path to save results (default: "results/tensors/shapley_accumulator.npz")
        verbose: Whether to print progress

    Returns:
        ShapleyAccumulator with accumulated statistics
    """
    # Extract hook configuration
    start_hook = config['hook_points']['start_hook']
    end_hook = config['hook_points']['end_hook']

    # Get hidden dimension from crosscoder
    hidden_dim = crosscoder.hidden_dim

    # Initialize accumulator
    accumulator = ShapleyAccumulator.create(hidden_dim)

    if verbose:
        print(f"\nAccumulating Shapley statistics over {num_stories} stories...")
        print(f"Processing {tokens_per_story} tokens per story")
        print("="*60)

    for story_idx in range(num_stories):
        if verbose:
            print(f"\nStory {story_idx + 1}/{num_stories}")

        # Get story text
        input_text = dataset[story_idx]['text']

        if verbose:
            print(f"  Text preview: {input_text[:50]}...")

        # Get activations
        feature_activations_SH, activations_SMLD = get_activations(input_text, model, crosscoder)

        # Limit tokens
        actual_tokens = min(tokens_per_story, feature_activations_SH.shape[0])

        for token_idx in range(actual_tokens):
            if verbose and token_idx % 5 == 0:
                print(f"  Token {token_idx + 1}/{actual_tokens}...", end=" ")

            # Prepare inputs
            prep = prepare_ts2_inputs(
                model=model,
                crosscoder=crosscoder,
                input_text=input_text,
                feature_activations_SH=feature_activations_SH,
                activations_SMLD=activations_SMLD,
                cfg=PrepConfig(
                    token_pos=token_idx,
                    start_hook=start_hook,
                    end_hook=end_hook,
                    top_k_players=top_k_players,
                    num_data_baselines=2,
                    add_zero_baseline=True,
                    batch_size_coalitions=32,
                    seed=0,
                ),
            )

            # Skip if no active features
            if prep.n == 0:
                continue

            # Compute Shapley-Taylor
            st2 = compute_st2_all_outputs(
                prepared=prep,
                st_cfg=ST2Config(budget=budget, batch_size=16, random_state=0),
            )

            # Add to accumulator (averaging across output dimensions)
            accumulator.add_sample(
                active_features=prep.active_idx,
                st2_result=st2,
                output_dim=None,  # average across all outputs
            )

        if verbose:
            print()  # newline after token progress

    # Print summary statistics
    if verbose:
        stats = accumulator.get_statistics()
        print("\n" + "="*60)
        print("ACCUMULATED STATISTICS")
        print("="*60)
        print(f"Total samples processed: {stats['total_samples']}")
        print(f"Number of active features: {stats['num_active_features']} / {hidden_dim}")
        print(f"Number of active feature pairs: {stats['num_active_pairs']}")
        print(f"\nMost frequent features: {stats['most_frequent_features']}")
        print(f"\nTop positive interactions:")
        for i, j, val in stats['strongest_positive_pairs'][:5]:
            print(f"  Features ({i}, {j}): {val:.6f}")
        print(f"\nTop negative interactions:")
        for i, j, val in stats['strongest_negative_pairs'][:5]:
            print(f"  Features ({i}, {j}): {val:.6f}")

    # Save results if requested
    if save_results:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save all tensors and metadata
        np.savez(save_path,
                 # Core tensors
                 feature_counts=accumulator.feature_counts,
                 pairwise_counts=accumulator.pairwise_counts,
                 pairwise_signed_sum=accumulator.pairwise_signed_sum,
                 pairwise_abs_sum=accumulator.pairwise_abs_sum,
                 single_indices_sum=accumulator.single_indices_sum,
                 # Metadata
                 hidden_dim=accumulator.hidden_dim,
                 total_samples=accumulator.total_samples,
                 num_stories=num_stories,
                 tokens_per_story=tokens_per_story,
                 start_hook=start_hook,
                 end_hook=end_hook,
                 crosscoder_name=config['crosscoder']['wandb_name'],
                 top_k_players=top_k_players,
                 budget=budget)

        if verbose:
            print(f"\nResults saved to: {save_path}")

    return accumulator


def reconstruct_and_compare(prepared: PreparedTS2, st2: ST2Result) -> ReconReport:
    # ST-II reconstruction at the grand coalition uses:
    # y_hat = baseline + sum_i v({i}) + sum_{i<j} v({i,j})   (STII is efficient up to order-2 on pure ≤2-order functions)  # noqa: E501
    # Our InteractionValues(2) returns a full matrix; sum upper triangle (excluding diag).
    singles_sum = st2.singles_OD.sum(axis=1)                                    # [D]
    pairs_sum   = np.triu(st2.pairs_Oij, k=1).sum(axis=(1, 2))                   # [D]
    recon_O     = st2.baseline_O + singles_sum + pairs_sum                       # [D]
    true_O      = st2.full_O                                                     # [D]

    abs_err_O = np.abs(recon_O - true_O)
    denom     = np.maximum(np.abs(true_O), 1e-8)
    rel_err_O = abs_err_O / denom

    l2_abs = float(np.linalg.norm(recon_O - true_O))
    l2_rel = float(np.linalg.norm(recon_O - true_O) / (np.linalg.norm(true_O) + 1e-8))

    return ReconReport(
        recon_O=recon_O, true_O=true_O, abs_err_O=abs_err_O, rel_err_O=rel_err_O, l2_abs=l2_abs, l2_rel=l2_rel
    )


def main(config_path="config.yaml"):
    # Load configuration
    config = load_config(config_path)
    device = setup_device(config)

    print(f"Loading configuration from {config_path}")
    print(f"Using device: {device}")

    # Load dataset
    dataset_config = config['dataset']
    dataset = load_dataset(
        dataset_config['name'],
        split=dataset_config['split']
    )

    # Filter to training examples if specified
    if dataset_config.get('filter_training', False):
        dataset = dataset.filter(lambda x: x['is_training'] == True)
        print(f"Filtered dataset to {len(dataset)} training examples")

    # Load LLM
    model_config = config['model']
    llm = build_llm_lora(
        base_model_repo=model_config['base_model_repo'],
        lora_model_repo=model_config['lora_model_repo'],
        cache_dir=model_config.get('cache_dir'),
        device=device,
        dtype=model_config.get('dtype'),
    )
    print(f"Loaded LLM: {model_config['base_model_repo']} with LoRA: {model_config['lora_model_repo']}")

    # Load crosscoder (using default entity and project)
    crosscoder_config = config['crosscoder']
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc",
        "sleeper-model-diffing",
        crosscoder_config['wandb_name'],
        crosscoder_config['artifacts_dir'],
        device
    )
    print(f"Loaded crosscoder from W&B: {crosscoder_config['wandb_name']}")


    #Loading done



    # Get hook points configuration
    hook_config = config.get('hook_points', {})
    available_hooks = hook_config.get('available_hooks', [])
    start_hook = hook_config.get('start_hook', 'blocks.0.hook_resid_pre')
    end_hook = hook_config.get('end_hook', 'blocks.0.hook_resid_post')

    # Validate hook points
    if available_hooks:
        validate_hook_points(start_hook, end_hook, available_hooks)

    # Get Shapley parameters
    shapley_config = config.get('shapley', {})
    num_stories = shapley_config.get('num_stories', 5)

    print(f"\nReady to calculate Shapley indices:")
    print(f"  - Processing {num_stories} stories")
    print(f"  - Start hook: {start_hook}")
    print(f"  - End hook: {end_hook}")
    print(f"  - Using {shapley_config.get('num_samples', 500)} samples for approximation")

    # 0) Get sample text
    input_text = dataset[0]['text']
    print(f"Story text: {input_text[:50]}...")

    # Run evaluation across first 20 tokens
    relative_errors, num_features, token_strings = evaluate_relative_errors_across_tokens(
        model=llm,
        crosscoder=crosscoder,
        input_text=input_text,
        num_tokens=3,  # Test with just 3 tokens
        start_hook=start_hook,
        end_hook=end_hook,
        top_k_players=10,
        budget=512,
        save_path="results/relative_errors_3tokens.png",
        verbose=True,
    )

    # Accumulate Shapley statistics over dataset
    print("\n" + "="*60)
    print("ACCUMULATING SHAPLEY STATISTICS OVER DATASET")
    print("="*60)

    accumulator = accumulate_shapley_over_dataset(
        model=llm,
        crosscoder=crosscoder,
        dataset=dataset,
        num_stories=num_stories,
        tokens_per_story=3,  # Test with just 3 tokens
        config=config,
        top_k_players=10,
        budget=512,
        save_results=True,  # Will save to results/tensors/shapley_accumulator.npz
        save_path="results/tensors/shapley_accumulator_3tokens.npz",
        verbose=True,
    )

    # 4) VERIFICATION: Compute ground truth directly and compare
    # direct_ground_truth = compute_ground_truth_directly(
    #     model=llm,
    #     crosscoder=crosscoder,
    #     tokens_1S=tokenize_to_device(input_text, llm),
    #     feature_activations_SH=feature_activations_SH,
    #     token_pos=3,  # Same as PrepConfig
    #     layer_idx=0,  # Same as PrepConfig
    #     active_idx=prep.active_idx,
    #     shapley_ground_truth=report.true_O,  # Pass Shapley ground truth for comparison
    #     verbose=False,  # Print verification info
    # )



# st2_sae_pipeline.py
# Minimal, clean scaffold for Shapley–Taylor (order≤2) on SAE latents inside TransformerLens.
# Assumes: PyTorch, einops, shapiq, TransformerLens, your SAE (crosscoder).
                                                  # noqa: E402

                


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='Calculate Shapley indices for SAE features')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    main(args.config)



