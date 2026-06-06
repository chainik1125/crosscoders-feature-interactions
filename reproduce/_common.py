"""Shared helpers for the GPU reproduction scripts (fig2, fig4).

Everything here is behaviour-preserving relative to the original inline code in
fig2_ablation.py / fig4_sleeper.py — it is the same loading, harvesting and
per-feature MLP-preactivation math, factored out so each figure script only
expresses what is unique to it.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from einops import einsum, rearrange
from huggingface_hub import hf_hub_download, snapshot_download

from model_diffing.models.crosscoder import AcausalCrosscoder
from sleepers.scripts.llms import build_llm_lora

HF_REPO = "dmanningcoe/crosscoders-feature-interactions"
BASE_MODEL = "roneneldan/TinyStories-Instruct-33M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_crosscoder(folder: str):
    """Load a published crosscoder from the Hub. The HF folders are post_fork's
    native save format, so `AcausalCrosscoder.load` reads them directly (no shim).
    Returns (crosscoder, hookpoints)."""
    root = snapshot_download(HF_REPO, allow_patterns=f"{folder}/*")
    cc = AcausalCrosscoder.load(Path(root) / folder, device=DEVICE).to(DEVICE)
    cfg = yaml.unsafe_load(open(hf_hub_download(HF_REPO, f"{folder}/experiment_config.yaml")))
    return cc, cfg["hookpoints"]


def build_model(lora: str):
    return build_llm_lora(base_model_repo=BASE_MODEL, lora_model_repo=lora,
                          cache_dir=None, device=DEVICE, dtype=None)


@torch.no_grad()
def harvest(model, hook_names, prompt):
    """Run the model on one prompt and stack the hookpoint activations into the
    crosscoder's (S, 1, L, D) layout. Returns (tokens, model_loss, acts_SXD)."""
    tokens = model.to_tokens(prompt)[:, :128]
    loss, cache = model.run_with_cache(tokens, names_filter=hook_names, return_type="loss")
    acts_BSLD = torch.stack([cache[n] for n in hook_names], dim=2)
    acts_SXD = rearrange(acts_BSLD.unsqueeze(2), "b s m l d -> (b s) m l d")
    return tokens, loss.item(), acts_SXD


def feature_preact(cc, enc_SH, mlp, block):
    """Per-feature contribution to each MLP-neuron pre-activation at `block`:
    p[s,n,h] = enc[s,h] * (W_in . W_dec[mlp-input hookpoint])[n,h]. Shape (S, N, H).
    Biases are intentionally excluded — callers add them as needed."""
    W_dec_HD = cc.W_dec_HXD[:, 0, 4 * block + 3, :]
    data_w_NH = einsum(mlp.W_in, W_dec_HD, "d_model d_mlp, hidden d_model -> d_mlp hidden")
    return enc_SH[:, None, :] * data_w_NH[None, :, :]
