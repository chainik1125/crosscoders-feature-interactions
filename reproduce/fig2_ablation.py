"""Reproduce Figure 2: MLP feature-ablation fidelity for computationally-sparse
crosscoders on TinyStories-33M.

The dominant feature at an MLP neuron is argmax_H |p_SNH| (per-feature pre-activation
contribution, from _common.feature_preact). We re-run the MLP keeping different
feature subsets, reconstruct the post-MLP residual, patch it into the model, and
report the fidelity (loss recovered) of each scheme:

    none            full reconstruction (all features)
    drop_random     zero a random active feature at each neuron
    drop_second     zero the 2nd-largest feature at each neuron
    drop_largest    zero the dominant feature at each neuron
    keep_largest    keep ONLY the dominant feature at each neuron

Fidelity  Phi = 1 - (L_scheme - L_model) / (L_zero - L_model),
with L_zero = zero-ablating all features (MLP output -> bias only).

Run on a GPU box with the post_fork `sleepers` env (see reproduce/README.md):
    HF_TOKEN=... python reproduce/fig2_ablation.py
Writes reproduce/out/fig2_ablation.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _common import DEVICE, build_model, feature_preact, harvest, load_crosscoder

LORA = "mars-jason-25/tiny-stories-33M-TSdata-ft1"
DATASET = "mars-jason-25/tiny_stories_instruct_sleeper_data"
SCHEMES = ["none", "drop_random", "drop_second", "drop_largest", "keep_largest"]


@torch.no_grad()
def ablated_post_mlp(cc, enc_SH, raw_SXD, model, block, scheme):
    """Reconstructed post-MLP residual for `block` under a feature-ablation scheme."""
    mlp = model.blocks[block].mlp
    p_SNH = feature_preact(cc, enc_SH, mlp, block)               # (S, N, H)

    if scheme == "zero":
        p_SNH = torch.zeros_like(p_SNH)
    elif scheme != "none":
        order = p_SNH.abs().argsort(dim=-1, descending=True)     # rank features per (S,N)
        largest = order[..., 0]
        if scheme == "keep_largest":
            mask = torch.zeros_like(p_SNH, dtype=torch.bool)
            mask.scatter_(-1, largest.unsqueeze(-1), True)
            p_SNH = p_SNH * mask
        else:
            if scheme == "drop_largest":
                tgt = largest
            elif scheme == "drop_second":
                tgt = order[..., 1]
            elif scheme == "drop_random":
                active = enc_SH != 0
                rnd = torch.rand(p_SNH.shape, device=p_SNH.device) * active[:, None, :]
                tgt = rnd.argmax(dim=-1)
            else:
                raise ValueError(scheme)
            p_SNH = p_SNH.scatter(-1, tgt.unsqueeze(-1), 0.0)

    b_dec_D = cc.b_dec_XD[0, 4 * block + 3, :]
    preact_SN = p_SNH.sum(dim=-1) + mlp.b_in[None, :] + (mlp.W_in.T @ b_dec_D)[None, :]
    post = mlp.act_fn(preact_SN) @ mlp.W_out + mlp.b_out
    resid_mid = cc._forward(raw_SXD).output_BXD[:, 0, 4 * block + 2, :]
    return resid_mid + post                                       # reconstructed resid_post


@torch.no_grad()
def fidelity_row(model, texts, cc, hooks, block, n):
    accum = {k: [] for k in SCHEMES + ["model", "zero"]}
    patch_hook = f"blocks.{block}.hook_resid_post"
    for prompt in texts[:n]:
        tokens, lm, raw_SXD = harvest(model, hooks, prompt)
        enc_SH = cc.forward_train(raw_SXD).hidden_BH
        accum["model"].append(lm)
        for scheme in SCHEMES + ["zero"]:
            rec_BSD = ablated_post_mlp(cc, enc_SH, raw_SXD, model, block, scheme).unsqueeze(0)
            L = model.run_with_hooks(
                tokens, return_type="loss",
                fwd_hooks=[(patch_hook, lambda acts, hook, _r=rec_BSD: _r)],
            ).item()
            accum[scheme].append(L)
    mean = {k: sum(v) / len(v) for k, v in accum.items()}
    lm, l0 = mean["model"], mean["zero"]
    return {s: 1 - (mean[s] - lm) / (l0 - lm) for s in SCHEMES}


def main(block: int = 1, n_texts: int = 100):
    from datasets import load_dataset

    texts = load_dataset(DATASET, split="test").filter(lambda x: x["is_training"])["text"]
    model = build_model(LORA)
    rows = {}
    for folder in ("tinystories_lambda0", "tinystories_lambda1000"):
        cc, hooks = load_crosscoder(folder)
        rows[folder] = fidelity_row(model, texts, cc, hooks, block, n_texts)
        print(folder, {k: round(v, 3) for k, v in rows[folder].items()})

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(SCHEMES))
    for i, (folder, r) in enumerate(rows.items()):
        ax.bar([xi + i * 0.4 for xi in x], [r[s] for s in SCHEMES], width=0.4,
               label=folder.replace("tinystories_", ""))
    ax.set_xticks([xi + 0.2 for xi in x])
    ax.set_xticklabels(SCHEMES, rotation=20)
    ax.set_ylabel("fidelity Φ (loss recovered)")
    ax.set_title(f"Fig 2: MLP ablation fidelity, block {block} (TinyStories-33M)")
    ax.legend()
    fig.tight_layout()
    out = Path(__file__).parent / "out" / "fig2_ablation.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
