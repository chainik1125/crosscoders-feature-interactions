"""Reproduce Figure 1: the computationally-sparse crosscoder tradeoff sweep.

Pulls the canonical 2025-04-18 interaction-penalty sweep from wandb and plots,
versus the penalty strength lambda, the end-of-training reconstruction loss and
the dominant feature's share of the MLP-neuron L1 norm (train/mean_max_ratio_mlp).

This is a pure-metrics figure (no GPU / model needed). Run:
    WANDB_API_KEY=... python reproduce/fig1_sweep.py
Writes reproduce/out/fig1_sweep.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import wandb

ENTITY = "dmitry2-uiuc"
PROJECT = "sleeper-model-diffing"

# Canonical 2025-04-18 sweep (verified vs paper anchors: λ0→.305, λ1000→.816, λ2000→.923).
# λ -> finished run id (λ=1000 is ckubmeg1; the same-date bbnfhse5 FAILED — do not use).
SWEEP = {
    0: "86u64trx", 10: "hhm6y0s6", 20: "ni4z2dkr", 50: "bn2qo3w9",
    100: "vh2bylhi", 200: "7avbfdww", 500: "x21ussr1", 1000: "ckubmeg1",
    2000: "bn1xtudv", 10000: "b5l291e5",
}
SHARE = "train/mean_max_ratio_mlp"
RECON = "train/reconstruction_loss"


def fetch() -> list[tuple[int, float, float]]:
    api = wandb.Api()
    rows = []
    for lam, rid in sorted(SWEEP.items()):
        s = api.run(f"{ENTITY}/{PROJECT}/{rid}").summary
        rows.append((lam, float(s[SHARE]), float(s[RECON])))
    return rows


def plot(rows: list[tuple[int, float, float]], out: Path) -> None:
    lam = [max(r[0], 1) for r in rows]  # 1 stands in for λ=0 on the log axis
    share = [r[1] for r in rows]
    recon = [r[2] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xscale("log")
    ax1.set_xlabel(r"interaction penalty $\lambda$ (0 shown at 1)")
    ln1 = ax1.plot(lam, share, "o-", color="C0", label="dominant feature L1 share")
    ax1.set_ylabel("dominant feature share of neuron $L^1$ norm", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.axhline(0.30, ls=":", color="C0", alpha=0.5)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(lam, recon, "s--", color="C3", label="reconstruction loss")
    ax2.set_ylabel("reconstruction loss", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    ax1.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2], loc="center left")
    ax1.set_title("Fig 1: computationally-sparse crosscoder tradeoff (TinyStories-33M, 2025-04-18 sweep)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    for lam_, sh, rc in rows:
        print(f"  λ={lam_:<6} share={sh:.3f}  recon={rc:.0f}")


if __name__ == "__main__":
    plot(fetch(), Path(__file__).parent / "out" / "fig1_sweep.pdf")
