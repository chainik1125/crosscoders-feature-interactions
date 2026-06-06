"""Transfer trained crosscoders from wandb artifacts to a Hugging Face model repo.

Pulls each wandb model-checkpoint artifact, picks the final-epoch `.pt`, writes a
`config.json` alongside it (so downstream loading needs no hardcoded params), and
uploads to `HF_REPO` under one folder per crosscoder. Nothing is kept on disk.

Auth: expects WANDB_API_KEY and HF_TOKEN in the environment (both present here).

Usage:
    python reproduce/transfer_wandb_to_hf.py            # transfer all in REGISTRY
    python reproduce/transfer_wandb_to_hf.py crosscoder_S_3072   # just one
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import wandb
from huggingface_hub import HfApi

WANDB_ENTITY = "dmitry2-uiuc"
HF_REPO = "dmanningcoe/crosscoders-feature-interactions"

# folder_in_hf -> (wandb_project, run_id, paper_role)
# NOTE: the mainline λ-sweep BatchTopK/1536 runs are NOT yet identified (blocker 1
# in reproduce/README.md). The entries below are the confirmed `published_models`
# crosscoders (JumpReLU/3072 sleeper-diffing set). Add the mainline runs once known:
#   "tinystories_lambda0":    ("<project>", "<run_id>", "Figs 1-3 unpenalized λ=0"),
#   "tinystories_lambda200":  ("<project>", "<run_id>", "Fig 1 sweep λ=200"),
#   "tinystories_lambda1000": ("<project>", "<run_id>", "Figs 1-3 penalized λ=1000"),
#   "tinystories_lambda2000": ("<project>", "<run_id>", "Fig 1 sweep λ=2000"),
REGISTRY: dict[str, tuple[str, str, str]] = {
    # --- Fig 4 sleeper crosscoder: CONFIRMED from shapley_new/sleeper_ints_config.yaml
    #   (xc_copy/detection @ 2025-09-24 "starting experiments on interactions in sleepers").
    #   sleeper cc = wel7i9u0 (trained on sleeper LoRA TSdata-sleeper); base cc = 86u64trx
    #   (= tinystories_lambda0). NB kv0gxxb7 is a BASE (ft1) crosscoder, NOT the sleeper one.
    "sleeper_xcoder":     ("sleeper-model-diffing", "wel7i9u0", "Fig 4 sleeper crosscoder (trained on TSdata-sleeper model, λ=0)"),
    # "sleeper_xcoder_alt": ("sleeper-model-diffing", "6f4kzfbo", "Fig 4 sleeper alternate (commented out)"),

    # --- 'published_models' S/D/M/DF/MF set (JumpReLU/3072/5-resid) — model-diffing/feature_analysis ---
    "crosscoder_S_3072":  ("published_models", "b1exnef6", "feature-analysis S (base LoRA ft1)"),
    "crosscoder_D_3072":  ("published_models", "u8ah75j0", "feature-analysis D"),
    "crosscoder_M_3072":  ("published_models", "2vxm3g8l", "feature-analysis M (sleeper LoRA)"),
    "crosscoder_DF_3072": ("published_models", "w7xq09ps", "feature-analysis DF"),
    "crosscoder_MF_3072": ("published_models", "ffii9281", "feature-analysis MF"),

    # --- MAINLINE λ-sweep (Figs 1-3): CANONICAL 2025-04-18 sweep, all FINISHED ---
    #   sleeper-model-diffing, penalty = config train.lam_n. Auto-matched by
    #   train/mean_max_ratio_mlp to paper Fig 1 (λ0→30%, λ1000→80%, λ2000→92%, +25% recon).
    #   NB λ=1000 is ckubmeg1 (finished); bbnfhse5 at the same λ/date FAILED — do not use.
    "tinystories_lambda0":     ("sleeper-model-diffing", "86u64trx", "Figs 1-3 unpenalized λ=0 (share .305)"),
    "tinystories_lambda10":    ("sleeper-model-diffing", "hhm6y0s6", "Fig 1 sweep λ=10 (.344)"),
    "tinystories_lambda20":    ("sleeper-model-diffing", "ni4z2dkr", "Fig 1 sweep λ=20 (.367)"),
    "tinystories_lambda50":    ("sleeper-model-diffing", "bn2qo3w9", "Fig 1 sweep λ=50 (.405)"),
    "tinystories_lambda100":   ("sleeper-model-diffing", "vh2bylhi", "Fig 1 sweep λ=100 (.447)"),
    "tinystories_lambda200":   ("sleeper-model-diffing", "7avbfdww", "Fig 1 sweep λ=200 (.516)"),
    "tinystories_lambda500":   ("sleeper-model-diffing", "x21ussr1", "Fig 1 sweep λ=500 (.640)"),
    "tinystories_lambda1000":  ("sleeper-model-diffing", "ckubmeg1", "Figs 1-3 penalized λ=1000 (.816)"),
    "tinystories_lambda2000":  ("sleeper-model-diffing", "bn1xtudv", "Fig 1 sweep λ=2000 (.923)"),
    "tinystories_lambda10000": ("sleeper-model-diffing", "b5l291e5", "Fig 1 sweep λ=10000 (.930)"),
}


def transfer_one(api: wandb.Api, hf: HfApi, folder: str, project: str, run_id: str, role: str) -> None:
    run = api.run(f"{WANDB_ENTITY}/{project}/{run_id}")
    arts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not arts:
        print(f"  !! {folder}: no model artifact on run {run_id}, skipping", flush=True)
        return
    art = arts[-1]

    # skip_cache=True => never populates the persistent wandb cache (disk-safe);
    # tempdir is removed when the with-block exits, so peak usage ≈ one checkpoint.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(art.download(root=tmp, skip_cache=True))
        pts = list(d.rglob("*.pt"))
        if not pts:
            print(f"  !! {folder}: artifact has no .pt file, skipping", flush=True)
            return
        ckpt = max(pts, key=lambda p: p.stat().st_size)  # the model weights

        # provenance.json records which wandb run + paper role this is
        prov = {
            "paper_role": role,
            "wandb": f"{WANDB_ENTITY}/{project}/{run_id}",
            "lam_n": run.config.get("train", {}).get("lam_n"),
            "hidden_dim": run.config.get("crosscoder", {}).get("hidden_dim"),
            "n_hookpoints": len(run.config.get("hookpoints", [])),
        }
        (d / "provenance.json").write_text(json.dumps(prov, indent=2))

        # upload model.pt as <folder>/model.pt plus any yaml configs + provenance
        uploads = [(ckpt, "model.pt")] + [
            (y, y.name) for y in d.rglob("*.yaml")
        ] + [(d / "provenance.json", "provenance.json")]
        for path, name in uploads:
            hf.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{folder}/{name}",
                repo_id=HF_REPO,
                repo_type="model",
            )
        extras = ", ".join(n for _, n in uploads[1:])
        print(f"  ✓ {folder}: model.pt (+{extras}) -> {HF_REPO}", flush=True)


def main() -> None:
    wanted = sys.argv[1:] or list(REGISTRY)
    api = wandb.Api()
    hf = HfApi()
    hf.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=True)
    print(f"Transferring {len(wanted)} crosscoder(s) -> {HF_REPO}")
    for folder in wanted:
        if folder not in REGISTRY:
            print(f"  ?? unknown '{folder}', skipping")
            continue
        transfer_one(api, hf, folder, *REGISTRY[folder])


if __name__ == "__main__":
    main()
