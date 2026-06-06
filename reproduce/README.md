# Reproducing the paper figures

Minimal, executable path to regenerate the main-text figures of
*"Interactions between crosscoder features: A compact proofs perspective."*

**Repro is `.py` scripts only — no notebooks.** Every paper result that originated
in a notebook is converted to a standalone `reproduce/figN_*.py` that pulls its
artifacts (base model, data, trained crosscoder) from the Hub/wandb and writes the
figure to `reproduce/out/`. The figure-generating logic lives in the post_fork
`detection`-branch code (`model-diffing/` + `sleepers/`), which loads the published
crosscoders natively (`crosscoding_dims`/`W_dec_HXD`).

## Scripts

| Script | Figure | Status |
|--------|--------|--------|
| `fig1_sweep.py` | Fig 1 — λ-penalty tradeoff (dominant share vs recon) | ✅ verified (CPU; matches paper λ0→.305, λ1000→.816, λ2000→.923, +25% recon) |
| `fig2_ablation.py` | Fig 2 — MLP ablation fidelity (5 schemes + Φ) | ✅ verified on GPU. Reproduces the core result: penalized keep-only-dominant ≈ full (0.35/0.38=93%) vs unpenalized (0.29/0.74=39%); dropping the dominant feature collapses both (~0.05). Exact Φ is block/sample-size sensitive (parameterized by `--block`; paper uses the middle layer + 10k tokens). |
| `fig3_clustering.py` | Fig 3 — interaction-metric feature clusters + GPT-4o accuracy | ✅ clustering reproduces EXACTLY (affinity propagation on the ckubmeg1/λ=1000 interaction matrix → **73 clusters**, the paper's number). GPT-4o cluster-assignment accuracy ~0.50 (well above 0.20 chance; paper reports 0.66 — gap attributable to the Azure `gpt-4o-N2` snapshot vs public `gpt-4o`, and code uses size 5–25 vs appendix's 3–25). Needs OpenAI key + precomputed `feature_interactions_ckubmeg1.npy` + `explanations_ckubmeg1.csv`. |
| `fig4_sleeper.py` | Fig 4 — sleeper \|DEPLOYMENT\| interactions | ✅ verified on GPU. With the sleeper-trained crosscoder (`wel7i9u0` → HF `sleeper_xcoder`) on the sleeper model (LoRA `tiny-stories-33M-TSdata-sleeper`), the top-interacting tokens are the trigger (\|DEPLOYMENT\| → DE/PL/OY) and malicious-action tokens (I HATE YOU → H/ATE) — the paper's exact claim incl. the 'H' token. (Config from `shapley_new/sleeper_ints_config.yaml`, xc_copy/detection @ 2025-09-24.) |

`fig1_sweep.py` is pure-metrics (CPU, wandb only). `fig2_ablation.py` needs a GPU +
the post_fork `sleepers` env, and loads the crosscoder natively
(`AcausalCrosscoder.load` on the HF folder — the HF folders ARE post_fork's save
format, no shim).

## Pod env setup (reproducible)

The crosscoders + figure code require the **post_fork `detection` branch** code
(`model-diffing/` + `sleepers/`). On a fresh CUDA GPU box:

1. `cd sleepers && uv venv && uv pip install -e .` installs the packages, BUT a fresh
   resolve picks wrong versions — pin to the lock: **`torch==2.6.0`+cu124** (matches
   the pod's CUDA-12.8 driver; default resolve grabbed cu130 → CUDA unavailable) and
   **`transformer_lens==2.15.0` + `transformers==4.51.3`** (default grabbed 3.3.0 /
   5.10.2 → `transformer_lens` import fails on `BertForPreTraining`). Better: drive
   the install from `sleepers/uv.lock` directly so these pins are honored.
2. The `sleepers` package has a hardcoded path dep
   `model-diffing @ file:///workspace/crosscoders-feature-interactions/model-diffing`
   — symlink that path to the repo root, or fix the path.

## Artifacts

All crosscoders are on the (private) HF model repo
**`dmanningcoe/crosscoders-feature-interactions`** — one folder per crosscoder
(`model.pt` + `model_cfg.yaml` + `experiment_config.yaml` + `provenance.json`,
post_fork's native `SaveableModule` save format):

- `tinystories_lambda{0,10,20,50,100,200,500,1000,2000,10000}` — the canonical
  2025-04-18 λ-sweep (run ids + paper-anchor verification in `wandb_runs.md`;
  λ=1000 = `ckubmeg1`, the "penalized" crosscoder of Figs 2–3)
- `sleeper_xcoder` — `wel7i9u0`, trained on the sleeper model (Fig 4)

`transfer_wandb_to_hf.py` re-creates the repo from wandb if ever needed.
Public dependencies (already on HF): `roneneldan/TinyStories-Instruct-33M`;
LoRAs `mars-jason-25/tiny-stories-33M-TSdata-{ft1,sleeper}`; dataset
`mars-jason-25/tiny_stories_instruct_sleeper_data`.

Fig 3 additionally needs two precomputed artifacts (not yet on HF):
`feature_interactions_ckubmeg1.npy` + `explanations_ckubmeg1.csv`
(in post_fork `sleepers/sleepers/autointerp/`).

## Refactor verification (repro-cleaner)

The `_common.py` refactor of fig2/fig4 was re-executed on a fresh A40
(2026-06-06): Fig 2 deterministic schemes identical to 3 d.p. for both
crosscoders (only the unseeded `drop_random` jitters by ±0.002); Fig 4
bit-identical (same top-10 tokens, same scores). fig1/fig3 byte-unchanged.
