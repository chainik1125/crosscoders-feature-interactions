# wandb run inventory (dmitry2-uiuc)

All training logs to **`dmitry2-uiuc/sleeper-model-diffing`** (`build_wandb_run` in
`model_diffing/utils.py`). The interaction-penalty strength is the **`lam_n`** field
under `train` (runs named `lambda_n{lam_n}_S_{date}`). Mainline geometry = hidden_dim
1536, 17 hookpoints (resid_pre/ln1/resid_mid/ln2 × blocks 0–3 + resid_post).

## Activation: BatchTopK (verified)

Checkpoint state_dicts hold only 5 tensors (W_enc_MLDH, W_dec_HMLD, b_enc_H, b_dec_MLD,
folded_scaling_factors_ML) and **no learnable `log_threshold`** → BatchTopK, not
JumpReLU. The `crosscoder.jumprelu` field in wandb configs is stale default metadata.

## Dominant-share metric

Paper "dominant feature's share of L1 norm" = wandb **`train/mean_max_ratio_mlp`**.
Reconstruction loss = `train/reconstruction_loss`.

## Mainline λ-sweep (Figs 1–3) — CANONICAL set = 2025-04-18 sweep

Auto-matched across all 203 finished+metric runs (38 λ values). The **2025-04-18**
sweep is complete, all FINISHED, monotonic in dominant share, consistent recon, and
hits every paper anchor (λ0→30%, λ1000→80%, λ2000→92%, recon λ0→2000 = +25%):

| λ (`lam_n`) | run id | `mean_max_ratio_mlp` | recon | paper anchor |
|------|----------|------|------|--------------|
| 0    | `86u64trx` | 0.305 | 1962 | "30%" ✓ |
| 10   | `hhm6y0s6` | 0.344 | 2186 | |
| 20   | `ni4z2dkr` | 0.367 | 1867 | |
| 50   | `bn2qo3w9` | 0.405 | 1956 | |
| 100  | `vh2bylhi` | 0.447 | 1849 | |
| 200  | `7avbfdww` | 0.516 | 1769 | |
| 500  | `x21ussr1` | 0.640 | 2044 | |
| 1000 | `ckubmeg1` | 0.816 | 2363 | "80%" ✓ |
| 2000 | `bn1xtudv` | 0.923 | 2451 | "92%, +25% recon" ✓ |
| 10000| `b5l291e5` | 0.930 | 4268 | |

WARNING: at λ=1000/04-18 there are TWO runs — `ckubmeg1` (finished, USE) and
`bbnfhse5` (FAILED, lastStep −1, do not use). Earlier draft wrongly used bbnfhse5.
Other dates (03-12 etc.) have recon ~10k–13k (different normalization/unfolded scaling)
— not the paper set. Full raw dump: `b0kykwo07` scan (203 rows) saved in chat history.

### Older raw run lists (superseded by table above)

| λ (`lam_n`) | run id (2025-04-18) | run id (2025-04-22) |
|------|---------------------|---------------------|
| 0    | `l2ez7y09` (also 79lgt82k, 86u64trx) | — |
| 10   | `7ltititr`          | `ef3m3eq3` |
| 20   | `6stnd7g8`          | — |
| 50   | `gq4t4hwa`          | `wlkl2xjv` |
| 100  | `8ks8fsiv`          | `baikb1tl` |
| 200  | `mgr4gbpe`          | — |
| 500  | `5z7a6gbj`          | `9dsfm9u7` |
| 1000 | `bbnfhse5`          | `v7128kc4` (also v20s0hyh) |
| 2000 | `bn1xtudv`          | `z21ddlef` |
| 10000| `b5l291e5`          | `mvb2hc1q` |

Paper-referenced points: λ=0 (unpenalized baseline), λ=200, λ=1000 (the "penalized"
crosscoder used in Figs 2–3 and the tables), λ=2000 (92% share / 25% loss claim).

## Fig 4 sleeper crosscoder

`dmitry2-uiuc/sleeper-model-diffing/kv0gxxb7` (active in `sleepers/.../mlp_analysis.ipynb`;
commented alt `6f4kzfbo`). 1536-dim, 17 hookpoints. Also on disk at
`code/data/sleeper_xcoders/crosscoder_S/model_epoch_20.pt`.

## NOT the paper mainline (don't use for Figs 1–3)

`dmitry2-uiuc/published_models` — `crosscoder_{S,D,M,DF,MF}_3072`: JumpReLU, hidden
3072, 5 resid hookpoints, LoRA model-diffing set (feature_analysis.ipynb). Different
experiment.

Open question: config `crosscoder` dict still lists `jumprelu`; paper Methods say
BatchTopK (K=20). Confirm the activation actually used for the λ-sweep runs (may be a
stale config field with topk set via `train.c`/`k`).
