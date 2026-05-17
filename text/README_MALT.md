# MALT-EM — SCGM with transferred soft superclasses

MALT-EM (Macro-Anchored Latent Transfer with EM) adapts SCGM-G to a **target** corpus of text embeddings. The coarse macro label is not observed on the target; it is replaced by transferred soft responsibilities `p0(y|x)` from source anchors learned on BTP.

## Idea

1. Train SCGM-Text on source BTP → anchors `mu_y^s`, projector `f_s`.
2. For each target embedding `e_i^t`, compute `p0_i = softmax(cos(f_s(e_i^t), mu^s) / tau_macro)`.
3. Learn target projector `f_t`, anchors `mu_y^t`, latent motifs `nu_k^t`.
4. **E-step** (full train set): Sinkhorn on scores  
   `P[i,k] = p_t(z=k|x_i) * exp(sum_m p0_i(m) log p_t(y=m|z=k))`.
5. **M-step**: fix `q` from E-step; minimize  
   `L_EM = -mean_i sum_k q_ik [ log p_t(z=k|x_i) + sum_m p0_i(m) log p_t(y=m|z=k) ]`  
   plus optional anchor / diversity / macro consistency / balance terms.

| Aspect | SCGM (source) | MALT-EM (target) |
|--------|----------------|------------------|
| Coarse supervision | observed `y` | soft `p0(y|x)` |
| Latent inference | full E-step Sinkhorn | full E-step Sinkhorn |
| M-step | `q` fixed | `q` fixed |
| Transfer | — | anchor + transferred `p0` |
| Fine motifs | source domain | re-estimated on target |

## Commands

From `text/`:

```bash
cd text
python scripts/train_malt_target.py --config configs/malt_btp_to_mettalurgie_qwen06.yaml
```

Export:

```bash
python scripts/export_malt_outputs.py \
  --checkpoint runs/malt_btp_to_mettalurgie_qwen06/best_model.pt \
  --source_checkpoint runs/scgm_text_qwen06/best_model.pt \
  --output_dir runs/malt_btp_to_mettalurgie_qwen06/exports
```

Evaluation:

```bash
python scripts/evaluate_malt_transfer.py \
  --exports_dir runs/malt_btp_to_mettalurgie_qwen06/exports \
  --output_dir runs/malt_btp_to_mettalurgie_qwen06/evaluation \
  --label_col pred_label
```

**Evaluation outputs** (`evaluation/`):

- `metrics_table.csv` — rows: `Embedding brut` (if `--emb_csv` or `raw_embeddings.npy`), `MALT_source`, `MALT_adapted` with `eta2_macro_balanced`, `eta2_weighted`, inertias `T_*`/`W_*`/`B_*`, RankMe, C1, C10
- `metrics_summary.json` — `{ "metrics_table": [...], "malt_diagnostics": { ... } }`
- `macro_transition_matrix.csv`, `z_macro_affiliation.csv`, `anchor_drift.csv`

Classifier metrics, subtype ARI/NMI, and clustering scores are no longer computed at evaluation.

Notebook: `notebooks/02_malt_btp_to_mettalurgie_transfer.ipynb`

## Key hyperparameters

- `n_iter_estep` — run full E-step every N epochs (and epoch 1).
- `em_q_mode` — `hard` (one-hot `q`) or `soft` (Sinkhorn matrix).
- `sinkhorn_lmd` — Sinkhorn temperature (same role as SCGM `lmd`).
- `beta_anchor`, `beta_div`, `beta_macro`, `beta_balance` — M-step regularizers.
- `init_q_mode` — `source_scores`, `p0_block`, `kmeans`, `random`.

## Outputs

Training run directory:

- `best_model.pt`, `last_model.pt` (includes `q_em` in checkpoint)
- `p0_y_target.npy`, `q_em_final.npy`, `z_hat_em_final.npy`
- `metrics/train_log.csv`, `epoch_metrics.jsonl`, `logs.csv`
- `q_epoch_XXXX.npy` when `save_q_every_estep=true`

Export directory:

- `metadata_with_malt_predictions.csv` — `p0_*`, `pt_*`, `z_hat`, `q_z_hat`, …
- `pt_y_given_z.npy`, `nu_target.npy`, `macro_transition_p0_to_pt.csv`

## Module map

- `malt_text/malt_em_model.py` — target model + `compute_all_probs`
- `malt_text/malt_em_estep.py` — full-dataset E-step
- `malt_text/malt_em_losses.py` — M-step losses
- `malt_text/malt_em_training.py` — training loop

## Limitations

- E-step memory scales as `O(N * K)` for the score matrix.
- Checkpoints from pre–MALT-EM training runs are not compatible (re-train required).
- `sinkhornknopp` prints iteration logs to stdout (inherited from SCGM).
