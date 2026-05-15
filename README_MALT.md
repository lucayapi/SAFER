# MALT — Macro-Anchored Latent Transfer

MALT est une extension de transfert macro-ancré pour SCGM-Text. Elle n’est pas l’implémentation image SCGM d’origine.

## Idée scientifique

1. Entraîner SCGM sur le corpus source BTP.
2. Transférer les ancres macro A0/A1/B/C vers un corpus cible.
3. Ne pas transférer directement les motifs fins source.
4. Réapprendre localement un ensemble global de `K` motifs latents cibles.
5. Inférer la relation macro via `p_t(y|z)`.
6. Utiliser des pseudo-labels macro souples `p0(y|x)` induits par les ancres source.

Cette version conserve un **seul K global** (par défaut 32), comme SCGM classique. Il n’y a pas de partition `K_A0`, `K_A1`, etc.

## Formules principales

- Projection source gelée : `v_j^s = f_s(e_j^t)`
- Pseudo-labels macro : `p0(y|x) = softmax_m(cos(v^s, mu_m^s) / tau_macro)`
- Projection cible : `v_j^t = f_t(e_j^t)`
- Motifs latents : `p_t(z|x) = softmax_k(cos(v^t, nu_k^t) / tau_z)`
- Macro conditionnelle : `p_t(y|z) = softmax_m(cos(nu_k^t, mu_m^t) / tau_yz)`
- Macro marginale : `p_t(y|x) = sum_k p_t(y|z=k) p_t(z|x)`

## Loss

`L = L_softmacro + beta_latent L_latent + beta_anchor L_anchor + beta_div L_div`

Les termes sont loggés séparément. Des flags d’ablation (`--disable_softmacro`, etc.) sont prévus dans le code.

Le notebook `02_malt_btp_to_mettalurgie_transfer.ipynb` utilise par défaut **`FORCE_RETRAIN = True`** : chaque exécution relance l’entraînement si `RUN_TRAINING=True`. Mettre **`FORCE_RETRAIN = False`** pour sauter lorsque `best_model.pt` existe déjà (reprendre export / figures sans réentraîner).

L’évaluation exportée peut inclure un **ARI global** (`ari_z_vs_pred_subtype_micro`) entre **`z_hat`** et une colonne sous-type (ex. `pred_subtype`) **uniquement** si vous passez `--subtype_col` à une colonne présente dans les métadonnées : vue *micro* sur les segments renseignés. Le notebook `02_malt_btp_to_mettalurgie_transfer` **n’utilise pas** ce signal par défaut.

## Commandes

Entraînement :

```bash
python scripts/train_malt_target.py \
  --config configs/malt_btp_to_mettalurgie_qwen06.yaml
```

Export :

```bash
python scripts/export_malt_outputs.py \
  --checkpoint runs/malt_btp_to_mettalurgie_qwen06/best_model.pt \
  --source_checkpoint runs/scgm_text_qwen06/best_model.pt \
  --output_dir runs/malt_btp_to_mettalurgie_qwen06/exports
```

Évaluation :

```bash
python scripts/evaluate_malt_transfer.py \
  --exports_dir runs/malt_btp_to_mettalurgie_qwen06/exports \
  --output_dir runs/malt_btp_to_mettalurgie_qwen06/evaluation
```

Notebook :

```bash
jupyter lab notebooks/02_malt_btp_to_mettalurgie_transfer.ipynb
```

## Prérequis

- Checkpoint source SCGM BTP (`runs/scgm_text_qwen06/best_model.pt`).
- CSV cible métallurgie (`dataset/data_metallurgie.csv` ou variante orthographique).
- Embeddings cible Qwen 0.6B alignés sur `doc_id`.
- Si `pred_label` est présent sur la cible, il sert uniquement au diagnostic.

## Limites

- Les pseudo-labels `p0` ne sont pas une vérité terrain experte.
- Les métriques de classification cible restent diagnostiques si les labels sont automatiques.
- Le transfert macro ne garantit pas l’alignement sémantique fine des motifs latents.
