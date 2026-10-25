# GOLD decisions: source-only LODO experiments

These two campaigns evaluate annotation-level decisions independently. They do
not use segmenter predictions and do not report end-to-end extraction scores.

- **Facthood:** classify an exact GOLD factual span as `KEEP` (including role
  `AUTRE`) or an explicitly annotated `excluded_span` as `REJECT`. Unannotated
  gaps are not fabricated as negatives.
- **Roles:** assign `A0`, `A1`, `B`, `C`, or `AUTRE` to an exact GOLD factual
  span. This isolates role discrimination from boundary errors.

Both use the frozen six-corpus leave-one-domain-out IDs from the RQ1 bundle:
five source corpora for model fitting, source validation for method/epoch and
threshold selection, and the sixth corpus only for final evaluation. The three
comparisons are word/bigram TF-IDF with class-weighted logistic regression,
frozen Qwen3 embeddings with logistic regression, and Qwen3 fine-tuning (binary
BCE for facthood; five-class cross-entropy for roles). The fine-tuned model is
run once with seed 42 for each held-out target; the deterministic feature
baselines also run once per target. This is a single-run comparison and does
not estimate variability across training seeds.
No supervised contrastive or SoftTriple run is included. Fine-tuning and
excluded-span encoding use the recorded Qwen3 snapshot; the smoke job compares
one recomputed factual vector per corpus with its supplied CSV vector and stops
if their cosine similarity is below 0.999.

Balanced accuracy is the primary score for both decisions. Macro-F1 is a
secondary summary. Facthood outputs also report KEEP and REJECT recall, since
balanced accuracy alone can hide which side of the gate is failing. Role
outputs report recall for each of the five roles. Per-example test predictions,
class frequencies, confusion matrices, split IDs/provenance hashes, and scores
are retained under the run directory for later narrative-level uncertainty
analysis. The exact split records remain in the private RQ1 bundle; the run
directory records their manifest/signature hashes rather than copying private
split files.

## Files required on the cluster

1. Push the repository code, including `scripts/decision_gold_models.py` and
   the Slurm files in this directory.
2. Provide the six GOLD v3 CSV files in `text/dataset/` and
   `text/dataset/gold_v3_manifest.json`.
3. Provide the private RQ1 annotation/split bundle and set `ROLE_GOLD_BUNDLE`
   if it is outside the script's recognized default paths. It must contain
   `manifest.json`, `complete.json`, and the six `folds/target_<corpus>/`
   directories.
4. Transfer the six frozen Qwen3 CSV exports to
   `text/embeddings/gold_v3/qwen3/{btp,metallurgie,caou,bois_textille,service_temporaire,nicollin}.csv`.
   Each must have `doc_id` followed by 1,024 `dim_*` columns in the GOLD row
   order. Preparation verifies their IDs, order, schema, and hashes.
5. Ensure the métier stop-word list is present at
   `text/macro_transfer/stop_metier.txt`. It is used only by TF-IDF. If your
   cluster checkout does not contain it, copy that file manually to the same
   relative path; the campaign does not upload it.
6. Use the project's `text/.venv` (or set `ROLE_GOLD_VENV`) with the existing
   PyTorch/Transformers/scikit-learn dependencies. The public
   `Qwen/Qwen3-Embedding-0.6B` model must be in the shared Hugging Face cache or
   downloadable from the cluster. GPU work requests one A100/H100 and up to
   96 GB host RAM per task.

The facthood campaign first makes frozen embeddings for explicit excluded
spans, because the existing Qwen3 CSVs contain only factual units. It then
runs one GPU smoke job. Each successful launch starts six LODO fit tasks (one
per held-out domain, using seed 42), with no more than three GPU tasks running
at once, and finishes with a summary job. The role campaign uses the
same sequence without the extra exclusion-embedding array. A launch guard
prevents starting either campaign while the other is still queued or running,
so their combined GPU concurrency stays at three.

From the cluster repository root (`SAFER/`), submit the two decisions
separately with:

```bash
sbatch text/jobs/decision_gold_facthood.sh
sbatch text/jobs/decision_gold_roles.sh
```

The first command launches only the binary keep/reject campaign. The second
launches only five-role classification. If a campaign is interrupted, wait for
its remaining jobs to leave the queue, then rerun the same command; completed
fold/seed outputs are verified and skipped.
