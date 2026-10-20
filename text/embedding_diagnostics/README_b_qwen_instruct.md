# Task-instructed embeddings for B event mechanisms

This is a separate Qwen3-Embedding export for the valid factual units labelled
B in the Building construction and masonry subfamily. It does not overwrite
the original Qwen or E5 exports and does not use UMAP or an LLM annotator.

Every unit is encoded with the same English task instruction, followed by the
Qwen3 model-card input form `Instruct: ...\nQuery:{text}`. English follows the
model card's recommendation for multilingual instructions. The export uses
the last non-padding token pooling and L2 normalization from the model-card
recipe. Since the existing general-purpose export used the project's generic
mean-pooling pipeline with no instruction prefix, this is a new task-adapted
representation as a whole; differences cannot be attributed to the instruction
alone.

## Run the four role-specific exports together

The combined Slurm job processes B, A0, A1 and C. It loads Qwen once, applies
the matching instruction for each role, and writes separate CSVs and manifests.
From the project `text/` directory:

```bash
cd /home/aayapi/SAFER/text
sbatch jobs/export_all_roles_instruct.sh
```

The Slurm log prints valid-unit counts for each role and a batch progress bar.
All outputs are written under
`embeddings/instructed/btp_building_construction_and_masonry/`. Each JSON
manifest records the role-specific instruction, input hash, row count, pooling
rule, dimensions, and export hash. The older B-only job remains available if a
later rerun of B alone is needed.

Before submission, a data-only dry run can confirm the number of selected
units for all four roles without loading the model:

```bash
python embedding_diagnostics/export_b_instruct_embeddings.py \
  --config embedding_diagnostics/config_qwen_instruct_all_roles.yaml --dry-run
```

The full export loads Qwen3-Embedding-0.6B and therefore needs access to its
Hugging Face weights, typically through the mesocentre cache or permitted
download access.
