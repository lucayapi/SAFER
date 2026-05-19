"""Dimensions d'embedding pour normaliser RankMe (rankme / d)."""

from __future__ import annotations

QWEN3_EMBEDDING_06B_DIM = 1024
SCGM_DEFAULT_HIDIM = 128

# Libellés affichés (notebook 01) → d pour rankme_over_d si absent des CSV
METHOD_LABEL_EMBEDDING_DIM: dict[str, int] = {
    "Embedding brut": QWEN3_EMBEDDING_06B_DIM,
    "Batch Triplet": QWEN3_EMBEDDING_06B_DIM,
    "SupCon": QWEN3_EMBEDDING_06B_DIM,
    "SoftTriple": QWEN3_EMBEDDING_06B_DIM,
    "SCGM": SCGM_DEFAULT_HIDIM,
}


def embedding_dim_for_display_label(method_label: str) -> int:
    name = str(method_label).strip()
    for suffix in ("_btp", "_test", "_BTP", "_TEST"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    if name in METHOD_LABEL_EMBEDDING_DIM:
        return METHOD_LABEL_EMBEDDING_DIM[name]
    if name.startswith("SCGM"):
        return SCGM_DEFAULT_HIDIM
    return QWEN3_EMBEDDING_06B_DIM
