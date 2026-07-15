"""Templates markdown + code pour la section BERTopic des notebooks de vue."""

from __future__ import annotations

from typing import Literal

ViewKind = Literal["contrastive", "macro_ft", "baseline"]

DEFAULT_SECTION_NUM: dict[ViewKind, int] = {
    "contrastive": 6,
    "macro_ft": 8,
    "baseline": 0,
}


def bertopic_md(section_num: int, *, view_kind: ViewKind = "contrastive") -> str:
    emb_note = {
        "contrastive": "Embeddings = projection contrastive (`projected_*.npy`).",
        "macro_ft": "Embeddings = projection CE fine-tuné (`projected_*.npy`).",
        "baseline": "Embeddings = Qwen brut (`dim_*` dans le CSV d'embeddings).",
    }[view_kind]
    return f"""## {section_num}. Topic modeling BERTopic (intra-macro)

Segmentation du corpus choisi en sous-corpus **A0 / A1 / B / C** (classe prédite par défaut).
{emb_note}
Hyperparamètres : `configs/bertopic_notebook.yaml` (+ `bertopic_macro_shared.yaml`).

Les sorties sont écrites sous `{{RESULTS_DIR}}/bertopic_notebook/<corpus>/` et consommées par le notebook **06** (réseau bayésien macro-contraint).

Paramètres modifiables ci-dessous : corpus, mode segmentation, UMAP, `min_topic_size`, relance (`RESTIMATE_BERTOPIC`).
Prérequis LLM : variable d'environnement `OPENAI_API_KEY`.
"""


def bertopic_params_snippet(
    *,
    view_kind: ViewKind,
    default_corpus: str = "metallurgie",
) -> str:
    if view_kind == "baseline":
        return """BERTOPIC_CORPUS = corpus_id  # corpus test de la section courante
BERTOPIC_SEGMENT_MODE = "predicted"  # predicted | true_label
RESTIMATE_BERTOPIC = False
BERTOPIC_UMAP_ENABLED = None       # True / False pour surcharger le YAML
BERTOPIC_MIN_TOPIC_SIZE = None     # int pour surcharger le YAML
"""
    return f"""BERTOPIC_CORPUS = "{default_corpus}"
BERTOPIC_SEGMENT_MODE = "predicted"  # predicted | true_label
RESTIMATE_BERTOPIC = False
BERTOPIC_UMAP_ENABLED = None       # True / False pour surcharger le YAML
BERTOPIC_MIN_TOPIC_SIZE = None     # int pour surcharger le YAML
"""


def _preds_block(view_kind: ViewKind) -> str:
    if view_kind == "macro_ft":
        return """
preds_bertopic = None
_pred_cache = RESULTS_DIR / "transfer" / "target_macro_predictions.csv"
if _pred_cache.is_file():
    _cached = pd.read_csv(_pred_cache)
    if "pred_macro" in _cached.columns:
        preds_bertopic = _cached
if preds_bertopic is None and RUN_INFERENCE and ART.checkpoint_dir is not None:
    from supervised_macro_ft.notebook_viz import build_prediction_df
    from safer_core.test_corpus import resolve_test_corpus

    _data_path = (
        DATA_CSV
        if BERTOPIC_CORPUS == "btp"
        else resolve_test_corpus(BERTOPIC_CORPUS, anchor=ROOT).data_csv
    )
    _meta_full = pd.read_csv(_data_path)
    preds_bertopic = build_prediction_df(
        ART.checkpoint_dir,
        _meta_full,
        _meta_full[TEXT_COL].astype(str).tolist(),
        label_col=LABEL_COL,
        text_col=TEXT_COL,
        device=INFER_DEVICE,
        batch_size=INFER_BATCH_SIZE,
        results_dir=RESULTS_DIR,
        corpus_id=BERTOPIC_CORPUS,
        anchor=ROOT,
        backbone_emb_csv=BACKBONE_EMB_CSV,
    )
"""
    if view_kind == "baseline":
        return """
preds_bertopic = pd.read_csv(TRANSFER_DIR / "target_macro_predictions.csv")
"""
    return """
preds_bertopic = None
"""


def bertopic_code(
    view_kind: ViewKind,
    *,
    method_expr: str,
    method_key_expr: str | None = None,
    results_dir_expr: str = "RESULTS_DIR",
    label_col_expr: str = "LABEL_COL",
    text_col_expr: str = "TEXT_COL",
    seed_expr: str = "42",
) -> str:
    method_key = method_key_expr or method_expr
    preds = _preds_block(view_kind)
    return f"""
{bertopic_params_snippet(view_kind=view_kind)}

from macro_transfer.notebook_bertopic import (
    bertopic_run_dir,
    display_notebook_bertopic_results,
    load_notebook_bertopic_config,
    run_notebook_bertopic,
)

cfg_full = load_notebook_bertopic_config(anchor=ROOT)
nb_cfg = cfg_full.get("notebook") or {{}}
bertopic_out = bertopic_run_dir(
    {results_dir_expr},
    BERTOPIC_CORPUS,
    output_subdir=str(nb_cfg.get("output_subdir", "bertopic_notebook")),
)
{preds}
_assign = bertopic_out / "topics_bertopic" / "assignments.csv"
if RESTIMATE_BERTOPIC or not _assign.is_file():
    bt_cfg = dict(cfg_full)
    if BERTOPIC_UMAP_ENABLED is not None:
        bt_cfg.setdefault("bertopic", {{}}).setdefault("umap", {{}})["enabled"] = bool(BERTOPIC_UMAP_ENABLED)
    if BERTOPIC_MIN_TOPIC_SIZE is not None:
        bt_cfg.setdefault("bertopic", {{}})["min_topic_size"] = int(BERTOPIC_MIN_TOPIC_SIZE)
    bertopic_cfg, topics_export_cfg, topic_judge_cfg = (
        bt_cfg.get("bertopic"),
        bt_cfg.get("topics_export"),
        bt_cfg.get("topic_judge"),
    )
    run_notebook_bertopic(
        {results_dir_expr},
        BERTOPIC_CORPUS,
        method_name={method_expr},
        view_kind="{view_kind}",
        segment_mode=BERTOPIC_SEGMENT_MODE or str(nb_cfg.get("segment_mode", "predicted")),
        label_col={label_col_expr},
        text_col={text_col_expr},
        preds=preds_bertopic,
        bertopic_cfg=bertopic_cfg,
        topics_export_cfg=topics_export_cfg,
        topic_judge_cfg=topic_judge_cfg,
        anchor=ROOT,
        method_key={method_key},
        seed=int({seed_expr}),
        export_for_bn=bool(nb_cfg.get("export_for_bn", True)),
    )
    print("BERTopic terminé :", bertopic_out)
else:
    print("Cache BERTopic :", bertopic_out)

display_notebook_bertopic_results(bertopic_out)
print("→ Entrée notebook 06 BN :", bertopic_out)
"""
