"""Orchestration BERTopic pour notebooks de vue + export compatible réseau bayésien."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from macro_transfer.bertopic_config import enrich_run_config_bertopic, resolve_bertopic_run_config
from macro_transfer.bertopic_exports import export_bertopic_datamaps_from_run
from macro_transfer.constants import MACRO_NAMES
from macro_transfer.intra_bertopic import fit_bertopic_per_macro
from macro_transfer.topics_export import build_macro_topic_test_table
from safer_core.classification_eval import (
    DEFAULT_CLASSIFIER,
    fit_logistic_on_embeddings,
)
from safer_core.classification_metrics import build_gating_from_predictions
from safer_core.io import load_yaml
from safer_core.paths import TEXT_ROOT, resolve_repo_path
from safer_core.test_corpus import (
    resolve_final_embeddings_csv_in_dir,
    resolve_projected_embeddings_in_dir,
    resolve_test_corpus,
)

DEFAULT_NOTEBOOK_BERTOPIC_YAML = "configs/bertopic_notebook.yaml"
TRANSFER_PREDS_NAME = "target_macro_predictions.csv"


def load_notebook_bertopic_config(*, anchor: Optional[Path] = None) -> Dict[str, Any]:
    """Charge ``bertopic_notebook.yaml`` fusionné avec ``bertopic_macro_shared.yaml``."""
    root = Path(anchor or TEXT_ROOT).resolve()
    path = resolve_repo_path(DEFAULT_NOTEBOOK_BERTOPIC_YAML, repo_root=root)
    raw = load_yaml(path)
    return enrich_run_config_bertopic(raw, anchor=root)


def bertopic_run_dir(results_dir: Path, corpus_id: str, *, output_subdir: str = "bertopic_notebook") -> Path:
    return Path(results_dir).resolve() / output_subdir / str(corpus_id)


def build_gating_from_true_labels(
    meta: pd.DataFrame,
    label_col: str = "pred_label",
    *,
    macros: Sequence[str] = MACRO_NAMES,
) -> pd.DataFrame:
    """Gating one-hot compatible ``fit_bertopic_per_macro`` (mode vraie classe)."""
    labels = meta[label_col].astype(str)
    out = pd.DataFrame(index=meta.index)
    out["m_hat"] = labels
    out["ambiguous"] = False
    out["q_conf"] = 1.0
    for m in macros:
        out[f"p_{m}"] = (labels == m).astype(float)
    return out


def resolve_gating(
    meta: pd.DataFrame,
    preds: Optional[pd.DataFrame],
    *,
    mode: str = "predicted",
    label_col: str = "pred_label",
    macros: Sequence[str] = MACRO_NAMES,
) -> pd.DataFrame:
    if str(mode).strip().lower() == "true_label":
        return build_gating_from_true_labels(meta, label_col, macros=macros)
    if preds is None or preds.empty:
        raise ValueError("Mode 'predicted' requiert un DataFrame de prédictions (pred_macro, prob_*).")
    return build_gating_from_predictions(preds, macros)


def _align_meta_embeddings(
    meta: pd.DataFrame,
    embeddings: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    meta = meta.reset_index(drop=True)
    emb = np.asarray(embeddings, dtype=np.float64)
    if len(meta) != emb.shape[0]:
        n = min(len(meta), emb.shape[0])
        meta = meta.iloc[:n].copy()
        emb = emb[:n]
    return meta, emb


def load_projected_corpus_pair(
    results_dir: Path,
    corpus_id: str,
    *,
    method_key: Optional[str] = None,
    anchor: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    pair = resolve_projected_embeddings_in_dir(
        results_dir, corpus_id, method=method_key, anchor=anchor
    )
    if pair is None:
        raise FileNotFoundError(f"Embeddings projetés absents pour {corpus_id} sous {results_dir}")
    projected = np.load(pair[0])
    meta = pd.read_csv(pair[1])
    meta, emb = _align_meta_embeddings(meta, projected)
    return emb, meta


def load_qwen_corpus_embeddings(
    corpus_id: str,
    *,
    anchor: Optional[Path] = None,
    label_col: str = "pred_label",
) -> Tuple[np.ndarray, pd.DataFrame]:
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    from scgm_text.utils_io import create_doc_id_if_missing

    root = Path(anchor or TEXT_ROOT).resolve()
    spec = resolve_test_corpus(corpus_id, anchor=root) if corpus_id != "btp" else None
    if corpus_id == "btp":
        data_path = root / "dataset" / "data_btp.csv"
        emb_path = root / "embeddings" / "Qwen3-Embedding-0.6B_btp.csv"
    else:
        data_path = spec.data_csv
        emb_path = spec.emb_csv
    meta = create_doc_id_if_missing(pd.read_csv(data_path))
    slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
    merged, dim_cols = merge_metadata_with_embeddings(slim, str(emb_path), strict=False)
    return merged[dim_cols].to_numpy(dtype=np.float64), merged


def predictions_from_lr_on_projected(
    results_dir: Path,
    corpus_id: str,
    *,
    method_key: str,
    anchor: Optional[Path] = None,
    label_col: str = "pred_label",
    seed: int = 42,
) -> pd.DataFrame:
    """Fit logistic sur BTP projeté, prédit le corpus cible (mode contrastif / macro_ft)."""
    z_btp, meta_btp = load_projected_corpus_pair(results_dir, "btp", method_key=method_key, anchor=anchor)
    z_tgt, meta_tgt = load_projected_corpus_pair(results_dir, corpus_id, method_key=method_key, anchor=anchor)
    from scgm_text.data_metadata import LABEL2ID

    y_train = meta_btp[label_col].astype(str).map(LABEL2ID).astype(int).to_numpy()
    y_eval = meta_tgt[label_col].astype(str).to_numpy()
    pipe = fit_logistic_on_embeddings(z_btp, y_train, classifier=DEFAULT_CLASSIFIER, seed=seed)
    from macro_transfer.supervised_baseline import _predict_with_probs

    pred_macro, probs, _, conf, _ = _predict_with_probs(pipe, z_tgt, list(MACRO_NAMES))
    rows: Dict[str, Any] = {}
    for col in meta_tgt.columns:
        rows[col] = meta_tgt[col].tolist()
    rows["pred_macro"] = list(pred_macro)
    rows["confidence"] = list(conf)
    for i, m in enumerate(MACRO_NAMES):
        rows[f"prob_{m}"] = probs[:, i].tolist()
    return pd.DataFrame(rows)


def build_transfer_predictions_csv(
    meta: pd.DataFrame,
    preds: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Écrit ``transfer/target_macro_predictions.csv`` pour le staging BN."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [c for c in ("accident_id", "fact_id", "doc_id", "sentence", "pred_label") if c in meta.columns]
    pred_cols = [c for c in ("pred_macro", "confidence") if c in preds.columns]
    prob_cols = [f"prob_{m}" for m in MACRO_NAMES if f"prob_{m}" in preds.columns]
    df = meta[base_cols].copy() if base_cols else meta.copy()
    for c in pred_cols + prob_cols:
        if c in preds.columns and len(preds) == len(df):
            df[c] = preds[c].values
    if "pred_macro" in df.columns and "m_hat" not in df.columns:
        df["m_hat"] = df["pred_macro"].astype(str)
    if "confidence" in df.columns and "q_conf" not in df.columns:
        df["q_conf"] = pd.to_numeric(df["confidence"], errors="coerce")
    df.to_csv(out_path, index=False)
    return out_path


def build_notebook_bertopic_summary_table(
    macro_topic_counts: Mapping[str, Mapping[str, Any]],
    assignments: pd.DataFrame,
    themes: pd.DataFrame,
) -> pd.DataFrame:
    """Tableau récap : macro, n_units, n_topics, outliers, bruit %, plus gros topic."""
    return build_macro_topic_test_table(dict(macro_topic_counts), assignments, themes)


def run_notebook_bertopic(
    results_dir: Path,
    corpus_id: str,
    *,
    method_name: str,
    view_kind: str = "contrastive",
    segment_mode: str = "predicted",
    label_col: str = "pred_label",
    text_col: str = "sentence",
    preds: Optional[pd.DataFrame] = None,
    bertopic_cfg: Optional[Mapping[str, Any]] = None,
    topics_export_cfg: Optional[Mapping[str, Any]] = None,
    topic_judge_cfg: Optional[Mapping[str, Any]] = None,
    anchor: Optional[Path] = None,
    method_key: Optional[str] = None,
    seed: int = 42,
    export_for_bn: bool = True,
) -> Path:
    """
    Lance BERTopic intra-macro et écrit sous ``{results_dir}/bertopic_notebook/{corpus}/``.

    Structure exportée (consommée par ``stage_bn_exports_from_bertopic_run``) :
    - ``topics_bertopic/assignments.csv``, ``themes_by_macro.csv``
    - ``transfer/target_macro_predictions.csv``
    - ``bertopic/<macro>/`` (modèles + DataMapPlot)
    """
    root = Path(anchor or TEXT_ROOT).resolve()
    results_dir = Path(results_dir).resolve()
    cfg_bundle = dict(bertopic_cfg or {})
    if not cfg_bundle:
        full = load_notebook_bertopic_config(anchor=root)
        bertopic_cfg, topics_export_cfg, topic_judge_cfg = resolve_bertopic_run_config(full, anchor=root)
    else:
        topics_export_cfg = dict(topics_export_cfg or {})
        topic_judge_cfg = dict(topic_judge_cfg or {})

    nb_cfg = dict((load_notebook_bertopic_config(anchor=root).get("notebook") or {}))
    out_root = bertopic_run_dir(results_dir, corpus_id, output_subdir=str(nb_cfg.get("output_subdir", "bertopic_notebook")))
    out_root.mkdir(parents=True, exist_ok=True)

    if view_kind == "baseline":
        z, meta = load_qwen_corpus_embeddings(corpus_id, anchor=root, label_col=label_col)
    else:
        z, meta = load_projected_corpus_pair(results_dir, corpus_id, method_key=method_key, anchor=root)

    if segment_mode == "predicted" and preds is None and view_kind != "baseline":
        from safer_core.classification_eval import load_saved_predictions

        preds = load_saved_predictions(results_dir, corpus_id)
        if preds is None:
            preds = predictions_from_lr_on_projected(
                results_dir,
                corpus_id,
                method_key=method_key or method_name,
                anchor=root,
                label_col=label_col,
                seed=seed,
            )
    if preds is None and segment_mode == "predicted" and view_kind == "baseline":
        raise ValueError("Mode predicted en baseline : fournir ``preds`` (sortie classifieur sklearn).")

    gating = resolve_gating(meta, preds, mode=segment_mode, label_col=label_col)

    if export_for_bn:
        build_transfer_predictions_csv(
            meta,
            preds if preds is not None else pd.DataFrame({"pred_macro": meta[label_col].astype(str)}),
            out_root / "transfer" / TRANSFER_PREDS_NAME,
        )

    bertopic_cfg_dict = dict(bertopic_cfg or {})
    diagnostics = dict(bertopic_cfg_dict.get("diagnostics") or {})
    if nb_cfg.get("save_datamap", True):
        diagnostics["save_datamap"] = True
    bertopic_cfg_dict["diagnostics"] = diagnostics

    themes, assignments, partial = fit_bertopic_per_macro(
        z,
        meta,
        gating,
        method=method_name,
        bertopic_cfg=bertopic_cfg_dict,
        output_dir=out_root / "topics_bertopic",
        legacy_output_dir=out_root / "topics_bertopic",
        per_macro_output_root=out_root / "bertopic",
        run_output_root=out_root,
        sentence_col=text_col,
        top_k_words=int((topics_export_cfg or {}).get("top_k_words", 12)),
        top_k_sentences=int((topics_export_cfg or {}).get("top_k_sentences", 5)),
        repo_anchor=root,
        corpus_id=corpus_id,
    )

    try:
        export_bertopic_datamaps_from_run(
            out_root,
            meta,
            z,
            text_col=text_col,
            fig_dir=out_root / "figures",
            assignments_path=out_root / "topics_bertopic" / "assignments.csv",
        )
    except Exception:
        pass

    manifest = {
        "method": method_name,
        "corpus_id": corpus_id,
        "view_kind": view_kind,
        "segment_mode": segment_mode,
        "bertopic_run_dir": str(out_root),
        "n_units": int(len(meta)),
        "macro_topic_counts": partial.get("macro_topic_counts", {}),
        "warnings": partial.get("warnings", []),
    }
    (out_root / "bertopic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = build_notebook_bertopic_summary_table(
        partial.get("macro_topic_counts", {}), assignments, themes
    )
    summary.to_csv(out_root / "topics_summary_by_macro.csv", index=False)
    if not themes.empty:
        themes_out = out_root / "topics_bertopic" / "themes_by_macro.csv"
        themes_out.parent.mkdir(parents=True, exist_ok=True)
        themes.to_csv(themes_out, index=False)
    return out_root


def display_notebook_bertopic_results(
    bertopic_run_dir: Path,
    *,
    macros: Sequence[str] = MACRO_NAMES,
) -> None:
    """Affiche tableau récap, détail topics, barres effectifs et DataMapPlot."""
    import matplotlib.pyplot as plt

    from IPython.display import display
    from macro_transfer.notebook_viz import show_bertopic_datamaps_inline

    root = Path(bertopic_run_dir).resolve()
    summary_path = root / "topics_summary_by_macro.csv"
    themes_path = root / "topics_bertopic" / "themes_by_macro.csv"

    if summary_path.is_file():
        print("=== Récapitulatif BERTopic par macro ===")
        display(pd.read_csv(summary_path))
    if themes_path.is_file():
        themes = pd.read_csv(themes_path)
        cols = [c for c in ("macro", "topic_id", "theme_label", "n_units", "top_words") if c in themes.columns]
        print("=== Topics (libellés LLM) ===")
        display(themes[cols].sort_values(["macro", "topic_id"]))

        if {"macro", "topic_id", "n_units"}.issubset(themes.columns):
            fig, ax = plt.subplots(figsize=(9, 4))
            sub = themes.sort_values("n_units", ascending=True)
            labels = sub.apply(
                lambda r: f"{r['macro']}·T{int(r['topic_id'])}", axis=1
            )
            ax.barh(labels.astype(str), sub["n_units"].astype(float))
            ax.set_xlabel("Effectif")
            ax.set_title("Effectifs par topic")
            plt.tight_layout()
            plt.show()

    show_bertopic_datamaps_inline(root, macros=macros)
