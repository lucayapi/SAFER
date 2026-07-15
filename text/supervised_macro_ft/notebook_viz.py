"""Visualisations notebook pour supervised_macro_ft (métriques, t-SNE, prédictions CE)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES
from safer_core.brand_style import (
    ACCENT_BLUE,
    ACCENT_RED,
    DEEP_BLUE,
    HERO_BLUE,
    apply_matplotlib_brand,
    bar_kwargs,
    macro_color_map,
    matplotlib_heatmap_cmap,
    style_axes,
    style_figure,
)
from safer_core.classification_eval import EMBEDDING_STEMS

MACRO_COLOR_HEX: Dict[str, str] = macro_color_map()


@dataclass
class MacroFtArtifacts:
    """Artefacts d'un run supervised_macro_ft (train BTP)."""

    results_dir: Path
    cv_per_fold: Optional[pd.DataFrame] = None
    cv_summary: Optional[pd.DataFrame] = None
    cross_domain: Optional[pd.DataFrame] = None
    all_test_metrics: Optional[pd.DataFrame] = None
    metrics_by_corpus: Dict[str, pd.DataFrame] = field(default_factory=dict)
    train_history: Optional[pd.DataFrame] = None
    train_summary: Optional[Dict[str, Any]] = None
    checkpoint_dir: Optional[Path] = None
    projected_corpora: List[str] = field(default_factory=list)
    tuning_grid: Optional[pd.DataFrame] = None
    missing: List[str] = field(default_factory=list)


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def validate_results_dir(results_dir: str | Path) -> Path:
    """Vérifie qu'un dossier ressemble à un run macro_ft."""
    root = Path(results_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier absent : {root}")
    markers = (
        root / "train_summary.json",
        root / "cv" / "cv_summary.csv",
        root / "kfold_summary.csv",
        root / "metrics" / "cross_domain_generalization.csv",
    )
    if not any(p.is_file() for p in markers):
        raise FileNotFoundError(
            f"Dossier non reconnu comme run supervised_macro_ft : {root}\n"
            "Attendu : train_summary.json, cv/cv_summary.csv ou metrics/cross_domain_generalization.csv"
        )
    return root


def discover_projected_corpora(results_dir: str | Path) -> List[str]:
    emb_dir = Path(results_dir) / "embeddings"
    if not emb_dir.is_dir():
        return []
    stems: List[str] = []
    for stem in EMBEDDING_STEMS:
        if (emb_dir / f"projected_{stem}.npy").is_file():
            stems.append(stem)
    return stems


def load_supervised_macro_ft_train_history(train_out: str | Path) -> pd.DataFrame:
    """Charge l'historique epoch par epoch (CV + fit final BTP)."""
    root = Path(train_out).resolve()
    frames: list[pd.DataFrame] = []
    cv_path = root / "cv" / "train_history.csv"
    final_path = root / "train_history_final.csv"
    if cv_path.is_file():
        frames.append(pd.read_csv(cv_path))
    if final_path.is_file():
        frames.append(pd.read_csv(final_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_macro_ft_artifacts(results_dir: str | Path) -> MacroFtArtifacts:
    """Charge métriques, historique et chemins utiles depuis un dossier de run."""
    root = validate_results_dir(results_dir)
    art = MacroFtArtifacts(results_dir=root)

    art.cv_per_fold = _read_optional_csv(root / "cv" / "cv_per_fold.csv")
    art.cv_summary = _read_optional_csv(root / "cv" / "cv_summary.csv")
    if art.cv_summary is None:
        art.cv_summary = _read_optional_csv(root / "kfold_summary.csv")

    art.cross_domain = _read_optional_csv(root / "metrics" / "cross_domain_generalization.csv")
    art.all_test_metrics = _read_optional_csv(root / "metrics" / "all_test_corpora_metrics.csv")

    btp_cls = _read_optional_csv(root / "metrics" / "metrics_classification_btp.csv")
    if btp_cls is not None:
        art.metrics_by_corpus["btp"] = btp_cls
    for stem in EMBEDDING_STEMS:
        if stem == "btp":
            continue
        m = _read_optional_csv(root / "metrics" / f"metrics_classification_test_{stem}.csv")
        if m is not None:
            art.metrics_by_corpus[stem] = m

    art.train_history = load_supervised_macro_ft_train_history(root)
    summary_path = root / "train_summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            art.train_summary = json.load(f)

    ckpt = root / "checkpoints" / "best_model"
    if ckpt.is_dir() and (ckpt / "config.json").is_file():
        art.checkpoint_dir = ckpt
    else:
        art.missing.append(str(ckpt))

    art.projected_corpora = discover_projected_corpora(root)

    tuning_path = root / "tuning" / "grid_summary.csv"
    if not tuning_path.is_file() and root.parent.name == "combos":
        tuning_path = root.parent.parent / "grid_summary.csv"
    art.tuning_grid = _read_optional_csv(tuning_path)

    return art


def _metric_to_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "—":
            return np.nan
        head = text.split("±")[0].strip()
        try:
            x = float(head)
            return x if np.isfinite(x) else np.nan
        except ValueError:
            return np.nan
    try:
        x = float(value)
        if np.isnan(x):
            return np.nan
        return x
    except (TypeError, ValueError):
        return np.nan


def _metric_display_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "—":
            return "—"
        if "±" in text:
            return text
        try:
            x = float(text)
            return f"{x:.3f}" if np.isfinite(x) else "—"
        except ValueError:
            return text
    try:
        x = float(value)
        if not np.isfinite(x):
            return "—"
        return f"{x:.3f}"
    except (TypeError, ValueError):
        return "—"


def _metric_series_as_float(s: pd.Series) -> pd.Series:
    return s.map(_metric_to_float)


def style_metrics_table(
    df: pd.DataFrame,
    metric_cols: Sequence[str] = ("balanced_accuracy", "macro_f1", "accuracy"),
) -> "pd.io.formats.style.Styler":
    """Styler pandas : format décimal + surbrillance max par colonne métrique."""
    present = [c for c in metric_cols if c in df.columns]
    formatters = {
        c: "{:.3f}"
        for c in present
        if pd.api.types.is_numeric_dtype(df[c])
    }
    styler = df.style.format(formatters, na_rep="—")

    def _highlight_max(s: pd.Series) -> List[str]:
        nums = _metric_series_as_float(s)
        if nums.notna().sum() == 0:
            return [""] * len(s)
        best = nums.max()
        return [
            "background-color: #E8F4FD; font-weight: 600"
            if np.isfinite(v) and v == best
            else ""
            for v in nums
        ]

    for col in present:
        styler = styler.apply(_highlight_max, subset=[col])
    return styler


def export_metrics_latex_table(
    df: pd.DataFrame,
    metric_cols: Sequence[str] = ("balanced_accuracy", "macro_f1", "accuracy"),
) -> str:
    """Export LaTeX avec meilleures valeurs en gras."""
    present = [c for c in metric_cols if c in df.columns]
    if not present:
        return ""

    label_cols = [c for c in ("phase", "corpus") if c in df.columns]
    display_df = df.copy()
    for col in present:
        display_df[col] = display_df[col].map(_metric_display_value)

    def _row_label(row: pd.Series) -> str:
        if label_cols:
            return " / ".join(str(row[c]) for c in label_cols)
        return str(row.iloc[0])

    def _winner_indices(values: Sequence[Any]) -> set[int]:
        s = pd.Series([_metric_to_float(v) for v in values], dtype="float64")
        if s.notna().sum() == 0:
            return set()
        best = s.max()
        return set(s.index[s == best].tolist())

    bold_cols: Dict[str, set[int]] = {c: _winner_indices(df[c]) for c in present}
    lines = ["\\begin{tabular}{l" + "r" * len(present) + "}", "\\toprule"]
    header = " & ".join(["run"] + present) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for i, row in display_df.iterrows():
        cells = [_row_label(row)]
        for col in present:
            val = str(row[col])
            if i in bold_cols.get(col, set()):
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def plot_cv_metrics_bars(
    cv_per_fold: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
    filename: str = "cv_per_fold_metrics.png",
    show: bool = True,
) -> Optional[Path]:
    """Barres balanced_accuracy et macro_f1 par fold."""
    if cv_per_fold is None or cv_per_fold.empty:
        return None
    fold_col = "fold" if "fold" in cv_per_fold.columns else "fold_id"
    if fold_col not in cv_per_fold.columns:
        return None

    metrics = [c for c in ("balanced_accuracy", "macro_f1") if c in cv_per_fold.columns]
    if not metrics:
        return None

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]
    style_figure(fig)

    folds = cv_per_fold[fold_col].astype(str).tolist()
    x = np.arange(len(folds))
    for ax, metric in zip(axes, metrics):
        vals = cv_per_fold[metric].astype(float).values
        ax.bar(x, vals, **bar_kwargs())
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {f}" for f in folds])
        ax.set_ylim(0, 1)
        style_axes(ax, title=metric.replace("_", " ").title())
    fig.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_cross_domain_bars(
    cross_domain: pd.DataFrame,
    *,
    test_corpora: Sequence[str] = ("metallurgie", "caou"),
    fig_dir: Optional[Path] = None,
    filename: str = "cross_domain_bars.png",
    show: bool = True,
) -> Optional[Path]:
    """Barres CV BA vs OOD BA par corpus."""
    if cross_domain is None or cross_domain.empty:
        return None
    row = cross_domain.iloc[0]
    labels: List[str] = []
    values: List[float] = []
    if np.isfinite(_metric_to_float(row.get("cv_ba_mean", np.nan))):
        labels.append("CV (BTP)")
        values.append(_metric_to_float(row["cv_ba_mean"]))
    for cid in test_corpora:
        key = f"balanced_accuracy_{cid}"
        if key in row.index and np.isfinite(_metric_to_float(row[key])):
            labels.append(str(cid))
            values.append(_metric_to_float(row[key]))

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    style_figure(fig)
    x = np.arange(len(labels))
    ax.bar(x, values, **bar_kwargs())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    style_axes(ax, title="Balanced accuracy — CV vs OOD")
    fig.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_supervised_macro_ft_train_history(
    history_df: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
    filename: str = "train_history_curves.png",
    show: bool = True,
) -> Optional[Path]:
    """Courbes train_loss et val macro F1 / BA par epoch (CV + fit final), charte SAFER."""
    if history_df.empty or "epoch" not in history_df.columns:
        return None

    apply_matplotlib_brand()

    if "phase" in history_df.columns:
        cv = history_df[history_df["phase"] == "cv"].copy()
        final = history_df[history_df["phase"] == "final"].copy()
    else:
        cv = (
            history_df[history_df.get("fold", -1) >= 0].copy()
            if "fold" in history_df.columns
            else history_df.copy()
        )
        final = (
            history_df[history_df.get("fold", -1) == -1].copy()
            if "fold" in history_df.columns
            else pd.DataFrame()
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    style_figure(fig)

    ax = axes[0]
    if not cv.empty and "train_loss" in cv.columns:
        for _, sub in cv.groupby("fold"):
            ax.plot(sub["epoch"], sub["train_loss"], alpha=0.3, linewidth=1, color=HERO_BLUE)
        mean_loss = cv.groupby("epoch")["train_loss"].mean()
        ax.plot(mean_loss.index, mean_loss.values, color=ACCENT_RED, linewidth=2, label="CV moyenne")
    if not final.empty and "train_loss" in final.columns:
        ax.plot(
            final["epoch"],
            final["train_loss"],
            color=DEEP_BLUE,
            linewidth=2,
            linestyle="--",
            label="fit final (100 % BTP)",
        )
    style_axes(ax, title="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    ax = axes[1]
    val_col = next(
        (c for c in ("val_balanced_accuracy", "val_macro_f1") if c in cv.columns),
        None,
    )
    if not cv.empty and val_col is not None:
        val_cv = cv.dropna(subset=[val_col])
        if not val_cv.empty:
            for _, sub in val_cv.groupby("fold"):
                ax.plot(sub["epoch"], sub[val_col], alpha=0.3, linewidth=1, color=HERO_BLUE)
            grp = val_cv.groupby("epoch")[val_col]
            mean_v = grp.mean()
            std_v = grp.std(ddof=0).fillna(0.0)
            ax.plot(mean_v.index, mean_v.values, color=DEEP_BLUE, linewidth=2, label="CV moyenne")
            ax.fill_between(
                mean_v.index,
                mean_v - std_v,
                mean_v + std_v,
                color=HERO_BLUE,
                alpha=0.15,
                label="±1 écart-type",
            )
    ylab = "Balanced accuracy" if val_col == "val_balanced_accuracy" else "Macro F1"
    style_axes(ax, title=f"Val {ylab} (CV)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylab)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_confusion_matrix_brand(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    title: str = "Matrice de confusion",
    fig_dir: Optional[Path] = None,
    filename: str = "confusion_matrix.png",
    show: bool = True,
    macros: Sequence[str] = MACRO_NAMES,
) -> Optional[Path]:
    """Heatmap confusion matrix (charte SAFER)."""
    from sklearn.metrics import confusion_matrix

    labels = list(macros)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    apply_matplotlib_brand()
    fig, ax = plt.subplots(figsize=(6, 5))
    style_figure(fig)
    im = ax.imshow(cm, cmap=matplotlib_heatmap_cmap(), aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=DEEP_BLUE, fontsize=11)
    style_axes(ax, title=title)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_calibration_histograms(
    pred_df: pd.DataFrame,
    *,
    true_col: str = "true_macro",
    confidence_col: str = "confidence",
    entropy_col: str = "entropy",
    fig_dir: Optional[Path] = None,
    filename: str = "calibration_histograms.png",
    show: bool = True,
    macros: Sequence[str] = MACRO_NAMES,
) -> Optional[Path]:
    """Histogrammes confidence / entropy par macro vraie."""
    if pred_df.empty or true_col not in pred_df.columns:
        return None
    apply_matplotlib_brand()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    style_figure(fig)

    for ax, col, title in zip(
        axes,
        (confidence_col, entropy_col),
        ("Confiance", "Entropie"),
    ):
        if col not in pred_df.columns:
            ax.set_visible(False)
            continue
        for macro in macros:
            sub = pred_df[pred_df[true_col].astype(str) == str(macro)]
            if sub.empty:
                continue
            ax.hist(
                sub[col].astype(float),
                bins=20,
                alpha=0.55,
                label=str(macro),
                color=MACRO_COLOR_HEX.get(str(macro), HERO_BLUE),
                edgecolor="white",
            )
        style_axes(ax, title=title)
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
    fig.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def get_misclassification_sample(
    pred_df: pd.DataFrame,
    *,
    true_col: str = "true_macro",
    pred_col: str = "pred_macro",
    text_col: str = "sentence",
    margin_col: str = "margin",
    n: int = 15,
) -> pd.DataFrame:
    """Échantillon d'erreurs triées par marge faible (incertitude)."""
    if pred_df.empty:
        return pd.DataFrame()
    err = pred_df[pred_df[true_col].astype(str) != pred_df[pred_col].astype(str)].copy()
    if err.empty:
        return err
    if margin_col in err.columns:
        err = err.sort_values(margin_col, ascending=True)
    cols = [c for c in (text_col, true_col, pred_col, "confidence", margin_col) if c in err.columns]
    return err[cols].head(int(n))


def load_raw_backbone_embeddings(
    results_dir: str | Path,
    corpus_id: str,
    *,
    anchor: Optional[Path] = None,
    backbone_emb_csv: Optional[str | Path] = None,
) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], List[str]]:
    """
    Charge h Qwen brut : cache run > CSV config > registre test_corpora.

    Retourne (hidden, metadata_df, missing_paths).
    """
    from safer_core.paths import TEXT_ROOT
    from safer_core.test_corpus import resolve_test_corpus
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    from scgm_text.utils_io import create_doc_id_if_missing

    root = Path(results_dir).resolve()
    repo = Path(anchor or TEXT_ROOT).resolve()
    missing: List[str] = []

    if corpus_id == "btp":
        cache_path = root / "cache" / "backbone_hidden.npy"
        data_csv = repo / "dataset" / "data_btp.csv"
        if cache_path.is_file():
            hidden = np.load(cache_path)
            meta_path = root / "embeddings" / "projected_btp_metadata.csv"
            if meta_path.is_file() and len(hidden) == len(pd.read_csv(meta_path)):
                return np.asarray(hidden, dtype=np.float64), pd.read_csv(meta_path), missing
            if data_csv.is_file():
                meta = create_doc_id_if_missing(pd.read_csv(data_csv))
                if len(hidden) == len(meta):
                    return np.asarray(hidden, dtype=np.float64), meta, missing
                missing.append(
                    f"cache {cache_path} n={len(hidden)} != data_csv n={len(meta)}"
                )
        if backbone_emb_csv:
            emb_path = Path(backbone_emb_csv)
            if not emb_path.is_absolute():
                emb_path = repo / emb_path
        else:
            emb_path = repo / "embeddings" / "Qwen3-Embedding-0.6B_btp.csv"
        if not emb_path.is_file():
            missing.append(str(emb_path))
            return None, None, missing
        if not data_csv.is_file():
            missing.append(str(data_csv))
            return None, None, missing
        meta = create_doc_id_if_missing(pd.read_csv(data_csv))
        slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
        try:
            merged, dim_cols = merge_metadata_with_embeddings(slim, str(emb_path))
        except ValueError as exc:
            missing.append(str(exc))
            return None, None, missing
        return merged[dim_cols].to_numpy(dtype=np.float64), merged, missing

    cache_path = root / "cache" / f"backbone_hidden_{corpus_id}.npy"
    if cache_path.is_file():
        hidden = np.load(cache_path)
        meta_path = root / "embeddings" / f"projected_{corpus_id}_metadata.csv"
        if meta_path.is_file() and len(hidden) == len(pd.read_csv(meta_path)):
            return np.asarray(hidden, dtype=np.float64), pd.read_csv(meta_path), missing

    try:
        spec = resolve_test_corpus(corpus_id, anchor=repo)
    except KeyError as exc:
        missing.append(str(exc))
        return None, None, missing

    if not spec.data_csv.is_file():
        missing.append(str(spec.data_csv))
        return None, None, missing
    if not spec.emb_csv.is_file():
        missing.append(str(spec.emb_csv))
        return None, None, missing

    meta = create_doc_id_if_missing(pd.read_csv(spec.data_csv))
    slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
    try:
        merged, dim_cols = merge_metadata_with_embeddings(slim, str(spec.emb_csv))
    except ValueError as exc:
        missing.append(str(exc))
        return None, None, missing
    return merged[dim_cols].to_numpy(dtype=np.float64), merged, missing


def _load_macro_ft_tokenizer(backbone_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def build_prediction_df(
    checkpoint_dir: str | Path,
    meta_df: pd.DataFrame,
    texts: Sequence[str],
    *,
    label_col: str,
    text_col: str = "sentence",
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 256,
    macros: Sequence[str] = MACRO_NAMES,
    results_dir: Optional[str | Path] = None,
    corpus_id: Optional[str] = None,
    anchor: Optional[Path] = None,
    backbone_emb_csv: Optional[str | Path] = None,
    prefer_hidden_cache: bool = True,
) -> pd.DataFrame:
    """Inférence CE : vraie classe + prédiction + scores.

    Essaie d'abord le cache backbone (``cache/backbone_hidden*.npy``) si le backbone
    est gelé ; sinon, ou si le cache/CSV est absent ou désaligné, repasse par
    ``predict_corpus`` (forward Qwen + checkpoint).
    """
    import torch

    from supervised_macro_ft.checkpoint_io import load_checkpoint, read_checkpoint_config
    from supervised_macro_ft.embedding_cache import predict_from_hidden_matrix
    from supervised_macro_ft.inference import predict_corpus

    ckpt_dir = Path(checkpoint_dir)
    cfg = read_checkpoint_config(ckpt_dir)
    max_len = int(cfg.get("max_seq_length", max_length))
    model = load_checkpoint(ckpt_dir, device=device)
    dev = torch.device(device)

    hidden: Optional[np.ndarray] = None
    frozen_backbone = not bool(cfg.get("backbone_trainable", False))
    if prefer_hidden_cache and frozen_backbone and results_dir is not None and corpus_id is not None:
        hidden, _, missing = load_raw_backbone_embeddings(
            results_dir,
            str(corpus_id),
            anchor=anchor,
            backbone_emb_csv=backbone_emb_csv if corpus_id == "btp" else None,
        )
        if hidden is not None and len(hidden) != len(texts):
            print(
                f"[macro_ft viz] cache hidden n={len(hidden)} != corpus n={len(texts)} "
                f"— fallback predict_corpus ({corpus_id})"
            )
            hidden = None
        elif hidden is None and missing:
            print(
                f"[macro_ft viz] cache/CSV backbone indisponible pour {corpus_id} "
                f"— fallback predict_corpus (forward Qwen)"
            )
            for msg in missing[:2]:
                print(f"  · {msg[:200]}")

    if hidden is not None:
        pred_macro, probs, confidence, margin, entropy = predict_from_hidden_matrix(
            model,
            hidden,
            macros=macros,
            batch_size=batch_size,
            device=dev,
        )
    else:
        tokenizer = _load_macro_ft_tokenizer(str(cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B")))
        pred_macro, probs, confidence, margin, entropy = predict_corpus(
            model,
            tokenizer,
            texts,
            macros=macros,
            max_length=max_len,
            batch_size=batch_size,
            device=dev,
        )
    out = meta_df.copy()
    if text_col not in out.columns and len(texts) == len(out):
        out[text_col] = list(texts)
    out["true_macro"] = out[label_col].astype(str).values
    out["pred_macro"] = pred_macro
    out["confidence"] = confidence
    out["margin"] = margin
    out["entropy"] = entropy
    for i, m in enumerate(macros):
        if i < probs.shape[1]:
            out[f"prob_{m}"] = probs[:, i]
    return out


def _tsne_2d(
    X: np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = len(X)
    perplexity = min(30, max(5, n - 1))
    return TSNE(
        n_components=2,
        random_state=int(seed),
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(X)


def plot_raw_vs_projected_tsne_pair(
    raw_emb: np.ndarray,
    projected_emb: np.ndarray,
    meta_df: pd.DataFrame,
    label_col: str,
    *,
    corpus_name: str = "",
    fig_dir: Optional[Path] = None,
    filename: str = "raw_vs_projected_tsne.png",
    max_points: int = 8000,
    seed: int = 42,
    show: bool = True,
    macros: Sequence[str] = MACRO_NAMES,
) -> Optional[Path]:
    """Figure 1×2 : t-SNE embedding brut vs projeté (même sous-échantillon, vraies classes)."""
    from scgm_text.notebook_viz import sample_projection_indices

    if len(raw_emb) != len(projected_emb) or len(raw_emb) != len(meta_df):
        raise ValueError("raw, projected et metadata doivent avoir la même longueur")

    idx = sample_projection_indices(meta_df, label_col, max_points=max_points, seed=seed)
    raw_sub = np.asarray(raw_emb[idx], dtype=np.float64)
    proj_sub = np.asarray(projected_emb[idx], dtype=np.float64)
    sub_df = meta_df.iloc[idx].reset_index(drop=True)

    apply_matplotlib_brand()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    style_figure(fig)

    for ax, X, subtitle in zip(
        axes,
        (raw_sub, proj_sub),
        ("Embedding brut Qwen", "Embedding projeté ψ(h)"),
    ):
        tsne_xy = _tsne_2d(X, seed=seed)
        for macro in macros:
            mask = sub_df[label_col].astype(str).values == str(macro)
            if not mask.any():
                continue
            ax.scatter(
                tsne_xy[mask, 0],
                tsne_xy[mask, 1],
                s=5,
                alpha=0.45,
                label=str(macro),
                c=MACRO_COLOR_HEX.get(str(macro), HERO_BLUE),
            )
        style_axes(ax, title=f"{subtitle}" + (f" — {corpus_name}" if corpus_name else ""))
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_raw_embeddings_pca_tsne(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    label_col: str,
    *,
    corpus_name: str = "",
    fig_dir: Optional[Path] = None,
    filename: str = "raw_pca_tsne.png",
    max_points: int = 12000,
    seed: int = 42,
    show: bool = True,
    macros: Sequence[str] = MACRO_NAMES,
) -> Optional[Path]:
    """PCA + t-SNE sur embeddings backbone bruts, charte SAFER."""
    from sklearn.decomposition import PCA
    from scgm_text.notebook_viz import sample_projection_indices

    if len(embeddings) != len(meta_df):
        raise ValueError("embeddings et metadata non alignés")

    idx = sample_projection_indices(meta_df, label_col, max_points=max_points, seed=seed)
    sample_x = np.asarray(embeddings[idx], dtype=np.float64)
    sample_df = meta_df.iloc[idx].reset_index(drop=True)
    pca_xy = PCA(n_components=2, random_state=seed).fit_transform(sample_x)
    tsne_xy = _tsne_2d(sample_x, seed=seed)

    apply_matplotlib_brand()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    style_figure(fig)
    for ax, coords, subtitle in zip(
        axes,
        (pca_xy, tsne_xy),
        ("PCA 2D", "t-SNE 2D"),
    ):
        for macro in macros:
            mask = sample_df[label_col].astype(str).values == str(macro)
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=5,
                alpha=0.45,
                label=str(macro),
                c=MACRO_COLOR_HEX.get(str(macro), HERO_BLUE),
            )
        style_axes(ax, title=f"{subtitle} — {corpus_name} (brut Qwen)" if corpus_name else subtitle)
        ax.set_xlabel(f"{subtitle.split()[0]} 1")
        ax.set_ylabel(f"{subtitle.split()[0]} 2")
        ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_tuning_grid_bars(
    grid_df: pd.DataFrame,
    *,
    fig_dir: Optional[Path] = None,
    filename: str = "tuning_top_combos.png",
    top_n: int = 12,
    show: bool = True,
) -> Optional[Path]:
    """Barres horizontales des meilleurs combos tuning."""
    if grid_df is None or grid_df.empty or "selection_score" not in grid_df.columns:
        return None
    top = grid_df.sort_values("selection_score", ascending=False).head(int(top_n))
    id_col = "combo_id" if "combo_id" in top.columns else top.columns[0]

    apply_matplotlib_brand()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    style_figure(fig)
    y = np.arange(len(top))
    ax.barh(y, top["selection_score"].astype(float), **bar_kwargs())
    ax.set_yticks(y)
    ax.set_yticklabels(top[id_col].astype(str))
    style_axes(ax, title="Grille tuning — top combos")
    ax.set_xlabel("selection_score (balanced accuracy CV)")
    fig.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_tsne_true_vs_pred_brand(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    *,
    title: str = "",
    fig_dir: Optional[Path] = None,
    filename: str = "tsne_true_vs_pred.png",
    max_points: int = 8000,
    seed: int = 42,
    show: bool = True,
    macros: Sequence[str] = MACRO_NAMES,
) -> Optional[Path]:
    """t-SNE 2D : vraie classe vs prédite (mêmes coordonnées), charte SAFER."""
    h = np.asarray(embeddings, dtype=np.float64)
    if len(h) != len(df):
        raise ValueError(f"embeddings ({len(h)}) et metadata ({len(df)}) non alignés")
    if true_col not in df.columns or pred_col not in df.columns:
        raise KeyError(f"Colonnes manquantes : {true_col!r}, {pred_col!r}")

    from scgm_text.notebook_viz import sample_projection_indices

    idx = sample_projection_indices(df, true_col, max_points=max_points, seed=seed)
    X = h[idx]
    sub = df.iloc[idx].reset_index(drop=True)
    tsne_xy = _tsne_2d(X, seed=seed)

    apply_matplotlib_brand()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    style_figure(fig)
    for ax, col, subtitle in zip(axes, (true_col, pred_col), ("Vraie classe", "Classe prédite")):
        for macro in macros:
            mask = sub[col].astype(str).values == str(macro)
            if not mask.any():
                continue
            ax.scatter(
                tsne_xy[mask, 0],
                tsne_xy[mask, 1],
                s=5,
                alpha=0.45,
                label=str(macro),
                c=MACRO_COLOR_HEX.get(str(macro), HERO_BLUE),
            )
        style_axes(ax, title=f"{subtitle}" + (f" — {title}" if title else ""))
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()

    out_path: Optional[Path] = None
    if fig_dir is not None:
        out = Path(fig_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path
