"""Chargement des exports MALT et construction du tableau de topics MALT."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .ctfidf import top_words_ctfidf
from .paths import find_repo_root, resolve_repo_path
from .topic_cleaning import clean_top_words, preprocess_for_ctfidf

REQUIRED_EXPORT_FILES = (
    "metadata_with_malt_predictions.csv",
    "target_projected_source.npy",
    "target_projected_adapted.npy",
    "p0_y_target.npy",
    "pt_y_target.npy",
    "pt_z_target.npy",
    "pt_y_given_z.npy",
    "z_assignments_target.csv",
)

MACRO_NAMES = ("A0", "A1", "B", "C")


def check_malt_export_files(exports_dir: str | Path, *, repo_root: Path | None = None) -> None:
    d = resolve_repo_path(exports_dir, repo_root)
    missing = [f for f in REQUIRED_EXPORT_FILES if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Exports MALT incomplets dans "
            f"{d.resolve()}.\nFichiers manquants :\n" + "\n".join(f"  - {m}" for m in missing)
        )


def _argmax_macro_row(row: np.ndarray, id2label: dict[int, str]) -> str:
    return id2label[int(np.argmax(row))]


def enrich_metadata_from_numpy(
    meta: pd.DataFrame,
    exports_dir: Path,
) -> pd.DataFrame:
    """Reconstruit colonnes manquantes depuis les .npy (alignement ligne = ligne)."""
    out = meta.copy()
    n = len(out)
    id2label = {0: "A0", 1: "A1", 2: "B", 3: "C"}

    p0 = np.load(exports_dir / "p0_y_target.npy")
    if p0.shape[0] != n:
        raise ValueError(f"p0_y_target.npy ({p0.shape[0]} lignes) ≠ metadata ({n}).")
    for i, name in enumerate(MACRO_NAMES):
        col = f"p0_{name}"
        if col not in out.columns:
            out[col] = p0[:, i]
    if "p0_macro_name" not in out.columns:
        out["p0_macro_id"] = np.argmax(p0, axis=1)
        out["p0_macro_name"] = [_argmax_macro_row(p0[j], id2label) for j in range(n)]

    pt = np.load(exports_dir / "pt_y_target.npy")
    if pt.shape[0] != n:
        raise ValueError(f"pt_y_target.npy ({pt.shape[0]} lignes) ≠ metadata ({n}).")
    for i, name in enumerate(MACRO_NAMES):
        col = f"pt_{name}"
        if col not in out.columns:
            out[col] = pt[:, i]
    if "pt_macro_name" not in out.columns:
        out["pt_macro_id"] = np.argmax(pt, axis=1)
        out["pt_macro_name"] = [_argmax_macro_row(pt[j], id2label) for j in range(n)]

    pz = np.load(exports_dir / "pt_z_target.npy")
    if pz.shape[0] != n:
        raise ValueError(f"pt_z_target.npy ({pz.shape[0]} lignes) ≠ metadata ({n}).")
    if "z_hat" not in out.columns:
        out["z_hat"] = np.argmax(pz, axis=1)
    if "z_confidence" not in out.columns:
        out["z_confidence"] = np.max(pz, axis=1)

    prob_y_z = np.load(exports_dir / "pt_y_given_z.npy")
    if "z_dominant_macro" not in out.columns:
        z_hat_arr = out["z_hat"].to_numpy(dtype=int)
        id2 = {0: "A0", 1: "A1", 2: "B", 3: "C"}
        out["z_dominant_macro"] = [_argmax_macro_row(prob_y_z[int(z)], id2) for z in z_hat_arr]

    return out


def _rank_sentences_in_cluster_by_pz(
    docs: Sequence[str],
    idx: np.ndarray,
    pz: np.ndarray,
    z_id: int,
    top_k: int,
) -> List[str]:
    scores = pz[idx, z_id]
    order_local = np.argsort(-scores)[:top_k]
    return [str(docs[idx[j]]) for j in order_local]


def _rank_sentences_centroid(
    sentences: Sequence[str],
    emb: np.ndarray,
    nu_z: np.ndarray,
    top_k: int,
) -> List[str]:
    c = nu_z.reshape(1, -1)
    dists = np.linalg.norm(emb - c, axis=1)
    order = np.argsort(dists)[:top_k]
    return [str(sentences[i]) for i in order]


def build_malt_topics_df(
    meta: pd.DataFrame,
    docs: Sequence[str],
    projected_adapted: np.ndarray,
    pz: np.ndarray,
    prob_y_z: np.ndarray,
    nu: np.ndarray | None,
    stopwords_domain: Set[str],
    text_col: str,
    n_top_words: int,
    n_representative_sentences: int,
    z_col: str = "z_hat",
) -> pd.DataFrame:
    z_hat = meta[z_col].to_numpy(dtype=int)
    z_conf = meta["z_confidence"].to_numpy(dtype=float) if "z_confidence" in meta.columns else None
    rows: List[dict] = []
    n_total = max(1, len(docs))

    active_z = sorted(int(z) for z in np.unique(z_hat))
    for z_id in active_z:
        mask = z_hat == z_id
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        sents = [str(docs[i]) for i in idx]
        sub_emb = projected_adapted[mask]
        id2 = {0: "A0", 1: "A1", 2: "B", 3: "C"}
        macro = _argmax_macro_row(prob_y_z[z_id], id2)

        tw_lists = top_words_ctfidf([sents], n_words=n_top_words, preprocessor=preprocess_for_ctfidf)
        raw = tw_lists[0] if tw_lists else []
        top_w = clean_top_words(raw, stopwords_domain)[:n_top_words]

        if pz is not None and pz.shape[1] > z_id:
            top_s = _rank_sentences_in_cluster_by_pz(docs, idx, pz, z_id, n_representative_sentences)
        elif nu is not None and z_id < nu.shape[0]:
            top_s = _rank_sentences_centroid(sents, sub_emb, nu[z_id], n_representative_sentences)
        else:
            top_s = sents[:n_representative_sentences]

        mean_conf = float(np.mean(z_conf[mask])) if z_conf is not None else float(np.mean(np.max(pz[mask], axis=1)))

        n_d = int(mask.sum())
        rows.append(
            {
                "method": "MALT adapted + c-TF-IDF",
                "topic_id": str(z_id),
                "macro": macro,
                "n_docs": n_d,
                "coverage_docs": n_d / n_total,
                "top_words": " ".join(top_w),
                "top_sentences": " || ".join(top_s),
                "mean_confidence": mean_conf,
                "source": "MALT adapted",
            }
        )

    return pd.DataFrame(rows)


def load_embeddings_and_metadata(
    exports_dir: str | Path,
    text_col: str,
    repo_root: Path | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    d = resolve_repo_path(exports_dir, repo_root)
    check_malt_export_files(d)
    meta = pd.read_csv(d / "metadata_with_malt_predictions.csv")
    meta = enrich_metadata_from_numpy(meta, d)
    emb_src = np.load(d / "target_projected_source.npy")
    emb_adapt = np.load(d / "target_projected_adapted.npy")
    if len(meta) != emb_src.shape[0] or len(meta) != emb_adapt.shape[0]:
        raise ValueError(
            f"Alignement : metadata={len(meta)}, source_emb={emb_src.shape[0]}, adapted_emb={emb_adapt.shape[0]}"
        )
    extras = {
        "p0": np.load(d / "p0_y_target.npy"),
        "pt": np.load(d / "pt_y_target.npy"),
        "pz": np.load(d / "pt_z_target.npy"),
        "prob_y_z": np.load(d / "pt_y_given_z.npy"),
    }
    nu_path = d / "nu_target.npy"
    extras["nu"] = np.load(nu_path) if nu_path.is_file() else None
    return meta, emb_src, emb_adapt, extras


def build_topics_by_macro_qualitative(
    malt_df: pd.DataFrame,
    bertopic_df: pd.DataFrame,
    kmeans_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []
    for macro in MACRO_NAMES:
        for label, df in (
            ("MALT adapted + c-TF-IDF", malt_df),
            ("BERTopic intra-macro via p0", bertopic_df),
            ("KMeans intra-macro + c-TF-IDF", kmeans_df),
        ):
            if df is None or df.empty:
                continue
            sub = df.loc[df["macro"].astype(str) == macro].sort_values("n_docs", ascending=False)
            for _, r in sub.iterrows():
                sents = [s.strip() for s in str(r.get("top_sentences", "")).split(" || ")]
                sents = [s for s in sents if s][:3]
                while len(sents) < 3:
                    sents.append("")
                rows.append(
                    {
                        "macro": macro,
                        "method": label,
                        "topic_id": r.get("topic_id", ""),
                        "n_docs": int(r.get("n_docs", 0)),
                        "top_words": r.get("top_words", ""),
                        "top_sentence_1": sents[0],
                        "top_sentence_2": sents[1],
                        "top_sentence_3": sents[2],
                    }
                )
    return pd.DataFrame(rows)


def write_topic_comparison_report_md(
    path: str | Path,
    comparison_metrics_df: pd.DataFrame,
    stopwords_path: str,
    n_docs: int,
    figure_dir: str,
) -> None:
    from datetime import date

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = comparison_metrics_df.copy()
    lines = [
        f"# Rapport de comparaison thématique ({date.today().isoformat()})",
        "",
        f"- Nombre de documents (texte valide) : **{n_docs}**",
        "- Méthodes : MALT adapté + c-TF-IDF ; BERTopic intra-macro via p0 ; KMeans intra-macro + c-TF-IDF",
        f"- Stopwords métier : `{stopwords_path}`",
        "",
        "## Métriques principales",
        "",
    ]
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    def best_on(metric: str, higher: bool = True) -> str:
        if df.empty or metric not in df.columns:
            return "n/a"
        s = df[metric].astype(float)
        if s.isna().all():
            return "n/a"
        idx = int(s.idxmax()) if higher else int(s.idxmin())
        return str(df.loc[idx, "method"])

    lines.extend(
        [
            "## Meilleures méthodes (automatique)",
            "",
            f"- Meilleur **C_v** : {best_on('cv', True)}",
            f"- Meilleur **NPMI** : {best_on('npmi', True)}",
            f"- Meilleure **Topic Diversity** : {best_on('topic_diversity', True)}",
            f"- Meilleure **Coverage** : {best_on('coverage', True)}",
            "",
            "## Commentaires automatiques",
            "",
        ]
    )
    malt = df.loc[df["method"].str.contains("MALT", na=False)]
    bert = df.loc[df["method"].str.contains("BERTopic", na=False)]
    km = df.loc[df["method"].str.contains("KMeans", na=False)]
    if not malt.empty and not df.empty:
        best_cv = best_on("cv", True)
        if "MALT" in best_cv:
            lines.append("- MALT obtient la meilleure cohérence (C_v) parmi les méthodes comparées.")
    if not bert.empty:
        best_div = best_on("topic_diversity", True)
        if "BERTopic" in best_div:
            lines.append("- BERTopic intra-macro présente la meilleure diversité lexicale des top words.")
    if not km.empty:
        best_cov = best_on("coverage", True)
        if "KMeans" in best_cov:
            lines.append("- KMeans intra-macro maximise la couverture documentaire (hors outliers BERTopic).")
    low_nt = df.loc[df["n_topics"] < 5, "method"]
    if len(low_nt):
        lines.append(f"- Attention : peu de topics pour : {', '.join(low_nt.astype(str).tolist())}.")
    high_red = df.loc[df["redundancy"] > 0.85, "method"]
    if len(high_red):
        lines.append(f"- Redondance élevée (>0.85) pour : {', '.join(high_red.astype(str).tolist())}.")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Dossier : `{figure_dir}`",
            "- Fichiers : cv_by_method.png, npmi_by_method.png, topic_diversity_by_method.png, redundancy_by_method.png, coverage_by_method.png, quality_bubble_plot.png, n_topics_by_method.png, topic_size_boxplot.png, macro_topic_count_heatmap.png",
            "",
            "## Limites",
            "",
            "- C_v et NPMI sont des proxys automatiques ; ils ne remplacent pas une validation experte.",
            "- Les pseudo-macros p0/pt sont induites par le modèle source / MALT, pas une vérité terrain annotée.",
            "",
        ]
    )
    p.write_text("\n".join(lines), encoding="utf-8")

