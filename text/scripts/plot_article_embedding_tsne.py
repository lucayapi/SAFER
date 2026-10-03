"""Create a balanced, four-panel t-SNE candidate for the article appendix.

The plot compares Qwen3 / Construction representations for the same labelled
units from Construction and the Company corpus. Each method is projected
independently; coordinates must not be compared across panels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
TEXT = ROOT / "text"
SEED = 2026
SAMPLE_PER_ROLE_PER_CORPUS = 203
ROLES = ["A0", "A1", "B", "C"]
CORPORA = {
    "Construction": {
        "id": "btp",
        "metadata": TEXT / "output/supcon/macro_ft_tuning/combos/full_yes/embeddings/projected_btp_metadata.csv",
        "frozen": TEXT / "embeddings/qwen3/btp.csv",
        "dataset": TEXT / "dataset/data_btp.csv",
    },
    "Company corpus": {
        "id": "nicollin",
        "metadata": TEXT / "output/supcon/macro_ft_tuning/combos/full_yes/embeddings/projected_nicollin_metadata.csv",
        "frozen": TEXT / "embeddings/qwen3/nicollin.csv",
        "dataset": TEXT / "dataset/data_nicollin.csv",
    },
}
METHODS = {
    "Frozen embeddings": None,
    "Cross-entropy": TEXT / "output/article_visualization/cross_entropy_qwen3_construction_tsne/embeddings",
    "Supervised contrastive learning": TEXT / "output/supcon/macro_ft_tuning/combos/full_yes/embeddings",
    "SoftTriple": TEXT / "output/softtriple/macro_ft_tuning/combos/full_yes/embeddings",
}
OUTPUT = TEXT / "output/article_visualization/candidate_embedding_geometry"
ROLE_COLORS = {
    "A0": "#4477AA",
    "A1": "#EE9944",
    "B": "#228833",
    "C": "#AA3377",
}
CORPUS_MARKERS = {"Construction": "o", "Company corpus": "^"}
METHOD_FILES = {
    "Cross-entropy": {"btp": "projected_btp.npy", "nicollin": "projected_nicollin.npy"},
    "Supervised contrastive learning": {"btp": "projected_btp.npy", "nicollin": "projected_nicollin.npy"},
    "SoftTriple": {"btp": "projected_btp.npy", "nicollin": "projected_nicollin.npy"},
}


def load_canonical_metadata(corpus_name: str) -> pd.DataFrame:
    spec = CORPORA[corpus_name]
    meta = pd.read_csv(spec["metadata"], usecols=["accident_id", "doc_id", "sentence", "pred_label"])
    if meta["doc_id"].duplicated().any():
        raise ValueError(f"{corpus_name}: duplicate doc_id in representation metadata")
    if not set(meta["pred_label"].dropna().unique()).issubset(ROLES):
        raise ValueError(f"{corpus_name}: unexpected role value")

    # The exported labels and text must agree with the source corpus after the
    # same labelled-unit filtering and sequential doc_id assignment.
    data = pd.read_csv(
        spec["dataset"],
        usecols=["accident_id", "fact_id", "sentence", "pred_label", "pred_ok"],
    )
    data.insert(1, "doc_id", np.arange(1, len(data) + 1, dtype=np.int64))
    pred_ok = data["pred_ok"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "t"})
    data = data.loc[pred_ok & data["pred_label"].isin(ROLES)].reset_index(drop=True)
    check = meta.merge(
        data[["accident_id", "doc_id", "sentence", "pred_label"]],
        on="doc_id",
        suffixes=("_export", "_source"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not check["_merge"].eq("both").all():
        raise ValueError(f"{corpus_name}: metadata doc_id coverage differs from the source corpus")
    for col in ("accident_id", "sentence", "pred_label"):
        if not check[f"{col}_export"].astype(str).equals(check[f"{col}_source"].astype(str)):
            raise ValueError(f"{corpus_name}: exported {col} does not align with source rows")
    return meta


def sample_metadata(metadata: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    samples = {}
    for corpus, meta in metadata.items():
        chunks = []
        for role in ROLES:
            eligible = meta.loc[meta["pred_label"].eq(role)]
            if len(eligible) < SAMPLE_PER_ROLE_PER_CORPUS:
                raise ValueError(
                    f"{corpus}/{role}: need {SAMPLE_PER_ROLE_PER_CORPUS} units, found {len(eligible)}"
                )
            take = rng.choice(eligible.index.to_numpy(), SAMPLE_PER_ROLE_PER_CORPUS, replace=False)
            chunks.append(meta.loc[take])
        sample = pd.concat(chunks, ignore_index=True).sort_values("doc_id").reset_index(drop=True)
        sample["corpus"] = corpus
        samples[corpus] = sample
    return samples


def load_selected_csv_vectors(path: Path, selected_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    header = pd.read_csv(path, nrows=0)
    dim_cols = [c for c in header.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError(f"No embedding columns found in {path}")
    pieces = []
    for chunk in pd.read_csv(path, usecols=["doc_id", *dim_cols], chunksize=2048):
        part = chunk.loc[chunk["doc_id"].isin(selected_ids)]
        if len(part):
            pieces.append(part)
    if not pieces:
        raise ValueError(f"No selected doc_id values were found in {path}")
    selected = pd.concat(pieces, ignore_index=True).sort_values("doc_id")
    if selected["doc_id"].duplicated().any() or set(selected["doc_id"].astype(int)) != selected_ids:
        raise ValueError(f"Selected embedding IDs do not align in {path}")
    return selected["doc_id"].to_numpy(dtype=np.int64), selected[dim_cols].to_numpy(dtype=np.float32)


def load_selected_model_vectors(method: str, corpus: str, meta: pd.DataFrame) -> np.ndarray:
    corpus_id = CORPORA[corpus]["id"]
    root = METHODS[method]
    if root is None:
        ids, vectors = load_selected_csv_vectors(
            CORPORA[corpus]["frozen"], set(meta["doc_id"].astype(int))
        )
        positions = pd.Series(np.arange(len(ids)), index=ids.astype(int))
        return vectors[positions.loc[meta["doc_id"].astype(int)].to_numpy()]

    array_path = root / METHOD_FILES[method][corpus_id]
    metadata_path = root / f"projected_{corpus_id}_metadata.csv"
    exported_meta = pd.read_csv(metadata_path, usecols=["accident_id", "doc_id", "sentence", "pred_label"])
    vectors = np.load(array_path, mmap_mode="r")
    if len(vectors) != len(exported_meta):
        raise ValueError(f"{method}/{corpus}: vector and metadata row counts differ")
    canonical = meta.sort_values("doc_id").reset_index(drop=True)
    exported_meta = exported_meta.sort_values("doc_id").reset_index(drop=True)
    for col in ("accident_id", "doc_id", "sentence", "pred_label"):
        if not canonical[col].astype(str).equals(exported_meta[col].astype(str)):
            raise ValueError(f"{method}/{corpus}: {col} differs from the canonical unit order")
    positions = pd.Series(np.arange(len(exported_meta)), index=exported_meta["doc_id"].astype(int))
    selected_positions = positions.loc[meta["doc_id"].astype(int)].to_numpy()
    return np.asarray(vectors[selected_positions], dtype=np.float32)


def build_figure(sampled: dict[str, pd.DataFrame], metadata: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    plot_meta = pd.concat(sampled.values(), ignore_index=True)
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.75), constrained_layout=False)
    axes = axes.ravel()
    plt.rcParams.update({"font.family": "serif", "font.size": 8.4, "axes.labelsize": 8.4})

    for ax, (method, _) in zip(axes, METHODS.items()):
        corpus_arrays = []
        corpus_frames = []
        for corpus, frame in sampled.items():
            canonical = metadata[corpus]
            vectors = load_selected_model_vectors(method, corpus, canonical)
            ids = frame["doc_id"].astype(int).to_numpy()
            meta_positions = pd.Series(np.arange(len(canonical)), index=canonical["doc_id"].astype(int))
            selected = np.asarray(vectors[meta_positions.loc[ids].to_numpy()], dtype=np.float32)
            if selected.shape[0] != len(frame):
                raise ValueError(f"{method}/{corpus}: selected row count is not aligned")
            corpus_arrays.append(selected)
            corpus_frames.append(frame.reset_index(drop=True))

        x = np.vstack(corpus_arrays)
        panel_meta = pd.concat(corpus_frames, ignore_index=True)
        x = normalize(x, norm="l2")
        coords = TSNE(
            n_components=2,
            perplexity=30,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            max_iter=1200,
            random_state=SEED,
            method="barnes_hut",
        ).fit_transform(x)

        for role in ROLES:
            for corpus in CORPUS_MARKERS:
                mask = panel_meta["pred_label"].eq(role).to_numpy() & panel_meta["corpus"].eq(corpus).to_numpy()
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=9, alpha=0.54, linewidths=0.25,
                    c=ROLE_COLORS[role], marker=CORPUS_MARKERS[corpus],
                    edgecolors="#333333" if corpus == "Company corpus" else "none",
                    rasterized=True,
                )

        role_silhouette = float(silhouette_score(x, panel_meta["pred_label"], metric="cosine"))
        domain_silhouette = float(silhouette_score(x, panel_meta["corpus"], metric="cosine"))
        results.append({
            "method": method,
            "n_per_role_per_corpus": SAMPLE_PER_ROLE_PER_CORPUS,
            "n_total": len(panel_meta),
            "embedding_dimension": int(x.shape[1]),
            "role_silhouette_cosine": role_silhouette,
            "corpus_silhouette_cosine": domain_silhouette,
        })
        panel_letter = chr(ord("a") + len(results) - 1)
        ax.set_title(f"({panel_letter}) {method}", loc="left", fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#aaaaaa")
        ax.spines[["bottom", "left"]].set_linewidth(0.55)

    from matplotlib.lines import Line2D

    role_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=ROLE_COLORS[r], markeredgecolor="none", label=r)
        for r in ROLES
    ]
    corpus_handles = [
        Line2D([0], [0], marker=CORPUS_MARKERS[c], linestyle="", markersize=5,
               markerfacecolor="#777777", markeredgecolor="#333333" if c == "Company corpus" else "none",
               label=c)
        for c in CORPUS_MARKERS
    ]
    fig.legend(handles=role_handles, title="Annotated role", loc="lower center",
               bbox_to_anchor=(0.43, 0.025), ncol=4, frameon=False,
               columnspacing=1.2, handletextpad=0.35, title_fontsize=8, fontsize=8)
    fig.legend(handles=corpus_handles, title="Corpus", loc="lower center",
               bbox_to_anchor=(0.75, 0.025), ncol=2, frameon=False,
               columnspacing=1.0, handletextpad=0.35, title_fontsize=8, fontsize=8)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.97, bottom=0.15, hspace=0.22, wspace=0.14)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "embedding_structure_tsne_candidate.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "embedding_structure_tsne_candidate.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(results)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {name: load_canonical_metadata(name) for name in CORPORA}
    # Verify there are exactly the same accepted units for every exported method.
    for corpus, frame in metadata.items():
        print(f"{corpus}: {len(frame)} labelled units; role counts={frame['pred_label'].value_counts().to_dict()}")
    sampled = sample_metadata(metadata)
    manifest = pd.concat(sampled.values(), ignore_index=True)
    manifest.to_csv(OUTPUT / "tsne_sample_manifest.csv", index=False)
    summary = build_figure(sampled, metadata)
    summary.to_csv(OUTPUT / "embedding_geometry_summary.csv", index=False)
    (OUTPUT / "figure_notes.json").write_text(
        json.dumps({
            "question": "Do semantic roles and corpus identity form visible local structure in the Qwen3/Construction representations?",
            "sampling": f"{SAMPLE_PER_ROLE_PER_CORPUS} annotated units per role and corpus, shared across methods",
            "t_sne": {"metric": "cosine after row L2 normalization", "perplexity": 30, "seed": SEED, "iterations": 1200},
            "warning": "Each panel is independently fitted. Distances and orientations across panels are not comparable. The sample is role-balanced and does not reflect corpus prevalence.",
            "methods": list(METHODS),
            "corpora": list(CORPORA),
        }, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved candidate figure in {OUTPUT}")


if __name__ == "__main__":
    main()
