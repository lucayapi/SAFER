"""Reproducible, prediction-only transfer-error diagnostics.

Run from text/: python scripts/build_transfer_error_diagnostics.py
No model is fitted. Frozen embeddings are used only for source-neighbour
diagnostics and are never treated as adapted-run representations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replication.runner import load_replication_config, training_seeds
from scgm_text.data_metadata import load_filtered_metadata

ROLES = ("A0", "A1", "B", "C")
METHODS = ("cross_entropy", "supcon", "softtriple")
BACKBONES = ("qwen3", "multilingual_e5_large", "camembert_base")
SOURCES = ("btp", "metallurgie")
CORPORA = ("btp", "metallurgie", "caou", "nicollin")
DATA_PATHS = {corpus: ROOT / "dataset" / f"data_{corpus}.csv" for corpus in CORPORA}
RECIPES = {
    "backbone_source_factorial": ROOT / "output/replication_recipes/backbone_source_factorial.yaml",
    "camembert_source_factorial": ROOT / "output/replication_recipes/camembert_source_factorial.yaml",
    "camembert_metallurgie_repair": ROOT / "output/replication_recipes/camembert_metallurgie_repair.yaml",
}
REQUIRED = {"accident_id", "fact_id", "true_macro", "pred_macro", "confidence"}


def model_location(backbone: str, source: str, method: str) -> tuple[Path, Path, str]:
    if backbone == "camembert_base":
        campaign = "camembert_metallurgie_repair" if source == "metallurgie" else "camembert_source_factorial"
        stem = "camembert"
    else:
        campaign = "backbone_source_factorial"
        stem = backbone
    return RECIPES[campaign], ROOT / "output" / campaign, f"{stem}_{source}_{method}"


def key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["accident_id", "fact_id"]])


def canonical_metadata() -> dict[str, pd.DataFrame]:
    result = {}
    for corpus, path in DATA_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = load_filtered_metadata(str(path), text_col="sentence")
        for column in ("accident_id", "fact_id"):
            frame[column] = frame[column].astype(str)
        frame["pred_label"] = frame["pred_label"].astype(str)
        if frame.duplicated(["accident_id", "fact_id"]).any():
            raise ValueError(f"Duplicate accident_id/fact_id in {path}")
        if not set(frame.pred_label).issubset(ROLES):
            raise ValueError(f"Unexpected roles in {path}")
        result[corpus] = frame.set_index(["accident_id", "fact_id"], verify_integrity=True).sort_index()
    return result


def read_predictions(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=lambda col: col in REQUIRED,
                        dtype={"accident_id": str, "fact_id": str})
    if REQUIRED - set(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(REQUIRED - set(frame.columns))}")
    if frame.duplicated(["accident_id", "fact_id"]).any():
        raise ValueError(f"Duplicate accident_id/fact_id in {path}")
    frame = frame.set_index(["accident_id", "fact_id"], verify_integrity=True).sort_index()
    if not frame.index.equals(expected.index):
        raise ValueError(f"Unit keys do not match prepared corpus in {path}")
    if not np.array_equal(frame.true_macro.to_numpy(dtype=str), expected.pred_label.to_numpy(dtype=str)):
        raise ValueError(f"True labels do not match prepared corpus in {path}")
    if not set(frame.pred_macro.astype(str)).issubset(ROLES):
        raise ValueError(f"Unknown predicted role in {path}")
    if not np.isfinite(frame.confidence.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite confidence in {path}")
    return frame


def recorded_ba(run_dir: Path, target: str) -> float:
    path = run_dir / "metrics" / f"metrics_classification_test_{target}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 1 or "balanced_accuracy" not in frame:
        raise ValueError(f"Invalid recorded BA in {path}")
    return float(frame.balanced_accuracy.iloc[0])


def prepare_campaigns() -> tuple[dict, list[int]]:
    configs = {key: load_replication_config(path) for key, path in RECIPES.items()}
    seeds = training_seeds(configs["backbone_source_factorial"])
    if len(seeds) != 5 or any(training_seeds(cfg) != seeds for cfg in configs.values()):
        raise ValueError("Campaigns do not share the expected five training seeds")
    return configs, seeds


def bootstrap_overlap(errors: np.ndarray, groups: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """One accident resample shared by all five seeds and three methods."""
    codes, ids = pd.factorize(groups.astype(str), sort=True)
    if len(ids) == 0:
        raise ValueError("Empty target corpus")
    # Columns: denominator, three-way error count per seed, any error per seed,
    # and number of units with three-way errors in every seed.
    all_wrong = (errors.sum(axis=1) == 3).astype(np.int32)
    any_wrong = (errors.sum(axis=1) > 0).astype(np.int32)
    persistent = (all_wrong.sum(axis=0) == errors.shape[0]).astype(np.int32)
    stats = np.zeros((len(ids), 2 + 2 * errors.shape[0]), dtype=np.int32)
    np.add.at(stats[:, 0], codes, 1)
    np.add.at(stats[:, 1], codes, persistent)
    for seed_index in range(errors.shape[0]):
        np.add.at(stats[:, 2 + seed_index], codes, all_wrong[seed_index])
        np.add.at(stats[:, 2 + errors.shape[0] + seed_index], codes, any_wrong[seed_index])
    rng = np.random.default_rng(seed)
    draws_common = np.empty(n_boot)
    draws_conditional = np.empty(n_boot)
    draws_persistent = np.empty(n_boot)
    n_groups = len(ids)
    p = np.full(n_groups, 1.0 / n_groups)
    for start in range(0, n_boot, 128):
        stop = min(start + 128, n_boot)
        weights = rng.multinomial(n_groups, p, size=stop - start)
        totals = weights @ stats
        total_units = totals[:, 0]
        common = totals[:, 2:2 + errors.shape[0]].mean(axis=1)
        any_count = totals[:, 2 + errors.shape[0]:].mean(axis=1)
        draws_common[start:stop] = common / total_units
        draws_conditional[start:stop] = np.divide(common, any_count, out=np.zeros_like(common), where=any_count > 0)
        draws_persistent[start:stop] = totals[:, 1] / total_units
    return {
        "common_unit_ci_low": float(np.quantile(draws_common, 0.025)),
        "common_unit_ci_high": float(np.quantile(draws_common, 0.975)),
        "common_among_any_ci_low": float(np.quantile(draws_conditional, 0.025)),
        "common_among_any_ci_high": float(np.quantile(draws_conditional, 0.975)),
        "persistent_unit_ci_low": float(np.quantile(draws_persistent, 0.025)),
        "persistent_unit_ci_high": float(np.quantile(draws_persistent, 0.975)),
    }


def factorial_ss(frame: pd.DataFrame, response: str, factors: tuple[str, ...]) -> pd.DataFrame:
    """Orthogonal descriptive sums of squares for a fully balanced design."""
    levels = [frame[factor].nunique() for factor in factors]
    expected_cells = math.prod(levels)
    sizes = frame.groupby(list(factors), observed=True).size()
    if len(sizes) != expected_cells or sizes.nunique() != 1:
        raise ValueError(f"Unbalanced factorial design: {factors}")
    grand_mean = float(frame[response].mean())
    effects: dict[tuple[str, ...], pd.Series] = {}
    results = []
    for size in range(1, len(factors) + 1):
        for subset in itertools.combinations(factors, size):
            means = frame.groupby(list(subset), observed=True)[response].mean()
            effect = means.copy() - grand_mean
            for smaller_size in range(1, size):
                for smaller in itertools.combinations(subset, smaller_size):
                    previous = effects[smaller]
                    if len(smaller) == 1:
                        values = [previous.loc[key[subset.index(smaller[0])]] for key in means.index]
                    else:
                        values = [previous.loc[tuple(key[subset.index(name)] for name in smaller)] for key in means.index]
                    effect -= np.asarray(values)
            effects[subset] = effect
            if len(subset) == 1:
                individual = frame[subset[0]].map(effect).to_numpy(dtype=float)
            else:
                individual = np.array([effect.loc[tuple(row)] for row in frame[list(subset)].itertuples(index=False, name=None)])
            results.append({"term": " × ".join(subset), "order": size, "sum_squares": float(np.square(individual).sum())})
    total = float(np.square(frame[response].to_numpy() - grand_mean).sum())
    remainder = max(0.0, total - sum(row["sum_squares"] for row in results))
    results.append({"term": "seed/run residual", "order": 0, "sum_squares": remainder})
    out = pd.DataFrame(results)
    out["share_total_variation"] = out.sum_squares / total if total > 0 else np.nan
    out["total_sum_squares"] = total
    return out.sort_values("sum_squares", ascending=False, ignore_index=True)


def validate_embedding(corpus: str, backbone: str, metadata: pd.DataFrame) -> np.ndarray:
    path = ROOT / "embeddings" / backbone / f"{corpus}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    emb = pd.read_csv(path)
    expected_cols = [f"dim_{i:04d}" for i in range(1, 1025)]
    if list(emb.columns) != ["doc_id", *expected_cols]:
        raise ValueError(f"Wrong embedding schema: {path}")
    if len(metadata) != len(emb):
        raise ValueError(f"Wrong embedding row count: {path}")
    # Exports follow the original filtered CSV order, while the analysis uses
    # accident_id/fact_id order. Reorder by unique doc_id through metadata.
    ids = metadata.doc_id.to_numpy(dtype=np.int64)
    exported_ids = pd.Index(emb.doc_id.to_numpy(dtype=np.int64))
    if exported_ids.has_duplicates or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Non-unique doc_id in {path} or prepared metadata")
    positions = exported_ids.get_indexer(ids)
    if np.any(positions < 0) or not np.array_equal(np.sort(exported_ids.to_numpy()), np.sort(ids)):
        raise ValueError(f"doc_id alignment failed for {path}")
    vectors = emb[expected_cols].to_numpy(dtype=np.float32)[positions]
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite vectors in {path}")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"Zero-length vectors in {path}")
    return vectors / norms[:, None]


def nearest_source(source_vectors: np.ndarray, target_vectors: np.ndarray,
                   source_roles: np.ndarray, *, k: int = 10, block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine search in bounded blocks; source vectors stay source-only."""
    n = len(target_vectors)
    distances = np.empty(n, dtype=np.float32)
    entropies = np.empty(n, dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        similarities = target_vectors[start:stop] @ source_vectors.T
        indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        top_sim = np.take_along_axis(similarities, indices, axis=1)
        distances[start:stop] = 1.0 - top_sim.mean(axis=1)
        labels = source_roles[indices]
        counts = np.stack([(labels == role).sum(axis=1) for role in range(4)], axis=1)
        probs = counts / k
        entropies[start:stop] = -(np.where(probs > 0, probs * np.log2(np.maximum(probs, 1e-12)), 0)).sum(axis=1)
        if start == 0 or stop == n or (start // block) % 32 == 0:
            print(f"[neighbors] {stop}/{n}", flush=True)
    return distances, entropies


def one_run(path: Path, expected: pd.DataFrame, target: str) -> tuple[pd.DataFrame, float]:
    manifest = path / "run_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    details = json.loads(manifest.read_text(encoding="utf-8"))
    if details.get("status") != "complete" or target not in details.get("test_corpora", []):
        raise ValueError(f"Incomplete or wrong-target run: {manifest}")
    predictions = read_predictions(path / "predictions" / f"predictions_{target}.csv", expected)
    actual = float(balanced_accuracy_score(predictions.true_macro, predictions.pred_macro))
    recorded = recorded_ba(path, target)
    if not np.isclose(actual, recorded, atol=1e-8, rtol=0):
        raise ValueError(f"BA mismatch in {path}: recomputed={actual}, recorded={recorded}")
    return predictions, actual


def load_recorded_intervals() -> pd.DataFrame:
    frames = []
    for campaign in RECIPES:
        path = ROOT / "output" / f"{campaign}_analysis" / "ood_metrics_by_target.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["analysis_file"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    # Original CamemBERT metallurgy analysis belongs to the erroneous runs.
    bad = (combined.backbone == "camembert_base") & (combined.source_corpus == "metallurgie") & (combined.analysis_file.str.contains("camembert_source_factorial_analysis"))
    combined = combined.loc[~bad & combined.method.isin(METHODS)].copy()
    if combined.duplicated(["backbone", "source_corpus", "method", "evaluation_corpus"]).any():
        raise ValueError("Duplicate recorded interval condition")
    return combined


def save_tables(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)
    with pd.ExcelWriter(out / "diagnostics.xlsx", engine="openpyxl") as writer:
        for name, frame in tables.items():
            if name in {"unit_profiles", "neighbor_indicators"}:
                continue
            frame.to_excel(writer, sheet_name=name[:31], index=False)


def text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.07, .93, title, fontsize=18, weight="bold", va="top")
    y = .84
    for paragraph in paragraphs:
        import textwrap
        wrapped = textwrap.fill(paragraph, width=115)
        fig.text(.07, y, wrapped, fontsize=11, va="top", linespacing=1.5)
        y -= .055 * (wrapped.count("\n") + 1) + .035
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf: PdfPages, title: str, frame: pd.DataFrame, columns: list[str], n: int = 24) -> None:
    display = frame[columns].head(n).copy()
    display.replace({"btp": "Construction", "caou": "Chemistry–plastics",
                     "metallurgie": "Metallurgy", "nicollin": "Company"}, inplace=True)
    for col in display.select_dtypes(include=["float"]).columns:
        display[col] = display[col].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, pad=16)
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_report(out: Path, tables: dict[str, pd.DataFrame], config: dict) -> None:
    overlap = tables["overlap_summary"]
    variance = tables["ba_variance"]
    profile = tables["error_profiles"]
    neighbor = tables["neighbor_associations"]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    with PdfPages(out / "transfer_error_diagnostics.pdf") as pdf:
        text_page(pdf, "Transfer-error diagnostics", [
            "Prediction-only analysis of CE, SupCon and SoftTriple across Qwen3, multilingual E5-large and CamemBERT, two training sources, three unseen targets and five paired training seeds. No model was retrained.",
            "The principal questions are whether objectives fail on the same factual units, which observed factors account for balanced-accuracy variation, and what describes units repeatedly misclassified by all three methods.",
            "Roles are A0 (work situation), A1 (explicitly reported unfavourable condition), B (accident event or deviation) and C (reported consequence). The four evaluation corpora are Construction, Metallurgy, Chemistry–plastics and an independent Company corpus.",
            "Outputs: this PDF, corresponding CSV tables, diagnostics.xlsx, a local review CSV, figures, and a manifest of inputs and parameters. Company narratives are never quoted in this PDF.",
        ])
        text_page(pdf, "Definitions and uncertainty", [
            "For each backbone, training source, target and seed, a unit is common-error when CE, SupCon and SoftTriple all predict the wrong role. Category counts 0, 1, 2 and 3 describe how many of these methods err. The unit denominator is all labelled target units; the any-error denominator includes units missed by at least one method. The error-event denominator counts individual wrong predictions across methods.",
            "A persistent common error is a unit missed by all three methods in all five seeds. A stable correct unit is correctly classified by all three in all five seeds. These labels describe the observed runs, not intrinsic unit difficulty.",
            "Reported overlap intervals use 2,000 accident-level bootstrap resamples, with the same sampled accidents used across the three methods and five seeds. Standard deviations across seeds are listed separately. Balanced-accuracy intervals are taken from the existing campaign analyses after reproducing their point estimates from saved predictions.",
        ])
        text_page(pdf, "Further definitions", [
            "The factorial sums of squares are descriptive for the balanced common-target design; they are not causal effects or population-level variance components. Frozen nearest-neighbour distance and label entropy are diagnostic indicators, not geometry of the adapted models.",
            "For each target unit, the source-neighbour diagnostic uses exact cosine similarity in the frozen encoder space. Distance is one minus the mean cosine similarity of the ten nearest labelled source units. Label entropy is the Shannon entropy, in bits, of their four role counts; it ranges from zero to two.",
        ])
        method_terms = variance.loc[variance.term.str.contains("method"), "share_total_variation"].sum()
        company_a1 = tables["persistent_by_role_target"].query("target == 'nicollin' and role == 'A1'").iloc[0]
        text_page(pdf, "Main findings from the observed runs", [
            f"Across the 18 backbone-source-target conditions, the three objectives jointly miss an average {overlap.common_among_any.mean():.1%} of units with at least one error (range {overlap.common_among_any.min():.1%}–{overlap.common_among_any.max():.1%}). These are condition averages, not a pooled unit-level estimate.",
            f"On balanced accuracy for the two common targets, the target main term accounts for {variance.set_index('term').loc['target','share_total_variation']:.1%} of observed variation and the backbone term for {variance.set_index('term').loc['backbone','share_total_variation']:.1%}. The method main term accounts for {variance.set_index('term').loc['method','share_total_variation']:.1%}; all terms involving method together account for {method_terms:.1%}. Interactions and seed variation are shown in the following tables.",
            f"The Company corpus has {int(company_a1.n_units)} A1 units. Averaged over the six backbone-source conditions, {company_a1.persistent_fraction:.1%} of these are missed by all three methods in all five seeds. The small support makes small differences sensitive to individual accidents.",
        ])
        target = overlap.groupby("target", as_index=False).agg(common_unit_fraction=("common_unit_fraction", "mean"),
             common_among_any=("common_among_any", "mean"), persistent_unit_fraction=("persistent_unit_fraction", "mean"))
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(target)); width = .25
        for j, col in enumerate(["common_unit_fraction", "common_among_any", "persistent_unit_fraction"]):
            ax.bar(x + (j-1)*width, target[col], width, label=col.replace("_", " "))
        display_targets = {"btp": "Construction", "caou": "Chemistry–plastics",
                           "metallurgie": "Metallurgy", "nicollin": "Company"}
        ax.set_xticks(x, [display_targets[item] for item in target.target])
        ax.set_ylim(0, .6); ax.set_ylabel("Fraction")
        ax.set_title("Method overlap and persistent errors by target")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(figures / "overlap_by_target.png", dpi=200)
        pdf.savefig(fig); plt.close(fig)
        table_page(pdf, "Overlap by backbone, source and target", overlap,
                   ["backbone", "source", "target", "common_unit_fraction", "common_among_any", "persistent_unit_fraction"], 18)
        table_page(pdf, "Accident-bootstrap intervals for common-error unit share", overlap,
                   ["backbone", "source", "target", "common_unit_fraction", "common_unit_ci_low", "common_unit_ci_high"], 18)
        fig, ax = plt.subplots(figsize=(10, 6))
        main = variance.loc[(variance.order == 1) | (variance.term == "seed/run residual")].sort_values("share_total_variation")
        ax.barh(main.term, main.share_total_variation)
        ax.set_xlabel("Share of observed BA variation")
        ax.set_title("Balanced common-target factorial design: main terms and run residual")
        fig.tight_layout(); fig.savefig(figures / "ba_variance.png", dpi=200); pdf.savefig(fig); plt.close(fig)
        table_page(pdf, "Largest descriptive BA components", variance,
                   ["term", "sum_squares", "share_total_variation"], 16)
        ba_compact = tables["ba_summary"].groupby(["backbone", "source", "target"], as_index=False).agg(
            ba_mean=("ba_mean", "mean"), min_ba=("ba_mean", "min"), max_ba=("ba_mean", "max"))
        table_page(pdf, "BA by condition, averaged over the three objectives", ba_compact,
                   ["backbone", "source", "target", "ba_mean", "min_ba", "max_ba"], 18)
        text_page(pdf, "Reading the balanced-accuracy tables", [
            "The preceding table averages CE, SupCon and SoftTriple within each backbone-source-target cell. It is a compact descriptive view; the CSV and workbook retain method-specific five-seed means, seed SDs and target-accident bootstrap intervals.",
            "BTP→Metallurgy and Metallurgy→BTP are different transfer directions and do not enter the two-common-target factorial decomposition. They are reported as separate rows and in the method-specific CSV.",
        ])
        table_page(pdf, "Persistent errors by true role and target", tables["persistent_by_role_target"],
                   ["target", "role", "n_units", "n_conditions", "persistent_fraction", "common_error_mean_fraction"], 16)
        role_target = tables["role_recall"].groupby(["target", "role"], as_index=False).agg(recall_mean=("recall_mean", "mean"), support=("support", "first"))
        table_page(pdf, "Recall by target and role (mean across observed conditions)", role_target,
                   ["target", "role", "support", "recall_mean"], 16)
        heat = role_target.pivot(index="target", columns="role", values="recall_mean").reindex(
            index=["btp", "metallurgie", "caou", "nicollin"], columns=ROLES)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heat.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_xticks(range(4), ROLES)
        ax.set_yticks(range(4), ["Construction", "Metallurgy", "Chemistry–plastics", "Company"])
        for i in range(4):
            for j in range(4):
                value = heat.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if value >= .65 else "black")
        ax.set_title("Mean role recall across observed adapted conditions")
        fig.colorbar(im, ax=ax, label="Recall")
        fig.tight_layout(); fig.savefig(figures / "role_recall_heatmap.png", dpi=200)
        pdf.savefig(fig); plt.close(fig)
        table_page(pdf, "Largest descriptive role-recall components", tables["role_variance"],
                   ["term", "sum_squares", "share_total_variation"], 12)
        profile_compact = tables["unit_profiles"].loc[
            tables["unit_profiles"].status.isin(["persistent", "stable_correct"])].groupby(
            ["target", "status"], as_index=False).agg(n_unit_conditions=("status", "size"),
             mean_words=("n_words", "mean"), mean_confidence=("mean_confidence", "mean"))
        table_page(pdf, "Unit profiles: persistent versus stable correct", profile_compact,
                   ["target", "status", "n_unit_conditions", "mean_words", "mean_confidence"], 8)
        confusions = tables["persistent_confusions"].groupby(
            ["target", "role", "modal_wrong_prediction"], as_index=False).agg(n_unit_conditions=("n_units", "sum"))
        confusions = confusions.sort_values("n_unit_conditions", ascending=False)
        table_page(pdf, "Frequent persistent-error confusions", confusions,
                   ["target", "role", "modal_wrong_prediction", "n_unit_conditions"], 16)
        if len(neighbor):
            neighbor_compact = tables["unit_profiles"].loc[
                tables["unit_profiles"].status.isin(["persistent", "stable_correct"]) &
                tables["unit_profiles"].backbone.isin(BACKBONES[:2])].groupby(
                ["backbone", "target", "status"], as_index=False).agg(n_unit_conditions=("status", "size"),
                mean_cosine_distance=("mean_cosine_distance", "mean"), mean_label_entropy=("label_entropy", "mean"))
            table_page(pdf, "Frozen source-neighbour indicators", neighbor_compact,
                       ["backbone", "target", "status", "n_unit_conditions", "mean_cosine_distance", "mean_label_entropy"], 16)
            wide = neighbor_compact.pivot(index=["backbone", "target"], columns="status",
                                          values="mean_cosine_distance")
            delta = wide["persistent"] - wide["stable_correct"]
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f"{b.replace('multilingual_e5_large','E5-large')} → {display_targets[t]}" for b, t in delta.index]
            ax.barh(labels, delta.to_numpy())
            ax.axvline(0, color="black", linewidth=.8)
            ax.set_xlabel("Persistent minus stable-correct mean cosine distance")
            ax.set_title("Frozen-embedding source-neighbour diagnostic")
            fig.tight_layout(); fig.savefig(figures / "neighbor_distance_difference.png", dpi=200)
            pdf.savefig(fig); plt.close(fig)
        table_page(pdf, "One-run Triplet reference (exploratory)", tables["triplet_exploratory"],
                   ["target", "balanced_accuracy", "accuracy", "macro_f1", "n_units"], 3)
        text_page(pdf, "Interpretation boundaries", [
            "Common errors show robustness to the three evaluated objectives, not to all modelling choices. Triplet has only one run and is not included in the five-seed overlap or variance conclusions.",
            "The Company corpus differs from EPICEA in reporting context, population, narrative length, role prevalence and segmentation settings. No single factor can be isolated as the cause of its transfer gap.",
            "Confidences are not calibrated across the three strategies; compare them within a strategy or treat their pooled means as descriptive only. Annotation on prepared factual units is conditional on the segmentation and does not measure end-to-end extraction from raw narratives.",
            "The source-neighbour analysis uses frozen Qwen3 and E5 embeddings. It does not reveal where the corresponding adapted run placed a unit. Association between neighbour indicators and errors cannot establish a causal mechanism or concept shift.",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/diagnostics")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--neighbor-block", type=int, default=256)
    parser.add_argument("--skip-neighbors", action="store_true", help="Only for a quick validation run")
    parser.add_argument("--report-only", action="store_true", help="Re-render PDF and workbook from validated CSV tables")
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("n-bootstrap must be at least 100")
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        table_paths = list((out / "tables").glob("*.csv"))
        if not table_paths:
            raise FileNotFoundError(out / "tables")
        tables = {path.stem: pd.read_csv(path) for path in table_paths}
        needed = {"overlap_summary", "ba_variance", "error_profiles", "neighbor_associations",
                  "persistent_by_role_target", "role_recall", "persistent_confusions", "triplet_exploratory"}
        if needed - set(tables):
            raise ValueError(f"Missing report tables: {sorted(needed - set(tables))}")
        role = tables["persistent_by_condition_role"]
        tables["persistent_by_role_target"] = role.groupby(["target", "role"], as_index=False).agg(
            n_units=("n_units", "first"), n_conditions=("n_units", "size"),
            persistent_fraction=("persistent_fraction", "mean"),
            common_error_mean_fraction=("common_error_mean_fraction", "mean"))
        if "file_inventory" not in tables:
            inventory = tables["input_inventory"]
            used = set(DATA_PATHS.values()) | set(RECIPES.values())
            for col in ("prediction_file", "manifest_file", "recipe_file"):
                used |= {Path(path) for path in inventory[col].dropna()}
            used |= {ROOT / "output" / f"{campaign}_analysis" / "ood_metrics_by_target.csv" for campaign in RECIPES}
            used |= {ROOT / "output/batch_triplet/macro_ft_tuning/combos/full_no/predictions" / f"predictions_{target}.csv"
                     for target in ("metallurgie", "caou", "nicollin")}
            if len(tables["neighbor_indicators"]):
                used |= {ROOT / "embeddings" / backbone / f"{corpus}.csv"
                         for backbone in BACKBONES[:2] for corpus in CORPORA}
            tables["file_inventory"] = pd.DataFrame([{"path": str(path), "bytes": path.stat().st_size}
                for path in sorted(used, key=str)])
        save_tables(out, tables)
        parameters = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
        make_report(out, tables, parameters)
        print(f"[report] {out / 'transfer_error_diagnostics.pdf'}", flush=True)
        return
    metadata = canonical_metadata()
    _, seeds = prepare_campaigns()
    recorded_intervals = load_recorded_intervals()
    records, overlaps, seed_counts, units, recalls = [], [], [], [], []
    inventory = []
    for backbone in BACKBONES:
        for source in SOURCES:
            for target in (corpus for corpus in CORPORA if corpus != source):
                print(f"[analysis] {backbone} {source} -> {target}", flush=True)
                expected = metadata[target]
                n_units = len(expected)
                errors = np.zeros((len(seeds), len(METHODS), n_units), dtype=np.uint8)
                confidence = np.zeros((len(seeds), len(METHODS), n_units), dtype=np.float32)
                pred_codes = np.zeros((len(seeds), len(METHODS), n_units), dtype=np.uint8)
                for s_index, seed in enumerate(seeds):
                    for m_index, method in enumerate(METHODS):
                        recipe, root, model_id = model_location(backbone, source, method)
                        run_dir = root / model_id / f"seed_{seed}"
                        frame, ba = one_run(run_dir, expected, target)
                        pred = frame.pred_macro.to_numpy(dtype=str)
                        truth = frame.true_macro.to_numpy(dtype=str)
                        errors[s_index, m_index] = (pred != truth)
                        pred_codes[s_index, m_index] = pd.Categorical(pred, categories=ROLES).codes
                        confidence[s_index, m_index] = frame.confidence.to_numpy(dtype=np.float32)
                        records.append({"backbone": backbone, "source": source, "target": target,
                                        "method": method, "seed": seed, "balanced_accuracy": ba, "n_units": n_units})
                        inventory.append({"backbone": backbone, "source": source, "target": target,
                                          "method": method, "seed": seed, "prediction_file": str(run_dir / 'predictions' / f'predictions_{target}.csv'),
                                          "manifest_file": str(run_dir / 'run_manifest.json'), "recipe_file": str(recipe)})
                        for role in ROLES:
                            mask = truth == role
                            recalls.append({"backbone": backbone, "source": source, "target": target,
                                            "method": method, "seed": seed, "role": role,
                                            "support": int(mask.sum()), "recall": float((pred[mask] == role).mean())})
                counts = errors.sum(axis=1)
                for s_index, seed in enumerate(seeds):
                    categories = np.bincount(counts[s_index], minlength=4)
                    any_units = int(n_units - categories[0])
                    events = int(counts[s_index].sum())
                    if int(categories.sum()) != n_units:
                        raise ValueError("Error-category count mismatch")
                    seed_counts.append({"backbone": backbone, "source": source, "target": target, "seed": seed,
                         **{f"wrong_{k}": int(categories[k]) for k in range(4)}, "n_units": n_units,
                         "any_error_units": any_units, "error_events": events,
                         "common_fraction_units": categories[3] / n_units,
                         "common_fraction_any": categories[3] / any_units if any_units else 0.0,
                         "common_fraction_events": 3 * categories[3] / events if events else 0.0})
                persistent = (counts == 3).all(axis=0)
                stable_correct = (counts == 0).all(axis=0)
                bootstrap = bootstrap_overlap(errors, expected.index.get_level_values(0).to_numpy(),
                    n_boot=args.n_bootstrap, seed=2027 + BACKBONES.index(backbone)*100 + SOURCES.index(source)*10 + CORPORA.index(target))
                group_seed = [row for row in seed_counts if row["backbone"] == backbone and row["source"] == source and row["target"] == target]
                overlaps.append({"backbone": backbone, "source": source, "target": target,
                    "n_units": n_units, "n_accidents": expected.index.get_level_values(0).nunique(),
                    "common_unit_fraction": float(np.mean([row["common_fraction_units"] for row in group_seed])),
                    "common_unit_seed_sd": float(np.std([row["common_fraction_units"] for row in group_seed], ddof=1)),
                    "common_among_any": float(np.mean([row["common_fraction_any"] for row in group_seed])),
                    "common_error_event_fraction": float(np.mean([row["common_fraction_events"] for row in group_seed])),
                    "persistent_unit_fraction": float(persistent.mean()),
                    "n_persistent": int(persistent.sum()), "n_stable_correct": int(stable_correct.sum()), **bootstrap})
                truth_codes = pd.Categorical(expected.pred_label, categories=ROLES).codes
                all_wrong_seeds = (counts == 3).sum(axis=0)
                most_common_pred = np.empty(n_units, dtype=object)
                for i in range(n_units):
                    wrong = pred_codes[:, :, i][errors[:, :, i].astype(bool)]
                    most_common_pred[i] = ROLES[int(np.bincount(wrong, minlength=4).argmax())] if len(wrong) else ""
                profile = expected[["doc_id", "pred_label", "sentence"]].copy().reset_index()
                profile.rename(columns={"pred_label": "role"}, inplace=True)
                profile["backbone"] = backbone
                profile["source"] = source
                profile["target"] = target
                profile["common_error_seeds"] = all_wrong_seeds
                profile["any_error_seeds"] = (counts > 0).sum(axis=0)
                profile["persistent_common_error"] = persistent
                profile["stable_correct"] = stable_correct
                profile["status"] = np.where(persistent, "persistent", np.where(stable_correct, "stable_correct", "mixed"))
                profile["mean_wrong_methods"] = counts.mean(axis=0)
                profile["mean_confidence"] = confidence.mean(axis=(0, 1))
                profile["modal_wrong_prediction"] = most_common_pred
                profile["n_words"] = profile.sentence.astype(str).str.split().str.len()
                profile["n_characters"] = profile.sentence.astype(str).str.len()
                if not np.array_equal(truth_codes, pd.Categorical(profile.role, categories=ROLES).codes):
                    raise ValueError("Role order changed while assembling unit profiles")
                units.append(profile)
    runs = pd.DataFrame(records)
    if len(runs) != 270:
        raise ValueError(f"Expected 270 target-run observations (90 runs x 3 targets), got {len(runs)}")
    overlap = pd.DataFrame(overlaps)
    seed_counts_df = pd.DataFrame(seed_counts)
    unit_profiles = pd.concat(units, ignore_index=True)
    recalls_df = pd.DataFrame(recalls)
    ba_summary = runs.groupby(["backbone", "source", "target", "method"], as_index=False).agg(
        ba_mean=("balanced_accuracy", "mean"), ba_seed_sd=("balanced_accuracy", "std"))
    interval = recorded_intervals.rename(columns={"source_corpus": "source", "evaluation_corpus": "target"})
    ba_summary = ba_summary.merge(interval[["backbone", "source", "target", "method", "balanced_accuracy_mean", "ci_low", "ci_high", "n_seeds"]],
                                  on=["backbone", "source", "target", "method"], how="left", validate="one_to_one")
    if ba_summary.ci_low.isna().any() or (ba_summary.n_seeds != 5).any() or not np.allclose(ba_summary.ba_mean, ba_summary.balanced_accuracy_mean, atol=1e-8):
        raise ValueError("Saved bootstrap summary does not match recomputed BA")
    ba_summary.drop(columns="balanced_accuracy_mean", inplace=True)
    common = runs.loc[runs.target.isin(["caou", "nicollin"])].copy()
    ba_variance = factorial_ss(common, "balanced_accuracy", ("target", "backbone", "source", "method"))
    role_common = recalls_df.loc[recalls_df.target.isin(["caou", "nicollin"])].copy()
    role_variance = factorial_ss(role_common, "recall", ("target", "role", "backbone", "source", "method"))
    role_recall = recalls_df.groupby(["backbone", "source", "target", "method", "role"], as_index=False).agg(
        recall_mean=("recall", "mean"), recall_seed_sd=("recall", "std"), support=("support", "first"))
    persistent_role = unit_profiles.groupby(["backbone", "source", "target", "role"], as_index=False).agg(
        n_units=("role", "size"), persistent_fraction=("persistent_common_error", "mean"),
        common_error_mean_fraction=("common_error_seeds", lambda x: float(x.mean()/5)))
    persistent_role_target = persistent_role.groupby(["target", "role"], as_index=False).agg(
        n_units=("n_units", "first"), n_conditions=("n_units", "size"),
        persistent_fraction=("persistent_fraction", "mean"),
        common_error_mean_fraction=("common_error_mean_fraction", "mean"))
    error_profiles = unit_profiles.groupby(["backbone", "source", "target", "role", "status"], as_index=False).agg(
        n_units=("role", "size"), mean_words=("n_words", "mean"), median_words=("n_words", "median"),
        mean_characters=("n_characters", "mean"), mean_confidence=("mean_confidence", "mean"))
    confusion = unit_profiles.loc[unit_profiles.persistent_common_error].groupby(
        ["backbone", "source", "target", "role", "modal_wrong_prediction"], as_index=False).size().rename(columns={"size": "n_units"})
    neighbors = []
    if not args.skip_neighbors:
        for backbone in BACKBONES[:2]:
            vectors = {corpus: validate_embedding(corpus, backbone, metadata[corpus].reset_index()) for corpus in CORPORA}
            for source in SOURCES:
                source_labels = pd.Categorical(metadata[source].pred_label, categories=ROLES).codes
                for target in (corpus for corpus in CORPORA if corpus != source):
                    print(f"[neighbors] {backbone} {source} -> {target}", flush=True)
                    distance, entropy = nearest_source(vectors[source], vectors[target], source_labels,
                                                        block=args.neighbor_block)
                    table = metadata[target][["doc_id"]].reset_index()
                    table["backbone"] = backbone; table["source"] = source; table["target"] = target
                    table["mean_cosine_distance"] = distance
                    table["label_entropy"] = entropy
                    neighbors.append(table)
    neighbor_indicators = pd.concat(neighbors, ignore_index=True) if neighbors else pd.DataFrame()
    if len(neighbor_indicators):
        unit_profiles = unit_profiles.merge(neighbor_indicators,
            on=["accident_id", "fact_id", "doc_id", "backbone", "source", "target"],
            how="left", validate="many_to_one")
        if unit_profiles.loc[unit_profiles.backbone.isin(BACKBONES[:2]), "mean_cosine_distance"].isna().any():
            raise ValueError("Missing nearest-neighbor diagnostic for a Qwen3/E5 target unit")
        neighbor_associations = unit_profiles.loc[unit_profiles.backbone.isin(BACKBONES[:2])].groupby(
            ["backbone", "source", "target", "role", "status"], as_index=False).agg(
            n_units=("role", "size"), mean_cosine_distance=("mean_cosine_distance", "mean"),
            mean_label_entropy=("label_entropy", "mean"))
    else:
        neighbor_associations = pd.DataFrame(columns=["backbone", "source", "target", "role", "status", "n_units", "mean_cosine_distance", "mean_label_entropy"])
    triplet_rows = []
    triplet_root = ROOT / "output/batch_triplet/macro_ft_tuning/combos/full_no/predictions"
    for target in ("metallurgie", "caou", "nicollin"):
        frame = read_predictions(triplet_root / f"predictions_{target}.csv", metadata[target])
        triplet_rows.append({"target": target, "balanced_accuracy": balanced_accuracy_score(frame.true_macro, frame.pred_macro),
           "accuracy": float((frame.true_macro == frame.pred_macro).mean()),
           "macro_f1": float(f1_score(frame.true_macro, frame.pred_macro, labels=ROLES, average="macro")),
           "n_units": len(frame), "run_count": 1})
    triplet = pd.DataFrame(triplet_rows)
    used_files = set(DATA_PATHS.values()) | set(RECIPES.values())
    used_files |= {Path(row["prediction_file"]) for row in inventory}
    used_files |= {Path(row["manifest_file"]) for row in inventory}
    used_files |= {Path(path) for path in recorded_intervals.analysis_file.unique()}
    used_files |= {triplet_root / f"predictions_{target}.csv" for target in ("metallurgie", "caou", "nicollin")}
    if not args.skip_neighbors:
        used_files |= {ROOT / "embeddings" / backbone / f"{corpus}.csv"
                       for backbone in BACKBONES[:2] for corpus in CORPORA}
    file_inventory = pd.DataFrame([{"path": str(path), "bytes": path.stat().st_size}
                                   for path in sorted(used_files, key=str)])
    review = unit_profiles.loc[unit_profiles.persistent_common_error].sort_values(
        ["target", "role", "mean_confidence"], ascending=[True, True, False]).groupby(
        ["backbone", "source", "target", "role"], group_keys=False).head(20).copy()
    review.to_csv(out / "units_for_expert_review_local.csv", index=False)
    safe_unit_profiles = unit_profiles.drop(columns="sentence")
    tables = {"run_scores": runs, "ba_summary": ba_summary, "overlap_per_seed": seed_counts_df,
              "overlap_summary": overlap, "ba_variance": ba_variance, "role_variance": role_variance,
              "role_recall": role_recall, "persistent_by_role_target": persistent_role_target,
              "persistent_by_condition_role": persistent_role, "error_profiles": error_profiles,
              "persistent_confusions": confusion, "neighbor_associations": neighbor_associations,
              "neighbor_indicators": neighbor_indicators, "triplet_exploratory": triplet,
              "unit_profiles": safe_unit_profiles, "input_inventory": pd.DataFrame(inventory),
              "file_inventory": file_inventory}
    save_tables(out, tables)
    config = {"seeds": seeds, "n_bootstrap": args.n_bootstrap, "neighbor_k": 10,
              "neighbor_metric": "mean cosine distance; exact top 10", "neighbor_block": args.neighbor_block,
              "neighbor_backbones": list(BACKBONES[:2]) if not args.skip_neighbors else [],
              "triplet_scope": "one-run exploratory Qwen3 BTP only", "camembert_metallurgie": "repaired campaign only"}
    (out / "parameters.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    make_report(out, tables, config)
    print(f"[done] {out / 'transfer_error_diagnostics.pdf'}", flush=True)


if __name__ == "__main__":
    main()
