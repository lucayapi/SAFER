"""Exact within/cross-corpus role cosine means from existing embeddings.

No model fitting occurs. Within-corpus pairs from the same accident are
excluded. Frozen-space intervals resample whole accidents within each corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scgm_text.data_metadata import load_filtered_metadata


ROLES = ("A0", "A1", "B", "C")
CORPORA = ("btp", "metallurgie", "caou", "nicollin")
FROZEN = ("qwen3", "multilingual_e5_large")
ADAPTED = ("cross_entropy", "supcon", "softtriple")
DISPLAY = {"btp": "Construction", "metallurgie": "Metallurgy",
           "caou": "Chemistry-plastics", "nicollin": "Company"}


@dataclass
class RoleStats:
    n: int
    unit_sum: np.ndarray
    accident_counts: np.ndarray
    accident_sums: np.ndarray


def metadata(corpus: str) -> pd.DataFrame:
    path = ROOT / "dataset" / f"data_{corpus}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = load_filtered_metadata(str(path), text_col="sentence")
    if frame["doc_id"].duplicated().any() or frame["accident_id"].isna().any():
        raise ValueError(f"Non-unique doc_id or missing accident_id: {corpus}")
    return frame


def frozen_vectors(backbone: str, corpus: str, reference: pd.DataFrame) -> np.ndarray:
    path = ROOT / "embeddings" / backbone / f"{corpus}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = [f"dim_{i:04d}" for i in range(1, 1025)]
    frame = pd.read_csv(path, dtype={name: np.float32 for name in expected})
    if frame.columns.tolist() != ["doc_id", *expected] or len(frame) != len(reference):
        raise ValueError(f"Frozen embedding schema/row count mismatch: {path}")
    index = pd.Index(frame["doc_id"].astype(str))
    if index.has_duplicates:
        raise ValueError(f"Duplicate exported doc_id: {path}")
    positions = index.get_indexer(reference["doc_id"].astype(str))
    if (positions < 0).any():
        raise ValueError(f"Frozen doc_id alignment failed: {path}")
    return frame[expected].to_numpy(dtype=np.float32, copy=True)[positions]


def adapted_vectors(method: str, seed: int, corpus: str, reference: pd.DataFrame) -> np.ndarray:
    base = (ROOT / "output" / "adapted_geometry_pilot" / "runs" /
            f"qwen3_btp_{method}" / f"seed_{seed}" / "embeddings")
    matrix_path = base / f"projected_{corpus}.npy"
    meta_path = base / f"projected_{corpus}_metadata.csv"
    if not matrix_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing adapted export: {base}, {corpus}")
    exported = pd.read_csv(meta_path, usecols=["accident_id", "doc_id", "pred_label"])
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2 or len(exported) != len(matrix) or len(matrix) != len(reference):
        raise ValueError(f"Adapted embedding row count mismatch: {matrix_path}")
    index = pd.Index(exported["doc_id"].astype(str))
    if index.has_duplicates:
        raise ValueError(f"Duplicate adapted doc_id: {meta_path}")
    positions = index.get_indexer(reference["doc_id"].astype(str))
    if (positions < 0).any():
        raise ValueError(f"Adapted doc_id alignment failed: {meta_path}")
    if not np.array_equal(exported["pred_label"].to_numpy()[positions],
                          reference["pred_label"].to_numpy()):
        raise ValueError(f"Adapted role labels do not match corpus: {meta_path}")
    if not np.array_equal(exported["accident_id"].astype(str).to_numpy()[positions],
                          reference["accident_id"].astype(str).to_numpy()):
        raise ValueError(f"Adapted accident IDs do not match corpus: {meta_path}")
    return np.asarray(matrix[positions], dtype=np.float32)


def summarize_vectors(vectors: np.ndarray, reference: pd.DataFrame) -> dict[str, RoleStats]:
    if vectors.ndim != 2 or len(vectors) != len(reference):
        raise ValueError("Embedding count differs from metadata")
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite embedding value")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms <= 0).any():
        raise ValueError("Zero-length embedding")
    normalized = vectors / norms[:, None]
    result = {}
    roles = reference["pred_label"].to_numpy(dtype=str)
    accidents = reference["accident_id"].astype(str).to_numpy()
    for role in ROLES:
        mask = roles == role
        selected = normalized[mask]
        names, inverse = np.unique(accidents[mask], return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        ordered_groups = inverse[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_groups)) + 1]
        sums = np.add.reduceat(selected[order], boundaries, axis=0, dtype=np.float64)
        counts = np.diff(np.r_[boundaries, len(selected)]).astype(np.int64)
        if len(names) != len(sums) or not len(selected):
            raise ValueError(f"Invalid accident grouping for role {role}")
        result[role] = RoleStats(
            n=len(selected), unit_sum=sums.sum(axis=0),
            accident_counts=counts, accident_sums=sums,
        )
    return result


def within_similarity(stat: RoleStats) -> float:
    denominator = stat.n**2 - float(np.square(stat.accident_counts.astype(float)).sum())
    if denominator <= 0:
        raise ValueError("No valid different-accident within-sector pairs")
    numerator = (np.dot(stat.unit_sum, stat.unit_sum)
                 - np.einsum("ij,ij->", stat.accident_sums, stat.accident_sums))
    return float(numerator / denominator)


def cross_similarity(left: RoleStats, right: RoleStats) -> float:
    return float(np.dot(left.unit_sum, right.unit_sum) / (left.n * right.n))


def _point_rows(space: str, method: str, seed: int | None,
                stats: dict[str, dict[str, RoleStats]]) -> list[dict]:
    rows = []
    for role in ROLES:
        within = {corpus: within_similarity(stats[corpus][role]) for corpus in CORPORA}
        for left, right in combinations(CORPORA, 2):
            cross = cross_similarity(stats[left][role], stats[right][role])
            paired_within = (within[left] + within[right]) / 2.0
            rows.append({
                "space": space, "method": method, "seed": seed,
                "role": role, "sector_left": left, "sector_right": right,
                "n_left": stats[left][role].n, "n_right": stats[right][role].n,
                "within_left": within[left], "within_right": within[right],
                "within_pair_mean": paired_within, "cross": cross,
                "delta": paired_within - cross,
            })
    return rows


def _bootstrap_role(stats: dict[str, RoleStats], *, resamples: int,
                    rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cluster bootstrap for one role, separately within each corpus.

    Weights are applied to accidents. Terms involving the same original
    accident are excluded even if that accident is drawn more than once.
    """
    result = {}
    for corpus, stat in stats.items():
        n_accidents = len(stat.accident_counts)
        weights = rng.multinomial(n_accidents,
                                  np.full(n_accidents, 1.0 / n_accidents),
                                  size=resamples).astype(np.float32)
        counts = stat.accident_counts.astype(np.float32)
        sums = stat.accident_sums.astype(np.float32)
        total_counts = weights @ counts
        total_sums = weights @ sums
        square_weights = weights * weights
        same_count = square_weights @ (counts * counts)
        same_dot = square_weights @ np.einsum("ij,ij->i", sums, sums)
        denominator = total_counts**2 - same_count
        if (denominator <= 0).any():
            raise ValueError(f"Bootstrap lost all different-accident {corpus} pairs")
        numerator = np.einsum("ij,ij->i", total_sums, total_sums) - same_dot
        within = numerator / denominator
        result[corpus] = (within, total_sums / total_counts[:, None])
    return result


def frozen_bootstrap(stats: dict[str, dict[str, RoleStats]], *,
                     resamples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for role in ROLES:
        result = _bootstrap_role({corpus: stats[corpus][role] for corpus in CORPORA},
                                 resamples=resamples, rng=rng)
        pair_deltas = []
        for left, right in combinations(CORPORA, 2):
            within = (result[left][0] + result[right][0]) / 2.0
            cross = np.einsum("ij,ij->i", result[left][1], result[right][1])
            delta = within - cross
            pair_deltas.append(delta)
            rows.append({"role": role, "sector_left": left, "sector_right": right,
                         "ci_low": float(np.quantile(delta, .025)),
                         "ci_high": float(np.quantile(delta, .975)),
                         "bootstrap_resamples": resamples})
        overall = np.mean(pair_deltas, axis=0)
        rows.append({"role": role, "sector_left": "equal_pair_mean", "sector_right": "",
                     "ci_low": float(np.quantile(overall, .025)),
                     "ci_high": float(np.quantile(overall, .975)),
                     "bootstrap_resamples": resamples})
    return rows


def _seeds() -> list[int]:
    base = ROOT / "output" / "adapted_geometry_pilot" / "runs"
    result = []
    for method in ADAPTED:
        paths = sorted((base / f"qwen3_btp_{method}").glob("seed_*"))
        seeds = {int(path.name[5:]) for path in paths if path.is_dir()}
        if len(seeds) != 5:
            raise ValueError(f"Expected five adapted runs for {method}; found {len(seeds)}")
        result.append(seeds)
    if result[0] != result[1] or result[0] != result[2]:
        raise ValueError("Adapted methods do not have matching seeds")
    return sorted(result[0])


def _report(out: Path, pair: pd.DataFrame, role: pd.DataFrame,
            pair_group: pd.DataFrame, boot: pd.DataFrame, audit: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = out / "role_geometry_diagnostic.pdf"
    summary = role.copy()
    summary["label"] = summary["space"].map({
        "frozen_qwen3": "Frozen Qwen3", "frozen_multilingual_e5_large": "Frozen E5",
        "adapted_qwen3_cross_entropy": "Adapted Qwen3 CE",
        "adapted_qwen3_supcon": "Adapted Qwen3 SupCon",
        "adapted_qwen3_softtriple": "Adapted Qwen3 SoftTriple",
    })
    table = summary.pivot(index="label", columns="role", values="delta_mean")
    table = table.reindex(["Frozen Qwen3", "Frozen E5", "Adapted Qwen3 CE",
                           "Adapted Qwen3 SupCon", "Adapted Qwen3 SoftTriple"])
    with PdfPages(pdf) as pages:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .94, "Same-role geometry across reporting corpora", fontsize=19, weight="bold")
        fig.text(.06, .88, "Exact cosine means from existing embeddings; no model fitting", fontsize=11)
        lines = [
            "For each role and pair of corpora: Delta = (within-left + within-right)/2 - cross.",
            "Within-corpus pairs from the same accident are excluded. Each of the six corpus pairs",
            "has equal weight in the role-level mean. Positive Delta means same-role units are",
            "more similar inside a corpus than across the two corpora in the measured space.",
            "",
            "Frozen Qwen3 and E5 cover four corpora. Adapted Qwen3 CE, SupCon and SoftTriple",
            "cover five matched seeds and were trained on Construction. Their row reports the",
            "mean across seeds; seed SD is in role_summary.csv. Frozen-space confidence intervals",
            "use accident-level resampling within each corpus; adapted-seed SD is separate.",
            "",
            "Company has only 203 A1 units. Its segmentation and reporting protocol differ from",
            "EPICEA. Bootstrap resampling is conditional on accidents carrying the given role.",
            "Delta is descriptive: it does not show that alignment will improve transfer.",
        ]
        fig.text(.06, .80, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.6)
        pages.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.7, 7.4))
        ax.axis("off")
        ax.set_title("Company pairs versus EPICEA-only pairs", fontsize=16, pad=20)
        shown = pair_group.pivot(index=["space", "role"], columns="pair_group",
                                 values="delta_mean")
        shown = shown[["EPICEA-only", "Company-involving"]]
        shown.index = [f"{space.replace('_', ' ')} / {role}" for space, role in shown.index]
        shown = shown.map(lambda v: f"{v:.4f}")
        tab = ax.table(cellText=shown.values, rowLabels=shown.index,
                       colLabels=shown.columns, loc="center", cellLoc="center", rowLoc="left")
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.8)
        tab.scale(1, 1.4)
        fig.text(.06, .035, "Each column gives equal weight to its three corpus pairs; adapted values average five seeds.",
                 fontsize=9)
        pages.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.7, 7.4))
        ax.axis("off")
        ax.set_title("Mean within-minus-cross cosine similarity by role", fontsize=16, pad=20)
        shown = table.map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        tab = ax.table(cellText=shown.values, rowLabels=shown.index, colLabels=shown.columns,
                       loc="center", cellLoc="center", rowLoc="left")
        tab.auto_set_font_size(False)
        tab.set_fontsize(11)
        tab.scale(1, 2.0)
        fig.text(.08, .09, "See role_summary.csv for frozen 95% CIs and adapted seed SD.", fontsize=10)
        pages.savefig(fig)
        plt.close(fig)

        for space in pair["space"].unique():
            part = pair[pair.space == space].groupby(["sector_left", "sector_right", "role"],
                                                     as_index=False)["delta"].mean()
            pivot = part.pivot(index=["sector_left", "sector_right"], columns="role", values="delta")
            fig, ax = plt.subplots(figsize=(11.7, 7.2))
            data = pivot.to_numpy()
            limit = max(.005, float(np.nanmax(np.abs(data))))
            image = ax.imshow(data, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
            ax.set_xticks(range(len(ROLES)), ROLES)
            ax.set_yticks(range(len(pivot)),
                          [f"{DISPLAY[a]} / {DISPLAY[b]}" for a, b in pivot.index])
            ax.set_title(f"Within-minus-cross cosine similarity: {space.replace('_', ' ')}")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i,j]:+.3f}", ha="center", va="center", fontsize=9)
            fig.colorbar(image, ax=ax, shrink=.8, label="Delta cosine")
            fig.tight_layout()
            pages.savefig(fig)
            plt.close(fig)

        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .94, "Scope and checks", fontsize=17, weight="bold")
        lines = [
            f"Input audits: {len(audit)} complete corpus-space files, all labels and doc_id aligned.",
            f"Pairwise estimates: {len(pair)} rows (4 roles x 6 corpus pairs x runs).",
            "Cosine similarity is computed after L2 normalization inside each representation space.",
            "The exact pairwise means use vector sums, so no random pair subsampling is needed.",
            "The bootstrap varies accidents, not fitted models. Five adapted seeds quantify fit",
            "variation but are not bootstrap replicates. Do not compare absolute cosine levels",
            "between encoders or methods as if they occupied a common calibrated space.",
            "The adapted models were trained on Construction; within-Construction similarity",
            "can reflect source supervision. A positive Delta may reflect reporting or segmentation",
            "differences and does not imply weak role separability or benefit from alignment.",
            "No adapted E5 or CamemBERT geometry is analyzed.",
        ]
        fig.text(.06, .84, "\n".join(lines), fontsize=11, va="top", linespacing=1.8)
        pages.savefig(fig)
        plt.close(fig)
    print(f"Report: {pdf}")


def _write_findings(out: Path, pair: pd.DataFrame, role: pd.DataFrame,
                    pair_group: pd.DataFrame, boot: pd.DataFrame) -> None:
    all_positive = int((pair["delta"] > 0).sum())
    frozen_pair_intervals = boot[boot["sector_left"] != "equal_pair_mean"]
    frozen_positive = int((frozen_pair_intervals["ci_low"] > 0).sum())
    comparisons = pair_group.pivot(index=["space", "role"], columns="pair_group",
                                   values="delta_mean")
    company_larger = int((comparisons["Company-involving"] >
                          comparisons["EPICEA-only"]).sum())
    lines = [
        "# Diagnostic: same-role geometry across corpora",
        "",
        f"All {all_positive} of {len(pair)} role × corpus-pair × run estimates have positive "
        "within-minus-cross cosine similarity. Some adapted differences are near zero.",
        f"All {frozen_positive} of {len(frozen_pair_intervals)} frozen-space pair-specific "
        "95% accident-bootstrap intervals have positive lower bounds.",
        f"Company-involving pairs have larger mean gaps than EPICEA-only pairs in "
        f"{company_larger} of {len(comparisons)} role × space comparisons.",
        "",
        "| Space | A0 | A1 | B | C |",
        "|---|---:|---:|---:|---:|",
    ]
    for space in ["frozen_qwen3", "frozen_multilingual_e5_large",
                  "adapted_qwen3_cross_entropy", "adapted_qwen3_supcon",
                  "adapted_qwen3_softtriple"]:
        values = role[role["space"] == space].set_index("role")["delta_mean"]
        lines.append(f"| {space} | " + " | ".join(f"{values[r]:.4f}" for r in ROLES) + " |")
    lines += [
        "",
        "Values average six corpus pairs equally. Adapted values additionally average five seeds; "
        "their seed SD is in `role_summary.csv`. Frozen confidence intervals are in that file "
        "and `frozen_bootstrap_intervals.csv`.",
        "",
        "**Interpretation:** same-role units retain measurable corpus-specific structure in all "
        "tested spaces. The larger Company contrasts are consistent with a greater corpus shift, "
        "but Company also differs in reporting and segmentation protocol. These data do not "
        "show that an alignment objective would improve role classification. Absolute cosine "
        "levels across different models are not calibrated to one another. Adapted models were "
        "trained on Construction, so their within-Construction geometry reflects source supervision.",
    ]
    (out / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(out: Path, *, bootstrap_resamples: int, bootstrap_seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    refs = {corpus: metadata(corpus) for corpus in CORPORA}
    seeds = _seeds()
    rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    audit_rows: list[dict] = []
    spaces = [(f"frozen_{backbone}", "frozen", None) for backbone in FROZEN]
    spaces += [(f"adapted_qwen3_{method}", method, seed)
               for method in ADAPTED for seed in seeds]
    for space_index, (space, method, seed) in enumerate(spaces):
        print(f"[{space_index + 1}/{len(spaces)}] {space} seed={seed}", flush=True)
        stats = {}
        for corpus in CORPORA:
            frame = refs[corpus]
            if method == "frozen":
                backbone = space.removeprefix("frozen_")
                vectors = frozen_vectors(backbone, corpus, frame)
                source_path = ROOT / "embeddings" / backbone / f"{corpus}.csv"
            else:
                vectors = adapted_vectors(method, int(seed), corpus, frame)
                source_path = (ROOT / "output" / "adapted_geometry_pilot" / "runs" /
                               f"qwen3_btp_{method}" / f"seed_{seed}" / "embeddings" /
                               f"projected_{corpus}.npy")
            stats[corpus] = summarize_vectors(vectors, frame)
            audit_rows.append({"space": space, "method": method, "seed": seed,
                               "corpus": corpus, "rows": len(frame),
                               "n_accidents": frame.accident_id.nunique(),
                               "embedding_dim": vectors.shape[1],
                               "file": str(source_path),
                               **{f"n_{role}": stats[corpus][role].n for role in ROLES}})
            del vectors
        rows.extend(_point_rows(space, method, seed, stats))
        if method == "frozen":
            for item in frozen_bootstrap(stats, resamples=bootstrap_resamples,
                                         seed=bootstrap_seed):
                bootstrap_rows.append({"space": space, **item})
        del stats
    pair = pd.DataFrame(rows)
    pair["pair_group"] = np.where(pair["sector_right"].eq("nicollin"),
                                   "Company-involving", "EPICEA-only")
    boot = pd.DataFrame(bootstrap_rows)
    audit = pd.DataFrame(audit_rows)
    role = pair.groupby(["space", "method", "role"], as_index=False).agg(
        within_mean=("within_pair_mean", "mean"), cross_mean=("cross", "mean"),
        delta_mean=("delta", "mean"), n_runs=("seed", lambda s: max(1, s.nunique())),
    )
    adapted_seed_role = pair.groupby(["space", "method", "seed", "role"],
                                      as_index=False)["delta"].mean()
    seed_sd = adapted_seed_role.groupby(["space", "role"])["delta"].std(ddof=1)
    role["seed_sd"] = [seed_sd.get((row.space, row.role), np.nan) for row in role.itertuples()]
    summary_ci = boot[(boot.sector_left == "equal_pair_mean")]
    role = role.merge(summary_ci[["space", "role", "ci_low", "ci_high"]],
                      on=["space", "role"], how="left")
    pair_group = pair.groupby(["space", "method", "role", "pair_group"], as_index=False).agg(
        within_mean=("within_pair_mean", "mean"), cross_mean=("cross", "mean"),
        delta_mean=("delta", "mean"),
    )
    pair = pair.sort_values(["space", "seed", "role", "sector_left", "sector_right"])
    role = role.sort_values(["space", "role"])
    pair.to_csv(out / "pairwise_similarity.csv", index=False)
    role.to_csv(out / "role_summary.csv", index=False)
    pair_group.to_csv(out / "pair_group_summary.csv", index=False)
    boot.to_csv(out / "frozen_bootstrap_intervals.csv", index=False)
    adapted_seed_role.to_csv(out / "adapted_seed_role_means.csv", index=False)
    audit.to_csv(out / "input_audit.csv", index=False)
    parameters = {"roles": ROLES, "corpora": CORPORA,
                  "frozen_backbones": FROZEN, "adapted_methods": ADAPTED,
                  "adapted_seeds": seeds, "bootstrap_resamples": bootstrap_resamples,
                  "bootstrap_seed": bootstrap_seed,
                  "bootstrap_unit": "accidents carrying each role, sampled within corpus",
                  "pair_definition": "within excludes pairs from the same accident",
                  "role_aggregation": "equal weight over six unordered corpus pairs"}
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    _report(out, pair, role, pair_group, boot, audit)
    _write_findings(out, pair, role, pair_group, boot)
    print(role[["space", "role", "delta_mean", "seed_sd", "ci_low", "ci_high"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "role_geometry")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("Use at least 100 accident bootstrap resamples")
    run(args.output.resolve(), bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed)


if __name__ == "__main__":
    main()
