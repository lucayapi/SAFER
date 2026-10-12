"""Prepare, fit and analyze the seven-source-mixture Company transfer study.

Preparation uses EPICEA labels only. Company labels are read for a frozen
reference key set and for evaluation after fitting; they never select a model.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scgm_text.data_metadata import load_filtered_metadata
from safer_core.test_corpus import resolve_test_corpus


ROLES = ("A0", "A1", "B", "C")
SOURCES = ("btp", "metallurgie", "caou")
CONDITIONS = {
    "btp": ("btp",),
    "metallurgie": ("metallurgie",),
    "caou": ("caou",),
    "btp_metallurgie": ("btp", "metallurgie"),
    "btp_caou": ("btp", "caou"),
    "metallurgie_caou": ("metallurgie", "caou"),
    "btp_metallurgie_caou": SOURCES,
}
PRED_COLS = ("accident_id", "fact_id", "true_macro", "pred_macro")


def _absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _config(path: str | Path) -> dict:
    with _absolute(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_source(path: Path, domain: str) -> pd.DataFrame:
    frame = load_filtered_metadata(str(path), text_col="sentence")
    if frame.empty or frame["accident_id"].isna().any() or frame["fact_id"].isna().any():
        raise ValueError(f"Missing source rows or identifiers: {domain}")
    frame["accident_id"] = domain + ":" + frame["accident_id"].astype(str)
    keys = frame[["accident_id", "fact_id"]].astype(str)
    if keys.duplicated().any():
        raise ValueError(f"Duplicate (accident_id, fact_id) in {domain}")
    return frame.drop(columns=["label_id", "row_id"], errors="ignore")


def _sample_nested(
    frame: pd.DataFrame,
    budgets: tuple[int, ...],
    reference_mix: np.ndarray,
    *,
    seed: int,
    candidates: int,
) -> dict[int, pd.DataFrame]:
    """Select whole-accident nested prefixes close to each unit budget.

    Candidate orderings are scored only with source counts and labels.
    Exact quotas are impossible when accident lengths differ.
    """
    group_roles = pd.crosstab(frame["accident_id"], frame["pred_label"])
    group_roles = group_roles.reindex(columns=ROLES, fill_value=0)
    groups = group_roles.index.to_numpy(dtype=str)
    counts = group_roles.to_numpy(dtype=np.int32)
    total_available = int(counts.sum())
    if max(budgets) > total_available:
        raise ValueError(f"Requested {max(budgets)} units from {total_available}")
    rng = np.random.default_rng(seed)
    best_score = np.inf
    best_choices: dict[int, np.ndarray] | None = None
    for _ in range(candidates):
        order = rng.permutation(len(groups))
        cumulative = np.cumsum(counts[order], axis=0)
        totals = cumulative.sum(axis=1)
        choices = {}
        score = 0.0
        for budget in budgets:
            near = int(np.searchsorted(totals, budget))
            candidate_positions = [i for i in (near - 1, near) if 0 <= i < len(totals)]
            position = min(candidate_positions, key=lambda i: abs(int(totals[i]) - budget))
            selected_counts = cumulative[position]
            actual = int(totals[position])
            mix = selected_counts / actual
            score += 10.0 * abs(actual - budget) / budget + float(np.abs(mix - reference_mix).sum())
            choices[budget] = order[: position + 1]
        if score < best_score:
            best_score, best_choices = score, choices
    assert best_choices is not None
    samples = {budget: frame.loc[frame["accident_id"].isin(groups[best_choices[budget]])].copy()
               for budget in budgets}
    previous: set[str] = set()
    for budget in sorted(budgets):
        current = set(samples[budget]["accident_id"].astype(str))
        if not previous.issubset(current):
            raise AssertionError("Nested accident samples are not nested")
        previous = current
    return samples


def _key_table(frame: pd.DataFrame) -> pd.DataFrame:
    key = frame[["accident_id", "fact_id", "pred_label"]].copy()
    key.columns = ["accident_id", "fact_id", "true_macro"]
    key["accident_id"] = key["accident_id"].astype(str)
    key["fact_id"] = key["fact_id"].astype(str)
    if key[["accident_id", "fact_id"]].duplicated().any():
        raise ValueError("Duplicate target keys")
    return key.sort_values(["accident_id", "fact_id"]).reset_index(drop=True)


def prepare(cfg: dict, run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    registry_target = resolve_test_corpus("nicollin", anchor=ROOT, require_files=True).data_csv
    if _absolute(cfg["target_file"]).resolve() != registry_target.resolve():
        raise ValueError("Configured target differs from the nicollin test registry")
    sources = {name: _read_source(_absolute(cfg["source_files"][name]), name) for name in SOURCES}
    target = load_filtered_metadata(str(_absolute(cfg["target_file"])), text_col="sentence")
    reference = _key_table(target)
    run_dir.mkdir(parents=True)
    (run_dir / "inputs").mkdir()
    reference.to_csv(run_dir / "target_keys.csv", index=False)
    mix_counts = sum((sources[s]["pred_label"].value_counts().reindex(ROLES, fill_value=0)
                      for s in SOURCES))
    reference_mix = mix_counts.to_numpy(dtype=float) / float(mix_counts.sum())
    budget = int(cfg["controlled_units"])
    if budget % 6:
        raise ValueError("controlled_units must be divisible by 6")
    per_source_budgets = (budget // 3, budget // 2, budget)
    sample_rows: list[dict] = []
    tasks: list[dict] = []
    seeds = [int(s) for s in cfg["training_seeds"]]
    for arm, draws in (("controlled", int(cfg["controlled_draws"])), ("all_data", 1)):
        for draw in range(draws):
            if arm == "controlled":
                samples = {
                    source: _sample_nested(
                        sources[source], per_source_budgets, reference_mix,
                        seed=20261012 + 1009 * draw + 37 * i,
                        candidates=int(cfg["sample_candidates"]),
                    )
                    for i, source in enumerate(SOURCES)
                }
            for condition, domains in CONDITIONS.items():
                pieces = []
                for domain in domains:
                    quota = budget // len(domains) if arm == "controlled" else len(sources[domain])
                    piece = samples[domain][quota] if arm == "controlled" else sources[domain]
                    pieces.append(piece)
                    counts = piece["pred_label"].value_counts().reindex(ROLES, fill_value=0)
                    max_role_deviation_pp = 100.0 * float(
                        np.max(np.abs(counts.to_numpy(dtype=float) / len(piece) - reference_mix))
                    )
                    if arm == "controlled" and max_role_deviation_pp > 5.0:
                        raise ValueError(
                            f"Source role mix exceeds 5 percentage points from reference: "
                            f"{domain}, draw={draw}, {max_role_deviation_pp:.2f} pp"
                        )
                    sample_rows.append({
                        "arm": arm, "draw": draw, "condition": condition, "domain": domain,
                        "requested_units": quota, "actual_units": len(piece),
                        "accidents": piece["accident_id"].nunique(),
                        "max_role_share_deviation_pp": max_role_deviation_pp,
                        **{f"n_{role}": int(counts[role]) for role in ROLES},
                    })
                combined = pd.concat(pieces, ignore_index=True)
                if combined[["accident_id", "fact_id"]].astype(str).duplicated().any():
                    raise AssertionError(f"Duplicate mixed-source keys: {condition}")
                if arm == "controlled" and abs(len(combined) - budget) > 50:
                    raise ValueError(f"Unit budget deviates by more than 50: {condition} draw={draw}")
                input_path = run_dir / "inputs" / f"{arm}_{condition}_draw{draw}.csv"
                combined.to_csv(input_path, index=False)
                for seed in seeds:
                    tasks.append({
                        "task_id": len(tasks), "arm": arm, "condition": condition,
                        "draw": draw, "seed": seed, "source_csv": str(input_path),
                        "run_dir": str(run_dir / "runs" / arm / condition / f"draw_{draw}_seed_{seed}"),
                    })
    pd.DataFrame(sample_rows).to_csv(run_dir / "source_samples.csv", index=False)
    pd.DataFrame(tasks).to_csv(run_dir / "tasks.csv", index=False)
    (run_dir / "campaign_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "source_reference_mix.json").write_text(
        json.dumps(dict(zip(ROLES, reference_mix.tolist())), indent=2), encoding="utf-8"
    )
    expected = len(CONDITIONS) * (int(cfg["controlled_draws"]) + 1) * len(seeds)
    if len(tasks) != expected:
        raise AssertionError(f"Expected {expected} tasks, got {len(tasks)}")
    print(f"Prepared {len(tasks)} fits, {len(reference)} Company units, outputs: {run_dir}")


def _read_tasks(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "tasks.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    tasks = pd.read_csv(path)
    if tasks["task_id"].tolist() != list(range(len(tasks))):
        raise ValueError("Noncontiguous task IDs")
    return tasks


def _aligned_predictions(pred_path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    frame = pd.read_csv(pred_path, dtype={"accident_id": str, "fact_id": str})
    if not set(PRED_COLS).issubset(frame.columns):
        raise ValueError(f"Missing prediction columns: {pred_path}")
    frame = frame[list(PRED_COLS)].copy()
    if frame[["accident_id", "fact_id"]].duplicated().any():
        raise ValueError(f"Duplicate target predictions: {pred_path}")
    frame = frame.sort_values(["accident_id", "fact_id"]).reset_index(drop=True)
    if len(frame) != len(reference) or not frame[list(PRED_COLS[:3])].equals(reference):
        raise ValueError(f"Prediction keys/labels do not match target reference: {pred_path}")
    if not frame["pred_macro"].isin(ROLES).all():
        raise ValueError(f"Invalid predicted role: {pred_path}")
    return frame


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["true_macro"].to_numpy()
    p = frame["pred_macro"].to_numpy()
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=ROLES, average="macro", zero_division=0)),
    }
    precision = precision_score(y, p, labels=ROLES, average=None, zero_division=0)
    recall = recall_score(y, p, labels=ROLES, average=None, zero_division=0)
    for idx, role in enumerate(ROLES):
        metrics[f"{role.lower()}_precision"] = float(precision[idx])
        metrics[f"{role.lower()}_recall"] = float(recall[idx])
    return metrics


def fit(cfg: dict, run_dir: Path, task_id: int) -> None:
    from replication.runner import set_global_training_seed
    from supervised_macro_ft.train_runner import run_supervised_macro_ft_training

    tasks = _read_tasks(run_dir)
    if not 0 <= task_id < len(tasks):
        raise IndexError(task_id)
    task = tasks.iloc[task_id].to_dict()
    out = Path(task["run_dir"])
    pred_path = out / "predictions" / "predictions_nicollin.csv"
    done_path = out / "done.json"
    reference = pd.read_csv(run_dir / "target_keys.csv", dtype={"accident_id": str, "fact_id": str})
    if done_path.is_file():
        _aligned_predictions(pred_path, reference)
        print(f"Verified completed task {task_id}: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    if pred_path.exists():
        pred_path.unlink()  # Prevent a stale partial run from passing validation.
    seed = int(task["seed"])
    set_global_training_seed(seed)
    with _absolute(cfg["base_model_config"]).open(encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg["source_corpus"] = f"multisource_{task['condition']}"
    model_cfg["method_name"] = "multisource_ce"
    model_cfg["test_corpora"] = ["nicollin"]
    model_cfg["data"].update({"data_csv": task["source_csv"], "text_col": "sentence"})
    model_cfg["model"].update({
        "backbone_name": cfg["backbone_name"], "backbone_trainable": True,
        "train_last_n_layers": None, "use_projector": False, "projection": None,
        "n_classes": 4, "class_weight": "balanced", "oversampling": False,
        "max_seq_length": int(cfg["max_seq_length"]), "pooling": "mean",
        "gradient_checkpointing": True, "cache_backbone_embeddings": False,
        "standardize_backbone": False,
    })
    model_cfg["training"].update({
        "skip_cross_validation": True, "seed": seed, "epochs": int(cfg["epochs"]),
        "batch_size": int(cfg["batch_size"]), "lr_backbone": float(cfg["lr_backbone"]),
        "lr_head": float(cfg["lr_head"]), "weight_decay": float(cfg["weight_decay"]),
        "use_amp": False, "final_epochs_from_cv": False,
    })
    model_cfg["exports"].update({"save_btp_embeddings": False, "save_checkpoint": False})
    (out / "fit_config.json").write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")
    run_supervised_macro_ft_training(None, cfg=model_cfg, output_dir_override=out)
    predictions = _aligned_predictions(pred_path, reference)
    # Keep the target-level evidence while avoiding repeated raw narrative exports.
    predictions.to_csv(pred_path, index=False)
    for generated in (out / "embeddings", out / "transfer"):
        if generated.exists():
            if not generated.resolve().is_relative_to(out.resolve()):
                raise ValueError(f"Unsafe generated output path: {generated}")
            shutil.rmtree(generated)
    source_pred = out / "predictions" / f"predictions_{model_cfg['source_corpus']}.csv"
    if source_pred.exists():
        source_pred.unlink()
    metrics = {**{k: task[k] for k in ("task_id", "arm", "condition", "draw", "seed")},
               **_metrics(predictions), "n_target": len(predictions)}
    (out / "target_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    done_path.write_text(json.dumps({"task_id": task_id, "predictions": str(pred_path)}), encoding="utf-8")
    print(json.dumps(metrics))


def _run_tp_by_accident(frame: pd.DataFrame, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    group = pd.Categorical(frame["accident_id"], categories=group_ids).codes
    if (group < 0).any():
        raise ValueError("Unknown accident in predictions")
    true = frame["true_macro"].to_numpy()
    pred = frame["pred_macro"].to_numpy()
    tp = np.zeros((n_groups, len(ROLES)), dtype=np.float64)
    for role_idx, role in enumerate(ROLES):
        np.add.at(tp[:, role_idx], group, ((true == role) & (pred == role)).astype(float))
    return tp


def analyze(cfg: dict, run_dir: Path) -> None:
    tasks = _read_tasks(run_dir)
    reference = pd.read_csv(run_dir / "target_keys.csv", dtype={"accident_id": str, "fact_id": str})
    groups = np.sort(reference["accident_id"].unique())
    group_idx = pd.Categorical(reference["accident_id"], categories=groups).codes
    true_counts = np.zeros((len(groups), len(ROLES)), dtype=np.float64)
    for j, role in enumerate(ROLES):
        np.add.at(true_counts[:, j], group_idx, (reference["true_macro"].to_numpy() == role).astype(float))
    if (true_counts.sum(axis=0) == 0).any():
        raise ValueError("A role has zero target support")
    per_run = []
    tp_sums: dict[tuple[str, str], np.ndarray] = {}
    tp_counts: dict[tuple[str, str], int] = {}
    for task in tasks.to_dict("records"):
        out = Path(task["run_dir"])
        if not (out / "done.json").is_file():
            raise FileNotFoundError(f"Task incomplete: {task['task_id']}")
        frame = _aligned_predictions(out / "predictions" / "predictions_nicollin.csv", reference)
        measured = _metrics(frame)
        saved = json.loads((out / "target_metrics.json").read_text(encoding="utf-8"))
        if abs(measured["balanced_accuracy"] - float(saved["balanced_accuracy"])) > 1e-10:
            raise ValueError(f"Metric mismatch: task {task['task_id']}")
        per_run.append({**{k: task[k] for k in ("task_id", "arm", "condition", "draw", "seed")}, **measured})
        key = (task["arm"], task["condition"])
        tp = _run_tp_by_accident(frame, groups, len(groups))
        tp_sums[key] = tp_sums.get(key, np.zeros_like(tp)) + tp
        tp_counts[key] = tp_counts.get(key, 0) + 1
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    run_df = pd.DataFrame(per_run)
    run_df.to_csv(analysis_dir / "per_run.csv", index=False)
    summaries = []
    for (arm, condition), part in run_df.groupby(["arm", "condition"], sort=False):
        row = {"arm": arm, "condition": condition, "n_runs": len(part),
               "n_draws": part["draw"].nunique()}
        for metric in ("balanced_accuracy", "accuracy", "macro_f1") + tuple(
            f"{role.lower()}_{name}" for role in ROLES for name in ("precision", "recall")
        ):
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_sd_runs"] = part[metric].std(ddof=1)
        draw_means = part.groupby("draw")["balanced_accuracy"].mean()
        row["balanced_accuracy_sd_draw_means"] = draw_means.std(ddof=1) if len(draw_means) > 1 else np.nan
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values(["arm", "condition"])
    summary.to_csv(analysis_dir / "summary.csv", index=False)

    # Paired accident bootstrap: every condition sees the identical target draw.
    n_resamples = int(cfg["bootstrap_resamples"])
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    weights = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)),
                              size=n_resamples).astype(np.float32)
    denominators = weights @ true_counts.astype(np.float32)
    if (denominators == 0).any():
        raise ValueError("A bootstrap sample omitted a target role")
    boot: dict[tuple[str, str], np.ndarray] = {}
    point: dict[tuple[str, str], float] = {}
    for key, tp_sum in tp_sums.items():
        tp_mean = (tp_sum / tp_counts[key]).astype(np.float32)
        boot[key] = ((weights @ tp_mean) / denominators).mean(axis=1)
        point[key] = float((tp_mean.sum(axis=0) / true_counts.sum(axis=0)).mean())
    contrast_rows = []
    for arm in ("controlled", "all_data"):
        baseline = (arm, "btp")
        for condition, domains in CONDITIONS.items():
            key = (arm, condition)
            if key == baseline:
                continue
            for comparator, other_boot, other_point in (
                ("btp", boot[baseline], point[baseline]),
                ("mean_constituent_singles",
                 np.mean([boot[(arm, d)] for d in domains], axis=0),
                 float(np.mean([point[(arm, d)] for d in domains]))),
            ):
                if comparator == "mean_constituent_singles" and len(domains) == 1:
                    continue
                diff = boot[key] - other_boot
                contrast_rows.append({
                    "arm": arm, "condition": condition, "comparator": comparator,
                    "delta_ba": point[key] - other_point,
                    "ci_low": float(np.quantile(diff, 0.025)),
                    "ci_high": float(np.quantile(diff, 0.975)),
                    "bootstrap_resamples": n_resamples,
                })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(analysis_dir / "paired_accident_bootstrap.csv", index=False)
    paired_rows = []
    for (arm, draw, seed), part in run_df.groupby(["arm", "draw", "seed"]):
        values = part.set_index("condition")["balanced_accuracy"]
        if set(values.index) != set(CONDITIONS):
            raise ValueError(f"Incomplete paired runs: {arm}, draw={draw}, seed={seed}")
        for condition, domains in CONDITIONS.items():
            if condition == "btp":
                continue
            paired_rows.append({
                "arm": arm, "draw": draw, "seed": seed, "condition": condition,
                "delta_vs_btp": float(values[condition] - values["btp"]),
                "delta_vs_mean_constituent_singles": (
                    float(values[condition] - values[list(domains)].mean()) if len(domains) > 1 else np.nan
                ),
            })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(analysis_dir / "paired_run_contrasts.csv", index=False)
    sample = pd.read_csv(run_dir / "source_samples.csv")
    samples = sample.groupby(["arm", "condition", "draw"], as_index=False).agg(
        actual_units=("actual_units", "sum"), accidents=("accidents", "sum")
    )
    samples.to_csv(analysis_dir / "sample_sizes.csv", index=False)
    lines = [
        "# Multi-source transfer to Company",
        "",
        f"Qwen3 CE; {len(tasks)} fitted models; Company n={len(reference)} units, "
        f"A1 n={int((reference['true_macro'] == 'A1').sum())}. ",
        "All hyperparameters were fixed before this campaign. Company was used only for evaluation.",
        "Controlled samples comprise whole accidents and approximately 6,000 units per condition; see sample_sizes.csv.",
        "The all-data arm uses every available EPICEA unit. Both arms use the same three training seeds; "
        "the controlled arm additionally varies the source-accident draw five times.",
        "Bootstrap intervals resample Company accidents conditional on fitted models; run and draw SDs are separate.",
        "Pair/triple versus the mean of constituent single-source models is the pre-specified diversity contrast. "
        "The BTP contrast is also supplied for direct reference.",
        "These comparisons describe observed source mixtures; neither an improvement nor a decline isolates a causal "
        "effect of sector diversity because source texts and role mixtures differ.",
        "",
        "## Files",
        "- summary.csv: BA, accuracy, macro-F1, precision/recall for every role, SD across runs and draw means.",
        "- paired_accident_bootstrap.csv: paired BA contrasts with 95% accident-bootstrap intervals.",
        "- paired_run_contrasts.csv: paired BA differences for each draw and training seed.",
        "- per_run.csv: each training run, to inspect seed and draw variation.",
        "- sample_sizes.csv and ../source_samples.csv: actual source sizes and class counts.",
    ]
    (analysis_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analysis complete: {analysis_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "count", "fit", "analyze"))
    parser.add_argument("--config", default="configs/multisource_dg.yaml")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = _config(args.config)
    if args.phase == "prepare":
        prepare(cfg, run_dir)
    elif args.phase == "count":
        print(len(_read_tasks(run_dir)))
    elif args.phase == "fit":
        if args.task_index is None:
            parser.error("fit requires --task-index")
        fit(cfg, run_dir, args.task_index)
    else:
        analyze(cfg, run_dir)


if __name__ == "__main__":
    main()
