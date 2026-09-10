"""Agrégation multi-seeds et bootstrap apparié au niveau accident_id."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from replication.runner import output_root, run_dir_for, training_seeds

LABELS: Tuple[str, ...] = ("A0", "A1", "B", "C")
METRIC_FUNCTIONS = {
    "balanced_accuracy": lambda truth, pred: float(balanced_accuracy_score(truth, pred)),
    "macro_f1": lambda truth, pred: float(f1_score(truth, pred, labels=list(LABELS), average="macro", zero_division=0)),
}


def _key_columns(frame: pd.DataFrame) -> List[str]:
    if "fact_id" in frame.columns:
        return ["accident_id", "fact_id"]
    return ["accident_id", "sentence"]


def _load_prediction(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prédictions requises absentes : {path}")
    frame = pd.read_csv(path)
    required = {"accident_id", "true_macro", "pred_macro"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} ne contient pas {sorted(missing)}")
    keys = _key_columns(frame)
    if frame.duplicated(keys).any():
        raise ValueError(f"Clé prédiction non unique dans {path}: {keys}")
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _aligned_predictions(
    config: Mapping[str, Any],
    *,
    model_id: str,
    corpus: str,
) -> List[pd.DataFrame]:
    frames = [
        _load_prediction(run_dir_for(config, model_id, seed) / "predictions" / f"predictions_{corpus}.csv")
        for seed in training_seeds(config)
    ]
    reference = frames[0]
    keys = _key_columns(reference)
    ref_key = reference[keys].astype(str).agg("\u241f".join, axis=1)
    for frame in frames[1:]:
        frame_keys = _key_columns(frame)
        if frame_keys != keys:
            raise ValueError(f"Colonnes clé différentes entre seeds pour {model_id}/{corpus}.")
        candidate_key = frame[keys].astype(str).agg("\u241f".join, axis=1)
        if not candidate_key.equals(ref_key):
            raise ValueError(f"Les unités diffèrent entre seeds pour {model_id}/{corpus}.")
        if not frame["true_macro"].astype(str).equals(reference["true_macro"].astype(str)):
            raise ValueError(f"Les vérités terrain diffèrent entre seeds pour {model_id}/{corpus}.")
    return frames


def _score(truth: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    try:
        return METRIC_FUNCTIONS[metric](truth, prediction)
    except KeyError as exc:
        raise ValueError(f"Métrique bootstrap inconnue : {metric}") from exc


def _percentile_interval(values: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def analyze_replications(config: Mapping[str, Any], *, destination: str | Path | None = None) -> Dict[str, Path]:
    """Produit les IC bootstrap et comparaisons appariées à partir des prédictions."""
    bootstrap = dict(config.get("bootstrap") or {})
    n_resamples = int(bootstrap.get("n_resamples", 2000))
    if n_resamples < 100:
        raise ValueError("bootstrap.n_resamples doit être >= 100.")
    confidence_level = float(bootstrap.get("confidence_level", 0.95))
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap.confidence_level doit être entre 0 et 1.")
    metrics = [str(metric) for metric in bootstrap.get("metrics", ["balanced_accuracy", "macro_f1"])]
    unknown = set(metrics) - set(METRIC_FUNCTIONS)
    if unknown:
        raise ValueError(f"Métriques bootstrap non supportées : {sorted(unknown)}")
    model_ids = [str(key) for key in config["models"]]
    corpora = [str(value) for value in config["training"]["test_corpora"]]
    seeds = training_seeds(config)
    destination = Path(destination) if destination is not None else output_root(config).parent / "replication_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(bootstrap.get("seed", 2027)))

    score_rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []
    difference_rows: List[Dict[str, Any]] = []

    for corpus in corpora:
        per_model = {model: _aligned_predictions(config, model_id=model, corpus=corpus) for model in model_ids}
        reference = per_model[model_ids[0]][0]
        keys = _key_columns(reference)
        reference_keys = reference[keys].astype(str).agg("␟".join, axis=1)
        for model_id, frames in per_model.items():
            candidate = frames[0]
            candidate_keys = candidate[keys].astype(str).agg("␟".join, axis=1)
            if not candidate_keys.equals(reference_keys):
                raise ValueError(f"Les unités diffèrent entre méthodes pour {corpus}: {model_id}.")
            if not candidate["true_macro"].astype(str).equals(reference["true_macro"].astype(str)):
                raise ValueError(f"Les vérités terrain diffèrent entre méthodes pour {corpus}: {model_id}.")
        truth = reference["true_macro"].astype(str).to_numpy()
        accidents = reference["accident_id"].astype(str).to_numpy()
        unique_accidents = np.unique(accidents)
        if len(unique_accidents) < 2:
            raise ValueError(f"Bootstrap impossible pour {corpus}: moins de deux accidents.")
        group_rows = {accident: np.flatnonzero(accidents == accident) for accident in unique_accidents}
        sampled_indices = [
            np.concatenate([group_rows[accident] for accident in rng.choice(unique_accidents, size=len(unique_accidents), replace=True)])
            for _ in range(n_resamples)
        ]

        bootstrap_scores: Dict[Tuple[str, str], np.ndarray] = {}
        full_seed_scores: Dict[Tuple[str, str], np.ndarray] = {}
        for model_id, frames in per_model.items():
            predictions = [frame["pred_macro"].astype(str).to_numpy() for frame in frames]
            for metric in metrics:
                seed_scores = np.asarray([_score(truth, prediction, metric) for prediction in predictions])
                full_seed_scores[(model_id, metric)] = seed_scores
                for seed, score in zip(seeds, seed_scores):
                    score_rows.append({
                        "corpus": corpus, "model": model_id, "training_seed": int(seed),
                        "metric": metric, "score": float(score), "n_units": int(len(truth)),
                        "n_accidents": int(len(unique_accidents)),
                    })
                values = np.empty(n_resamples, dtype=float)
                for index, rows in enumerate(sampled_indices):
                    values[index] = float(np.mean([_score(truth[rows], prediction[rows], metric) for prediction in predictions]))
                bootstrap_scores[(model_id, metric)] = values
                lo, hi = _percentile_interval(values, confidence_level)
                ci_rows.append({
                    "corpus": corpus, "model": model_id, "metric": metric,
                    "mean_across_seeds": float(seed_scores.mean()),
                    "std_across_seeds": float(seed_scores.std(ddof=1)) if len(seed_scores) > 1 else 0.0,
                    "ci_low": lo, "ci_high": hi, "confidence_level": confidence_level,
                    "n_resamples": n_resamples, "n_seeds": len(seeds),
                    "n_units": int(len(truth)), "n_accidents": int(len(unique_accidents)),
                    "resampling_unit": "accident_id",
                })
        for model_a, model_b in itertools.combinations(model_ids, 2):
            for metric in metrics:
                differences = bootstrap_scores[(model_a, metric)] - bootstrap_scores[(model_b, metric)]
                lo, hi = _percentile_interval(differences, confidence_level)
                difference_rows.append({
                    "corpus": corpus, "metric": metric, "model_a": model_a, "model_b": model_b,
                    "difference_a_minus_b": float(
                        np.mean(full_seed_scores[(model_a, metric)] - full_seed_scores[(model_b, metric)])
                    ),
                    "bootstrap_mean_difference": float(differences.mean()), "ci_low": lo, "ci_high": hi,
                    "confidence_level": confidence_level, "n_resamples": n_resamples,
                    "paired_on": "same accident_id resamples and same training seeds",
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                })

    paths = {
        "seed_scores": destination / "seed_scores.csv",
        "bootstrap_ci": destination / "bootstrap_ci.csv",
        "paired_differences": destination / "paired_bootstrap_differences.csv",
        "manifest": destination / "analysis_manifest.json",
    }
    pd.DataFrame(score_rows).to_csv(paths["seed_scores"], index=False)
    pd.DataFrame(ci_rows).to_csv(paths["bootstrap_ci"], index=False)
    pd.DataFrame(difference_rows).to_csv(paths["paired_differences"], index=False)
    paths["manifest"].write_text(json.dumps({
        "seeds": seeds, "n_resamples": n_resamples, "confidence_level": confidence_level,
        "metrics": metrics, "corpora": corpora, "models": model_ids,
        "resampling_unit": "accident_id",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_intervals(pd.DataFrame(ci_rows), destination / "bootstrap_intervals.png")
    return paths


def _plot_intervals(frame: pd.DataFrame, destination: Path) -> None:
    """Figure légère, utile pour le rapport; l'analyse reste valide sans Matplotlib."""
    if frame.empty:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    sns.set_theme(style="whitegrid")
    metrics = list(frame["metric"].drop_duplicates())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        subset = frame[frame["metric"] == metric].copy()
        labels = [f"{row.corpus}\n{row.model}" for row in subset.itertuples()]
        means = subset["mean_across_seeds"].to_numpy()
        low = subset["ci_low"].to_numpy()
        high = subset["ci_high"].to_numpy()
        positions = np.arange(len(subset))
        axis.errorbar(positions, means, yerr=[means - low, high - means], fmt="o", capsize=4, color="#386cb0")
        axis.set_xticks(positions, labels, rotation=45, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(metric.replace("_", " "))
        axis.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
