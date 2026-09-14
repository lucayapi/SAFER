"""IC bootstrap pour une évaluation cible, au niveau des accidents.

Cette fonction est volontairement indépendante des entraînements. Elle prend les
prédictions déjà produites par chaque modèle et rééchantillonne des accidents
entiers : toutes les unités factuelles tirées avec un ``accident_id`` restent
donc ensemble dans le même échantillon bootstrap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


MACROS: tuple[str, ...] = ("A0", "A1", "B", "C")
SUPPORTED_METRICS: tuple[str, ...] = ("balanced_accuracy", "macro_f1")


def _metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(
            f1_score(y_true, y_pred, labels=list(MACROS), average="macro", zero_division=0)
        )
    raise ValueError(f"Métrique bootstrap non supportée : {metric!r}")


def _prediction_signature(predictions_by_model: Mapping[str, pd.DataFrame], settings: Mapping) -> str:
    digest = hashlib.sha256(json.dumps(dict(settings), sort_keys=True).encode("utf-8"))
    for model in sorted(predictions_by_model):
        frame = predictions_by_model[model]
        digest.update(model.encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes()
        )
        digest.update("\x1e".join(map(str, frame.columns)).encode("utf-8"))
    return digest.hexdigest()


def _validate_predictions(
    predictions_by_model: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str], tuple[str, ...]]:
    if not predictions_by_model:
        raise ValueError("Aucune prédiction fournie pour le bootstrap cible.")
    required = {"accident_id", "true_macro", "pred_macro"}
    raw_frames = list(predictions_by_model.values())
    # Les sorties du projet contiennent normalement fact_id. Si ce n'est pas
    # le cas, sentence reste une clé de contrôle utile; l'ordre complet sert de
    # dernier filet de sécurité pour les anciens exports.
    if all("fact_id" in frame.columns and frame["fact_id"].notna().all() for frame in raw_frames):
        alignment_columns: tuple[str, ...] = ("accident_id", "fact_id")
    elif all("sentence" in frame.columns and frame["sentence"].notna().all() for frame in raw_frames):
        alignment_columns = ("accident_id", "sentence")
    else:
        alignment_columns = ("accident_id",)

    prepared: dict[str, pd.DataFrame] = {}
    reference_keys: pd.DataFrame | None = None
    reference_truth: pd.Series | None = None
    for model, frame in predictions_by_model.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{model} : colonnes requises absentes pour le bootstrap : {sorted(missing)}")
        copy = frame.loc[:, [*alignment_columns, "true_macro", "pred_macro"]].copy().reset_index(drop=True)
        if copy[["accident_id", "true_macro", "pred_macro"]].isna().any().any():
            raise ValueError(f"{model} : accident_id, vérité terrain ou prédiction manquant.")
        copy["accident_id"] = copy["accident_id"].astype(str)
        copy["true_macro"] = copy["true_macro"].astype(str)
        copy["pred_macro"] = copy["pred_macro"].astype(str)
        if copy["accident_id"].str.strip().eq("").any():
            raise ValueError(f"{model} : accident_id vide, bootstrap groupé impossible.")
        # L'ordre des lignes doit être identique : les prédictions sont générées
        # pour le même corpus par les classifieurs comparés.
        keys = copy.loc[:, list(alignment_columns)]
        if reference_keys is None:
            reference_keys, reference_truth = keys, copy["true_macro"]
        elif not keys.equals(reference_keys) or not copy["true_macro"].equals(reference_truth):
            raise ValueError("Les prédictions des modèles ne sont pas alignées sur les mêmes unités cibles.")
        prepared[str(model)] = copy
    reference = next(iter(prepared.values()))
    accidents = sorted(reference["accident_id"].unique().tolist())
    if len(accidents) < 2:
        raise ValueError("Bootstrap cible impossible : au moins deux accident_id sont nécessaires.")
    return reference, accidents, alignment_columns


def bootstrap_target_predictions(
    predictions_by_model: Mapping[str, pd.DataFrame],
    *,
    destination: str | Path,
    n_resamples: int = 2000,
    seed: int = 2027,
    confidence_level: float = 0.95,
    metrics: Sequence[str] = SUPPORTED_METRICS,
    force: bool = False,
) -> pd.DataFrame:
    """Calcule et met en cache les IC percentile de l'évaluation OOD.

    Le fichier ``destination`` est un CSV avec une ligne par modèle et métrique.
    Son manifeste voisin garantit que le cache correspond aux prédictions et aux
    paramètres de bootstrap actuels.
    """
    if n_resamples < 100:
        raise ValueError("n_resamples doit être >= 100 pour un IC bootstrap exploitable.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level doit être strictement entre 0 et 1.")
    metrics = tuple(map(str, metrics))
    unsupported = set(metrics) - set(SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"Métriques bootstrap non supportées : {sorted(unsupported)}")

    reference, accidents, alignment_columns = _validate_predictions(predictions_by_model)
    destination = Path(destination)
    settings = {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "metrics": list(metrics),
        "resampling_unit": "accident_id",
    }
    signature = _prediction_signature(predictions_by_model, settings)
    manifest_path = destination.with_name(f"{destination.stem}_manifest.json")
    if not force and destination.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") == signature:
                return pd.read_csv(destination)
        except (OSError, ValueError):
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    truth = reference["true_macro"].to_numpy()
    rows_by_accident = {
        accident: np.flatnonzero(reference["accident_id"].to_numpy() == accident)
        for accident in accidents
    }
    rng = np.random.default_rng(seed)
    samples = [
        np.concatenate([rows_by_accident[accident] for accident in rng.choice(accidents, size=len(accidents), replace=True)])
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict] = []
    for model, frame in predictions_by_model.items():
        prediction = frame["pred_macro"].astype(str).to_numpy()
        for metric in metrics:
            values = np.asarray([_metric(truth[index], prediction[index], metric) for index in samples])
            rows.append({
                "model": str(model),
                "metric": metric,
                "point_estimate": _metric(truth, prediction, metric),
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(np.quantile(values, alpha)),
                "ci_high": float(np.quantile(values, 1.0 - alpha)),
                "confidence_level": float(confidence_level),
                "n_resamples": int(n_resamples),
                "bootstrap_seed": int(seed),
                "n_units": int(len(reference)),
                "n_accidents": int(len(accidents)),
                "resampling_unit": "accident_id",
                "alignment_key": "+".join(alignment_columns),
            })
    result = pd.DataFrame(rows)
    result.to_csv(destination, index=False)
    manifest_path.write_text(json.dumps({"signature": signature, **settings}, indent=2), encoding="utf-8")
    return result


def plot_target_bootstrap_intervals(
    intervals: pd.DataFrame, *, destination: str | Path, title: str
) -> Path:
    """Trace les scores ponctuels et IC pour les modèles d'un même corpus cible."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    required = {"model", "metric", "point_estimate", "ci_low", "ci_high"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"Tableau d'IC incomplet : {sorted(missing)}")
    metrics = list(intervals["metric"].drop_duplicates())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    colors = sns.color_palette("deep", n_colors=max(len(intervals["model"].unique()), 1))
    for axis, metric in zip(axes[0], metrics):
        subset = intervals.loc[intervals["metric"].eq(metric)].reset_index(drop=True)
        means = subset["point_estimate"].to_numpy(float)
        low = subset["ci_low"].to_numpy(float)
        high = subset["ci_high"].to_numpy(float)
        positions = np.arange(len(subset))
        axis.errorbar(positions, means, yerr=[means - low, high - means], fmt="none", ecolor="#333333", capsize=4)
        axis.scatter(positions, means, color=colors[:len(subset)], zorder=3)
        axis.set_xticks(positions, subset["model"], rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("score")
        axis.set_title(metric.replace("_", " "))
    fig.suptitle(title)
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return destination
