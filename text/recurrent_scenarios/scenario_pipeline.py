"""Auditable implementation of the recurrent-accident protocol.

Workflow:
1. theme discovery — UMAP--HDBSCAN candidates per role, DBCV / ``S_R``;
2. Pareto screening on (``S_R``, DBCV), then geometric knee-point selection;
3. UMAP seed sensitivity on the selected partition;
4. topic dictionary / labels (results notebook) on the frozen partition;
5. corpus-specific latent BN (Structural EM with ``Z``) and constrained MPE scenarios.

Independent from SCGM / contrastive / BERTopic training pipelines. Inputs are the
annotated unit table and frozen Qwen embedding export.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

ROLES = ("A0", "A1", "B", "C")
ROLE_RANK = {role: index for index, role in enumerate(ROLES)}
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
PARAMETER_KEYS = (
    "umap_n_neighbors",
    "umap_n_components",
    "umap_min_dist",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "hdbscan_cluster_selection_method",
)


def _log_progress(message: str) -> None:
    print(f"[theme-discovery] {message}", flush=True)


@dataclass
class PreparedData:
    units: pd.DataFrame
    embeddings: np.ndarray
    input_summary: pd.DataFrame


@dataclass
class PartitionResult:
    role: str
    assignments: pd.DataFrame
    topics: pd.DataFrame
    edges: pd.DataFrame
    replications: pd.DataFrame


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_n_workers(config: Mapping[str, Any]) -> int:
    """Resolve the outer parallel worker count from the allocated CPU budget.

    Slurm exposes the allocation through ``SLURM_CPUS_PER_TASK``.  Outside
    Slurm, the local CPU count is used.  One core is deliberately reserved for
    the parent process, logging and filesystem writes.
    """
    parallel_cfg = config.get("parallel", {})
    if not bool(parallel_cfg.get("enabled", True)):
        return 1
    configured = parallel_cfg.get("n_workers", "auto")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    available = int(allocated) if allocated else int(os.cpu_count() or 1)
    workers = available - 1 if configured in (None, "", "auto") else int(configured)
    max_workers = parallel_cfg.get("max_workers")
    if max_workers not in (None, ""):
        workers = min(workers, int(max_workers))
    return max(1, workers)


def _bn_parallel_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return BN-specific parallel settings, falling back to global settings."""
    bayesian_networks = config.get("bayesian_networks", {})
    nested = bayesian_networks.get("parallel", {}) if isinstance(bayesian_networks, Mapping) else {}
    if isinstance(nested, Mapping) and nested:
        return nested
    return config.get("parallel", {})


def resolve_bn_n_workers(config: Mapping[str, Any]) -> int:
    """Resolve the worker count for independent latent-BN fits."""
    bayesian_networks = config.get("bayesian_networks", {})
    has_nested_settings = isinstance(bayesian_networks, Mapping) and bool(bayesian_networks.get("parallel"))
    if not has_nested_settings and "parallel" not in config:
        return 1
    worker_config = dict(config)
    worker_config["parallel"] = dict(_bn_parallel_config(config))
    return resolve_n_workers(worker_config)


def _parallel_map(
    function: Any,
    tasks: Sequence[Any],
    config: Mapping[str, Any],
    *,
    progress_label: str | None = None,
) -> list[Any]:
    """Execute independent clustering tasks with bounded outer parallelism."""
    if not tasks:
        return []
    workers = resolve_n_workers(config)
    if workers <= 1:
        iterator: Iterable[Any] = (function(task) for task in tasks)
    else:
        iterator = None
    try:
        from joblib import Parallel, delayed
    except ImportError as error:
        raise ImportError("joblib est requis lorsque le parallélisme est activé avec plusieurs workers.") from error
    parallel_cfg = config.get("parallel", {})
    backend = str(parallel_cfg.get("backend", "loky"))
    if workers > 1:
        parallel_kwargs: dict[str, Any] = {
            "n_jobs": workers,
            "backend": backend,
            "batch_size": 1,
            "verbose": 0,
            "return_as": "generator",
        }
        iterator = Parallel(
            **parallel_kwargs,
        )(
            delayed(function)(task) for task in tasks
        )
    if progress_label and config.get("validation", config.get("pareto", {})).get("show_progress", True):
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                iterator,
                total=len(tasks),
                desc=progress_label,
                unit="task",
                file=sys.stdout,
                leave=True,
            )
        except ImportError:
            pass
    return list(iterator)


def select_dataset_config(config: Mapping[str, Any], dataset_id: str | None = None) -> dict[str, Any]:
    """Select one registered corpus without changing the source YAML file."""
    selected = json.loads(json.dumps(config))
    data_cfg = selected.setdefault("data", {})
    chosen_id = str(dataset_id or data_cfg.get("dataset_id", "caou")).strip().lower()
    registry = data_cfg.get("dataset_registry", {})
    if chosen_id not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Dataset inconnu {chosen_id!r}. Disponibles : {available}")
    data_cfg["dataset_id"] = chosen_id
    data_cfg["units_path"] = registry[chosen_id]["units_path"]
    data_cfg["embeddings_path"] = registry[chosen_id]["embeddings_path"]
    return selected


def resolve_config_paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    base = config_path.parent.resolve()
    for section, key in (
        ("data", "units_path"),
        ("data", "embeddings_path"),
        ("data", "output_dir"),
        ("topics", "stopwords_file"),
    ):
        value = resolved.get(section, {}).get(key)
        if value:
            value = str(value).format(dataset_id=resolved.get("data", {}).get("dataset_id", "custom"))
            path = Path(value)
            if path.is_absolute() and not path.is_file() and key in {"units_path", "embeddings_path"}:
                registry = resolved.get("data", {}).get("dataset_registry", {})
                dataset_id = str(resolved.get("data", {}).get("dataset_id", ""))
                fallback = registry.get(dataset_id, {}).get(
                    "units_path" if key == "units_path" else "embeddings_path"
                )
                if fallback:
                    path = (base / str(fallback)).resolve()
            elif not path.is_absolute():
                path = (base / value).resolve()
            resolved[section][key] = str(path)
    return resolved


_BN_CONFIG_KEYS_FROM_RESOLVED = frozenset({
    "min_theme_support_count",
    "d_max",
    "include_all_retained_factors",
    "latent_scope",
    "alpha",
    "probability_floor",
})


def load_bn_analysis_config(
    config_path: Path,
    dataset_id: str,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Load BN notebook config with portable data paths.

    ``config_resolved.yaml`` from a prior discovery run may contain machine-specific
    absolute paths and an outdated K grid (without K=1). Data paths and the BN
    selection grid always come from ``config.yaml``; only a small allow-list of
    discovery-tuned keys may be merged from ``config_resolved.yaml``.
    """

    config = resolve_config_paths(
        select_dataset_config(load_yaml_config(config_path), dataset_id),
        config_path,
    )
    if run_dir is None:
        return config
    resolved_path = run_dir / "config_resolved.yaml"
    if resolved_path.is_file():
        stored = load_yaml_config(resolved_path)
        for section in ("validation", "screening", "parallel", "runtime"):
            if section in stored:
                config[section] = stored[section]
        if "bayesian_networks" in stored:
            bn = dict(config.get("bayesian_networks", {}))
            for key, value in stored["bayesian_networks"].items():
                if key in _BN_CONFIG_KEYS_FROM_RESOLVED:
                    bn[key] = value
            config["bayesian_networks"] = bn
        if "topics" in stored:
            topics = dict(config.get("topics", {}))
            for key, value in stored["topics"].items():
                if key != "stopwords_file":
                    topics[key] = value
            config["topics"] = topics
    latent_states = config.get("bayesian_networks", {}).get("latent_states", [])
    if 1 not in {int(value) for value in latent_states}:
        warnings.warn(
            "K=1 is missing from latent_states; the BIC plot cannot compare against a homogeneous reference model.",
            RuntimeWarning,
            stacklevel=2,
        )
    return config


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def load_units(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    path = Path(data_cfg["units_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Table des unités introuvable : {path}")
    frame = _read_table(path)
    required = [data_cfg["accident_id_col"], data_cfg["fact_id_col"], data_cfg["text_col"], data_cfg["role_col"]]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.name}: {missing}")
    frame = frame.copy()
    frame["_row_order"] = np.arange(len(frame), dtype=int)
    frame["_accident_id"] = frame[data_cfg["accident_id_col"]].astype(str)
    frame["_fact_id"] = frame[data_cfg["fact_id_col"]].astype(str)
    frame["_text"] = frame[data_cfg["text_col"]].fillna("").astype(str)
    frame["_role"] = frame[data_cfg["role_col"]].astype(str).str.strip()
    if data_cfg.get("keep_valid_only", True) and data_cfg.get("valid_col") in frame.columns:
        frame = frame.loc[_as_bool(frame[data_cfg["valid_col"]])].copy()
    frame = frame.loc[frame["_role"].isin(ROLES) & frame["_text"].str.strip().ne("")].copy()
    if frame["_fact_id"].duplicated().any():
        duplicates = int(frame["_fact_id"].duplicated().sum())
        raise ValueError(f"fact_id non uniques après filtrage: {duplicates}")
    frame = frame.reset_index(drop=True)
    rows = [
        {"metric": "n_units", "value": int(len(frame))},
        {"metric": "n_accidents", "value": int(frame["_accident_id"].nunique())},
        *[
            {"metric": f"n_units_{role}", "value": int((frame["_role"] == role).sum())}
            for role in ROLES
        ],
        *[
            {"metric": f"n_accidents_{role}", "value": int(frame.loc[frame["_role"] == role, "_accident_id"].nunique())}
            for role in ROLES
        ],
    ]
    return frame, pd.DataFrame(rows)


def load_embeddings(config: Mapping[str, Any], units: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    data_cfg = config["data"]
    path = Path(data_cfg["embeddings_path"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        embeddings = _read_table(path)
        dimension_columns = [column for column in embeddings.columns if str(column).startswith("dim_")]
        if not dimension_columns:
            dimension_columns = [column for column in embeddings.columns if str(column).isdigit()]
        if not dimension_columns:
            raise ValueError(f"Aucune colonne dimensionnelle trouvée dans {path}")
        id_column = "doc_id" if "doc_id" in embeddings.columns else None
        if id_column is not None and embeddings[id_column].astype(str).isin(units["_fact_id"]).all():
            lookup = embeddings.assign(_fact_id=embeddings[id_column].astype(str)).set_index("_fact_id")
            missing = sorted(set(units["_fact_id"]) - set(lookup.index))
            if missing:
                raise ValueError(f"Embeddings absents pour {len(missing)} fact_id")
            matrix = lookup.loc[units["_fact_id"], dimension_columns].to_numpy(dtype=np.float32)
        elif len(embeddings) == len(units):
            matrix = embeddings[dimension_columns].to_numpy(dtype=np.float32)
            raise ValueError("Impossible d'aligner les embeddings et les unités par fact_id ou par ordre.")
    else:
        model_name = str(data_cfg.get("encoder_name", "Qwen/Qwen3-Embedding-0.6B"))
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("sentence-transformers est requis si embeddings_path est absent.") from error
        model = SentenceTransformer(model_name)
        matrix = model.encode(
            units["_text"].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        np.save(cache_dir / "embeddings_encoded.npy", matrix)
    if data_cfg.get("normalize_embeddings", True):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
    if not np.isfinite(matrix).all():
        raise ValueError("Les embeddings contiennent des valeurs non finies.")
    return matrix


def prepare_data(config: Mapping[str, Any], run_dir: Path) -> PreparedData:
    units, summary = load_units(config)
    embeddings = load_embeddings(config, units, run_dir / "embeddings")
    if len(embeddings) != len(units):
        raise ValueError("Le nombre d'embeddings ne correspond pas au nombre d'unités.")
    summary = pd.concat(
        [summary, pd.DataFrame([{"metric": "embedding_dimension", "value": int(embeddings.shape[1])}])],
        ignore_index=True,
    )
    summary.to_csv(run_dir / "audit_input_summary.csv", index=False)
    return PreparedData(units=units, embeddings=embeddings, input_summary=summary)


def _grid_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _to_python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _to_python_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, list):
        return [_to_python_value(item) for item in value]
    return value


def _normalize_hdbscan_min_samples(value: Any, default: int) -> int:
    """Return an integer min_samples value after CSV/YAML round-tripping."""
    value = _to_python_value(value)
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}):
        return int(default)
    return int(value)


def parameter_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the shared UMAP--HDBSCAN candidate grid from ``config["screening"]``."""
    screening_cfg = config["screening"]
    umap_cfg = screening_cfg["umap"]
    hdbscan_cfg = screening_cfg["hdbscan"]
    keys = [
        ("umap_n_neighbors", _grid_values(umap_cfg["n_neighbors"])),
        ("umap_n_components", _grid_values(umap_cfg["n_components"])),
        ("umap_min_dist", _grid_values(umap_cfg["min_dist"])),
        ("hdbscan_min_cluster_size", _grid_values(hdbscan_cfg["min_cluster_size"])),
        ("hdbscan_min_samples", _grid_values(hdbscan_cfg["min_samples"])),
        ("hdbscan_cluster_selection_method", _grid_values(hdbscan_cfg["cluster_selection_method"])),
    ]
    names = [name for name, _ in keys]
    values = [vals for _, vals in keys]
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def _sample_accidents(units: pd.DataFrame, fraction: float, rng: np.random.Generator) -> set[str]:
    accidents = units["_accident_id"].drop_duplicates().to_numpy()
    size = max(1, int(round(len(accidents) * float(fraction))))
    return set(rng.choice(accidents, size=min(size, len(accidents)), replace=False).tolist())



def _fit_cluster_with_embedding(
    texts: Sequence[str],
    embeddings: np.ndarray,
    params: Mapping[str, Any],
    random_state: int,
    config: Mapping[str, Any],
    *,
    return_membership_strength: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one UMAP-HDBSCAN candidate and retain the fitted UMAP coordinates."""
    try:
        import hdbscan
        import umap
    except ImportError as error:
        raise ImportError("umap-learn et hdbscan sont requis pour le clustering.") from error
    if len(embeddings) < 3:
        labels = np.full(len(embeddings), -1, dtype=int)
        reduced = np.asarray(embeddings)
        if return_membership_strength:
            return labels, reduced, np.zeros(len(labels), dtype=np.float32)
        return labels, reduced
    screening_cfg = config["screening"]
    umap_cfg = screening_cfg["umap"]
    hdbscan_cfg = screening_cfg["hdbscan"]
    n_neighbors = min(int(params["umap_n_neighbors"]), max(2, len(embeddings) - 1))
    n_components = min(int(params["umap_n_components"]), max(2, len(embeddings) - 1))
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=float(params["umap_min_dist"]),
        metric=str(umap_cfg.get("metric", "cosine")),
        random_state=int(random_state),
        low_memory=bool(umap_cfg.get("low_memory", True)),
        n_jobs=int(umap_cfg.get("n_jobs", 1)),
    )
    min_cluster_size = min(int(params["hdbscan_min_cluster_size"]), max(2, len(embeddings)))
    min_samples = _normalize_hdbscan_min_samples(params.get("hdbscan_min_samples"), min_cluster_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=str(params["hdbscan_cluster_selection_method"]),
        metric=str(hdbscan_cfg.get("metric", "euclidean")),
        prediction_data=bool(hdbscan_cfg.get("prediction_data", True)),
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    labels = clusterer.fit_predict(reduced_embeddings)
    labels = np.asarray(labels, dtype=int)
    reduced_embeddings = np.asarray(reduced_embeddings)
    if return_membership_strength:
        strengths = np.asarray(
            getattr(clusterer, "probabilities_", np.where(labels >= 0, 1.0, 0.0)),
            dtype=np.float32,
        )
        return labels, reduced_embeddings, strengths
    return labels, reduced_embeddings


def _screening_metrics(
    role_units: pd.DataFrame,
    selected_indices: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    selected_labels = np.asarray(labels, dtype=int)
    valid = selected_labels >= 0
    valid_labels = selected_labels[valid]
    supports = pd.Series(dtype=float)
    if valid.any():
        support_frame = pd.DataFrame({
            "topic": valid_labels,
            "accident_id": role_units.iloc[selected_indices].loc[valid, "_accident_id"].to_numpy(),
        })
        supports = support_frame.groupby("topic")["accident_id"].nunique()
    return {
        "n_units_sampled": int(len(selected_indices)),
        "n_accidents_sampled": int(role_units.iloc[selected_indices]["_accident_id"].nunique()),
        "n_clusters": int(valid_labels.size and len(np.unique(valid_labels))),
        "n_units_noise": int((~valid).sum()),
        "noise_fraction": float((~valid).mean()) if len(selected_labels) else 1.0,
        "coverage": float(valid.mean()) if len(selected_labels) else 0.0,
        "median_accident_support": float(supports.median()) if not supports.empty else 0.0,
        "mean_accident_support": float(supports.mean()) if not supports.empty else 0.0,
        "min_accident_support": int(supports.min()) if not supports.empty else 0,
        "max_accident_support": int(supports.max()) if not supports.empty else 0,
        "n_single_accident_clusters": int((supports == 1).sum()) if not supports.empty else 0,
    }


def _dbcv(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the reference HDBSCAN DBCV score, returning NaN when undefined."""
    valid = np.asarray(labels) >= 0
    if valid.sum() < 3 or np.unique(np.asarray(labels)[valid]).size < 2:
        return float("nan")
    try:
        from hdbscan.validity import validity_index

        score = validity_index(np.asarray(X, dtype=np.float64), np.asarray(labels, dtype=int), metric="euclidean")
        return float(score) if np.isfinite(score) else float("nan")
    except (ImportError, ValueError, RuntimeError, FloatingPointError):
        return float("nan")


def _candidate_metrics(role_units: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
    """Return diagnostics used for both pathology checks and audit tables."""
    metrics = _screening_metrics(role_units, np.arange(len(role_units)), labels)
    valid = np.asarray(labels, dtype=int) >= 0
    sizes = pd.Series(labels[valid]).value_counts() if valid.any() else pd.Series(dtype=int)
    metrics.update({
        "median_cluster_size": float(sizes.median()) if not sizes.empty else 0.0,
        "min_cluster_size_observed": int(sizes.min()) if not sizes.empty else 0,
        "max_cluster_size_observed": int(sizes.max()) if not sizes.empty else 0,
    })
    return metrics


def _evaluate_candidate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Fit and score one full-data candidate in an isolated worker."""
    labels, reduced, membership_strength = _fit_cluster_with_embedding(
        (),
        task["embeddings"],
        task["params"],
        int(task["random_state"]),
        task["config"],
        return_membership_strength=True,
    )
    dbcv_indices = np.arange(len(labels))
    sample_size = task.get("dbcv_sample_size")
    if sample_size is not None and int(sample_size) < len(labels):
        rng = np.random.default_rng(int(task["random_state"]) + 100000)
        dbcv_indices = np.sort(rng.choice(len(labels), int(sample_size), replace=False))
    metrics = _candidate_metrics_from_accident_ids(task["accident_ids"], labels)
    return {
        "index": int(task["index"]),
        "configuration_id": str(task["configuration_id"]),
        "params": dict(task["params"]),
        "labels": labels.astype(np.int32),
        "membership_strength": membership_strength.astype(np.float32),
        "reduced": np.asarray(reduced, dtype=np.float32),
        "dbcv_umap": _dbcv(reduced[dbcv_indices], labels[dbcv_indices]),
        "dbcv_n_units": int(len(dbcv_indices)),
        "random_state": int(task["random_state"]),
        "metrics": metrics,
    }


def _candidate_metrics_from_accident_ids(accident_ids: Sequence[str], labels: np.ndarray) -> dict[str, Any]:
    """Compute candidate diagnostics without serializing the full unit table."""
    selected_labels = np.asarray(labels, dtype=int)
    valid = selected_labels >= 0
    valid_labels = selected_labels[valid]
    accidents = np.asarray(accident_ids, dtype=str)
    supports = pd.Series(dtype=float)
    if valid.any():
        supports = pd.DataFrame({
            "topic": valid_labels,
            "accident_id": accidents[valid],
        }).groupby("topic")["accident_id"].nunique()
    sizes = pd.Series(valid_labels).value_counts() if valid.any() else pd.Series(dtype=int)
    return {
        "n_units_sampled": int(len(selected_labels)),
        "n_accidents_sampled": int(np.unique(accidents).size),
        "n_clusters": int(len(np.unique(valid_labels))) if valid.any() else 0,
        "n_units_noise": int((~valid).sum()),
        "noise_fraction": float((~valid).mean()) if len(selected_labels) else 1.0,
        "coverage": float(valid.mean()) if len(selected_labels) else 0.0,
        "median_accident_support": float(supports.median()) if not supports.empty else 0.0,
        "mean_accident_support": float(supports.mean()) if not supports.empty else 0.0,
        "min_accident_support": int(supports.min()) if not supports.empty else 0,
        "max_accident_support": int(supports.max()) if not supports.empty else 0,
        "n_single_accident_clusters": int((supports == 1).sum()) if not supports.empty else 0,
        "median_cluster_size": float(sizes.median()) if not sizes.empty else 0.0,
        "min_cluster_size_observed": int(sizes.min()) if not sizes.empty else 0,
        "max_cluster_size_observed": int(sizes.max()) if not sizes.empty else 0,
    }


def _evaluate_stability_task(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fit one resampled candidate and return cluster-wise Jaccard rows."""
    selected_indices = np.asarray(task["selected_indices"], dtype=int)
    resampled_labels, _ = _fit_cluster_with_embedding(
        (),
        np.asarray(task["embeddings"])[selected_indices],
        task["params"],
        int(task["random_state"]),
        task["config"],
    )
    reference = np.load(task["reference_path"])
    local_reference = np.asarray(reference)[selected_indices]
    rows = []
    for cluster_label in task["reference_clusters"]:
        rows.append({
            "role": str(task["role"]),
            "configuration_id": str(task["configuration_id"]),
            "repetition": int(task["repetition"]),
            "cluster_label": int(cluster_label),
            "n_reference_units": int((local_reference == cluster_label).sum()),
            "best_jaccard": _best_jaccard(local_reference, resampled_labels, int(cluster_label)),
        })
    return rows


def _configuration_id(role: str, index: int) -> str:
    return f"{role}_cfg_{int(index):03d}"


def _validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve validation settings with backward-compatible ``pareto`` fallback."""
    values = dict(config.get("validation") or config.get("pareto") or {})
    values.setdefault("random_state", int(config.get("random_state", 42)))
    values.setdefault("resampling_fraction", 0.8)
    values.setdefault("n_resampling", 30)
    values.setdefault("dbcv_sample_size", None)
    values.setdefault("selection_metric", "pareto_geometric_knee")
    values.setdefault("show_progress", True)
    seed_cfg = dict(values.get("seed_sensitivity") or {})
    seed_cfg.setdefault("enabled", True)
    seed_cfg.setdefault("seeds", list(range(1, 11)))
    values["seed_sensitivity"] = seed_cfg
    return values


def _discovery_role_dir(output_dir: Path, role: str) -> Path:
    return Path(output_dir) / "discovery" / role


def _resolve_partition_artifact_paths(run_dir: Path, role: str, configuration_id: str) -> tuple[Path, Path]:
    """Prefer frozen selected artifacts, then discovery candidates, then legacy pareto paths."""
    candidates = [
        (
            run_dir / "discovery" / role / "selected" / "labels.npy",
            run_dir / "discovery" / role / "selected" / "membership_strength.npy",
        ),
        (
            run_dir / "discovery" / role / "candidate_partitions" / f"{configuration_id}_labels.npy",
            run_dir / "discovery" / role / "candidate_partitions" / f"{configuration_id}_membership_strength.npy",
        ),
        (
            run_dir / "pareto" / role / "candidate_partitions" / f"{configuration_id}_labels.npy",
            run_dir / "pareto" / role / "candidate_partitions" / f"{configuration_id}_membership_strength.npy",
        ),
    ]
    for labels_path, strength_path in candidates:
        if labels_path.is_file() and strength_path.is_file():
            return labels_path, strength_path
    raise FileNotFoundError(
        f"Artefacts de partition absents pour {role}/{configuration_id} "
        f"sous discovery/ ou pareto/ dans {run_dir}"
    )


def evaluate_candidates(
    role: str,
    units: pd.DataFrame,
    embeddings: np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    reestimate: bool = False,
) -> pd.DataFrame:
    """Fit every declared full-data candidate and persist labels plus DBCV diagnostics."""
    role_mask = units["_role"].eq(role).to_numpy()
    role_units = units.loc[role_mask].reset_index(drop=True)
    role_embeddings = np.asarray(embeddings[role_mask])
    validation_cfg = _validation_config(config)
    role_dir = _discovery_role_dir(output_dir, role)
    candidate_dir = role_dir / "candidate_partitions"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = role_dir / "candidate_metrics.csv"
    expected_random_state = int(validation_cfg["random_state"])
    if metrics_path.is_file() and not reestimate:
        cached = pd.read_csv(metrics_path)
        membership_files_present = all(
            (candidate_dir / f"{configuration_id}_membership_strength.npy").is_file()
            for configuration_id in cached.get("configuration_id", pd.Series(dtype=str)).astype(str)
        )
        cache_has_expected_seed = (
            "random_state" in cached.columns
            and cached["random_state"].astype(int).eq(expected_random_state).all()
        )
        if {"configuration_id", "dbcv_umap"}.issubset(cached.columns) and membership_files_present and cache_has_expected_seed:
            _log_progress(f"[{role}] candidats: cache réutilisé ({metrics_path})")
            return cached.loc[:, ~cached.columns.str.contains("semantic", case=False)]
    plan = parameter_plan(config)
    dbcv_sample_size = validation_cfg.get("dbcv_sample_size")
    tasks = [
        {
            "index": index,
            "role": role,
            "configuration_id": _configuration_id(role, index),
            "params": params,
            "embeddings": role_embeddings,
            "accident_ids": role_units["_accident_id"].astype(str).to_numpy(),
            "random_state": expected_random_state,
            "config": config,
            "dbcv_sample_size": dbcv_sample_size,
        }
        for index, params in enumerate(plan)
    ]
    _log_progress(
        f"[{role}] candidats: démarrage de {len(tasks)} configurations "
        f"avec {resolve_n_workers(config)} workers"
    )
    results = _parallel_map(
        _evaluate_candidate_task,
        tasks,
        config,
        progress_label=f"{role} candidats",
    )
    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: int(item["index"])):
        configuration_id = str(result["configuration_id"])
        labels = result["labels"]
        reduced = result["reduced"]
        membership_strength = result["membership_strength"]
        np.save(candidate_dir / f"{configuration_id}_labels.npy", labels.astype(np.int32))
        np.save(candidate_dir / f"{configuration_id}_membership_strength.npy", membership_strength.astype(np.float32))
        np.save(candidate_dir / f"{configuration_id}_umap.npy", reduced.astype(np.float32))
        rows.append({
            "role": role,
            "configuration_id": configuration_id,
            **result["params"],
            **result["metrics"],
            "dbcv_umap": result["dbcv_umap"],
            "dbcv_n_units": result["dbcv_n_units"],
            "random_state": result["random_state"],
        })
    result = pd.DataFrame(rows)
    result.to_csv(metrics_path, index=False)
    _log_progress(f"[{role}] candidats: terminé ({len(result)} configurations)")
    (role_dir / "candidate_metadata.json").write_text(
        json.dumps({
            "version": "discovery_candidates_v1_membership_strength",
            "role": role,
            "n_candidates": len(result),
            "n_workers": resolve_n_workers(config),
        }, indent=2),
        encoding="utf-8",
    )
    return result


# Backward-compatible alias
evaluate_pareto_candidates = evaluate_candidates


def _best_jaccard(reference: np.ndarray, candidate: np.ndarray, reference_label: int) -> float:
    """Best-match Jaccard of one reference cluster against all non-noise candidate clusters.

    Reference and candidate label arrays must be aligned on the same factual units
    (same length, same row order). HDBSCAN noise (``-1``) is never used as a match
    target; reference members that are noise under the candidate remain in the
    reference set and therefore stay in the Jaccard denominator.
    """
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            "Jaccard alignment error: reference and candidate label vectors differ "
            f"in shape ({reference.shape} vs {candidate.shape}). "
            "Seed sensitivity must reuse the same role corpus/order as the reference partition."
        )
    reference_set = set(np.flatnonzero(reference == reference_label).tolist())
    if not reference_set:
        return 0.0
    best = 0.0
    for label in np.unique(candidate[candidate >= 0]):
        candidate_set = set(np.flatnonzero(candidate == label).tolist())
        union = reference_set | candidate_set
        if union:
            best = max(best, len(reference_set & candidate_set) / len(union))
    return float(best)


def _aggregate_resampling_stability(
    theme_frame: pd.DataFrame,
    *,
    n_repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate factor-level mean Jaccard and observability into configuration S_R."""
    if theme_frame.empty:
        empty_theme = pd.DataFrame(
            columns=[
                "role", "configuration_id", "cluster_label", "theme_stability",
                "observability", "n_observable_reps", "n_repetitions",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=["role", "configuration_id", "stability", "n_themes", "n_repetitions"]
        )
        return empty_theme, empty_summary

    observable = theme_frame.loc[theme_frame["n_reference_units"].astype(int) > 0].copy()
    if observable.empty:
        empty_theme = pd.DataFrame(
            columns=[
                "role", "configuration_id", "cluster_label", "theme_stability",
                "observability", "n_observable_reps", "n_repetitions",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=["role", "configuration_id", "stability", "n_themes", "n_repetitions"]
        )
        return empty_theme, empty_summary

    theme_summary = (
        observable.groupby(["role", "configuration_id", "cluster_label"], as_index=False)
        .agg(
            theme_stability=("best_jaccard", "mean"),
            n_observable_reps=("repetition", "nunique"),
        )
    )
    theme_summary["n_repetitions"] = int(n_repetitions)
    theme_summary["observability"] = theme_summary["n_observable_reps"] / float(max(1, n_repetitions))
    theme_frame = theme_frame.merge(
        theme_summary[
            ["role", "configuration_id", "cluster_label", "theme_stability", "observability", "n_observable_reps", "n_repetitions"]
        ],
        on=["role", "configuration_id", "cluster_label"],
        how="left",
    )
    summary = (
        theme_summary.groupby(["role", "configuration_id"], as_index=False)
        .agg(
            stability=("theme_stability", "mean"),
            n_themes=("cluster_label", "nunique"),
        )
    )
    summary["n_repetitions"] = int(n_repetitions)
    return theme_frame, summary


def evaluate_resampling_stability(
    role: str,
    units: pd.DataFrame,
    embeddings: np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    candidates: pd.DataFrame,
    *,
    reestimate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit each candidate on grouped accident subsamples and compute clusterwise Jaccard."""
    role_mask = units["_role"].eq(role).to_numpy()
    role_units = units.loc[role_mask].reset_index(drop=True)
    role_embeddings = np.asarray(embeddings[role_mask])
    validation_cfg = _validation_config(config)
    role_dir = _discovery_role_dir(output_dir, role)
    role_dir.mkdir(parents=True, exist_ok=True)
    theme_path = role_dir / "stability_theme.csv"
    summary_path = role_dir / "stability_summary.csv"
    n_repetitions = int(validation_cfg["n_resampling"])
    fraction = float(validation_cfg["resampling_fraction"])
    random_state = int(validation_cfg["random_state"])
    stability_metadata_path = role_dir / "stability_metadata.json"
    expected_stability_metadata = {
        "version": "mean_sr_observability_v2_fixed_umap_seed",
        "random_state": random_state,
        "n_repetitions": n_repetitions,
        "resampling_fraction": fraction,
        "aggregation": "mean_over_observable_replicates",
        "umap_random_state_policy": "fixed_primary_seed_during_resampling",
    }
    cached_stability_metadata = {}
    if stability_metadata_path.is_file():
        try:
            cached_stability_metadata = json.loads(stability_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_stability_metadata = {}
    if (
        theme_path.is_file()
        and summary_path.is_file()
        and cached_stability_metadata == expected_stability_metadata
        and not reestimate
    ):
        _log_progress(f"[{role}] resampling: cache réutilisé ({summary_path})")
        return pd.read_csv(theme_path), pd.read_csv(summary_path)
    candidate_dir = role_dir / "candidate_partitions"
    all_tasks: list[dict[str, Any]] = []
    candidates = candidates.reset_index(drop=True)
    for _, candidate in candidates.iterrows():
        configuration_id = str(candidate["configuration_id"])
        labels_path = candidate_dir / f"{configuration_id}_labels.npy"
        reference = np.load(labels_path).astype(int)
        params = {key: _to_python_value(candidate[key]) for key in PARAMETER_KEYS}
        reference_clusters = [int(label) for label in np.unique(reference) if label >= 0]
        for repetition in range(n_repetitions):
            rng = np.random.default_rng(random_state + ROLE_RANK[role] * 10000 + repetition)
            selected_accidents = _sample_accidents(role_units, fraction, rng)
            selected_indices = np.flatnonzero(role_units["_accident_id"].isin(selected_accidents).to_numpy())
            if len(selected_indices) < 3:
                continue
            all_tasks.append({
                "role": role,
                "configuration_id": configuration_id,
                "repetition": repetition,
                "reference_path": str(labels_path),
                "reference_clusters": reference_clusters,
                "selected_indices": selected_indices,
                "params": params,
                "embeddings": role_embeddings,
                "random_state": random_state,
                "config": config,
            })
    _log_progress(
        f"[{role}] resampling: démarrage de {n_repetitions} réplications "
        f"pour {len(candidates)} configurations ({len(all_tasks)} tâches)"
    )
    task_results = _parallel_map(
        _evaluate_stability_task,
        all_tasks,
        config,
        progress_label=f"{role} resampling",
    )
    all_rows = [row for task_rows in task_results for row in task_rows]
    theme_frame = pd.DataFrame(all_rows)
    theme_frame, summary = _aggregate_resampling_stability(theme_frame, n_repetitions=n_repetitions)
    theme_frame.to_csv(theme_path, index=False)
    summary.to_csv(summary_path, index=False)
    stability_metadata_path.write_text(json.dumps(expected_stability_metadata, indent=2), encoding="utf-8")
    _log_progress(
        f"[{role}] resampling: terminé ({len(summary)} résumés, "
        f"{len(all_tasks)} tâches)"
    )
    return theme_frame, summary


def select_configuration_by_stability(merged: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Legacy alias: Pareto + geometric knee selection (returns table and selected id)."""
    table, selected_id, _rule = select_configuration_for_role(merged)
    return table, selected_id


def select_configuration_for_role(
    merged: pd.DataFrame,
    semantic_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Select via Pareto front and geometric knee point (deterministic, no LLM).

    Returns ``(table, selected_id, selection_rule)`` where ``selection_rule`` is
    ``single_pareto`` or ``geometric_knee``.
    """
    from pareto_knee_selection import select_knee_configuration

    return select_knee_configuration(merged)


def materialize_selected_partition(
    role: str,
    units: pd.DataFrame,
    configuration_id: str,
    output_dir: Path,
    theme_stability: pd.DataFrame | None = None,
) -> None:
    """Copy the selected candidate artifacts into discovery/<role>/selected/."""
    role_dir = _discovery_role_dir(output_dir, role)
    candidate_dir = role_dir / "candidate_partitions"
    selected_dir = role_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    labels_path = candidate_dir / f"{configuration_id}_labels.npy"
    strength_path = candidate_dir / f"{configuration_id}_membership_strength.npy"
    umap_path = candidate_dir / f"{configuration_id}_umap.npy"
    if not labels_path.is_file() or not strength_path.is_file():
        raise FileNotFoundError(f"Missing candidate artifacts for {role}/{configuration_id}")
    labels = np.load(labels_path).astype(np.int32)
    strengths = np.load(strength_path).astype(np.float32)
    np.save(selected_dir / "labels.npy", labels)
    np.save(selected_dir / "membership_strength.npy", strengths)
    if umap_path.is_file():
        np.save(selected_dir / "umap.npy", np.load(umap_path).astype(np.float32))

    role_units = units.loc[units["_role"].eq(role)].reset_index(drop=True)
    if len(role_units) != len(labels):
        raise ValueError(f"Label/unit mismatch for {role}/{configuration_id}")
    assignments = role_units[["_accident_id", "_fact_id", "_text"]].copy()
    assignments.rename(
        columns={"_accident_id": "accident_id", "_fact_id": "fact_id", "_text": "sentence"},
        inplace=True,
    )
    assignments["role"] = role
    assignments["configuration_id"] = configuration_id
    assignments["topic_id"] = [f"{role}_{int(label):03d}" if int(label) >= 0 else "" for label in labels]
    assignments["membership_strength"] = strengths
    assignments.to_csv(selected_dir / "topic_assignments.csv", index=False)

    stability_lookup = {}
    observability_lookup = {}
    if theme_stability is not None and not theme_stability.empty:
        selected_themes = theme_stability[
            theme_stability["configuration_id"].astype(str).eq(str(configuration_id))
        ]
        if "theme_stability" in selected_themes.columns:
            stability_lookup = {
                int(row["cluster_label"]): float(row["theme_stability"])
                for _, row in selected_themes.drop_duplicates("cluster_label").iterrows()
            }
        if "observability" in selected_themes.columns:
            observability_lookup = {
                int(row["cluster_label"]): float(row["observability"])
                for _, row in selected_themes.drop_duplicates("cluster_label").iterrows()
            }
    topic_rows = []
    for label in sorted(int(value) for value in np.unique(labels) if int(value) >= 0):
        mask = labels == label
        topic_rows.append({
            "topic_id": f"{role}_{label:03d}",
            "role": role,
            "configuration_id": configuration_id,
            "cluster_label": label,
            "n_units": int(mask.sum()),
            "n_accidents": int(role_units.loc[mask, "_accident_id"].nunique()),
            "theme_stability": stability_lookup.get(label, np.nan),
            "observability": observability_lookup.get(label, np.nan),
        })
    pd.DataFrame(topic_rows).to_csv(selected_dir / "topics.csv", index=False)
    (selected_dir / "selection_metadata.json").write_text(
        json.dumps({
            "version": "pareto_knee_selected_partition_v1",
            "role": role,
            "configuration_id": configuration_id,
        }, indent=2),
        encoding="utf-8",
    )


def write_stability_landscape_figure(
    selection_tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    roles: Sequence[str] | None = None,
    filename: str = "stability_landscape_all_roles.png",
) -> None:
    """Scatter DBCV versus S_R with Pareto front and geometric knee (raw space)."""
    from pareto_knee_selection import plot_pareto_raw

    plot_pareto_raw(selection_tables, output_dir, roles=tuple(roles or ROLES), filename=filename)


def write_pareto_normalized_knee_figure(
    selection_tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    roles: Sequence[str] | None = None,
    filename: str = "pareto_normalized_knee_all_roles.png",
) -> None:
    """Normalized objective space with reference line and knee projection."""
    from pareto_knee_selection import plot_pareto_normalized_with_knee

    plot_pareto_normalized_with_knee(
        selection_tables,
        output_dir,
        roles=tuple(roles or ROLES),
        filename=filename,
    )


def write_factor_stability_figure(
    role: str,
    theme_stability: pd.DataFrame,
    configuration_id: str,
    output_dir: Path,
) -> None:
    """Legacy hook kept for compatibility; combined figures are written after all roles."""
    del role, theme_stability, configuration_id, output_dir


def write_factor_resampling_manuscript_figures(
    theme_tables: Mapping[str, pd.DataFrame],
    selections: Mapping[str, str],
    output_dir: Path,
) -> None:
    """Write ``factor_resampling_A0.png`` and ``factor_resampling_A1_B_C.png``."""
    from manuscript_reporting import (
        FIGURE_FACTOR_RESAMPLING_A0,
        FIGURE_FACTOR_RESAMPLING_A1_B_C,
        plot_factor_resampling_multi_panel,
        plot_factor_resampling_reproducibility,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "A0" in selections and "A0" in theme_tables:
        plot_factor_resampling_reproducibility(
            theme_tables["A0"],
            role="A0",
            configuration_id=str(selections["A0"]),
            output_path=output_dir / FIGURE_FACTOR_RESAMPLING_A0,
        )
    combo_roles = [role for role in ("A1", "B", "C") if role in selections and role in theme_tables]
    if combo_roles:
        plot_factor_resampling_multi_panel(
            {role: theme_tables[role] for role in combo_roles},
            roles=combo_roles,
            configuration_ids={role: str(selections[role]) for role in combo_roles},
            output_path=output_dir / FIGURE_FACTOR_RESAMPLING_A1_B_C,
        )


def write_umap_seed_sensitivity_all_roles_figure(output_dir: Path, *, run_dir: Path | None = None) -> None:
    """Write the Jaccard-only seed figure and complementary seed summary table."""
    from manuscript_reporting import (
        FIGURE_UMAP_SEED_SENSITIVITY,
        build_seed_sensitivity_summary_all_roles,
        plot_umap_seed_sensitivity_all_roles,
    )

    if run_dir is None:
        run_dir = Path(output_dir).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_umap_seed_sensitivity_all_roles(
        run_dir,
        output_path=output_dir / FIGURE_UMAP_SEED_SENSITIVITY,
    )
    summary = build_seed_sensitivity_summary_all_roles(run_dir)
    if summary.empty:
        return
    tables_dir = Path(run_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    display_cols = [
        "Role",
        "Reference_K",
        "K_range",
        "DBCV_range",
        "Unassigned_fraction_range",
        "Mean_Jaccard_range",
    ]
    cols = [c for c in display_cols if c in summary.columns]
    summary[cols].to_csv(tables_dir / "seed_sensitivity_summary_all_roles.csv", index=False)
    summary[cols].to_csv(output_dir / "seed_sensitivity_summary_all_roles.csv", index=False)


def _evaluate_seed_task(task: Mapping[str, Any]) -> dict[str, Any]:
    labels, reduced, membership_strength = _fit_cluster_with_embedding(
        (),
        task["embeddings"],
        task["params"],
        int(task["seed"]),
        task["config"],
        return_membership_strength=True,
    )
    reference = np.asarray(task["reference_labels"], dtype=int)
    labels = np.asarray(labels, dtype=int)
    if labels.shape != reference.shape:
        raise ValueError(
            "Seed sensitivity label length mismatch: "
            f"reference={reference.shape}, alternative_seed={labels.shape}, "
            f"n_embeddings={len(task['embeddings'])}."
        )
    rows = []
    for cluster_label in task["reference_clusters"]:
        rows.append({
            "cluster_label": int(cluster_label),
            "best_jaccard": _best_jaccard(reference, labels, int(cluster_label)),
        })
    metrics = _candidate_metrics_from_accident_ids(task["accident_ids"], labels)
    return {
        "seed": int(task["seed"]),
        "labels": np.asarray(labels, dtype=np.int32),
        "membership_strength": np.asarray(membership_strength, dtype=np.float32),
        "n_clusters": int(metrics["n_clusters"]),
        "noise_fraction": float(metrics["noise_fraction"]),
        "dbcv_umap": float(_dbcv(reduced, labels)),
        "factor_rows": rows,
    }


def evaluate_seed_sensitivity(
    role: str,
    units: pd.DataFrame,
    embeddings: np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    configuration_id: str,
    candidate_row: Mapping[str, Any],
    *,
    reestimate: bool = False,
) -> pd.DataFrame:
    """Refit the selected configuration under alternative UMAP seeds."""
    validation_cfg = _validation_config(config)
    seed_cfg = validation_cfg["seed_sensitivity"]
    if not bool(seed_cfg.get("enabled", True)):
        return pd.DataFrame()
    seeds = [int(seed) for seed in seed_cfg.get("seeds", [])]
    primary_seed = int(validation_cfg["random_state"])
    seeds = [seed for seed in seeds if seed != primary_seed]
    role_dir = _discovery_role_dir(output_dir, role)
    selected_dir = role_dir / "selected"
    seed_dir = role_dir / "seed_sensitivity"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seed_dir / "seed_summary.csv"
    factor_path = seed_dir / "seed_factor_jaccard.csv"
    metadata_path = seed_dir / "seed_metadata.json"
    expected_metadata = {
        "version": "umap_seed_sensitivity_v2_aligned_corpus",
        "configuration_id": configuration_id,
        "primary_seed": primary_seed,
        "seeds": seeds,
    }
    if (
        summary_path.is_file()
        and factor_path.is_file()
        and metadata_path.is_file()
        and not reestimate
    ):
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_metadata = {}
        if cached_metadata == expected_metadata:
            _log_progress(f"[{role}] seed sensitivity: cache réutilisé")
            return pd.read_csv(summary_path)

    role_mask = units["_role"].eq(role).to_numpy()
    role_units = units.loc[role_mask].reset_index(drop=True)
    role_embeddings = np.asarray(embeddings[role_mask])
    reference_path = selected_dir / "labels.npy"
    if not reference_path.is_file():
        reference_path = role_dir / "candidate_partitions" / f"{configuration_id}_labels.npy"
    reference = np.load(reference_path).astype(int)
    if len(reference) != len(role_units) or len(reference) != len(role_embeddings):
        raise ValueError(
            f"[{role}] seed sensitivity corpus mismatch for {configuration_id}: "
            f"reference_labels={len(reference)}, role_units={len(role_units)}, "
            f"role_embeddings={len(role_embeddings)}. "
            "Refuse to compare partitions on different factual-unit sets."
        )
    reference_clusters = [int(label) for label in np.unique(reference) if label >= 0]
    missing_params = [key for key in PARAMETER_KEYS if key not in candidate_row]
    if missing_params:
        raise ValueError(
            f"[{role}] selected configuration missing parameters for seed sensitivity: {missing_params}"
        )
    params = {key: _to_python_value(candidate_row[key]) for key in PARAMETER_KEYS}
    tasks = [
        {
            "seed": seed,
            "params": params,
            "embeddings": role_embeddings,
            "accident_ids": role_units["_accident_id"].astype(str).to_numpy(),
            "reference_labels": reference,
            "reference_clusters": reference_clusters,
            "config": config,
        }
        for seed in seeds
    ]
    _log_progress(f"[{role}] seed sensitivity: {len(tasks)} seeds for {configuration_id}")
    results = _parallel_map(_evaluate_seed_task, tasks, config, progress_label=f"{role} seeds")
    factor_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: int(item["seed"])):
        seed = int(result["seed"])
        np.save(seed_dir / f"seed_{seed}_labels.npy", result["labels"])
        np.save(seed_dir / f"seed_{seed}_membership_strength.npy", result["membership_strength"])
        seed_jaccards = []
        for row in result["factor_rows"]:
            factor_rows.append({
                "role": role,
                "configuration_id": configuration_id,
                "seed": seed,
                "cluster_label": int(row["cluster_label"]),
                "best_jaccard": float(row["best_jaccard"]),
            })
            seed_jaccards.append(float(row["best_jaccard"]))
        summary_rows.append({
            "role": role,
            "configuration_id": configuration_id,
            "seed": seed,
            "seed_stability": float(np.mean(seed_jaccards)) if seed_jaccards else np.nan,
            "n_clusters": int(result["n_clusters"]),
            "noise_fraction": float(result["noise_fraction"]),
            "dbcv_umap": float(result["dbcv_umap"]),
        })
    factor_frame = pd.DataFrame(factor_rows)
    summary = pd.DataFrame(summary_rows)
    if not factor_frame.empty:
        factor_means = (
            factor_frame.groupby(["role", "configuration_id", "cluster_label"], as_index=False)["best_jaccard"]
            .mean()
            .rename(columns={"best_jaccard": "seed_theme_stability"})
        )
        factor_means.to_csv(seed_dir / "seed_theme_stability.csv", index=False)
    summary.to_csv(summary_path, index=False)
    factor_frame.to_csv(factor_path, index=False)
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")

    if not summary.empty:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.5, 4.2))
        axis.plot(summary["seed"], summary["seed_stability"], marker="o", color="#4C78A8")
        axis.axhline(1.0, color="#AAAAAA", linewidth=0.8, linestyle="--", alpha=0.7)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Mean best-match Jaccard vs $s_0$")
        axis.set_xlabel("Alternative UMAP seed")
        axis.set_title(f"{role}: membership stability under alternative UMAP seeds")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        from manuscript_reporting import save_manuscript_figure

        save_manuscript_figure(figure, seed_dir / f"seed_sensitivity_{role}.png", dpi=220)
        plt.close(figure)
    return summary



def load_topic_stopwords(config: Mapping[str, Any]) -> set[str]:
    topics_cfg = config.get("topics", {})
    stopwords = {
        str(word).strip().lower()
        for word in topics_cfg.get("stopwords", [])
        if str(word).strip()
    }
    stopwords.update(
        str(word).strip().lower()
        for word in topics_cfg.get("additional_stopwords", [])
        if str(word).strip()
    )
    stopwords_file = topics_cfg.get("stopwords_file")
    if stopwords_file:
        path = Path(stopwords_file)
        if not path.is_file():
            local_candidate = Path(__file__).resolve().parent / path.name
            if local_candidate.is_file():
                path = local_candidate
            if not local_candidate.is_file():
                raise FileNotFoundError(f"Fichier de stopwords métier introuvable : {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip().lower()
            if line and not line.startswith("#"):
                stopwords.add(line)
    return stopwords


def _tokenize(text: str, stopwords: set[str], ngram_range: tuple[int, int] = (1, 1)) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(text)) if token.lower() not in stopwords]
    minimum, maximum = ngram_range
    features: list[str] = []
    for size in range(int(minimum), int(maximum) + 1):
        features.extend(" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1))
    return features


def build_topic_dictionary(
    prepared: PreparedData,
    partition_results: Mapping[str, PartitionResult],
    config: Mapping[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    topics_cfg = config.get("topics", {})
    stopwords = load_topic_stopwords(config)
    top_words = int(topics_cfg.get("top_words", 12))
    top_sentences = int(topics_cfg.get("top_sentences", 5))
    ngram_range = tuple(int(value) for value in topics_cfg.get("n_gram_range", [1, 1]))
    min_topic_frequency = int(topics_cfg.get("min_topic_frequency", 1))
    idf_smoothing = float(topics_cfg.get("idf_smoothing", 1.0))
    topic_payloads: list[tuple[str, pd.Series, dict[str, int]]] = []
    for role, result in partition_results.items():
        for _, topic in result.topics.iterrows():
            topic_id = str(topic["topic_id"])
            subset = result.assignments[result.assignments["topic_id"] == topic_id]
            token_counts: dict[str, int] = {}
            for text in subset["sentence"]:
                for token in _tokenize(text, stopwords, ngram_range):
                    token_counts[token] = token_counts.get(token, 0) + 1
            topic_payloads.append((topic_id, topic, token_counts))
    document_frequency: dict[str, int] = {}
    for _, _, token_counts in topic_payloads:
        for token in token_counts:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    document_frequency = {
        token: frequency
        for token, frequency in document_frequency.items()
        if frequency >= min_topic_frequency
    }
    n_topics = max(1, len(topic_payloads))
    rows: list[dict[str, Any]] = []
    embedding_lookup = {
        fact_id: prepared.embeddings[index]
        for index, fact_id in enumerate(prepared.units["_fact_id"].astype(str))
    }
    for topic_id, topic, token_counts in topic_payloads:
        role = str(topic["role"])
        result = partition_results[role]
        subset = result.assignments[result.assignments["topic_id"] == topic_id].copy()
        total_tokens = max(1, sum(token_counts.values()))
        scored_terms = {
            token: (count / total_tokens) * math.log(
                idf_smoothing + n_topics / max(1, document_frequency[token])
            )
            for token, count in token_counts.items()
            if token in document_frequency
        }
        ranked_words = sorted(scored_terms.items(), key=lambda item: (-item[1], item[0]))[:top_words]
        row = topic.to_dict()
        row["top_terms"] = ", ".join(f"{word} ({score:.4f})" for word, score in ranked_words)
        row["label"] = "; ".join(word for word, _ in ranked_words[:5])
        valid_subset = subset[subset["fact_id"].astype(str).isin(embedding_lookup)].reset_index(drop=True)
        vectors = np.asarray([embedding_lookup[fact_id] for fact_id in valid_subset["fact_id"].astype(str)])
        if len(vectors):
            centroid = vectors.mean(axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
            similarities = vectors @ centroid
            central_order = np.argsort(-similarities)
            boundary_order = np.argsort(similarities)
            row["central_sentence"] = str(valid_subset.iloc[int(central_order[0])]["sentence"])
            row["boundary_sentences"] = " || ".join(valid_subset.iloc[boundary_order[:top_sentences]]["sentence"].astype(str).tolist())
        else:
            row["central_sentence"] = ""
            row["boundary_sentences"] = ""
        row["representative_sentences"] = " || ".join(subset["sentence"].astype(str).head(top_sentences).tolist())
        rows.append(row)
    dictionary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(output_dir / "topic_dictionary.csv", index=False)
    return dictionary



# ---------------------------------------------------------------------------
# Frozen-theme latent BN analysis
# ---------------------------------------------------------------------------

def _role_color(role: str) -> str:
    from manuscript_reporting import role_color

    return role_color(role)


BN_ROLE_ARCS = (("A0", "A1"), ("A0", "B"), ("A1", "B"), ("B", "C"))


def is_latent_conditioned(role: str, latent_scope: str) -> bool:
    """Return whether node probabilities are indexed by the latent family Z."""

    return latent_scope == "all_roles" or role in {"A0", "A1"}


@dataclass
class StructuralEMResult:
    """One fitted latent BN candidate, including its selection diagnostics."""

    nodes: list[str]
    roles: dict[str, str]
    edges: list[tuple[str, str]]
    n_states: int
    weights: np.ndarray
    responsibilities: np.ndarray
    upstream_probabilities: dict[tuple[str, int, tuple[int, ...]], float]
    downstream_probabilities: dict[tuple[str, tuple[int, ...]], float]
    log_likelihood: float
    bic: float
    n_iter: int
    converged: bool
    seed: int
    initialization: str
    model: Any | None = None
    iteration_history: list[dict[str, Any]] | None = None
    last_loglik_delta: float = math.inf
    relative_loglik_delta: float = math.inf
    same_graph: bool = False
    edges_added_last: int = 0
    edges_removed_last: int = 0

    @property
    def parent_map(self) -> dict[str, list[str]]:
        parents = {node: [] for node in self.nodes}
        for parent, child in self.edges:
            parents.setdefault(child, []).append(parent)
        for node in parents:
            parents[node].sort()
        return parents

    @property
    def effective_sizes(self) -> np.ndarray:
        return self.responsibilities.sum(axis=0)


def _bn_config(config: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(config.get("bayesian_networks", {}))
    values.setdefault("include_all_retained_factors", True)
    values.setdefault("min_theme_support_count", 1)
    values.setdefault("d_max", 2)
    values.setdefault("latent_states", list(range(1, 11)))
    values.setdefault("extended_latent_states", [12, 15])
    values.setdefault("run_extended_k_if_boundary", True)
    values.setdefault("n_initializations", 20)
    values.setdefault("em_max_iter", 500)
    values.setdefault("em_tol", 1e-6)
    values.setdefault("graph_stability_patience", 5)
    values.setdefault("structure_max_iter", 100)
    values.setdefault("structure_epsilon", 1e-6)
    values.setdefault("alpha", 0.5)
    values.setdefault("probability_floor", 1e-12)
    values.setdefault("top_m_mpe", 3)
    values.setdefault("near_equivalent_mpe_log_gap", 0.10)
    values.setdefault("median_max_posterior_warning", 0.70)
    values.setdefault("min_parent_configuration_support_warning", 5)
    values.setdefault("article_edge_stability_threshold", 0.60)
    values.setdefault("article_bn_max_edges", 15)
    values.setdefault("write_diagnostic_bn_figures", False)
    values.setdefault("min_latent_effective_n", 25)
    values.setdefault("latent_scope", "upstream_only")
    values.setdefault("latent_scope_sensitivity", {"enabled": False, "alternatives": ["upstream_only", "all_roles"]})
    values.setdefault("mpe_required_roles", ["B", "C"])
    values.setdefault("mpe_upstream_any_roles", ["A0", "A1"])
    values.setdefault("mpe_optional_roles", [])
    values.setdefault("mpe_compute_free_diagnostic", True)
    values.setdefault("bn_structure_bootstrap", {
        "enabled": True, "n_resamples": 30, "sample_fraction": 0.80, "fraction": 0.80,
        "n_initializations_per_resample": 3, "random_state": 2026,
    })
    values.setdefault("show_progress", False)
    return values


def _latent_scope(config: Mapping[str, Any]) -> str:
    scope = str(_bn_config(config).get("latent_scope", "upstream_only"))
    if scope not in {"upstream_only", "all_roles"}:
        raise ValueError(f"latent_scope inconnu: {scope}")
    return scope


def _bn_logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    from scipy.special import logsumexp

    return logsumexp(values, axis=axis)


def _validate_bn_units(units: pd.DataFrame) -> pd.DataFrame:
    required = {"_accident_id", "_fact_id", "_role"}
    missing = required.difference(units.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour l'analyse BN: {sorted(missing)}")
    frame = units.copy()
    frame["_fact_id"] = frame["_fact_id"].astype(str)
    frame["_accident_id"] = frame["_accident_id"].astype(str)
    frame["_role"] = frame["_role"].astype(str)
    if frame["_fact_id"].duplicated().any():
        duplicated = frame.loc[frame["_fact_id"].duplicated(), "_fact_id"].head().tolist()
        raise ValueError(f"fact_id non uniques dans les unités BN: {duplicated}")
    invalid_roles = sorted(set(frame["_role"]) - set(ROLES))
    if invalid_roles:
        raise ValueError(f"Rôles inconnus dans les unités BN: {invalid_roles}")
    if frame["_accident_id"].eq("").any() or frame["_accident_id"].isna().any():
        raise ValueError("Des unités BN n'ont pas d'accident_id valide.")
    return frame


def build_frozen_bn_inputs(
    units: pd.DataFrame,
    run_dir: Path,
    partition_selections: Mapping[str, str],
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load one explicitly selected partition per role.

    Labels come from ``discovery/<role>/selected/`` or candidate artifacts.
    No clustering or resampling is performed here. The returned matrix has one
    row per accident and one binary variable per retained non-noise topic.

    Note: matrix value **0 = factor not observed** in the available narrative,
    not physical absence of the factor.
    """

    units = _validate_bn_units(units)
    cfg = _bn_config(config)
    include_all = bool(cfg["include_all_retained_factors"])
    min_support = int(cfg["min_theme_support_count"])
    selections = {role: str(partition_selections.get(role, "")).strip() for role in ROLES}
    missing = [role for role, value in selections.items() if not value]
    if missing:
        raise ValueError("Une configuration BN doit être choisie pour chaque rôle: " + ", ".join(missing))

    dictionary_paths = [
        run_dir / "topics_manual" / "topic_dictionary_with_llm_labels.csv",
        run_dir / "topics_manual" / "topic_dictionary_all_selected.csv",
    ]
    dictionary = next((pd.read_csv(path) for path in dictionary_paths if path.is_file()), pd.DataFrame())
    all_rows: list[dict[str, Any]] = []
    matrix = pd.DataFrame({"accident_id": sorted(units["_accident_id"].unique())})
    n_accidents = len(matrix)

    for role in ROLES:
        configuration_id = selections[role]
        labels_path, strength_path = _resolve_partition_artifact_paths(run_dir, role, configuration_id)
        role_units = units[units["_role"].eq(role)].reset_index(drop=True)
        labels = np.load(labels_path).astype(int)
        strengths = np.load(strength_path).astype(float)
        if len(labels) != len(role_units) or len(strengths) != len(role_units):
            raise ValueError(f"Décalage labels/unités pour {role}/{configuration_id}")
        if role_units["_fact_id"].duplicated().any():
            raise ValueError(f"fact_id non uniques dans le rôle {role}")

        for label in sorted(int(value) for value in np.unique(labels) if int(value) >= 0):
            topic_id = f"{role}_{label:03d}"
            mask = labels == label
            support_accidents = int(role_units.loc[mask, "_accident_id"].nunique())
            if not dictionary.empty and {"role", "configuration_id", "topic_id"}.issubset(dictionary.columns):
                topic_rows = dictionary[
                    dictionary["role"].astype(str).eq(role)
                    & dictionary["configuration_id"].astype(str).eq(configuration_id)
                    & dictionary["topic_id"].astype(str).eq(topic_id)
                ]
            else:
                topic_rows = pd.DataFrame()
            metadata = topic_rows.iloc[0].to_dict() if not topic_rows.empty else {}
            variable_name = f"{role}__T{label + 1:02d}"
            included = include_all or support_accidents >= min_support
            all_rows.append({
                "variable_name": variable_name,
                "topic_id": topic_id,
                "role": role,
                "topic_label": metadata.get("llm_label", metadata.get("label", topic_id)),
                "configuration_id": configuration_id,
                "n_units": int(mask.sum()),
                "n_accidents": support_accidents,
                "mean_membership_strength": float(np.mean(strengths[mask])) if mask.any() else 0.0,
                "included_in_bn": included,
            })
            if included:
                present = set(role_units.loc[mask, "_accident_id"].astype(str))
                matrix[variable_name] = matrix["accident_id"].isin(present).astype(np.int8)

    dictionary_out = pd.DataFrame(all_rows)
    if dictionary_out.empty:
        raise ValueError("Aucun thème non bruit n'a été trouvé dans les partitions sélectionnées.")
    n_retained_factors = len(dictionary_out)
    included = dictionary_out[dictionary_out["included_in_bn"]].copy()
    excluded = dictionary_out[~dictionary_out["included_in_bn"]].copy()
    n_bn_factors = len(included)
    if include_all and n_bn_factors != n_retained_factors:
        raise ValueError(
            f"Inventaire discovery ({n_retained_factors} facteurs retenus) ≠ facteurs BN ({n_bn_factors}). "
            "Vérifier include_all_retained_factors et min_theme_support_count."
        )
    included_names = included["variable_name"].tolist()
    matrix = matrix[["accident_id", *included_names]]
    roles = dict(zip(included["variable_name"], included["role"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_out.to_csv(output_dir / "theme_dictionary.csv", index=False)
    dictionary_out.to_csv(output_dir / "theme_support.csv", index=False)
    matrix.to_parquet(output_dir / "accident_factor_matrix.parquet", index=False)
    excluded.to_csv(output_dir / "excluded_themes.csv", index=False)
    (output_dir / "variable_roles.json").write_text(json.dumps(roles, indent=2), encoding="utf-8")

    prevalence_rows = []
    for _, row in included.iterrows():
        variable = str(row["variable_name"])
        observation_prevalence = float(matrix[variable].mean()) if variable in matrix.columns else 0.0
        prevalence_rows.append({
            "variable_name": variable,
            "topic_id": row["topic_id"],
            "role": row["role"],
            "topic_label": row["topic_label"],
            "observation_prevalence": observation_prevalence,
            "n_accidents_with_factor": int(matrix[variable].sum()) if variable in matrix.columns else 0,
            "n_accidents_total": n_accidents,
        })
    pd.DataFrame(prevalence_rows).to_csv(output_dir / "factor_prevalence.csv", index=False)
    pd.DataFrame([{
        "n_accidents": n_accidents,
        "n_retained_factors": n_retained_factors,
        "n_bn_factors": n_bn_factors,
        "n_excluded_factors": len(excluded),
        "include_all_retained_factors": include_all,
        "min_theme_support_count": min_support,
        "roles": json.dumps({role: int((included["role"] == role).sum()) for role in ROLES}),
    }]).to_csv(output_dir / "input_summary.csv", index=False)
    return matrix, included, excluded, roles


def write_bn_accident_inclusion_audit(
    units: pd.DataFrame,
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace which accidents enter the BN matrix and why."""

    units = _validate_bn_units(units)
    factor_columns = [column for column in matrix.columns if column != "accident_id"]
    included_ids = set(matrix["accident_id"].astype(str))
    all_accidents = sorted(units["_accident_id"].astype(str).unique())
    rows = []
    for accident_id in all_accidents:
        included = accident_id in included_ids
        if included:
            row_data = matrix.loc[matrix["accident_id"].astype(str).eq(accident_id)].iloc[0]
            role_counts = {
                role: int(sum(int(row_data[column]) for column in factor_columns if roles.get(column) == role))
                for role in ROLES
            }
            n_observed = int(row_data[factor_columns].sum())
            exclusion_reason = ""
        else:
            role_counts = {role: 0 for role in ROLES}
            n_observed = 0
            exclusion_reason = "not_in_matrix"
        rows.append({
            "accident_id": accident_id,
            "included_in_bn": included,
            "exclusion_reason": exclusion_reason,
            "n_observed_factors": n_observed,
            "n_A0": role_counts["A0"],
            "n_A1": role_counts["A1"],
            "n_B": role_counts["B"],
            "n_C": role_counts["C"],
        })
    audit = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "total_accidents_available": len(all_accidents),
        "included_accidents": int(audit["included_in_bn"].sum()),
        "excluded_accidents": int((~audit["included_in_bn"]).sum()),
        "exclusions_by_reason": json.dumps(
            audit.loc[~audit["included_in_bn"], "exclusion_reason"].value_counts().to_dict()
        ),
    }])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "bn_accident_inclusion_audit.csv", index=False)
    summary.to_csv(output_dir / "bn_input_audit_summary.csv", index=False)
    return audit, summary


def _allowed_bn_edges(roles: Mapping[str, str]) -> list[tuple[str, str]]:
    return [
        (parent, child)
        for parent, parent_role in roles.items()
        for child, child_role in roles.items()
        if (parent_role, child_role) in BN_ROLE_ARCS
    ]


def _bn_parent_map(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    result = {node: [] for node in nodes}
    for parent, child in edges:
        result.setdefault(child, []).append(parent)
    for node in result:
        result[node].sort()
    return result


def _bn_parameter_count(nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], n_states: int, latent_scope: str = "upstream_only") -> int:
    parents = _bn_parent_map(nodes, edges)
    count = n_states - 1
    for node in nodes:
        cardinality = n_states if is_latent_conditioned(roles[node], latent_scope) else 1
        count += cardinality * (2 ** len(parents[node]))
    return int(count)


def _bn_mle_parameters(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], tau: np.ndarray, n_states: int, floor: float = 1e-12, latent_scope: str = "upstream_only") -> tuple[np.ndarray, dict[tuple[str, int, tuple[int, ...]], float], dict[tuple[str, tuple[int, ...]], float]]:
    index = {node: position for position, node in enumerate(nodes)}
    parents = _bn_parent_map(nodes, edges)
    weights = tau.sum(axis=0)
    omega = weights / max(weights.sum(), floor)
    upstream: dict[tuple[str, int, tuple[int, ...]], float] = {}
    downstream: dict[tuple[str, tuple[int, ...]], float] = {}
    for node in nodes:
        parent_indices = [index[parent] for parent in parents[node]]
        for parent_values in itertools.product((0, 1), repeat=len(parent_indices)):
            mask = np.ones(len(data), dtype=bool)
            for column, value in zip(parent_indices, parent_values):
                mask &= data[:, column] == value
            if is_latent_conditioned(roles[node], latent_scope):
                for state in range(n_states):
                    mass = tau[:, state] * mask
                    denominator = float(mass.sum())
                    numerator = float((mass * data[:, index[node]]).sum())
                    upstream[(node, state, tuple(parent_values))] = numerator / denominator if denominator > floor else 0.5
            else:
                denominator = float(mask.sum())
                numerator = float((mask * data[:, index[node]]).sum())
                downstream[(node, tuple(parent_values))] = numerator / denominator if denominator > floor else 0.5
    return omega, upstream, downstream


def _bn_log_joint(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], weights: np.ndarray, upstream: Mapping[tuple[str, int, tuple[int, ...]], float], downstream: Mapping[tuple[str, tuple[int, ...]], float], floor: float = 1e-12, latent_scope: str = "upstream_only") -> np.ndarray:
    index = {node: position for position, node in enumerate(nodes)}
    parents = _bn_parent_map(nodes, edges)
    output = np.broadcast_to(
        np.log(np.clip(np.asarray(weights, dtype=float), floor, 1.0))[None, :],
        (len(data), len(weights)),
    ).copy()
    for node in nodes:
        node_index = index[node]
        node_values = data[:, node_index].astype(bool)
        node_parents = parents[node]
        parent_indices = [index[parent] for parent in node_parents]
        combinations = list(itertools.product((0, 1), repeat=len(parent_indices)))
        if parent_indices:
            codes = sum(
                data[:, parent_index].astype(int) * (2 ** (len(parent_indices) - offset - 1))
                for offset, parent_index in enumerate(parent_indices)
            )
        else:
            codes = np.zeros(len(data), dtype=int)
        if is_latent_conditioned(roles[node], latent_scope):
            for state in range(len(weights)):
                probabilities = np.asarray(
                    [upstream.get((node, state, values), 0.5) for values in combinations],
                    dtype=float,
                )
                probabilities = np.clip(probabilities, floor, 1.0 - floor)[codes]
                output[:, state] += np.where(
                    node_values,
                    np.log(probabilities),
                    np.log1p(-probabilities),
                )
        else:
            probabilities = np.asarray(
                [downstream.get((node, values), 0.5) for values in combinations],
                dtype=float,
            )
            probabilities = np.clip(probabilities, floor, 1.0 - floor)[codes]
            contribution = np.where(
                node_values,
                np.log(probabilities),
                np.log1p(-probabilities),
            )
            output += contribution[:, None]
    return output


def _bn_local_bic(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], node: str, parent_nodes: Sequence[str], tau: np.ndarray, n_states: int, floor: float, latent_scope: str = "upstream_only") -> float:
    index = {name: position for position, name in enumerate(nodes)}
    parent_indices = [index[parent] for parent in parent_nodes]
    node_values = data[:, index[node]].astype(float)
    combinations = list(itertools.product((0, 1), repeat=len(parent_indices)))
    if parent_indices:
        codes = sum(
            data[:, parent_index].astype(int) * (2 ** (len(parent_indices) - offset - 1))
            for offset, parent_index in enumerate(parent_indices)
        )
    else:
        codes = np.zeros(len(data), dtype=int)
    q_value = 0.0
    for combination_index, _ in enumerate(combinations):
        mask = codes == combination_index
        if is_latent_conditioned(roles[node], latent_scope):
            for state in range(n_states):
                mass = tau[:, state] * mask
                denominator = float(mass.sum())
                numerator = float((mass * node_values).sum())
                probability = numerator / denominator if denominator > floor else 0.5
                probability = float(np.clip(probability, floor, 1.0 - floor))
                q_value += numerator * math.log(probability)
                q_value += (denominator - numerator) * math.log1p(-probability)
        else:
            denominator = float(mask.sum())
            numerator = float((mask * node_values).sum())
            probability = numerator / denominator if denominator > floor else 0.5
            probability = float(np.clip(probability, floor, 1.0 - floor))
            q_value += numerator * math.log(probability)
            q_value += (denominator - numerator) * math.log1p(-probability)
    cardinality = n_states if is_latent_conditioned(roles[node], latent_scope) else 1
    parameter_count = cardinality * (2 ** len(parent_indices))
    return -2.0 * q_value + parameter_count * math.log(max(len(data), 1))


def _bn_expected_bic(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], tau: np.ndarray, n_states: int, floor: float, latent_scope: str = "upstream_only") -> float:
    weights = tau.sum(axis=0)
    weight_q = float(np.sum(tau * np.log(np.clip(weights / max(weights.sum(), floor), floor, 1.0))[None, :]))
    parents = _bn_parent_map(nodes, edges)
    local_scores = sum(
        _bn_local_bic(data, nodes, roles, node, parents[node], tau, n_states, floor, latent_scope)
        for node in nodes
    )
    return -2.0 * weight_q + (n_states - 1) * math.log(max(len(data), 1)) + local_scores


def _random_bn_edges(nodes: Sequence[str], roles: Mapping[str, str], d_max: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    candidates = _allowed_bn_edges(roles)
    rng.shuffle(candidates)
    parents = {node: 0 for node in nodes}
    selected = []
    for edge in candidates:
        if parents[edge[1]] < d_max and bool(rng.integers(0, 2)):
            selected.append(edge)
            parents[edge[1]] += 1
    return sorted(selected)


def _structure_step(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], tau: np.ndarray, n_states: int, d_max: int, epsilon: float, max_iter: int, floor: float, latent_scope: str = "upstream_only") -> list[tuple[str, str]]:
    current = set(edges)
    allowed = set(_allowed_bn_edges(roles))
    parents = _bn_parent_map(nodes, sorted(current))
    local_cache: dict[tuple[str, tuple[str, ...]], float] = {}

    def local_score(node: str, parent_nodes: Sequence[str]) -> float:
        key = (node, tuple(sorted(parent_nodes)))
        if key not in local_cache:
            local_cache[key] = _bn_local_bic(data, nodes, roles, node, key[1], tau, n_states, floor, latent_scope)
        return local_cache[key]

    def graph_score(edges_for_graph: set[tuple[str, str]]) -> float:
        graph_parents = _bn_parent_map(nodes, sorted(edges_for_graph))
        weights = tau.sum(axis=0)
        weight_q = float(np.sum(tau * np.log(np.clip(weights / max(weights.sum(), floor), floor, 1.0))[None, :]))
        return (
            -2.0 * weight_q
            + (n_states - 1) * math.log(max(len(data), 1))
            + sum(local_score(node, graph_parents[node]) for node in nodes)
        )

    for _ in range(max_iter):
        base_score = graph_score(current)
        moves: list[tuple[set[tuple[str, str]], str, tuple[str, ...]]] = []
        for edge in sorted(allowed - current):
            if len(parents[edge[1]]) < d_max:
                new_parents = tuple(sorted((*parents[edge[1]], edge[0])))
                moves.append((current | {edge}, edge[1], new_parents))
        for edge in sorted(current):
            new_parents = tuple(parent for parent in parents[edge[1]] if parent != edge[0])
            moves.append((current - {edge}, edge[1], new_parents))
        if not moves:
            break
        current_local = {
            node: local_score(node, parents[node])
            for node in nodes
        }
        scored = []
        for candidate, changed_node, changed_parents in moves:
            candidate_score = base_score - current_local[changed_node] + local_score(changed_node, changed_parents)
            scored.append((candidate, candidate_score))
        candidate, candidate_score = min(scored, key=lambda item: item[1])
        if base_score - candidate_score <= epsilon:
            break
        current = candidate
        parents = _bn_parent_map(nodes, sorted(current))
    return sorted(current)


def _fit_bn_k1(
    data: np.ndarray,
    nodes: list[str],
    roles: dict[str, str],
    seed: int,
    initialization: str,
    config: Mapping[str, Any],
    progress_callback: Callable[[int, float, int], None] | None = None,
) -> StructuralEMResult:
    """Fit K=1 reference model: no latent heterogeneity (P(Z=1)=1, tau=1)."""

    cfg = _bn_config(config)
    latent_scope = _latent_scope(config)
    rng = np.random.default_rng(seed)
    d_max = int(cfg["d_max"])
    floor = float(cfg["probability_floor"])
    n_states = 1
    edges = [] if initialization == "empty" else _random_bn_edges(nodes, roles, d_max, rng)
    tau = np.ones((len(data), 1), dtype=float)
    previous_ll = -np.inf
    converged = False
    iteration_history: list[dict[str, Any]] = []
    last_loglik_delta = math.inf
    relative_loglik_delta = math.inf
    same_graph = False
    edges_added_last = 0
    edges_removed_last = 0
    graph_stable_streak = 0
    for iteration in range(1, int(cfg["em_max_iter"]) + 1):
        weights = np.array([1.0], dtype=float)
        _, upstream, downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor, latent_scope)
        log_joint = _bn_log_joint(data, nodes, roles, edges, weights, upstream, downstream, floor, latent_scope)
        observed_ll = float(_bn_logsumexp(log_joint, axis=1).sum())
        new_edges = _structure_step(data, nodes, roles, edges, tau, n_states, d_max, float(cfg["structure_epsilon"]), int(cfg["structure_max_iter"]), floor, latent_scope)
        same_graph = new_edges == edges
        delta = abs(observed_ll - previous_ll) if np.isfinite(previous_ll) else np.inf
        relative_delta = delta / max(abs(previous_ll), 1.0) if np.isfinite(previous_ll) else np.inf
        old_edge_set = set(edges)
        new_edge_set = set(new_edges)
        edges_added_last = len(new_edge_set - old_edge_set)
        edges_removed_last = len(old_edge_set - new_edge_set)
        graph_stable_streak = graph_stable_streak + 1 if same_graph else 0
        last_loglik_delta = delta
        relative_loglik_delta = relative_delta
        stopped_by_tolerance = graph_stable_streak >= int(cfg["graph_stability_patience"]) and relative_delta < float(cfg["em_tol"])
        iteration_history.append({
            "iteration": iteration,
            "log_likelihood": observed_ll,
            "loglik_delta": delta,
            "relative_loglik_delta": relative_delta,
            "same_graph": same_graph,
            "graph_stable_streak": graph_stable_streak,
            "edges": len(new_edges),
            "edges_added": edges_added_last,
            "edges_removed": edges_removed_last,
            "stopped_by_tolerance": stopped_by_tolerance,
            "stopped_by_max_iterations": False,
        })
        edges = new_edges
        tau = np.ones((len(data), 1), dtype=float)
        if progress_callback is not None:
            progress_callback(iteration, observed_ll, len(edges))
        if stopped_by_tolerance:
            converged = True
            break
        previous_ll = observed_ll
    if not converged and iteration_history:
        iteration_history[-1]["stopped_by_max_iterations"] = True
    final_weights = np.array([1.0], dtype=float)
    _, final_upstream, final_downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor, latent_scope)
    final_log_joint = _bn_log_joint(data, nodes, roles, edges, final_weights, final_upstream, final_downstream, floor, latent_scope)
    final_ll = float(_bn_logsumexp(final_log_joint, axis=1).sum())
    bic = -2.0 * final_ll + _bn_parameter_count(nodes, roles, edges, n_states, latent_scope) * math.log(max(len(data), 1))
    return StructuralEMResult(
        nodes, roles, edges, n_states, final_weights, tau,
        final_upstream, final_downstream, final_ll, bic, iteration,
        converged, seed, initialization, None, iteration_history,
        last_loglik_delta, relative_loglik_delta, same_graph,
        edges_added_last, edges_removed_last,
    )


def _fit_structural_em_initialization(
    data: np.ndarray,
    nodes: list[str],
    roles: dict[str, str],
    n_states: int,
    seed: int,
    initialization: str,
    config: Mapping[str, Any],
    progress_callback: Callable[[int, float, int], None] | None = None,
) -> StructuralEMResult:
    if n_states == 1:
        return _fit_bn_k1(data, nodes, roles, seed, initialization, config, progress_callback)
    cfg = _bn_config(config)
    latent_scope = _latent_scope(config)
    rng = np.random.default_rng(seed)
    d_max = int(cfg["d_max"])
    floor = float(cfg["probability_floor"])
    edges = [] if initialization == "empty" else _random_bn_edges(nodes, roles, d_max, rng)
    tau = rng.dirichlet(np.ones(n_states), size=len(data))
    previous_ll = -np.inf
    converged = False
    iteration_history: list[dict[str, Any]] = []
    last_loglik_delta = math.inf
    relative_loglik_delta = math.inf
    same_graph = False
    edges_added_last = 0
    edges_removed_last = 0
    graph_stable_streak = 0
    for iteration in range(1, int(cfg["em_max_iter"]) + 1):
        weights, upstream, downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor, latent_scope)
        log_joint = _bn_log_joint(data, nodes, roles, edges, weights, upstream, downstream, floor, latent_scope)
        observed_ll = float(_bn_logsumexp(log_joint, axis=1).sum())
        new_tau = np.exp(log_joint - _bn_logsumexp(log_joint, axis=1)[:, None])
        new_edges = _structure_step(data, nodes, roles, edges, new_tau, n_states, d_max, float(cfg["structure_epsilon"]), int(cfg["structure_max_iter"]), floor, latent_scope)
        same_graph = new_edges == edges
        delta = abs(observed_ll - previous_ll) if np.isfinite(previous_ll) else np.inf
        relative_delta = delta / max(abs(previous_ll), 1.0) if np.isfinite(previous_ll) else np.inf
        old_edge_set = set(edges)
        new_edge_set = set(new_edges)
        edges_added_last = len(new_edge_set - old_edge_set)
        edges_removed_last = len(old_edge_set - new_edge_set)
        graph_stable_streak = graph_stable_streak + 1 if same_graph else 0
        last_loglik_delta = delta
        relative_loglik_delta = relative_delta
        iteration_history.append({
            "iteration": iteration,
            "log_likelihood": observed_ll,
            "loglik_delta": delta,
            "relative_loglik_delta": relative_delta,
            "same_graph": same_graph,
            "graph_stable_streak": graph_stable_streak,
            "edges": len(new_edges),
            "edges_added": edges_added_last,
            "edges_removed": edges_removed_last,
            "stopped_by_tolerance": graph_stable_streak >= int(cfg["graph_stability_patience"]) and relative_delta < float(cfg["em_tol"]),
            "stopped_by_max_iterations": False,
        })
        tau = new_tau
        edges = new_edges
        if progress_callback is not None:
            progress_callback(iteration, observed_ll, len(edges))
        if graph_stable_streak >= int(cfg["graph_stability_patience"]) and relative_delta < float(cfg["em_tol"]):
            converged = True
            break
        previous_ll = observed_ll
    if not converged and iteration_history:
        iteration_history[-1]["stopped_by_max_iterations"] = True
    final_weights, final_upstream, final_downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor, latent_scope)
    final_log_joint = _bn_log_joint(data, nodes, roles, edges, final_weights, final_upstream, final_downstream, floor, latent_scope)
    tau = np.exp(final_log_joint - _bn_logsumexp(final_log_joint, axis=1)[:, None])
    final_weights, final_upstream, final_downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor, latent_scope)
    final_log_joint = _bn_log_joint(data, nodes, roles, edges, final_weights, final_upstream, final_downstream, floor, latent_scope)
    final_ll = float(_bn_logsumexp(final_log_joint, axis=1).sum())
    bic = -2.0 * final_ll + _bn_parameter_count(nodes, roles, edges, n_states, latent_scope) * math.log(max(len(data), 1))
    return StructuralEMResult(
        nodes, roles, edges, n_states, final_weights, tau,
        final_upstream, final_downstream, final_ll, bic, iteration,
        converged, seed, initialization, None, iteration_history,
        last_loglik_delta, relative_loglik_delta, same_graph,
        edges_added_last, edges_removed_last,
    )


def _select_latent_result(results: Sequence[StructuralEMResult], config: Mapping[str, Any], diagnostic_output_dir: Path | None = None, primary_grid: Sequence[int] | None = None, eligible_k: Sequence[int] | None = None) -> tuple[StructuralEMResult, pd.DataFrame, list[str]]:
    minimum = float(_bn_config(config)["min_latent_effective_n"])
    latent_scope = _latent_scope(config)
    selection_warnings: list[str] = []
    rows = []
    admissible = []
    for result in results:
        effective = result.effective_sizes
        min_effective_n = float(effective.min())
        min_weight = float(result.weights.min())
        rejection_reasons = []
        if not result.converged:
            rejection_reasons.append("not_converged")
        if result.n_states > 1:
            if min_effective_n < minimum:
                rejection_reasons.append("min_effective_n_below_threshold")
            if min_weight <= 0:
                rejection_reasons.append("non_positive_weight")
        valid = not rejection_reasons
        history = result.iteration_history or []
        tau = result.responsibilities
        if result.n_states == 1:
            entropy = 0.0
            normalized_entropy = 0.0
        else:
            entropy = float(-np.sum(tau * np.log(np.clip(tau, 1e-300, 1.0))))
            normalized_entropy = entropy / max(len(tau) * math.log(max(result.n_states, 2)), 1e-12)
        max_posterior = float(np.max(tau, axis=1).mean())
        n_parameters = _bn_parameter_count(result.nodes, result.roles, result.edges, result.n_states, latent_scope)
        rows.append({
            "K": result.n_states,
            "seed": result.seed,
            "initialization": result.initialization,
            "converged": result.converged,
            "n_iter": result.n_iter,
            "log_likelihood": result.log_likelihood,
            "bic": result.bic,
            "number_parameters": n_parameters,
            "min_effective_n": min_effective_n,
            "min_N_eff": min_effective_n,
            "min_weight": min_weight,
            "min_family_weight": min_weight,
            "admissible": valid,
            "rejection_reason": "|".join(rejection_reasons),
            "edges": len(result.edges),
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "mean_max_posterior": max_posterior,
            "last_loglik_delta": result.last_loglik_delta,
            "relative_loglik_delta": result.relative_loglik_delta,
            "same_graph": result.same_graph,
            "edges_added_last": result.edges_added_last,
            "edges_removed_last": result.edges_removed_last,
            "n_admissible_runs": np.nan,
            "convergence_rate": np.nan,
            "last_10_log_likelihoods": json.dumps([item["log_likelihood"] for item in history[-10:]]),
            "last_10_loglik_deltas": json.dumps([item["loglik_delta"] for item in history[-10:]]),
            "last_10_edge_counts": json.dumps([item["edges"] for item in history[-10:]]),
            "full_iteration_history": json.dumps(history),
        })
        if valid:
            admissible.append(result)
    selection = pd.DataFrame(rows).sort_values(["K", "bic"], na_position="last").reset_index(drop=True)
    selection["selected_for_K"] = False
    selection["selected_final"] = False
    if diagnostic_output_dir is not None:
        diagnostic_output_dir.mkdir(parents=True, exist_ok=True)
        selection.to_csv(diagnostic_output_dir / "all_initializations.csv", index=False)
    if not admissible:
        summary = selection["rejection_reason"].value_counts().to_dict()
        raise RuntimeError(f"Aucune initialisation Structural-EM n'est admissible. Motifs: {summary}")
    per_k = {result.n_states: min((item for item in admissible if item.n_states == result.n_states), key=lambda item: item.bic, default=None) for result in admissible}
    per_k = {key: value for key, value in per_k.items() if value is not None}
    if not per_k:
        raise RuntimeError("Aucun K ne satisfait les critères de convergence et de taille effective.")
    selection_pool = {key: value for key, value in per_k.items() if eligible_k is None or key in set(eligible_k)}
    if not selection_pool:
        raise RuntimeError("Aucun K admissible dans la grille de sélection.")
    selected = min(selection_pool.values(), key=lambda item: item.bic)
    for result in per_k.values():
        selection.loc[(selection["K"] == result.n_states) & (selection["seed"] == result.seed), "selected_for_K"] = True
    selection["selected_final"] = (selection["K"] == selected.n_states) & (selection["seed"] == selected.seed)

    grid = list(primary_grid) if primary_grid else sorted(selection_pool)
    if selected.n_states == min(grid):
        message = "Le K sélectionné est sur la borne inférieure de la grille primaire."
        selection_warnings.append(message)
        warnings.warn(message, RuntimeWarning)
        if selected.n_states == 1:
            k1_message = (
                "Selected K=1: the data do not support additional latent accident-family "
                "heterogeneity under the tested model."
            )
            selection_warnings.append(k1_message)
            warnings.warn(k1_message, RuntimeWarning)
    if selected.n_states == max(grid):
        message = "Le K sélectionné est sur la borne supérieure de la grille primaire."
        selection_warnings.append(message)
        warnings.warn(message, RuntimeWarning)
    for k_value in grid:
        n_admissible = int(selection.loc[(selection["K"] == k_value) & selection["admissible"]].shape[0])
        if n_admissible == 0:
            message = f"Aucun fit admissible pour K={k_value}."
            selection_warnings.append(message)
            warnings.warn(message, RuntimeWarning)
        elif n_admissible < 2 and k_value == selected.n_states:
            message = f"Moins de 2 initialisations convergentes pour K*={k_value}."
            selection_warnings.append(message)
            warnings.warn(message, RuntimeWarning)
    best_bics = sorted((result.bic for result in selection_pool.values()))
    if len(best_bics) >= 2:
        delta_bic = best_bics[1] - best_bics[0]
        close_k = [k for k, result in selection_pool.items() if result.bic - best_bics[0] < 2.0]
        if len(close_k) > 1:
            message = f"Plusieurs K avec delta_BIC < 2: {close_k}."
            selection_warnings.append(message)
            warnings.warn(message, RuntimeWarning)
    if diagnostic_output_dir is not None:
        selection.groupby("K", as_index=False).agg(
            best_bic=("bic", "min"), best_log_likelihood=("log_likelihood", "max"),
            n_admissible=("admissible", "sum"), min_effective_n=("min_effective_n", "max"),
        ).to_csv(diagnostic_output_dir / "K_selection.csv", index=False)
    return selected, selection, selection_warnings


def _build_pgmpy_model(result: StructuralEMResult, smooth: bool, alpha: float, latent_scope: str = "upstream_only") -> Any:
    try:
        from pgmpy.factors.discrete import TabularCPD
        try:
            from pgmpy.models import DiscreteBayesianNetwork
        except ImportError:
            from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork
    except ImportError as error:
        raise ImportError("pgmpy est requis pour construire le BN final.") from error

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(["Z", *result.nodes])
    model.add_edges_from(result.edges)
    for node in result.nodes:
        if is_latent_conditioned(result.roles[node], latent_scope):
            model.add_edge("Z", node)
    parents = _bn_parent_map(result.nodes, result.edges)
    cpds = [TabularCPD("Z", result.n_states, np.asarray(result.weights, dtype=float).reshape(-1, 1).tolist())]
    for node in result.nodes:
        evidence = [*parents[node]]
        if is_latent_conditioned(result.roles[node], latent_scope):
            evidence = ["Z", *evidence]
        columns = list(itertools.product(range(result.n_states) if evidence and evidence[0] == "Z" else range(2), *([range(2)] * (len(evidence) - 1)))) if evidence else [()]
        values = np.zeros((2, len(columns)), dtype=float)
        for column, assignment in enumerate(columns):
            if evidence and evidence[0] == "Z":
                state = int(assignment[0])
                parent_values = tuple(int(value) for value in assignment[1:])
            else:
                state = 0
                parent_values = tuple(int(value) for value in assignment)
            if is_latent_conditioned(result.roles[node], latent_scope):
                mle = result.upstream_probabilities.get((node, state, parent_values), 0.5)
            else:
                mle = result.downstream_probabilities.get((node, parent_values), 0.5)
            if smooth:
                # Equivalent posterior smoothing is applied to the stored MLE
                # only through a neutral fallback; exact counts are handled in
                # finalize_latent_bn below when available.
                probability = min(max(float(mle), alpha / (1.0 + 2.0 * alpha)), 1.0 - alpha / (1.0 + 2.0 * alpha))
            else:
                probability = float(mle)
            values[:, column] = [1.0 - probability, probability]
        cpd = TabularCPD(node, 2, values, evidence=evidence or None, evidence_card=[result.n_states if evidence and evidence[0] == "Z" else 2] + [2] * (len(evidence) - 1) if evidence else None)
        cpds.append(cpd)
    model.add_cpds(*cpds)
    model.check_model()
    return model


def _run_structural_em_grid(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    latent_states: Sequence[int],
) -> list[StructuralEMResult]:
    """Fit Structural EM for all (K, initialization) pairs in the grid."""

    cfg = _bn_config(config)
    nodes = list(roles)
    data = matrix[nodes].to_numpy(dtype=np.int8)
    n_initializations = int(cfg["n_initializations"])
    base_seed = int(config.get("random_state", 42))
    tasks = [
        (
            int(n_states),
            init_index,
            "empty" if init_index == 0 else "random",
            base_seed + int(n_states) * 100_000 + init_index,
        )
        for n_states in latent_states
        for init_index in range(n_initializations)
    ]
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    workers = resolve_bn_n_workers(config)
    progress = tqdm(
        total=len(tasks),
        desc="Structural EM",
        unit="modèle",
        file=sys.stdout,
        disable=not bool(cfg["show_progress"]) or tqdm is None or workers > 1,
    ) if tqdm is not None else None
    try:
        if workers > 1:
            from joblib import Parallel, delayed

            parallel_cfg = _bn_parallel_config(config)
            iterator = Parallel(
                n_jobs=workers,
                backend=str(parallel_cfg.get("backend", "loky")),
                batch_size=1,
                verbose=0,
                return_as="generator",
            )(
                delayed(_fit_structural_em_initialization)(
                    data,
                    nodes,
                    dict(roles),
                    n_states,
                    seed,
                    initialization,
                    config,
                )
                for n_states, init_index, initialization, seed in tasks
            )
        else:
            def sequential_iterator() -> Iterable[StructuralEMResult]:
                for n_states, init_index, initialization, seed in tasks:
                    def update_progress(iteration: int, log_likelihood: float, n_edges: int) -> None:
                        if progress is not None:
                            progress.set_postfix_str(
                                f"K={n_states} init={init_index + 1}/{n_initializations} "
                                f"EM={iteration} arcs={n_edges}"
                            )

                    yield _fit_structural_em_initialization(
                        data,
                        nodes,
                        dict(roles),
                        n_states,
                        seed,
                        initialization,
                        config,
                        progress_callback=update_progress,
                    )
                    if progress is not None:
                        progress.update(1)

            iterator = sequential_iterator()
        if progress is None and tqdm is not None and bool(cfg["show_progress"]):
            progress = tqdm(iterator, total=len(tasks), desc="Structural EM", unit="modèle", file=sys.stdout, leave=True)
            iterator = progress
        return list(iterator)
    finally:
        if progress is not None:
            progress.close()


def fit_latent_bn_analysis(matrix: pd.DataFrame, roles: Mapping[str, str], config: Mapping[str, Any], output_dir: Path) -> tuple[StructuralEMResult, pd.DataFrame, list[str]]:
    """Fit and select the single constrained latent BN using Structural EM."""

    cfg = _bn_config(config)
    primary_states = [int(value) for value in cfg["latent_states"]]
    results = _run_structural_em_grid(matrix, roles, config, primary_states)
    selected, selection, selection_warnings = _select_latent_result(results, config, output_dir, primary_grid=primary_states)

    if bool(cfg.get("run_extended_k_if_boundary")) and selected.n_states == max(primary_states):
        extended_states = [int(value) for value in cfg.get("extended_latent_states", [12, 15])]
        extended_results = _run_structural_em_grid(matrix, roles, config, extended_states)
        results.extend(extended_results)
        extended_rows = []
        for result in extended_results:
            effective = result.effective_sizes
            tau = result.responsibilities
            entropy = float(-np.sum(tau * np.log(np.clip(tau, 1e-300, 1.0))))
            extended_rows.append({
                "K": result.n_states,
                "seed": result.seed,
                "bic": result.bic,
                "admissible": result.converged and float(effective.min()) >= float(cfg["min_latent_effective_n"]),
                "diagnostic_only": True,
                "entropy": entropy,
            })
        if extended_rows:
            extended_frame = pd.DataFrame(extended_rows)
            output_dir.mkdir(parents=True, exist_ok=True)
            extended_frame.to_csv(output_dir / "K_selection_extended.csv", index=False)
            selection_warnings.append(
                f"Grille étendue exécutée (K={extended_states}) car K*={selected.n_states} est sur la borne primaire."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _, selection, _ = _select_latent_result(results, config, output_dir, primary_grid=primary_states, eligible_k=primary_states)
    return selected, selection, selection_warnings


def finalize_latent_bn(result: StructuralEMResult, matrix: pd.DataFrame, roles: Mapping[str, str], config: Mapping[str, Any]) -> StructuralEMResult:
    """Apply post-selection Beta smoothing and construct the pgmpy model."""

    cfg = _bn_config(config)
    latent_scope = _latent_scope(config)
    data = matrix[result.nodes].to_numpy(dtype=np.int8)
    tau = result.responsibilities
    weights, upstream, downstream = _bn_mle_parameters(data, result.nodes, roles, result.edges, tau, result.n_states, float(cfg["probability_floor"]), latent_scope)
    alpha = float(cfg["alpha"])
    parents = _bn_parent_map(result.nodes, result.edges)
    index = {node: position for position, node in enumerate(result.nodes)}
    smoothed_upstream = {}
    smoothed_downstream = {}
    for key, probability in upstream.items():
        node, state, parent_values = key
        mask = np.ones(len(data), dtype=bool)
        for parent, value in zip(parents[node], parent_values):
            mask &= data[:, index[parent]] == value
        mass = tau[:, state] * mask
        denominator = float(mass.sum())
        numerator = float((mass * data[:, index[node]]).sum())
        smoothed_upstream[key] = (numerator + alpha) / (denominator + 2.0 * alpha)
    for key, probability in downstream.items():
        node, parent_values = key
        mask = np.ones(len(data), dtype=bool)
        for parent, value in zip(parents[node], parent_values):
            mask &= data[:, index[parent]] == value
        denominator = float(mask.sum())
        numerator = float((mask * data[:, index[node]]).sum())
        smoothed_downstream[key] = (numerator + alpha) / (denominator + 2.0 * alpha)
    final = StructuralEMResult(
        result.nodes, dict(roles), result.edges, result.n_states, weights, tau,
        smoothed_upstream, smoothed_downstream, result.log_likelihood,
        result.bic, result.n_iter, result.converged, result.seed,
        result.initialization, None, result.iteration_history,
        result.last_loglik_delta, result.relative_loglik_delta,
        result.same_graph, result.edges_added_last, result.edges_removed_last,
    )
    final.model = _build_pgmpy_model(final, smooth=False, alpha=alpha, latent_scope=latent_scope)
    return final


def _scenario_state_space(result: StructuralEMResult, roles: Mapping[str, str]) -> tuple[list[str], list[str], list[str]]:
    upstream = [node for node in result.nodes if roles[node] in {"A0", "A1"}]
    events = [node for node in result.nodes if roles[node] == "B"]
    consequences = [node for node in result.nodes if roles[node] == "C"]
    return upstream, events, consequences


def _exact_constrained_mpe(
    result: StructuralEMResult,
    roles: Mapping[str, str],
    state: int,
    config: Mapping[str, Any],
    forbidden: Sequence[Mapping[str, int]] | None = None,
    *,
    apply_role_constraints: bool = True,
) -> dict[str, int]:
    """Solve the exact role-constrained MPE as a local-factor MILP.

    Default role constraints: at least one factor in A0∪A1, at least one in B, at least one in C.
    Set ``apply_role_constraints=False`` for the unconstrained (free) MPE diagnostic.
    """

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as error:
        raise RuntimeError("scipy.optimize.milp est requis pour calculer le MPE exact.") from error

    cfg = _bn_config(config)
    latent_scope = _latent_scope(config)
    required_roles = [str(role) for role in cfg["mpe_required_roles"]]
    nodes = list(result.nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    parents = _bn_parent_map(nodes, result.edges)
    objective: list[float] = [0.0] * len(nodes)
    local_assignments: list[tuple[str, tuple[int, ...], int, float]] = []
    floor = 1e-12

    for node in nodes:
        parent_nodes = parents[node]
        for parent_values in itertools.product((0, 1), repeat=len(parent_nodes)):
            if is_latent_conditioned(roles[node], latent_scope):
                probability_one = result.upstream_probabilities.get((node, int(state), tuple(parent_values)), 0.5)
            else:
                probability_one = result.downstream_probabilities.get((node, tuple(parent_values)), 0.5)
            probability_one = float(np.clip(probability_one, floor, 1.0 - floor))
            for node_value in (0, 1):
                probability = probability_one if node_value else 1.0 - probability_one
                local_assignments.append((node, tuple(parent_values), node_value, -math.log(probability)))

    objective.extend(item[3] for item in local_assignments)
    n_variables = len(objective)
    rows: list[tuple[int, int, float]] = []
    lower: list[float] = []
    upper: list[float] = []
    local_offset = len(nodes)
    assignment_offsets: dict[str, list[int]] = {node: [] for node in nodes}

    def add_constraint(entries: Mapping[int, float], lower_bound: float, upper_bound: float) -> None:
        row_index = len(lower)
        rows.extend((row_index, column, coefficient) for column, coefficient in entries.items())
        lower.append(lower_bound)
        upper.append(upper_bound)

    for offset, (node, parent_values, node_value, _) in enumerate(local_assignments):
        assignment_offsets[node].append(local_offset + offset)

    for node in nodes:
        node_assignments = assignment_offsets[node]
        add_constraint({column: 1.0 for column in node_assignments}, 1.0, 1.0)
        node_one_assignments = [
            local_offset + index
            for index, (assignment_node, _, node_value, _) in enumerate(local_assignments)
            if assignment_node == node and node_value == 1
        ]
        entries = {node_index[node]: 1.0}
        entries.update({column: -1.0 for column in node_one_assignments})
        add_constraint(entries, 0.0, 0.0)
        parent_nodes = parents[node]
        for parent_position, parent in enumerate(parent_nodes):
            parent_one_assignments = [
                local_offset + index
                for index, (assignment_node, parent_values, _, _) in enumerate(local_assignments)
                if assignment_node == node and parent_values[parent_position] == 1
            ]
            entries = {node_index[parent]: 1.0}
            entries.update({column: -1.0 for column in parent_one_assignments})
            add_constraint(entries, 0.0, 0.0)

    if apply_role_constraints:
        for role in required_roles:
            group = {node for node in nodes if roles[node] == role}
            if not group:
                raise ValueError(f"Le MPE contraint nécessite au moins un thème {role}.")
            add_constraint({node_index[node]: 1.0 for node in group}, 1.0, np.inf)

        upstream_any_roles = [str(role) for role in cfg.get("mpe_upstream_any_roles", ["A0", "A1"])]
        if upstream_any_roles:
            upstream_group = {node for node in nodes if roles[node] in upstream_any_roles}
            if not upstream_group:
                raise ValueError(
                    f"Le MPE contraint nécessite au moins un thème parmi {upstream_any_roles}."
                )
            add_constraint({node_index[node]: 1.0 for node in upstream_group}, 1.0, np.inf)

    for previous in forbidden or []:
        n_zeros = sum(1 for node in nodes if int(previous[node]) == 0)
        entries = {node_index[node]: float(2 * int(previous[node]) - 1) for node in nodes}
        add_constraint(entries, -np.inf, float(len(nodes) - 1 - n_zeros))

    constraint_matrix = lil_matrix((len(lower), n_variables), dtype=float)
    for row_index, column, coefficient in rows:
        constraint_matrix[row_index, column] = coefficient
    solution = milp(
        c=np.asarray(objective, dtype=float),
        integrality=np.ones(n_variables, dtype=np.int8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(constraint_matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"mip_rel_gap": 0.0},
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"Le MPE MILP exact a échoué pour la famille {state + 1}: {solution.message}")
    assignment = np.rint(solution.x[:len(nodes)]).astype(int)
    return {node: int(assignment[node_index[node]]) for node in nodes}


def _exact_free_mpe(
    result: StructuralEMResult,
    roles: Mapping[str, str],
    state: int,
    config: Mapping[str, Any],
) -> dict[str, int]:
    """Unconstrained MPE diagnostic (no upstream/B/C role minima)."""

    return _exact_constrained_mpe(result, roles, state, config, apply_role_constraints=False)


def _mpe_role_constraint_satisfied(assignment: Mapping[str, int], roles: Mapping[str, str], config: Mapping[str, Any]) -> bool:
    cfg = _bn_config(config)
    for role in cfg.get("mpe_required_roles", ["B", "C"]):
        if not any(assignment.get(node, 0) for node, node_role in roles.items() if node_role == role):
            return False
    upstream_roles = {str(role) for role in cfg.get("mpe_upstream_any_roles", ["A0", "A1"])}
    if upstream_roles and not any(assignment.get(node, 0) for node, node_role in roles.items() if node_role in upstream_roles):
        return False
    return True


def _exact_constrained_mpe_top_m(
    result: StructuralEMResult,
    roles: Mapping[str, str],
    state: int,
    config: Mapping[str, Any],
    matrix: pd.DataFrame | None = None,
    responsibilities: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return up to ``top_m_mpe`` distinct constrained MPE solutions via no-good cuts."""

    top_m = int(_bn_config(config)["top_m_mpe"])
    nodes = result.nodes
    solutions: list[dict[str, int]] = []
    ranked: list[dict[str, Any]] = []
    for rank in range(1, top_m + 1):
        try:
            assignment = _exact_constrained_mpe(result, roles, state, config, forbidden=solutions)
        except RuntimeError:
            break
        if assignment in solutions:
            break
        solutions.append(assignment)
        log_probability = math.log(max(_family_probability(result, assignment, state, config), 1e-300))
        positive = [node for node in nodes if assignment[node]]
        family_support = np.nan
        global_support = np.nan
        if matrix is not None and positive:
            positive_mask = matrix[positive].all(axis=1).to_numpy(dtype=bool)
            global_support = float(positive_mask.mean())
            if responsibilities is not None:
                family_support = float(
                    np.sum(responsibilities[:, state] * positive_mask)
                    / max(responsibilities[:, state].sum(), 1e-12)
                )
        ranked.append({
            "family_id": state + 1,
            "mpe_rank": rank,
            "log_probability": log_probability,
            "probability": math.exp(log_probability),
            "mpe_probability": math.exp(log_probability),
            "n_positive_factors": len(positive),
            "A0_factors": ";".join(node for node in positive if roles[node] == "A0"),
            "A1_factors": ";".join(node for node in positive if roles[node] == "A1"),
            "B_factors": ";".join(node for node in positive if roles[node] == "B"),
            "C_factors": ";".join(node for node in positive if roles[node] == "C"),
            "positive_factors": ";".join(positive),
            "family_support": family_support,
            "global_support": global_support,
            **{f"{node}_value": int(assignment[node]) for node in nodes},
        })
    return ranked


def _family_probability(result: StructuralEMResult, values: Mapping[str, int], state: int, config: Mapping[str, Any]) -> float:
    frame = np.asarray([[int(values[node]) for node in result.nodes]], dtype=np.int8)
    latent_scope = _latent_scope(config)
    floor = float(_bn_config(config)["probability_floor"])
    log_joint = _bn_log_joint(
        frame, result.nodes, result.roles, result.edges, result.weights,
        result.upstream_probabilities, result.downstream_probabilities, floor, latent_scope,
    )[0, state]
    return float(np.exp(log_joint - math.log(max(float(result.weights[state]), floor))))


def _theme_label_map(theme_dictionary: pd.DataFrame | None) -> dict[str, str]:
    if theme_dictionary is None or theme_dictionary.empty:
        return {}
    if not {"variable_name", "topic_label"}.issubset(theme_dictionary.columns):
        return {}
    return {
        str(row.variable_name): str(row.topic_label)
        for row in theme_dictionary[["variable_name", "topic_label"]].itertuples(index=False)
    }


def _write_recurrent_scenarios_graph(
    scenarios: pd.DataFrame,
    result: StructuralEMResult,
    label_map: Mapping[str, str],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as error:
        warnings.warn(f"Graphe des scénarios non généré: {error}", RuntimeWarning)
        return
    if scenarios.empty:
        return
    from scenario_figures import normalize_scenario_table, render_compact_scenarios_figure

    scenario_table = scenarios.copy()
    scenario_table["scenario_id"] = scenario_table["family_id"].map(lambda value: f"S{int(value)}")
    scenario_table["latent_family"] = scenario_table["family_id"].map(lambda value: str(int(value)))
    scenario_table["heading"] = scenario_table.apply(
        lambda row: " -> ".join(
            str(row[column])
            for column in ("A0_labels", "A1_labels", "B_labels", "C_labels")
            if str(row.get(column, "")).strip() and str(row.get(column, "")).strip() != "—"
        ),
        axis=1,
    )
    scenario_table["factor_codes"] = scenario_table.apply(
        lambda row: ";".join(
            str(row[column])
            for column in ("A0_factors", "A1_factors", "B_factors", "C_factors")
            if str(row.get(column, "")).strip()
        ),
        axis=1,
    )
    scenario_table = scenario_table.rename(columns={
        "A0_labels": "A0_label",
        "A1_labels": "A1_label",
        "B_labels": "B_label",
        "C_labels": "C_label",
        "family_positive_support": "family_support",
        "global_positive_support": "global_support",
        "omega": "omega",
        "N_eff": "N_eff",
    })
    render_compact_scenarios_figure(
        scenario_table,
        output_dir,
        learned_edges=result.edges,
        stem="recurrent_scenarios_compact",
    )


def _select_prototype(
    matrix: pd.DataFrame,
    nodes: Sequence[str],
    responsibilities: np.ndarray,
    state: int,
    positive: Sequence[str],
) -> tuple[int, bool, float, list[str], int, int]:
    """Pick an observed accident prototype aligned with MPE rank 1."""

    data = matrix[list(nodes)].to_numpy(dtype=np.int8)
    positive_list = list(positive)
    n_mpe = len(positive_list)
    if positive_list:
        exact_mask = data[:, [nodes.index(node) for node in positive_list]].all(axis=1)
    else:
        exact_mask = np.ones(len(matrix), dtype=bool)
    exact_indices = np.flatnonzero(exact_mask)
    if len(exact_indices):
        prototype_index = int(exact_indices[np.argmax(responsibilities[exact_indices, state])])
        return prototype_index, True, 1.0, [], n_mpe, n_mpe
    coverage_scores = []
    for index in range(len(matrix)):
        observed = {node for node in positive_list if data[index, nodes.index(node)] == 1}
        coverage = len(observed) / max(len(positive_list), 1)
        coverage_scores.append((coverage, float(responsibilities[index, state]), index))
    _, _, prototype_index = max(coverage_scores, key=lambda item: (item[0], item[1]))
    observed = {node for node in positive_list if data[prototype_index, nodes.index(node)] == 1}
    missing = [node for node in positive_list if node not in observed]
    coverage = len(observed) / max(len(positive_list), 1)
    return int(prototype_index), False, coverage, missing, n_mpe, len(observed)


def extract_latent_bn_scenarios(result: StructuralEMResult, matrix: pd.DataFrame, roles: Mapping[str, str], units: pd.DataFrame, config: Mapping[str, Any], output_dir: Path, theme_dictionary: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract constrained MPEs, supports and observed accident prototypes."""

    cfg = _bn_config(config)
    latent_scope = _latent_scope(config)
    label_map = _theme_label_map(theme_dictionary)
    nodes = result.nodes
    data = matrix[nodes].to_numpy(dtype=np.int8)
    floor = float(cfg["probability_floor"])
    log_joint = _bn_log_joint(
        data, nodes, roles, result.edges, result.weights,
        result.upstream_probabilities, result.downstream_probabilities, floor, latent_scope,
    )
    responsibilities = np.exp(log_joint - _bn_logsumexp(log_joint, axis=1)[:, None])
    responsibilities_frame = pd.DataFrame(responsibilities, columns=[f"family_{state + 1}" for state in range(result.n_states)])
    responsibilities_frame.insert(0, "accident_id", matrix["accident_id"].astype(str).to_numpy())

    family_sizes = pd.DataFrame({"family_id": np.arange(1, result.n_states + 1), "omega": result.weights, "N_eff": responsibilities.sum(axis=0)})
    profile_rows = []
    for state in range(result.n_states):
        for node in nodes:
            try:
                from pgmpy.inference import VariableElimination
                query = VariableElimination(result.model).query([node], evidence={"Z": state}, show_progress=False)
                marginal = float(query.values[1])
            except Exception as error:
                raise RuntimeError(
                    f"L'inférence exacte du profil a échoué pour la famille {state + 1}, variable {node}."
                ) from error
            profile_rows.append({"family_id": state + 1, "variable_name": node, "role": roles[node], "probability": marginal})
    profiles = pd.DataFrame(profile_rows)
    upstream, events, consequences = _scenario_state_space(result, roles)
    if not upstream or not events or not consequences:
        raise ValueError("Le MPE contraint nécessite au moins un thème upstream, B et C.")
    scenario_rows = []
    support_rows = []
    prototype_rows = []
    mpe_ranked_rows: list[dict[str, Any]] = []
    mpe_free_diagnostic_rows: list[dict[str, Any]] = []
    scenario_warnings: list[str] = []
    latent_heterogeneity = result.n_states > 1
    mpe_gap_threshold = float(cfg.get("near_equivalent_mpe_log_gap", 0.10))
    if not latent_heterogeneity:
        scenario_warnings.append(
            "Selected K=1: the data do not support additional latent accident-family "
            "heterogeneity under the tested model."
        )
    for state in range(result.n_states):
        ranked = _exact_constrained_mpe_top_m(result, roles, state, config, matrix, responsibilities)
        if not ranked:
            raise RuntimeError(f"Aucun MPE trouvé pour la famille {state + 1}.")
        log_gap_12 = np.nan
        log_gap_23 = np.nan
        if len(ranked) >= 2:
            log_gap_12 = float(ranked[0]["log_probability"] - ranked[1]["log_probability"])
            if log_gap_12 < mpe_gap_threshold:
                scenario_warnings.append(
                    f"Famille {state + 1}: MPE rank 1 et 2 quasi équivalents (log gap={log_gap_12:.4f})."
                )
        if len(ranked) >= 3:
            log_gap_23 = float(ranked[1]["log_probability"] - ranked[2]["log_probability"])
        for row in ranked:
            row["mpe_log_gap_1_2"] = log_gap_12 if row["mpe_rank"] == 1 else np.nan
            row["mpe_log_gap_2_3"] = log_gap_23 if row["mpe_rank"] == 1 else np.nan
            if row["mpe_rank"] == 2 and len(ranked) >= 2:
                row["probability_ratio_2_to_1"] = math.exp(ranked[1]["log_probability"] - ranked[0]["log_probability"])
            if row["mpe_rank"] == 3 and len(ranked) >= 3:
                row["probability_ratio_3_to_1"] = math.exp(ranked[2]["log_probability"] - ranked[0]["log_probability"])
            mpe_ranked_rows.append(row)
        best = {node: int(ranked[0][f"{node}_value"]) for node in nodes}
        mpe_probability = float(ranked[0]["mpe_probability"])
        positive = [node for node in nodes if best[node]]
        positive_mask = matrix[positive].all(axis=1).to_numpy(dtype=bool) if positive else np.zeros(len(matrix), dtype=bool)
        n_exact_matching = int(positive_mask.sum())
        family_support = float(np.sum(responsibilities[:, state] * positive_mask) / max(responsibilities[:, state].sum(), 1e-12))
        global_support = float(positive_mask.mean())
        posterior_weighted_exact = float(np.sum(responsibilities[:, state] * positive_mask) / max(responsibilities[:, state].sum(), 1e-12))
        support_enrichment = family_support / max(global_support, 1e-6)
        prototype_index, exact_match, prototype_coverage, missing_factors, n_mpe_pos, n_matched = _select_prototype(
            matrix, nodes, responsibilities, state, positive,
        )
        if latent_heterogeneity and not exact_match:
            scenario_warnings.append(f"Famille {state + 1}: aucun prototype exact observé pour le MPE rank 1.")
        prototype_accident = str(matrix.iloc[prototype_index]["accident_id"])
        rank2 = ranked[1] if len(ranked) > 1 else None
        rank3 = ranked[2] if len(ranked) > 2 else None
        scenario_kind = "latent_family" if latent_heterogeneity else "global_descriptive"
        scenario_rows.append({
            "family_id": state + 1,
            "scenario_id": "global" if not latent_heterogeneity else f"S{state + 1}",
            "scenario_kind": scenario_kind,
            "latent_heterogeneity_supported": latent_heterogeneity,
            "N_eff": float(responsibilities[:, state].sum()),
            "omega": float(result.weights[state]),
            "A0_factors": ";".join(node for node in positive if roles[node] == "A0"),
            "A1_factors": ";".join(node for node in positive if roles[node] == "A1"),
            "B_factors": ";".join(node for node in positive if roles[node] == "B"),
            "C_factors": ";".join(node for node in positive if roles[node] == "C"),
            "A0_labels": ";".join(label_map.get(node, node) for node in positive if roles[node] == "A0"),
            "A1_labels": ";".join(label_map.get(node, node) for node in positive if roles[node] == "A1"),
            "B_labels": ";".join(label_map.get(node, node) for node in positive if roles[node] == "B"),
            "C_labels": ";".join(label_map.get(node, node) for node in positive if roles[node] == "C"),
            "mpe_probability": mpe_probability,
            "mpe_method": "exact_milp_(A0|A1)+B+C",
            "family_positive_support": family_support,
            "global_positive_support": global_support,
            "support_enrichment_ratio": support_enrichment,
            "n_exact_matching_accidents": n_exact_matching,
            "posterior_weighted_exact_support": posterior_weighted_exact,
            "mpe_log_gap_1_2": log_gap_12,
            "mpe_log_gap_2_3": log_gap_23,
            "mpe_rank2_probability": float(rank2["mpe_probability"]) if rank2 else np.nan,
            "mpe_rank3_probability": float(rank3["mpe_probability"]) if rank3 else np.nan,
            "prototype_accident_id": prototype_accident,
            "prototype_exact_mpe_match": exact_match,
            "prototype_mpe_coverage": prototype_coverage,
            "missing_mpe_factors": ";".join(missing_factors),
            "prototype_probability": _family_probability(result, dict(zip(nodes, data[prototype_index])), state, config),
            "prototype_posterior_membership": float(responsibilities[prototype_index, state]),
        })
        support_rows.append({
            "family_id": state + 1,
            "positive_factors": ";".join(positive),
            "family_positive_support": family_support,
            "global_positive_support": global_support,
            "support_enrichment_ratio": support_enrichment,
            "n_exact_matching_accidents": n_exact_matching,
            "mpe_probability": mpe_probability,
            "mpe_method": "exact_milp_(A0|A1)+B+C",
        })
        if cfg.get("mpe_compute_free_diagnostic", True):
            try:
                free_assignment = _exact_free_mpe(result, roles, state, config)
                constrained_log_p = math.log(max(_family_probability(result, best, state, config), 1e-300))
                free_log_p = math.log(max(_family_probability(result, free_assignment, state, config), 1e-300))
                free_positive = [node for node in nodes if free_assignment[node]]
                mpe_free_diagnostic_rows.append({
                    "family_id": state + 1,
                    "constrained_log_probability": constrained_log_p,
                    "free_log_probability": free_log_p,
                    "log_probability_gain_free": free_log_p - constrained_log_p,
                    "same_assignment": all(free_assignment[node] == best[node] for node in nodes),
                    "constrained_n_positive": len(positive),
                    "free_n_positive": len(free_positive),
                    "free_satisfies_role_constraints": _mpe_role_constraint_satisfied(free_assignment, roles, config),
                    "constrained_upstream_role": next(
                        (roles[node] for node in positive if roles[node] in {"A0", "A1"}), "",
                    ),
                    "free_upstream_role": next(
                        (roles[node] for node in free_positive if roles[node] in {"A0", "A1"}), "",
                    ),
                })
            except Exception as error:
                scenario_warnings.append(f"Famille {state + 1}: diagnostic MPE libre indisponible ({error}).")
        prototype_units = units[units["_accident_id"].astype(str).eq(prototype_accident)].copy()
        prototype_rows.append({
            "family_id": state + 1,
            "scenario_id": "global" if not latent_heterogeneity else f"S{state + 1}",
            "accident_id": prototype_accident,
            "prototype_accident_id": prototype_accident,
            "prototype_exact_mpe_match": exact_match,
            "prototype_mpe_coverage": prototype_coverage,
            "missing_mpe_factors": ";".join(missing_factors),
            "n_mpe_positive_factors": n_mpe_pos,
            "n_matched_positive_factors": n_matched,
            "probability": _family_probability(result, dict(zip(nodes, data[prototype_index])), state, config),
            "posterior_membership": float(responsibilities[prototype_index, state]),
            "prototype_posterior_membership": float(responsibilities[prototype_index, state]),
            "fact_ids": ";".join(prototype_units["_fact_id"].astype(str)),
            "sentences": " || ".join(prototype_units.get("_text", pd.Series(dtype=str)).astype(str)[:20]),
        })
    scenarios = pd.DataFrame(scenario_rows)
    supports = pd.DataFrame(support_rows)
    prototypes = pd.DataFrame(prototype_rows)
    mpe_ranked = pd.DataFrame(mpe_ranked_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    responsibilities_frame.to_parquet(output_dir / "posterior_responsibilities.parquet", index=False)
    responsibilities_frame.to_csv(output_dir / "posterior_responsibilities.csv", index=False)
    family_sizes.to_csv(output_dir / "latent_family_sizes.csv", index=False)
    profiles.to_csv(output_dir / "family_factor_profiles.csv", index=False)
    scenarios.to_csv(output_dir / "recurrent_scenarios.csv", index=False)
    supports.to_csv(output_dir / "scenario_support.csv", index=False)
    prototypes.to_csv(output_dir / "scenario_prototypes.csv", index=False)
    prototypes.to_csv(output_dir / "prototypes.csv", index=False)
    mpe_ranked.to_csv(output_dir / "mpe_ranked_solutions.csv", index=False)
    if mpe_free_diagnostic_rows:
        pd.DataFrame(mpe_free_diagnostic_rows).to_csv(output_dir / "mpe_free_diagnostic.csv", index=False)
    from scenario_latex_export import write_recurrent_scenarios_latex, write_recurrent_scenarios_article

    write_recurrent_scenarios_latex(scenarios, result, matrix, roles, responsibilities, label_map, output_dir)
    write_recurrent_scenarios_article(scenarios, output_dir)
    for warning in scenario_warnings:
        warnings.warn(warning, RuntimeWarning)
    _write_recurrent_scenarios_graph(scenarios, result, label_map, figures_dir)
    return scenarios, supports, prototypes, profiles


def _short_semantic_label(label: str, max_length: int = 32) -> str:
    text = re.sub(r"\s+", " ", str(label)).strip()
    return text if len(text) <= max_length else text[: max_length - 1].rstrip() + "…"


def _edge_conditional_contrast_signed(result: StructuralEMResult, parent: str, child: str) -> float:
    """Mean signed contrast: P(child=1|parent=1,…) − P(child=1|parent=0,…) averaged over other parents."""
    parents = _bn_parent_map(result.nodes, result.edges)[child]
    parent_position = parents.index(parent)
    contrasts = []
    other_positions = [index for index in range(len(parents)) if index != parent_position]
    for other_values in itertools.product((0, 1), repeat=len(other_positions)):
        parent_values = [0] * len(parents)
        for index, value in zip(other_positions, other_values):
            parent_values[index] = value
        probabilities = []
        for parent_value in (0, 1):
            parent_values[parent_position] = parent_value
            key_values = tuple(parent_values)
            if result.roles[child] in {"A0", "A1"}:
                probability = sum(
                    float(result.weights[state])
                    * float(result.upstream_probabilities.get((child, state, key_values), 0.5))
                    for state in range(result.n_states)
                )
            else:
                probability = float(result.downstream_probabilities.get((child, key_values), 0.5))
            probabilities.append(probability)
        contrasts.append(probabilities[1] - probabilities[0])
    return float(np.mean(contrasts)) if contrasts else 0.0


def _edge_conditional_strength(result: StructuralEMResult, parent: str, child: str) -> float:
    """Legacy alias: absolute mean conditional contrast (magnitude only)."""
    return abs(_edge_conditional_contrast_signed(result, parent, child))


def _write_conceptual_bn_architecture(output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as error:
        warnings.warn(f"Architecture conceptuelle non générée: {error}", RuntimeWarning)
        return
    graph = nx.DiGraph([
        ("Z", "A0"), ("Z", "A1"), ("A0", "A1"),
        ("A0", "B"), ("A1", "B"), ("B", "C"),
    ])
    positions = {"Z": (0.0, 1.0), "A0": (1.0, 1.5), "A1": (1.0, 0.5), "B": (2.0, 1.0), "C": (3.0, 1.0)}
    labels = {"Z": "Z\nFamille latente", "A0": "A0\nContexte", "A1": "A1\nCondition adverse", "B": "B\nÉvénement", "C": "C\nConséquence"}
    figure, axis = plt.subplots(figsize=(12, 5))
    nx.draw_networkx(
        graph,
        positions,
        labels=labels,
        ax=axis,
        node_color=[_role_color(node) for node in graph.nodes],
        node_size=2800,
        font_size=10,
        arrows=True,
        arrowsize=22,
        edge_color="#555555",
        width=2.0,
    )
    axis.set_title("Architecture conceptuelle du modèle bayésien latent")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "conceptual_bn_architecture.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def _write_simplified_learned_bn_graph(result: StructuralEMResult, label_map: Mapping[str, str], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as error:
        warnings.warn(f"Réseau appris simplifié non généré: {error}", RuntimeWarning)
        return
    graph = nx.DiGraph()
    graph.add_nodes_from(result.nodes)
    graph.add_edges_from(result.edges)
    strengths = {(parent, child): _edge_conditional_strength(result, parent, child) for parent, child in result.edges}
    max_strength = max(strengths.values(), default=1.0)
    positions = {}
    for role_index, role in enumerate(ROLES):
        role_nodes = [node for node in result.nodes if result.roles[node] == role]
        center = (len(role_nodes) - 1) / 2
        for node_index, node in enumerate(sorted(role_nodes)):
            positions[node] = (role_index, center - node_index)
    labels = {node: _short_semantic_label(label_map.get(node, node)) for node in result.nodes}
    figure, axis = plt.subplots(figsize=(18, max(9, len(result.nodes) * 0.22)))
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_color=[_role_color(result.roles[node]) for node in graph.nodes],
        node_size=1700,
        edgecolors="#444444",
        linewidths=0.7,
    )
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axis, font_size=7)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        arrows=True,
        arrowsize=16,
        edge_color="#555555",
        width=[1.0 + 4.0 * strengths[edge] / max_strength for edge in graph.edges],
        connectionstyle="arc3,rad=0.03",
    )
    axis.set_title("Réseau appris simplifié — arcs observés entre facteurs")
    axis.text(0.5, -0.04, "Épaisseur : contraste conditionnel moyen | aucune arête Z affichée", transform=axis.transAxes, ha="center", fontsize=9)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "learned_bn_simplified.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def write_final_bn_outputs(
    result: StructuralEMResult,
    output_dir: Path,
    theme_dictionary: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    label_map = _theme_label_map(theme_dictionary)
    learned_edges = pd.DataFrame(result.edges, columns=["parent", "child"])
    learned_edges["parent_role"] = learned_edges["parent"].map(result.roles)
    learned_edges["child_role"] = learned_edges["child"].map(result.roles)
    learned_edges["parent_label"] = learned_edges["parent"].map(lambda node: label_map.get(node, node))
    learned_edges["child_label"] = learned_edges["child"].map(lambda node: label_map.get(node, node))
    learned_edges["conditional_strength"] = [
        _edge_conditional_strength(result, parent, child)
        for parent, child in result.edges
    ]
    learned_edges.to_csv(output_dir / "learned_edges.csv", index=False)
    learned_edges.to_csv(output_dir / "learned_bn_edges.csv", index=False)
    _write_conceptual_bn_architecture(figures_dir)
    write_diagnostic_figures = bool(_bn_config(config or {}).get("write_diagnostic_bn_figures", False))
    diagnostic_dir = figures_dir / "diagnostics"
    if write_diagnostic_figures:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        _write_simplified_learned_bn_graph(result, label_map, diagnostic_dir)
    rows = []
    parents = _bn_parent_map(result.nodes, result.edges)
    for node in result.nodes:
        for state in range(result.n_states) if result.roles[node] in {"A0", "A1"} else [None]:
            for parent_values in itertools.product((0, 1), repeat=len(parents[node])):
                probability = result.upstream_probabilities.get((node, state, parent_values)) if state is not None else result.downstream_probabilities.get((node, parent_values))
                rows.append({"variable_name": node, "role": result.roles[node], "family_id": None if state is None else state + 1, "parent_values": "|".join(map(str, parent_values)), "P(value=1)": probability, "P(value=0)": 1.0 - probability})
    pd.DataFrame(rows).to_csv(output_dir / "final_CPT_summary.csv", index=False)
    if not write_diagnostic_figures:
        return
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        graph = nx.DiGraph()
        graph.add_nodes_from(["Z", *result.nodes])
        graph.add_edges_from(result.edges)
        graph.add_edges_from(("Z", node) for node in result.nodes if result.roles[node] in {"A0", "A1"})
        positions = {"Z": (-1.0, 0.5)}
        for role_index, role in enumerate(ROLES):
            role_nodes = [node for node in result.nodes if result.roles[node] == role]
            for index, node in enumerate(sorted(role_nodes)):
                positions[node] = (role_index, -float(index))
        figure, axis = plt.subplots(figsize=(16, max(8, len(result.nodes) * 0.18)))
        labels = {node: f"{node}\n{label_map.get(node, 'Famille latente')}" if node != "Z" else "Z\nFamille latente" for node in graph.nodes}
        nx.draw_networkx(graph, positions, ax=axis, labels=labels, node_size=900, font_size=7, node_color=[_role_color(result.roles.get(node, "Z")) for node in graph.nodes], arrows=True)
        axis.axis("off")
        figure.tight_layout()
        from manuscript_reporting import save_manuscript_figure
        figure.savefig(diagnostic_dir / "final_bn_full.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        save_manuscript_figure(figure, diagnostic_dir / "final_bn_full.pdf", dpi=220)
        figure.savefig(diagnostic_dir / "final_bn.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
    except Exception as error:
        warnings.warn(f"Figure BN diagnostic non générée: {error}", RuntimeWarning)


def run_frozen_bn_analysis(config: Mapping[str, Any], run_dir: Path, partition_selections: Mapping[str, str], output_dir: Path, units: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run the complete downstream analysis for one dataset."""

    if units is None:
        units, _ = load_units(config)
    matrix, included, excluded, roles = build_frozen_bn_inputs(units, run_dir, partition_selections, config, output_dir)
    write_bn_accident_inclusion_audit(units, matrix, roles, output_dir)
    selected, selection, selection_warnings = fit_latent_bn_analysis(matrix, roles, config, output_dir)
    final = finalize_latent_bn(selected, matrix, roles, config)
    write_final_bn_outputs(final, output_dir, included, config)
    scenarios, supports, prototypes, profiles = extract_latent_bn_scenarios(final, matrix, roles, units, config, output_dir, included)

    from bn_reporting import (
        run_latent_scope_sensitivity,
        write_bn_reporting,
    )

    factor_prevalence_path = output_dir / "factor_prevalence.csv"
    factor_prevalence = pd.read_csv(factor_prevalence_path) if factor_prevalence_path.is_file() else None
    responsibilities = pd.read_parquet(output_dir / "posterior_responsibilities.parquet")
    diagnostic = write_bn_reporting(
        final,
        selection,
        scenarios,
        prototypes,
        profiles,
        matrix,
        roles,
        config,
        output_dir,
        selection_warnings=selection_warnings,
        factor_prevalence=factor_prevalence,
        responsibilities=responsibilities,
        theme_dictionary=included,
    )
    run_latent_scope_sensitivity(matrix, roles, config, output_dir)
    return {
        "matrix": matrix,
        "theme_dictionary": included,
        "excluded_themes": excluded,
        "selection": selection,
        "selection_warnings": selection_warnings,
        "result": final,
        "scenarios": scenarios,
        "supports": supports,
        "prototypes": prototypes,
        "profiles": profiles,
        "diagnostic": diagnostic,
    }


def load_selected_configurations(run_dir: Path) -> dict[str, str]:
    """Load role -> configuration_id from selected_configurations.csv."""
    path = Path(run_dir) / "selected_configurations.csv"
    if not path.is_file():
        raise FileNotFoundError(f"selected_configurations.csv introuvable: {path}")
    frame = pd.read_csv(path)
    if not {"role", "configuration_id"}.issubset(frame.columns):
        raise ValueError("selected_configurations.csv must contain role and configuration_id")
    return {
        str(row["role"]): str(row["configuration_id"])
        for _, row in frame.iterrows()
        if str(row["role"]) in ROLES and str(row["configuration_id"]).strip()
    }


def run_theme_discovery(
    config_path: Path,
    *,
    dataset_id: str | None = None,
    reestimate: bool = False,
    stage: str = "all",
    run_dir: Path | None = None,
    chat_completion: Callable[..., str] | None = None,
) -> Path:
    """Run theme discovery with Pareto screening and geometric knee selection.

    Stages:
    - ``metrics``: candidate DBCV and accident-level ``S_R``;
    - ``select``: Pareto + geometric knee, materialize partitions, figures;
    - ``seed``: UMAP seed sensitivity for selected configurations;
    - ``all``: metrics then select then seed.

    ``evaluate`` is accepted as a deprecated alias for ``select``.
    """
    del chat_completion
    if stage == "evaluate":
        warnings.warn(
            "stage=evaluate is deprecated (semantic evaluation removed); using stage=select.",
            stacklevel=2,
        )
        stage = "select"
    if stage not in {"all", "metrics", "select", "seed"}:
        raise ValueError("stage must be one of: all, metrics, select, seed")

    raw_config = select_dataset_config(load_yaml_config(config_path), dataset_id)
    config = resolve_config_paths(raw_config, config_path)

    if run_dir is not None:
        output_base = Path(run_dir).expanduser().resolve()
    else:
        output_base = Path(config["data"]["output_dir"])
        if (
            stage in {"all", "metrics"}
            and output_base.exists()
            and any(output_base.iterdir())
            and not config.get("runtime", {}).get("overwrite", False)
        ):
            output_base = output_base.with_name(
                output_base.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )

    if stage in {"select", "seed"} and not output_base.is_dir():
        raise FileNotFoundError(f"Existing run directory not found: {output_base}")
    output_base.mkdir(parents=True, exist_ok=True)
    _log_progress(
        f"START dataset={config['data'].get('dataset_id')} stage={stage} "
        f"output={output_base} workers={resolve_n_workers(config)}"
    )

    if stage in {"all", "metrics"}:
        (output_base / "config_resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (output_base / "parallel_runtime.json").write_text(
            json.dumps({
                "n_workers": resolve_n_workers(config),
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "backend": config.get("parallel", {}).get("backend", "loky"),
                "inner_umap_n_jobs": config.get("screening", {}).get("umap", {}).get("n_jobs", 1),
            }, indent=2),
            encoding="utf-8",
        )

    candidate_tables: dict[str, pd.DataFrame] = {}
    theme_tables: dict[str, pd.DataFrame] = {}
    prepared: PreparedData | None = None

    if stage in {"all", "metrics"}:
        prepared = prepare_data(config, output_base)
        _log_progress(
            f"données prêtes: {len(prepared.units)} unités, "
            f"{prepared.units['_accident_id'].nunique()} accidents"
        )
        for role in ROLES:
            candidates = evaluate_candidates(
                role, prepared.units, prepared.embeddings, config, output_base, reestimate=reestimate
            )
            theme_frame, summary = evaluate_resampling_stability(
                role, prepared.units, prepared.embeddings, config, output_base, candidates, reestimate=reestimate
            )
            candidate_tables[role] = candidates.merge(summary, on=["role", "configuration_id"], how="left")
            theme_tables[role] = theme_frame
    elif stage in {"select", "seed"}:
        for role in ROLES:
            role_dir = _discovery_role_dir(output_base, role)
            candidate_path = role_dir / "candidate_metrics.csv"
            stability_path = role_dir / "stability_summary.csv"
            theme_path = role_dir / "stability_theme.csv"
            missing = [str(path) for path in (candidate_path, stability_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot run evaluate/select/seed stage; missing metric artifacts: " + ", ".join(missing)
                )
            candidates = pd.read_csv(candidate_path)
            summary = pd.read_csv(stability_path)
            required = {"role", "configuration_id"}
            if not required.issubset(candidates.columns) or not required.issubset(summary.columns):
                raise ValueError(f"Invalid cached metric tables for role {role}")
            candidate_tables[role] = candidates.merge(summary, on=["role", "configuration_id"], how="left")
            theme_tables[role] = pd.read_csv(theme_path) if theme_path.is_file() else pd.DataFrame()

    if stage == "metrics":
        _log_progress(f"DONE stage=metrics output={output_base}")
        return output_base

    from pareto_knee_selection import (
        identify_pareto_front,
        print_role_selection_summary,
        summarize_selected_configurations,
    )

    for role in ROLES:
        candidate_tables[role] = identify_pareto_front(candidate_tables[role])
        marked = candidate_tables[role]
        marked.loc[marked["is_pareto"]].to_csv(
            _discovery_role_dir(output_base, role) / "pareto_front.csv",
            index=False,
        )

    validation_cfg = _validation_config(config)
    selection_tables: dict[str, pd.DataFrame] = {}
    selections: dict[str, str] = {}
    selection_rules: dict[str, str] = {}

    if stage in {"all", "select"}:
        if prepared is None:
            prepared = prepare_data(config, output_base)
        _log_progress("Sélection: front Pareto + geometric knee point")
        selected_rows = []
        for role in ROLES:
            table, selected_id, rule = select_configuration_for_role(candidate_tables[role])
            if not selected_id:
                n_pareto = int(table["is_pareto"].fillna(False).astype(bool).sum()) if "is_pareto" in table.columns else 0
                raise RuntimeError(
                    f"Aucune configuration sélectionnable pour le rôle {role} "
                    f"(pareto={n_pareto}, rule={rule})."
                )
            selection_tables[role] = table
            selections[role] = selected_id
            selection_rules[role] = rule
            table.to_csv(_discovery_role_dir(output_base, role) / "selection_table.csv", index=False)
            pareto_only = table.loc[table["is_pareto"]].copy()
            pareto_only.to_csv(_discovery_role_dir(output_base, role) / "pareto_candidates.csv", index=False)
            materialize_selected_partition(
                role, prepared.units, selected_id, output_base, theme_tables.get(role)
            )
            print_role_selection_summary(role, table, selected_id=selected_id)
            row = table.loc[table["configuration_id"].astype(str).eq(selected_id)].iloc[0].to_dict()
            selected_rows.append({
                "role": role,
                "configuration_id": selected_id,
                "selection_rule": rule,
                "stability": row.get("stability"),
                "dbcv_umap": row.get("dbcv_umap"),
                "stability_normalized": row.get("stability_normalized"),
                "dbcv_normalized": row.get("dbcv_normalized"),
                "knee_distance": row.get("knee_distance"),
                "n_clusters": row.get("n_clusters"),
                "noise_fraction": row.get("noise_fraction"),
                "coverage": row.get("coverage"),
                **{key: row.get(key) for key in PARAMETER_KEYS},
            })
            _log_progress(
                f"[{role}] sélectionné {selected_id} via {rule} "
                f"(S_R={row.get('stability')}, DBCV={row.get('dbcv_umap')}, "
                f"knee={row.get('knee_distance')})"
            )
        selected_frame = pd.DataFrame(selected_rows)
        selected_frame.to_csv(output_base / "selected_configurations.csv", index=False)
        summarize_selected_configurations(selection_tables, parameter_keys=PARAMETER_KEYS).to_csv(
            output_base / "selected_configurations_summary.csv",
            index=False,
        )
        write_stability_landscape_figure(selection_tables, output_base / "figures")
        write_pareto_normalized_knee_figure(selection_tables, output_base / "figures")
        write_factor_resampling_manuscript_figures(theme_tables, selections, output_base / "figures")
        (output_base / "theme_discovery_manifest.json").write_text(
            json.dumps({
                "version": "pareto_geometric_knee_seed_sensitivity_v1",
                "dataset_id": config["data"].get("dataset_id"),
                "selection_metric": "pareto_geometric_knee",
                "selection_rules": selection_rules,
                "selected_configurations": selections,
                "n_workers": resolve_n_workers(config),
            }, indent=2),
            encoding="utf-8",
        )
    else:
        selections = load_selected_configurations(output_base)
        for role in ROLES:
            selection_tables[role] = candidate_tables[role]

    if stage == "select":
        _log_progress(f"DONE stage=select output={output_base}")
        return output_base

    # seed stage
    if prepared is None:
        prepared = prepare_data(config, output_base)
    if bool(validation_cfg["seed_sensitivity"].get("enabled", True)):
        _log_progress("Seed sensitivity sur les configurations sélectionnées")
        for role in ROLES:
            configuration_id = selections[role]
            table = candidate_tables[role]
            row = table.loc[table["configuration_id"].astype(str).eq(configuration_id)]
            if row.empty:
                raise ValueError(f"Configuration sélectionnée introuvable dans les candidats: {role}/{configuration_id}")
            evaluate_seed_sensitivity(
                role,
                prepared.units,
                prepared.embeddings,
                config,
                output_base,
                configuration_id,
                row.iloc[0].to_dict(),
                reestimate=reestimate,
            )
        write_umap_seed_sensitivity_all_roles_figure(output_base / "figures", run_dir=output_base)
    _log_progress(f"DONE stage={stage} output={output_base}")
    return output_base


__all__ = [
    "ROLES",
    "PreparedData",
    "PartitionResult",
    "StructuralEMResult",
    "load_yaml_config",
    "select_dataset_config",
    "resolve_config_paths",
    "load_bn_analysis_config",
    "prepare_data",
    "parameter_plan",
    "load_topic_stopwords",
    "build_topic_dictionary",
    "build_frozen_bn_inputs",
    "load_selected_configurations",
    "fit_latent_bn_analysis",
    "finalize_latent_bn",
    "extract_latent_bn_scenarios",
    "write_final_bn_outputs",
    "run_frozen_bn_analysis",
    "run_theme_discovery",
    "resolve_n_workers",
    "evaluate_candidates",
    "evaluate_pareto_candidates",
    "evaluate_resampling_stability",
    "select_configuration_by_stability",
    "select_configuration_for_role",
    "materialize_selected_partition",
    "evaluate_seed_sensitivity",
    "write_stability_landscape_figure",
    "write_pareto_normalized_knee_figure",
    "write_factor_stability_figure",
    "write_factor_resampling_manuscript_figures",
    "write_umap_seed_sensitivity_all_roles_figure",
]
