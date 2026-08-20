"""Auditable Python implementation of the recurrent-accident protocol.

The module is deliberately independent from the production training pipelines. It
accepts the existing annotated unit table and frozen embedding export, then writes
all intermediate objects needed to audit the analysis.
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
from typing import Any, Iterable, Mapping, Sequence

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
    if progress_label and config.get("pareto", {}).get("show_progress", True):
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
            resolved[section][key] = str((base / value).resolve())
    return resolved


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
        else:
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


def _same_parameter_value(left: Any, right: Any) -> bool:
    return json.dumps(_to_python_value(left), sort_keys=True, ensure_ascii=False) == json.dumps(_to_python_value(right), sort_keys=True, ensure_ascii=False)


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


def parameter_plan(
    config: Mapping[str, Any],
    *,
    role: str | None = None,
    apply_selection: bool = True,
    budget_override: int | None = None,
) -> list[dict[str, Any]]:
    screening_cfg = config["screening"]
    cfg = screening_cfg
    umap_cfg = cfg["umap"]
    hdbscan_cfg = cfg["hdbscan"]
    keys = [
        ("umap_n_neighbors", _grid_values(umap_cfg["n_neighbors"])),
        ("umap_n_components", _grid_values(umap_cfg["n_components"])),
        ("umap_min_dist", _grid_values(umap_cfg["min_dist"])),
        ("hdbscan_min_cluster_size", _grid_values(hdbscan_cfg["min_cluster_size"])),
        ("hdbscan_min_samples", _grid_values(hdbscan_cfg["min_samples"])),
        ("hdbscan_cluster_selection_method", _grid_values(hdbscan_cfg["cluster_selection_method"])),
    ]
    names = [name for name, _ in keys]
    values = [values for _, values in keys]
    combinations = [dict(zip(names, combination)) for combination in itertools.product(*values)]
    selection = None
    if True:
        if False:
            combinations = [
                combination
                for combination in combinations
                if any(
                    all(
                        key in selected
                        and _same_parameter_value(combination[key], selected[key])
                        for key in PARAMETER_KEYS
                    )
                    for selected in selection
                )
            ]
            if not combinations:
                raise ValueError(
                    f"Aucune configuration sélectionnée ne correspond à la grille pour le rôle {role or 'global'}."
                )
    budget = budget_override
    if selection is not None:
        budget = None
    if budget is not None and int(budget) < len(combinations):
        indices = np.linspace(0, len(combinations) - 1, int(budget), dtype=int)
        combinations = [combinations[int(index)] for index in np.unique(indices)]
    return combinations


def _sample_accidents(units: pd.DataFrame, fraction: float, rng: np.random.Generator) -> set[str]:
    accidents = units["_accident_id"].drop_duplicates().to_numpy()
    size = max(1, int(round(len(accidents) * float(fraction))))
    return set(rng.choice(accidents, size=min(size, len(accidents)), replace=False).tolist())


def _fit_cluster(
    texts: Sequence[str],
    embeddings: np.ndarray,
    params: Mapping[str, Any],
    random_state: int,
    config: Mapping[str, Any],
) -> np.ndarray:
    labels, _ = _fit_cluster_with_embedding(texts, embeddings, params, random_state, config)
    return labels


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


def screen_clustering_parameters(
    role: str,
    units: pd.DataFrame,
    embeddings: np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    reestimate: bool = False,
) -> pd.DataFrame:
    """Run the UMAP-HDBSCAN diagnostics for the shared candidate grid.

    Each candidate configuration is fitted once on a deterministic accident sample.
    This stage reports usability diagnostics only; it does not select a best score.
    """
    role_mask = units["_role"].eq(role).to_numpy()
    role_units = units.loc[role_mask].reset_index(drop=True)
    screening_dir = output_dir / "screening"
    screening_dir.mkdir(parents=True, exist_ok=True)
    cache_path = screening_dir / f"parameter_screening_{role}.csv"
    required_columns = {
        "role",
        "configuration_id",
        *PARAMETER_KEYS,
        "n_clusters",
        "noise_fraction",
        "coverage",
        "median_accident_support",
        "n_single_accident_clusters",
        "random_state",
    }
    if cache_path.is_file() and not reestimate:
        cached = pd.read_csv(cache_path)
        expected_random_state = int(config.get("random_state", 42))
        if required_columns.issubset(cached.columns) and cached["random_state"].astype(int).eq(expected_random_state).all():
            return cached
    role_embeddings = embeddings[role_mask]
    screening_cfg = config.get("screening", {})
    fraction = float(screening_cfg.get("sampling_fraction", 1.0))
    rng = np.random.default_rng(int(config.get("random_state", 42)) + 10000 + ROLE_RANK[role])
    accidents = _sample_accidents(role_units, fraction, rng)
    selected = role_units["_accident_id"].isin(accidents).to_numpy()
    selected_indices = np.flatnonzero(selected)
    plan = parameter_plan(
        config,
        apply_selection=False,
        budget_override=None,
    )
    rows: list[dict[str, Any]] = []
    iterator: Iterable[tuple[int, dict[str, Any]]] = enumerate(plan)
    if screening_cfg.get("show_progress", True):
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(plan), desc=f"Screening {role}", unit="config", leave=True)
        except ImportError:
            pass
    for configuration_index, params in iterator:
        labels = np.full(len(selected_indices), -1, dtype=int)
        if len(selected_indices) >= 3:
            labels = _fit_cluster(
                role_units.iloc[selected_indices]["_text"].tolist(),
                role_embeddings[selected_indices],
                params,
                int(config.get("random_state", 42)),
                config,
            )
        metrics = _screening_metrics(role_units, selected_indices, labels)
        rows.append({
            "role": role,
            "configuration_id": f"{role}_cfg_{configuration_index:03d}",
            "random_state": int(config.get("random_state", 42)),
            **params,
            **metrics,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return frame


def mark_admissible_configurations(
    frame: pd.DataFrame,
    rules: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Add transparent admissibility flags without optimizing a single configuration."""
    rules = dict(rules or {})
    result = frame.copy()
    admissible = pd.Series(True, index=result.index)
    rule_columns = {
        "max_noise_fraction": ("noise_fraction", "le"),
        "min_coverage": ("coverage", "ge"),
        "min_median_accident_support": ("median_accident_support", "ge"),
        "min_clusters": ("n_clusters", "ge"),
        "max_clusters": ("n_clusters", "le"),
        "max_single_accident_clusters": ("n_single_accident_clusters", "le"),
    }
    for rule_name, (column, operator) in rule_columns.items():
        threshold = rules.get(rule_name)
        if threshold is None:
            continue
        if operator == "le":
            admissible &= result[column] <= float(threshold)
        else:
            admissible &= result[column] >= float(threshold)
    result["admissible"] = admissible.astype(bool)
    result["admissibility_reason"] = np.where(result["admissible"], "passes configured rules", "fails configured rules")
    return result


def resolve_admissibility_rules(
    config: Mapping[str, Any],
    role: str,
    n_accidents_role: int,
) -> dict[str, Any]:
    """Resolve role-specific admissibility thresholds from the configuration."""
    admissibility_cfg = config.get("screening", {}).get("admissibility", {})
    role_cfg = admissibility_cfg.get("by_role", {}).get(role, {})
    rules = {
        key: admissibility_cfg.get(key)
        for key in (
            "max_noise_fraction",
            "min_coverage",
            "min_median_accident_support",
            "min_clusters",
            "max_clusters",
            "max_single_accident_clusters",
        )
    }
    rules.update({key: value for key, value in role_cfg.items() if key in rules})
    if rules["min_median_accident_support"] is None:
        floor = role_cfg.get(
            "min_median_accident_support_floor",
            admissibility_cfg.get("min_median_accident_support_floor"),
        )
        fraction = role_cfg.get(
            "min_median_accident_support_fraction_of_role_accidents",
            admissibility_cfg.get("min_median_accident_support_fraction_of_role_accidents"),
        )
        denominator = role_cfg.get(
            "support_denominator",
            admissibility_cfg.get("support_denominator", "max_clusters"),
        )
        if fraction is not None:
            denominator_value = rules.get("max_clusters") if denominator == "max_clusters" else float(denominator)
            denominator_value = max(1.0, float(denominator_value or 1.0))
            computed = math.ceil(float(fraction) * int(n_accidents_role) / denominator_value)
            rules["min_median_accident_support"] = max(int(floor or 0), computed)
        elif floor is not None:
            rules["min_median_accident_support"] = int(floor)
    return rules


def select_admissible_parameter_combinations(
    screening_results: Mapping[str, pd.DataFrame],
    rules: Mapping[str, Any] | None = None,
    rules_by_role: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Convert configured screening rules into role-specific candidate plans."""
    selected: dict[str, list[dict[str, Any]]] = {}
    for role, frame in screening_results.items():
        role_rules = rules_by_role.get(role, {}) if rules_by_role is not None else rules
        marked = mark_admissible_configurations(frame, role_rules)
        combinations = []
        for _, row in marked.loc[marked["admissible"]].iterrows():
            combinations.append({key: _to_python_value(row[key]) for key in PARAMETER_KEYS})
        if not combinations:
            raise ValueError(f"Aucune configuration admissible pour le rôle {role} avec les règles fournies.")
        selected[role] = combinations
    return selected


def write_screening_figures(
    screening_results: Mapping[str, pd.DataFrame],
    output_dir: Path,
    rules: Mapping[str, Any] | None = None,
) -> None:
    """Write the four-panel parameter-region diagnostic and per-role copies."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {role: mark_admissible_configurations(frame, rules) for role, frame in screening_results.items()}
    all_frames = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
    screening_dir = output_dir.parent / "screening"
    screening_dir.mkdir(parents=True, exist_ok=True)
    for role, frame in frames.items():
        frame.to_csv(screening_dir / f"parameter_screening_{role}_marked.csv", index=False)
    if not all_frames.empty:
        all_frames.to_csv(screening_dir / "parameter_screening_all_roles.csv", index=False)
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis_r")
    markers = {5: "o", 10: "s", 15: "^"}
    for axis, role in zip(axes.flat, ROLES):
        frame = frames.get(role, pd.DataFrame())
        if frame.empty:
            axis.set_title(role)
            axis.text(0.5, 0.5, "No screening results", ha="center", va="center", transform=axis.transAxes)
            continue
        support = frame["median_accident_support"].to_numpy(dtype=float)
        size = 45.0 + 180.0 * np.sqrt(support / max(1.0, float(np.nanmax(support))))
        for n_components, marker in markers.items():
            subset = frame[frame["umap_n_components"] == n_components]
            if subset.empty:
                continue
            subset_positions = subset.index.to_numpy()
            axis.scatter(
                subset["umap_n_neighbors"],
                subset["hdbscan_min_cluster_size"],
                c=subset["noise_fraction"],
                s=size[subset_positions],
                marker=marker,
                cmap=cmap,
                norm=norm,
                alpha=0.82,
                edgecolors=np.where(subset["admissible"], "black", "white"),
                linewidths=np.where(subset["admissible"], 2.0, 0.6),
            )
            for _, row in subset.iterrows():
                axis.annotate(
                    f"K={int(row['n_clusters'])}",
                    (row["umap_n_neighbors"], row["hdbscan_min_cluster_size"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_title(role)
        axis.set_xlabel("UMAP n_neighbors")
        axis.set_ylabel("HDBSCAN min_cluster_size")
        axis.grid(alpha=0.2)
    if axes.size:
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        figure.colorbar(scalar, ax=axes, shrink=0.8, label="Noise fraction")
        legend = [
            Line2D([0], [0], marker=marker, color="w", label=f"UMAP dimension = {dimension}", markerfacecolor="grey", markersize=8)
            for dimension, marker in markers.items()
        ]
        legend.append(Line2D([0], [0], marker="o", color="black", label="Admissible configuration", markerfacecolor="white", markersize=8, linewidth=0))
        figure.legend(handles=legend, loc="upper center", ncol=4, frameon=False)
    figure.suptitle("UMAP-HDBSCAN parameter screening by role")
    figure.savefig(output_dir / "parameter_screening_region_all_roles.png", dpi=250, bbox_inches="tight")
    plt.close(figure)


def consensus_role(
    role: str,
    units: pd.DataFrame,
    embeddings: np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    reestimate: bool = False,
) -> PartitionResult:
    role_mask = units["_role"].eq(role).to_numpy()
    role_units = units.loc[role_mask].reset_index(drop=True)
    role_embeddings = embeddings[role_mask]
    clustering_embeddings = role_embeddings
    n_units = len(role_units)
    if n_units > int(config["consensus"].get("max_dense_units", 6000)) and int(config["consensus"].get("consensus_knn", 50)) <= 0:
        raise ValueError("Legacy consensus execution is no longer supported.")
    role_dir = output_dir / "clustering" / role
    role_dir.mkdir(parents=True, exist_ok=True)
    cache_files = [
        role_dir / "topic_assignments.csv",
        role_dir / "topics.csv",
        role_dir / "coassociation_edges.csv",
        role_dir / "replications.csv",
        role_dir / "cache_metadata.json",
    ]
    if not reestimate and all(path.is_file() for path in cache_files):
        metadata = json.loads((role_dir / "cache_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("version") == "direct_umap_hdbscan_consensus_v1":
            return PartitionResult(
                role=role,
                assignments=pd.read_csv(role_dir / "topic_assignments.csv"),
                topics=pd.read_csv(role_dir / "topics.csv"),
                edges=pd.read_csv(role_dir / "coassociation_edges.csv"),
                replications=pd.read_csv(role_dir / "replications.csv"),
            )
    plan = parameter_plan(config, role=role)
    consensus_cfg = config["consensus"]
    repetitions = int(consensus_cfg.get("n_repetitions", 100))
    rng = np.random.default_rng(int(consensus_cfg.get("random_state", 42)) + ROLE_RANK[role])
    pairs = _neighbor_pairs(clustering_embeddings, int(consensus_cfg.get("consensus_knn", 50)))
    co_present = np.zeros(len(pairs), dtype=np.int32)
    co_cluster = np.zeros(len(pairs), dtype=np.int32)
    rep_rows: list[dict[str, Any]] = []
    labels_store = np.full((repetitions, n_units), -1, dtype=np.int16)
    repetition_iterator: Iterable[int] = range(repetitions)
    if consensus_cfg.get("show_progress", True):
        try:
            from tqdm.auto import tqdm

            repetition_iterator = tqdm(
                repetition_iterator,
                desc=f"Consensus {role}",
                unit="rep",
                leave=True,
            )
        except ImportError:
            pass
    for repetition in repetition_iterator:
        params = plan[repetition % len(plan)]
        accident_sample = _sample_accidents(role_units, float(consensus_cfg.get("sampling_fraction", 0.8)), rng)
        selected = role_units["_accident_id"].isin(accident_sample).to_numpy()
        selected_indices = np.flatnonzero(selected)
        labels = np.full(n_units, -1, dtype=int)
        if len(selected_indices) >= 3:
            labels[selected_indices] = _fit_cluster(
                role_units.loc[selected_indices, "_text"].tolist(),
                clustering_embeddings[selected_indices],
                params,
                int(consensus_cfg.get("random_state", 42)) + repetition,
                config,
            )
        labels_store[repetition] = labels.astype(np.int16)
        valid_selected = labels[selected_indices] >= 0
        run_coverage = float(valid_selected.mean()) if len(valid_selected) else 0.0
        if len(pairs):
            both_present = selected[pairs[:, 0]] & selected[pairs[:, 1]]
            if consensus_cfg.get("ignore_noise_pairs", True):
                both_present &= (labels[pairs[:, 0]] >= 0) & (labels[pairs[:, 1]] >= 0)
            same_cluster = both_present & (labels[pairs[:, 0]] == labels[pairs[:, 1]])
            run_stability = float(same_cluster.sum() / max(1, both_present.sum()))
            co_present += both_present.astype(np.int32)
            co_cluster += same_cluster.astype(np.int32)
        else:
            run_stability = 0.0
        valid_labels = labels[labels >= 0]
        rep_rows.append({
            "role": role,
            "repetition": repetition,
            "n_accidents_sampled": len(accident_sample),
            "n_units_sampled": int(selected.sum()),
            "n_clusters": int(len(set(valid_labels.tolist()))) if len(valid_labels) else 0,
            "noise_fraction_sampled": float(np.mean(labels[selected_indices] < 0)) if len(selected_indices) else 1.0,
            "coverage": run_coverage,
            "stability": run_stability,
            **params,
        })
    if config.get("runtime", {}).get("save_intermediate_assignments", True):
        np.save(role_dir / "replication_labels.npy", labels_store)
    replications = pd.DataFrame(rep_rows)
    replications.to_csv(role_dir / "replications.csv", index=False)
    similarity = co_cluster / np.maximum(co_present, 1)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as error:
        raise ImportError("scipy est requis pour la partition de consensus.") from error
    threshold = float(consensus_cfg.get("consensus_edge_threshold", 0.6))
    keep = (co_present > 0) & (similarity >= threshold)
    graph = coo_matrix((np.ones(int(keep.sum())), (pairs[keep, 0], pairs[keep, 1])), shape=(n_units, n_units)).tocsr()
    graph = graph + graph.T
    _, labels = connected_components(graph, directed=False, return_labels=True)
    topic_sizes = pd.Series(labels).value_counts()
    order = topic_sizes.index.tolist()
    remap = {int(old): index for index, old in enumerate(order)}
    topic_numbers = np.array([remap[int(label)] for label in labels], dtype=int)
    edge_frame = pd.DataFrame({
        "role": role,
        "unit_i": pairs[:, 0] if len(pairs) else [],
        "unit_j": pairs[:, 1] if len(pairs) else [],
        "co_present": co_present,
        "co_cluster": co_cluster,
        "similarity": similarity,
        "selected_at_threshold": keep,
    })
    edge_frame.to_csv(role_dir / "coassociation_edges.csv", index=False)
    assignment_rows = role_units[["_accident_id", "_fact_id", "_text"]].copy()
    assignment_rows["role"] = role
    assignment_rows["topic_id"] = [f"{role}_{number:03d}" for number in topic_numbers]
    assignment_rows.rename(columns={"_accident_id": "accident_id", "_fact_id": "fact_id", "_text": "sentence"}, inplace=True)
    assignment_rows.to_csv(role_dir / "topic_assignments.csv", index=False)
    topic_rows = []
    for number in sorted(set(topic_numbers.tolist())):
        mask = topic_numbers == number
        topic_id = f"{role}_{number:03d}"
        topic_stability = _topic_stability(number, topic_numbers, pairs, similarity)
        topic_rows.append({
            "topic_id": topic_id,
            "role": role,
            "n_units": int(mask.sum()),
            "n_accidents": int(role_units.loc[mask, "_accident_id"].nunique()),
            "stability": topic_stability,
            "consensus_threshold": threshold,
        })
    topics = pd.DataFrame(topic_rows)
    topics.to_csv(role_dir / "topics.csv", index=False)
    (role_dir / "cache_metadata.json").write_text(
        json.dumps({"version": "direct_umap_hdbscan_consensus_v1", "role": role}, indent=2),
        encoding="utf-8",
    )
    return PartitionResult(role=role, assignments=assignment_rows, topics=topics, edges=edge_frame, replications=replications)


def _topic_stability(topic_number: int, labels: np.ndarray, pairs: np.ndarray, similarity: np.ndarray) -> float:
    if len(pairs) == 0:
        return 0.0
    same = (labels[pairs[:, 0]] == topic_number) & (labels[pairs[:, 1]] == topic_number)
    return float(np.mean(similarity[same])) if same.any() else 0.0


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


def evaluate_pareto_candidates(
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
    pareto_cfg = config.get("pareto", {})
    pareto_dir = output_dir / "pareto" / role
    candidate_dir = pareto_dir / "candidate_partitions"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = pareto_dir / "candidate_metrics.csv"
    expected_random_state = int(pareto_cfg.get("random_state", config.get("random_state", 42)))
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
    clustering_input = role_embeddings
    plan = parameter_plan(
        config,
        role=None,
        apply_selection=False,
        budget_override=None,
    )
    random_state = expected_random_state
    dbcv_sample_size = pareto_cfg.get("dbcv_sample_size")
    tasks = [
        {
            "index": index,
            "role": role,
            "configuration_id": _configuration_id(role, index),
            "params": params,
            "embeddings": clustering_input,
            "accident_ids": role_units["_accident_id"].astype(str).to_numpy(),
            "random_state": random_state,
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
    (pareto_dir / "candidate_metadata.json").write_text(
        json.dumps({
            "version": "pareto_candidates_parallel_v3_membership_strength",
            "role": role,
            "n_candidates": len(result),
            "n_workers": resolve_n_workers(config),
        }, indent=2),
        encoding="utf-8",
    )
    return result


def _best_jaccard(reference: np.ndarray, candidate: np.ndarray, reference_label: int) -> float:
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
    pareto_cfg = config.get("pareto", {})
    pareto_dir = output_dir / "pareto" / role
    theme_path = pareto_dir / "stability_theme.csv"
    summary_path = pareto_dir / "stability_summary.csv"
    n_repetitions = int(pareto_cfg.get("n_resampling", 30))
    fraction = float(pareto_cfg.get("resampling_fraction", 0.8))
    random_state = int(pareto_cfg.get("random_state", config.get("random_state", 42)))
    stability_metadata_path = pareto_dir / "stability_metadata.json"
    expected_stability_metadata = {
        "version": "fixed_main_seed_v1",
        "random_state": random_state,
        "n_repetitions": n_repetitions,
        "resampling_fraction": fraction,
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
    candidate_dir = pareto_dir / "candidate_partitions"
    clustering_input = role_embeddings
    all_tasks: list[dict[str, Any]] = []
    candidates = candidates.reset_index(drop=True)
    accident_ids = role_units["_accident_id"].astype(str).to_numpy()
    for candidate_index, candidate in candidates.iterrows():
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
                "embeddings": clustering_input,
                "random_state": random_state + repetition,
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
    if theme_frame.empty:
        summary = pd.DataFrame(columns=["role", "configuration_id", "stability", "n_themes", "n_repetitions"])
    else:
        theme_summary = (
            theme_frame.groupby(["role", "configuration_id", "cluster_label"], as_index=False)["best_jaccard"]
            .median()
            .rename(columns={"best_jaccard": "theme_stability"})
        )
        theme_frame = theme_frame.merge(theme_summary, on=["role", "configuration_id", "cluster_label"], how="left")
        summary = (
            theme_summary.groupby(["role", "configuration_id"], as_index=False)["theme_stability"]
            .median()
            .rename(columns={"theme_stability": "stability"})
        )
        summary["n_themes"] = theme_summary.groupby(["role", "configuration_id"])["cluster_label"].nunique().to_numpy()
        summary["n_repetitions"] = n_repetitions
    theme_frame.to_csv(theme_path, index=False)
    summary.to_csv(summary_path, index=False)
    stability_metadata_path.write_text(json.dumps(expected_stability_metadata, indent=2), encoding="utf-8")
    _log_progress(
        f"[{role}] resampling: terminé ({len(summary)} résumés, "
        f"{len(all_tasks)} tâches)"
    )
    return theme_frame, summary


def _pareto_mask(values: pd.DataFrame, objectives: Sequence[str]) -> pd.Series:
    mask = pd.Series(True, index=values.index)
    usable = values[list(objectives)].notna().all(axis=1)
    indices = values.index[usable].tolist()
    for left in indices:
        for right in indices:
            if left == right:
                continue
            left_values = values.loc[left, list(objectives)].to_numpy(dtype=float)
            right_values = values.loc[right, list(objectives)].to_numpy(dtype=float)
            if np.all(right_values >= left_values) and np.any(right_values > left_values):
                mask.loc[left] = False
                break
    mask.loc[~usable] = False
    return mask


def _ordered_pareto_export(table: pd.DataFrame) -> pd.DataFrame:
    """Order the complete per-role Pareto export without dropping diagnostics."""
    preferred = [
        "role", "configuration_id", "pareto_non_dominated", "candidate_only",
        "dbcv_umap", "stability",
        "n_clusters", "noise_fraction", "coverage", "median_accident_support",
        "mean_accident_support", "min_accident_support", "max_accident_support",
        "n_single_accident_clusters", "median_cluster_size",
        "min_cluster_size_observed", "max_cluster_size_observed",
        "n_units_sampled", "n_units_noise", "n_accidents_sampled",
        "n_themes", "n_repetitions", "dbcv_n_units",
        *PARAMETER_KEYS, "random_state",
    ]
    columns = [column for column in preferred if column in table.columns]
    columns.extend(column for column in table.columns if column not in columns)
    return table.loc[:, columns]


def select_pareto_partitions(
    role: str,
    candidates: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Identify the D_U/S_R Pareto frontier without selecting a partition."""
    _log_progress(f"[{role}] Pareto: démarrage sur {len(candidates)} configurations")
    pareto_dir = output_dir / "pareto" / role
    pareto_dir.mkdir(parents=True, exist_ok=True)
    result = candidates.copy()
    pool = result
    result["pareto_pool"] = result["configuration_id"].isin(pool["configuration_id"])
    result["pareto_non_dominated"] = False
    if not pool.empty:
        result.loc[pool.index, "pareto_non_dominated"] = _pareto_mask(pool, ["dbcv_umap", "stability"])
    pareto = result[result["pareto_non_dominated"]].copy()
    agreement = pd.DataFrame()
    result["candidate_only"] = True
    result.to_csv(pareto_dir / "pareto_selection_table.csv", index=False)
    _ordered_pareto_export(pareto.assign(candidate_only=True)).to_csv(
        pareto_dir / "pareto_frontier.csv", index=False
    )
    _ordered_pareto_export(pareto.assign(candidate_only=True)).to_csv(
        pareto_dir / "pareto_optimal_configurations.csv", index=False
    )
    selected_id = ""
    selected_metrics = ""
    _log_progress(
        f"[{role}] Pareto: terminé — {len(pareto)} non-dominées, "
        f"sélection={selected_id or 'aucune'}{selected_metrics}"
    )
    return result, agreement, ""


def _symmetric_partition_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    def directional(source: np.ndarray, target: np.ndarray) -> float:
        labels = [int(label) for label in np.unique(source) if label >= 0]
        if not labels:
            return 0.0
        scores = []
        for label in labels:
            source_set = set(np.flatnonzero(source == label).tolist())
            best = 0.0
            for target_label in np.unique(target[target >= 0]):
                target_set = set(np.flatnonzero(target == target_label).tolist())
                union = source_set | target_set
                if union:
                    best = max(best, len(source_set & target_set) / len(union))
            scores.append(best)
        return float(np.mean(scores))
    return float((directional(left, right) + directional(right, left)) / 2.0)


def build_selected_partition_results(
    prepared: PreparedData,
    config: Mapping[str, Any],
    output_dir: Path,
    selections: Mapping[str, str],
    stability_themes: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, PartitionResult]:
    """Materialize the manually selected Pareto partition for downstream analysis."""
    results: dict[str, PartitionResult] = {}
    for role in ROLES:
        role_mask = prepared.units["_role"].eq(role).to_numpy()
        role_units = prepared.units.loc[role_mask].reset_index(drop=True)
        pareto_dir = output_dir / "pareto" / role
        labels = np.load(pareto_dir / "selected_labels.npy").astype(int)
        selected_id = selections[role]
        theme_stability = {}
        if stability_themes and role in stability_themes and not stability_themes[role].empty:
            grouped = stability_themes[role].groupby("cluster_label")["theme_stability"].first()
            theme_stability = {int(key): float(value) for key, value in grouped.items()}
        topic_ids = np.where(labels >= 0, [f"{role}_{int(label):03d}" for label in labels], "")
        assignments = role_units[["_accident_id", "_fact_id", "_text"]].copy()
        assignments["role"] = role
        assignments["topic_id"] = topic_ids
        assignments.rename(columns={"_accident_id": "accident_id", "_fact_id": "fact_id", "_text": "sentence"}, inplace=True)
        topic_rows = []
        for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
            mask = labels == label
            topic_rows.append({
                "topic_id": f"{role}_{label:03d}",
                "role": role,
                "n_units": int(mask.sum()),
                "n_accidents": int(role_units.loc[mask, "_accident_id"].nunique()),
                "stability": theme_stability.get(label, np.nan),
                "selected_configuration_id": selected_id,
            })
        topics = pd.DataFrame(topic_rows)
        role_dir = output_dir / "clustering" / role
        role_dir.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(role_dir / "topic_assignments.csv", index=False)
        topics.to_csv(role_dir / "topics.csv", index=False)
        (role_dir / "cache_metadata.json").write_text(json.dumps({"version": "pareto_medoid_partition_v1", "role": role, "configuration_id": selected_id}, indent=2), encoding="utf-8")
        results[role] = PartitionResult(role=role, assignments=assignments, topics=topics, edges=pd.DataFrame(), replications=pd.DataFrame())
    return results


def write_pareto_figure(
    selection_tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    show_all_candidates: bool = True,
    roles: Sequence[str] | None = None,
    filename: str = "pareto_validation_all_roles.png",
) -> None:
    """Save the two-objective D_U versus S_R Pareto scatter plot."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_roles = tuple(roles or ROLES)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_columns = 2 if len(plot_roles) > 1 else 1
    n_rows = math.ceil(len(plot_roles) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 10) if len(plot_roles) > 1 else (8, 6),
        squeeze=False,
    )
    method_markers = {"leaf": "o"}
    pareto_color = "#f28e2b"
    dominated_color = "#9e9e9e"
    for axis, role in zip(axes.flat, plot_roles):
        frame = selection_tables.get(role, pd.DataFrame()).copy()
        if frame.empty:
            axis.set_title(role)
            axis.text(0.5, 0.5, "Aucune configuration", ha="center", va="center")
            continue
        frame["hdbscan_cluster_selection_method"] = frame["hdbscan_cluster_selection_method"].astype(str)
        base = frame if show_all_candidates else frame[frame["pareto_pool"]]
        valid = base["dbcv_umap"].notna() & base["stability"].notna()
        base = base[valid].copy()
        pareto_mask = base["pareto_non_dominated"].fillna(False).astype(bool)
        dominated = base[~pareto_mask]
        frontier = base[pareto_mask].sort_values("dbcv_umap")
        for method, marker in method_markers.items():
            subset = dominated[dominated["hdbscan_cluster_selection_method"].eq(method)]
            if not subset.empty:
                axis.scatter(
                    subset["dbcv_umap"],
                    subset["stability"],
                    marker=marker,
                    facecolors=dominated_color,
                    edgecolors="black",
                    alpha=0.65,
                    s=48,
                    linewidths=0.9,
                    zorder=1,
                )
            subset = frontier[frontier["hdbscan_cluster_selection_method"].eq(method)]
            if not subset.empty:
                axis.scatter(
                    subset["dbcv_umap"],
                    subset["stability"],
                    marker=marker,
                    facecolors=pareto_color,
                    edgecolors="black",
                    alpha=0.95,
                    s=68,
                    linewidths=0.8,
                    zorder=4,
                )
        axis.set_title(role)
        axis.set_xlabel("UMAP-space DBCV $D_U$")
        axis.set_ylabel("Accident-level resampling stability $S_R$")
        axis.grid(alpha=0.25)
    for axis in list(axes.flat)[len(plot_roles):]:
        axis.remove()
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", label="Dominated configurations", markerfacecolor=dominated_color, markeredgecolor="black", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", color="black", label="Pareto frontier", markerfacecolor=pareto_color, markeredgecolor="black", markersize=7),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Pareto validation of UMAP--HDBSCAN partitions", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_dir / filename, dpi=250, bbox_inches="tight")
    plt.close(figure)


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


def build_accident_topic_matrix(
    prepared: PreparedData,
    partition_results: Mapping[str, PartitionResult],
    topic_dictionary: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    topics_cfg = config.get("topics", {})
    min_support = int(topics_cfg.get("min_accident_support", 20))
    max_topics = topics_cfg.get("max_topics_per_role")
    selected_parts: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    for role in ROLES:
        role_topics = topic_dictionary[topic_dictionary["role"] == role].copy()
        role_topics = role_topics[role_topics["n_accidents"] >= min_support].sort_values("n_accidents", ascending=False)
        if max_topics is not None:
            role_topics = role_topics.head(int(max_topics))
        selected_parts.append(role_topics)
        selected_rows.extend(role_topics.to_dict("records"))
    selected = pd.DataFrame(selected_rows)
    accident_ids = prepared.units["_accident_id"].drop_duplicates().sort_values().tolist()
    matrix = pd.DataFrame({"accident_id": accident_ids})
    for topic_id in selected.get("topic_id", pd.Series(dtype=str)).astype(str):
        role = topic_id.split("_", 1)[0]
        assignments = partition_results[role].assignments
        present = assignments.loc[assignments["topic_id"] == topic_id, "accident_id"].drop_duplicates()
        matrix[topic_id] = matrix["accident_id"].isin(set(present)).astype(int)
    topic_columns = [column for column in selected.get("topic_id", pd.Series(dtype=str)).astype(str) if column in matrix.columns]
    variable_columns = [column for column in topic_columns if matrix[column].nunique(dropna=False) > 1]
    selected = selected[selected["topic_id"].astype(str).isin(variable_columns)].copy() if not selected.empty else selected
    matrix = matrix.drop(columns=[column for column in topic_columns if column not in variable_columns])
    for role in ROLES:
        role_cols = [column for column in variable_columns if column.startswith(f"{role}_")]
        matrix[f"n_topics_{role}"] = matrix[role_cols].sum(axis=1) if role_cols else 0
    matrix["n_topics_observed"] = matrix[[f"n_topics_{role}" for role in ROLES]].sum(axis=1)
    matrix["incomplete_report"] = (matrix[[f"n_topics_{role}" for role in ROLES]] == 0).any(axis=1).astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / "accident_topic_matrix.csv", index=False)
    selected.to_csv(output_dir / "selected_topics.csv", index=False)
    variable_map = {str(row["topic_id"]): str(row["role"]) for _, row in selected.iterrows()}
    (output_dir / "variable_macro_map.json").write_text(json.dumps(variable_map, indent=2), encoding="utf-8")
    return matrix, selected, variable_map


def descriptive_tables(
    matrix: pd.DataFrame,
    selected_topics: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    topic_columns = [column for column in selected_topics.get("topic_id", pd.Series(dtype=str)).astype(str) if column in matrix.columns]
    rng = np.random.default_rng(int(config.get("random_state", 42)))
    n_boot = int(config.get("descriptive", {}).get("bootstrap_repetitions", 1000))
    n_accidents = len(matrix)
    frequency_rows = []
    values = matrix[topic_columns].to_numpy(dtype=float) if topic_columns else np.empty((n_accidents, 0))
    for index, topic_id in enumerate(topic_columns):
        observed = values[:, index]
        estimates = np.empty(n_boot, dtype=float)
        for bootstrap_index in range(n_boot):
            sample = rng.integers(0, n_accidents, size=n_accidents)
            estimates[bootstrap_index] = observed[sample].mean()
        frequency_rows.append({
            "topic_id": topic_id,
            "n_accidents": int(observed.sum()),
            "frequency": float(observed.mean()),
            "bootstrap_low": float(np.quantile(estimates, 0.025)),
            "bootstrap_high": float(np.quantile(estimates, 0.975)),
        })
    frequencies = pd.DataFrame(frequency_rows).sort_values("frequency", ascending=False)
    frequencies.to_csv(output_dir / "topic_frequencies.csv", index=False)
    pair_rows = []
    for left_index, left in enumerate(topic_columns):
        for right in topic_columns[left_index + 1 :]:
            support_left = float(matrix[left].mean())
            support_right = float(matrix[right].mean())
            joint = float((matrix[left] * matrix[right]).mean())
            pair_rows.append({
                "topic_left": left,
                "topic_right": right,
                "role_left": left.split("_", 1)[0],
                "role_right": right.split("_", 1)[0],
                "support_left": support_left,
                "support_right": support_right,
                "joint_frequency": joint,
                "lift": joint / max(support_left * support_right, 1e-12),
                "n_joint_accidents": int((matrix[left] * matrix[right]).sum()),
            })
    pairs = pd.DataFrame(pair_rows).sort_values("lift", ascending=False) if pair_rows else pd.DataFrame()
    pairs.to_csv(output_dir / "topic_cooccurrence_lift.csv", index=False)
    return {"frequencies": frequencies, "cooccurrence_lift": pairs}


def _role_color(role: str) -> str:
    return {"A0": "#A6CEE3", "A1": "#FDBF6F", "B": "#B2DF8A", "C": "#FB9A99", "Z": "#CAB2D6"}.get(role, "#DDDDDD")


def write_descriptive_figures(tables: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    frequencies = tables["frequencies"].head(30)
    if not frequencies.empty:
        fig, axis = plt.subplots(figsize=(10, max(5, len(frequencies) * 0.24)))
        plot = frequencies.sort_values("frequency")
        axis.barh(plot["topic_id"], plot["frequency"] * 100, color="#4C78A8")
        axis.set_xlabel("Accidents containing topic (%)")
        axis.set_ylabel("Topic")
        axis.set_title("Consensus topic frequency")
        fig.tight_layout()
        fig.savefig(output_dir / "topic_frequencies.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    pairs = tables["cooccurrence_lift"]
    if not pairs.empty:
        top = pairs.head(30)
        pivot = top.pivot_table(index="topic_left", columns="topic_right", values="lift", aggfunc="max")
        fig, axis = plt.subplots(figsize=(10, 7))
        sns.heatmap(pivot, ax=axis, cmap="viridis", annot=False)
        axis.set_title("Cross-topic lift")
        fig.tight_layout()
        fig.savefig(output_dir / "topic_lift_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_consensus_figures(
    partition_results: Mapping[str, PartitionResult],
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Write the protocol's stability/coverage/granularity and co-association figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    tradeoff_frames = []
    for role, result in partition_results.items():
        repetitions = result.replications.copy()
        repetitions["configuration"] = (
            repetitions["umap_n_neighbors"].astype(str)
            + "/"
            + repetitions["hdbscan_min_cluster_size"].astype(str)
        )
        grouped = repetitions.groupby("configuration", sort=False).agg(
            stability=("stability", "mean"),
            coverage=("coverage", "mean"),
            n_topics=("n_clusters", "mean"),
        ).reset_index()
        grouped["role"] = role
        tradeoff_frames.append(grouped)
        _plot_tradeoff(grouped, role, output_dir / f"consensus_tradeoff_{role}.png")
    if tradeoff_frames:
        combined = pd.concat(tradeoff_frames, ignore_index=True)
        combined.to_csv(output_dir / "consensus_tradeoff_summary.csv", index=False)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
        for axis, role in zip(axes.flat, ROLES):
            role_frame = combined[combined["role"] == role]
            _plot_tradeoff(role_frame, role, None, axis=axis)
        fig.suptitle("Granularity–coverage–stability trade-off by role")
        fig.tight_layout()
        fig.savefig(output_dir / "consensus_tradeoff_all_roles.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
    max_units = int(config.get("figures", {}).get("coassociation_plot_max_units", 250))
    roles = config.get("figures", {}).get("coassociation_roles", list(ROLES))
    for role in roles:
        if role in partition_results:
            _plot_coassociation(partition_results[role], max_units, output_dir / f"coassociation_matrix_{role}.png")


def _plot_tradeoff(frame: pd.DataFrame, role: str, output_path: Path | None, axis: Any = None) -> None:
    import matplotlib.pyplot as plt

    if frame.empty:
        return
    owns_figure = axis is None
    if owns_figure:
        _, axis = plt.subplots(figsize=(11, 6))
    x = np.arange(len(frame))
    axis.plot(x, frame["stability"], marker="o", label="Stability")
    axis.plot(x, 1.0 - frame["coverage"], marker="s", label="Not assigned")
    right_axis = axis.twinx()
    right_axis.plot(x, frame["n_topics"], marker="^", linestyle="--", label="Number of topics")
    axis.set_title(f"Granularity–coverage–stability trade-off — {role}")
    axis.set_xlabel("UMAP n_neighbors / HDBSCAN min_cluster_size")
    axis.set_ylabel("Proportion")
    right_axis.set_ylabel("Number of topics")
    axis.set_xticks(x, frame["configuration"], rotation=45, ha="right")
    axis.set_ylim(0, 1)
    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    axis.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    axis.grid(axis="y", alpha=0.25)
    if owns_figure:
        axis.figure.tight_layout()
        axis.figure.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(axis.figure)


def _plot_coassociation(result: PartitionResult, max_units: int, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    assignments = result.assignments.sort_values("topic_id")
    selected_indices = assignments.index.to_numpy()[: min(max_units, len(assignments))]
    index_lookup = {int(original): position for position, original in enumerate(selected_indices)}
    matrix = np.zeros((len(selected_indices), len(selected_indices)), dtype=float)
    np.fill_diagonal(matrix, 1.0)
    edges = result.edges
    for _, edge in edges.iterrows():
        left = index_lookup.get(int(edge["unit_i"]))
        right = index_lookup.get(int(edge["unit_j"]))
        if left is not None and right is not None:
            matrix[left, right] = float(edge["similarity"])
            matrix[right, left] = float(edge["similarity"])
    fig, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="Greys",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Co-association frequency"},
    )
    axis.set_title(f"Co-association matrix — {result.role} (illustration)")
    axis.set_xlabel("Units ordered by consensus topic")
    axis.set_ylabel("Units ordered by consensus topic")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Frozen-theme latent BN analysis
# ---------------------------------------------------------------------------

BN_ROLE_ARCS = (("A0", "A1"), ("A0", "B"), ("A1", "B"), ("B", "C"))


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
    values.setdefault("min_theme_support_count", 20)
    values.setdefault("d_max", 2)
    values.setdefault("latent_states", list(range(2, 9)))
    values.setdefault("n_initializations", 20)
    values.setdefault("em_max_iter", 500)
    values.setdefault("em_tol", 1e-6)
    values.setdefault("structure_max_iter", 100)
    values.setdefault("structure_epsilon", 1e-6)
    values.setdefault("alpha", 0.5)
    values.setdefault("probability_floor", 1e-12)
    values.setdefault("exact_inference", True)
    values.setdefault("top_m_mpe", 3)
    values.setdefault("min_latent_effective_n", 25)
    return values


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
    """Load one explicitly selected Pareto partition per role.

    The labels are read from Notebook 1 candidate artifacts. No clustering or
    resampling is performed here. The returned matrix contains one row per
    accident and one binary variable per sufficiently supported theme.
    """

    units = _validate_bn_units(units)
    selections = {role: str(partition_selections.get(role, "")).strip() for role in ROLES}
    missing = [role for role, value in selections.items() if not value]
    if missing:
        raise ValueError("Une configuration BN doit être choisie pour chaque rôle: " + ", ".join(missing))

    dictionary_paths = [
        run_dir / "topics_manual" / "topic_dictionary_with_llm_labels.csv",
        run_dir / "topics_manual" / "topic_dictionary_all_selected.csv",
    ]
    dictionary = next((pd.read_csv(path) for path in dictionary_paths if path.is_file()), pd.DataFrame())
    min_support = int(_bn_config(config)["min_theme_support_count"])
    all_rows: list[dict[str, Any]] = []
    matrix = pd.DataFrame({"accident_id": sorted(units["_accident_id"].unique())})

    for role in ROLES:
        configuration_id = selections[role]
        labels_path = run_dir / "pareto" / role / "candidate_partitions" / f"{configuration_id}_labels.npy"
        strength_path = run_dir / "pareto" / role / "candidate_partitions" / f"{configuration_id}_membership_strength.npy"
        if not labels_path.is_file() or not strength_path.is_file():
            raise FileNotFoundError(f"Artefacts Pareto absents pour {role}/{configuration_id}")
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
            topic_rows = dictionary[
                dictionary.get("role", pd.Series(dtype=str)).astype(str).eq(role)
                & dictionary.get("configuration_id", pd.Series(dtype=str)).astype(str).eq(configuration_id)
                & dictionary.get("topic_id", pd.Series(dtype=str)).astype(str).eq(topic_id)
            ]
            metadata = topic_rows.iloc[0].to_dict() if not topic_rows.empty else {}
            variable_name = f"{role}__T{label + 1:02d}"
            all_rows.append({
                "variable_name": variable_name,
                "topic_id": topic_id,
                "role": role,
                "topic_label": metadata.get("llm_label", metadata.get("label", topic_id)),
                "configuration_id": configuration_id,
                "n_units": int(mask.sum()),
                "n_accidents": support_accidents,
                "mean_membership_strength": float(np.mean(strengths[mask])) if mask.any() else 0.0,
                "included_in_bn": support_accidents >= min_support,
            })
            if support_accidents >= min_support:
                present = set(role_units.loc[mask, "_accident_id"].astype(str))
                matrix[variable_name] = matrix["accident_id"].isin(present).astype(np.int8)

    dictionary_out = pd.DataFrame(all_rows)
    if dictionary_out.empty:
        raise ValueError("Aucun thème non bruit n'a été trouvé dans les partitions sélectionnées.")
    included = dictionary_out[dictionary_out["included_in_bn"]].copy()
    excluded = dictionary_out[~dictionary_out["included_in_bn"]].copy()
    included_names = included["variable_name"].tolist()
    matrix = matrix[["accident_id", *included_names]]
    roles = dict(zip(included["variable_name"], included["role"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_out.to_csv(output_dir / "theme_dictionary.csv", index=False)
    dictionary_out.to_csv(output_dir / "theme_support.csv", index=False)
    matrix.to_parquet(output_dir / "accident_factor_matrix.parquet", index=False)
    excluded.to_csv(output_dir / "excluded_themes.csv", index=False)
    (output_dir / "variable_roles.json").write_text(json.dumps(roles, indent=2), encoding="utf-8")
    return matrix, included, excluded, roles


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


def _bn_parameter_count(nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], n_states: int) -> int:
    parents = _bn_parent_map(nodes, edges)
    return int((n_states - 1) + sum(n_states * (2 ** len(parents[node])) for node in nodes if roles[node] in {"A0", "A1"}) + sum(2 ** len(parents[node]) for node in nodes if roles[node] in {"B", "C"}))


def _bn_mle_parameters(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], tau: np.ndarray, n_states: int, floor: float = 1e-12) -> tuple[np.ndarray, dict[tuple[str, int, tuple[int, ...]], float], dict[tuple[str, tuple[int, ...]], float]]:
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
            if roles[node] in {"A0", "A1"}:
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


def _bn_log_joint(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], weights: np.ndarray, upstream: Mapping[tuple[str, int, tuple[int, ...]], float], downstream: Mapping[tuple[str, tuple[int, ...]], float], floor: float = 1e-12) -> np.ndarray:
    index = {node: position for position, node in enumerate(nodes)}
    parents = _bn_parent_map(nodes, edges)
    output = np.zeros((len(data), len(weights)), dtype=float)
    for row_index, row in enumerate(data):
        for state in range(len(weights)):
            value = math.log(max(float(weights[state]), floor))
            for node in nodes:
                parent_values = tuple(int(row[index[parent]]) for parent in parents[node])
                probability = upstream.get((node, state, parent_values)) if roles[node] in {"A0", "A1"} else downstream.get((node, parent_values))
                probability = min(max(float(probability if probability is not None else 0.5), floor), 1.0 - floor)
                value += math.log(probability if int(row[index[node]]) else 1.0 - probability)
            output[row_index, state] = value
    return output


def _bn_expected_bic(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], tau: np.ndarray, n_states: int, floor: float) -> float:
    weights, upstream, downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor)
    log_joint = _bn_log_joint(data, nodes, roles, edges, weights, upstream, downstream, floor)
    q_value = float(np.sum(tau * log_joint))
    return -2.0 * q_value + _bn_parameter_count(nodes, roles, edges, n_states) * math.log(max(len(data), 1))


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


def _structure_step(data: np.ndarray, nodes: Sequence[str], roles: Mapping[str, str], edges: Sequence[tuple[str, str]], tau: np.ndarray, n_states: int, d_max: int, epsilon: float, max_iter: int, floor: float) -> list[tuple[str, str]]:
    current = set(edges)
    allowed = set(_allowed_bn_edges(roles))
    for _ in range(max_iter):
        base_score = _bn_expected_bic(data, nodes, roles, sorted(current), tau, n_states, floor)
        parents = _bn_parent_map(nodes, sorted(current))
        moves: list[set[tuple[str, str]]] = []
        for edge in sorted(allowed - current):
            if len(parents[edge[1]]) < d_max:
                moves.append(current | {edge})
        for edge in sorted(current):
            moves.append(current - {edge})
        if not moves:
            break
        scored = [(candidate, _bn_expected_bic(data, nodes, roles, sorted(candidate), tau, n_states, floor)) for candidate in moves]
        candidate, candidate_score = min(scored, key=lambda item: item[1])
        if base_score - candidate_score <= epsilon:
            break
        current = candidate
    return sorted(current)


def _fit_structural_em_initialization(data: np.ndarray, nodes: list[str], roles: dict[str, str], n_states: int, seed: int, initialization: str, config: Mapping[str, Any]) -> StructuralEMResult:
    cfg = _bn_config(config)
    rng = np.random.default_rng(seed)
    d_max = int(cfg["d_max"])
    floor = float(cfg["probability_floor"])
    edges = [] if initialization == "empty" else _random_bn_edges(nodes, roles, d_max, rng)
    tau = rng.dirichlet(np.ones(n_states), size=len(data))
    previous_ll = -np.inf
    converged = False
    for iteration in range(1, int(cfg["em_max_iter"]) + 1):
        weights, upstream, downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor)
        log_joint = _bn_log_joint(data, nodes, roles, edges, weights, upstream, downstream, floor)
        observed_ll = float(_bn_logsumexp(log_joint, axis=1).sum())
        new_tau = np.exp(log_joint - _bn_logsumexp(log_joint, axis=1)[:, None])
        new_edges = _structure_step(data, nodes, roles, edges, new_tau, n_states, d_max, float(cfg["structure_epsilon"]), int(cfg["structure_max_iter"]), floor)
        same_graph = new_edges == edges
        delta = abs(observed_ll - previous_ll) if np.isfinite(previous_ll) else np.inf
        tau = new_tau
        edges = new_edges
        if same_graph and delta < float(cfg["em_tol"]):
            converged = True
            break
        previous_ll = observed_ll
    final_weights, final_upstream, final_downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor)
    final_log_joint = _bn_log_joint(data, nodes, roles, edges, final_weights, final_upstream, final_downstream, floor)
    tau = np.exp(final_log_joint - _bn_logsumexp(final_log_joint, axis=1)[:, None])
    final_weights, final_upstream, final_downstream = _bn_mle_parameters(data, nodes, roles, edges, tau, n_states, floor)
    final_log_joint = _bn_log_joint(data, nodes, roles, edges, final_weights, final_upstream, final_downstream, floor)
    final_ll = float(_bn_logsumexp(final_log_joint, axis=1).sum())
    bic = -2.0 * final_ll + _bn_parameter_count(nodes, roles, edges, n_states) * math.log(max(len(data), 1))
    return StructuralEMResult(nodes, roles, edges, n_states, final_weights, tau, final_upstream, final_downstream, final_ll, bic, iteration, converged, seed, initialization)


def _select_latent_result(results: Sequence[StructuralEMResult], config: Mapping[str, Any]) -> tuple[StructuralEMResult, pd.DataFrame]:
    minimum = float(_bn_config(config)["min_latent_effective_n"])
    rows = []
    admissible = []
    for result in results:
        effective = result.effective_sizes
        valid = result.converged and bool(np.all(effective >= minimum)) and bool(np.all(result.weights > 0))
        rows.append({"K": result.n_states, "seed": result.seed, "initialization": result.initialization, "converged": result.converged, "n_iter": result.n_iter, "log_likelihood": result.log_likelihood, "bic": result.bic, "min_effective_n": float(effective.min()), "admissible": valid, "edges": len(result.edges)})
        if valid:
            admissible.append(result)
    if not admissible:
        raise RuntimeError("Aucune initialisation Structural-EM n'est admissible.")
    per_k = {result.n_states: min((item for item in admissible if item.n_states == result.n_states), key=lambda item: item.bic, default=None) for result in admissible}
    per_k = {key: value for key, value in per_k.items() if value is not None}
    if not per_k:
        raise RuntimeError("Aucun K ne satisfait les critères de convergence et de taille effective.")
    selected = min(per_k.values(), key=lambda item: item.bic)
    selection = pd.DataFrame(rows).sort_values(["K", "bic"], na_position="last").reset_index(drop=True)
    selection["selected_for_K"] = False
    for result in per_k.values():
        selection.loc[(selection["K"] == result.n_states) & (selection["seed"] == result.seed), "selected_for_K"] = True
    selection["selected_final"] = (selection["K"] == selected.n_states) & (selection["seed"] == selected.seed)
    if selected.n_states == max(per_k):
        warnings.warn("Le K sélectionné est sur la borne supérieure de la grille.", RuntimeWarning)
    return selected, selection


def _build_pgmpy_model(result: StructuralEMResult, smooth: bool, alpha: float) -> Any:
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
        if result.roles[node] in {"A0", "A1"}:
            model.add_edge("Z", node)
    parents = _bn_parent_map(result.nodes, result.edges)
    cpds = [TabularCPD("Z", result.n_states, np.asarray(result.weights, dtype=float).reshape(-1, 1).tolist())]
    for node in result.nodes:
        evidence = [*parents[node]]
        if result.roles[node] in {"A0", "A1"}:
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
            mle = result.upstream_probabilities.get((node, state, parent_values), 0.5) if result.roles[node] in {"A0", "A1"} else result.downstream_probabilities.get((node, parent_values), 0.5)
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


def fit_latent_bn_analysis(matrix: pd.DataFrame, roles: Mapping[str, str], config: Mapping[str, Any], output_dir: Path) -> tuple[StructuralEMResult, pd.DataFrame]:
    """Fit and select the single constrained latent BN using Structural EM."""

    cfg = _bn_config(config)
    nodes = list(roles)
    data = matrix[nodes].to_numpy(dtype=np.int8)
    results: list[StructuralEMResult] = []
    n_initializations = int(cfg["n_initializations"])
    base_seed = int(config.get("random_state", 42))
    for n_states in [int(value) for value in cfg["latent_states"]]:
        for init_index in range(n_initializations):
            initialization = "empty" if init_index == 0 else "random"
            seed = base_seed + n_states * 100_000 + init_index
            results.append(_fit_structural_em_initialization(data, nodes, dict(roles), n_states, seed, initialization, config))
    selected, selection = _select_latent_result(results, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / "all_initializations.csv", index=False)
    selection.groupby("K", as_index=False).agg(
        best_bic=("bic", "min"), best_log_likelihood=("log_likelihood", "max"),
        n_admissible=("admissible", "sum"), min_effective_n=("min_effective_n", "max"),
    ).to_csv(output_dir / "K_selection.csv", index=False)
    return selected, selection


def finalize_latent_bn(result: StructuralEMResult, matrix: pd.DataFrame, roles: Mapping[str, str], config: Mapping[str, Any]) -> StructuralEMResult:
    """Apply post-selection Beta smoothing and construct the pgmpy model."""

    cfg = _bn_config(config)
    data = matrix[result.nodes].to_numpy(dtype=np.int8)
    tau = result.responsibilities
    weights, upstream, downstream = _bn_mle_parameters(data, result.nodes, roles, result.edges, tau, result.n_states, float(cfg["probability_floor"]))
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
    final = StructuralEMResult(result.nodes, dict(roles), result.edges, result.n_states, weights, tau, smoothed_upstream, smoothed_downstream, result.log_likelihood, result.bic, result.n_iter, result.converged, result.seed, result.initialization)
    final.model = _build_pgmpy_model(final, smooth=False, alpha=alpha)
    return final


def _scenario_state_space(result: StructuralEMResult, roles: Mapping[str, str]) -> tuple[list[str], list[str], list[str]]:
    upstream = [node for node in result.nodes if roles[node] in {"A0", "A1"}]
    events = [node for node in result.nodes if roles[node] == "B"]
    consequences = [node for node in result.nodes if roles[node] == "C"]
    return upstream, events, consequences


def _family_probability(result: StructuralEMResult, values: Mapping[str, int], state: int, config: Mapping[str, Any]) -> float:
    frame = np.asarray([[int(values[node]) for node in result.nodes]], dtype=np.int8)
    log_joint = _bn_log_joint(frame, result.nodes, result.roles, result.edges, result.weights, result.upstream_probabilities, result.downstream_probabilities, float(_bn_config(config)["probability_floor"]))[0, state]
    return float(np.exp(log_joint - math.log(max(float(result.weights[state]), float(_bn_config(config)["probability_floor"])))))


def extract_latent_bn_scenarios(result: StructuralEMResult, matrix: pd.DataFrame, roles: Mapping[str, str], units: pd.DataFrame, config: Mapping[str, Any], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract constrained MPEs, supports and observed accident prototypes."""

    cfg = _bn_config(config)
    nodes = result.nodes
    data = matrix[nodes].to_numpy(dtype=np.int8)
    log_joint = _bn_log_joint(data, nodes, roles, result.edges, result.weights, result.upstream_probabilities, result.downstream_probabilities, float(cfg["probability_floor"]))
    responsibilities = np.exp(log_joint - _bn_logsumexp(log_joint, axis=1)[:, None])
    responsibilities_frame = pd.DataFrame(responsibilities, columns=[f"family_{state + 1}" for state in range(result.n_states)])
    responsibilities_frame.insert(0, "accident_id", matrix["accident_id"].astype(str).to_numpy())

    family_sizes = pd.DataFrame({"family_id": np.arange(1, result.n_states + 1), "omega": result.weights, "N_eff": responsibilities.sum(axis=0)})
    profile_rows = []
    for state in range(result.n_states):
        for node in nodes:
            marginal = float(result.model.predict if False else np.mean(np.exp(log_joint[:, state] - np.log(np.maximum(result.weights[state], cfg["probability_floor"]))) * data[:, nodes.index(node)]))
            # The exact marginal is obtained from pgmpy when available below.
            try:
                from pgmpy.inference import VariableElimination
                query = VariableElimination(result.model).query([node], evidence={"Z": state}, show_progress=False)
                marginal = float(query.values[1])
            except Exception:
                marginal = float(np.average(data[:, nodes.index(node)], weights=responsibilities[:, state]))
            profile_rows.append({"family_id": state + 1, "variable_name": node, "role": roles[node], "probability": marginal})
    profiles = pd.DataFrame(profile_rows)
    upstream, events, consequences = _scenario_state_space(result, roles)
    if not upstream or not events or not consequences:
        raise ValueError("Le MPE contraint nécessite au moins un thème upstream, B et C.")
    exact_limit = int(cfg.get("exact_mpe_max_variables", 20))
    scenario_rows = []
    support_rows = []
    prototype_rows = []
    for state in range(result.n_states):
        candidates: list[dict[str, int]] = []
        exact = bool(cfg.get("exact_inference", True)) and len(nodes) <= exact_limit
        if exact:
            for bits in itertools.product((0, 1), repeat=len(nodes)):
                values = dict(zip(nodes, bits))
                if any(values[node] for node in upstream) and any(values[node] for node in events) and any(values[node] for node in consequences):
                    candidates.append(values)
        else:
            warnings.warn(f"MPE exact non utilisé pour la famille {state + 1}; repli approximatif explicite.", RuntimeWarning)
            ranked = profiles[profiles["family_id"].eq(state + 1)].sort_values("probability", ascending=False)
            top = ranked.groupby(ranked["role"].isin({"A0", "A1", "B", "C"})).head(3)
            candidates = []
            for upstream_node in ranked[ranked["role"].isin({"A0", "A1"})]["variable_name"].head(3):
                for event_node in ranked[ranked["role"].eq("B")]["variable_name"].head(3):
                    for consequence_node in ranked[ranked["role"].eq("C")]["variable_name"].head(3):
                        candidates.append({node: int(node in {upstream_node, event_node, consequence_node}) for node in nodes})
        if not candidates:
            continue
        probabilities = [_family_probability(result, candidate, state, config) for candidate in candidates]
        best_index = int(np.argmax(probabilities))
        best = candidates[best_index]
        positive = [node for node in nodes if best[node]]
        positive_mask = matrix[positive].all(axis=1).to_numpy(dtype=bool)
        family_support = float(np.sum(responsibilities[:, state] * positive_mask) / max(responsibilities[:, state].sum(), 1e-12))
        global_support = float(positive_mask.mean())
        exact_vector = np.all(data == np.asarray([best[node] for node in nodes]), axis=1)
        exact_vector_support = float(np.sum(responsibilities[:, state] * exact_vector) / max(responsibilities[:, state].sum(), 1e-12))
        hard_family = np.argmax(responsibilities, axis=1) == state
        candidate_indices = np.flatnonzero(hard_family)
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(matrix))
        prototype_index = max(candidate_indices, key=lambda index: _family_probability(result, dict(zip(nodes, data[index])), state, config))
        prototype_accident = str(matrix.iloc[prototype_index]["accident_id"])
        scenario_rows.append({"family_id": state + 1, "N_eff": float(responsibilities[:, state].sum()), "omega": float(result.weights[state]), "A0_factors": ";".join(node for node in positive if roles[node] == "A0"), "A1_factors": ";".join(node for node in positive if roles[node] == "A1"), "B_factors": ";".join(node for node in positive if roles[node] == "B"), "C_factors": ";".join(node for node in positive if roles[node] == "C"), "mpe_probability": float(probabilities[best_index]), "family_positive_support": family_support, "global_positive_support": global_support, "exact_vector_support": exact_vector_support, "prototype_accident_id": prototype_accident, "prototype_probability": _family_probability(result, dict(zip(nodes, data[prototype_index])), state, config), "prototype_posterior_membership": float(responsibilities[prototype_index, state]), "mpe_exact": exact})
        support_rows.append({"family_id": state + 1, "positive_factors": ";".join(positive), "family_positive_support": family_support, "global_positive_support": global_support, "exact_vector_support": exact_vector_support, "mpe_probability": float(probabilities[best_index])})
        prototype_units = units[units["_accident_id"].astype(str).eq(prototype_accident)].copy()
        prototype_rows.append({"family_id": state + 1, "accident_id": prototype_accident, "probability": _family_probability(result, dict(zip(nodes, data[prototype_index])), state, config), "posterior_membership": float(responsibilities[prototype_index, state]), "fact_ids": ";".join(prototype_units["_fact_id"].astype(str)), "sentences": " || ".join(prototype_units.get("_text", pd.Series(dtype=str)).astype(str)[:20])})
    scenarios = pd.DataFrame(scenario_rows)
    supports = pd.DataFrame(support_rows)
    prototypes = pd.DataFrame(prototype_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    responsibilities_frame.to_parquet(output_dir / "posterior_responsibilities.parquet", index=False)
    family_sizes.to_csv(output_dir / "latent_family_sizes.csv", index=False)
    profiles.to_csv(output_dir / "family_factor_profiles.csv", index=False)
    scenarios.to_csv(output_dir / "recurrent_scenarios.csv", index=False)
    supports.to_csv(output_dir / "scenario_support.csv", index=False)
    prototypes.to_csv(output_dir / "prototypes.csv", index=False)
    return scenarios, supports, prototypes, profiles


def write_final_bn_outputs(result: StructuralEMResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.edges, columns=["parent", "child"]).assign(parent_role=lambda frame: frame["parent"].map(result.roles), child_role=lambda frame: frame["child"].map(result.roles)).to_csv(output_dir / "learned_edges.csv", index=False)
    rows = []
    parents = _bn_parent_map(result.nodes, result.edges)
    for node in result.nodes:
        for state in range(result.n_states) if result.roles[node] in {"A0", "A1"} else [None]:
            for parent_values in itertools.product((0, 1), repeat=len(parents[node])):
                probability = result.upstream_probabilities.get((node, state, parent_values)) if state is not None else result.downstream_probabilities.get((node, parent_values))
                rows.append({"variable_name": node, "role": result.roles[node], "family_id": None if state is None else state + 1, "parent_values": "|".join(map(str, parent_values)), "P(value=1)": probability, "P(value=0)": 1.0 - probability})
    pd.DataFrame(rows).to_csv(output_dir / "final_CPT_summary.csv", index=False)
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
        nx.draw_networkx(graph, positions, ax=axis, node_size=500, font_size=7, node_color=[_role_color(result.roles.get(node, "Z")) for node in graph.nodes], arrows=True)
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / "final_bn.png", dpi=220, bbox_inches="tight")
        plt.close(figure)
    except Exception as error:
        warnings.warn(f"Figure BN non générée: {error}", RuntimeWarning)


def run_frozen_bn_analysis(config: Mapping[str, Any], run_dir: Path, partition_selections: Mapping[str, str], output_dir: Path, units: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run the complete downstream analysis for one dataset."""

    if units is None:
        units, _ = load_units(config)
    matrix, included, excluded, roles = build_frozen_bn_inputs(units, run_dir, partition_selections, config, output_dir)
    selected, selection = fit_latent_bn_analysis(matrix, roles, config, output_dir)
    final = finalize_latent_bn(selected, matrix, roles, config)
    write_final_bn_outputs(final, output_dir)
    scenarios, supports, prototypes, profiles = extract_latent_bn_scenarios(final, matrix, roles, units, config, output_dir)
    return {"matrix": matrix, "theme_dictionary": included, "excluded_themes": excluded, "selection": selection, "result": final, "scenarios": scenarios, "supports": supports, "prototypes": prototypes, "profiles": profiles}


def run_theme_discovery(
    config_path: Path,
    *,
    dataset_id: str | None = None,
    reestimate: bool = False,
    stage: str = "all",
    run_dir: Path | None = None,
) -> Path:
    """Run the theme-discovery workflow.

    ``metrics`` computes or reuses candidate and resampling metrics,
    ``pareto`` regenerates only the Pareto selection from an existing run,
    and ``all`` performs both stages. Partition selection is manual and occurs
    in the results notebook.
    """
    if stage not in {"all", "metrics", "pareto"}:
        raise ValueError("stage must be one of: all, metrics, pareto")
    if stage == "pareto" and run_dir is None:
        raise ValueError("run_dir is required when stage='pareto'")

    raw_config = select_dataset_config(load_yaml_config(config_path), dataset_id)
    config = resolve_config_paths(raw_config, config_path)

    if run_dir is not None:
        output_base = Path(run_dir).expanduser().resolve()
    else:
        output_base = Path(config["data"]["output_dir"])
        if output_base.exists() and any(output_base.iterdir()) and not config.get("runtime", {}).get("overwrite", False):
            output_base = output_base.with_name(output_base.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    if stage == "pareto" and not output_base.is_dir():
        raise FileNotFoundError(f"Existing run directory not found: {output_base}")
    output_base.mkdir(parents=True, exist_ok=True)
    _log_progress(
        f"START dataset={config['data'].get('dataset_id')} stage={stage} "
        f"output={output_base} workers={resolve_n_workers(config)}"
    )

    if stage != "pareto":
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
    stability_themes: dict[str, pd.DataFrame] = {}
    stability_summaries: dict[str, pd.DataFrame] = {}
    selections: dict[str, str] = {}
    selection_tables: dict[str, pd.DataFrame] = {}
    agreements: dict[str, pd.DataFrame] = {}

    prepared: PreparedData | None = None
    if stage in {"all", "metrics"}:
        prepared = prepare_data(config, output_base)
        _log_progress(
            f"données prêtes: {len(prepared.units)} unités, "
            f"{prepared.units['_accident_id'].nunique()} accidents"
        )
        for role in ROLES:
            candidates = evaluate_pareto_candidates(
                role, prepared.units, prepared.embeddings, config, output_base, reestimate=reestimate
            )
            theme_frame, summary = evaluate_resampling_stability(
                role, prepared.units, prepared.embeddings, config, output_base, candidates, reestimate=reestimate
            )
            merged = candidates.merge(summary, on=["role", "configuration_id"], how="left")
            candidate_tables[role] = merged
            stability_themes[role] = theme_frame
            stability_summaries[role] = summary
    else:
        for role in ROLES:
            candidate_path = output_base / "pareto" / role / "candidate_metrics.csv"
            stability_path = output_base / "pareto" / role / "stability_summary.csv"
            missing = [str(path) for path in (candidate_path, stability_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot run Pareto-only stage; missing metric artifacts: " + ", ".join(missing)
                )
            candidates = pd.read_csv(candidate_path)
            summary = pd.read_csv(stability_path)
            required = {"role", "configuration_id"}
            if not required.issubset(candidates.columns) or not required.issubset(summary.columns):
                raise ValueError(f"Invalid cached metric tables for role {role}: missing role/configuration_id")
            candidate_tables[role] = candidates.merge(summary, on=["role", "configuration_id"], how="left")
            stability_summaries[role] = summary

    if stage == "metrics":
        _log_progress(f"DONE stage=metrics output={output_base}")
        return output_base

    _log_progress("Pareto: sélection des partitions non-dominées par rôle")
    for role in ROLES:
        merged = candidate_tables[role]
        table, agreement, selected_id = select_pareto_partitions(
            role,
            merged,
            output_base,
        )
        selection_tables[role] = table
        agreements[role] = agreement
        selections[role] = selected_id
    pareto_frontiers = []
    for role in ROLES:
        table = selection_tables.get(role, pd.DataFrame())
        if table.empty:
            continue
        frontier = table[table["pareto_non_dominated"].fillna(False)].copy()
        if not frontier.empty:
            pareto_frontiers.append(_ordered_pareto_export(frontier))
    if pareto_frontiers:
        pd.concat(pareto_frontiers, ignore_index=True).to_csv(
            output_base / "pareto_frontier_all_roles.csv", index=False
        )
    write_pareto_figure(selection_tables, output_base / "figures")
    summary_rows = []
    for role in ROLES:
        table = selection_tables.get(role, pd.DataFrame())
        summary_rows.append({
            "Role": role,
            "n_candidates": int(len(table)),
            "n_pareto": int(table["pareto_non_dominated"].sum()) if not table.empty else 0,
        })
    pareto_summary = pd.DataFrame(summary_rows)
    pareto_summary.to_csv(output_base / "pareto_selection_summary.csv", index=False)
    selected_metrics = pareto_summary
    _log_progress("métriques des configurations Pareto retenues:\n" + selected_metrics.to_string(index=False))
    if stage == "manual_selection" and prepared is not None:
        _log_progress("matérialisation: thèmes sélectionnés et dictionnaire")
        partition_results = build_selected_partition_results(
            prepared, config, output_base, selections, stability_themes
        )
        topic_dictionary = build_topic_dictionary(prepared, partition_results, config, output_base / "topics")
        pd.DataFrame([{"role": role, "configuration_id": selections[role]} for role in ROLES]).to_csv(
            output_base / "selected_configurations.csv", index=False
        )
        (output_base / "theme_discovery_manifest.json").write_text(
            json.dumps({
                "version": "parallel_du_sr_discovery_v1",
                "dataset_id": config["data"].get("dataset_id"),
                "selection_objectives": ["dbcv_umap", "stability"],
                "n_workers": resolve_n_workers(config),
                "n_topics": int(len(topic_dictionary)),
                "selected_configurations": selections,
            }, indent=2),
            encoding="utf-8",
        )
    _log_progress(f"DONE stage={stage} output={output_base}")
    return output_base


__all__ = [
    "ROLES", "PreparedData", "PartitionResult", "StructuralEMResult", "load_yaml_config",
    "select_dataset_config", "resolve_config_paths", "prepare_data", "parameter_plan",
    "build_topic_dictionary",
    "build_accident_topic_matrix", "descriptive_tables", "build_frozen_bn_inputs",
    "fit_latent_bn_analysis", "finalize_latent_bn", "extract_latent_bn_scenarios",
    "run_frozen_bn_analysis", "run_theme_discovery", "resolve_n_workers",
    "screen_clustering_parameters", "mark_admissible_configurations", "resolve_admissibility_rules", "load_topic_stopwords",
    "select_admissible_parameter_combinations", "write_screening_figures",
    "evaluate_pareto_candidates", "evaluate_resampling_stability",
    "select_pareto_partitions", "build_selected_partition_results", "write_pareto_figure",
]
