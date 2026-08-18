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
import platform
import re
import sys
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


@dataclass
class BinaryBNParameters:
    nodes: list[str]
    edges: list[tuple[str, str]]
    probabilities: dict[str, dict[tuple[int, ...], float]]

    @property
    def parents(self) -> dict[str, list[str]]:
        result = {node: [] for node in self.nodes}
        for parent, child in self.edges:
            result.setdefault(child, []).append(parent)
        return result

    def log_probability_matrix(self, data: pd.DataFrame) -> np.ndarray:
        values = data.reindex(columns=self.nodes, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=int)
        parent_map = self.parents
        logs = np.zeros(len(values), dtype=float)
        for row_index, row in enumerate(values):
            log_value = 0.0
            for node_index, node in enumerate(self.nodes):
                parent_key = tuple(int(row[self.nodes.index(parent)]) for parent in parent_map.get(node, []))
                probability = float(self.probabilities[node].get(parent_key, 0.5))
                probability = min(max(probability, 1e-12), 1.0 - 1e-12)
                log_value += math.log(probability if int(row[node_index]) else 1.0 - probability)
            logs[row_index] = log_value
        return logs

    def probability(self, row: Mapping[str, int]) -> float:
        frame = pd.DataFrame([{node: int(row.get(node, 0)) for node in self.nodes}])
        return float(np.exp(self.log_probability_matrix(frame)[0]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [list(edge) for edge in self.edges],
            "probabilities": {
                node: {"|".join(map(str, key)): value for key, value in values.items()}
                for node, values in self.probabilities.items()
            },
        }


@dataclass
class LatentMixtureBN:
    nodes: list[str]
    edges: list[tuple[str, str]]
    n_states: int
    weights: np.ndarray | None = None
    components: list[BinaryBNParameters] | None = None
    log_likelihood: float = float("nan")
    n_iter: int = 0

    def fit(
        self,
        data: pd.DataFrame,
        *,
        equivalent_sample_size: float = 5.0,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        random_state: int = 42,
    ) -> "LatentMixtureBN":
        rng = np.random.default_rng(random_state)
        n_rows = len(data)
        responsibilities = rng.dirichlet(np.ones(self.n_states), size=n_rows)
        previous = -np.inf
        for iteration in range(max_iter):
            weights = responsibilities.sum(axis=0) + 1e-12
            self.weights = weights / weights.sum()
            self.components = [
                fit_binary_bn_parameters(
                    data,
                    self.edges,
                    equivalent_sample_size=equivalent_sample_size,
                    sample_weights=responsibilities[:, state],
                )
                for state in range(self.n_states)
            ]
            component_logs = np.column_stack(
                [component.log_probability_matrix(data) for component in self.components]
            )
            joint_logs = component_logs + np.log(self.weights)[None, :]
            normalizer = _logsumexp(joint_logs, axis=1)
            responsibilities = np.exp(joint_logs - normalizer[:, None])
            current = float(np.mean(normalizer))
            if abs(current - previous) < tolerance:
                self.n_iter = iteration + 1
                self.log_likelihood = current
                return self
            previous = current
        self.n_iter = max_iter
        self.log_likelihood = float(previous)
        return self

    def log_probability_matrix(self, data: pd.DataFrame) -> np.ndarray:
        if self.weights is None or self.components is None:
            raise RuntimeError("Le modèle latent doit être ajusté avant le calcul de probabilité.")
        component_logs = np.column_stack(
            [component.log_probability_matrix(data) for component in self.components]
        )
        return _logsumexp(component_logs + np.log(self.weights)[None, :], axis=1)

    def probability(self, row: Mapping[str, int]) -> float:
        frame = pd.DataFrame([{node: int(row.get(node, 0)) for node in self.nodes}])
        return float(np.exp(self.log_probability_matrix(frame)[0]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [list(edge) for edge in self.edges],
            "n_states": self.n_states,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "n_iter": self.n_iter,
            "log_likelihood": self.log_likelihood,
            "components": [component.to_dict() for component in self.components or []],
        }


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.exp(values - maximum).sum(axis=axis))


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
    }
    if cache_path.is_file() and not reestimate:
        cached = pd.read_csv(cache_path)
        if required_columns.issubset(cached.columns):
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
                int(config.get("random_state", 42)) + configuration_index,
                config,
            )
        metrics = _screening_metrics(role_units, selected_indices, labels)
        rows.append({
            "role": role,
            "configuration_id": f"{role}_cfg_{configuration_index:03d}",
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
    if metrics_path.is_file() and not reestimate:
        cached = pd.read_csv(metrics_path)
        membership_files_present = all(
            (candidate_dir / f"{configuration_id}_membership_strength.npy").is_file()
            for configuration_id in cached.get("configuration_id", pd.Series(dtype=str)).astype(str)
        )
        if {"configuration_id", "dbcv_umap"}.issubset(cached.columns) and membership_files_present:
            _log_progress(f"[{role}] candidats: cache réutilisé ({metrics_path})")
            return cached.loc[:, ~cached.columns.str.contains("semantic", case=False)]
    clustering_input = role_embeddings
    plan = parameter_plan(
        config,
        role=None,
        apply_selection=False,
        budget_override=None,
    )
    random_state = int(pareto_cfg.get("random_state", config.get("random_state", 42)))
    dbcv_sample_size = pareto_cfg.get("dbcv_sample_size")
    tasks = [
        {
            "index": index,
            "role": role,
            "configuration_id": _configuration_id(role, index),
            "params": params,
            "embeddings": clustering_input,
            "accident_ids": role_units["_accident_id"].astype(str).to_numpy(),
            "random_state": random_state + index,
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
    if theme_path.is_file() and summary_path.is_file() and not reestimate:
        _log_progress(f"[{role}] resampling: cache réutilisé ({summary_path})")
        return pd.read_csv(theme_path), pd.read_csv(summary_path)
    candidate_dir = pareto_dir / "candidate_partitions"
    clustering_input = role_embeddings
    n_repetitions = int(pareto_cfg.get("n_resampling", 30))
    fraction = float(pareto_cfg.get("resampling_fraction", 0.8))
    random_state = int(pareto_cfg.get("random_state", config.get("random_state", 42)))
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
            rng = np.random.default_rng(random_state + ROLE_RANK[role] * 10000 + candidate_index * 100 + repetition)
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
                "random_state": random_state + repetition + candidate_index,
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
    _log_progress(
        f"[{role}] resampling: terminé ({len(summary)} résumés, "
        f"{len(all_tasks)} tâches)"
    )
    (pareto_dir / "stability_metadata.json").write_text(
        json.dumps({
            "version": "resampling_stability_parallel_v2",
            "role": role,
            "n_tasks": len(all_tasks),
            "n_workers": resolve_n_workers(config),
            "n_repetitions": n_repetitions,
        }, indent=2),
        encoding="utf-8",
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


def learn_constrained_edges(data: pd.DataFrame, variable_macro_map: Mapping[str, str], config: Mapping[str, Any]) -> list[tuple[str, str]]:
    from bn_pipeline.bn_structure import learn_macro_constrained_structure

    nodes = list(variable_macro_map)
    model, edges = learn_macro_constrained_structure(
        data[nodes],
        dict(variable_macro_map),
        max_indegree=int(config["bayesian_networks"].get("max_indegree", 3)),
        disallow_a0_to_b_direct=not bool(config["bayesian_networks"].get("include_a0_to_b_direct", True)),
        ensure_macro_chain_backbone=bool(config["bayesian_networks"].get("ensure_macro_chain_backbone", False)),
    )
    del model
    return [(str(parent), str(child)) for parent, child in edges]


def fit_binary_bn_parameters(
    data: pd.DataFrame,
    edges: Sequence[tuple[str, str]],
    *,
    equivalent_sample_size: float = 5.0,
    sample_weights: np.ndarray | None = None,
) -> BinaryBNParameters:
    nodes = list(data.columns)
    numeric = data.reindex(columns=nodes, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 1).astype(int)
    weights = np.ones(len(numeric), dtype=float) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    parent_map = {node: [] for node in nodes}
    for parent, child in edges:
        parent_map.setdefault(child, []).append(parent)
    probabilities: dict[str, dict[tuple[int, ...], float]] = {}
    for node in nodes:
        parents = parent_map.get(node, [])
        probabilities[node] = {}
        parent_combinations = itertools.product((0, 1), repeat=len(parents))
        for parent_values in parent_combinations:
            mask = np.ones(len(numeric), dtype=bool)
            for parent, value in zip(parents, parent_values):
                mask &= numeric[parent].to_numpy() == value
            total = float(weights[mask].sum())
            positive = float((weights[mask] * numeric.loc[mask, node].to_numpy()).sum())
            probabilities[node][tuple(parent_values)] = (positive + equivalent_sample_size * 0.5) / (total + equivalent_sample_size)
    return BinaryBNParameters(nodes=nodes, edges=list(edges), probabilities=probabilities)


def fit_no_z_model(data: pd.DataFrame, variable_macro_map: Mapping[str, str], config: Mapping[str, Any]) -> tuple[list[tuple[str, str]], BinaryBNParameters]:
    edges = learn_constrained_edges(data, variable_macro_map, config)
    return edges, fit_binary_bn_parameters(data, edges, equivalent_sample_size=float(config["bayesian_networks"].get("equivalent_sample_size", 5)))


def fit_latent_model(data: pd.DataFrame, edges: Sequence[tuple[str, str]], config: Mapping[str, Any], n_states: int, random_state: int = 42) -> LatentMixtureBN:
    model = LatentMixtureBN(nodes=list(data.columns), edges=list(edges), n_states=int(n_states))
    return model.fit(
        data,
        equivalent_sample_size=float(config["bayesian_networks"].get("equivalent_sample_size", 5)),
        max_iter=int(config["bayesian_networks"].get("latent_max_iter", 100)),
        tolerance=float(config["bayesian_networks"].get("latent_tolerance", 1e-4)),
        random_state=random_state,
    )


def evaluate_bn_cv(matrix: pd.DataFrame, variable_macro_map: Mapping[str, str], config: Mapping[str, Any], output_dir: Path) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold

    nodes = list(variable_macro_map)
    data = matrix[nodes].copy()
    groups = matrix["accident_id"].astype(str)
    n_folds = int(config["bayesian_networks"].get("n_folds", 5))
    n_folds = min(n_folds, groups.nunique())
    if n_folds < 2:
        raise ValueError("Au moins deux accidents sont nécessaires pour la validation croisée BN.")
    rows = []
    splitter = GroupKFold(n_splits=n_folds)
    latent_states = _grid_values(config["bayesian_networks"].get("latent_states", [2, 3, 4]))
    for fold, (train_index, test_index) in enumerate(splitter.split(data, groups=groups)):
        train, test = data.iloc[train_index], data.iloc[test_index]
        edges, no_z = fit_no_z_model(train, variable_macro_map, config)
        rows.append({"model": "BN_without_Z", "fold": fold, "latent_states": None, "log_likelihood": float(no_z.log_probability_matrix(test).mean()), "n_train": len(train), "n_test": len(test), "n_edges": len(edges)})
        for n_states in latent_states:
            latent = fit_latent_model(train, edges, config, int(n_states), random_state=int(config.get("random_state", 42)) + fold + int(n_states))
            rows.append({"model": "BN_with_Z", "fold": fold, "latent_states": int(n_states), "log_likelihood": float(latent.log_probability_matrix(test).mean()), "n_train": len(train), "n_test": len(test), "n_edges": len(edges), "em_iterations": latent.n_iter})
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "cv_log_likelihood.csv", index=False)
    result.groupby(["model", "latent_states"], dropna=False)["log_likelihood"].agg(["mean", "std"]).reset_index().to_csv(output_dir / "cv_log_likelihood_summary.csv", index=False)
    return result


def bootstrap_arc_stability(matrix: pd.DataFrame, variable_macro_map: Mapping[str, str], config: Mapping[str, Any], output_dir: Path) -> pd.DataFrame:
    repetitions = int(config["bayesian_networks"].get("bootstrap_repetitions", 100))
    rng = np.random.default_rng(int(config.get("random_state", 42)) + 1000)
    accidents = matrix["accident_id"].astype(str).drop_duplicates().to_numpy()
    rows = []
    for repetition in range(repetitions):
        sampled = rng.choice(accidents, size=len(accidents), replace=True)
        sampled_matrix = pd.concat([matrix[matrix["accident_id"].astype(str) == accident] for accident in sampled], ignore_index=True)
        edges = learn_constrained_edges(sampled_matrix[list(variable_macro_map)], variable_macro_map, config)
        for edge in edges:
            rows.append({"model": "BN_without_Z", "repetition": repetition, "parent": edge[0], "child": edge[1], "present": 1})
            rows.append({"model": "BN_with_Z", "repetition": repetition, "parent": edge[0], "child": edge[1], "present": 1})
    all_edges = list(itertools.permutations(variable_macro_map, 2))
    for model_name in ("BN_without_Z", "BN_with_Z"):
        for repetition in range(repetitions):
            seen = {(row["parent"], row["child"]) for row in rows if row["model"] == model_name and row["repetition"] == repetition}
            for parent, child in all_edges:
                if (parent, child) not in seen:
                    rows.append({"model": model_name, "repetition": repetition, "parent": parent, "child": child, "present": 0})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["model", "parent", "child"], as_index=False)["present"].mean().rename(columns={"present": "bootstrap_frequency"})
    summary["stable_at_threshold"] = summary["bootstrap_frequency"] >= float(config["bayesian_networks"].get("bootstrap_arc_threshold", 0.75))
    detail.to_csv(output_dir / "arc_stability_repetitions.csv", index=False)
    summary.to_csv(output_dir / "arc_stability.csv", index=False)
    return summary


def extract_scenarios(
    matrix: pd.DataFrame,
    selected_topics: pd.DataFrame,
    variable_macro_map: Mapping[str, str],
    no_z: BinaryBNParameters,
    latent_models: Mapping[int, LatentMixtureBN],
    config: Mapping[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    by_role = {role: selected_topics.loc[selected_topics["role"] == role, "topic_id"].astype(str).tolist() for role in ROLES}
    combinations = itertools.product(*(by_role[role] for role in ROLES))
    topic_columns = list(variable_macro_map)
    rows = []
    for selected in combinations:
        row = {column: 0 for column in topic_columns}
        for topic_id in selected:
            row[topic_id] = 1
        exact_mask = (matrix[topic_columns] == pd.Series(row)).all(axis=1)
        partial_mask = matrix[list(selected)].sum(axis=1) == len(selected)
        no_z_probability = no_z.probability(row)
        latent_probabilities = {str(state): model.probability(row) for state, model in latent_models.items()}
        rows.append({
            "scenario_id": 0,
            "A0": selected[0], "A1": selected[1], "B": selected[2], "C": selected[3],
            "model_probability_without_Z": no_z_probability,
            "model_probability_with_Z_max": max(latent_probabilities.values()) if latent_probabilities else float("nan"),
            "support_exact": int(exact_mask.sum()),
            "support_exact_share": float(exact_mask.mean()),
            "support_partial": int(partial_mask.sum()),
            "support_partial_share": float(partial_mask.mean()),
            "latent_probabilities": json.dumps(latent_probabilities),
        })
    scenarios = pd.DataFrame(rows)
    if scenarios.empty:
        scenarios = pd.DataFrame(columns=["scenario_id", "A0", "A1", "B", "C"])
    else:
        scenarios = scenarios[(scenarios["support_exact"] >= int(config["bayesian_networks"].get("min_scenario_support", 3))) | (scenarios["support_partial"] >= int(config["bayesian_networks"].get("min_scenario_support", 3)))]
        scenarios = scenarios.sort_values(["model_probability_without_Z", "support_partial"], ascending=False).head(int(config["bayesian_networks"].get("top_scenarios", 30))).reset_index(drop=True)
        scenarios["scenario_id"] = np.arange(1, len(scenarios) + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / "scenario_catalog.csv", index=False)
    return scenarios


def write_network_figure(edges: Sequence[tuple[str, str]], variable_macro_map: Mapping[str, str], arc_stability: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(variable_macro_map)
    graph.add_edges_from(edges)
    latent_graph = graph.copy()
    latent_graph.add_node("Z", role="Z")
    root_nodes = [node for node in graph.nodes if variable_macro_map.get(node) in {"A0", "A1"} and graph.in_degree(node) == 0]
    latent_graph.add_edges_from(("Z", node) for node in root_nodes)
    positions = _fixed_network_positions(list(graph.nodes), variable_macro_map)
    latent_positions = dict(positions)
    latent_positions["Z"] = (0.5, 1.12)
    stability = {(row["parent"], row["child"]): float(row["bootstrap_frequency"]) for _, row in arc_stability[arc_stability["model"] == "BN_without_Z"].iterrows()}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, model_graph, title, positions_for_graph in ((axes[0], graph, "Constrained BN without Z", positions), (axes[1], latent_graph, "Constrained latent BN with Z", latent_positions)):
        nx.draw_networkx_nodes(model_graph, positions_for_graph, ax=axis, node_color=[_role_color(variable_macro_map.get(node, "Z")) for node in model_graph.nodes], node_size=900, edgecolors="black")
        nx.draw_networkx_labels(model_graph, positions_for_graph, ax=axis, font_size=8)
        widths = [1.0 + 4.0 * stability.get(edge, 0.0) for edge in model_graph.edges]
        colors = ["#6A3D9A" if edge[0] == "Z" else "#555555" for edge in model_graph.edges]
        nx.draw_networkx_edges(model_graph, positions_for_graph, ax=axis, arrows=True, arrowsize=16, width=widths, edge_color=colors, connectionstyle="arc3,rad=0.04")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _fixed_network_positions(nodes: Sequence[str], variable_macro_map: Mapping[str, str]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for role in ROLES:
        role_nodes = sorted(node for node in nodes if variable_macro_map.get(node) == role)
        for index, node in enumerate(role_nodes):
            positions[node] = (ROLE_RANK[role], 1.0 - (index + 1) / (len(role_nodes) + 1))
    return positions


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


def save_models(
    no_z_edges: Sequence[tuple[str, str]],
    no_z: BinaryBNParameters,
    latent_models: Mapping[int, LatentMixtureBN],
    variable_macro_map: Mapping[str, str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bn_without_z_edges.json").write_text(json.dumps([list(edge) for edge in no_z_edges], indent=2), encoding="utf-8")
    (output_dir / "bn_without_z_parameters.json").write_text(json.dumps(no_z.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "variable_macro_map.json").write_text(json.dumps(dict(variable_macro_map), indent=2), encoding="utf-8")
    for n_states, model in latent_models.items():
        (output_dir / f"bn_with_z_{n_states}_states.json").write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


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


def write_audit_report(config: Mapping[str, Any], prepared: PreparedData, selected_topics: pd.DataFrame, cv: pd.DataFrame, arc_stability: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Recurrent accident scenarios — audit report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"Python: {platform.python_version()}",
        "",
        "## Scope",
        "",
        "The run covers intra-role consensus themes, accident-level binary aggregation, descriptive co-occurrence/lift, and constrained BN comparison with and without latent Z. Inter-sector transfer and alternative methods are excluded.",
        "",
        "## Input",
        "",
        f"- Units: {len(prepared.units)}",
        f"- Accidents: {prepared.units['_accident_id'].nunique()}",
        f"- Embedding dimension: {prepared.embeddings.shape[1]}",
        f"- Selected BN topics: {len(selected_topics)}",
        "",
        "## Central diagnostics",
        "",
        cv.groupby(["model", "latent_states"], dropna=False)["log_likelihood"].agg(["mean", "std"]).to_string(index=False),
        "",
        "Arc stability summary:",
        arc_stability.groupby("model")["bootstrap_frequency"].agg(["mean", "median"]).to_string(),
        "",
        "## Interpretation guardrails",
        "",
        "- A zero means that a selected topic was not observed in the available units for the accident.",
        "- BN arcs are conditional dependencies under structural constraints, not causal proof.",
        "- Z is a latent family variable in a mixture of constrained process networks; its arcs are not accident-process causes.",
        "- Scenario support and stability must be checked before prevention-oriented interpretation.",
    ]
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "ROLES", "PreparedData", "PartitionResult", "LatentMixtureBN", "load_yaml_config",
    "select_dataset_config", "resolve_config_paths", "prepare_data", "parameter_plan",
    "build_topic_dictionary",
    "build_accident_topic_matrix", "descriptive_tables", "fit_no_z_model", "fit_latent_model",
    "evaluate_bn_cv", "bootstrap_arc_stability", "extract_scenarios", "run_theme_discovery", "resolve_n_workers",
    "screen_clustering_parameters", "mark_admissible_configurations", "resolve_admissibility_rules", "load_topic_stopwords",
    "select_admissible_parameter_combinations", "write_screening_figures",
    "evaluate_pareto_candidates", "evaluate_resampling_stability",
    "select_pareto_partitions", "build_selected_partition_results", "write_pareto_figure",
]
