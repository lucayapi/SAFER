"""Build post-run results notebooks for recurrent-accident discovery (one per corpus)."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"
DATASETS = ("caou", "btp", "metallurgie")


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


def build_cells(dataset_id: str) -> list[dict]:
    return [
    markdown(
        f"""
# Recurrent accident scenarios — results notebook ({dataset_id})

This notebook starts after theme discovery (`run_theme_discovery.py`) for corpus
`{dataset_id}`. It does not rerun UMAP, HDBSCAN, resampling or configuration selection.

Reporting structure (manuscript-oriented):

1. Corpus summary by role
2. Pareto-front and knee-selection tables
3. Pareto figures, resampling and seed-sensitivity diagnostics (membership-strength plots → appendix)
4. Topic dictionary, LLM labels and retained-factors summary (Table 4.5)
5. Narrative inspection and 2-D UMAP maps
6. Continue with `recurrent_scenarios_bn_analysis_{dataset_id}.ipynb`

Editable below: representative sentences, role prompts, OpenAI model,
and 2-D UMAP display settings. The run directory is fixed to this corpus.
        """
    ),
    code(
        f"""
from pathlib import Path
import json
import os
import sys
import textwrap
import warnings

import numpy as np

if int(np.__version__.split(".")[0]) >= 2:
    raise ImportError(
        "numpy>=2 détecté (kernel Anaconda/base). "
        "Ce projet attend numpy==1.26.4 (text/requirements.txt). "
        "Activez le venv du projet (text/.venv) et sélectionnez ce kernel dans Jupyter."
    )

import pandas as pd
import yaml
import matplotlib.pyplot as plt
from IPython.display import Image, display
from tqdm.auto import tqdm

SCENARIO_DIR = Path.cwd()
scenario_candidates = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
    Path.cwd() / "text" / "recurrent_scenarios",
    Path.cwd().parent / "recurrent_scenarios",
    Path.cwd().parent.parent / "recurrent_scenarios",
]
SCENARIO_DIR = next(
    (path.resolve() for path in scenario_candidates if (path / "scenario_pipeline.py").is_file()),
    SCENARIO_DIR.resolve(),
)

# Load secrets without printing or storing them in the notebook outputs.
try:
    from dotenv import load_dotenv

    env_candidates = [
        Path.cwd() / ".env",
        SCENARIO_DIR.parent / ".env",
        SCENARIO_DIR.parent.parent / ".env",
    ]
    for env_path in env_candidates:
        if env_path.is_file():
            load_dotenv(env_path, override=False)
except ImportError:
    warnings.warn("python-dotenv n'est pas installé; les variables d'environnement existantes seront utilisées.")

DATASET_ID = {dataset_id!r}
DISCOVERY_RUN_NAME = "theme_discovery_audit"
RUN_DIRECTORY = f"runs/{{DISCOVERY_RUN_NAME}}/{{DATASET_ID}}"
OPENAI_ENABLED = True
OPENAI_MODEL = "gpt-5.6-luna"
OPENAI_REASONING_EFFORT = "low"
OPENAI_MAX_OUTPUT_TOKENS = 4000
LLM_OUTPUT_LANGUAGE = "English"
LLM_FORCE_REFRESH = False  # True = ignore llm_theme_labels.csv and relabel every topic
N_REPRESENTATIVE_SENTENCES = 50
MAX_TOPICS_PER_ROLE_FOR_LLM = None
FALLBACK_TOPICS_PER_ROLE = None
UMAP_2D_N_NEIGHBORS = 15
UMAP_2D_MIN_DIST = 0.1
UMAP_2D_RANDOM_STATE = 42
MAX_POINTS_PER_ROLE_PLOT = 12000
UMAP_2D_FIGSIZE = (14, 12)
UMAP_2D_FIGSIZE_COMPACT = (10, 10)
DATAMAP_LABEL_WRAP_WIDTH = 22
DATAMAP_LABEL_FONT_SIZE = 9
DATAMAP_DYNAMIC_LABEL_SIZE = False
DATAMAP_NOISE_LABEL = "Noise"
UMAP_2D_DPI = 300
print("OPENAI_API_KEY available:", bool(os.environ.get("OPENAI_API_KEY")))

ROLE_PROMPTS = {{
    "A0": '''You name A0 semantic factors (work situation before the accident). Each label must denote a precise, observable motif: concrete activity, workstation type, equipment, location, or operational setup. Forbidden: meta or vague labels ("work context", "work environment", "professional activity", "work in general"). The label must distinguish this theme from others in the same role. 4 to 10 words, grounded in the examples. No consequence or accident event.''',
    "A1": '''You name A1 semantic factors (adverse condition or hazard before the event). Each label must name the hazard, failure, missing protection, or dangerous condition specifically. Forbidden: meta labels ("adverse factor", "dangerous condition", "risk", "general hazard"). 4 to 10 words, grounded in the examples. Not the event or injury unless indispensable to distinguish the condition.''',
    "B": '''You name B semantic factors (immediate event or mechanism). Each label must describe what happened or the mechanism concretely (fall, impact, entrapment, unintended start, etc.). Forbidden: meta labels ("accident event", "accident", "incident", "general deviation"). 4 to 10 words, grounded in the examples. Not upstream context or final injury.''',
    "C": '''You name C semantic factors (consequence or injury). Each label must specify the nature of harm or body location (injury type, body part, severity if visible in examples). Forbidden: meta labels ("consequence", "injury", "lesion", "damage" alone). 4 to 10 words, grounded in the examples. Not the cause or work context.''',
}}

def resolve_run_directory(value):
    requested = Path(value).expanduser()
    if requested.is_absolute():
        candidates = [requested]
    else:
        candidates = [
            SCENARIO_DIR / requested,
            SCENARIO_DIR / "runs" / requested,
            SCENARIO_DIR.parent / requested,
        ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    searched = "\\n".join(f"- {{candidate.resolve()}}" for candidate in candidates)
    available = sorted(path.name for path in (SCENARIO_DIR / "runs").glob("**/*") if path.is_dir())
    available_text = ", ".join(available[-20:]) if available else "aucun dossier de run trouvé"
    raise FileNotFoundError(
        f"Run directory not found for RUN_DIRECTORY={{value!r}}.\\n"
        f"Paths searched:\\n{{searched}}\\n"
        f"Recent run directories: {{available_text}}"
    )

RUN_DIR = resolve_run_directory(RUN_DIRECTORY)
ROLES = ("A0", "A1", "B", "C")
print("Dataset:", DATASET_ID)
print("Scenario directory:", SCENARIO_DIR)
print("Run directory:", RUN_DIR)

def read_json(name, default=None):
    path = RUN_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def local_path(value):
    path = Path(value).expanduser()
    candidates = [path]
    parts = path.parts
    if "text" in parts:
        text_index = parts.index("text")
        candidates.append(SCENARIO_DIR.parent / Path(*parts[text_index + 1:]))
    candidates.extend([
        SCENARIO_DIR / path.name,
        SCENARIO_DIR.parent / path.name,
        SCENARIO_DIR.parent.parent / path.name,
    ])
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate.resolve()
    return path


def load_run_config():
    yaml_path = RUN_DIR / "config_resolved.yaml"
    if yaml_path.is_file():
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {{}}
    else:
        loaded = read_json("theme_discovery_parameters.json", {{}})
    if "data" not in loaded:
        raise ValueError(
            f"La configuration du run ne contient pas la section 'data' : {{yaml_path}}"
        )
    data_cfg = loaded["data"]
    for key in ("units_path", "embeddings_path"):
        if data_cfg.get(key):
            data_cfg[key] = str(local_path(data_cfg[key]))
    topics_cfg = loaded.setdefault("topics", {{}})
    if topics_cfg.get("stopwords_file"):
        topics_cfg["stopwords_file"] = str(local_path(topics_cfg["stopwords_file"]))
    return loaded
        """
    ),
    markdown(
        """
## 1. Corpus description

Role-conditioned factual units retained for theme discovery. *Contributing accidents*
counts distinct accident identifiers with at least one valid unit in the role.
        """
    ),
    code(
        """
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from scenario_pipeline import load_units
from manuscript_reporting import (
    KNEE_TABLE_COLUMNS,
    PARETO_TABLE_COLUMNS,
    RETAINED_FACTOR_COLUMNS,
    FIGURE_FACTOR_RESAMPLING_A0,
    FIGURE_FACTOR_RESAMPLING_A1_B_C,
    FIGURE_RETAINED_FACTORS_A0,
    FIGURE_UMAP_SEED_SENSITIVITY,
    ROLE_RETAINED_FACTOR_TITLES_SHORT,
    appendix_figure_path,
    retained_factors_figure_name,
    membership_strength_figure_name,
    build_corpus_summary_from_audit,
    build_corpus_summary_table,
    build_knee_selection_table,
    build_pareto_front_summary_table,
    build_retained_factors_summary_table,
    build_seed_sensitivity_summary_all_roles,
    build_selected_configuration_table,
    coalesce_text,
    display_manuscript_table,
    sanitize_label_text,
    plot_factor_resampling_multi_panel,
    plot_factor_resampling_reproducibility,
    plot_membership_strength_by_factor,
    plot_umap_seed_sensitivity_all_roles,
)
from umap_datamapplot import build_llm_label_lookup, build_topic_label_array, plot_role_topic_datamap

config = load_run_config()
config.setdefault("topics", {})["top_sentences"] = max(
    int(config.get("topics", {}).get("top_sentences", 5)),
    N_REPRESENTATIVE_SENTENCES,
)
manifest = read_json("theme_discovery_manifest.json", {})
parallel = read_json("parallel_runtime.json", {})

(RUN_DIR / "tables").mkdir(parents=True, exist_ok=True)
summary_path = RUN_DIR / "audit_input_summary.csv"
units_preview, live_summary = load_units(config)
live_corpus = build_corpus_summary_table(units_preview)
expected_n = None
try:
    expected_n = int(
        (config.get("expected_inventory") or {})
        .get(str(config.get("data", {}).get("dataset_id", "")).lower(), {})
        .get("n_accidents")
    )
except (TypeError, ValueError):
    expected_n = None

if summary_path.is_file():
    audit_summary = pd.read_csv(summary_path)
    corpus_summary = build_corpus_summary_from_audit(audit_summary)
    cached_n = None
    if "metric" in audit_summary.columns:
        hit = audit_summary.loc[audit_summary["metric"].astype(str).eq("n_accidents"), "value"]
        if not hit.empty:
            cached_n = int(float(hit.iloc[0]))
    live_n = int(units_preview["_accident_id"].nunique())
    if cached_n is not None and cached_n != live_n:
        print(
            f"WARNING: audit_input_summary.csv has n_accidents={cached_n} but "
            f"current units file has {live_n}. Using live units for corpus summary."
        )
        corpus_summary = live_corpus
        audit_summary = live_summary
        if expected_n is not None and live_n == expected_n:
            # refresh stale audit so later cells stay consistent
            emb_hit = pd.read_csv(summary_path)
            emb_rows = emb_hit.loc[emb_hit["metric"].astype(str).eq("embedding_dimension")]
            refresh = live_summary.copy()
            if not emb_rows.empty:
                refresh = pd.concat([refresh, emb_rows], ignore_index=True)
            refresh.to_csv(summary_path, index=False)
            print("Refreshed stale audit_input_summary.csv from current units.")
elif expected_n is not None and int(units_preview["_accident_id"].nunique()) != expected_n:
    print(
        f"WARNING: units n_accidents={units_preview['_accident_id'].nunique()} "
        f"!= expected_inventory ({expected_n})."
    )
    corpus_summary = live_corpus
    audit_summary = live_summary
else:
    corpus_summary = live_corpus
    audit_summary = live_summary
corpus_summary.to_csv(RUN_DIR / "tables" / "corpus_summary_by_role.csv", index=False)
display_manuscript_table(corpus_summary)
        """
    ),
    markdown(
        """
## 2. Pareto-optimal configurations (raw objectives)

Configurations on the role-specific Pareto front, sorted by increasing DBCV.
Candidate IDs (`A0-P1`, …) are reporting aliases ordered on the front; hyperparameters
and raw $S_R$/DBCV values come from the discovery run. Objectives are shown with
four decimal places so that near-ties on the front remain distinguishable.
        """
    ),
    code(
        """
pareto_front_tables = []
knee_selection_tables = []
selected_rows = []
for role in ROLES:
    selection_path = RUN_DIR / "discovery" / role / "selection_table.csv"
    if not selection_path.is_file():
        print(f"Missing selection_table.csv for {role}")
        continue
    selection_table = pd.read_csv(selection_path)
    pareto_table = build_pareto_front_summary_table(selection_table, role=role)
    knee_table = build_knee_selection_table(selection_table, role=role)
    selected_table = build_selected_configuration_table(selection_table, role=role)
    pareto_front_tables.append(pareto_table.drop(columns=["_configuration_id", "_candidate_rank"], errors="ignore"))
    knee_selection_tables.append(knee_table)
    if not selected_table.empty:
        selected_rows.append(selected_table)

if pareto_front_tables:
    pareto_all = pd.concat(pareto_front_tables, ignore_index=True)
    pareto_all.to_csv(RUN_DIR / "tables" / "pareto_front_summary_all_roles.csv", index=False)
    print("### Pareto front — all roles")
    display_manuscript_table(pareto_all, columns=PARETO_TABLE_COLUMNS)
else:
    print("No Pareto tables available.")
        """
    ),
    markdown(
        """
## 3. Geometric knee selection (normalized Pareto objectives)

Normalized scores are computed on the Pareto set only. $d_K$ is the signed
perpendicular distance toward the ideal point; **Selected = Yes** marks the
geometric knee configuration retained for the role.
        """
    ),
    code(
        """
for role in ROLES:
    selection_path = RUN_DIR / "discovery" / role / "selection_table.csv"
    if not selection_path.is_file():
        continue
    knee_table = build_knee_selection_table(pd.read_csv(selection_path), role=role)
    knee_selection_tables.append(knee_table)

if knee_selection_tables:
    knee_all = pd.concat(knee_selection_tables, ignore_index=True)
    knee_all.to_csv(RUN_DIR / "tables" / "knee_selection_all_roles.csv", index=False)
    print("### Geometric knee selection — all roles")
    display_manuscript_table(knee_all, columns=KNEE_TABLE_COLUMNS)

selected_path = RUN_DIR / "selected_configurations.csv"
if selected_path.is_file():
    selected = pd.read_csv(selected_path)
    display_cols = [
        c for c in [
            "role", "configuration_id", "selection_rule",
            "stability", "dbcv_umap",
            "stability_normalized", "dbcv_normalized", "knee_distance",
            "n_clusters", "noise_fraction",
        ]
        if c in selected.columns
    ]
    display_manuscript_table(selected[display_cols] if display_cols else selected)
else:
    raise FileNotFoundError("selected_configurations.csv manquant — lancer le job discovery (stage select/all).")
        """
    ),
    markdown(
        """
## 4. Pareto figures

**Figure 4.2 (raw, principal)** — les 36 configurations candidates dans $(\\mathrm{DBCV}, S_R)$,
avec le front de Pareto, la configuration retenue (étoile, *Selected configuration*) et les candidats dominés (gris).
Panneaux : **(a) A0 – Work context**, **(b) A1 – Adverse condition**, **(c) B – Event/deviation**,
**(d) C – Consequence**. Les limites des axes sont **spécifiques à chaque rôle** : ne pas comparer
visuellement les distances entre panneaux.

**Figure 4.2 (normalisé, complément)** — espace normalisé, droite de référence et configuration
retenue (*Selected configuration*), avec les mêmes titres de panneaux. Seuls les rôles dont le front
de Pareto comporte **plus d'une** configuration sont affichés (typiquement A0 et A1). Les rôles B et C
sont absents lorsque leur front se réduit à une seule configuration, retenue directement sans calcul de knee.
        """
    ),
    code(
        """
from scenario_pipeline import write_pareto_normalized_knee_figure, write_stability_landscape_figure

(RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)
selection_tables = {}
for role in ROLES:
    selection_path = RUN_DIR / "discovery" / role / "selection_table.csv"
    if selection_path.is_file():
        selection_tables[role] = pd.read_csv(selection_path)

figure_path = RUN_DIR / "figures" / "stability_landscape_all_roles.png"
normalized_path = RUN_DIR / "figures" / "pareto_normalized_knee_all_roles.png"

if selection_tables:
    write_stability_landscape_figure(selection_tables, RUN_DIR / "figures")
    write_pareto_normalized_knee_figure(selection_tables, RUN_DIR / "figures")
    print("Regenerated:", figure_path.name, "and", normalized_path.name)

if figure_path.is_file():
    display(Image(filename=str(figure_path)))
else:
    print("Missing:", figure_path)

if normalized_path.is_file():
    display(Image(filename=str(normalized_path)))
else:
    print("Missing:", normalized_path)
        """
    ),
    markdown("## 5. Run manifest and technical diagnostics"),
    code(
        """
display(pd.DataFrame([{
    "dataset": manifest.get("dataset_id", config.get("data", {}).get("dataset_id")),
    "selection_metric": manifest.get("selection_metric", "pareto_geometric_knee"),
    "n_workers": parallel.get("n_workers", manifest.get("n_workers")),
    "slurm_cpus_per_task": parallel.get("slurm_cpus_per_task"),
    "backend": parallel.get("backend"),
    "inner_umap_n_jobs": parallel.get("inner_umap_n_jobs"),
}]))
if summary_path.is_file():
    display(audit_summary)
        """
    ),
    markdown("## 6. Frozen themes and audit dictionary"),
    code(
        """
from scenario_pipeline import PartitionResult, PreparedData, build_topic_dictionary, load_embeddings, load_selected_configurations, load_units

PARTITION_SELECTION = load_selected_configurations(RUN_DIR)
PARTITION_SELECTIONS = {role: [PARTITION_SELECTION[role]] for role in ROLES}
missing_selection_roles = [role for role in ROLES if role not in PARTITION_SELECTION]
if missing_selection_roles:
    raise ValueError(
        "selected_configurations.csv incomplet pour : "
        + ", ".join(missing_selection_roles)
    )
display(pd.read_csv(RUN_DIR / "selected_configurations.csv"))
print("Frozen partitions:", PARTITION_SELECTION)

units, _ = load_units(config)
embedding_cache = RUN_DIR / "embeddings" / "embeddings_encoded.npy"
embeddings = np.load(embedding_cache) if embedding_cache.is_file() else load_embeddings(config, units, RUN_DIR / "embeddings")
if len(embeddings) != len(units):
    raise ValueError(f"Embedding/unit mismatch: {len(embeddings)} embeddings for {len(units)} units")

manual_results = {}
partition_frames = {}
for role in ROLES:
    configuration_id = str(PARTITION_SELECTION[role])
    selected_dir = RUN_DIR / "discovery" / role / "selected"
    labels_path = selected_dir / "labels.npy"
    strength_path = selected_dir / "membership_strength.npy"
    if not labels_path.is_file() or not strength_path.is_file():
        labels_path = RUN_DIR / "discovery" / role / "candidate_partitions" / f"{configuration_id}_labels.npy"
        strength_path = RUN_DIR / "discovery" / role / "candidate_partitions" / f"{configuration_id}_membership_strength.npy"
    if not labels_path.is_file() or not strength_path.is_file():
        raise FileNotFoundError(f"Missing selected/candidate artifacts for {role}/{configuration_id}")
    role_units = units[units["_role"].astype(str).eq(role)].reset_index(drop=True)
    assignments = role_units[["_accident_id", "_fact_id", "_text"]].copy()
    assignments.rename(
        columns={"_accident_id": "accident_id", "_fact_id": "fact_id", "_text": "sentence"},
        inplace=True,
    )
    assignments["role"] = role
    labels = np.load(labels_path).astype(int)
    strengths = np.load(strength_path).astype(float)
    if len(assignments) != len(labels) or len(labels) != len(strengths):
        raise ValueError(f"Label/assignment mismatch for {role}/{configuration_id}")
    assignments["topic_id"] = [f"{role}_{label:03d}" if label >= 0 else "" for label in labels]
    assignments["membership_strength"] = strengths
    topic_rows = []
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        mask = labels == label
        topic_rows.append({
            "topic_id": f"{role}_{label:03d}",
            "role": role,
            "n_units": int(mask.sum()),
            "n_accidents": int(assignments.loc[mask, "accident_id"].nunique()),
            "selected_configuration_id": configuration_id,
        })
    manual_results[role] = PartitionResult(
        role=role,
        assignments=assignments,
        topics=pd.DataFrame(topic_rows),
        edges=pd.DataFrame(),
        replications=pd.DataFrame(),
    )

    candidate_id = configuration_id
    candidate_frame = assignments[["accident_id", "fact_id", "sentence"]].copy()
    candidate_frame["Topic"] = labels
    candidate_frame["membership_strength"] = strengths
    candidate_frame["partition_id"] = candidate_id
    candidate_frame["role"] = role
    partition_frames[(role, candidate_id)] = candidate_frame

prepared_manual = PreparedData(units=units, embeddings=embeddings, input_summary=pd.DataFrame())
topics = build_topic_dictionary(prepared_manual, manual_results, config, RUN_DIR / "topics_manual")
topics["selected_configuration_id"] = topics["role"].map(PARTITION_SELECTION)
representative_rows = []
for (role, configuration_id), frame in partition_frames.items():
    for topic in sorted(int(value) for value in frame["Topic"].unique() if int(value) >= 0):
        subset = frame[frame["Topic"].eq(topic)].sort_values("membership_strength", ascending=False).head(N_REPRESENTATIVE_SENTENCES)
        representative_rows.extend(
            {
                "role": role,
                "configuration_id": configuration_id,
                "topic_id": f"{role}_{topic:03d}",
                "fact_id": row["fact_id"],
                "accident_id": row["accident_id"],
                "sentence": row["sentence"],
                "membership_strength": row["membership_strength"],
            }
            for _, row in subset.iterrows()
        )
representatives_by_membership = pd.DataFrame(representative_rows)
(RUN_DIR / "topics_manual").mkdir(parents=True, exist_ok=True)
representatives_by_membership.to_csv(RUN_DIR / "topics_manual" / "representatives_by_membership.csv", index=False)
primary_representatives = representatives_by_membership[
    representatives_by_membership["configuration_id"].astype(str).eq(representatives_by_membership["role"].map(PARTITION_SELECTION).astype(str))
]
representative_lookup = primary_representatives.groupby("topic_id")["sentence"].apply(lambda values: " || ".join(values.astype(str).tolist())).to_dict()
topics["representative_sentences"] = topics["topic_id"].map(representative_lookup).fillna(topics.get("representative_sentences", ""))
topics.to_csv(RUN_DIR / "topics_manual" / "topic_dictionary.csv", index=False)

# Build a catalog for every selected partition. The configuration ID is
# part of the topic identity for labels and plots because topic numbers can be
# reused by different partitions.
topic_catalog_rows = []
primary_topic_lookup = topics.set_index("topic_id").to_dict("index") if not topics.empty else {}
for (role, configuration_id), frame in partition_frames.items():
    for topic in sorted(int(value) for value in frame["Topic"].unique() if int(value) >= 0):
        subset = frame[frame["Topic"].eq(topic)].sort_values("membership_strength", ascending=False)
        topic_id = f"{role}_{topic:03d}"
        primary_row = primary_topic_lookup.get(topic_id, {}) if str(configuration_id) == str(PARTITION_SELECTION[role]) else {}
        topic_catalog_rows.append({
            "topic_id": topic_id,
            "role": role,
            "configuration_id": str(configuration_id),
            "n_units": int(len(subset)),
            "n_accidents": int(subset["accident_id"].nunique()),
            "top_terms": str(primary_row.get("top_terms", "")),
            "label": str(primary_row.get("label", "")),
            "representative_sentences": " || ".join(subset["sentence"].astype(str).head(N_REPRESENTATIVE_SENTENCES).tolist()),
        })
topic_catalog = pd.DataFrame(topic_catalog_rows)
topic_catalog.to_csv(RUN_DIR / "topics_manual" / "topic_dictionary_all_selected.csv", index=False)
selected_topics = topics.copy()
display(topics)

for role in ROLES:
    assignments = manual_results[role].assignments
    print(role, "assignments:", assignments.shape, "configuration:", PARTITION_SELECTION[role])
    display(assignments.head())
        """
    ),
    markdown(
        """
## 7. Membership-strength distributions (appendix)

Saved under ``figs_ch4/appendix/`` as ``membership_strength_A0.png``, …,
``membership_strength_C.png`` (no ``factor`` in the filename). The main text
relies on the **Median membership strength** column in Table 4.5 instead of
displaying these boxplots in the body.
        """
    ),
    code(
        """
for role in ROLES:
    configuration_id = str(PARTITION_SELECTION[role])
    frame = partition_frames[(role, configuration_id)]
    output_path = appendix_figure_path(RUN_DIR, membership_strength_figure_name(role))
    plot_membership_strength_by_factor(
        frame,
        role=role,
        configuration_id=configuration_id,
        output_path=output_path,
        show_unit_annotations=False,
    )
    legacy_path = RUN_DIR / "figures" / f"membership_strength_factors_{role}.png"
    if legacy_path.is_file():
        legacy_path.unlink()
        print(f"Removed legacy figure: {{legacy_path.name}}")
    noise_count = int(frame["Topic"].astype(int).lt(0).sum())
    print(f"Appendix saved: {output_path.name}", end="")
    if noise_count:
        print(f"  ({noise_count} noise units excluded)")
    else:
        print()
        """
    ),
    markdown(
        """
## 8. Factor-level accident-resampling reproducibility

**Figure 4.4** — Factor-level accident-resampling reproducibility for A0 (separate panel)
and for A1, B and C (combined three-panel figure). Each row represents one retained
factor and shows the distribution of its best-match Jaccard similarity across
accident-level resamples. Factors are ordered by decreasing mean reproducibility
$S_{cg}$ within each role (highest at the top). The red dot marks $S_{cg}$; exact
values appear in the retained-factors table.
        """
    ),
    code(
        """
from scenario_pipeline import write_factor_resampling_manuscript_figures

(RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)
theme_by_role = {}
for role in ROLES:
    theme_path = RUN_DIR / "discovery" / role / "stability_theme.csv"
    if theme_path.is_file():
        theme_by_role[role] = pd.read_csv(theme_path)
    else:
        print(f"Missing stability_theme.csv for {role}")

if theme_by_role:
    write_factor_resampling_manuscript_figures(theme_by_role, PARTITION_SELECTION, RUN_DIR / "figures")
    print("Wrote:", RUN_DIR / "figures" / FIGURE_FACTOR_RESAMPLING_A0)
    print("Wrote:", RUN_DIR / "figures" / FIGURE_FACTOR_RESAMPLING_A1_B_C)

a0_path = RUN_DIR / "figures" / FIGURE_FACTOR_RESAMPLING_A0
combo_path = RUN_DIR / "figures" / FIGURE_FACTOR_RESAMPLING_A1_B_C
if a0_path.is_file():
    display(Image(filename=str(a0_path)))
else:
    print("Missing:", a0_path)

if combo_path.is_file():
    display(Image(filename=str(combo_path)))
else:
    print(
        "Missing:", combo_path,
        "\\nNeed stability_theme.csv for A1/B/C and selected configuration rows.",
        {role: PARTITION_SELECTION.get(role) for role in ("A1", "B", "C")},
    )

resampling_summaries = []
for role in ROLES:
    if role not in theme_by_role:
        continue
    configuration_id = str(PARTITION_SELECTION[role])
    selected_theme = theme_by_role[role][
        theme_by_role[role]["configuration_id"].astype(str).eq(configuration_id)
    ].copy()
    if selected_theme.empty:
        print(f"No resampling rows for selected configuration {role}/{configuration_id}")
        continue
    factor_summary = (
        selected_theme.groupby("cluster_label", as_index=False)
        .agg(
            S_cg=("theme_stability", "first"),
            B_cg_over_B=("observability", "first"),
            n_replicates=("repetition", "nunique"),
        )
        .sort_values("S_cg", ascending=False)
    )
    factor_summary.insert(0, "role", role)
    resampling_summaries.append(factor_summary)

if resampling_summaries:
    pd.concat(resampling_summaries, ignore_index=True).to_csv(
        RUN_DIR / "tables" / "factor_resampling_summary_all_roles.csv", index=False
    )
        """
    ),
    markdown(
        """
## 9. Sensitivity of the selected partitions to UMAP random seeds

**Figure 4.5** — ``umap_seed_sensitivity_all_roles.png``: for each role, mean best-match
Jaccard between the alternative-seed partition and the reference partition obtained with
the primary seed $s_0$ (configuration $c_r^{\\star}$ held fixed). Panel titles:
**(a) A0 – Work context**, **(b) A1 – Adverse condition**, **(c) B – Event/deviation**,
**(d) C – Consequence**.

The primary seed $s_0$ remains the unique reference; alternative seeds 1–10 are **not**
competing configurations and are never selected. $K$, DBCV and the unassigned fraction are
reported in the complementary table (reference $K$ at $s_0$, ranges over alternative seeds).
        """
    ),
    code(
        """
from manuscript_reporting import build_seed_sensitivity_summary_all_roles

seed_figure_path = RUN_DIR / "figures" / FIGURE_UMAP_SEED_SENSITIVITY
seed_figure = plot_umap_seed_sensitivity_all_roles(RUN_DIR, output_path=seed_figure_path)
if seed_figure is not None:
    display(Image(filename=str(seed_figure_path)))
    plt.close(seed_figure)
else:
    print("Missing:", seed_figure_path.name)

combined = build_seed_sensitivity_summary_all_roles(RUN_DIR)
if not combined.empty:
    display_cols = [
        "Role",
        "Reference_K",
        "K_range",
        "DBCV_range",
        "Unassigned_fraction_range",
        "Mean_Jaccard_range",
    ]
    table = combined[[c for c in display_cols if c in combined.columns]].copy()
    (RUN_DIR / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(RUN_DIR / "tables" / "seed_sensitivity_summary_all_roles.csv", index=False)
    display(table)
    for _, row in table.iterrows():
        print(
            f"{row['Role']}: mean best-match Jaccard ranged from "
            f"{row['Mean_Jaccard_range']} across alternative seeds "
            f"(reference K={row['Reference_K']})."
        )
else:
    print("No seed sensitivity summary available.")
        """
    ),
    markdown(
        """
## 10. Natural-language theme labels with the OpenAI API

The prompts are defined in the first code cell, separately for A0, A1, B and
C. For each role, the API receives the topic identifiers, top words and the
configured number of representative sentences. The output is requested as
JSON and saved locally. The API key is read only from the environment variable
OPENAI_API_KEY; it is never read from the run directory or written to the
notebook. In PowerShell, define it before launching Jupyter with
`$env:OPENAI_API_KEY = "votre-cle"`.
If the key, package or request is unavailable, the notebook keeps the
top-word label and continues without stopping the analysis.

Set `LLM_FORCE_REFRESH = True` in the first code cell to delete
`topics_manual/llm_theme_labels.csv` and relabel every topic (ignore cache).

Model defaults follow the annotation pipeline: `gpt-5.6-luna` with
`reasoning_effort` and `max_completion_tokens` (no custom temperature).

Labels are requested in **English** (`LLM_OUTPUT_LANGUAGE = "English"`) so that
Table 4.5 and the UMAP figures use the same wording. Set `LLM_FORCE_REFRESH = True`
to relabel existing topics after changing the language or prompts.
        """
    ),
    code(
        """
def topic_representatives(row, n_sentences):
    values = []
    for column in ("central_sentence", "representative_sentences", "boundary_sentences"):
        value = str(row.get(column, ""))
        if value and value.lower() != "nan":
            values.extend(part.strip() for part in value.split(" || ") if part.strip())
    unique_values = list(dict.fromkeys(values))
    return unique_values[:int(n_sentences)]


def topic_records_for_role(role):
    role_topics = topic_catalog[topic_catalog["role"].astype(str).eq(role)].copy()
    if not role_topics.empty:
        role_topics = role_topics.sort_values(["configuration_id", "n_accidents"], ascending=[True, False])
    elif FALLBACK_TOPICS_PER_ROLE is not None:
        role_topics = role_topics.sort_values("n_accidents", ascending=False).head(int(FALLBACK_TOPICS_PER_ROLE))
    if MAX_TOPICS_PER_ROLE_FOR_LLM is not None:
        role_topics = role_topics.groupby("configuration_id", group_keys=False).head(int(MAX_TOPICS_PER_ROLE_FOR_LLM))
    records = []
    for _, row in role_topics.iterrows():
        records.append({
            "topic_id": str(row["topic_id"]),
            "configuration_id": str(row["configuration_id"]),
            "top_words": str(row.get("top_terms", row.get("label", ""))),
            "representative_sentences": topic_representatives(row, N_REPRESENTATIVE_SENTENCES),
        })
    return records


from llm_labeling import (
    complete_theme_label_json,
    extract_theme_item,
    is_valid_llm_cache_row,
    normalize_llm_fields,
    parse_llm_payload,
)
from scgm_text.openai_theme_labels import _get_client, load_openai_dotenv


def request_role_labels(client, role, records):
    prompt = ROLE_PROMPTS[role]
    instruction = f'''
Return a single JSON object with a `themes` array containing one element per `topic_id`.
Write the label, description and evidence in {LLM_OUTPUT_LANGUAGE}.

Rules for `label`:
- 4 to 10 words, nominal phrase suitable for a figure legend.
- Must name a specific factual motif visible in the examples (object, action,
  equipment, location, mechanism or injury depending on the role).
- Forbidden: generic labels that do not discriminate themes (e.g. only the role name,
  "work context", "adverse factor", "event", "consequence", "accident", "risk").
- If examples are heterogeneous, choose the most specific wording supported by at
  least two examples; do not summarize the whole A0/A1/B/C role.
- Do not invent information absent from the keywords and sentences provided.

`description`: one sentence explaining the motif. `evidence`: 2 to 4 short text
fragments from the examples (quotes or minimal paraphrases).

Role-specific guidance:
{prompt}

Topics to analyse:
{json.dumps(records, ensure_ascii=False, indent=2)}
'''
    response = complete_theme_label_json(
        client,
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful occupational-accident analyst. Return valid JSON only."},
            {"role": "user", "content": instruction},
        ],
        reasoning_effort=OPENAI_REASONING_EFFORT,
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )
    return parse_llm_payload(response)


llm_cache_path = RUN_DIR / "topics_manual" / "llm_theme_labels.csv"
if LLM_FORCE_REFRESH and llm_cache_path.is_file():
    llm_cache_path.unlink()
    print("LLM cache cleared:", llm_cache_path)
llm_cache = pd.read_csv(llm_cache_path) if llm_cache_path.is_file() else pd.DataFrame()
if not llm_cache.empty and "configuration_id" not in llm_cache.columns:
    # Backward compatibility with the previous cache format, which contained
    # only labels for the primary partition of each role.
    llm_cache["configuration_id"] = llm_cache["role"].map(PARTITION_SELECTION)
for column in ("topic_id", "role", "configuration_id", "llm_label", "llm_description", "llm_evidence"):
    if column not in llm_cache.columns:
        llm_cache[column] = ""
for column in ("llm_label", "llm_description", "llm_evidence"):
    llm_cache[column] = llm_cache[column].map(sanitize_label_text)
llm_cache["cache_key"] = (
    llm_cache["role"].astype(str) + "::" +
    llm_cache["configuration_id"].astype(str) + "::" +
    llm_cache["topic_id"].astype(str)
)
llm_cache = llm_cache.drop_duplicates("cache_key", keep="last")
cached_by_key = llm_cache.set_index("cache_key").to_dict("index") if not llm_cache.empty else {}
llm_rows = []
new_llm_rows = []
llm_status = {}
if not topic_catalog.empty and OPENAI_ENABLED and os.environ.get("OPENAI_API_KEY"):
    try:
        load_openai_dotenv()
        client = _get_client()
        for role in ROLES:
            records = topic_records_for_role(role)
            if not records:
                continue
            failed_count = 0
            missing_count = 0
            cached_count = 0
            for record in tqdm(records, desc=f"OpenAI labels {role}", unit="topic"):
                cache_key = f"{role}::{record['configuration_id']}::{record['topic_id']}"
                cached = cached_by_key.get(cache_key)
                if cached and is_valid_llm_cache_row(cached):
                    cached_count += 1
                    llm_rows.append({
                        "topic_id": record["topic_id"],
                        "role": role,
                        "configuration_id": record["configuration_id"],
                        **normalize_llm_fields(cached),
                    })
                    continue
                try:
                    response_items = request_role_labels(client, role, [record])
                    item = extract_theme_item(response_items, record)
                    fields = normalize_llm_fields(item)
                    if not fields["llm_label"]:
                        # Retry once with a minimal prompt when the model omits the label.
                        retry_instruction = f'''
Return JSON {{"themes": [{{"topic_id": "{record["topic_id"]}", "label": "...", "description": "...", "evidence": "..."}}]}}.
The label field must contain 4 to 10 words in {LLM_OUTPUT_LANGUAGE}, factual and specific.
topic_id={record["topic_id"]}
top_words={json.dumps(record.get("top_words", ""), ensure_ascii=False)}
examples={json.dumps(record.get("representative_sentences", []), ensure_ascii=False)}
'''
                        retry_text = complete_theme_label_json(
                            client,
                            model=OPENAI_MODEL,
                            messages=[
                                {"role": "system", "content": "Retournez uniquement un JSON valide."},
                                {"role": "user", "content": retry_instruction},
                            ],
                            reasoning_effort=OPENAI_REASONING_EFFORT,
                            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                        )
                        retry_items = parse_llm_payload(retry_text)
                        fields = normalize_llm_fields(extract_theme_item(retry_items, record))
                    if not fields["llm_label"]:
                        missing_count += 1
                        warnings.warn(
                            f"OpenAI returned no usable label for {role}/{record['topic_id']} "
                            f"(response items={len(response_items)})"
                        )
                    llm_rows.append({
                        "topic_id": record["topic_id"],
                        "role": role,
                        "configuration_id": record["configuration_id"],
                        **fields,
                    })
                except Exception as error:
                    failed_count += 1
                    new_llm_rows.append({
                        "topic_id": record["topic_id"],
                        "role": role,
                        "configuration_id": record["configuration_id"],
                        "llm_label": "",
                        "llm_description": "",
                        "llm_evidence": "",
                    })
                    warnings.warn(f"OpenAI labelling failed for {role}/{record['topic_id']}: {error}")
            if failed_count:
                llm_status[role] = f"completed_with_failures:{failed_count}"
            elif missing_count:
                llm_status[role] = f"completed_with_missing_topics:{missing_count}"
            elif cached_count == len(records):
                llm_status[role] = f"loaded_from_cache:{cached_count}"
            elif cached_count:
                llm_status[role] = f"completed_with_cache_hits:{cached_count}"
            else:
                llm_status[role] = "completed"
    except Exception as error:
        llm_status["global"] = f"unavailable: {error}"
        warnings.warn(f"OpenAI labelling unavailable: {error}")
else:
    llm_status["global"] = "skipped: OPENAI_ENABLED is false or OPENAI_API_KEY is missing"

if not llm_rows and not new_llm_rows and not llm_cache.empty:
    llm_rows = llm_cache.to_dict("records")
llm_labels = pd.DataFrame(
    llm_rows + new_llm_rows,
    columns=["topic_id", "role", "configuration_id", "llm_label", "llm_description", "llm_evidence"],
)
if not llm_labels.empty:
    for column in ("llm_label", "llm_description", "llm_evidence"):
        llm_labels[column] = llm_labels[column].map(sanitize_label_text)
if not llm_labels.empty:
    llm_labels = llm_labels[llm_labels["llm_label"].map(sanitize_label_text).astype(bool)].copy()
    llm_labels["cache_key"] = (
        llm_labels["role"].astype(str) + "::" +
        llm_labels["configuration_id"].astype(str) + "::" +
        llm_labels["topic_id"].astype(str)
    )
    llm_labels = llm_labels.drop_duplicates("cache_key", keep="last").drop(columns=["cache_key"])
(RUN_DIR / "topics_manual").mkdir(parents=True, exist_ok=True)
llm_labels.to_csv(RUN_DIR / "topics_manual" / "llm_theme_labels.csv", index=False)
(RUN_DIR / "topics_manual" / "llm_theme_labels_status.json").write_text(json.dumps(llm_status, indent=2, ensure_ascii=False), encoding="utf-8")

theme_labels = topic_catalog.copy()
if not llm_labels.empty:
    theme_labels = theme_labels.merge(llm_labels, on=["topic_id", "role", "configuration_id"], how="left")
else:
    theme_labels["llm_label"] = ""
    theme_labels["llm_description"] = ""
    theme_labels["llm_evidence"] = ""
for column in ("llm_label", "llm_description", "llm_evidence"):
    theme_labels[column] = theme_labels[column].map(sanitize_label_text)
for column in ("label", "top_terms"):
    if column not in theme_labels.columns:
        theme_labels[column] = ""
    theme_labels[column] = theme_labels[column].map(sanitize_label_text)
theme_labels["plot_label"] = theme_labels.apply(
    lambda row: coalesce_text(row.get("llm_label"), default=str(row.get("topic_id", ""))),
    axis=1,
)
theme_labels.to_csv(RUN_DIR / "topics_manual" / "topic_dictionary_with_llm_labels.csv", index=False)
display(theme_labels[[column for column in ["topic_id", "role", "plot_label", "llm_description", "top_terms", "representative_sentences"] if column in theme_labels.columns]])
        """
    ),
    markdown(
        """
## 11. Summary of retained factors (Table 4.5)

Illustrative summary after LLM labelling: units, contributing accidents, resampling
stability $S_{cg}$, median membership strength, and illustrative content.
        """
    ),
    code(
        """
factor_stability_by_role = []
for role in ROLES:
    theme_path = RUN_DIR / "discovery" / role / "stability_theme.csv"
    if not theme_path.is_file():
        continue
    theme_stability = pd.read_csv(theme_path)
    configuration_id = str(PARTITION_SELECTION[role])
    selected = theme_stability[
        theme_stability["configuration_id"].astype(str).eq(configuration_id)
    ].drop_duplicates("cluster_label")
    if not selected.empty:
        factor_stability_by_role.append(selected)
factor_stability = pd.concat(factor_stability_by_role, ignore_index=True) if factor_stability_by_role else pd.DataFrame()

primary_theme_labels = theme_labels[
    theme_labels["configuration_id"].astype(str).eq(theme_labels["role"].map(PARTITION_SELECTION).astype(str))
]
retained_factors_table = build_retained_factors_summary_table(
    primary_theme_labels,
    partition_frames,
    factor_stability=factor_stability if not factor_stability.empty else None,
    partition_selection=PARTITION_SELECTION,
)
retained_factors_table.to_csv(RUN_DIR / "tables" / "retained_factors_summary.csv", index=False)
display_manuscript_table(retained_factors_table, columns=RETAINED_FACTOR_COLUMNS)
        """
    ),
    markdown(
        """
## 12. Narrative text coloured by topic

Each **selected** partition is inspected on the source accident narrative. Colours
match the retained factors; legend and inline highlights use **LLM labels only**
(fallback: neutral topic id if labelling was skipped). Noise (`-1`) remains uncoloured.
        """
    ),
    code(
        """
# Choose one accident independently for each role. None displays the first
# accident in the corresponding role dataframe by default.
ACCIDENT_ID_A0 = None
ACCIDENT_ID_A1 = None
ACCIDENT_ID_B = None
ACCIDENT_ID_C = None
ACCIDENT_IDS_TO_DISPLAY = {
    "A0": ACCIDENT_ID_A0,
    "A1": ACCIDENT_ID_A1,
    "B": ACCIDENT_ID_B,
    "C": ACCIDENT_ID_C,
}
display(pd.DataFrame({"role": list(ACCIDENT_IDS_TO_DISPLAY), "selected_accident_id": list(ACCIDENT_IDS_TO_DISPLAY.values())}))
        """
    ),
    code(
        """
import html as html_lib
from IPython.display import HTML


def show_colored_text_inline_v5(
    frame,
    accident_id,
    *,
    text_col="sentence",
    topic_col="Topic",
    label_col="topic_label",
    show_legend=True,
    legend_max_items=80,
    cmap_name="tab20",
    sep=" ",
    font_size_px=11,
    line_height=1.55,
    highlight_style="border",
    pastel_strength=0.82,
    border_width_px=2,
    container_max_width_px=1100,
    legend_font_size_px=9,
    legend_title="Thèmes",
    legend_max_height_px=260,
    legend_item_min_width_px=220,
):
    data = frame[frame["accident_id"].astype(str).eq(str(accident_id))].copy()
    if data.empty:
        print(f"Aucune donnée pour accident_id={accident_id}")
        return
    topics_in_text = data[topic_col].fillna(-1).astype(int).to_numpy()
    topic_ids = sorted(topic for topic in np.unique(topics_in_text) if topic != -1)
    cmap = plt.get_cmap(cmap_name, max(1, len(topic_ids)))

    def rgba_to_hex(rgba):
        return "#{:02x}{:02x}{:02x}".format(*(int(channel * 255) for channel in rgba[:3]))

    def hex_to_rgb(value):
        value = value.lstrip("#")
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))

    def pastel(value):
        red, green, blue = hex_to_rgb(value)
        return "rgb({},{},{})".format(
            int(red + (255 - red) * pastel_strength),
            int(green + (255 - green) * pastel_strength),
            int(blue + (255 - blue) * pastel_strength),
        )

    colors = {topic: rgba_to_hex(cmap(index)) for index, topic in enumerate(topic_ids)}
    backgrounds = {topic: pastel(colors[topic]) for topic in topic_ids}
    labels = {}
    for topic in topic_ids:
        subset = data[data[topic_col].fillna(-1).astype(int).eq(topic)]
        value = subset[label_col].iloc[0] if label_col in subset.columns else ""
        labels[topic] = str(value).strip() if str(value).strip() else f"Topic {topic}"

    parts = []
    for _, row in data.iterrows():
        sentence = str(row.get(text_col, "")).strip()
        if not sentence:
            continue
        topic = int(row.get(topic_col, -1))
        escaped = html_lib.escape(sentence)
        if topic < 0:
            parts.append(f"<span style='color:#111'>{escaped}</span>")
            continue
        color = colors.get(topic, "#999999")
        background = backgrounds.get(topic, "#f5f5f5")
        if highlight_style == "bar":
            style = f"border-left:{border_width_px}px solid {color};padding:1px 7px;"
        else:
            style = f"background:{background};padding:1px 6px;border-radius:8px;box-shadow:0 0 0 {border_width_px}px {color} inset;"
        parts.append(f"<span title='{html_lib.escape(labels.get(topic, f'Topic {topic}'))}' style='{style}'>{escaped}</span>")

    legend = ""
    if show_legend and topic_ids:
        items = []
        for topic in topic_ids[:legend_max_items]:
            items.append(
                f"<div style='display:flex;align-items:flex-start;gap:8px;padding:5px 7px;border-radius:10px;background:{backgrounds[topic]};min-width:{legend_item_min_width_px}px;'>"
                f"<div style='width:10px;height:10px;background:{colors[topic]};border-radius:3px;margin-top:2px;flex:0 0 auto;'></div>"
                f"<div style='font-size:{legend_font_size_px}px;line-height:1.25;overflow-wrap:anywhere;'>{html_lib.escape(labels[topic])}</div></div>"
            )
        legend = f"<div style='margin-top:10px;padding:10px 12px;border:1px solid #eee;border-radius:14px;background:#fbfbfb;'><div style='font-weight:800;margin-bottom:8px;'>{html_lib.escape(legend_title)}</div><div style='display:flex;flex-wrap:wrap;gap:8px;max-height:{legend_max_height_px}px;overflow:auto;'>{''.join(items)}</div></div>"

    display(HTML(f"<div style='max-width:{container_max_width_px}px;'><div style='font-size:{font_size_px}px;line-height:{line_height};white-space:pre-wrap;border:1px solid #e0e0e0;padding:12px;border-radius:14px;background:#fff;'>{sep.join(parts)}</div>{legend}</div>"))


def show_partition_narratives(role, configuration_id, accident_ids=None):
    frame = partition_frames[(role, str(configuration_id))].copy()
    label_lookup = build_llm_label_lookup(theme_labels, role=role, configuration_id=str(configuration_id))
    frame["topic_label"] = frame["Topic"].map(
        lambda value: label_lookup.get(f"{role}_{int(value):03d}", f"Topic {int(value)}")
        if int(value) >= 0
        else "Bruit / non assigné"
    )
    requested = accident_ids if accident_ids is not None else ACCIDENT_IDS_TO_DISPLAY.get(role)
    if requested is None or requested == "":
        requested = frame["accident_id"].drop_duplicates().head(1).tolist()
    elif isinstance(requested, (str, int)):
        requested = [requested]
    print(f"{role} — accidents disponibles : {frame['accident_id'].drop_duplicates().astype(str).tolist()[:20]}")
    for accident_id in requested:
        display(HTML(f"<h4>{role} — partition {configuration_id} — accident {html_lib.escape(str(accident_id))}</h4>"))
        show_colored_text_inline_v5(frame, accident_id)


for role in ROLES:
    configuration_id = str(PARTITION_SELECTION[role])
    print(f"{role} — partition sélectionnée {configuration_id}")
    show_partition_narratives(role, configuration_id, accident_ids=ACCIDENT_IDS_TO_DISPLAY.get(role))
        """
    ),
    markdown(
        """
## 13. Two-dimensional UMAP topic map (DataMapPlot)

Publication-style 2-D UMAP for **qualitative interpretation only** (not used for
clustering, selection or validation). Rendered with **DataMapPlot**: soft cluster glow,
exterior labels with leader lines, and **English LLM factor names** matching Table 4.5.
One figure per role: ``retained_factors_A0.png``, ``retained_factors_A1.png``,
``retained_factors_B.png``, ``retained_factors_C.png``. Grey points indicate HDBSCAN noise.
        """
    ),
    code(
        """
import umap

assignments = pd.concat(
    [manual_results[role].assignments[["fact_id", "role", "topic_id"]] for role in ROLES],
    ignore_index=True,
).drop_duplicates("fact_id")
assignments["fact_id"] = assignments["fact_id"].astype(str)
unit_index = pd.DataFrame({"fact_id": units["_fact_id"].astype(str), "embedding_index": np.arange(len(units))})
plot_frame = assignments.merge(unit_index, on="fact_id", how="inner")
plot_frame["topic_id"] = plot_frame["topic_id"].fillna("").astype(str)

(RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)


def render_role_umap_panel(role, *, output_path, figsize, tight_crop=False):
    configuration_id = str(PARTITION_SELECTION[role])
    role_units = units[units["_role"].eq(role)].reset_index(drop=True)
    role_embeddings = embeddings[units["_role"].to_numpy() == role]
    if len(role_embeddings) < 3:
        return None
    reducer = umap.UMAP(
        n_neighbors=min(int(UMAP_2D_N_NEIGHBORS), max(2, len(role_embeddings) - 1)),
        n_components=2,
        min_dist=float(UMAP_2D_MIN_DIST),
        metric="cosine",
        random_state=int(UMAP_2D_RANDOM_STATE),
    )
    coordinates = reducer.fit_transform(role_embeddings)
    role_plot = plot_frame[plot_frame["role"].eq(role)].copy()
    role_plot = role_plot.merge(
        pd.DataFrame({"fact_id": role_units["_fact_id"].astype(str), "local_index": np.arange(len(role_units))}),
        on="fact_id",
        how="inner",
    )
    if len(role_plot) > int(MAX_POINTS_PER_ROLE_PLOT):
        role_plot = role_plot.sample(int(MAX_POINTS_PER_ROLE_PLOT), random_state=int(UMAP_2D_RANDOM_STATE))
    coordinate_lookup = dict(enumerate(coordinates))
    role_plot["x"] = role_plot["local_index"].map(lambda index: coordinate_lookup[int(index)][0])
    role_plot["y"] = role_plot["local_index"].map(lambda index: coordinate_lookup[int(index)][1])
    role_plot = role_plot.dropna(subset=["x", "y"])
    retained_ids = set(
        theme_labels.loc[
            theme_labels["role"].astype(str).eq(role)
            & theme_labels["configuration_id"].astype(str).eq(configuration_id),
            "topic_id",
        ].astype(str)
    )
    role_plot["plot_topic_id"] = role_plot["topic_id"].where(role_plot["topic_id"].isin(retained_ids), "")
    label_lookup = build_llm_label_lookup(theme_labels, role=role, configuration_id=configuration_id)
    datamap_labels = build_topic_label_array(
        role_plot["plot_topic_id"],
        label_lookup,
        noise_label=DATAMAP_NOISE_LABEL,
    )
    coords = role_plot[["x", "y"]].to_numpy(dtype=float)
    figure = plot_role_topic_datamap(
        coords,
        datamap_labels,
        output_path=output_path,
        figsize=figsize,
        dpi=float(UMAP_2D_DPI),
        label_wrap_width=int(DATAMAP_LABEL_WRAP_WIDTH),
        label_font_size=float(DATAMAP_LABEL_FONT_SIZE),
        dynamic_label_size=bool(DATAMAP_DYNAMIC_LABEL_SIZE),
        noise_label=DATAMAP_NOISE_LABEL,
        title_fontsize=11.0,
        tight_crop=tight_crop,
    )
    plt.close(figure)
    return output_path

for role in ROLES:
    figure_path = RUN_DIR / "figures" / retained_factors_figure_name(role)
    figsize = UMAP_2D_FIGSIZE if role == "A0" else UMAP_2D_FIGSIZE_COMPACT
    tight_crop = role in {"B", "C"}
    rendered = render_role_umap_panel(
        role,
        output_path=figure_path,
        figsize=figsize,
        tight_crop=tight_crop,
    )
    if rendered is not None and figure_path.is_file():
        print("Wrote:", figure_path.name)
        display(Image(filename=str(figure_path)))
    else:
        print(f"Skipped UMAP map for {role}")
        """
    ),
    markdown("## 14. Export locations"),
    code(
        """
outputs = [
    "tables/corpus_summary_by_role.csv",
    "tables/pareto_front_summary_all_roles.csv",
    "tables/knee_selection_all_roles.csv",
    "tables/factor_resampling_summary_all_roles.csv",
    "tables/seed_sensitivity_summary_all_roles.csv",
    "tables/retained_factors_summary.csv",
    "config_resolved.yaml", "parallel_runtime.json", "selected_configurations.csv",
    "figures/stability_landscape_all_roles.png", "figures/pareto_normalized_knee_all_roles.png",
    "figures/factor_resampling_A0.png", "figures/factor_resampling_A1_B_C.png",
    "figures/umap_seed_sensitivity_all_roles.png",
    "figures/retained_factors_A0.png", "figures/retained_factors_A1.png",
    "figures/retained_factors_B.png", "figures/retained_factors_C.png",
    "figs_ch4/appendix/membership_strength_A0.png",
    "figs_ch4/appendix/membership_strength_A1.png",
    "figs_ch4/appendix/membership_strength_B.png",
    "figs_ch4/appendix/membership_strength_C.png",
    "topics_manual/topic_dictionary.csv",
    "topics_manual/topic_dictionary_all_selected.csv", "topics_manual/representatives_by_membership.csv",
    "topics_manual/llm_theme_labels.csv", "topics_manual/topic_dictionary_with_llm_labels.csv",
    "discovery/A0/candidate_partitions/<configuration_id>_labels.npy",
    "discovery/A0/candidate_partitions/<configuration_id>_membership_strength.npy",
]
display(pd.DataFrame({"path": outputs, "exists": [(RUN_DIR / path).exists() for path in outputs]}))
        """
    ),
]


def build_notebook(dataset_id: str) -> dict:
    return {
        "cells": build_cells(dataset_id),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id in DATASETS:
        path = NOTEBOOK_DIR / f"topic_modeling_results_{dataset_id}.ipynb"
        path.write_text(
            json.dumps(build_notebook(dataset_id), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"Notebook written: {path}")
    # Keep a generic alias pointing at the default corpus for discoverability.
    alias = NOTEBOOK_DIR / "topic_modeling_results.ipynb"
    alias.write_text(
        json.dumps(build_notebook("caou"), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Notebook written: {alias} (alias caou)")


if __name__ == "__main__":
    main()
