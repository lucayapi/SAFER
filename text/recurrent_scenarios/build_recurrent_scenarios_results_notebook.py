"""Build the post-run results notebook for recurrent-accident discovery."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "topic_modeling_results.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


cells = [
    markdown(
        """
# Recurrent accident scenarios — results notebook

This notebook reads a completed run and does not rerun the discovery pipeline.
The run directory, the number of representative sentences, the role prompts,
the OpenAI model and the two-dimensional UMAP settings are editable in the
configuration cell below.
        """
    ),
    code(
        """
from pathlib import Path
import json
import os
import sys
import textwrap
import warnings

import numpy as np
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

# Use either an absolute path or a path relative to this project directory.
# Examples: "runs/theme_discovery_audit/caou" or
# "runs/theme_discovery_audit/caou/audit_caou_20260812T150807Z".
RUN_DIRECTORY = "runs/theme_discovery_audit/caou"
OPENAI_ENABLED = True
OPENAI_MODEL = "gpt-4o-mini"
LLM_OUTPUT_LANGUAGE = "Français"
N_REPRESENTATIVE_SENTENCES = 25
MAX_TOPICS_PER_ROLE_FOR_LLM = None
FALLBACK_TOPICS_PER_ROLE = None
UMAP_2D_N_NEIGHBORS = 15
UMAP_2D_MIN_DIST = 0.1
UMAP_2D_RANDOM_STATE = 42
MAX_POINTS_PER_ROLE_PLOT = 12000
print("OPENAI_API_KEY disponible :", bool(os.environ.get("OPENAI_API_KEY")))

ROLE_PROMPTS = {
    "A0": '''Vous analysez des thèmes d'accidents du travail décrivant le contexte de travail, l'activité, le lieu, les équipements ou l'environnement opérationnel. Donnez à chaque thème un intitulé court et naturel. Ne décrivez pas la conséquence de l'accident et n'inventez aucune information absente des exemples.''',
    "A1": '''Vous analysez des thèmes d'accidents du travail décrivant des conditions défavorables, des dangers, des situations dangereuses, des protections absentes ou des équipements défectueux. Donnez à chaque thème un intitulé court et naturel. Ne décrivez pas l'événement ou la blessure, sauf si cela est nécessaire pour distinguer la condition défavorable.''',
    "B": '''Vous analysez des thèmes d'accidents du travail décrivant l'événement, la déviation, la perte de contrôle ou le mécanisme de l'accident. Donnez à chaque thème un intitulé court et naturel. Concentrez-vous sur ce qui s'est produit, et non uniquement sur le contexte de travail ou la blessure finale.''',
    "C": '''Vous analysez des thèmes d'accidents du travail décrivant les conséquences, les blessures, les parties du corps touchées ou les dommages. Donnez à chaque thème un intitulé court et naturel. Concentrez-vous sur la conséquence observée plutôt que sur sa cause.''',
}

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
    searched = "\\n".join(f"- {candidate.resolve()}" for candidate in candidates)
    available = sorted(path.name for path in (SCENARIO_DIR / "runs").glob("**/*") if path.is_dir())
    available_text = ", ".join(available[-20:]) if available else "aucun dossier de run trouvé"
    raise FileNotFoundError(
        f"Run directory not found for RUN_DIRECTORY={value!r}.\\n"
        f"Paths searched:\\n{searched}\\n"
        f"Recent run directories: {available_text}"
    )

RUN_DIR = resolve_run_directory(RUN_DIRECTORY)
ROLES = ("A0", "A1", "B", "C")
print("Scenario directory:", SCENARIO_DIR)
print("Run directory:", RUN_DIR)
        """
    ),
    markdown("## 1. Run manifest and resolved parallelism"),
    code(
        """
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
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    else:
        loaded = read_json("theme_discovery_parameters.json", {})
    if "data" not in loaded:
        raise ValueError(
            f"La configuration du run ne contient pas la section 'data' : {yaml_path}"
        )
    data_cfg = loaded["data"]
    for key in ("units_path", "embeddings_path"):
        if data_cfg.get(key):
            data_cfg[key] = str(local_path(data_cfg[key]))
    topics_cfg = loaded.setdefault("topics", {})
    if topics_cfg.get("stopwords_file"):
        topics_cfg["stopwords_file"] = str(local_path(topics_cfg["stopwords_file"]))
    return loaded

manifest = read_json("theme_discovery_manifest.json", {})
parallel = read_json("parallel_runtime.json", {})
config = load_run_config()
config.setdefault("topics", {})["top_sentences"] = max(
    int(config.get("topics", {}).get("top_sentences", 5)),
    N_REPRESENTATIVE_SENTENCES,
)
display(pd.DataFrame([{
    "dataset": manifest.get("dataset_id", config.get("data", {}).get("dataset_id")),
    "selection_objectives": ", ".join(manifest.get("selection_objectives", ["dbcv_umap", "stability"])),
    "n_workers": parallel.get("n_workers", manifest.get("n_workers")),
    "slurm_cpus_per_task": parallel.get("slurm_cpus_per_task"),
    "backend": parallel.get("backend"),
    "inner_umap_n_jobs": parallel.get("inner_umap_n_jobs"),
}]))
summary_path = RUN_DIR / "audit_input_summary.csv"
if summary_path.is_file():
    display(pd.read_csv(summary_path))
        """
    ),
    markdown("## 2. Pareto candidates"),
    code(
        """
selection_rows = []
for role in ROLES:
    path = RUN_DIR / "pareto" / role / "pareto_frontier.csv"
    if path.is_file():
        print(f"{role} — Pareto frontier")
        frontier = pd.read_csv(path)
        chosen_ids = []
        frontier_ids = set(frontier["configuration_id"].astype(str))
        selection_rows.extend(
            {
                "role": role,
                "configuration_id": chosen_id,
                "is_pareto_candidate": chosen_id in frontier_ids,
            }
            for chosen_id in chosen_ids
        )
        display(frontier)
        print()
selection_check = pd.DataFrame(selection_rows)
display(selection_check)
if False and (
    selection_check.empty
    or set(selection_check["role"]) != set(ROLES)
    or not selection_check["is_pareto_candidate"].all()
):
    raise ValueError(
        "PARTITION_SELECTIONS doit contenir au moins un configuration_id Pareto valide pour chaque rôle."
    )
        """
    ),
    markdown(
        """
## 3. Validation landscapes

The figures show dominated configurations and the Pareto frontier in the
(D_U, S_R) plane. The axes are UMAP-space DBCV and accident-level
resampling stability; the plot contains only the fixed extraction strategy.
        """
    ),
    code(
        """
figure_path = RUN_DIR / "figures" / "pareto_validation_all_roles.png"
if figure_path.is_file():
    display(Image(filename=str(figure_path)))
for role in ROLES:
    figure_path = RUN_DIR / "figures" / f"pareto_validation_{role}.png"
    if figure_path.is_file():
        display(Image(filename=str(figure_path)))
        """
    ),
    markdown("## 4. Candidate diagnostics and stability"),
    code(
        """
for role in ROLES:
    metrics_path = RUN_DIR / "pareto" / role / "candidate_metrics.csv"
    stability_path = RUN_DIR / "pareto" / role / "stability_summary.csv"
    theme_path = RUN_DIR / "pareto" / role / "stability_theme.csv"
    print(f"### {role}")
    frontier_path = RUN_DIR / "pareto" / role / "pareto_frontier.csv"
    if frontier_path.is_file():
        pareto = pd.read_csv(frontier_path)
        if "pareto_non_dominated" in pareto.columns:
            is_nondominated = pareto["pareto_non_dominated"].astype(str).str.lower().isin({"true", "1", "yes"})
            pareto = pareto.loc[is_nondominated].copy()
        pareto = pareto.sort_values("coverage", ascending=False, na_position="last")
        print("Pareto non-dominated candidates — sorted by coverage (descending)")
        display(pareto)
    elif metrics_path.is_file():
        # Fallback for runs created before pareto_frontier.csv was exported.
        metrics = pd.read_csv(metrics_path)
        if stability_path.is_file():
            metrics = metrics.merge(pd.read_csv(stability_path), on=["role", "configuration_id"], how="left")
        display(metrics.sort_values("coverage", ascending=False, na_position="last"))
    if theme_path.is_file():
        print("Theme-level median Jaccard stability")
        display(pd.read_csv(theme_path).sort_values(["configuration_id", "theme_stability"], ascending=[True, False]))
        """
    ),
    code(
        """
# Choose one or more Pareto configurations per role after inspecting sections 2--4.
PARTITION_SELECTIONS = {
    "A0": [],  # Example: ["A0_cfg_005", "A0_cfg_012"]
    "A1": [],  # Example: ["A1_cfg_033"]
    "B": [],   # Example: ["B_cfg_029", "B_cfg_041"]
    "C": [],   # Example: ["C_cfg_015"]
}
PARTITION_SELECTION = {role: (values[0] if values else "") for role, values in PARTITION_SELECTIONS.items()}

selection_rows = []
for role in ROLES:
    frontier_path = RUN_DIR / "pareto" / role / "pareto_frontier.csv"
    frontier = pd.read_csv(frontier_path) if frontier_path.is_file() else pd.DataFrame()
    frontier_ids = set(frontier.get("configuration_id", pd.Series(dtype=str)).astype(str))
    for configuration_id in PARTITION_SELECTIONS.get(role, []):
        selection_rows.append({
            "role": role,
            "configuration_id": str(configuration_id),
            "is_pareto_candidate": str(configuration_id) in frontier_ids,
        })
selection_check = pd.DataFrame(selection_rows)
display(selection_check)
if (
    selection_check.empty
    or set(selection_check["role"]) != set(ROLES)
    or not selection_check["is_pareto_candidate"].all()
):
    raise ValueError("PARTITION_SELECTIONS doit contenir au moins un configuration_id Pareto valide pour chaque rôle.")
        """
    ),
    markdown("## 5. Frozen themes and audit dictionary"),
    code(
        """
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))
from scenario_pipeline import PartitionResult, PreparedData, build_topic_dictionary, load_embeddings, load_units

missing_selection_roles = [role for role in ROLES if not PARTITION_SELECTIONS.get(role)]
if missing_selection_roles:
    raise ValueError(
        "Renseigne PARTITION_SELECTIONS avant cette cellule pour les rôles : "
        + ", ".join(missing_selection_roles)
        + ". Utilise les configuration_id présents dans pareto_frontier.csv."
    )

units, _ = load_units(config)
embedding_cache = RUN_DIR / "embeddings" / "embeddings_encoded.npy"
embeddings = np.load(embedding_cache) if embedding_cache.is_file() else load_embeddings(config, units, RUN_DIR / "embeddings")
if len(embeddings) != len(units):
    raise ValueError(f"Embedding/unit mismatch: {len(embeddings)} embeddings for {len(units)} units")

manual_results = {}
partition_frames = {}
for role in ROLES:
    configuration_id = str(PARTITION_SELECTION[role])
    labels_path = RUN_DIR / "pareto" / role / "candidate_partitions" / f"{configuration_id}_labels.npy"
    strength_path = RUN_DIR / "pareto" / role / "candidate_partitions" / f"{configuration_id}_membership_strength.npy"
    if not labels_path.is_file() or not strength_path.is_file():
        raise FileNotFoundError(f"Missing candidate artifacts for {role}/{configuration_id}")
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

    for candidate_id in [str(value) for value in PARTITION_SELECTIONS[role]]:
        candidate_labels_path = RUN_DIR / "pareto" / role / "candidate_partitions" / f"{candidate_id}_labels.npy"
        candidate_strength_path = RUN_DIR / "pareto" / role / "candidate_partitions" / f"{candidate_id}_membership_strength.npy"
        candidate_labels = np.load(candidate_labels_path).astype(int)
        candidate_strength = np.load(candidate_strength_path).astype(float)
        if len(candidate_labels) != len(assignments) or len(candidate_strength) != len(assignments):
            raise ValueError(f"Label/assignment mismatch for {role}/{candidate_id}")
        candidate_frame = assignments[["accident_id", "fact_id", "sentence"]].copy()
        candidate_frame["Topic"] = candidate_labels
        candidate_frame["membership_strength"] = candidate_strength
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

# Build a catalog for every selected Pareto partition. The configuration ID is
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
## 6. Membership-strength distributions

These scores are HDBSCAN membership strengths, not posterior probabilities.
They are shown for every selected Pareto partition. The histogram describes
the distribution over units, while the horizontal barplot compares the maximum,
mean and median strength for each topic. Noise units are excluded from topic
summaries but remain visible in the diagnostic count.
        """
    ),
    code(
        """
from matplotlib.ticker import MaxNLocator


membership_summaries = {}
for (role, configuration_id), frame in partition_frames.items():
    valid = frame[frame["Topic"].astype(int).ge(0)].copy()
    valid["Topic"] = valid["Topic"].astype(int)
    summary = valid.groupby("Topic")["membership_strength"].agg(
        n_units="size",
        max_strength="max",
        mean_strength="mean",
        median_strength="median",
    ).sort_values("max_strength", ascending=False)
    summary.insert(0, "role", role)
    summary.insert(1, "configuration_id", configuration_id)
    membership_summaries[(role, configuration_id)] = summary.reset_index()
    display(summary.reset_index())

    figure, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.7]})
    strengths = valid["membership_strength"].astype(float)
    axes[0].hist(strengths, bins=np.linspace(0, 1, 21), color="#4C78A8", edgecolor="white")
    axes[0].set_title(f"{role} — {configuration_id}: distribution")
    axes[0].set_xlabel("HDBSCAN membership strength")
    axes[0].set_ylabel("Number of units")
    axes[0].set_xlim(0, 1)
    axes[0].xaxis.set_major_locator(MaxNLocator(6))

    plot_summary = summary.sort_values("max_strength", ascending=True)
    positions = np.arange(len(plot_summary))
    axes[1].barh(positions, plot_summary["max_strength"], color="#F28E2B", alpha=0.85, label="Maximum")
    axes[1].scatter(plot_summary["mean_strength"], positions, color="#4C78A8", s=28, label="Moyenne", zorder=3)
    axes[1].scatter(plot_summary["median_strength"], positions, color="#59A14F", s=28, label="Médiane", zorder=3)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([f"Topic {int(topic)}" for topic in plot_summary.index])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Membership strength")
    axes[1].set_title(f"{role} — {configuration_id}: strength by topic")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="x", alpha=0.2)
    figure.tight_layout()
    (RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(RUN_DIR / "figures" / f"membership_strength_{role}_{configuration_id}.png", dpi=250, bbox_inches="tight")
    display(figure)
    plt.close(figure)

membership_summary_table = pd.concat(membership_summaries.values(), ignore_index=True) if membership_summaries else pd.DataFrame()
membership_summary_table.to_csv(RUN_DIR / "topics_manual" / "membership_strength_summary.csv", index=False)
        """
    ),
    markdown(
        """
## 7. Natural-language theme labels with the OpenAI API

The prompts are defined in the first code cell, separately for A0, A1, B and
C. For each role, the API receives the topic identifiers, top words and the
configured number of representative sentences. The output is requested as
JSON and saved locally. The API key is read only from the environment variable
OPENAI_API_KEY; it is never read from the run directory or written to the
notebook. In PowerShell, define it before launching Jupyter with
`$env:OPENAI_API_KEY = "votre-cle"`.
If the key, package or request is unavailable, the notebook keeps the
top-word label and continues without stopping the analysis.
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


def parse_llm_payload(raw_text):
    payload = json.loads(raw_text)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("themes", payload.get("labels", []))
    else:
        items = []
    return items if isinstance(items, list) else []


def request_role_labels(client, role, records):
    prompt = ROLE_PROMPTS[role]
    instruction = f'''
Retournez un seul objet JSON contenant un tableau `themes`, avec un élément
pour chaque `topic_id`. Rédigez l'intitulé, la description et les éléments de
preuve en {LLM_OUTPUT_LANGUAGE}. L'intitulé doit être court, précis et adapté à
une figure. La description doit tenir en une phrase. Utilisez uniquement les
mots-clés et les phrases représentatives fournis.
Chaque élément doit contenir : `topic_id`, `label`, `description` et `evidence`.

Consigne spécifique au rôle :
{prompt}

Thèmes à analyser :
{json.dumps(records, ensure_ascii=False, indent=2)}
'''
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Vous êtes un analyste prudent spécialisé dans les accidents du travail. Retournez uniquement un JSON valide."},
            {"role": "user", "content": instruction},
        ],
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content or "{}"
    return parse_llm_payload(raw_text)


llm_cache_path = RUN_DIR / "topics_manual" / "llm_theme_labels.csv"
llm_cache = pd.read_csv(llm_cache_path) if llm_cache_path.is_file() else pd.DataFrame()
if not llm_cache.empty and "configuration_id" not in llm_cache.columns:
    # Backward compatibility with the previous cache format, which contained
    # only labels for the primary partition of each role.
    llm_cache["configuration_id"] = llm_cache["role"].map(PARTITION_SELECTION)
for column in ("topic_id", "role", "configuration_id", "llm_label", "llm_description", "llm_evidence"):
    if column not in llm_cache.columns:
        llm_cache[column] = ""
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
        from openai import OpenAI

        client = OpenAI()
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
                if cached and str(cached.get("llm_label", "")).strip():
                    cached_count += 1
                    llm_rows.append({
                        "topic_id": record["topic_id"],
                        "role": role,
                        "configuration_id": record["configuration_id"],
                        "llm_label": str(cached.get("llm_label", "")).strip(),
                        "llm_description": str(cached.get("llm_description", "")).strip(),
                        "llm_evidence": str(cached.get("llm_evidence", "")).strip(),
                    })
                    continue
                try:
                    response_items = request_role_labels(client, role, [record])
                    by_topic = {str(item.get("topic_id")): item for item in response_items if item.get("topic_id")}
                    item = by_topic.get(record["topic_id"], {})
                    if not item:
                        missing_count += 1
                    llm_rows.append({
                        "topic_id": record["topic_id"],
                        "role": role,
                        "configuration_id": record["configuration_id"],
                        "llm_label": str(item.get("label", "")).strip(),
                        "llm_description": str(item.get("description", "")).strip(),
                        "llm_evidence": str(item.get("evidence", "")).strip(),
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
    list(llm_cache.drop(columns=["cache_key"], errors="ignore").to_dict("records"))
    + llm_rows
    + new_llm_rows,
    columns=["topic_id", "role", "configuration_id", "llm_label", "llm_description", "llm_evidence"],
)
if not llm_labels.empty:
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
    theme_labels[column] = theme_labels[column].fillna("").astype(str).str.strip()
for column in ("label", "top_terms"):
    if column not in theme_labels.columns:
        theme_labels[column] = ""
theme_labels["plot_label"] = theme_labels["llm_label"].fillna("").astype(str).str.strip()
empty_labels = theme_labels["plot_label"].eq("")
theme_labels.loc[empty_labels, "plot_label"] = theme_labels.loc[empty_labels, "label"].fillna("").astype(str)
empty_labels = theme_labels["plot_label"].eq("")
theme_labels.loc[empty_labels, "plot_label"] = theme_labels.loc[empty_labels, "top_terms"].fillna("").astype(str)
theme_labels.to_csv(RUN_DIR / "topics_manual" / "topic_dictionary_with_llm_labels.csv", index=False)
display(theme_labels[[column for column in ["topic_id", "role", "plot_label", "llm_description", "top_terms", "representative_sentences"] if column in theme_labels.columns]])
        """
    ),
    markdown(
        """
## 8. Narrative text coloured by topic

Each selected non-dominated partition can be inspected directly on the source
accident narrative. The colours are local to the displayed partition and the
legend uses the LLM label when available, otherwise a transparent topic ID or
top-term label. Noise (`-1`) remains uncoloured.
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
    partition_labels = theme_labels[
        theme_labels["role"].astype(str).eq(role)
        & theme_labels["configuration_id"].astype(str).eq(str(configuration_id))
    ].copy()
    label_lookup = partition_labels.set_index("topic_id")["plot_label"].astype(str).to_dict() if not partition_labels.empty else {}
    frame["topic_label"] = frame["Topic"].map(lambda value: label_lookup.get(f"{role}_{int(value):03d}", f"Topic {int(value)}") if int(value) >= 0 else "Bruit / non assigné")
    requested = accident_ids if accident_ids is not None else ACCIDENT_IDS_TO_DISPLAY.get(role)
    if requested is None or requested == "":
        requested = frame["accident_id"].drop_duplicates().head(1).tolist()
    elif isinstance(requested, (str, int)):
        requested = [requested]
    print(f"{role} — accidents disponibles : {frame['accident_id'].drop_duplicates().astype(str).tolist()[:20]}")
    for accident_id in requested:
        display(HTML(f"<h4>{role} — partition {configuration_id} — accident {html_lib.escape(str(accident_id))}</h4>"))
        show_colored_text_inline_v5(frame, accident_id)


for (role, configuration_id) in partition_frames:
    print(f"{role} — partition {configuration_id}")
    show_partition_narratives(role, configuration_id, accident_ids=ACCIDENT_IDS_TO_DISPLAY.get(role))
        """
    ),
    markdown(
        """
## 9. Two-dimensional UMAP topic map

This is a descriptive BERTopic-style map. A separate two-dimensional UMAP is
fit for visualization only; it is not used for clustering or model selection.
Points are colored by retained topic, while noise and topics not retained for
downstream modelling are shown in light gray. The labels are the OpenAI labels
when available and otherwise the top-word labels.
        """
    ),
    code(
        """
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))
from scenario_pipeline import load_embeddings, load_units


def load_assignments():
    frames = [manual_results[role].assignments[["fact_id", "role", "topic_id"]] for role in ROLES]
    return pd.concat(frames, ignore_index=True).drop_duplicates("fact_id")


assignments = load_assignments()
assignments["fact_id"] = assignments["fact_id"].astype(str)
unit_index = pd.DataFrame({"fact_id": units["_fact_id"].astype(str), "embedding_index": np.arange(len(units))})
plot_frame = assignments.merge(unit_index, on="fact_id", how="inner")
if not selected_topics.empty and "topic_id" in selected_topics.columns:
    retained_topic_ids = set(selected_topics["topic_id"].astype(str))
else:
    retained_topic_ids = set(theme_labels["topic_id"].astype(str))
plot_frame["topic_id"] = plot_frame["topic_id"].fillna("").astype(str)
plot_frame["plot_topic_id"] = plot_frame["topic_id"].where(plot_frame["topic_id"].isin(retained_topic_ids), "")
primary_theme_labels = theme_labels[
    theme_labels["configuration_id"].astype(str).eq(theme_labels["role"].map(PARTITION_SELECTION).astype(str))
]
label_lookup = primary_theme_labels.set_index("topic_id")["plot_label"].astype(str).to_dict()

import umap


def shorten_label(value, width=32):
    value = str(value).strip()
    return textwrap.shorten(value, width=width, placeholder="…") or "Topic"


def plot_role_umap(role):
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
    figure, axis = plt.subplots(figsize=(12, 9))
    noise = role_plot[role_plot["plot_topic_id"].eq("")]
    if not noise.empty:
        axis.scatter(noise["x"], noise["y"], s=6, c="#7f7f7f", alpha=0.45, linewidths=0, rasterized=True, label="Bruit / non assigné")
    active_topics = [topic_id for topic_id in sorted(role_plot["plot_topic_id"].unique()) if topic_id]
    colors = plt.get_cmap("tab20", max(1, len(active_topics)))
    for index, topic_id in enumerate(active_topics):
        subset = role_plot[role_plot["plot_topic_id"].eq(topic_id)]
        color = colors(index)
        axis.scatter(subset["x"], subset["y"], s=8, color=color, alpha=0.72, linewidths=0, rasterized=True)
        center_x = float(subset["x"].mean())
        center_y = float(subset["y"].mean())
        axis.text(
            center_x,
            center_y,
            shorten_label(label_lookup.get(topic_id, topic_id)),
            fontsize=9,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": color},
        )
    axis.set_title(f"{role} — partition {PARTITION_SELECTION[role]}")
    axis.set_xlabel("UMAP-1")
    axis.set_ylabel("UMAP-2")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    return figure


(RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)
role_figures = {}
for role in ROLES:
    figure = plot_role_umap(role)
    if figure is not None:
        output_path = RUN_DIR / "figures" / f"umap_topics_2d_{role}.png"
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
        role_figures[role] = output_path
        display(figure)
        plt.close(figure)
        """
    ),
    markdown("## 10. Export locations"),
    code(
        """
outputs = [
    "config_resolved.yaml", "parallel_runtime.json", "pareto_selection_summary.csv",
    "figures/pareto_validation_all_roles.png", "figures/umap_topics_2d_A0.png",
    "figures/umap_topics_2d_A1.png", "figures/umap_topics_2d_B.png",
    "figures/umap_topics_2d_C.png", "topics_manual/topic_dictionary.csv",
    "topics_manual/topic_dictionary_all_selected.csv", "topics_manual/representatives_by_membership.csv",
    "topics_manual/llm_theme_labels.csv", "topics_manual/topic_dictionary_with_llm_labels.csv",
    "pareto/A0/candidate_partitions/<configuration_id>_labels.npy",
    "pareto/A0/candidate_partitions/<configuration_id>_membership_strength.npy",
]
display(pd.DataFrame({"path": outputs, "exists": [(RUN_DIR / path).exists() for path in outputs]}))
        """
    ),
]


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(
    json.dumps(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)
print(f"Notebook written: {NOTEBOOK_PATH}")
