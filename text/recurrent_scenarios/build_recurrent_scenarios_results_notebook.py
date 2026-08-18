"""Build the post-run results notebook for recurrent-accident discovery."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "recurrent_scenarios_results.ipynb"


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
N_REPRESENTATIVE_SENTENCES = 5
MAX_TOPICS_PER_ROLE_FOR_LLM = None
FALLBACK_TOPICS_PER_ROLE = None
# Choose one Pareto configuration per role after inspecting the diagnostics.
# These example IDs are editable and are validated against pareto_frontier.csv.
PARTITION_SELECTION = {
    "A0": "A0_cfg_005",
    "A1": "A1_cfg_033",
    "B": "B_cfg_029",
    "C": "C_cfg_015",
}
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
    for key in ("units_path", "embeddings_path", "stopwords_file"):
        if data_cfg.get(key):
            data_cfg[key] = str(local_path(data_cfg[key]))
    return loaded

manifest = read_json("theme_discovery_manifest.json", {})
parallel = read_json("parallel_runtime.json", {})
config = load_run_config()
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
    markdown("## 2. Pareto candidates and manual partition selection"),
    code(
        """
selection_rows = []
for role in ROLES:
    path = RUN_DIR / "pareto" / role / "pareto_frontier.csv"
    if path.is_file():
        print(f"{role} — Pareto frontier")
        frontier = pd.read_csv(path)
        chosen_id = PARTITION_SELECTION.get(role, "")
        selection_rows.append({"role": role, "configuration_id": chosen_id, "is_pareto_candidate": chosen_id in set(frontier["configuration_id"].astype(str))})
        display(frontier)
        print()
selection_check = pd.DataFrame(selection_rows)
display(selection_check)
if not selection_check["is_pareto_candidate"].all():
    raise ValueError("Each PARTITION_SELECTION entry must be a configuration_id present in that role's pareto_frontier.csv.")
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
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        if stability_path.is_file():
            metrics = metrics.merge(pd.read_csv(stability_path), on=["role", "configuration_id"], how="left")
        display(metrics.sort_values(["dbcv_umap", "stability"], ascending=False))
    if theme_path.is_file():
        print("Theme-level median Jaccard stability")
        display(pd.read_csv(theme_path).sort_values(["configuration_id", "theme_stability"], ascending=[True, False]))
        """
    ),
    markdown("## 5. Frozen themes and audit dictionary"),
    code(
        """
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))
from scenario_pipeline import PartitionResult, PreparedData, build_topic_dictionary, load_embeddings, load_units

units, _ = load_units(config)
embedding_cache = RUN_DIR / "embeddings" / "embeddings_encoded.npy"
embeddings = np.load(embedding_cache) if embedding_cache.is_file() else load_embeddings(config, units, RUN_DIR / "embeddings")
if len(embeddings) != len(units):
    raise ValueError(f"Embedding/unit mismatch: {len(embeddings)} embeddings for {len(units)} units")

manual_results = {}
for role in ROLES:
    configuration_id = str(PARTITION_SELECTION[role])
    assignments_path = RUN_DIR / "clustering" / role / "topic_assignments.csv"
    labels_path = RUN_DIR / "pareto" / role / "candidate_partitions" / f"{configuration_id}_labels.npy"
    if not assignments_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Missing candidate artifacts for {role}/{configuration_id}")
    assignments = pd.read_csv(assignments_path)
    labels = np.load(labels_path).astype(int)
    if len(assignments) != len(labels):
        raise ValueError(f"Label/assignment mismatch for {role}/{configuration_id}")
    assignments["topic_id"] = [f"{role}_{label:03d}" if label >= 0 else "" for label in labels]
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

prepared_manual = PreparedData(units=units, embeddings=embeddings, input_summary=pd.DataFrame())
topics = build_topic_dictionary(prepared_manual, manual_results, config, RUN_DIR / "topics_manual")
topics["selected_configuration_id"] = topics["role"].map(PARTITION_SELECTION)
topics.to_csv(RUN_DIR / "topics_manual" / "topic_dictionary.csv", index=False)
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
## 6. Natural-language theme labels with the OpenAI API

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
    role_topics = topics[topics["role"].astype(str).eq(role)].copy()
    if not selected_topics.empty and "topic_id" in selected_topics.columns:
        selected_ids = set(selected_topics["topic_id"].astype(str))
        role_topics = role_topics[role_topics["topic_id"].astype(str).isin(selected_ids)]
    elif FALLBACK_TOPICS_PER_ROLE is not None:
        role_topics = role_topics.sort_values("n_accidents", ascending=False).head(int(FALLBACK_TOPICS_PER_ROLE))
    if MAX_TOPICS_PER_ROLE_FOR_LLM is not None:
        role_topics = role_topics.sort_values("n_accidents", ascending=False).head(int(MAX_TOPICS_PER_ROLE_FOR_LLM))
    records = []
    for _, row in role_topics.iterrows():
        records.append({
            "topic_id": str(row["topic_id"]),
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


llm_rows = []
llm_status = {}
if not topics.empty and OPENAI_ENABLED and os.environ.get("OPENAI_API_KEY"):
    try:
        from openai import OpenAI

        client = OpenAI()
        for role in ROLES:
            records = topic_records_for_role(role)
            if not records:
                continue
            failed_count = 0
            missing_count = 0
            for record in tqdm(records, desc=f"OpenAI labels {role}", unit="topic"):
                try:
                    response_items = request_role_labels(client, role, [record])
                    by_topic = {str(item.get("topic_id")): item for item in response_items if item.get("topic_id")}
                    item = by_topic.get(record["topic_id"], {})
                    if not item:
                        missing_count += 1
                    llm_rows.append({
                        "topic_id": record["topic_id"],
                        "role": role,
                        "llm_label": str(item.get("label", "")).strip(),
                        "llm_description": str(item.get("description", "")).strip(),
                        "llm_evidence": str(item.get("evidence", "")).strip(),
                    })
                except Exception as error:
                    failed_count += 1
                    llm_rows.append({
                        "topic_id": record["topic_id"],
                        "role": role,
                        "llm_label": "",
                        "llm_description": "",
                        "llm_evidence": "",
                    })
                    warnings.warn(f"OpenAI labelling failed for {role}/{record['topic_id']}: {error}")
            if failed_count:
                llm_status[role] = f"completed_with_failures:{failed_count}"
            elif missing_count:
                llm_status[role] = f"completed_with_missing_topics:{missing_count}"
            else:
                llm_status[role] = "completed"
    except Exception as error:
        llm_status["global"] = f"unavailable: {error}"
        warnings.warn(f"OpenAI labelling unavailable: {error}")
else:
    llm_status["global"] = "skipped: OPENAI_ENABLED is false or OPENAI_API_KEY is missing"

llm_labels = pd.DataFrame(llm_rows, columns=["topic_id", "role", "llm_label", "llm_description", "llm_evidence"])
(RUN_DIR / "topics_manual").mkdir(parents=True, exist_ok=True)
llm_labels.to_csv(RUN_DIR / "topics_manual" / "llm_theme_labels.csv", index=False)
(RUN_DIR / "topics_manual" / "llm_theme_labels_status.json").write_text(json.dumps(llm_status, indent=2, ensure_ascii=False), encoding="utf-8")

theme_labels = topics.copy()
if not llm_labels.empty:
    theme_labels = theme_labels.merge(llm_labels, on=["topic_id", "role"], how="left")
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
## 7. Two-dimensional UMAP topic map

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
label_lookup = theme_labels.set_index("topic_id")["plot_label"].astype(str).to_dict()

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
    markdown("## 8. Export locations"),
    code(
        """
outputs = [
    "config_resolved.yaml", "parallel_runtime.json", "pareto_selection_summary.csv",
    "figures/pareto_validation_all_roles.png", "figures/umap_topics_2d_A0.png",
    "figures/umap_topics_2d_A1.png", "figures/umap_topics_2d_B.png",
    "figures/umap_topics_2d_C.png", "topics_manual/topic_dictionary.csv",
    "topics_manual/llm_theme_labels.csv", "topics_manual/topic_dictionary_with_llm_labels.csv",
    "pareto/A0/candidate_partitions/<configuration_id>_labels.npy", "clustering/A0/topic_assignments.csv",
    "clustering/A1/topic_assignments.csv", "clustering/B/topic_assignments.csv",
    "clustering/C/topic_assignments.csv",
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
