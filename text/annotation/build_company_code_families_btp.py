"""Create the BTP company-family notebook from the common family workflow."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "company_code_families_caou.ipynb"
TARGET = ROOT / "company_code_families_btp.ipynb"


def lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip().splitlines()]


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook["cells"][0]["source"] = lines(
        "# Macro-activités des entreprises — Bâtiment et travaux publics"
    )
    notebook["cells"][1]["source"] = lines(
        """
## Objectif et principe

Le fichier source contient deux systèmes de codification : les codes antérieurs
à 2015 sont des **codes risque historiques** et les codes plus récents des
**codes APE/NAF**. Ils ne sont donc pas ramenés artificiellement à une division
NAF commune.

Chaque accident reçoit trois variables de traçabilité : `company_code`,
`company_label` et `coding_system`. La variable `macro_activity` est construite
à partir du **libellé d'activité**, au moyen de règles explicites et ordonnées.
Elle définit des contextes BTP homogènes pour l'analyse locale des motifs.

Les familles comptant au moins `MIN_CANDIDATE_SIZE` accidents sont exportées
comme candidates à l'analyse locale. Les autres restent visibles dans
`macro_activity_raw`, mais leur famille finale est `Other activities`. Chaque
ligne exportée conserve `accident_id`, ce qui maintient le lien avec les unités
factuelles et les embeddings.
        """
    )
    notebook["cells"][2]["source"] = lines(
        """
from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)

DATASET = "btp"
DATA_PATH = Path("data/btp_sentence_accidents.csv")
EXPORT_PATH = Path("../dataset/company_family_btp.csv")
MIN_CANDIDATE_SIZE = 75
TAXONOMY_VERSION = "label_based_btp_macro_activity_v1"
        """
    )
    notebook["cells"][4]["source"] = lines(
        r'''
TAXONOMY_RULES = (
    ("building_construction_and_masonry", "Building construction and masonry", r"gros.{0,6}uvre|ma.onnerie|construction d.autres b.timents|maisons individuelles"),
    ("roofing_and_waterproofing", "Roofing and waterproofing", r"couverture|.tanch"),
    ("civil_engineering_and_networks", "Civil engineering and networks", r"ouvrages d.art|chauss|sols sportifs|pavage|lignes .lectriques|t.l.communication|travaux urbains|canalisations . grande distance|voies ferr|maritimes|fluviaux|r.seaux (?:.lectriques|pour fluides)|routes et autoroutes|tunnels"),
    ("earthworks_foundations_and_demolition", "Earthworks, foundations and demolition", r"terrass|forages|fondations|d.molition|sondages"),
    ("electrical_installation", "Electrical installation", r"installation .lectrique|pose d.enseignes|paratonnerres|antennes"),
    ("plumbing_heating_and_hvac", "Plumbing, heating and HVAC", r"plomberie|sanitaires|installation d.eau et de gaz|.quipements thermiques|climatisation|a.raulique|frigorifique"),
    ("carpentry_and_joinery", "Carpentry and joinery", r"menuiserie|charpente"),
    ("building_finishing_and_insulation", "Building finishing and insulation", r"peinture|ravalement|am.nagement int.rieur|isolation|pl.trerie|rev.tement|finitions|ass.chement|ignifugation"),
    ("metal_construction_and_assembly", "Metal construction and assembly", r"construction m.tallique|structures m.talliques|m.tallerie|serrurerie"),
    ("construction_equipment_with_operator", "Construction equipment with operator", r"location de mat.riel.*b.timent|location avec op.rateur"),
    ("engineering_and_project_support", "Engineering and project support", r"ing.nierie|architecture|topographie|m.tr.s"),
)


parsed = accidents["company_code_raw"].astype("string").str.extract(
    r"^\s*(?P<company_code>(?:\d{4}[A-Z]|\d{3}[A-Z]{2}))\s*-\s*(?P<company_label>.+?)\s*$"
)
accidents["company_code"] = parsed["company_code"].astype("string")
accidents["company_label"] = parsed["company_label"].astype("string")

# The format identifies the source system; historical codes are never read as NAF.
accidents["coding_system"] = pd.Series("unclassified", index=accidents.index, dtype="string")
accidents.loc[accidents["company_code"].str.fullmatch(r"\d{3}[A-Z]{2}").fillna(False), "coding_system"] = "historical_risk_code"
accidents.loc[accidents["company_code"].str.fullmatch(r"\d{4}[A-Z]").fillna(False), "coding_system"] = "NAF_APE"


def assign_macro_activity(label: object) -> tuple[str, str]:
    if pd.isna(label):
        return "Other activities", "missing_or_unparseable_label"
    label_text = str(label).lower()
    for rule_id, macro_activity, pattern in TAXONOMY_RULES:
        if re.search(pattern, label_text):
            return macro_activity, rule_id
    return "Other activities", "no_taxonomy_rule_matched"


assignments = accidents["company_label"].map(assign_macro_activity)
accidents[["macro_activity_raw", "macro_activity_rule"]] = pd.DataFrame(
    assignments.tolist(), index=accidents.index
)
accidents["macro_activity_raw"] = accidents["macro_activity_raw"].astype("string")
accidents["macro_activity_rule"] = accidents["macro_activity_rule"].astype("string")

raw_counts = accidents["macro_activity_raw"].value_counts()
accidents["macro_activity_size_before_other"] = accidents["macro_activity_raw"].map(raw_counts).astype("Int64")
accidents["macro_activity"] = accidents["macro_activity_raw"].copy()
small_or_other = (
    accidents["macro_activity_raw"].eq("Other activities")
    | accidents["macro_activity_size_before_other"].lt(MIN_CANDIDATE_SIZE)
)
accidents.loc[small_or_other, "macro_activity"] = "Other activities"
accidents["is_local_analysis_candidate"] = ~accidents["macro_activity"].eq("Other activities")
accidents["family"] = accidents["macro_activity"]
accidents["family_grouping_reason"] = pd.Series(
    "Rule applied to company_label: " + accidents["macro_activity_rule"],
    index=accidents.index,
    dtype="string",
)
accidents.loc[small_or_other, "family_grouping_reason"] = (
    "Other activities: raw macro-activity is unmatched or below "
    + str(MIN_CANDIDATE_SIZE)
    + " accidents; raw assignment retained in macro_activity_raw."
)
accidents["taxonomy_version"] = TAXONOMY_VERSION
        '''
    )
    notebook["cells"][10]["source"] = lines(
        """
# Build topic-modeling inputs for selected BTP macro-activity families.
# The generated unit and embedding files remain aligned through fact_id / doc_id.
# Re-run this cell after changing the taxonomy or the threshold above.

FAMILIES_TO_EXPORT = None  # None = all local-analysis candidates; or e.g. ["Electrical installation"]
INCLUDE_OTHER_ACTIVITIES = False
SOURCE_UNITS_PATH = Path(f"../dataset/data_{DATASET}.csv")
SOURCE_EMBEDDINGS_PATH = Path(f"../embeddings/Qwen3-Embedding-0.6B_{DATASET}.csv")
FAMILY_DATASET_DIR = Path(f"../dataset/families/{DATASET}")
FAMILY_EMBEDDING_DIR = Path(f"../embeddings/families/{DATASET}")

if not SOURCE_UNITS_PATH.is_file() or not SOURCE_EMBEDDINGS_PATH.is_file():
    raise FileNotFoundError("Source units or embeddings file is missing.")


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def corpus_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


selected_families = sorted(
    family_export.loc[family_export["is_local_analysis_candidate"], "macro_activity"].dropna().unique().tolist()
)
if INCLUDE_OTHER_ACTIVITIES:
    selected_families.append("Other activities")
if FAMILIES_TO_EXPORT is not None:
    requested = set(FAMILIES_TO_EXPORT)
    unknown = requested.difference(selected_families)
    if unknown:
        raise ValueError(f"Unknown or ineligible families: {sorted(unknown)}")
    selected_families = [family for family in selected_families if family in requested]
if not selected_families:
    raise ValueError("No family selected for export.")

units_source = pd.read_csv(SOURCE_UNITS_PATH, dtype={"accident_id": "string", "fact_id": "string"})
required_unit_columns = {"accident_id", "fact_id", "sentence", "pred_label"}
missing_unit_columns = required_unit_columns.difference(units_source.columns)
if missing_unit_columns:
    raise ValueError(f"Missing unit columns: {sorted(missing_unit_columns)}")

family_lookup = family_export[["accident_id", "macro_activity", "family", "is_local_analysis_candidate"]].copy()
family_lookup["accident_id"] = family_lookup["accident_id"].astype("string")
units_source["accident_id"] = units_source["accident_id"].astype("string")
units_source["fact_id"] = units_source["fact_id"].astype("string")
units_with_family = units_source.merge(family_lookup, on="accident_id", how="inner", validate="many_to_one")

# Match the exact filtering performed by recurrent_scenarios.load_units.
ready_mask = units_with_family["sentence"].fillna("").astype(str).str.strip().ne("")
ready_mask &= units_with_family["pred_label"].astype(str).str.strip().isin({"A0", "A1", "B", "C"})
if "pred_ok" in units_with_family.columns:
    ready_mask &= is_true(units_with_family["pred_ok"])
topic_ready_units = units_with_family.loc[ready_mask].copy()
if topic_ready_units["fact_id"].duplicated().any():
    raise ValueError("fact_id must be unique after topic-input filtering.")

embeddings_source = pd.read_csv(SOURCE_EMBEDDINGS_PATH, dtype={"doc_id": "string"})
if "doc_id" not in embeddings_source.columns:
    raise ValueError("Embeddings must contain doc_id.")
if embeddings_source["doc_id"].duplicated().any():
    raise ValueError("Embedding doc_id must be unique.")

FAMILY_DATASET_DIR.mkdir(parents=True, exist_ok=True)
FAMILY_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
manifest_rows = []
for family in selected_families:
    corpus_id = f"{DATASET}_{corpus_slug(family)}"
    family_units = topic_ready_units.loc[topic_ready_units["macro_activity"].eq(family)].copy()
    fact_ids = family_units["fact_id"].astype("string")
    family_embeddings = embeddings_source.loc[embeddings_source["doc_id"].isin(fact_ids)].copy()

    missing_embeddings = set(fact_ids).difference(set(family_embeddings["doc_id"]))
    if missing_embeddings:
        raise ValueError(f"{corpus_id}: {len(missing_embeddings)} units have no embedding.")
    if len(family_embeddings) != len(family_units):
        raise ValueError(f"{corpus_id}: units / embeddings row count mismatch.")

    units_path = FAMILY_DATASET_DIR / f"data_{corpus_id}.csv"
    embeddings_path = FAMILY_EMBEDDING_DIR / f"Qwen3-Embedding-0.6B_{corpus_id}.csv"
    family_units.to_csv(units_path, index=False)
    family_embeddings.to_csv(embeddings_path, index=False)
    manifest_rows.append({
        "dataset_id": corpus_id,
        "macro_activity": family,
        "n_accidents": int(family_units["accident_id"].nunique()),
        "n_units": int(len(family_units)),
        "n_embeddings": int(len(family_embeddings)),
        "units_path": str(units_path),
        "embeddings_path": str(embeddings_path),
    })

family_topic_manifest = pd.DataFrame(manifest_rows).sort_values("dataset_id", kind="stable")
manifest_path = FAMILY_DATASET_DIR / "topic_modeling_manifest.csv"
family_topic_manifest.to_csv(manifest_path, index=False)
display(family_topic_manifest)
print(f"Manifest written: {manifest_path.resolve()}")
print("Use one dataset_id with: DATASET=<dataset_id> sbatch jobs/run_recurrent_scenarios_theme_discovery.sh")
        """
    )
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook written: {TARGET}")


if __name__ == "__main__":
    main()
