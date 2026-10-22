"""Recode factual units into explicit, generic accident components.

The selected leaf clusters inform the component vocabulary but are not assumed
to be homogeneous. Every valid factual unit, including HDBSCAN noise, is
examined by the same role-specific text rules. The output retains the source
cluster and original sentence for auditing. No LLM or embedding is used here.

Run from text/:
  python bayesian_network/curate_leaf_components.py \
    --source-run bayesian_network/runs/<leaf-run> \
    --output-dir bayesian_network/runs/curated_leaf_components_qwen3_4b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


TEXT_ROOT = Path(__file__).resolve().parents[1]
ROLES = ("A0", "A1", "B", "C")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (TEXT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(value: str) -> str:
    value = str(value).lower().replace("\x92", "'").replace("’", "'")
    return "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def load_definitions(path: Path) -> list[dict]:
    definitions = yaml.safe_load(path.read_text(encoding="utf-8"))["components"]
    ids = [entry["id"] for entry in definitions]
    if len(set(ids)) != len(ids) or not all(entry["role"] in ROLES for entry in definitions):
        raise ValueError("Component IDs must be unique and roles valid")
    for entry in definitions:
        if not entry.get("patterns") or not entry.get("label"):
            raise ValueError(f"Missing patterns or label: {entry['id']}")
        entry["compiled"] = [re.compile(pattern) for pattern in entry["patterns"]]
        entry["compiled_exclusions"] = [
            re.compile(pattern) for pattern in entry.get("exclude_patterns", [])
        ]
    return definitions


def load_source(source: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("The source discovery run must be complete")
    paths = [source / "discovery" / role / "selected" / "topic_assignments.csv" for role in ROLES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    frames = [pd.read_csv(path, dtype={"accident_id": str, "fact_id": str},
                          keep_default_na=False) for path in paths]
    units = pd.concat(frames, ignore_index=True)
    required = {"accident_id", "fact_id", "sentence", "role", "topic_id"}
    if not required.issubset(units):
        raise ValueError(f"Selected assignments need {sorted(required)}")
    if units["fact_id"].duplicated().any() or units["accident_id"].eq("").any():
        raise ValueError("Unit identifiers must be unique and accidents present")
    if units["role"].value_counts().to_dict().keys() != set(ROLES):
        raise ValueError("The selected assignments must contain exactly four valid roles")
    if len(units) != 16189 or units["accident_id"].nunique() != 2382:
        raise ValueError("Unexpected source corpus size: inspect the source run before curation")
    return units, {"source_manifest_sha256": sha256(source / "manifest.json"),
                   "selected_assignments_sha256": {role: sha256(path)
                                                   for role, path in zip(ROLES, paths)}}


def recode(units: pd.DataFrame, definitions: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_role = {role: [entry for entry in definitions if entry["role"] == role]
               for role in ROLES}
    unit_rows = []
    matches = []
    for row in units.itertuples(index=False):
        sentence = normalise(row.sentence)
        found = [
            entry["id"] for entry in by_role[row.role]
            if any(pattern.search(sentence) for pattern in entry["compiled"])
            and not any(pattern.search(sentence)
                        for pattern in entry["compiled_exclusions"])
        ]
        unit_rows.append({"accident_id": row.accident_id, "fact_id": row.fact_id,
                          "role": row.role, "source_topic_id": row.topic_id,
                          "sentence": row.sentence, "component_ids": ";".join(found),
                          "component_count": len(found)})
        matches.extend({"accident_id": row.accident_id, "fact_id": row.fact_id,
                        "role": row.role, "source_topic_id": row.topic_id,
                        "component_id": component, "sentence": row.sentence}
                       for component in found)
    return pd.DataFrame(unit_rows), pd.DataFrame(matches)


def build_outputs(units: pd.DataFrame, matches: pd.DataFrame,
                  definitions: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ids = [entry["id"] for entry in definitions]
    accidents = pd.Index(sorted(units["accident_id"].unique()), name="accident_id")
    matrix = pd.DataFrame(0, index=accidents, columns=ids, dtype=np.uint8)
    if not matches.empty:
        pairs = matches[["accident_id", "component_id"]].drop_duplicates()
        for component_id, group in pairs.groupby("component_id"):
            matrix.loc[group["accident_id"], component_id] = 1
    dictionary = pd.DataFrame([{"factor_id": entry["id"], "role": entry["role"],
                                "llm_label": entry["label"],
                                "llm_description": entry["description"],
                                "label_source": "explicit_text_rules",
                                "unit_count": int((matches["component_id"] == entry["id"]).sum()),
                                "accident_count": int(matrix[entry["id"]].sum()),
                                "accident_percent": round(100 * matrix[entry["id"]].mean(), 2)}
                               for entry in definitions])
    by_role = {}
    for role in ROLES:
        group = units[units["role"] == role]
        selected = matches[matches["role"] == role]
        by_role[role] = {"units": len(group), "accidents_with_role": int(group["accident_id"].nunique()),
                         "source_clustered_units": int(group["source_topic_id"].ne("").sum()),
                         "recoded_units": int(group["component_count"].gt(0).sum()),
                         "recoded_source_noise_units": int((group["component_count"].gt(0) &
                                                            group["source_topic_id"].eq("")).sum()),
                         "unassigned_units": int(group["component_count"].eq(0).sum()),
                         "components": len([d for d in definitions if d["role"] == role]),
                         "accidents_with_component": int(selected["accident_id"].nunique())}
    audit = {"accidents": len(matrix), "units": len(units), "component_count": len(ids),
             "units_with_component": int(units["component_count"].gt(0).sum()),
             "units_without_component": int(units["component_count"].eq(0).sum()),
             "accidents_with_component": int(matrix.any(axis=1).sum()),
             "accidents_without_component": int((~matrix.any(axis=1)).sum()),
             "multi_component_units": int(units["component_count"].gt(1).sum()),
             "role_breakdown": by_role}
    return matrix.reset_index(), dictionary, audit


def write_review_samples(units: pd.DataFrame, matches: pd.DataFrame,
                         definitions: list[dict], output: Path) -> None:
    samples = []
    for entry in definitions:
        group = matches[matches["component_id"] == entry["id"]]
        if group.empty:
            continue
        pick = group.sample(min(12, len(group)), random_state=42)
        samples.append(pick.assign(review_bucket=entry["id"]))
    for role in ROLES:
        group = units[(units["role"] == role) & (units["component_count"] == 0)]
        if group.empty:
            continue
        pick = group.sample(min(30, len(group)), random_state=42)
        pick = pick.rename(columns={"source_topic_id": "source_topic_id"})
        pick["component_id"] = ""
        pick["review_bucket"] = f"{role}_UNASSIGNED"
        samples.append(pick)
    pd.concat(samples, ignore_index=True, sort=False).to_csv(
        output / "review_samples.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--definitions", default="bayesian_network/component_definitions_leaf.yaml")
    parser.add_argument("--min-component-accidents", type=int, default=12,
                        help="Minimum distinct accidents for a model component")
    parser.add_argument("--force", action="store_true",
                        help="Replace a completed curation after codebook or code changes")
    args = parser.parse_args()
    source, output, codebook = map(resolve, (args.source_run, args.output_dir, args.definitions))
    if source == output or output.is_relative_to(source):
        raise ValueError("The curated output must be separate from the source discovery run")
    definitions = load_definitions(codebook)
    source_units, provenance = load_source(source)
    prior_path = output / "curation_manifest.json"
    if prior_path.is_file() and not args.force:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        expected = {
            "status": "complete",
            "source_run": str(source),
            "source": provenance,
            "definitions_sha256": sha256(codebook),
            "code_sha256": sha256(Path(__file__)),
            "min_component_accidents": args.min_component_accidents,
        }
        same_inputs = all(prior.get(key) == value for key, value in expected.items())
        required = {
            "matrix_sha256": output / "accident_factor_matrix.csv",
            "dictionary_sha256": output / "factor_dictionary.csv",
            "unit_audit_sha256": output / "unit_component_audit.csv",
        }
        same_outputs = all(path.is_file() and prior.get(key) == sha256(path)
                           for key, path in required.items())
        if same_inputs and same_outputs:
            print(f"Reusing completed curated matrix: {output}")
            return
        raise ValueError(
            f"Existing curation differs from the requested inputs: {output}. "
            "Choose another output directory or pass --force before downstream modelling."
        )
    units, matches = recode(source_units, definitions)
    candidate_counts = (matches.groupby("component_id")["accident_id"].nunique()
                        .to_dict() if not matches.empty else {})
    dropped = [
        {"factor_id": entry["id"], "role": entry["role"],
         "label": entry["label"], "accident_count": int(candidate_counts.get(entry["id"], 0))}
        for entry in definitions
        if candidate_counts.get(entry["id"], 0) < args.min_component_accidents
    ]
    definitions = [entry for entry in definitions
                   if candidate_counts.get(entry["id"], 0) >= args.min_component_accidents]
    retained = {entry["id"] for entry in definitions}
    matches = matches[matches["component_id"].isin(retained)].copy()
    grouped = matches.groupby("fact_id")["component_id"].agg(lambda values: ";".join(values))
    units["component_ids"] = units["fact_id"].map(grouped).fillna("")
    units["component_count"] = units["component_ids"].map(
        lambda value: 0 if not value else len(value.split(";"))
    )
    matrix, dictionary, audit = build_outputs(units, matches, definitions)
    audit["min_component_accidents"] = args.min_component_accidents
    audit["excluded_components"] = dropped
    values = matrix[dictionary["factor_id"].tolist()].astype(np.int64)
    accident_count = values.sum(axis=1)
    role_count = pd.DataFrame({
        role: values[[entry["id"] for entry in definitions
                      if entry["role"] == role]].any(axis=1)
        for role in ROLES
    }).sum(axis=1)
    audit["components_per_accident"] = {
        str(int(count)): int(frequency)
        for count, frequency in accident_count.value_counts().sort_index().items()
    }
    audit["roles_per_accident"] = {
        str(int(count)): int(frequency)
        for count, frequency in role_count.value_counts().sort_index().items()
    }
    if dictionary["accident_count"].eq(0).any():
        empty = dictionary.loc[dictionary["accident_count"].eq(0), "factor_id"].tolist()
        raise ValueError(f"Components without an observed occurrence: {empty}")
    output.mkdir(parents=True, exist_ok=True)
    units.to_csv(output / "unit_component_audit.csv", index=False, encoding="utf-8")
    matches.to_csv(output / "unit_component_assignments.csv", index=False, encoding="utf-8")
    matrix.to_csv(output / "accident_factor_matrix.csv", index=False)
    dictionary.to_csv(output / "factor_dictionary.csv", index=False, encoding="utf-8")
    pd.DataFrame({"accident_id": matrix["accident_id"],
                  "component_count": accident_count,
                  "role_count": role_count}).to_csv(
        output / "accident_component_counts.csv", index=False)
    cooccurrence = values.T.dot(values)
    cooccurrence.to_csv(output / "component_cooccurrence.csv", index=True)
    pd.DataFrame(dropped, columns=["factor_id", "role", "label", "accident_count"]).to_csv(
        output / "excluded_components.csv", index=False, encoding="utf-8")
    write_review_samples(units, matches, definitions, output)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    curated = {"status": "complete", "method": "role_specific_explicit_text_recoding",
               "review_scope": "rules_and_stratified_samples; no expert adjudication claimed",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_run": str(source), "source": provenance,
               "definitions_sha256": sha256(codebook),
               "code_sha256": sha256(Path(__file__)),
               "min_component_accidents": args.min_component_accidents,
               "matrix_sha256": sha256(output / "accident_factor_matrix.csv"),
               "dictionary_sha256": sha256(output / "factor_dictionary.csv"),
               "unit_audit_sha256": sha256(output / "unit_component_audit.csv")}
    (output / "curation_manifest.json").write_text(
        json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"status": "complete",
        "kind": "curated_accident_component_matrix", "source_run": str(source),
        "curation_manifest_sha256": sha256(output / "curation_manifest.json")}, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Curated matrix: {output}")


if __name__ == "__main__":
    main()
