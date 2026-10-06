"""Name selected accident factors after discovery, without changing the partition.

The prompts and output columns follow the results notebook.  A content hash is
stored with each label so a topic number reused by a later fit cannot silently
inherit an unrelated label.  Successful calls are saved immediately, allowing a
Slurm job interrupted during annotation to resume without repeating them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from llm_labeling import (
    complete_theme_label_json,
    extract_theme_item,
    is_valid_llm_cache_row,
    normalize_llm_fields,
    parse_llm_payload,
)
from manuscript_reporting import sanitize_label_text

TEXT_ROOT = Path(__file__).resolve().parent.parent
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scgm_text.openai_theme_labels import _get_client, load_openai_dotenv


ROLE_PROMPTS = {
    "A0": """You name A0 semantic factors (work situation before the accident). Each label must denote the dominant pre-accident work situation: a concrete activity, workstation type, equipment, location, or operational setup. Do not label victim age, seniority, occupation alone, uncertainty about the accident, witness availability, emergency response, or post-accident actions as A0 unless they genuinely describe the dominant pre-accident context. Forbidden: meta or vague labels (\"work context\", \"work environment\", \"professional activity\", \"work in general\"). No consequence or accident event.""",
    "A1": """You name A1 semantic factors (adverse condition or hazard before the event). Each label must name the dominant hazard, failure, missing protection, or dangerous condition present before the event. Do not describe an unsafe action occurring during the event as an A1 condition. Forbidden: meta labels (\"adverse factor\", \"dangerous condition\", \"risk\", \"general hazard\"). Not the event or injury unless indispensable to distinguish the condition.""",
    "B": """You name B semantic factors (immediate event or mechanism). Each label must describe an actual change, deviation, loss of control, contact, fall, ignition, entrapment, unintended movement, or other immediate mechanism. A routine activity such as welding, cutting, grinding, machining, or driving is not by itself an event/deviation. If the examples mainly describe an activity rather than an event, retain the best factual label but set role_fit to \"partial\" or \"poor\". Forbidden: meta labels (\"accident event\", \"accident\", \"incident\", \"general deviation\"). Not upstream context or final injury.""",
    "C": """You name C semantic factors (consequence or injury). Each label must specify the dominant harm, affected body region, or severity when consistently supported. Do not create artificial distinctions based only on wording, hospitalization, resuscitation, reporting, or timing of death when the underlying consequence is the same; different clusters may legitimately receive similar or identical labels. Forbidden: meta labels (\"consequence\", \"injury\", \"lesion\", \"damage\" alone). Not the cause or work context.""",
}


def check_annotation_credentials(config: Mapping[str, Any]) -> None:
    """Fail before the expensive discovery stage if annotation cannot run."""
    if not config.get("factor_annotation", {}).get("enabled", False):
        return
    load_openai_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "factor_annotation.enabled is true but OPENAI_API_KEY is missing. "
            "Add it to text/.env before submitting STAGE=all."
        )


def _examples(row: Mapping[str, Any], limit: int) -> dict[str, list[str]]:
    def split(value: Any) -> list[str]:
        value = str(value or "")
        return [] if value.lower() == "nan" else [part.strip() for part in value.split(" || ") if part.strip()]

    central = split(row.get("central_sentence", ""))[:1]
    representatives = [value for value in split(row.get("representative_sentences", "")) if value not in central]
    boundary = [value for value in split(row.get("boundary_sentences", "")) if value not in central and value not in representatives]
    boundary_budget = min(len(boundary), max(1, limit // 4))
    selected_representatives = representatives[: max(0, limit - len(central) - boundary_budget)]
    remaining = limit - len(central) - len(selected_representatives)
    return {
        "central": central,
        "representative": selected_representatives,
        "boundary": boundary[: min(len(boundary), boundary_budget + remaining)],
    }


def _catalog(prepared, selections: Mapping[str, str], config: Mapping[str, Any], run_dir: Path) -> pd.DataFrame:
    from scenario_pipeline import PartitionResult, ROLES, build_topic_dictionary

    annotation_dir = run_dir / "topics_manual"
    partitions: dict[str, PartitionResult] = {}
    representative_rows: list[dict[str, Any]] = []
    limit = int(config.get("factor_annotation", {}).get("representative_sentences", 50))
    for role in ROLES:
        selected_dir = run_dir / "discovery" / role / "selected"
        assignments = pd.read_csv(selected_dir / "topic_assignments.csv", dtype={"fact_id": str}).fillna({"topic_id": ""})
        role_units = prepared.units.loc[prepared.units["_role"].eq(role)].reset_index(drop=True)
        if len(assignments) != len(role_units) or not assignments["fact_id"].astype(str).reset_index(drop=True).eq(role_units["_fact_id"].astype(str)).all():
            raise ValueError(f"Selected factor assignments no longer align with factual units for {role}")
        if not assignments["configuration_id"].astype(str).eq(str(selections[role])).all():
            raise ValueError(f"Selected configuration mismatch for {role}")
        topics = pd.read_csv(selected_dir / "topics.csv")
        partitions[role] = PartitionResult(role=role, assignments=assignments, topics=topics,
                                            edges=pd.DataFrame(), replications=pd.DataFrame())
        for topic_id, subset in assignments.loc[assignments["topic_id"].astype(bool)].groupby("topic_id", sort=True):
            strongest = subset.sort_values("membership_strength", ascending=False).head(limit)
            for row in strongest.itertuples(index=False):
                representative_rows.append({
                    "role": role,
                    "configuration_id": str(selections[role]),
                    "topic_id": str(topic_id),
                    "fact_id": str(row.fact_id),
                    "accident_id": str(row.accident_id),
                    "sentence": str(row.sentence),
                    "membership_strength": float(row.membership_strength),
                })

    dictionary = build_topic_dictionary(prepared, partitions, config, annotation_dir)
    representatives = pd.DataFrame(representative_rows)
    representatives.to_csv(annotation_dir / "representatives_by_membership.csv", index=False)
    if dictionary.empty:
        raise ValueError("No selected factors available for annotation")
    lookup = representatives.groupby("topic_id")["sentence"].apply(
        lambda values: " || ".join(values.astype(str).tolist())
    ).to_dict()
    dictionary["representative_sentences"] = dictionary["topic_id"].map(lookup).fillna(
        dictionary.get("representative_sentences", "")
    )
    dictionary["selected_configuration_id"] = dictionary["role"].map(selections)
    dictionary.to_csv(annotation_dir / "topic_dictionary.csv", index=False)
    catalog = dictionary.copy()
    catalog["configuration_id"] = catalog["role"].map(selections)
    catalog.to_csv(annotation_dir / "topic_dictionary_all_selected.csv", index=False)
    return catalog


def _record(row: Mapping[str, Any], *, language: str, prompt_version: str,
            model: str, limit: int) -> dict[str, Any]:
    role = str(row["role"])
    examples = _examples(row, limit)
    record = {
        "topic_id": str(row["topic_id"]),
        "role": role,
        "configuration_id": str(row["configuration_id"]),
        "top_words": str(row.get("top_terms", row.get("label", ""))),
        "examples": examples,
    }
    source = {
        "fact_ids": sorted(str(value) for value in row["fact_ids"]),
        "record": record,
        "role_prompt": ROLE_PROMPTS[role],
        "language": language,
        "prompt_version": prompt_version,
        "model": model,
    }
    record["source_fingerprint"] = hashlib.sha256(
        json.dumps(source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return record


def _request(client, record: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, str]:
    role = record["role"]
    language = str(settings.get("output_language", "English"))
    instruction = f'''
Return a single JSON object with a `themes` array containing one element per `topic_id`.
Write the label, description and evidence in {language}.

The factual-unit cluster has already been constructed and must not be modified.
Your task is only to describe its dominant semantic content.

Use the sentences as the primary evidence. Keywords are auxiliary cues only and
must not override the dominant meaning of the examples. Central and
representative sentences define the main semantic motif. Boundary sentences are
provided mainly to assess scope and heterogeneity; they must not determine the
label unless they are consistent with the central examples.

Rules for `label`:
- 4 to 10 words, nominal phrase suitable for a figure legend.
- Name the dominant factual motif visible across the examples (object, action,
  equipment, location, mechanism or injury depending on the role).
- Forbidden: generic labels that do not discriminate themes (e.g. only the role name,
  "work context", "adverse factor", "event", "consequence", "accident", "risk").
- Do not select a narrow sub-theme supported by only a few examples.
- If no specific motif dominates, use a broader factual label rather than inventing
  specificity. Labels do not need to be unique across topics.
- Do not invent information absent from the provided material.

`description`: one sentence explaining what is recurrent across the cluster;
do not imply causality.

`evidence`: 2 to 4 short fragments that directly support the dominant motif.
Use fragments from distinct accident narratives whenever possible.

`role_fit`: `good` when the dominant motif clearly corresponds to the requested
accident-process role; `partial` when it is coherent but only partly corresponds;
`poor` when it mostly belongs to another role or describes narrative/reporting
information rather than the requested process component.

`heterogeneity`: `low`, `moderate`, or `high`, according to how consistently the
examples support one dominant factual motif.

Role-specific guidance:
{ROLE_PROMPTS[role]}

Topics to analyse:
{json.dumps([{key: record[key] for key in ("topic_id", "configuration_id", "top_words", "examples")}], ensure_ascii=False, indent=2)}
'''
    kwargs = {
        "model": str(settings.get("model", "gpt-5.6-luna")),
        "reasoning_effort": str(settings.get("reasoning_effort", "low")),
        "max_output_tokens": int(settings.get("max_output_tokens", 4000)),
    }
    messages = [
        {"role": "system", "content": "You are a careful occupational-accident analyst. Return valid JSON only."},
        {"role": "user", "content": instruction},
    ]
    items = parse_llm_payload(complete_theme_label_json(client, messages=messages, **kwargs))
    fields = normalize_llm_fields(extract_theme_item(items, record))
    if not fields["llm_label"]:
        retry = f'''Return JSON {{"themes": [{{"topic_id": "{record["topic_id"]}", "label": "...", "description": "...", "evidence": ["..."], "role_fit": "good|partial|poor", "heterogeneity": "low|moderate|high"}}]}}.
The label field must contain 4 to 10 words in {language}, factual and specific.
topic_id={record["topic_id"]}
top_words={json.dumps(record["top_words"], ensure_ascii=False)}
examples={json.dumps(record["examples"], ensure_ascii=False)}'''
        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": retry},
        ]
        items = parse_llm_payload(complete_theme_label_json(client, messages=messages, **kwargs))
        fields = normalize_llm_fields(extract_theme_item(items, record))
    if not fields["llm_label"] or not fields["llm_role_fit"] or not fields["llm_heterogeneity"]:
        raise ValueError(f"Incomplete label response for {role}/{record['topic_id']}")
    return fields


def _write_cache(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["topic_id", "role", "configuration_id", "llm_label", "llm_description",
               "llm_evidence", "llm_role_fit", "llm_heterogeneity", "prompt_version",
               "model", "source_fingerprint"]
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
    temporary.replace(path)


def annotate_selected_factors(prepared, selections: Mapping[str, str], config: Mapping[str, Any],
                              run_dir: Path, *, requester: Callable | None = None) -> Path:
    """Write a newly labelled selected-factor catalog; resume valid cached rows."""
    check_annotation_credentials(config)
    settings = config.get("factor_annotation", {})
    run_dir = Path(run_dir)
    catalog = _catalog(prepared, selections, config, run_dir)
    annotation_dir = run_dir / "topics_manual"
    cache_path = annotation_dir / "llm_theme_labels.csv"
    old = pd.read_csv(cache_path).fillna("") if cache_path.is_file() else pd.DataFrame()
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in old.to_dict("records"):
        key = (str(row.get("role", "")), str(row.get("configuration_id", "")), str(row.get("topic_id", "")))
        cache[key] = row

    assignments_by_role = {}
    for role in ROLE_PROMPTS:
        path = run_dir / "discovery" / role / "selected" / "topic_assignments.csv"
        assignments_by_role[role] = pd.read_csv(path, dtype={"fact_id": str}).fillna({"topic_id": ""})

    limit = int(settings.get("representative_sentences", 50))
    language = str(settings.get("output_language", "English"))
    version = str(settings.get("prompt_version", "dominant-motif-role-fit-v1"))
    model = str(settings.get("model", "gpt-5.6-luna"))
    client = _get_client() if requester is None else None
    requested = requester or (lambda record: _request(client, record, settings))
    current_keys = []
    errors = []
    new_count = 0
    reused_count = 0
    for row in catalog.to_dict("records"):
        role = str(row["role"])
        topic_id = str(row["topic_id"])
        config_id = str(row["configuration_id"])
        key = (role, config_id, topic_id)
        current_keys.append(key)
        fact_ids = assignments_by_role[role].loc[
            assignments_by_role[role]["topic_id"].eq(topic_id), "fact_id"
        ].astype(str).tolist()
        record = _record({**row, "fact_ids": fact_ids}, language=language,
                         prompt_version=version, model=model, limit=limit)
        cached = cache.get(key, {})
        if (cached.get("prompt_version") == version
                and cached.get("model") == model
                and cached.get("source_fingerprint") == record["source_fingerprint"]
                and is_valid_llm_cache_row(cached)
                and normalize_llm_fields(cached)["llm_role_fit"]
                and normalize_llm_fields(cached)["llm_heterogeneity"]):
            reused_count += 1
            continue
        try:
            fields = normalize_llm_fields(requested(record))
            if not fields.get("llm_label") or not fields.get("llm_role_fit") or not fields.get("llm_heterogeneity"):
                raise ValueError("LLM response lacks a label, role fit, or heterogeneity")
            cache[key] = {
                "topic_id": topic_id, "role": role, "configuration_id": config_id,
                **fields, "prompt_version": version, "model": model,
                "source_fingerprint": record["source_fingerprint"],
            }
            _write_cache(list(cache.values()), cache_path)
            new_count += 1
            print(f"[factor-annotation] {role}/{topic_id}: {fields['llm_label']}", flush=True)
        except Exception as error:
            errors.append(f"{role}/{topic_id}: {error}")
            print(f"[factor-annotation] failed {role}/{topic_id}: {error}", flush=True)

    status = {"n_factors": len(current_keys), "new_labels": new_count,
              "cache_hits": reused_count, "errors": errors, "model": model,
              "prompt_version": version}
    (annotation_dir / "llm_theme_labels_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"Factor annotation incomplete ({len(errors)} failures); rerun STAGE=all without REESTIMATE=1 to resume")
    labels = pd.DataFrame([cache[key] for key in current_keys])
    _write_cache(labels.to_dict("records"), cache_path)
    merged = catalog.merge(labels, on=["topic_id", "role", "configuration_id"], how="left", validate="one_to_one")
    merged["plot_label"] = merged["llm_label"].map(sanitize_label_text)
    result = annotation_dir / "topic_dictionary_with_llm_labels.csv"
    merged.to_csv(result, index=False)
    print(f"[factor-annotation] {len(labels)} selected factors labelled: {result}", flush=True)
    return result


def main() -> None:
    """Resume annotation alone when discovery already finished."""
    from scenario_pipeline import (
        PreparedData, load_bn_analysis_config, load_embeddings,
        load_selected_configurations, load_units,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = load_bn_analysis_config(args.config.resolve(), args.dataset, run_dir)
    check_annotation_credentials(config)
    units, _ = load_units(config)
    embeddings = load_embeddings(config, units, run_dir / "embeddings")
    prepared = PreparedData(units=units, embeddings=embeddings, input_summary=pd.DataFrame())
    selections = load_selected_configurations(run_dir)
    annotate_selected_factors(prepared, selections, config, run_dir)


if __name__ == "__main__":
    main()
