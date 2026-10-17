"""Meaningful data-integrity checks for the six-domain SaT experiment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sat_lodo.py"
SPEC = importlib.util.spec_from_file_location("sat_lodo", SCRIPT)
assert SPEC and SPEC.loader
sat_lodo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sat_lodo)


def annotation(raw: str, ident: str) -> dict:
    first = "Un salarié travaillait"
    second = "Puis il est tombé"
    first_start = raw.index(first)
    second_start = raw.index(second)
    return {"document_id": ident, "annotation_source": "OPENAI_BATCH_PREANNOTATION",
            "guide_version": "narrative_v2", "issues": [], "excluded_spans": [],
            "units": [
                {"start": first_start, "end": first_start + len(first), "text": first,
                 "role": "A0", "assertion": "AFFIRME", "uncertain": False},
                {"start": second_start, "end": second_start + len(second), "text": second,
                 "role": "B", "assertion": "AFFIRME", "uncertain": False}]}


def fixture_corpora(root: Path) -> None:
    for corpus in sat_lodo.CORPORA:
        docs = root / "data" / corpus / "documents.jsonl"
        anns = root / "preannotations" / corpus
        docs.parent.mkdir(parents=True)
        anns.mkdir(parents=True)
        with docs.open("w", encoding="utf-8") as stream:
            for n in range(5):
                ident = f"{corpus}_{n}"
                gap = ".\n" if corpus == "nicollin" else ". "
                raw = f"Un salarié travaillait{gap}Puis il est tombé."
                stream.write(json.dumps({"document_id": ident, "raw_text": raw},
                                        ensure_ascii=False) + "\n")
                (anns / f"{ident}.json").write_text(
                    json.dumps(annotation(raw, ident), ensure_ascii=False), encoding="utf-8")


def test_eligibility_rejects_omissions_and_uncertainty() -> None:
    raw = "Un salarié travaillait. Puis il est tombé."
    ann = annotation(raw, "x")
    assert sat_lodo.validate_record(raw, ann) is None
    chunks, replacements = sat_lodo.training_chunks(raw, ann["units"])
    assert "".join(chunks) == raw and replacements == 0
    assert chunks[0] == "Un salarié travaillait"
    assert chunks[1].startswith(". Puis")
    ann["units"][1]["start"] += 1
    assert sat_lodo.validate_record(raw, ann) == "inexact_text"
    ann = annotation(raw, "x")
    ann["units"][0]["uncertain"] = True
    assert sat_lodo.validate_record(raw, ann) == "uncertain_unit"
    ann = annotation(raw, "x")
    ann["units"].pop()
    assert sat_lodo.validate_record(raw, ann) == "informative_gap"


def test_prepare_is_deterministic_and_has_no_target_leakage(tmp_path: Path) -> None:
    import torch

    root = tmp_path / "annotation"
    fixture_corpora(root)
    out = tmp_path / "bundle"
    args = SimpleNamespace(annotation_root=str(root), allow_fixture_counts=True,
                           out=str(out), seed=42)
    sat_lodo.prepare(args)
    manifest = sat_lodo.verify_bundle(out)
    assert manifest["seed"] == 42
    assert all(manifest["source_report"][c]["eligible"] == 5 for c in sat_lodo.CORPORA)
    assert manifest["source_report"]["nicollin"]["training_newline_characters_replaced"] == 5
    for target in sat_lodo.CORPORA:
        folder = out / "folds" / f"target_{target}"
        train = sat_lodo.read_jsonl(folder / "train.jsonl")
        val = sat_lodo.read_jsonl(folder / "val.jsonl")
        test = sat_lodo.read_jsonl(folder / "test.jsonl")
        assert len(train) == 20 and len(val) == 5 and len(test) == 5
        assert all(row["corpus"] != target for row in train + val)
        assert all(row["corpus"] == target for row in test)
        assert set(r["document_id"] for r in train).isdisjoint(r["document_id"] for r in val)
        packed = torch.load(folder / "training.pth", weights_only=False)
        section = packed["fr"]["sentence"][sat_lodo.DATASET_NAME]
        assert len(section["meta"]["train_data"]) == len(train)
        assert len(section["data"]) == len(val)
        assert all("\n" not in chunk for narrative in section["meta"]["train_data"]
                   for chunk in narrative)
    ids = sat_lodo.partition([f"id{i}" for i in range(101)], "btp", 42)
    assert ids == sat_lodo.partition([f"id{i}" for i in reversed(range(101))], "btp", 42)
    assert len(ids[0]) == 80 and len(ids[1]) == 21
    target_folder = out / "folds" / "target_btp"
    (target_folder / "test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modified bundle file"):
        sat_lodo.verify_bundle(out)


def test_lora_config_keeps_the_pretrained_single_output_head(tmp_path: Path, monkeypatch) -> None:
    config = sat_lodo.base_config(tmp_path, "nicollin", tmp_path / "trainer")
    assert config["model_name_or_path"] == "segment-any-text/sat-12l-sm"
    assert config["use_auxiliary"] is False
    assert config["do_auxiliary_training"] is False
    assert not config.get("clf_from_scratch", False)

    wrong_template = tmp_path / "wrong_template.json"
    wrong_template.write_text(json.dumps({**config, "use_auxiliary": True}), encoding="utf-8")
    monkeypatch.setattr(sat_lodo, "TEMPLATE", wrong_template)
    with pytest.raises(ValueError, match="labels"):
        sat_lodo.base_config(tmp_path, "nicollin", tmp_path / "trainer")


def test_preflight_blocks_missing_sixth_domain(tmp_path: Path) -> None:
    root = tmp_path / "annotation"
    fixture_corpora(root)
    folder = root / "preannotations" / "service_temporaire"
    for path in folder.glob("*.json"):
        ann = json.loads(path.read_text(encoding="utf-8"))
        ann["annotation_source"] = "LOCAL_RULE_ASSISTED_PREANNOTATION"
        path.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    args = SimpleNamespace(annotation_root=str(root), allow_fixture_counts=True)
    with pytest.raises(RuntimeError, match="service_temporaire"):
        sat_lodo.preflight(args)


def test_vendored_trainer_detects_changed_or_extra_files(tmp_path: Path) -> None:
    repo = tmp_path / "wtpsplit"
    script = repo / "wtpsplit" / "train" / "train_lora.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('trainer')\n", encoding="utf-8")
    revision = sat_lodo.VENDOR_REVISION
    (repo / "SAFER_VENDOR.json").write_text(json.dumps({
        "revision": revision,
        "files": {"wtpsplit/train/train_lora.py": sat_lodo.digest(script)},
    }), encoding="utf-8")
    assert sat_lodo.git_revision(repo) == revision
    script.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        sat_lodo.git_revision(repo)
    script.write_text("print('trainer')\n", encoding="utf-8")
    (repo / "unexpected.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="files differ"):
        sat_lodo.git_revision(repo)


def test_vendor_hashes_accept_only_line_ending_conversion_for_text(tmp_path: Path) -> None:
    import hashlib

    repo = tmp_path / "wtpsplit"
    repo.mkdir()
    text_file = repo / ".gitattributes"
    binary_file = repo / "image.png"
    text_file.write_bytes(b"first\r\nsecond\r\n")
    binary_file.write_bytes(b"\x89PNG\x00\r\n")
    manifest = {"revision": sat_lodo.VENDOR_REVISION,
                "hash_mode": "lf_normalized_text", "binary_files": ["image.png"],
                "files": {
                    ".gitattributes": hashlib.sha256(b"first\nsecond\n").hexdigest(),
                    "image.png": sat_lodo.digest(binary_file)}}
    (repo / "SAFER_VENDOR.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert sat_lodo.git_revision(repo) == sat_lodo.VENDOR_REVISION
    text_file.write_bytes(b"first\nsecond\n")
    assert sat_lodo.git_revision(repo) == sat_lodo.VENDOR_REVISION
    text_file.write_bytes(b"first\nchanged\n")
    with pytest.raises(RuntimeError, match="changed"):
        sat_lodo.git_revision(repo)
    text_file.write_bytes(b"first\nsecond\n")
    binary_file.write_bytes(b"\x89PNG\x00\n")
    with pytest.raises(RuntimeError, match="changed"):
        sat_lodo.git_revision(repo)


def test_exact_metric_and_threshold_selection_use_only_validation(tmp_path: Path) -> None:
    import sqlite3

    raw = "Un salarié travaillait. Puis il est tombé."
    ann = annotation(raw, "x")
    row = {"corpus": "btp", "document_id": "x", "raw_text": raw,
           "units": ann["units"]}
    db = sqlite3.connect(tmp_path / "predictions.sqlite3")
    db.execute("CREATE TABLE predictions (model TEXT, corpus TEXT, document_id TEXT, spans_json TEXT)")
    ref = [[u["start"], u["end"]] for u in ann["units"]]
    spans = {f"{threshold:.2f}": (ref if threshold == 0.25 else [[0, len(raw)]])
             for threshold in sat_lodo.THRESHOLDS}
    db.execute("INSERT INTO predictions VALUES(?,?,?,?)",
               ("base", "btp", "x", json.dumps(spans)))
    selected, curve = sat_lodo.select_threshold([row], db, "base")
    assert selected == 0.25 and len(curve) == len(sat_lodo.THRESHOLDS)
    assert sat_lodo.counts_for([row], db, "base", 0.25)["exact_f1"] == 1
    assert sat_lodo.counts_for([row], db, "base", 0.10)["exact_f1"] == 0
    db.close()


def test_export_loss_history_keeps_train_and_validation_values(tmp_path: Path) -> None:
    import csv

    state = tmp_path / "trainer_state.json"
    state.write_text(json.dumps({"log_history": [
        {"step": 50, "epoch": 0.5, "loss": 0.8, "learning_rate": 0.0002},
        {"step": 100, "epoch": 1.0, "eval_loss": 0.6,
         "eval_safer_lodo/fr/loss": 0.6, "eval_runtime": 2.0},
        {"step": 100, "epoch": 1.0, "train_loss": 0.7},
    ]}), encoding="utf-8")
    output = tmp_path / "loss_history.csv"
    sat_lodo.export_loss_history(state, output)
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["step"], row["metric"], row["value"]) for row in rows] == [
        ("50", "loss", "0.8"), ("100", "eval_loss", "0.6"),
        ("100", "eval_safer_lodo/fr/loss", "0.6"), ("100", "train_loss", "0.7")]
