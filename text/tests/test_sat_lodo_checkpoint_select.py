"""Checkpoint conversion and selection for the SaT RQ1 rerun."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "scripts"))

import sat_lodo_checkpoint_select as selection  # noqa: E402


class CheckpointSelectionTests(unittest.TestCase):
    def test_checkpoint_and_threshold_tie_break(self) -> None:
        rows = [
            {"candidate": "checkpoint-500", "step": 500, "threshold": 0.1,
             "source_macro_span_f1": 0.3},
            {"candidate": "checkpoint-1000", "step": 1000, "threshold": 0.25,
             "source_macro_span_f1": 0.3},
            {"candidate": "checkpoint-500", "step": 500, "threshold": 0.25,
             "source_macro_span_f1": 0.3},
            {"candidate": "checkpoint-500", "step": 500, "threshold": 0.3,
             "source_macro_span_f1": 0.2},
        ]
        self.assertEqual(selection.best_row(rows)["candidate"], "checkpoint-500")
        self.assertEqual(selection.best_row(rows)["threshold"], 0.25)

    def test_curve_matches_existing_source_validation_rule(self) -> None:
        bundle = selection.base.BUNDLE
        folder = bundle / "folds" / "target_btp"
        val = selection.base.read_jsonl(folder / "val.jsonl")
        sample = [next(row for row in val if row["corpus"] == source and row["units"])
                  for source in selection.base.CORPORA if source != "btp"]
        with closing(sqlite3.connect(bundle / "base_probabilities.sqlite3")) as db:
            curve = selection.validation_curve(sample, db, "base", 0, 0.0)
            chosen, existing = selection.base.choose_thresholds(sample, db, db)
        self.assertEqual(selection.best_row(curve)["threshold"], chosen["M1"])
        base_curve = {row["threshold"]: row["source_macro_span_f1"]
                      for row in existing if row["model"] == "M1"}
        self.assertEqual({row["threshold"]: row["source_macro_span_f1"] for row in curve},
                         base_curve)

    def test_checkpoint_export_keeps_lora_and_head_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-500"
            final = root / "final"
            checkpoint.mkdir()
            final.mkdir()
            adapter = {"roberta.encoder.layer.0.loras.text.lora_A.weight": torch.ones(2, 3)}
            head = {"classifier.weight": torch.ones(1, 3),
                    "classifier.bias": torch.zeros(1)}
            torch.save({"backbone." + key: value for key, value in (adapter | head).items()},
                       checkpoint / "pytorch_model.bin")
            torch.save(adapter, final / "pytorch_adapter.bin")
            torch.save(head, final / "pytorch_model_head.bin")
            (final / "adapter_config.json").write_text("{}", encoding="utf-8")
            (final / "head_config.json").write_text("{}", encoding="utf-8")
            exported = root / "exported"
            first = selection.export_checkpoint_adapter(checkpoint, exported, final)
            self.assertEqual(first, selection.export_checkpoint_adapter(checkpoint, exported, final))
            self.assertTrue(selection.complete_adapter(exported))
            self.assertTrue(torch.equal(torch.load(exported / "pytorch_adapter.bin",
                                                  weights_only=True)[next(iter(adapter))],
                                        next(iter(adapter.values()))))
            self.assertEqual(set(torch.load(exported / "pytorch_model_head.bin",
                                                weights_only=True)), set(head))

    def test_missing_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Path(temporary)
            (trainer / "trainer_state.json").write_text(
                json.dumps({"global_step": 1000, "epoch": 2}), encoding="utf-8")
            folder = trainer / "checkpoint-1000"
            folder.mkdir()
            (folder / "pytorch_model.bin").touch()
            (folder / "trainer_state.json").write_text(
                json.dumps({"epoch": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing checkpoints"):
                selection.training_candidates(trainer)

    def test_final_adapter_is_a_candidate_between_save_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Path(temporary)
            (trainer / "trainer_state.json").write_text(
                json.dumps({"global_step": 750, "epoch": 3.0}), encoding="utf-8")
            folder = trainer / "checkpoint-500"
            folder.mkdir()
            (folder / "pytorch_model.bin").write_bytes(b"saved")
            (folder / "trainer_state.json").write_text(
                json.dumps({"epoch": 2.0}), encoding="utf-8")
            self.assertEqual([(row["candidate"], row["step"])
                              for row in selection.training_candidates(trainer)],
                             [("checkpoint-500", 500), ("final", 750)])

    def test_resume_uses_latest_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Path(temporary)
            for step in (500, 1000):
                folder = trainer / f"checkpoint-{step}"
                folder.mkdir()
                for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt",
                             "trainer_state.json"):
                    (folder / name).write_bytes(b"saved")
            (trainer / "checkpoint-1000" / "optimizer.pt").unlink()
            self.assertEqual(selection.latest_resumable_checkpoint(trainer).name,
                             "checkpoint-500")


if __name__ == "__main__":
    unittest.main()
