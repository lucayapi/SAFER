"""Checks for the isolated full-parameter elementary SaT campaign."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "scripts"))

import sat_elementary_full3_lodo as experiment  # noqa: E402


class FullThreeEpochTests(unittest.TestCase):
    def test_config_uses_full_model_and_three_epochs(self):
        with patch.object(experiment.base, "trainer_config", return_value={}):
            config = experiment.config_for(Path("bundle"), "btp", Path("trainer"),
                                           Path("model"), Path("tokenizer"), Path("data"))
        self.assertIs(config["train_adapter"], False)
        self.assertEqual(config["num_train_epochs"], 3)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["save_total_limit"], 3)
        self.assertLess(config["learning_rate"], 3e-4)

    def test_source_selection_prefers_best_epoch_then_earlier_tie(self):
        rows = [{"candidate": "epoch_1", "epoch": 1, "threshold": 0.25,
                 "macro_elementary_exact_f1": 0.51},
                {"candidate": "epoch_2", "epoch": 2, "threshold": 0.25,
                 "macro_elementary_exact_f1": 0.52},
                {"candidate": "epoch_3", "epoch": 3, "threshold": 0.25,
                 "macro_elementary_exact_f1": 0.52}]
        self.assertEqual(experiment.best_candidate(rows)["candidate"], "epoch_2")

    def test_requires_exactly_three_completed_epoch_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Path(directory)
            for epoch in (1, 2, 3):
                folder = trainer / f"checkpoint-{epoch * 10}"
                folder.mkdir()
                (folder / "trainer_state.json").write_text(
                    json.dumps({"global_step": epoch * 10, "epoch": float(epoch)}),
                    encoding="utf-8")
                (folder / "model.safetensors").write_bytes(b"weights")
            self.assertEqual([e for e, _ in experiment.epoch_checkpoints(trainer)], [1, 2, 3])
            (trainer / "checkpoint-20" / "trainer_state.json").unlink()
            with self.assertRaises(ValueError):
                experiment.epoch_checkpoints(trainer)


if __name__ == "__main__":
    unittest.main()
