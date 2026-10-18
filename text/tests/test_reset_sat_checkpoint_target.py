"""The Slurm restart must remove only the requested experiment target."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import reset_sat_checkpoint_target as reset


class ResetTargetTests(unittest.TestCase):
    def test_removes_only_selected_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "manifest.json").write_text(
                json.dumps({"fold_order": list(reset.CORPORA)}), encoding="utf-8")
            runs = root / "safer_sat_lodo" / "runs" / reset.RUN_NAME
            metallurgy = runs / "target_metallurgie"
            btp = runs / "target_btp"
            metallurgy.mkdir(parents=True)
            btp.mkdir()
            (metallurgy / "partial_checkpoint.bin").write_bytes(b"partial")
            (btp / "completed.txt").write_text("keep", encoding="utf-8")
            self.assertEqual(reset.reset_target(bundle, runs, 1), metallurgy)
            self.assertFalse(metallurgy.exists())
            self.assertEqual((btp / "completed.txt").read_text(encoding="utf-8"), "keep")
            # Repeating the reset before the trainer recreates the folder is safe.
            self.assertEqual(reset.reset_target(bundle, runs, 1), metallurgy)

    def test_rejects_unrelated_root_and_changed_corpus_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = bundle / "manifest.json"
            manifest.write_text(json.dumps({"fold_order": list(reset.CORPORA)}), encoding="utf-8")
            wrong = root / "other_experiment" / "runs" / reset.RUN_NAME
            target = wrong / "target_metallurgie"
            target.mkdir(parents=True)
            (target / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected checkpoint-selection run root"):
                reset.reset_target(bundle, wrong, 1)
            self.assertTrue((target / "keep.txt").exists())
            manifest.write_text(json.dumps({"fold_order": list(reversed(reset.CORPORA))}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Bundle corpus order"):
                reset.reset_target(bundle, wrong, 1)
            self.assertTrue((target / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
