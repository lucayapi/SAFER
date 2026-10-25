"""Source isolation and validation checks for the Nicollin ablation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "scripts"))

import sat_elementary_no_nicollin as experiment  # noqa: E402


class NoNicollinTests(unittest.TestCase):
    def test_all_six_targets_remain_and_nicollin_is_never_a_source(self):
        self.assertEqual(len(experiment.base.CORPORA), 6)
        for target in experiment.base.CORPORA:
            sources = experiment.source_domains(target)
            self.assertNotIn(target, sources)
            self.assertNotIn("nicollin", sources)
            self.assertEqual(len(sources), 5 if target == "nicollin" else 4)

    def test_source_validation_rejects_nicollin(self):
        sources = experiment.source_domains("btp")
        candidate = {"candidate": "final", "step": 1000, "epoch": 1.0}
        rows = [{"corpus": source} for source in sources] + [{"corpus": "nicollin"}]
        with patch.object(experiment.base, "get_probs") as probabilities:
            with self.assertRaisesRegex(ValueError, "forbidden source"):
                experiment.validation_curve(rows, object(), candidate, sources)
        probabilities.assert_not_called()

    def test_prepare_uses_only_allowed_source_domains(self):
        bundle = experiment.elementary.BASE_BUNDLE
        if not (bundle / "complete.json").is_file():
            self.skipTest("The private GOLD bundle is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared"
            experiment.prepare(bundle, prepared)
            manifest = experiment.check_prepared(bundle, prepared)
            for target in experiment.base.CORPORA:
                report = manifest["source_report"][target]
                self.assertEqual(report["sources"], list(experiment.source_domains(target)))
                for partition in ("train", "val"):
                    self.assertNotIn("nicollin", report[partition])
                    self.assertEqual(set(report[partition]), set(report["sources"]))


if __name__ == "__main__":
    unittest.main()
