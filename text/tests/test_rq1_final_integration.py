"""Optional full report fixture using old caches as *synthetic* LODO inputs.

This test never publishes its scores; it only checks the report's I/O contract.
Run explicitly with RUN_RQ1_FINAL_INTEGRATION=1 after the local bundle exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rq1_final as campaign
import rq1_final_report as report
import sat_lodo_gold_v3 as sat


@unittest.skipUnless(os.environ.get("RUN_RQ1_FINAL_INTEGRATION") == "1" and
                     campaign.OUTPUT.is_dir(), "Explicit local integration run only")
class FinalReportIntegrationTests(unittest.TestCase):
    def test_complete_report_uses_only_common_tests(self):
        bundle = campaign.OUTPUT
        manifest = campaign.check_bundle(bundle)
        with tempfile.TemporaryDirectory(prefix="rq1_final_fixture_") as temporary:
            root = Path(temporary)
            lodo, out = root / "synthetic_lodo", root / "report"
            summary = []
            baseline = json.loads((report.BASELINE_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
            source_thresholds = {row["target"]: row["M2_threshold"] for row in baseline["thresholds"]}
            with sqlite3.connect(bundle / "base_probabilities.sqlite3") as baseline:
                for target in sat.CORPORA:
                    run = lodo / f"target_{target}"
                    run.mkdir(parents=True)
                    source_run = campaign.SINGLE_RUNS / f"source_{target}"
                    old_selection = json.loads((source_run / "selection.json").read_text(encoding="utf-8"))
                    rows = sat.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
                    chosen = {"M2_threshold": source_thresholds[target],
                              "M3_threshold": old_selection["M2_threshold"],
                              "M3_candidate": old_selection["M2_candidate"]}
                    adapter = run / "selected_adapter"
                    adapter.mkdir()
                    for filename in ("adapter_config.json", "head_config.json",
                                     "pytorch_adapter.bin", "pytorch_model_head.bin"):
                        (adapter / filename).write_bytes((target + filename).encode("utf-8"))
                    chosen["M3_weights"] = {
                        filename: sat.digest(adapter / filename)
                        for filename in ("pytorch_adapter.bin", "pytorch_model_head.bin")}
                    campaign.json_file(run / "selection.json", chosen)
                    predictions = {
                        "M1": report.probability_predictions(baseline, rows, sat.DEFAULT),
                        "M2": report.probability_predictions(baseline, rows, chosen["M2_threshold"]),
                    }
                    with sqlite3.connect(source_run / "heldout_probabilities.sqlite3") as adapted:
                        predictions["M3"] = report.probability_predictions(
                            adapted, rows, chosen["M3_threshold"])
                    hashes = {}
                    for model, spans in predictions.items():
                        path = run / "predictions" / f"{model}.jsonl"
                        report.write_predictions(path, rows, spans,
                                                 sat.DEFAULT if model == "M1" else
                                                 chosen["M2_threshold"] if model == "M2" else
                                                 chosen["M3_threshold"])
                        hashes[model] = sat.digest(path)
                    campaign.json_file(run / "metrics.json", {
                        "target": target, "n_test": len(rows), "selection": chosen,
                        "bundle": {"bundle_manifest_sha256": sat.digest(bundle / "manifest.json")},
                        "selection_sha256": sat.digest(run / "selection.json"),
                        "prediction_sha256": hashes})
                    summary.append({"target": target})
            campaign.json_file(lodo / "cluster_complete.json", {
                "campaign": campaign.NAME,
                "model": sat.MODEL,
                "bundle_manifest_sha256": sat.digest(bundle / "manifest.json"),
                "gold_snapshot_sha256": manifest["gold_snapshot_sha256"],
                "common_split_sha256": manifest["common_split_sha256"],
                "results": summary})
            transfer_files = {
                path.relative_to(lodo).as_posix(): sat.digest(path)
                for path in lodo.rglob("*") if path.is_file()}
            campaign.json_file(lodo / "transfer_manifest.json", {
                "campaign": campaign.NAME,
                "gold_snapshot_sha256": manifest["gold_snapshot_sha256"],
                "common_split_sha256": manifest["common_split_sha256"],
                "files": transfer_files})
            report.report(argparse.Namespace(bundle=bundle, single_bundle=campaign.SOURCE,
                                              single=campaign.SINGLE_RUNS, pysbd=report.PYSBD,
                                              lodo=lodo, output=out,
                                              baseline_output=report.BASELINE_OUTPUT))
            with (out / "table_A_lodo.csv").open(encoding="utf-8-sig", newline="") as stream:
                main = list(csv.DictReader(stream))
            self.assertEqual(len(main), 7)
            self.assertEqual(sum(int(row["n_test"]) for row in main[:-1]), 2690)
            with (out / "table_B_transfer_M3.csv").open(encoding="utf-8-sig", newline="") as stream:
                matrix = list(csv.DictReader(stream))
            self.assertEqual(len(matrix), 6)
            self.assertTrue(all(row[row["source"]] == "N/A" for row in matrix))
            self.assertTrue((out / "figure_role_recall.svg").is_file())
            self.assertTrue((out / "bootstrap_paired_2000.csv").is_file())
            (lodo / "target_btp" / "selected_adapter" / "pytorch_adapter.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "cluster transfer member"):
                report.report(argparse.Namespace(bundle=bundle, single_bundle=campaign.SOURCE,
                                                  single=campaign.SINGLE_RUNS, pysbd=report.PYSBD,
                                                  lodo=lodo, output=out,
                                                  baseline_output=report.BASELINE_OUTPUT))


if __name__ == "__main__":
    unittest.main()
