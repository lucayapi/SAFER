"""Checkpoint retention without running a GPU or the upstream trainer."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rq1_final as campaign
import rq1_final_lodo as lodo
import sat_lodo_gold_v3 as sat


class CheckpointRetentionTests(unittest.TestCase):
    def test_monitor_prunes_before_trainer_finishes(self):
        with tempfile.TemporaryDirectory(prefix="rq1_checkpoint_live_") as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            run = root / "target_btp"
            trainer = run / "trainer"
            trainer.mkdir(parents=True)
            script = repo / "fake_trainer.py"
            script.write_text(textwrap.dedent("""
                import json, sys, time
                from pathlib import Path
                config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
                trainer = Path(config['output_dir'])
                for step in (1000, 2000, 3000):
                    folder = trainer / f'checkpoint-{step}'
                    folder.mkdir()
                    for filename in ('pytorch_model.bin', 'optimizer.pt', 'scheduler.pt'):
                        (folder / filename).write_bytes(f'{step}/{filename}'.encode())
                    (folder / 'trainer_state.json').write_text(
                        json.dumps({'global_step': step, 'epoch': step / 1000}), encoding='utf-8')
                    time.sleep(0.3)
                    if step == 2000 and (trainer / 'checkpoint-1000').exists():
                        sys.exit(9)
                """), encoding="utf-8")

            def export(folder, adapter, unused_template):
                adapter.mkdir(parents=True, exist_ok=True)
                for filename in lodo.ADAPTER_FILES:
                    (adapter / filename).write_bytes((folder.name + filename).encode())
                fingerprint = sat.digest(folder / "pytorch_model.bin")
                campaign.json_file(adapter / "checkpoint_source.json", {
                    "checkpoint_sha256": fingerprint})
                return fingerprint

            with (patch.object(lodo.checkpoint, "export_checkpoint_adapter", side_effect=export),
                  patch.object(lodo.sat.legacy, "official_trainer", return_value=script),
                  patch.object(lodo, "POLL_SECONDS", 0.02)):
                lodo.run_trainer_with_harvest(repo, {"output_dir": str(trainer)},
                                              run / "config.json", run, root / "template")
            self.assertFalse((trainer / "checkpoint-1000").exists())
            self.assertFalse((trainer / "checkpoint-2000").exists())
            self.assertTrue((trainer / "checkpoint-3000").is_dir())

    def test_export_then_prune_and_keep_selected_adapter(self):
        with tempfile.TemporaryDirectory(prefix="rq1_checkpoint_") as temporary:
            run = Path(temporary) / "target_btp"
            trainer = run / "trainer"
            trainer.mkdir(parents=True)
            template = run / "template"
            template.mkdir()
            for step in (1000, 2000):
                folder = trainer / f"checkpoint-{step}"
                folder.mkdir()
                for filename in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt"):
                    (folder / filename).write_bytes(f"{step}/{filename}".encode())
                campaign.json_file(folder / "trainer_state.json", {
                    "global_step": step, "epoch": step / 1000})

            def export(folder, adapter, unused_template):
                adapter.mkdir(parents=True, exist_ok=True)
                for filename in lodo.ADAPTER_FILES:
                    (adapter / filename).write_bytes((folder.name + filename).encode())
                fingerprint = sat.digest(folder / "pytorch_model.bin")
                campaign.json_file(adapter / "checkpoint_source.json", {
                    "checkpoint_sha256": fingerprint})
                return fingerprint

            with patch.object(lodo.checkpoint, "export_checkpoint_adapter", side_effect=export):
                lodo.harvest_checkpoints(run, trainer, template)
            self.assertFalse((trainer / "checkpoint-1000").exists())
            self.assertTrue((trainer / "checkpoint-2000").is_dir())
            self.assertTrue((run / "checkpoint_adapters" / "checkpoint-1000" /
                             "candidate.json").is_file())
            for filename in lodo.ADAPTER_FILES:
                (trainer / "safer_lodo" / "fr").mkdir(parents=True, exist_ok=True)
                (trainer / "safer_lodo" / "fr" / filename).write_bytes(filename.encode())
            campaign.json_file(trainer / "trainer_state.json", {
                "global_step": 2500, "epoch": 2.5})
            choices = lodo.candidates_from_exports(run, trainer)
            self.assertEqual([item["candidate"] for item in choices],
                             ["checkpoint-1000", "checkpoint-2000", "final"])
            source = choices[0]["adapter"]
            selected = run / "selected_adapter"
            weights = lodo.adapter_hashes(source)
            lodo.persist_selected_adapter(source, selected, weights)
            selection = {"M3_weights": weights}
            campaign.json_file(run / "selection.json", selection)
            pred_dir = run / "predictions"
            pred_dir.mkdir()
            predictions = {}
            for model in ("M1", "M2", "M3"):
                path = pred_dir / f"{model}.jsonl"
                path.write_text("{}\n", encoding="utf-8")
                predictions[model] = sat.digest(path)
            campaign.json_file(run / "metrics.json", {
                "selection_sha256": sat.digest(run / "selection.json"),
                "prediction_sha256": predictions})
            lodo.cleanup_full_states(run, selection)
            self.assertFalse((trainer / "checkpoint-2000").exists())
            self.assertFalse((run / "checkpoint_adapters").exists())
            self.assertFalse((trainer / "safer_lodo" / "fr").exists())
            self.assertEqual(lodo.adapter_hashes(selected), weights)
            self.assertEqual(json.loads((run / "storage_cleanup.json").read_text(
                encoding="utf-8"))["full_checkpoints_remaining"], 0)

    def test_rejects_deletion_outside_trainer(self):
        with tempfile.TemporaryDirectory(prefix="rq1_checkpoint_") as temporary:
            root = Path(temporary)
            trainer = root / "trainer"
            trainer.mkdir()
            outsider = root / "checkpoint-1000"
            outsider.mkdir()
            with self.assertRaisesRegex(ValueError, "outside the trainer"):
                lodo.remove_checkpoint(trainer, outsider)
            self.assertTrue(outsider.exists())


if __name__ == "__main__":
    unittest.main()
