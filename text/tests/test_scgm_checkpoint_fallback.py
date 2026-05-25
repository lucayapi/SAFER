"""Tests repli best_model.pt depuis last_model.pt."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scripts.train_scgm_text import ensure_best_checkpoint_file


def test_ensure_best_copies_from_last(tmp_path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    last = ckpt / "last_model.pt"
    last.write_bytes(b"fake")
    ensure_best_checkpoint_file(str(ckpt))
    assert (ckpt / "best_model.pt").is_file()
    assert (ckpt / "best_model.pt").read_bytes() == b"fake"
