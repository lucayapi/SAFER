"""Tests write_bn_compat_arrays (staging macro_transfer → BN)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import importlib.util
from pathlib import Path

_staging_path = Path(__file__).resolve().parents[1] / "bn_pipeline" / "staging_macro_transfer.py"
_spec = importlib.util.spec_from_file_location("staging_macro_transfer", _staging_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
write_bn_compat_arrays = _mod.write_bn_compat_arrays


def test_write_bn_compat_arrays(tmp_path):
    n = 5
    meta = pd.DataFrame(
        {
            "accident_id": ["a1"] * n,
            "fact_id": range(n),
            "pred_label": ["A0", "A1", "B", "C", "A0"],
            "p_A0": [0.4, 0.1, 0.1, 0.1, 0.5],
            "p_A1": [0.2, 0.5, 0.1, 0.1, 0.2],
            "p_B": [0.2, 0.2, 0.6, 0.1, 0.1],
            "p_C": [0.2, 0.2, 0.2, 0.7, 0.2],
            "m_hat": ["A0", "A1", "B", "C", "A0"],
            "q_conf": [0.4, 0.5, 0.6, 0.7, 0.5],
        }
    )
    exports = tmp_path / "bn_exports"
    exports.mkdir()
    write_bn_compat_arrays(exports, meta)

    pt = np.load(exports / "pt_y_target.npy")
    pz = np.load(exports / "pt_z_target.npy")
    assert pt.shape == (n, 4)
    assert pz.shape == (n, 4)
    zdf = pd.read_csv(exports / "z_assignments_target.csv")
    assert len(zdf) == n
    assert "z_hat" in zdf.columns
