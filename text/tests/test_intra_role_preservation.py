"""Tests IPR (Intra-role Preservation Ratio)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from metrics.intra_role_preservation import (
    DEFAULT_BASELINE_LABEL,
    compute_ipr_columns,
    compute_ipr_from_geometry_rows,
    ipr_mean,
    ipr_r_from_rho,
    ipr_display_table,
    rho_r,
)
from metrics.compare_display import (
    enrich_geometry_with_ipr,
    joint_eta2_ipr_table,
)


def _row(method: str, t: float, w_a0: float, w_a1: float = 2.0, w_b: float = 2.0, w_c: float = 2.0):
    return {
        "method": method,
        "T_macro_balanced": t,
        "W_A0": w_a0,
        "W_A1": w_a1,
        "W_B": w_b,
        "W_C": w_c,
        "eta2_macro_balanced": 0.1,
    }


def test_rho_r_basic():
    r = rho_r(_row("x", 10.0, 2.0))
    assert abs(r["A0"] - 5.0) < 1e-9


def test_ipr_a0_doubled_w():
    raw = _row(DEFAULT_BASELINE_LABEL, 10.0, 2.0)
    method = _row("SCGM", 10.0, 4.0)
    rho_b = rho_r(raw)
    rho_m = rho_r(method)
    ipr = ipr_r_from_rho(rho_b, rho_m)
    assert abs(ipr["A0"] - 2.0) < 1e-9


def test_ipr_mean_excludes_nan_role():
    ipr = {"A0": 2.0, "A1": float("nan"), "B": 1.0, "C": 1.0}
    assert abs(ipr_mean(ipr) - (2.0 + 1.0 + 1.0) / 3.0) < 1e-9


def test_compute_ipr_columns_no_baseline():
    df = pd.DataFrame([_row("SCGM", 10.0, 2.0)])
    out = compute_ipr_columns(df)
    assert out["IPR_mean"].isna().all()


def test_compute_ipr_columns_with_baseline():
    df = pd.DataFrame(
        [
            _row(DEFAULT_BASELINE_LABEL, 10.0, 2.0, w_a1=2.0, w_b=2.0, w_c=2.0),
            _row("SCGM", 10.0, 4.0, w_a1=2.0, w_b=2.0, w_c=2.0),
        ]
    )
    out = compute_ipr_columns(df)
    scgm = out[out["method"] == "SCGM"].iloc[0]
    assert abs(scgm["IPR_A0"] - 2.0) < 1e-9
    assert abs(scgm["IPR_mean"] - 1.25) < 1e-9  # mean(2,1,1,1)


def test_ipr_display_table_rounding():
    df = compute_ipr_columns(
        pd.DataFrame(
            [
                _row(DEFAULT_BASELINE_LABEL, 10.0, 2.0),
                _row("SCGM", 10.0, 4.0),
            ]
        )
    )
    disp = ipr_display_table(df)
    assert "IPR_mean" in disp.columns
    assert disp.loc[disp["method"] == "SCGM", "IPR_A0"].iloc[0] == 2.0


def test_compute_ipr_from_geometry_rows():
    raw = {"T_macro_balanced": 10.0, "W_A0": 2.0, "W_A1": 2.0, "W_B": 2.0, "W_C": 2.0}
    method = {"T_macro_balanced": 10.0, "W_A0": 4.0, "W_A1": 2.0, "W_B": 2.0, "W_C": 2.0}
    ipr = compute_ipr_from_geometry_rows(raw, method)
    assert abs(ipr["IPR_A0"] - 2.0) < 1e-9
    assert abs(ipr["IPR_mean"] - 1.25) < 1e-9


def test_joint_eta2_ipr_table():
    df = pd.DataFrame(
        [
            {
                "method": DEFAULT_BASELINE_LABEL,
                "T_macro_balanced": 10.0,
                "W_A0": 2.0,
                "W_A1": 2.0,
                "W_B": 2.0,
                "W_C": 2.0,
                "eta2_macro_balanced": 0.5,
            },
            {
                "method": "SCGM",
                "T_macro_balanced": 10.0,
                "W_A0": 4.0,
                "W_A1": 2.0,
                "W_B": 2.0,
                "W_C": 2.0,
                "eta2_macro_balanced": 0.6,
            },
        ]
    )
    joint = joint_eta2_ipr_table(df)
    assert "eta2_macro_balanced_perc" in joint.columns
    assert "IPR_mean" in joint.columns
