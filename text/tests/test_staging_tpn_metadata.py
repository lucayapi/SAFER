"""Staging BN : chemin metadata TPN."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_staging_path = Path(__file__).resolve().parents[1] / "bn_pipeline" / "staging_macro_transfer.py"
_spec = importlib.util.spec_from_file_location("staging_macro_transfer", _staging_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
METADATA_BY_METHOD = _mod.METADATA_BY_METHOD
resolve_transfer_metadata_path = _mod.resolve_transfer_metadata_path
_normalize_fsp_metadata_for_bn = _mod._normalize_fsp_metadata_for_bn


@pytest.mark.parametrize(
    "method",
    ["tpn_full_softtriple", "tpn_full_supcon", "tpn_full_batch_triplet", "tpn_full_scgm_text"],
)
def test_tpn_metadata_filename(method: str):
    assert METADATA_BY_METHOD[method] == "metadata_with_tpn_macro_probs.csv"


@pytest.mark.parametrize(
    "method",
    ["tpn_full_softtriple", "tpn_full_supcon"],
)
def test_resolve_transfer_metadata_path(method: str):
    mt = Path(f"output_test/metallurgie/macro_transfer/{method}")
    p = resolve_transfer_metadata_path(mt, method)
    assert p.name == "metadata_with_tpn_macro_probs.csv"
    assert "transfer" in str(p)


def test_fsp_metadata_filename_and_path():
    method = "frozen_source_prototypes/scgm"
    assert METADATA_BY_METHOD[method] == "target_macro_predictions.csv"
    mt = Path("output_test/metallurgie/macro_transfer/frozen_source_prototypes/scgm")
    p = resolve_transfer_metadata_path(mt, method)
    assert p.name == "target_macro_predictions.csv"
    assert "transfer" in str(p)


def test_normalize_fsp_metadata_for_bn():
    import pandas as pd

    df = pd.DataFrame(
        {
            "pred_macro": ["A0", "B"],
            "confidence": [0.9, 0.8],
            "prob_A0": [0.9, 0.1],
            "prob_A1": [0.05, 0.1],
            "prob_B": [0.03, 0.7],
            "prob_C": [0.02, 0.1],
        }
    )
    out = _normalize_fsp_metadata_for_bn(df)
    assert "m_hat" in out.columns
    assert "q_conf" in out.columns
    for col in ("p_A0", "p_A1", "p_B", "p_C"):
        assert col in out.columns
