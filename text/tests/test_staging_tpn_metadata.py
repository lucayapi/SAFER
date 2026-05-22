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


@pytest.mark.parametrize(
    "method",
    ["tpn_softtriple", "tpn_supcon", "tpn_batch_triplet", "tpn_scgm_text"],
)
def test_tpn_metadata_filename(method: str):
    assert METADATA_BY_METHOD[method] == "metadata_with_tpn_macro_probs.csv"


@pytest.mark.parametrize(
    "method",
    ["tpn_softtriple", "tpn_supcon"],
)
def test_resolve_transfer_metadata_path(method: str):
    mt = Path(f"output_test/metallurgie/macro_transfer/{method}")
    p = resolve_transfer_metadata_path(mt, method)
    assert p.name == "metadata_with_tpn_macro_probs.csv"
    assert "transfer" in str(p)
