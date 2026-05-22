"""Tests encodage modulaire TPN (dispatch + validation config)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from macro_transfer.tpn_encode import (
    CONTRASTIVE_ENCODERS,
    encode_corpus_for_tpn,
    resolve_tpn_checkpoint,
    tpn_method_name,
    validate_encoder_name,
)


def test_tpn_method_name():
    assert tpn_method_name("softtriple") == "tpn_softtriple"
    assert tpn_method_name("supcon") == "tpn_supcon"
    assert tpn_method_name("tpn_batch_triplet") == "tpn_batch_triplet"


def test_validate_encoder_name_strips_tpn_prefix():
    assert validate_encoder_name("tpn_supcon") == "supcon"


def test_validate_encoder_name_rejects_unknown():
    with pytest.raises(ValueError, match="non supporté"):
        validate_encoder_name("unknown_encoder")


def test_resolve_tpn_checkpoint_from_method_cfg():
    ckpt = resolve_tpn_checkpoint(
        "supcon",
        {"checkpoint": "output/supcon/checkpoints/best_model"},
        {},
    )
    assert "supcon" in ckpt


def test_resolve_tpn_checkpoint_from_block():
    ckpt = resolve_tpn_checkpoint(
        "batch_triplet",
        {},
        {"batch_triplet": "output/batch_triplet/checkpoints/best_model"},
    )
    assert "batch_triplet" in ckpt


def test_resolve_tpn_checkpoint_missing():
    with pytest.raises(ValueError, match="Checkpoint manquant"):
        resolve_tpn_checkpoint("supcon", {}, {})


def test_scgm_requires_data_csv():
    with pytest.raises(ValueError, match="data_csv"):
        encode_corpus_for_tpn(
            "scgm_text",
            ["a"],
            "output/scgm_text/checkpoints/best_model.pt",
            emb_csv="embeddings/test.csv",
        )


@patch("macro_transfer.tpn_encode._encode_contrastive_for_tpn")
def test_dispatch_supcon(mock_encode: MagicMock):
    mock_encode.return_value = np.zeros((2, 8), dtype=np.float64)
    z = encode_corpus_for_tpn(
        "supcon",
        ["t1", "t2"],
        "output/supcon/checkpoints/best_model",
        contrastive_config=Path("configs/methods/supcon.yaml"),
        repo_anchor=Path(__file__).resolve().parents[1],
    )
    assert z.shape == (2, 8)
    mock_encode.assert_called_once()
    assert mock_encode.call_args[0][0] == "supcon"


@pytest.mark.parametrize("method", CONTRASTIVE_ENCODERS)
@patch("macro_transfer.tpn_encode._encode_contrastive_for_tpn")
def test_dispatch_all_contrastive(mock_encode: MagicMock, method: str):
    mock_encode.return_value = np.ones((1, 4), dtype=np.float64)
    z = encode_corpus_for_tpn(method, ["x"], f"output/{method}/ckpt")
    assert z.shape[0] == 1
    assert mock_encode.call_args[0][0] == method


@patch("macro_transfer.tpn_encode._encode_contrastive_for_tpn")
def test_contrastive_requires_nonempty_texts(mock_encode: MagicMock):
    with pytest.raises(ValueError, match="texts vide"):
        encode_corpus_for_tpn("softtriple", [], "output/softtriple/ckpt")
    mock_encode.assert_not_called()
