"""Tests affichage console loss SoftTriple par epoch."""

from __future__ import annotations

from contrastive_methods.training_softtriple import _print_softtriple_epoch


def test_print_softtriple_epoch_train_only(capsys):
    _print_softtriple_epoch(1, 3, 2.5)
    out = capsys.readouterr().out
    assert "[SoftTriple epoch=1/3]" in out
    assert "train_loss=2.5000" in out
    assert "val_loss" not in out


def test_print_softtriple_epoch_with_val(capsys):
    _print_softtriple_epoch(
        2,
        30,
        1.234,
        val_loss=1.102,
        val_geometry={"eta2_macro_balanced_perc": 11.5},
        selection_metric="eta2_macro_balanced_perc",
    )
    out = capsys.readouterr().out
    assert "val_loss=1.1020" in out
    assert "eta2_macro_balanced_perc=11.5000" in out
