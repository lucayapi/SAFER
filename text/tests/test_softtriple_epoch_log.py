"""Tests affichage console loss SoftTriple par epoch."""

from __future__ import annotations

from contrastive_methods.training_log import print_epoch_line


def test_print_epoch_line_train_only(capsys):
    print_epoch_line("SoftTriple", 1, 3, 2.5)
    out = capsys.readouterr().out
    assert "[SoftTriple epoch=1/3]" in out
    assert "train_loss=2.5000" in out
    assert "val_loss" not in out


def test_print_epoch_line_with_val(capsys):
    print_epoch_line("SoftTriple", 2, 30, 1.234, val_loss=1.102)
    out = capsys.readouterr().out
    assert "val_loss=1.1020" in out
