"""Vérifie que chaque grille tuning fusionne en config chargeable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import load_contrastive_config_from_dict, merge_config_dict
from contrastive_methods.tuning import (
    expand_grid,
    filter_softtriple_tuning_combos,
    validate_contrastive_grid_keys,
)
from safer_core.io import load_yaml
from safer_core.paths import TEXT_ROOT as PKG_ROOT
from scgm_text.tuning import expand_grid as scgm_expand_grid
from scgm_text.tuning import validate_scgm_grid_keys
from supervised_macro_ft.tuning import expand_grid as macro_expand_grid
from supervised_macro_ft.tuning import validate_macro_ft_grid_keys


@pytest.mark.parametrize(
    "method",
    ["softtriple", "supcon", "batch_triplet"],
)
def test_contrastive_grid_keys_and_all_combos_merge(method: str):
    spec_path = PKG_ROOT / f"configs/tuning/{method}_grid.yaml"
    spec = load_yaml(spec_path)
    base = load_yaml(PKG_ROOT / spec["base_config"])
    grid = spec.get("grid") or {}
    validate_contrastive_grid_keys(method, grid)
    combos = expand_grid(grid)
    if method == "softtriple":
        combos = filter_softtriple_tuning_combos(combos)
    assert combos, f"grille vide : {spec_path}"
    for overrides in combos[:3]:
        merged = merge_config_dict(base, overrides)
        cfg = load_contrastive_config_from_dict(method, merged)
        assert cfg.method_name == method
        assert cfg.learning_rate > 0
        assert cfg.epochs >= 1


def test_scgm_grid_keys_and_combos_merge():
    spec = load_yaml(PKG_ROOT / "configs/tuning/scgm_text_grid.yaml")
    base = load_yaml(PKG_ROOT / spec["base_config"])
    grid = spec.get("grid") or {}
    validate_scgm_grid_keys(grid)
    combos = scgm_expand_grid(grid)
    assert combos
    for overrides in combos:
        merged = merge_config_dict(base, overrides)
        assert merged["training"]["lr"] > 0
        assert merged["model"]["lmd"] > 0


@pytest.mark.parametrize(
    "grid_name,allowed_projections",
    [
        ("supervised_macro_ft_grid.yaml", ("linear", "mlp_sklearn")),
    ],
)
def test_macro_ft_grid_keys_and_combos_merge(grid_name, allowed_projections):
    spec = load_yaml(PKG_ROOT / f"configs/tuning/{grid_name}")
    base = load_yaml(PKG_ROOT / spec["base_config"])
    grid = spec.get("grid") or {}
    validate_macro_ft_grid_keys(grid)
    combos = macro_expand_grid(grid)
    assert combos
    for overrides in combos[:3]:
        merged = merge_config_dict(base, overrides)
        assert merged["model"]["projection"] in allowed_projections
        assert float(merged["training"]["lr_projector"]) > 0
