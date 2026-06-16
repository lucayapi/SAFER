"""La config FSP unique doit charger sans erreur YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from safer_core.io import load_yaml

TEXT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = TEXT_ROOT / "configs" / "frozen_source_prototypes.yaml"


def test_fsp_config_yaml_parses() -> None:
    data = load_yaml(CONFIG)
    assert data.get("method", {}).get("base_method") == "scgm_text"
    assert "checkpoints" in data
    assert data["checkpoints"]["scgm_text"]
    assert data["checkpoints"]["raw_embedding"] is None
    assert data["bertopic"]["enabled"] is True
    yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
