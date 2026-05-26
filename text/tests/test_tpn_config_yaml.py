"""Les configs TPN doivent charger sans erreur YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from safer_core.io import load_yaml

TEXT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [TEXT_ROOT / "configs" / "tpn_macro_transfer.yaml"]


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.name)
def test_tpn_config_yaml_parses(cfg_path: Path) -> None:
    data = load_yaml(cfg_path)
    assert "encoding" in data
    enc = data["encoding"]
    assert "encode_batch_size" in enc
    assert "log_every_batches" in enc
    assert "scgm_infer_batch_size" in enc
    assert "normalize_embeddings" in enc
    assert data.get("method", {}).get("base_method") == "scgm_text"
    # Re-parse brut pour détecter les ':' manquants (erreur utilisateur courante).
    yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def test_encode_batch_size_line_has_colon() -> None:
    main = TEXT_ROOT / "configs" / "tpn_macro_transfer.yaml"
    for i, line in enumerate(main.read_text(encoding="utf-8").splitlines(), start=1):
        if "encode_batch_size" in line and not line.strip().startswith("#"):
            assert ":" in line, f"Ligne {i} : ':' manquant dans {line!r}"
