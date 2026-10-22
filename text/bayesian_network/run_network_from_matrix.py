"""Fit both Bayesian networks from a documented accident-component matrix.

This entry point does not discover factors, encode text, or call an LLM.
Run from text/ with --source-run pointing to a completed, audited matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from . import run_bayesian_analysis as analysis
except ImportError:
    import run_bayesian_analysis as analysis


TEXT_ROOT = Path(__file__).resolve().parents[1]
ROLES = {"A0", "A1", "B", "C"}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (TEXT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curated_matrix(source: Path) -> tuple[pd.DataFrame, dict[str, str], dict]:
    manifest_path = source / "manifest.json"
    matrix_path = source / "accident_factor_matrix.csv"
    dictionary_path = source / "factor_dictionary.csv"
    curation_path = source / "curation_manifest.json"
    for path in (manifest_path, matrix_path, dictionary_path, curation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required curated input is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or curation.get("status") != "complete":
        raise ValueError("The source run and its component curation must both be complete")
    if manifest.get("curation_manifest_sha256") != sha256(curation_path):
        raise ValueError("The source manifest does not match the curation manifest")
    hashes = {"matrix_sha256": sha256(matrix_path),
              "dictionary_sha256": sha256(dictionary_path)}
    if any(curation.get(key) != value for key, value in hashes.items()):
        raise ValueError("The curation hashes no longer match the matrix and dictionary")

    frame = pd.read_csv(matrix_path, dtype={"accident_id": str})
    dictionary = pd.read_csv(dictionary_path, dtype=str).fillna("")
    if frame.columns[0] != "accident_id" or frame["accident_id"].isna().any():
        raise ValueError("The matrix must begin with nonmissing accident_id")
    if frame["accident_id"].duplicated().any():
        raise ValueError("Duplicate accidents in the matrix")
    factors = list(frame.columns[1:])
    if not factors or len(factors) != len(set(factors)):
        raise ValueError("The matrix needs unique factor columns")
    required = {"factor_id", "role", "llm_label"}
    if not required.issubset(dictionary.columns):
        raise ValueError(f"The dictionary needs columns {sorted(required)}")
    if (dictionary["factor_id"].duplicated().any() or
            set(factors) != set(dictionary["factor_id"])):
        raise ValueError("Dictionary IDs and matrix columns do not match exactly")
    if set(dictionary["role"]) != ROLES:
        raise ValueError("The component matrix must contain A0, A1, B and C")
    if dictionary["llm_label"].eq("").any():
        raise ValueError("Every component needs a nonempty display label")
    values = frame[factors].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError("The matrix must be binary")
    if len(frame) == 0:
        raise ValueError("The matrix contains no accidents")
    role_map = dict(zip(dictionary["factor_id"], dictionary["role"]))
    metadata = {"source_run": str(source), "accidents": len(frame),
                "components": len(factors), **hashes,
                "curation_sha256": sha256(curation_path)}
    return frame.set_index("accident_id"), role_map, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--config", default="bayesian_network/config_btp_building_construction_masonry_qwen3_4b_raw_leaf_umap_no_llm.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, config_path, output = map(resolve, (args.source_run, args.config, args.output_dir))
    matrix, role_map, metadata = load_curated_matrix(source)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps({
        **metadata, "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
        "network_code_sha256": sha256(Path(__file__).with_name("network_analysis.py")),
    }, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("analysis_hash") != signature:
            raise ValueError(f"Existing output uses a different input or code: {output}")
    manifest = {"status": "running", "analysis_hash": signature,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": metadata, "config_sha256": sha256(config_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    analysis.RUN_DIR = output
    print(f"[BN] {len(matrix)} accidents x {len(matrix.columns)} documented components", flush=True)
    try:
        analysis.run_network(matrix, role_map, config)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[BN] completed: {output}", flush=True)


if __name__ == "__main__":
    main()
