#!/usr/bin/env python3
"""Export task-instructed Qwen3 embeddings for one or more factual-unit roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_flag(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return cfg


def role_jobs(cfg: dict) -> dict[str, dict]:
    """Normalize multi-role configuration and the legacy single-role B config."""
    if "roles" in cfg:
        return cfg["roles"]
    return {
        str(cfg["input"]["selected_role"]): {
            "instruction": cfg["instruction"],
            **cfg["output"],
        }
    }


def load_role_units(cfg: dict, data: pd.DataFrame, role: str) -> pd.DataFrame:
    input_cfg = cfg["input"]
    source = (TEXT_ROOT / input_cfg["units_csv"]).resolve()
    required = [input_cfg[k] for k in ("fact_id_col", "role_col", "valid_col", "text_col")]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required input columns {missing} in {source}")
    mask = accepted_flag(data[input_cfg["valid_col"]]) & data[input_cfg["role_col"]].astype(str).str.strip().str.upper().eq(role.upper())
    selected = data.loc[mask, [input_cfg["fact_id_col"], input_cfg["text_col"]]].copy()
    selected.columns = ["doc_id", "text"]
    selected["text"] = selected["text"].fillna("").astype(str).str.strip()
    if selected.empty:
        raise ValueError(f"No valid role-{role} units found")
    if selected["doc_id"].duplicated().any():
        dupes = selected.loc[selected["doc_id"].duplicated(keep=False), "doc_id"].head(10).tolist()
        raise ValueError(f"fact_id must be unique for role-{role} doc_id export; duplicate examples: {dupes}")
    if selected["text"].eq("").any():
        raise ValueError(f"Some selected role-{role} units have empty text")
    try:
        selected["doc_id"] = selected["doc_id"].astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("fact_id values must be numeric to match the existing doc_id convention") from exc
    return selected.reset_index(drop=True)


def load_encoder(cfg: dict):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_cfg = cfg["model"]
    model_name = str(model_cfg["name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This export is configured for a GPU Slurm job; no CUDA device is available")
    precision = str(model_cfg.get("precision", "bfloat16")).lower()
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16 if precision == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype).to(device)
    model.eval()
    return torch, tokenizer, model, device


def encode(texts: list[str], instruction: str, cfg: dict, encoder) -> np.ndarray:
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    torch, tokenizer, model, device = encoder
    model_cfg = cfg["model"]
    max_length = int(model_cfg["max_length"])
    batch_size = int(model_cfg["batch_size"])
    eod_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eod_id is None or eod_id < 0:
        raise ValueError("Qwen tokenizer does not provide <|endoftext|>")

    results = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Qwen instructed role-{cfg['_active_role']} encoding", unit="batch"):
        batch_texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts[start : start + batch_size]]
        tokenized = tokenizer(batch_texts, padding=False, truncation=True, max_length=max_length - 2)
        for token_ids, attention in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            token_ids.append(eod_id)
            attention.append(1)
        batch = tokenizer.pad(tokenized, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**batch, return_dict=True)
            # The official Qwen3-Embedding model-card recipe uses the last
            # non-padding token; left padding makes it the final column.
            emb = output.last_hidden_state[:, -1]
            emb = F.normalize(emb.float(), p=2, dim=1)
        results.append(emb.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(results, axis=0)


def export_role(role: str, role_cfg: dict, cfg: dict, source: Path, units: pd.DataFrame, vectors: np.ndarray) -> None:
    if vectors.ndim != 2 or vectors.shape[0] != len(units) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid role-{role} embedding array shape or values: {vectors.shape}")
    dest = (TEXT_ROOT / role_cfg["embeddings_csv"]).resolve()
    manifest_dest = (TEXT_ROOT / role_cfg["manifest_json"]).resolve()
    if not dest.is_relative_to(TEXT_ROOT) or not manifest_dest.is_relative_to(TEXT_ROOT):
        raise ValueError("Output paths must remain under text/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest_dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    temp_manifest = manifest_dest.with_name(f".{manifest_dest.name}.{uuid.uuid4().hex}.tmp")
    frame = pd.DataFrame(vectors, columns=[f"dim_{i:04d}" for i in range(1, vectors.shape[1] + 1)])
    frame.insert(0, "doc_id", units["doc_id"].to_numpy(dtype=np.int64))
    instruction = str(role_cfg["instruction"]).strip()
    try:
        frame.to_csv(temp, index=False, float_format="%.9g")
        record = {
            "model": cfg["model"]["name"],
            "instruction": instruction,
            "input_template": "Instruct: {instruction}\\nQuery:{text}",
            "pooling": "last non-padding token; official Qwen3-Embedding model-card recipe",
            "normalization": "L2",
            "max_length": int(cfg["model"]["max_length"]),
            "batch_size": int(cfg["model"]["batch_size"]),
            "source_path": str(source),
            "source_sha256": sha256(source),
            "selected_role": role,
            "rows": int(len(units)),
            "dimensions": int(vectors.shape[1]),
            "doc_id_sha256": hashlib.sha256(units["doc_id"].astype(str).str.cat(sep="\n").encode()).hexdigest(),
            "embedding_sha256": sha256(temp),
        }
        temp_manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, dest)
        os.replace(temp_manifest, manifest_dest)
    finally:
        temp.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
    print(f"Wrote role-{role}: {len(units)} embeddings x {vectors.shape[1]} dimensions to {dest}", flush=True)
    print(f"Manifest: {manifest_dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("embedding_diagnostics/config_b_qwen_mechanism_instruct.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Validate selected role rows without loading the model")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (TEXT_ROOT / args.config).resolve()
    cfg = read_config(cfg_path)
    input_cfg = cfg["input"]
    source = (TEXT_ROOT / input_cfg["units_csv"]).resolve()
    data = pd.read_csv(source, dtype={input_cfg["fact_id_col"]: str})
    jobs = role_jobs(cfg)
    selected = {role: load_role_units(cfg, data, role) for role in jobs}
    print(f"Input: {source}")
    for role, role_cfg in jobs.items():
        units = selected[role]
        print(f"Selected valid role-{role} units: {len(units)}; unique doc_id={units['doc_id'].nunique()}")
        print(f"Instruction [{role}]: {str(role_cfg['instruction']).strip()}")
    if args.dry_run:
        return

    encoder = load_encoder(cfg)  # One model load shared across all requested roles.
    for role, role_cfg in jobs.items():
        print(f"Starting instructed embedding export for role-{role}", flush=True)
        vectors = encode(selected[role]["text"].tolist(), str(role_cfg["instruction"]).strip(), {**cfg, "_active_role": role}, encoder)
        export_role(role, role_cfg, cfg, source, selected[role], vectors)


if __name__ == "__main__":
    main()
