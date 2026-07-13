"""Exporte les embeddings d'encodeur figé pour un ou plusieurs corpus (registre test_corpora.yaml)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset
from contrastive_methods.export import export_text_embeddings
from safer_core.test_corpus import (
    default_test_corpus_id,
    list_test_corpus_ids,
    resolve_test_corpus,
)

DEFAULT_CONFIG = ROOT_DIR / "configs" / "export_embeddings.yaml"


def load_export_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config encodage introuvable : {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config invalide (dict attendu) : {config_path}")
    return cfg


def resolve_corpora_ids(
    cfg: Dict[str, Any],
    *,
    corpus: Optional[str] = None,
    all_corpora: bool = False,
) -> List[str]:
    if corpus:
        return [str(corpus)]
    corpora_cfg = cfg.get("corpora")
    if all_corpora or corpora_cfg == "all":
        return list_test_corpus_ids()
    if isinstance(corpora_cfg, list) and corpora_cfg:
        return [str(c) for c in corpora_cfg]
    return [default_test_corpus_id()]


def build_contrastive_config(cfg: Dict[str, Any], data_csv: Path) -> ContrastiveConfig:
    return ContrastiveConfig(
        method_name="raw_export",
        dataset_path=data_csv,
        text_col=str(cfg.get("text_col", "sentence")),
        label_col=str(cfg.get("label_col", "pred_label")),
        group_col=str(cfg.get("group_col", "accident_id")),
        pred_ok_col=str(cfg.get("pred_ok_col", "pred_ok")),
        backbone_name=str(cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B")),
        encode_batch_size=int(cfg.get("encode_batch_size", 16)),
        max_seq_length=int(cfg.get("max_seq_length", 256)),
        use_prompt=bool(cfg.get("use_prompt", False)),
        use_projector=False,
        backbone_trainable=False,
    )


def export_corpus(
    corpus_id: str,
    cfg: Dict[str, Any],
    *,
    force: bool = False,
    anchor: Optional[Path] = None,
) -> Path:
    spec = resolve_test_corpus(
        corpus_id,
        anchor=anchor or ROOT_DIR,
        require_files=True,
        require_emb_csv=False,
    )
    dest = spec.emb_csv
    skip_existing = bool(cfg.get("skip_existing", True))
    if dest.is_file() and skip_existing and not force:
        print(f"[skip] corpus={spec.id} déjà présent : {dest}", flush=True)
        return dest

    contrastive_cfg = build_contrastive_config(cfg, spec.data_csv)
    dataset = prepare_text_dataset(contrastive_cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    export_text_embeddings(
        contrastive_cfg,
        dataset,
        dest,
        batch_size=contrastive_cfg.encode_batch_size,
        show_progress=True,
    )
    print(f"[ok] corpus={spec.id} → {dest} ({len(dataset)} lignes)", flush=True)
    return dest


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export embeddings encodeur pour corpus du registre.")
    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG.relative_to(ROOT_DIR)),
        help="YAML d'encodage (backbone, corpora, text_col, …)",
    )
    p.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Un seul corpus (id registre) ; sinon liste du YAML ou --all",
    )
    p.add_argument("--all", action="store_true", help="Encoder tous les corpus du registre.")
    p.add_argument("--backbone_name", type=str, default=None, help="Override backbone HF.")
    p.add_argument("--force", action="store_true", help="Ré-encoder même si emb_csv existe.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    cfg = load_export_config(config_path)
    if args.backbone_name:
        cfg = dict(cfg)
        cfg["backbone_name"] = args.backbone_name

    corpus_ids = resolve_corpora_ids(cfg, corpus=args.corpus, all_corpora=args.all)
    if not corpus_ids:
        raise ValueError("Aucun corpus à encoder.")

    print(
        f"[export] backbone={cfg.get('backbone_name')} corpora={corpus_ids} "
        f"text_col={cfg.get('text_col', 'sentence')}",
        flush=True,
    )
    for cid in corpus_ids:
        export_corpus(cid, cfg, force=bool(args.force), anchor=ROOT_DIR)


if __name__ == "__main__":
    main()
