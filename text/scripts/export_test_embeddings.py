"""Exporte les embeddings Qwen figés pour un corpus (wrapper → export_corpus_embeddings)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from safer_core.test_corpus import default_test_corpus_id


def _load_export_module():
    mod_path = ROOT_DIR / "scripts" / "export_corpus_embeddings.py"
    spec = importlib.util.spec_from_file_location("export_corpus_embeddings", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export embeddings Qwen (rétrocompat — délègue à export_corpus_embeddings.py)",
    )
    p.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=f"Identifiant registre test_corpora.yaml (défaut : {default_test_corpus_id()})",
    )
    p.add_argument("--data_csv", type=str, default=None, help="Ignoré (rétrocompat).")
    p.add_argument("--output_csv", type=str, default=None, help="Ignoré (rétrocompat).")
    p.add_argument("--backbone_name", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None, help="→ encode_batch_size")
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "--config",
        "configs/export_embeddings.yaml",
        "--corpus",
        str(args.corpus or default_test_corpus_id()),
    ]
    if args.backbone_name:
        argv.extend(["--backbone_name", args.backbone_name])
    if args.force:
        argv.append("--force")

    cfg_overrides = {}
    if args.batch_size is not None:
        cfg_overrides["encode_batch_size"] = args.batch_size
    if args.max_seq_length is not None:
        cfg_overrides["max_seq_length"] = args.max_seq_length

    if cfg_overrides:
        import yaml

        cfg_path = ROOT_DIR / "configs" / "export_embeddings.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cfg.update(cfg_overrides)
        if args.backbone_name:
            cfg["backbone_name"] = args.backbone_name
        tmp_cfg = ROOT_DIR / "configs" / ".export_embeddings_cli_override.yaml"
        tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        argv[argv.index("configs/export_embeddings.yaml")] = str(
            tmp_cfg.relative_to(ROOT_DIR)
        )

    if args.data_csv or args.output_csv:
        print(
            "[warn] --data_csv / --output_csv ignorés : chemins résolus via test_corpora.yaml",
            flush=True,
        )

    export_mod = _load_export_module()
    export_mod.main(argv)


if __name__ == "__main__":
    main()
