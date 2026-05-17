#!/usr/bin/env python3
"""Enrichit themes_by_z.csv via OpenAI (hors notebook / nœud avec accès Internet)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scgm_text.openai_theme_labels import (  # noqa: E402
    enrich_themes_by_z_openai,
    load_openai_dotenv,
    probe_openai_connectivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produit topics/themes_by_z_openai.csv à partir de themes_by_z.csv (API OpenAI)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resultats/scgm_text",
        help="Racine du run SCGM (contient topics/themes_by_z.csv).",
    )
    parser.add_argument(
        "--themes_csv",
        type=str,
        default=None,
        help="Défaut : <output_dir>/topics/themes_by_z.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Défaut : <output_dir>/topics/themes_by_z_openai.csv",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n-example-texts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="En cas d'échec sur un z, libellé de secours (top_words) au lieu d'arrêter.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Teste uniquement la connectivité API puis quitte (0 = OK).",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Limite de topics (debug).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (ROOT / out_root).resolve()

    themes_in = Path(args.themes_csv) if args.themes_csv else out_root / "topics" / "themes_by_z.csv"
    themes_out = Path(args.output_csv) if args.output_csv else out_root / "topics" / "themes_by_z_openai.csv"

    if not themes_in.is_file():
        print(f"[erreur] Fichier introuvable : {themes_in}", file=sys.stderr)
        print("Lancez d'abord : python scripts/export_scgm_text_outputs.py --output_dir", themes_in.parents[1])
        return 1

    load_openai_dotenv()
    if not probe_openai_connectivity(timeout=min(30.0, args.timeout)):
        print(
            "[erreur] API OpenAI inaccessible (timeout). Sur HPC2, lancez ce script sur le "
            "nœud de login (hpclogin01), pas depuis JupyterHub / un nœud GPU.",
            file=sys.stderr,
        )
        print("  bash jobs/enrich_scgm_themes_openai.sh", file=sys.stderr)
        return 2

    if args.probe_only:
        print("Connectivité OpenAI OK.")
        return 0

    print(f"Entrée  : {themes_in}")
    print(f"Sortie  : {themes_out}")
    print(f"Modèle  : {args.model} | timeout={args.timeout}s | n_example_texts={args.n_example_texts}")

    enrich_themes_by_z_openai(
        themes_in,
        themes_out,
        model=args.model,
        temperature=args.temperature,
        n_example_texts=args.n_example_texts,
        request_timeout=args.timeout,
        skip_on_error=args.skip_on_error,
        max_rows=args.max_rows,
    )
    print(f"OK — écrit {themes_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
