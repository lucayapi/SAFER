"""Déplace runs/ et outputs/ connus vers text/resultats/ (symlinks ou copie)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safer_core.paths import TEXT_ROOT, _LEGACY_TO_RESULTATS


def _migrate_one(src_rel: str, dst_rel: str, *, dry_run: bool, use_symlink: bool) -> None:
    src = TEXT_ROOT / src_rel
    dst = TEXT_ROOT / dst_rel
    if not src.exists():
        print(f"  skip (absent): {src_rel}")
        return
    if dst.exists():
        print(f"  skip (déjà là): {dst_rel}")
        return
    print(f"  {src_rel} -> {dst_rel}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    else:
        shutil.copytree(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Migrer runs/outputs legacy vers resultats/.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Créer un lien symbolique au lieu de copier (économise l'espace).",
    )
    args = p.parse_args()
    print("Migration legacy -> resultats/")
    for src_rel, dst_rel in _LEGACY_TO_RESULTATS.items():
        _migrate_one(src_rel, dst_rel, dry_run=args.dry_run, use_symlink=args.symlink)
    print("Terminé.")


if __name__ == "__main__":
    main()
