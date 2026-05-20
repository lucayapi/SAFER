"""Helper CLI : affiche data_csv et emb_csv pour un corpus test (usage jobs shell)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.test_corpus import resolve_test_corpus


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus", nargs="?", default=None)
    p.add_argument("--field", choices=("data", "emb", "both"), default="both")
    args = p.parse_args()
    spec = resolve_test_corpus(args.corpus)
    if args.field in ("data", "both"):
        print(spec.data_csv)
    if args.field in ("emb", "both"):
        print(spec.emb_csv)


if __name__ == "__main__":
    main()
