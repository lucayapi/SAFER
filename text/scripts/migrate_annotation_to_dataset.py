#!/usr/bin/env python3
"""CLI : migration annotation/outputs → dataset/ (voir annotation/migrate_to_dataset.py)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.migrate_to_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
