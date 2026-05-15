"""Repair truncated .ipynb files (invalid JSON from huge Plotly/HTML outputs)."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
CELL_START = re.compile(r'\n  \{\n   "cell_type"')

METADATA_SUFFIX = """
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"""


def repair(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        json.loads(text)
        print(f"  OK (already valid): {path.name}")
        return True
    except json.JSONDecodeError as parse_err:
        print(f"  Repairing {path.name}: {parse_err}")
        cut_pos = parse_err.pos

    backup = path.with_suffix(".ipynb.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  Backup: {backup.name}")

    starts = [m.start() for m in CELL_START.finditer(text)]
    if len(starts) < 2:
        print(f"  ERROR: cannot find cell boundaries in {path.name}")
        return False
    idx = max(i for i, s in enumerate(starts) if s < cut_pos)
    prefix = text[: starts[idx]].rstrip().rstrip(",")
    nb = json.loads(prefix + METADATA_SUFFIX)
    dropped = len(starts) - idx
    print(f"  -> {len(nb['cells'])} cells kept, {dropped} truncated cell(s) removed")

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    json.loads(path.read_text(encoding="utf-8"))
    print(f"  -> written {path.stat().st_size // 1024} KB")
    return True


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    ok = True
    for path in paths:
        if path.suffix != ".ipynb":
            continue
        print(path.name)
        ok = repair(path) and ok
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
