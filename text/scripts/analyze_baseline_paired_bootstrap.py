"""Compare TF-IDF + LR et embeddings Qwen figés + LR par bootstrap apparié."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_transfer.target_bootstrap import paired_bootstrap_target_difference


CORPORA = ("metallurgie", "caou", "nicollin")
MODEL_A = "frozen_embeddings_lr"
MODEL_B = "tfidf_lr"


def _paths(corpus: str, *, tfidf_root: Path, frozen_root: Path) -> dict[str, Path]:
    return {
        MODEL_A: frozen_root / corpus / "supervised_baseline" / "transfer" / "models" / "logistic_regression" / "target_macro_predictions.csv",
        MODEL_B: tfidf_root / "targets" / corpus / "models" / "logistic_regression" / "predictions.csv",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap apparié : embeddings figés + LR moins TF-IDF + LR.")
    parser.add_argument("--tfidf-root", default="output/tfidf_baseline")
    parser.add_argument("--frozen-root", default="output_test")
    parser.add_argument("--output-dir", default="output/baseline_paired_bootstrap")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    tfidf_root = Path(args.tfidf_root)
    frozen_root = Path(args.frozen_root)
    output_dir = Path(args.output_dir)
    rows = []
    source_paths: dict[str, dict[str, str]] = {}
    for corpus in CORPORA:
        paths = _paths(corpus, tfidf_root=tfidf_root, frozen_root=frozen_root)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{corpus} : prédictions baseline absentes : {missing}")
        frames = {name: pd.read_csv(path) for name, path in paths.items()}
        result = paired_bootstrap_target_difference(
            frames,
            model_a=MODEL_A,
            model_b=MODEL_B,
            destination=output_dir / "per_corpus" / f"paired_bootstrap_{corpus}.csv",
            n_resamples=args.n_resamples,
            seed=args.seed,
            confidence_level=args.confidence_level,
            force=args.force,
        )
        result.insert(0, "corpus", corpus)
        rows.append(result)
        source_paths[corpus] = {name: str(path) for name, path in paths.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(rows, ignore_index=True)
    destination = output_dir / "paired_bootstrap_differences.csv"
    combined.to_csv(destination, index=False)
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps({
            "model_a": MODEL_A,
            "model_b": MODEL_B,
            "metrics": ["balanced_accuracy", "macro_f1"],
            "corpora": list(CORPORA),
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "confidence_level": args.confidence_level,
            "resampling_unit": "accident_id",
            "source_predictions": source_paths,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"paired_differences: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
