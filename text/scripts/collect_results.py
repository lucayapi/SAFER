"""Agrège metrics_geometry par méthode (BTP et corpus de test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metrics.compare_display import (
    EMBEDDING_COMPARE_METHODS,
    METHOD_DISPLAY,
    RAW_TEST_RESULTS_KEY,
    collect_kfold_btp_comparison,
    fill_eta2_macro_balanced_perc,
    method_label,
    normalize_method_display_name,
    order_methods,
)
from metrics.geometry import METRICS_TABLE_COLUMNS
from safer_core.paths import ensure_comparisons_dirs
from safer_core.test_corpus import (
    default_test_corpus_id,
    list_test_corpus_ids,
    output_test_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str, default="output")
    p.add_argument("--output", type=str, default=None, help="Alias BTP (legacy)")
    p.add_argument(
        "--test-corpus",
        type=str,
        default=None,
        help="Corpus test pour raw_embedding_test/<id>/ (défaut : registre)",
    )
    p.add_argument(
        "--all-test-corpora",
        action="store_true",
        help="Écrit aussi embedding_geometry_comparison_test_<corpus>.csv par id registre",
    )
    return p.parse_args()


def _load_metrics_file(path: Path) -> dict | None:
    if path.suffix == ".csv" and path.is_file():
        df = pd.read_csv(path)
        if len(df):
            return df.iloc[0].to_dict()
    if path.suffix == ".json" and path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_method_row(method_dir: Path) -> dict | None:
    """BTP : metrics_geometry_btp.csv puis repli metrics_geometry.*"""
    return _load_method_row_for_corpus(method_dir, "btp")


def _results_dir_for_method(
    root: Path,
    method_key: str,
    corpus: str,
    *,
    test_corpus_id: str | None = None,
) -> Path:
    if corpus == "test":
        cid = test_corpus_id or default_test_corpus_id()
        if method_key == "raw_embedding":
            return root / cid / "raw_embedding"
        return root / cid / method_key
    return root / method_key


def _load_method_row_for_corpus(method_dir: Path, corpus: str) -> dict | None:
    metrics_dir = method_dir / "metrics"
    if corpus == "btp":
        candidates = (
            "metrics_geometry_btp.csv",
            "metrics_geometry.csv",
            "metrics_geometry.json",
        )
    elif corpus == "test":
        candidates = (
            "metrics_geometry_test.csv",
            "metrics_geometry.csv",
            "metrics_geometry.json",
        )
    else:
        raise ValueError(f"corpus inconnu : {corpus}")
    for name in candidates:
        row = _load_metrics_file(metrics_dir / name)
        if row is not None:
            return row
    return None


def _normalize_comparison_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = fill_eta2_macro_balanced_perc(df)
    for col in METRICS_TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan") if col not in ("method", "macros_ignored", "macros_valid") else ""
    return df[METRICS_TABLE_COLUMNS]


def collect_embedding_comparison(
    root: Path,
    *,
    corpus: str,
    method_keys: tuple[str, ...] = EMBEDDING_COMPARE_METHODS,
    test_corpus_id: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for key in method_keys:
        method_dir = _results_dir_for_method(root, key, corpus, test_corpus_id=test_corpus_id)
        if not method_dir.is_dir():
            continue
        row = _load_method_row_for_corpus(method_dir, corpus)
        if row is None:
            continue
        raw_name = row.get("method")
        if raw_name in (None, "", key):
            row["method"] = method_label(key)
        else:
            row["method"] = normalize_method_display_name(str(raw_name), key)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=METRICS_TABLE_COLUMNS)
    return order_methods(_normalize_comparison_df(rows))


def collect_all_methods_btp(root: Path) -> pd.DataFrame:
    """Toutes les méthodes sous output/ (hors comparisons) — usage legacy."""
    rows: list[dict] = []
    if not root.is_dir():
        return pd.DataFrame(columns=METRICS_TABLE_COLUMNS)
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "comparisons":
            continue
        row = _load_method_row(method_dir)
        if row is None:
            continue
        key = method_dir.name
        if row.get("method") in (None, "", key):
            row["method"] = method_label(key)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=METRICS_TABLE_COLUMNS)
    return _normalize_comparison_df(rows)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Écrit : {path} ({len(df)} méthodes)")


def main() -> None:
    args = parse_args()
    root = (ROOT_DIR / args.results_root).resolve()
    test_root = output_test_root()
    comp = ensure_comparisons_dirs()
    tables = comp / "tables"

    test_corpus = args.test_corpus or default_test_corpus_id()
    df_btp = collect_embedding_comparison(root, corpus="btp")
    df_test = collect_embedding_comparison(test_root, corpus="test", test_corpus_id=test_corpus)

    if df_btp.empty and df_test.empty:
        df_legacy = collect_all_methods_btp(root)
        if df_legacy.empty:
            print("Aucune métrique trouvée sous", root)
            return
        df_btp = df_legacy

    btp_path = tables / "embedding_geometry_comparison_btp.csv"
    test_path = tables / "embedding_geometry_comparison_test.csv"
    alias_path = Path(args.output) if args.output else tables / "embedding_geometry_comparison.csv"

    df_kfold_btp = collect_kfold_btp_comparison(root)
    kfold_btp_path = tables / "embedding_geometry_comparison_btp_kfold.csv"

    if not df_btp.empty:
        _write_table(df_btp, btp_path)
        _write_table(df_btp, alias_path)
    if not df_kfold_btp.empty:
        _write_table(df_kfold_btp, kfold_btp_path)
    elif not df_btp.empty:
        print(
            f"(absent) {kfold_btp_path} — pas de metrics/kfold_summary.csv "
            "(relancer train K-fold n_folds>1)"
        )
    if not df_test.empty:
        _write_table(df_test, test_path)
    elif not df_btp.empty:
        print(
            f"(absent) {test_path} — relancer l'éval test "
            f"(TEST_CORPUS={test_corpus}, fit final + embeddings test)"
        )

    if args.all_test_corpora:
        for cid in list_test_corpus_ids():
            df_c = collect_embedding_comparison(root, corpus="test", test_corpus_id=cid)
            if not df_c.empty:
                per_corpus = tables / f"embedding_geometry_comparison_test_{cid}.csv"
                _write_table(df_c, per_corpus)


if __name__ == "__main__":
    main()
