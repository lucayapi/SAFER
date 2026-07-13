"""Migration des exports annotés (XLSX) vers ``dataset/data_<corpus>.csv``."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from annotation.article_stats import discover_annotation_runs
from annotation.export_io import (
    ANNOTATION_TABLE_ENGINE,
    ANNOTATION_TABLE_SUFFIX,
    attach_accident_summary_column,
    reorder_annotation_output_columns,
)
from annotation.prompts.v13_two_pass_ambiguity_context import LABELS

TEXT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = TEXT_ROOT / "dataset"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# run_id annotation → id registre configs/test_corpora.yaml
RUN_ID_TO_DATASET: dict[str, str] = {
    "run_all_btp": "btp",
    "run_all_metallurgie": "metallurgie",
    "run_all_caou_chimie_plas": "caou",
}

REQUIRED_COLUMNS = ("accident_id", "sentence", "pred_label", "pred_ok")
RECOMMENDED_COLUMNS = ("fact_id", "accident_summary")


@dataclass(frozen=True)
class MigrationPlan:
    run_id: str
    dataset_id: str
    source_xlsx: Path
    dest_csv: Path
    backup_csv: Optional[Path] = None


def resolve_dataset_id(run_id: str, override: Optional[str] = None) -> str:
    if override:
        return str(override)
    if run_id in RUN_ID_TO_DATASET:
        return RUN_ID_TO_DATASET[run_id]
    if run_id.startswith("run_all_"):
        return run_id.removeprefix("run_all_")
    raise KeyError(
        f"Pas de mapping dataset pour run_id={run_id!r}. "
        f"Connus : {', '.join(sorted(RUN_ID_TO_DATASET))}. "
        "Utilisez --dataset-id."
    )


def find_best_annotated_xlsx(run_dir: Path, *, prefer_final: bool = True) -> Optional[Path]:
    """Préfère ``*__annotated_final.xlsx``, sinon ``*__annotated.xlsx`` le plus récent."""
    if not run_dir.is_dir():
        return None

    if prefer_final:
        finals = sorted(run_dir.glob(f"*__annotated_final{ANNOTATION_TABLE_SUFFIX}"))
        if finals:
            return max(finals, key=lambda path: path.stat().st_mtime)

    matches = sorted(run_dir.glob(f"*__annotated{ANNOTATION_TABLE_SUFFIX}"))
    # Exclure annotated_final déjà traité ci-dessus.
    matches = [p for p in matches if "__annotated_final" not in p.stem]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_annotated_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine=ANNOTATION_TABLE_ENGINE)
    df = attach_accident_summary_column(df)
    return reorder_annotation_output_columns(df)


def validate_annotation_table(df: pd.DataFrame, *, source: Path) -> list[str]:
    """Retourne la liste des problèmes bloquants."""
    issues: list[str] = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        issues.append(f"{source.name} : colonnes manquantes {missing}")
        return issues

    if df.empty:
        issues.append(f"{source.name} : tableau vide")
        return issues

    for col in RECOMMENDED_COLUMNS:
        if col not in df.columns:
            issues.append(f"{source.name} : colonne recommandée absente : {col}")

    labels = df["pred_label"].astype(str).str.strip()
    bad_labels = sorted(set(labels.unique()) - set(LABELS) - {"", "nan", "None"})
    if bad_labels:
        issues.append(f"{source.name} : pred_label inattendus (hors A0/A1/B/C) : {bad_labels[:10]}")

    return issues


def preview_stats(df: pd.DataFrame) -> dict[str, Any]:
    ok = df["pred_ok"]
    if ok.dtype != bool:
        ok = ok.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "t"})
    valid = df.loc[ok & df["pred_label"].astype(str).str.strip().isin(LABELS)]
    return {
        "n_rows": int(len(df)),
        "n_pred_ok": int(ok.sum()),
        "n_trainable": int(len(valid)),
        "n_accidents": int(df["accident_id"].nunique()) if "accident_id" in df.columns else 0,
    }


def build_migration_plan(
    run: dict[str, Any],
    *,
    dataset_id: Optional[str] = None,
    prefer_final: bool = True,
    dataset_dir: Path = DATASET_DIR,
) -> MigrationPlan:
    run_id = str(run["run_id"])
    run_dir = Path(run["run_dir"])
    source = find_best_annotated_xlsx(run_dir, prefer_final=prefer_final)
    if source is None:
        source = Path(run.get("annotated_path") or "")
    if not source.is_file():
        raise FileNotFoundError(f"Aucun XLSX annoté pour {run_id} dans {run_dir}")

    ds_id = resolve_dataset_id(run_id, dataset_id)
    dest = dataset_dir / f"data_{ds_id}.csv"
    return MigrationPlan(
        run_id=run_id,
        dataset_id=ds_id,
        source_xlsx=source,
        dest_csv=dest,
    )


def migrate_one(
    plan: MigrationPlan,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    df = load_annotated_table(plan.source_xlsx)
    issues = validate_annotation_table(df, source=plan.source_xlsx)
    if issues:
        raise ValueError("\n".join(issues))

    stats = preview_stats(df)
    backup_path: Optional[Path] = None

    if dry_run:
        return {
            "run_id": plan.run_id,
            "dataset_id": plan.dataset_id,
            "source": str(plan.source_xlsx),
            "dest": str(plan.dest_csv),
            "dry_run": True,
            **stats,
        }

    plan.dest_csv.parent.mkdir(parents=True, exist_ok=True)
    if backup and plan.dest_csv.is_file():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = plan.dest_csv.with_name(f"{plan.dest_csv.stem}__backup_{stamp}.csv")
        shutil.copy2(plan.dest_csv, backup_path)

    df.to_csv(plan.dest_csv, index=False, encoding="utf-8")
    return {
        "run_id": plan.run_id,
        "dataset_id": plan.dataset_id,
        "source": str(plan.source_xlsx),
        "dest": str(plan.dest_csv),
        "backup": str(backup_path) if backup_path else None,
        "dry_run": False,
        **stats,
    }


def select_runs(
    outputs_dir: Path,
    *,
    run_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    runs = discover_annotation_runs(outputs_dir)
    if not run_ids:
        # Par défaut : runs connus du registre dataset, s'ils ont un export.
        wanted = set(RUN_ID_TO_DATASET)
        runs = [r for r in runs if r["run_id"] in wanted]
        if not runs:
            runs = discover_annotation_runs(outputs_dir)
        return runs

    selected = []
    known = {r["run_id"]: r for r in runs}
    for rid in run_ids:
        if rid not in known:
            run_dir = outputs_dir / rid
            xlsx = find_best_annotated_xlsx(run_dir)
            if xlsx is None:
                raise FileNotFoundError(f"Run {rid!r} : pas de *__annotated.xlsx dans {run_dir}")
            selected.append(
                {
                    "run_id": rid,
                    "corpus": rid,
                    "run_dir": run_dir,
                    "annotated_path": xlsx,
                    "run_config": {},
                }
            )
        else:
            selected.append(known[rid])
    return selected


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Migre les XLSX annotés (annotation/outputs) vers dataset/data_<corpus>.csv",
    )
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Racine des runs d'annotation (défaut : annotation/outputs)",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Dossier cible dataset/ (défaut : text/dataset)",
    )
    p.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        metavar="RUN_ID",
        help="Run à migrer (répétable). Défaut : btp, metallurgie, caou si présents.",
    )
    p.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Override id corpus cible (un seul run à la fois).",
    )
    p.add_argument(
        "--pass1-only",
        action="store_true",
        help="Ignorer *__annotated_final.xlsx et prendre la passe 1.",
    )
    p.add_argument("--dry-run", action="store_true", help="Valider sans écrire les CSV.")
    p.add_argument("--no-backup", action="store_true", help="Ne pas sauvegarder l'ancien CSV.")
    p.add_argument(
        "--validate-load",
        action="store_true",
        help="Après export, tester load_filtered_metadata sur chaque CSV.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.dataset_id and args.run_ids and len(args.run_ids) > 1:
        print("Erreur : --dataset-id compatible avec un seul --run-id.", file=sys.stderr)
        return 2

    outputs_dir = args.outputs_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()

    try:
        runs = select_runs(outputs_dir, run_ids=args.run_ids)
    except FileNotFoundError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    if not runs:
        print(f"Aucune run migrable dans {outputs_dir}", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    errors = 0
    for run in runs:
        try:
            plan = build_migration_plan(
                run,
                dataset_id=args.dataset_id if len(runs) == 1 else None,
                prefer_final=not args.pass1_only,
                dataset_dir=dataset_dir,
            )
            result = migrate_one(
                plan,
                dry_run=bool(args.dry_run),
                backup=not args.no_backup,
            )
            if args.validate_load and not args.dry_run:
                from scgm_text.data_metadata import load_filtered_metadata

                meta = load_filtered_metadata(str(plan.dest_csv))
                result["n_loaded_trainable"] = int(len(meta))
            results.append(result)
            mode = "DRY-RUN" if result["dry_run"] else "OK"
            print(
                f"[{mode}] {result['run_id']} → {result['dest']} "
                f"({result['n_rows']} lignes, {result['n_trainable']} exploitables) "
                f"← {Path(result['source']).name}",
                flush=True,
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            errors += 1
            print(f"[ERREUR] {run.get('run_id', run)} : {exc}", file=sys.stderr)

    if errors:
        return 1
    if args.dry_run:
        print(f"\n{len(results)} migration(s) prête(s). Relancez sans --dry-run pour écrire.")
    else:
        print(
            f"\n{len(results)} CSV écrit(s) dans {dataset_dir}. "
            "Pensez à relancer export_corpus_embeddings (FORCE=1 si besoin).",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
