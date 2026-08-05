"""Tuning contrastif par architecture avec recherche LR post-évaluation.

Les paramètres de la loss et de l'entraînement viennent du fichier méthode.
Seuls l'architecture de l'encodeur et les paramètres de la régression
logistique sont explorés ici.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from contrastive_methods.config import load_contrastive_config_from_dict, merge_config_dict
from contrastive_methods.eval_corpus import run_final_classification_eval
from contrastive_methods.kfold_train import get_contrastive_runner, run_kfold_loop
from safer_core.io import ensure_dir, load_yaml, save_config_resolved
from safer_core.paths import TEXT_ROOT


ARCHITECTURE_ORDER = ("last_1", "last_2", "last_3", "full")
PROJECTOR_ORDER = (True, False)
CV_METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


def expand_grid(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expandit une grille de valeurs simples en dictionnaires d'overrides."""
    if not grid:
        return [{}]
    keys = sorted(grid)
    values = [value if isinstance(value, list) else [value] for value in (grid[key] for key in keys)]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def architecture_name(train_last_n_layers: Optional[int], use_projector: bool) -> str:
    scope = "full" if train_last_n_layers is None else f"last_{int(train_last_n_layers)}"
    return f"{scope}_{'yes' if use_projector else 'no'}"


def _normalise_variant(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "last1_yes": "last_1_yes",
        "last1_no": "last_1_no",
        "last2_yes": "last_2_yes",
        "last2_no": "last_2_no",
        "last3_yes": "last_3_yes",
        "last3_no": "last_3_no",
        "full_projector": "full_yes",
        "full_no_projector": "full_no",
    }
    return aliases.get(text, text)


def parse_variants(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    raw: List[str] = list(values or [])
    if not raw and os.environ.get("VARIANTS"):
        raw = os.environ["VARIANTS"].replace(",", " ").split()
    if not raw:
        return None
    selected = {_normalise_variant(value) for value in raw}
    valid = {architecture_name(n, projector) for n in (1, 2, 3, None) for projector in PROJECTOR_ORDER}
    unknown = selected - valid
    if unknown:
        raise ValueError(f"Variantes inconnues : {sorted(unknown)}. Choix : {sorted(valid)}")
    return selected


def _architecture_overrides(combo: Mapping[str, Any]) -> Dict[str, Any]:
    overrides = dict(combo)
    overrides.setdefault("model.backbone_trainable", True)
    overrides.setdefault("model.train_last_n_layers", None)
    overrides.setdefault("model.use_projector", True)
    overrides.setdefault("model.projection", "mlp_sklearn")
    return overrides


def _variant_from_overrides(overrides: Mapping[str, Any]) -> str:
    return architecture_name(
        overrides.get("model.train_last_n_layers"),
        bool(overrides.get("model.use_projector", True)),
    )


def _scope_label(train_last_n_layers: Optional[int]) -> str:
    if train_last_n_layers is None:
        return "Full encoder"
    layer = int(train_last_n_layers)
    return f"Last {layer} layer" if layer == 1 else f"Last {layer} layers"


def _apply_full_overrides(
    overrides: Dict[str, Any],
    full_training_overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    if overrides.get("model.train_last_n_layers") is not None:
        return overrides
    merged = dict(overrides)
    merged.update(dict(full_training_overrides))
    return merged


def _encoder_scope_key(overrides: Mapping[str, Any]) -> str:
    layers = overrides.get("model.train_last_n_layers")
    return "full" if layers is None else f"last_{int(layers)}"


def _apply_scope_epoch_override(
    overrides: Mapping[str, Any],
    epochs_by_encoder_scope: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(overrides)
    scope = _encoder_scope_key(merged)
    if scope in epochs_by_encoder_scope:
        merged["training.epochs"] = int(epochs_by_encoder_scope[scope])
    return merged


def _metric_value(metrics_dir: Path, filename: str, metric: str = "balanced_accuracy") -> float:
    path = metrics_dir / filename
    if not path.is_file():
        return float("nan")
    try:
        frame = pd.read_csv(path)
        return float(frame.iloc[0].get(metric, float("nan"))) if not frame.empty else float("nan")
    except (OSError, ValueError, TypeError, IndexError):
        return float("nan")


def _contrastive_combo_complete(combo_dir: Path, test_corpora: Sequence[str]) -> bool:
    metrics_dir = combo_dir / "metrics"
    required = [
        combo_dir / "best_logistic_params.json",
        combo_dir / "cv" / "cv_per_fold.csv",
        combo_dir / "cv" / "cv_summary.csv",
        combo_dir / "cv" / "logistic_grid_summary.csv",
        metrics_dir / "metrics_classification_btp.csv",
    ]
    required.extend(
        metrics_dir / f"metrics_classification_test_{corpus_id}.csv"
        for corpus_id in test_corpora
    )
    return all(path.is_file() for path in required)


def _completed_contrastive_row(
    combo_dir: Path,
    cfg,
    merged_overrides: Mapping[str, Any],
    variant: str,
) -> Optional[Dict[str, Any]]:
    if not _contrastive_combo_complete(combo_dir, cfg.test_corpora_list()):
        return None
    try:
        cv_df = pd.read_csv(combo_dir / "cv" / "cv_summary.csv")
        best_lr = json.loads((combo_dir / "best_logistic_params.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None
    if cv_df.empty:
        return None
    best_cv = cv_df.iloc[0].to_dict()
    metrics_dir = combo_dir / "metrics"
    row: Dict[str, Any] = {
        "method": cfg.method_name,
        "variant": variant,
        "encoder_scope": _scope_label(merged_overrides.get("model.train_last_n_layers")),
        "projector": "Yes" if merged_overrides.get("model.use_projector") else "No",
        "train_last_n_layers": merged_overrides.get("model.train_last_n_layers"),
        "projection": merged_overrides.get("model.projection") if merged_overrides.get("model.use_projector") else None,
        "embedding_dim": 128 if merged_overrides.get("model.use_projector") else 1024,
        "combo_id": variant,
        "combo_dir": str(combo_dir),
        "cv_ba_mean": best_cv.get("mean_balanced_accuracy"),
        "cv_ba_std": best_cv.get("std_balanced_accuracy"),
        "cv_protocol": "full_pipeline",
        "cv_accuracy_mean": best_cv.get("mean_accuracy"),
        "cv_macro_f1_mean": best_cv.get("mean_macro_f1"),
        "best_lr_C": best_lr.get("C"),
        "best_lr_penalty": best_lr.get("penalty"),
        "best_lr_solver": best_lr.get("solver"),
        "best_lr_class_weight": best_lr.get("class_weight"),
        "final_epochs_used": cfg.epochs,
        "ba_btp": _metric_value(metrics_dir, "metrics_classification_btp.csv"),
    }
    for corpus_id in cfg.test_corpora_list():
        row[f"ba_ood_{corpus_id}"] = _metric_value(
            metrics_dir, f"metrics_classification_test_{corpus_id}.csv"
        )
    ood = [
        value for key, value in row.items()
        if key.startswith("ba_ood_") and math.isfinite(float(value))
    ]
    row["ba_ood_avg"] = sum(ood) / len(ood) if ood else float("nan")
    row["ba_ood_worst"] = min(ood) if ood else float("nan")
    return row
def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_safe(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{str(key): _finite_or_none(value) for key, value in row.items()} for row in rows]


def _variant_key(row: Mapping[str, Any]) -> str:
    if row.get("variant"):
        return str(row["variant"])
    layers = row.get("train_last_n_layers")
    scope = str(row.get("encoder_scope", "")).strip().lower()
    if scope.startswith("full"):
        layers = None
    elif layers is None or (isinstance(layers, float) and math.isnan(layers)):
        for candidate in (1, 2, 3):
            if f"last {candidate}" in scope:
                layers = candidate
                break
    projector = str(row.get("projector", "yes")).strip().lower() in ("yes", "true", "1")
    return architecture_name(layers, projector)


def _merge_partial_summary(summary_path: Path, new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = {_variant_key(row) for row in new_rows}
    previous: List[Dict[str, Any]] = []
    if summary_path.is_file():
        try:
            previous = pd.read_csv(summary_path).to_dict(orient="records")
        except (OSError, pd.errors.ParserError):
            previous = []
    retained = [row for row in previous if _variant_key(row) not in selected]
    merged = retained + new_rows
    order = {name: index for index, name in enumerate(
        architecture_name(n, projector)
        for n in (1, 2, 3, None)
        for projector in PROJECTOR_ORDER
    )}
    return sorted(merged, key=lambda row: (order.get(_variant_key(row), 999), str(row.get("combo_id", ""))))


def _build_lr_summary(
    fold_rows: List[Dict[str, Any]],
    classifier_grid: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    best_index = -1
    best_score = float("-inf")
    for index, params in enumerate(classifier_grid):
        values: Dict[str, Any] = {"lr_index": index, **dict(params)}
        for metric in CV_METRICS:
            key = f"lr_{index}_val_{metric}"
            series = pd.to_numeric(pd.DataFrame(fold_rows).get(key), errors="coerce") if fold_rows else pd.Series(dtype=float)
            series = series.dropna()
            values[f"mean_val_{metric}"] = float(series.mean()) if not series.empty else float("nan")
            values[f"std_val_{metric}"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        score = values.get("mean_val_balanced_accuracy", float("nan"))
        if math.isfinite(float(score)) and float(score) > best_score:
            best_score = float(score)
            best_index = index
        rows.append(values)
    if best_index < 0:
        raise RuntimeError("Aucune configuration LR n'a produit une balanced_accuracy valide.")
    return rows, best_index, dict(classifier_grid[best_index])


def _run_full_pipeline_group_cv(
    cfg,
    runner,
    combo_dir: Path,
    classifier_grid: Sequence[Mapping[str, Any]],
    variant: str,
) -> List[Dict[str, Any]]:
    """CV honnête: nouvel encodeur contrastif entraîné dans chaque fold."""
    def fold_dir_fn(fold_id: int) -> str:
        return str(combo_dir / "folds" / f"fold_{fold_id}")

    fold_rows, _ = run_kfold_loop(
        cfg,
        runner,
        fold_dir_fn=fold_dir_fn,
        log_prefix=f"architecture/{variant}",
        save_tables=False,
        post_eval_grid=[dict(params) for params in classifier_grid],
    )
    return fold_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def run_architecture_tuning(method_name: str, argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Tuning architecture + LR contrastif : {method_name}")
    parser.add_argument("--grid-config", default=f"configs/tuning/{method_name}_macro_ft_grid.yaml")
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--skip-final-fit", action="store_true")
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Nombre de folds de la CV groupée (défaut: 3).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument(
        "--refit",
        type=lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Force le recalcul des variantes existantes (défaut: true).",
    )
    args, _ = parser.parse_known_args(argv)
    spec = load_yaml(TEXT_ROOT / args.grid_config)
    method_name = str(spec.get("method_name", method_name))
    base_path = TEXT_ROOT / str(spec.get("base_config", f"configs/methods/{method_name}.yaml"))
    base_raw = load_yaml(base_path)
    seed = int(args.seed if args.seed is not None else spec.get("seed", base_raw.get("seed", 42)))
    base_raw = merge_config_dict(base_raw, {"seed": seed})
    if os.environ.get("TEST_CORPORA"):
        spec = {**spec, "test_corpora": [c.strip() for c in os.environ["TEST_CORPORA"].split(",") if c.strip()]}
    elif os.environ.get("TEST_CORPUS"):
        spec = {**spec, "test_corpora": [os.environ["TEST_CORPUS"]]}

    architecture_grid = spec.get("architecture_grid") or {}
    architecture_combos = [_architecture_overrides(combo) for combo in expand_grid(architecture_grid)]
    expected = {architecture_name(n, projector) for n in (1, 2, 3, None) for projector in PROJECTOR_ORDER}
    actual = {_variant_from_overrides(combo) for combo in architecture_combos}
    if actual != expected:
        raise ValueError(f"La grille architecture doit produire exactement 8 variantes, obtenu : {sorted(actual)}")
    all_architecture_combos = list(architecture_combos)
    selected = parse_variants(args.variants)
    if selected is not None:
        architecture_combos = [combo for combo in architecture_combos if _variant_from_overrides(combo) in selected]
    if args.max_combos is not None:
        architecture_combos = architecture_combos[: args.max_combos]

    classifier_grid = [dict(params) for params in expand_grid(spec.get("logistic_grid") or {})]
    if not classifier_grid:
        raise ValueError("logistic_grid ne peut pas être vide")
    folds_override = args.n_folds
    if folds_override is None and os.environ.get("N_FOLDS"):
        folds_override = int(os.environ["N_FOLDS"])
    n_folds = int(folds_override if folds_override is not None else spec.get("n_folds", 3))
    if n_folds < 2:
        raise ValueError("n_folds doit être >= 2 pour la CV LR.")
    selection_metric = str(spec.get("selection_metric", "balanced_accuracy"))
    full_training_overrides = dict(spec.get("full_training_overrides") or {})
    epochs_by_encoder_scope = dict(spec.get("epochs_by_encoder_scope") or {})
    tuning_root = TEXT_ROOT / str(spec.get("output_dir", f"output/{method_name}/macro_ft_tuning"))
    ensure_dir(tuning_root / "combos")
    runner = get_contrastive_runner(method_name)
    new_rows: List[Dict[str, Any]] = []

    for architecture in architecture_combos:
        variant = _variant_from_overrides(architecture)
        combo_id = variant
        combo_dir = tuning_root / "combos" / combo_id
        if not args.refit:
            cfg_existing_overrides = _apply_full_overrides(architecture, full_training_overrides)
            cfg_existing_overrides = _apply_scope_epoch_override(
                cfg_existing_overrides, epochs_by_encoder_scope
            )
            merged_existing = merge_config_dict(base_raw, cfg_existing_overrides)
            merged_existing.update({"method_name": method_name, "seed": seed, "n_folds": n_folds})
            merged_existing["output_dir"] = str(combo_dir)
            merged_existing["final_fit_full_data"] = True
            existing_cfg = load_contrastive_config_from_dict(method_name, merged_existing, config_path=str(args.grid_config))
            existing_row = _completed_contrastive_row(
                combo_dir, existing_cfg, cfg_existing_overrides, variant
            )
            if existing_row is not None:
                new_rows.append(existing_row)
                print(f"[architecture/{variant}] variante complète détectée: skip refit", flush=True)
                continue
        merged_overrides = _apply_full_overrides(architecture, full_training_overrides)
        merged_overrides = _apply_scope_epoch_override(
            merged_overrides, epochs_by_encoder_scope
        )
        merged = merge_config_dict(base_raw, merged_overrides)
        merged.update({"method_name": method_name, "seed": seed, "n_folds": n_folds})
        merged["output_dir"] = str(combo_dir)
        merged["final_fit_full_data"] = True
        merged["selection_metric"] = selection_metric
        if spec.get("test_corpora"):
            merged["test_corpora"] = list(spec["test_corpora"])
        cfg = load_contrastive_config_from_dict(method_name, merged, config_path=str(args.grid_config))

        print(
            f"[architecture/{variant}] CV complète: nouvel encodeur par fold…",
            flush=True,
        )
        fold_rows = _run_full_pipeline_group_cv(
            cfg, runner, combo_dir, classifier_grid, variant
        )
        cv_dir = combo_dir / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fold_rows).to_csv(cv_dir / "cv_per_fold.csv", index=False)
        lr_rows, best_index, best_lr = _build_lr_summary(fold_rows, classifier_grid)
        pd.DataFrame(lr_rows).to_csv(cv_dir / "logistic_grid_summary.csv", index=False)
        _write_json(combo_dir / "best_logistic_params.json", best_lr)

        best_cv = lr_rows[best_index]
        cv_summary = pd.DataFrame([{
            "model": f"{method_name}_{variant}",
            "n_folds": len(fold_rows),
            "cv_protocol": "full_pipeline",
            "mean_balanced_accuracy": best_cv.get("mean_val_balanced_accuracy"),
            "std_balanced_accuracy": best_cv.get("std_val_balanced_accuracy"),
            "mean_accuracy": best_cv.get("mean_val_accuracy"),
            "std_accuracy": best_cv.get("std_val_accuracy"),
            "mean_macro_f1": best_cv.get("mean_val_macro_f1"),
            "std_macro_f1": best_cv.get("std_val_macro_f1"),
        }])
        cv_summary.to_csv(cv_dir / "cv_summary.csv", index=False)

        if not args.skip_final_fit:
            print(f"[architecture/{variant}] Fit final sur 100 % BTP…", flush=True)
            fold_epochs = [
                int(row["best_epoch"])
                for row in fold_rows
                if row.get("best_epoch") is not None and int(row["best_epoch"]) > 0
            ]
            final_epochs = cfg.epochs
            if cfg.final_epochs_from_cv and fold_epochs:
                final_epochs = max(1, int(round(statistics.median(fold_epochs))))
            cfg_final = dataclasses.replace(cfg, epochs=final_epochs)
            cfg_final.extra = dict(cfg.extra)
            cfg_final.extra["final_epochs_used"] = final_epochs
            result = runner(cfg_final)
            checkpoint = result.output_root / "checkpoints" / "best_model"
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"Checkpoint final absent : {checkpoint}")
            run_final_classification_eval(
                cfg_final,
                checkpoint,
                result.output_root,
                cv_summary=cv_summary,
                classifier_overrides=best_lr,
            )
            save_config_resolved(
                {
                    **merged,
                    "architecture_variant": variant,
                    "best_logistic_params": best_lr,
                    "best_epoch_by_fold": fold_epochs,
                    "final_epochs_used": final_epochs,
                    "tuning_grid_config": args.grid_config,
                },
                result.output_root,
            )

        metrics_dir = combo_dir / "metrics"
        row: Dict[str, Any] = {
            "method": method_name,
            "variant": variant,
            "encoder_scope": _scope_label(merged_overrides.get("model.train_last_n_layers")),
            "projector": "Yes" if merged_overrides.get("model.use_projector") else "No",
            "train_last_n_layers": merged_overrides.get("model.train_last_n_layers"),
            "projection": merged_overrides.get("model.projection") if merged_overrides.get("model.use_projector") else None,
            "embedding_dim": 128 if merged_overrides.get("model.use_projector") else 1024,
            "combo_id": combo_id,
            "combo_dir": str(combo_dir),
            "cv_ba_mean": best_cv.get("mean_val_balanced_accuracy"),
            "cv_ba_std": best_cv.get("std_val_balanced_accuracy"),
            "cv_protocol": "full_pipeline",
            "cv_accuracy_mean": best_cv.get("mean_val_accuracy"),
            "cv_macro_f1_mean": best_cv.get("mean_val_macro_f1"),
            "best_lr_C": best_lr.get("C"),
            "best_lr_penalty": best_lr.get("penalty"),
            "best_lr_solver": best_lr.get("solver"),
            "best_lr_class_weight": best_lr.get("class_weight"),
            "final_epochs_used": final_epochs if not args.skip_final_fit else None,
            "ba_btp": _metric_value(metrics_dir, "metrics_classification_btp.csv"),
        }
        for corpus_id in cfg.test_corpora_list():
            row[f"ba_ood_{corpus_id}"] = _metric_value(
                metrics_dir,
                f"metrics_classification_test_{corpus_id}.csv",
            )
        ood = [
            value
            for key, value in row.items()
            if key.startswith("ba_ood_") and math.isfinite(float(value))
        ]
        row["ba_ood_avg"] = sum(ood) / len(ood) if ood else float("nan")
        row["ba_ood_worst"] = min(ood) if ood else float("nan")
        new_rows.append(row)

    if not args.refit:
        known_variants = {_variant_key(row) for row in new_rows}
        for architecture in all_architecture_combos:
            variant = _variant_from_overrides(architecture)
            if variant in known_variants:
                continue
            merged_overrides = _apply_full_overrides(architecture, full_training_overrides)
            merged_overrides = _apply_scope_epoch_override(
                merged_overrides, epochs_by_encoder_scope
            )
            combo_dir = tuning_root / "combos" / variant
            merged_existing = merge_config_dict(base_raw, merged_overrides)
            merged_existing.update({"method_name": method_name, "seed": seed, "n_folds": n_folds})
            merged_existing["output_dir"] = str(combo_dir)
            merged_existing["final_fit_full_data"] = True
            existing_cfg = load_contrastive_config_from_dict(method_name, merged_existing, config_path=str(args.grid_config))
            existing_row = _completed_contrastive_row(
                combo_dir, existing_cfg, merged_overrides, variant
            )
            if existing_row is not None:
                new_rows.append(existing_row)
                known_variants.add(variant)

    summary_path = tuning_root / "grid_summary.csv"
    all_rows = _merge_partial_summary(summary_path, new_rows)
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    pd.DataFrame(all_rows).to_csv(tuning_root / "results_summary.csv", index=False)
    _write_json(tuning_root / "results_summary.json", _json_safe(all_rows))
    print(f"[architecture] Résumé : {summary_path}", flush=True)
    return 0
