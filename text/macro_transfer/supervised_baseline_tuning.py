"""Tuning hyperparamètres de la baseline sklearn (notebook 07b).

Grille **par famille de modèle** (LR, RF, XGB) sur BTP via GroupKFold,
sélection sur balanced accuracy, puis évaluation OOD avec les meilleurs params.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from macro_transfer.supervised_baseline import (
    aggregate_cv_metrics,
    evaluate_all_models_on_test,
    export_all_models_test_results,
    fit_final_and_predict_test,
    load_supervised_datasets,
    merge_model_registry,
    run_model_group_kfold_cv,
    save_supervised_run_manifest,
    select_best_model,
    supervised_baseline_output_dir,
    summarize_cross_domain_generalization,
)
from safer_core.io import ensure_dir, load_yaml
from safer_core.paths import TEXT_ROOT, resolve_repo_path

logger = logging.getLogger(__name__)

TUNABLE_MODEL_KEYS: Tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)


def supervised_baseline_tuned_output_dir(
    corpus_id: str,
    *,
    anchor: Optional[Path] = None,
) -> Path:
    """``output_test/<corpus>/supervised_baseline_tuned/`` (n'écrase pas le 07)."""
    root = anchor or TEXT_ROOT
    return (root / "output_test" / str(corpus_id) / "supervised_baseline_tuned").resolve()


def supervised_baseline_tuning_dir(
    *,
    cv_corpus: str = "metallurgie",
    anchor: Optional[Path] = None,
) -> Path:
    """Dossier grille partagé : ``output_test/<cv_corpus>/supervised_baseline/tuning/``."""
    return supervised_baseline_output_dir(cv_corpus, anchor=anchor) / "tuning"


def expand_param_grid(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Produit cartésien d'un dict de listes (param → valeurs)."""
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values_list = [v if isinstance(v, list) else [v] for v in (grid[k] for k in keys)]
    return [dict(zip(keys, vals)) for vals in itertools.product(*values_list)]


def combo_id(model_key: str, params: Mapping[str, Any]) -> str:
    payload = {"model": model_key, **{k: params[k] for k in sorted(params)}}
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    parts = [model_key]
    for key in sorted(params):
        val = params[key]
        if val is None:
            parts.append(f"{key}None")
        elif isinstance(val, float):
            parts.append(f"{key}{val:g}")
        else:
            parts.append(f"{key}{val}")
    readable = "_".join(parts)[:90]
    return f"{readable}_{digest}"


def _normalize_param_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
        return None
    return value


def normalize_param_overrides(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(k): _normalize_param_value(v) for k, v in dict(params or {}).items()}


def build_tuned_registry_from_best_rows(
    best_by_model: Mapping[str, Mapping[str, Any]],
    *,
    base_registry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Construit un ``MODEL_REGISTRY`` avec les meilleurs params par modèle."""
    base = merge_model_registry(base_registry)
    out: Dict[str, Dict[str, Any]] = {}
    for model_key, row in best_by_model.items():
        if model_key not in base:
            continue
        block = deepcopy(base[model_key])
        params = dict(block.get("params") or {})
        raw = row.get("best_params") or row.get("params") or {}
        if isinstance(raw, str):
            raw = json.loads(raw)
        params.update(normalize_param_overrides(dict(raw)))
        block["params"] = params
        out[model_key] = block
    return out


def run_model_param_grid_cv(
    model_key: str,
    param_grid: Mapping[str, Any],
    *,
    X,
    y,
    groups,
    macros: Sequence[str],
    n_folds: int,
    seed: int,
    use_scaler: bool,
    base_params: Mapping[str, Any],
    selection_metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """CV GroupKFold pour chaque combo de ``param_grid`` d'un modèle."""
    rows: List[Dict[str, Any]] = []
    combos = expand_param_grid(param_grid)
    metric_col = (
        selection_metric
        if selection_metric.startswith("mean_")
        else f"mean_{selection_metric}"
    )
    for overrides in combos:
        params = {**dict(base_params), **normalize_param_overrides(overrides)}
        cid = combo_id(model_key, overrides)
        fold_rows = run_model_group_kfold_cv(
            model_key,
            X,
            y,
            groups,
            macros=macros,
            n_folds=n_folds,
            seed=seed,
            params=params,
            use_scaler=use_scaler,
        )
        summary = aggregate_cv_metrics(fold_rows)
        if summary.empty:
            continue
        srow = summary.iloc[0].to_dict()
        score = float(srow.get(metric_col, float("nan")))
        rows.append(
            {
                "combo_id": cid,
                "model": model_key,
                "best_params": json.dumps(normalize_param_overrides(overrides), default=str),
                "selection_score": score,
                **{k: srow[k] for k in srow if k != "model"},
            }
        )
        logger.info(
            "combo %s | %s = %.4f",
            cid,
            metric_col,
            score if score == score else float("nan"),
        )
    return pd.DataFrame(rows)


def select_best_row_per_model(grid_summary: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Meilleure ligne (max selection_score) par modèle."""
    if grid_summary.empty:
        return {}
    best: Dict[str, Dict[str, Any]] = {}
    for model_key, grp in grid_summary.groupby("model", sort=False):
        ordered = grp.sort_values("selection_score", ascending=False, kind="mergesort")
        best[str(model_key)] = ordered.iloc[0].to_dict()
    return best


def export_tuning_artifacts(
    tuning_dir: Path,
    grid_summary: pd.DataFrame,
    best_by_model: Mapping[str, Mapping[str, Any]],
    *,
    selection_metric: str,
    n_folds: int,
    seed: int,
) -> Dict[str, Any]:
    ensure_dir(tuning_dir)
    summary_path = tuning_dir / "grid_summary.csv"
    grid_summary.to_csv(summary_path, index=False)

    overall = None
    if best_by_model:
        overall = max(
            best_by_model.values(),
            key=lambda r: float(r.get("selection_score", float("-inf"))),
        )

    best_payload = {
        "selection_metric": selection_metric,
        "n_folds": n_folds,
        "seed": seed,
        "best_model": None if overall is None else str(overall.get("model")),
        "best_combo_id": None if overall is None else str(overall.get("combo_id")),
        "best_selection_score": None
        if overall is None
        else float(overall.get("selection_score", float("nan"))),
        "best_by_model": {
            mk: {
                "combo_id": row.get("combo_id"),
                "selection_score": float(row.get("selection_score", float("nan"))),
                "params": json.loads(row["best_params"])
                if isinstance(row.get("best_params"), str)
                else dict(row.get("best_params") or {}),
            }
            for mk, row in best_by_model.items()
        },
    }
    best_path = tuning_dir / "best_combo.json"
    best_path.write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return best_payload


def load_tuning_artifacts(tuning_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    summary_path = Path(tuning_dir) / "grid_summary.csv"
    best_path = Path(tuning_dir) / "best_combo.json"
    if not summary_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(
            f"Artefacts de tuning manquants sous {tuning_dir} "
            "(attendu : grid_summary.csv, best_combo.json)."
        )
    grid = pd.read_csv(summary_path)
    best = json.loads(best_path.read_text(encoding="utf-8"))
    return grid, best


def compare_default_vs_tuned_cv(
    default_cv_summary: pd.DataFrame,
    tuned_cv_summary: pd.DataFrame,
    *,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Table article : BA CV défaut (07) vs BA CV tuné (07b)."""
    mean_col = f"mean_{metric}"
    std_col = f"std_{metric}"
    rows: List[Dict[str, Any]] = []
    models = sorted(
        set(default_cv_summary.get("model", pd.Series(dtype=str)).astype(str))
        | set(tuned_cv_summary.get("model", pd.Series(dtype=str)).astype(str))
    )
    def _lookup(df: pd.DataFrame, model: str) -> Optional[pd.Series]:
        if df.empty or "model" not in df.columns:
            return None
        hit = df.loc[df["model"].astype(str) == model]
        return None if hit.empty else hit.iloc[0]

    for model in models:
        d = _lookup(default_cv_summary, model)
        t = _lookup(tuned_cv_summary, model)
        d_mean = float(d[mean_col]) if d is not None and mean_col in d else float("nan")
        d_std = float(d[std_col]) if d is not None and std_col in d else float("nan")
        t_mean = float(t[mean_col]) if t is not None and mean_col in t else float("nan")
        t_std = float(t[std_col]) if t is not None and std_col in t else float("nan")
        rows.append(
            {
                "model": model,
                "cv_ba_default": d_mean,
                "cv_ba_default_std": d_std,
                "cv_ba_tuned": t_mean,
                "cv_ba_tuned_std": t_std,
                "delta_ba": t_mean - d_mean
                if t_mean == t_mean and d_mean == d_mean
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def export_final_results_table(
    tuning_dir: Path,
    best_by_model: Mapping[str, Mapping[str, Any]],
    *,
    best_model: str,
    ood_ba_by_corpus: Mapping[str, Mapping[str, float]],
    cross_domain: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Tableau final : meilleur HP par modèle + BA CV + BA OOD.

    Écrit ``results_summary.csv`` et ``best_hyperparams.json`` sous ``tuning_dir``.
    """
    ensure_dir(tuning_dir)
    rows: List[Dict[str, Any]] = []
    corpora = list(ood_ba_by_corpus.keys())

    cross_by_model: Dict[str, Mapping[str, Any]] = {}
    if cross_domain is not None and not cross_domain.empty and "model" in cross_domain.columns:
        for _, crow in cross_domain.iterrows():
            cross_by_model[str(crow["model"])] = crow.to_dict()

    for model_key, row in best_by_model.items():
        params = row.get("best_params") or row.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        params = normalize_param_overrides(dict(params))
        out_row: Dict[str, Any] = {
            "model": model_key,
            "is_best_overall": str(model_key) == str(best_model),
            "combo_id": row.get("combo_id"),
            "best_params": json.dumps(params, ensure_ascii=False, default=str),
            "cv_balanced_accuracy": float(row.get("selection_score", float("nan"))),
            "cv_ba_std": float(row.get("std_balanced_accuracy", float("nan"))),
            "cv_accuracy": float(row.get("mean_accuracy", float("nan"))),
        }
        ood_vals: List[float] = []
        for corpus_id in corpora:
            ba = float(ood_ba_by_corpus.get(corpus_id, {}).get(model_key, float("nan")))
            out_row[f"ba_ood_{corpus_id}"] = ba
            if ba == ba:
                ood_vals.append(ba)
        out_row["ba_ood_avg"] = float(sum(ood_vals) / len(ood_vals)) if ood_vals else float("nan")
        out_row["ba_ood_worst"] = float(min(ood_vals)) if ood_vals else float("nan")
        cref = cross_by_model.get(str(model_key))
        if cref:
            if "ba_ood_avg" in cref and out_row["ba_ood_avg"] != out_row["ba_ood_avg"]:
                out_row["ba_ood_avg"] = float(cref["ba_ood_avg"])
            if "ba_ood_worst" in cref and out_row["ba_ood_worst"] != out_row["ba_ood_worst"]:
                out_row["ba_ood_worst"] = float(cref["ba_ood_worst"])
        rows.append(out_row)

    summary = pd.DataFrame(rows)
    if not summary.empty and "cv_balanced_accuracy" in summary.columns:
        summary = summary.sort_values(
            "cv_balanced_accuracy", ascending=False, kind="mergesort"
        ).reset_index(drop=True)
    summary_path = tuning_dir / "results_summary.csv"
    summary.to_csv(summary_path, index=False)

    best_hp = {
        str(mk): normalize_param_overrides(
            json.loads(row["best_params"])
            if isinstance(row.get("best_params"), str)
            else dict(row.get("best_params") or row.get("params") or {})
        )
        for mk, row in best_by_model.items()
    }
    (tuning_dir / "best_hyperparams.json").write_text(
        json.dumps(
            {
                "best_model": best_model,
                "best_hyperparams_by_model": best_hp,
                "results_summary_csv": str(summary_path),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Tableau final → %s", summary_path)
    return summary


def export_source_predictions_for_tuned_models(
    out_dir: Path,
    model_keys: Sequence[str],
    model_registry: Mapping[str, Mapping[str, Any]],
    X_btp,
    y_btp,
    btp_meta: pd.DataFrame,
    *,
    macros: Sequence[str],
    seed: int,
    text_col: str,
    group_col: str,
    label_col: str,
    method_prefix: str = "supervised_macro_baseline_tuned",
    best_model: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Fit final 100 % BTP → prédictions **train** (source) par modèle.

    Écrit :
    - ``transfer/models/<model>/source_macro_predictions.csv``
    - ``transfer/models/<model>/source_metrics.json`` (+ confusion si labels)
    - ``transfer/source_macro_predictions.csv`` pour le best model
    """
    ensure_dir(out_dir)
    transfer = Path(out_dir) / "transfer"
    models_root = transfer / "models"
    preds_by_model: Dict[str, pd.DataFrame] = {}

    for model_key in model_keys:
        spec = model_registry[model_key]
        _, preds, metrics = fit_final_and_predict_test(
            str(model_key),
            X_btp,
            y_btp,
            X_btp,
            btp_meta,
            macros=macros,
            seed=seed,
            params=spec.get("params"),
            use_scaler=spec.get("use_scaler"),
            method_name=f"{method_prefix}/{model_key}/train",
            text_col=text_col,
            group_col=group_col,
            label_col=label_col,
        )
        preds_by_model[str(model_key)] = preds
        model_dir = models_root / str(model_key)
        ensure_dir(model_dir)
        preds.to_csv(model_dir / "source_macro_predictions.csv", index=False)
        metrics_json = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
        (model_dir / "source_metrics.json").write_text(
            json.dumps(metrics_json, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        cm = metrics.get("_confusion_matrix")
        if cm is not None:
            pd.DataFrame(cm, index=list(macros), columns=list(macros)).to_csv(
                model_dir / "source_confusion_matrix.csv"
            )
        logger.info(
            "Preds train %s → %s (%d lignes)",
            model_key,
            model_dir / "source_macro_predictions.csv",
            len(preds),
        )

    if best_model and str(best_model) in preds_by_model:
        ensure_dir(transfer)
        preds_by_model[str(best_model)].to_csv(
            transfer / "source_macro_predictions.csv", index=False
        )
    return preds_by_model


def run_supervised_baseline_tuning(
    config_path: str | Path,
    *,
    anchor: Optional[Path] = None,
) -> Dict[str, Any]:
    """Point d'entrée YAML : grille CV BTP + eval OOD des meilleurs params."""
    root = anchor or TEXT_ROOT
    cfg_path = resolve_repo_path(str(config_path), repo_root=root)
    tune_cfg = load_yaml(cfg_path)

    base_rel = str(tune_cfg.get("base_config") or "configs/supervised_macro_baseline.yaml")
    base_cfg = load_yaml(resolve_repo_path(base_rel, repo_root=root))

    n_folds = int(tune_cfg.get("n_folds") or base_cfg.get("n_folds") or 7)
    seed = int(tune_cfg.get("seed") or base_cfg.get("seed") or 42)
    selection_metric = str(
        tune_cfg.get("selection_metric")
        or base_cfg.get("selection_metric")
        or "balanced_accuracy"
    )
    cv_corpus = str(tune_cfg.get("cv_corpus") or base_cfg.get("corpus") or "metallurgie")
    test_corpora = list(
        tune_cfg.get("test_corpora")
        or ["metallurgie", "caou", "nicollin"]
    )
    grids = dict(tune_cfg.get("grids") or {})
    model_keys = [k for k in TUNABLE_MODEL_KEYS if k in grids]
    if not model_keys:
        raise ValueError("Aucune grille sous `grids:` pour logistic_regression / random_forest / xgboost.")

    registry = merge_model_registry(base_cfg.get("models"))
    data_cfg = {
        **base_cfg,
        "corpus": cv_corpus,
        "n_folds": n_folds,
        "seed": seed,
        "selection_metric": selection_metric,
    }
    data = load_supervised_datasets(data_cfg, anchor=root)
    X_btp, y_btp, groups_btp = data["X_btp"], data["y_btp"], data["groups_btp"]
    macros = data["macros"]

    tuning_dir = resolve_repo_path(
        str(
            tune_cfg.get("output_dir")
            or f"output_test/{cv_corpus}/supervised_baseline/tuning"
        ),
        repo_root=root,
    )
    ensure_dir(tuning_dir)

    all_rows: List[pd.DataFrame] = []
    for model_key in model_keys:
        block = registry[model_key]
        logger.info("=== Grille %s (%d combos) ===", model_key, len(expand_param_grid(grids[model_key])))
        df_model = run_model_param_grid_cv(
            model_key,
            grids[model_key],
            X=X_btp,
            y=y_btp,
            groups=groups_btp,
            macros=macros,
            n_folds=n_folds,
            seed=seed,
            use_scaler=bool(block.get("use_scaler", False)),
            base_params=dict(block.get("params") or {}),
            selection_metric=selection_metric,
        )
        all_rows.append(df_model)

    grid_summary = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    best_by_model = select_best_row_per_model(grid_summary)
    best_payload = export_tuning_artifacts(
        tuning_dir,
        grid_summary,
        best_by_model,
        selection_metric=selection_metric,
        n_folds=n_folds,
        seed=seed,
    )
    tuned_registry = build_tuned_registry_from_best_rows(best_by_model, base_registry=registry)
    MODEL_KEYS = list(tuned_registry.keys())

    # Synthèse CV = meilleures lignes de grille (évite de rejouer 3 × K folds).
    cv_rows: List[Dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        row = dict(best_by_model[model_key])
        row["model"] = model_key
        cv_rows.append(row)
    cv_summary = pd.DataFrame(cv_rows)
    # Colonnes attendues par select_best_model / summarize_cross_domain
    if "mean_balanced_accuracy" not in cv_summary.columns and "selection_score" in cv_summary.columns:
        cv_summary["mean_balanced_accuracy"] = cv_summary["selection_score"]
    if "std_balanced_accuracy" not in cv_summary.columns:
        cv_summary["std_balanced_accuracy"] = float("nan")
    best_model = select_best_model(cv_summary, selection_metric=selection_metric)

    cv_out = supervised_baseline_tuned_output_dir(cv_corpus, anchor=root)
    # Export CV allégé : une ligne synthèse par modèle (pas de per-fold ici).
    ensure_dir(cv_out / "cv")
    cv_summary.to_csv(cv_out / "cv" / "cv_summary.csv", index=False)
    # per_fold vide mais fichier présent pour compat cache notebook 07
    pd.DataFrame(columns=["model", "fold", "accuracy", "balanced_accuracy", "macro_f1"]).to_csv(
        cv_out / "cv" / "cv_per_fold.csv", index=False
    )
    save_supervised_run_manifest(
        cv_out,
        best_model=best_model,
        selection_metric=selection_metric,
        seed=seed,
        n_folds=n_folds,
        test_corpus=cv_corpus,
        model_keys=MODEL_KEYS,
    )
    (cv_out / "tuning_ref.json").write_text(
        json.dumps(
            {"tuned": True, "tuning_dir": str(tuning_dir), "best_combo": best_payload},
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    for corpus_id in test_corpora:
        data_c = load_supervised_datasets({**data_cfg, "corpus": corpus_id}, anchor=root)
        out_dir = supervised_baseline_tuned_output_dir(corpus_id, anchor=root)
        preds_by_model, metrics_by_model = evaluate_all_models_on_test(
            MODEL_KEYS,
            tuned_registry,
            X_btp,
            y_btp,
            data_c["X_test"],
            data_c["test_meta"],
            macros=macros,
            seed=seed,
            text_col=data_c["target_text_col"],
            group_col=data_c["target_group_col"],
            label_col=data_c["target_label_col"],
            method_prefix="supervised_macro_baseline_tuned",
        )
        export_all_models_test_results(
            out_dir,
            preds_by_model,
            metrics_by_model,
            macros=macros,
            best_model=best_model,
        )
        save_supervised_run_manifest(
            out_dir,
            best_model=best_model,
            selection_metric=selection_metric,
            seed=seed,
            n_folds=n_folds,
            test_corpus=corpus_id,
            model_keys=MODEL_KEYS,
        )

    export_train = bool(tune_cfg.get("export_train_predictions", True))
    train_preds: Dict[str, pd.DataFrame] = {}
    if export_train:
        train_out = supervised_baseline_tuned_output_dir("btp", anchor=root)
        train_preds = export_source_predictions_for_tuned_models(
            train_out,
            MODEL_KEYS,
            tuned_registry,
            X_btp,
            y_btp,
            data["btp_meta"],
            macros=macros,
            seed=seed,
            text_col=str(data["text_col"]),
            group_col=str(data["group_col"]),
            label_col=str(data["label_col"]),
            method_prefix="supervised_macro_baseline_tuned",
            best_model=best_model,
        )
        save_supervised_run_manifest(
            train_out,
            best_model=best_model,
            selection_metric=selection_metric,
            seed=seed,
            n_folds=n_folds,
            test_corpus="btp",
            model_keys=MODEL_KEYS,
        )
        (train_out / "tuning_ref.json").write_text(
            json.dumps(
                {"tuned": True, "tuning_dir": str(tuning_dir), "split": "train_source"},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    ood_ba = _load_ood_ba_tuned(test_corpora, MODEL_KEYS, anchor=root)
    cross = summarize_cross_domain_generalization(
        cv_summary, ood_ba, model_keys=MODEL_KEYS
    )
    cross_path = cv_out / "cross_domain_generalization.csv"
    cross.to_csv(cross_path, index=False)
    # Copie aussi sous tuning_dir pour un seul point d'entrée job.
    cross.to_csv(tuning_dir / "cross_domain_generalization.csv", index=False)

    results_summary = export_final_results_table(
        tuning_dir,
        best_by_model,
        best_model=best_model,
        ood_ba_by_corpus=ood_ba,
        cross_domain=cross,
    )

    logger.info("Tuning terminé. Grille : %s", tuning_dir)
    logger.info("Meilleur modèle : %s", best_model)
    logger.info("Résultats : %s", tuning_dir / "results_summary.csv")
    return {
        "tuning_dir": tuning_dir,
        "grid_summary": grid_summary,
        "best_payload": best_payload,
        "cv_summary": cv_summary,
        "best_model": best_model,
        "tuned_registry": tuned_registry,
        "cross_domain": cross,
        "results_summary": results_summary,
        "train_predictions": train_preds,
    }


def _load_ood_ba_tuned(
    test_corpora: Sequence[str],
    model_keys: Sequence[str],
    *,
    anchor: Path,
) -> Dict[str, Dict[str, float]]:
    """BA OOD depuis ``supervised_baseline_tuned`` : ``{corpus: {model: ba}}``."""
    out: Dict[str, Dict[str, float]] = {}
    keys = {str(m) for m in model_keys}
    for corpus_id in test_corpora:
        metrics_path = (
            supervised_baseline_tuned_output_dir(corpus_id, anchor=anchor)
            / "transfer"
            / "all_models_test_metrics.csv"
        )
        if not metrics_path.is_file():
            continue
        df = pd.read_csv(metrics_path)
        if "model" not in df.columns or "balanced_accuracy" not in df.columns:
            continue
        by_model: Dict[str, float] = {}
        for _, row in df.iterrows():
            mk = str(row["model"])
            if mk in keys:
                by_model[mk] = float(row["balanced_accuracy"])
        if by_model:
            out[str(corpus_id)] = by_model
    return out


def run_supervised_baseline_tuning_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tune baseline supervisée sklearn (07b).")
    parser.add_argument(
        "--config",
        "--grid-config",
        dest="config",
        default="configs/tuning/supervised_macro_baseline_grid.yaml",
        help="YAML de grille (alias --grid-config pour jobs/_tune_common.sh)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_supervised_baseline_tuning(args.config)
    summary = result.get("results_summary")
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        print("\n=== Résultats finaux (best HP / modèle) ===", flush=True)
        cols = [
            c
            for c in (
                "model",
                "is_best_overall",
                "cv_balanced_accuracy",
                "ba_ood_avg",
                "ba_ood_worst",
                "best_params",
            )
            if c in summary.columns
        ]
        print(summary[cols].to_string(index=False), flush=True)
        print(f"\nArtefacts : {result['tuning_dir']}", flush=True)
        print(f"  - grid_summary.csv", flush=True)
        print(f"  - results_summary.csv", flush=True)
        print(f"  - best_combo.json / best_hyperparams.json", flush=True)
        print(
            "  - preds train : output_test/btp/supervised_baseline_tuned/transfer/",
            flush=True,
        )
        print(
            "  - preds OOD  : output_test/<corpus>/supervised_baseline_tuned/transfer/",
            flush=True,
        )
    return 0
