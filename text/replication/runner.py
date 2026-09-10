"""Exécute une réplication gelée d'un modèle macro.

Chaque réplication sépare volontairement la seed qui crée les partitions BTP
(``split_seed``) de celle qui pilote les aléas de l'apprentissage
(``training_seed``). Les sorties sont autonomes afin d'être rapatriées depuis
le Mésocentre sans checkpoints antérieurs.
"""

from __future__ import annotations

import copy
import json
import random
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from safer_core.io import ensure_dir, load_yaml, save_config_resolved
from safer_core.kfold_eval import group_kfold_splits
from safer_core.paths import TEXT_ROOT

REQUIRED_PREDICTION_COLUMNS = {"accident_id", "true_macro", "pred_macro", "prob_A0", "prob_A1", "prob_B", "prob_C"}


def load_replication_config(path: str | Path) -> Dict[str, Any]:
    """Charge et valide la configuration unique de la campagne."""
    config = load_yaml(path)
    training = config.get("training")
    models = config.get("models")
    if not isinstance(training, Mapping) or not isinstance(models, Mapping) or not models:
        raise ValueError("replication_config.yaml doit définir les sections training et models.")
    n_seeds = int(training.get("n_seeds", 0))
    if n_seeds < 1:
        raise ValueError("training.n_seeds doit être >= 1.")
    if int(training.get("n_folds", 0)) < 2:
        raise ValueError("training.n_folds doit être >= 2.")
    return config


def training_seeds(config: Mapping[str, Any]) -> List[int]:
    """Génère une liste stable de seeds à partir d'un unique entier maître."""
    training = dict(config["training"])
    n_seeds = int(training["n_seeds"])
    generator = int(training.get("seed_generator", 2026))
    return random.Random(generator).sample(range(1, 2_147_483_647), n_seeds)


def task_matrix(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Retourne une tâche par couple modèle × seed, dans un ordre reproductible."""
    return [
        {"model_id": str(model_id), "training_seed": int(seed)}
        for model_id in config["models"]
        for seed in training_seeds(config)
    ]


def output_root(config: Mapping[str, Any]) -> Path:
    raw = str(config.get("output_root", "output/replications"))
    return Path(raw) if Path(raw).is_absolute() else TEXT_ROOT / raw


def run_dir_for(config: Mapping[str, Any], model_id: str, training_seed: int) -> Path:
    return output_root(config) / str(model_id) / f"seed_{int(training_seed)}"


def set_global_training_seed(seed: int) -> None:
    """Initialise explicitement Python, NumPy et PyTorch avant chaque fit."""
    import os

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Répétable d'une exécution à l'autre; les opérations non déterministes
        # non supportées gardent leur comportement PyTorch habituel.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _model_config(config: Mapping[str, Any], model_id: str) -> Dict[str, Any]:
    models = dict(config["models"])
    if model_id not in models:
        raise ValueError(f"Modèle inconnu : {model_id}. Disponibles : {', '.join(models)}")
    spec = models[model_id]
    if not isinstance(spec, Mapping):
        raise ValueError(f"La définition de {model_id} doit être un mapping YAML.")
    runner = str(spec.get("runner", "")).strip()
    if runner not in {"contrastive", "supervised_macro_ft"}:
        raise ValueError(f"{model_id}.runner doit valoir contrastive ou supervised_macro_ft.")
    return dict(spec)


def _base_with_recipe(spec: Mapping[str, Any]) -> Dict[str, Any]:
    base_path = TEXT_ROOT / str(spec["base_config"])
    if not base_path.is_file():
        raise FileNotFoundError(f"Configuration méthode absente : {base_path}")
    return _deep_merge(load_yaml(base_path), dict(spec.get("overrides") or {}))


def _write_fold_partitions(
    raw_cfg: Mapping[str, Any],
    *,
    n_folds: int,
    split_seed: int,
    destination: Path,
) -> None:
    """Exporte les partitions afin de prouver l'absence de fuite entre accidents."""
    from scgm_text.dataset_text_raw import TextRawDataset

    data = dict(raw_cfg.get("data") or {})
    csv_key = "dataset_path" if "dataset_path" in data else "data_csv"
    csv_value = data.get(csv_key, "dataset/data_btp.csv")
    csv_path = Path(str(csv_value))
    if not csv_path.is_absolute():
        csv_path = TEXT_ROOT / csv_path
    dataset = TextRawDataset(
        str(csv_path),
        label_col=str(data.get("label_col", "pred_label")),
        pred_ok_col=str(data.get("pred_ok_col", "pred_ok")),
        group_col=str(data.get("group_col", "accident_id")),
        text_col=data.get("text_col", "sentence"),
    )
    groups = np.asarray(dataset.get_groups()).astype(str)
    rows: List[Dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(group_kfold_splits(groups, n_folds, split_seed)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(f"Fuite accident_id dans le fold {fold}: {sorted(overlap)[:3]}")
        rows.extend(
            {"fold": fold, "partition": "train", "row_index": int(i), "accident_id": groups[i]}
            for i in train_idx
        )
        rows.extend(
            {"fold": fold, "partition": "validation", "row_index": int(i), "accident_id": groups[i]}
            for i in val_idx
        )
    ensure_dir(destination.parent)
    pd.DataFrame(rows).to_csv(destination, index=False)


def _require_predictions(run_dir: Path, corpora: Iterable[str]) -> None:
    for corpus in corpora:
        path = run_dir / "predictions" / f"predictions_{corpus}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Prédictions absentes pour {corpus}: {path}")
        columns = set(pd.read_csv(path, nrows=1).columns)
        missing = REQUIRED_PREDICTION_COLUMNS - columns
        if missing:
            raise ValueError(f"Colonnes absentes dans {path}: {sorted(missing)}")


def _run_contrastive(
    raw: Dict[str, Any],
    *,
    model_id: str,
    run_dir: Path,
    training_seed: int,
    split_seed: int,
    n_folds: int,
    classifier: Mapping[str, Any],
) -> Dict[str, Any]:
    import dataclasses

    from contrastive_methods.config import config_to_resolved_dict, load_contrastive_config_from_dict
    from contrastive_methods.eval_corpus import run_final_classification_eval
    from contrastive_methods.kfold_train import get_contrastive_runner, run_kfold_loop

    method_name = str(raw.get("method_name") or raw.get("method"))
    raw = _deep_merge(raw, {
        "method_name": method_name,
        "output_dir": str(run_dir),
        "seed": int(training_seed),
        "split_seed": int(split_seed),
        "n_folds": int(n_folds),
        "final_fit_full_data": True,
    })
    cfg = load_contrastive_config_from_dict(method_name, raw, config_path="replication_config.yaml")
    runner = get_contrastive_runner(method_name)

    set_global_training_seed(training_seed)
    fold_rows, _ = run_kfold_loop(
        cfg,
        runner,
        fold_dir_fn=lambda fold: str(run_dir / "folds" / f"fold_{fold}"),
        log_prefix=f"replication/{model_id}/seed_{training_seed}",
        save_tables=False,
        post_eval_grid=[dict(classifier)],
        cleanup_fold_outputs=True,
    )
    cv_dir = ensure_dir(run_dir / "cv")
    pd.DataFrame(fold_rows).to_csv(cv_dir / "cv_per_fold.csv", index=False)
    required = [f"lr_0_val_{metric}" for metric in ("accuracy", "balanced_accuracy", "macro_f1")]
    if not fold_rows or any(key not in fold_rows[0] for key in required):
        raise RuntimeError("La post-évaluation LR gelée n'a pas produit toutes les métriques CV.")
    cv_summary = pd.DataFrame([{
        "model": model_id,
        "n_folds": len(fold_rows),
        "split_seed": int(split_seed),
        "training_seed": int(training_seed),
        "mean_balanced_accuracy": float(pd.DataFrame(fold_rows)["lr_0_val_balanced_accuracy"].mean()),
        "std_balanced_accuracy": float(pd.DataFrame(fold_rows)["lr_0_val_balanced_accuracy"].std(ddof=1)),
        "mean_accuracy": float(pd.DataFrame(fold_rows)["lr_0_val_accuracy"].mean()),
        "std_accuracy": float(pd.DataFrame(fold_rows)["lr_0_val_accuracy"].std(ddof=1)),
        "mean_macro_f1": float(pd.DataFrame(fold_rows)["lr_0_val_macro_f1"].mean()),
        "std_macro_f1": float(pd.DataFrame(fold_rows)["lr_0_val_macro_f1"].std(ddof=1)),
        "dispersion": "écart-type entre folds BTP",
    }])
    cv_summary.to_csv(cv_dir / "cv_summary.csv", index=False)
    (run_dir / "best_logistic_params.json").write_text(
        json.dumps(dict(classifier), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    epochs = [int(row["best_epoch"]) for row in fold_rows if row.get("best_epoch")]
    final_epochs = max(1, int(round(statistics.median(epochs)))) if epochs and cfg.final_epochs_from_cv else cfg.epochs
    final_cfg = dataclasses.replace(cfg, epochs=final_epochs, final_fit_full_data=True)
    final_cfg.extra = dict(cfg.extra)
    final_cfg.extra["final_epochs_used"] = final_epochs
    set_global_training_seed(training_seed + 1_000_003)
    result = runner(final_cfg)
    checkpoint = result.output_root / "checkpoints" / "best_model"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint final absent : {checkpoint}")
    run_final_classification_eval(final_cfg, checkpoint, result.output_root, cv_summary=cv_summary, classifier_overrides=classifier)
    resolved = config_to_resolved_dict(final_cfg)
    resolved.update({
        "replication_model_id": model_id,
        "training_seed": int(training_seed),
        "split_seed": int(split_seed),
        "final_fit_seed": int(training_seed + 1_000_003),
        "best_logistic_params": dict(classifier),
        "final_epochs_used": int(final_epochs),
    })
    save_config_resolved(resolved, run_dir)
    return {"final_epochs_used": int(final_epochs), "checkpoint_dir": str(checkpoint)}


def _run_supervised_macro_ft(
    raw: Dict[str, Any],
    *,
    model_id: str,
    run_dir: Path,
    training_seed: int,
    split_seed: int,
    n_folds: int,
) -> Dict[str, Any]:
    from supervised_macro_ft.train_runner import run_supervised_macro_ft_training

    training = dict(raw.get("training") or {})
    training.update({"seed": int(training_seed), "split_seed": int(split_seed), "n_folds": int(n_folds)})
    raw = {**raw, "training": training, "output_dir": str(run_dir), "method_name": "supervised_macro_ft"}
    set_global_training_seed(training_seed)
    result = run_supervised_macro_ft_training(None, cfg=raw, output_dir_override=run_dir)
    resolved = copy.deepcopy(raw)
    resolved.update({"replication_model_id": model_id, "training_seed": int(training_seed), "split_seed": int(split_seed)})
    save_config_resolved(resolved, run_dir)
    return {"final_epochs_used": result.get("final_epochs_used"), "checkpoint_dir": result.get("checkpoint_dir")}


def run_replication(
    config_path: str | Path,
    *,
    model_id: str,
    training_seed: int,
    refit: bool = False,
) -> Path:
    """Lance une tâche complète CV BTP + fit BTP + évaluation OOD."""
    config = load_replication_config(config_path)
    spec = _model_config(config, model_id)
    run_dir = run_dir_for(config, model_id, training_seed)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file() and not refit:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            _require_predictions(run_dir, config["training"]["test_corpora"])
            print(f"[replication] déjà complète : {run_dir}", flush=True)
            return run_dir

    ensure_dir(run_dir)
    raw = _base_with_recipe(spec)
    training = dict(config["training"])
    split_seed = int(training.get("split_seed", 42))
    n_folds = int(training["n_folds"])
    corpora = [str(item) for item in training["test_corpora"]]
    raw["test_corpora"] = corpora
    _write_fold_partitions(raw, n_folds=n_folds, split_seed=split_seed, destination=run_dir / "cv" / "fold_partitions.csv")
    manifest = {
        "status": "running", "model_id": model_id, "runner": spec["runner"],
        "training_seed": int(training_seed), "split_seed": split_seed,
        "n_folds": n_folds, "test_corpora": corpora,
        "recipe_path": str(Path(config_path).resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if spec["runner"] == "contrastive":
        classifier = dict(spec.get("classifier") or {})
        if not classifier:
            raise ValueError(f"{model_id} doit définir classifier pour l'évaluation contrastive.")
        details = _run_contrastive(raw, model_id=model_id, run_dir=run_dir, training_seed=training_seed,
                                   split_seed=split_seed, n_folds=n_folds, classifier=classifier)
    else:
        details = _run_supervised_macro_ft(raw, model_id=model_id, run_dir=run_dir,
                                           training_seed=training_seed, split_seed=split_seed, n_folds=n_folds)
    _require_predictions(run_dir, corpora)
    manifest.update({"status": "complete", **details})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _cleanup_completed_run(run_dir, dict(config.get("storage") or {}), manifest)
    return run_dir


def _cleanup_completed_run(run_dir: Path, storage: Mapping[str, Any], manifest: Dict[str, Any]) -> None:
    """Supprime les artefacts lourds seulement après exports complets validés."""
    if not bool(storage.get("cleanup_after_completion", False)):
        return
    removable = []
    if not bool(storage.get("keep_final_checkpoint", False)):
        removable.append("checkpoints")
    if not bool(storage.get("keep_embeddings", False)):
        removable.extend(["embeddings", "cache"])
    removed = []
    for name in removable:
        target = run_dir / name
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(name)
    manifest["cleanup"] = {
        "enabled": True,
        "removed_after_completion": removed,
        "fold_outputs_removed_during_cv": True,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
