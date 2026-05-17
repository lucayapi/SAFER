import os
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    StratifiedGroupKFold = None
    HAS_STRATIFIED_GROUP_KFOLD = False

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers


DEFAULT_CONTEXTUAL_PROMPT_TEMPLATE = (
    "Tâche: encoder une unité factuelle d'accident avec son contexte narratif.\n"
    "Récit: {accident_summary}\n"
    "Phrase: {sentence}"
)


# =========================================================
# Utils
# =========================================================
def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_hf_cache(hf_cache_folder: str = "./hf_cache") -> None:
    ensure_dir(hf_cache_folder)
    os.environ["HF_HOME"] = os.path.abspath(hf_cache_folder)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(
        os.environ["HF_HOME"], "sentence_transformers"
    )
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")


def get_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _append_suffix_to_path(filepath: str, suffix: str) -> str:
    p = Path(filepath)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x is None:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text != "" else None


# =========================================================
# Data cleaning
# =========================================================
def clean_supervised_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    group_col: str,
    unit_id_col: Optional[str] = None,
) -> pd.DataFrame:
    required = {text_col, label_col, group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    work = df.copy()
    work = work[work[label_col].notna() & work[group_col].notna()].copy()

    if unit_id_col is not None and unit_id_col in work.columns:
        work["unit_id"] = work[unit_id_col].astype(str)
    else:
        work["unit_id"] = np.arange(len(work)).astype(str)

    work[group_col] = work[group_col].astype(str)
    work[text_col] = work[text_col].fillna("").astype(str).str.strip()
    work[label_col] = work[label_col].astype(str)

    work = work[work[text_col] != ""].copy()
    work = work.reset_index(drop=True)

    if work.empty:
        raise ValueError("Aucune donnée valide après nettoyage.")

    if work[label_col].nunique() < 2:
        raise ValueError("Il faut au moins 2 classes distinctes dans le dataset.")

    return work


def build_effective_model_input_text(
    df: pd.DataFrame,
    text_col: str,
    use_contextual_prompt_with_summary: bool = False,
    summary_col: Optional[str] = None,
    contextual_prompt_template: str = DEFAULT_CONTEXTUAL_PROMPT_TEMPLATE,
    fixed_instruction_prefix: Optional[str] = None,
    output_col: str = "__model_input_text__",
) -> pd.DataFrame:
    """
    Construit la colonne texte réellement encodée par le modèle.

    Cas possibles :
    1) phrase seule
    2) fixed_prefix + phrase
    3) prompt structuré (récit + phrase)
    4) fixed_prefix + prompt structuré (récit + phrase)
    """
    work = df.copy()

    if text_col not in work.columns:
        raise ValueError(f"Colonne texte introuvable: {text_col}")

    sentences = work[text_col].fillna("").astype(str).str.strip()
    fixed_instruction_prefix = _normalize_optional_text(fixed_instruction_prefix)

    if use_contextual_prompt_with_summary:
        if summary_col is None:
            raise ValueError(
                "summary_col doit être fourni si use_contextual_prompt_with_summary=True."
            )
        if summary_col not in work.columns:
            raise ValueError(
                f"summary_col='{summary_col}' introuvable dans le DataFrame."
            )

        summaries = work[summary_col].fillna("").astype(str).str.strip()

        base_texts = [
            contextual_prompt_template.format(
                accident_summary=summary,
                sentence=sentence,
            )
            for summary, sentence in zip(summaries.tolist(), sentences.tolist())
        ]
    else:
        base_texts = sentences.tolist()

    if fixed_instruction_prefix is not None:
        final_texts = [
            f"{fixed_instruction_prefix}\n\n{txt}".strip()
            for txt in base_texts
        ]
    else:
        final_texts = [str(txt).strip() for txt in base_texts]

    work[output_col] = pd.Series(final_texts, index=work.index).fillna("").astype(str).str.strip()

    if (work[output_col] == "").any():
        raise ValueError("Certaines entrées du texte modèle final sont vides.")

    return work


def hf_dataset_from_df(df: pd.DataFrame, text_col: str) -> Dataset:
    return Dataset.from_dict(
        {
            "sentence": df[text_col].tolist(),
            "label": df["label"].tolist(),
        }
    )


# =========================================================
# Diagnostics helpers
# =========================================================
def build_fold_label_diagnostics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> Dict[str, Any]:
    train_counts = train_df[label_col].astype(str).value_counts().sort_index()
    test_counts = test_df[label_col].astype(str).value_counts().sort_index()

    train_labels = set(train_counts.index.tolist())
    test_labels = set(test_counts.index.tolist())

    singleton_labels_train = sorted(train_counts[train_counts == 1].index.tolist())
    singleton_labels_test = sorted(test_counts[test_counts == 1].index.tolist())

    diagnostics = {
        "label_distribution_train": {str(k): int(v) for k, v in train_counts.items()},
        "label_distribution_test": {str(k): int(v) for k, v in test_counts.items()},
        "n_singleton_labels_train": int((train_counts == 1).sum()),
        "n_singleton_labels_test": int((test_counts == 1).sum()),
        "singleton_labels_train": singleton_labels_train,
        "singleton_labels_test": singleton_labels_test,
        "classes_absent_from_train": sorted(list(test_labels - train_labels)),
        "classes_absent_from_test": sorted(list(train_labels - test_labels)),
    }
    return diagnostics


# =========================================================
# K-fold helpers
# =========================================================
def make_group_kfold_splits(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    n_splits: int = 4,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits doit être >= 2.")

    if df[group_col].nunique() < n_splits:
        raise ValueError(
            f"Nombre de groupes insuffisant pour {n_splits} folds : "
            f"{df[group_col].nunique()} groupes seulement."
        )

    X_dummy = np.zeros(len(df))
    y = df[label_col].astype(str).to_numpy()
    groups = df[group_col].astype(str).to_numpy()

    if HAS_STRATIFIED_GROUP_KFOLD:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            splits = list(splitter.split(X_dummy, y=y, groups=groups))
            print(f"[{stamp()}] K-fold splitter = StratifiedGroupKFold")
            return splits
        except Exception as e:
            print(
                f"[{stamp()}] [WARN] StratifiedGroupKFold indisponible/échoué "
                f"({e}). Fallback sur GroupKFold."
            )

    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(X_dummy, y=y, groups=groups))
    print(f"[{stamp()}] K-fold splitter = GroupKFold")
    return splits


# =========================================================
# Aggregation helpers
# =========================================================
def _aggregate_scalar(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    cleaned = []
    for v in values:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        cleaned.append(float(v))

    if len(cleaned) == 0:
        return {"mean": None, "std": None, "n_folds": 0}

    arr = np.asarray(cleaned, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_folds": int(len(arr)),
    }


def aggregate_kfold_final_metrics_simple(
    fold_metrics: List[Dict[str, Any]],
    eval_label_cols: List[str],
    selection_label_col: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "test_separation": {},
        "model_selection_metric": {
            "name": f"test_separation.{selection_label_col}.delta_ratio",
            "stats": None,
        },
    }

    for label_col in eval_label_cols:
        out["test_separation"][label_col] = {}
        for metric_name in ["S", "W", "B", "delta_ratio", "delta_pct"]:
            vals = []
            for fm in fold_metrics:
                vals.append(
                    fm.get("test_separation", {})
                      .get(label_col, {})
                      .get(metric_name, None)
                )
            out["test_separation"][label_col][metric_name] = _aggregate_scalar(vals)

    out["model_selection_metric"]["stats"] = (
        out["test_separation"]
           .get(selection_label_col, {})
           .get("delta_ratio", {"mean": None, "std": None, "n_folds": 0})
    )

    return out


# =========================================================
# Embeddings + metrics
# =========================================================
def load_model(
    model_name_or_path: str,
    device: str,
    hf_cache_folder: str,
) -> SentenceTransformer:
    setup_hf_cache(hf_cache_folder)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    st_kwargs = dict(
        model_name_or_path=model_name_or_path,
        cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
        device=device,
    )
    if hf_token:
        st_kwargs["token"] = hf_token
    return SentenceTransformer(**st_kwargs)


def encode_texts(
    texts: List[str],
    model_name_or_path: str,
    batch_size: int = 128,
    normalize: bool = True,
    device: Optional[str] = None,
    hf_cache_folder: str = "./hf_cache",
) -> np.ndarray:
    device = get_device(device)
    model = load_model(model_name_or_path, device, hf_cache_folder)

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return emb


def separation_metric_cosine(
    X: np.ndarray,
    y: np.ndarray,
    singleton_policy: str = "zero",
    eps: float = 1e-12,
) -> Dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    mask_finite = np.isfinite(X).all(axis=1)
    X = X[mask_finite]
    y = y[mask_finite]

    if len(X) < 2:
        return {
            "n": len(X),
            "n_classes": len(np.unique(y)),
            "S": np.nan,
            "W": np.nan,
            "B": np.nan,
            "delta_ratio": np.nan,
            "delta_pct": np.nan,
        }

    sq_norms = np.einsum("ij,ij->i", X, X)
    norms = np.sqrt(np.maximum(sq_norms, eps))

    mask_nonzero = norms > eps
    X = X[mask_nonzero]
    y = y[mask_nonzero]
    norms = norms[mask_nonzero]

    n = len(X)
    if n < 2:
        return {
            "n": n,
            "n_classes": len(np.unique(y)),
            "S": np.nan,
            "W": np.nan,
            "B": np.nan,
            "delta_ratio": np.nan,
            "delta_pct": np.nan,
        }

    Xn = X / norms[:, None]

    sum_all = Xn.sum(axis=0)
    self_all = np.einsum("ij,ij->i", Xn, Xn).sum()
    mean_sim_all = (sum_all @ sum_all - self_all) / (n * (n - 1))
    S = 1.0 - mean_sim_all

    classes, inv = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    counts = np.bincount(inv)

    class_sums = np.zeros((n_classes, Xn.shape[1]), dtype=np.float64)
    np.add.at(class_sums, inv, Xn)

    sqnorm_class_sums = np.einsum("ij,ij->i", class_sums, class_sums)

    Wc = np.full(n_classes, np.nan, dtype=np.float64)
    valid = counts >= 2

    mean_sim_c = np.full(n_classes, np.nan, dtype=np.float64)
    mean_sim_c[valid] = (
        (sqnorm_class_sums[valid] - counts[valid]) /
        (counts[valid] * (counts[valid] - 1))
    )
    Wc[valid] = 1.0 - mean_sim_c[valid]

    if singleton_policy == "zero":
        Wc[~valid] = 0.0
        W = np.sum((counts / n) * Wc)
    elif singleton_policy == "drop":
        if valid.sum() == 0:
            W = np.nan
        else:
            weights = counts[valid] / counts[valid].sum()
            W = np.sum(weights * Wc[valid])
    else:
        raise ValueError("singleton_policy doit être 'zero' ou 'drop'")

    if abs(S) < eps:
        S = 0.0
    if abs(W) < eps:
        W = 0.0

    B = S - W

    if abs(S) < eps:
        delta_ratio = np.nan
        delta_pct = np.nan
    else:
        delta_ratio = B / S
        delta_pct = 100.0 * delta_ratio

    return {
        "n": int(n),
        "n_classes": int(n_classes),
        "S": float(S),
        "W": float(W),
        "B": float(B),
        "delta_ratio": float(delta_ratio),
        "delta_pct": float(delta_pct),
    }


def compute_separation_metrics_for_model(
    model_name_or_path: str,
    df: pd.DataFrame,
    text_col: str,
    label_cols: List[str],
    batch_size: int = 128,
    normalize: bool = True,
    singleton_policy: str = "zero",
    device: Optional[str] = None,
    hf_cache_folder: str = "./hf_cache",
) -> Dict[str, Dict[str, float]]:
    if len(df) == 0:
        return {}

    X = encode_texts(
        texts=df[text_col].tolist(),
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        normalize=normalize,
        device=device,
        hf_cache_folder=hf_cache_folder,
    )

    results = {}
    for col in label_cols:
        if col not in df.columns:
            continue
        y = df[col].astype(str).to_numpy()
        results[col] = separation_metric_cosine(
            X=X,
            y=y,
            singleton_policy=singleton_policy,
        )

    return results


def export_embeddings_csv(
    df: pd.DataFrame,
    model_name_or_path: str,
    output_csv: str,
    text_col: str,
    extra_cols: Optional[List[str]] = None,
    batch_size: int = 128,
    normalize: bool = True,
    device: Optional[str] = None,
    hf_cache_folder: str = "./hf_cache",
) -> str:
    ensure_dir(str(Path(output_csv).parent))

    X = encode_texts(
        texts=df[text_col].tolist(),
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        normalize=normalize,
        device=device,
        hf_cache_folder=hf_cache_folder,
    )

    dim_cols = [f"dim_{j:04d}" for j in range(1, X.shape[1] + 1)]
    out = pd.DataFrame(X, columns=dim_cols)

    if extra_cols is None:
        extra_cols = []

    keep_cols = [c for c in extra_cols if c in df.columns]
    for c in reversed(keep_cols):
        out.insert(0, c, df[c].values)

    out.to_csv(output_csv, index=False)
    print(f"[{stamp()}] Embeddings exportés: {output_csv}")
    return output_csv


# =========================================================
# SupCon loss
# =========================================================
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss :
    - une phrase par exemple
    - positifs = même classe dans le batch
    """

    def __init__(
        self,
        model: SentenceTransformer,
        temperature: float = 0.07,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = float(temperature)
        self.normalize_embeddings = bool(normalize_embeddings)

    def forward(self, sentence_features, labels: Optional[torch.Tensor] = None):
        if labels is None:
            raise ValueError("SupConLoss nécessite des labels.")

        embeddings = self.model(sentence_features[0])["sentence_embedding"]

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        labels = labels.view(-1)
        device = embeddings.device
        batch_size = embeddings.size(0)

        if batch_size < 2:
            return embeddings.sum() * 0.0

        logits = torch.matmul(embeddings, embeddings.T) / self.temperature

        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        logits = logits.masked_fill(eye_mask, float("-inf"))

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(device)
        positive_mask = positive_mask & (~eye_mask)

        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchor_mask = positives_per_anchor > 0

        if not valid_anchor_mask.any():
            return embeddings.sum() * 0.0

        log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = logits - log_denom

        mean_log_prob_pos = (
            (positive_mask.float() * log_prob).sum(dim=1) /
            positives_per_anchor.clamp(min=1).float()
        )

        loss = -mean_log_prob_pos[valid_anchor_mask].mean()
        return loss


# =========================================================
# Simple fold runner
# =========================================================
def _run_single_supcon_fold_simple(
    fold_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_root: str,
    base_model_name: str,
    text_col: str,
    original_text_col: str,
    label_col: str,
    group_col: str,
    unit_id_col: Optional[str],
    summary_col: Optional[str],
    use_contextual_prompt_with_summary: bool,
    contextual_prompt_template: str,
    fixed_instruction_prefix: Optional[str],
    eval_label_cols: List[str],
    selection_label_col: str,
    hf_cache_folder: str,
    min_count_per_class_in_train: int,
    batch_size_train: int,
    batch_size_eval: int,
    batch_size_encode: int,
    num_train_epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    normalize_for_eval: bool,
    singleton_policy: str,
    max_eval_triplets: int,
    supcon_temperature: float,
    supcon_normalize_train_embeddings: bool,
    seed: int,
    device: Optional[str],
    export_full_embeddings_csv: Optional[str],
) -> Dict[str, Any]:
    ensure_dir(output_root)

    fixed_instruction_prefix = _normalize_optional_text(fixed_instruction_prefix)
    use_fixed_instruction_prefix = fixed_instruction_prefix is not None

    fold_label_diagnostics = build_fold_label_diagnostics(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
    )

    unique_labels = sorted(train_df[label_col].unique())
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["label"] = train_df[label_col].map(label2id)
    test_df["label"] = test_df[label_col].map(label2id)

    n_test_rows_before_mapping = int(len(test_df))

    train_df = train_df[train_df["label"].notna()].copy().reset_index(drop=True)
    test_df = test_df[test_df["label"].notna()].copy().reset_index(drop=True)

    n_test_rows_after_mapping = int(len(test_df))
    n_test_rows_dropped_unseen_labels = n_test_rows_before_mapping - n_test_rows_after_mapping

    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    if train_df.empty:
        raise ValueError(f"Fold {fold_id} | Train vide.")
    if train_df[label_col].nunique() < 2:
        raise ValueError(f"Fold {fold_id} | Moins de 2 classes dans le train.")

    print(
        f"[{stamp()}] Fold {fold_id} | "
        f"train rows={len(train_df):,} | test rows={len(test_df):,}"
    )

    split_dir = os.path.join(output_root, "splits")
    ensure_dir(split_dir)

    train_csv = os.path.join(split_dir, "train.csv")
    test_csv = os.path.join(split_dir, "test.csv")

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    with open(os.path.join(output_root, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_root, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_root, "fold_label_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(fold_label_diagnostics, f, ensure_ascii=False, indent=2)

    split_info = {
        "fold_id": fold_id,
        "base_model_name": base_model_name,
        "text_col_used_by_model": text_col,
        "original_text_col": original_text_col,
        "summary_col": summary_col,
        "use_contextual_prompt_with_summary": use_contextual_prompt_with_summary,
        "contextual_prompt_template": (
            contextual_prompt_template if use_contextual_prompt_with_summary else None
        ),
        "use_fixed_instruction_prefix": use_fixed_instruction_prefix,
        "fixed_instruction_prefix": fixed_instruction_prefix,
        "label_col": label_col,
        "group_col": group_col,
        "eval_label_cols": eval_label_cols,
        "selection_label_col": selection_label_col,
        "selection_metric": f"test_separation.{selection_label_col}.delta_ratio",
        "seed": seed,
        "min_count_per_class_in_train": min_count_per_class_in_train,
        "n_rows_train": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "n_rows_test_before_train_label_mapping": n_test_rows_before_mapping,
        "n_rows_test_after_train_label_mapping": n_test_rows_after_mapping,
        "n_test_rows_dropped_unseen_labels": int(n_test_rows_dropped_unseen_labels),
        "n_groups_train": int(train_df[group_col].nunique()) if group_col in train_df.columns else None,
        "n_groups_test": int(test_df[group_col].nunique()) if group_col in test_df.columns else None,
        "n_classes_train": int(train_df["label"].nunique()),
        "n_classes_test": int(test_df["label"].nunique()),
        "label_distribution_train": fold_label_diagnostics["label_distribution_train"],
        "label_distribution_test": fold_label_diagnostics["label_distribution_test"],
        "n_singleton_labels_train": fold_label_diagnostics["n_singleton_labels_train"],
        "n_singleton_labels_test": fold_label_diagnostics["n_singleton_labels_test"],
        "singleton_labels_train": fold_label_diagnostics["singleton_labels_train"],
        "singleton_labels_test": fold_label_diagnostics["singleton_labels_test"],
        "classes_absent_from_train": fold_label_diagnostics["classes_absent_from_train"],
        "classes_absent_from_test": fold_label_diagnostics["classes_absent_from_test"],
        "train_loss_name": "SupConLoss",
        "supcon_temperature": float(supcon_temperature),
        "supcon_normalize_train_embeddings": bool(supcon_normalize_train_embeddings),
    }
    with open(os.path.join(output_root, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    train_dataset = hf_dataset_from_df(train_df, text_col=text_col)

    print(f"[{stamp()}] Fold {fold_id} | Loading base model: {base_model_name}")
    model = load_model(base_model_name, get_device(device), hf_cache_folder)

    train_loss = SupConLoss(
        model=model,
        temperature=supcon_temperature,
        normalize_embeddings=supcon_normalize_train_embeddings,
    )

    use_bf16 = bool(
        str(get_device(device)).startswith("cuda")
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = bool(str(get_device(device)).startswith("cuda") and not use_bf16)

    steps_per_epoch = max(
        1,
        math.ceil(len(train_df) / max(1, batch_size_train * gradient_accumulation_steps))
    )

    print(
        f"[{stamp()}] Fold {fold_id} | steps_per_epoch={steps_per_epoch} | "
        f"epochs={num_train_epochs} | temperature={supcon_temperature}"
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_root,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        fp16=use_fp16,
        bf16=use_bf16,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 2),
        report_to=[],
        seed=seed + fold_id,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    print(f"[{stamp()}] Fold {fold_id} | Start training")
    trainer.train()

    final_model_dir = os.path.join(output_root, "final_model")
    trainer.model.save_pretrained(final_model_dir)
    print(f"[{stamp()}] Fold {fold_id} | Final model saved to: {final_model_dir}")

    test_sep = compute_separation_metrics_for_model(
        model_name_or_path=final_model_dir,
        df=test_df,
        text_col=text_col,
        label_cols=eval_label_cols,
        batch_size=batch_size_eval,
        normalize=normalize_for_eval,
        singleton_policy=singleton_policy,
        device=device,
        hf_cache_folder=hf_cache_folder,
    )

    selection_score = (
        test_sep.get(selection_label_col, {})
                .get("delta_ratio", None)
    )

    final_metrics = {
        "fold_id": fold_id,
        "selection_label_col": selection_label_col,
        "selection_metric": f"test_separation.{selection_label_col}.delta_ratio",
        "selection_score": selection_score,
        "eval_label_cols": eval_label_cols,
        "test_separation": test_sep,
        "train_loss_name": "SupConLoss",
        "supcon_temperature": float(supcon_temperature),
        "use_fixed_instruction_prefix": use_fixed_instruction_prefix,
        "fixed_instruction_prefix": fixed_instruction_prefix,
    }

    with open(os.path.join(output_root, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print(f"[{stamp()}] Fold {fold_id} | Final metrics:")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))

    exported_embeddings_csv = None
    if export_full_embeddings_csv is not None:
        fold_export_csv = _append_suffix_to_path(
            export_full_embeddings_csv,
            f"__fold{fold_id}"
        )

        extra_cols = _dedupe_keep_order(
            [
                "unit_id",
                group_col,
                original_text_col,
                summary_col,
                text_col,
                label_col,
                "label",
            ] + [c for c in eval_label_cols if c in train_df.columns and c != label_col]
        )

        export_embeddings_csv(
            df=pd.concat([train_df, test_df], axis=0, ignore_index=True),
            model_name_or_path=final_model_dir,
            output_csv=fold_export_csv,
            text_col=text_col,
            extra_cols=extra_cols,
            batch_size=batch_size_encode,
            normalize=normalize_for_eval,
            device=device,
            hf_cache_folder=hf_cache_folder,
        )
        exported_embeddings_csv = fold_export_csv

    del trainer, model
    gc.collect()
    if str(get_device(device)).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "fold_id": fold_id,
        "output_root": output_root,
        "final_model_dir": final_model_dir,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "final_metrics": final_metrics,
        "supcon_temperature": float(supcon_temperature),
        "full_embeddings_csv": exported_embeddings_csv,
    }


# =========================================================
# K-fold simple SupCon
# =========================================================
def finetune_supcon_embedder_research_kfold_simple(
    input_csv: str,
    output_root: str,
    base_model_name: str,
    text_col: str = "sentence",
    label_col: str = "pred_label",
    group_col: str = "doc_id",
    unit_id_col: Optional[str] = None,
    summary_col: Optional[str] = "accident_summary",
    use_contextual_prompt_with_summary: bool = False,
    contextual_prompt_template: str = DEFAULT_CONTEXTUAL_PROMPT_TEMPLATE,
    fixed_instruction_prefix: Optional[str] = None,
    eval_label_cols: Optional[List[str]] = None,
    selection_label_col: Optional[str] = None,
    hf_cache_folder: str = "./hf_cache",
    n_splits: int = 4,
    min_count_per_class_in_train: int = 4,
    batch_size_train: int = 32,
    batch_size_eval: int = 64,
    batch_size_encode: int = 128,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    normalize_for_eval: bool = True,
    singleton_policy: str = "zero",
    max_eval_triplets: int = 3000,
    supcon_temperature: float = 0.07,
    supcon_normalize_train_embeddings: bool = True,
    seed: int = 42,
    device: Optional[str] = None,
    export_full_embeddings_csv: Optional[str] = None,
) -> Dict[str, Any]:
    set_global_seed(seed)
    setup_hf_cache(hf_cache_folder)
    ensure_dir(output_root)

    if eval_label_cols is None:
        eval_label_cols = [label_col]

    if selection_label_col is None:
        selection_label_col = label_col

    device = get_device(device)
    fixed_instruction_prefix = _normalize_optional_text(fixed_instruction_prefix)
    use_fixed_instruction_prefix = fixed_instruction_prefix is not None

    print(f"[{stamp()}] Device = {device}")

    print(f"[{stamp()}] Chargement CSV: {input_csv}")
    raw_df = pd.read_csv(input_csv)

    cleaned_df = clean_supervised_dataframe(
        df=raw_df,
        text_col=text_col,
        label_col=label_col,
        group_col=group_col,
        unit_id_col=unit_id_col,
    )

    print(
        f"[{stamp()}] Dataset nettoyé (avant filtrage global) | rows={len(cleaned_df):,} | "
        f"groups={cleaned_df[group_col].nunique():,} | classes={cleaned_df[label_col].nunique():,}"
    )

    global_counts = cleaned_df[label_col].value_counts()
    keep_labels = global_counts[global_counts >= min_count_per_class_in_train].index.tolist()
    dropped_global = global_counts[global_counts < min_count_per_class_in_train].sort_index()

    if len(dropped_global) > 0:
        print(
            f"[{stamp()}] [WARN] Filtrage global | {len(dropped_global)} labels supprimés "
            f"car < {min_count_per_class_in_train} exemples dans tout le dataset."
        )

    df = cleaned_df[cleaned_df[label_col].isin(keep_labels)].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset vide après filtrage global des classes rares.")
    if df[label_col].nunique() < 2:
        raise ValueError("Moins de 2 classes après filtrage global des classes rares.")

    model_input_text_col = "__model_input_text__"
    df = build_effective_model_input_text(
        df=df,
        text_col=text_col,
        use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
        summary_col=summary_col,
        contextual_prompt_template=contextual_prompt_template,
        fixed_instruction_prefix=fixed_instruction_prefix,
        output_col=model_input_text_col,
    )

    print(
        f"[{stamp()}] Dataset après filtrage global | rows={len(df):,} | "
        f"groups={df[group_col].nunique():,} | classes={df[label_col].nunique():,}"
    )
    print(
        f"[{stamp()}] Texte utilisé par le modèle = "
        f"{'prompt+récit+phrase' if use_contextual_prompt_with_summary else 'phrase seule'}"
    )
    print(
        f"[{stamp()}] Fixed instruction prefix = "
        f"{'activé' if use_fixed_instruction_prefix else 'désactivé'}"
    )

    splits = make_group_kfold_splits(
        df=df,
        group_col=group_col,
        label_col=label_col,
        n_splits=n_splits,
        seed=seed,
    )

    all_fold_results = []
    all_fold_final_metrics = []

    global_info = {
        "input_csv": input_csv,
        "base_model_name": base_model_name,
        "text_col": text_col,
        "summary_col": summary_col,
        "model_input_text_col": model_input_text_col,
        "use_contextual_prompt_with_summary": use_contextual_prompt_with_summary,
        "contextual_prompt_template": (
            contextual_prompt_template if use_contextual_prompt_with_summary else None
        ),
        "use_fixed_instruction_prefix": use_fixed_instruction_prefix,
        "fixed_instruction_prefix": fixed_instruction_prefix,
        "label_col": label_col,
        "group_col": group_col,
        "eval_label_cols": eval_label_cols,
        "selection_label_col": selection_label_col,
        "selection_metric": f"test_separation.{selection_label_col}.delta_ratio",
        "seed": seed,
        "n_splits": n_splits,
        "min_count_per_class_in_train": min_count_per_class_in_train,
        "supcon_temperature": float(supcon_temperature),
        "supcon_normalize_train_embeddings": bool(supcon_normalize_train_embeddings),
        "n_rows_total_before_global_filtering": int(len(cleaned_df)),
        "n_rows_total": int(len(df)),
        "n_groups_total": int(df[group_col].nunique()),
        "n_classes_total": int(df[label_col].nunique()),
        "global_label_counts_after_filtering": {
            str(k): int(v) for k, v in df[label_col].value_counts().sort_index().items()
        },
        "dropped_labels_below_global_min_count": {
            str(k): int(v) for k, v in dropped_global.items()
        },
        "train_loss_name": "SupConLoss",
    }
    with open(os.path.join(output_root, "kfold_info.json"), "w", encoding="utf-8") as f:
        json.dump(global_info, f, ensure_ascii=False, indent=2)

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        print("=" * 80)
        print(f"[{stamp()}] START FOLD {fold_id}/{n_splits}")
        print("=" * 80)

        fold_root = os.path.join(output_root, f"fold_{fold_id}")
        ensure_dir(fold_root)

        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        fold_result = _run_single_supcon_fold_simple(
            fold_id=fold_id,
            train_df=train_df,
            test_df=test_df,
            output_root=fold_root,
            base_model_name=base_model_name,
            text_col=model_input_text_col,
            original_text_col=text_col,
            label_col=label_col,
            group_col=group_col,
            unit_id_col=unit_id_col,
            summary_col=summary_col,
            use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
            contextual_prompt_template=contextual_prompt_template,
            fixed_instruction_prefix=fixed_instruction_prefix,
            eval_label_cols=eval_label_cols,
            selection_label_col=selection_label_col,
            hf_cache_folder=hf_cache_folder,
            min_count_per_class_in_train=min_count_per_class_in_train,
            batch_size_train=batch_size_train,
            batch_size_eval=batch_size_eval,
            batch_size_encode=batch_size_encode,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            normalize_for_eval=normalize_for_eval,
            singleton_policy=singleton_policy,
            max_eval_triplets=max_eval_triplets,
            supcon_temperature=supcon_temperature,
            supcon_normalize_train_embeddings=supcon_normalize_train_embeddings,
            seed=seed,
            device=device,
            export_full_embeddings_csv=export_full_embeddings_csv,
        )

        all_fold_results.append(fold_result)
        all_fold_final_metrics.append(fold_result["final_metrics"])

    aggregate_metrics = aggregate_kfold_final_metrics_simple(
        fold_metrics=all_fold_final_metrics,
        eval_label_cols=eval_label_cols,
        selection_label_col=selection_label_col,
    )

    with open(os.path.join(output_root, "kfold_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_fold_results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_root, "kfold_aggregate_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, ensure_ascii=False, indent=2)

    print(f"[{stamp()}] K-fold aggregate metrics:")
    print(json.dumps(aggregate_metrics, ensure_ascii=False, indent=2))

    print(
        f"[{stamp()}] Selection metric = "
        f"mean test delta_ratio on '{selection_label_col}' = "
        f"{aggregate_metrics['model_selection_metric']['stats']['mean']}"
    )

    return {
        "output_root": output_root,
        "n_splits": n_splits,
        "fold_results": all_fold_results,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_metrics_json": os.path.join(output_root, "kfold_aggregate_metrics.json"),
        "fold_results_json": os.path.join(output_root, "kfold_results.json"),
        "selection_score_mean_test_delta_ratio": (
            aggregate_metrics["model_selection_metric"]["stats"]["mean"]
        ),
    }


# =========================================================
# Final training on full data (SupCon)
# =========================================================
def train_final_supcon_model_on_full_data(
    input_csv: str,
    output_root: str,
    base_model_name: str,
    text_col: str = "sentence",
    label_col: str = "pred_label",
    group_col: str = "doc_id",
    unit_id_col: Optional[str] = None,
    summary_col: Optional[str] = "accident_summary",
    use_contextual_prompt_with_summary: bool = False,
    contextual_prompt_template: str = DEFAULT_CONTEXTUAL_PROMPT_TEMPLATE,
    fixed_instruction_prefix: Optional[str] = None,
    hf_cache_folder: str = "./hf_cache",
    min_count_per_class_in_train: int = 4,
    batch_size_train: int = 32,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    supcon_temperature: float = 0.07,
    supcon_normalize_train_embeddings: bool = True,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    set_global_seed(seed)
    setup_hf_cache(hf_cache_folder)
    ensure_dir(output_root)

    device = get_device(device)
    fixed_instruction_prefix = _normalize_optional_text(fixed_instruction_prefix)
    use_fixed_instruction_prefix = fixed_instruction_prefix is not None

    print(f"[{stamp()}] Device = {device}")

    raw_df = pd.read_csv(input_csv)
    df = clean_supervised_dataframe(
        df=raw_df,
        text_col=text_col,
        label_col=label_col,
        group_col=group_col,
        unit_id_col=unit_id_col,
    )

    counts = df[label_col].value_counts()
    keep_labels = counts[counts >= min_count_per_class_in_train].index.tolist()
    dropped_labels = counts[counts < min_count_per_class_in_train].sort_index()

    df = df[df[label_col].isin(keep_labels)].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset vide après filtrage des labels rares.")
    if df[label_col].nunique() < 2:
        raise ValueError("Moins de 2 classes après filtrage.")

    model_input_text_col = "__model_input_text__"
    df = build_effective_model_input_text(
        df=df,
        text_col=text_col,
        use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
        summary_col=summary_col,
        contextual_prompt_template=contextual_prompt_template,
        fixed_instruction_prefix=fixed_instruction_prefix,
        output_col=model_input_text_col,
    )

    unique_labels = sorted(df[label_col].unique())
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    df["label"] = df[label_col].map(label2id).astype(int)

    with open(os.path.join(output_root, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_root, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    train_dataset = hf_dataset_from_df(df, text_col=model_input_text_col)

    model = load_model(base_model_name, get_device(device), hf_cache_folder)
    train_loss = SupConLoss(
        model=model,
        temperature=supcon_temperature,
        normalize_embeddings=supcon_normalize_train_embeddings,
    )

    use_bf16 = bool(
        str(get_device(device)).startswith("cuda")
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = bool(str(get_device(device)).startswith("cuda") and not use_bf16)

    steps_per_epoch = max(
        1,
        math.ceil(len(df) / max(1, batch_size_train * gradient_accumulation_steps))
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_root,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size_train,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        fp16=use_fp16,
        bf16=use_bf16,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 2),
        report_to=[],
        seed=seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    print(f"[{stamp()}] Start final SupCon training on full dataset")
    trainer.train()

    final_model_dir = os.path.join(output_root, "final_model_full_data")
    trainer.model.save_pretrained(final_model_dir)
    print(f"[{stamp()}] Final full-data SupCon model saved to: {final_model_dir}")

    training_info = {
        "base_model_name": base_model_name,
        "text_col": text_col,
        "summary_col": summary_col,
        "model_input_text_col": model_input_text_col,
        "use_contextual_prompt_with_summary": use_contextual_prompt_with_summary,
        "contextual_prompt_template": (
            contextual_prompt_template if use_contextual_prompt_with_summary else None
        ),
        "use_fixed_instruction_prefix": use_fixed_instruction_prefix,
        "fixed_instruction_prefix": fixed_instruction_prefix,
        "label_col": label_col,
        "group_col": group_col,
        "n_rows": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "batch_size_train": batch_size_train,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "supcon_temperature": float(supcon_temperature),
        "supcon_normalize_train_embeddings": bool(supcon_normalize_train_embeddings),
        "seed": seed,
        "dropped_labels_below_global_min_count": {
            str(k): int(v) for k, v in dropped_labels.items()
        },
        "global_label_counts_after_filtering": {
            str(k): int(v) for k, v in df[label_col].value_counts().sort_index().items()
        },
        "train_loss_name": "SupConLoss",
    }
    with open(os.path.join(output_root, "final_training_info.json"), "w", encoding="utf-8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)

    del trainer, model
    gc.collect()
    if str(get_device(device)).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "final_model_dir": final_model_dir,
        "n_rows": int(len(df)),
        "n_classes": int(df["label"].nunique()),
    }