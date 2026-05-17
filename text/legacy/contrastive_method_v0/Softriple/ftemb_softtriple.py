import os
import gc
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    StratifiedGroupKFold = None
    HAS_STRATIFIED_GROUP_KFOLD = False


# =========================================================
# Utils
# =========================================================
def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Optional[str]) -> None:
    """
    Crée un dossier si le chemin est non vide.
    Corrige le cas os.path.dirname("file.csv") == "".
    """
    if path is None:
        return
    path = str(path).strip()
    if path == "":
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_python(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(x) for x in obj]
    return obj


def json_dumps_pretty(obj: Any) -> str:
    return json.dumps(to_python(obj), ensure_ascii=False, indent=2)


def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def build_effective_model_input_text(
    sentence: str,
    summary: Optional[str] = None,
    use_contextual_prompt_with_summary: bool = False,
    contextual_prompt_template: Optional[str] = None,
    fixed_instruction_prefix: Optional[str] = None,
) -> str:
    """
    Construit le texte réellement donné au modèle.

    Modes possibles :
    1. phrase seule ;
    2. résumé + phrase via template ;
    3. préfixe d'instruction + texte.
    """
    sentence = "" if sentence is None else str(sentence).strip()
    summary = "" if summary is None else str(summary).strip()

    if use_contextual_prompt_with_summary:
        if contextual_prompt_template is None:
            raise ValueError(
                "contextual_prompt_template doit être fourni si "
                "use_contextual_prompt_with_summary=True."
            )
        text = contextual_prompt_template.format(
            accident_summary=summary,
            sentence=sentence,
        )
    else:
        text = sentence

    if fixed_instruction_prefix is not None and str(fixed_instruction_prefix).strip() != "":
        text = f"{fixed_instruction_prefix.strip()}\n\n{text}"

    return text.strip()


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return x / norms


def pairwise_cosine_distance_matrix(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    if normalize:
        x = l2_normalize_np(x)
    sim = np.clip(x @ x.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_separation_metrics(
    embeddings: np.ndarray,
    labels: List[Any],
    normalize_for_eval: bool = True,
    singleton_policy: str = "zero",
) -> Dict[str, Any]:
    """
    Métrique de séparation utilisée dans tes expériences.

    S = distance moyenne globale entre toutes les paires.
    W = distance moyenne intra-classe pondérée.
    B = S - W.
    delta_ratio = B / S.

    Interprétation : plus delta_ratio est élevé, plus les classes sont compactes
    relativement à la dispersion globale.
    """
    if len(embeddings) == 0:
        return {
            "n_samples": 0,
            "n_classes": 0,
            "global_mean_distance": None,
            "within_mean_distance": None,
            "between_proxy_distance": None,
            "delta_ratio": None,
        }

    labels = list(labels)
    n = len(labels)
    dist = pairwise_cosine_distance_matrix(embeddings, normalize=normalize_for_eval)

    iu = np.triu_indices(n, k=1)
    global_pairs = dist[iu]
    global_mean = float(global_pairs.mean()) if global_pairs.size > 0 else 0.0

    label_to_indices: Dict[Any, List[int]] = {}
    for i, y in enumerate(labels):
        label_to_indices.setdefault(y, []).append(i)

    within_values = []
    within_weights = []

    for _, idxs in label_to_indices.items():
        if len(idxs) < 2:
            if singleton_policy == "zero":
                within_values.append(0.0)
                within_weights.append(1.0)
            elif singleton_policy == "drop":
                pass
            else:
                raise ValueError("singleton_policy doit être 'zero' ou 'drop'.")
            continue

        sub = dist[np.ix_(idxs, idxs)]
        iu_sub = np.triu_indices(len(idxs), k=1)
        vals = sub[iu_sub]
        if vals.size > 0:
            within_values.append(float(vals.mean()))
            within_weights.append(float(vals.size))

    if len(within_values) == 0:
        within_mean = 0.0
    else:
        within_mean = float(np.average(within_values, weights=within_weights))

    between_proxy = global_mean - within_mean
    delta_ratio = None
    if global_mean > 1e-12:
        delta_ratio = float(between_proxy / global_mean)

    return {
        "n_samples": int(n),
        "n_classes": int(len(label_to_indices)),
        "global_mean_distance": float(global_mean),
        "within_mean_distance": float(within_mean),
        "between_proxy_distance": float(between_proxy),
        "delta_ratio": delta_ratio,
    }


def aggregate_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agrège les métriques de séparation sur les folds.

    fold_metrics: liste de dicts de forme :
    {
      "test_separation": {
         "pred_label": {...},
         "pred_subtype": {...}
      }
    }
    """
    agg: Dict[str, Any] = {"test_separation": {}}

    if len(fold_metrics) == 0:
        return agg

    label_names = set()
    for fm in fold_metrics:
        label_names.update(fm.get("test_separation", {}).keys())

    keys = [
        "global_mean_distance",
        "within_mean_distance",
        "between_proxy_distance",
        "delta_ratio",
        "n_samples",
        "n_classes",
    ]

    for label_name in sorted(label_names):
        agg["test_separation"][label_name] = {}
        for key in keys:
            vals = []
            for fm in fold_metrics:
                v = (
                    fm.get("test_separation", {})
                    .get(label_name, {})
                    .get(key, None)
                )
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(float(v))
            if len(vals) == 0:
                agg["test_separation"][label_name][key] = {"mean": None, "std": None}
            else:
                agg["test_separation"][label_name][key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                }

    return agg


def _build_class_id_to_label(label_to_id: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in label_to_id.items()}


def _reshape_softtriple_centers(
    centers: torch.Tensor,
    num_classes: int,
    centers_per_class: int,
) -> torch.Tensor:
    if centers.dim() == 3:
        return centers

    if centers.dim() == 2:
        expected = num_classes * centers_per_class
        if centers.shape[0] != expected:
            raise ValueError(
                f"Nombre de centres incompatible: got {centers.shape[0]}, "
                f"expected {expected} = {num_classes} * {centers_per_class}"
            )
        return centers.view(num_classes, centers_per_class, centers.shape[1])

    raise ValueError(
        f"Format inattendu pour les centres SoftTriple: shape={tuple(centers.shape)}"
    )


def compute_softtriple_assignment_metadata(
    embeddings: np.ndarray,
    centers: torch.Tensor,
    num_classes: int,
    centers_per_class: int,
    class_id_to_label: Optional[Dict[int, str]] = None,
    normalize_embeddings: bool = True,
    normalize_centers: bool = True,
    batch_size: int = 2048,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Pour chaque embedding, assigne :
    - la meilleure classe SoftTriple ;
    - le meilleur centre dans cette classe ;
    - le score de similarité avec ce centre.

    Cela permet ensuite d'analyser les sous-zones apprises par SoftTriple.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings doit être 2D [N, D], reçu shape={embeddings.shape}"
        )

    centers = _reshape_softtriple_centers(
        centers=centers,
        num_classes=num_classes,
        centers_per_class=centers_per_class,
    ).detach().to(device)

    if normalize_centers:
        centers = F.normalize(centers, p=2, dim=-1)

    emb_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
    if normalize_embeddings:
        emb_t = F.normalize(emb_t, p=2, dim=-1)

    n = emb_t.shape[0]

    pred_class_idx_all = []
    pred_center_idx_all = []
    pred_center_score_all = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z = emb_t[start:end]

        # [B, C, K]
        scores = torch.einsum("bd,ckd->bck", z, centers)

        # meilleur centre par classe -> [B, C]
        class_scores, center_idx_per_class = scores.max(dim=2)

        # meilleure classe -> [B]
        pred_class_idx = class_scores.argmax(dim=1)

        batch_idx = torch.arange(pred_class_idx.shape[0], device=device)
        pred_center_idx = center_idx_per_class[batch_idx, pred_class_idx]
        pred_center_score = scores[batch_idx, pred_class_idx, pred_center_idx]

        pred_class_idx_all.append(pred_class_idx.detach().cpu())
        pred_center_idx_all.append(pred_center_idx.detach().cpu())
        pred_center_score_all.append(pred_center_score.detach().cpu())

    pred_class_idx = torch.cat(pred_class_idx_all).numpy()
    pred_center_idx = torch.cat(pred_center_idx_all).numpy()
    pred_center_score = torch.cat(pred_center_score_all).numpy()

    pred_center_global = pred_class_idx * centers_per_class + pred_center_idx

    if class_id_to_label is None:
        pred_macro = [str(int(x)) for x in pred_class_idx]
    else:
        pred_macro = [class_id_to_label[int(x)] for x in pred_class_idx]

    return pd.DataFrame(
        {
            "softtriple_pred_class_idx": pred_class_idx.astype(int),
            "softtriple_pred_macro": pred_macro,
            "softtriple_pred_center": pred_center_idx.astype(int),
            "softtriple_pred_center_global": pred_center_global.astype(int),
            "softtriple_pred_center_score": pred_center_score.astype(float),
        }
    )


# =========================================================
# Data
# =========================================================
class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        unit_ids: List[Any],
    ):
        self.texts = texts
        self.labels = labels
        self.unit_ids = unit_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
            "unit_id": self.unit_ids[idx],
        }


@dataclass
class PreparedSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]


# =========================================================
# Model
# =========================================================
class HFTextEncoder(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        hf_cache_folder: Optional[str] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(
            base_model_name,
            cache_dir=hf_cache_folder,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=hf_cache_folder,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.encoder = AutoModel.from_pretrained(
            base_model_name,
            cache_dir=hf_cache_folder,
            trust_remote_code=True,
            config=config,
        )

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Impossible d'inférer hidden_size depuis le config du modèle.")

        self.embedding_dim = int(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            x = mean_pooling(outputs.last_hidden_state, attention_mask)
            return x

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        raise ValueError("Le modèle HF ne retourne ni last_hidden_state ni pooler_output.")


class SoftTripleLoss(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        centers_per_class: int = 5,
        gamma: float = 0.1,
        la: float = 10.0,
        delta: float = 0.01,
        tau: float = 0.0,
        normalize_embeddings: bool = True,
        normalize_centers: bool = True,
        center_max_similarity: float = 0.50,
    ):
        """
        SoftTriple Loss avec plusieurs centres par classe.

        Paramètres importants :
        - centers_per_class : nombre de centres par classe.
        - gamma : température du soft-assignment intra-classe.
        - la : facteur d'échelle des logits.
        - delta : marge appliquée à la vraie classe.
        - tau : poids de la régularisation des centres.
        - center_max_similarity : seuil de similarité cosinus au-dessus duquel
          deux centres d'une même classe sont considérés comme trop proches.

        Correction importante :
        La version initiale ajoutait tau * distance moyenne entre centres.
        Comme la loss est minimisée, cela rapprochait les centres.
        Ici, on pénalise plutôt les centres trop similaires :
            penalty = relu(cos_sim - center_max_similarity)^2
        Ainsi, tau > 0 évite le collapse des centres intra-classe sans forcer
        inutilement les centres à devenir opposés.
        """
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.centers_per_class = int(centers_per_class)
        self.gamma = float(gamma)
        self.la = float(la)
        self.delta = float(delta)
        self.tau = float(tau)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.normalize_centers = bool(normalize_centers)
        self.center_max_similarity = float(center_max_similarity)

        centers = torch.randn(num_classes, centers_per_class, embedding_dim) * 0.02
        centers = F.normalize(centers, p=2, dim=-1)
        self.centers = nn.Parameter(centers)

    def _get_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            return F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _get_centers(self) -> torch.Tensor:
        if self.normalize_centers:
            return F.normalize(self.centers, p=2, dim=-1)
        return self.centers

    def compute_relaxed_class_similarity(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne :
        - relaxed_sim: [B, C]
        - raw_sim: [B, C, K]

        Pour chaque classe, on combine les similarités aux K centres via un
        soft-assignment. Cela donne une similarité classe plus souple qu'un
        simple prototype unique.
        """
        z = self._get_embeddings(embeddings)
        centers = self._get_centers()

        raw_sim = torch.einsum("bd,ckd->bck", z, centers)

        if self.centers_per_class == 1:
            relaxed_sim = raw_sim.squeeze(-1)
        else:
            q = F.softmax(raw_sim / max(self.gamma, 1e-8), dim=2)
            relaxed_sim = (q * raw_sim).sum(dim=2)

        return relaxed_sim, raw_sim

    def regularization(self) -> torch.Tensor:
        """
        Régularisation de diversité intra-classe des centres.

        Objectif : éviter que les K centres d'une même classe apprennent tous
        le même prototype.

        On ne pénalise que les paires de centres trop proches, c'est-à-dire
        avec une similarité cosinus supérieure à center_max_similarity.

        Si tau == 0 ou centers_per_class == 1, aucune régularisation.
        """
        if self.tau <= 0.0 or self.centers_per_class <= 1:
            return torch.tensor(0.0, device=self.centers.device)

        centers = self._get_centers()
        penalties = []

        iu = torch.triu_indices(
            self.centers_per_class,
            self.centers_per_class,
            offset=1,
            device=centers.device,
        )

        for c in range(self.num_classes):
            wc = centers[c]
            sim = torch.clamp(wc @ wc.T, -1.0, 1.0)
            pair_sim = sim[iu[0], iu[1]]
            penalty = F.relu(pair_sim - self.center_max_similarity).pow(2)
            if penalty.numel() > 0:
                penalties.append(penalty.mean())

        if len(penalties) == 0:
            return torch.tensor(0.0, device=self.centers.device)

        reg = torch.stack(penalties).mean()
        return self.tau * reg

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        relaxed_sim, _ = self.compute_relaxed_class_similarity(embeddings)
        logits = self.la * relaxed_sim

        batch_idx = torch.arange(labels.shape[0], device=labels.device)

        # Marge sur la vraie classe.
        # On clone pour éviter les effets de bord in-place sur un tenseur utilisé par autograd.
        logits = logits.clone()
        logits[batch_idx, labels] = self.la * (relaxed_sim[batch_idx, labels] - self.delta)

        ce = F.cross_entropy(logits, labels)
        reg = self.regularization()
        loss = ce + reg

        stats = {
            "loss_total": float(loss.detach().cpu().item()),
            "loss_ce": float(ce.detach().cpu().item()),
            "loss_reg": float(reg.detach().cpu().item()),
        }
        return loss, stats


# =========================================================
# Training / encoding helpers
# =========================================================
def make_collate_fn(tokenizer, max_length: Optional[int] = None):
    if max_length is None:
        tokenizer_max_length = getattr(tokenizer, "model_max_length", 512)
        if tokenizer_max_length is None or tokenizer_max_length > 100000:
            tokenizer_max_length = 512
        max_length = min(int(tokenizer_max_length), 512)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        unit_ids = [x["unit_id"] for x in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "texts": texts,
            "unit_ids": unit_ids,
        }

    return collate


def encode_texts(
    model: HFTextEncoder,
    texts: List[str],
    batch_size_encode: int,
    device: str,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    model.eval()
    collate_fn = make_collate_fn(model.tokenizer)
    ds = TextDataset(texts=texts, labels=[0] * len(texts), unit_ids=list(range(len(texts))))
    dl = DataLoader(ds, batch_size=batch_size_encode, shuffle=False, collate_fn=collate_fn)

    out = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            z = model(input_ids=input_ids, attention_mask=attention_mask)
            if normalize_embeddings:
                z = F.normalize(z, p=2, dim=-1)
            out.append(z.detach().cpu().numpy())

    if len(out) == 0:
        return np.zeros((0, model.embedding_dim), dtype=np.float32)

    return np.concatenate(out, axis=0).astype(np.float32)


def train_one_fold(
    train_df: pd.DataFrame,
    label_to_id: Dict[str, int],
    base_model_name: str,
    text_col: str,
    label_col: str,
    summary_col: Optional[str],
    use_contextual_prompt_with_summary: bool,
    contextual_prompt_template: Optional[str],
    fixed_instruction_prefix: Optional[str],
    hf_cache_folder: Optional[str],
    batch_size_train: int,
    num_train_epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    softtriple_centers_per_class: int,
    softtriple_gamma: float,
    softtriple_lambda: float,
    softtriple_delta: float,
    softtriple_tau: float,
    softtriple_normalize_train_embeddings: bool,
    softtriple_normalize_centers: bool,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    set_global_seed(seed)

    train_df = train_df.copy().reset_index(drop=True)
    texts = []
    labels = []
    unit_ids = []

    for _, row in train_df.iterrows():
        summary_value = None
        if summary_col is not None and summary_col in row.index:
            summary_value = row[summary_col]

        text = build_effective_model_input_text(
            sentence=row[text_col],
            summary=summary_value,
            use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
            contextual_prompt_template=contextual_prompt_template,
            fixed_instruction_prefix=fixed_instruction_prefix,
        )
        texts.append(text)
        labels.append(label_to_id[str(row[label_col])])
        unit_ids.append(row.get("doc_id", None))

    model = HFTextEncoder(
        base_model_name=base_model_name,
        hf_cache_folder=hf_cache_folder,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)

    loss_module = SoftTripleLoss(
        embedding_dim=model.embedding_dim,
        num_classes=len(label_to_id),
        centers_per_class=softtriple_centers_per_class,
        gamma=softtriple_gamma,
        la=softtriple_lambda,
        delta=softtriple_delta,
        tau=softtriple_tau,
        normalize_embeddings=softtriple_normalize_train_embeddings,
        normalize_centers=softtriple_normalize_centers,
    ).to(device)

    ds = TextDataset(texts=texts, labels=labels, unit_ids=unit_ids)
    collate_fn = make_collate_fn(model.tokenizer)
    dl = DataLoader(
        ds,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    optimizer = AdamW(
        list(model.parameters()) + list(loss_module.parameters()),
        lr=learning_rate,
        weight_decay=0.0,
    )

    total_steps = max(
        1,
        math.ceil(len(dl) * num_train_epochs / max(gradient_accumulation_steps, 1)),
    )
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    loss_module.train()

    history = []
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    for epoch in range(num_train_epochs):
        epoch_losses = []
        epoch_ce_losses = []
        epoch_reg_losses = []

        for step, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            z = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, stats = loss_module(z, y)
            scaled_loss = loss / max(gradient_accumulation_steps, 1)
            scaled_loss.backward()

            if (step + 1) % max(gradient_accumulation_steps, 1) == 0 or (step + 1) == len(dl):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_ce_losses.append(stats["loss_ce"])
            epoch_reg_losses.append(stats["loss_reg"])

        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else None,
                "mean_train_ce_loss": float(np.mean(epoch_ce_losses)) if len(epoch_ce_losses) > 0 else None,
                "mean_train_reg_loss": float(np.mean(epoch_reg_losses)) if len(epoch_reg_losses) > 0 else None,
                "global_step": int(global_step),
            }
        )

    return {
        "model": model,
        "softtriple_loss": loss_module,
        "train_history": history,
    }


# =========================================================
# Split prep
# =========================================================
def prepare_dataframe(
    input_csv: str,
    text_col: str,
    label_col: str,
    group_col: str,
    unit_id_col: str,
    summary_col: Optional[str],
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    needed = [text_col, label_col, group_col]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Colonne requise absente du CSV: {c}")

    if unit_id_col not in df.columns:
        df[unit_id_col] = np.arange(len(df))

    if summary_col is not None and summary_col not in df.columns:
        df[summary_col] = ""

    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[label_col] = df[label_col].astype(str)
    df[group_col] = df[group_col].astype(str)
    df[unit_id_col] = df[unit_id_col]

    if summary_col is not None and summary_col in df.columns:
        df[summary_col] = df[summary_col].fillna("").astype(str)

    df = df[df[text_col].str.strip() != ""].reset_index(drop=True)
    return df


def make_group_stratified_splits(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_groups = df[group_col].nunique()
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} est supérieur au nombre de groupes uniques={n_groups}. "
            f"Réduis n_splits ou vérifie la colonne {group_col}."
        )

    if HAS_STRATIFIED_GROUP_KFOLD:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(sgkf.split(df, y=df[label_col], groups=df[group_col]))
        return splits

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(df, y=df[label_col], groups=df[group_col]))
    return splits


def build_label_mapping(train_labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted(set(train_labels))
    label_to_id = {y: i for i, y in enumerate(uniq)}
    id_to_label = {i: y for y, i in label_to_id.items()}
    return label_to_id, id_to_label


def filter_train_test_by_min_count(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    min_count_per_class_in_train: int,
) -> PreparedSplit:
    counts = train_df[label_col].value_counts()
    keep_labels = counts[counts >= min_count_per_class_in_train].index.tolist()

    train_df = train_df[train_df[label_col].isin(keep_labels)].copy()
    test_df = test_df[test_df[label_col].isin(keep_labels)].copy()

    label_to_id, id_to_label = build_label_mapping(train_df[label_col].astype(str).tolist())

    return PreparedSplit(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )


# =========================================================
# Evaluation / export
# =========================================================
def build_effective_texts_for_df(
    df: pd.DataFrame,
    text_col: str,
    summary_col: Optional[str],
    use_contextual_prompt_with_summary: bool,
    contextual_prompt_template: Optional[str],
    fixed_instruction_prefix: Optional[str],
) -> List[str]:
    texts = []
    for _, row in df.iterrows():
        summary_value = None
        if summary_col is not None and summary_col in row.index:
            summary_value = row[summary_col]

        texts.append(
            build_effective_model_input_text(
                sentence=row[text_col],
                summary=summary_value,
                use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
                contextual_prompt_template=contextual_prompt_template,
                fixed_instruction_prefix=fixed_instruction_prefix,
            )
        )
    return texts


def export_fold_embeddings_with_softtriple_metadata(
    export_rows: List[pd.DataFrame],
    df_split: pd.DataFrame,
    texts: List[str],
    embeddings: np.ndarray,
    split_name: str,
    fold_idx: int,
    eval_label_cols: List[str],
    unit_id_col: str,
    group_col: str,
    text_col: str,
    softtriple_loss: SoftTripleLoss,
    label_to_id: Dict[str, int],
    softtriple_centers_per_class: int,
    softtriple_normalize_train_embeddings: bool,
    softtriple_normalize_centers: bool,
    device: str,
) -> None:
    dim_cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
    export_df = pd.DataFrame(embeddings, columns=dim_cols)

    base_cols = {
        "fold": fold_idx,
        "split": split_name,
        unit_id_col: df_split[unit_id_col].tolist(),
        group_col: df_split[group_col].tolist(),
        text_col: df_split[text_col].tolist(),
        "effective_text": texts,
    }

    for col, vals in base_cols.items():
        export_df.insert(len(export_df.columns), col, vals)

    for lc in eval_label_cols:
        if lc in df_split.columns:
            export_df[lc] = df_split[lc].tolist()

    class_id_to_label = _build_class_id_to_label(label_to_id)
    center_meta_df = compute_softtriple_assignment_metadata(
        embeddings=embeddings,
        centers=softtriple_loss.centers,
        num_classes=len(label_to_id),
        centers_per_class=softtriple_centers_per_class,
        class_id_to_label=class_id_to_label,
        normalize_embeddings=softtriple_normalize_train_embeddings,
        normalize_centers=softtriple_normalize_centers,
        batch_size=2048,
        device=device,
    )

    export_df = pd.concat(
        [export_df.reset_index(drop=True), center_meta_df.reset_index(drop=True)],
        axis=1,
    )
    export_rows.append(export_df)


# =========================================================
# Public API
# =========================================================
def finetune_softtriple_embedder_research_kfold_simple(
    input_csv: str,
    output_root: str,
    base_model_name: str,
    text_col: str,
    label_col: str,
    group_col: str,
    unit_id_col: str,
    summary_col: Optional[str],
    use_contextual_prompt_with_summary: bool,
    contextual_prompt_template: Optional[str],
    fixed_instruction_prefix: Optional[str],
    eval_label_cols: List[str],
    selection_label_col: str,
    hf_cache_folder: Optional[str],
    n_splits: int,
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
    softtriple_centers_per_class: int,
    softtriple_gamma: float,
    softtriple_lambda: float,
    softtriple_delta: float,
    softtriple_tau: float,
    softtriple_normalize_train_embeddings: bool,
    softtriple_normalize_centers: bool,
    seed: int,
    device: str,
    export_full_embeddings_csv: Optional[str] = None,
) -> Dict[str, Any]:
    del batch_size_eval, max_eval_triplets

    ensure_dir(output_root)
    set_global_seed(seed)

    df = prepare_dataframe(
        input_csv=input_csv,
        text_col=text_col,
        label_col=label_col,
        group_col=group_col,
        unit_id_col=unit_id_col,
        summary_col=summary_col,
    )

    splits = make_group_stratified_splits(
        df=df,
        label_col=label_col,
        group_col=group_col,
        n_splits=n_splits,
        seed=seed,
    )

    fold_results = []
    export_rows: List[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n[{stamp()}] ===== Fold {fold_idx}/{len(splits)} =====")

        train_df_raw = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df_raw = df.iloc[test_idx].copy().reset_index(drop=True)

        prepared = filter_train_test_by_min_count(
            train_df=train_df_raw,
            test_df=test_df_raw,
            label_col=label_col,
            min_count_per_class_in_train=min_count_per_class_in_train,
        )

        train_df = prepared.train_df
        test_df = prepared.test_df
        label_to_id = prepared.label_to_id
        id_to_label = prepared.id_to_label

        if len(train_df) == 0 or len(test_df) == 0 or len(label_to_id) < 2:
            print(f"[WARN] Fold {fold_idx} ignoré (train/test vide ou <2 classes).")
            continue

        trained = train_one_fold(
            train_df=train_df,
            label_to_id=label_to_id,
            base_model_name=base_model_name,
            text_col=text_col,
            label_col=label_col,
            summary_col=summary_col,
            use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
            contextual_prompt_template=contextual_prompt_template,
            fixed_instruction_prefix=fixed_instruction_prefix,
            hf_cache_folder=hf_cache_folder,
            batch_size_train=batch_size_train,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            softtriple_centers_per_class=softtriple_centers_per_class,
            softtriple_gamma=softtriple_gamma,
            softtriple_lambda=softtriple_lambda,
            softtriple_delta=softtriple_delta,
            softtriple_tau=softtriple_tau,
            softtriple_normalize_train_embeddings=softtriple_normalize_train_embeddings,
            softtriple_normalize_centers=softtriple_normalize_centers,
            seed=seed + fold_idx,
            device=device,
        )

        model: HFTextEncoder = trained["model"]
        softtriple_loss: SoftTripleLoss = trained["softtriple_loss"]

        train_texts = build_effective_texts_for_df(
            df=train_df,
            text_col=text_col,
            summary_col=summary_col,
            use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
            contextual_prompt_template=contextual_prompt_template,
            fixed_instruction_prefix=fixed_instruction_prefix,
        )
        test_texts = build_effective_texts_for_df(
            df=test_df,
            text_col=text_col,
            summary_col=summary_col,
            use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
            contextual_prompt_template=contextual_prompt_template,
            fixed_instruction_prefix=fixed_instruction_prefix,
        )

        train_embeddings = encode_texts(
            model=model,
            texts=train_texts,
            batch_size_encode=batch_size_encode,
            device=device,
            normalize_embeddings=normalize_for_eval,
        )
        test_embeddings = encode_texts(
            model=model,
            texts=test_texts,
            batch_size_encode=batch_size_encode,
            device=device,
            normalize_embeddings=normalize_for_eval,
        )

        test_sep = {}
        for eval_col in eval_label_cols:
            if eval_col in test_df.columns:
                test_sep[eval_col] = compute_separation_metrics(
                    embeddings=test_embeddings,
                    labels=test_df[eval_col].astype(str).tolist(),
                    normalize_for_eval=normalize_for_eval,
                    singleton_policy=singleton_policy,
                )

        fold_result = {
            "fold_idx": fold_idx,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_classes_train": int(len(label_to_id)),
            "train_history": trained["train_history"],
            "test_separation": test_sep,
        }
        fold_results.append(fold_result)

        if export_full_embeddings_csv is not None:
            export_fold_embeddings_with_softtriple_metadata(
                export_rows=export_rows,
                df_split=train_df,
                texts=train_texts,
                embeddings=train_embeddings,
                split_name="train",
                fold_idx=fold_idx,
                eval_label_cols=eval_label_cols,
                unit_id_col=unit_id_col,
                group_col=group_col,
                text_col=text_col,
                softtriple_loss=softtriple_loss,
                label_to_id=label_to_id,
                softtriple_centers_per_class=softtriple_centers_per_class,
                softtriple_normalize_train_embeddings=softtriple_normalize_train_embeddings,
                softtriple_normalize_centers=softtriple_normalize_centers,
                device=device,
            )
            export_fold_embeddings_with_softtriple_metadata(
                export_rows=export_rows,
                df_split=test_df,
                texts=test_texts,
                embeddings=test_embeddings,
                split_name="test",
                fold_idx=fold_idx,
                eval_label_cols=eval_label_cols,
                unit_id_col=unit_id_col,
                group_col=group_col,
                text_col=text_col,
                softtriple_loss=softtriple_loss,
                label_to_id=label_to_id,
                softtriple_centers_per_class=softtriple_centers_per_class,
                softtriple_normalize_train_embeddings=softtriple_normalize_train_embeddings,
                softtriple_normalize_centers=softtriple_normalize_centers,
                device=device,
            )

        fold_dir = os.path.join(output_root, f"fold_{fold_idx}")
        ensure_dir(fold_dir)

        model.encoder.save_pretrained(os.path.join(fold_dir, "hf_model"))
        model.tokenizer.save_pretrained(os.path.join(fold_dir, "hf_model"))
        torch.save(
            {
                "softtriple_state_dict": softtriple_loss.state_dict(),
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "embedding_dim": model.embedding_dim,
                "num_classes": len(label_to_id),
                "centers_per_class": softtriple_centers_per_class,
                "softtriple_gamma": softtriple_gamma,
                "softtriple_lambda": softtriple_lambda,
                "softtriple_delta": softtriple_delta,
                "softtriple_tau": softtriple_tau,
                "softtriple_normalize_train_embeddings": softtriple_normalize_train_embeddings,
                "softtriple_normalize_centers": softtriple_normalize_centers,
                "center_max_similarity": softtriple_loss.center_max_similarity,
            },
            os.path.join(fold_dir, "softtriple_state.pt"),
        )
        with open(os.path.join(fold_dir, "fold_result.json"), "w", encoding="utf-8") as f:
            f.write(json_dumps_pretty(fold_result))

        del model
        del softtriple_loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(fold_results) == 0:
        raise ValueError("Aucun fold valide n'a été produit.")

    aggregate_metrics = aggregate_fold_metrics(fold_results)

    selection_score_mean_test_delta_ratio = (
        aggregate_metrics.get("test_separation", {})
        .get(selection_label_col, {})
        .get("delta_ratio", {})
        .get("mean", None)
    )

    if export_full_embeddings_csv is not None and len(export_rows) > 0:
        ensure_dir(os.path.dirname(export_full_embeddings_csv))
        export_df_all = pd.concat(export_rows, axis=0, ignore_index=True)
        export_df_all.to_csv(export_full_embeddings_csv, index=False)

    result = {
        "aggregate_metrics": aggregate_metrics,
        "aggregate_metrics_json": json_dumps_pretty(aggregate_metrics),
        "fold_results": fold_results,
        "fold_results_json": json_dumps_pretty(fold_results),
        "selection_score_mean_test_delta_ratio": selection_score_mean_test_delta_ratio,
    }

    with open(os.path.join(output_root, "aggregate_metrics.json"), "w", encoding="utf-8") as f:
        f.write(result["aggregate_metrics_json"])

    with open(os.path.join(output_root, "fold_results.json"), "w", encoding="utf-8") as f:
        f.write(result["fold_results_json"])

    return result


def train_final_softtriple_model_on_full_data(
    input_csv: str,
    output_root: str,
    base_model_name: str,
    text_col: str,
    label_col: str,
    group_col: str,
    unit_id_col: str,
    summary_col: Optional[str],
    use_contextual_prompt_with_summary: bool,
    contextual_prompt_template: Optional[str],
    fixed_instruction_prefix: Optional[str],
    hf_cache_folder: Optional[str],
    min_count_per_class_in_train: int,
    batch_size_train: int,
    num_train_epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    softtriple_centers_per_class: int,
    softtriple_gamma: float,
    softtriple_lambda: float,
    softtriple_delta: float,
    softtriple_tau: float,
    softtriple_normalize_train_embeddings: bool,
    softtriple_normalize_centers: bool,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    ensure_dir(output_root)
    set_global_seed(seed)

    df = prepare_dataframe(
        input_csv=input_csv,
        text_col=text_col,
        label_col=label_col,
        group_col=group_col,
        unit_id_col=unit_id_col,
        summary_col=summary_col,
    )

    counts = df[label_col].value_counts()
    keep_labels = counts[counts >= min_count_per_class_in_train].index.tolist()
    df = df[df[label_col].isin(keep_labels)].copy().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Aucune donnée restante après filtrage min_count_per_class_in_train.")

    label_to_id, id_to_label = build_label_mapping(df[label_col].astype(str).tolist())

    trained = train_one_fold(
        train_df=df,
        label_to_id=label_to_id,
        base_model_name=base_model_name,
        text_col=text_col,
        label_col=label_col,
        summary_col=summary_col,
        use_contextual_prompt_with_summary=use_contextual_prompt_with_summary,
        contextual_prompt_template=contextual_prompt_template,
        fixed_instruction_prefix=fixed_instruction_prefix,
        hf_cache_folder=hf_cache_folder,
        batch_size_train=batch_size_train,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        softtriple_centers_per_class=softtriple_centers_per_class,
        softtriple_gamma=softtriple_gamma,
        softtriple_lambda=softtriple_lambda,
        softtriple_delta=softtriple_delta,
        softtriple_tau=softtriple_tau,
        softtriple_normalize_train_embeddings=softtriple_normalize_train_embeddings,
        softtriple_normalize_centers=softtriple_normalize_centers,
        seed=seed,
        device=device,
    )

    model: HFTextEncoder = trained["model"]
    softtriple_loss: SoftTripleLoss = trained["softtriple_loss"]

    model.encoder.save_pretrained(os.path.join(output_root, "hf_model"))
    model.tokenizer.save_pretrained(os.path.join(output_root, "hf_model"))

    state = {
        "softtriple_state_dict": softtriple_loss.state_dict(),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "embedding_dim": model.embedding_dim,
        "num_classes": len(label_to_id),
        "centers_per_class": softtriple_centers_per_class,
        "softtriple_gamma": softtriple_gamma,
        "softtriple_lambda": softtriple_lambda,
        "softtriple_delta": softtriple_delta,
        "softtriple_tau": softtriple_tau,
        "softtriple_normalize_train_embeddings": softtriple_normalize_train_embeddings,
        "softtriple_normalize_centers": softtriple_normalize_centers,
        "center_max_similarity": softtriple_loss.center_max_similarity,
        "train_history": trained["train_history"],
        "n_train": int(len(df)),
        "base_model_name": base_model_name,
        "label_col": label_col,
        "text_col": text_col,
        "summary_col": summary_col,
        "use_contextual_prompt_with_summary": use_contextual_prompt_with_summary,
        "contextual_prompt_template": contextual_prompt_template if use_contextual_prompt_with_summary else None,
        "fixed_instruction_prefix": fixed_instruction_prefix,
        "seed": seed,
    }

    torch.save(state, os.path.join(output_root, "softtriple_state.pt"))

    result = {
        "output_root": output_root,
        "hf_model_dir": os.path.join(output_root, "hf_model"),
        "softtriple_state_path": os.path.join(output_root, "softtriple_state.pt"),
        "n_train": int(len(df)),
        "n_classes_train": int(len(label_to_id)),
        "train_history": trained["train_history"],
    }

    with open(os.path.join(output_root, "training_summary.json"), "w", encoding="utf-8") as f:
        f.write(json_dumps_pretty(result))

    del model
    del softtriple_loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result