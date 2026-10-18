#!/usr/bin/env python3
"""Five-role GOLD v3 transfer experiments; no target labels enter model selection.

The frozen document partitions come from the final RQ1 bundle.  A task trains
one model once and evaluates only the held-out test partition of each target.
All outputs live outside Git and are bound to dataset/split fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEXT = Path(__file__).resolve().parents[1]
ROOT = TEXT.parent
sys.path.insert(0, str(TEXT))

from scgm_text.data_metadata import GOLD_V3_SNAPSHOT_SHA256, LABEL2ID, load_filtered_metadata
from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings

CORPORA = ("btp", "metallurgie", "caou_chimie_plas", "bois_textille", "service_temporaire", "nicollin")
CSV_NAME = {name: ("caou" if name == "caou_chimie_plas" else name) for name in CORPORA}
ROLES = tuple(LABEL2ID)
METHODS = ("ce", "supcon", "softtriple")
SEEDS = (42, 1729, 20261018)
CS = (0.01, 0.1, 1.0, 10.0)
BACKBONE = "intfloat/multilingual-e5-large"
PREFIX = "query: "
MAX_LENGTH = 256
EPOCHS = 15
PATIENCE = 3


def checkpoint_policy() -> str:
    """Keep selected weights, or discard them after verified test predictions."""
    policy = os.getenv("ROLE_GOLD_CHECKPOINT_POLICY", "prune").strip().lower()
    if policy not in ("prune", "keep"):
        raise ValueError("ROLE_GOLD_CHECKPOINT_POLICY must be 'prune' or 'keep'")
    return policy


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bundle_path() -> Path:
    if os.getenv("ROLE_GOLD_BUNDLE"):
        return Path(os.environ["ROLE_GOLD_BUNDLE"]).expanduser().resolve()
    candidates = (
        ROOT / "annotation/rq1_final_v1/bundle",
        ROOT / "annotation/lodo_sat_transfer/rq1_final_v1/bundle",
        ROOT / "annotation/lodo_sat_transfer/rq1_final_v1",
    )
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError("RQ1 private bundle absent; set ROLE_GOLD_BUNDLE")


def output_root(experiment: str) -> Path:
    if experiment not in ("single_source", "lodo"):
        raise ValueError(experiment)
    root = Path(os.getenv("ROLE_GOLD_RUN_ROOT", os.path.join(
        os.getenv("SCRATCH", str(ROOT)), "safer_role_gold_v3")))
    return root.expanduser().resolve() / experiment


def source_partition(bundle: Path, corpus: str, name: str) -> list[str]:
    # Every final LODO fold records the same train/val IDs for a source.
    holder = next(target for target in CORPORA if target != corpus)
    ids = read_json(bundle / "folds" / f"target_{holder}" / "split_ids.json")
    return [str(value) for value in ids[name][corpus]]


def load_gold(bundle: Path) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    manifest = read_json(TEXT / "dataset/gold_v3_manifest.json")
    source = read_json(bundle / "manifest.json")
    bundle_complete = read_json(bundle / "complete.json")
    if bundle_complete.get("manifest_sha256") != digest(bundle / "manifest.json"):
        raise ValueError("RQ1 bundle manifest is incomplete or changed")
    if (manifest.get("snapshot_sha256") != GOLD_V3_SNAPSHOT_SHA256
            or source.get("gold_snapshot_sha256") != GOLD_V3_SNAPSHOT_SHA256
            or source.get("experiment") != "rq1_segmentation_final_v1"):
        raise ValueError("GOLD v3 or RQ1 split provenance changed")
    for corpus in CORPORA:
        relative = f"folds/target_{corpus}/split_ids.json"
        path = bundle / relative
        if not path.is_file() or digest(path) != source["files"].get(relative):
            raise ValueError(f"RQ1 split changed: {path}")
    frames: dict[str, pd.DataFrame] = {}
    split: dict[str, dict[str, list[str]]] = {}
    all_groups = set()
    total = 0
    for corpus in CORPORA:
        stem = CSV_NAME[corpus]
        file = TEXT / "dataset" / f"data_{stem}.csv"
        entry = manifest["corpora"][stem]
        if not file.is_file() or digest(file) != entry["sha256"]:
            raise ValueError(f"GOLD CSV missing or changed: {file}")
        df = pd.read_csv(file, dtype={"accident_id": str})
        if len(df) != entry["units"] or not df["pred_ok"].astype(str).str.lower().eq("true").all():
            raise ValueError(f"Incomplete GOLD CSV: {corpus}")
        if not df["pred_label"].isin(ROLES).all() or df["doc_id"].duplicated().any():
            raise ValueError(f"Invalid labels or duplicate fact ID: {corpus}")
        if not df.apply(lambda r: isinstance(r.sentence, str) and
                        r.accident_summary[int(r.start):int(r.end)] == r.sentence, axis=1).all():
            raise ValueError(f"Invalid GOLD offsets: {corpus}")
        df["label_id"] = df["pred_label"].map(LABEL2ID).astype(int)
        df["corpus"] = corpus
        frames[corpus] = df
        split[corpus] = {name: source_partition(bundle, corpus, name)
                         for name in ("train", "val")}
        for other in CORPORA:
            if other == corpus:
                continue
            other_ids = read_json(bundle / "folds" / f"target_{other}" / "split_ids.json")
            for name in ("train", "val"):
                if split[corpus][name] != [str(x) for x in other_ids[name][corpus]]:
                    raise ValueError(f"Source partition differs by RQ1 fold: {corpus}/{name}")
        own = read_json(bundle / "folds" / f"target_{corpus}" / "split_ids.json")
        split[corpus]["test"] = [str(x) for x in own["test"][corpus]]
        groups = set(df["accident_id"])
        by_part = [set(split[corpus][part]) for part in ("train", "val", "test")]
        partition_groups = set.union(*by_part)
        if (len(partition_groups) != sum(map(len, by_part))
                or len(partition_groups) != entry["documents"]
                or len(groups) != entry["documents_with_units"]
                or groups - partition_groups
                or len(partition_groups - groups) != entry["documents_without_units"]):
            # Narratives with no GOLD unit have no classification row; retain
            # their IDs in the frozen split but do not invent a training fact.
            raise ValueError(f"Narrative split/CSV mismatch: {corpus}")
        if any((corpus, group) in all_groups for group in groups):
            raise ValueError(f"Repeated accident across corpus: {corpus}")
        all_groups.update((corpus, group) for group in groups)
        for part in ("train", "val", "test"):
            subset = df[df["accident_id"].isin(split[corpus][part])]
            if subset.empty or set(subset["pred_label"]) != set(ROLES):
                raise ValueError(f"Empty or missing a role in {corpus}/{part}")
        total += len(df)
    if total != 147487:
        raise ValueError(f"Expected 147487 GOLD facts, got {total}")
    split_hash = digest(bundle / "manifest.json")
    return frames, split, {"gold_sha256": GOLD_V3_SNAPSHOT_SHA256,
                           "bundle_manifest_sha256": split_hash,
                           "csv_sha256": {c: manifest["corpora"][CSV_NAME[c]]["sha256"] for c in CORPORA}}


def selection(experiment: str, fold: str, frames: dict, split: dict):
    sources = (fold,) if experiment == "single_source" else tuple(c for c in CORPORA if c != fold)
    targets = tuple(c for c in CORPORA if c != fold) if experiment == "single_source" else (fold,)
    def part(c: str, p: str) -> pd.DataFrame:
        return frames[c][frames[c]["accident_id"].isin(split[c][p])].copy()
    train = pd.concat([part(c, "train") for c in sources], ignore_index=True)
    val = pd.concat([part(c, "val") for c in sources], ignore_index=True)
    tests = {c: part(c, "test") for c in targets}
    for c, df in tests.items():
        if set(df["accident_id"]) & set(train["accident_id"]):
            raise ValueError(f"Target leak: {c}")
    return train, val, tests


def task_spec(index: int) -> tuple[str, int, str]:
    if not 0 <= index < 54:
        raise ValueError("Training index must be 0..53")
    method = METHODS[index % 3]
    fold = CORPORA[(index // 3) % 6]
    seed = SEEDS[index // 18]
    return fold, seed, method


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, recall_score
    labels = list(range(5))
    recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    return {"balanced_accuracy": float(recall.mean()),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "recall_by_role": dict(zip(ROLES, map(float, recall))),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "n_units": int(len(y))}


def predict_lr(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(x).astype(int)
    raw = model.predict_proba(x)
    probs = np.zeros((len(x), 5), dtype=np.float32)
    probs[:, model.classes_.astype(int)] = raw
    return pred, probs


def select_lr(x_train, y_train, x_val, y_val, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    best = None
    for c in CS:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=1000, random_state=seed))
        model.fit(x_train, y_train)
        pred = model.predict(x_val).astype(int)
        value = scores(y_val, pred)["balanced_accuracy"]
        if best is None or value > best[0]:
            best = (value, c, model)
    assert best is not None
    return best


def save_predictions(root: Path, target: str, df: pd.DataFrame,
                     pred: np.ndarray, probs: np.ndarray, *, method: str, seed: int) -> dict:
    y = df["label_id"].to_numpy(dtype=int)
    if len(pred) != len(df) or probs.shape != (len(df), 5):
        raise ValueError("Prediction count mismatch")
    dest = root / f"predictions_{target}.csv"
    out = df[["corpus", "accident_id", "doc_id", "fact_id", "start", "end", "pred_label"]].copy()
    out["prediction"] = [ROLES[i] for i in pred]
    for idx, label in enumerate(ROLES):
        out[f"prob_{label}"] = probs[:, idx]
    tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, dest)
    result = scores(y, pred)
    result.update({"target": target,
                   "method": method, "seed": seed, "predictions_sha256": digest(dest)})
    write_json(root / f"metrics_{target}.json", result)
    return result


def check_space(out: Path, *, min_free_gib: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out)
    available = usage.free / 2**30
    if available < min_free_gib:
        raise OSError(f"Only {available:.1f} GiB free at {out}; need {min_free_gib:.1f} GiB")
    return {"free_gib": round(available, 2), "minimum_gib": min_free_gib}


def prepare(args) -> None:
    bundle = bundle_path()
    frames, split, signatures = load_gold(bundle)
    root = output_root(args.experiment)
    def remaining_jobs(path: Path) -> int:
        return sum(not (path / "trained" / fold / method /
                        f"seed_{seed}" / "complete.json").is_file()
                   for fold in CORPORA for method in METHODS for seed in SEEDS)
    other = output_root("lodo" if args.experiment == "single_source" else "single_source")
    remaining = remaining_jobs(root)
    if (other / "campaign.json").exists() and not (other / "results/manifest.json").exists():
        remaining += remaining_jobs(other)
    policy = checkpoint_policy()
    # With pruning, only the two simultaneously running jobs need room for
    # their selected E5 weights. Keeping all 54 models requires much more.
    minimum = (max(16.0, args.min_free_gib) if policy == "prune" else
               max(12.0, (args.min_free_gib or 170.0) * remaining / 54))
    disk = check_space(root, min_free_gib=minimum)
    embedding_dir = TEXT / "embeddings/gold_v3/e5"
    missing_embeddings = sum(not (embedding_dir / f"{CSV_NAME[corpus]}.csv").is_file()
                             for corpus in CORPORA)
    if missing_embeddings and shutil.disk_usage(TEXT).free < 6 * 2**30:
        raise OSError("Less than 6 GiB free for missing GOLD E5 embedding exports")
    record = {"experiment": args.experiment, "corpora": list(CORPORA),
              "roles": list(ROLES), "seeds": list(SEEDS), "methods": list(METHODS),
              "backbone": BACKBONE, "input_prefix": PREFIX, "max_length": MAX_LENGTH,
              "bundle": str(bundle), "signatures": signatures,
              "splits": {c: {p: {"n_narratives": len(ids), "n_units": int(frames[c]["accident_id"].isin(ids).sum())}
                             for p, ids in split[c].items()} for c in CORPORA}}
    existing = root / "campaign.json"
    if existing.is_file():
        if read_json(existing) != record:
            raise ValueError(f"Existing campaign differs from GOLD/splits/configuration: {root}")
    else:
        write_json(existing, record)
    print(json.dumps({"campaign": str(root), "disk": disk,
                      "checkpoint_policy": policy,
                      "remaining_training_tasks_on_shared_filesystem": remaining,
                      "missing_e5_exports": missing_embeddings,
                      "units": sum(map(len, frames.values()))}), flush=True)


def verified_campaign(experiment: str):
    bundle = bundle_path()
    root = output_root(experiment)
    frames, split, signatures = load_gold(bundle)
    record = read_json(root / "campaign.json")
    if record["signatures"] != signatures or record["experiment"] != experiment:
        raise ValueError("Campaign does not match current GOLD/splits")
    return root, frames, split


def complete_and_verified(dest: Path, targets, campaign: Path, *, checkpoint: bool) -> bool:
    marker = dest / "complete.json"
    if not marker.is_file():
        return False
    try:
        record = read_json(marker)
        if set(record["targets"]) != set(targets):
            return False
        if record.get("campaign_sha256") != digest(campaign):
            return False
        if checkpoint:
            if not record.get("checkpoint_files_sha256"):
                return False
            model_dir = dest / "best_model"
            if model_dir.exists() or record.get("checkpoint_retention") != "predictions_only":
                for name, expected in record["checkpoint_files_sha256"].items():
                    artifact = model_dir / name
                    if not artifact.is_file() or digest(artifact) != expected:
                        return False
            for target in targets:
                metrics = read_json(dest / f"metrics_{target}.json")
                if metrics["predictions_sha256"] != record["targets"][target]["predictions_sha256"]:
                    return False
        else:
            classifier = dest / "classifier.joblib"
            if not classifier.is_file() or digest(classifier) != record["classifier_sha256"]:
                return False
        return all((dest / f"predictions_{target}.csv").is_file()
                   and digest(dest / f"predictions_{target}.csv") ==
                   record["targets"][target]["predictions_sha256"] for target in targets)
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return False


def prune_verified_checkpoint(dest: Path, targets, campaign: Path) -> None:
    """Free disk only after the selected model and every prediction are verified."""
    if not complete_and_verified(dest, targets, campaign, checkpoint=True):
        raise ValueError(f"Refusing to delete an unverified checkpoint: {dest}")
    model_dir = dest / "best_model"
    if not model_dir.exists():
        return
    if model_dir.resolve().parent != dest.resolve() or model_dir.is_symlink():
        raise ValueError(f"Unsafe checkpoint cleanup path: {model_dir}")
    marker = dest / "complete.json"
    record = read_json(marker)
    record["checkpoint_retention"] = "predictions_only"
    write_json(marker, record)
    shutil.rmtree(model_dir)


def frozen(args) -> None:
    root, frames, split = verified_campaign(args.experiment)
    fold = CORPORA[args.fold_index]
    dest = root / "frozen" / fold
    expected_targets = tuple(c for c in CORPORA if c != fold) if args.experiment == "single_source" else (fold,)
    if complete_and_verified(dest, expected_targets, root / "campaign.json", checkpoint=False):
        print(f"[skip] frozen/{fold}", flush=True)
        return
    train, val, tests = selection(args.experiment, fold, frames, split)
    by_corpus = {}
    for corpus in (fold,) if args.experiment == "single_source" else CORPORA:
        emb = TEXT / "embeddings/gold_v3/e5" / f"{CSV_NAME[corpus]}.csv"
        meta = load_filtered_metadata(str(TEXT / "dataset" / f"data_{CSV_NAME[corpus]}.csv"), text_col="sentence")
        merged, dims = merge_metadata_with_embeddings(meta, str(emb), strict=True, corpus_id=corpus)
        by_corpus[corpus] = merged.set_index("doc_id")[dims]
    # Tests from domains not in the training set also need verified embeddings.
    for corpus in tests:
        if corpus not in by_corpus:
            emb = TEXT / "embeddings/gold_v3/e5" / f"{CSV_NAME[corpus]}.csv"
            meta = load_filtered_metadata(str(TEXT / "dataset" / f"data_{CSV_NAME[corpus]}.csv"), text_col="sentence")
            merged, dims = merge_metadata_with_embeddings(meta, str(emb), strict=True, corpus_id=corpus)
            by_corpus[corpus] = merged.set_index("doc_id")[dims]
    def vectors(df):
        chunks = [by_corpus[c].loc[df.loc[df.corpus.eq(c), "doc_id"].tolist()].to_numpy(dtype=np.float32)
                  for c in df.corpus.unique()]
        # Preserve interleaved row order across domains.
        result = np.empty((len(df), chunks[0].shape[1]), dtype=np.float32)
        for c, chunk in zip(df.corpus.unique(), chunks):
            result[np.flatnonzero(df.corpus.eq(c).to_numpy())] = chunk
        return result
    score, c, model = select_lr(vectors(train), train.label_id.to_numpy(),
                                vectors(val), val.label_id.to_numpy(), SEEDS[0])
    dest.mkdir(parents=True, exist_ok=True)
    from joblib import dump
    dump(model, dest / "classifier.joblib")
    result = {}
    for target, test in tests.items():
        pred, probs = predict_lr(model, vectors(test))
        result[target] = save_predictions(dest, target, test, pred, probs, method="frozen", seed=SEEDS[0])
    write_json(dest / "complete.json", {"validation_ba": score, "selected_C": c,
                                        "targets": result, "campaign_sha256": digest(root / "campaign.json"),
                                        "classifier_sha256": digest(dest / "classifier.joblib")})
    print(f"[done] frozen/{fold} validation_BA={score:.4f} C={c}", flush=True)


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def text_loader(df: pd.DataFrame, tokenizer, *, batch_size: int, shuffle: bool = False,
                sampler=None):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Facts(Dataset):
        def __init__(self, frame):
            self.text = frame["sentence"].tolist()
            self.labels = frame["label_id"].astype(int).tolist()

        def __len__(self):
            return len(self.text)

        def __getitem__(self, index):
            return self.text[index], self.labels[index]

    def collate(items):
        # Apply E5's prefix once.  The legacy TextLabelDataset plus its collator
        # can apply it twice, so this experiment intentionally owns this path.
        tokens = tokenizer([PREFIX + text for text, _ in items], padding=True,
                           truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
                "labels": torch.tensor([label for _, label in items], dtype=torch.long)}

    if sampler is not None:
        return DataLoader(Facts(df), batch_sampler=sampler, collate_fn=collate,
                          num_workers=2, pin_memory=True)
    return DataLoader(Facts(df), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=2, pin_memory=True)


def ce_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    model.eval()
    parts = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model.forward_logits(batch["input_ids"].to(device),
                                              batch["attention_mask"].to(device))
            parts.append(torch.softmax(logits.float(), -1).cpu().numpy())
    probs = np.vstack(parts)
    return probs.argmax(axis=1), probs


def metric_embeddings(model, df: pd.DataFrame, tokenizer, device, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    model.eval()
    chunks = []
    loader = text_loader(df, tokenizer, batch_size=batch_size)
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = model({"input_ids": batch["input_ids"].to(device),
                           "attention_mask": batch["attention_mask"].to(device)})
            chunks.append(F.normalize(z.float(), p=2, dim=1).cpu().numpy())
    return np.vstack(chunks)


def train_model(method: str, seed: int, train: pd.DataFrame, val: pd.DataFrame,
                tests: dict[str, pd.DataFrame], dest: Path, *, smoke: bool = False) -> None:
    import torch
    from transformers import AutoTokenizer
    from tqdm import tqdm

    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig
    from contrastive_methods.hf_training_common import load_contrastive_checkpoint, save_contrastive_checkpoint
    from contrastive_methods.losses.softtriple import SoftTripleLoss
    from contrastive_methods.losses.supcon_hobbit import SupConEmbeddingLoss
    from contrastive_methods.samplers.pk_batch_sampler import build_pk_batch_sampler, resolve_balanced_pk_params
    from supervised_macro_ft.checkpoint_io import load_checkpoint, save_checkpoint
    from supervised_macro_ft.model import SupervisedMacroModel

    if not smoke and complete_and_verified(dest, tests, dest.parents[3] / "campaign.json", checkpoint=True):
        if checkpoint_policy() == "prune":
            prune_verified_checkpoint(dest, tests, dest.parents[3] / "campaign.json")
        if checkpoint_policy() == "prune" or (dest / "best_model").is_dir():
            print(f"[skip] {dest}", flush=True)
            return
    if not torch.cuda.is_available():
        raise RuntimeError("Full E5 fine-tuning requires a GPU")
    seed_all(seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = dest / "best_model"
    if not smoke and checkpoint_policy() == "prune":
        check_space(dest, min_free_gib=8.0)
    epochs = 1 if smoke else EPOCHS
    if method == "ce":
        model = SupervisedMacroModel(backbone_name=BACKBONE, num_classes=5,
                                     pooling="mean", backbone_trainable=True,
                                     train_last_n_layers=None, gradient_checkpointing=True,
                                     projection=None, dropout=0.1).to(device)
        from supervised_macro_ft.train_loop import build_class_weights, build_optimizer
        optimizer = build_optimizer(model, {"lr_backbone": 2e-5, "lr_head": 1e-3,
                                            "weight_decay": 0.01})
        weights = build_class_weights(train.label_id.tolist(), 5, "balanced").to(device)
        loss_module = torch.nn.CrossEntropyLoss(weight=weights)
        metric_cfg = None
        sampler = None
    else:
        metric_cfg = ContrastiveConfig(method_name=method, dataset_path=Path("unused"),
                                      backbone_name=BACKBONE, input_prefix=PREFIX,
                                      backbone_trainable=True, train_last_n_layers=None,
                                      cache_backbone_embeddings=False, max_seq_length=MAX_LENGTH,
                                      use_projector=True, projection="linear", hiddim=128,
                                      use_amp=True, encode_batch_size=32, seed=seed)
        model = ContrastiveEncoder(EncoderConfig(
            backbone_name=BACKBONE, max_seq_length=MAX_LENGTH, backbone_trainable=True,
            train_last_n_layers=None, cache_backbone_embeddings=False,
            use_projector=True, projection="linear", hiddim=128)).to(device)
        if method == "supcon":
            loss_module = SupConEmbeddingLoss(temperature=0.07, normalize_embeddings=True)
            params = resolve_balanced_pk_params(train.label_id.tolist(), 20, seed=seed)
            sampler = build_pk_batch_sampler(train.label_id.tolist(), params)
        else:
            loss_module = SoftTripleLoss(embedding_dim=128, num_classes=5,
                                         centers_per_class=5, gamma=0.1, la=10.0,
                                         delta=0.01, tau=0.2, normalize_embeddings=True,
                                         normalize_centers=True, distance_metric="cosine",
                                         center_regularization_type="merge_l21").to(device)
            sampler = None
        groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad],
                   "lr": 2e-5}]
        if model.projector is not None:
            groups.append({"params": list(model.projector.parameters()), "lr": 1e-3})
        loss_parameters = list(loss_module.parameters())
        if loss_parameters:
            groups.append({"params": loss_parameters, "lr": 1e-3})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    if not any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("E5 backbone is unexpectedly frozen")
    transformer_layers = model.backbone._get_transformer_layers()
    if not transformer_layers:
        raise RuntimeError("Cannot identify E5 transformer layers for adaptation audit")
    witness = next(p for p in transformer_layers[-1].parameters() if p.requires_grad)
    initial_witness = witness.detach().float().cpu().clone()
    train_loader = text_loader(train, tokenizer, batch_size=20 if method == "supcon" else 16,
                               shuffle=sampler is None, sampler=sampler)
    val_loader = text_loader(val, tokenizer, batch_size=32)
    best_ba, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        if method == "softtriple":
            loss_module.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total, count = 0.0, 0
        progress = tqdm(train_loader, desc=f"{method} seed={seed} epoch={epoch + 1}/{epochs}",
                        file=sys.stdout, mininterval=30.0, dynamic_ncols=False)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                if method == "ce":
                    loss = loss_module(model.forward_logits(x, mask), y)
                else:
                    z = model({"input_ids": x, "attention_mask": mask})
                    out = loss_module(z, y)
                    loss = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite {method} loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
            if count % 25 == 0:
                progress.set_postfix(loss=f"{total / count:.4f}")
        if method == "ce":
            pred, _ = ce_predictions(model, val_loader, device)
            val_score = scores(val.label_id.to_numpy(), pred)["balanced_accuracy"]
            selected_c = None
            classifier = None
        else:
            x_train = metric_embeddings(model, train, tokenizer, device, 32)
            x_val = metric_embeddings(model, val, tokenizer, device, 32)
            val_score, selected_c, classifier = select_lr(
                x_train, train.label_id.to_numpy(), x_val, val.label_id.to_numpy(), seed)
        row = {"epoch": epoch + 1, "train_loss": total / max(count, 1),
               "val_balanced_accuracy": val_score, "selected_C": selected_c}
        history.append(row)
        print(json.dumps({"method": method, "seed": seed, **row}), flush=True)
        if val_score > best_ba + 1e-12:
            best_ba, best_epoch, stale = val_score, epoch + 1, 0
            if not smoke and checkpoint_policy() == "prune":
                check_space(dest, min_free_gib=8.0)
            if method == "ce":
                save_checkpoint(model, checkpoint, config={"backbone_name": BACKBONE,
                    "num_classes": 5, "pooling": "mean", "projection": None,
                    "input_prefix": PREFIX, "max_seq_length": MAX_LENGTH})
            else:
                extra = {"softtriple_state.pt": loss_module.state_dict()} if method == "softtriple" else None
                save_contrastive_checkpoint(model, checkpoint, extra_state=extra)
                from joblib import dump
                dump(classifier, checkpoint / "classifier.joblib")
                write_json(checkpoint / "selection.json", {"selected_C": selected_c,
                                                              "validation_ba": best_ba})
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError("No selected checkpoint")
    if torch.equal(initial_witness, witness.detach().float().cpu()):
        raise RuntimeError("Full fine-tuning did not update the E5 backbone")
    write_json(dest / "history.json", history)
    del model, optimizer, loss_module
    torch.cuda.empty_cache()
    result = {}
    if method == "ce":
        model = load_checkpoint(checkpoint, device="cuda")
        for target, test in tests.items():
            loader = text_loader(test, tokenizer, batch_size=32)
            pred, probs = ce_predictions(model, loader, device)
            result[target] = save_predictions(dest, target, test, pred, probs, method=method, seed=seed)
    else:
        from joblib import load
        model = load_contrastive_checkpoint(metric_cfg, checkpoint, "cuda")
        classifier = load(checkpoint / "classifier.joblib")
        for target, test in tests.items():
            x = metric_embeddings(model, test, tokenizer, device, 32)
            pred, probs = predict_lr(classifier, x)
            result[target] = save_predictions(dest, target, test, pred, probs, method=method, seed=seed)
    checkpoint_names = (["hf_model.bin", "classifier_head.pt"] if method == "ce"
                        else ["contrastive_encoder.pt", "classifier.joblib"]
                        + (["softtriple_state.pt"] if method == "softtriple" else []))
    write_json(dest / "complete.json", {"method": method, "seed": seed,
        "best_epoch": best_epoch, "validation_ba": best_ba,
        "selected_C": read_json(checkpoint / "selection.json")["selected_C"] if method != "ce" else None,
        "checkpoint": str(checkpoint),
        "checkpoint_files_sha256": {name: digest(checkpoint / name) for name in checkpoint_names},
        "targets": result,
        "campaign_sha256": digest(dest.parents[3] / "campaign.json") if not smoke else None})
    if not smoke and checkpoint_policy() == "prune":
        prune_verified_checkpoint(dest, tests, dest.parents[3] / "campaign.json")
    print(f"[done] {dest} validation_BA={best_ba:.4f}", flush=True)


def train(args) -> None:
    root, frames, split = verified_campaign(args.experiment)
    fold, seed, method = task_spec(args.task_index)
    train_df, val_df, tests = selection(args.experiment, fold, frames, split)
    dest = root / "trained" / fold / method / f"seed_{seed}"
    train_model(method, seed, train_df, val_df, tests, dest)


def smoke(args) -> None:
    root, frames, split = verified_campaign(args.experiment)
    train_df, val_df, tests = selection(args.experiment, CORPORA[0], frames, split)
    def tiny(df, n):
        return pd.concat([df[df.pred_label.eq(role)].head(n) for role in ROLES], ignore_index=True)
    train_df, val_df = tiny(train_df, 5), tiny(val_df, 2)
    tests = {next(iter(tests)): tiny(next(iter(tests.values())), 2)}
    for method in METHODS:
        train_model(method, SEEDS[0], train_df, val_df, tests,
                    root / "_smoke" / method, smoke=True)
        import torch
        torch.cuda.empty_cache()
    smoke_dir = (root / "_smoke").resolve()
    if not smoke_dir.is_relative_to(root.resolve()):
        raise ValueError("Unsafe smoke cleanup path")
    shutil.rmtree(smoke_dir)
    write_json(root / "smoke_complete.json", {"methods": list(METHODS),
                                               "campaign_sha256": digest(root / "campaign.json")})


def summary(args) -> None:
    root, frames, split = verified_campaign(args.experiment)
    rows, role_rows, confusion_rows = [], [], []
    import pandas as pd
    for fold in CORPORA:
        expected = tuple(c for c in CORPORA if c != fold) if args.experiment == "single_source" else (fold,)
        train_df, val_df, _ = selection(args.experiment, fold, frames, split)
        candidates = [("frozen", SEEDS[0], root / "frozen" / fold)]
        candidates += [(method, seed, root / "trained" / fold / method / f"seed_{seed}")
                       for method in METHODS for seed in SEEDS]
        for method, seed, path in candidates:
            complete = read_json(path / "complete.json")
            if not complete_and_verified(path, expected, root / "campaign.json",
                                         checkpoint=method != "frozen"):
                raise ValueError(f"Incomplete or changed model/predictions: {path}")
            if set(complete["targets"]) != set(expected):
                raise ValueError(f"Missing target predictions: {path}")
            for target in expected:
                metrics = read_json(path / f"metrics_{target}.json")
                predictions = path / f"predictions_{target}.csv"
                if digest(predictions) != metrics["predictions_sha256"]:
                    raise ValueError(f"Changed predictions: {predictions}")
                frame = pd.read_csv(predictions)
                test = frames[target][frames[target]["accident_id"].isin(split[target]["test"])]
                if (len(frame) != len(test) or
                    frame["doc_id"].tolist() != test["doc_id"].tolist() or
                    frame["pred_label"].tolist() != test["pred_label"].tolist()):
                    raise ValueError(f"Test units differ: {path}/{target}")
                rows.append({"experiment": args.experiment, "source": fold if args.experiment == "single_source" else "five_sources",
                             "target": target, "method": method, "seed": seed,
                             "balanced_accuracy": metrics["balanced_accuracy"],
                             "macro_f1": metrics["macro_f1"], "n_units": metrics["n_units"],
                             "n_train_units": len(train_df), "n_val_units": len(val_df)})
                for role, recall in metrics["recall_by_role"].items():
                    role_rows.append({"source": fold, "target": target, "method": method,
                                      "seed": seed, "role": role, "recall": recall})
                for i, actual in enumerate(ROLES):
                    for j, predicted in enumerate(ROLES):
                        confusion_rows.append({"source": fold, "target": target,
                            "method": method, "seed": seed, "gold_role": actual,
                            "predicted_role": predicted,
                            "count": metrics["confusion_matrix"][i][j]})
    result = root / "results"
    result.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(result / "scores.csv", index=False)
    pd.DataFrame(role_rows).to_csv(result / "recall_by_role.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(result / "confusion_matrices.csv", index=False)
    grouped = frame.groupby(["source", "target", "method"], as_index=False).agg(
        ba_mean=("balanced_accuracy", "mean"), ba_sd=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        n_seeds=("seed", "nunique"))
    grouped[["ba_sd", "macro_f1_sd"]] = grouped[["ba_sd", "macro_f1_sd"]].fillna(0)
    grouped.to_csv(result / "scores_by_transfer.csv", index=False)
    if args.experiment == "single_source":
        for method in ("frozen", *METHODS):
            matrix = grouped[grouped.method.eq(method)].pivot(index="source", columns="target", values="ba_mean")
            matrix = matrix.reindex(index=CORPORA, columns=CORPORA)
            matrix.to_csv(result / f"matrix_{method}_ba.csv", na_rep="N/A")
        per_source = grouped.groupby(["source", "method"], as_index=False).agg(
            macro_target_ba=("ba_mean", "mean"), macro_target_f1=("macro_f1_mean", "mean"))
        per_source.to_csv(result / "macro_by_source.csv", index=False)
    else:
        grouped.groupby("method", as_index=False).agg(
            macro_target_ba=("ba_mean", "mean"),
            macro_target_f1=("macro_f1_mean", "mean")).to_csv(result / "macro_lodo.csv", index=False)
    write_json(result / "manifest.json", {"experiment": args.experiment,
        "gold_sha256": GOLD_V3_SNAPSHOT_SHA256,
        "campaign_sha256": digest(root / "campaign.json"), "rows": len(frame),
        "methods": ["frozen", *METHODS], "seeds": list(SEEDS),
        "note": "Single-source and LODO training volumes differ; results use GOLD units, not predicted segments."})
    print(f"[done] {result} ({len(frame)} transfer scores)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "frozen", "train", "smoke", "summary"))
    parser.add_argument("--experiment", choices=("single_source", "lodo"), required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=0)
    args = parser.parse_args()
    {"prepare": prepare, "frozen": frozen, "train": train,
     "smoke": smoke, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
