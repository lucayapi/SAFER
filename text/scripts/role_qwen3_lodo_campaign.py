#!/usr/bin/env python3
"""Five-role, annotated-unit LODO campaign on narrative_only_lodo_v1.

The target domain is never used for model or epoch selection.  Each source
domain contributes one balanced-accuracy value to the selection mean.
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

import decision_gold_models as old_qwen
import role_gold_campaign as old_role
from scgm_text.data_metadata import LABEL2ID

CORPORA = old_role.CORPORA
CSV_NAME = old_role.CSV_NAME
ROLES = tuple(LABEL2ID)
SEEDS = old_role.SEEDS
METHODS = ("ce", "supcon", "softtriple")
CS = old_role.CS
EPOCHS = 3
BACKBONE = old_qwen.BACKBONE
REVISION = old_qwen.BACKBONE_REVISION
MAX_LENGTH = old_qwen.MAX_LENGTH
TFIDF_PARAMS = {"analyzer": "word", "token_pattern": r"(?u)\b\w\w+\b",
                "ngram_range": (1, 2), "min_df": 2, "max_df": 1.0,
                "max_features": 50000, "sublinear_tf": True, "norm": "l2",
                "lowercase": True, "strip_accents": None}


def bundle() -> Path:
    return Path(os.environ.get("ROLE_QWEN3_BUNDLE", ROOT / "annotation/lodo_sat_transfer/narrative_only_lodo_v1")).resolve()


def root() -> Path:
    return Path(os.environ.get("ROLE_QWEN3_RUN_ROOT", Path(os.environ.get("SCRATCH", ROOT)) /
                               "safer_role_qwen3_narrative_lodo_3epoch_v1")).resolve()


def digest(path: Path) -> str:
    return old_role.digest(path)


def read(path: Path) -> dict:
    return old_role.read_json(path)


def write(path: Path, value: object) -> None:
    old_role.write_json(path, value)


def load_data(*, check_qwen: bool) -> tuple[dict, dict, dict]:
    """Verify dataset, split, labels, document isolation, and export identity."""
    b = bundle()
    manifest_path = b / "manifest.json"
    m = read(manifest_path)
    complete = read(b / "complete.json")
    if (m.get("version") != "sat_narrative_only_lodo_v1" or
            complete.get("manifest_sha256") != digest(manifest_path)):
        raise ValueError("Wrong or incomplete narrative-only LODO bundle")
    # The bundle contains the manifest used to construct these exact folds.
    # A separately copied manifest under text/dataset may be stale, so verify
    # the *actual six CSV files* against the bundled authoritative hashes.
    source_manifest = b / "source_dataset_manifest.json"
    source_hash = digest(source_manifest)
    if (source_hash != m.get("narrative_only_manifest_sha256") or
            source_hash != m["files"].get("source_dataset_manifest.json")):
        raise ValueError("Bundled clean-dataset manifest hash differs from LODO manifest: "
                         f"actual={source_hash}, expected={m.get('narrative_only_manifest_sha256')}")
    clean = read(source_manifest)
    if clean.get("total_units") != 147487 or clean.get("total_documents") != 17927:
        raise ValueError("Bundled clean-dataset counts differ from 147487 facts / 17927 narratives")
    clean_path = TEXT / "dataset/narrative_only_v1/narrative_only_manifest.json"
    if clean_path.is_file() and digest(clean_path) != source_hash:
        print(f"[preflight] Local dataset manifest differs from the frozen LODO copy: {clean_path}. "
              "The six CSV hashes will be checked against the LODO copy.", flush=True)
    frames, splits, signatures = {}, {}, {"bundle_manifest_sha256": digest(manifest_path),
        "clean_manifest_sha256": source_hash, "clean_csv_sha256": {},
        "split_sha256": {}, "qwen_sha256": {}, "original_csv_sha256": {}}
    for corpus in CORPORA:
        stem = CSV_NAME[corpus]
        entry = clean["corpora"][stem]
        csv = TEXT / "dataset/narrative_only_v1" / entry["file"]
        csv_hash = digest(csv)
        if csv_hash != entry["csv_sha256"]:
            raise ValueError(f"Cleaned CSV differs from frozen LODO source: {csv}; "
                             f"actual={csv_hash}, expected={entry['csv_sha256']}")
        df = pd.read_csv(csv, dtype={"accident_id": str, "doc_id": str})
        if len(df) != entry["units"] or df.doc_id.duplicated().any() or df.accident_id.isna().any():
            raise ValueError(f"Wrong unit count or duplicate ID: {corpus}")
        if set(df.pred_label) != set(ROLES) or not df.pred_ok.astype(str).str.lower().eq("true").all():
            raise ValueError(f"Missing role or invalid factual annotation: {corpus}")
        if not all(isinstance(r.sentence, str) and isinstance(r.accident_summary, str) and
                   r.accident_summary[int(r.start):int(r.end)] == r.sentence
                   for r in df.itertuples(index=False)):
            raise ValueError(f"Cleaned offsets/text disagree: {corpus}")
        original_path = TEXT / "dataset" / f"data_{stem}.csv"
        original = pd.read_csv(original_path, usecols=["accident_id", "doc_id", "sentence", "pred_label"],
                               dtype={"accident_id": str, "doc_id": str})
        fields = ["accident_id", "doc_id", "sentence", "pred_label"]
        if not original[fields].equals(df[fields]):
            raise ValueError(f"Cleaned units differ from original embeddings source: {corpus}")
        df["label_id"] = df.pred_label.map(LABEL2ID).astype(int)
        df["corpus"] = corpus
        frames[corpus] = df
        signatures["clean_csv_sha256"][corpus] = digest(csv)
        signatures["original_csv_sha256"][corpus] = digest(original_path)
        own_split_file = b / "folds" / f"target_{corpus}" / "split_ids.json"
        relative = own_split_file.relative_to(b).as_posix()
        if digest(own_split_file) != m["files"].get(relative):
            raise ValueError(f"Changed split IDs: {corpus}")
        signatures["split_sha256"][corpus] = digest(own_split_file)
        own = read(own_split_file)
        test = [str(x) for x in own["test"][corpus]]
        splits[corpus] = {"test": test}
        # A source's train/val IDs must be identical in all five target rotations.
        source_folds = []
        for target in CORPORA:
            if target == corpus:
                continue
            path = b / "folds" / f"target_{target}" / "split_ids.json"
            source_folds.append(read(path))
        for part in ("train", "val"):
            values = [[str(x) for x in fold[part][corpus]] for fold in source_folds]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"Inconsistent source {part} IDs: {corpus}")
            splits[corpus][part] = values[0]
        id_sets = [set(splits[corpus][part]) for part in ("train", "val", "test")]
        # The corpus's own target test contains every document; its source
        # train/val partition is used only in the five *other* rotations.
        if (id_sets[0] & id_sets[1] or id_sets[0] | id_sets[1] != id_sets[2] or
                len(id_sets[2]) != entry["documents"] or
                not set(df.accident_id).issubset(id_sets[2]) or
                len(test) != entry["documents"]):
            raise ValueError(f"Narrative split leakage/mismatch: {corpus}")
        for part in ("train", "val", "test"):
            subset = df[df.accident_id.isin(id_sets[("train", "val", "test").index(part)])]
            if subset.empty or set(subset.pred_label) != set(ROLES):
                raise ValueError(f"Missing roles in {corpus}/{part}")
        if check_qwen:
            qwen_path = old_qwen.qwen_path(corpus)
            header = pd.read_csv(qwen_path, nrows=0).columns.tolist()
            ids = pd.read_csv(qwen_path, usecols=["doc_id"], dtype={"doc_id": str}).doc_id.tolist()
            if (len(header) != 1025 or header[0] != "doc_id" or
                    not all(name.startswith("dim_") for name in header[1:]) or
                    len(set(header)) != len(header) or ids != df.doc_id.tolist()):
                raise ValueError(f"Qwen3 export schema/doc_id order mismatch: {corpus}")
            signatures["qwen_sha256"][corpus] = digest(qwen_path)
    if sum(len(df) for df in frames.values()) != 147487:
        raise ValueError("Expected 147487 factual units")
    return frames, splits, signatures


def selection(target: str, frames: dict, splits: dict):
    sources = [c for c in CORPORA if c != target]
    part = lambda c, p: frames[c][frames[c].accident_id.isin(splits[c][p])].copy()
    train = pd.concat([part(c, "train") for c in sources], ignore_index=True)
    val = pd.concat([part(c, "val") for c in sources], ignore_index=True)
    test = part(target, "test")
    assert len(test) == len(frames[target])
    return train, val, test


def macro_source_ba(frame: pd.DataFrame, predictions: np.ndarray, target: str) -> tuple[float, dict]:
    if len(frame) != len(predictions):
        raise ValueError("Validation prediction count differs")
    per_source = {}
    for corpus in CORPORA:
        if corpus == target:
            continue
        mask = frame.corpus.eq(corpus).to_numpy()
        per_source[corpus] = old_role.scores(frame.label_id.to_numpy()[mask], predictions[mask])["balanced_accuracy"]
    if len(per_source) != 5:
        raise ValueError("Need five source validation domains")
    return float(np.mean(list(per_source.values()))), per_source


def choose(current: dict | None, candidate: dict) -> bool:
    if current is None:
        return True
    if candidate["source_macro_ba"] > current["source_macro_ba"] + 1e-12:
        return True
    if abs(candidate["source_macro_ba"] - current["source_macro_ba"]) <= 1e-12:
        return (float(candidate.get("C") or 0), int(candidate.get("epoch") or 0)) < \
               (float(current.get("C") or 0), int(current.get("epoch") or 0))
    return False


def campaign() -> Path:
    path = root() / "campaign.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run preparation first: {path}")
    record = read(path)
    _, _, signatures = load_data(check_qwen=True)
    if record["signatures"] != signatures or record["script_sha256"] != digest(Path(__file__)):
        raise ValueError("Campaign inputs or code changed since preflight")
    return path


def prepare(_args) -> None:
    frames, splits, signatures = load_data(check_qwen=True)
    expected = {"version": "role_qwen3_narrative_lodo_3epoch_v1", "signatures": signatures,
                "script_sha256": digest(Path(__file__)), "roles": list(ROLES),
                "backbone": BACKBONE, "revision": REVISION, "epochs": EPOCHS,
                "seeds": list(SEEDS), "methods": list(METHODS), "C_grid": list(CS),
                "selection": "mean of five source-domain balanced accuracies",
                "splits": {c: {p: {"documents": len(splits[c][p]),
                                     "facts": int(frames[c].accident_id.isin(splits[c][p]).sum())}
                               for p in ("train", "val", "test")} for c in CORPORA}}
    path = root() / "campaign.json"
    if path.is_file() and read(path) != expected:
        raise ValueError(f"Existing campaign binding differs: {path}")
    if not path.exists():
        write(path, expected)
    print(json.dumps({"campaign": str(path), "facts": 147487, "targets": list(CORPORA)}), flush=True)


def verified_done(dest: Path, expected: Path) -> bool:
    marker = dest / "complete.json"
    if not marker.is_file():
        return False
    try:
        record = read(marker)
        return (record["campaign_sha256"] == digest(expected) and
                all(digest(dest / name) == sha for name, sha in record["files_sha256"].items()))
    except (KeyError, OSError, ValueError):
        return False


def save_result(dest: Path, test: pd.DataFrame, pred: np.ndarray, probs: np.ndarray,
                target: str, method: str, seed: int | None, chosen: dict, history: list) -> None:
    if len(pred) != len(test) or probs.shape != (len(test), 5) or not np.isfinite(probs).all():
        raise ValueError("Invalid test predictions")
    dest.mkdir(parents=True, exist_ok=True)
    output = test[["corpus", "accident_id", "doc_id", "fact_id", "pred_label"]].copy()
    output["prediction"] = [ROLES[i] for i in pred]
    for i, role in enumerate(ROLES):
        output[f"prob_{role}"] = probs[:, i]
    tmp = dest / "predictions.tmp.csv"
    output.to_csv(tmp, index=False)
    os.replace(tmp, dest / "predictions.csv")
    metrics = old_role.scores(test.label_id.to_numpy(), pred)
    metrics.update({"target": target, "method": method, "seed": seed,
                    "selection": chosen, "predictions_sha256": digest(dest / "predictions.csv")})
    write(dest / "metrics.json", metrics)
    write(dest / "history.json", history)
    files = ("predictions.csv", "metrics.json", "history.json")
    write(dest / "complete.json", {"campaign_sha256": digest(campaign()),
         "files_sha256": {name: digest(dest / name) for name in files},
         "checkpoint_retention": "predictions_only"})
    if not verified_done(dest, campaign()):
        raise RuntimeError(f"Result verification failed: {dest}")


def vectors(frame: pd.DataFrame, by_corpus: dict) -> np.ndarray:
    # doc_id is only unique within a corpus, so index within each corpus first.
    result = np.empty((len(frame), 1024), dtype=np.float32)
    for corpus in frame.corpus.unique():
        mask = frame.corpus.eq(corpus).to_numpy()
        result[mask] = by_corpus[corpus].loc[frame.loc[mask, "doc_id"].astype(str)].to_numpy(dtype=np.float32)
    if result.shape != (len(frame), 1024) or not np.isfinite(result).all():
        raise ValueError("Invalid frozen Qwen3 feature matrix")
    return result


def baseline(_args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from macro_transfer.tfidf_baseline import fit_vectorizer, normalize_stopwords

    binding = campaign()
    frames, splits, _ = load_data(check_qwen=True)
    stops = normalize_stopwords(old_qwen.STOPWORDS, TFIDF_PARAMS)
    for target in CORPORA:
        train, val, test = selection(target, frames, splits)
        for method in ("tfidf_lr", "qwen3_frozen_lr"):
            dest = root() / "baselines" / target / method
            if verified_done(dest, binding):
                print(f"[skip] {dest}", flush=True)
                continue
            if method == "tfidf_lr":
                vectorizer, x_train = fit_vectorizer(train.sentence.tolist(), TFIDF_PARAMS, stops, "source train")
                x_val, x_test = vectorizer.transform(val.sentence.tolist()), vectorizer.transform(test.sentence.tolist())
                extra = {"vocabulary_size": len(vectorizer.vocabulary_), "stopwords_sha256": digest(old_qwen.STOPWORDS)}
            else:
                by_corpus = {}
                for corpus in CORPORA:
                    path = old_qwen.qwen_path(corpus)
                    d = pd.read_csv(path, dtype={"doc_id": str})
                    by_corpus[corpus] = d.set_index("doc_id")
                x_train, x_val, x_test = vectors(train, by_corpus), vectors(val, by_corpus), vectors(test, by_corpus)
                extra = {"dimensions": 1024}
            best, best_model, history = None, None, []
            for c in CS:
                lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=SEEDS[0])
                model = make_pipeline(StandardScaler(), lr) if method == "qwen3_frozen_lr" else lr
                model.fit(x_train, train.label_id.to_numpy())
                score, per_source = macro_source_ba(val, model.predict(x_val).astype(int), target)
                candidate = {"C": c, "epoch": None, "source_macro_ba": score,
                             "source_ba": per_source}
                history.append(candidate)
                if choose(best, candidate):
                    best, best_model = candidate, model
            probs = np.zeros((len(test), 5), dtype=np.float32)
            probs[:, best_model.classes_.astype(int) if method == "tfidf_lr" else
                  best_model[-1].classes_.astype(int)] = best_model.predict_proba(x_test)
            pred = probs.argmax(axis=1)
            save_result(dest, test, pred, probs, target, method, None,
                        {**best, **extra}, history)
            print(f"[done] {target}/{method}: BA={old_role.scores(test.label_id.to_numpy(), pred)['balanced_accuracy']:.4f}", flush=True)
            del x_train, x_val, x_test, best_model
            if method == "qwen3_frozen_lr":
                del by_corpus


def spec(index: int) -> tuple[str, int, str]:
    return old_role.task_spec(index)


def text_loader(frame, tokenizer, *, batch_size, shuffle=False, sampler=None):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Facts(Dataset):
        def __init__(self, source):
            self.text = source.sentence.astype(str).tolist()
            self.labels = source.label_id.astype(int).tolist()
        def __len__(self):
            return len(self.text)
        def __getitem__(self, index):
            return self.text[index], self.labels[index]

    def collate(items):
        # Qwen3 exports were created without E5's `query: ` prefix.
        encoded = tokenizer([value for value, _ in items], padding=True,
                            truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
                "labels": torch.tensor([label for _, label in items], dtype=torch.long)}

    kwargs = {"batch_sampler": sampler} if sampler is not None else {
        "batch_size": batch_size, "shuffle": shuffle}
    return DataLoader(Facts(frame), collate_fn=collate, num_workers=2,
                      pin_memory=True, **kwargs)


def metric_embeddings(model, frame, tokenizer, device, batch_size):
    import torch
    import torch.nn.functional as F
    model.eval()
    chunks = []
    with torch.inference_mode():
        for batch in text_loader(frame, tokenizer, batch_size=batch_size):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model({"input_ids": batch["input_ids"].to(device),
                           "attention_mask": batch["attention_mask"].to(device)})
            chunks.append(F.normalize(z.float(), p=2, dim=1).cpu().numpy())
    return np.vstack(chunks)


def train_one(target: str, seed: int, method: str, *, smoke: bool = False) -> None:
    import torch
    from transformers import AutoTokenizer
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig
    from contrastive_methods.hf_training_common import load_contrastive_checkpoint, save_contrastive_checkpoint
    from contrastive_methods.losses.softtriple import SoftTripleLoss
    from contrastive_methods.losses.supcon_hobbit import SupConEmbeddingLoss
    from contrastive_methods.samplers.pk_batch_sampler import build_pk_batch_sampler, resolve_balanced_pk_params
    from supervised_macro_ft.checkpoint_io import load_checkpoint, save_checkpoint
    from supervised_macro_ft.model import SupervisedMacroModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from joblib import dump, load

    binding = campaign()
    dest = root() / "trained" / target / method / f"seed_{seed}"
    if not smoke and verified_done(dest, binding):
        checkpoint = dest / "best_model"
        if checkpoint.is_dir():
            if checkpoint.is_symlink() or checkpoint.resolve().parent != dest.resolve():
                raise RuntimeError("Unsafe checkpoint cleanup path")
            shutil.rmtree(checkpoint)
        print(f"[skip] {dest}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Full Qwen3 fine-tuning needs CUDA")
    frames, splits, _ = load_data(check_qwen=False)
    train, val, test = selection(target, frames, splits)
    if smoke:
        # Real labels from every source, small enough to test all three objectives.
        train = pd.concat([train[train.pred_label.eq(r)].head(4) for r in ROLES], ignore_index=True)
        val = pd.concat([val[(val.corpus == c) & val.pred_label.eq(r)].head(1)
                         for c in CORPORA if c != target for r in ROLES], ignore_index=True)
        test = test.head(8).copy()
        dest = root() / "smoke" / method
    old_role.seed_all(seed)
    device = torch.device("cuda")
    model_ref = old_qwen.backbone_path()
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if method == "ce":
        model = SupervisedMacroModel(backbone_name=model_ref, num_classes=5, pooling="mean",
            backbone_trainable=True, train_last_n_layers=None, gradient_checkpointing=True,
            projection=None, dropout=0.1).to(device)
        counts = np.bincount(train.label_id.to_numpy(), minlength=5).astype(np.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(5*counts), device=device))
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 2e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3}], weight_decay=0.01)
        sampler = None
    else:
        model = ContrastiveEncoder(EncoderConfig(backbone_name=model_ref,
            max_seq_length=MAX_LENGTH, backbone_trainable=True, train_last_n_layers=None,
            cache_backbone_embeddings=False, use_projector=True,
            projection="linear", hiddim=128)).to(device)
        if method == "supcon":
            loss_fn = SupConEmbeddingLoss(temperature=0.07, normalize_embeddings=True)
            sampler = build_pk_batch_sampler(train.label_id.tolist(),
                resolve_balanced_pk_params(train.label_id.tolist(), 10, seed=seed))
        else:
            loss_fn = SoftTripleLoss(embedding_dim=128, num_classes=5, centers_per_class=5,
                gamma=0.1, la=10.0, delta=0.01, tau=0.2, normalize_embeddings=True,
                normalize_centers=True, distance_metric="cosine",
                center_regularization_type="merge_l21").to(device)
            sampler = None
        groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 2e-5},
                  {"params": model.projector.parameters(), "lr": 1e-3}]
        if list(loss_fn.parameters()):
            groups.append({"params": loss_fn.parameters(), "lr": 1e-3})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    if not all(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("Qwen3 backbone is not fully trainable")
    layers = model.backbone._get_transformer_layers()
    witness = next(p for p in layers[-1].parameters() if p.requires_grad)
    initial = witness.detach().float().cpu().clone()
    loader = text_loader(train, tokenizer, batch_size=10 if method == "supcon" else 4,
                         shuffle=sampler is None, sampler=sampler)
    val_loader = text_loader(val, tokenizer, batch_size=16)
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = dest / "best_model"
    best, history = None, []
    for epoch in range(1 if smoke else EPOCHS):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total, count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accumulation = 1 if method == "supcon" else 4
        for batch_index, batch in enumerate(loader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x, mask, y = (batch["input_ids"].to(device), batch["attention_mask"].to(device),
                              batch["labels"].to(device))
                if method == "ce":
                    loss = loss_fn(model.forward_logits(x, mask), y)
                else:
                    output = loss_fn(model({"input_ids": x, "attention_mask": mask}), y)
                    loss = output[0] if isinstance(output, tuple) else output
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {method} loss at epoch {epoch + 1}")
            group_size = min(accumulation, len(loader) - (batch_index // accumulation) * accumulation)
            (loss / group_size).backward()
            total += float(loss.detach().cpu()); count += 1
            if (batch_index + 1) % accumulation == 0 or (batch_index + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if method == "ce":
            val_pred, _ = old_role.ce_predictions(model, val_loader, device)
            score, per_source = macro_source_ba(val, val_pred, target)
            row = {"epoch": epoch + 1, "C": None, "source_macro_ba": score,
                   "source_ba": per_source, "train_loss": total / count}
            candidates = [(row, None)]
        else:
            x_train = metric_embeddings(model, train, tokenizer, device, 32)
            x_val = metric_embeddings(model, val, tokenizer, device, 32)
            candidates = []
            for c in CS:
                classifier = make_pipeline(StandardScaler(), LogisticRegression(
                    C=c, class_weight="balanced", max_iter=1000, random_state=seed))
                classifier.fit(x_train, train.label_id.to_numpy())
                score, per_source = macro_source_ba(val, classifier.predict(x_val).astype(int), target)
                candidates.append(({"epoch": epoch + 1, "C": c, "source_macro_ba": score,
                                   "source_ba": per_source, "train_loss": total / count}, classifier))
        for row, classifier in candidates:
            history.append(row)
            if choose(best, row):
                best = row
                if method == "ce":
                    save_checkpoint(model, checkpoint, config={"backbone_name": model_ref,
                        "num_classes": 5, "pooling": "mean", "projection": None,
                        "input_prefix": "", "max_seq_length": MAX_LENGTH})
                else:
                    save_contrastive_checkpoint(model, checkpoint)
                    dump(classifier, checkpoint / "classifier.joblib")
        print(json.dumps({"method": method, "seed": seed, "epoch": epoch+1,
                          "best_source_ba": best["source_macro_ba"]}), flush=True)
    if torch.equal(initial, witness.detach().float().cpu()):
        raise RuntimeError(f"Smoke/train failed to update full Qwen3 encoder: {method}")
    del layers, witness, model, optimizer, loss_fn
    torch.cuda.empty_cache()
    if method == "ce":
        selected = load_checkpoint(checkpoint, device="cuda")
        pred, probs = old_role.ce_predictions(selected, text_loader(test, tokenizer, batch_size=16), device)
    else:
        cfg = ContrastiveConfig(method_name=method, dataset_path=Path("unused"),
            backbone_name=model_ref, input_prefix="", backbone_trainable=True,
            train_last_n_layers=None, cache_backbone_embeddings=False,
            max_seq_length=MAX_LENGTH, use_projector=True, projection="linear",
            hiddim=128, use_amp=True, encode_batch_size=32, seed=seed)
        selected = load_contrastive_checkpoint(cfg, checkpoint, "cuda")
        classifier = load(checkpoint / "classifier.joblib")
        x_test = metric_embeddings(selected, test, tokenizer, device, 32)
        pred, probs = old_role.predict_lr(classifier, x_test)
    del selected
    torch.cuda.empty_cache()
    save_result(dest, test, pred, probs, target, method, seed, best, history)
    # The completion marker and predictions are checked before deleting weights.
    if checkpoint.resolve().parent != dest.resolve() or checkpoint.is_symlink():
        raise RuntimeError("Unsafe checkpoint cleanup path")
    shutil.rmtree(checkpoint)
    if not verified_done(dest, binding):
        raise RuntimeError("Result changed after checkpoint cleanup")
    print(f"[done] {dest}", flush=True)


def smoke(_args) -> None:
    campaign()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    frames, _, _ = load_data(check_qwen=True)
    old_qwen.verify_qwen_export_revision(frames, "cuda")
    for method in METHODS:
        train_one("btp", SEEDS[0], method, smoke=True)
    write(root() / "smoke_complete.json", {"campaign_sha256": digest(campaign()),
          "methods": list(METHODS)})


def train(args) -> None:
    marker = read(root() / "smoke_complete.json")
    if marker.get("campaign_sha256") != digest(campaign()) or marker.get("methods") != list(METHODS):
        raise ValueError("Run the GPU smoke first")
    target, seed, method = spec(args.task_index)
    train_one(target, seed, method)


def summary(_args) -> None:
    binding = campaign()
    frames, _, _ = load_data(check_qwen=False)
    rows = []
    for target in CORPORA:
        for method in ("tfidf_lr", "qwen3_frozen_lr", *METHODS):
            for seed in ((None,) if method.endswith("_lr") else SEEDS):
                dest = (root() / "baselines" / target / method if seed is None else
                        root() / "trained" / target / method / f"seed_{seed}")
                if not verified_done(dest, binding):
                    raise ValueError(f"Missing or changed result: {dest}")
                metrics = read(dest / "metrics.json")
                predictions = pd.read_csv(dest / "predictions.csv", dtype={"doc_id": str})
                expected = frames[target]
                if (len(predictions) != len(expected) or
                        predictions.doc_id.tolist() != expected.doc_id.tolist() or
                        predictions.pred_label.tolist() != expected.pred_label.tolist()):
                    raise ValueError(f"Predictions are not on the complete target: {dest}")
                predicted_ids = predictions.prediction.map(LABEL2ID)
                if predicted_ids.isna().any():
                    raise ValueError(f"Invalid role prediction: {dest}")
                recomputed = old_role.scores(expected.label_id.to_numpy(), predicted_ids.to_numpy(dtype=int))
                for metric in ("balanced_accuracy", "macro_f1"):
                    if abs(metrics[metric] - recomputed[metric]) > 1e-10:
                        raise ValueError(f"Stored {metric} differs from predictions: {dest}")
                rows.append({"target": target, "method": method, "seed": seed,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"], "n_units": metrics["n_units"]})
    out = root() / "results"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "scores_by_run.csv", index=False)
    grouped = table.groupby(["target", "method"], as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        n_runs=("balanced_accuracy", "size"))
    grouped[["balanced_accuracy_sd", "macro_f1_sd"]] = grouped[["balanced_accuracy_sd", "macro_f1_sd"]].fillna(0)
    grouped.to_csv(out / "scores_by_target.csv", index=False)
    by_seed = table.groupby(["method", "seed"], dropna=False, as_index=False).agg(
        macro_target_balanced_accuracy=("balanced_accuracy", "mean"),
        macro_target_f1=("macro_f1", "mean"), n_targets=("target", "nunique"))
    if not by_seed.n_targets.eq(6).all():
        raise ValueError("Incomplete six-target macro average")
    by_seed.to_csv(out / "macro_by_seed.csv", index=False)
    macro = by_seed.groupby("method", as_index=False).agg(
        macro_target_ba_mean=("macro_target_balanced_accuracy", "mean"),
        macro_target_ba_sd=("macro_target_balanced_accuracy", "std"),
        macro_target_f1_mean=("macro_target_f1", "mean"),
        macro_target_f1_sd=("macro_target_f1", "std"),
        n_seeds=("macro_target_balanced_accuracy", "size"))
    macro[["macro_target_ba_sd", "macro_target_f1_sd"]] = macro[["macro_target_ba_sd", "macro_target_f1_sd"]].fillna(0)
    macro.to_csv(out / "macro_six_targets.csv", index=False)
    write(out / "manifest.json", {"campaign_sha256": digest(binding), "runs": len(rows),
        "targets": list(CORPORA), "methods": ["tfidf_lr", "qwen3_frozen_lr", *METHODS],
        "note": "Seed standard deviations are computed within target; macro scores average six target means."})
    print(f"[done] {out}: {len(rows)} results", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "baseline", "summary"):
        sub.add_parser(name)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke, "baseline": baseline,
     "train": train, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
