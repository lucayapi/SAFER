"""Baseline lexicale du notebook 07c : TF-IDF, CV groupée et transfert OOD.

Les vectoriseurs sont ajustés sur le train uniquement. Les caches CV, modèle
final et corpus cible sont indépendants et identifiés par leur contenu.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from safer_core.kfold_eval import group_kfold_splits
from scgm_text.data_metadata import load_filtered_metadata

MACROS = ("A0", "A1", "B", "C")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def cache_ready(directory: Path, signature: str, files: list[str]) -> bool:
    manifest = directory / "manifest.json"
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("signature") == signature and all((directory / name).is_file() for name in files)


def mark_complete(directory: Path, signature: str) -> None:
    # Écrit en dernier : une exécution interrompue n'est pas un cache complet.
    write_json(directory / "manifest.json", {"signature": signature})


def normalize_stopwords(path: Path, tfidf_params: dict) -> list[str]:
    """Même prétraitement/tokenisation que TF-IDF, sans générer des n-grams."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de stopwords introuvable : {path}")
    probe = TfidfVectorizer(**tfidf_params)
    preprocess, tokenize = probe.build_preprocessor(), probe.build_tokenizer()
    words = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            words.update(tokenize(preprocess(entry)))
    if not words:
        raise ValueError(f"Aucun stopword exploitable dans {path}")
    return sorted(words)


def fit_vectorizer(texts, params: dict, stopwords: list[str], context: str):
    vectorizer = TfidfVectorizer(**params, stop_words=stopwords or None, dtype=np.float32)
    try:
        matrix = vectorizer.fit_transform(texts).tocsr()
    except ValueError as exc:
        raise ValueError(
            f"TF-IDF ({context}) : {exc}. Vérifier les textes, stopwords, min_df, max_df et ngram_range."
        ) from exc
    return vectorizer, matrix


def load_text_corpus(spec: dict):
    """Réutilise les filtres du projet ; conserve les textes vides pour l'évaluation."""
    path = Path(spec["dataset_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Corpus introuvable : {path}")
    columns = {k: spec.get(k, v) for k, v in {
        "text_col": "sentence", "label_col": "pred_label",
        "group_col": "accident_id", "pred_ok_col": "pred_ok",
    }.items()}
    header = pd.read_csv(path, nrows=0).columns
    missing = set(columns.values()) - set(header)
    if missing:
        raise ValueError(f"{path.name} : colonnes absentes {sorted(missing)}")
    n_raw = sum(len(chunk) for chunk in pd.read_csv(path, usecols=[columns["label_col"]], chunksize=50000))
    frame = load_filtered_metadata(str(path), **columns)
    if frame.empty:
        raise ValueError(f"{path.name} : aucune unité après filtrage pred_ok / labels")
    group = frame[columns["group_col"]]
    if group.isna().any() or group.astype(str).str.strip().eq("").any():
        raise ValueError(f"{path.name} : accident_id manquant ; CV groupée impossible")
    frame[columns["text_col"]] = frame[columns["text_col"]].fillna("").astype(str)
    frame[columns["label_col"]] = frame[columns["label_col"]].astype(str).str.strip()
    summary = {
        "n_raw": n_raw, "n_kept": len(frame), "n_excluded": n_raw - len(frame),
        "n_accidents": group.nunique(),
        "n_empty_text": int(frame[columns["text_col"]].str.strip().eq("").sum()),
    }
    summary.update(frame[columns["label_col"]].value_counts().reindex(MACROS, fill_value=0).to_dict())
    return frame, summary


def build_classifier(model_key: str, params: dict, seed: int, n_jobs: int):
    kwargs = {"random_state": seed, **params}
    if model_key == "logistic_regression":
        return LogisticRegression(**kwargs)
    if model_key == "random_forest":
        return RandomForestClassifier(**{"n_jobs": n_jobs, **kwargs})
    if model_key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("XGBoost est requis pour le troisième modèle : pip install xgboost") from exc
        return XGBClassifier(**{"n_jobs": n_jobs, "tree_method": "hist", **kwargs})
    raise ValueError(f"Modèle inconnu : {model_key}")


def aligned_probabilities(model, matrix) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=float)
    probabilities = np.zeros((matrix.shape[0], len(MACROS)), dtype=float)
    for column, label_id in enumerate(model.classes_):
        probabilities[:, int(label_id)] = raw[:, column]
    return probabilities


def classification_results(y_true, probabilities):
    y_pred = probabilities.argmax(axis=1)
    scores = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=range(4), average="macro", zero_division=0)),
    }
    report = pd.DataFrame(classification_report(
        y_true, y_pred, labels=range(4), target_names=MACROS, output_dict=True, zero_division=0,
    )).T
    counts = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=range(4)), index=MACROS, columns=MACROS)
    normalized = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return scores, report, counts, normalized


def prediction_frame(metadata, probabilities, zero_mask, *, corpus: str, model_key: str, label_col: str):
    out = metadata.copy()
    out.insert(0, "corpus", corpus)
    out["method"] = f"tfidf_{model_key}"
    out["true_macro"] = out[label_col]
    out["pred_macro"] = np.asarray(MACROS)[probabilities.argmax(axis=1)]
    for idx, label in enumerate(MACROS):
        out[f"prob_{label}"] = probabilities[:, idx]
    ordered = np.sort(probabilities, axis=1)
    out["confidence"] = ordered[:, -1]
    out["margin"] = ordered[:, -1] - ordered[:, -2]
    out["entropy"] = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1)
    out["zero_tfidf"] = zero_mask
    return out


class TfidfExperiment:
    """Une instance par exécution du notebook ; paramètres immuables après création."""

    def __init__(self, config: dict, *, restimate: bool = False):
        self.config = deepcopy(config)
        self.root = Path(config["output_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.restimate = restimate
        self.seed = int(config["seed"])
        self.specs = {"btp": config["source"], **config["targets"]}
        if "btp" in config["targets"]:
            raise ValueError("BTP est la source, pas un corpus cible OOD")
        self.models = deepcopy(config["models"])
        if not self.models or set(self.models) - set(MODEL_NAMES):
            raise ValueError("Choisir au moins un modèle parmi LR, Random Forest et XGBoost")
        if config["selection_metric"] not in METRICS:
            raise ValueError(f"selection_metric doit appartenir à {METRICS}")
        params = dict(config["tfidf"])
        if params.get("analyzer", "word") != "word":
            raise ValueError("Ce notebook utilise des n-grams de mots : analyzer='word'")
        self.stopwords = normalize_stopwords(Path(config["stopwords_file"]), params) if config["use_stopwords"] else []
        self.metadata, self.file_hashes, summaries = {}, {}, []
        for corpus, spec in self.specs.items():
            print(f"Chargement : {corpus}", flush=True)
            self.metadata[corpus], summary = load_text_corpus(spec)
            summaries.append({"corpus": corpus, **summary})
            self.file_hashes[corpus] = file_digest(Path(spec["dataset_path"]))
        self.data_summary = pd.DataFrame(summaries)
        self.data_summary.to_csv(self.root / "data_summary.csv", index=False)
        self.runtime = {pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}
        if "xgboost" in self.models:
            self.runtime["xgboost"] = version("xgboost")
        stop_hash = file_digest(Path(config["stopwords_file"])) if config["use_stopwords"] else None
        self.feature_signature = fingerprint({
            "source": config["source"], "source_hash": self.file_hashes["btp"],
            "tfidf": params, "stopwords_hash": stop_hash, "stopwords": self.stopwords,
            "runtime": self.runtime, "code": file_digest(Path(__file__)),
            "loader": file_digest(Path(__file__).parents[1] / "scgm_text" / "data_metadata.py"),
            "metadata_utils": file_digest(Path(__file__).parents[1] / "scgm_text" / "utils_io.py"),
            "splitter": file_digest(Path(__file__).parents[1] / "safer_core" / "kfold_eval.py"),
        })
        source = self.metadata["btp"]
        self.y = source["label_id"].to_numpy()
        groups = source[config["source"].get("group_col", "accident_id")].astype(str).to_numpy()
        n_folds = int(config["n_folds"])
        if n_folds < 2 or n_folds > len(np.unique(groups)):
            raise ValueError(f"n_folds={n_folds} incompatible avec {len(np.unique(groups))} accidents")
        self.splits = group_kfold_splits(groups, n_folds, self.seed)
        assignment = source[[config["source"].get("group_col", "accident_id"), "doc_id"]].copy()
        for fold_id, (train, val) in enumerate(self.splits):
            absent = set(range(4)) - set(self.y[train])
            if absent:
                raise ValueError(f"Fold {fold_id + 1} : classes absentes du train : {[MACROS[i] for i in sorted(absent)]}")
            if set(groups[train]) & set(groups[val]):
                raise ValueError(f"Fold {fold_id + 1} : fuite d'accidents train/validation")
            assignment.loc[val, "fold_id"] = fold_id
        (self.root / "cv").mkdir(exist_ok=True)
        assignment.to_csv(self.root / "cv" / "fold_assignments.csv", index=False)
        write_json(self.root / "config_resolved.json", {
            **config, "data_sha256": self.file_hashes, "stopwords_sha256": stop_hash,
            "normalized_stopwords": self.stopwords, "runtime": self.runtime,
            "feature_signature": self.feature_signature,
        })
        self._fold_features = {}
        self._final_models = {}
        self._final_features = None
        self._target_features = {}
        self.cv_results = {}
        self.test_results = {}

    def _texts(self, corpus):
        return self.metadata[corpus][self.specs[corpus].get("text_col", "sentence")].tolist()

    def model_signature(self, model_key):
        return fingerprint({"features": self.feature_signature, "model": model_key,
                            "params": self.models[model_key], "seed": self.seed,
                            "n_jobs": self.config.get("n_jobs", 4)})

    def cv_signature(self, model_key):
        return fingerprint({"model": self.model_signature(model_key), "n_folds": len(self.splits)})

    def fold_features(self, fold_id):
        """Vectorisation partagée entre les modèles, jamais entre train et validation."""
        if fold_id not in self._fold_features:
            train, val = self.splits[fold_id]
            texts = np.asarray(self._texts("btp"), dtype=object)
            vectorizer, x_train = fit_vectorizer(texts[train], self.config["tfidf"], self.stopwords, f"fold {fold_id + 1}")
            x_val = vectorizer.transform(texts[val]).tocsr()
            self._fold_features[fold_id] = (vectorizer, x_train, x_val)
        return self._fold_features[fold_id]

    def _estimator(self, model_key):
        return build_classifier(model_key, self.models[model_key], self.seed, self.config.get("n_jobs", 4))

    def run_cv(self, model_key):
        directory = self.root / "cv" / model_key
        signature = self.cv_signature(model_key)
        files = ["cv_per_fold.csv", "oof_predictions.csv"]
        if not self.restimate and cache_ready(directory, signature, files):
            print(f"{MODEL_NAMES[model_key]} : CV rechargée", flush=True)
            result = pd.read_csv(directory / files[0])
        else:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "manifest.json").unlink(missing_ok=True)
            rows, predictions = [], []
            for fold_id, (train, val) in enumerate(self.splits):
                print(f"{MODEL_NAMES[model_key]} : fold {fold_id + 1}/{len(self.splits)}", flush=True)
                started = time.perf_counter()
                _, x_train, x_val = self.fold_features(fold_id)
                model = self._estimator(model_key)
                model.fit(x_train, self.y[train])
                probs = aligned_probabilities(model, x_val)
                scores, _, _, _ = classification_results(self.y[val], probs)
                zeros = x_val.getnnz(axis=1) == 0
                rows.append({"model": model_key, "fold_id": fold_id, "n_train": len(train),
                             "n_val": len(val), "n_features": x_train.shape[1],
                             "n_zero_train": int((x_train.getnnz(axis=1) == 0).sum()),
                             "n_zero_val": int(zeros.sum()), "wall_time_sec": time.perf_counter() - started, **scores})
                pred = prediction_frame(self.metadata["btp"].iloc[val], probs, zeros, corpus="btp",
                                        model_key=model_key, label_col=self.config["source"].get("label_col", "pred_label"))
                pred["fold_id"] = fold_id
                predictions.append(pred)
            result = pd.DataFrame(rows)
            result.to_csv(directory / files[0], index=False)
            pd.concat(predictions).sort_index().to_csv(directory / files[1], index=False)
            mark_complete(directory, signature)
        self.cv_results[model_key] = result
        return result

    def summarize_cv(self):
        missing = set(self.models) - set(self.cv_results)
        if missing:
            raise ValueError(f"Exécuter la CV des modèles : {sorted(missing)}")
        all_rows = pd.concat([self.cv_results[key] for key in self.models], ignore_index=True)
        rows = []
        for key in self.models:
            frame = self.cv_results[key]
            row = {"model": key, "n_folds": len(frame)}
            for metric in METRICS:
                row[f"mean_{metric}"] = frame[metric].mean()
                row[f"std_{metric}"] = frame[metric].std(ddof=1)
            rows.append(row)
        self.cv_summary = pd.DataFrame(rows)
        metric = self.config["selection_metric"]
        self.best_model = str(self.cv_summary.loc[self.cv_summary[f"mean_{metric}"].idxmax(), "model"])
        all_rows.to_csv(self.root / "cv" / "cv_per_fold.csv", index=False)
        self.cv_summary.to_csv(self.root / "cv" / "cv_summary.csv", index=False)
        write_json(self.root / "selection.json", {"best_model": self.best_model, "selection_metric": metric,
                                                  "selection_data": "CV BTP uniquement"})
        return self.cv_summary

    def final_features(self):
        if self._final_features is None:
            directory = self.root / "models"
            directory.mkdir(exist_ok=True)
            files = ["vectorizer.joblib", "vocabulary.csv"]
            if not self.restimate and cache_ready(directory, self.feature_signature, files):
                vectorizer = joblib.load(directory / files[0])
                matrix = vectorizer.transform(self._texts("btp")).tocsr()
            else:
                (directory / "manifest.json").unlink(missing_ok=True)
                vectorizer, matrix = fit_vectorizer(self._texts("btp"), self.config["tfidf"], self.stopwords, "BTP complet")
                joblib.dump(vectorizer, directory / files[0])
                pd.DataFrame({"term": vectorizer.get_feature_names_out(), "idf": vectorizer.idf_}).to_csv(directory / files[1], index=False)
                mark_complete(directory, self.feature_signature)
            self._final_features = vectorizer, matrix
        return self._final_features

    def fit_final(self):
        # Les matrices CV ne sont plus nécessaires pendant le fit final.
        self._fold_features.clear()
        vectorizer, matrix = self.final_features()
        for key in self.models:
            if key in self._final_models:
                continue
            directory = self.root / "models" / key
            signature = self.model_signature(key)
            if not self.restimate and cache_ready(directory, signature, ["classifier.joblib"]):
                print(f"{MODEL_NAMES[key]} : modèle final rechargé", flush=True)
                model = joblib.load(directory / "classifier.joblib")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "manifest.json").unlink(missing_ok=True)
                print(f"{MODEL_NAMES[key]} : entraînement final sur tout le BTP", flush=True)
                model = self._estimator(key)
                model.fit(matrix, self.y)
                joblib.dump(model, directory / "classifier.joblib")
                mark_complete(directory, signature)
            self._final_models[key] = model
        return {"n_features": matrix.shape[1], "n_zero_train": int((matrix.getnnz(axis=1) == 0).sum()),
                "n_stopwords": len(self.stopwords)}

    def describe_tfidf_corpora(self):
        """Décrit les matrices avec le vocabulaire appris sur BTP uniquement.

        La transformation des corpus cibles est volontairement faite avec le
        vectoriseur BTP final : cette cellule documente la couverture lexicale
        réelle du transfert, sans jamais apprendre de vocabulaire ou d'IDF sur
        les cibles.
        """
        vectorizer, source_matrix = self.final_features()
        vocabulary_size = int(len(vectorizer.vocabulary_))
        rows = []
        for corpus in self.specs:
            matrix = source_matrix if corpus == "btp" else vectorizer.transform(self._texts(corpus)).tocsr()
            if corpus != "btp":
                self._target_features[corpus] = matrix
            n_docs, n_features = matrix.shape
            nonzero_by_document = matrix.getnnz(axis=1)
            denominator = int(n_docs) * int(n_features)
            rows.append({
                "corpus": corpus,
                "n_documents": int(n_docs),
                "n_features": int(n_features),
                "vocabulary_size_btp": vocabulary_size,
                "nnz": int(matrix.nnz),
                "density_pct": 100.0 * float(matrix.nnz) / denominator if denominator else 0.0,
                "mean_nonzero_features_per_document": float(nonzero_by_document.mean()) if n_docs else 0.0,
                "median_nonzero_features_per_document": float(np.median(nonzero_by_document)) if n_docs else 0.0,
                "max_nonzero_features_per_document": int(nonzero_by_document.max(initial=0)),
                "n_zero_tfidf": int((nonzero_by_document == 0).sum()),
                "zero_tfidf_pct": 100.0 * float((nonzero_by_document == 0).mean()) if n_docs else 0.0,
            })
        summary = pd.DataFrame(rows)
        summary.to_csv(self.root / "tfidf_corpus_statistics.csv", index=False)
        return summary

    def evaluate_target(self, corpus):
        if not hasattr(self, "best_model"):
            raise ValueError("Exécuter summarize_cv() avant l'évaluation cible")
        if corpus not in self.config["targets"]:
            raise ValueError(f"Corpus cible non configuré : {corpus}")
        self.fit_final()
        if corpus not in self._target_features:
            vectorizer, _ = self.final_features()
            self._target_features[corpus] = vectorizer.transform(self._texts(corpus)).tocsr()
        matrix = self._target_features[corpus]
        zeros = matrix.getnnz(axis=1) == 0
        print(f"{corpus} : {int(zeros.sum())}/{len(zeros)} unités à vecteur nul (conservées)", flush=True)
        target_root = self.root / "targets" / corpus
        result = {}
        for key, model in self._final_models.items():
            directory = target_root / "models" / key
            signature = fingerprint({"model": self.model_signature(key), "target": self.specs[corpus],
                                     "target_hash": self.file_hashes[corpus]})
            files = ["predictions.csv", "metrics.json", "classification_report.csv",
                     "confusion_matrix.csv", "confusion_matrix_normalized.csv"]
            if not self.restimate and cache_ready(directory, signature, files):
                scores = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                print(f"{corpus} / {key} : évaluation rechargée", flush=True)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "manifest.json").unlink(missing_ok=True)
                probs = aligned_probabilities(model, matrix)
                scores, report, counts, normalized = classification_results(self.metadata[corpus]["label_id"].to_numpy(), probs)
                scores.update({"n_eval": len(zeros), "n_zero_tfidf": int(zeros.sum())})
                report.to_csv(directory / "classification_report.csv")
                counts.to_csv(directory / "confusion_matrix.csv")
                normalized.to_csv(directory / "confusion_matrix_normalized.csv")
                prediction_frame(self.metadata[corpus], probs, zeros, corpus=corpus, model_key=key,
                                 label_col=self.specs[corpus].get("label_col", "pred_label")).to_csv(directory / "predictions.csv", index=False)
                write_json(directory / "metrics.json", scores)
                mark_complete(directory, signature)
            result[key] = {"scores": scores, "directory": directory}
        summary = pd.DataFrame([{"model": key, **record["scores"]} for key, record in result.items()])
        summary.to_csv(target_root / "all_models_test_metrics.csv", index=False)
        self.test_results[corpus] = summary
        return summary, result

    def summarize_cross_domain(self):
        missing = set(self.config["targets"]) - set(self.test_results)
        if missing:
            raise ValueError(f"Corpus non évalués : {sorted(missing)}")
        summary = self.cv_summary.copy()
        ba_columns = []
        for corpus, frame in self.test_results.items():
            summary = summary.merge(frame[["model", *METRICS]].rename(
                columns={metric: f"{metric}_{corpus}" for metric in METRICS}), on="model", validate="one_to_one")
            ba_columns.append(f"balanced_accuracy_{corpus}")
        summary["ba_ood_avg"] = summary[ba_columns].mean(axis=1)
        summary["ba_ood_worst"] = summary[ba_columns].min(axis=1)
        summary["selected_on_btp_cv"] = summary["model"].eq(self.best_model)
        summary.to_csv(self.root / "cross_domain_generalization.csv", index=False)
        return summary


def plot_cv_summary(summary: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4), sharey=True)
    colors = sns.color_palette("deep", len(summary))
    for ax, metric in zip(axes, METRICS):
        ax.bar(range(len(summary)), summary[f"mean_{metric}"], yerr=summary[f"std_{metric}"], capsize=4, color=colors)
        ax.set_xticks(range(len(summary)), [MODEL_NAMES[key] for key in summary["model"]], rotation=25, ha="right")
        ax.set_title(metric)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Score — moyenne ± écart-type entre folds")
    fig.suptitle("CV groupée BTP — TF-IDF")
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "cv_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def show_target_results(corpus: str, summary: pd.DataFrame, results: dict, best_model: str, output_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import Markdown, display

    display(Markdown(f"### {corpus} — meilleur modèle CV BTP : **{MODEL_NAMES[best_model]}**"))
    display(summary.style.format({metric: "{:.3f}" for metric in METRICS}))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, record in results.items():
        directory = record["directory"]
        display(Markdown(f"#### {MODEL_NAMES[key]} — résultats par classe"))
        report = pd.read_csv(directory / "classification_report.csv", index_col=0)
        display(report.loc[list(MACROS)].style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, filename, title, fmt in zip(axes,
                ("confusion_matrix.csv", "confusion_matrix_normalized.csv"),
                ("Effectifs", "Proportion par classe vraie"), (".0f", ".1%")):
            matrix = pd.read_csv(directory / filename, index_col=0)
            sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", ax=ax, vmin=0,
                        vmax=1 if "normalized" in filename else None)
            ax.set(title=title, xlabel="Prédit", ylabel="Vrai")
        fig.suptitle(f"{corpus} — {MODEL_NAMES[key]}")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_{key}.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
