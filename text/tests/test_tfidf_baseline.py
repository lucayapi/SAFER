"""Contrôles du protocole lexical, des caches et du notebook 07c."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from macro_transfer import tfidf_baseline as baseline


@pytest.fixture
def config(tmp_path):
    rows = []
    words = ["préparation consigne", "machine protection", "chute contact", "blessure fracture"]
    for accident in range(8):
        for label_id, label in enumerate(baseline.MACROS):
            rows.append({"sentence": f"SALARIÉ {words[label_id]} groupe{accident}",
                         "pred_label": label, "pred_ok": True,
                         "accident_id": f"accident{accident}", "fact_id": f"f{label_id}"})
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    target = tmp_path / "target.csv"
    target_rows = pd.DataFrame(rows[:8])
    target_rows["sentence"] += " exclusifcible"
    target_rows.loc[0, "sentence"] = "SALARIÉ"
    target_rows.to_csv(target, index=False)
    stops = tmp_path / "stop_metier.txt"
    stops.write_text("# métier\nSALARIÉ\n\n", encoding="utf-8")
    return {
        "source": {"dataset_path": str(source)},
        "targets": {"metallurgie": {"dataset_path": str(target)}},
        "output_dir": str(tmp_path / "output"),
        "tfidf": {"analyzer": "word", "ngram_range": (1, 2), "min_df": 1,
                  "max_df": 1.0, "max_features": 50000, "sublinear_tf": True,
                  "norm": "l2", "lowercase": True, "strip_accents": None},
        "models": {
            "logistic_regression": {"C": 1.0, "class_weight": "balanced", "max_iter": 200},
            "random_forest": {"n_estimators": 5, "class_weight": "balanced"},
            "xgboost": {"n_estimators": 5, "max_depth": 2, "objective": "multi:softprob", "eval_metric": "mlogloss"},
        },
        "n_folds": 2, "seed": 42, "n_jobs": 1, "selection_metric": "balanced_accuracy",
        "use_stopwords": True, "stopwords_file": str(stops),
    }


def test_stopwords_ngrams_and_sparse_input(config):
    stops = baseline.normalize_stopwords(Path(config["stopwords_file"]), config["tfidf"])
    vectorizer, matrix = baseline.fit_vectorizer(
        ["SALARIÉ préparation consigne", "machine protection"], config["tfidf"], stops, "test")
    assert sparse.isspmatrix_csr(matrix)
    assert "salarié" not in vectorizer.vocabulary_
    assert "préparation consigne" in vectorizer.vocabulary_
    assert "préparation" in vectorizer.vocabulary_
    assert np.allclose(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel(), 1)
    bigrams = {**config["tfidf"], "ngram_range": (2, 2)}
    vectorizer, _ = baseline.fit_vectorizer(["préparation consigne"], bigrams, [], "test")
    assert set(vectorizer.vocabulary_) == {"préparation consigne"}


def test_group_and_vocabulary_leakage(config):
    experiment = baseline.TfidfExperiment(config)
    groups = experiment.metadata["btp"]["accident_id"].to_numpy()
    texts = np.asarray(experiment._texts("btp"))
    for fold_id, (train, val) in enumerate(experiment.splits):
        assert not set(groups[train]) & set(groups[val])
        vectorizer, x_train, x_val = experiment.fold_features(fold_id)
        assert sparse.issparse(x_train) and sparse.issparse(x_val)
        held_out_word = texts[val[0]].split()[-1]
        assert held_out_word not in vectorizer.vocabulary_
        assert "exclusifcible" not in vectorizer.vocabulary_
        assert "salarié" not in vectorizer.vocabulary_
    vectorizer, _ = experiment.final_features()
    vocabulary_before = dict(vectorizer.vocabulary_)
    idf_before = vectorizer.idf_.copy()
    vectorizer.transform(experiment._texts("metallurgie"))
    assert vectorizer.vocabulary_ == vocabulary_before
    np.testing.assert_array_equal(vectorizer.idf_, idf_before)
    assert "exclusifcible" not in vocabulary_before


def test_probabilities_reordered_and_four_class_metrics():
    class ReorderedEstimator:
        classes_ = np.array([2, 0, 3, 1])

        def predict_proba(self, matrix):
            return np.array([[0.1, 0.7, 0.05, 0.15]])

    probs = baseline.aligned_probabilities(ReorderedEstimator(), sparse.csr_matrix([[1]]))
    np.testing.assert_allclose(probs, [[0.7, 0.15, 0.1, 0.05]])
    scores, report, counts, normalized = baseline.classification_results([0], probs)
    assert scores["accuracy"] == 1
    assert scores["macro_f1"] == 0.25  # ensemble fixe de quatre classes
    assert counts.to_numpy().sum() == 1
    assert report.loc["A0", "support"] == 1
    assert normalized.loc["C"].sum() == 0


def test_three_models_exports_and_cache(config, monkeypatch):
    experiment = baseline.TfidfExperiment(config)
    for key in config["models"]:
        experiment.run_cv(key)
    summary = experiment.summarize_cv()
    best_before_test = experiment.best_model
    tfidf_statistics = experiment.describe_tfidf_corpora()
    assert tfidf_statistics.loc[0, "vocabulary_size_btp"] > 0
    assert set(tfidf_statistics["corpus"]) == {"btp", "metallurgie"}
    assert (experiment.root / "tfidf_corpus_statistics.csv").is_file()
    assert tfidf_statistics.loc[tfidf_statistics["corpus"].eq("metallurgie"), "n_zero_tfidf"].item() == 1
    experiment.fit_final()
    final_models = dict(experiment._final_models)
    result, records = experiment.evaluate_target("metallurgie")
    cross = experiment.summarize_cross_domain()
    assert len(result) == len(summary) == len(cross) == 3
    assert experiment.best_model == best_before_test
    assert experiment._final_models == final_models
    assert cross.selected_on_btp_cv.sum() == 1
    for key, record in records.items():
        directory = record["directory"]
        preds = pd.read_csv(directory / "predictions.csv")
        assert len(preds) == 8
        assert preds.zero_tfidf.sum() == 1
        np.testing.assert_allclose(preds[[f"prob_{m}" for m in baseline.MACROS]].sum(axis=1), 1, atol=1e-6)
        assert record["scores"]["accuracy"] == pytest.approx(preds.true_macro.eq(preds.pred_macro).mean())
        assert pd.read_csv(directory / "confusion_matrix.csv", index_col=0).to_numpy().sum() == len(preds)
        oof = pd.read_csv(experiment.root / "cv" / key / "oof_predictions.csv")
        assert len(oof) == 32 and oof.doc_id.nunique() == 32

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Le cache compatible doit éviter tout nouvel entraînement")

    monkeypatch.setattr(baseline, "build_classifier", unexpected_fit)
    reloaded = baseline.TfidfExperiment(config)
    for key in config["models"]:
        reloaded.run_cv(key)
    reloaded.summarize_cv()
    reloaded.evaluate_target("metallurgie")
    pd.testing.assert_frame_equal(summary, reloaded.cv_summary, check_exact=False, atol=1e-12)
    directory = records["logistic_regression"]["directory"]
    signature = json.loads((directory / "manifest.json").read_text())["signature"]
    (directory / "predictions.csv").unlink()
    assert not baseline.cache_ready(directory, signature, ["predictions.csv", "metrics.json"])


def test_cache_invalidation_is_scoped(config):
    original = baseline.TfidfExperiment(config)
    changed_model_cfg = copy.deepcopy(config)
    changed_model_cfg["models"]["logistic_regression"]["C"] = 0.1
    changed = baseline.TfidfExperiment(changed_model_cfg)
    assert changed.cv_signature("logistic_regression") != original.cv_signature("logistic_regression")
    assert changed.cv_signature("random_forest") == original.cv_signature("random_forest")
    new_target = pd.read_csv(config["targets"]["metallurgie"]["dataset_path"])
    new_target.loc[0, "sentence"] = "autre texte cible"
    new_target.to_csv(config["targets"]["metallurgie"]["dataset_path"], index=False)
    changed = baseline.TfidfExperiment(config)
    assert changed.file_hashes["metallurgie"] != original.file_hashes["metallurgie"]
    assert changed.cv_signature("logistic_regression") == original.cv_signature("logistic_regression")
    Path(config["stopwords_file"]).write_text("salarié\nprotection\n", encoding="utf-8")
    changed = baseline.TfidfExperiment(config)
    assert changed.feature_signature != original.feature_signature
    changed_tfidf = copy.deepcopy(config)
    changed_tfidf["tfidf"]["ngram_range"] = (2, 2)
    assert baseline.TfidfExperiment(changed_tfidf).feature_signature != changed.feature_signature
    source = pd.read_csv(config["source"]["dataset_path"])
    source.loc[0, "sentence"] += " modification"
    source.to_csv(config["source"]["dataset_path"], index=False)
    assert baseline.TfidfExperiment(config).feature_signature != changed.feature_signature


def test_filtering_and_empty_texts(config):
    path = config["source"]["dataset_path"]
    frame = pd.read_csv(path)
    frame.loc[0, "sentence"] = None
    frame.loc[1, "pred_ok"] = False
    frame.loc[2, "pred_label"] = "INVALID"
    frame.to_csv(path, index=False)
    metadata, summary = baseline.load_text_corpus(config["source"])
    assert summary["n_raw"] == 32 and summary["n_excluded"] == 2
    assert summary["n_empty_text"] == 1 and len(metadata) == 30


def test_explicit_failures(config):
    with pytest.raises(FileNotFoundError, match="stopwords"):
        baseline.normalize_stopwords(Path(config["stopwords_file"]).with_name("absent.txt"), config["tfidf"])
    with pytest.raises(ValueError, match="TF-IDF.*min_df"):
        baseline.fit_vectorizer(["salarié"], config["tfidf"], ["salarié"], "test")
    with pytest.raises(FileNotFoundError, match="Corpus"):
        baseline.load_text_corpus({"dataset_path": "absent.csv"})
    source = pd.read_csv(config["source"]["dataset_path"])
    source = source.loc[source.pred_label != "C"]
    source.to_csv(config["source"]["dataset_path"], index=False)
    with pytest.raises(ValueError, match="Fold.*classes absentes"):
        baseline.TfidfExperiment(config)


def test_notebook_structure_and_editable_defaults():
    import nbformat
    from scripts.build_notebook_07c_supervised_macro_tfidf_baseline import build_notebook

    notebook = build_notebook()
    nbformat.validate(notebook)
    code_cells = {cell.metadata.tags[0]: cell.source for cell in notebook.cells if cell.cell_type == "code"}
    for source in code_cells.values():
        compile(source, "07c", "exec")
    context = {"TEXT_ROOT": Path("text")}
    exec(code_cells["tfidf_parameters"], context)
    assert context["TFIDF_PARAMS"]["ngram_range"] == (1, 2)
    assert "\b" not in context["TFIDF_PARAMS"]["token_pattern"]  # pas de caractère backspace
    assert "préparation" in baseline.TfidfVectorizer(**context["TFIDF_PARAMS"]).build_analyzer()("préparation consigne")
    assert set(code_cells) >= {"parameters", "tfidf_parameters", "cv_logistic_regression",
                               "cv_random_forest", "cv_xgboost", "tfidf_statistics",
                               "target_evaluation", "cross_domain"}


def test_notebook_executes_top_to_bottom(config, tmp_path):
    import sys

    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    from scripts.build_notebook_07c_supervised_macro_tfidf_baseline import build_notebook

    notebook = build_notebook()
    column_defaults = {"text_col": "sentence", "label_col": "pred_label",
                       "group_col": "accident_id", "pred_ok_col": "pred_ok"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tag = cell.metadata.tags[0]
        if tag == "parameters":
            cell.source += (
                f"\nN_FOLDS = 2\nN_JOBS = 1\nTEST_CORPORA = ['metallurgie']"
                f"\nOUTPUT_DIR = Path({config['output_dir']!r})"
                f"\nSOURCE_CFG = {dict(column_defaults, **config['source'])!r}\n"
            )
        elif tag == "tfidf_parameters":
            cell.source += f"\nTFIDF_PARAMS = {config['tfidf']!r}\nSTOPWORDS_FILE = Path({config['stopwords_file']!r})"
        elif tag.startswith("model_"):
            key = tag.removeprefix("model_")
            cell.source = f"MODEL_REGISTRY[{key!r}] = {config['models'][key]!r}"
        elif tag == "load_data":
            targets = {key: {**column_defaults, **value} for key, value in config["targets"].items()}
            cell.source = cell.source.replace("CONFIG = {", f"TARGET_SPECS = {targets!r}\nCONFIG = {{")
    # Utilise l'environnement du test, pas un kernel Python global de la machine.
    manager = KernelManager(kernel_name="python3", connection_file=str(tmp_path / "kernel.json"))
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=180,
                            resources={"metadata": {"path": str(Path(__file__).resolve().parents[1])}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, tmp_path / "07c_smoke_executed.ipynb")
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
    root = Path(config["output_dir"])
    assert (root / "cross_domain_generalization.csv").is_file()
    assert len(list((root / "targets" / "metallurgie" / "figures").glob("*.png"))) == 3
