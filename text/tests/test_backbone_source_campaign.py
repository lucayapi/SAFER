from __future__ import annotations

from pathlib import Path

from replication.runner import _model_config, load_replication_config, task_matrix
from safer_core.embedding_paths import embedding_export_path


def test_embedding_exports_are_backbone_isolated(tmp_path: Path) -> None:
    qwen = embedding_export_path("caou", backbone_name="Qwen/Qwen3-Embedding-0.6B", output_root=tmp_path)
    e5 = embedding_export_path("caou", backbone_name="intfloat/multilingual-e5-large", output_root=tmp_path)
    assert qwen != e5
    assert qwen == tmp_path / "qwen3" / "caou.csv"
    assert e5 == tmp_path / "multilingual_e5_large" / "caou.csv"


def test_e5_export_config_passes_query_prefix_to_encoder() -> None:
    import yaml
    from scripts.export_corpus_embeddings import build_contrastive_config

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/export_embeddings_multilingual_e5_large.yaml").read_text(encoding="utf-8"))
    encoder_cfg = build_contrastive_config(cfg, root / "dataset/data_btp.csv")
    assert encoder_cfg.input_prefix == "query: "


def test_factorial_recipe_has_16_conditions_and_no_source_target() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_replication_config(root / "output/replication_recipes/backbone_source_factorial.yaml")
    assert len(cfg["models"]) == 16
    assert len(task_matrix(cfg)) == 80
    for model_id in cfg["models"]:
        spec = _model_config(cfg, model_id)
        source = spec["source_corpus"]
        requested = spec.get("test_corpora", cfg["training"]["test_corpora"])
        targets = [str(x) for x in requested if str(x) != source]
        assert source in {"btp", "metallurgie"}
        assert source not in targets


def test_camembert_recipe_has_40_runs_and_separate_embedding_exports() -> None:
    import yaml
    from safer_core.embedding_paths import embedding_export_path

    root = Path(__file__).resolve().parents[1]
    config = load_replication_config(root / "output/replication_recipes/camembert_source_factorial.yaml")
    assert len(config["models"]) == 8
    assert len(task_matrix(config)) == 40
    assert embedding_export_path("btp", backbone_name="almanach/camembert-base", output_root="embeddings") == root / "embeddings/camembert_base/btp.csv"
    export_cfg = yaml.safe_load((root / "configs/export_embeddings_camembert.yaml").read_text(encoding="utf-8"))
    assert export_cfg["backbone_name"] == "almanach/camembert-base"
    assert export_cfg["input_prefix"] == ""
    for model_id in config["models"]:
        spec = _model_config(config, model_id)
        backbone = spec.get("backbone_name") or spec.get("overrides", {}).get("model", {}).get("backbone_name")
        assert backbone == "almanach/camembert-base"
        assert spec["source_corpus"] in {"btp", "metallurgie"}
        requested = spec.get("test_corpora", config["training"]["test_corpora"])
        assert spec["source_corpus"] not in [x for x in requested if x != spec["source_corpus"]]


def test_e5_recipes_keep_prefix_for_all_adapted_methods() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_replication_config(root / "output/replication_recipes/backbone_source_factorial.yaml")
    for model_id in cfg["models"]:
        if not model_id.startswith("multilingual_e5_large"):
            continue
        spec = _model_config(cfg, model_id)
        if spec["runner"] == "frozen":
            assert spec["input_prefix"] == "query: "
        else:
            assert spec["overrides"]["model"]["input_prefix"] == "query: "


def test_full_encoder_final_recipe_keeps_the_reduced_training_budget() -> None:
    from scripts.apply_backbone_source_tuning import _apply_scope_training_overrides

    cross_entropy: dict = {"overrides": {"training": {"epochs": 30, "use_amp": True, "lr_backbone": 2e-5}}}
    _apply_scope_training_overrides(cross_entropy, method="cross_entropy", scope=None)
    assert cross_entropy["overrides"]["training"] == {
        "epochs": 3,
        "use_amp": False,
        "lr_backbone": 2e-6,
    }

    supcon: dict = {"overrides": {"training": {"epochs": 30, "use_amp": True, "learning_rate": 2e-5}}}
    _apply_scope_training_overrides(supcon, method="supcon", scope=None)
    assert supcon["overrides"]["training"] == {
        "epochs": 3,
        "use_amp": False,
        "learning_rate": 2e-6,
    }

    _apply_scope_training_overrides(supcon, method="supcon", scope=2)
    assert supcon["overrides"]["training"] == {
        "epochs": 15,
        "use_amp": True,
        "learning_rate": 2e-5,
    }


def test_ood_common_target_summary_uses_both_target_frames() -> None:
    import numpy as np
    import pandas as pd
    from scripts.analyze_backbone_source_campaign import _multi_target_summary

    frame_a = pd.DataFrame({"accident_id": ["a", "b", "c", "d"], "true_macro": ["A0", "A1", "B", "C"], "pred_macro": ["A0", "A1", "B", "C"]})
    frame_b = pd.DataFrame({"accident_id": ["e", "f", "g", "h"], "true_macro": ["A0", "A1", "B", "C"], "pred_macro": ["A0", "A0", "B", "C"]})
    result = _multi_target_summary({"caou": [frame_a], "nicollin": [frame_b]}, rng=np.random.default_rng(4), n_boot=40, confidence=0.95)
    assert result["n_seeds"] == 1
    assert result["balanced_accuracy_mean"] == 0.875
    assert result["ci_low"] <= result["balanced_accuracy_mean"] <= result["ci_high"]
