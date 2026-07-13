"""Chargement et validation des configs contrastives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from safer_core.io import load_yaml
from safer_core.paths import TEXT_ROOT
from safer_core.text_columns import warn_if_prompt_enabled


@dataclass
class ContrastiveConfig:
    method_name: str
    dataset_path: Path
    text_col: str = "sentence"
    label_col: str = "pred_label"
    group_col: str = "accident_id"
    pred_ok_col: str = "pred_ok"
    output_dir: str = ""
    seed: int = 42
    backbone_name: str = "Qwen/Qwen3-Embedding-0.6B"
    max_seq_length: int = 256
    batch_size: int = 16
    eval_batch_size: int = 16
    encode_batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 2.0e-5
    val_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    use_prompt: bool = False
    # softtriple
    centers_per_class: int = 5
    softtriple_gamma: float = 0.1
    softtriple_lambda: float = 10.0
    softtriple_delta: float = 0.01
    softtriple_tau: float = 0.01
    center_max_similarity: float = 0.5
    center_min_distance: float = 0.3
    center_regularization_type: str = "none"
    export_effective_centers: bool = False
    effective_center_distance_threshold: float = 0.05
    effective_center_similarity_threshold: float = 0.995
    # backbone / projecteur (encodeur HF unifié)
    backbone_trainable: bool = False
    train_last_n_layers: Optional[int] = None
    cache_backbone_embeddings: bool = True
    use_projector: bool = True
    projection: str = "linear"
    hiddim: int = 128
    # supcon (HobbitLong / SupContrast)
    supcon_temperature: float = 0.07
    supcon_base_temperature: float = 0.07
    supcon_contrast_mode: str = "all"
    supcon_normalize_embeddings: bool = True
    # batch triplet (Sentence Transformers)
    batch_triplet_margin: Optional[float] = None
    triplet_log_diagnostics: bool = False
    triplet_diagnostics_every_steps: int = 50
    triplet_diagnostics_eps: float = 1e-6
    triplet_loss_type: str = "soft_margin"
    # distance (SupCon, SoftTriple, batch triplet)
    distance_metric: str = "euclidean"
    final_fit_full_data: bool = False
    selection_metric: str = "balanced_accuracy"
    n_folds: int = 1
    test_corpus: Optional[str] = None
    test_corpora: Optional[List[str]] = None
    test_dataset_path: Optional[Path] = None
    # post-évaluation classification (logistic sklearn)
    post_eval_enabled: bool = True
    post_eval_classifier: str = "logistic_regression"
    post_eval_class_weight: Optional[str] = None
    post_eval_oversampling: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def test_data_csv(self) -> Path:
        if self.test_dataset_path is not None:
            return self.test_dataset_path
        from safer_core.test_corpus import resolve_test_corpus

        return resolve_test_corpus(self.test_corpus).data_csv

    @property
    def resolved_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        return f"output/{self.method_name}"

    def test_corpora_list(self) -> List[str]:
        if self.test_corpora:
            return [str(c) for c in self.test_corpora]
        if self.test_corpus:
            return [str(self.test_corpus)]
        return ["metallurgie"]


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    block = raw.get(name)
    return block if isinstance(block, dict) else {}


def resolve_center_regularization_type(
    explicit: Optional[str],
    tau: float,
) -> str:
    """
    Rétrocompatibilité SoftTriple :
    - si center_regularization_type absent du YAML : tau <= 0 → none, tau > 0 → diversity
    - sinon : valeur explicite (none | merge_l21 | diversity)
    """
    if explicit is None or str(explicit).strip() == "":
        return "none" if float(tau) <= 0.0 else "diversity"
    reg = str(explicit).strip().lower()
    valid = {"none", "merge_l21", "diversity"}
    if reg not in valid:
        raise ValueError(
            f"center_regularization_type invalide : {explicit!r} (attendu : {sorted(valid)})"
        )
    return reg


def load_contrastive_config(
    method_name: str,
    config_path: str | Path | None = None,
    raw: Dict[str, Any] | None = None,
) -> ContrastiveConfig:
    path = Path(config_path) if config_path else TEXT_ROOT / f"configs/methods/{method_name}.yaml"
    if raw is None:
        raw = load_yaml(path)
    data = _section(raw, "data")
    model = _section(raw, "model")
    training = _section(raw, "training")
    post_eval = _section(raw, "post_eval")
    softtriple = _section(raw, "softtriple")
    supcon = _section(raw, "supcon")
    batch_triplet = _section(raw, "batch_triplet")

    flat = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    for block in (data, model, training, flat):
        pass

    def pick(*keys: str, default: Any = None, sources: tuple = ()) -> Any:
        for src in sources:
            if not isinstance(src, dict):
                continue
            for key in keys:
                if key in src:
                    return src[key]
        return default

    dataset_rel = pick(
        "dataset_path",
        default=f"dataset/data_btp.csv",
        sources=(data, raw),
    )
    use_prompt = bool(pick("use_prompt", default=False, sources=(raw, training)))

    _metric_default = "cosine" if method_name == "supcon" else "euclidean"

    cfg = ContrastiveConfig(
        method_name=str(pick("method_name", default=method_name, sources=(raw,))),
        dataset_path=TEXT_ROOT / str(dataset_rel),
        text_col=str(pick("text_col", default="sentence", sources=(data, raw))),
        label_col=str(pick("label_col", default="pred_label", sources=(data, raw))),
        group_col=str(pick("group_col", default="accident_id", sources=(data, raw))),
        pred_ok_col=str(pick("pred_ok_col", default="pred_ok", sources=(data, raw))),
        output_dir=str(pick("output_dir", default="", sources=(raw, training))),
        seed=int(pick("seed", default=42, sources=(training, raw))),
        backbone_name=str(
            pick("backbone_name", default="Qwen/Qwen3-Embedding-0.6B", sources=(model, raw))
        ),
        max_seq_length=int(pick("max_seq_length", default=256, sources=(model, training, raw))),
        batch_size=int(pick("batch_size", default=16, sources=(training, raw))),
        eval_batch_size=int(
            pick("eval_batch_size", default=16, sources=(training, raw))
        ),
        encode_batch_size=int(
            pick("encode_batch_size", default=16, sources=(training, raw))
        ),
        epochs=int(pick("epochs", default=30, sources=(training, raw))),
        learning_rate=float(pick("learning_rate", default=2.0e-5, sources=(training, raw))),
        val_ratio=float(pick("val_ratio", default=0.1, sources=(data, training, raw))),
        gradient_accumulation_steps=int(
            pick("gradient_accumulation_steps", default=1, sources=(training, raw))
        ),
        use_prompt=use_prompt,
        centers_per_class=int(
            pick("centers_per_class", default=5, sources=(softtriple, raw))
        ),
        softtriple_gamma=float(pick("gamma", default=0.1, sources=(softtriple,))),
        softtriple_lambda=float(
            pick("lambda", "la", default=10.0, sources=(softtriple,))
        ),
        softtriple_delta=float(pick("delta", default=0.01, sources=(softtriple,))),
        softtriple_tau=float(pick("tau", default=0.01, sources=(softtriple,))),
        center_max_similarity=float(
            pick("center_max_similarity", default=0.5, sources=(softtriple,))
        ),
        center_min_distance=float(
            pick("center_min_distance", default=0.3, sources=(softtriple,))
        ),
        supcon_temperature=float(
            pick("temperature", default=0.07, sources=(supcon,))
        ),
        supcon_base_temperature=float(
            pick(
                "base_temperature",
                default=float(pick("temperature", default=0.07, sources=(supcon,))),
                sources=(supcon,),
            )
        ),
        supcon_contrast_mode=str(
            pick("contrast_mode", default="all", sources=(supcon,))
        ),
        supcon_normalize_embeddings=bool(
            pick("normalize_embeddings", default=True, sources=(supcon,))
        ),
        batch_triplet_margin=(
            float(m)
            if (m := pick("triplet_margin", default=None, sources=(batch_triplet,))) is not None
            else None
        ),
        triplet_log_diagnostics=bool(
            pick("log_diagnostics", default=False, sources=(batch_triplet,))
        ),
        triplet_diagnostics_every_steps=int(
            pick("diagnostics_every_steps", default=50, sources=(batch_triplet,))
        ),
        triplet_diagnostics_eps=float(
            pick("diagnostics_eps", default=1e-6, sources=(batch_triplet,))
        ),
        triplet_loss_type=str(
            pick("loss_type", default="soft_margin", sources=(batch_triplet,))
        ),
        distance_metric=str(
            pick(
                "distance_metric",
                default=_metric_default,
                sources=(training, raw),
            )
        ),
        final_fit_full_data=bool(
            pick("final_fit_full_data", default=False, sources=(training, raw))
        ),
        center_regularization_type=resolve_center_regularization_type(
            pick("center_regularization_type", default=None, sources=(softtriple,)),
            float(pick("tau", default=0.01, sources=(softtriple,))),
        ),
        export_effective_centers=bool(
            pick("export_effective_centers", default=True, sources=(softtriple,))
        ),
        effective_center_distance_threshold=float(
            pick("effective_center_distance_threshold", default=0.05, sources=(softtriple,))
        ),
        effective_center_similarity_threshold=float(
            pick("effective_center_similarity_threshold", default=0.995, sources=(softtriple,))
        ),
        backbone_trainable=bool(
            pick("backbone_trainable", default=False, sources=(model, raw))
        ),
        train_last_n_layers=pick("train_last_n_layers", default=None, sources=(model, raw)),
        cache_backbone_embeddings=bool(
            pick("cache_backbone_embeddings", default=True, sources=(model, raw))
        ),
        use_projector=bool(
            pick("use_projector", default=True, sources=(model, raw))
        ),
        projection=str(
            pick("projection", default="linear", sources=(model, raw))
        ),
        hiddim=int(
            pick("hiddim", default=128, sources=(model, raw))
        ),
        selection_metric=str(
            pick("selection_metric", default="balanced_accuracy", sources=(raw, training))
        ),
        n_folds=int(pick("n_folds", default=1, sources=(raw, training))),
        test_corpus=(
            str(tc)
            if (tc := pick("test_corpus", default=None, sources=(data, raw, training)))
            else None
        ),
        test_corpora=(
            [str(c) for c in corpora]
            if (corpora := pick("test_corpora", default=None, sources=(data, raw, training)))
            else None
        ),
        test_dataset_path=(
            TEXT_ROOT / str(test_rel)
            if (test_rel := pick("test_dataset_path", default=None, sources=(data, raw, training)))
            else None
        ),
        post_eval_enabled=bool(
            pick("enabled", default=True, sources=(post_eval,))
        ),
        post_eval_classifier=str(
            pick("classifier", default="logistic_regression", sources=(post_eval, raw))
        ),
        post_eval_class_weight=pick("class_weight", default=None, sources=(post_eval, raw)),
        post_eval_oversampling=bool(
            pick("oversampling", default=False, sources=(post_eval, raw))
        ),
        extra={"raw": raw, "config_path": str(path)},
    )
    from contrastive_methods.config_validation import validate_contrastive_config

    cfg = validate_contrastive_config(cfg)
    import os

    env_corpus = os.environ.get("TEST_CORPUS")
    if env_corpus:
        cfg.test_corpus = str(env_corpus)
        cfg.test_corpora = None
        cfg.test_dataset_path = None
    env_corpora = os.environ.get("TEST_CORPORA")
    if env_corpora:
        cfg.test_corpora = [c.strip() for c in env_corpora.split(",") if c.strip()]
        cfg.test_corpus = None
        cfg.test_dataset_path = None
    return cfg


def load_contrastive_config_from_dict(
    method_name: str,
    raw: Dict[str, Any],
    *,
    config_path: str = "",
) -> ContrastiveConfig:
    """Charge une config à partir d'un dict déjà fusionné (tuning grid)."""
    cfg = load_contrastive_config(method_name, config_path=config_path or None, raw=raw)
    cfg.extra["raw"] = raw
    if config_path:
        cfg.extra["config_path"] = config_path
    return cfg


def merge_config_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne des overrides en notation pointée (ex. training.lr) dans base."""
    import copy

    merged = copy.deepcopy(base)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = merged
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return merged


def config_to_resolved_dict(cfg: ContrastiveConfig) -> Dict[str, Any]:
    raw = dict(cfg.extra.get("raw", {}))
    raw.update(
        {
            "method_name": cfg.method_name,
            "dataset_path": str(cfg.dataset_path.relative_to(TEXT_ROOT))
            if cfg.dataset_path.is_relative_to(TEXT_ROOT)
            else str(cfg.dataset_path),
            "text_col": cfg.text_col,
            "label_col": cfg.label_col,
            "group_col": cfg.group_col,
            "output_dir": cfg.resolved_output_dir,
            "seed": cfg.seed,
            "backbone_name": cfg.backbone_name,
            "max_seq_length": cfg.max_seq_length,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "val_ratio": cfg.val_ratio,
        }
    )
    return raw
