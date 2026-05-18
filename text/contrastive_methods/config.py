"""Chargement et validation des configs contrastives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

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
    warmup_ratio: float = 0.1
    val_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    use_prompt: bool = False
    use_fixed_instruction_prefix: bool = False
    fixed_instruction_prefix: Optional[str] = None
    use_contextual_prompt_with_summary: bool = False
    # softtriple
    centers_per_class: int = 5
    softtriple_gamma: float = 0.1
    softtriple_lambda: float = 10.0
    softtriple_delta: float = 0.01
    softtriple_tau: float = 0.01
    center_max_similarity: float = 0.5
    center_min_distance: float = 0.3
    # supcon
    supcon_temperature: float = 0.07
    supcon_normalize_embeddings: bool = True
    # distance (SupCon, SoftTriple, batch triplet)
    distance_metric: str = "euclidean"
    final_fit_full_data: bool = False
    selection_metric: str = "delta_macro_pct"
    n_folds: int = 1
    test_dataset_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def test_data_csv(self) -> Path:
        if self.test_dataset_path is not None:
            return self.test_dataset_path
        return TEXT_ROOT / "dataset/test/data_metallurgie.csv"

    @property
    def resolved_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        return f"resultats/{self.method_name}"


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    block = raw.get(name)
    return block if isinstance(block, dict) else {}


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
    warn_if_prompt_enabled(use_prompt)

    fixed_prefix = pick("fixed_instruction_prefix", default=None, sources=(training, raw))
    use_fixed = bool(
        pick(
            "use_fixed_instruction_prefix",
            default=False,
            sources=(training, raw),
        )
    )
    if use_fixed and fixed_prefix is None:
        fixed_prefix = (
            "Represent this workplace accident narrative for semantic retrieval "
            "and safety classification."
        )

    return ContrastiveConfig(
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
        warmup_ratio=float(pick("warmup_ratio", default=0.1, sources=(training, raw))),
        val_ratio=float(pick("val_ratio", default=0.1, sources=(data, training, raw))),
        gradient_accumulation_steps=int(
            pick("gradient_accumulation_steps", default=1, sources=(training, raw))
        ),
        gradient_checkpointing=bool(
            pick("gradient_checkpointing", default=False, sources=(training, raw))
        ),
        use_prompt=use_prompt,
        use_fixed_instruction_prefix=use_fixed,
        fixed_instruction_prefix=str(fixed_prefix) if fixed_prefix else None,
        use_contextual_prompt_with_summary=bool(
            pick(
                "use_contextual_prompt_with_summary",
                default=False,
                sources=(training, raw),
            )
        ),
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
        supcon_normalize_embeddings=bool(
            pick("normalize_embeddings", default=True, sources=(supcon,))
        ),
        distance_metric=str(
            pick(
                "distance_metric",
                default="euclidean",
                sources=(training, supcon, softtriple, batch_triplet, raw),
            )
        ),
        final_fit_full_data=bool(
            pick("final_fit_full_data", default=False, sources=(training, raw))
        ),
        selection_metric=str(
            pick("selection_metric", default="delta_macro_pct", sources=(raw, training))
        ),
        n_folds=int(pick("n_folds", default=1, sources=(raw, training))),
        test_dataset_path=(
            TEXT_ROOT / str(test_rel)
            if (test_rel := pick("test_dataset_path", default=None, sources=(data, raw, training)))
            else None
        ),
        extra={"raw": raw, "config_path": str(path)},
    )


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
