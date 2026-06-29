import torch

from scgm_text.projection import build_embedding_projector, normalize_projection_name
from scgm_text.scgm_text_model import SCGMTextModel


def test_fc_alias_is_linear():
    assert normalize_projection_name("fc", None) == "linear"


def test_ln_gelu_and_residual_aliases():
    assert normalize_projection_name("ln_gelu", None) == "ln_gelu"
    assert normalize_projection_name("residual", None) == "residual"


def test_ln_gelu_projector_output_dim():
    proj = build_embedding_projector("ln_gelu", 64, 24, proj_hidden=32)
    assert proj(torch.randn(3, 64)).shape == (3, 24)


def test_residual_projector_output_dim():
    proj = build_embedding_projector(
        "residual", 64, 24, proj_bottleneck=16, proj_alpha=0.1
    )
    assert proj(torch.randn(3, 64)).shape == (3, 24)


def test_mlp_projector_changes_dim():
    model = SCGMTextModel(
        hiddim=16,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="mlp",
    )
    batch = {
        "input_ids": torch.randint(1, 50, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    out = model(batch)
    assert out.shape == (2, 16)
