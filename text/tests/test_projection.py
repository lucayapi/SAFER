import torch

from scgm_text.projection import normalize_projection_name
from scgm_text.scgm_text_model import SCGMTextModel


def test_fc_alias_is_linear():
    assert normalize_projection_name("fc", None) == "linear"


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
