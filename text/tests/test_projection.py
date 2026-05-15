import pytest
import torch

from scgm_text.projection import build_embedding_projector, normalize_projection_name
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet


def test_projection_output_dims():
    x = torch.randn(4, 32)
    for name, hiddim in [("linear", 16), ("mlp", 16)]:
        proj = build_embedding_projector(name, 32, hiddim)
        out = proj(x)
        assert out.shape == (4, hiddim)

    proj_id = build_embedding_projector("identity", 32, 32)
    assert proj_id(x).shape == (4, 32)

    with pytest.raises(ValueError):
        build_embedding_projector("identity", 32, 16)


def test_model_loss_components():
    model = SCGMEmbeddingNet(32, 16, 4, 8, projection="mlp")
    features = model(torch.randn(6, 32))
    q = torch.zeros(6, 8)
    q[torch.arange(6), torch.randint(0, 8, (6,))] = 1.0
    y = torch.zeros(6, 4)
    y[torch.arange(6), torch.randint(0, 4, (6,))] = 1.0
    out = model.loss(features, q, y, tau=0.1, alpha=0.5)
    assert len(out) == 7
