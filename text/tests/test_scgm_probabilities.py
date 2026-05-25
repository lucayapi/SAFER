import torch

from scgm_text.scgm_text_model import SCGMTextModel


def _model():
    return SCGMTextModel(
        hiddim=16,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    )


def test_compute_latent_sinkhorn_scores_shape():
    model = _model()
    x = torch.randn(5, 16)
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    y = torch.zeros(5, 4)
    y[torch.arange(5), torch.randint(0, 4, (5,))] = 1.0
    score, _, _ = model.compute_latent_sinkhorn_scores(x, y, tau=0.1)
    assert score.shape == (5, 8)
