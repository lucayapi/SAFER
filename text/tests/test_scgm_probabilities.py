import numpy as np
import torch

from scgm_text.scgm_embedding_model import SCGMEmbeddingNet


def _model(n_class=4, n_sub=8, hiddim=16, projection="linear"):
    return SCGMEmbeddingNet(
        input_dim=32,
        hiddim=hiddim,
        num_classes=n_class,
        num_subclasses=n_sub,
        projection=projection,
    )


def test_pred_probabilities_sum_to_one():
    model = _model()
    features = model(torch.randn(12, 32))
    tau = 0.1
    prob_y_x, prob_z_x, prob_y_z = model.pred(features, tau)
    assert prob_y_x.shape == (12, 4)
    assert prob_z_x.shape == (12, 8)
    assert prob_y_z.shape == (8, 4)
    np.testing.assert_allclose(prob_y_x.sum(dim=1).detach().numpy(), 1.0, rtol=1e-4)
    np.testing.assert_allclose(prob_z_x.sum(dim=1).detach().numpy(), 1.0, rtol=1e-4)
    np.testing.assert_allclose(prob_y_z.sum(dim=1).detach().numpy(), 1.0, rtol=1e-4)


def test_sinkhorn_score_shape_and_differs_from_macro_margin():
    model = _model()
    features = model(torch.randn(10, 32))
    y = torch.zeros(10, 4)
    y.scatter_(1, torch.randint(0, 4, (10, 1)), 1.0)
    tau = 0.1
    score, _, prob_z = model.compute_latent_sinkhorn_scores(features, y, tau)
    prob_y_x, _, _ = model.pred(features, tau)
    assert score.shape == (10, 8)
    assert prob_z.shape == (10, 8)
    assert prob_y_x.shape == (10, 4)
    assert not torch.allclose(score.sum(1), prob_y_x.sum(1), atol=1e-3)
