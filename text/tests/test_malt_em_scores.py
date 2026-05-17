import numpy as np
import torch

from malt_text.malt_em_estep import compute_malt_em_scores, compute_soft_macro_compatibility


def test_soft_macro_compatibility_shape():
    p0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    prob_y_z = torch.tensor(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    psi = compute_soft_macro_compatibility(p0, prob_y_z)
    assert psi.shape == (2, 4)
    assert torch.all(psi > 0)


def test_one_hot_p0_matches_classic_score():
    n, k, c = 5, 4, 4
    prob_z_x = torch.softmax(torch.randn(n, k), dim=1)
    prob_y_z = torch.softmax(torch.randn(k, c), dim=1)
    y_ids = torch.tensor([0, 1, 2, 3, 0])
    p0 = torch.zeros(n, c)
    p0[torch.arange(n), y_ids] = 1.0
    scores_soft, _ = compute_malt_em_scores(prob_z_x, prob_y_z, p0)
    scores_classic = prob_z_x * prob_y_z[:, y_ids].transpose(0, 1)
    assert torch.allclose(scores_soft, scores_classic, atol=1e-4)


def test_scores_positive_finite():
    p0 = torch.softmax(torch.randn(8, 4), dim=1)
    prob_z_x = torch.softmax(torch.randn(8, 32), dim=1)
    prob_y_z = torch.softmax(torch.randn(32, 4), dim=1)
    scores, log_scores = compute_malt_em_scores(prob_z_x, prob_y_z, p0)
    assert scores.shape == (8, 32)
    assert torch.isfinite(scores).all()
    assert (scores > 0).all()
