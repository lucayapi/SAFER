import torch

from malt_text.malt_em_losses import MALTEMLossComputer


def test_em_loss_finite():
    n, k, c = 16, 8, 4
    p0 = torch.softmax(torch.randn(n, c), dim=1)
    q = torch.softmax(torch.randn(n, k), dim=1)
    prob_z_x = torch.softmax(torch.randn(n, k), dim=1)
    prob_y_z = torch.softmax(torch.randn(k, c), dim=1)
    prob_y_x = prob_z_x @ prob_y_z
    computer = MALTEMLossComputer()
    out = computer.forward(
        p0=p0,
        q=q,
        prob_z_x=prob_z_x,
        prob_y_z=prob_y_z,
        prob_y_x=prob_y_x,
        mu_target=torch.randn(c, 16),
        mu_source=torch.randn(c, 16),
        nu=torch.randn(k, 16),
    )
    assert torch.isfinite(out.loss_total)
    assert torch.isfinite(out.loss_em)
    assert torch.isfinite(out.loss_z)
    assert torch.isfinite(out.loss_yz)
    assert abs(out.loss_em.item() - (out.loss_z.item() + out.loss_yz.item())) < 1e-5
