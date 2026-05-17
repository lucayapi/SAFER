import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from malt_text.malt_em_estep import hard_q_from_argmax, run_malt_em_estep
from malt_text.malt_em_model import MALTEMTargetModel
from scgm_text.sinkhorn_estep import sinkhorn_assign


def test_sinkhorn_assign_shape():
    scores = np.random.rand(20, 8).astype(np.float64) + 0.01
    q, z_hat, diag = sinkhorn_assign(scores, lmd=25.0)
    assert q.shape == (20, 8)
    assert z_hat.shape == (20,)
    assert "sinkhorn_n_active_z" in diag


def test_hard_q_one_hot():
    z = np.array([0, 2, 1, 3])
    q = hard_q_from_argmax(z, 4)
    assert q.shape == (4, 4)
    assert np.allclose(q.sum(axis=1), 1.0)


def test_dataset_return_index():
    class _MiniDS(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, i):
            return {"embedding": torch.zeros(4), "index": torch.tensor(i)}

    ds = _MiniDS()
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=lambda b: {
        "embedding": torch.stack([x["embedding"] for x in b]),
        "index": torch.stack([x["index"] for x in b]),
    })))
    assert batch["index"].shape == (2,)


def test_run_malt_em_estep_mini():
    n, d, k, c = 24, 8, 4, 4
    model = MALTEMTargetModel(input_dim=d, hiddim=8, num_classes=c, num_subclasses=k, projection="linear")
    x = torch.randn(n, d)
    ds = TensorDataset(x, torch.arange(n))
    loader = DataLoader(ds, batch_size=8)
    p0 = np.ones((n, c), dtype=np.float32) / c
    q, z_hat, diag = run_malt_em_estep(
        model=model,
        data_loader=loader,
        p0_all=p0,
        tau_z=0.1,
        tau_yz=0.1,
        sinkhorn_lmd=25.0,
        device=torch.device("cpu"),
        q_mode="hard",
    )
    assert q.shape == (n, k)
    assert z_hat.shape == (n,)
    assert diag["n_active_z"] >= 1.0
