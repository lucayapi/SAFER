from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_transfer.tpn_full_encoder import TPNBatch, compute_tpn_full_encoder_losses


class _LowConfModel(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.device_obj = torch.device("cpu")
        self.param = nn.Parameter(torch.randn(dim))

    def encode_texts_batch(self, texts):
        n = len(texts)
        # Presque identiques -> softmax proche uniforme, faible confiance.
        x = self.param.unsqueeze(0).repeat(n, 1)
        noise = torch.linspace(0.0, 1e-4, n).unsqueeze(1)
        return F.normalize(x + noise, p=2, dim=-1)


def test_pseudo_label_threshold_filters_target():
    model = _LowConfModel()
    src = TPNBatch(texts=["a0", "a1", "b", "c"], labels=torch.tensor([0, 1, 2, 3]))
    tgt = TPNBatch(texts=["t1", "t2", "t3", "t4"])
    losses = compute_tpn_full_encoder_losses(
        model,
        src,
        tgt,
        tpn_cfg={
            "objective": "standard_tpn",
            "tau": 0.3,
            "distance_metric": "euclidean",
            "assignment_mode": "soft",
            "pseudo_label_threshold": 0.99,
            "target_weight_st": 1.0,
        },
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 1.0, "ent": 0.01, "div": 0.01, "reg": 0.0},
        n_macros=4,
    )
    assert losses["pseudo_coverage"] < 1.0
