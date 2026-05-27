from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_transfer.tpn_full_encoder import TPNBatch, compute_tpn_full_encoder_losses


class _DummyModel(nn.Module):
    def __init__(self, dim: int = 12) -> None:
        super().__init__()
        self.device_obj = torch.device("cpu")
        self.embed = nn.Embedding(256, dim)
        self.proj = nn.Linear(dim, dim)

    def encode_texts_batch(self, texts):
        ids = []
        for t in texts:
            arr = [ord(c) % 256 for c in str(t)[:8]]
            if not arr:
                arr = [0]
            ids.append(arr + [0] * (8 - len(arr)))
        x = torch.tensor(ids, dtype=torch.long, device=self.device_obj)
        h = self.embed(x).mean(dim=1)
        h = self.proj(h)
        return F.normalize(h, p=2, dim=-1)


def test_tpn_full_loss_has_grad():
    model = _DummyModel()
    src = TPNBatch(texts=["a0 txt", "a1 txt", "b txt", "c txt"], labels=torch.tensor([0, 1, 2, 3]))
    tgt = TPNBatch(texts=["target 1", "target 2", "target 3", "target 4"])
    losses = compute_tpn_full_encoder_losses(
        model,
        src,
        tgt,
        tpn_cfg={
            "tau": 0.3,
            "distance_metric": "euclidean",
            "assignment_mode": "soft",
            "pseudo_label_threshold": 0.0,
            "target_weight_st": 1.0,
            "src_classifier_prototypes": "source",
        },
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 1.0, "ent": 0.01, "div": 0.01, "preserve": 0.0},
        n_macros=4,
    )
    assert losses["loss_total"].requires_grad
    losses["loss_total"].backward()
    grad_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sum += float(p.grad.detach().abs().sum().item())
    assert grad_sum > 0.0
