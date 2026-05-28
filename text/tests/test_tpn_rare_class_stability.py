from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_transfer.tpn_full_encoder import (
    TPNBatch,
    _BalancedMacroBatchSampler,
    build_target_pseudo_mask,
    compute_tpn_full_encoder_losses,
)


def _make_imbalanced_labels() -> np.ndarray:
    parts = [
        np.full(100, 0, dtype=np.int64),
        np.full(80, 1, dtype=np.int64),
        np.full(3, 2, dtype=np.int64),
        np.full(50, 3, dtype=np.int64),
    ]
    return np.concatenate(parts)


def test_balanced_sampler_with_replacement_contains_all_macros():
    labels = _make_imbalanced_labels()
    sampler = _BalancedMacroBatchSampler(
        labels,
        batch_size=16,
        drop_last=True,
        seed=123,
        min_per_macro=1,
        with_replacement=True,
    )
    batches = list(sampler)
    assert batches, "Le sampler doit produire au moins un batch"
    for batch in batches[:20]:
        batch_labels = labels[np.asarray(batch, dtype=np.int64)]
        for macro_id in range(4):
            assert int((batch_labels == macro_id).sum()) >= 1


def test_balanced_sampler_raises_if_macro_absent():
    labels = np.concatenate(
        [
            np.full(100, 0, dtype=np.int64),
            np.full(80, 1, dtype=np.int64),
            np.full(50, 3, dtype=np.int64),
        ]
    )
    with pytest.raises(ValueError, match="Macros absentes"):
        _BalancedMacroBatchSampler(
            labels,
            batch_size=16,
            drop_last=True,
            min_per_macro=1,
            with_replacement=True,
        )


def _q_with_rare_b_never_top1() -> torch.Tensor:
    q = torch.zeros(32, 4)
    q[:, 0] = 0.70
    q[:, 1] = 0.20
    q[:, 2] = 0.05
    q[:, 3] = 0.05
    q[8:12, 0] = 0.20
    q[8:12, 1] = 0.20
    q[8:12, 2] = 0.35
    q[8:12, 3] = 0.25
    return q


def test_per_class_topk_gives_nonzero_mass_to_rare_macro():
    q = _q_with_rare_b_never_top1()
    mask = build_target_pseudo_mask(
        q,
        strategy="per_class_topk",
        assignment_mode="soft",
        global_threshold=0.6,
        min_per_macro=4,
        min_confidence=0.25,
    )
    q_masked = q * mask
    mass_b = float(q_masked[:, 2].sum().item())
    assert mass_b > 0.0


def test_global_threshold_soft_mask_is_binary():
    q = torch.tensor(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.3, 0.3, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    mask = build_target_pseudo_mask(
        q,
        strategy="global_threshold",
        assignment_mode="soft",
        global_threshold=0.5,
        min_per_macro=1,
        min_confidence=0.25,
    )
    uniq = set(torch.unique(mask).tolist())
    assert uniq.issubset({0.0, 1.0})
    q_masked = q * mask
    keep = torch.tensor([1.0, 0.0], dtype=torch.float32).unsqueeze(1)
    expected = q * keep
    assert torch.allclose(q_masked, expected)
    assert not torch.allclose(q_masked, (q * q) * keep)


def test_global_threshold_can_drop_rare_macro():
    q = _q_with_rare_b_never_top1()
    mask = build_target_pseudo_mask(
        q,
        strategy="global_threshold",
        assignment_mode="hard",
        global_threshold=0.6,
        min_per_macro=4,
        min_confidence=0.25,
    )
    q_masked = q * mask
    mass_b = float(q_masked[:, 2].sum().item())
    assert mass_b == 0.0


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


def test_full_encoder_loss_still_has_grad():
    model = _DummyModel()
    src = TPNBatch(texts=["a0 txt", "a1 txt", "b txt", "c txt"], labels=torch.tensor([0, 1, 2, 3]))
    tgt = TPNBatch(texts=[f"target {i}" for i in range(8)])
    losses = compute_tpn_full_encoder_losses(
        model,
        src,
        tgt,
        tpn_cfg={
            "objective": "standard_tpn",
            "tau": 0.3,
            "distance_metric": "euclidean",
            "assignment_mode": "soft",
            "pseudo_label_strategy": "per_class_topk",
            "pseudo_label_threshold": 0.6,
            "pseudo_label_min_confidence": 0.25,
            "pseudo_label_min_per_macro": 2,
            "target_weight_st": 1.0,
        },
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 1.0, "ent": 0.01, "div": 0.01, "reg": 0.0},
        n_macros=4,
    )
    assert losses["loss_total"].requires_grad
    losses["loss_total"].backward()
    grad_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sum += float(p.grad.detach().abs().sum().item())
    assert grad_sum > 0.0


def test_standard_tpn_loss_only_uses_src_proto_kl():
    model = _DummyModel()
    src = TPNBatch(texts=["a0 txt", "a1 txt", "b txt", "c txt"], labels=torch.tensor([0, 1, 2, 3]))
    tgt = TPNBatch(texts=[f"target {i}" for i in range(8)])
    losses = compute_tpn_full_encoder_losses(
        model,
        src,
        tgt,
        tpn_cfg={
            "objective": "standard_tpn",
            "tau": 0.3,
            "distance_metric": "euclidean",
            "assignment_mode": "soft",
            "pseudo_label_strategy": "per_class_topk",
            "pseudo_label_min_confidence": 0.25,
            "pseudo_label_min_per_macro": 2,
            "target_weight_st": 1.0,
            "src_classifier_prototypes": "source",
        },
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 1.0, "ent": 999.0, "div": 999.0, "reg": 999.0},
        n_macros=4,
    )
    expected = losses["loss_src"] + losses["loss_proto"] + losses["loss_kl"]
    assert torch.allclose(losses["loss_total"], expected, atol=1e-6, rtol=1e-6)


def test_invalid_prototypes_excluded_from_proto_loss():
    model = _DummyModel()
    src = TPNBatch(texts=["a0 txt", "a1 txt", "b txt", "c txt"], labels=torch.tensor([0, 1, 2, 3]))
    tgt = TPNBatch(texts=[f"target {i}" for i in range(8)])
    losses = compute_tpn_full_encoder_losses(
        model,
        src,
        tgt,
        tpn_cfg={
            "objective": "standard_tpn",
            "tau": 0.3,
            "distance_metric": "euclidean",
            "assignment_mode": "soft",
            "pseudo_label_strategy": "global_threshold",
            "pseudo_label_threshold": 2.0,
            "target_weight_st": 1.0,
        },
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 1.0},
        n_macros=4,
    )
    assert losses["proto_valid_terms"] >= 0
    assert torch.isfinite(losses["loss_proto"])


def test_no_zero_prototype_in_kl_softmax():
    model = _DummyModel()
    src = TPNBatch(texts=["a0 txt", "a1 txt", "b txt", "c txt"], labels=torch.tensor([0, 1, 2, 3]))
    tgt = TPNBatch(texts=[f"target {i}" for i in range(8)])
    losses = compute_tpn_full_encoder_losses(
        model,
        src,
        tgt,
        tpn_cfg={
            "objective": "standard_tpn",
            "tau": 0.3,
            "distance_metric": "euclidean",
            "assignment_mode": "soft",
            "pseudo_label_strategy": "global_threshold",
            "pseudo_label_threshold": 2.0,
            "target_weight_st": 1.0,
        },
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 1.0},
        n_macros=4,
    )
    assert torch.isfinite(losses["loss_kl"])
