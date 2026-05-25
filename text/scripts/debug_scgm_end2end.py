"""Smoke test: SCGM end2end forward, loss, backward, backbone gradients."""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scgm_text.scgm_text_model import SCGMTextModel
from scgm_text.training_diagnostics import grad_norm_by_group, print_end2end_startup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hiddim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    model = SCGMTextModel(
        hiddim=args.hiddim,
        num_classes=4,
        num_subclasses=8,
        backbone_model_name_or_path="__test_dummy__",
        projection="linear",
    ).to(device)
    print_end2end_startup(model)

    batch_size = args.batch_size
    batch = {
        "input_ids": torch.randint(1, 50, (batch_size, 12), device=device),
        "attention_mask": torch.ones(batch_size, 12, dtype=torch.long, device=device),
        "label_ids": torch.randint(0, 4, (batch_size,), device=device),
        "indices": torch.arange(batch_size, device=device),
    }
    q = torch.zeros(batch_size, 8, device=device)
    q[torch.arange(batch_size), torch.randint(0, 8, (batch_size,), device=device)] = 1.0
    y = torch.zeros(batch_size, 4, device=device)
    y[torch.arange(batch_size), batch["label_ids"]] = 1.0

    model.train()
    features = model(batch)
    loss, *_ = model.loss(features, q, y, tau=0.1, alpha=0.5)
    print(f"loss={float(loss):.4f}", flush=True)
    loss.backward()
    norms = grad_norm_by_group(model)
    for name, val in norms.items():
        print(f"[SCGM DEBUG] grad_norm_{name}={val:.6e}", flush=True)
    assert norms["backbone"] > 0.0, "grad_norm_backbone must be > 0"
    assert norms["projector"] > 0.0, "grad_norm_projector must be > 0"
    assert norms["head"] > 0.0, "grad_norm_head must be > 0"
    print("[SCGM DEBUG] end2end gradient check OK", flush=True)


if __name__ == "__main__":
    main()
