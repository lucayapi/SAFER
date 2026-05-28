from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_transfer.tpn_full_encoder import train_tpn_full_encoder


class _SmokeModel(nn.Module):
    def __init__(self, dim: int = 10) -> None:
        super().__init__()
        self.device_obj = torch.device("cpu")
        self.base_method = "hf"
        self.backbone_name = "dummy"
        self.max_seq_length = 16
        self.pooling = "mean"
        self.emb = nn.Embedding(256, dim)
        self.lin = nn.Linear(dim, dim)

    def encode_texts_batch(self, texts):
        ids = []
        for t in texts:
            arr = [ord(c) % 256 for c in str(t)[:10]]
            if not arr:
                arr = [0]
            ids.append(arr + [0] * (10 - len(arr)))
        x = torch.tensor(ids, dtype=torch.long, device=self.device_obj)
        h = self.emb(x).mean(dim=1)
        h = self.lin(h)
        return F.normalize(h, p=2, dim=-1)


def test_full_encoder_smoke_cpu(tmp_path: Path):
    source_df = pd.DataFrame(
        {
            "sentence": [
                "src a0 1",
                "src a0 2",
                "src a1 1",
                "src a1 2",
                "src b 1",
                "src b 2",
                "src c 1",
                "src c 2",
            ],
            "pred_label": ["A0", "A0", "A1", "A1", "B", "B", "C", "C"],
            "pred_ok": [True] * 8,
            "accident_id": list(range(8)),
        }
    )
    target_df = pd.DataFrame(
        {
            "sentence": [f"tgt {i}" for i in range(8)],
            "pred_label": ["A0", "A1", "B", "C", "A0", "A1", "B", "C"],
            "pred_ok": [True] * 8,
            "accident_id": list(range(8)),
        }
    )
    model = _SmokeModel()
    out = train_tpn_full_encoder(
        model=model,
        source_df=source_df,
        target_df=target_df,
        out_dir=tmp_path / "run",
        tpn_cfg={
            "objective": "standard_tpn",
            "tau": 0.3,
            "distance_metric": "euclidean",
            "assignment_mode": "soft",
            "pseudo_label_threshold": 0.0,
            "target_weight_st": 1.0,
            "src_classifier_prototypes": "source",
        },
        full_cfg={
            "source_text_col": "sentence",
            "target_text_col": "sentence",
            "source_batch_size": 4,
            "target_batch_size": 4,
            "drop_last": True,
            "balanced_source_batches": True,
            "epochs": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "fp16": False,
            "bf16": False,
            "prototype_mode": "batch",
            "seed": 42,
        },
        loss_weights={"src": 1.0, "proto": 1.0, "kl": 1.0, "ent": 0.01, "div": 0.01, "reg": 0.0},
        label_col="pred_label",
        target_label_col="pred_label",
        pred_ok_col_target="pred_ok",
    )
    assert Path(out["training_log_path"]).is_file()
    assert (tmp_path / "run" / "transfer" / "metrics_tpn_full.json").is_file()
