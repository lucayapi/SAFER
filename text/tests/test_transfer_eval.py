"""Tests métriques transfert."""

from __future__ import annotations

import pandas as pd

from macro_transfer.transfer_eval import evaluate_transfer_classification


def test_transfer_eval_metrics():
    meta = pd.DataFrame(
        {
            "pred_label": ["A0", "A1", "B", "C", "A0"],
            "pred_ok": [True, True, True, True, True],
            "m_hat": ["A0", "A1", "B", "A0", "A0"],
            "q_conf": [0.9, 0.8, 0.7, 0.6, 0.55],
        }
    )
    m = evaluate_transfer_classification(meta)
    assert m["n_eval"] == 5
    assert m["accuracy"] == 0.8
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert "confusion" in m
