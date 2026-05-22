"""Tests diagnostics compression intra-macro."""

from __future__ import annotations

import numpy as np

from macro_transfer.macro_compression import compute_macro_compression_diagnostics


def test_compression_ratio_computed():
    rng = np.random.default_rng(0)
    n = 40
    d = 8
    h_init = rng.standard_normal((n, d))
    labels = ["A0"] * 20 + ["B"] * 20
    h_adapt = h_init.copy()
    h_adapt[:20] = h_init[:20].mean(axis=0)  # A0 compressé vers le centroïde

    df = compute_macro_compression_diagnostics(h_init, h_adapt, labels, macros=["A0", "B"])
    a0 = df.loc[df["macro"] == "A0"].iloc[0]
    b_row = df.loc[df["macro"] == "B"].iloc[0]
    assert a0["n_units"] == 20
    assert a0["W_init"] > 0
    assert a0["W_adapt"] < a0["W_init"]
    assert a0["compression_ratio"] < 1.0
    assert np.isclose(b_row["compression_ratio"], 1.0, rtol=0.01)
