"""Tests statistiques d'accord d'annotation (IAA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from annotation.agreement_stats import (
    agreement_metrics_from_paired,
    bootstrap_kappa_ci,
    build_agreement_artifacts,
    build_agreement_table,
    confusion_matrix_from_paired,
    corpus_key_from_run_id,
    disagreement_subset,
    export_agreement_latex,
    export_confusion_latex,
    observed_agreement,
    pair_labels,
    role_counts_in_sample,
    cohen_kappa,
)


def test_corpus_key_from_run_id():
    assert corpus_key_from_run_id("run_all_btp") == "btp"
    assert corpus_key_from_run_id("run_partial_nicollin") == "nicollin"
    assert corpus_key_from_run_id("other") is None


def test_pair_labels_inner_join():
    df_all = pd.DataFrame(
        [
            {"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True},
            {"accident_id": "A1", "fact_id": 2, "pred_label": "B", "pred_ok": True},
            {"accident_id": "A2", "fact_id": 1, "pred_label": "C", "pred_ok": True},
        ]
    )
    df_partial = pd.DataFrame(
        [
            {"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True},
            {"accident_id": "A1", "fact_id": 2, "pred_label": "A1", "pred_ok": True},
        ]
    )
    paired = pair_labels(df_all, df_partial)
    assert len(paired) == 2
    assert set(paired["fact_id"].astype(str)) == {"1", "2"}


def test_identical_labels_perfect_agreement():
    y = ["A0", "A1", "B", "C"] * 5
    assert observed_agreement(y, y) == pytest.approx(1.0)
    assert cohen_kappa(y, y) == pytest.approx(1.0)
    paired = pd.DataFrame(
        {
            "accident_id": [f"a{i}" for i in range(len(y))],
            "fact_id": list(range(len(y))),
            "label_all": y,
            "label_partial": y,
        }
    )
    metrics = agreement_metrics_from_paired(paired, corpus="Test", n_boot=50, seed=0)
    assert metrics["observed_agreement"] == pytest.approx(1.0)
    assert metrics["kappa"] == pytest.approx(1.0)


def test_random_labels_low_kappa():
    rng = np.random.RandomState(0)
    labels = np.array(["A0", "A1", "B", "C"])
    y1 = rng.choice(labels, size=200)
    y2 = rng.choice(labels, size=200)
    assert cohen_kappa(y1, y2) < 0.2


def test_bootstrap_kappa_ci_narrative_level():
    y1 = ["A0", "A1", "B", "C"] * 3
    y2 = ["A0", "A1", "B", "C"] * 3
    accident_ids = [f"n{i // 2}" for i in range(len(y1))]
    lo, hi = bootstrap_kappa_ci(
        y1, y2, accident_ids=accident_ids, n_boot=100, seed=0
    )
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= hi


def test_agreement_metrics_requires_accident_id():
    paired = pd.DataFrame(
        {
            "fact_id": [1],
            "label_all": ["A0"],
            "label_partial": ["A0"],
        }
    )
    with pytest.raises(KeyError, match="accident_id"):
        agreement_metrics_from_paired(paired, corpus="Test", n_boot=10, seed=0)


def test_disagreement_subset_b_c():
    paired = pd.DataFrame(
        {
            "accident_id": ["a0", "a1", "a2", "a3"],
            "fact_id": [1, 1, 1, 1],
            "sentence": ["s0", "s1", "s2", "s3"],
            "label_all": ["B", "C", "B", "A0"],
            "label_partial": ["C", "B", "B", "A0"],
        }
    )
    bc = disagreement_subset(paired, labels=("B", "C"))
    assert len(bc) == 2
    assert set(bc["disagreement"]) == {"B → C", "C → B"}
    all_dd = disagreement_subset(paired)
    assert len(all_dd) == 2
    paired = pd.DataFrame(
        {
            "accident_id": ["a0", "a1", "a2", "a3", "a4"],
            "fact_id": [1, 1, 1, 1, 1],
            "label_all": ["A0", "A0", "B", "C", "A1"],
            "label_partial": ["A0", "A1", "B", "C", "A1"],
        }
    )
    all_dd = disagreement_subset(paired)
    assert len(all_dd) == 1
    assert all_dd["disagreement"].iloc[0] == "A0 → A1"
    counts = role_counts_in_sample(paired)
    assert counts == {"n_A0": 2, "n_A1": 1, "n_B": 1, "n_C": 1}
    metrics = agreement_metrics_from_paired(paired, corpus="Test", n_boot=20, seed=0)
    assert metrics["n_A0"] == 2
    assert metrics["n_B"] == 1
    cm = confusion_matrix_from_paired(paired)
    assert cm.loc["A0", "A0"] == 1
    assert cm.loc["A0", "A1"] == 1
    assert cm.loc["A1", "A1"] == 1
    assert int(cm.to_numpy().sum()) == 5


def test_disagreement_subset_all_labels():
    paired = pd.DataFrame(
        {
            "accident_id": ["a0", "a1", "a2", "a3"],
            "fact_id": [1, 1, 1, 1],
            "label_all": ["A0", "A1", "B", "C"],
            "label_partial": ["A1", "A1", "C", "A0"],
        }
    )
    dd = disagreement_subset(paired)
    assert len(dd) == 3
    assert set(dd["disagreement"]) == {"A0 → A1", "B → C", "C → A0"}
    bc_only = disagreement_subset(paired, labels=("B", "C"))
    assert len(bc_only) == 1
    assert bc_only["disagreement"].iloc[0] == "B → C"


def test_export_agreement_latex_contains_kappa_and_overall(tmp_path):
    pairs = []
    for corpus_key, corpus_name, labels_a, labels_b in [
        (
            "btp",
            "Construction",
            ["A0", "A0", "B", "C"],
            ["A0", "A1", "B", "C"],
        ),
    ]:
        all_path = tmp_path / f"{corpus_key}_all.xlsx"
        partial_path = tmp_path / f"{corpus_key}_partial.xlsx"
        pd.DataFrame(
            {
                "accident_id": ["a0", "a1", "a2", "a3"],
                "fact_id": [1, 1, 1, 1],
                "pred_label": labels_a,
                "pred_ok": True,
            }
        ).to_excel(all_path, index=False)
        pd.DataFrame(
            {
                "accident_id": ["a0", "a1", "a2", "a3"],
                "fact_id": [1, 1, 1, 1],
                "pred_label": labels_b,
                "pred_ok": True,
            }
        ).to_excel(partial_path, index=False)
        pairs.append(
            {
                "corpus_key": corpus_key,
                "corpus": corpus_name,
                "all_path": all_path,
                "partial_path": partial_path,
            }
        )

    table, confusion, paired = build_agreement_artifacts(
        pairs, n_boot=30, seed=1, include_overall=True
    )
    assert "Overall" in set(table["corpus"])
    assert "n_A0" in table.columns
    assert int(table.loc[table["corpus"] == "Construction", "n_A0"].iloc[0]) == 2
    assert "Construction" in confusion
    assert confusion["Construction"].shape == (4, 4)
    assert "Construction" in paired
    bc = disagreement_subset(paired["Construction"], labels=("B", "C"))
    # labels_a/b in fixture: A0,A0,B,C vs A0,A1,B,C → no B↔C swap
    assert bc.empty
    all_dd = disagreement_subset(paired["Construction"])
    assert len(all_dd) == 1
    assert all_dd["disagreement"].iloc[0] == "A0 → A1"
    tex = export_agreement_latex(table)
    assert r"Cohen's $\kappa$" in tex
    assert "Overall" in tex
    assert "Construction" in tex
    assert r"$n_{\mathrm{A0}}$" in tex
    cm_tex = export_confusion_latex(confusion["Construction"], corpus="Construction")
    assert "Confusion matrix" in cm_tex
    assert "A0" in cm_tex
