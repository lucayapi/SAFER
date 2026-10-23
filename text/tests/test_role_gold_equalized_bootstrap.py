import pandas as pd

from text.scripts.role_gold_compare_equalized import _bootstrap_target


def _prediction_frame(prefix: str, flip: bool = False) -> pd.DataFrame:
    rows = []
    roles = ["A0", "A1", "B", "C", "AUTRE"]
    for accident in range(10):
        for index, role in enumerate(roles):
            prediction = roles[(index + 1) % len(roles)] if flip else role
            rows.append({"accident_id": f"{prefix}{accident}",
                         "doc_id": f"{prefix}{accident}_{index}",
                         "pred_label": role,
                         "prediction": prediction})
    return pd.DataFrame(rows)


def test_paired_cluster_bootstrap_is_reproducible_and_matches_point_scores():
    # The same test narratives and gold labels are required across every model;
    # source prefixes therefore identify different fitted models, not different
    # document IDs or accident IDs.
    gold = _prediction_frame("a")
    worse = gold.copy()
    worse["prediction"] = _prediction_frame("a", flip=True)["prediction"].to_numpy()
    single = {"source_1": [gold, worse], "source_2": [gold.copy(), gold.copy()]}
    lodo = [gold.copy(), gold.copy(), gold.copy()]
    result1 = _bootstrap_target(single, lodo, n_boot=100, seed=101)
    result2 = _bootstrap_target(single, lodo, n_boot=100, seed=101)

    for metric in ("ba", "macro_f1"):
        values = result1[metric]
        repeat = result2[metric]
        assert abs(values["single_point"] - 0.75) < 1e-12
        assert values["lodo_point"] == 1.0
        assert values["delta_point"] == values["lodo_point"] - values["single_point"]
        assert (values["delta"] == repeat["delta"]).all()


def test_bootstrap_rejects_different_test_or_gold_rows():
    reference = _prediction_frame("a")
    changed = reference.copy()
    changed.loc[0, "pred_label"] = "AUTRE"
    try:
        _bootstrap_target({"source": [reference]}, [changed], n_boot=5, seed=1)
    except ValueError as exc:
        assert "tested documents" in str(exc)
    else:
        raise AssertionError("Expected mismatched GOLD predictions to be rejected")
