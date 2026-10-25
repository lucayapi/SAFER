"""Protocol checks for the five-role, narrative-only campaign."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import role_qwen3_lodo_campaign as campaign


def test_task_array_covers_each_target_seed_and_method_once():
    tasks = [campaign.spec(index) for index in range(54)]
    assert len(set(tasks)) == 54
    assert set(tasks) == {(target, seed, method) for target in campaign.CORPORA
                          for seed in campaign.SEEDS for method in campaign.METHODS}


def test_selection_uses_equal_source_domain_weight_and_stable_ties():
    target = campaign.CORPORA[0]
    sources = campaign.CORPORA[1:]
    rows = [(source, role) for source in sources for role in campaign.ROLES]
    frame = pd.DataFrame(rows, columns=["corpus", "pred_label"])
    frame["label_id"] = frame.pred_label.map(campaign.LABEL2ID)
    pred = frame.label_id.to_numpy().copy()
    mask = frame.corpus.eq(sources[0]).to_numpy()
    pred[mask] = (pred[mask] + 1) % 5
    score, by_source = campaign.macro_source_ba(frame, pred, target)
    assert score == 0.8
    assert by_source[sources[0]] == 0.0
    assert all(by_source[source] == 1.0 for source in sources[1:])
    best = {"source_macro_ba": 0.7, "C": 1.0, "epoch": 1}
    assert campaign.choose(best, {"source_macro_ba": 0.7, "C": 0.1, "epoch": 3})
    assert not campaign.choose(best, {"source_macro_ba": 0.7, "C": 1.0, "epoch": 2})


def test_frozen_vectors_keep_corpus_identity_with_repeated_doc_ids():
    frame = pd.DataFrame({"corpus": ["btp", "metallurgie", "btp"],
                          "doc_id": ["1", "1", "2"]})
    frozen = {"btp": pd.DataFrame(np.stack([np.ones(1024), np.ones(1024) * 2]),
                                   index=["1", "2"]),
              "metallurgie": pd.DataFrame(np.ones((1, 1024)) * 3, index=["1"])}
    matrix = campaign.vectors(frame, frozen)
    assert matrix.shape == (3, 1024)
    assert matrix[:, 0].tolist() == [1.0, 3.0, 2.0]


def test_stale_local_manifest_is_allowed_only_when_csvs_match(monkeypatch, capsys):
    real_digest = campaign.digest
    local_manifest = campaign.TEXT / "dataset/narrative_only_v1/narrative_only_manifest.json"

    def stale_manifest(path):
        return "0" * 64 if Path(path) == local_manifest else real_digest(path)

    monkeypatch.setattr(campaign, "digest", stale_manifest)
    frames, _, signatures = campaign.load_data(check_qwen=False)
    assert sum(len(frame) for frame in frames.values()) == 147487
    assert signatures["clean_manifest_sha256"] == real_digest(campaign.bundle() / "source_dataset_manifest.json")
    assert "Local dataset manifest differs" in capsys.readouterr().out

    btp_csv = campaign.TEXT / "dataset/narrative_only_v1/data_btp.csv"
    def changed_csv(path):
        return "1" * 64 if Path(path) == btp_csv else stale_manifest(path)

    monkeypatch.setattr(campaign, "digest", changed_csv)
    import pytest
    with pytest.raises(ValueError, match="Cleaned CSV differs from frozen LODO source"):
        campaign.load_data(check_qwen=False)
