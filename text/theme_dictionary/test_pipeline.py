import json

import pandas as pd
import yaml

from theme_dictionary import run


def test_full_synthetic_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text('{"status":"complete"}', encoding="utf-8")
    pd.DataFrame({
        "accident_id": [f"A{i}" for i in range(10)],
        "A0_000": [1] * 7 + [0] * 3,
    }).to_csv(source / "accident_factor_matrix.csv", index=False)
    pd.DataFrame({
        "factor_id": ["A0_000"], "role": ["A0"],
        "llm_label": ["Construction work"], "llm_description": ["Work on a building site"],
    }).to_csv(source / "factor_dictionary.csv", index=False)
    config = {
        "data": {"source_run": str(source)},
        "output_dir": str(tmp_path / "outputs"),
        "model": {
            "max_theme_length": 4, "min_observed_count": 3,
            "overlap_values": [3, 1],
            "initial_scales": [0.25, 0.5, 1.0], "em_tolerance": 1e-8,
            "em_max_iterations": 100,
            "max_groups_per_component": 1000, "max_total_groups": 10000,
        },
        "parallel": {"workers": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["theme_dictionary.run", "--config", str(config_path)])
    run.main()
    paths = list((tmp_path / "outputs").iterdir())
    assert len(paths) == 1
    output = paths[0]
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    assert len(pd.read_csv(output / "accident_partitions_k3.csv")) == 10
    assert len(pd.read_csv(output / "accident_partitions_k1.csv")) == 10
    assert len(pd.read_csv(output / "theme_catalog_k3.csv")) == 1
    assert len(pd.read_csv(output / "theme_counts_by_size_roles_k3.csv")) == 1
    assert len(pd.read_csv(output / "basket_fit_k3.csv")) == 2
    assert json.loads((output / "diagnostics_k3.json").read_text())["all_em_starts_converged"]
    assert len(pd.read_csv(output / "method_comparison.csv")) == 3
    assert "Methods compared" in (output / "results_report.md").read_text(encoding="utf-8")
    assert json.loads((output / "summary_k3.json").read_text())["overlap_index"] == 0
    run.main()  # Completed checkpoints are reused on re-submission.
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    for k in (3, 1):
        monkeypatch.setattr("sys.argv", [
            "theme_dictionary.run", "--config", str(config_path), "--k", str(k)
        ])
        run.main()
        separate = output / f"k{k}"
        manifest = json.loads((separate / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["overlap_values"] == [k]
        assert (separate / f"theme_catalog_k{k}.csv").exists()
        assert not (separate / f"theme_catalog_k{1 if k == 3 else 3}.csv").exists()
