"""Summarise paired scenario recovery across report-sampling fractions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FRACTIONS = (0.6, 0.7, 0.8, 0.9, 1.0)
MAIN_THRESHOLD = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=200)
    args = parser.parse_args()
    root = args.output_root.resolve()
    summary_rows = []
    control = {"fraction": 1.0, "expected_replicates": 1}

    for fraction in FRACTIONS:
        tag = f"{fraction:.1f}".replace(".", "p")
        folder = root / f"fraction_{tag}"
        design = json.loads((folder / "design.json").read_text(encoding="utf-8"))
        expected_replicates = 1 if fraction == 1.0 else args.replicates
        if design["fraction"] != fraction or design["replicates"] != expected_replicates:
            raise ValueError(f"Unexpected design in {folder}")
        scenarios = pd.read_csv(folder / f"scenario_recovery_tau_{MAIN_THRESHOLD:.1f}.csv")
        edges = pd.read_csv(folder / f"edge_recovery_tau_{MAIN_THRESHOLD:.1f}.csv")
        factors = pd.read_csv(folder / f"factor_recovery_tau_{MAIN_THRESHOLD:.1f}.csv")
        for object_type, table in (("scenario", scenarios), ("relationship", edges)):
            # The runner defines paired_difference as fixed minus reconstructed.
            # Reverse it here so a negative value means recovery fell after refit.
            difference = table["reconstructed_recovery"] - table["fixed_recovery"]
            summary_rows.append({
                "fraction": fraction,
                "replicates": expected_replicates,
                "object_type": object_type,
                "n_reference_objects": len(table),
                "median_fixed_recovery": table["fixed_recovery"].median(),
                "median_reconstructed_recovery": table["reconstructed_recovery"].median(),
                "median_reconstructed_minus_fixed": difference.median(),
                "objects_lower": int(difference.lt(0).sum()),
                "objects_tied": int(difference.eq(0).sum()),
                "objects_higher": int(difference.gt(0).sum()),
            })
        if fraction == 1.0:
            control.update({
                "n_reference_factors": len(factors),
                "factors_recovered_at_tau_0_5": int(factors["recovery"].ge(1.0).sum()),
                "n_reference_relationships": len(edges),
                "fixed_relationships_recovered": int(edges["fixed_recovery"].ge(1.0).sum()),
                "reconstructed_relationships_recovered": int(edges["reconstructed_recovery"].ge(1.0).sum()),
                "n_reference_scenarios": len(scenarios),
                "fixed_scenarios_recovered": int(scenarios["fixed_recovery"].ge(1.0).sum()),
                "reconstructed_scenarios_recovered": int(scenarios["reconstructed_recovery"].ge(1.0).sum()),
            })
            control["all_reference_objects_recovered_after_refit"] = all((
                control["factors_recovered_at_tau_0_5"] == control["n_reference_factors"],
                control["reconstructed_relationships_recovered"] == control["n_reference_relationships"],
                control["reconstructed_scenarios_recovered"] == control["n_reference_scenarios"],
            ))

    pd.DataFrame(summary_rows).to_csv(root / "fraction_curve_summary.csv", index=False)
    (root / "full_sample_identity_check.json").write_text(
        json.dumps(control, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"Full-sample identity check: {control['all_reference_objects_recovered_after_refit']}", flush=True)


if __name__ == "__main__":
    main()
