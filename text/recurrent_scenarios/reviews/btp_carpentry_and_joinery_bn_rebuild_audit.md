# Check of the rebuilt exact-BN results

Compared `runs/theme_discovery_rebuild/btp_carpentry_and_joinery` with `runs/theme_discovery_audit/btp_carpentry_and_joinery` after the new exact-BN job completed. This check does not cover paired factor reconstruction, whose new outputs are not yet present in the rebuild directory.

## Reproduced numerical reference

- The selected A0/A1/B/C partitions and the 417-by-42 accident-factor matrix are identical; the matrix Parquet files have the same SHA-256 hash.
- `global_bn_summary.csv`, `bn_bootstrap_edges.csv`, `bn_final_cpts.csv`, `bn_selected_parent_sets.csv`, `bn_conditional_contrasts.csv` and `scenario_threshold_summary.csv` have identical SHA-256 hashes. The selected network has 18 relationships, 65 Bernoulli parameters and BIC 9449.605927.
- The 80 recurrent scenario IDs, their supports, confidence and lift agree. Recomputed BN-implied scenario probabilities differ only at floating-point rounding scale (typically around 1e-17); the six manuscript examples have identical fitted counts to six decimals.

| Manuscript | Scenario ID | Reports | Confidence | Lift | BN-implied reports |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | SC_00197 | 23 | 0.958 | 4.757 | 18.103 |
| S2 | SC_00221 | 7 | 0.538 | 2.673 | 7.820 |
| S3 | SC_04076 | 5 | 0.833 | 4.455 | 0.617 |
| S4 | SC_00211 | 5 | 0.417 | 2.286 | 2.137 |
| S5 | SC_00173 | 5 | 1.000 | 4.964 | 1.791 |
| S6 | SC_00045 | 5 | 0.625 | 3.103 | 1.655 |

The `S1`–`S6` labels in the manuscript **must not** be equated with the six rows of the automatically generated `scenarios_article.csv`; that export chooses a different set of examples.

## Reporting differences to reconcile

- The earlier run used a bootstrap-frequency threshold of 0.60 for displayed stable relationships; the rebuild configuration uses 0.50. This changes the reported count from 9 to 12 stable relationships, although the full selected network and all bootstrap frequencies are unchanged. The manuscript currently describes the 0.60 convention. The differing threshold is also included in the exact-BN fingerprint, explaining why the manifest fingerprints differ while the matrix is identical.
- The new generated CSVs contain the new LLM factor labels. This changes text columns and figure labels, not the scenario definitions. One path-status field changes from `PARTIAL` to `COMPLETE` under the lower stability threshold.
- The automatically generated network and scenario diagrams truncate several labels; use neither image directly in the article without rebuilding its presentation. The manuscript's own TikZ scenario figure uses a separate, correctly mapped selection.
- Inspection of the supporting factual units showed that the manuscript's previous injury labels were overly specific for S1, S3 and S4. The narrative, TikZ figure, factor list and relationship labels were corrected to describe the observed groups more accurately. This is a semantic correction; none of the computed values changed.

## Full-sample reconstruction check

The one-replicate, 100%-report refit is now available in
`runs/theme_discovery_rebuild/btp_carpentry_and_joinery/identity_check`.
It uses all 417 reports, the selected role-specific clustering parameters,
UMAP seed 42, and matching threshold 0.50. The reference branch recovers all
18 archived relationships by construction, and the script verifies the
reference network and scenario counts against the archived matrix before
refitting factors.

The reconstructed branch does **not** reproduce the reference partition
exactly: 40 of 42 reference factors meet Jaccard 0.50, but only 3 of 42 have
Jaccard 1. The two factors below 0.50 are A0__T11 (0) and A0__T12 (0.360).
The refit finds 24 A0, 11 A1, 2 B and 5 C factors, compared with 25, 11, 2
and 4 reference factors. It recovers 13 of 18 reference relationships at the
matching threshold; the reconstructed network has 22 relationships. Of the
80 reference scenarios, 64 meet the article's recovery criterion after
reconstruction, while only 29 retain exactly the reference report count.
For example, S1 remains above the recovery threshold but its count changes
from 23 to 22. Thus a 1.000 recovery entry for a selected scenario is not
evidence of exact identity.

The previous focused diagnosis in
`runs/theme_discovery_audit/btp_carpentry_and_joinery/diagnostic_reconstruction/diagnosis.json`
found that refitting HDBSCAN on the archived UMAP coordinates reproduces the
archived C-role labels exactly, whereas recomputing UMAP on the same ordered
full data gave five C clusters rather than four. Repeated fits within that
diagnostic agreed with one another. This localises the observed partition
change to the UMAP stage but does not yet identify why the archived UMAP
coordinates and a new fit differ.

Do not interpret the 100%-report output as an estimated recovery frequency
or launch the 200-replicate reconstruction solely to validate exact
reproducibility. First establish the source of the UMAP difference or define
and document a computational environment in which the reference fit can be
reproduced. The generated `observed_results.md` says "500 replicates" in a
stale pilot message; the recorded rebuild configuration prespecifies 200.
