# Prioritized manuscript review: carpentry and joinery application

Date: 29 September 2026

## Scope and audit method

This review covers the accessible chapter sources: applications.tex and
text/recurrent_scenarios/bn_explain.tex, plus the resolved carpentry run and
its Bayesian-network outputs.

The manuscript master file, bibliography, preamble, appendices, and most
manuscript figure assets were not present in the reviewed workspace. The
review therefore does not certify a complete LaTeX build or bibliography. It
does verify internal references in the accessible TeX sources, the availability
of graphics at the paths used by applications.tex, and the main reported
numbers against the saved carpentry run.

The numerical audit found agreement for the main values: 417 accidents, 42
factors, 4,961 evaluated parent sets, 18 full-sample edges, 5,328 candidate
scenarios, 905 observed candidates, and 80 closed scenarios at n >= 5. The
run records 500 bootstrap resamples and 9 of 18 full-sample edges with
selection frequency >= 0.60.

## Overall assessment

The chapter has a strong analytical architecture. It separates empirical
recurrence, association-rule measures, direct conditional dependence,
bootstrap stability, and BN-implied recurrence. The role ordering makes the
exact BIC optimization claim transparent, and the repeated non-causal
qualification is appropriate.

The most likely reviewer concerns are reproducibility of the compiled
manuscript, the operational rule for sparse CPTs, the completeness of
stability reporting, transparency of selection choices, and the evidence
offered for sensitivity to the recurrence threshold.

## P0 - correct before circulation or submission

### P0.1 Undefined method cross-reference

Evidence: applications.tex, lines 513-514, refers to
sec:chap4_global_bn_learning. No such label occurs in the reviewed source set.
The exact-learning subsection in bn_explain.tex is labelled
sec:chap4_global_bn_exact at line 543.

Risk: the compiled PDF will contain an unresolved reference unless a separate
unavailable source defines the label.

Action: replace the reference with sec:chap4_global_bn_exact, or add the
intended label to the exact-learning subsection. Check the final PDF for all
unresolved references.

### P0.2 Missing graphic assets in the current compilation tree

Evidence: six of seven includegraphics paths in applications.tex are absent
from figs_ch4 in this workspace: stability_landscape_all_roles.png,
pareto_normalized_tchebycheff_all_roles.png, factor_resampling_A0.png,
factor_resampling_A1_B_C.png, umap_seed_sensitivity_all_roles.png, and
retained_factors_A1.png. The observed-versus-BN-implied figure is present.

Risk: this blocks a reproducible local build if the workspace reflects the
submission tree.

Action: place the assets in figs_ch4, use a documented graphicspath, or add a
build step that copies figures from the analysis run. Compile from a clean
clone before submission.

## P1 - major scientific reporting issues

### P1.1 The CPT regularization decision rule is underspecified

Evidence: bn_explain.tex, lines 672-681, says MLEs are retained only if a
"prespecified support criterion" is met. In applications.tex, lines 516-530,
the smallest selected CPT count is 4 and two rows have count <= 5, but MLEs
are retained because no selected row is unobserved. The resolved configuration
sets cpt_regularization_min_observed_count to 0.

Reviewer concern: the method sounds stricter than the rule used in the
application. Sparse rows and boundary MLEs can affect model-implied scenario
probabilities.

Action: state the operative rule exactly. Suggested wording: "Post-selection
regularization was triggered only for an unobserved selected CPT row
(threshold = 0). All selected rows were observed; MLEs were therefore used."
Report the two rows with count <= 5 and the three boundary estimates as
support limitations. Add a Jeffreys-prior sensitivity for the scenarios with
the largest BN discrepancies, or justify why it is unnecessary.

### P1.2 Stability is described, but its empirical result is not reported

Evidence: the subsection Bayesian-network estimation and stability assessment
gives the bootstrap procedure and threshold, but no number of resamples or
summary of the stable-edge result. The saved run used 500 resamples and has 9
of 18 full-sample edges with bootstrap frequency >= 0.60.

Reviewer concern: the section title promises an assessment, yet the reader
cannot judge the amount of stable structure without reopening the output.

Action: add this result sentence after the stability-threshold paragraph:
"Across 500 accident-level bootstrap resamples, 9 of the 18 dependencies
selected in the full sample had selection frequency at least 0.60." Include
the generated stable-network figure and state that edge width encodes
selection frequency and line style encodes the conditional contrast pattern.

### P1.3 Threshold sensitivity does not yet test stability of principal scenarios

Evidence: bn_explain.tex, lines 93-97, says sensitivity is examined to
determine whether principal recurrent configurations depend strongly on n_min.
The application table reports only how many patterns remain at n >= 3, 5, 8,
and 10.

Reviewer concern: changing the number of retained scenarios does not show
whether the substantive scenarios used for interpretation persist.

Action: retain the count table and add a nested-membership table for the
highlighted scenarios, or a compact overlap analysis. State whether the
prevention and BN-discrepancy examples persist at n >= 8 and n >= 10.

### P1.4 The eight labelled discrepancy scenarios need a stated selection rule

Evidence: the observed-versus-BN-implied figure and table identify S1-S8, but
the current caption only says that labels refer to the table.

Reviewer concern: without the rule, the labelled examples can look selected
after inspection.

Action: add a table note: "S1-S3 are the three largest positive count
discrepancies; S4-S6 are the three largest negative discrepancies; S7-S8 are
the nearest-to-equality scenarios among those with recurrence at least equal
to the median." This is the rule used to generate the figure.

### P1.5 Study-population selection needs an auditable definition

Evidence: applications.tex, lines 13-22, calls carpentry and joinery a
sufficiently represented and occupationally specific group, but does not state
the company activity codes, inclusion and exclusion rules, or a minimum-size
criterion.

Reviewer concern: the family is analytically sensible, but selection can
otherwise look discretionary and cannot be reproduced from EPICEA.

Action: provide the company-code family mapping in methods or appendix, state
the eligibility rule, and give the number of accidents removed by each
restriction. If the family was selected substantively rather than through a
numerical threshold, say so plainly.

## P2 - important interpretation and presentation improvements

### P2.1 Show the stable network before scenario-level complementarity

Evidence: the stability section has a CPT-support table but no network result.
A stable-network figure already exists in the saved run.

Action: place it at the end of the BN subsection. Use semantic labels, or
retain factor IDs with a compact nearby key. The generated figure uses IDs
only, which makes it hard to read without returning to the long inventory.

### P2.2 Keep discrepancy descriptive, and add optional uncertainty support

Evidence: the text appropriately states that the discrepancy is not a p-value.
However, 74 of 80 retained scenarios lie above the equality line.

Action: keep the current non-test language. In an appendix, add a bootstrap
or held-out diagnostic for the distribution of observed-minus-implied counts,
if feasible. Report it as robustness, not significance. Otherwise, avoid
language suggesting a general lack of fit beyond this constrained,
within-sample diagnostic.

### P2.3 Make qualitative semantic review reproducible

Evidence: the factor section says representative factual units were reviewed
qualitatively, but does not state how many units per factor were reviewed, how
they were selected, or whether review was blinded to robustness metrics.

Action: add a short appendix protocol: number of representatives, ranking
rule, reviewers, and treatment of ambiguity. The existing cautious discussion
of broad and mixed factors is a good basis for this addition.

## P3 - clarity and economy

### P3.1 Reduce numerical duplication around CPT support

The prose immediately before the CPT-support table repeats most table values.
Keep the interpretive conclusion in prose and let the table carry the full
counts.

### P3.2 Standardize terminology after definition

The application alternates among configuration, pattern, and scenario. Define
the technical distinction once, then use scenario for retained, interpreted
configurations; reserve candidate configuration and closed pattern for their
technical stages.

### P3.3 Keep the revised expected-count language

The horizontal axis now uses "BN-implied expected recurrence, mu_s
(accidents)", which correctly describes mu_s as an expected count. Maintain
that wording in captions, tables, and prose.

## Recommended revision order

1. Restore the missing reference and graphic assets, then run a clean complete
   manuscript build.
2. Specify the CPT regularization rule and report the 500-resample, 9-of-18
   stability result with the stable-network figure.
3. Strengthen threshold sensitivity and state the figure-label selection rule.
4. Document company-code family selection and the semantic-review protocol.
5. Add optional discrepancy uncertainty diagnostics and make terminology edits.

## Strengths to preserve

- The role-constrained ordering and proof of exact BIC optimization are clear.
- The distinction among recurrence, association, conditional dependence,
  bootstrap stability, and BN discrepancy is unusually careful.
- The chapter repeatedly rejects causal interpretation where the design cannot
  support it.
- The principal application counts checked in this review agree with the saved
  analysis outputs.
