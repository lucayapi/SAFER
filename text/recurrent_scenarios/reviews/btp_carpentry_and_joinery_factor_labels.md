# Editorial review of the 42 selected factor labels

Source: `runs/theme_discovery_rebuild/btp_carpentry_and_joinery/topics_manual/topic_dictionary_with_llm_labels.csv` (42 factors: 25 A0, 11 A1, 2 B, 4 C). This review concerns wording and interpretation only. Factor IDs, assignments, and the running statistical analysis are unchanged. Proposed wording is for human review before manuscript use; it is not a new factor annotation.

## Labels used in the manuscript's S1--S6

These are the six scenarios defined by `sections/application.tex` and `figures/scenario_diagram.tex`. They differ from the six highest-support scenarios automatically exported in `bn_results_exact/scenarios/scenarios_article.csv`. The factor IDs below were checked against `recurrent_scenarios_all.csv` using the manuscript's counts, confidence, lift and fitted counts.

| Factor | Current label | Suggested manuscript label | Assessment |
| --- | --- | --- | --- |
| A0_024 (S1) | Woodworking and Carpentry Machine Operations | Woodworking with powered cutting machines | The representative units describe several saws and woodworking machines. Avoid treating all units as carpentry. |
| B_001 (S1, S2, S5, S6) | Woodpiece kickback causing hand contact | Wood kickback and hand contact with cutting tools | The joint motif is well supported; some units describe slipping rather than a clear kickback. |
| C_000 (S1, S2, S5, S6) | Severe Finger and Hand Amputations | Severe finger and hand injuries | Cuts and fractures also occur in this group. The current label implies that every case is an amputation. |
| A1_002 (S2) | Inadequate Machine Guarding and Protection | Missing or ineffective machine guards | The representative units describe missing or ineffective guards on machinery. |
| A0_017 (S3) | Roofing replacement and installation work | Roof covering removal and replacement | Supported by the descriptions of replacing roofing materials. |
| A1_007 (S3) | Deteriorated and fragile roofing surfaces | Keep, with caution | Fragile roofing is supported; not every unit describes deterioration. |
| C_003 (S3) | Fatal or life-threatening traumatic injuries | **Review substantive interpretation** | Several highly representative units describe emergency transport, resuscitation or hospital care rather than a specified injury or death. The manuscript figure calls this a fatal consequence; verify the five supporting reports directly before retaining that wording. |
| A1_001 (S4) | Unsecured and incomplete scaffolding systems | Incomplete or unsecured scaffolding | The representative units describe scaffold assembly, access and missing safeguards. |
| C_002 (S4) | Multiple fractures and traumatic bodily injuries | Fractures and other traumatic injuries | Some representative units describe a single fracture or another traumatic injury. |
| A0_021 (S5) | Carpentry and roofing trade work | **Review substantive interpretation** | Representative units chiefly identify the worker's occupation and age, not a specific pre-accident work situation. This is one of the two apparent recovery increases discussed in the manuscript. |
| A0_005 (S6) | Wood cutting with powered saws | Keep | The representative units describe powered saws and wood cutting. |
| B_000 (S3, S4) | Falls from elevated work surfaces | Falls from height | Supported and more concise. |

The automatically exported top-six scenario list also contains A0_006, A0_015, A0_023, A0_000 and C_001. In particular, A0_023 is **not** the manuscript's S4. Its semantic weakness remains relevant to any broader interpretation of the complete scenario catalogue, but it is not a flaw in the manuscript's scaffolding example.

Direct checks of the supporting factual units clarify three labels. S1 has 23 supporting reports, but its C factor also includes cuts and fractures; the title must not imply 23 amputations. Among the five S3 reports, the C units include polytrauma, fractures, cardiorespiratory arrest and a cervical injury; the combination is not a five-report fatality scenario. Among the five S4 reports, three C units explicitly mention fractures, one describes multiple organ injuries and one describes abrasions. In S5, all five A0 units used for this factor describe the worker's occupation, rather than the task being performed. These checks concern the recorded sentences, not independently verified clinical outcomes.

## Other labels to revise before manuscript use

| Factor | Suggested wording or action | Reason |
| --- | --- | --- |
| A0_001 | Manual guidance of wood at saws | The units include more than one saw type and hand-guided operation. |
| A0_002 | **Do not interpret as a substantive work-situation factor** | Work times and site preparation are mixed with other narrative timing and emergency-response statements; the generated assessment itself reports partial role fit and high heterogeneity. |
| A0_008 | Material dimensions and cutting tasks | Wood, panels and sheet metal occur; the examples often record dimensions rather than a performed cut. |
| A0_010 | **Review before substantive interpretation** | Many representative units concern the timing or organization of a job, rather than construction or roofing as such. |
| A0_011 | **Do not interpret as a substantive work-situation factor** | Investigation comments and uncertainty about accident circumstances are reporting content, not a pre-accident work situation. |
| A0_012 | Positioning and access during elevated work | Corrects “Charpentry” and reflects ladders, roof structures and workers' positions across the examples. |
| A0_013 | Work near elevated surfaces and openings | More concrete than “exposed surfaces”; verify against mixed indoor and roof examples. |
| A0_014 | Scaffold setup and use | The units cover presence, assembly, replacement and use of scaffolds, not all elevated work. |
| A0_016 | Construction schedules and project duration | The representative units chiefly concern duration, deadlines and staging. |
| A0_021 | **Do not interpret as a substantive work-situation factor** | The group chiefly records occupation and age, with limited information on the work being performed. |
| A0_022 | **Do not interpret as a substantive work-situation factor** | The group chiefly records job titles and employer activities, with limited information on a particular work situation. |
| A1_001 | Incomplete or unsecured scaffolding | Shorter and faithful to the examples. |
| A1_002 | Missing or ineffective machine guards | Avoids the broad and redundant “guarding and protection”. |
| A1_003 | **Keep broad; review heterogeneity** | The model marks high heterogeneity; a precise title would overstate cohesion. |
| A1_004 | **Keep broad; review heterogeneity** | The group combines unsuitable equipment, spatial constraints and missing protective arrangements. |
| A1_008 | Unsafe roof access and unprotected openings | Reflects the examples more closely than “elevated work areas”. |
| A1_010 | Missing guards and protective barriers | Covers both machines and openings without pretending these are one particular safeguard. |
| C_002 | Fractures and other traumatic injuries | Some representative units describe a single fracture or another traumatic injury. |

The remaining generated labels are usable as descriptive shorthand after normal copy editing (sentence case, consistent spelling and British English). These proposals should be checked against the full cluster content before being frozen in the article. A descriptive label does not itself validate a cluster's accident-process role.
