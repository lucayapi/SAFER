# RQ1 — inventaire legacy

Les expériences ci-dessous restent à leur emplacement pour permettre l'audit.
Leurs scores ne sont **pas** importés dans les tableaux de la campagne
`rq1_segmentation_final_v1`. Les seules réutilisations autorisées sont le GOLD
`narrative_v3`, le partage 70/15/15 vérifié, les probabilités de SaT non adapté,
les prédictions PySBD et les six adaptateurs mono-source après vérification de
leurs empreintes et de leur sélection sur validation source.

| Ancienne expérience | Emplacement | Raison du classement |
| --- | --- | --- |
| Préannotations directes et campagnes API | `annotation/runs/`, `annotation/audit/` | Historique de construction du GOLD ; ni tests ni scores finaux |
| Calibration SaT initiale | `annotation/sat_threshold_calibration_20261016/`, `annotation/rq1_sat_v3_20261017/` | Partitions et tests différents du protocole commun |
| Comparaison SaT/préannotations | `annotation/analysis_sat_vs_preannotation/` | Ancienne référence, avant GOLD v3 |
| LODO initial | `safer_sat_lodo/runs/sat12l_sm_gold_v3_rq1/` et variantes | Test sur domaine cible entier, incompatible avec les 15 % communs |
| Mono-source et in-domain | `safer_sat_indomain/runs/sat12l_sm_gold_v3_indomain_v1/` | Anciens tableaux exclus ; adaptateurs et caches audités comme intrants figés |
| PySBD initial | `annotation/rq1_pysbd_v3_20261017/` | Ancien tableau exclu ; cache de prédictions vérifié puis rescoring commun |

Les scripts historiques `sat_lodo_gold_v3.py`, `sat_lodo_checkpoint_select.py`,
`sat_indomain_v3.py`, `sat_indomain_soft_span.py`, `rq1_pysbd_baseline.py` et
`annotation/rq1_calibrate_sat_v3.py` restent en place. La nouvelle campagne
écrit uniquement dans son dossier privé et applique sa nomenclature propre :
M0 PySBD, M1 SaT préentraîné, M2 SaT calibré sur sources, M3 SaT + LoRA.

Les récits, caches et adaptateurs sont ignorés par Git. Cet inventaire ne copie
aucune donnée privée dans le dépôt.

