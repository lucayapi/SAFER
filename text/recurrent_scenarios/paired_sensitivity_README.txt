Contrôle des paramètres de reconstruction pour l'article JRSS C
=============================================================

Depuis text/, sur le cluster :

    sbatch jobs/run_recurrent_scenarios_parameter_sensitivity.sh

Un seul job Slurm, 30 CPU, 200 réplications par politique. Les politiques sont
traitées successivement ; les réplications d'une politique sont parallélisées.
Les threads internes sont limités à un par worker. Aucun GPU requis.

Paramètres : recurrent_scenarios/config.yaml, section paired_recovery.
Le nombre de réplications, la fraction, la graine et les workers sont communs.
La liste sensitivity.parameter_policies définit les trois comparaisons :

  fixed                 paramètres absolus sélectionnés, analyse de référence ;
  scaled_cluster_size   seule la taille minimale du cluster est ajustée ;
  scaled_density        taille minimale et min_samples sont ajustés.

L'ajustement multiplie le paramètre par la proportion effectivement retenue
d'unités textuelles du rôle, puis arrondit à l'entier le plus proche (demi vers
le haut). Minimum : 2 pour la taille de cluster, 1 pour min_samples.
Les paramètres UMAP et sa graine restent identiques. Aucun hyperparamètre n'est
resélectionné ; les facteurs de référence restent les mêmes.

Chaque politique reçoit exactement les mêmes sous-échantillons d'accidents.
Le programme vérifie leur identité ainsi que celle des résultats à facteurs
fixes. Les contrastes finaux comparent la récupération APRÈS reconstruction
sous chaque politique à celle après reconstruction avec paramètres fixes.
Une différence positive indique une récupération plus fréquente après ajustement.

Sortie :
  recurrent_scenarios/runs/theme_discovery_audit/btp_carpentry_and_joinery/
      paired_parameter_sensitivity/

  fixed/, scaled_cluster_size/, scaled_density/ : résultats détaillés ;
  policy_comparison.csv : différences appariées par scénario/relation + erreur MC ;
  policy_summary.csv : médianes par politique et seuil d'appariement ;
  comparison_audit.json : vérifications et provenance du code.

Les anciens résultats paired_recovery/ sont conservés. La politique fixed est
recalculée dans le nouveau dossier pour garder un contrôle homogène des sources.
Un job interrompu reprend les réplications déjà terminées si le design concorde.

Pilote séparé (vérification technique, pas de résultats à citer) :
    N_REPLICATES=2 OUTPUT_DIR=recurrent_scenarios/runs/theme_discovery_audit/btp_carpentry_and_joinery/paired_parameter_sensitivity_pilot sbatch jobs/run_recurrent_scenarios_parameter_sensitivity.sh

Ces nouvelles expériences doivent être exécutées avant d'en décrire les
résultats dans le manuscrit. Les tableaux ajoutés au manuscrit proviennent
exclusivement des 200 réplications déjà archivées.
