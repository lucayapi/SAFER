# Finir la classification LODO : CE et Batch-hard Triplet

La trace `CrossEntropyLoss ... expected scalar type Float but found Double`
concerne **CE**. SupCon et SoftTriple ont déjà leurs 18 résultats complets
chacun. Le script original de la campagne reste inchangé pour que son
empreinte et les 48 résultats existants restent vérifiables. Cette extension
entraîne seulement CE et Batch-hard Triplet sur les mêmes six plis, avec les
graines 42, 1729 et 20261018, trois époques et une sélection sur la moyenne
des balanced accuracy des cinq validations sources.

Depuis la racine `SAFER/` sur le cluster, après avoir copié les cinq nouveaux
fichiers `role_qwen3_lodo_missing.py` et `role_qwen3_lodo_missing_*.sh` :

```bash
bash text/jobs/role_qwen3_lodo_missing_launch.sh
```

Le lanceur soumet un smoke GPU, un array de 36 tâches (6 cibles × 3 graines ×
2 méthodes, au plus 3 GPU simultanés), puis une synthèse CPU dépendante de
la réussite de tout l'array. Les tâches terminées et vérifiées sont ignorées
à la relance. Les nouveaux résultats sont rangés sous
`$SCRATCH/safer_role_qwen3_narrative_lodo_3epoch_v1/missing_ce_batch_triplet_v1/`.
Les sorties précédentes ne sont ni modifiées ni réentraînées.

L'extension conserve le même `campaign.json`, le même environnement
`text/.venv` et vérifie l'empreinte du script original. Elle enregistre sa
propre empreinte dans son manifeste. La synthèse finale exige les 84 résultats
(48 existants + 36 nouveaux) et vérifie les métriques depuis les prédictions.
