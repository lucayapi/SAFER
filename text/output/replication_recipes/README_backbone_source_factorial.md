# Campagne backbone x domaine source

1. Exporter les CSV figÃ©s, dans des rÃ©pertoires disjoints :

   ```bash
   cd ~/SAFER/text
   ALL_CORPORA=1 BACKBONE_NAME=Qwen/Qwen3-Embedding-0.6B BACKBONE_ID=qwen3 sbatch jobs/export_corpus_embeddings.sh
   ALL_CORPORA=1 BACKBONE_NAME=intfloat/multilingual-e5-large BACKBONE_ID=multilingual_e5_large INPUT_PREFIX='query: ' sbatch jobs/export_corpus_embeddings.sh
   ```

2. Produire les neuf nouvelles sÃ©lections source-only pour les mÃ©thodes adaptÃ©es :

   ```bash
   bash jobs/submit_backbone_source_tuning.sh
   ```

   La soumission est chaÃ®nÃ©e en trois groupes de trois jobs : E5/BTP, puis
   Qwen3/mÃ©tallurgie, puis E5/mÃ©tallurgie. Chaque groupe commence seulement
   aprÃ¨s la rÃ©ussite du groupe prÃ©cÃ©dent.

   Les trois sÃ©lections Qwen3/BTP de l'Ã©tude principale sont rÃ©utilisÃ©es.
   Les configs gÃ©nÃ©rÃ©es ont `test_corpora: []`; les cibles OOD ne sont donc
   jamais utilisÃ©es par la CV de sÃ©lection. Appliquer les choix retenus :

   ```bash
   python scripts/apply_backbone_source_tuning.py --write
   ```

3. Lancer les 80 fits finaux :

   ```bash
   CONFIG=output/replication_recipes/backbone_source_factorial.yaml \
     bash jobs/run_backbone_source_campaign.sh
   ```

4. AprÃ¨s complÃ©tion de toutes les tÃ¢ches :

   ```bash
   python scripts/analyze_backbone_source_campaign.py \
     --config output/replication_recipes/backbone_source_factorial.yaml
   ```

   Les sorties sont dans `output/backbone_source_factorial_analysis/` :
   `ood_summary.csv`, `ood_balanced_accuracy.png`, un heatmap par cible,
   `per_role_metrics.csv`, l'inventaire de prÃ©dictions et les comparaisons de
   mÃ©thodes. Les IC sont bootstrapÃ©s au niveau `accident_id`.
