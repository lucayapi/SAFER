# SAFER

Analyse de récits d'accidents du travail par modèles hiérarchiques (SCGM), transfert macro-ancré (MALT) et réseaux bayésiens exploratoires.

## Structure du dépôt

Le code du pipeline se trouve dans [`text/`](text/) :

- **SCGM-Text** — apprentissage hiérarchique sur embeddings de récits BTP
- **MALT** — transfert macro-ancré vers d'autres corpus (ex. métallurgie)
- **topic_eval** — comparaison MALT / BERTopic / KMeans
- **notebooks** — expériences reproductibles (01 à 04)

## Démarrage rapide

```bash
cd text
pip install -r requirements.txt
python scripts/train_scgm_text.py -h
```

## Documentation

- [Adaptation SCGM texte BTP](text/README_TEXT_ADAPTATION.md) (modes strict / pragmatique vs SCGM-G officiel)
- [MALT — transfert macro-ancré](text/README_MALT.md)
- [Comparaison de méthodes de topics](text/README_TOPIC_COMPARISON.md)
- [Réseaux bayésiens à partir de MALT](text/README_BN_MALT.md)
- [Notebooks](text/notebooks/README.md)

## Données et artefacts

Les embeddings, checkpoints (`runs/`), figures et exports (`outputs/`) ne sont **pas** versionnés (voir [`.gitignore`](.gitignore)). Placez vos CSV d'embeddings dans `text/embeddings/` en local.

Les métadonnées segmentées (`text/dataset/data_btp.csv`, `data_metallurgie.csv`) sont versionnées ; les fichiers `.env` et clés API ne doivent jamais être commités.
