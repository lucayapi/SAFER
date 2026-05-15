# SAFER

Analyse de récits d'accidents du travail par modèles hiérarchiques (SCGM), transfert macro-ancré (MALT) et réseaux bayésiens exploratoires.

## Structure du dépôt

| Dossier | Contenu |
|---------|---------|
| [`text/`](text/) | Pipeline actif : SCGM-Text, MALT, comparaison de topics, BN (BTP → métallurgie) |
| [`images/`](images/) | Code historique SCGM image (ICLR 2022, BREEDS / CIFAR / tieredImageNet) |

## Démarrage rapide (texte)

```bash
cd text
pip install -r requirements.txt
python scripts/train_scgm_text.py -h
```

Documentation détaillée :

- [Adaptation SCGM texte BTP](text/README_TEXT_ADAPTATION.md)
- [MALT — transfert macro-ancré](text/README_MALT.md)
- [Comparaison de méthodes de topics](text/README_TOPIC_COMPARISON.md)
- [Réseaux bayésiens à partir de MALT](text/README_BN_MALT.md)
- [Notebooks](text/notebooks/README.md)

## Code image (legacy)

```bash
cd images
pip install -r requirements_legacy_image.txt
python train_scgm_g.py -h
```

Voir [images/README.md](images/README.md) (article ICLR 2022).

## Données et artefacts

Les embeddings, checkpoints (`runs/`), figures et exports (`outputs/`) ne sont **pas** versionnés (voir [`.gitignore`](.gitignore)). Placez vos CSV d'embeddings dans `text/embeddings/` en local.
