# Legacy Image SCGM

Le code original du dépôt SCGM reste présent pour la reproductibilité de l'article image.

## Périmètre legacy

- [`dataset/dataset_breeds.py`](dataset/dataset_breeds.py)
- [`dataset/dataset_cifar.py`](dataset/dataset_cifar.py)
- [`dataset/dataset_tiered_imagenet.py`](dataset/dataset_tiered_imagenet.py)
- [`scgm_g/scgm_resnet.py`](scgm_g/scgm_resnet.py)
- [`scgm_a/`](scgm_a/)
- [`train_scgm_a.py`](train_scgm_a.py)
- [`train_scgm_g.py`](train_scgm_g.py)
- [`test_scgm_a.py`](test_scgm_a.py)
- [`test_fg_scgm_a.py`](test_fg_scgm_a.py)
- [`test_scgm_g.py`](test_scgm_g.py)
- [`test_fg_scgm_g.py`](test_fg_scgm_g.py)
- [`eval/`](eval/)
- [`vis.py`](vis.py)

## Adaptation texte actuelle

Pour le pipeline BTP sur embeddings texte fixes, utiliser [`README_TEXT_ADAPTATION.md`](README_TEXT_ADAPTATION.md), [`scgm_text/`](scgm_text/) et [`scripts/`](scripts/).

Les dépendances image historiques sont listées dans [`requirements_legacy_image.txt`](requirements_legacy_image.txt).
