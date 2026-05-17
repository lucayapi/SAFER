# SAFER

Analyse de récits d'accidents du travail par modèles hiérarchiques (SCGM), transfert macro-ancré (MALT) et réseaux bayésiens exploratoires.

Le pipeline se trouve dans [`text/`](text/). **Documentation complète : [`text/README.md`](text/README.md).**

## Démarrage rapide

```bash
cd text
pip install -r requirements.txt
python scripts/train_scgm_text.py -h
```

Les artefacts (`text/resultats/`, `text/embeddings/`) ne sont pas versionnés. Les métadonnées (`text/dataset/`) le sont. Ne jamais committer `.env` ni de clés API.
