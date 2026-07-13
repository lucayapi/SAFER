# Données d'entrée pour l'annotation

Placez ici vos fichiers CSV à annoter (non versionnés si données sensibles).

## Colonnes requises

| Colonne | Obligatoire | Description |
|---------|-------------|-------------|
| `accident_id` | oui | Identifiant unique de l'accident |
| `sentence` | oui | Unité factuelle à annoter |
| `fact_id` | recommandé | Identifiant de l'unité dans l'accident |
| `accident_summary` | recommandé | Résumé global (désambiguïsation) |

## Exemple

Dans le notebook `annotate_factual_units.ipynb` :

```python
INPUT_CSV = "mon_corpus.csv"  # relatif à annotation/data/
```

Les CSV du projet (`dataset/data_btp.csv`) utilisent déjà `accident_summary` comme colonne de résumé.

Les **sorties** du pipeline (`annotation/outputs/`) sont exportées en **XLSX** (snapshot, annotated, summary, accident_outcomes). Le cache de reprise reste en JSONL.
