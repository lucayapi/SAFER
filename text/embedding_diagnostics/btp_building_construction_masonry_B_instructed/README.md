# Diagnostic des voisins cosinus pour les unités B

## Question

Les embeddings Qwen instruits rapprochent-ils des unités B qui décrivent le
même mécanisme accidentel ?

## Données et calcul

- Unités : `text/dataset/families/btp/data_btp_building_construction_and_masonry.csv`.
- Vecteurs : `text/embeddings/btp_building_construction_and_masonry/Qwen3-Embedding-0.6B_btp_B_mechanism_instruct.csv`.
- Consigne : celle indiquée dans le manifeste de l'export Qwen, centrée sur
  l'événement physique, son déclencheur immédiat et l'objet ou équipement
  impliqué.
- Jointure explicite `fact_id = doc_id` : 3 116 unités B pour 3 116 vecteurs,
  sans doublon ni perte ; dimension 1 024.
- Cinquante références ont été sélectionnées par couverture gloutonne de
  l'espace cosinus, avec au plus une référence par accident.
- Pour chacune, les dix voisins cosinus les plus proches ont été cherchés
  parmi les autres unités B. Toutes les unités provenant du même accident que
  la référence ont été exclues.

## Résultats exploratoires

Les 500 paires (50 références par 10 rangs) ont été lues et classées par un
seul annotateur : 189 « même mécanisme », 157 « mécanisme apparenté », 68
« différent » et 86 « détails insuffisants ». Parmi les 414 paires où le texte
permet un jugement, 189/414 (45,7 %) décrivent le même mécanisme et 346/414
(83,6 %) un mécanisme identique ou apparenté. Au rang 1, le même mécanisme est
retrouvé pour 26/43 paires évaluables (60,5 %), et le même ou un mécanisme
apparenté pour 39/43 (90,7 %).

Les récupérations les plus convaincantes concernent les chutes de hauteur,
les contacts avec des lignes électriques, les rafales de vent, les ruptures de
plaques translucides en toiture, les banches qui basculent, et les
défaillances de supports ou d'appareils de levage. Les rangs plus éloignés
élargissent progressivement la similarité à des mécanismes voisins. D'autres
requêtes récupèrent surtout une conséquence ou un élément de récit : secours,
bruit entendu, malaise, ensevelissement, ou accident sans cause précisée.

La similarité cosinus moyenne est de 0,783 au rang 1 et de 0,730 sur les dix
rangs. Ces scores ne mesurent pas la précision. Trois paires sont des
reprises textuelles exactes de l'ancre.

## Fichiers

- `results/B_reference_units.csv` : les 50 unités de référence.
- `results/B_top10_cosine_neighbours_for_review.csv` : les 500 paires sans
  annotations.
- `results/B_top10_cosine_neighbours_reviewed.csv` : les 500 paires et le
  classement descriptif (« same », « related », « different » ou
  « insufficient detail »).
- `results/B_top1_mechanism_review.csv` : sous-table des meilleurs voisins,
  dérivée du classement cohérent des 500 paires.
- `results/B_precision_at_k_summary.csv` : résumé descriptif cumulé aux rangs
  1, 3, 5 et 10.
- `results/B_top10_similarity_summary.csv` : résumé des scores par référence.
- `results/audit.json` et `results/top1_review_summary.json` : audit des
  entrées, alignement, réglages et résumé du classement.

## Limites d'interprétation

Le classement des 500 paires est une lecture exploratoire par un seul
annotateur ; il n'est pas inter-annotateur et les catégories de rôle B peuvent
contenir des événements, conséquences ou éléments de récit sans mécanisme
explicite. Les pourcentages sont des accords descriptifs avec une grille
qualitative, pas une validation indépendante. La sélection des références et
la recherche des voisins utilisent le même espace d'embedding : ce diagnostic
décrit la cohérence interne de l'espace, sans constituer une évaluation
indépendante.

## Reproduction

Depuis la racine du dépôt :

```powershell
text/.venv/Scripts/python.exe text/embedding_diagnostics/btp_building_construction_masonry_B_instructed/run_diagnostic.py
```
