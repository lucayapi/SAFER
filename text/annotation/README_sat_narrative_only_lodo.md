# Plis LODO communs pour M0–M3

Le paquet `annotation/lodo_sat_transfer/narrative_only_lodo_v1/` est la source
de données figée pour comparer M0–M3. Il contient six rotations leave-one-
corpus-out, les récits nettoyés, les unités factuelles GOLD, leurs rôles, les
index d’identifiants des documents et les données d’entraînement SaT.
Les spans explicitement exclus ont été retirés du texte fourni aux modèles.

Pour chaque cible, les cinq autres corpus sont les sources. La partition
d’entraînement est celle de 80 % des récits source; les 20 % restants servent
à la validation source; le corpus cible entier reste réservé au test. Le
`split_ids.json` de chaque pli permet de vérifier l’identité des documents.
Toutes les méthodes doivent lire ces mêmes fichiers et ne pas recréer les plis.

La métrique de segmentation est le F1 exact normalisé aux bords : on ignore
uniquement les espaces Unicode et la ponctuation Unicode au début et à la fin
des spans GOLD et prédits. Tout leur contenu interne doit correspondre
exactement. Les anciens scores obtenus sur les récits non nettoyés ou avec un
exact-span F1 non normalisé ne sont pas directement comparables.

M0 PySBD et M1 SaT préentraîné n’ont pas de paramètres ajustés sur les sources.
M2 sélectionne son seuil sur la validation source. M3 s’entraîne sur les unités
factuelles GOLD des sources puis sélectionne son seuil sur cette même
validation. Les méthodes peuvent différer; les plis et le test doivent rester
identiques.

## Paquet autonome

Le paquet contient déjà les six plis, leurs index, les données d’entraînement
SaT, les révisions immuables du modèle et du tokenizer, ainsi que les sommes de
contrôle. Il n’y a pas d’étape de préparation qui nécessite un autre paquet.
Le contrôle local est :

```powershell
.\text\.venv\Scripts\python.exe text\scripts\sat_narrative_only_lodo.py check
.\text\.venv\Scripts\python.exe -m pytest text\tests\test_sat_narrative_only_metrics.py -q -p no:cacheprovider
```

Transférer le dossier complet
`annotation/lodo_sat_transfer/narrative_only_lodo_v1/` vers
`$SCRATCH/safer_private/narrative_only_lodo_v1/`, ainsi que le code SAFER à jour
`text/scripts/sat_narrative_only_lodo.py`,
`text/scripts/sat_narrative_only_runtime.py`, les jobs
`text/jobs/sat_narrative_only_*` et `_sat_narrative_only_env.sh`. Le CSV nettoyé
n’a pas besoin d’être transféré séparément : les récits et offsets sont dans
les plis.

Depuis la racine SAFER sur le cluster :

```bash
bash text/jobs/sat_narrative_only_launch.sh
```

Cela soumet un smoke GPU, six tâches cibles (au plus trois simultanées), puis
un résumé. Les résultats SaT+LoRA sont écrits dans
`$SCRATCH/safer_sat_lodo/runs/narrative_only_lodo_v1/`. Les évaluateurs M0–M2
doivent eux aussi lire les plis de ce paquet pour que le tableau M0–M3 soit
comparable.
