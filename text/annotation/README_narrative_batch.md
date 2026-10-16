# Préannotation indépendante de récits par Batch

Cette campagne est **séparée** de l'annotation des segments SaT existants.
Les sorties sont des propositions LLM, pas une référence humaine. Le guide
`prompts/narrative_units_v0.txt` reste un guide pilote à valider humainement.

Depuis `text/`, préparer 20 récits du corpus bois-textile, sans appel API :

```bash
python scripts/annotate_narratives_batch.py prepare --corpus bois_textille --run-id pilote_20_v0 --limit 20
```

Le nom `--corpus` détermine le CSV recherché :
`annotation/data/<corpus>_sentence_accidents.csv`. Pour un nom différent du
fichier, ajouter `--input-csv CHEMIN`. La commande affiche le dossier créé,
par exemple `output/narrative_annotation/bois_textille/pilote_20_v0/`. Vérifier
`manifest.json`, `documents.jsonl` et `batch_input.jsonl` avant la soumission.
Le fichier envoyé au modèle ne contient que le récit et le prompt ; les
segments SaT sont conservés uniquement dans `documents.jsonl` local.

Soumettre le pilote, puis consulter son état, télécharger et valider :

```bash
python scripts/annotate_narratives_batch.py submit --run-dir output/narrative_annotation/bois_textille/pilote_20_v0
python scripts/annotate_narratives_batch.py status --run-dir output/narrative_annotation/bois_textille/pilote_20_v0
python scripts/annotate_narratives_batch.py download --run-dir output/narrative_annotation/bois_textille/pilote_20_v0
python scripts/annotate_narratives_batch.py validate --run-dir output/narrative_annotation/bois_textille/pilote_20_v0
```

Une fois le guide et la qualité du pilote vérifiés, préparer séparément le
corpus complet avec `--all` et un nouveau `--run-id`. Cette option explicite
évite d'envoyer 1 920 récits par erreur. Un Batch traite une requête par récit ;
il ne nécessite pas de GPU ni de job SLURM. L'API Batch peut prendre jusqu'à
24 heures. Ne pas resoumettre un dossier déjà soumis : `manifest.json`
conserve l'identifiant du batch.

`quality_report.jsonl` indique les réponses absentes ou rejetées et signale
les caractères alphanumériques non couverts par les unités ou exclusions,
`quality_summary.json` donne les totaux, et `llm_annotations.jsonl` contient
uniquement les propositions techniquement valides avec
`review_status=PENDING`. Les contrôles automatiques ne jugent pas l'exactitude
sémantique ; les annotations de référence exigent une validation humaine.

Si beaucoup de réponses échouent à cause des offsets produits par le modèle,
examiner les extraits exacts sans modifier les sorties brutes :

```bash
python scripts/annotate_narratives_batch.py diagnose-offsets --run-dir output/narrative_annotation/bois_textille/pilote_20_v0
```

`offset_diagnostics.jsonl` contient les positions originales et les positions
retrouvées de façon univoque dans le récit ; `offset_summary.json` donne les
totaux. Ces positions sont seulement des propositions à relire. Les textes
introuvables et les chevauchements avec `excluded_spans` restent à résoudre.

Pour créer des brouillons alignés à partir des réponses déjà téléchargées,
sans nouvel appel API ni modification de `batch_output.jsonl` :

```bash
python scripts/annotate_narratives_batch.py reconcile --run-dir output/narrative_annotation/bois_textille/pilote_20_v0
```

`reconciled_annotations.jsonl` contient seulement les récits dont toutes les
unités peuvent être retrouvées littéralement et sans ambiguïté, dans l'ordre,
et dont les intervalles ne se chevauchent pas. Chaque unité conserve ses offsets
proposés par le modèle dans `offset_provenance`; `provenance` relie le brouillon
au batch et au prompt. `reconciliation_report.jsonl` donne les positions
proposées et les motifs de blocage pour tous les récits ;
`reconciliation_summary.json` donne les totaux. Les exclusions du prompt v0
n'ont pas de texte cité : leurs offsets ne peuvent pas être vérifiés par cette
méthode. Tous les brouillons restent `review_status=PENDING` et
`needs_review=true`; ils ne constituent pas une référence humaine.
