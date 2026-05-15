# Comparaison MALT vs BERTopic intra-macro vs KMeans + c-TF-IDF

## Objectif

Ce dossier et le notebook `notebooks/03_compare_malt_bertopic_kmeans_topics.ipynb` comparent la **qualité thématique** de trois approches sur le corpus cible (métallurgie) à partir des exports MALT déjà produits :

1. **MALT adapté + c-TF-IDF** : topics = clusters latents `z_hat` ; mots représentatifs par c-TF-IDF intra-cluster ; phrases proches des maxima de `pt_z` (ou du centroïde `nu` dans l’espace projeté adapté).
2. **BERTopic intra-macro via p0** : même chaîne de transfert que MALT pour obtenir `p0(y|x)` sur la cible, **sans adaptation MALT** ; découpage **uniquement** par `p0_macro_name` (A0, A1, B, C) ; un modèle BERTopic par macro sur les **embeddings projetés source** (`target_projected_source.npy` par défaut).
3. **KMeans intra-macro + c-TF-IDF** : même découpage macro que (2) ; KMeans (ou MiniBatch) par macro sur les mêmes embeddings ; top words par **c-TF-IDF** ; phrases les plus proches des centroïdes.

La comparaison porte surtout sur **C_v**, **NPMI**, **Topic Diversity**, **Redundancy**, **Coverage**, avec une section **macro** secondaire (distributions `p0` / `pt`, nombre de topics par macro).

## Package `topic_eval/`

| Module | Rôle |
|--------|------|
| `topic_cleaning.py` | Stopwords FR (NLTK si dispo) + métier ; nettoyage des top words uniquement. |
| `ctfidf.py` | c-TF-IDF par classe (topic). |
| `metrics_topic_quality.py` | Diversité, redondance, couverture, C_v / NPMI (gensim). |
| `bertopic_baseline.py` | BERTopic indépendant par macro. |
| `kmeans_ctfidf_baseline.py` | KMeans + c-TF-IDF par macro. |
| `compare_topics.py` | Vérification des fichiers d’export, enrichissement CSV / `.npy`, construction du tableau MALT, tableaux qualitatifs, rapport Markdown. |
| `visualization.py` | Sauvegarde des figures comparatives. |

## Stopwords métier

Liste dans `outputs/topic_comparison/stopwords_domain.txt` (un mot par ligne). Ces formes sont **retirées uniquement des top words** (pas des phrases sources).

## Métriques (rappel)

- **Topic Diversity** : nombre de mots uniques dans les top words / (`n_topics` × `N_TOP_WORDS`). Plus haut = mieux.
- **Redundancy** : `1 - Topic Diversity` (plus bas = mieux). Variante Jaccard moyenne entre paires de topics disponible dans le tableau détaillé.
- **Coverage** : documents assignés à un topic valide / total ; pour BERTopic, les documents en topic **-1** (outliers) sont exclus du numérateur.
- **C_v / NPMI** : `gensim.models.CoherenceModel` (nécessite `pip install gensim`). Le dictionnaire est construit sur les **textes tokenisés** plus les **listes de top words** (pseudo-documents), pour que les mots c-TF-IDF MALT absents du vocabulaire brut des phrases ne fassent pas tomber la cohérence en NaN.

## Lancement

Depuis la racine du dépôt (avec l’environnement Python activé) :

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
jupyter notebook notebooks/03_compare_malt_bertopic_kmeans_topics.ipynb
```

Paramètres **Papermill** : première cellule du notebook (`MALT_EXPORTS_DIR`, `OUTPUT_DIR`, `TEXT_COL`, etc.).

Chemins par défaut :

- `MALT_EXPORTS_DIR = runs/malt_btp_to_mettalurgie_qwen06/exports`
- `OUTPUT_DIR = outputs/topic_comparison`

Ces chemins sont **relatifs à la racine du dépôt** (dossier contenant `topic_eval/`), pas au répertoire courant du carnet (ex. `notebooks/`).

## Dépendances

Toutes les dépendances (dont `topic_eval`, BERTopic, gensim, etc.) sont listées dans le fichier unique **`requirements.txt`** à la racine du dépôt (hors stack image historique : `requirements_legacy_image.txt`).

## Limites

- C_v et NPMI restent des **proxys** ; une lecture experte des topics et des phrases reste indispensable.
- `p0` / `pt` sont des **pseudo-labels** modèle-dépendants, pas une annotation gold.
- BERTopic peut produire un topic outliers (-1) ; les métriques de couverture en tiennent compte.
