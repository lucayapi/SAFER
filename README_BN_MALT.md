# BN MALT — Réseaux bayésiens à partir des motifs MALT

## Objectif

Transformer les sorties **MALT** (motifs latents `z`, confiances, `p(y|z)`) en variables **discrètes au niveau accident**, puis apprendre des **réseaux bayésiens** avec **pgmpy** pour :

- quantifier des dépendances conditionnelles (sous contraintes d’ordre macro A0 → A1 → B → C → gravité) ;
- explorer des **scénarios** typiques et des requêtes du type \(P(B\mid A1)\) sur les variables binaires de présence de motif.

## Interprétation critique

- **\(X_{i,k}=0\)** signifie que le motif \(k\) n’apparaît pas dans le récit de l’accident \(i\) **au-dessus du seuil de confiance** : ce n’est **pas** une preuve d’absence physique du facteur.
- Un **arc** dans le BN indique une **dépendance conditionnelle** apprise (souvent avec un prior bayésien), **pas** une relation causale démontrée.

## Données d’entrée

Exports MALT (par défaut `runs/malt_btp_to_mettalurgie_qwen06/exports`) :

- `metadata_with_malt_predictions.csv` (avec **`accident_id`** obligatoire) ;
- `pt_z_target.npy`, `pt_y_target.npy`, `pt_y_given_z.npy`, `z_assignments_target.csv`.

Les chemins `MALT_EXPORTS_DIR` et `OUTPUT_DIR` du notebook sont **relatifs à la racine du dépôt** (dossier contenant `topic_eval/`), pas au répertoire `notebooks/`, grâce à `topic_eval.paths.resolve_repo_path`.

## Lancement

```bash
pip install -r requirements.txt
jupyter lab notebooks/04_malt_to_bayesian_network.ipynb
```

### Noyau Jupyter (`ModuleNotFoundError: pgmpy`)

L’installation doit cibler **le même interpréteur** que celui du noyau (voir *Kernel → Change kernel* dans Jupyter / VS Code).

**Important :** si vous avez installé les paquets dans le `.venv` du dépôt mais que le carnet utilise encore **Python Anaconda** (`…\anaconda3\python.exe`), vous continuerez à avoir **NumPy 2.x** et des erreurs matplotlib. Soit vous corrigez **Anaconda** (`numpy<2` sur cet interpréteur), soit vous sélectionnez le noyau **`.venv`** après `pip install -r requirements.txt` dans ce venv.

1. Dans une cellule du notebook : `import sys; print(sys.executable)`
2. Dans un terminal :  
   `"<chemin affiché>" -m pip install "numpy<2" --force-reinstall`  
   puis `pip install -r requirements.txt` si besoin.
3. **Redémarrer le noyau** (*Kernel → Restart*), puis réexécuter depuis le haut.

Si `pip` échoue sous Windows avec *« fichier utilisé par un autre processus »* (souvent sur `scipy`), fermez Jupyter / VS Code, réessayez, ou utilisez un environnement virtuel dédié.

### NumPy 2.x vs matplotlib / Anaconda (`numpy.core.multiarray failed to import`)

Si vous avez installé **pgmpy 1.x**, `pip` peut avoir mis **NumPy 2.4+**, alors que le **matplotlib** (ou d’autres wheels) du base conda a été compilé pour **NumPy 1.x** → erreur à l’import de `matplotlib`.

Ce dépôt fixe **`numpy>=1.24,<2.0`** et **`pgmpy>=0.1.23,<1.0`** pour rester cohérent avec la stack classique.

**Réparer un environnement Anaconda déjà cassé** (adapter si vous utilisez `conda` / `mamba`) :

```bash
conda install "numpy<2" "matplotlib" "scipy" --yes
pip install "numpy<2" "pgmpy>=0.1.23,<1.0"
```

Puis **redémarrer le noyau** Jupyter.

Si vous tenez à **NumPy 2** et **pgmpy 1.x**, il faut plutôt **mettre à jour** matplotlib/scipy/pandas avec des builds compatibles NumPy 2 (souvent `conda update --all` ou un env neuf).

### Papermill (exemple)

```bash
papermill notebooks/04_malt_to_bayesian_network.ipynb \
  outputs/bn_malt/executed_bn_notebook.ipynb \
  -p MALT_EXPORTS_DIR runs/malt_btp_to_mettalurgie_qwen06/exports \
  -p OUTPUT_DIR outputs/bn_malt \
  -p CONFIDENCE_THRESHOLD 0.50 \
  -p MIN_TOPIC_ACCIDENT_SUPPORT 20 \
  -p MAX_TOPICS_PER_MACRO 6
```

## Sorties

Sous `outputs/bn_malt/` : `tables/`, `figures/`, `models/`, `reports/` (voir notebook pour la liste des fichiers CSV, LaTeX, PNG, HTML optionnel).

## Dépendances optionnelles

- **pyvis** : visualisation interactive ; peut être omise si l’installation échoue.
- **plotly** : Sankey / HTML interactif (déjà optionnel dans le code).

## Régénérer le notebook

```bash
python scripts/build_notebook_04_malt_bn.py
```
