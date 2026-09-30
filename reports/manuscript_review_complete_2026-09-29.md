# Audit du manuscrit complet : priorités avant soumission

Date : 29 septembre 2026

## Périmètre et méthode

L'audit couvre le projet complet `C:\\Users\\aho\\Documents\\manuscript de thèse\\Manuscript_de_thèse_AHO` : `main.tex`, les chapitres 1 à 4, les annexes, la bibliographie, les tables incluses et les graphiques. Il remplace le rapport antérieur, limité au dossier de travail SAFER.

Les contrôles mécaniques portent sur les inclusions TeX, labels, renvois, citations, ressources graphiques et contenus inachevés. Ils trouvent 164 labels sans doublon, 91 renvois dont 3 sans cible, 201 clés de citation dont 3 invalides, et 24 graphiques inclus tous présents. Les annexes et la bibliographie ont donc été examinées.

Ce contrôle est statique : l'environnement ne contient ni moteur LaTeX ni log de compilation récent. Il faut donc achever par une compilation propre de `main.tex` et une vérification visuelle du PDF. Les P0 sont toutefois directement vérifiables dans les sources.

## Appréciation générale

Le manuscrit est scientifiquement cohérent. Sa progression -- prévision temporelle, structuration des récits, puis facteurs et scénarios récurrents -- est nette. Les chapitres 3 et 4 distinguent avec soin performance prédictive, généralisation, récurrence, association, dépendance conditionnelle et causalité. Cette prudence interprétative est une force à conserver.

## P0 -- corriger avant diffusion du PDF

### P0.1 Front matter inachevé dans le document compilé

**Preuve.** `main.tex:610-614` inclut `ak.tex` et `resume_substantiel_fr.tex`. Ces fichiers contiennent respectivement « à rédiger » (`ak.tex:6`) et « à rédiger .. » (`resume_substantiel_fr.tex:13`).

**Risque.** Les placeholders sont réellement intégrés au manuscrit, avant les résumés. Ils signalent une version non finalisée et peuvent contrevenir aux exigences de l'établissement.

**Correction.** Rédiger les remerciements et le résumé substantiel français, ou retirer provisoirement leurs `input` de `main.tex`. Vérifier ensuite table des matières, signets et pagination.

### P0.2 Annexe du chapitre 2 non constituée et renvois sans cible

**Preuve.** `appendix.tex:2-9` ne contient pour le chapitre 2 qu'une liste textuelle de rubriques, sans sections, figures, tableaux ni labels. Les références `Plot_series` (`chapter2.tex:535`), `forecast_series` (`chapter2.tex:747`) et `app:hp_grids` (`chapter2.tex:553`) n'ont aucune cible dans le projet.

**Risque.** Le PDF produira des « ?? » ou des renvois invalides. Le lecteur ne peut pas vérifier les séries complémentaires, prévisions détaillées et grilles d'hyperparamètres annoncées.

**Correction.** Remplacer la liste par les vraies sections annexes de l'article, y insérer les figures/tableaux annoncés et poser les trois labels. Si ces éléments sont retirés, supprimer les trois renvois et reformuler les phrases associées.

### P0.3 Trois citations ne correspondent pas aux clés bibliographiques

**Preuve.** `chapter2.tex:67` cite `Cheng2012`, alors que la clé est `cheng2012` (`bibliography.bib:411`). `chapter2.tex:161` cite `koc2023`, alors que la clé est `Koc2023` (`bibliography.bib:866`). `chapter2.tex:321` cite `bai2018tcn`, alors que les clés présentes sont `Bai2018_TCN` et `Bai2018TCN` (`bibliography.bib:1150,1227`).

**Risque.** Les clés BibTeX sont distinctes selon la casse : citations absentes ou warnings de bibliographie.

**Correction.** Employer les clés existantes avec leur casse exacte, puis recompiler après BibTeX jusqu'à stabilisation de tous les renvois.

### P0.4 Marqueur Markdown laissé dans le chapitre 4

**Preuve.** `chapter4.tex:529` contient littéralement trois accents graves suivis de `latex`, au début de « Reproducibility under accident-level subsampling ».

**Risque.** Ce marqueur n'a pas de signification LaTeX et peut s'imprimer ou dégrader la page.

**Correction.** Supprimer uniquement cette ligne : le contenu qui suit est déjà une sous-section LaTeX valide.

## P1 -- risque scientifique ou de reproductibilité élevé

### P1.1 Découpage entraînement-test du chapitre 2 ambigu

**Preuve.** La base couvre janvier 2019 à octobre 2022 (`chapter2.tex:442`), tandis que le texte indique : « training period spans from 2019 to 2022, while data from 2022 onward were used for testing » (`chapter2.tex:552`). Cette rédaction ne démontre pas un chevauchement effectif, mais elle ne permet pas non plus d'identifier sans ambiguïté la date de coupure au sein de 2022.

**Enjeu.** Le chapitre défend une évaluation réellement hors échantillon. Un rapporteur peut demander la date de coupure, les effectifs par split et la garantie que calibration du seuil et choix de modèle n'utilisent pas le test.

**Correction.** Indiquer les dates exactes train/test, les nombres de jours ou semaines, et rappeler que tous les paramètres sont figés avant le test.

### P1.2 Paramètres numériques des analyses de stabilité du chapitre 4 incomplets

**Preuve.** La méthode définit `M`, `rho`, des seeds UMAP et `B_boot` (`chapter4.tex:551-555`, `1038-1065`, `2009-2024`), sans donner les valeurs effectivement utilisées dans la recherche de configuration ni la section BN (`chapter4.tex:2415-2442`, `2839-2928`).

**Enjeu.** Ces paramètres déterminent la précision des scores de reproductibilité et des fréquences de sélection. Ils sont indispensables pour reproduire les figures et interpréter le seuil de stabilité.

**Correction.** Ajouter un tableau ou une phrase avec `M`, `rho`, seed de référence, seeds alternatifs, `B_boot`, seed bootstrap et `tau_stab = 0.60`. Donner aussi le nombre d'arêtes parmi les 18 atteignant le seuil.

### P1.3 Population « carpentry and joinery » insuffisamment auditable

**Preuve.** Le groupe est présenté comme suffisamment représenté et spécifique (`chapter4.tex:2343-2357`), mais sans codes d'activité, table de correspondance, inclusions/exclusions ni règle de sélection.

**Enjeu.** Les 417 accidents fondent toutes les estimations du chapitre 4. Sans définition vérifiable, l'échantillon n'est pas reproductible et son domaine de généralisation reste flou.

**Correction.** Ajouter une table « code activité brut -> famille », les effectifs à chaque étape (6 040 -> familles éligibles -> 417), les exclusions, et préciser si ce choix de famille était a priori.

### P1.4 Revue qualitative des facteurs insuffisamment décrite

**Preuve.** La cohérence sémantique est vérifiée sur « a set of representative factual units » (`chapter4.tex:2786-2792`), sans taille de l'échantillon, règle de sélection, nombre d'évaluateurs ni traitement de l'ambiguïté.

**Enjeu.** Les labels et l'interprétation préventive reposent en partie sur cette étape. Elle peut être perçue comme ad hoc si elle n'est pas traçable.

**Correction.** Mettre en annexe : nombre d'unités vues par facteur, règle de sélection, relecteurs et traitement des désaccords. Conserver l'information déjà présente selon laquelle cette revue ne modifie pas les clusters.

## P2 -- améliorations importantes de présentation et d'interprétation

### P2.1 Sensibilité au seuil de récurrence

Le chapitre annonce une analyse de sensibilité à `n_min` (`chapter4.tex:1341-1345`) mais l'application ne montre que les nombres de configurations à 3, 5, 8 et 10 accidents (`chapter4.tex:2939-2976`). Ajouter la persistance des scénarios centraux aux seuils 8 et 10, ou un indice de recouvrement entre inventaires. Cela répond à la question substantielle : les conclusions changent-elles avec le seuil ?

### P2.2 Règle de sélection des huit scénarios annotés du scatter plot

La figure/table de récurrence BN annotent S1--S8 (`chapter4.tex:3264-3367`) sans règle de sélection. Ajouter une phrase : S1--S3 sont les plus grands écarts positifs, S4--S6 les plus grands écarts négatifs, S7--S8 sont les scénarios proches de la diagonale selon une règle annoncée. Le texte actuel sur le caractère descriptif et non testé est juste et doit être conservé (`chapter4.tex:3394-3416`).

### P2.3 Sensibilité BN aux faibles cellules

La table BN rapporte deux lignes de CPT avec `N <= 5`, un minimum de 4 et trois MLE de frontière (`chapter4.tex:2851-2895`). Le texte est prudent, mais les probabilités BN alimentent ensuite les écarts de récurrence. Ajouter en annexe une sensibilité avec lissage de Jeffreys ou Laplace pour les scénarios aux plus grands écarts. Elle ne doit ni modifier le BIC ni être interprétée causalement.

### P2.4 Passage éditorial complet du chapitre 2

Exemples visibles : mélange français/anglais dans une équation (`chapter2.tex:337`, « avec »), confusion Figure/Table (`chapter2.tex:747` commente une figure dont le label commence par `tab`), accord « one day exceed » (`chapter2.tex:76,126`) et répétition de la phrase annonçant la table des variables (`chapter2.tex:482`). Une passe éditoriale complète des captions, labels, accords et majuscules est recommandée.

### P2.5 Déclaration de reproductibilité de thèse

Ajouter une déclaration concise donnant le dépôt du code, les paramètres, l'environnement et, si les données ne sont pas diffusables, les métadonnées, scripts de préparation et procédure d'accès. Cela consolidera les chapitres 2 à 4 sans exposer de récits sensibles.

## P3 -- harmonisation finale

Choisir une variante anglaise uniforme, puis réserver « candidate configuration », « closed pattern » et « scenario » à leurs usages techniques définis. Lorsque les tableaux donnent déjà les comptes, privilégier dans le texte leur interprétation. La section BN de récurrence va déjà dans cette direction.

## Ordre de révision conseillé

1. Supprimer les placeholders et le marqueur Markdown ; corriger les trois renvois et trois citations invalides.
2. Reconstituer les annexes du chapitre 2, puis compiler `main.tex` dans une arborescence propre jusqu'à disparition des warnings de références et citations.
3. Clarifier le split temporel du chapitre 2, puis compléter stabilité, population et revue qualitative du chapitre 4.
4. Ajouter les analyses ciblées : persistance des scénarios, règles de labellisation du scatter, sensibilité CPT si utile.
5. Réaliser la relecture éditoriale et la vérification visuelle finales.

## Forces à préserver

- Progression nette entre les trois niveaux d'analyse de la thèse.
- Chapitre 3 : sélection des modèles sans utiliser les domaines cibles.
- Chapitre 4 : distinction rigoureuse entre récurrence, lift, dépendance conditionnelle, stabilité bootstrap et écart BN ; absence d'interprétation causale indue.
- Discussion explicite des limites in-sample et de la sélection des scénarios.
