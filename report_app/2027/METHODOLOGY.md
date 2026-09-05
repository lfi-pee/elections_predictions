# Législatives 2027 — note de méthode (1 page)

Prévision **géographique** par circonscription, ancrée sur les sondages. Ni sondage maison ni
simulation de campagne : un modèle qui apprend *où* chaque bloc sur- ou sous-performe, puis
applique le niveau national que vous posez au curseur.

## 1. Prédiction du 1er tour (par bureau de vote)
- **Cible** : parts de bloc (Gauche / Centre+Droite / Extrême-Droite) + abstention, par bureau.
- **Espace de travail** : *déviations* au national (score du bureau − moyenne nationale du
  scrutin), stables d'un scrutin à l'autre.
- **Modèle** : Ridge + PCA, un réglage par bloc, entraîné sur les législatives **2002→2024**.
  Prédicteurs : surtout l'héritage de vote local (déviations 2024 puis 2022) + 52 indicateurs
  INSEE. Sortie servie : la déviation par bureau (le *motif spatial*), pas un score figé.
- **Niveau national** : posé par l'utilisateur (curseurs), présélections = sondages publiés
  agrégés (2025-2026). `score = national(curseur) + déviation(modèle)`.
- **Couplage participation (γ)** : baisser l'abstention réaffecte les revenants selon la courbe
  γ mesurée en 2024 (l'électeur de retour penche à gauche) → relève les parts *effectives* de
  gauche. Effet FORT sur les sièges de gauche, dans toutes les configurations.

## 2. Modèle de 2nd tour (jouabilité & sièges)
Qualification au seuil des **12,5 % des inscrits** ; puis reports, tous **mesurés sur données
réelles** (réglables au curseur) :
- **Désistement « front républicain »** (mécanisme dominant) — mesuré sur les 271 triangulaires
  face au RN de 2024 (`desist_2024_measure.py`) : ~69 % au survivant anti-RN, 17 % au RN, dans
  celles où un pôle s'est effectivement retiré. Défaut **0,52** = force inconditionnelle (toutes
  triangulaires, dont ~⅓ maintenues) qui reproduit le **nombre réel** de sièges RN 2024 (109
  contre 109), sans biais.
- **Reports Centre+Droite** décomposés Ensemble (barrage) / LR (ambivalent, bascule au RN sous
  « union des droites »).
- **Réunification d'une gauche divisée** (`reunif`, défaut 0,72) — mesurée sur 2012 (gauche
  divisée : PS/Front de Gauche/EELV) : régression 0,69–0,73 (`reunif_measure.py`).
- **Partage gauche radicale (LFI) / soc-dém** — la part LFI du VOTE (curseur, ~37 % sondé) est
  répartie par circo selon le **motif réel des européennes 2024** (le scrutin divisé le plus
  récent ; `radical_spatial.py`), dispersion appliquée telle quelle (RAD_GAIN 1,0). La part de
  SIÈGES en diffère (~27 % des sièges de gauche à LFI en divisé — le pôle le plus fort rafle le
  siège).

## 3. Incertitude
- **Locale** : intervalles conformes par bureau (validation croisée), ramenés au grain circo par
  le rapport observé σ_circo/σ_bv (≈0,7, pas 1/√N : erreurs corrélées dans une circo).
- **Fourchette de sièges** : Monte-Carlo (bruit gaussien indépendant par circo/bloc). Ne borne
  que l'erreur locale — pas l'incertitude nationale (posée) ni structurelle (hypothèses de report).

## 3 bis. Voix régionalistes : attribution et portée résiduelle
- **Le trou** : le modèle range chaque candidat dans l'un de **trois blocs**. Les nuances
  ministérielles `REG` (régionaliste), `DIV` et `DSV` n'y entrent pas — 1,77 % du vote national,
  mais la force **dominante** dans une vingtaine de circonscriptions. Cas extrême, Cayenne :
  **25,6 %** du vote rattaché, la gauche à **1,2 %** alors qu'elle détient le siège.
- **Réparation automatique** (`cross_type_ridge._apply_candidate_lineage`) : un candidat codé
  « Autre » reprend le bloc où le **même** candidat était codé à un scrutin antérieur. Elle
  rattrape 34 circos (Nadeau, Molac, Dupont-Aignan, Gokel, Beaudet, Sempastous) et échoue sur les
  élus codés `REG` **à tous** leurs scrutins : il n'y a aucun codage antérieur à retrouver.
- **Attribution explicite** (`data/nuance/attribution_regionalistes_2024.csv`) : pour ceux-là,
  une décision par candidat, adossée à un fait vérifiable et par ordre de priorité — le **groupe
  parlementaire rejoint**, l'**investiture de coalition** (NFP 2024), le groupe d'un **mandat
  antérieur**, l'alignement du **parti**. Dix candidats rattachés à la Gauche : Castor et Rimane
  (GDR, Guyane), Nilor (LFI) et William (apparenté Socialistes) et Carole (PALIMA, Martinique),
  Tjibaou et Naisseline (UC-FLNKS, Nouvelle-Calédonie), Le Gayic, Chailloux et Reid Arbelot
  (Tavini, Polynésie). Cayenne passe de **1,3 % à 64,1 %** de gauche.
- **Non-attributions assumées**, inscrites dans la même table avec leur motif : la mouvance
  **autonomiste corse** (Colombani, Castellani, Acquaviva, Colonna) siège au groupe **LIOT**,
  territorial et non aligné — la ranger dans un bloc déformerait quatre circonscriptions ; une
  candidate **sans étiquette** n'ayant donné aucune consigne de second tour ; les « divers »
  locaux sans alignement établi. Leurs voix restent exclues et renormalisées, comme le modèle
  traite déjà les divers au niveau national.
- **Couverture résiduelle** : mesurée circo par circo sur le fichier officiel du ministère
  (`coverage.json`, produit par `src/attribution_2027.py`), pour les **577** circonscriptions.
  Seuil de marquage = 100 − la plus large demi-largeur à 90 % servie (±9,7 pts) → **90,3 %** :
  en deçà, l'erreur de nomenclature dépasse à elle seule l'incertitude annoncée.
- **Le marquage suit la CHAÎNE, pas la mesure.** Réparer la donnée 2024 ne répare pas la
  prévision : les déviations 2027 servies descendent du modèle entraîné avec l'ancienne
  nomenclature. Tant que `summary.attribution_applied` n'est pas posé par une reconstruction,
  le marquage reste au niveau d'avant — **19 circos**. Après `./rebuild_2027.sh`, il tombe à
  **11** : les quatre corses et sept où subsistent des « divers » sans alignement établi.
- **Ce qui reste marqué** est grisé sur la carte (liseré tireté), sans score ni siège dans le
  panneau, et porte une colonne `fiabilite` dans l'export CSV. Ces circos restent comptées dans
  les totaux de sièges — les en retirer fausserait l'Assemblée.
- **Effet sur la validation** : le vrai vainqueur 2024 était lui aussi lu à travers la
  nomenclature (`backtest_2024_seats`), si bien que les circonscriptions gagnées par un élu codé
  `REG` sortaient **silencieusement** du backtest — la justesse était calculée sur 501 circos.
  La table y est désormais branchée ; le chiffre se met à jour à la prochaine reconstruction.

## 4. Validation sur 2024
- **À l'aveugle (chaîne complète)** : 2024 **retiré de l'entraînement**, prévision du 1er tour →
  modèle de sièges → vrais sièges → **~78 %** des circos disputées (`backtest_2024_endtoend.py`).
- **Oracle (modèle de sièges seul, parts réelles)** : ~82 % des disputées ; **RN sans biais**
  (109 sièges projetés contre 109 réels).
- **1er tour (sièges sûrs, gagnés au 1er tour)** : 99 % ; **ensemble des 577 : ~84 %**
  (`backtest_2024_firstround.py`).
- **Réserve** : le test à l'aveugle retire 2024 de l'entraînement, ce qui sous-estime le RN
  (~70 rejoués vs 109) ; la prévision 2027 garde 2024 en mémoire, donc n'a pas ce handicap.

## 5. Reproductibilité
- **Une commande** rebâtit chaque nombre servi depuis les données brutes : `./rebuild_2027.sh`.
- **Parité Python ↔ JavaScript** : `test_parity_2027.py` exécute le vrai `compute.js` (Node) et
  vérifie qu'il calcule exactement le même modèle de sièges que Python (constantes + 720 cas).
- **Aucun chiffre figé** : `test_no_hardcoded_2027.py` vérifie que les statistiques affichées
  égalent les données servies.
- **Garde-fou de publication** : `test_coverage_2027.py` exécute le vrai JS du site et vérifie
  que les circos hors nomenclature sont bien grisées, sans score ni siège annoncés ; la parité
  du marquage Python ↔ JS est couverte par `test_parity_2027.py`.
- **Sources sondages** : liens en pied de page du site. Baromètre législatif gelé depuis
  oct. 2025 (suivi reporté sur la présidentielle) — présélections = tendance agrégée
  PolitPro / Toute l'Europe (2026).

*Limites* : **19 circonscriptions hors nomenclature de blocs** (§3 bis — 11 après reconstruction) — aucune prévision par
circo n'y est publiable ; géométrie outre-mer/étranger (encarts) moins validée ; part LFI en
sièges bornée par l'arithmétique d'une compétition divisée (≠ répartition d'union négociée) ;
sondages non rafraîchis depuis fin 2025.
