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
- **Niveau national** : posé par l'utilisateur (curseurs), présélections = **sondages publiés du
  1er tour de la présidentielle 2027** (Wikipédia, tous instituts, douze derniers mois, moyenne
  simple des sondages ramenés à 100 sur trois blocs — `scenarios_2027.anchor_from_polls`). C'est
  l'estimateur validé du modèle : pour toutes les législatives d'apprentissage (2007→2022), la
  fenêtre d'un an de l'Étape 1 ne contenait que des sondages présidentiels (la législative suit
  la présidentielle de six semaines) ; son erreur LOO par bloc est celle citée en §4. Le
  baromètre législatif « hypothèse dissolution » est gelé depuis octobre 2025 et ne sert plus.
  `score = national(curseur) + déviation(modèle)`.
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
- **Partage gauche radicale (LFI) / soc-dém** — le niveau national de la part LFI du VOTE
  vient du curseur, initialisé à **k × part de Mélenchon dans le vote de gauche** aux sondages
  présidentiels 2027 (douze derniers mois). La décote k (« candidat → parti », `lfi_pres_discount`)
  est mesurée en validation croisée **à l'horizon de la prévision** : pour les trois
  présidentielles suivies d'un scrutin à gauche divisée (2012→legi 2012, 2017→legi 2017,
  2022→euro 2024), on lit Mélenchon/gauche dans les sondages des douze mois finissant h mois
  avant le 1er tour (h = distance actuelle entre le dernier sondage 2027 et avril 2027, servie
  dans `summary.anchor.horizon_months`) et on le rapporte à la part LFI/gauche du scrutin
  suivant. Mélenchon montant tard dans chaque campagne, k dépend de h : ≈0,9 à sept mois
  (plis 0,76–1,02 ; RMSE LOO 0,05 contre 0,15 pour une moyenne plate), ≈0,75 à la veille du
  vote. La décote est recalculée à chaque reconstruction, donc glisse avec la campagne. Trois
  plis : une direction validée, une précision mince. Le **profil local vient du premier tour
  de la présidentielle 2022**, dernière présidentielle disponible avant la cible 2027 dans
  les données (`radical_spatial.select_source`). Il mesure les voix de **Mélenchon parmi
  l'ensemble des voix de gauche**, puis leur écart à la moyenne pondérée par ces voix ; ce
  n'est pas une mesure directe du vote législatif LFI. Les européennes 2024 ne sont pas la
  source de ce profil. Les résultats présidentiels s'arrêtent à la **commune** : 537 des 577
  circonscriptions sont mesurées sur les communes entièrement contenues en elles ; les 40
  autres — les 18 de Paris, les 7 de Marseille, celles de Lyon, Nice, Toulouse, Montpellier,
  Nantes, Strasbourg, Bordeaux, Tours, Saint-Étienne, Toulon, Boulogne-Billancourt et
  Saint-Denis de La Réunion — sont entièrement *à l'intérieur* d'une grande commune que le
  scrutin ne découpe pas et reçoivent l'écart de **cette commune**, identique pour toutes les
  circonscriptions de la ville : exact pour la commune, muet sur les écarts internes. Ces
  villes votant Mélenchon nettement plus que la moyenne dans la gauche (Paris 71 %, Marseille
  79 %, contre 68 % en France), c'est un repère plus juste que l'écart nul (part nationale)
  servi auparavant.
  Formule : `part locale = borne(part nationale + RAD_GAIN × écart local, 0,05, 0,95)` ;
  dispersion appliquée telle quelle (RAD_GAIN 1,0). La part de
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

## 3 ter. Négociation des circonscriptions (page « Négocier », vu de LFI)
- **La question** : dans une gauche unie (une candidature par circo), quelles circos LFI
  doit-elle demander pour élire le plus de député·es ? Le score de jouabilité (§2) dit où *la
  gauche* peut gagner, sans connaître l'étiquette. Tout est vu de LFI : aucun autre parti n'entre
  dans un calcul. Quatre prédictions du modèle par circo (`negotiation_2027.py`,
  `data/negotiation.json`), et rien d'autre à gauche du tableau : **p_lfi** = P(siège avec une
  candidature LFI) — la valeur ; **q_lfi** = P(LFI seule se qualifie au 2nd tour si la gauche se
  divise) — l'option extérieure de LFI ; **q_oth** = P(le reste de la gauche seul se qualifie,
  même division) — celle du partenaire, mesurée à l'identique ; **p_left** = P(siège avec la
  candidature d'union MOYENNE, report moyen 2024, étiquette quelconque) — la valeur du siège
  pour l'union, affichée mais absente de la règle des postures. La **part de Mélenchon**
  dans le vote de gauche (présidentielle 2022, part brute, moyenne nationale ~68 %) est servie à
  DROITE, comme argument visible par tous : elle n'entre dans aucune posture.
- **Report vers une candidature LFI, MESURÉ** (`label_effect_2024.py`) : le parti de chaque
  candidat·e d'union 2024 est connu par la répartition des circos du NFP (data.gouv, 546 circos :
  FI 229, PS 175, écologistes 92, PCF 50). Dans les **142 duels** union–RN de 2024 (centre-droit
  éliminé), un·e candidat·e LFI a récupéré **53 %** des voix libérées contre 59 % pour le·la
  candidat·e moyen·ne de l'union (écart −0,05 ; vs non-LFI −0,08, IC 95 % bootstrap [−0,11 ;
  −0,06] ; −1,4 pt d'inscrits de marge à marge de 1er tour égale ; présent dans les trois
  terciles de force de la gauche). Un **taux** de report, insensible au fait que LFI ait reçu des
  circos plus dures. Injecté dans `seat_winner` comme décalage additif `cd2l_delta` du taux
  centre-droit → gauche ; 0 = le modèle moyen de la carte, calibré sur les 109 sièges RN de 2024.
- **Incertitude nationale** : 6 000 tirages Monte-Carlo. Le niveau national n'est PAS tiré
  autour de l'ancre brute. Sur les 6 législatives T1 mesurées (2002→2024, erreurs LOO de
  `bayesian_polls`, convention prédit − réel), les sondages ont sur-estimé l'extrême droite de
  +4,1 pts en moyenne et sous-estimé le centre-droit de 3,5 — une erreur de CENTRE, qu'aucune
  largeur d'intervalle ne rattrape. Elle est donc corrigée, mais **rétractée vers zéro** par un
  facteur scalaire λ = 0,36 (Efron–Morris : la part de ‖biais‖² qui subsiste une fois retiré le
  bruit d'échantillonnage tr(Σ)/n ; si les biais ne dépassent pas leur propre bruit, λ = 0 et
  rien n'est corrigé). Six scrutins ne suffisent pas à parier sur le chiffre brut, et 2024 —
  hors échantillon — est parti dans l'autre sens. Correction appliquée : ED −1,5 · C+D +1,2 ·
  G +0,2 pt. Autour de ce centre corrigé, l'erreur est tirée dans sa **loi mesurée** :
  écart-type G 6,5 · C+D 6,7 · ED 5,8 pts, et les corrélations réelles entre blocs
  (G/C+D −0,61 · G/ED −0,41 · C+D/ED −0,47) — une part surestimée est prise à une autre, et
  c'est ce partage qui décide des qualifications au 2<sup>d</sup> tour. Les erreurs sont
  exprimées en parts des trois blocs, donc somment à zéro : le tirage respecte le total imposé
  **sans renormalisation**, et ce qui sort vaut exactement ce qui est visé (contrôlé à chaque
  build). La version précédente tirait les trois blocs indépendamment puis renormalisait : cette
  projection rabotait 14 à 22 % de la dispersion et rendait toutes les paires à peu près
  également anticorrélées, là où gauche et centre-droit le sont de loin le plus. Voir
  `poll_error_model.py`.
- **Incertitude locale** : bruit par circo (σ = demi-largeur circo 90 % / 1,645 : G 4,6 ·
  C+D 5,9 · ED 3,7 pts). Abstention et bloc « Autre » tenus fixes à la référence du scénario
  (γ = identité) : ils ne sont ni dans le chiffre ni dans les bornes.
- **Bornes affichées** : sous chaque probabilité, son intervalle de Wilson à 95 % sur les
  6 000 tirages (±1,3 pt au plus large). C'est l'erreur de **simulation** et elle seule — de
  combien le chiffre bougerait à graine différente. L'incertitude de l'**élection** (sondages,
  erreur locale) est déjà dans le point estimé ; la remettre autour la compterait deux fois.
  Ces bornes valent pour un chiffre **pris seul** : les quatre colonnes sortent des mêmes
  tirages, donc un écart entre deux d'entre elles est bien mieux connu — l'entête du CSV donne
  le chiffre exact, servi et non écrit à la main. Conséquence sur le rang : une circonscription
  est interchangeable avec une dizaine de voisines, au bruit de simulation près (critère sur
  l'ÉCART, `rank_blur`, pas sur les bornes marginales — les circos partagent les tirages
  nationaux, donc la variance de l'écart demande leur covariance).
- **Groupes** : *acquis* = député·e sortant·e LFI (71 ; `deputes_an.py`, open data AN) ; *gauche
  hors union* = siège pris en 2024 par une candidature codée à gauche mais hors union (DVG,
  régionaliste…) dont le titulaire ne siège pas dans un groupe de gauche (La Rochelle/Falorni,
  Orthez/Habib, Les Abymes/Serva…) : le bloc de gauche prédit inclut ses voix, qui ne se
  reporteraient pas sur une candidature d'union → sorti du classement, signalé (7) ; *sans
  enjeu* = p_lfi < 5 % ; *en jeu* = le reste, classé par p_lfi décroissant ; *non mesurée* (§3 bis).
- **Valeur, options extérieures, postures** : pour une grille de parts nationales
  LFI-dans-la-gauche (25→55 %, ancre sondages présidentiels décotée, curseur de la page), le
  modèle rejoue une gauche DIVISÉE (LFI d'un côté, PS·PP·Écologistes·PCF de l'autre, motif
  Mélenchon, `split_outcome`) et sert **deux options extérieures mesurées à l'identique sur les
  deux pôles** : `q_lfi` (LFI seule atteint le 2nd tour) et `q_oth` (le reste de la gauche seul
  l'atteint). Avec `p_lfi` (LFI gagne le siège en portant la candidature unique), ce sont les
  **trois probabilités du même Monte-Carlo** dont se déduisent les postures, et rien d'autre :
  *rien à jouer* (p_lfi < 5 %, c'est-à-dire le groupe « sans enjeu » : LFI ne gagne pas ce siège,
  rien à demander ni à céder) ; *exiger* (p_lfi ≥ 5 % et q_lfi ≥ 50 % : LFI tient le siège sans
  l'accord) ; *monnaie d'échange* (p_lfi ≥ 5 %, q_lfi < 50 % ≤ q_oth : l'option extérieure est du
  côté du partenaire, LFI ne peut pas exiger ce siège et devra le céder — c'est un vrai siège,
  donc la concession a un prix) ; *obtenir* (p_lfi ≥ 5 % et aucun des deux pôles ne tient le
  siège seul : personne ne peut se passer de l'accord, il se gagne à la table). Aucun seuil
  nouveau : 5 % et 50 % sont ceux déjà posés, et le 5 % est **le même** que celui du groupe, de
  sorte que classement et posture ne peuvent pas se contredire. `p_left` (la gauche unie gagne le
  siège, candidature d'union moyenne) est **affichée sans entrer dans la règle** : elle situe la
  valeur du siège pour l'union. Ce n'est pas la chance du partenaire — aucune chance n'est
  calculée pour un autre parti. Le survol d'une pastille redonne les chiffres de la ligne, tous
  présents en colonne.
  *Correction (2026-09-18, seconde).* La règle testait `p_left < 5 %` pour « rien à jouer » alors
  que le groupe testait `p_lfi < 5 %`. Les deux quantités ne diffèrent que de l'écart d'étiquette,
  plus petit que le bruit de simulation : 8 circonscriptions tombaient de part et d'autre et
  affichaient « obtenir » (revendiquer) tout en étant classées imprenables, argumentaire compris.
  Le garde-fou aligne la posture sur le groupe. `seat_winner` étant croissante en `cd2l_delta` et
  les deux probabilités sortant des mêmes tirages, p_lfi ≤ p_left partout (invariant testé) :
  `p_left < 5 %` était donc devenu du code mort, et a été supprimé de la règle. Les postures de
  demande vivent désormais **exclusivement** dans le groupe « en jeu », vérifié à chaque cran du
  curseur de part nationale.
  *Correction (2026-09-18).* La règle précédente opposait `p_lfi` (candidature LFI) à `p_left`
  (candidature d'union moyenne). L'effet d'étiquette mesuré sur 2024 étant faible, ces deux
  quantités ne se séparent que de 0,6 pt en médiane et 3,0 pts au maximum : « monnaie d'échange »
  n'y capturait que 8 circonscriptions à cheval sur le seuil de 5 %, toutes ingagnables (gauche
  23-26 % contre RN 44-51 %) — un artefact d'arrondi, pas une catégorie politique. Les deux
  options extérieures, elles, se séparent vraiment. Un test interdit le retour en arrière.
  Vérification externe : les 108 circonscriptions « monnaie d'échange » portent très
  majoritairement une investiture NFP 2024 du partenaire (55 PS, 25 Écologistes, 8 PCF, 19 LFI).
- **Ordre de lecture** : par p_lfi décroissant, parce que ce qui se négocie est un NOMBRE de
  circos et qu'à nombre donné chaque circo vaut pour LFI exactement sa chance d'y élire un·e
  député·e. Aucun score composite : il cacherait le raisonnement.
- **Deux groupes de colonnes, un seul tableau** : à gauche *notre lecture* (posture, p_lfi,
  p_left, q_lfi, q_oth — que des probabilités simulées) ; à droite *les chiffres à mettre sur la table* : sortant·e, parti NFP et sort du siège
  2024, gauche 2024 **ventilée par nuance** (UG = candidature NFP avec son parti ; DVG/EXG/ECO…
  = gauche hors NFP), LFI seule 2017 (nuance FI), écart Mélenchon, extrapolation « 2024 +
  évolution nationale » (4 blocs, plancher 0, renormalisation) **répartie entre LFI, PS,
  Écologistes et PCF** : LFI par la règle de la force réelle (part nationale de LFI dans la
  gauche, filtre, ancre sondages par défaut, + RAD_GAIN × écart Mélenchon 2022, bornée
  [0,05 ; 0,95]) ; le reste selon la seule enquête législative qui sépare PS/EELV/PCF (Ifop
  3-4 juin 2025 : 12/5/3, `data/polls/legislatives/legislatives_2027_partis_gauche.csv`, hors
  ancre du modèle), chaque part décalée de l'écart local de son·sa candidat·e présidentiel·le 2022
  (Hidalgo, Jadot, Roussel ; `radical_spatial.left_presidential_shares`), bornée [0,02 ; 0,96],
  renormalisée. Aucune sortie du modèle à droite : le score de gauche prévu ne
  reste que dans le CSV. Vue par défaut : toutes les circonscriptions (« rien à jouer »
  recouvre exactement le groupe « sans enjeu » ; « exiger », « obtenir » et « monnaie d'échange »
  vivent exclusivement dans le groupe « en jeu »). Le partage LFI / reste du calcul simple est un
  chiffre de DROITE (argument), pas une entrée du modèle ni de la posture. Colonnes redimensionnables ; le CSV porte les colonnes détaillées (sensibilités
  « sondages exacts » et « droites unies », gauche 2022/2017, extrapolation en %).
- **Garde-fous** : `test_negotiation_2027.py` (partition des groupes, postures reproductibles
  depuis les valeurs servies, q_lfi croissant avec la part LFI, la courbe ne contient que les
  circos en jeu, aucun champ partenaire servi, la page ne fige aucun chiffre) ;
  `test_negotiation_page_2027.py` (rendu réel : 577 lignes, postures survolables miroir du
  Python, curseur de part LFI, tri, colonnes redimensionnables, export CSV, 0 erreur JS, pas de
  défilement horizontal à 1440 px).

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
- **Sources sondages** : `data/polls/presidentielle/2027/presidentielle_2027_t1_tidy.csv`
  (page Wikipédia des sondages présidentiels 2027, relevée par `src/scrape_pres_2027.py` ; la
  date du relevé est en tête du fichier), fenêtre et effectifs servis dans `summary.json`
  (`anchor`). Séries historiques 2012 et 2017 (mêmes pages Wikipédia, `src/scrape_pres_history.py`)
  et nsppolls 2022 pour la décote k(h). Le baromètre législatif 2025 n'est conservé qu'en rappel.

*Limites* : **19 circonscriptions hors nomenclature de blocs** (§3 bis — 11 après reconstruction) — aucune prévision par
circo n'y est publiable ; géométrie outre-mer/étranger (encarts) moins validée ; part LFI en
sièges bornée par l'arithmétique d'une compétition divisée (la répartition d'union négociée se lit
sur la page « Négocier », §3 ter) ; report LFI mesuré sur une seule élection (2024) ;
l'ancre présidentielle hérite du biais connu des sondages présidentiels sur une législative
(extrême droite surestimée dans les cinq plis historiques, +2,7 à +11,6 pts) — l'erreur servie
au Monte-Carlo le couvre, aucune correction de biais n'est appliquée (hors ensemble validé).
