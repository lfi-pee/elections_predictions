# Plan du livrable — brief autonome (à lire en entier avant de produire)

> Ce document est la spécification complète. Il encode le destinataire, la
> philosophie, les contraintes dures et les **erreurs de jugement déjà commises
> à ne pas refaire**. Un agent qui démarre en contexte neuf doit pouvoir produire
> le bon livrable à partir de ce seul fichier.

---

## 1. Destinataire & nature du livrable

Le client est **un parti politique** (décideurs de campagne, pas des
statisticiens). Le livrable est **un site web unique, en français** — une page
qui se parcourt d'une seule descente d'écran. Plus de PDF : le produit qu'on vend
est **un instrument de décision qu'on peut interroger**, et un PDF ne peut pas
être interrogé — il ne peut que *représenter* une interrogation. Le médium doit
*être* le produit. C'est un objet de persuasion stratégique, premium, slick —
mais c'est désormais un objet **vivant** : on bouge le national, les 69 000
bureaux répondent sous les yeux du décideur.

## 2. Objectif unique

Le lecteur doit refermer l'onglet en pensant : **« il me faut leur algorithme. »**
Tout sert cela. On ne vend pas « un modèle précis » — on vend un instrument qui
dit des choses qu'aucun sondage national ne sait dire, **au bureau de vote près**,
et qu'on peut *interroger en direct*.

## 3. Philosophie : l'honnêteté est le moteur du « waouh »

Deux exigences tenues ensemble, et la première nourrit la seconde :

- **Honnêteté absolue.** Chaque chiffre est réel et vérifié dans les données
  avant d'être écrit. Chaque limite est assumée. La rigueur (aucun réglage sur la
  validation, intervalles à garantie de couverture stratifiés par territoire,
  modes d'échec documentés) *est* l'argument de vente — c'est ce qui manque aux
  prévisions des instituts. Sur le web, l'honnêteté devient tangible : des
  **calques** « là où on est le moins sûr » et des pastilles de couverture qu'on
  peut afficher d'un clic font *partie* de la démonstration.
- **Effet « waouh ».** Le rendu doit être le plus slick et convaincant possible.
  Le « waouh » vient de *montrer* et de *laisser manipuler* (carte zoomable,
  curseurs de scénario, panneau d'explication au clic), jamais de surpromettre.
  On persuade par la preuve interrogeable, pas par l'emphase.

Posture de production : ingénieur principal senior — concis, on planifie avant de
coder, on vérifie avant d'affirmer.

## 4. Contraintes dures (non négociables)

1. **Court : une seule descente d'écran.** Densité d'information ≈ deux pages, mais
   on porte beaucoup plus *parce que* le détail se range dans les états
   d'interaction (survol, clic, zoom, calque) au lieu de coûter de l'encre. Pas de
   scrollytelling fleuve : trois bandes verticales, point final.
2. **Aucune liste, aucune puce, aucune énumération à puces dans le rendu.** Le
   texte est un liant court, en phrases, entre les visuels. *(Cette règle vaut
   pour le site rendu, pas pour le présent brief.)*
3. **Dense mais lisible.** Une douzaine de panneaux composés en grille magazine
   pleine largeur (1400 px+) ; chaque panneau petit, titré d'une ligne, une idée.
4. **Jamais 69 000 bureaux jetés d'un coup à l'écran.** La règle « agréger +
   zoomer » du print survit ici sous forme de **niveau de détail** : choroplèthe
   agrégé (commune/canton) dézoomé, dissolution vers les polygones bureau par
   bureau en zoomant. On ne charge jamais plus que le viewport (tuiles
   vectorielles, voir §8). On entre par **recherche** ou par zoom, pas par un
   nuage national illisible.
5. **Langue : français** — texte et étiquettes de figures.
6. **Décision avant technique.** L'accroche est en langage de décideur
   (« bloc en tête correctement appelé dans 82 % des bureaux »), pas en R².
7. **Périmètre strict.** Le volet « archétypes de communes / score
   Gauche × Abstention » vient d'un **autre projet** : exclu, et déjà **supprimé
   du dépôt** (`vis_*.png` retirés). Ne pas le réintroduire. Tout ce que montre le
   site sort du seul modèle prédictif et de ses sorties.

## 5. Erreurs de jugement déjà commises — NE PAS refaire

- ❌ Avoir gardé le PDF comme livrable. → **Le PDF combattait le pitch : on vend
  un instrument interrogeable, le web l'incarne. Le companion QR du plan
  précédent était le symptôme — la démo vivante devient la page elle-même.**
- ❌ Avoir compris « 20 pages ». → **C'est court : une descente d'écran.**
- ❌ Avoir intégré le projet archétypes / Gauche × Abstention. → **Autre projet,
  exclu et supprimé.**
- ❌ Avoir proposé trop peu de visuels et trop de texte. → **Densité de
  data-journalisme, ~12 panneaux, prose minimale.**
- ❌ Avoir voulu une carte nationale de tous les bureaux. → **Niveau de détail :
  agréger au dézoom, dissoudre au zoom, entrer par recherche.**
- ❌ Avoir mis en avant le R² et la rigueur technique d'abord. → **Mettre en avant
  l'action et le langage de décision ; le R² est une note de bas de page.**
- ❌ Avoir utilisé des listes / structure en puces dans le rendu. → **Zéro liste.**
- ❌ Avoir affirmé des chiffres avant de les calculer. → **Vérifier dans les
  données d'abord (voir §6).**
- ❌ Avoir confondu « interactif » et « long ». → **L'interactivité achète de la
  densité sans coûter de la hauteur ; rester court reste la discipline.**
- ❌ Avoir fait de la carte le héros permanent des trois bandes. → **Elle mangeait
  ~60 % en continu et étouffait les autres vizs. La carte domine en Bande 1 puis
  se docke en rail collant au scroll : elle reste vivante mais rend la place aux
  panneaux (voir §7, principe de mise en page).**
- ❌ Avoir gardé des panneaux de statisticien (hexbin prédit-vs-réel, barres de
  couverture conforme, barres de R²) comme preuves de Bande 1. → **Le décideur ne
  lit pas un nuage hexbin ni un R². Hexbin remplacé par la *précision-par-marge*
  (le bon appel est quasi certain quand l'écart est net, plus dur sur le terrain
  serré — preuve ET récit du combat en une image) ; couverture conforme démotée en
  pastille numérique compacte (« on annonce 90 %, le réel y tombe ≥ 93,5 % ») ; R²
  réduit à une note dans le panneau d'honnêteté et la légende de méthode.**
- ❌ Avoir enterré l'argument de vente. → **« D'où vient l'incertitude » (part
  locale vs national) était un calque basculable en bas de Bande 3 alors que c'est
  *le* fossé (« pourquoi pas juste un sondage ? »). Promu en panneau de Bande 1,
  intitulé « Pas un sondage déguisé ».**
- ❌ Avoir donné à l'Extrême Droite un quasi-noir (`#3A3A4A`) et à l'Abstention un
  gris : sur la choroplèthe, deux blocs sur quatre lisaient comme « donnée
  manquante », et ED est souvent en tête. → **ED passe à un violet saturé
  (`#6A4C93`), distinct du bleu C+D, de l'orange Gauche et du gris Abstention.**
- ❌ Avoir promis un brushing pleine page (brosser un nuage filtre tout). → **Geste
  invisible pour ce public, source de bugs d'état. On garde le seul lien intuitif
  carte ↔ panneau (survol d'un panneau → la carte se recadre/surligne).**
- ❌ Avoir calculé le point de bascule par bureau sans jamais l'assembler en
  *cible*. → **Le pitch vend un instrument de décision : on ajoute un calque
  « cibles » (Bande 2) qui éclaire les bureaux qu'un effort national plausible
  (+4 pts ED) ferait passer en tête, chiffrés en bureaux et en électeurs — le
  « où agir » que rien d'autre ne donne.**
- ❌ Avoir laissé le décideur sans réponse **en circonscriptions**. → **Tout le
  produit parlait « bureaux basculés » alors qu'un directeur de campagne compte en
  sièges. Le code de circonscription est *nul* dans le scrutin 2024 — c'est pourquoi
  il dormait. Correctif honnête : on reprend la carte bureau→circonscription du
  législatif 2022 (`CIRCO_SRC`, découpage stable depuis 2012, 99,9 % des bureaux
  couverts) et on remonte chaque bureau à sa circonscription. Le panneau « rapport de
  force » (Bande 3) rend le bloc *dominant au premier tour, sur l'agrégat des bureaux*
  de chacune des 577 circonscriptions — explicitement pas une projection de siège à
  deux tours (le modèle ne voit que le premier tour, `T1_TYPES` « T2 has runoff
  dynamics » : ni qualification, ni désistement, ni report de voix) : la rigueur est de
  le dire. Le même cadran national déplace les bureaux *et* les circonscriptions, en
  direct.**
- ⚠️ Avoir d'abord remplacé la **mini-cascade SHAP** par une **phrase de décideur**
  (« L'Extrême Droite ressort ici surtout par son vote Extrême Droite passé, tempéré
  par le chômage »). → **Rejeté par le client : faute de chiffres, la phrase ne se
  décide pas et lisse les variables. On **rétablit la viz quantitative** des
  contributions — sans le défaut d'origine (noms bruts type « Gauche lag1 »). Chaque
  bureau sert ses **6 contributions dominantes** (top-6 SHAP du bloc en tête) en
  **barres divergentes signées**, libellées en clair par `report_shap.pretty_label`
  (« Vote Gauche (n-1) », « Chômage », « Diplômés du supérieur », « Position
  (latitude) »…) et chiffrées en **points d'écart au national** — exactement la
  grandeur que le panneau explique. Le piège n'était pas le *quantitatif* (le décideur
  veut le chiffre) mais le *jargon* : on garde l'un, on tue l'autre.**
- ❌ Avoir gardé « Le paysage national » en **panneau autonome**. → **Il redisait en
  agrégat ce que la carte montre déjà. Fondu dans le panneau « Pas un sondage déguisé »
  sous la forme d'une barre « la réalité, bureau par bureau » (partage G/CD/ED), qui
  *renforce* le duel au lieu d'occuper une case pour rien. Le duel passe en `span8`,
  héros argumentaire de la Bande 1.**
- ❌ Avoir **fragmenté l'honnêteté** en quatre petits objets dispersés (calque
  « moins sûr », pastille de couverture, frise de méthode, notes de R²) alors que le §3
  en fait *le moteur du waouh*. → **Les deux panneaux d'honnêteté de la Bande 3 fusionnent
  en un seul « Notre honnêteté » (`span12`) : calque d'incertitude + pastille de
  couverture côte à côte, un seul moment décisif.**
- ❌ Avoir tourné le cadran de scénario en *ajoutant* des points à un bloc sans en
  retirer aux autres. → **Les blocs G/CD/ED sont des parts de suffrages exprimés et
  somment à ~100 % par bureau (l'abstention est un axe distinct) ; pousser ED de
  +3 sans rien retirer fabriquait 103 % de votes — physiquement impossible, et en
  contradiction frontale avec « honnêteté absolue ». Correctif : le swing est
  *conservé* — `appliedⱼ = dⱼ − Σ_{k≠j} d_k·wⱼ/(1−w_k)`, avec `w` = parts nationales
  des blocs. Un gain d'un bloc est financé par les autres au prorata de leur taille ;
  les trois deltas somment à 0. Même formule dans `report_data.conserved` (Python,
  source de vérité du compteur et de `ed_tip`) et dans `config.js appliedDeltas`
  (carte + compteur client). Conséquence : les chiffres montent (+3 ED → 6 923 bascules
  de bureau et non 4 624), parce qu'à mesure qu'ED progresse ses rivaux reculent — la
  cohérence interne est exacte. (Le compte « bureaux basculés » a depuis cédé sa place
  de *cible* aux électeurs convaincables — erreur suivante ; la conservation du swing
  reste vraie pour le stress-test ED de la Bande 3.)**
- ❌ Avoir vendu la cible comme un **compte de bureaux basculés**. → **Le client est un
  parti de gauche : un bureau n'est pas un siège (le siège se joue à la circonscription,
  à deux tours), et un bureau basculé n'est pas un électeur gagné. La bonne unité est
  l'**électeur individuel convaincable**. Le total national d'un basculement est de la
  pure arithmétique (+1 pt national ≈ 0,48 M voix) — le modèle ne l'invente pas ; son
  apport unique est la **répartition spatiale** de chaque électeur gagnable. La Bande 2
  cesse de compter des bureaux et chiffre **un seul gisement défendable** : la
  **mobilisation** (abstentionnistes qui penchent à gauche, voir bullet suivant). Le
  scénario ED (Bande 3) reste, mais comme **stress-test de marée adverse**, plus comme
  cible.**
- ❌ Avoir chiffré la gauche des abstentionnistes par le **partage des exprimés locaux**
  (circulaire), *puis* par la **propension démographique** (attente SHAP du modèle Gauche,
  `report_potential`). → **Les deux sont insuffisants. Le partage des exprimés est
  circulaire ; la propension démographique est une attente de niveau *exprimé*, sans
  rapport avec la façon dont penche l'abstentionniste *marginal*, et sans identification.
  L'investigation `MOVABILITY.md` (§11) a mesuré la seule quantité **identifiée et
  stable** : **γ = part de gauche du votant marginal**, lue sur les vraies hausses de
  participation (`movability_turnout`). Mobilisables(b) = abstentionnistes(b) × γ(b),
  jamais γ = part exprimée. Le « potentiel latent » (résidu jumeaux) et le canal
  persuasion β ont été testés et **abandonnés** : non identifiés / non stables (§2, §10).
  `report_potential.py` et `potential.parquet` supprimés.**
- ❌ Avoir rendu les **circonscriptions seulement en agrégat** (deux compteurs, deux
  barres empilées) sans jamais dire *lesquelles*. → **Le décideur compte en sièges ; un
  compteur « 75 basculent » sans liste nommée n'est pas interrogeable. Ajout d'un panneau
  « les circonscriptions sur le fil » (`span12`, Bande 3) : les sièges à la marge la plus
  mince sous le scénario courant, *nommés* par leur commune la plus peuplée (`circo.json`
  champs `id`/`nm`, ancre calculée dans `report_data.circo_rollup`), réordonnés en direct,
  les basculés surlignés. Marge / bloc / bascule recalculés client depuis les parts + swing
  conservé. Le compteur dit *combien*, la liste dit *lesquelles*.**
- ❌ Avoir laissé la **courbe de bascule aveugle aux sièges** (bureaux seuls) alors que le
  climax vend « un cadran, bureaux *et* circonscriptions ». → **Courbe à deux unités :
  bureaux (trait plein, échelle de gauche) et circonscriptions (pointillé violet, échelle
  de droite) superposés sur le même balayage −4…+6 pts, avec axes Y chiffrés (l'ancienne
  courbe n'avait aucune échelle Y). Le « un geste, deux unités » devient littéral.**
- ❌ Avoir laissé la **conservation du swing invisible** : trois curseurs indépendants,
  la somme-nulle absorbée en silence par le calcul. → **Le §5 fait de l'honnêteté le
  moteur du waouh, donc la conservation doit se *voir*. Témoin « effet net, à somme nulle »
  sous les curseurs : +3 ED y affiche « G −1,5 · CD −1,5 · ED +3,0 » — le décideur voit
  qu'un bloc est financé par les autres, jamais conjuré.**
- ❌ Avoir fait de la carte un **score de partis** (« bloc en tête, gauche vs autres ») par
  défaut. → **Le client est un parti de gauche : ce qui l'intéresse est le gisement
  mobilisable, pas le horse-race. La carte colore désormais **par défaut la mobilisation**
  (abstentionnistes × γ, teinte Gauche pâle→saturé) ; le **bloc en tête devient un calque
  optionnel** (case « afficher le bloc en tête — le duel sondage vs nous » dans le hero, plus
  le calque honnêteté). Le **survol** et la **légende** n'affichent plus un score de partis
  mais *expliquent le score de mobilisation* : `mv = abstentionnistes × γ`, où γ est la part
  de gauche du votant marginal — et la base d'abstentionnistes est la **réelle** (`ab`/`cab`,
  `act_AB`), cohérente avec le `mv` du modèle (pas l'abstention prédite, qui décalait le γ
  affiché de 2-3 pts). Le panneau au clic le décompose **avec le SHAP du modèle Gauche**
  (« pourquoi ce bureau est mobilisable » : ce qui règle son niveau de gauche, donc γ), en
  phrase directionnelle honnête (un bureau de droite « voit son niveau de gauche tiré vers le
  bas par… », jamais « penche à gauche par… »). La légende (`#legend`) devient *pilotée par le
  mode*.**

---

## 6. Faits vérifiés dans les données (socle du document)

Calculés sur `data/predictions_with_intervals.csv` (validation 2024 Législatives
T1, une seule passe, jamais ré-ajustée). À recalculer/citer tels quels.

- **Bloc en tête correctement appelé dans 81,6 % des 69 358 bureaux.** C'est
  l'accroche. Le R² par bloc (Gauche 0,74 · Centre+Droite 0,61 · Extrême Droite
  0,80 · Abstention 0,74) passe en appui discret.
- **Terrain du combat :** 11 080 bureaux à marge < 3 pts (16,0 %), **17 841 à
  marge < 5 pts (25,7 %)**, 26 968 à marge < 8 pts (38,9 %). Terrain fini,
  nommable, à chiffrer aussi en électeurs via `inscrits`.
- **La précision suit la marge (preuve honnête, pas un hexbin) :** le bon appel
  monte avec l'écart bloc 1 − bloc 2 — **47 % sous 1 pt, 67 % à 3–5 pts, 86 % à
  8–12 pts, 95 % au-delà de 12 pts.** Une seule image porte l'accroche (81,6 %),
  l'aveu (on est plus faible sur le serré) et le récit du combat (le serré, c'est
  le terrain). Calculée dans `report_data.accuracy_by_margin`, tracée client.
- **Électeurs convaincables — mobilisation (le seul gisement défendable) :**
  abstentionnistes qui penchent à gauche = abstentionnistes × **γ(b)**, où **γ est la
  part de gauche du *votant marginal*** — l'ex-abstentionniste qui se déplace quand la
  participation monte — lue sur les vraies hausses de participation par décile de niveau
  de gauche (`movability_turnout`, voir `MOVABILITY.md` §11). **4,71 M en métropole**,
  sur 14,8 M d'abstentionnistes (γ moyen 31,7 %) — abstention et niveau de gauche
  **prédits** (`pred_*`, pas les résultats observés : le livrable montre ce que
  l'instrument *prévoit*, pas ce qu'on a vu après coup) — des non-votants qu'aucun
  sondage (qui n'interroge que les votants probables) ne voit. Quantité **identifiée**
  (pas le partage
  des exprimés locaux, circulaire) et **stable dans le temps** (courbe γ ancienne vs
  récente corrélée à +0,96). **Validée hors échantillon** : la courbe γ(niveau) bat
  l'hypothèse de swing uniforme pour prédire le mouvement du scrutin suivant, et les
  variantes ML plus riches (γ par démographie / lags — GBM, forêt, linéaire — ou par
  historique propre du bureau) ont **toutes été testées et écartées** : elles décrivent
  finement l'élection sur laquelle on les ajuste mais **ne transfèrent pas** au scrutin
  suivant — or l'objectif est de *prédire* (`MOVABILITY.md` §13). Calculé dans
  `report_targets.left_gain`, champ `summary.left_gain`.
- **« Potentiel latent » abandonné (verdict honnête) :** l'ancien gisement (attente
  démographique − réalisé) était un **résidu non identifié, de signe ambigu** (un bureau
  sous ses jumeaux a *durablement* refusé sa démographie → le moins mobilisable, pas le
  plus) qui ne repose sur **aucun signal spatial vérifiable** (`MOVABILITY.md` §2/§10).
  Retiré plutôt que maquillé : un gisement spatial inventé serait la malhonnêteté même que
  le §3 proscrit. Le canal *persuasion* (chargement β sur la marée) a aussi été testé et
  **échoue hors échantillon** — on ne bat pas le swing uniforme (`MOVABILITY.md` §10).
- **Où déployer (en électeurs, pas en bureaux, métropole seule) :** le gisement de
  mobilisation se concentre dans des villes nommables — **Paris 134 k, Marseille 74 k,
  Toulouse 31 k, Lyon 27 k, Nice 26 k, Montpellier 24 k, Nantes 23 k, Lille 19 k,
  Strasbourg 18 k, Rennes 15 k, Le Havre 14 k, Reims 13 k.** L'outre-mer et l'étranger sont
  exclus de l'ordre de déploiement (non démarchables, erreur concentrée).
  `summary.left_gain.deployment`, top 12 par abstentionnistes mobilisables.
- **Honnêteté du chiffre :** le total national d'un basculement est arithmétique (+1 pt
  gauche ≈ 0,48 M voix) — le modèle ne le crée pas. Son apport est la *répartition*
  spatiale du gisement. Et l'orientation des abstentionnistes est *modélisée* par γ (lue
  sur des hausses de participation réelles), jamais observée bureau par bureau : on le dit.
- **Le sondage vs nous (l'argument de vente, rendu viscéral) :** un sondage national
  n'a qu'un favori — le même bloc partout (l'Extrême Droite, 33,5 % de moyenne
  pondérée). Ce favori unique ne désigne le bon bloc en tête que dans **48,2 % des
  bureaux** ; notre carte, lue bureau par bureau, y arrive à **81,6 %**. Deux barres
  côte à côte portent l'écart mieux que la part d'incertitude abstraite
  (`summary.flat_poll`, calculé sur les vrais résultats 2024).
- **Moteur de scénarios (l'arme décisive) :** prédiction = moyenne nationale +
  écart local, donc on tourne le cadran national et le local répond. Vérifié :
  **+3 pts d'Extrême Droite au national → 6 923 bureaux basculent de bloc en
  tête (10,0 %).** Aucun sondage ne donne ce « et si… ? » au bureau de vote. Le swing
  est **conservé** : les +3 pts d'ED sont financés par les autres blocs au prorata de
  leur taille (voir §5), jamais ajoutés ex nihilo.
- **Rapport de force en circonscriptions (la monnaie du décideur) :** chaque bureau est
  remonté à sa circonscription (carte bureau→circo du législatif **2022**, découpage
  stable depuis 2012, **99,9 % des 69 358 bureaux couverts**, 577 circonscriptions). Le
  **bloc dominant au premier tour, sur l'agrégat des bureaux** d'une circonscription se
  répartit aujourd'hui en **Gauche 190 · Centre+Droite 166 · Extrême Droite 221**. Le
  même cadran national agit dessus : **+3 pts d'Extrême Droite → 75 circonscriptions
  changent de bloc dominant.** Honnêteté assumée — et cohérente avec le modèle lui-même,
  **entraîné et validé sur le seul premier tour** (`T1_TYPES`, commentaire « T2 has
  runoff dynamics ») : c'est un rapport de force de **premier tour**, **pas une
  projection de siège à deux tours** (aucune qualification, aucun désistement, aucun
  report de voix modélisés) — et on l'écrit. Calculé dans `report_data.circo_rollup`,
  recomposé client en direct depuis `circo.json` (parts agrégées par circonscription).
- **Données sous-jacentes :** 56 élections (1999–2026), 52 indicateurs INSEE,
  16 M d'adresses d'électeurs géocodées → 72 795 bureaux à coordonnées vérifiées
  (94,7 % sources exactes, zéro repli sur le centre de la France). Intervalles
  conformes 80/90/95 % à garantie de couverture, **stratifiés par classe de
  territoire** (≥ 90 % d'actuels dans la bande 90 %).
- **Provenance de l'incertitude (l'argument de vente, chiffré) :** la prédiction
  d'un bureau = chiffre national + écart local, et on sait dire **ce que chaque
  source pèse**. *Par bureau*, la part de l'incertitude qui vient du national
  (sondages) plutôt que de notre lecture locale est de **7 % (Gauche), 16 %
  (Centre+Droite), 48 % (Extrême Droite), 60 % (Abstention)** — le reste, c'est
  l'information de terrain qu'aucun sondage ne porte. *Au national*, c'est
  l'inverse : l'erreur locale s'annule sur 69 000 bureaux, il ne reste que celle
  des sondages — donc on est au pire aussi bon qu'eux, jamais pire. (Mesuré en
  rejouant la calibration conforme avec la vraie moyenne nationale vs l'estimation
  sondagière : `src/conformal.py`, mode oracle vs réaliste.)
- **Limites assumées :** territoires particuliers (DOM-TOM, Corse, étranger)
  concentrent l'erreur ; élections de rupture (2007, 2017) plus dures ;
  l'abstention n'a aucun sondage direct (traitée par un modèle de participation,
  ce qui explique sa forte part d'incertitude « national » ci-dessus).

---

## 7. Maquette du site (une descente, trois bandes)

Bandeau de tête, fixé et discret : titre décision, le **81,6 %** et la légende des
blocs. Le lien entre vues est **intuitif et unique** : les curseurs de scénario
national et les calques (mobilisation, honnêteté) recomposent la
carte en direct, et la carte répond au survol / clic — pas de brushing pleine page
(geste invisible, source de bugs).

**Principe de mise en page — la carte qui se docke (à ne pas oublier).** Une seule
carte MapLibre, persistante et toujours vivante (clic, survol, curseurs continuent
de la piloter), mais qui **change de statut au scroll** au lieu de rester héros
permanent. Erreur précédente : la carte posée en héros des trois bandes mangeait
~60 % en continu et étouffait les autres vizs. Correctif : elle domine **seulement**
en Bande 1 ; dès qu'on scrolle, elle se **replie en rail collant** (sticky, ~30-35 %
de large, hauteur réduite) sur un côté et **rend la colonne principale** aux
panneaux. Elle ne disparaît pas — elle se **subordonne** : reste le **miroir
spatial** de la viz qu'on lit (le panneau survolé recadre/surligne la carte). On
gagne ainsi la place pour une vraie grille magazine **sans rallonger la page** : on
remplit la largeur, pas la hauteur. La discipline « une descente » tient.

### Bande 1 — « La carte que les sondages ne donnent pas » (le héros vivant)

Plein cadre (le **seul** moment où la carte a le droit de dominer), **la carte
MapLibre au bureau de vote**, démarrée **zoomée sur une métropole** (Lyon par
défaut) — mosaïque rue par rue immédiate, preuve de finesse. **Par défaut elle colore
la mobilisation** (abstentionnistes × γ, teinte Gauche pâle→saturé), pas le bloc en
tête : le client est un parti de gauche, le gisement prime sur le horse-race. Le **bloc
en tête** reste accessible en **calque** (case « afficher le bloc en tête — le duel
sondage vs nous » dans le hero). Une **barre de recherche** (« tape une commune ») fait
*voler* la caméra ailleurs ; au dézoom, la carte agrège en choroplèthe commune/canton.
**Survol** d'un bureau → infobulle qui *explique le score affiché* : en mobilisation,
« N électeurs mobilisables · A abstentionnistes × γ % de gauche » plus la phrase SHAP
« pourquoi ce bureau penche (ou non) à gauche » ; en calque bloc, « bloc en tête · marge ».
La **légende** suit le mode. **Clic** → le **panneau d'instrument** (décrit §7-bis). **Au scroll vers la Bande 2, la carte se docke** : la colonne
libérée déroule une grille de preuve en **langage de décideur**, pas de
statisticien — **« La précision suit la marge »** (barres d'exactitude par tranche de
marge en `span4` ; le terrain serré, sous 5 pts, s'allume en orange — l'aveu honnête
qu'on est plus faible là où les blocs se tiennent) et, en **héros argumentaire
(`span8`)**, **« Pas un sondage déguisé »**. Ce dernier ouvre sur une barre **« la réalité, bureau par bureau »**
(partage G/CD/ED des bureaux en tête — le contrepoint au favori unique, qui *remplace*
l'ancien panneau autonome « paysage national »), puis le **duel sondage vs nous** —
deux barres : le favori unique d'un sondage national tape juste dans **48,2 %** des
bureaux, notre carte **81,6 %** — puis détaille d'où vient l'écart (part locale vs
nationale). Une **frise pleine largeur**
(56 élections, 16 M d'adresses, 52 indicateurs, 69 358 bureaux) signe le socle. La
couverture conforme n'est plus un graphe : elle devient une **pastille numérique**
en Bande 3.

### Bande 2 — « Combien d'électeurs, et où » (le gisement de voix)

Colonne principale aux panneaux, **carte en rail collant** à côté (elle réagit, ne
domine plus). La bande ne compte plus des **bureaux** — un bureau n'est ni un siège
ni un électeur — mais des **électeurs convaincables**, en personnes et localisés. Un
**seul calque** (basculable), un seul gisement défendable, son compteur en chiffre
plein :

Le calque **« mobilisation »** allume les **abstentionnistes qui penchent à gauche**
(`national.json` champ `mv`, abstentionnistes × **γ**, la part de gauche du votant
marginal) — **4,71 M en métropole** : des non-votants qu'aucun sondage (qui n'interroge
que les votants probables) ne voit, à faire venir aux urnes. La teinte porte la densité
d'électeurs gagnables ; le rail recadre sur le foyer survolé. *(L'ancien second calque
« potentiel latent » a été retiré : résidu non identifié, sans signal spatial vérifiable
— `MOVABILITY.md` §2/§10.)*

Sous le calque, le panneau pleine largeur **« Où déployer »** classe les
**communes par abstentionnistes mobilisables** (`summary.left_gain.deployment`, top 12) —
**Paris 134 k, Marseille 74 k, Toulouse 31 k, Lyon 27 k, Nice 26 k…** — métropole seule
(outre-mer/étranger exclus : non démarchables). L'ordre de déploiement est nommé et
chiffré en personnes, pas une teinte abstraite. Une note honnête tient en une ligne : le
total national d'un gain est arithmétique (+1 pt ≈ 0,48 M voix) ; ce que le modèle ajoute,
c'est *où* sont ces électeurs — et leur orientation est modélisée par γ (lue sur des
hausses de participation réelles), pas observée bureau par bureau.

### Bande 3 — « Et si… ? » (l'instrument qu'on interroge) — le climax

Les **curseurs de scénario** (G / C+D / ED / Abstention au national) recomposent
**instantanément** les 69 000 bureaux (recomposition client `moyenne nationale
ajustée + écart local`, swing conservé). Sous les curseurs, un **témoin « effet net,
à somme nulle »** rend la conservation *visible* : pousser ED de +3 affiche en clair
« Gauche −1,5 · Centre+Droite −1,5 · Extrême Droite +3,0 » — le décideur voit que le
gain d'un bloc est financé par les autres, jamais conjuré (au repos : « un seul cadran
à somme nulle, tournez un curseur »). **Deux compteurs côte à côte** sur le panneau
sombre : **bureaux basculés** (**6 923** à +3 pts ED, avec bande d'incertitude « ≈ N
bureaux au seuil, fragiles ») **et circonscriptions** dont le bloc dominant change
(**75** à +3 pts ED, sur 577) — le même geste répond en bureaux *et* en sièges. À côté,
un **balayage à deux unités** : faire glisser le national de −4 à +6 pts trace la
**courbe de bascule** où **deux** réponses se superposent — bureaux basculés (trait
plein, échelle de gauche) *et* circonscriptions basculées (pointillé violet, échelle de
droite) — le « un cadran, deux unités » rendu littéral. En regard, le **« rapport de
force »** — deux barres empilées G/CD/ED des circonscriptions, *aujourd'hui* vs *sous
le scénario courant* (**bloc dominant au premier tour**, agrégat des bureaux ; le modèle
ne voit que le T1, donc pas une projection de siège à deux tours — c'est dit) — puis,
en `span12`, **« les circonscriptions sur le fil »** : la liste *nommée* (commune la
plus peuplée de chaque circo, p. ex. *Le Havre · circ. 7*) des sièges à la marge la
plus mince **sous le scénario courant**, réordonnée en direct, celles qui *viennent de
basculer* surlignées. Le compteur dit *combien* ; cette liste dit *lesquelles* — la
monnaie du décideur (le siège) enfin **interrogeable**, pas seulement agrégée.
Le rail collant reflète chaque réglage en direct. L'honnêteté qui referme la bande tient
désormais en **un seul panneau « Notre honnêteté » (`span12`)** : à gauche le calque
carte des intervalles élargis sur les territoires particuliers (R² en note), à droite
la couverture conforme en **pastille numérique** (« on annonce 80/90/95 %, le réel y
tombe ≥ 88,1 / 93,5 / 96,3 % »), pas en graphe. La partition local vs national (promue
en Bande 1) se réannote sur la **moustache d'intervalle** du panneau-instrument (7-bis).
Enfin, juste avant le pied de page, une bande **« Ce qu'il faut retenir »** (`span12`,
trois colonnes, pas une liste à puces) fixe les trois choses à emporter : **81,6 % vs
48,2 %**, le **cadran unique** qui déplace 69 358 bureaux et 577 circonscriptions, et
**où déployer** — la discipline contre la surcharge des ~12 panneaux.

### Pied de page — « Comment ça marche » (la signature de méthode)

**Panneau de méthode (compact), tout en bas du site, après les trois bandes :**
c'est la dernière chose que voit le décideur, la preuve de sérieux qui referme la
démonstration. Un **schéma de méthode** lisible en cinq secondes, qui rend la
rigueur *visible* sans jargon. Deux objets seulement :

- Un **flux en deux temps** dessiné de gauche à droite : *national* (sondages +
  modèle de participation pour l'abstention, qui n'a pas de sondage) **→ + écart
  local** (lecture du bureau : démographie INSEE compressée + héritage du vote
  passé) **→ prédiction du bureau → intervalle conforme**. C'est la même équation
  que le moteur de scénarios (Bande 3) — le décideur voit que le cadran qu'il
  tourne agit sur la première boîte, et que tout le reste est notre apport local.
- Une **frise validation croisée / test** : les scrutins passés (2002–2022) en
  points, **un retiré à tour de rôle** (validation croisée leave-one-election-out :
  c'est ainsi qu'on *choisit* le modèle, jamais sur 2024), puis **un seul point
  distinct, 2024**, séparé, marqué « tenu à l'écart · testé une fois ». L'image
  porte à elle seule l'argument d'honnêteté du §3 : la sélection se fait par
  validation croisée sur le passé, le test 2024 est une passe unique jamais
  réajustée. Le **81,6 %** et les R² par bloc (hors échantillon 2024) se posent ici
  en légende discrète, pas en titre.

Pleine largeur (`span12`), SVG centré plafonné, titré d'une ligne : il referme la
descente sans la rallonger. C'est le raccord du § « d'où vient l'incertitude ? » de
la Bande 3 — local vs national — réexprimé en schéma de méthode.

### 7-bis. Le panneau « interroger l'instrument » (au clic sur un bureau)

Glisse depuis la droite, ne quitte pas la page. En-tête : **commune + n° de BV +
inscrits**. Les **4 blocs** en barres, **prédiction vs réel** côte à côte, avec
**moustaches d'intervalle conforme** (bande 90 %). **Bloc en tête prédit + marge**
sur le 2ᵉ, avec pastille de robustesse fondée sur la marge (« rang net » ≥ 8 pts,
« serré » 3–8, « disputé » < 3) — pas sur le recouvrement des intervalles
marginaux, qui surestime massivement l'incertitude de rang. **Pourquoi *ce* bureau
dévie du national**, en **viz quantitative** : les 6 contributions dominantes du bloc en
tête, en **barres divergentes signées** chiffrées en points d'écart au national, avec
des **libellés lisibles** (`report_shap.pretty_label` : « Vote Gauche (n-1) »,
« Chômage », « Diplômés du supérieur », « Position (latitude) »… — jamais « Gauche
lag1 »). Le décideur voit *combien* chaque facteur pèse et *dans quel sens*, pas une
phrase qui lisse les chiffres. **En mode mobilisation (défaut), le « pourquoi » bascule
sur la mobilisabilité** : section **« Pourquoi ce bureau est mobilisable »** — l'équation
en clair `N électeurs = A abstentionnistes × γ %` (base d'abstentionnistes **réelle**,
cohérente avec le `mv` du modèle), une **phrase de décideur** (« doit son niveau de gauche
surtout à son héritage de vote à gauche et son taux de diplômés », ou pour un bureau de
droite « voit son niveau de gauche tiré vers le bas par… » — `report_shap.explain_left`,
directionnelle et honnête), et les **barres SHAP du modèle Gauche** : ce qui règle le
niveau de gauche du bureau (`gdrivers`), donc γ, donc le gisement. **Réponse au scénario
courant** : ce bureau bascule-t-il sous le réglage des curseurs — et
son **point de bascule propre** (« +4,2 pts ED au national le feraient tomber »),
le seuil nommable qui transforme la carte en instrument.

---

## 8. Production (telle que construite)

Site statique (aucun backend) : **MapLibre GL JS** + fond raster CARTO, via CDN.
L'environnement n'avait ni tippecanoe ni node : plutôt que de risquer une chaîne
de tuiles C++, on tient *l'intention* de PMTiles (jamais 69 000 polygones en
mémoire) avec un schéma de fichiers statiques. Vue nationale = couche
**symboles proportionnels par commune** (35 221 centroïdes, **couleur = mobilisation
par défaut** — bloc en tête en calque —, taille = inscrits) chargée en une fois. Vue
zoomée = **polygones bureau par département**, simplifiés (shapely, tol. ~15 m) et
chargés à la demande (`bv/<dept>.geojson`, 1–3 Mo, ≤ 10 départements gardés en mémoire ;
propriétés `mv` mobilisables, `ab` abstentionnistes **réels** pour le γ du survol, `w`
phrase SHAP « pourquoi mobilisable »). **Recherche**
de commune côté client (index hors-ligne, privé — pas de géocodeur externe).
**Recomposition de scénario 100 % client** : les décalages bruts des curseurs
passent par `appliedDeltas` (swing conservé, §5) qui les transforme en deltas à
somme nulle, puis `fill-color` est recalculé via une expression MapLibre `moyenne +
écart local + delta appliqué`, instantané ; les **compteurs de bascule sont exacts
sur les 69 358 bureaux** via `national.json` (tableaux compacts à 4 décimales,
champs `pg/pc/pe/ins/m`, le point de bascule ED `t` et l'index commune `ci` (legs du
stress-test ED), plus **`mv`** — abstentionnistes mobilisables par bureau
(abstentionnistes × γ), qui alimente le **calque mobilisation** de la Bande 2, ~3,5 Mo). Le **rapport de
force en circonscriptions** se recompose pareil, client, depuis `circo.json` (parts
agrégées `g`/`c`/`e` par circonscription, plus `id` = code `dépt-circ` et `nm` = commune
la plus peuplée de la circo comme ancre lisible — alimente la liste **« sur le fil »** ;
~30 Ko). Marge, bloc en tête et bascules par circonscription se calculent **en direct
côté client** depuis ces parts + les deltas conservés ; la **courbe de bascule** superpose
bureaux (`summary.flip_curve`, Python) et circonscriptions (recomptées client sur le même
balayage). Les poids de swing `w` viennent de
`summary.swing`, identiques côté Python. Vérifié au navigateur (Playwright,
Chromium headless) : zéro erreur JS, +3 pts ED → 6 923 bascules de bureau **et 75
circonscriptions** (swing conservé, identiques à `report_data.flip_curve` /
`circo_rollup`), calque mobilisation réactif (basculable on/off) — identique aux chiffres
Python. **Carte qui se docke (§7)** : une seule
instance MapLibre, conteneur en `position: sticky` dont la taille passe par paliers
selon le scroll (IntersectionObserver, pas de listener coûteux) ; `map.resize()` à
chaque palier — MapLibre garde l'état (caméra, couches, expressions de scénario)
intact, donc la carte reste vivante en rail comme en plein cadre. Aucun coût de
chargement supplémentaire. **Mode par défaut `mobil`** (`config.js APP.state.mode`) :
les couches naissent peintes en mobilisation, le bloc en tête et l'incertitude sont des
calques (`setMode` + `updateLegend`, légende `#legend` pilotée par le mode). **Cache-busting
des données** : `loadJSON` suffixe chaque fichier servi de `?v=APP.DATAV` (bumpé à chaque
changement de données) — un navigateur ne sert jamais un `national.json` / geojson périmé
(le bug « NaN abstentionnistes × 0 % » venait d'un geojson en cache sans le champ `ab`).

Pipeline `src/` (point d'entrée `report_build.py`) : `report_data.py` joint
`predictions_with_intervals.csv` aux inscrits + noms commune/canton/circo
(`general_results.parquet`) et au géocodage, calcule marge bloc 1 / 2, bon appel,
**précision-par-marge**, **duel sondage vs nous** (`flat_poll` : exactitude du favori
national unique vs la nôtre, sur les vrais résultats), **point de bascule ED par
bureau à swing conservé** (`conserved`, §5 — les rivaux reculent quand ED monte, sert
le stress-test ED de la Bande 3), largeur d'intervalle (incertitude) **et rapport de
force en circonscriptions** (`circo_rollup` : carte
bureau→circo reprise du législatif **2022**, `CIRCO_SRC`, agrégat des parts pondéré par
les inscrits → bloc dominant par circonscription, base et bascules au scénario par
défaut) → table maître + agrégats + `summary.json` (dont `swing`, `circo` et
**`left_gain`**) + `national.json` (champs compacts `pg/pc/pe/ins/m/t/ci` legs ED
**+ `mv`**, abstentionnistes mobilisables par bureau) + `circo.json` (parts
agrégées par circonscription). En amont, `movability_turnout.py`
estime **γ**, la part de gauche du *votant marginal*, par décile de niveau de gauche du
bureau (différences premières intra-type sur les vraies hausses de participation —
identifiée, stable +0,96 ; voir `MOVABILITY.md` §11) ; `report_targets.py` en tire le
**gisement mobilisation** (abstentionnistes × γ(b)), le **résumé `summary.left_gain`**
(total métropole + déploiement par commune, outre-mer/étranger exclus) et le tableau par
bureau **`mv`** ajouté à `national.json` pour le calque carte client. *(Le canal
persuasion β — `movability.py` — et l'ancien « potentiel latent » par attente SHAP
— `report_potential.py`, supprimé — ont été testés et abandonnés : non identifiés / non
stables, `MOVABILITY.md` §2/§10.)* `report_geo.py` découpe les contours en GeoJSON par
département (propriétés légères, dont `mv` mobilisables — **couleur par défaut**, `ab`
abstentionnistes réels pour le γ du survol, `w` phrase « pourquoi mobilisable » lue dans
`why_left.json`, et `t` legs du stress-test ED) ; il s'exécute **après `report_shap.py`**
(pour disposer de `why_left.json`). `aggregate_communes` ajoute `cmv`/`cab` (mobilisables
et abstentionnistes réels agrégés) à `communes.json` pour le survol de la vue dézoomée. `report_figs.py`
ne produit plus que deux choses : `coverage.json` (couverture empirique, servie en
**pastille numérique**) et le **schéma de méthode** SVG (flux deux temps + frise
apprentissage/test, R² en légende). Précision-par-marge, déploiement par commune,
courbe de bascule, rapport de force et pastille de couverture sont **tracés client**
(SVG / barres en palette, réactifs aux curseurs). `report_shap.py` réutilise
`train_and_explain` (modèles pré-enregistrés, désormais à **noms de features bruts**)
pour les contributions top-6 par bureau — `drivers` du **bloc en tête** (calque bloc) **et
`gdrivers` du modèle Gauche** (toujours exporté : il règle le niveau de gauche, donc γ,
donc la mobilisation). Chaque entrée `[libellé, valeur]` où `pretty_label` rend le nom
lisible (« Vote Gauche (n-1) », « Chômage »…) et la valeur est la contribution SHAP en
points. `explain_left(gdrivers)` en tire une **phrase directionnelle honnête** (`wleft`,
dupliquée dans `why_left.json` pour le survol via `report_geo`) — « doit son niveau de
gauche surtout à… » si les deux plus gros moteurs le tirent vers le haut, sinon « voit son
niveau de gauche tiré vers le bas par… » : jamais affirmer qu'un bureau de droite « penche
à gauche ». Le panneau-instrument trace les barres en **divergentes signées** (couleur du
bloc en tête, ou Gauche en mode mobilisation ; la phrase de décideur, jugée trop lisse pour
le bloc en tête, *revient* pour la mobilisation où elle dit l'essentiel). `mob` (mobilisables
réels) est aussi joint à chaque enregistrement de détail pour l'équation `N = A × γ`. Le
**waterfall de diagnostic**
(`shap_waterfall._clean_name`, hors livrable) lit les mêmes noms bruts et les nettoie à
l'affichage (« vote Extrême Droite (n-1) »), sans jamais imprimer « lag ».
`report_app/` (HTML/CSS/JS, sans build). Vérification navigateur :
`src/verify_report.py` (Playwright headless) — zéro exception JS, 7 barres de
précision, pastille 80/90/95, 4 barres de provenance, **2 barres du duel sondage
vs nous (48,2 % / 81,6 %)**, la barre **« réalité partagée »** (panneau autonome
« paysage national » supprimé), **1 gisement d'électeurs** (mobilisation 4,71 M)
et **12 communes de déploiement**, le **mode par défaut `mobil`** avec le **calque
bloc en tête** basculable (`#lead` : `mobil` → `lead` → `mobil`),
**2 barres de rapport de
force** (aujourd'hui / scénario), la liste **« sur le fil »** (12 circonscriptions, zéro
surlignée au repos, ≥ 1 surlignée à +3 pts ED), et le double compteur **+3 pts ED →
6 923 bascules de bureau et 75 circonscriptions** (swing conservé, identiques aux
chiffres Python).

Vérification finale : chaque chiffre du site recoupé contre `algorithm.md` /
`preconisations.md` et recalculé dans les données. `ruff format --check` propre
sur le Python.

Palette neutre par défaut : Gauche `#E4572E` · Centre+Droite `#4A90D9` · Extrême
Droite `#6A4C93` (violet saturé, distinct des autres — surtout pas un quasi-noir
qui lirait « donnée manquante » sur la carte) · Abstention `#9AA0A6` · encre
`#1A1A2E` (texte uniquement) sur papier `#FAFAF7`.

**La carte n'est pas binaire : la teinte porte la marge.** Plutôt que trois aplats
pleins (un bureau gagné d'1 pt aussi tranché qu'un raz-de-marée), chaque bloc a une
**variante pâle** (sa teinte fondue ~78 % vers le papier) et la couleur
**interpole de pâle (marge 0) à plein (marge ≥ 12 pts)** : `#F5D5C9 → #E4572E`,
`#CFE1F2 → #4A90D9`, `#DBD2E8 → #6A4C93`. La marge est calculée en direct dans
l'expression MapLibre (`top − second`, via `max`/`min`), donc un bureau qu'un
curseur vient de faire basculer ressort *pâle* — l'incertitude est lisible sur la
carte elle-même, pas seulement dans les panneaux. Légende : pastille dégradée
« pâle = serré ».

**Robustesse de la carte qui se docke.** Une seule instance MapLibre ; le passage
plein-cadre ↔ rail ne se déclenche qu'**au changement d'état** (IntersectionObserver
sur `#hero-end`), jamais à chaque frame de scroll, et `map.resize()` est appelé par
brèves rafales (`resizeBurst`, ~14 frames) à ce seul moment — pas de listener
coûteux. Repli mobile : la grille magazine s'effondre en une colonne (≤ 720 px), le
hero rétrécit, la mini-carte passe à 200×150 px ; le site reste lisible au doigt.

## 9. Décisions ouvertes (confirmer, sinon défauts)

- **Parti & couleur d'accent** — sinon palette neutre ci-dessus.
- **Rapport de force en circonscriptions** — défaut : **bloc dominant au premier tour,
  agrégat des bureaux** (honnête, déjà construit, swing conservé, cohérent avec le
  modèle qui ne voit que le T1). À confirmer si le client veut aller jusqu'à une
  **projection de siège à deux tours** — hors périmètre actuel (exige de modéliser la
  qualification, les désistements et le report de voix du T1 vers le T2, non fait et
  non validé) ; à ne pas vendre comme un pronostic de sièges sans ce travail.
- **Scénario mis en avant** — défaut +3 pts ED (le plus parlant sur 2024).
- **Caméra d'entrée & zooms** — défaut métropole = Lyon ; circonscription
  disputée = la plus serrée selon les marges calculées.
- **Hébergement / diffusion** — site statique : un lien privé suffit-il, ou faut-il
  un accès protégé par mot de passe ?
- **Mention de confidentialité** du pied de page.
