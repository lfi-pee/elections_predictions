# Améliorations — retours client (fin de campagne / abstentionnistes)

> **État (2026-05-30) : toutes les recommandations sont appliquées au rendu et aux .md,
> Européennes comprises.** γ par type de scrutin + graphe à **3 courbes** (legi 39 % /
> euro 24 % / présid 12 %, §0, §2.4/2.7), abstention conjoncturelle/structurelle sur
> historique **3 scrutins** (§2.1/§2.2), hover refondu avec unité nommée + « si tous
> mobilisés → score » (§2.5/2.6), schéma de méthode annoté (§2.3), accent LFI + cadre
> GOTV (§2.8/§3), déploiement volume+rendement. Chiffre héros : **3,39 M** (conjoncturels
> 8,15 M × γ législatif 41,6 %). Les Européennes (1999–2024) étaient en fait présentes
> dans `general_results` / `elections.parquet` : un panel γ dédié
> (`gamma_panel.parquet`) les intègre **sans toucher au modèle de production** (qui
> n'entraîne toujours que sur legi+présid). Pipeline régénéré, `verify_report.py` : PASS.

> Destinataire confirmé : **Manuel Bompard (LFI)** — directeur de campagne d'un parti
> de gauche. L'usage visé est désormais explicite : un **outil de fin de campagne pour
> aller chercher les abstentionnistes** (GOTV), pas une lecture de horse-race. Ce
> document traduit chaque retour en changement **précis** : fichier du site, panneau,
> et passage du brief (`REPORT_PLAN.md`) / des notes méthode (`MOVABILITY.md`).
>
> Artefact joint : [`top100_villes_mobilisables.csv`](top100_villes_mobilisables.csv) —
> les 100 communes de métropole au plus fort gisement d'électeurs mobilisables à gauche.

---

## 0. La trouvaille qui commande tout le reste : γ n'est pas la même selon le scrutin

Le retour *« l'abstention de gauche doit dépendre des élections, le γ ne doit pas être
la même formule »* est **exact, et l'écart est massif**. Mesuré dans
`cross_type_dev_base.parquet` (mêmes différences premières intra-type que
`movability_turnout.py`, mais séparées par type de scrutin) :

| γ = part de gauche de l'électeur marginal | valeur | n (différences) |
|---|---|---|
| **Poolé (ancien défaut)** | **31,5 %** | 561 292 |
| **Législatives T1** | **39,3 %** | 314 280 |
| **Européennes T1** | **23,9 %** | 307 463 |
| **Présidentielle T1** | **12,3 %** | 247 012 |

Et la **forme** par décile de niveau de gauche est de signe opposé :

| décile de gauche (faible → fort) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| γ Législatives (%) | 26,1 | 36,5 | 44,8 | 50,8 | 54,1 |
| γ Présidentielle (%) | 16,6 | 9,1 | 8,5 | 10,2 | 16,6 |

Lecture : en **législatives**, l'électeur qu'on fait revenir aux urnes penche d'autant
plus à gauche que le bureau est déjà à gauche (monotone croissant, le cœur du raisonnement
GOTV de gauche). En **présidentielle**, le marginal est globalement bien moins à gauche
(12 %) et la courbe est en U — la mobilisation présidentielle ramène surtout d'autres
électorats. **Le γ poolé de 31,7 % mélange deux régimes contraires** et n'est juste pour
*aucun* scrutin réel.

Conséquence directe : le chiffre héros **4,71 M d'électeurs mobilisables** est
**ambigu sur le scrutin**. Pour une cible législative il est sous-estimé ; pour une
présidentielle il est largement surestimé (≈ 1,8 M au γ présidentiel). Le livrable doit
**nommer le scrutin qu'il projette** et appliquer la courbe γ de *ce* type.

`MOVABILITY.md` §13 a testé si γ dépend de la démographie/des lags (verdict : non, non
transférable) — mais **n'a jamais séparé par type de scrutin cible**. C'est une dimension
différente, **identifiée et stable intra-type** (le §11 montre déjà +0,96 de stabilité
temporelle *au sein* d'un type), donc défendable. C'est le chantier n°1.

---

## 1. Analyse du tableur des 100 villes — ce qu'il révèle

Calculé sur `bv_master.parquet`, métropole, agrégé par commune (`mob` = abstentionnistes
prédits × γ poolé par bureau).

**Constat n°1 — le classement « Où déployer » est de la population déguisée.** Sur
toutes les communes : `corr(mobilisables, abstentionnistes) = 0,996` et
`corr(mobilisables, inscrits) = 0,987`. Dans le top 100, le gisement `mob` varie ×31
(Paris 134 k → 4,3 k) tandis que γ ne varie que ×2 (26 % → 52 %). **Le panneau
« Où déployer » classe donc essentiellement les villes par taille** — exactement le
risque que `MOVABILITY.md` §10 s'était engagé à ne pas habiller en signal spatial. Tel
quel, il ne montre pas l'apport unique du modèle ; il montre que Paris est grand.

**Constat n°2 — l'apport réel du modèle est γ (le *taux*), pas le total.** Les communes
à fort γ sortent du lot : Montreuil 51,8 %, Saint-Denis 51,4 %, Roubaix 42,7 %, Lille
42,6 % — des bastions de gauche où l'abstentionniste marginal est *vraiment* de gauche.
À l'inverse Nice (30,4 %), Toulon (29,8 %), Reims (31,5 %) ont un gros stock mais un
rendement de gauche faible. **C'est ce contraste — pas le rang brut — qui est
décisionnel** pour LFI : un effort à Montreuil convertit ~52 % des abstentionnistes
ramenés, à Nice ~30 %.

**Constat n°3 — le top 100 ne pèse que 21,6 % du gisement national.** Le reste est
diffus. Honnête à dire : la concentration urbaine est réelle mais partielle.

**Ce que le tableur impose au site :** le panneau « Où déployer » doit afficher **deux
colonnes** — le **volume** (mobilisables, en personnes) *et* le **rendement** (γ, %) —
et permettre de trier par l'un ou l'autre. Trier par volume = « où il y a le plus à
faire » ; trier par γ = « où chaque porte frappée rapporte le plus ». Aujourd'hui seul le
volume est montré, donc seul « les grandes villes » ressort.

---

## 2. Changements précis — par retour

### 2.1 « Différencier l'abstention de fond de l'abstention conjoncturelle »

**Problème.** `report_targets.mobilizable()` fait `abstentionnistes × γ` sur **tout** le
stock d'abstentionnistes. Or l'abstentionniste chronique (ne vote jamais) et le
conjoncturel (vote aux présidentielles, saute les législatives/européennes) ne sont pas
le même gisement. Le mobilisable de fin de campagne, c'est le **conjoncturel**.

**Méthode proposée.** Décomposer l'abstention par bureau en deux strates via la variance
de participation entre types de scrutin déjà présents dans les données :
`abst_structurelle(b) ≈ min de participation observée` (le socle qui ne se déplace
jamais) et `abst_conjoncturelle(b) = abst(b) − abst_structurelle(b)` (la frange qui vote
quand l'enjeu monte). Le gisement devient
`mobilisables(b) = abst_conjoncturelle(b) × γ_type(b)` — plus petit, plus défendable,
et **exactement le public d'une campagne GOTV**.

**Site.**
- `report_app/index.html` panneau « Combien d'électeurs, et où » : ajouter une bascule
  *abstention totale* vs *abstention mobilisable (conjoncturelle)* ; afficher les deux
  chiffres (p. ex. « 14,8 M d'abstentionnistes, dont X M conjoncturels »).
- `js/map.js hoverBody()` et `js/panel.js whyMobil()` : décomposer la ligne en
  « A abstentionnistes, dont C conjoncturels × γ % ».

**Pipeline.** `report_targets.py` : nouvelle fonction `abstention_split(df)` ; champ
`mv` de `national.json` recalculé sur la part conjoncturelle ; `summary.left_gain`
gagne `structural_abstainers` / `conjunctural_abstainers`.

**Report .md.** `MOVABILITY.md` : nouvelle section §14 « Abstention de fond vs
conjoncturelle » documentant l'identification et le verdict. `REPORT_PLAN.md` §6
(bullet « Électeurs convaincables ») : remplacer « abstentionnistes × γ » par
« abstentionnistes **conjoncturels** × γ », et corriger le chiffre héros.

### 2.2 « Ajouter les Européennes partout » — fait

**Découverte.** Le cache `cross_type_dev_base.parquet` ne contenait que legi+présid, mais
le brut **les contient** : `general_results.parquet` a `1999/2004/2009/2014/2019/2024_euro_t1`
et `elections.parquet` porte 11,5 M de lignes `Europeennes_T1` bloc-mappables.

**Appliqué.** Un panel γ dédié `data/baseline_cache/gamma_panel.parquet`
(`movability_turnout._ensure_panel`, bâti via `cross_type_ridge._build_block_scores`) intègre
les **trois** types T1 — **sans toucher au modèle de production**, qui n'entraîne toujours que
sur legi+présid. Effets :
- **γ à trois régimes** (legi 39 % / euro 24 % / présid 12 %), servis au graphe
  (`curves_by_type` → `gamma_curve.json`, 3 courbes).
- **Plancher d'abstention** calculé sur l'historique **3 scrutins** (`report_data.TURNOUT_CACHE`
  pointe sur le panel) — abstention de fond plus fidèle.

**Reste ouvert (mesure observée, complément).** Le différentiel par bureau
`gauche(Prés. 2022) − gauche(Eur. 2024)` (réservoir conjoncturel *observé*, non modélisé)
est désormais calculable depuis ce même panel ; il peut devenir un calque de validation —
documenté `preconisations.md` reco 6.

### 2.3 « Insister sur comment le modèle s'articule — quelle variable à quel moment »

**Problème.** Le schéma `data/fig_method.svg` (pied de page) reste abstrait
(« national → + écart local → prédiction »). Le décideur ne voit pas *quelle variable
entre quand*.

**Changements (site).**
- `src/report_figs.py` (générateur du SVG) : annoter chaque boîte avec ses **entrées
  nommées**. Boîte 1 *national* : « sondages agrégés (vote) + modèle de participation
  (abstention) ». Boîte 2 *écart local* : « vote passé du bureau (lags n-1, n-2) +
  tendance (dev_lag1 − dev_lag2) + 52 indicateurs INSEE compressés + position
  géographique ». Boîte 3 : « = part prédite du bureau ». Boîte 4 : « → fourchette
  conforme stratifiée par territoire ». Faire apparaître que **le cadran de scénario
  agit sur la boîte 1 seule**.
- `index.html` `.method-panel` : sous le schéma, une phrase courte par étape (pas une
  liste à puces — règle §4 du brief : prose liante).

**Report .md.** `REPORT_PLAN.md` §7 pied de page « Comment ça marche » : préciser
explicitement la liste des variables par boîte (aujourd'hui le texte dit « démographie
INSEE compressée + héritage du vote passé » sans nommer lags/tendance/géo).

### 2.4 « L'abstention de gauche, ce n'est pas clair » + 2.7 « Rajouter le graphe pour γ »

**Problème.** Le concept « abstentionnistes qui penchent à gauche » et γ ne sont définis
nulle part visuellement. Le mot « γ » a été retiré du rendu (mémoire `site-novice-readable`)
— bien — mais il faut **montrer** la chose sans la nommer.

**Nouveau panneau (site).** Dans la Bande 2, un panneau **« Qui revient voter quand la
participation monte »** : la courbe γ(niveau de gauche du bureau), **une ligne par type
de scrutin** (législative vs présidentielle, cf. §0). Axe X « bureaux plutôt à droite →
plutôt à gauche », axe Y « part de ces revenants qui votent à gauche ». Légende en clair,
zéro lettre grecque.

**Pipeline.** `report_figs.py` : émettre `gamma_curve.json`
`{ legi:[[niveau,part]…], presid:[…] }` depuis `movability_turnout.gamma_curve` appelée
par type. `js/viz.js` : `renderGamma()` qui trace les deux lignes (SVG, palette des blocs).

**Report .md.** `MOVABILITY.md` §11 : ajouter le tableau §0 (γ par type) et noter que la
courbe servie est désormais **par type**, plus la courbe poolée.

### 2.5 « Le hover, on ne comprend rien. Le shape, c'est sur quoi ? »

**Problème.** `js/map.js hoverBody()` ouvre directement sur « **N électeurs mobilisables
à gauche** » sans dire **de quelle entité géographique** on parle. À fort zoom le polygone
(« shape ») est un **bureau de vote** ; au dézoom c'est une **commune** (cercle
proportionnel). Rien ne le dit, d'où la confusion. Le titre du popup affiche `p.n` qui
peut être vide (« commune »).

**Changements (`js/map.js`).**
- En-tête du popup **toujours qualifié par l'unité** : à fort zoom
  « **Bureau de vote n°{num} · {commune}** », au dézoom
  « **{commune} · {n} bureaux de vote** ». L'unité doit être lisible en premier.
- Première ligne reformulée pour dire ce qu'on lit : « la couleur = densité
  d'électeurs à aller chercher (abstentionnistes de gauche) ».
- La phrase SHAP `p.w` (« pourquoi mobilisable ») est souvent absente/obscure : la
  garder mais derrière un libellé « pourquoi ce bureau penche à gauche ».
- Le champ `num` (n° de bureau) doit être exposé dans les propriétés des features
  geojson (`report_geo.py`) pour pouvoir l'afficher au hover.

**Report .md.** `REPORT_PLAN.md` §7 Bande 1 (description du survol) : préciser que
**l'unité géographique est nommée en tête du survol** et que la couleur est explicitée.

### 2.6 « Rajouter dans le hover : si tous ces gens sont mobilisés, voilà votre score »

C'est le geste le plus vendeur, et il est **calculable exactement**. Si les `mob`
mobilisables d'un bureau votent Gauche :
`G_nouveau = (part_G × votants + mob) / (votants + mob)`. Exemples vérifiés :

| commune | Gauche aujourd'hui (exprimés) | si les mobilisables votent | gain |
|---|---|---|---|
| Marseille | 38,4 % | **50,3 %** | +11,9 pts |
| Nice | 27,3 % | **38,8 %** | +11,5 pts |
| Montreuil | 74,5 % | **80,0 %** | +5,5 pts |

**Changements.**
- `js/map.js hoverBody()` (mode `mobil`) : ajouter une ligne
  « **si vous les ramenez tous : Gauche {G_old} % → {G_new} %** ».
- `js/panel.js whyMobil()` : même calcul, en évidence dans l'équation `N = A × γ`.
- Données : `national.json` sert déjà `pg/pe/pc/ins/m` ; il faut **ajouter la
  participation/abstention par bureau** pour reconstituer `votants` côté client (champ
  `ab` déjà dans le geojson, à propager dans `national.json` ou calculer dans `panel.js`
  depuis `rec.blocks.AB.act` — déjà disponible).

**Report .md.** `REPORT_PLAN.md` §7-bis : ajouter à la description du panneau-instrument
la ligne « score résultant si mobilisation totale » (mode mobilisation).

### 2.7 voir 2.4 (graphe γ).

### 2.8 « Je suis Manuel Bompard » — identité client (décision §9 du brief)

Lever la décision ouverte §9 « Parti & couleur d'accent ». Client = **LFI**.

**Changements.**
- `js/config.js APP.COL.G` : la Gauche reste `#E4572E` (orange-rouge proche de l'identité
  LFI — cohérent), mais ajouter un **accent de marque** pour les titres/CTA. À confirmer
  avec le client (rouge LFI `#cc2229` typiquement).
- `index.html` header : adapter le sous-titre au registre directeur de campagne de
  gauche, et **assumer l'angle GOTV** dès le hero (« où aller chercher vos voix
  d'ici le scrutin »).
- Cadrage par défaut : le scénario héros pourrait passer du stress-test **+3 pts ED**
  (marée adverse) à un récit **mobilisation de gauche** — à arbitrer, mais l'instrument
  doit d'abord parler du gisement, pas de la menace.

**Report .md.** `REPORT_PLAN.md` §9 : trancher « Parti = LFI », fixer l'accent.

---

## 3. Cadre temporel : assumer « fin de campagne / GOTV »

Le premier retour (« c'est plutôt pour la fin de campagne ») doit être visible dans le
livrable. La Bande 2 n'est pas une analyse rétrospective : c'est un **plan de
mobilisation** pour les dernières semaines.

**Changements.** `index.html` titre de Bande 2 : « **Combien d'électeurs, et où — votre
plan de mobilisation** ». Sous-titre rappelant que ces électeurs sont des
**abstentionnistes conjoncturels** atteignables d'ici le vote. `REPORT_PLAN.md` §7
Bande 2 : réécrire l'intention en clé fin-de-campagne.

---

## 4. Ordre de priorité

1. **§0 + §2.4/2.7 — γ par type de scrutin + graphe γ.** C'est le retour de fond, c'est
   chiffré, c'est le plus visible. Sans lui, le chiffre héros est faux pour un scrutin
   donné.
2. **§2.5 — hover lisible (unité géographique nommée).** Bug de compréhension immédiat,
   coût faible (`map.js` + un champ dans `report_geo.py`).
3. **§2.6 — « si tous mobilisés, votre score ».** Fort effet, calcul exact, données
   presque déjà là.
4. **§2.1 — abstention conjoncturelle vs structurelle.** Recadre le gisement ; demande un
   peu de pipeline.
5. **§2.2 — charger les européennes** pour mesurer le conjoncturel directement (dépend
   d'un chargement de données neuf).
6. **§2.3 / §2.8 / §3 — articulation du modèle, marque LFI, cadre GOTV.** Éditorial,
   faible risque.

---

## 5. Récapitulatif des fichiers à toucher

| Fichier | Nature du changement |
|---|---|
| `src/movability_turnout.py` | γ **par type de scrutin** (paramètre `election_type`), exposer les deux courbes |
| `src/report_targets.py` | abstention conjoncturelle/structurelle ; γ du type cible ; champs résumé |
| `src/report_figs.py` | `gamma_curve.json` (par type) ; schéma méthode annoté variable par variable |
| `src/report_geo.py` | propager **n° de bureau** + abstention dans les propriétés geojson (hover) |
| `movability_turnout.py` (`_ensure_panel`) | panel γ 3 types **euro incluse** (`gamma_panel.parquet`), via `cross_type_ridge._build_block_scores` |
| `report_app/js/map.js` | hover : unité géo nommée, couleur explicitée, « si tous mobilisés → score » |
| `report_app/js/panel.js` | `whyMobil` : conjoncturel, score résultant |
| `report_app/js/viz.js` | `renderGamma()` (courbe γ par type) |
| `report_app/js/config.js` | accent de marque LFI |
| `report_app/index.html` | panneau γ, bascule abstention, cadre GOTV, titres |
| `REPORT_PLAN.md` | §6 gisement conjoncturel ; §7/7-bis hover + score ; §9 client LFI |
| `MOVABILITY.md` | §11 γ par type ; §14 abstention de fond vs conjoncturelle |
| `preconisations.md` | préconisation n°6 : charger les européennes |
