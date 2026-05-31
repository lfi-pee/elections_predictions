# Mesure de la « mobilité vers la gauche » par bureau — note de conception

> Note d'ingénierie / méthodologie (interne). Objet : remplacer le gisement
> « potentiel latent » de la Bande 2 par une mesure défendable de combien
> d'électeurs supplémentaires pourraient voter à gauche, localisée. Rédigée avant
> implémentation — c'est la spécification de la méthode, pas encore du code.

---

## 1. Objectif (et ce qu'il n'est pas)

On veut **maximiser le total national de voix de gauche**, pas gagner des bureaux
ni des sièges. La sortie est donc un **nombre d'électeurs gagnables par bureau**,
agrégeable au national et classable par commune — assorti d'une **hypothèse
d'identification explicite**. La tête de liste / le point de bascule / la
circonscription ne nous intéressent pas ici : on compte des électeurs.

## 2. Ce qu'on écarte — et pourquoi (à ne pas réintroduire)

- **Résidu démographique (« jumeaux »)** : `attente_démo − réalisé`. Relabelle la
  variance que la démographie n'explique pas en « potentiel ». Aucune identification ;
  et le **signe est faux** — le résidu est la composante *persistante* (un bureau loin
  sous ses jumeaux a durablement refusé sa démographie → le moins mobilisable, pas le
  plus). Privilégier la démo comme « vérité » et l'histoire de vote comme « bruit à
  retirer » est arbitraire.
- **Distance à la tête / point de bascule** : on compte des électeurs, pas des têtes.
- **Largeur de l'intervalle conforme comme « influençabilité »** : erreur de catégorie.
  L'IC mesure notre *ignorance* du score (épistémique), pas la *réponse* du score à un
  levier (causal). Les IC les plus larges sont sur l'outre-mer/Corse/étranger — les pires
  zones du modèle, pas les plus gagnables. Variance ≠ contrôlabilité.
- **Régression naïve `score = f(participation)`** : la participation est *endogène* au
  score (vague, candidat, saillance font monter les deux). La pente observée n'est pas la
  pente causale de mobilisation.

## 3. La mesure retenue — le chargement sur la marée de gauche (β_b)

Décomposition panel de la part de gauche du bureau *b* sur les scrutins comparables :

```
LeftShare(b,t) = α_b + β(x_{b,t}) · L_t + ε(b,t)
```

- `L_t` = **marée commune de gauche** à l'élection *t*, agrégat **leave-b-out**
  (national ou régional, **par type de scrutin**).
- `α_b` = niveau de base propre au bureau.
- `β(x_{b,t})` = **le chargement sur la marée = la mobilité vers la gauche**. C'est une
  dérivée : quand la marée monte d'un point, la part de gauche de *b* monte de `β` points.

**Mobilité = β.** En électeurs, pour une poussée plausible `ΔL` :

```
gagnables(b) = β(x_{b,t}) · ΔL · électeurs_b
```

**Déploiement** : classer les bureaux par `β · électeurs_b` (forte réponse × forte
population) ; agréger au national ; décomposer par commune.

Pourquoi c'est le bon objet, et pourquoi il survit à tout le §2 :

- **Du changement, pas du niveau** → n'affirme aucun niveau « mérité ». Un bureau sous
  ses jumeaux mais `β ≈ 0` est lu, correctement, comme durablement inerte. Corrige le
  signe faux du résidu.
- **Des électeurs, pas une tête** → `β·ΔL·électeurs` est un effectif, sans seuil.
- **Une réponse, pas une incertitude** → pente structurelle, sans rapport avec l'IC.
- **Identification propre** : `L_t` est un agrégat leave-one-out et un bureau est une part
  infinitésimale du national → pas de simultanéité (la maladie qui tuait `f(participation)`).
  C'est l'identification *shift-share / facteur commun* standard.

## 4. Estimation

- **Marée `L_t`** : fournie par l'**ancre nationale** que le modèle de production estime
  déjà par scrutin (meilleur régresseur qu'une part nationale brute).
- **Pentes hétérogènes, rétrécies** : OLS par bureau sur une poignée de scrutins est trop
  bruité → modèle à **pentes aléatoires** avec **rétrécissement (empirical Bayes)** vers
  des groupes de pairs. Sans cela, le haut du classement n'est que du bruit (régression
  vers la moyenne).
- **Directionnel** : « mobilité *vers la gauche* » = réponse aux marées **montantes** de
  gauche (pente sur les transitions favorables, ou `β` distinct au-dessus/sous la moyenne
  de marée), pas une élasticité symétrique.

## 5. Dynamique temporelle — contrainte dure, à ne pas perdre

Le modèle actuel tire sa force de ses **features dynamiques** : lags de vote
(`*_lag1/2`), lags de déviation (`dev_*_lag1/2`), **tendance** (`dev_lag1 − dev_lag2`),
encodage du **type de scrutin**. Ils suivent la *trajectoire* du bureau, pas un instantané.

La mobilité doit s'appuyer dessus, **pas les aplatir** : `β` est **dépendant de l'état
dynamique** — `β(x_{b,t})`, fonction de la représentation dynamique (un bureau déjà en
dérive à gauche ne répond pas comme un bureau plat). On estime sur le **panel** (plusieurs
scrutins), jamais sur une coupe transversale figée. La trajectoire/momentum est un prédicteur
de `β`, pas un parasite à retirer.

## 6. Rôle du modèle de production — échafaudage, pas estimateur

- **Il ne peut PAS estimer β.** Sa forme `ancre nationale + écart local figé` *est* une
  hypothèse de **swing uniforme** : `β ≡ 1` pour tous, par construction (c'est tout le
  moteur additif des scénarios). En lire la mobilité rend une constante. Perturber ses
  entrées rend l'**artefact de forme** (le swing uniforme), pas la dérivée causale — même
  piège que la contrefactuelle SHAP : *sensibilité d'un prédicteur ajusté ≠ dérivée causale*.
  Un modèle optimisé pour la justesse du **niveau** n'a aucune obligation d'avoir les bonnes
  **pentes** structurelles.
- **Mais il est l'échafaudage idéal :**
  1. il définit la **marée `L_t`** (ancre nationale par scrutin) ;
  2. sa **représentation par bureau** (features dynamiques §5 + démo) = covariables idéales
     pour mutualiser et rétrécir les `β` (le §4 « prédire β_b à partir des features ») ;
  3. son **moteur swing-conservé** convertit `β·ΔL` en voix **multi-blocs cohérentes**
     (un gain financé par les autres, somme nulle — pas de voix créées).
- **Le geste** : on **relâche le swing uniforme**. Nouveau modèle, plus petit (pentes
  hétérogènes dépendantes de l'état) sur les **mêmes données + même représentation**. On
  ne jette pas le prédicteur : on lui ajoute la pente qui lui manque.

## 7. Hypothèse d'identification (assumée, écrite noir sur blanc)

Le mouvement induit par campagne **ressemble** au mouvement induit par la marée historique :
la sensibilité passée aux vagues de gauche se transfère à une poussée délibérée. Hypothèse
**unique, directionnelle, falsifiable** — bien plus faible que « la démographie est la vraie
propension et l'histoire est du bruit ». C'est le seul endroit où vit le saut causal, au lieu
d'être étalé dans un résidu non identifié.

## 8. Validation — backtest temporel

Estimer `β` sur les scrutins **jusqu'à 2022**, prédire les **gains réels de gauche en 2024**
(hors échantillon, déjà notre jeu de validation). Tracer la **courbe de calibration**
prévu-vs-réalisé des gains, et comparer la sélection « haute mobilité » à un témoin. Le résidu
démographique ne peut **pas** produire cette courbe (il n'a aucune notion de « a-t-il bougé ») ;
`β`, si.

## 9. Sorties attendues

- `β_b` (rétréci, directionnel, dépendant de l'état) par bureau.
- `gagnables(b) = β_b · ΔL · électeurs_b` pour une `ΔL` plausible, agrégat national +
  déploiement par commune.
- Le tout assorti de l'**hypothèse §7** et de la **calibration §8**, et — comme le reste du
  livrable — assumé honnêtement : c'est une *réponse modélisée à une marée*, pas une mesure
  d'électeurs déjà acquis.

## 10. Résultat empirique — verdict du backtest (§8)

Implémenté dans `src/movability.py` et **testé hors échantillon. Le verdict est négatif,
et il faut l'assumer.**

Différences premières intra-type, β rétréci, prédisant le mouvement legi d'un scrutin au
suivant à partir des seuls scrutins antérieurs :

| transition | ΔL national | RMSE swing uniforme (β≡1) | RMSE β hétérogène | corr. part hétérogène |
|---|---|---|---|---|
| 2012→2017 (β≤2012) | −19,7 | **10,35** | 12,84 | +0,121 |
| 2017→2022 (β≤2017) | +3,9 | 10,10 | **10,09** | +0,060 |
| 2022→2024 (β≤2022) | −2,7 | **7,83** | 7,90 | −0,040 |

Mis en commun par **département** (le bruit par bureau moyenné) : pire encore sur la grosse
vague (RMSE 16,2 vs 10,3), corrélation hétérogène ≈ 0 partout.

**Lecture.** Le chargement hétérogène existe — la corrélation est *positive et significative*
sur la grande vague 2012→2017 (+0,12 sur 62 000 bureaux) — mais il est **trop faible et trop
bruité pour être exploité** : utilisé comme multiplicateur, β fait jeu égal au mieux, et
*dégrade* la prédiction sur les grands mouvements (le bruit d'estimation × ΔL explose). Le
regroupement ne le sauve pas, parce que les chargements sont **non stationnaires** : la
période 2012–2024 est une recomposition (effondrement du PS, montée LFI/RN, Macron), donc « la
sensibilité d'un bureau à la marée de gauche » d'hier ne porte pas sur demain — *ce que « la
gauche » désigne et qui y répond ont changé*. Le passé anti-prédit le futur en agrégat.

**Conséquences, honnêtes :**
- Le **swing uniforme** (β≡1) — précisément l'hypothèse du modèle de production — est le bon
  défaut empirique. On ne le bat pas de façon fiable.
- À β≈1, `gagnables(b) = exprimés_b·ΔL/100` est de l'**arithmétique proportionnelle à la
  population** : la liste de déploiement se réduit aux plus grandes villes (Paris, Marseille,
  Lyon…) et n'ajoute rien que la taille de l'électorat. C'est exactement ce que le livrable dit
  déjà ne *pas* être un apport unique du modèle.
- **Il ne faut pas** remplacer la Bande 2 par un compte « électeurs reconquérables » fondé sur
  β : ce serait habiller en signal spatial ce qui est de l'arithmétique nationale — la
  malhonnêteté même que toute cette démarche cherchait à éviter.

**Donc** : la « mobilité vers la gauche » par bureau, mesurée proprement, n'est pas un objet
identifiable et stable sur ces données. Le résultat défendable est *négatif* — et le dire est
l'apport. Le potentiel latent / les électeurs reconquérables de la Bande 2 ne reposent sur
aucun signal spatial vérifiable ; à conserver, ils doivent être présentés comme de
l'arithmétique de swing uniforme, pas comme une cartographie du gisement.

## 11. Le canal mobilisation : un remainder, lui, identifiable ET stable (γ)

Le verdict négatif du §10 porte sur le **canal persuasion** (réponse de la part de gauche
des *exprimés* à la marée). Il ne dit rien du **canal mobilisation** (les abstentionnistes).
Testé séparément dans `src/movability_turnout.py`, par différences premières intra-type :

`γ = d(gauche % inscrits) / d(participation)` = **part de gauche du votant marginal**
(l'ex-abstentionniste qui se déplace quand la participation monte) — quantité *identifiée*
(lue sur les vraies hausses de participation), pas le partage des exprimés (circulaire).

γ par décile de niveau de gauche du bureau :

| décile (G moyen) | 12 | 20 | 24 | 27 | 31 | 35 | 39 | 44 | 51 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| **γ (%)** | 23 | 29 | 30 | 30 | 30 | 32 | 33 | 33 | 36 | **47** |
| **γ − G** | +11 | +9 | +6 | +3 | 0 | −3 | −6 | −10 | −14 | **−17** |

**Trois faits, tous les trois importants :**
1. **L'intuition est vraie, directionnellement** : γ croît avec le niveau de gauche du bureau
   (23 % → 47 % ; pente +0,38). Il y a donc bien un **remainder mobilisable propre au bureau**,
   et il augmente avec le vote de gauche. « Pas de remainder » était faux — on n'avait testé que
   la persuasion.
2. **Mais c'est saturant** : `γ − G` passe de +11 (à droite, le marginal est *plus* à gauche que
   les votants) à −17 (en bastion, les abstentionnistes restants sont *bien moins* à gauche que
   les votants — la gauche a déjà capté ses électeurs faciles). Donc la part exprimée **surestime**
   le remainder mobilisable jusqu'à 17 pts en bastion : c'est exactement la circularité rejetée au
   §5, désormais chiffrée.
3. **Et c'est stable dans le temps** — contrairement au chargement β : la courbe γ(décile)
   ancienne (≤2012) vs récente (>2012) corrèle à **+0,96**. C'est une régularité comportementale,
   pas un artefact de période.

**Conséquence opératoire (le chemin défendable pour la Bande 2)** : le gisement mobilisation se
chiffre **mobilisables_gauche(b) = abstentionnistes conjoncturels(b) × γ(b)** (voir §14 pour le
split conjoncturel/fond), où `γ(b)` vient de la courbe participation **du type de scrutin projeté**
(§15), fonction du niveau de gauche du bureau, **jamais `γ = G`**. Identifié, stable, honnête sur la
saturation — plus petit que l'ancienne méthode en bastion, plus grand à droite. C'est la version qui
survit à l'examen.

## 12. Statut en production (résolu)

C'est **γ (§11) qui est câblé dans le livrable**, et lui seul. `movability_turnout.py`
estime la courbe γ(niveau de gauche) et l'expose ; `report_targets.py` chiffre le gisement
mobilisation de la Bande 2 = abstentionnistes × γ(b), servi par bureau (`national.json.mv`).

Ce qui est **abandonné, à ne pas réintroduire** (et pourquoi, en une ligne chacun) :
- le **chargement persuasion β** (`movability.py`) — non stationnaire, ne bat pas le swing
  uniforme hors échantillon (§10) ;
- le **potentiel latent** = attente démographique − réalisé — résidu non identifié, de signe
  ambigu, sans signal spatial vérifiable (§2) ; `report_potential.py` / `potential.parquet`
  supprimés ;
- **γ = part de gauche des exprimés** (`γ = G`) — circulaire, surestime jusqu'à 17 pts en
  bastion (§11).

Décisions encore ouvertes (réglages, pas méthode) : `ΔL` de référence pour le chiffrage
(poussée nationale plausible, à fixer avec le client comme le +3 ED de la Bande 3) ;
finesse du binning de la courbe γ (déciles validés, 20 bins servis pour lisser).

## 13. γ « dépend-il de tout » (démographie + lags + momentum) ? — testé, verdict négatif

Objection naturelle : la mobilité devrait dépendre de *tout* — démographie INSEE, lags de
vote, momentum — pas du seul niveau de gauche. Testé proprement dans
`src/movability_gamma_rich.py`. γ reste **identifié sur les vraies hausses de participation**
(cible `r = ΔLR/ΔT`, poids `ΔT²` → la prédiction d'une feuille = la pente sans constante = γ) ;
un HistGBM apprend γ(x) sur **72 features pré-swing** (niveau passé, lags G/CD/ED/Abst,
dev-lags, momentum, géo, 52 indicateurs INSEE) — la démographie comme *prédicteur de la pente
identifiée*, jamais substitut de niveau. Backtest hors échantillon (apprend ≤ fit_max, prédit
le mouvement legi suivant), RMSE de ΔLR :

| transition | γ plat | **courbe 1-D γ(niveau)** | γ riche (tout) | het_corr 1-D / riche |
|---|---|---|---|---|
| ≤2018 → legi 2022 | 5,65 | **5,34** | 6,36 | **+0,29** / −0,03 |
| ≤2022 → legi 2024 | 6,48 | **5,46** | 7,14 | **+0,43** / +0,01 |

**Deux faits.** (1) La **courbe 1-D bat le swing uniforme** hors échantillon (5,46 vs 6,48 en
2024, −16 % de RMSE ; het_corr fortement positif) : le niveau de gauche est un signal réel et
*transférable* pour la composition du votant marginal. (2) **« Tout » fait *pire* que plat**, et
loin derrière la courbe — alors que le GBM dispose pourtant de `prevG` et *pourrait* reproduire
la courbe puis y ajouter. Son het_corr ≈ 0 : la démographie/les lags n'apportent **aucune
hétérogénéité transférable**. Même cause que β (§10) : sur une décennie de recomposition, la
correspondance démographie→réponse est **non stationnaire** — le modèle ajuste l'ère
d'apprentissage, la relation ne tient pas au scrutin suivant, la variance d'estimation dégrade
la prédiction. **Le seul slice de « tout » qui transfère est le niveau de gauche, déjà utilisé.**
Garder γ à une dimension n'est pas un raccourci : c'est le plafond de ce qui est estimable et
stable sur ces données. (Distinguer du classement de déploiement, lui dominé par la *masse
d'abstentionnistes* — γ varie ×2,5, les effectifs ×82 ; voir la note opératoire.)

**Y a-t-il un juste milieu (GBM plus régularisé, mélange) ? Testé aussi — non.** Balayage de
régularisation + un mélange de principe (`courbe 1-D` en socle, GBM régularisé sur le *résidu*,
donc il ne peut qu'*ajouter*), RMSE de ΔLR hors échantillon :

| méthode | ≤2018→2022 | ≤2022→2024 |
|---|---|---|
| **courbe 1-D** | **5,34** | **5,46** |
| gbm-moyen → fort → très-fort | 6,36 · 6,74 · 6,77 | 7,14 · 6,61 · 6,54 |
| courbe 1-D + gbm-résidu | 6,63 | 7,25 |

Trois enseignements : (1) **régulariser pousse le GBM vers le *plat*, pas vers la courbe** — en
2024 il s'asymptote à ≈ 6,5 (≈ flat), jamais vers 5,46 ; un apprenant flexible avec 72 features
ne *redécouvre* pas de façon fiable la seule relation monotone stable. (2) En 2022, **plus de
régularisation = pire** (jusqu'à het négatif) : le GBM verrouille un motif démographique stable
en apprentissage mais *de mauvais sens* hors échantillon. (3) Le **mélange socle+résidu dégrade
la courbe** (5,46 → 7,25) : comme il ne peut qu'ajouter, cela prouve que le poids validé de tout
signal plus riche est **≤ 0**. Conclusion robuste : **coder en dur la seule relation stable bat
laisser un modèle la découvrir** — quand le reste est du bruit non stationnaire, la flexibilité
est un défaut, pas un atout. La courbe 1-D reste la version livrée.

**Et le « X vs Y » par historique propre de la station ? Testé — non identifiable.** Objection
de fond : un bureau peu à gauche peut l'être (X) parce qu'on n'y vote pas à gauche, ou (Y) parce
que la gauche y est *démobilisée* ; même niveau, mobilisable opposé. Pour les séparer sans passer
par la démographie (§13 supra) ni par le résidu jumeaux (§2), on laisse **l'historique de
participation propre à chaque station** estimer son γ_b (pente sans constante sur ses propres
hausses de participation), rétréci en **empirical Bayes vers la courbe 1-D** — donc une station
sans signal *retombe* sur la courbe, et seule une station dont l'histoire porte un vrai signal
s'en écarte. Backtest ≤2022 → 2024 : **τ² = 0,00000**. La variance inter-station de γ *au-delà de
la courbe*, une fois retranché le bruit d'estimation (≈ 3–6 hausses utilisables par bureau), est
**nulle** : k = 0 pour 100 % des bureaux, l'estimateur retombe *intégralement* sur la courbe
(RMSE identique à la 4ᵉ décimale). Ce n'est pas une hypothèse qu'on impose — c'est la
décomposition de variance qui le **découvre** : il n'y a pas assez de hausses de participation par
bureau pour estimer son penchant marginal assez précisément pour distinguer X de Y. L'intuition
est peut-être vraie dans le monde ; elle est **non identifiable sur ces données**. Quatre routes
indépendantes (GBM démographique, GBM régularisé, mélange socle+résidu, EB par station) butent sur
le même mur : aucun signal par-bureau transférable au-delà du niveau de gauche. *(Limite de ces
données — peu de scrutins comparables par bureau ; un historique plus long pourrait rouvrir la
question.)*

**« Le GBM était peut-être le mauvais algorithme ? » Testé — non plus** (`movability_gamma_algos.py`).
On couvre tout le spectre tabulaire (linéaire / bagging / boosting) et on corrige la validation :
le Ridge est réglé en **leave-one-election-out** (récompense le transfert inter-scrutin, pas
l'ajustement intra-ère, qui était le défaut de l'arrêt précoce aléatoire du GBM). RMSE de ΔLR hors
échantillon :

| algorithme | ≤2018→2022 | ≤2022→2024 | het OOS |
|---|---|---|---|
| **courbe 1-D (niveau)** | **5,34** | **5,46** | +0,29 / +0,43 |
| GBM (boosting, §13) | 6,36 | 7,14 | ≈ 0 |
| forêt aléatoire (bagging) | 6,34 | 7,68 | ≈ 0 |
| Ridge (linéaire, x-élection) | 6,00 | **11,49** | +0,06 / **−0,12** |

Les trois familles échouent, het ≈ 0, toutes battues par la courbe — quand linéaire, bagging et
boosting s'accordent, ce n'est **pas** l'algorithme (un réseau de neurones ne diffère pas, et ne
bat pas les GBM sur tabulaire). Le **blow-up du Ridge à 11,49 en 2024** est le plus instructif : la
relation linéaire features → γ la plus *stable sur 2002–2022* pointe **de mauvais sens** en 2024 —
« stable dans l'ère d'apprentissage » n'est pas « stable vers une recomposition ». Diagnostic :
**non-stationarité** de la relation features → γ (« ouvrier ⇒ mobilisable-gauche » vrai au temps du
PS industriel, faux à mesure que la gauche se recompose), insensible au choix d'algorithme — la
flexibilité sur-ajuste l'ère et empire le transfert. Seule la relation *niveau* → γ transfère
(loi de saturation structurelle, +0,96), parce qu'elle ne dépend pas de *quelles* démographies
définissent « la gauche » à une date donnée. Ce qui rouvrirait la question n'est pas un meilleur
algorithme mais **plus de données par unité** (histoire plus longue) ou une méthode d'*invariance*
(IRM) — qui ne peut toutefois que *sélectionner* un signal stable existant, ici quasi nul.

**« Si c'est de la non-stationarité, trouvons la variable qui la capture. »** Objection juste —
testée (`movability_gamma_nonstat.py`). (1) *Structure* : les vecteurs de coefs features→γ par
transition ne corrèlent qu'à **+0,18** entre scrutins (et **+0,21** au sein du même sens de marée) —
le motif démographique est *redessiné* à chaque scrutin, la seule dimension fortement stable
(niveau→γ, +0,96) étant déjà la courbe. (2) *Payoff* : on donne au modèle le **contexte national**
(marée ΔG) **et ses interactions** features×marée, et même **la vraie marée 2024** (test le plus
généreux) — RMSE 2024 : courbe **5,46** vs marée+interactions **10,09** (het **−0,21**). Conditionner
sur la marée ne restaure *pas* la stationarité. **La raison de fond** : la variable qui « capturerait »
l'instabilité est de **niveau-scrutin** (même valeur pour tous les bureaux d'une élection) ; pour
estimer comment elle module une carte features→γ de dimension 72, l'unité de réplication est
**l'élection**, et on en a **~9–14**. On ne peut pas apprendre une loi de modulation niveau-scrutin
sur neuf points et l'extrapoler à un régime 2024 inédit — c'est pourquoi le modèle à interactions
sur-ajuste et explose. L'intuition est correcte *en principe* (la non-stationarité est de
l'hétérogénéité non modélisée le long d'un axe) ; mais ici l'axe est (a) **haute dimension** (corr.
+0,18, non concentrée) et (b) **indexé par l'élection**, dont il faudrait des centaines et dont on a
une douzaine. Ça ne s'ingénie pas avec un meilleur modèle — seulement avec beaucoup plus d'élections.

**Within-élection vs across-élection — la démographie *marche*, mais pas pour prédire**
(`movability_gamma_within.py`). Avec 60 000 bureaux, la relation transversale démographie→γ est
bien dotée — il fallait le vérifier proprement : on coupe une transition en deux moitiés de bureaux,
on apprend sur l'une, on teste sur l'autre (mêmes bureaux test pour comparer *within* et *across*).
RMSE de ΔLR sur la moitié 2024 tenue à l'écart (34 781 bureaux) :

| modèle | RMSE | extra-het |
|---|---|---|
| niveau (courbe) | 5,46 | — |
| **démographie apprise sur l'autre moitié de *2024*** (within) | **4,18** | **+0,55** |
| démographie apprise sur les scrutins *passés* (across) | 11,15 | dir. +0,37 mais décalibré |

**Verdict net, et il tranche le débat** : *dans* une élection, la démographie **bat largement** le
niveau (5,46 → 4,18, signal validé sur bureaux tenus à l'écart) — donc « ce n'est pas que le niveau »
est **vrai en transversal**. Mais le **but est de prédire le scrutin suivant**, et là le même modèle
démographique appris sur le passé **explose** (11,15 ≫ 5,46) : la structure démographique de γ est
propre à chaque élection. Conséquence pour le livrable (un *instrument de prévision*, pas une
description rétrospective) : utiliser la démographie reviendrait à présenter la structure *propre à
2024* comme une loi — impressionnant sur 2024, pire que rien pour le scrutin d'après. **Seule la
courbe de niveau transfère** (+0,96) ; c'est donc, pour l'objectif de prévision, non un compromis
prudent mais le **seul choix défendable**. La richesse démographique est un mirage prédictif.

## 14. Abstention de fond vs conjoncturelle (retour client, câblé)

Retour client (fin de campagne) : on ne mobilise pas l'**abstentionniste de fond** (chronique,
qui ne vote jamais même quand l'enjeu monte). Le gisement de fin de campagne, c'est la frange
**conjoncturelle** — ceux qui reviennent voter quand la participation grimpe. Multiplier *tout*
le stock d'abstentionnistes par γ comptait des électeurs inatteignables.

**Identification.** Plancher d'abstention par bureau `abst_floor(b) = plus bas niveau
d'abstention démontré sur les Législatives T1 **strictement passées**` (= le meilleur niveau de
participation que le bureau a *réellement atteint*, hors élection cible) ;
`conjoncturelle(b) = max(0, abstention prédite(b) − abst_floor(b))`. Le gisement devient
`mobilisables(b) = inscrits(b) · conjoncturelle(b)/100 · γ(b)`. Câblé dans
`report_data.attach_abst_floor` (`gamma_panel.parquet` filtré sur `TARGET_TYPE` et
`date_float < TARGET_FLOOR_CUTOFF`) et `report_targets.conjunctural_pct`. Effet : 14,8 M
d'abstentionnistes → **0,57 M conjoncturels** (14,3 M de fond) ; gisement mobilisation
**0,25 M en métropole**.

**Deux correctifs de défendabilité (2026-05-31).** Le plancher initial (min poolé sur les trois
types) avait deux failles, découvertes en réponse à la question « ne faut-il pas renormaliser le
plancher par la prédiction ? » :

1. **Confusion de régimes.** Pooler legi+présid+euro laissait l'abstention présidentielle (forte
   participation) dominer le `min` : le plancher d'une projection *législative* incluait le
   décalage structurel legi↔présid, des électeurs qu'une campagne législative ne ramène pas.
   ⇒ on restreint au seul type projeté (`TARGET_TYPE`).
2. **Fuite de l'élection cible.** La cible (2024 legi) est **le législatif le plus mobilisé** du
   panel (abst. nationale 31 % vs 33,8 % pour le meilleur passé). Inclure 2024 faisait valoir le
   `min` à la valeur **observée 2024** pour **56 % des bureaux**, si bien que `conjoncturelle =
   pred − min ≈ pred − observé₂₀₂₄` = le **résidu de prédiction**, clippé positif (corr 0,75 ;
   64 % du gisement). On prédisait 2024 *et* on lisait 2024 dans le plancher. ⇒ on exclut la
   cible (`date_float < TARGET_FLOOR_CUTOFF`). corr résidu : 0,75 → 0,40.

**Renormalisation national+local testée puis écartée.** La voie « propre » suggérée — travailler
en écart à la moyenne nationale pour que le national s'annule (`plancher = ancre_proj +
min_passé[ AB − ancre_obs ]`), donc un signal purement local — a été implémentée et stress-testée.
Elle **échoue la validité faciale** : estimer l'écart local sur ~5 scrutins bruités produit des
planchers **hors bornes** (jusqu'à **−17 %** d'abstention) et des poches **« 70 % mobilisable »**
sur des bureaux isolés (la queue est immatérielle au national, 1,9 % du total, mais le produit
laisse **cliquer chaque bureau** — un plancher négatif affiché détruit la confiance). Sa
corrélation spatiale avec le plancher retenu n'est que +0,68 : la renormalisation *change* le
classement, mais au prix d'artefacts. **Verdict :** on garde le plancher en **niveau observé**
(∈ [0, prédiction], toujours valide, plafond *réellement atteint*). Le climat national de l'année
du min n'est pas retiré — assumé : un niveau que le bureau *a atteint* reste atteignable. C'est le
choix conservateur (0,25 M, le plus petit des candidats sans fuite ; min légi avec fuite donnait
0,56 M, renorm 0,34–0,61 M selon l'ancre). γ, lui, reste quasi ponctuel (bootstrap 95 %
[42,2 ; 42,6 %]) : toute la sensibilité du gisement est dans le plancher, pas dans γ. Balayage
reproductible dans `src/floor_sensitivity.py`.

**Historique du plancher.** Mesuré sur les Législatives T1 passées du panel `gamma_panel.parquet`
(§15) — ~5 scrutins/bureau. Le panel 3 types sert toujours l'estimation de γ (§15), mais **pas**
le plancher. Complément possible : `gauche(Prés. 2022) − gauche(Eur. 2024)` par bureau = le
conjoncturel de gauche *mesuré* (non modélisé), calculable depuis le même panel
(`preconisations.md`, reco 6).

## 15. γ dépend du type de scrutin (retour client, chiffré)

Retour client : *« le γ ne doit pas être la même formule selon les élections. »* Exact, et l'écart
est massif. Mêmes différences premières intra-type, **séparées par type de scrutin cible** :

| γ = part de gauche du votant marginal | valeur | n diffs |
|---|---|---|
| poolé (ancien défaut) | 31,5 % | 561 292 |
| **Législatives T1** | **39,3 %** | 314 280 |
| **Européennes T1** | **23,9 %** | 307 463 |
| **Présidentielle T1** | **12,3 %** | 247 012 |

Trois régimes de participation distincts. Les courbes γ(niveau de gauche) sont de **forme
contrastée** : législatives monotone croissante (24 → 56 %, l'électeur ramené penche d'autant
plus à gauche que le bureau l'est), présidentielle en U bas (18 → 8 → 19 %), européennes
intermédiaire mais montant fort en bastion (2 → 59 %). La courbe poolée mélangeait ces régimes
et n'était juste pour aucun scrutin réel.

**Données.** Les Européennes (1999–2024) sont bien dans le brut (`general_results`,
`elections.parquet`) ; elles n'étaient pas dans le cache du modèle (`cross_type_dev_base.parquet`,
legi+présid seuls). On les intègre via un **panel γ dédié** `data/baseline_cache/gamma_panel.parquet`
(`movability_turnout._ensure_panel` → `cross_type_ridge._build_block_scores` sur les trois types
T1), **sans toucher au modèle de production** (qui n'entraîne toujours que sur legi+présid). Le
plancher d'abstention (§14) lit aussi ce panel ⇒ abstention de fond sur historique 3 scrutins.

**Distinction avec §13.** §13 a montré que γ par *démographie / lags / momentum* ne transfère pas
(non stationnaire). Le **type de scrutin** est une autre dimension : c'est un régime de
participation, identifié et **stable intra-type** (la stabilité +0,96 du §11 vaut *au sein* d'un
type). On code donc en dur la courbe du type projeté (`report_data.TARGET_TYPE = "Legislatives_T1"`,
`movability_turnout.fit_gamma(election_type=…)`) et on sert les deux courbes au site
(`movability_turnout.curves_by_type` → `gamma_curve.json`, panneau « Qui revient voter quand la
participation monte »). Pour projeter une présidentielle, basculer `TARGET_TYPE` — le chiffre héros
change de régime (≈ 1,8 M au γ présidentiel), ce qui est l'honnêteté même : **le gisement n'existe
qu'au regard d'un scrutin.**
