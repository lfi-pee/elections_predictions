# L'élection au bureau de vote près — site interactif

Site statique (aucun backend) qui présente les prédictions du modèle au bureau
de vote, en français, pour un décideur de campagne. Une carte MapLibre sert de
colonne vertébrale ; le récit et les commandes se déroulent dans le rail de
droite, sur une seule descente d'écran.

## Reconstruire les données

```bash
uv run python -m src.report_build      # ~5 min (dont SHAP)
```

Produit dans `report_app/data/` :

| Fichier | Rôle |
|---|---|
| `summary.json` | chiffres d'accroche + courbe de bascule + précision-par-marge + cibles |
| `communes.json` | agrégat commune (couche nationale + recherche), ~6 Mo |
| `national.json` | tableaux compacts par bureau `pg/pc/pe/ins/m/t` (compteurs live exacts), ~2,5 Mo |
| `provenance.json` | part locale vs national de l'incertitude, par bloc |
| `coverage.json` | couverture conforme empirique (servie en pastille numérique) |
| `bv/<dept>.geojson` | polygones bureau, simplifiés, chargés à la demande (dont `t` pour les cibles) |
| `detail/<dept>.json` | réel / intervalles / SHAP par bureau (panneau au clic) |
| `fig_method.svg` | schéma de méthode (seule figure statique restante) |

## Servir

```bash
cd report_app && python -m http.server 8000   # http://localhost:8000
```

Fond de carte vectoriel OpenFreeMap (données OpenStreetMap) + bibliothèque MapLibre
via CDN (accès réseau requis à l'affichage) : ni clé ni quota, contrairement aux
tuiles raster CARTO qui exigent désormais une clé d'API. La recherche de commune
et la recomposition de scénario s'exécutent intégralement côté client.

## Comment ça marche

La prédiction est *moyenne nationale + écart local*. Les curseurs ajoutent un
décalage national à chaque bureau et la carte se recolore instantanément
(`fill-color` recalculé côté client) ; les compteurs de bascule sont exacts sur
les 69 358 bureaux via `national.json`. La finesse au bureau n'est jamais un
nuage national : on agrège en symboles par commune au dézoom, on dissout en
polygones bureau au zoom, et on entre par recherche — seul le département actif
est en mémoire.

## Négocier les circonscriptions (LFI) — 2027

`2027/negotiation.html`, accessible depuis la carte 2027 : pour une gauche unie, la chance de
siège de chaque circonscription **avec une étiquette LFI** et **avec une autre étiquette de
gauche**, leur différence (le prix pour l'union), le **rapport de force** (qui, seul, se
qualifierait au 2nd tour, selon la part nationale de LFI réglable), une **posture** par circo
(exiger / disputer / obtenir / difficile / monnaie d'échange / rien), un classement, une courbe
« combien en demander » et l'export CSV. L'effet d'étiquette est mesuré sur le second tour 2024 (parti de
chaque candidat·e d'union connu par la répartition du NFP, `data/nuance/nfp_repartition_2024.csv`) ;
les député·es en exercice viennent de l'open data de l'Assemblée (`data/nuance/deputes_2026.csv`).

```bash
python3 -m src.label_effect_2024     # mesure → 2027/data/label_effect_2024.json
python3 -m src.deputes_an            # sortant·es → data/nuance/deputes_2026.csv
python3 -m src.negotiation_2027      # Monte-Carlo (~20 s) → 2027/data/negotiation.json
python3 -m src.test_negotiation_2027 && python3 -m src.test_negotiation_page_2027
```

Méthode détaillée : `2027/METHODOLOGY.md` §3 ter. `rebuild_2027.sh` enchaîne ces étapes.

## Comparateur 2027

`2027/comparison.html`, accessible depuis la carte 2027, compare les scores de premier
tour (toute la gauche, LFI seule, autre gauche, centre+droite, extrême droite) : prédiction
2027 du modèle, variante réglable (niveau de toute la gauche, part de LFI dans la gauche,
abstention), législatives 2017/2022/2024, et deux extrapolations simples de 2024 (« 2024 +
évolution nationale », en points ; « 2024 × évolution nationale », en %). Tri par défaut :
prédiction 2027 décroissante. « LFI seule » n'est séparable qu'en 2017 (nuance FI).
Les colonnes historiques utilisent les suffrages exprimés au dénominateur et les
identifiants de circonscription du scrutin (2017/2022) ou le fichier officiel par
circonscription (2024). Les nuances et attributions sont détaillées dans le JSON.
La référence est le scénario par défaut des données servies. La variante utilise dès
l'ouverture l'abstention nationale observée au premier tour de 2024 (33,29 %), avec
les niveaux de vote de référence conservés avant couplage de participation. Le menu
propose aussi la participation de 2022 (52,49 % d'abstention) et la référence identique.
Les taux sont calculés depuis les totaux d'abstentions et d'inscrits des résultats,
puis embarqués avec leur source dans le JSON ; ce sont des hypothèses de sensibilité.
Modifier un curseur passe en réglage personnalisé ; réappliquer le préréglage restaure
le dernier choix. Le libellé de colonne et le CSV identifient le préréglage actif. Les deux modèles simples utilisent le
niveau national effectif de référence pour les trois blocs et le résidu Autre.
Les projections non publiables selon la couverture du site restent vides.

```bash
python -m src.report_comparison_2027  # reconstruit comparison_history.json
```

L'export CSV conserve le filtre, le tri, le bloc observé, les paramètres et les
sources. La reconstruction complète `rebuild_2027.sh` inclut cet export historique.
