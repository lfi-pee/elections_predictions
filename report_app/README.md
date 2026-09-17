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

`2027/negotiation.html`, accessible depuis la carte 2027 : un seul tableau, vu de LFI. Par
circonscription, la **chance d'élire un·e député·e LFI** si LFI porte la candidature unique de la
gauche, la **force réelle** (sans accord, LFI seule atteindrait-elle le 2nd tour ? selon la part
nationale de LFI, réglable), et une **posture** (exiger / obtenir / monnaie d'échange / rien)
calculée depuis les prédictions du modèle seulement (dont, pour les circos hors de portée de
LFI, la chance de la gauche unie avec une candidature moyenne) ; à droite les chiffres à mettre
sur la table (sortant·e, parti NFP et sort du siège 2024, gauche 2024 ventilée par nuance, LFI
seule 2017, part de Mélenchon dans le vote de gauche 2022 avec la moyenne nationale, calcul
simple 2027 réparti LFI / PS / Écologistes / PCF ; aucune sortie du modèle). La posture
n'utilise jamais un chiffre de droite.
Colonnes redimensionnables, export CSV.
Le report vers une candidature LFI est mesuré sur le second tour 2024 (parti de chaque
candidat·e d'union connu par la répartition du NFP, `data/nuance/nfp_repartition_2024.csv`) ;
les député·es en exercice viennent de l'open data de l'Assemblée (`data/nuance/deputes_2026.csv`).

```bash
python3 -m src.label_effect_2024     # mesure → 2027/data/label_effect_2024.json
python3 -m src.deputes_an            # sortant·es → data/nuance/deputes_2026.csv
python3 -m src.negotiation_2027      # Monte-Carlo (~20 s) → 2027/data/negotiation.json
python3 -m src.test_negotiation_2027 && python3 -m src.test_negotiation_page_2027
```

Méthode détaillée : `2027/METHODOLOGY.md` §3 ter. `rebuild_2027.sh` enchaîne ces étapes.

## Résultats passés par circonscription (données)

`2027/data/comparison_history.json` (produit par `report_comparison_2027`) : parts de bloc au
1er tour des législatives 2017/2022/2024 par circonscription (LFI seule séparable en 2017,
nuance FI). Il n'y a plus de page dédiée : ces colonnes sont intégrées au tableau de la page
« Négocier », dans le groupe « arguments à mettre sur la table », avec les deux extrapolations
simples de 2024 (« 2024 + évolution nationale », « 2024 × évolution nationale »).

```bash
python -m src.report_comparison_2027  # reconstruit comparison_history.json
```


