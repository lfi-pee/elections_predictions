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

Carte de fond CARTO + bibliothèque MapLibre via CDN (accès réseau requis à
l'affichage). La recherche de commune et la recomposition de scénario s'exécutent
intégralement côté client.

## Comment ça marche

La prédiction est *moyenne nationale + écart local*. Les curseurs ajoutent un
décalage national à chaque bureau et la carte se recolore instantanément
(`fill-color` recalculé côté client) ; les compteurs de bascule sont exacts sur
les 69 358 bureaux via `national.json`. La finesse au bureau n'est jamais un
nuage national : on agrège en symboles par commune au dézoom, on dissout en
polygones bureau au zoom, et on entre par recherche — seul le département actif
est en mémoire.
