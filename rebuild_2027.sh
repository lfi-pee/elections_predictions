#!/usr/bin/env bash
# Reconstruction COMPLÈTE du site 2027 depuis les données brutes — une commande.
#
#   ./rebuild_2027.sh
#
# Reproduit chaque nombre servi (report_app/2027/data/*.json) à partir du cache d'élections/
# démographie, puis VÉRIFIE que Python et le JavaScript du site calculent le même modèle.
# Aucune valeur affichée n'est saisie à la main : tout descend de ce pipeline.
#
# Variables :
#   SKIP_FORECAST=1  saute la ré-estimation du modèle de déviation (étape lente) et réutilise
#                    data/predictions_2027.csv — utile quand seul le post-traitement a changé.
set -euo pipefail
cd "$(dirname "$0")"

# Node (pour le test de parité JS) : PATH ou installation locale.
export PATH="$HOME/.local/node/bin:$PATH"

echo "══ 1/6  Modèle de déviation par bureau → data/predictions_2027.csv"
if [ "${SKIP_FORECAST:-0}" = "1" ]; then
  echo "   (sauté : SKIP_FORECAST=1 ; réutilise predictions_2027.csv)"
else
  python3 -u -m src.forecast_2027
fi

echo "══ 2/6  Données servies (summary / circo / communes) + backtests + motif radical + γ"
python3 -u -m src.report_data_2027

echo "══ 3/6  Parité Python ↔ JavaScript (le site calcule-t-il le même modèle ?)"
if command -v node >/dev/null 2>&1; then
  python3 -u -m src.test_parity_2027
else
  echo "   ⚠ Node absent — parité NON vérifiée. Installez Node pour la garantie de non-dérive."
fi

echo "══ 4/6  Cohérence : aucun chiffre affiché n'est figé (doit venir des données servies)"
python3 -u -m src.test_no_hardcoded_2027

echo "══ 5/6  Attribution des voix régionalistes (Guyane, Antilles, Pacifique) → couverture"
python3 -u -m src.attribution_2027
python3 -u -m src.test_attribution_2027

echo "══ 6/6  Garde-fou : aucun score publié là où la nomenclature ne couvre TOUJOURS pas"
echo "         l'électorat (Corse : mouvance LIOT, hors axe G/CD/ED)"
python3 -u -m src.coverage_2027
if command -v node >/dev/null 2>&1; then
  python3 -u -m src.test_coverage_2027
else
  echo "   ⚠ Node absent — rendu du garde-fou NON vérifié."
fi

echo "══ Validation 2024 (rappel) :"
python3 -u -m src.backtest_2024_firstround

echo "✅ Reconstruction terminée. Géométrie (circo.geojson, insets) = couche stable, non"
echo "   régénérée ici ; la rebâtir seulement si le découpage change :"
echo "     python3 -u -m src.report_geo_2027 && python3 -u -m src.report_geo_circo_2027"
