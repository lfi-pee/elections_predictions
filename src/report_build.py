"""Point d'entrée unique : reconstruit tout le socle de données du site.

    uv run python -m src.report_build           # tout
    uv run python -m src.report_build --no-shap  # sans le précalcul SHAP (lent)
    uv run python -m src.report_build --no-prov  # sans la provenance (relance conforme)

Étapes : scan contours → table maître + agrégats (dont gisement mobilisation γ) →
figures → provenance de l'incertitude → polygones par département → détail SHAP par
bureau. Puis servir `report_app/` en statique.
"""

from __future__ import annotations

import sys

from src import (
    report_data,
    report_figs,
    report_geo,
    report_provenance,
    report_shap,
)


def main() -> None:
    skip_shap = "--no-shap" in sys.argv
    skip_prov = "--no-prov" in sys.argv
    report_geo.scan()
    report_data.build()
    report_figs.build()
    if not skip_prov:
        report_provenance.build()
    if not skip_shap:
        report_shap.build()
    report_geo.export()
    print("\nPrêt. Servir : cd report_app && python -m http.server 8000")


if __name__ == "__main__":
    main()
