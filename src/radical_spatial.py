"""Motif spatial de la part RADICALE (LFI) DANS le vote de gauche, par circonscription.

Le modèle de sièges divise le bloc de gauche en pôles (radical/LFI vs sociaux-démocrates) selon
une part `rad`. Longtemps posée quasi CONSTANTE (curseur ± 0,006·dG, écart-type ~0,02), elle
variait en réalité beaucoup d'une circo à l'autre (bastions vs reste). On mesure ce motif sur
2017 — la seule législative à gauche réellement divisée avec des étiquettes de parti distinctes
(FI / SOC / ECO / COM / …) — et on l'exporte comme DÉVIATION recentrée (moyenne pondérée 0), à
ajouter au curseur (qui, lui, fixe la MOYENNE nationale, d'après les sondages).

`RAD_GAIN` amplifie la dispersion 2017 mesurée (×3 ≈ écart-type effectif ~0,50). À la part LFI
sondée (~0,37), la part de SIÈGES LFI passe ainsi de ~17 % (modèle cassé, falaise à 0,5) à ~21 %
sur la carte 2027 par défaut (~31 % sur les parts réelles 2024), courbe désormais LISSE. On ne
va pas au-delà : la mécanique de 1er tour divisé PLAFONNE vers ~29 % (au-delà, augmenter le gain
ne fait que saturer les bornes). Le repère 2024 (42 %) est un plafond d'UNION négociée,
structurellement INATTEIGNABLE en compétition divisée — ~21-29 % encadre la part réelle de
sièges LFI de la gauche divisée de 2017. Voir src/lfi_split_validate.py.

    python3 -u -m src.radical_spatial   # imprime moyenne/écart-type + couverture
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.reunif_measure import _commune2circo, CAND

# Pôles au sein de la gauche 2017.
_RAD = {"FI", "FG", "COM", "EXG", "DXG"}
_SD = {"SOC", "VEC", "ECO", "ECOLO", "DVG", "RDG", "PRG", "GEN", "UG", "NUP"}

# Amplification de la dispersion 2017 mesurée (cf. calibrage, docstring). rad_circo =
# clamp(curseur + RAD_GAIN · déviation_2017, 0.05, 0.95).
RAD_GAIN = 3.0


def radical_deviation() -> dict[str, float]:
    """{circo → déviation de la part radicale-dans-la-gauche} (recentrée, moyenne pondérée ≈ 0),
    mesurée sur 2017_legi_t1, circo au découpage actuel (mapping commune majoritaire)."""
    c2c = _commune2circo(drop_split=True)
    c = pd.read_parquet(CAND, columns=["id_election", "code_commune", "nuance", "voix"])
    c = c[c.id_election == "2017_legi_t1"].copy()
    c["circo"] = c.code_commune.astype(str).map(c2c)
    c = c.dropna(subset=["circo"])
    c["p"] = c.nuance.map(lambda n: "R" if n in _RAD else ("S" if n in _SD else None))
    tab = c.dropna(subset=["p"]).groupby(["circo", "p"]).voix.sum().unstack("p").fillna(0.0)
    tab["left"] = tab.get("R", 0.0) + tab.get("S", 0.0)
    tab = tab[tab.left > 0]
    tab["rs"] = tab["R"] / tab["left"]
    mean = float(np.average(tab.rs, weights=tab.left))
    return {ci: float(rs - mean) for ci, rs in tab.rs.items()}


if __name__ == "__main__":
    d = radical_deviation()
    v = np.array(list(d.values()))
    print(f"circos couvertes : {len(d)}")
    print(f"déviation part radicale — écart-type {v.std():.3f}, min {v.min():+.3f}, max {v.max():+.3f}")
    print(f"RAD_GAIN = {RAD_GAIN} → écart-type effectif appliqué ≈ {RAD_GAIN * v.std():.3f}")
