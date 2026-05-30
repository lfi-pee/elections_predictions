"""Canal mobilisation : la part de gauche du votant marginal (γ), identifiée et stable.

Voir `MOVABILITY.md` §11. γ = d(gauche % inscrits)/d(participation) = part de gauche
de l'électeur *marginal* (l'ex-abstentionniste qui se déplace quand la participation
monte) — quantité **identifiée** (lue sur les vraies hausses de participation), pas le
partage des exprimés (circulaire). On l'estime par différences premières intra-type,
on la lit par décile de niveau de gauche du bureau, et on s'en sert pour chiffrer le
gisement mobilisation : `mobilisables(b) = abstentionnistes(b) × γ(level_b)`.

γ croît avec le niveau de gauche (23 % → 47 %) mais sature (γ − G : +11 à droite,
−17 en bastion), et la courbe γ(décile) ancienne vs récente corrèle à +0,96 : c'est
une régularité comportementale exploitable, contrairement au chargement β (§10).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data/baseline_cache/cross_type_dev_base.parquet")


def panel_diffs() -> pd.DataFrame:
    """Différences premières intra-type : ΔT (participation) et ΔLR (gauche % inscrits),
    avec le niveau de gauche G du bureau. Filtre les mouvements de participation > 0,5 pt."""
    df = pd.read_parquet(
        CACHE,
        columns=["location", "election_type", "date_float", "Gauche", "Abstention"],
    )
    df["T"] = 100.0 - df["Abstention"]
    df["LR"] = df["T"] * df["Gauche"] / 100.0
    df = df.sort_values(["location", "election_type", "date_float"])
    g = df.groupby(["location", "election_type"], sort=False)
    dT = (df["T"] - g["T"].shift(1)).to_numpy()
    dLR = (df["LR"] - g["LR"].shift(1)).to_numpy()
    date = df["date_float"].to_numpy()
    m = ~np.isnan(dT) & ~np.isnan(dLR) & (np.abs(dT) > 0.5)
    return pd.DataFrame(
        {"dT": dT[m], "dLR": dLR[m], "G": df["Gauche"].to_numpy()[m], "date": date[m]}
    )


def gamma_curve(t: pd.DataFrame, nbins: int = 10) -> pd.DataFrame:
    """γ (part de gauche du votant marginal, %) par bin de niveau de gauche."""
    t = t.assign(bin=pd.qcut(t["G"], nbins, labels=False, duplicates="drop"))
    rows = []
    for b, grp in t.groupby("bin"):
        sxx = float((grp.dT**2).sum())
        sxy = float((grp.dT * grp.dLR).sum())
        rows.append(
            (
                int(b),
                float(grp.G.mean()),
                (sxy / sxx) * 100 if sxx else np.nan,
                len(grp),
            )
        )
    return pd.DataFrame(rows, columns=["decile", "G_moyen", "gamma_pct", "n"])


def fit_gamma(nbins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Courbe γ(niveau de gauche) prête à l'application : (niveaux, γ %) croissants."""
    res = gamma_curve(panel_diffs(), nbins).sort_values("G_moyen")
    return res.G_moyen.to_numpy(), res.gamma_pct.to_numpy()


def apply_gamma(curve: tuple[np.ndarray, np.ndarray], levels: np.ndarray) -> np.ndarray:
    """γ(level) par bureau, interpolé sur la courbe (extrapolation plate aux bords)."""
    g, gv = curve
    return np.clip(np.interp(levels, g, gv), 0.0, 100.0)


def slope(res: pd.DataFrame) -> float:
    return float(np.polyfit(res.G_moyen, res.gamma_pct, 1, w=np.sqrt(res.n))[0])


def run() -> None:
    t = panel_diffs()
    res = gamma_curve(t)
    print(
        "γ = part de gauche de l'électeur marginal (mobilisé), par décile de gauche :"
    )
    print(f"{'décile':>6} {'G moyen':>9} {'γ (%)':>8} {'γ − G':>8} {'n diffs':>9}")
    for r in res.itertuples():
        print(
            f"{r.decile:>6} {r.G_moyen:>9.1f} {r.gamma_pct:>8.1f} "
            f"{r.gamma_pct - r.G_moyen:>+8.1f} {r.n:>9,}"
        )
    print(f"\nPente γ vs niveau de gauche (tout) : {slope(res):+.3f} pt/pt.")

    early, late = gamma_curve(t[t.date <= 2012.6]), gamma_curve(t[t.date > 2012.6])
    print(
        f"Pente γ — transitions ≤2012 : {slope(early):+.3f} | >2012 : {slope(late):+.3f}"
    )
    merged = early.merge(late, on="decile", suffixes=("_e", "_l"))
    rho = float(np.corrcoef(merged.gamma_pct_e, merged.gamma_pct_l)[0, 1])
    print(
        f"Corrélation des courbes γ(décile) ancienne vs récente : {rho:+.3f} "
        "(proche de 1 ⇒ régularité stable, contrairement au chargement β)."
    )


if __name__ == "__main__":
    run()
