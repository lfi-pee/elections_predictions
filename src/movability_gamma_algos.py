"""« GBM était-il le mauvais algorithme ? » — test d'autres biais inductifs et,
surtout, d'une **validation inter-élection** (et non aléatoire) pour la cartographie
features → γ.

Le GBM (§13) a échoué hors échantillon avec un het_corr ≈ 0 *malgré* un bon ajustement
en apprentissage : symptôme de **non-stationarité** (la relation features → γ change
d'un scrutin à l'autre), pas de manque de capacité. Deux objections légitimes restent :
(1) mauvais *biais* — on teste un linéaire (Ridge) et une forêt aléatoire (bagging),
qui couvrent le spectre tabulaire avec le boosting ; (2) mauvaise *validation* — l'arrêt
précoce du GBM se faisait sur un découpage *aléatoire* (récompense l'ajustement
intra-ère). Ici Ridge est réglé en **leave-one-election-out** : on ne récompense que ce
qui *transfère* d'un scrutin à l'autre. Si tous échouent comme le GBM, ce n'est pas
l'algorithme — c'est qu'il n'y a pas de signal features → γ stable au-delà du niveau.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.movability_gamma_rich import LEGI, curve_1d, flat, load

ALPHAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]


def ridge_xelec(
    tr: pd.DataFrame, feats: list[str], te: pd.DataFrame
) -> tuple[float, np.ndarray]:
    """Ridge sur features standardisées, α choisi en leave-one-election-out
    (RMSE de ΔLR groupée sur le scrutin tenu à l'écart) → ne garde que ce qui transfère."""
    med = tr[feats].median()
    sc = StandardScaler().fit(tr[feats].fillna(med))
    Z = sc.transform(tr[feats].fillna(med))
    y, w = tr.r.to_numpy(), tr.w.to_numpy()
    dT, dLR = tr.dT.to_numpy(), tr.dLR.to_numpy()
    elections = sorted(tr.date_float.unique())

    def cv(alpha: float) -> float:
        sse, n = 0.0, 0
        for e in elections:
            m = (tr.date_float != e).to_numpy()
            if m.all() or not m.any():
                continue
            g = np.clip(
                Ridge(alpha=alpha).fit(Z[m], y[m], sample_weight=w[m]).predict(Z[~m]),
                0,
                1,
            )
            sse += float(np.sum((dLR[~m] - g * dT[~m]) ** 2))
            n += int((~m).sum())
        return np.sqrt(sse / n)

    best = min(ALPHAS, key=cv)
    model = Ridge(alpha=best).fit(Z, y, sample_weight=w)
    pred = model.predict(sc.transform(te[feats].fillna(med)))
    return best, np.clip(pred, 0, 1)


def rf_oos(tr: pd.DataFrame, feats: list[str], te: pd.DataFrame) -> np.ndarray:
    med = tr[feats].median()
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=500,
        max_samples=0.15,
        n_jobs=-1,
        random_state=0,
    )
    rf.fit(tr[feats].fillna(med), tr.r, sample_weight=tr.w)
    return np.clip(rf.predict(te[feats].fillna(med)), 0, 1)


def run() -> None:
    o = load()
    print(
        f"{len(o.df):,} transitions | autres algos & validation inter-élection vs courbe"
    )
    for fit_max, d_to in [(2017.5, 2022.5), (2022.5, 2024.5)]:
        tr = o.df[o.df.date_float <= fit_max + 0.01]
        te = o.df[(o.df.election_type == LEGI) & np.isclose(o.df.date_float, d_to)]
        if not len(te):
            continue
        dT, actual = te.dT.to_numpy(), te.dLR.to_numpy()
        gf = flat(tr)
        rmse = lambda g: float(np.sqrt(np.mean((actual - g * dT) ** 2)))
        resid = actual - gf * dT
        het = lambda g: float(np.corrcoef((g - gf) * dT, resid)[0, 1])
        gc = curve_1d(tr, te.prevG.to_numpy())
        alpha, gr = ridge_xelec(tr, o.feats, te)
        grf = rf_oos(tr, o.feats, te)
        print(f"\n  ≤{fit_max:.0f} → legi {d_to:.0f} (n={len(te):,}) :")
        print(f"    flat                       RMSE {rmse(gf):7.4f}")
        print(
            f"    courbe1d (niveau)          RMSE {rmse(gc):7.4f}   het {het(gc):+.3f}"
        )
        print(
            f"    ridge (α={alpha:g}, x-élection) RMSE {rmse(gr):7.4f}   het {het(gr):+.3f}"
        )
        print(
            f"    forêt aléatoire            RMSE {rmse(grf):7.4f}   het {het(grf):+.3f}"
        )


if __name__ == "__main__":
    run()
