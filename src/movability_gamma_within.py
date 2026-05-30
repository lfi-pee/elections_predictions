"""La démographie dit-elle qui profite de la hausse de participation — *dans* une
élection (à travers les 60 000 bureaux), et est-ce que ça *transfère* d'une à l'autre ?

Correction d'un raccourci : l'argument « ~9 élections » vaut pour un modulateur de
niveau-scrutin. La relation transversale démographie → γ, elle, est estimée sur des
**dizaines de milliers de bureaux** — bien dotée. On sépare donc deux questions, sur les
**mêmes bureaux test** (moitié tenue à l'écart d'une transition) :
- **within** : on apprend γ(features) sur l'autre moitié des bureaux de *la même* élection ;
- **across** : on apprend sur les élections *passées* (≤ fit_max).
Si la démographie bat le niveau en *within* mais pas en *across*, alors elle porte bien un
signal transversal réel — mais **propre à chaque élection** (non transférable) : descriptif
oui, instrument validé non. `extra_het` = corrélation de l'ajustement démographique
(γ_rich − γ_niveau)·ΔT avec ce que le niveau a raté, sur les bureaux tenus à l'écart.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.movability_gamma_rich import LEGI, curve_1d, flat, load

ALPHA = 1000.0


def run() -> None:
    o = load()
    df = o.df
    med = df[o.feats].median()
    Z = StandardScaler().fit_transform(df[o.feats].fillna(med))
    y, w = df.r.to_numpy(), df.w.to_numpy()
    rng = np.random.RandomState(0)
    print(
        f"{len(df):,} transitions | within-élection (même scrutin) vs across (scrutins passés)"
    )

    for fit_max, d_to in [(2017.5, 2022.5), (2022.5, 2024.5)]:
        te = (df.election_type == LEGI) & np.isclose(df.date_float, d_to)
        idx = np.where(te.to_numpy())[0]
        if not len(idx):
            continue
        half = rng.rand(len(idx)) < 0.5
        A, B = idx[half], idx[~half]  # A = moitié apprentissage même scrutin ; B = test
        past = np.where((df.date_float <= fit_max + 0.01).to_numpy())[0]

        dT_B, dLR_B = df.dT.to_numpy()[B], df.dLR.to_numpy()[B]
        prevG_B = df.prevG.to_numpy()[B]
        gf = flat(df.iloc[past])
        rmse = lambda g: float(np.sqrt(np.mean((dLR_B - g * dT_B) ** 2)))
        resid = dLR_B - gf * dT_B
        het = lambda g: float(np.corrcoef((g - gf) * dT_B, resid)[0, 1])

        def fit_pred(train_idx, level_src):
            gc = curve_1d(df.iloc[level_src], prevG_B)
            gr = np.clip(
                Ridge(alpha=ALPHA)
                .fit(Z[train_idx], y[train_idx], sample_weight=w[train_idx])
                .predict(Z[B]),
                0,
                1,
            )
            return gc, gr

        gc_w, gr_w = fit_pred(
            A, A
        )  # within : tout depuis l'autre moitié du même scrutin
        gc_a, gr_a = fit_pred(past, past)  # across : tout depuis les scrutins passés

        # extra_het : l'ajustement démographique (rich − niveau) capte-t-il le résidu du niveau ?
        eh = lambda gr, gc: float(
            np.corrcoef((gr - gc) * dT_B, dLR_B - gc * dT_B)[0, 1]
        )
        print(
            f"\n  test = moitié tenue à l'écart de legi {d_to:.0f} ({len(B):,} bureaux) :"
        )
        print(f"    niveau (courbe)                      RMSE {rmse(gc_a):7.4f}")
        print(
            f"    WITHIN  démographie (même scrutin)   RMSE {rmse(gr_w):7.4f}   "
            f"het {het(gr_w):+.3f}   extra-het {eh(gr_w, gc_w):+.3f}"
        )
        print(
            f"    ACROSS  démographie (scrutins passés) RMSE {rmse(gr_a):7.4f}   "
            f"het {het(gr_a):+.3f}   extra-het {eh(gr_a, gc_a):+.3f}"
        )


if __name__ == "__main__":
    run()
