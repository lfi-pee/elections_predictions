"""« Si c'est de la non-stationarité, trouvons la variable qui la capture. »

Objection juste : si la cartographie features → γ n'est pas figée mais *tourne* avec un
observable (quelle gauche est en jeu, le sens de la marée nationale), alors cet observable
est la variable manquante — la conditionner restaure la stationarité. Deux tests :

1. **Diagnostic** : la relation features → γ (coefs Ridge standardisés) est-elle stable
   d'une transition à l'autre ? Corrélation moyenne des vecteurs de coefs par transition,
   et corrélation moyenne *au sein* des transitions de même sens de marée (montante /
   descendante). Proche de 0 ⇒ le motif est redessiné à chaque scrutin, rien à extrapoler.
2. **Payoff** : on inclut explicitement le **contexte national** (marée ΔG de la transition)
   et son **interaction** avec les features — γ via [Z, marée, Z×marée] — puis on apprend
   ≤2022 et on prédit 2024 *en lui donnant la vraie marée 2024* (le test le plus généreux :
   on lui offre le contexte que normalement on ignorerait). Si même là on ne bat pas la
   courbe de niveau, la marée nationale ne capture pas la non-stationarité.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.movability_gamma_rich import LEGI, curve_1d, flat, load

ALPHA = 1000.0
MIN_N = 2000


def transition_table(o, Z: np.ndarray) -> pd.DataFrame:
    """Un vecteur de coefs Ridge (features → r) par transition + sa marée ΔG nationale."""
    df = o.df
    key = df.election_type + "@" + df.date_float.round(1).astype(str)
    rows = []
    for lab, idx in df.groupby(key).indices.items():
        if len(idx) < MIN_N:
            continue
        coef = (
            Ridge(alpha=ALPHA)
            .fit(Z[idx], df.r.to_numpy()[idx], sample_weight=df.w.to_numpy()[idx])
            .coef_
        )
        rows.append(
            {
                "t": lab,
                "tide": float(df.dG.to_numpy()[idx].mean()),
                "n": len(idx),
                "coef": coef,
            }
        )
    return pd.DataFrame(rows)


def coef_stability(tab: pd.DataFrame) -> None:
    C = np.vstack(tab.coef.to_numpy())
    R = np.corrcoef(C)
    off = R[np.triu_indices_from(R, k=1)]
    rising = tab.tide.to_numpy() > 0
    within = []
    for grp in (rising, ~rising):
        if grp.sum() >= 2:
            sub = np.corrcoef(C[grp])
            within.append(sub[np.triu_indices_from(sub, k=1)].mean())
    print(
        f"  {len(tab)} transitions (n≥{MIN_N}) | marées ΔG de {tab.tide.min():+.1f} à {tab.tide.max():+.1f}"
    )
    print(f"  corr. moyenne des coefs features→γ entre transitions : {off.mean():+.3f}")
    print(
        f"  corr. moyenne au sein du même sens de marée           : {np.mean(within):+.3f}"
    )
    print("  (proche de 0 ⇒ le motif démographique est redessiné à chaque scrutin)")


def augment(Z: np.ndarray, tide: np.ndarray) -> np.ndarray:
    return np.hstack([Z, tide[:, None], Z * tide[:, None]])


def payoff(o, Z: np.ndarray, fit_max: float, d_to: float) -> None:
    df = o.df
    tr = df.date_float <= fit_max + 0.01
    te = (df.election_type == LEGI) & np.isclose(df.date_float, d_to)
    if not te.any():
        return
    tr, te = tr.to_numpy(), te.to_numpy()
    tide = df.dG.to_numpy()
    dT, actual = df.dT.to_numpy()[te], df.dLR.to_numpy()[te]
    gf = flat(df[tr])
    rmse = lambda g: float(np.sqrt(np.mean((actual - g * dT) ** 2)))
    resid = actual - gf * dT
    het = lambda g: float(np.corrcoef((g - gf) * dT, resid)[0, 1])
    gc = curve_1d(df[tr], df.prevG.to_numpy()[te])

    y, w = df.r.to_numpy(), df.w.to_numpy()
    m_z = Ridge(alpha=ALPHA).fit(Z[tr], y[tr], sample_weight=w[tr])
    g_z = np.clip(m_z.predict(Z[te]), 0, 1)
    Xa = augment(Z, tide)
    m_i = Ridge(alpha=ALPHA).fit(Xa[tr], y[tr], sample_weight=w[tr])
    g_i = np.clip(m_i.predict(Xa[te]), 0, 1)

    print(
        f"\n  ≤{fit_max:.0f} → legi {d_to:.0f} (marée 2024 ΔG={tide[te].mean():+.1f}) :"
    )
    print(f"    courbe1d (niveau)            RMSE {rmse(gc):7.4f}   het {het(gc):+.3f}")
    print(
        f"    ridge features seules        RMSE {rmse(g_z):7.4f}   het {het(g_z):+.3f}"
    )
    print(
        f"    ridge + marée + (features×marée) RMSE {rmse(g_i):7.4f}   het {het(g_i):+.3f}"
    )


def run() -> None:
    o = load()
    o.df["dG"] = o.df["Gauche"] - o.df["prevG"]  # marée locale ΔG (gauche exprimée)
    med = o.df[o.feats].median()
    Z = StandardScaler().fit_transform(o.df[o.feats].fillna(med))
    print(f"{len(o.df):,} transitions station×scrutin")
    print("\n[1] Stabilité de la cartographie features→γ d'une transition à l'autre :")
    coef_stability(transition_table(o, Z))
    print(
        "\n[2] La marée nationale + ses interactions capturent-elles la non-stationarité ?"
    )
    for fit_max, d_to in [(2017.5, 2022.5), (2022.5, 2024.5)]:
        payoff(o, Z, fit_max, d_to)


if __name__ == "__main__":
    run()
