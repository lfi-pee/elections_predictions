"""A/B des estimateurs du bloc « Autre » sous LOO (leave-one-legislative-election-out),
métrique = R² de part de vote hors-échantillon (comme src/autre_oof.py).

But : le ridge de production sur-régularise l'Autre (α global élevé → écrase les bastions,
R²≈0.14). L'Autre étant zéro-gonflé et dominé par la persistance, on teste des estimateurs
qui s'appuient sur le lag. On ne garde un variant que s'il bat 0.14 SANS dégrader les circos
quasi-nulles (métrique globale) — même porte que le reste du modèle.

    python3 -u -m src.autre_ab
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

from src import cross_type_dev as D
from src.forecast_2027 import PCA_K, _feat_cols, fit_block, _transform

TC = "Other"


def main():
    df, demo, _nm, _polls = D.load_cross_type_data(Path("data"))
    D.add_election_type_onehot(df)
    demo_cols, feat_all = _feat_cols(demo)
    legi = df[df.election_type == D.VAL_TYPE].copy()
    raw_lag = [f"{b}_lag{k}" for b in D.BLOCKS_ABS for k in (1, 2)]
    dev_lag = [f"dev_{b}_lag{k}" for b in D.BLOCKS_ABS for k in (1, 2)]
    legi = legi.dropna(subset=demo + raw_lag + dev_lag + ["natmean_Other", "Other"]).copy()
    dates = sorted(legi.date_float.unique())
    print(f"{len(legi):,} lignes legi, plis {[round(float(d),1) for d in dates]}")

    k = PCA_K[TC]

    def ridge_dev(train, held):
        scaler, pca, ridge = fit_block(TC, train, feat_all, demo_cols, k)
        dp = ridge.predict(_transform(scaler, pca, len(demo_cols), held, feat_all))
        ins = held["inscrits"].to_numpy(float)
        ins = np.where(np.isfinite(ins) & (ins > 0), ins, 1.0)
        return dp - np.average(dp, weights=ins)

    def persist(held, lags=("dev_Other_lag1",)):
        return np.mean([held[c].to_numpy(float) for c in lags], axis=0)

    # Estimateurs : renvoient la déviation prédite sur les lignes `held`.
    ESTIM = {
        "ridge (prod)":        lambda tr, he: ridge_dev(tr, he),
        "persist lag1":        lambda tr, he: persist(he, ("dev_Other_lag1",)),
        "persist lag1+2":      lambda tr, he: persist(he, ("dev_Other_lag1", "dev_Other_lag2")),
        "blend .5 ridge+lag1": lambda tr, he: 0.5 * ridge_dev(tr, he) + 0.5 * persist(he),
        "low-alpha ridge":     lambda tr, he: _low_alpha(tr, he, demo_cols, feat_all),
        "hurdle lag":          lambda tr, he: _hurdle(tr, he),
    }

    # LOO : accumule (actual_share, pred_share) hors-échantillon par estimateur.
    acc = {name: {"t": [], "p": []} for name in ESTIM}
    for d in dates:
        held = legi[np.isclose(legi.date_float, d)]
        train = legi[~np.isclose(legi.date_float, d)]
        if len(held) == 0 or len(train) == 0:
            continue
        nat = held["natmean_Other"].to_numpy(float)
        act = held["Other"].to_numpy(float)
        for name, fn in ESTIM.items():
            share = np.clip(fn(train, held) + nat, 0.0, 100.0)
            acc[name]["t"].extend(act)
            acc[name]["p"].extend(share)

    # R² global + R² sur les bastions (Other réel ≥ 15 %) + fausses alertes (Other réel < 5,
    # prédit > 10 → bruit métropolitain).
    print(f"\n{'estimateur':22}{'R² global':>11}{'R² bastions':>13}{'MAE':>8}{'FP%':>7}")
    for name in ESTIM:
        yt = np.array(acc[name]["t"]); yp = np.array(acc[name]["p"])
        r2 = r2_score(yt, yp)
        strong = yt >= 15
        r2s = r2_score(yt[strong], yp[strong]) if strong.sum() > 2 else float("nan")
        mae = float(np.mean(np.abs(yt - yp)))
        fp = float(np.mean((yt < 5) & (yp > 10)) * 100)
        print(f"  {name:20}{r2:11.3f}{r2s:13.3f}{mae:7.2f}p{fp:6.2f}%")


def _low_alpha(train, held, demo_cols, feat_all):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Xtr = train[feat_all].to_numpy(np.float64); Xhe = held[feat_all].to_numpy(np.float64)
    sc = StandardScaler().fit(Xtr); Xtr, Xhe = sc.transform(Xtr), sc.transform(Xhe)
    nd = len(demo_cols)
    pca = PCA(n_components=PCA_K[TC]).fit(Xtr[:, :nd])
    Xtr = np.hstack([pca.transform(Xtr[:, :nd]), Xtr[:, nd:]])
    Xhe = np.hstack([pca.transform(Xhe[:, :nd]), Xhe[:, nd:]])
    r = Ridge(alpha=1.0).fit(Xtr, train[f"dev_{TC}"].to_numpy(np.float64))
    dp = r.predict(Xhe)
    ins = held["inscrits"].to_numpy(float); ins = np.where(np.isfinite(ins) & (ins > 0), ins, 1.0)
    return dp - np.average(dp, weights=ins)


def _hurdle(train, held):
    """Là où le lag indique un territoire à Autre (Other_lag1 ≥ 8 %), on prédit par persistance ;
    ailleurs, déviation ≈ 0 (le national ~1,8 % suffit). Data-driven, pas de liste codée."""
    lag1_share = held["Other_lag1"].to_numpy(float)
    dev_lag1 = held["dev_Other_lag1"].to_numpy(float)
    return np.where(lag1_share >= 8.0, dev_lag1, 0.0)


if __name__ == "__main__":
    main()
