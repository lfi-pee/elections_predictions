"""Mobilité vers la gauche par bureau : le chargement β sur la marée commune.

Voir `MOVABILITY.md`. β_b = réponse de la part de gauche (exprimés) du bureau à la
marée nationale de gauche, estimée en **différences premières intra-type** (le bureau
est son propre témoin ; l'écart fixe et la dérive sont retirés), puis **rétrécie**
(empirical Bayes) vers une moyenne prédite par la représentation (démo + géo +
trajectoire `dev_lag`). On ne lit jamais β sur le modèle de production : sa forme
`ancre + écart figé` impose β≡1 (swing uniforme). On l'estime sur le panel.

En électeurs : gagnables(b, ΔL) = exprimés_b · β_b · ΔL / 100 (canal persuasion ;
la mobilisation/participation est un axe distinct). Validé hors échantillon : β
estimé ≤2022 doit battre le swing uniforme pour prédire le mouvement 2022→2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CACHE = Path("data/baseline_cache/cross_type_dev_base.parquet")
INDIC = Path("data/baseline_cache/cross_type_dev_indicators.txt")
GEN = Path("data/elections/agregees/general_results.parquet")
OUT = Path("data/report/movability.parquet")

SCORE, TIDE, TRAJ = "Gauche", "natmean_Gauche", "dev_Gauche_lag1"
GEO = ["latitude", "longitude"]
LEGI, PRES = "Legislatives_T1", "Presidentielle_T1"
VAL = 2024.5


@dataclass
class Slopes:
    loc: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    n: np.ndarray


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    indic = INDIC.read_text().strip().split("\n")
    cols = ["location", "election_type", "date_float", SCORE, TIDE, TRAJ, *GEO, *indic]
    df = pd.read_parquet(CACHE, columns=cols)
    return df, indic


def fd_slopes(df: pd.DataFrame, rising_only: bool = False) -> Slopes:
    """Pente β par bureau en différences premières intra-type, avec dérive (intercept).
    β = OLS de ΔG sur ΔL ; se via résidus ; dof = n−2 (n ≥ 3 requis)."""
    d = df.sort_values(["location", "election_type", "date_float"])
    g = d.groupby(["location", "election_type"], sort=False)
    dg = (d[SCORE] - g[SCORE].shift(1)).to_numpy()
    dl = (d[TIDE] - g[TIDE].shift(1)).to_numpy()
    loc = d["location"].to_numpy()
    m = ~np.isnan(dg) & ~np.isnan(dl)
    if rising_only:
        m &= dl > 0
    t = pd.DataFrame({"loc": loc[m], "x": dl[m], "y": dg[m]})
    a = t.groupby("loc").agg(
        n=("x", "size"),
        sx=("x", "sum"),
        sy=("y", "sum"),
        sxx=("x", lambda v: float((v * v).sum())),
        syy=("y", lambda v: float((v * v).sum())),
    )
    xy = t.assign(xy=t.x * t.y).groupby("loc").xy.sum()
    a["sxy"] = xy
    n = a.n.to_numpy().astype(float)
    det = n * a.sxx - a.sx**2
    det = det.to_numpy()
    ok = (n >= 3) & (det > 1e-9)
    beta = np.full(len(a), np.nan)
    se = np.full(len(a), np.inf)
    beta[ok] = (
        n[ok] * a.sxy.to_numpy()[ok] - a.sx.to_numpy()[ok] * a.sy.to_numpy()[ok]
    ) / det[ok]
    inter = (a.sy.to_numpy()[ok] - beta[ok] * a.sx.to_numpy()[ok]) / n[ok]
    rss = (
        a.syy.to_numpy()[ok]
        - inter * a.sy.to_numpy()[ok]
        - beta[ok] * a.sxy.to_numpy()[ok]
    )
    var_b = np.maximum(rss, 0) / (n[ok] - 2) * (n[ok] / det[ok])
    se[ok] = np.sqrt(np.maximum(var_b, 1e-12))
    return Slopes(a.index.to_numpy(), beta, se, n)


def shrink(s: Slopes, cov: pd.DataFrame) -> np.ndarray:
    """Empirical Bayes : β̂ = μ(x) + τ²/(τ²+se²)·(β_raw − μ(x)). μ = Ridge pondéré
    (1/se²) sur la représentation ; τ² par moments. Les bureaux à faible n (se=∞)
    retombent sur μ(x)."""
    raw = s.beta
    have = np.isfinite(raw) & np.isfinite(s.se)
    w = np.where(have, 1.0 / np.maximum(s.se**2, 1e-6), 0.0)
    X = StandardScaler().fit_transform(cov.to_numpy().astype(float))
    fill = np.nanmedian(raw[have])
    y = np.where(have, raw, fill)
    mu = Ridge(alpha=10.0).fit(X[have], y[have], sample_weight=w[have]).predict(X)
    r = raw - mu
    wv = w[have] / w[have].sum()
    tau2 = max(0.0, float(np.sum(wv * r[have] ** 2)) - float(np.mean(s.se[have] ** 2)))
    k = np.where(have, tau2 / (tau2 + s.se**2), 0.0)
    return mu + k * np.where(have, r, 0.0)


def per_bureau_cov(df: pd.DataFrame, indic: list[str], loc: np.ndarray) -> pd.DataFrame:
    last = (
        df.sort_values("date_float").groupby("location").tail(1).set_index("location")
    )
    traj = df.groupby("location")[TRAJ].mean()
    cov = last[[*GEO, *indic]].copy()
    cov["traj"] = traj
    cov = cov.reindex(loc)
    return cov.fillna(cov.median())


def context() -> pd.DataFrame:
    g = pq.read_table(
        GEN,
        columns=[
            "id_election",
            "id_brut_miom",
            "exprimes",
            "libelle_commune",
            "code_departement",
            "code_commune",
        ],
    ).to_pandas()
    g = g[g.id_election == "2024_legi_t1"].drop(columns="id_election")
    return g.rename(columns={"id_brut_miom": "location"}).drop_duplicates("location")


def mainland(dep: pd.Series) -> np.ndarray:
    d = dep.astype(str)
    return ~(d.str.startswith("Z") | d.str.match(r"9[789]")).to_numpy()


def backtest(
    df: pd.DataFrame, indic: list[str], fit_max: float, d_from: float, d_to: float
) -> dict[str, float]:
    """Hors échantillon : β estimé sur les scrutins ≤ fit_max, prédit le mouvement
    legi d_from→d_to. Compare β hétérogène (rétréci) au swing uniforme (β≡1).
    `het_corr` = corrélation de la part hétérogène (β−1)·ΔL avec le réel ΔG−ΔL."""
    tr = df[df.date_float <= fit_max + 0.01]
    s = fd_slopes(tr)
    beta = pd.Series(shrink(s, per_bureau_cov(tr, indic, s.loc)), index=s.loc)
    legi = df[df.election_type == LEGI]
    gf = legi[np.isclose(legi.date_float, d_from)].set_index("location")[SCORE]
    gt = legi[np.isclose(legi.date_float, d_to)].set_index("location")[SCORE]
    tide = legi.groupby("date_float")[TIDE].first()
    dl = float(
        tide[tide.index[np.isclose(tide.index, d_to)][0]]
        - tide[tide.index[np.isclose(tide.index, d_from)][0]]
    )
    idx = gf.index.intersection(gt.index).intersection(beta.index)
    actual = (gt.reindex(idx) - gf.reindex(idx)).to_numpy()
    b = beta.reindex(idx).to_numpy()
    rmse = lambda p: float(np.sqrt(np.nanmean((actual - p) ** 2)))
    het_a, het_p = actual - dl, (b - 1.0) * dl
    ok = np.isfinite(het_a) & np.isfinite(het_p)
    corr = float(np.corrcoef(het_p[ok], het_a[ok])[0, 1])
    return {
        "transition": f"{d_from:.0f}→{d_to:.0f}",
        "n": int(ok.sum()),
        "dL": round(dl, 2),
        "rmse_uniform": round(rmse(np.full_like(actual, dl)), 3),
        "rmse_beta": round(rmse(b * dl), 3),
        "het_corr": round(corr, 4),
    }


def grouped_backtest(
    df: pd.DataFrame, gmap: dict[str, str], fit_max: float, d_from: float, d_to: float
) -> dict[str, float]:
    """β mis en commun par groupe (ex. département) sur ≤ fit_max : la moyenne tue
    le bruit par bureau. Teste si la mobilité est une propriété *régionale* stable."""
    tr = df[df.date_float <= fit_max + 0.01].sort_values(
        ["location", "election_type", "date_float"]
    )
    g = tr.groupby(["location", "election_type"], sort=False)
    dg = (tr[SCORE] - g[SCORE].shift(1)).to_numpy()
    dl = (tr[TIDE] - g[TIDE].shift(1)).to_numpy()
    grp = tr["location"].map(gmap).to_numpy()
    m = ~np.isnan(dg) & ~np.isnan(dl) & pd.notna(grp)
    t = pd.DataFrame({"grp": grp[m], "x": dl[m], "y": dg[m]})
    a = t.groupby("grp").agg(
        n=("x", "size"),
        sx=("x", "sum"),
        sy=("y", "sum"),
        sxx=("x", lambda v: float((v * v).sum())),
    )
    a["sxy"] = t.assign(xy=t.x * t.y).groupby("grp").xy.sum()
    det = a.n * a.sxx - a.sx**2
    beta_g = (a.n * a.sxy - a.sx * a.sy) / det.where(det > 1e-9)
    legi = df[df.election_type == LEGI]
    gf = legi[np.isclose(legi.date_float, d_from)].set_index("location")[SCORE]
    gt = legi[np.isclose(legi.date_float, d_to)].set_index("location")[SCORE]
    tide = legi.groupby("date_float")[TIDE].first()
    dl0 = float(
        tide[tide.index[np.isclose(tide.index, d_to)][0]]
        - tide[tide.index[np.isclose(tide.index, d_from)][0]]
    )
    idx = gf.index.intersection(gt.index)
    actual = (gt.reindex(idx) - gf.reindex(idx)).to_numpy()
    b = pd.Series(idx.map(gmap)).map(beta_g).to_numpy()
    het_a, het_p = actual - dl0, (b - 1.0) * dl0
    ok = np.isfinite(het_a) & np.isfinite(het_p)
    rmse = lambda p: float(np.sqrt(np.nanmean((actual[ok] - p[ok]) ** 2)))
    return {
        "transition": f"{d_from:.0f}→{d_to:.0f}",
        "n_groups": int(beta_g.notna().sum()),
        "rmse_uniform": round(rmse(np.full_like(actual, dl0)), 3),
        "rmse_beta": round(rmse(b * dl0), 3),
        "het_corr": round(float(np.corrcoef(het_p[ok], het_a[ok])[0, 1]), 4),
    }


def build() -> None:
    df, indic = load_panel()
    s = fd_slopes(df)
    cov = per_bureau_cov(df, indic, s.loc)
    beta = shrink(s, cov)
    raw_dir = fd_slopes(df, rising_only=True)
    beta_dir = pd.Series(
        shrink(raw_dir, per_bureau_cov(df, indic, raw_dir.loc)), index=raw_dir.loc
    )

    out = pd.DataFrame(
        {"location": s.loc, "beta": beta, "beta_raw": s.beta, "se": s.se, "n": s.n}
    )
    out["beta_rising"] = beta_dir.reindex(s.loc).to_numpy()
    ctx = context()
    out = out.merge(ctx, on="location", how="left")
    out["ml"] = mainland(out.code_departement.fillna("Z"))
    out["gettable_per_pt"] = out.exprimes.fillna(0) * out.beta / 100.0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    ml = out[out.ml]
    nat = float(ml.gettable_per_pt.sum())
    amp = float((out.beta > 1).mean())
    print(
        f"β median {np.nanmedian(out.beta):.3f} | amplificateurs (β>1) {amp:.1%} "
        f"| β rising median {np.nanmedian(out.beta_rising):.3f}"
    )
    print(f"gagnables / +1 pt national gauche (métropole) : {nat:,.0f} électeurs")
    print("\nBACKTEST hors échantillon (β hétérogène vs swing uniforme β≡1) :")
    for fit_max, d_from, d_to in [
        (2012.5, 2012.5, 2017.5),
        (2017.5, 2017.5, 2022.5),
        (2022.5, 2022.5, VAL),
    ]:
        bt = backtest(df, indic, fit_max, d_from, d_to)
        win = "β GAGNE" if bt["rmse_beta"] < bt["rmse_uniform"] else "uniforme gagne"
        print(
            f"  {bt['transition']} (β≤{fit_max:.0f}) n={bt['n']:,} ΔL={bt['dL']:+} | "
            f"RMSE unif {bt['rmse_uniform']} vs β {bt['rmse_beta']} | "
            f"corr hétéro {bt['het_corr']:+.3f} → {win}"
        )
    gmap = out.set_index("location").code_departement.to_dict()
    print("\nBACKTEST groupé par DÉPARTEMENT (β mis en commun, bruit moyenné) :")
    for fit_max, d_from, d_to in [
        (2012.5, 2012.5, 2017.5),
        (2017.5, 2017.5, 2022.5),
        (2022.5, 2022.5, VAL),
    ]:
        bt = grouped_backtest(df, gmap, fit_max, d_from, d_to)
        win = "β GAGNE" if bt["rmse_beta"] < bt["rmse_uniform"] else "uniforme gagne"
        print(
            f"  {bt['transition']} ({bt['n_groups']} dépts) | RMSE unif {bt['rmse_uniform']} "
            f"vs β {bt['rmse_beta']} | corr hétéro {bt['het_corr']:+.3f} → {win}"
        )
    dep = (
        ml.groupby("libelle_commune")
        .gettable_per_pt.sum()
        .sort_values(ascending=False)
        .head(12)
    )
    print("\nDéploiement (top communes, électeurs / +1 pt) :")
    for nm, v in dep.items():
        print(f"  {nm:24s} {v:8,.0f}")


if __name__ == "__main__":
    build()
