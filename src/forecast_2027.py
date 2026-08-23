"""Prévision **2027** des Législatives (1er tour) au bureau de vote.

Différences avec la démonstration 2024 (`conformal.py`) :

- **2024 rejoint l'entraînement.** Il n'y a plus d'élection « de validation » tenue à
  l'écart : les six législatives 2002→2024 servent toutes à entraîner et à calibrer. La
  preuve de compétence reste la validation croisée (chaque scrutin retiré à tour de rôle),
  2024 en étant désormais un pli.

- **La cible 2027 est construite, pas observée.** Pour chaque bureau, on décale les
  déviations d'un scrutin : `dev_lag1(2027) = dev(2024 legi)`, `dev_lag2(2027) =
  dev_lag1(2024) = dev(2022 legi)`. La démographie (dernier millésime INSEE) est
  reportée telle quelle. Aucune reconstruction du cache lourd n'est nécessaire.

- **L'ancre nationale est un curseur.** Le modèle de déviation donne, par bureau et par
  bloc, `dev_pred_b` — **indépendant du niveau national**. Le site calcule en direct
  `pred_b = national_b(curseur) + dev_pred_b`, borné à [0, 100]. On exporte donc les
  `dev_pred_b` (et non des prédictions figées), plus les demi-largeurs conformes.

- **Intervalles = erreur de déviation seule (mode oracle).** Puisque le niveau national
  est *posé par l'utilisateur*, l'erreur d'estimation nationale ne revient pas au modèle :
  la fourchette ne borne que l'incertitude *locale* (résidu de déviation), calibrée par
  validation croisée sur les scrutins passés, stratifiée par territoire.

Sortie : `data/predictions_2027.csv` (une ligne par bureau × bloc) avec `dev_pred`,
`hw_80/hw_90/hw_95` (demi-largeurs), `lag_fallback`, et une prédiction de référence au
scénario par défaut (colonne `pred_ref`) pour contrôle.

    python3 -u -m src.forecast_2027
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from src.conformal import (
    ALPHA_GRID,
    INTERVAL_ALPHAS,
    per_territory_intervals,
    territory_class,
)
from src.cross_type_dev import BLOCKS_ABS, TARGET_COLS, load_cross_type_data
from src import scenarios_2027

VAL_TYPE = "Legislatives_T1"
TARGET_DATE = 2027.5  # législatives « à l'heure » : juin 2027 (année + 6/12)

# Config Ridge par bloc, identique au site 2024 (LOO-sélectionnée, cf. conformal.BEST_RIDGE) :
# légi-only, déviation-lags, PCA sur la démographie.
PCA_K = {
    "Gauche": 5,
    "Centre+Droite": 7,
    "Extreme_Droite": 5,
    "Abstention": 5,
}


def _feat_cols(demo_indicators: list[str]) -> tuple[list[str], list[str]]:
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    non_demo = dev_lag1 + dev_lag2
    return demo_indicators, demo_indicators + non_demo


def build_2027_rows(legi: pd.DataFrame, feat_all: list[str]) -> pd.DataFrame:
    """Construit les lignes 2027 à partir des lignes 2024 legi : décalage des déviations
    d'un scrutin (2024→lag1, 2022→lag2), démographie inchangée."""
    l24 = legi[np.isclose(legi["date_float"], 2024.5, atol=1e-2)].copy()
    f = l24.copy()
    for b in BLOCKS_ABS:
        # lag2(2027) = lag1(2024) = 2022 legi ; lag1(2027) = dev(2024) observé.
        f[f"dev_{b}_lag2"] = l24[f"dev_{b}_lag1"].to_numpy()
        f[f"dev_{b}_lag1"] = l24[f"dev_{b}"].to_numpy()
    f["date_float"] = TARGET_DATE
    # Un bureau dont la déviation 2024 ou 2022 manque bascule en repli (moins sûr).
    lag_cols = [f"dev_{b}_lag{k}" for b in BLOCKS_ABS for k in (1, 2)]
    prior_fb = l24["lag_fallback"].to_numpy() if "lag_fallback" in l24 else np.zeros(len(l24), bool)
    f["lag_fallback"] = prior_fb | f[lag_cols].isna().any(axis=1).to_numpy()
    return f


def fit_block(tc, train, feat_all, demo_cols, k):
    """Ajuste PCA(démo)+RidgeCV sur les déviations `dev_<tc>` de tous les plis legi.
    Renvoie (scaler, pca, ridge)."""
    scaler = StandardScaler().fit(train[feat_all].to_numpy(np.float64))
    X = scaler.transform(train[feat_all].to_numpy(np.float64))
    n_d = len(demo_cols)
    pca = PCA(n_components=k).fit(X[:, :n_d]) if k else None
    Xt = np.hstack([pca.transform(X[:, :n_d]), X[:, n_d:]]) if k else X
    ridge = RidgeCV(alphas=ALPHA_GRID).fit(Xt, train[f"dev_{tc}"].to_numpy(np.float64))
    return scaler, pca, ridge


def _transform(scaler, pca, n_d, frame, feat_all):
    X = scaler.transform(frame[feat_all].to_numpy(np.float64))
    return np.hstack([pca.transform(X[:, :n_d]), X[:, n_d:]]) if pca is not None else X


def loo_dev_residuals(tc, train, feat_all, demo_cols, k, alpha_):
    """Résidus de **déviation** hors-échantillon : chaque scrutin legi retiré à tour de
    rôle, on prédit sa déviation et on prend `dev_observé − dev_prédit` (mode oracle :
    le niveau national n'entre pas, il sera posé au curseur)."""
    n_d = len(demo_cols)
    dates = sorted(train["date_float"].unique())
    res, terr = [], []
    Xraw = StandardScaler().fit(train[feat_all].to_numpy(np.float64))
    X_all = Xraw.transform(train[feat_all].to_numpy(np.float64))
    dev_y = train[f"dev_{tc}"].to_numpy(np.float64)
    locs = train["location"].to_numpy()
    for d in dates:
        held = np.isclose(train["date_float"].to_numpy(), d, atol=1e-3)
        not_held = ~held
        if k:
            pca = PCA(n_components=k).fit(X_all[not_held][:, :n_d])
            Xtr = np.hstack([pca.transform(X_all[not_held][:, :n_d]), X_all[not_held][:, n_d:]])
            Xh = np.hstack([pca.transform(X_all[held][:, :n_d]), X_all[held][:, n_d:]])
        else:
            Xtr, Xh = X_all[not_held], X_all[held]
        r = Ridge(alpha=alpha_, solver="cholesky").fit(Xtr, dev_y[not_held])
        res.append(dev_y[held] - r.predict(Xh))
        terr.append(np.array([territory_class(loc) for loc in locs[held]]))
    return np.concatenate(res), np.concatenate(terr)


def main() -> None:
    t0 = time.time()
    data_dir = Path("data")
    print("Chargement des données (cache)…")
    df, demo_indicators, national_means, poll_feats = load_cross_type_data(data_dir)
    demo_cols, feat_all = _feat_cols(demo_indicators)

    legi = df[df["election_type"] == VAL_TYPE].copy()
    raw_lag = [f"{b}_lag{k}" for b in BLOCKS_ABS for k in (1, 2)]
    dev_lag = [f"dev_{b}_lag{k}" for b in BLOCKS_ABS for k in (1, 2)]
    legi = legi.dropna(subset=demo_indicators + raw_lag + dev_lag)
    print(f"  Legi complet : {len(legi):,} lignes, plis {sorted(legi.date_float.round(2).unique())}")

    f27 = build_2027_rows(legi, feat_all)
    # Les lignes 2027 doivent avoir toutes les features (démo + dev-lags décalés).
    f27 = f27.dropna(subset=feat_all)
    print(f"  Cible 2027 : {len(f27):,} bureaux (repli lag : {int(f27.lag_fallback.sum()):,})")

    frames = []
    ref = scenarios_2027.SCENARIOS
    ref_means = next(s for s in ref if s["key"] == scenarios_2027.DEFAULT_SCENARIO)["means"]
    ref_map = {"Gauche": "G", "Centre+Droite": "CD", "Extreme_Droite": "ED", "Abstention": "AB"}

    ins27 = f27["inscrits"].to_numpy(np.float64)
    ins27 = np.where(np.isfinite(ins27) & (ins27 > 0), ins27, 1.0)
    for tc in TARGET_COLS:
        k = PCA_K[tc]
        scaler, pca, ridge = fit_block(tc, legi, feat_all, demo_cols, k)
        dev_pred = ridge.predict(_transform(scaler, pca, len(demo_cols), f27, feat_all))
        # On **centre** la déviation prédite (moyenne pondérée par les inscrits = 0) : le
        # modèle ne fournit que le *motif spatial* de l'écart au national ; le niveau
        # national absolu est posé par le curseur. Ainsi `moyenne(pred_b) = curseur_b`
        # exactement — le curseur veut dire ce qu'il affiche.
        dev_pred = dev_pred - np.average(dev_pred, weights=ins27)

        cal_res, cal_terr = loo_dev_residuals(tc, legi, feat_all, demo_cols, k, ridge.alpha_)
        val_terr = np.array([territory_class(loc) for loc in f27["location"].to_numpy()])
        hw = {}
        for a in INTERVAL_ALPHAS:
            pct = int(100 * (1 - a))
            hw[pct], _ = per_territory_intervals(cal_res, cal_terr, val_terr, a)

        pred_ref = np.clip(dev_pred + ref_means[ref_map[tc]], 0.0, 100.0)
        frames.append(
            pd.DataFrame(
                {
                    "location": f27["location"].to_numpy(),
                    "block": tc,
                    "dev_pred": np.round(dev_pred, 4),
                    "pred_ref": np.round(pred_ref, 4),
                    "hw_80": np.round(hw[80], 4),
                    "hw_90": np.round(hw[90], 4),
                    "hw_95": np.round(hw[95], 4),
                    "lag_fallback": f27["lag_fallback"].to_numpy(),
                }
            )
        )
        print(
            f"  {tc:16s} α={ridge.alpha_:.1e}  dev∈[{dev_pred.min():.1f},{dev_pred.max():.1f}]  "
            f"médiane |résidu| CV = {np.median(np.abs(cal_res)):.1f}pp  hw90≈{np.median(hw[90]):.1f}pp"
        )

    out = pd.concat(frames, ignore_index=True)
    out_path = data_dir / "predictions_2027.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n  Écrit {len(out):,} lignes → {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
