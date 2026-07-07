"""Prépare le socle de données du site (table maître BV, agrégat commune, stats).

Sorties (cache `data/report/`, servi dans `report_app/data/`) :
- `bv_master.parquet` : une ligne par bureau (prédictions, intervalles, marge,
  bloc en tête, statut disputé, point de bascule ED, inscrits, centroïde).
- `communes.json` : agrégat commune pondéré par inscrits (couche nationale + index
  de recherche).
- `summary.json` : chiffres d'accroche vérifiés.

Le détail polygones par département est produit par `report_geo.py`, les
explications SHAP par `report_shap.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import movability_turnout, report_targets

PRED_CSV = Path("data/predictions_with_intervals.csv")
GEN = Path("data/elections/agregees/general_results.parquet")
CENTROIDS = Path("data/geo/bv_contour_centroids.parquet")
LOC_COORDS = Path("data/geo/location_coords.parquet")
COMMUNE_COORDS = Path("data/geo/commune_coords.parquet")
VAL_ELECTION = "2024_legi_t1"
# Les contours de circonscription sont stables 2012–2024 ; le scrutin 2024 ne porte
# pas le code, on reprend la carte bureau→circo du dernier législatif qui le porte.
CIRCO_SRC = "2022_legi_t1"

CACHE = Path("data/report")
SERVED = Path("report_app/data")
# Panel 3 types (legi + présid + **européennes**) : sert l'estimation de γ (`movability_turnout`).
# Le plancher d'abstention (`attach_abst_floor`) ne lit que le type projeté, hors élection
# cible, et renormalise dans le cadre national+local du modèle (voir la fonction).
TURNOUT_CACHE = Path("data/baseline_cache/gamma_panel.parquet")
# Le scrutin projeté par le livrable (Législatives T1) ⇒ courbe γ de CE type.
TARGET_TYPE = "Legislatives_T1"
# Le plancher exclut l'élection cible elle-même (sinon le min vaut sa valeur observée et la
# part « conjoncturelle » dégénère en résidu de prédiction) : on ne lit que les scrutins
# strictement antérieurs à cette date (date_float du scrutin 2024 visé = 2024,5).
TARGET_FLOOR_CUTOFF = 2024.0

BLOCKS = {
    "Gauche": "G",
    "Centre+Droite": "CD",
    "Extreme_Droite": "ED",
    "Abstention": "AB",
}
VOTE = ["G", "CD", "ED"]
LEVELS = (80, 90, 95)


def load_wide() -> pd.DataFrame:
    df = pd.read_csv(PRED_CSV)
    df["b"] = df["block"].map(BLOCKS)
    metrics = {"prediction": "pred", "actual": "act"}
    metrics |= {f"{s}_{lvl}": f"{s}{lvl}" for lvl in LEVELS for s in ("lower", "upper")}
    frames = {}
    for col, short in metrics.items():
        w = df.pivot(index="location", columns="b", values=col)
        frames |= {f"{short}_{b}": w[b] for b in w.columns}
    wide = pd.DataFrame(frames).reset_index()
    # lag_fallback is per-BV (same across blocks): attach once, don't pivot.
    if "lag_fallback" in df.columns:
        fb = df.groupby("location")["lag_fallback"].first().rename("lag_fallback")
        wide = wide.merge(fb, on="location", how="left")
        wide["lag_fallback"] = wide["lag_fallback"].fillna(False).astype(bool)
    else:
        wide["lag_fallback"] = False
    return wide


def load_context() -> pd.DataFrame:
    cols = [
        "id_election",
        "id_brut_miom",
        "inscrits",
        "code_departement",
        "code_commune",
        "libelle_commune",
        "code_canton",
        "libelle_canton",
        "code_circonscription",
    ]
    g = pq.read_table(GEN, columns=cols).to_pandas()
    g = g[g.id_election == VAL_ELECTION].drop(columns="id_election")
    return g.rename(columns={"id_brut_miom": "location"})


def attach_circo(df: pd.DataFrame) -> pd.DataFrame:
    g = pq.read_table(
        GEN,
        columns=[
            "id_election",
            "id_brut_miom",
            "code_departement",
            "code_circonscription",
        ],
    ).to_pandas()
    g = g[(g.id_election == CIRCO_SRC) & g.code_circonscription.notna()]
    g["circo"] = (
        g.code_departement.astype(str) + "-" + g.code_circonscription.astype(str)
    )
    m = g.drop_duplicates("id_brut_miom").set_index("id_brut_miom").circo
    return df.merge(m.rename("circo"), left_on="location", right_index=True, how="left")


def attach_centroid(df: pd.DataFrame) -> pd.DataFrame:
    c = pq.read_table(
        CENTROIDS, columns=["id_brut_miom", "latitude", "longitude"]
    ).to_pandas()
    c = c.rename(
        columns={"id_brut_miom": "location", "latitude": "lat", "longitude": "lon"}
    ).drop_duplicates("location")
    df = df.merge(c, on="location", how="left")
    miss = df.lat.isna()
    if miss.any():
        lc = (
            pq.read_table(LOC_COORDS)
            .to_pandas()
            .rename(columns={"latitude": "lat2", "longitude": "lon2"})
        )
        df = df.merge(lc, on="location", how="left")
        df.loc[miss, "lat"] = df.loc[miss, "lat2"]
        df.loc[miss, "lon"] = df.loc[miss, "lon2"]
        df = df.drop(columns=["lat2", "lon2"])
    df["has_contour"] = ~df.location.isin(_contourless())
    return df


def _contourless() -> set[str]:
    path = CACHE / "contourless.json"
    return set(json.loads(path.read_text())) if path.exists() else set()


def attach_abst_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Plancher d'abstention par bureau = **plus bas niveau d'abstention démontré** sur les
    législatives **strictement passées** (hors élection cible). Deux exigences tenues :

    - **Hors fuite.** La cible 2024 étant le législatif le plus mobilisé, l'inclure faisait
      valoir le min à sa valeur 2024 pour 56 % des bureaux → la part « conjoncturelle »
      dégénérait en **résidu de prédiction** (`pred − observé₂₀₂₄`). On exclut donc la cible.
    - **Validité faciale.** On garde le min en **niveau observé** (∈ [0, prédiction]), un
      plancher *réellement atteint* par le bureau, donc atteignable par construction. La
      renormalisation national+local (écart prédit − meilleur écart passé) a été testée et
      **écartée** : estimer l'écart local sur ~5 scrutins bruités produit des planchers
      hors bornes (jusqu'à −17 %) et des poches « 70 % mobilisable » sur des bureaux isolés —
      indéfendable sur un produit où chaque bureau se clique. Le climat national de l'année
      du min n'est pas retiré, et c'est assumé : un niveau que le bureau *a atteint* reste
      atteignable, quel que soit le climat de cette année-là. C'est le choix conservateur."""
    c = pd.read_parquet(
        TURNOUT_CACHE, columns=["location", "election_type", "date_float", "Abstention"]
    )
    c = c[(c.election_type == TARGET_TYPE) & (c.date_float < TARGET_FLOOR_CUTOFF)]
    floor = c.groupby("location").Abstention.min().rename("abst_floor")
    df = df.merge(floor, left_on="location", right_index=True, how="left")
    df["abst_floor"] = df.abst_floor.fillna(df.pred_AB).clip(upper=df.pred_AB)
    return df


def swing_weights(df: pd.DataFrame) -> np.ndarray:
    """National vote-share weights (G/CD/ED sum to 1): a national swing for one
    bloc is drawn from the others in proportion to their size."""
    ins = df.inscrits.to_numpy().astype(float)
    base = np.array([np.average(df[f"pred_{b}"], weights=ins) for b in VOTE])
    return base / base.sum()


def conserved(deltas: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Vote-share-conserving deltas: applied_j = d_j − Σ_{k≠j} d_k·w_j/(1−w_k).
    Each bloc's gain is funded proportionally by the others; the three sum to 0."""
    d = np.asarray(deltas, dtype=float)
    pull = np.array(
        [sum(d[k] * w[j] / (1 - w[k]) for k in range(3) if k != j) for j in range(3)]
    )
    return d - pull


def derive(df: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    pred = df[[f"pred_{b}" for b in VOTE]].to_numpy()
    order = np.argsort(-pred, axis=1)
    top, second = order[:, 0], order[:, 1]
    rows = np.arange(len(df))
    df["lead"] = [VOTE[i] for i in top]
    df["runner_up"] = [VOTE[i] for i in second]
    df["margin"] = pred[rows, top] - pred[rows, second]
    ed, w_e = VOTE.index("ED"), w[VOTE.index("ED")]
    cross = [
        (pred[:, j] - pred[:, ed]) / (1 + w[j] / (1 - w_e)) for j in range(3) if j != ed
    ]
    df["ed_tip"] = np.maximum.reduce(cross)
    widths = [df[f"upper90_{b}"] - df[f"lower90_{b}"] for b in VOTE]
    df["unc"] = np.mean(widths, axis=0)
    return df


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    """Médiane pondérée : la valeur où la moitié des *inscrits* est en deçà."""
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(0.5 * w.sum(), cum, x))


def _weights(df: pd.DataFrame) -> np.ndarray:
    """Poids = inscrits du bureau (le grain porteur d'électeurs). Un bureau vaut ce
    qu'il pèse en électeurs : c'est la métrique qui compte pour l'usage GOTV, et elle
    dégonfle la longue traîne des micro-bureaux dont la part observée est du bruit
    d'échantillonnage (n≈15 votants ⇒ ±13 pts sur une part à 50 %)."""
    return df.inscrits.to_numpy().astype(float)


def lead_accuracy(df: pd.DataFrame) -> float:
    """Part **des inscrits** dont le bureau voit son bloc en tête correctement prédit."""
    pred = df[[f"pred_{b}" for b in VOTE]].to_numpy().argmax(1)
    act = df[[f"act_{b}" for b in VOTE]].to_numpy().argmax(1)
    w = _weights(df)
    m = np.isfinite(w) & (w > 0)
    return float(np.average((pred == act)[m], weights=w[m]))


def r2_by_block(df: pd.DataFrame) -> dict[str, float]:
    """R² hors échantillon par bloc, **pondéré par les inscrits**, calculé sur les
    prédictions 2024 réellement servies (mêmes lignes que
    `predictions_with_intervals.csv`) — plus de valeurs en dur."""
    w0 = _weights(df)
    out: dict[str, float] = {}
    for b in BLOCKS.values():
        a, p = df[f"act_{b}"].to_numpy(), df[f"pred_{b}"].to_numpy()
        m = np.isfinite(a) & np.isfinite(p) & np.isfinite(w0) & (w0 > 0)
        a, p, w = a[m], p[m], w0[m]
        abar = np.average(a, weights=w)
        out[b] = round(
            1 - (w * (a - p) ** 2).sum() / (w * (a - abar) ** 2).sum(), 2
        )
    return out


def observed_lead(df: pd.DataFrame) -> dict[str, int]:
    """Nombre de bureaux où chaque bloc de parti arrive **réellement** en tête en 2024
    (résultat observé, pas la prédiction) — le contrepoint honnête au favori unique d'un
    sondage, et la vérité contre laquelle se mesure le 81,6 %."""
    act = df[[f"act_{b}" for b in VOTE]].to_numpy().argmax(1)
    return {b: int((act == i).sum()) for i, b in enumerate(VOTE)}


def flat_poll(df: pd.DataFrame, baselines: dict[str, float]) -> dict[str, object]:
    """Ce que ferait un sondage plat : attribuer partout le favori national. Justesse
    **pondérée par les inscrits**, comparable au 81,6 % du modèle."""
    fav = max(VOTE, key=lambda b: baselines[b])
    act = df[[f"act_{b}" for b in VOTE]].to_numpy().argmax(1)
    w = _weights(df)
    m = np.isfinite(w) & (w > 0)
    acc = float(np.average((act == VOTE.index(fav))[m], weights=w[m]))
    return {"bloc": fav, "accuracy": round(acc * 100, 1)}


def circo_rollup(
    df: pd.DataFrame, w: np.ndarray, reach: float
) -> tuple[dict[str, list[float]], dict[str, object]]:
    """Agrège chaque bureau en sa circonscription (part de suffrages pondérée par
    les inscrits) et rend le bloc en tête sur cet agrégat. Honnête : c'est le bloc
    dominant des bureaux de la circonscription, pas une projection de siège à deux
    tours. Le client recompose en direct depuis les parts ; Python tient la vérité
    des compteurs (base + bascules au scénario par défaut)."""
    sub = df[df.circo.notna()]
    g = sub.groupby("circo")
    labels = list(g.groups.keys())
    shares = np.array(
        [
            [
                np.average(grp[f"pred_{b}"], weights=grp.inscrits.astype(float))
                for b in VOTE
            ]
            for _, grp in g
        ]
    )
    anchors = [grp.groupby("libelle_commune").inscrits.sum().idxmax() for _, grp in g]
    base = shares.argmax(1)
    deltas = np.zeros(3)
    deltas[VOTE.index("ED")] = reach
    flipped = (shares + conserved(deltas, w)).argmax(1)
    arrays = {
        k: [round(float(v), 4) for v in shares[:, i]] for i, k in enumerate("gce")
    }
    arrays["id"] = labels
    arrays["nm"] = anchors
    stats = {
        "n": int(len(shares)),
        "covered_bv": int(len(sub)),
        "base": {b: int((base == i).sum()) for i, b in enumerate(VOTE)},
        "flip_default": int((flipped != base).sum()),
    }
    return arrays, stats


def aggregate_communes(df: pd.DataFrame) -> pd.DataFrame:
    cc = (
        pq.read_table(COMMUNE_COORDS)
        .to_pandas()
        .rename(
            columns={
                "code_commune": "code_commune",
                "nom": "nom_cc",
                "latitude": "clat",
                "longitude": "clon",
            }
        )
    )

    def wmean(col: str, g: pd.DataFrame) -> float:
        ww = g.inscrits.to_numpy().astype(float)
        return (
            float(np.average(g[col], weights=ww)) if ww.sum() else float(g[col].mean())
        )

    recs = []
    for code, g in df.groupby("code_commune"):
        means = {b: wmean(f"pred_{b}", g) for b in (*VOTE, "AB")}
        lead = max(VOTE, key=lambda b: means[b])
        recs.append(
            {
                "code_commune": code,
                "nom": g.libelle_commune.iloc[0],
                "dept": g.code_departement.iloc[0],
                "inscrits": int(g.inscrits.sum()),
                "n_bv": len(g),
                "lead": lead,
                "lat": float(g.lat.mean()),
                "lon": float(g.lon.mean()),
                "cmv": int(g.mob.sum()),
                "cab": int(round((g.inscrits * g.pred_AB / 100).sum())),
                "ccj": int(
                    round(
                        (
                            g.inscrits * (g.pred_AB - g.abst_floor).clip(lower=0) / 100
                        ).sum()
                    )
                ),
                **{f"p{b}": round(means[b], 2) for b in (*VOTE, "AB")},
            }
        )
    com = pd.DataFrame(recs).merge(
        cc[["code_commune", "clat", "clon"]], on="code_commune", how="left"
    )
    com.lat = com.lat.fillna(com.clat)
    com.lon = com.lon.fillna(com.clon)
    return com.drop(columns=["clat", "clon"]).dropna(subset=["lat", "lon"])


def build() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    SERVED.mkdir(parents=True, exist_ok=True)
    df = load_wide().merge(load_context(), on="location", how="left")
    df = attach_circo(df)
    df = attach_centroid(df)
    df = attach_abst_floor(df)
    w = swing_weights(df)
    df = derive(df, w)
    curve = movability_turnout.fit_gamma(election_type=TARGET_TYPE)
    left, mob = report_targets.left_gain(df, curve)
    df["mob"] = np.round(mob).astype(int)
    df.to_parquet(CACHE / "bv_master.parquet", index=False)
    (SERVED / "gamma_curve.json").write_text(
        json.dumps(movability_turnout.curves_by_type(), separators=(",", ":"))
    )

    baselines = {
        b: round(
            float(np.average(df[f"pred_{b}"], weights=df.inscrits.astype(float))), 2
        )
        for b in BLOCKS.values()
    }
    com = aggregate_communes(df).reset_index(drop=True)
    com.to_json(SERVED / "communes.json", orient="records")
    code2idx = {c: i for i, c in enumerate(com.code_commune)}
    circo_arrays, circo_stats = circo_rollup(df, w, reach=3)
    (SERVED / "circo.json").write_text(json.dumps(circo_arrays, separators=(",", ":")))

    pred = df[[f"pred_{b}" for b in VOTE]].to_numpy()
    margins = np.sort(pred, 1)[:, -1] - np.sort(pred, 1)[:, -2]
    battlefield = {
        str(t): {
            "bv": int((margins < t).sum()),
            "pct": round(float((margins < t).mean()) * 100, 1),
            "inscrits": int(df.inscrits[margins < t].sum()),
        }
        for t in (3, 5, 8)
    }
    summary = {
        "n_bv": int(len(df)),
        "lead_accuracy": round(lead_accuracy(df) * 100, 1),
        "total_inscrits": int(df.inscrits.sum()),
        "baselines": baselines,
        "battlefield": battlefield,
        "observed_lead": observed_lead(df),
        "left_gain": left,
        "flat_poll": flat_poll(df, baselines),
        "circo": circo_stats,
        "swing": {b: round(float(w[i]), 6) for i, b in enumerate(VOTE)},
        "r2": r2_by_block(df),
        "unc_median": round(
            _wmedian(df.unc.to_numpy(), df.inscrits.to_numpy().astype(float)), 1
        ),
    }
    (SERVED / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1)
    )
    national = {
        "pg": [round(v, 4) for v in df.pred_G],
        "pc": [round(v, 4) for v in df.pred_CD],
        "pe": [round(v, 4) for v in df.pred_ED],
        "ins": [int(v) for v in df.inscrits],
        "m": [round(v, 4) for v in margins],
        "t": [round(v, 4) for v in df.ed_tip],
        "ci": [int(code2idx.get(c, -1)) for c in df.code_commune],
        "mv": [int(round(v)) for v in mob],
    }
    (SERVED / "national.json").write_text(json.dumps(national, separators=(",", ":")))
    print(
        f"BV {len(df)} | acc {summary['lead_accuracy']}% | "
        f"jouables<5 {battlefield['5']['bv']} | communes {len(com)} | "
        f"circo {circo_stats['n']} (base ED {circo_stats['base']['ED']}, "
        f"+3 ED → {circo_stats['flip_default']} bascules)"
    )


if __name__ == "__main__":
    build()
