"""Prépare le socle de données du site (table maître BV, agrégat commune, stats).

Sorties (cache `data/report/`, servi dans `report_app/data/`) :
- `bv_master.parquet` : une ligne par bureau (prédictions, intervalles, marge,
  bloc en tête, statut disputé, point de bascule ED, inscrits, centroïde).
- `communes.json` : agrégat commune pondéré par inscrits (couche nationale + index
  de recherche).
- `summary.json` : chiffres d'accroche vérifiés + courbe de bascule.

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
    return pd.DataFrame(frames).reset_index()


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


def lead_accuracy(df: pd.DataFrame) -> float:
    pred = df[[f"pred_{b}" for b in VOTE]].to_numpy().argmax(1)
    act = df[[f"act_{b}" for b in VOTE]].to_numpy().argmax(1)
    return float((pred == act).mean())


def _targets(df: pd.DataFrame, reach: float) -> dict[str, int]:
    sel = (df.ed_tip > 0) & (df.ed_tip <= reach)
    fragile = sel & (df.ed_tip >= reach * 0.8)
    return {
        "bv": int(sel.sum()),
        "inscrits": int(df.inscrits[sel].sum()),
        "fragile": int(fragile.sum()),
    }


def flat_poll(df: pd.DataFrame, baselines: dict[str, float]) -> dict[str, object]:
    fav = max(VOTE, key=lambda b: baselines[b])
    act = df[[f"act_{b}" for b in VOTE]].to_numpy().argmax(1)
    acc = float((act == VOTE.index(fav)).mean())
    return {"bloc": fav, "accuracy": round(acc * 100, 1)}


def target_communes(
    df: pd.DataFrame, reach: float, top: int = 10
) -> list[dict[str, object]]:
    sel = (df.ed_tip > 0) & (df.ed_tip <= reach)
    g = (
        df[sel]
        .groupby(["libelle_commune", "code_departement"])
        .agg(bv=("inscrits", "size"), el=("inscrits", "sum"))
        .sort_values("el", ascending=False)
        .head(top)
        .reset_index()
    )
    return [
        {
            "nom": r.libelle_commune,
            "dept": r.code_departement,
            "bv": int(r.bv),
            "el": int(r.el),
        }
        for r in g.itertuples()
    ]


def accuracy_by_margin(df: pd.DataFrame) -> list[dict[str, float]]:
    pred = df[[f"pred_{b}" for b in VOTE]].to_numpy()
    act = df[[f"act_{b}" for b in VOTE]].to_numpy()
    margins = np.sort(pred, 1)[:, -1] - np.sort(pred, 1)[:, -2]
    correct = pred.argmax(1) == act.argmax(1)
    edges = [0, 1, 2, 3, 5, 8, 12, 100]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (margins >= lo) & (margins < hi)
        n = int(sel.sum())
        if not n:
            continue
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "acc": round(float(correct[sel].mean()) * 100, 1),
                "n": n,
            }
        )
    return out


def flip_curve(df: pd.DataFrame, w: np.ndarray) -> list[dict[str, float]]:
    pred = df[[f"pred_{b}" for b in VOTE]].to_numpy()
    base = pred.argmax(1)
    ed = VOTE.index("ED")
    out = []
    for d in np.round(np.arange(-4, 6.01, 0.5), 2):
        deltas = np.zeros(3)
        deltas[ed] = d
        flips = int(((pred + conserved(deltas, w)).argmax(1) != base).sum())
        out.append(
            {"shift": float(d), "flips": flips, "pct": round(flips / len(df) * 100, 2)}
        )
    return out


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
    w = swing_weights(df)
    df = derive(df, w)
    left, mob = report_targets.left_gain(df, movability_turnout.fit_gamma())
    df["mob"] = np.round(mob).astype(int)
    df.to_parquet(CACHE / "bv_master.parquet", index=False)

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
        "flip_curve": flip_curve(df, w),
        "accuracy_by_margin": accuracy_by_margin(df),
        "target_reach": 4,
        "targets": _targets(df, 4),
        "target_communes": target_communes(df, 4),
        "left_gain": left,
        "flat_poll": flat_poll(df, baselines),
        "circo": circo_stats,
        "swing": {b: round(float(w[i]), 6) for i, b in enumerate(VOTE)},
        "r2": {"G": 0.74, "CD": 0.61, "ED": 0.80, "AB": 0.74},
        "unc_median": round(float(df.unc.median()), 1),
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
