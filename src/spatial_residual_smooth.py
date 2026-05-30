"""Spatial residual smoother: Ridge + kernel regression on the geographic
residual field (untried variant of the GP-on-residuals idea).

`gp_residual_boost.py` runs an RBF kernel over the *demographic feature
vector* — the already-settled "nonlinear demographic→residual" test.
This module instead smooths Ridge OOF residuals over **lat/lon** with a
Gaussian kernel whose length scale ℓ (km) is chosen by the data, the
principled form of the spatial-diffusion prior.

Two variants isolate the question "does a neighbour add info beyond the
bureau's own history (already in dev lags)?":
  - self+nbr: include the bureau's own past residuals (per-BV residual RE)
  - nbr-only: exclude the query location (pure spatial diffusion)

Protocol mirrors the repo: pick ℓ by LOO over training elections, then a
single 2024 forward pass. Centroid-fallback (missing-geo) BVs are dropped
from source and query to avoid a fake megacluster.

Usage:
    python3 -u -m src.spatial_residual_smooth
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from src.cross_type_dev import (
    BLOCKS_ABS,
    ABBR,
    TARGET_COLS,
    add_election_type_onehot,
    estimate_national_abstention_from_gaps,
    load_cross_type_data,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.gp_residual_boost import (
    ALPHA_GRID,
    BEST_RIDGE,
    PREV_RAW,
    VAL_DATE,
    VAL_TYPE,
    _apply_pca,
    _build_fold_info,
    split_tv,
)

EARTH_KM = 6371.0
CENTROID = (46.2276, 2.2137)  # France fallback for missing geo
LENGTH_SCALES_KM = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
KERNEL_CUTOFF = 3.0  # ignore neighbours beyond 3ℓ (negligible weight)


@dataclass
class SmoothField:
    tree: BallTree
    coords_rad: np.ndarray
    values: np.ndarray
    loc_index: dict[str, int]


def _missing_geo(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.isclose(lat, CENTROID[0]) & np.isclose(lon, CENTROID[1])


def build_field(
    locations: np.ndarray, lat: np.ndarray, lon: np.ndarray, resid: np.ndarray
) -> SmoothField:
    """Per-location mean residual field (geo-valid locations only)."""
    keep = ~_missing_geo(lat, lon) & ~np.isnan(resid)
    frame = pd.DataFrame(
        {"loc": locations[keep], "lat": lat[keep], "lon": lon[keep], "r": resid[keep]}
    )
    agg = frame.groupby("loc", sort=False).agg(
        lat=("lat", "first"), lon=("lon", "first"), r=("r", "mean")
    )
    coords_rad = np.radians(agg[["lat", "lon"]].values)
    return SmoothField(
        tree=BallTree(coords_rad, metric="haversine"),
        coords_rad=coords_rad,
        values=agg["r"].values.astype(np.float64),
        loc_index={loc: i for i, loc in enumerate(agg.index)},
    )


def smooth_predict(
    field: SmoothField,
    q_loc: np.ndarray,
    q_lat: np.ndarray,
    q_lon: np.ndarray,
    ell_km: float,
    exclude_self: bool,
) -> np.ndarray:
    """Gaussian kernel regression of the residual field onto query points."""
    n = len(q_loc)
    out = np.zeros(n, dtype=np.float64)
    valid = ~_missing_geo(q_lat, q_lon)
    if not valid.any():
        return out

    q_rad = np.radians(np.column_stack([q_lat[valid], q_lon[valid]]))
    radius = KERNEL_CUTOFF * ell_km / EARTH_KM
    idx_list, dist_list = field.tree.query_radius(q_rad, r=radius, return_distance=True)

    q_self = [field.loc_index.get(loc, -1) for loc in q_loc[valid]]
    inv_2l2 = 1.0 / (2.0 * ell_km * ell_km)
    preds = np.zeros(valid.sum(), dtype=np.float64)
    for j, (idx, dist) in enumerate(zip(idx_list, dist_list)):
        if exclude_self and q_self[j] >= 0:
            mask = idx != q_self[j]
            idx, dist = idx[mask], dist[mask]
        if len(idx) == 0:
            continue
        d_km = dist * EARTH_KM
        w = np.exp(-(d_km * d_km) * inv_2l2)
        wsum = w.sum()
        if wsum > 0:
            preds[j] = float(np.dot(w, field.values[idx]) / wsum)
    out[valid] = preds
    return out


def _ridge_oof(
    X_tr_raw: np.ndarray,
    dev_y: np.ndarray,
    fold_masks: list[np.ndarray],
    fold_nats: list[dict[str, float]],
    n_demo: int,
    pca_k: int | None,
    alpha: float,
    tc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """OOF Ridge dev predictions + per-fold-nat absolute predictions."""
    oof_dev = np.full(len(dev_y), np.nan)
    oof_abs = np.full(len(dev_y), np.nan)
    for i, held in enumerate(fold_masks):
        not_held = ~held
        pca_fold = (
            PCA(n_components=pca_k).fit(X_tr_raw[not_held, :n_demo]) if pca_k else None
        )
        X_ft = _apply_pca(X_tr_raw[not_held], pca_fold, n_demo)
        X_fh = _apply_pca(X_tr_raw[held], pca_fold, n_demo)
        ridge = Ridge(alpha=alpha, solver="cholesky")
        ridge.fit(X_ft, dev_y[not_held])
        pred = ridge.predict(X_fh)
        oof_dev[held] = pred
        oof_abs[held] = pred + fold_nats[i][tc]
    return oof_dev, oof_abs


def run_block(
    tc: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feat_cols: list[str],
    demo_cols: list[str],
    nat_est: dict[str, float],
    national_means: pd.DataFrame,
    pca_k: int | None,
) -> dict[str, object]:
    n_demo = len(demo_cols)
    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(train[feat_cols].values.astype(np.float64))
    X_v_raw = scaler.transform(val[feat_cols].values.astype(np.float64))

    pca_full = PCA(n_components=pca_k).fit(X_tr_raw[:, :n_demo]) if pca_k else None
    X_tr = _apply_pca(X_tr_raw, pca_full, n_demo)
    X_v = _apply_pca(X_v_raw, pca_full, n_demo)

    dev_y = train[f"dev_{tc}"].values.astype(np.float64)
    y_tr = train[tc].values.astype(np.float64)
    y_v = val[tc].values.astype(np.float64)
    nat = nat_est.get(tc, 0.0)

    fold_masks, fold_nats = _build_fold_info(train, national_means)

    ridge_full = RidgeCV(alphas=ALPHA_GRID).fit(X_tr, dev_y)
    ridge_val_dev = ridge_full.predict(X_v)
    ridge_val_pred = ridge_val_dev + nat
    ridge_val_r2 = r2_score(y_v, ridge_val_pred)

    oof_dev, oof_abs = _ridge_oof(
        X_tr_raw,
        dev_y,
        fold_masks,
        fold_nats,
        n_demo,
        pca_k,
        ridge_full.alpha_,
        tc,
    )
    oof_resid = dev_y - oof_dev
    ridge_oof_r2 = r2_score(y_tr, oof_abs)

    tr_loc = train["location"].values
    tr_lat = train["latitude"].values.astype(np.float64)
    tr_lon = train["longitude"].values.astype(np.float64)
    v_loc = val["location"].values
    v_lat = val["latitude"].values.astype(np.float64)
    v_lon = val["longitude"].values.astype(np.float64)

    # ── LOO selection of ℓ per variant ──
    results: dict[str, dict[str, float]] = {}
    for variant, exclude_self in [("self+nbr", False), ("nbr-only", True)]:
        best = {"ell": None, "oof_r2": -np.inf}
        for ell in LENGTH_SCALES_KM:
            oof_sm = oof_abs.copy()
            for i, held in enumerate(fold_masks):
                src = ~held & ~np.isnan(oof_resid)
                field = build_field(
                    tr_loc[src], tr_lat[src], tr_lon[src], oof_resid[src]
                )
                add = smooth_predict(
                    field,
                    tr_loc[held],
                    tr_lat[held],
                    tr_lon[held],
                    ell,
                    exclude_self,
                )
                oof_sm[held] = oof_dev[held] + add + fold_nats[i][tc]
            r2 = r2_score(y_tr, oof_sm)
            if r2 > best["oof_r2"]:
                best = {"ell": ell, "oof_r2": r2}

        field_full = build_field(tr_loc, tr_lat, tr_lon, oof_resid)
        add_val = smooth_predict(
            field_full, v_loc, v_lat, v_lon, best["ell"], exclude_self
        )
        val_pred = ridge_val_pred + add_val
        results[variant] = {
            "ell": best["ell"],
            "oof_r2": best["oof_r2"],
            "val_r2": r2_score(y_v, val_pred),
            "n_geo_val": int((~_missing_geo(v_lat, v_lon)).sum()),
        }

    return {
        "ridge_oof_r2": ridge_oof_r2,
        "ridge_val_r2": ridge_val_r2,
        "variants": results,
        "n_val": len(y_v),
    }


def _datasets(data_dir: Path) -> tuple:
    # BEST_RIDGE only uses legi/ct datasets; the extended (~1.2M-row) build
    # is intentionally skipped to keep memory bounded.
    df, demo_ind, national_means, poll_feats = load_cross_type_data(data_dir)
    type_cols = add_election_type_onehot(df)

    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    est["Abstention"], _ = estimate_national_abstention_from_gaps(national_means)

    raw1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    d1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    d2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]

    df_v1 = df.dropna(subset=demo_ind)
    ct_v1_2 = df_v1.dropna(subset=raw1 + raw2 + d1 + d2)
    df_legi = df[df["election_type"] == VAL_TYPE].copy().dropna(subset=demo_ind)
    legi_v1_2 = df_legi.dropna(subset=raw1 + raw2 + d1 + d2)

    feat_maps = {
        "ct": (demo_ind, demo_ind + d1 + d2 + type_cols, national_means),
        "legi": (demo_ind, demo_ind + d1 + d2, national_means),
    }
    datasets = {"ct_v1_2": ct_v1_2, "legi_v1_2": legi_v1_2}
    return datasets, feat_maps, est


def main() -> None:
    t0 = time.time()
    datasets, feat_maps, est = _datasets(Path("data"))

    print("\n" + "=" * 72)
    print("Spatial residual smoother (Gaussian kernel on lat/lon, LOO-tuned ℓ)")
    print("=" * 72)

    rows: list[tuple] = []
    for tc in TARGET_COLS:
        ridge_name, data_key, feat_key, cfg = BEST_RIDGE[tc]
        demo_cols, all_cols, nm = feat_maps[feat_key]
        train, val = split_tv(datasets[data_key])
        ok_tr = train[all_cols].notna().all(axis=1)
        ok_v = val[all_cols].notna().all(axis=1)
        train_c, val_c = train[ok_tr].copy(), val[ok_v].copy()

        t1 = time.time()
        res = run_block(
            tc, train_c, val_c, all_cols, demo_cols, est, nm, cfg.get("pca_k")
        )
        sn = res["variants"]["self+nbr"]
        no = res["variants"]["nbr-only"]
        print(f"\n── {ABBR[tc]} ({tc}) — {ridge_name} ──")
        print(
            f"  train={len(train_c):,} val={len(val_c):,} "
            f"geo-val={sn['n_geo_val']:,}  ({time.time() - t1:.0f}s)"
        )
        print(
            f"  Ridge-only         OOF={res['ridge_oof_r2']:.4f}  "
            f"Val={res['ridge_val_r2']:.4f}"
        )
        print(
            f"  +self+nbr (ℓ={sn['ell']:>4}km) OOF={sn['oof_r2']:.4f}  "
            f"Val={sn['val_r2']:.4f}  Δval={sn['val_r2'] - res['ridge_val_r2']:+.4f}"
        )
        print(
            f"  +nbr-only (ℓ={no['ell']:>4}km) OOF={no['oof_r2']:.4f}  "
            f"Val={no['val_r2']:.4f}  Δval={no['val_r2'] - res['ridge_val_r2']:+.4f}"
        )
        rows.append((tc, res))

    print("\n" + "=" * 72)
    print("SUMMARY  (Δ vs Ridge-only; positive = spatial helps)")
    print("=" * 72)
    print(f"{'Block':16s} {'Ridge':>8s} {'self+nbr':>10s} {'nbr-only':>10s}")
    print("-" * 48)
    for tc, res in rows:
        sn = res["variants"]["self+nbr"]
        no = res["variants"]["nbr-only"]
        print(
            f"{tc:16s} {res['ridge_val_r2']:8.4f} "
            f"{sn['val_r2']:+8.4f}Δ{sn['val_r2'] - res['ridge_val_r2']:+.3f} "
            f"{no['val_r2']:+8.4f}Δ{no['val_r2'] - res['ridge_val_r2']:+.3f}"
        )
    print(f"\nTotal: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
