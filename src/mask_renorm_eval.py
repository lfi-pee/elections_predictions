"""Candidate-slate mask + renormalization: does it beat the baseline under LOO?

Fix #1 from the diagnosis: the deviation model emits a share for every block
regardless of whether that block fields a candidate in the circonscription. In
2024 (front républicain désistements, NFP unity) many seats had a missing block,
so the model's phantom share for the absent block is pure error and the present
blocks are correspondingly under-predicted.

Fix: derive per-(bureau, election) block presence from the CANDIDATE SLATE (known
ex ante — no vote-outcome leakage), then at prediction time
  - set the predicted share of an absent block to 0, and
  - (renorm variant) redistribute its positive predicted mass onto the present
    blocks in proportion, preserving the total expressed-vote share.

Selection rule (matches src/preregistered.py): pooled leave-one-election-out OOF
R², deviation model + each fold's ACTUAL national mean. A variant is adopted for
a block only if it beats the baseline OOF R². The 2024 val pass is reported too
but never drives the choice (pre-registered rule).

Usage:
    python3 -u -m src.mask_renorm_eval
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.cross_type_dev import (
    load_cross_type_data,
    VAL_DATE,
    VAL_TYPE,
    TARGET_COLS,
    BLOCKS_ABS,
)
from src.cross_type_ridge import (
    TARGET_BLOCKS,
    _vectorized_block_mapping,
    CANDIDATE_BLOCK_OVERRIDES,
)

VOTE = ["Gauche", "Centre+Droite", "Extreme_Droite"]
# From conformal.BEST_RIDGE (the served model): legi_v1_2 / legi feats, PCA-k per block.
PCA_K = {"Gauche": 5, "Centre+Droite": 7, "Extreme_Droite": 5}
ALPHA_GRID = np.logspace(-2, 6, 20)


def build_presence(data_dir):
    """Per (location, election_type, date_float) boolean presence for each VOTE
    block, from the candidate slate (Result rows). Same block mapping + 2024 T1
    overrides as _build_block_scores, so it is consistent with the targets."""
    el = pd.read_parquet(data_dir / "baseline_cache" / "elections.parquet")
    res = el[el["metric_type"] == "Result"][
        ["location", "election_type", "date_float", "party", "candidate"]
    ].copy()
    res["block"] = _vectorized_block_mapping(res["party"], res["candidate"])
    m = (res["election_type"] == "Legislatives_T1") & (
        res["date_float"].round(1) == 2024.5
    )
    if m.any():
        keyed = (
            res.loc[m, "location"].str[:2]
            + "|"
            + res.loc[m, "candidate"].str.strip().str.lower()
        )
        lookup = {f"{d}|{n}": b for (d, n), b in CANDIDATE_BLOCK_OVERRIDES.items()}
        res.loc[m, "block"] = keyed.map(lookup).fillna(res.loc[m, "block"])

    res = res[res["block"].isin(VOTE)]
    cnt = (
        res.groupby(["location", "election_type", "date_float", "block"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for b in VOTE:
        if b not in cnt.columns:
            cnt[b] = 0
    cnt["date_float"] = cnt["date_float"].round(5)
    return cnt


def align_presence(frame, presence):
    """Return an (n, 3) boolean array of block presence aligned to `frame` rows."""
    key = frame[["location", "election_type", "date_float"]].copy()
    key["date_float"] = key["date_float"].round(5)
    merged = key.merge(
        presence, on=["location", "election_type", "date_float"], how="left"
    )
    # If a (loc, election) is missing from the slate table, assume present (no mask).
    present = np.column_stack(
        [(merged[b].fillna(1).values > 0) for b in VOTE]
    )
    return present


def apply_mask(P, present, lam=0.0, weight="pred", nat=None):
    """Slate mask with partial redistribution.

    P: (n,3) predicted shares for VOTE blocks. present: (n,3) bool.
    Absent blocks are set to 0. A fraction `lam` of the absent blocks' positive
    predicted mass is redistributed onto the present blocks:
        lam=0 → mask-only (present untouched)
        lam=1 → full renormalization (all absent mass transferred)
    `weight` sets how the transferred mass splits across present blocks:
        "pred" → in proportion to each present block's own positive prediction
        "nat"  → in proportion to each present block's national mean (nat: (3,))
    Rationale: a missing block's voters scatter partly to Other/divers candidates
    and to abstention, so only a fraction transfers to the modeled survivors."""
    pos = np.clip(P, 0.0, None)
    absent_mass = np.where(~present, pos, 0.0).sum(axis=1)  # mass to redistribute
    if weight == "nat":
        w = np.where(present, np.asarray(nat)[None, :], 0.0)
    else:
        w = np.where(present, pos, 0.0)
    wsum = w.sum(axis=1)
    frac = np.divide(w, wsum[:, None], out=np.zeros_like(w), where=wsum[:, None] > 1e-9)
    out = np.where(present, pos + lam * absent_mass[:, None] * frac, 0.0)
    # rows with no present weight to receive the transfer: just mask
    bad = wsum <= 1e-9
    if bad.any():
        m = np.where(present, pos, 0.0)
        out[bad] = m[bad]
    return out


def r2_block(y, p, w=None):
    if w is None:
        return r2_score(y, p)
    return r2_score(y, p, sample_weight=w)


def main():
    t0 = time.time()
    data_dir = Path("data")

    df, demo_indicators, national_means, poll_feats = load_cross_type_data(data_dir)

    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]

    dl = (
        df[df["election_type"] == VAL_TYPE]
        .dropna(subset=demo_indicators)
        .dropna(subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)
        .copy()
    )
    # conformal "legi" feature map: demographics + dev lags (no geo/type one-hot)
    nd_legi = dev_lag1 + dev_lag2
    all_cols = demo_indicators + nd_legi
    n_demo = len(demo_indicators)
    ok = dl[all_cols].notna().all(axis=1)
    dl = dl[ok].reset_index(drop=True)

    val_mask = np.isclose(dl["date_float"], VAL_DATE, atol=1e-3) & (
        dl["election_type"] == VAL_TYPE
    )
    train = dl[~val_mask].reset_index(drop=True)
    val = dl[val_mask].reset_index(drop=True)
    print(f"train={len(train):,}  val={len(val):,}  feat={len(all_cols)}  n_demo={n_demo}")

    # inscrits weights
    ins = pd.read_parquet(data_dir / "baseline_cache" / "inscrits_lookup.parquet")
    ins_last = ins.sort_values("date_float").groupby("location")["inscrits"].last()
    w_val = val["location"].map(ins_last).fillna(0).values + 1.0

    # ── Presence ──
    presence = build_presence(data_dir)
    pres_tr = align_presence(train, presence)
    pres_val = align_presence(val, presence)
    print("\nPresence vs actual==0 (slate should predict the structural zeros):")
    for j, b in enumerate(VOTE):
        absent = ~pres_val[:, j]
        act0 = np.isclose(val[b].values, 0.0)
        print(
            f"  {b:15s} val: slate-absent={absent.mean():.4f}  "
            f"actual==0={act0.mean():.4f}  "
            f"absent&act0={np.mean(absent & act0):.4f}  "
            f"absent&act>0={np.mean(absent & ~act0):.4f}"
        )

    # ── Scale + fold structure ──
    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(train[all_cols].values.astype(np.float64))
    X_v_raw = scaler.transform(val[all_cols].values.astype(np.float64))

    train_dates = train["date_float"].values
    train_types = train["election_type"].values
    train_td = (
        train[["election_type", "date_float"]]
        .drop_duplicates()
        .sort_values("date_float")
        .values.tolist()
    )
    fold_masks, fold_nats = [], []
    for etype, ddate in train_td:
        fmask = np.isclose(train_dates, ddate, atol=1e-3) & (train_types == etype)
        fold_masks.append(fmask)
        nm_row = national_means[
            (national_means["election_type"] == etype)
            & np.isclose(national_means["date_float"], ddate, atol=1e-3)
        ]
        fold_nats.append({tc: float(nm_row[tc].iloc[0]) for tc in TARGET_COLS})

    # oracle val national means (isolates the deviation+mask effect from poll error)
    val_nat = {b: float(val[b].mean()) for b in VOTE}

    def apply_pca(Xtr, Xte, k):
        pca = PCA(n_components=k).fit(Xtr[:, :n_demo])
        return (
            np.hstack([pca.transform(Xtr[:, :n_demo]), Xtr[:, n_demo:]]),
            np.hstack([pca.transform(Xte[:, :n_demo]), Xte[:, n_demo:]]),
        )

    # ── Fit each block: OOF abs preds (oracle nat per fold) + val abs preds ──
    def fit_all(extra_tr=None, extra_val=None):
        """Fit each VOTE block; optional `extra_*` columns appended AFTER the
        demographics so they bypass PCA (like the dev lags). Returns (P_oof,P_val)."""
        Xtr = X_tr_raw if extra_tr is None else np.hstack([X_tr_raw, extra_tr])
        Xv = X_v_raw if extra_val is None else np.hstack([X_v_raw, extra_val])
        oof = {b: np.full(len(train), np.nan) for b in VOTE}
        vab = {}
        for b in VOTE:
            k = PCA_K[b]
            dev_y = train[f"dev_{b}"].values.astype(np.float64)
            Xtr_t, Xv_t = apply_pca(Xtr, Xv, k)
            ridge_full = RidgeCV(alphas=ALPHA_GRID).fit(Xtr_t, dev_y)
            vab[b] = ridge_full.predict(Xv_t) + val_nat[b]
            for i, held in enumerate(fold_masks):
                nh = ~held
                Xft, Xfh = apply_pca(Xtr[nh], Xtr[held], k)
                rid = Ridge(alpha=ridge_full.alpha_, solver="cholesky").fit(
                    Xft, dev_y[nh]
                )
                oof[b][held] = rid.predict(Xfh) + fold_nats[i][b]
        return (
            np.column_stack([oof[b] for b in VOTE]),
            np.column_stack([vab[b] for b in VOTE]),
        )

    P_oof, P_val = fit_all()
    Y_oof = np.column_stack([train[b].values for b in VOTE])
    Y_val = np.column_stack([val[b].values for b in VOTE])

    nat_vec = np.array([val_nat[b] for b in VOTE])
    lam_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]

    def sweep(title, P, Y, pres, w=None):
        print(f"\n{title}")
        print(
            f"  {'block':15s} {'baseline':>9s} | "
            + "  ".join(f"λ={l:<4g}" for l in lam_grid)
            + "  | best(pred)   best(nat)"
        )
        out = {}
        for j, b in enumerate(VOTE):
            base = r2_block(Y[:, j], P[:, j], w)
            r2_pred = [
                r2_block(Y[:, j], apply_mask(P, pres, lam=l, weight="pred")[:, j], w)
                for l in lam_grid
            ]
            r2_nat = [
                r2_block(
                    Y[:, j],
                    apply_mask(P, pres, lam=l, weight="nat", nat=nat_vec)[:, j],
                    w,
                )
                for l in lam_grid
            ]
            bp = int(np.argmax(r2_pred))
            bn = int(np.argmax(r2_nat))
            out[b] = dict(base=base, r2_pred=r2_pred, r2_nat=r2_nat, bp=bp, bn=bn)
            print(
                f"  {b:15s} {base:9.4f} | "
                + "  ".join(f"{v:6.4f}" for v in r2_pred)
                + f"  | λ={lam_grid[bp]:g}:{r2_pred[bp]:.4f}  "
                f"λ={lam_grid[bn]:g}:{r2_nat[bn]:.4f}"
            )
        return out

    print("\n" + "=" * 92)
    print("SELECTION CRITERION — pooled LOO OOF R² (oracle national per fold); weight=pred")
    print("=" * 92)
    oof = sweep("[unweighted OOF]", P_oof, Y_oof, pres_tr)

    print("\n" + "=" * 92)
    print("2024 VAL R² (oracle national; reported only, does NOT drive selection)")
    print("=" * 92)
    sweep("[unweighted val]", P_val, Y_val, pres_val)
    sweep("[inscrits-weighted val]", P_val, Y_val, pres_val, w=w_val)

    print("\n" + "=" * 92)
    print("DECISION (by unweighted OOF R², weight=pred redistribution):")
    for b in VOTE:
        d = oof[b]
        lam_best = lam_grid[d["bp"]]
        gain = d["r2_pred"][d["bp"]] - d["base"]
        gain_nat = d["r2_nat"][d["bn"]] - d["base"]
        print(
            f"  {b:15s} baseline={d['base']:.4f}  "
            f"best λ={lam_best:g} (pred) → {d['r2_pred'][d['bp']]:.4f} (Δ={gain:+.4f});  "
            f"nat-weight best λ={lam_grid[d['bn']]:g} → {d['r2_nat'][d['bn']]:.4f} (Δ={gain_nat:+.4f})"
        )

    # ── Why does redistribution fail? Is there any present-block uplift left to
    #    capture, or has the demographic deviation model already absorbed it? ──
    print("\n" + "=" * 92)
    print("WHY λ=0: present-block bias on absent-sibling rows (OOF, oracle national)")
    print("=" * 92)
    print(
        "  If the baseline is already unbiased where a sibling is absent, there is no"
        "\n  uplift left to redistribute — the demographic model captured it. Positive"
        "\n  mean residual would mean under-prediction (room for transfer)."
    )
    any_absent = (~pres_tr).any(axis=1)
    for j, b in enumerate(VOTE):
        rows_present = pres_tr[:, j]
        sib = rows_present & any_absent  # b present, some sibling absent
        full = rows_present & ~any_absent
        res = Y_oof[:, j] - P_oof[:, j]
        # mass the model assigns to the absent sibling(s) on those rows
        absmass = np.where(~pres_tr, np.clip(P_oof, 0, None), 0.0).sum(axis=1)
        print(
            f"  {b:15s} n(sib-absent)={sib.sum():6d}  "
            f"mean resid(sib-absent)={res[sib].mean():+.2f}pp  "
            f"mean resid(full slate)={res[full].mean():+.2f}pp  "
            f"mean absent-mass={absmass[sib].mean():.2f}pp"
        )
    print(
        "\n  → near-zero residual on sib-absent rows ⇒ no systematic uplift for the"
        "\n    survivors ⇒ transferring the absent mass only adds error. The votes a"
        "\n    missing block would have drawn do NOT flow to the modeled survivors"
        "\n    (they go to Other/divers and abstention), and what little demographic"
        "\n    tilt exists is already in the deviation prediction. Mask, do not renorm."
    )

    # ── Better normalization: let the model LEARN the signed slate effect ──
    #    Append per-block presence indicators as features (bypassing PCA) so each
    #    block's Ridge can learn its own signed adjustment when a sibling is absent;
    #    then still mask the absent block to 0. This is the direction-aware version
    #    that a mass-conserving renorm cannot express.
    print("\n" + "=" * 92)
    print("BETTER NORMALIZATION — learned slate features (presence indicators) + mask")
    print("=" * 92)
    P_oof_s, P_val_s = fit_all(pres_tr.astype(np.float64), pres_val.astype(np.float64))
    print(f"  {'block':15s} {'baseline':>9s} {'mask(λ=0)':>10s} {'slate-feat+mask':>16s}")
    for j, b in enumerate(VOTE):
        base = r2_block(Y_oof[:, j], P_oof[:, j])
        mask0 = r2_block(Y_oof[:, j], apply_mask(P_oof, pres_tr, lam=0.0)[:, j])
        slate = r2_block(Y_oof[:, j], apply_mask(P_oof_s, pres_tr, lam=0.0)[:, j])
        winner = "slate-feat" if slate > mask0 + 1e-4 else "mask-only"
        print(
            f"  {b:15s} {base:9.4f} {mask0:10.4f} {slate:16.4f}   → {winner} "
            f"(Δ vs mask={slate - mask0:+.4f})"
        )

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
