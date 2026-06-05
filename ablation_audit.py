"""Audit: which always-included feature GROUPS help test (val 2024) but NOT LOO?

preregistered.py LOO-selects pca_k and lag-count, but four groups are hardcoded
into every config and never put against their own absence: demographics,
dev_lag1, dev_lag2, type one-hot. We ablate each (matched rows) and report the
removal effect on LOO OOF and on val. Red flag = ΔVal ≫ 0 while ΔLOO ≈ 0
(the group's inclusion is justified by the test, not the LOO).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import numpy as np
from pathlib import Path

from src.cross_type_dev import (
    load_cross_type_data,
    add_election_type_onehot,
    BLOCKS_ABS,
    TARGET_COLS,
    VAL_DATE,
    VAL_TYPE,
    estimate_national_abstention_from_gaps,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.preregistered import run_loo_and_val


def main():
    dd = Path("data")
    df, demo, nm, pf = load_cross_type_data(dd)
    tcs = add_election_type_onehot(df)
    p24 = pf[
        np.isclose(pf.date_float, VAL_DATE, atol=0.1) & (pf.election_type == VAL_TYPE)
    ]
    est = {b: float(p24[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    est["Abstention"] = estimate_national_abstention_from_gaps(nm)[0]

    dl1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dl2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    rl1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    rl2 = [f"{b}_lag2" for b in BLOCKS_ABS]

    ct = df.dropna(subset=demo + rl1 + rl2 + dl1 + dl2)
    legi = df[df.election_type == VAL_TYPE].dropna(subset=demo + rl1 + rl2 + dl1 + dl2)
    nd = len(demo)

    # (label, dataset, feat_cols, cfg)
    runs = {
        # Cross-type PCA5 (selected for ED + Abstention)
        "CT full": (ct, demo + dl1 + dl2 + tcs, {"pca_k": 5, "n_demo": nd}),
        "CT -demo": (ct, dl1 + dl2 + tcs, {"n_demo": 0}),
        "CT -dl2": (ct, demo + dl1 + tcs, {"pca_k": 5, "n_demo": nd}),
        "CT -dl1": (ct, demo + dl2 + tcs, {"pca_k": 5, "n_demo": nd}),
        "CT -type": (ct, demo + dl1 + dl2, {"pca_k": 5, "n_demo": nd}),
        # Legi-only PCA5 (selected for Gauche; C+D uses PCA7 but group effect holds)
        "Legi full": (legi, demo + dl1 + dl2, {"pca_k": 5, "n_demo": nd}),
        "Legi -demo": (legi, dl1 + dl2, {"n_demo": 0}),
        "Legi -dl2": (legi, demo + dl1, {"pca_k": 5, "n_demo": nd}),
        "Legi -dl1": (legi, demo + dl2, {"pca_k": 5, "n_demo": nd}),
    }

    res = {}
    for name, (data, feats, cfg) in runs.items():
        print(f"  {name}...", flush=True)
        res[name] = run_loo_and_val(name, data, feats, est, nm, dict(cfg))

    def show(full, abl_names, blocks):
        print(f"\n{'removed group':16s} {'block':14s} {'ΔLOO':>8s} {'ΔVal':>8s}   flag")
        for ab in abl_names:
            grp = ab.split("-", 1)[1]
            for tc in blocks:
                dloo = res[full][tc]["oof_r2"] - res[ab][tc]["oof_r2"]
                dval = res[full][tc]["val_r2"] - res[ab][tc]["val_r2"]
                # group helps val but not LOO → snooping flag
                flag = "⚠ TEST-ONLY" if (dval > 0.003 and dloo <= 0.0005) else ""
                print(f"  {grp:14s} {tc:14s} {dloo:+8.4f} {dval:+8.4f}   {flag}")

    print(
        f"\n{'=' * 64}\nCROSS-TYPE PCA5 (ED, Abstention) — effect of REMOVING each group"
    )
    print("=" * 64)
    show("CT full", ["CT -demo", "CT -dl2", "CT -dl1", "CT -type"], TARGET_COLS)

    print(f"\n{'=' * 64}\nLEGI-ONLY PCA5 (Gauche, C+D) — effect of REMOVING each group")
    print("=" * 64)
    show("Legi full", ["Legi -demo", "Legi -dl2", "Legi -dl1"], TARGET_COLS)


if __name__ == "__main__":
    main()
