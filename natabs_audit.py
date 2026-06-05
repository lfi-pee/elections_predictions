"""Audit the national-abstention estimator: is the gap-model LOO-best, or was
its form chosen to land near the 2024 test (31%)?

Compares candidate national-abstention estimators by LOO RMSE over training
elections, and shows each one's 2024 prediction. If the gap-model is NOT the
LOO-best but happens to be closest to 2024 → that's a test-driven choice.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

from src.cross_type_dev import load_cross_type_data, VAL_DATE

ACTUAL_2024 = 31.0  # observed national abstention, Legi T1 2024 (test)


def main():
    _, _, nm, _ = load_cross_type_data(Path("data"))
    nm = nm.sort_values("date_float").reset_index(drop=True)
    train = nm[nm["date_float"] < VAL_DATE - 0.1].reset_index(drop=True)
    y = train["Abstention"].to_numpy()
    n = len(y)
    gap_all = nm["date_float"].diff().to_numpy()
    gap_train = gap_all[:n]  # gap of each training election (first is nan)
    last_train_date = float(train["date_float"].max())
    gap_2024 = VAL_DATE - last_train_date
    types = train["election_type"].to_numpy()

    def loo_rmse(pred_fn):
        preds = np.full(n, np.nan)
        for i in range(n):
            m = np.ones(n, bool)
            m[i] = False
            preds[i] = pred_fn(i, m)
        ok = ~np.isnan(preds)
        return float(np.sqrt(np.mean((y[ok] - preds[ok]) ** 2)))

    # Candidate estimators: (name, loo_pred_fn(i,mask), pred_2024_fn)
    def gap_fit(mask, x_new):
        ok = mask & ~np.isnan(gap_train)
        lr = LinearRegression().fit(gap_train[ok].reshape(-1, 1), y[ok])
        return float(lr.predict([[x_new]])[0])

    cands = {
        "gap-model (current)": (
            lambda i, m: (
                gap_fit(m, gap_train[i]) if not np.isnan(gap_train[i]) else np.nan
            ),
            gap_fit(np.ones(n, bool), gap_2024),
        ),
        "global mean": (lambda i, m: float(y[m].mean()), float(y.mean())),
        "same-type mean": (
            lambda i, m: float(y[m & (types == types[i])].mean()),
            float(y[types == "Legislatives_T1"].mean()),
        ),
        "last election (any)": (
            lambda i, m: float(y[i - 1]) if i > 0 else np.nan,
            float(y[-1]),
        ),
        "last same-type": (
            lambda i, m: _last_same(i, types, y),
            _last_same_global(types, y, "Legislatives_T1"),
        ),
    }

    print(f"{'estimator':24s} {'LOO RMSE':>9s} {'2024 pred':>10s} {'|err 2024|':>11s}")
    print("-" * 58)
    ranked = []
    for name, (fn, p2024) in cands.items():
        r = loo_rmse(fn)
        ranked.append((r, name, p2024))
        print(f"{name:24s} {r:9.2f} {p2024:10.1f} {abs(p2024 - ACTUAL_2024):11.1f}")
    print("-" * 58)
    best = min(ranked)
    print(f"\nLOO-best: {best[1]} (RMSE {best[0]:.2f}, 2024 pred {best[2]:.1f})")
    closest = min(ranked, key=lambda t: abs(t[2] - ACTUAL_2024))
    print(f"Closest-to-2024: {closest[1]} (pred {closest[2]:.1f})")
    print(
        "\nIf LOO-best == gap-model → LOO-justified."
        "\nIf gap-model is only the closest-to-2024 but not LOO-best → test-driven."
    )


def _last_same(i, types, y):
    for j in range(i - 1, -1, -1):
        if types[j] == types[i]:
            return float(y[j])
    return np.nan


def _last_same_global(types, y, t):
    idx = [k for k in range(len(y)) if types[k] == t]
    return float(y[idx[-1]]) if idx else np.nan


if __name__ == "__main__":
    main()
