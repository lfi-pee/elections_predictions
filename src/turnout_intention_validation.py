"""Validation: does pre-scrutin turnout-intention track real participation?

Audit/record only — NOT a production estimator. Production uses the published
"indice de participation" directly (see turnout_polls.py), exactly as the vote
blocks use raw vote-intention polls. This script documents that the intention
signal, calibrated to turnout in LOO, would have flagged the 2024 mobilisation
(abstention well below the historical 49.6% baseline), justifying that design.

Intention = "% top-of-scale certain to vote", extracted from OPEN CDSP codebooks
/ documentation PDFs (no restricted microdata):
  - 5-pt "tout à fait certain": PEF2002 V1 xq9b; PEF2007 V1 Q93; PEF2007 V3 Q93P3
    (Documentation_*.pdf tris à plat, files 1744 / 4673 / 4675).
  - 0-10 "note=10": ENEF2017 MQ0BIS (présid), OLEG0BIS (légis); ENEF2024 Y6CERT.
Gaps with no accessible pre-scrutin intention: législatives 2002, all of 2012.

Usage: python3 -m src.turnout_intention_validation
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class IntentionObs:
    election_type: str
    date_float: float
    intention_top: float  # % at the top of the certainty scale
    instrument: int  # 0 = 5-pt "tout à fait certain", 1 = 0-10 "note=10"
    source: str


SERIES: list[IntentionObs] = [
    IntentionObs("Presidentielle_T1", 2002.33, 84.4, 0, "PEF2002 V1 xq9b"),
    IntentionObs("Presidentielle_T1", 2007.33, 85.9, 0, "PEF2007 V1 Q93"),
    IntentionObs("Legislatives_T1", 2007.50, 69.5, 0, "PEF2007 V3 Q93P3"),
    IntentionObs("Presidentielle_T1", 2017.33, 74.0, 1, "ENEF2017 MQ0BIS"),
    IntentionObs("Legislatives_T1", 2017.50, 61.7, 1, "ENEF2017 OLEG0BIS"),
    IntentionObs("Legislatives_T1", 2024.50, 65.2, 1, "ENEF2024 Y6CERT"),
]


def real_turnout(
    national_means: pd.DataFrame, election_type: str, date: float
) -> float:
    m = national_means[
        (national_means["election_type"] == election_type)
        & (abs(national_means["date_float"] - date) < 0.3)
    ]
    return 100.0 - float(m["Abstention"].iloc[0])


def loo_calibration(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        preds[i] = LinearRegression().fit(X[tr], y[tr]).predict(X[i : i + 1])[0]
    return preds


def main() -> None:
    nm = pd.read_parquet(
        Path("data") / "baseline_cache" / "cross_type_dev_natmean.parquet"
    )
    df = pd.DataFrame(
        [
            (
                o.election_type,
                o.date_float,
                o.intention_top,
                o.instrument,
                real_turnout(nm, o.election_type, o.date_float),
                o.source,
            )
            for o in SERIES
        ],
        columns=["type", "date", "intent", "instr", "turnout", "source"],
    )
    X = df[["intent", "instr"]].to_numpy()
    y = df["turnout"].to_numpy()
    df["abst_real"] = (100 - y).round(1)
    df["abst_loo"] = (100 - loo_calibration(X, y)).round(1)

    print(
        df[
            ["type", "date", "intent", "instr", "abst_real", "abst_loo", "source"]
        ].to_string(index=False)
    )
    rmse = float(np.sqrt(np.mean((y - (100 - df["abst_loo"].to_numpy())) ** 2)))
    print(f"\nLOO RMSE (turnout) = {rmse:.1f} pp over {len(df)} elections")

    t24 = df[(df.type == "Legislatives_T1") & (abs(df.date - 2024.5) < 0.1)].iloc[0]
    print(
        f"\nLégis 2024: abstention réelle {t24.abst_real} | calibré LOO {t24.abst_loo} "
        f"| estimateur historique 49.6 (last-same-type)"
    )


if __name__ == "__main__":
    main()
