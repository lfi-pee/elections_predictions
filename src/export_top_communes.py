"""Export CSV des communes par gisement de mobilisation — cohérent avec le site.

Mobilisables(b) = abstentionnistes **conjoncturels** × γ(niveau de gauche), exactement la
grandeur du calque carte et du panneau « Où déployer ». Métropole seule (outre-mer/étranger
exclus : non démarchables). Lit la table maître produite par `report_data.build`.

    uv run python -m src.export_top_communes        # top 100 → top100_villes_mobilisables.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import report_targets
from src.report_data import CACHE

OUT = Path("top100_villes_mobilisables.csv")


def build(top: int = 100) -> pd.DataFrame:
    df = pd.read_parquet(CACHE / "bv_master.parquet")
    ml = report_targets.mainland(df)
    ins = df.inscrits.to_numpy().astype(float)
    df = df.assign(
        conj=ins * report_targets.conjunctural_pct(df) / 100.0,
        abst=ins * df.pred_AB.to_numpy() / 100.0,
    )[ml]
    g = (
        df.groupby(["libelle_commune", "code_departement"])
        .agg(
            mobilisables=("mob", "sum"),
            conjoncturels=("conj", "sum"),
            abstentionnistes=("abst", "sum"),
            inscrits=("inscrits", "sum"),
            bureaux=("location", "size"),
            **{f"pred_{b}": (f"pred_{b}", "mean") for b in ("G", "CD", "ED", "AB")},
        )
        .sort_values("mobilisables", ascending=False)
        .head(top)
        .reset_index()
        .rename(columns={"libelle_commune": "commune", "code_departement": "dept"})
    )
    g.insert(0, "rang", range(1, len(g) + 1))
    g["gamma_pct"] = (100 * g.mobilisables / g.conjoncturels).round(1)
    g["mobilisables_pct_inscrits"] = (100 * g.mobilisables / g.inscrits).round(1)
    for c in ("mobilisables", "conjoncturels", "abstentionnistes", "inscrits"):
        g[c] = g[c].round().astype(int)
    return g.round({f"pred_{b}": 1 for b in ("G", "CD", "ED", "AB")})


def main() -> None:
    g = build()
    g.to_csv(OUT, index=False)
    top = g.iloc[0]
    print(
        f"{len(g)} communes → {OUT} | tête : {top.commune} {top.mobilisables} mob (γ {top.gamma_pct} %)"
    )


if __name__ == "__main__":
    main()
