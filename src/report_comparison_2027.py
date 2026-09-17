"""Export historical first-round block shares for the neutral 2027 comparison page.

Run: python -m src.report_comparison_2027
2017/2022 use each election's bureau-to-constituency identifiers, not a commune proxy.
2024 uses the cached official constituency results. All shares retain all expressed votes
in their denominator; unclassified candidates remain in the residual category.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from src import attribution_2027 as attribution

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report_app/2027/data/comparison_history.json"
BLOCKS = ("G", "CD", "ED", "AU")


def historical(year: int, sets: dict) -> tuple[dict, dict, dict]:
    eid = f"{year}_legi_t1"
    base = ROOT / "data/elections/agregees"
    general = pd.read_parquet(base / "general_results.parquet", filters=[("id_election", "=", eid)],
                              columns=["id_brut_miom", "code_departement", "code_circonscription", "exprimes", "inscrits", "abstentions"])
    if general.code_circonscription.isna().any() or general.id_brut_miom.duplicated().any():
        raise ValueError(f"{eid}: incomplete or ambiguous constituency mapping")
    general["circo"] = [attribution.circo_id(d, c) for d, c in
                          zip(general.code_departement, general.code_circonscription)]
    candidates = pd.read_parquet(base / "candidats_results.parquet", filters=[("id_election", "=", eid)],
                                 columns=["id_brut_miom", "nuance", "voix"])
    candidates = candidates.merge(general[["id_brut_miom", "circo"]], on="id_brut_miom",
                                  how="left", validate="many_to_one")
    if candidates.circo.isna().any():
        raise ValueError(f"{eid}: candidate results without a constituency")
    labels = {nuance: block for block, nuances in sets.items() for nuance in nuances}
    candidates["block"] = candidates.nuance.map(labels).fillna("AU")
    votes = candidates.groupby(["circo", "block"]).voix.sum().unstack(fill_value=0).reindex(columns=BLOCKS, fill_value=0)
    expressed = general.groupby("circo").exprimes.sum()
    totals = votes.sum(axis=1)
    if not totals.equals(expressed.reindex(totals.index).astype(totals.dtype)):
        raise ValueError(f"{eid}: candidate votes disagree with expressed totals")
    shares = votes.div(expressed, axis=0) * 100
    rows = {cid: {b: round(float(row[b]), 6) for b in BLOCKS}
            for cid, row in shares.iterrows() if expressed[cid] > 0}
    national = {b: float(100 * votes[b].sum() / expressed.sum()) for b in BLOCKS}
    registered, abstentions = int(general.inscrits.sum()), int(general.abstentions.sum())
    assert 0 <= abstentions <= registered and registered > 0
    participation = dict(registered=registered, abstentions=abstentions,
                         abstention_pct=100 * abstentions / registered)
    return rows, national, participation


def build() -> None:
    sets = attribution.block_sets()
    elections = []
    for year in (2017, 2022):
        rows, national, participation = historical(year, sets)
        elections.append(dict(key=str(year), label=f"Législatives {year} · 1er tour", rows=rows,
                              national=national, participation=participation, source="Résultats du ministère de l’Intérieur, fichiers locaux candidats_results.parquet et general_results.parquet",
                              mapping="Nuances regroupées selon les ensembles du modèle ; autres nuances dans le résidu. Circonscription du bureau au scrutin concerné."))
    # Do not implicitly download or change the existing attribution/coverage artifacts.
    if not attribution.RESULTS.exists():
        raise FileNotFoundError(f"Official constituency results required: {attribution.RESULTS}")
    rows, national_votes = {}, dict.fromkeys(BLOCKS, 0.0)
    table = attribution.table_by_key()
    for cid, candidates in attribution.load_results().items():
        attribution.classify(candidates, sets, table, cid)
        shares, votes = dict.fromkeys(BLOCKS, 0.0), dict.fromkeys(BLOCKS, 0.0)
        for candidate in candidates:
            b = candidate["apres"] or "AU"
            shares[b] += candidate["pct"]
            votes[b] += candidate["voix"]
        if sum(votes.values()) > 0:
            rows[cid] = {b: round(shares[b], 6) for b in BLOCKS}
        for b in BLOCKS:
            national_votes[b] += votes[b]
    with attribution.RESULTS.open(encoding="utf-8") as f:
        official = list(csv.DictReader(f, delimiter=";"))
    total = sum(float(r["Exprimés"].replace(",", ".") or 0) for r in official)
    registered = sum(int(r["Inscrits"]) for r in official)
    abstentions = sum(int(r["Abstentions"]) for r in official)
    assert 0 <= abstentions <= registered and registered > 0
    elections.append(dict(key="2024", label="Législatives 2024 · 1er tour", rows=rows,
                          national={b: 100 * national_votes[b] / total for b in BLOCKS},
                          participation=dict(registered=registered, abstentions=abstentions,
                                             abstention_pct=100 * abstentions / registered),
                          source=attribution.RESULTS_URL,
                          mapping="Nuances du modèle, complétées par la table documentée d’attribution régionaliste 2024 ; autres candidats dans le résidu."))
    ids = json.loads((OUTPUT.parent / "circo.json").read_text())["id"]
    for election in elections:
        assert set(election["rows"]) <= set(ids), "Unknown constituency identifiers"
        for shares in election["rows"].values():
            assert all(0 <= v <= 100 for v in shares.values())
            # Official 2024 candidate totals differ by 1–2 votes in four constituencies.
            # Preserve the official expressed-vote denominator rather than renormalizing.
            assert abs(sum(shares.values()) - 100) < 0.01
        print(election["label"], len(election["rows"]), "circonscriptions ; abstention", round(election["participation"]["abstention_pct"], 3))
    payload = dict(unit="Pourcentage des suffrages exprimés, tous candidats au dénominateur",
                   block_nuances={b: sorted(nuances) for b, nuances in sets.items()}, elections=elections)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    build()
