"""Précalcule les explications par bureau pour le panneau « interroger ».

Réutilise `train_and_explain` (mêmes modèles pré-enregistrés que
`shap_waterfall.py`) : la matrice SHAP de tout l'ensemble de validation est
produite en une passe par bloc. On garde les k contributions dominantes par
bureau, fusionnées avec prédictions / réel / intervalles de la table maître, puis
on écrit `report_app/data/detail/<dept>.json` (chargé à la demande au clic).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.beat_it import build_extended_data
from src.cross_type_dev import (
    VAL_DATE,
    VAL_TYPE,
    estimate_national_abstention_from_gaps,
    load_cross_type_data,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.shap_waterfall import BEST_MODELS, train_and_explain

MASTER = Path("data/report/bv_master.parquet")
OUT = Path("report_app/data/detail")
WHY_LEFT = Path("data/report/why_left.json")
SHORT = {
    "Gauche": "G",
    "Centre+Droite": "CD",
    "Extreme_Droite": "ED",
    "Abstention": "AB",
}
TOP_K = 6
BLOC_FR = {
    "Gauche": "Gauche",
    "Centre+Droite": "Centre+Droite",
    "Extreme_Droite": "Extrême Droite",
    "Abstention": "Abstention",
}
# Nom de feature brut → libellé court et lisible pour la viz quantitative de
# contributions. Le reste retombe sur un nettoyage générique du nom de colonne.
LABEL = {
    "Taux_Chomage": "Chômage",
    "Pct_Bac_Plus_5": "Diplômés du supérieur",
    "Pct_Bac_Plus_3_4": "Diplômés (bac+3/4)",
    "Pct_Bac_Plus_2": "Diplômés (bac+2)",
    "Pct_Sans_Diplome": "Sans diplôme",
    "Pct_Proprietaires": "Propriétaires",
    "Pct_Locataires": "Locataires",
    "Pct_HLM": "Logements sociaux",
    "Pct_Cadres": "Cadres",
    "Pct_Ouvriers": "Ouvriers",
    "Pct_Employes": "Employés",
    "Pct_Prof_Intermediaires": "Professions interm.",
    "Pct_Agriculteurs": "Agriculteurs",
    "Pct_Retraites": "Retraités",
    "Pct_Etudiants": "Étudiants",
    "Pct_Immigres": "Immigrés",
    "Pct_Emploi_Industrie": "Emploi industriel",
    "Pct_Emploi_Agriculture": "Emploi agricole",
    "Pct_Familles_Monoparentales": "Familles monoparentales",
    "Pct_Familles_Nombreuses": "Familles nombreuses",
    "Pct_Menages_Seuls": "Personnes seules",
    "Pct_Suroccupation": "Suroccupation",
}


def pretty_label(raw: str) -> str:
    """Nom de feature brut → libellé court lisible (distinct par variable)."""
    if "_lag" in raw:
        bloc, n = raw.replace("dev_", "").split("_lag")
        rec = "n-1" if n == "1" else "n-2"
        if bloc == "Abstention":
            return f"Abstention ({rec})"
        return f"Vote {BLOC_FR.get(bloc, bloc)} ({rec})"
    if raw == "latitude":
        return "Position (latitude)"
    if raw == "longitude":
        return "Position (longitude)"
    if raw == "date_float":
        return "Date du scrutin"
    if raw.startswith("type_"):
        return "Type de scrutin"
    if raw in LABEL:
        return LABEL[raw]
    return raw.replace("Pct_", "").replace("Taux_", "").replace("_", " ")


def feat_value_str(raw: str, v: float) -> str:
    """Valeur de la variable telle que mesurée dans ce bureau, pour l'afficher sous le
    libellé. Votes passés = écart au national (en points) ; indicateurs INSEE = niveau
    local (en %) ; géographie / date / type de scrutin : non affichés (peu parlants)."""
    if "_lag" in raw:
        return f"{v:+.0f} pts vs national".replace("-", "−")
    if raw.startswith(("Pct_", "Taux_")):
        return f"{round(v)} %"
    return ""


def humanize_left(label: str) -> str:
    if label.startswith("Vote Gauche"):
        return "son héritage de vote à gauche"
    if label.startswith("Vote "):
        base = label.replace(" (n-1)", "").replace(" (n-2)", "").lower()
        return "le " + base
    if label.startswith("Position"):
        return "sa position géographique"
    if label == "Date du scrutin":
        return "la date du scrutin"
    if label == "Type de scrutin":
        return "le type de scrutin"
    if label.startswith("Abstention"):
        return "son niveau d'abstention"
    return "son taux de " + label.lower()


def explain_left(gdrivers: list[list]) -> str:
    """Phrase de décideur : ce qui règle le niveau de gauche du bureau (donc γ, donc le
    gisement mobilisable), à partir des contributions Gauche dominantes — directionnelle,
    le signe réel des deux plus gros moteurs décidant le sens (les barres portent le détail)."""
    if not gdrivers:
        return "un profil peu différencié"
    top = gdrivers[:2]
    labels = list(dict.fromkeys(humanize_left(e[0]) for e in top))
    joined = " et ".join(labels)
    raises = sum(e[1] for e in top) >= 0
    if raises:
        return "doit son niveau de gauche surtout à " + joined
    return "voit son niveau de gauche tiré vers le bas par " + joined


def national_estimates(poll_feats, national_means) -> dict[str, float]:
    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    est["Abstention"], _ = estimate_national_abstention_from_gaps(national_means)
    return est


def block_shap(block: str, ctx: dict) -> dict[str, list[list]]:
    # Ridge-only SHAP: matches the deployed predictions (conformal.py is Ridge-only),
    # so the per-bureau "why" explains the exact production model — not a boosted variant.
    info = train_and_explain(
        block,
        ctx["df"],
        ctx["df_ext"],
        ctx["demo"],
        ctx["ext"],
        ctx["nm"],
        ctx["ext_nm"],
        ctx["est"],
        ctx["est"],
        with_boost=False,
    )
    sv = info["shap_values"]
    names = info["feature_names"]
    vr = info["val_raw"]
    locs = info["val"]["location"].to_numpy()
    top = np.argsort(-np.abs(sv), axis=1)[:, :TOP_K]
    out = {}
    for i, loc in enumerate(locs):
        out[loc] = [
            [
                pretty_label(names[j]),
                round(float(sv[i, j]), 2),
                feat_value_str(names[j], float(vr[i, j])),
            ]
            for j in top[i]
        ]
    return out


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    df, demo, national_means, poll_feats = load_cross_type_data(data_dir)
    df_ext, ext, ext_nm, _ = build_extended_data(data_dir)
    ctx = {
        "df": df,
        "df_ext": df_ext,
        "demo": demo,
        "ext": ext,
        "nm": national_means,
        "ext_nm": ext_nm,
        "est": national_estimates(poll_feats, national_means),
    }

    shap_by_block = {SHORT[b]: block_shap(b, ctx) for b in BEST_MODELS}

    m = pd.read_parquet(MASTER)
    by_dept: dict[str, dict] = defaultdict(dict)
    why_left: dict[str, str] = {}
    for row in m.itertuples():
        gdrivers = shap_by_block["G"].get(row.location, [])
        why_left[row.location] = explain_left(gdrivers)
        rec = {
            "n": row.libelle_commune,
            "i": int(row.inscrits),
            "lead": row.lead,
            "ru": row.runner_up,
            "m": round(float(row.margin), 1),
            "u": round(float(row.unc), 0),
            "tip": round(float(row.ed_tip), 1),
            "mob": int(round(float(row.mob))),
            "conj": int(
                round(
                    float(row.inscrits) * max(0.0, row.pred_AB - row.abst_floor) / 100
                )
            ),
            "drivers": shap_by_block[row.lead].get(row.location, []),
            "gdrivers": gdrivers,
            "wleft": why_left[row.location],
            # Lower-confidence: lag features fell back to the commune aggregate
            # (own-BV history missing or from a reused precinct).
            "fb": int(bool(getattr(row, "lag_fallback", False))),
            "blocks": {},
        }
        for b in SHORT.values():
            rec["blocks"][b] = {
                "pred": round(float(getattr(row, f"pred_{b}")), 1),
                "act": round(float(getattr(row, f"act_{b}")), 1),
                "lo": round(float(getattr(row, f"lower90_{b}")), 1),
                "hi": round(float(getattr(row, f"upper90_{b}")), 1),
            }
        by_dept[row.code_departement][row.location] = rec

    for dept, recs in by_dept.items():
        (OUT / f"{dept}.json").write_text(
            json.dumps(recs, ensure_ascii=False, separators=(",", ":"))
        )
    WHY_LEFT.write_text(json.dumps(why_left, ensure_ascii=False, separators=(",", ":")))
    size = sum(p.stat().st_size for p in OUT.glob("*.json")) / 1e6
    print(f"detail: {len(m)} bureaux, {len(by_dept)} départements, {size:.1f} Mo")


if __name__ == "__main__":
    build()
