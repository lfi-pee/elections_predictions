"""Cross-Election-Type Ridge: train on ALL French election types.

Core idea (from algorithm.md):
  Train a single Ridge across legislatives, presidentielle, europeennes,
  regionales, departementales, cantonales, municipales — all mapped to
  3 political blocks + abstention.  This gives ~31 T1 election dates with
  different national contexts instead of 5, making national-level features
  (polls, national lags) learnable.

Feature vector (~85 dims):
  Demographics         ~47 dims  (last available census, median imputed)
  Geo                    2 dims  (latitude, longitude)
  Time                   1 dim   (date_float)
  National polls         3 dims  (block avg over 1-year window)
  Has-national-polls     1 dim   (indicator)
  Local BV lag 1         4 dims  (3 blocks + abstention, cross-type)
  Local BV lag 2         4 dims  (cross-type)
  National agg lag 1     4 dims  (same-type national mean)
  National agg lag 2     4 dims  (same-type)
  Election type one-hot  7 dims

Key design choices:
  - Local lags are CROSS-TYPE: a BV's lag1 is its most recent prior
    election of ANY type.
  - National lags are SAME-TYPE: national aggregate uses same-type
    previous elections.
  - Polls are ANY-TYPE: national T1 polls in 1-year window.
  - Election type one-hot lets Ridge learn type-specific intercepts.

Usage:
    python -m src.cross_type_ridge
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.load_elections import load_election_tokens
from src.load_demographics import load_demographic_tokens
from src.load_polls import load_poll_tokens

# ── Block mapping (extended for ALL election types) ──────────────────
# Standard nuance codes (legislatives, cantonales)
LEFT = {
    "SOC",
    "COM",
    "VEC",
    "ECO",
    "EXG",
    "DVG",
    "FI",
    "NUP",
    "FG",
    "RDG",
    "DXG",
    "UG",
    "LO",
    "LCR",
    "GEN",
    "PRG",
    # L-prefixed (regionales, europeennes, municipales)
    "LDVG",
    "LFI",
    "LCOM",
    "LECO",
    "LEXG",
    "LRDG",
    "LUG",
    "LVEC",
    "LUC",
    "LFG",
    "LPS",
    "LPC",
    "LVE",
    "LGA",
    "LXG",
    "LSOC",
    # Old europeennes codes
    "GAU",
}
CENTER_RIGHT = {
    "UMP",
    "LR",
    "DVD",
    "REM",
    "ENS",
    "UDI",
    "MDM",
    "UDF",
    "CEN",
    "DVC",
    "NCE",
    "UDFD",
    "RPR",
    "RPF",
    "HOR",
    # L-prefixed
    "LLR",
    "LREM",
    "LMDM",
    "LHOR",
    "LUDI",
    "LDVD",
    "LNC",
    "LUD",
    "LMAJ",
    "LDVC",
    "LUMP",
    "LUDF",
    "LCMD",
    "LCOP",
    "LDR",
    "LENS",
    # Old codes
    "DTE",
}
EXTREME_RIGHT = {
    "FN",
    "RN",
    "REC",
    "EXD",
    "MNR",
    "UXD",
    "DLF",
    "MPF",
    "LRN",
    "LREC",
    "LEXD",
    "LUXD",
    # L-prefixed
    "LFN",
    "LDLF",
    "LXD",
    # Old codes
    "FRN",
    "MNA",
}

# Presidentielle candidate abbreviations → block
_PRES_ABBREV_TO_BLOCK: dict[str, str] = {
    # Gauche
    "JOSP": "Gauche",
    "HOLL": "Gauche",
    "ROYA": "Gauche",
    "MELE": "Gauche",
    "POUT": "Gauche",
    "ARTH": "Gauche",
    "BUFF": "Gauche",
    "VOYN": "Gauche",
    "BOVE": "Gauche",
    "SCHI": "Gauche",
    "JOLY": "Gauche",
    "BESA": "Gauche",
    "LAGU": "Gauche",
    "HUE": "Gauche",
    "GLUC": "Gauche",
    "MAME": "Gauche",
    "TAUB": "Gauche",
    "CHEV": "Gauche",
    "HAMO": "Gauche",
    "HIDA": "Gauche",
    "JADO": "Gauche",
    "ROUS": "Gauche",
    # Centre+Droite
    "CHIR": "Centre+Droite",
    "SARK": "Centre+Droite",
    "BAYR": "Centre+Droite",
    "MADE": "Centre+Droite",
    "BOUT": "Centre+Droite",
    "LEPA": "Centre+Droite",
    "FILO": "Centre+Droite",
    "PECE": "Centre+Droite",
    "MACR": "Centre+Droite",
    "LASS": "Centre+Droite",
    # Extreme Droite
    "LEPE": "Extreme_Droite",
    "MEGR": "Extreme_Droite",
    "VILL": "Extreme_Droite",
    "DUPO": "Extreme_Droite",
    "ZEMM": "Extreme_Droite",
}

# Full candidate names → block (for NC-coded elections: pres 2017+, euro 2019)
_FULLNAME_TO_BLOCK: dict[str, str] = {
    # Presidentielle (firstname lastname format)
    "jean luc melenchon": "Gauche",
    "benoit hamon": "Gauche",
    "philippe poutou": "Gauche",
    "nathalie arthaud": "Gauche",
    "yannick jadot": "Gauche",
    "anne hidalgo": "Gauche",
    "fabien roussel": "Gauche",
    "emmanuel macron": "Centre+Droite",
    "francois fillon": "Centre+Droite",
    "valerie pecresse": "Centre+Droite",
    "jean lassalle": "Centre+Droite",
    "marine le pen": "Extreme_Droite",
    "eric zemmour": "Extreme_Droite",
    "nicolas dupont aignan": "Extreme_Droite",
    # Europeennes 2019 (lastname firstname format)
    "glucksmann raphael": "Gauche",
    "aubry manon": "Gauche",
    "jadot yannick": "Gauche",
    "hamon benoit": "Gauche",
    "brossat ian": "Gauche",
    "arthaud nathalie": "Gauche",
    "loiseau nathalie": "Centre+Droite",
    "bellamy francois xavier": "Centre+Droite",
    "lagarde jean christophe": "Centre+Droite",
    "bardella jordan": "Extreme_Droite",
    "dupont aignan nicolas": "Extreme_Droite",
    "philippot florian": "Extreme_Droite",
}

TARGET_BLOCKS = ["Gauche", "Centre+Droite", "Extreme_Droite"]
TARGET_COLS = TARGET_BLOCKS + ["Abstention"]

# Election types to include (T1 only — T2 has runoff dynamics)
T1_TYPES = [
    "Legislatives_T1",
    "Presidentielle_T1",
    "Europeennes_T1",
    "Regionales_T1",
    "Departementales_T1",
    "Cantonales_T1",
    "Municipales_T1",
]
# Canonical names for one-hot (merge Cantonales into Departementales)
TYPE_ONEHOT = [
    "Legislatives",
    "Presidentielle",
    "Europeennes",
    "Regionales",
    "Departementales",
    "Municipales",
]

# Poll-to-block mapping (reused from poll_ridge.py)
import re

_CODE_TO_BLOCK: dict[str, str] = {}
for _c in [
    "EXG",
    "NFP",
    "DVG",
    "ECO",
    "FI",
    "LFI",
    "PCF",
    "PS",
    "SOC",
    "COM",
    "VEC",
    "FG",
    "NPA",
    "LO",
    "GEN",
    "PRG",
    "NUP",
    "NUPES",
    "EÉLV",
    "EELV",
    "RDG",
    "DXG",
    "UG",
    "LCR",
    "REV",
    "LDVG",
    "LECO",
    "LEXG",
    "LRDG",
    "LUG",
    "LVEC",
    "LUC",
    "LFG",
    "GAUCHE",
]:
    _CODE_TO_BLOCK[_c] = "Gauche"
for _c in [
    "ENS",
    "LR",
    "DVD",
    "DVC",
    "RE",
    "REM",
    "LREM",
    "UMP",
    "UDF",
    "UDI",
    "MDM",
    "MODEM",
    "HOR",
    "RPR",
    "RPF",
    "NC",
    "NCE",
    "CEN",
    "UDFD",
    "RES",
    "REN",
    "ENSEMBLE",
    "UDC",
    "SE",
    "LLR",
    "LMDM",
    "LHOR",
    "LUDI",
    "LDVD",
    "LNC",
    "LUD",
    "LMAJ",
    "LDVC",
]:
    _CODE_TO_BLOCK[_c] = "Centre+Droite"
for _c in [
    "RN",
    "FN",
    "REC",
    "EXD",
    "MNR",
    "UXD",
    "DLF",
    "MPF",
    "UPF",
    "LRN",
    "LREC",
    "LEXD",
    "LUXD",
]:
    _CODE_TO_BLOCK[_c] = "Extreme_Droite"

_PARTY_NAME_TO_BLOCK: dict[str, str] = {
    "rassemblement national": "Extreme_Droite",
    "front national": "Extreme_Droite",
    "reconquête": "Extreme_Droite",
    "reconquête !": "Extreme_Droite",
    "debout la france": "Extreme_Droite",
    "mouvement pour la france": "Extreme_Droite",
    "mouvement national républicain": "Extreme_Droite",
    "parti socialiste": "Gauche",
    "france insoumise": "Gauche",
    "la france insoumise": "Gauche",
    "parti communiste français": "Gauche",
    "europe écologie les verts": "Gauche",
    "lutte ouvrière": "Gauche",
    "nouveau parti anticapitaliste": "Gauche",
    "parti radical de gauche": "Gauche",
    "génération.s": "Gauche",
    "les verts": "Gauche",
    "la république en marche": "Centre+Droite",
    "la république en marche !": "Centre+Droite",
    "les républicains": "Centre+Droite",
    "union pour un mouvement populaire": "Centre+Droite",
    "mouvement démocrate": "Centre+Droite",
    "union des démocrates et indépendants": "Centre+Droite",
    "horizons": "Centre+Droite",
    "union pour la démocratie française": "Centre+Droite",
    "rassemblement pour la république": "Centre+Droite",
}
_CANDIDATE_NAME_TO_BLOCK: dict[str, str] = {
    "MÉLENCHON": "Gauche",
    "MELENCHON": "Gauche",
    "HIDALGO": "Gauche",
    "JADOT": "Gauche",
    "POUTOU": "Gauche",
    "ARTHAUD": "Gauche",
    "ROUSSEL": "Gauche",
    "HOLLANDE": "Gauche",
    "ROYAL": "Gauche",
    "JOSPIN": "Gauche",
    "HAMON": "Gauche",
    "TAUBIRA": "Gauche",
    "BESANCENOT": "Gauche",
    "LAGUILLER": "Gauche",
    "CHEVÈNEMENT": "Gauche",
    "BUFFET": "Gauche",
    "VOYNET": "Gauche",
    "MAMÈRE": "Gauche",
    "BOVÉ": "Gauche",
    "SCHIVARDI": "Gauche",
    "MACRON": "Centre+Droite",
    "FILLON": "Centre+Droite",
    "PÉCRESSE": "Centre+Droite",
    "PECRESSE": "Centre+Droite",
    "SARKOZY": "Centre+Droite",
    "CHIRAC": "Centre+Droite",
    "BAYROU": "Centre+Droite",
    "BALLADUR": "Centre+Droite",
    "MADELIN": "Centre+Droite",
    "BOUTIN": "Centre+Droite",
    "LE PEN": "Extreme_Droite",
    "ZEMMOUR": "Extreme_Droite",
    "DUPONT-AIGNAN": "Extreme_Droite",
    "MÉGRET": "Extreme_Droite",
    "MEGRET": "Extreme_Droite",
    "VILLIERS": "Extreme_Droite",
}


def get_block(nuance: str, candidate: str = "") -> str:
    """Map party code to political block, handling all election type formats."""
    if pd.isna(nuance) or nuance == "":
        nuance = ""

    # 1. Direct match on standard/extended sets
    if nuance in LEFT:
        return "Gauche"
    if nuance in CENTER_RIGHT:
        return "Centre+Droite"
    if nuance in EXTREME_RIGHT:
        return "Extreme_Droite"

    # 2. Strip BC- prefix (departementales)
    if nuance.startswith("BC-"):
        inner = nuance[3:]
        if inner in LEFT:
            return "Gauche"
        if inner in CENTER_RIGHT:
            return "Centre+Droite"
        if inner in EXTREME_RIGHT:
            return "Extreme_Droite"
        # BC-UD → Union Droite, BC-UG → Union Gauche, BC-UGE → Union Gauche-Ecolo
        # BC-UCD → Union Centre-Droite
        if inner in ("UD", "UCD"):
            return "Centre+Droite"
        if inner in ("UGE",):
            return "Gauche"

    # 3. Presidentielle candidate abbreviations
    if nuance in _PRES_ABBREV_TO_BLOCK:
        return _PRES_ABBREV_TO_BLOCK[nuance]

    # 4. NC (non classé) — fall back to candidate full name
    if nuance == "NC" or nuance == "":
        cand = str(candidate).strip().lower() if candidate else ""
        if cand and cand in _FULLNAME_TO_BLOCK:
            return _FULLNAME_TO_BLOCK[cand]

    return "Other"


def _clean_poll_header(s: str) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"\[.*?\]", "", s).strip()
    return s


def _poll_token_to_block(party: str, candidate: str) -> str:
    if party and not pd.isna(party) and party != "":
        b = get_block(party)
        if b != "Other":
            return b
        if party in _CODE_TO_BLOCK:
            return _CODE_TO_BLOCK[party]
        b = _PARTY_NAME_TO_BLOCK.get(party.lower().strip())
        if b:
            return b
    if not candidate or pd.isna(candidate) or candidate == "":
        return "Other"
    cand = _clean_poll_header(candidate)
    if not cand:
        return "Other"
    if cand in _CODE_TO_BLOCK:
        return _CODE_TO_BLOCK[cand]
    first = re.split(r"[\s\(\-]", cand)[0].strip()
    if first in _CODE_TO_BLOCK:
        return _CODE_TO_BLOCK[first]
    paren = re.search(r"\(([^)]+)\)", cand)
    if paren:
        inner = paren.group(1).strip()
        inner_first = re.split(r"[\s\-]", inner)[0].strip()
        if inner_first in _CODE_TO_BLOCK:
            return _CODE_TO_BLOCK[inner_first]
    if cand in _CANDIDATE_NAME_TO_BLOCK:
        return _CANDIDATE_NAME_TO_BLOCK[cand]
    parts = cand.split()
    if len(parts) >= 2:
        for name, block in _CANDIDATE_NAME_TO_BLOCK.items():
            if name in cand:
                return block
    last = parts[-1] if parts else ""
    if last in _CANDIDATE_NAME_TO_BLOCK:
        return _CANDIDATE_NAME_TO_BLOCK[last]
    return "Other"


# ── Data loading ──────────────────────────────────────────────────────


def _load_cached(data_dir: Path):
    cache_dir = data_dir / "baseline_cache"
    elections_cache = cache_dir / "elections.parquet"
    demos_cache = cache_dir / "demographics.parquet"

    if elections_cache.exists() and demos_cache.exists():
        print("Loading elections/demographics from cache...", flush=True)
        elections = pd.read_parquet(elections_cache)
        demos = pd.read_parquet(demos_cache)
    else:
        print("Cache miss — full load (slow, one-time)...", flush=True)
        elections = load_election_tokens(data_dir)
        demos = load_demographic_tokens(data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        elections.to_parquet(elections_cache, index=False)
        demos.to_parquet(demos_cache, index=False)

    print(
        f"  Elections: {len(elections):,} rows, Demographics: {len(demos):,} rows",
        flush=True,
    )
    return elections, demos


# ── Feature builders ──────────────────────────────────────────────────


def _vectorized_block_mapping(party: pd.Series, candidate: pd.Series) -> pd.Series:
    """Vectorized block mapping for all election types."""
    # Build a single flat lookup dict: code → block
    code_to_block: dict[str, str] = {}
    for c in LEFT:
        code_to_block[c] = "Gauche"
    for c in CENTER_RIGHT:
        code_to_block[c] = "Centre+Droite"
    for c in EXTREME_RIGHT:
        code_to_block[c] = "Extreme_Droite"
    for c, b in _PRES_ABBREV_TO_BLOCK.items():
        code_to_block[c] = b

    # Step 1: Direct party code lookup
    block = party.map(code_to_block).fillna("Other")

    # Step 2: BC- prefix (departementales) — strip prefix and remap
    bc_mask = party.str.startswith("BC-", na=False) & (block == "Other")
    if bc_mask.any():
        inner = party[bc_mask].str[3:]
        bc_mapped = inner.map(code_to_block).fillna("Other")
        # Handle UD, UCD, UGE
        bc_mapped[inner == "UD"] = "Centre+Droite"
        bc_mapped[inner == "UCD"] = "Centre+Droite"
        bc_mapped[inner == "UGE"] = "Gauche"
        block[bc_mask] = bc_mapped

    # Step 3: NC — candidate full name lookup
    nc_mask = (party == "NC") & (block == "Other")
    if nc_mask.any():
        cand_lower = candidate[nc_mask].str.strip().str.lower()
        name_mapped = cand_lower.map(_FULLNAME_TO_BLOCK)
        block[nc_mask] = name_mapped.fillna("Other")

    return block


# ── Candidate-level block overrides ──────────────────────────────────
# A handful of 2024 Législatives T1 candidates were coded by the Ministry under a
# nuance that misroutes them. Each was verified individually (party + whether the
# RN stood aside for them), with sources — the "no-RN circo" heuristic over-flags
# (e.g. Durovray, mainstream LR), so this is a hand-checked allowlist, NOT a rule.
# Keyed on (département, lowercased "prénom nom"); scoped to 2024 Legislatives T1
# only (some names, e.g. Dupont-Aignan, also ran in présidentielles).
#   → Extrême Droite: RN-allied candidates coded DVD/DSV (RN fielded no one against
#     them): Mexis (Marne, LR-Ciotti), Dupont-Aignan (Essonne, DLF/RN pact),
#     Paul-Petit (Seine-et-Marne, "RÀD (RN)" union des droites).
#   → Centre+Droite: Beaudet (Essonne, ex-LR presidential-adjacent) coded DIV→Other.
#   → Gauche: Gokel (Nord, Parti Socialiste) coded DIV→Other.
CANDIDATE_BLOCK_OVERRIDES: dict[tuple[str, str], str] = {
    ("51", "adrien mexis"): "Extreme_Droite",
    ("91", "nicolas dupont aignan"): "Extreme_Droite",
    ("77", "vincent paul petit"): "Extreme_Droite",
    ("91", "stephane beaudet"): "Centre+Droite",
    ("59", "julien gokel"): "Gauche",
}


def _mapped_result_blocks(elections: pd.DataFrame) -> pd.DataFrame:
    """Result rows with a `block` column (party→block mapping + verified 2024 T1
    candidate overrides). Shared by _build_block_scores and build_slate_presence
    so the target shares and the slate-presence mask always use the same routing."""
    results = elections[elections["metric_type"] == "Result"].copy()
    results["block"] = _vectorized_block_mapping(results["party"], results["candidate"])

    # Apply individually-verified 2024 Legislatives T1 candidate overrides.
    m = (results["election_type"] == "Legislatives_T1") & (
        results["date_float"].round(1) == 2024.5
    )
    if m.any():
        keyed = (
            results.loc[m, "location"].str[:2]
            + "|"
            + results.loc[m, "candidate"].str.strip().str.lower()
        )
        lookup = {f"{d}|{n}": b for (d, n), b in CANDIDATE_BLOCK_OVERRIDES.items()}
        overridden = keyed.map(lookup)
        results.loc[m, "block"] = overridden.fillna(results.loc[m, "block"])
    return results


def build_slate_presence(elections: pd.DataFrame) -> pd.DataFrame:
    """Per (location, election_type, date_float) presence of each TARGET_BLOCK,
    derived from the candidate slate (who filed — known ex ante, no vote-outcome
    leakage). A block is 'present' iff ≥1 candidate routed to it appears on the
    ballot. Columns: location, election_type, date_float, present_<block> (bool).

    Used to mask the deviation model: a block absent from the ballot has an actual
    share of exactly 0, so its predicted share is forced to 0. LOO-selected over
    partial/full redistribution (see src/mask_renorm_eval.py) — the votes a missing
    block would draw do not flow to the modeled survivors, so no renorm is added."""
    results = _mapped_result_blocks(elections)
    sub = results[results["block"].isin(TARGET_BLOCKS)]
    cnt = (
        sub.groupby(["location", "election_type", "date_float", "block"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for b in TARGET_BLOCKS:
        col = cnt[b] if b in cnt.columns else 0
        cnt[f"present_{b}"] = (col > 0) if b in cnt.columns else False
    return cnt[
        ["location", "election_type", "date_float"]
        + [f"present_{b}" for b in TARGET_BLOCKS]
    ]


def _build_block_scores(elections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate-level results to block-level per (location, election_type, date)."""
    results = _mapped_result_blocks(elections)

    scores = (
        results.groupby(["location", "election_type", "date_float", "block"])["value"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for b in TARGET_BLOCKS:
        if b not in scores.columns:
            scores[b] = 0.0

    # Abstention
    abs_df = elections[
        (elections["metric_type"] == "Context")
        & (elections["candidate"] == "Abstention")
    ][["location", "election_type", "date_float", "value"]].rename(
        columns={"value": "Abstention"}
    )

    scores = scores.merge(
        abs_df, on=["location", "election_type", "date_float"], how="inner"
    )
    return scores


def _add_cross_type_local_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add 2 lags of local BV block scores across ALL election types.

    For each BV, sort all its elections chronologically (across types)
    and shift to get the most recent 1-2 prior elections.
    """
    df = df.sort_values(["location", "date_float"])
    lag_cols = TARGET_BLOCKS + ["Abstention"]
    for col in lag_cols:
        df[f"{col}_lag1"] = df.groupby("location")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("location")[col].shift(2)
    return df


def _build_same_type_national_agg(block_scores: pd.DataFrame) -> pd.DataFrame:
    """National mean block scores per (election_type, date)."""
    national = (
        block_scores.groupby(["election_type", "date_float"])[
            TARGET_BLOCKS + ["Abstention"]
        ]
        .mean()
        .reset_index()
        .sort_values(["election_type", "date_float"])
    )
    return national


def _add_same_type_national_lags(
    df: pd.DataFrame, national_agg: pd.DataFrame
) -> pd.DataFrame:
    """Add 2 lags of same-type national aggregate block scores."""
    # Build lag DataFrames via shift within each election type
    national_agg = national_agg.sort_values(["election_type", "date_float"])
    lag_cols = TARGET_BLOCKS + ["Abstention"]

    lag1_df = national_agg.copy()
    lag2_df = national_agg.copy()
    for col in lag_cols:
        lag1_df[f"national_{col}_lag1"] = lag1_df.groupby("election_type")[col].shift(1)
        lag2_df[f"national_{col}_lag2"] = lag2_df.groupby("election_type")[col].shift(2)

    lag1_cols = [f"national_{c}_lag1" for c in lag_cols]
    lag2_cols = [f"national_{c}_lag2" for c in lag_cols]

    df = df.merge(
        lag1_df[["election_type", "date_float"] + lag1_cols],
        on=["election_type", "date_float"],
        how="left",
    )
    df = df.merge(
        lag2_df[["election_type", "date_float"] + lag2_cols],
        on=["election_type", "date_float"],
        how="left",
    )

    return df


def _build_national_poll_features(
    polls: pd.DataFrame,
    election_dates: list[tuple[str, float]],
    window: float = 1.0,
) -> pd.DataFrame:
    """Average national poll block scores in [date-window, date] per election.

    Uses ALL T1 polls (any type) within the window, as polls reflect national mood
    regardless of which election they target.
    """
    polls = polls.copy()
    polls["block"] = polls.apply(
        lambda r: _poll_token_to_block(
            str(r.get("party", "")), str(r.get("candidate", ""))
        ),
        axis=1,
    )
    # Keep national T1 polls with a mapped block
    national = polls[
        (polls["location"] == "National")
        & (polls["block"].isin(TARGET_BLOCKS))
        & (~polls["election_type"].str.contains("T2", na=False))
    ]

    n_mapped = len(national)
    n_total = len(polls[polls["location"] == "National"])
    print(
        f"  National polls: {n_mapped:,} mapped to blocks / {n_total:,} total",
        flush=True,
    )

    rows = []
    for etype, date in sorted(set(election_dates)):
        mask = (national["date_float"] >= date - window) & (
            national["date_float"] <= date
        )
        w = national[mask]
        if len(w) == 0:
            rows.append(
                {
                    "election_type": etype,
                    "date_float": date,
                    "poll_Gauche": np.nan,
                    "poll_Centre+Droite": np.nan,
                    "poll_Extreme_Droite": np.nan,
                    "has_polls": 0.0,
                }
            )
            continue

        per_poll = (
            w.groupby(["date_float", "metric_type", "block"])["value"]
            .sum()
            .reset_index()
        )
        poll_totals = per_poll.groupby(["date_float", "metric_type"])[
            "value"
        ].transform("sum")
        per_poll["value"] = per_poll["value"] / poll_totals * 100.0
        avgs = per_poll.groupby("block")["value"].mean()

        rows.append(
            {
                "election_type": etype,
                "date_float": date,
                "poll_Gauche": avgs.get("Gauche", np.nan),
                "poll_Centre+Droite": avgs.get("Centre+Droite", np.nan),
                "poll_Extreme_Droite": avgs.get("Extreme_Droite", np.nan),
                "has_polls": 1.0,
            }
        )

    return pd.DataFrame(rows)


def _add_demographics(
    df: pd.DataFrame, demos: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Merge last-available demographics per commune via merge_asof."""
    all_indicators = demos["candidate"].unique().tolist()
    df["commune"] = df["location"].str.split("_").str[0]
    demos = demos.sort_values("availability_date")
    df = df.sort_values("date_float")

    for ind in all_indicators:
        d = (
            demos[demos["candidate"] == ind][["location", "availability_date", "value"]]
            .rename(columns={"location": "commune", "value": ind})
            .sort_values("availability_date")
            .dropna(subset=["availability_date"])
        )
        df = pd.merge_asof(
            df,
            d,
            left_on="date_float",
            right_on="availability_date",
            by="commune",
            direction="backward",
        )
        if "availability_date" in df.columns:
            df = df.drop(columns=["availability_date"])

    available = [i for i in all_indicators if i in df.columns and df[i].notna().any()]
    dropped = set(all_indicators) - set(available)
    if dropped:
        print(
            f"  Dropped {len(dropped)} all-NaN indicators: {sorted(dropped)}",
            flush=True,
        )
    for ind in available:
        df[ind] = df[ind].fillna(df[ind].median())

    return df, available


def _add_election_type_onehot(df: pd.DataFrame) -> list[str]:
    """Add one-hot columns for election type. Cantonales → Departementales."""
    df["_type_canonical"] = df["election_type"].str.replace("_T1", "")
    df.loc[df["_type_canonical"] == "Cantonales", "_type_canonical"] = "Departementales"

    onehot_cols = []
    for t in TYPE_ONEHOT:
        col = f"type_{t}"
        df[col] = (df["_type_canonical"] == t).astype(np.float64)
        onehot_cols.append(col)

    df.drop(columns=["_type_canonical"], inplace=True)
    return onehot_cols


# ── Evaluation ────────────────────────────────────────────────────────


def _run_ridge(df, feature_cols, target_cols, val_mask, label=""):
    alphas = np.logspace(-2, 6, 60)
    train = df[~val_mask]
    val = df[val_mask]

    # Remove rows with NaN in features
    feat_notna = train[feature_cols].notna().all(axis=1)
    train = train[feat_notna]
    feat_notna_v = val[feature_cols].notna().all(axis=1)
    val = val[feat_notna_v]

    X_tr = train[feature_cols].values.astype(np.float64)
    X_v = val[feature_cols].values.astype(np.float64)

    print(
        f"  Train: {len(X_tr):,}, Val: {len(X_v):,}, Features: {len(feature_cols)}",
        flush=True,
    )
    if len(X_v) == 0:
        print("  ERROR: 0 val samples", flush=True)
        return {}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_v = scaler.transform(X_v)

    results = {}
    for t_col in target_cols:
        y_tr = train[t_col].values
        y_v = val[t_col].values
        model = RidgeCV(alphas=alphas)
        model.fit(X_tr, y_tr)
        pred_v = model.predict(X_v)
        r2_tr = r2_score(y_tr, model.predict(X_tr))
        r2_v = r2_score(y_v, pred_v)
        mae_v = np.mean(np.abs(y_v - pred_v))
        results[t_col] = r2_v
        print(
            f"  {t_col:20s}  Train R²={r2_tr:.4f}  Val R²={r2_v:.4f}  "
            f"MAE={mae_v:.2f}pp  (alpha={model.alpha_:.1f})",
            flush=True,
        )

    return results


# ── Main ──────────────────────────────────────────────────────────────


def main():
    data_dir = Path("data")

    # 1. Load data
    elections, demos = _load_cached(data_dir)
    print("Loading polls...", flush=True)
    polls = load_poll_tokens(data_dir)
    print(f"  Polls: {len(polls):,} tokens", flush=True)

    # 2. Phase 1: Legislatives + Presidentielle T1 only (11 dates, similar dynamics)
    PHASE1_TYPES = ["Legislatives_T1", "Presidentielle_T1"]
    t1 = elections[elections["election_type"].isin(PHASE1_TYPES)].copy()
    print(f"\nPhase 1 elections: {len(t1):,} rows")
    for etype in PHASE1_TYPES:
        sub = t1[t1["election_type"] == etype]
        dates = sorted(sub["date_float"].unique())
        print(
            f"  {etype:30s}  {len(dates)} dates: {[round(float(d), 2) for d in dates]}"
        )

    val_date = 2024.5
    val_type = "Legislatives_T1"

    # 3. Build block scores per (BV, election_type, date)
    print("\nBuilding block scores...", flush=True)
    block_scores = _build_block_scores(t1)
    coverage = block_scores[TARGET_BLOCKS].sum(axis=1)
    print(f"  Block coverage: mean={coverage.mean():.1f}%")
    print(f"  Total BV x election rows: {len(block_scores):,}")

    # Diagnostic: coverage per type
    if "Other" in block_scores.columns:
        for etype in PHASE1_TYPES:
            sub = block_scores[block_scores["election_type"] == etype]
            mapped = sub[TARGET_BLOCKS].sum(axis=1).mean()
            other = sub["Other"].mean()
            print(f"    {etype:30s}  mapped={mapped:.1f}%  other={other:.1f}%")

    # 4. National aggregates (for normalization and lags)
    national_agg = _build_same_type_national_agg(block_scores)
    print("\n  National averages:")
    for _, row in national_agg.iterrows():
        print(
            f"    {row['election_type']:30s} {row['date_float']:.2f}:  "
            f"G={row['Gauche']:.1f}  C+D={row['Centre+Droite']:.1f}  "
            f"ED={row['Extreme_Droite']:.1f}  Abs={row['Abstention']:.1f}"
        )

    # 5. Normalize block scores to DELTAS from national mean
    #    This makes cross-type lags comparable: +5pp left in pres ≈ +5pp left in legi
    print("\nNormalizing to deltas from national mean...", flush=True)
    nat_rename = {}
    for col in TARGET_BLOCKS + ["Abstention"]:
        nat_rename[col] = f"{col}_nat_mean"
    nat_for_merge = national_agg.rename(columns=nat_rename)
    block_scores = block_scores.merge(
        nat_for_merge[["election_type", "date_float"] + list(nat_rename.values())],
        on=["election_type", "date_float"],
        how="left",
    )
    for col in TARGET_BLOCKS + ["Abstention"]:
        block_scores[f"{col}_delta"] = (
            block_scores[col] - block_scores[f"{col}_nat_mean"]
        )

    # 6. Cross-type local BV lags (on DELTAS — comparable across types)
    print("Building cross-type local BV lags (on deltas)...", flush=True)
    block_scores = block_scores.sort_values(["location", "date_float"])
    delta_cols = [f"{b}_delta" for b in TARGET_BLOCKS] + ["Abstention_delta"]
    for col in delta_cols:
        block_scores[f"{col}_lag1"] = block_scores.groupby("location")[col].shift(1)
        block_scores[f"{col}_lag2"] = block_scores.groupby("location")[col].shift(2)
    # Also keep raw cross-type lags for comparison
    for col in TARGET_BLOCKS + ["Abstention"]:
        block_scores[f"{col}_lag1"] = block_scores.groupby("location")[col].shift(1)
        block_scores[f"{col}_lag2"] = block_scores.groupby("location")[col].shift(2)

    df = block_scores

    # 7. Same-type national lags
    print("Building same-type national aggregate lags...", flush=True)
    df = _add_same_type_national_lags(df, national_agg)

    # 8. National poll features
    print("Building national poll features...", flush=True)
    election_dates = list(
        df[["election_type", "date_float"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    poll_feats = _build_national_poll_features(polls, election_dates)
    n_with_polls = (poll_feats["has_polls"] == 1.0).sum()
    print(f"  {n_with_polls}/{len(election_dates)} election dates have poll coverage")
    for _, row in poll_feats.iterrows():
        if row["has_polls"]:
            print(
                f"    {row['election_type']:30s} {row['date_float']:.2f}:  "
                f"G={row['poll_Gauche']:.1f}  C+D={row['poll_Centre+Droite']:.1f}  "
                f"ED={row['poll_Extreme_Droite']:.1f}"
            )

    df = df.merge(poll_feats, on=["election_type", "date_float"], how="left")
    for col in ["poll_Gauche", "poll_Centre+Droite", "poll_Extreme_Droite"]:
        df[col] = df[col].fillna(0.0)
    df["has_polls"] = df["has_polls"].fillna(0.0)

    # Poll deltas: poll_block - national_lag1_block
    for b in TARGET_BLOCKS:
        df[f"poll_delta_{b}"] = df[f"poll_{b}"] - df[f"national_{b}_lag1"]
    poll_delta_cols = [f"poll_delta_{b}" for b in TARGET_BLOCKS] + ["has_polls"]

    # 9. Geo coordinates
    geo = t1[["location", "latitude", "longitude"]].drop_duplicates("location")
    df = df.merge(geo, on="location", how="left")
    df["latitude"] = df["latitude"].fillna(46.2276)
    df["longitude"] = df["longitude"].fillna(2.2137)

    # 10. Election type indicator (just 1 feature: is_presidentielle)
    df["is_presidentielle"] = (df["election_type"] == "Presidentielle_T1").astype(
        np.float64
    )

    # 11. Demographics
    print("Merging demographics...", flush=True)
    df, demo_indicators = _add_demographics(df, demos)

    # 12. Clean up NaN in targets
    df = df.dropna(subset=TARGET_COLS)
    print(f"\nFinal dataset: {len(df):,} rows, {df['location'].nunique():,} unique BVs")

    # ── Validation mask: 2024 Legislatives T1 ─────────────────────────
    val_mask = np.isclose(df["date_float"], val_date, atol=1e-3) & (
        df["election_type"] == val_type
    )
    print(f"Val set: {val_mask.sum():,} BVs (2024 Legislatives T1)")
    print(f"Train set: {(~val_mask).sum():,} rows (Legi + Pres T1)")

    # ── Define feature groups ─────────────────────────────────────────
    demo_cols = demo_indicators
    geo_cols = ["latitude", "longitude"]
    time_cols = ["date_float"]
    poll_cols = [
        "poll_Gauche",
        "poll_Centre+Droite",
        "poll_Extreme_Droite",
        "has_polls",
    ]
    raw_lag1_cols = [f"{b}_lag1" for b in TARGET_BLOCKS] + ["Abstention_lag1"]
    raw_lag2_cols = [f"{b}_lag2" for b in TARGET_BLOCKS] + ["Abstention_lag2"]
    delta_lag1_cols = [f"{b}_delta_lag1" for b in TARGET_BLOCKS] + [
        "Abstention_delta_lag1"
    ]
    delta_lag2_cols = [f"{b}_delta_lag2" for b in TARGET_BLOCKS] + [
        "Abstention_delta_lag2"
    ]
    national_lag1_cols = [f"national_{b}_lag1" for b in TARGET_BLOCKS] + [
        "national_Abstention_lag1"
    ]
    national_lag2_cols = [f"national_{b}_lag2" for b in TARGET_BLOCKS] + [
        "national_Abstention_lag2"
    ]
    type_cols = ["is_presidentielle"]

    # ── Model A: Delta lags + polls + national lags + type indicator ──
    features_a = (
        demo_cols
        + geo_cols
        + time_cols
        + poll_cols
        + delta_lag1_cols
        + delta_lag2_cols
        + national_lag1_cols
        + national_lag2_cols
        + type_cols
    )
    print(f"\n{'=' * 70}")
    print(f"MODEL A: LEGI+PRES — demos + 2 DELTA lags + polls + national lags + type")
    print(f"  ({len(features_a)} features)")
    print(f"{'=' * 70}")
    _run_ridge(df, features_a, TARGET_COLS, val_mask, "A")

    # ── Model B: Raw lags + polls + national lags + type ──────────────
    features_b = (
        demo_cols
        + geo_cols
        + time_cols
        + poll_cols
        + raw_lag1_cols
        + raw_lag2_cols
        + national_lag1_cols
        + national_lag2_cols
        + type_cols
    )
    print(f"\n{'=' * 70}")
    print(f"MODEL B: LEGI+PRES — demos + 2 RAW lags + polls + national lags + type")
    print(f"  ({len(features_b)} features)")
    print(f"{'=' * 70}")
    _run_ridge(df, features_b, TARGET_COLS, val_mask, "B")

    # ── Model C: Poll deltas instead of absolute polls ────────────────
    features_c = (
        demo_cols
        + geo_cols
        + time_cols
        + poll_delta_cols
        + delta_lag1_cols
        + delta_lag2_cols
        + national_lag1_cols
        + national_lag2_cols
        + type_cols
    )
    print(f"\n{'=' * 70}")
    print(f"MODEL C: LEGI+PRES — demos + delta lags + POLL DELTAS + national lags")
    print(f"  ({len(features_c)} features)")
    print(f"{'=' * 70}")
    _run_ridge(df, features_c, TARGET_COLS, val_mask, "C")

    # ── Model D: No polls (ablation) ──────────────────────────────────
    features_d = (
        demo_cols
        + geo_cols
        + time_cols
        + delta_lag1_cols
        + delta_lag2_cols
        + national_lag1_cols
        + national_lag2_cols
        + type_cols
    )
    print(f"\n{'=' * 70}")
    print(f"MODEL D: LEGI+PRES — no polls")
    print(f"  ({len(features_d)} features)")
    print(f"{'=' * 70}")
    _run_ridge(df, features_d, TARGET_COLS, val_mask, "D")

    # ── Model E: 1 delta lag (more rows) ──────────────────────────────
    features_e = (
        demo_cols
        + geo_cols
        + time_cols
        + poll_cols
        + delta_lag1_cols
        + national_lag1_cols
        + type_cols
    )
    print(f"\n{'=' * 70}")
    print(f"MODEL E: LEGI+PRES — 1 delta lag + polls")
    print(f"  ({len(features_e)} features)")
    print(f"{'=' * 70}")
    _run_ridge(df, features_e, TARGET_COLS, val_mask, "E")

    # ── Model F: Legi-only baseline repro ─────────────────────────────
    legi_mask = df["election_type"] == "Legislatives_T1"
    df_legi = df[legi_mask].copy()
    val_mask_legi = np.isclose(df_legi["date_float"], val_date, atol=1e-3)
    # Rebuild lags within legislatives only
    df_legi = df_legi.sort_values(["location", "date_float"])
    for col in TARGET_BLOCKS + ["Abstention"]:
        df_legi[f"{col}_lag1"] = df_legi.groupby("location")[col].shift(1)
        df_legi[f"{col}_lag2"] = df_legi.groupby("location")[col].shift(2)

    features_f = demo_cols + geo_cols + time_cols + raw_lag1_cols + raw_lag2_cols
    print(f"\n{'=' * 70}")
    print(f"MODEL F: LEGI-ONLY BASELINE REPRO — demos + 2 same-type lags (no polls)")
    print(f"  ({len(features_f)} features)")
    print(f"{'=' * 70}")
    _run_ridge(df_legi, features_f, TARGET_COLS, val_mask_legi, "F")

    # ── Model G: Legi-only + polls ────────────────────────────────────
    features_g = (
        demo_cols + geo_cols + time_cols + raw_lag1_cols + raw_lag2_cols + poll_cols
    )
    print(f"\n{'=' * 70}")
    print(f"MODEL G: LEGI-ONLY + POLLS — demos + 2 same-type lags + polls")
    print(f"  ({len(features_g)} features)")
    print(f"{'=' * 70}")
    _run_ridge(df_legi, features_g, TARGET_COLS, val_mask_legi, "G")

    # ── Model H: Legi-only + poll deltas ──────────────────────────────
    features_h = (
        demo_cols
        + geo_cols
        + time_cols
        + raw_lag1_cols
        + raw_lag2_cols
        + poll_delta_cols
    )
    print(f"\n{'=' * 70}")
    print(f"MODEL H: LEGI-ONLY + POLL DELTAS")
    print(f"  ({len(features_h)} features)")
    print(f"{'=' * 70}")
    _run_ridge(df_legi, features_h, TARGET_COLS, val_mask_legi, "H")


if __name__ == "__main__":
    main()
