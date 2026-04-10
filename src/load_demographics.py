"""Load INSEE Census demographic data as universal tokens — multi-vintage.

Reads Census (activité, diplômes, population) commune-level data and
converts each indicator × commune into a DataToken with
metric_type='Demographics'.

**Multi-vintage**: scans ``data/demographics/census/{vintage}/`` for all
available year directories (2006-2022) and loads every one.  Each vintage
gets ``date_float`` and ``availability_date`` computed from the publication
calendar so the router can enforce temporal causality.

Publication calendar (approximate):
  Census vintage Y → pools surveys Y-4 to Y, centred ~Y-1.5
                   → detailed commune data published ~June Y+3
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date_float", "availability_date", "election_type", "location",
            "candidate", "party", "metric_type", "value",
            "latitude", "longitude",
        ]
    )


def _read_insee_file(path: Path) -> pd.DataFrame | None:
    """Read an INSEE data file (Excel or CSV) and return the data sheet."""
    if not path.exists():
        return None
    try:
        if path.suffix in (".xlsx", ".xls"):
            xls = pd.ExcelFile(path)
            # Try each sheet, with and without skiprows (old XLS have 5 header rows)
            for sheet_name in xls.sheet_names:
                for skip in [0, 5]:
                    try:
                        df = pd.read_excel(
                            xls, sheet_name=sheet_name,
                            skiprows=skip, dtype={"CODGEO": str},
                        )
                        if "CODGEO" in df.columns:
                            return df
                    except Exception:
                        continue
            return None
        else:
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(path, sep=sep, dtype={"CODGEO": str}, low_memory=False)
                    if "CODGEO" in df.columns:
                        return df
                except Exception:
                    continue
            return None
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None


def _find_col(df: pd.DataFrame, exact: str) -> str | None:
    """Find a column by exact name (case-insensitive)."""
    for col in df.columns:
        if col.upper() == exact.upper():
            return col
    return None


def _detect_prefix(df: pd.DataFrame) -> str | None:
    """Auto-detect the 2-digit vintage prefix from column names.

    Census columns follow patterns like P21_POP, C21_ACT_CSP1 etc.
    Returns the 2-digit string (e.g. '21') or None.
    """
    for col in df.columns:
        m = re.match(r'^[PC](\d{2})_', col)
        if m:
            return m.group(1)
    return None


def _safe_ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan)


def _make_tokens(
    codgeo: pd.Series,
    indicator: str,
    values: pd.Series,
    date_float: float,
    availability_date: float,
) -> pd.DataFrame:
    """Create token rows for a single indicator."""
    mask = values.notna() & np.isfinite(values)
    if not mask.any():
        return pd.DataFrame()
    return pd.DataFrame({
        "date_float": np.float32(date_float),
        "availability_date": np.float32(availability_date),
        "election_type": "",
        "location": codgeo[mask].values,
        "candidate": indicator,
        "party": "",
        "metric_type": "Demographics",
        "value": values[mask].astype(np.float32).values,
    })


def _glob_first(directory: Path, *patterns: str) -> Path | None:
    """Return the first file matching any of the glob patterns."""
    for pat in patterns:
        hits = sorted(directory.glob(pat))
        # Skip meta_ files
        hits = [h for h in hits if not h.name.startswith("meta_")]
        if hits:
            return hits[0]
    return None


# ═══════════════════════════════════════════════════════════════════════
# Census loaders — parametrised by vintage prefix
# ═══════════════════════════════════════════════════════════════════════

def _load_census_activity_vintage(
    vintage_dir: Path, date_float: float, avail_date: float,
) -> pd.DataFrame:
    """Unemployment rate, % ouvriers, % cadres from a single vintage dir."""
    path = _glob_first(
        vintage_dir,
        "*emploi*pop*activ*", "*emploi*pop*act*",
        "*activ*", "*ACT*", "*carac*emploi*", "*caract*emploi*",
    )
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return _empty_df()

    yy = _detect_prefix(df)
    if yy is None:
        return _empty_df()

    codgeo = df["CODGEO"].astype(str)
    frames: list[pd.DataFrame] = []

    # Unemployment rate = CHOM / ACT * 100
    chom = _find_col(df, f"P{yy}_CHOM1564")
    act = _find_col(df, f"P{yy}_ACT1564")
    if chom and act:
        rate = _safe_ratio(
            pd.to_numeric(df[chom], errors="coerce"),
            pd.to_numeric(df[act], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Taux_Chomage", rate, date_float, avail_date))

    # CSP breakdown — column pattern is C{yy}_ACT1564_CS{i}
    csp_series: dict[int, pd.Series] = {}
    for i in range(1, 7):
        col = _find_col(df, f"C{yy}_ACT1564_CS{i}")
        if col:
            csp_series[i] = pd.to_numeric(df[col], errors="coerce")

    if len(csp_series) >= 2:
        total_csp = sum(csp_series.values())
        # CS1=agriculteurs, CS2=artisans, CS3=cadres, CS4=prof.intermédiaires,
        # CS5=employés, CS6=ouvriers
        if 6 in csp_series:
            frames.append(_make_tokens(
                codgeo, "Pct_Ouvriers",
                _safe_ratio(csp_series[6], total_csp) * 100.0, date_float, avail_date,
            ))
        if 3 in csp_series:
            frames.append(_make_tokens(
                codgeo, "Pct_Cadres",
                _safe_ratio(csp_series[3], total_csp) * 100.0, date_float, avail_date,
            ))
        if 5 in csp_series:
            frames.append(_make_tokens(
                codgeo, "Pct_Employes",
                _safe_ratio(csp_series[5], total_csp) * 100.0, date_float, avail_date,
            ))
        if 4 in csp_series:
            frames.append(_make_tokens(
                codgeo, "Pct_Prof_Intermediaires",
                _safe_ratio(csp_series[4], total_csp) * 100.0, date_float, avail_date,
            ))
        if 1 in csp_series:
            frames.append(_make_tokens(
                codgeo, "Pct_Agriculteurs",
                _safe_ratio(csp_series[1], total_csp) * 100.0, date_float, avail_date,
            ))
        if 2 in csp_series:
            frames.append(_make_tokens(
                codgeo, "Pct_Artisans",
                _safe_ratio(csp_series[2], total_csp) * 100.0, date_float, avail_date,
            ))

    # Employment by sector
    emplt = _find_col(df, f"C{yy}_EMPLT")
    if emplt:
        emplt_total = pd.to_numeric(df[emplt], errors="coerce")
        for sector_suffix, indicator in [
            ("AGRI", "Pct_Emploi_Agriculture"),
            ("INDUS", "Pct_Emploi_Industrie"),
            ("CONST", "Pct_Emploi_Construction"),
            ("CTS", "Pct_Emploi_Tertiaire"),
        ]:
            sector_col = _find_col(df, f"C{yy}_EMPLT_{sector_suffix}")
            if sector_col:
                pct = _safe_ratio(pd.to_numeric(df[sector_col], errors="coerce"), emplt_total) * 100.0
                frames.append(_make_tokens(codgeo, indicator, pct, date_float, avail_date))

    # Retired, student, and other inactive rates
    retr = _find_col(df, f"P{yy}_RETR1564")
    pop1564 = _find_col(df, f"P{yy}_POP1564")
    if retr and pop1564:
        pct = _safe_ratio(
            pd.to_numeric(df[retr], errors="coerce"),
            pd.to_numeric(df[pop1564], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Retraites", pct, date_float, avail_date))

    etud = _find_col(df, f"P{yy}_ETUD1564")
    if etud and pop1564:
        pct = _safe_ratio(
            pd.to_numeric(df[etud], errors="coerce"),
            pd.to_numeric(df[pop1564], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Etudiants", pct, date_float, avail_date))

    ainact = _find_col(df, f"P{yy}_AINACT1564")
    if ainact and pop1564:
        pct = _safe_ratio(
            pd.to_numeric(df[ainact], errors="coerce"),
            pd.to_numeric(df[pop1564], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Autres_Inactifs", pct, date_float, avail_date))

    if not frames:
        return _empty_df()
    return pd.concat(frames, ignore_index=True)


def _load_census_education_vintage(
    vintage_dir: Path, date_float: float, avail_date: float,
) -> pd.DataFrame:
    """% sans diplôme, % bac+5 from a single vintage dir."""
    path = _glob_first(
        vintage_dir,
        "*diplom*", "*DIPL*", "*form*", "*FOR*",
    )
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return _empty_df()

    yy = _detect_prefix(df)
    if yy is None:
        return _empty_df()

    codgeo = df["CODGEO"].astype(str)
    frames: list[pd.DataFrame] = []

    denom_col = _find_col(df, f"P{yy}_NSCOL15P")
    if not denom_col:
        return _empty_df()
    denom = pd.to_numeric(df[denom_col], errors="coerce")

    # No diploma: DIPLMIN (2013+) or DIPL0 (2010-2012)
    nodip = _find_col(df, f"P{yy}_NSCOL15P_DIPLMIN") or _find_col(df, f"P{yy}_NSCOL15P_DIPL0")
    if nodip:
        pct = _safe_ratio(pd.to_numeric(df[nodip], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Sans_Diplome", pct, date_float, avail_date))

    # Higher education: SUP5 (2013+, bac+5 specifically) or SUP (2010-2012, all higher ed)
    sup5 = _find_col(df, f"P{yy}_NSCOL15P_SUP5") or _find_col(df, f"P{yy}_NSCOL15P_SUP")
    if sup5:
        pct = _safe_ratio(pd.to_numeric(df[sup5], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Bac_Plus_5", pct, date_float, avail_date))

    # BAC level
    bac = _find_col(df, f"P{yy}_NSCOL15P_BAC")
    if bac:
        pct = _safe_ratio(pd.to_numeric(df[bac], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Bac", pct, date_float, avail_date))

    # CAP/BEP level
    capbep = _find_col(df, f"P{yy}_NSCOL15P_CAPBEP")
    if capbep:
        pct = _safe_ratio(pd.to_numeric(df[capbep], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_CAP_BEP", pct, date_float, avail_date))

    # SUP2 (bac+2) — only available in 2017+ vintages
    sup2 = _find_col(df, f"P{yy}_NSCOL15P_SUP2")
    if sup2:
        pct = _safe_ratio(pd.to_numeric(df[sup2], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Bac_Plus_2", pct, date_float, avail_date))

    # SUP34 (bac+3/4) — only available in 2017+ vintages
    sup34 = _find_col(df, f"P{yy}_NSCOL15P_SUP34")
    if sup34:
        pct = _safe_ratio(pd.to_numeric(df[sup34], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Bac_Plus_3_4", pct, date_float, avail_date))

    # BEPC level
    bepc = _find_col(df, f"P{yy}_NSCOL15P_BEPC")
    if bepc:
        pct = _safe_ratio(pd.to_numeric(df[bepc], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_BEPC", pct, date_float, avail_date))

    if not frames:
        return _empty_df()
    return pd.concat(frames, ignore_index=True)


def _load_census_population_vintage(
    vintage_dir: Path, date_float: float, avail_date: float,
) -> pd.DataFrame:
    """% age 15-24, % age 60+, % immigrants from a single vintage dir."""
    path = _glob_first(
        vintage_dir,
        "*evol*struct*pop*", "*pop*", "*POP*",
    )
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return _empty_df()

    yy = _detect_prefix(df)
    if yy is None:
        return _empty_df()

    codgeo = df["CODGEO"].astype(str)
    frames: list[pd.DataFrame] = []

    pop_col = _find_col(df, f"P{yy}_POP")
    if not pop_col:
        return _empty_df()
    pop = pd.to_numeric(df[pop_col], errors="coerce")

    # Young population: try POP1524 (post-2017) then POP1529 (pre-2017)
    young = _find_col(df, f"P{yy}_POP1524") or _find_col(df, f"P{yy}_POP1529")
    if young:
        pct = _safe_ratio(pd.to_numeric(df[young], errors="coerce"), pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_18_24", pct, date_float, avail_date))

    # Elderly population: POP6074 + POP75P (post-2017) or POP7589 + POP90P (pre-2017)
    e60 = _find_col(df, f"P{yy}_POP6074")
    e75 = _find_col(df, f"P{yy}_POP75P")
    e7589 = _find_col(df, f"P{yy}_POP7589")
    e90 = _find_col(df, f"P{yy}_POP90P")
    elder_parts = []
    if e60:
        elder_parts.append(pd.to_numeric(df[e60], errors="coerce"))
    if e75:
        elder_parts.append(pd.to_numeric(df[e75], errors="coerce"))
    elif e7589 and e90:
        elder_parts.append(pd.to_numeric(df[e7589], errors="coerce"))
        elder_parts.append(pd.to_numeric(df[e90], errors="coerce"))
    if e60 and len(elder_parts) >= 2:
        elder = sum(elder_parts)
        pct = _safe_ratio(elder, pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_60_Plus", pct, date_float, avail_date))

    # Immigration: IRAN2 = immigrés (born foreign abroad)
    # POP01P_IRAN2 is available in population file from 2013+ vintages
    imm = _find_col(df, f"P{yy}_POP01P_IRAN2")
    pop01p = _find_col(df, f"P{yy}_POP01P")
    if imm and pop01p:
        pct = _safe_ratio(
            pd.to_numeric(df[imm], errors="coerce"),
            pd.to_numeric(df[pop01p], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Immigres", pct, date_float, avail_date))

    # Middle-age brackets: 30-44, 45-59
    m3044 = _find_col(df, f"P{yy}_POP3044")
    if m3044:
        pct = _safe_ratio(pd.to_numeric(df[m3044], errors="coerce"), pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_30_44", pct, date_float, avail_date))

    m4559 = _find_col(df, f"P{yy}_POP4559")
    if m4559:
        pct = _safe_ratio(pd.to_numeric(df[m4559], errors="coerce"), pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_45_59", pct, date_float, avail_date))

    # Under-15 (minors)
    m0014 = _find_col(df, f"P{yy}_POP0014")
    if m0014:
        pct = _safe_ratio(pd.to_numeric(df[m0014], errors="coerce"), pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_0_14", pct, date_float, avail_date))

    # Very elderly (75+)
    if e75:
        pct = _safe_ratio(pd.to_numeric(df[e75], errors="coerce"), pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_75_Plus", pct, date_float, avail_date))
    elif e7589 and e90:
        elder75 = (pd.to_numeric(df[e7589], errors="coerce") +
                   pd.to_numeric(df[e90], errors="coerce"))
        pct = _safe_ratio(elder75, pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_75_Plus", pct, date_float, avail_date))

    if not frames:
        return _empty_df()
    return pd.concat(frames, ignore_index=True)


def _load_census_housing_vintage(
    vintage_dir: Path, date_float: float, avail_date: float,
) -> pd.DataFrame:
    """% HLM, % propriétaires, % logements vacants from a single vintage dir."""
    path = _glob_first(
        vintage_dir,
        "*logement*", "*LOG*", "*log*",
    )
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return _empty_df()

    yy = _detect_prefix(df)
    if yy is None:
        return _empty_df()

    codgeo = df["CODGEO"].astype(str)
    frames: list[pd.DataFrame] = []

    # Total principal residences
    rp_col = _find_col(df, f"P{yy}_RP")
    if not rp_col:
        return _empty_df()
    rp = pd.to_numeric(df[rp_col], errors="coerce")

    # Owner-occupied
    prop = _find_col(df, f"P{yy}_RP_PROP")
    if prop:
        pct = _safe_ratio(pd.to_numeric(df[prop], errors="coerce"), rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Proprietaires", pct, date_float, avail_date))

    # HLM (social housing) — column name varies: LOCHLM (pre-2017) or LOCHLMV (2017+)
    hlm = _find_col(df, f"P{yy}_RP_LOCHLMV") or _find_col(df, f"P{yy}_RP_LOCHLM")
    if hlm:
        pct = _safe_ratio(pd.to_numeric(df[hlm], errors="coerce"), rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_HLM", pct, date_float, avail_date))

    # Private renters
    loc = _find_col(df, f"P{yy}_RP_LOC")
    if loc:
        pct = _safe_ratio(pd.to_numeric(df[loc], errors="coerce"), rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Locataires", pct, date_float, avail_date))

    # Vacant dwellings
    log_col = _find_col(df, f"P{yy}_LOG")
    vac = _find_col(df, f"P{yy}_LOGVAC")
    if log_col and vac:
        pct = _safe_ratio(
            pd.to_numeric(df[vac], errors="coerce"),
            pd.to_numeric(df[log_col], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Logements_Vacants", pct, date_float, avail_date))

    # Houses vs apartments
    maison = _find_col(df, f"P{yy}_RPMAISON")
    if maison:
        pct = _safe_ratio(pd.to_numeric(df[maison], errors="coerce"), rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Maisons", pct, date_float, avail_date))

    # Small dwellings (1-2 rooms)
    rp1 = _find_col(df, f"P{yy}_RP_1P")
    rp2 = _find_col(df, f"P{yy}_RP_2P")
    if rp1 and rp2:
        small = (pd.to_numeric(df[rp1], errors="coerce") +
                 pd.to_numeric(df[rp2], errors="coerce"))
        pct = _safe_ratio(small, rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Petits_Logements", pct, date_float, avail_date))

    # Large dwellings (5+ rooms)
    rp5 = _find_col(df, f"P{yy}_RP_5PP")
    if rp5:
        pct = _safe_ratio(pd.to_numeric(df[rp5], errors="coerce"), rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Grands_Logements", pct, date_float, avail_date))

    # Heating type — extract for primary prefix AND any embedded historical prefixes
    # (e.g. the 2022 file embeds P11_ and P16_ comparison columns for heating)
    _HEAT_SUFFIXES = [("CELEC", "Pct_Chauff_Elec"),
                      ("CFIOUL", "Pct_Chauff_Fioul"),
                      ("CGAZB", "Pct_Chauff_Gaz")]
    for heat_col_name, indicator in _HEAT_SUFFIXES:
        col = _find_col(df, f"P{yy}_RP_{heat_col_name}")
        if col:
            pct = _safe_ratio(pd.to_numeric(df[col], errors="coerce"), rp) * 100.0
            frames.append(_make_tokens(codgeo, indicator, pct, date_float, avail_date))

    # Extract heating from embedded historical prefixes (only in recent files)
    embedded_prefixes = set()
    for c in df.columns:
        m = re.match(r'^[PC](\d{2})_', c)
        if m:
            embedded_prefixes.add(m.group(1))
    embedded_prefixes.discard(yy)
    for emb_yy in sorted(embedded_prefixes):
        emb_rp_col = _find_col(df, f"P{emb_yy}_RP")
        if not emb_rp_col:
            continue
        emb_rp = pd.to_numeric(df[emb_rp_col], errors="coerce")
        emb_vintage = int(emb_yy) + 2000
        emb_date_float, emb_avail_date = _census_dates(emb_vintage)
        for heat_col_name, indicator in _HEAT_SUFFIXES:
            col = _find_col(df, f"P{emb_yy}_RP_{heat_col_name}")
            if col:
                pct = _safe_ratio(pd.to_numeric(df[col], errors="coerce"), emb_rp) * 100.0
                frames.append(_make_tokens(codgeo, indicator, pct, emb_date_float, emb_avail_date))

    # Old housing stock (pre-1945)
    # 2022+: ACH1919/ACH1945;  2013-2021: ACH19/ACH45
    ach_old = _find_col(df, f"P{yy}_RP_ACH1919") or _find_col(df, f"P{yy}_RP_ACH19")
    ach_45 = _find_col(df, f"P{yy}_RP_ACH1945") or _find_col(df, f"P{yy}_RP_ACH45")
    if ach_old:
        old_stock = pd.to_numeric(df[ach_old], errors="coerce")
        if ach_45:
            old_stock = old_stock + pd.to_numeric(df[ach_45], errors="coerce")
        pct = _safe_ratio(old_stock, rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Logements_Anciens", pct, date_float, avail_date))

    # Over-occupied dwellings
    # 2022+: SUROCC_MOD + SUROCC_ACC (moderate + acute);  2017-2021: HSTU1P_SUROCC (total)
    surocc_mod = _find_col(df, f"C{yy}_RP_SUROCC_MOD")
    surocc_acc = _find_col(df, f"C{yy}_RP_SUROCC_ACC")
    surocc_total = _find_col(df, f"C{yy}_RP_HSTU1P_SUROCC")
    if surocc_mod or surocc_acc:
        surocc = pd.Series(0.0, index=df.index)
        if surocc_mod:
            surocc = surocc + pd.to_numeric(df[surocc_mod], errors="coerce").fillna(0)
        if surocc_acc:
            surocc = surocc + pd.to_numeric(df[surocc_acc], errors="coerce").fillna(0)
        pct = _safe_ratio(surocc, rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Suroccupation", pct, date_float, avail_date))
    elif surocc_total:
        pct = _safe_ratio(pd.to_numeric(df[surocc_total], errors="coerce"), rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Suroccupation", pct, date_float, avail_date))

    # Free housing (gratuit)
    grat = _find_col(df, f"P{yy}_RP_GRAT")
    if grat:
        pct = _safe_ratio(pd.to_numeric(df[grat], errors="coerce"), rp) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Logement_Gratuit", pct, date_float, avail_date))

    if not frames:
        return _empty_df()
    return pd.concat(frames, ignore_index=True)


def _load_census_families_vintage(
    vintage_dir: Path, date_float: float, avail_date: float,
) -> pd.DataFrame:
    """% single-person households, % single-parent families from a single vintage dir."""
    path = _glob_first(
        vintage_dir,
        "*coupl*fam*men*", "*fam*men*", "*menage*", "*MEN*",
    )
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return _empty_df()

    yy = _detect_prefix(df)
    if yy is None:
        return _empty_df()

    codgeo = df["CODGEO"].astype(str)
    frames: list[pd.DataFrame] = []

    # Single-person households: MENPSEUL / MEN
    men = _find_col(df, f"C{yy}_MEN")
    seul = _find_col(df, f"C{yy}_MENPSEUL")
    if men and seul:
        pct = _safe_ratio(
            pd.to_numeric(df[seul], errors="coerce"),
            pd.to_numeric(df[men], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Menages_Seuls", pct, date_float, avail_date))

    # Single-parent families: FAMMONO / FAM
    fam = _find_col(df, f"C{yy}_FAM")
    mono = _find_col(df, f"C{yy}_FAMMONO")
    if fam and mono:
        pct = _safe_ratio(
            pd.to_numeric(df[mono], errors="coerce"),
            pd.to_numeric(df[fam], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Familles_Monoparentales", pct, date_float, avail_date))

    # Couples with / without children
    if fam:
        fam_total = pd.to_numeric(df[fam], errors="coerce")
        coupaenf = _find_col(df, f"C{yy}_COUPAENF")
        if coupaenf:
            pct = _safe_ratio(pd.to_numeric(df[coupaenf], errors="coerce"), fam_total) * 100.0
            frames.append(_make_tokens(codgeo, "Pct_Couples_Avec_Enfants", pct, date_float, avail_date))
        coupsenf = _find_col(df, f"C{yy}_COUPSENF")
        if coupsenf:
            pct = _safe_ratio(pd.to_numeric(df[coupsenf], errors="coerce"), fam_total) * 100.0
            frames.append(_make_tokens(codgeo, "Pct_Couples_Sans_Enfants", pct, date_float, avail_date))

    # Large families (3+ children under 25)
    ne24f3 = _find_col(df, f"C{yy}_NE24F3")
    ne24f4p = _find_col(df, f"C{yy}_NE24F4P")
    if fam and (ne24f3 or ne24f4p):
        fam_total = pd.to_numeric(df[fam], errors="coerce")
        large = pd.Series(0.0, index=df.index)
        if ne24f3:
            large = large + pd.to_numeric(df[ne24f3], errors="coerce").fillna(0)
        if ne24f4p:
            large = large + pd.to_numeric(df[ne24f4p], errors="coerce").fillna(0)
        pct = _safe_ratio(large, fam_total) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Familles_Nombreuses", pct, date_float, avail_date))

    # Marital status breakdown (% of 15+ population)
    pop15p = _find_col(df, f"P{yy}_POP15P")
    if pop15p:
        pop15p_val = pd.to_numeric(df[pop15p], errors="coerce")
        for status_col, indicator in [
            (f"P{yy}_POP15P_CELIBATAIRE", "Pct_Celibataires"),
            (f"P{yy}_POP15P_MARIEE", "Pct_Maries"),
            (f"P{yy}_POP15P_DIVORCEE", "Pct_Divorces"),
            (f"P{yy}_POP15P_VEUFS", "Pct_Veufs"),
            (f"P{yy}_POP15P_PACSEE", "Pct_Pacses"),
            (f"P{yy}_POP15P_CONCUB_UNION_LIBRE", "Pct_Union_Libre"),
        ]:
            col = _find_col(df, status_col)
            if col:
                pct = _safe_ratio(pd.to_numeric(df[col], errors="coerce"), pop15p_val) * 100.0
                frames.append(_make_tokens(codgeo, indicator, pct, date_float, avail_date))

    if not frames:
        return _empty_df()
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Multi-vintage Census orchestrator
# ═══════════════════════════════════════════════════════════════════════

# Publication calendar: census vintage Y → published ~June of Y+3
# Data centred ~Y-1.5  (pools surveys Y-4 to Y)
_CENSUS_PUB_OFFSET = 3   # years after vintage for publication
_CENSUS_PUB_MONTH = 0.5  # ~June = 0.5 into the year
_CENSUS_CENTER_OFFSET = 1.5  # data centre lag


def _census_dates(vintage: int) -> tuple[float, float]:
    """Return (date_float, availability_date) for a census vintage."""
    date_float = vintage - _CENSUS_CENTER_OFFSET
    availability_date = vintage + _CENSUS_PUB_OFFSET + _CENSUS_PUB_MONTH
    return date_float, availability_date


def _load_all_census(demo_dir: Path) -> pd.DataFrame:
    """Load all census vintages.  Returns tokens DataFrame."""
    census_dir = demo_dir / "census"
    if not census_dir.exists():
        return _empty_df()

    # Discover vintage directories
    vintage_dirs = sorted(
        d for d in census_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )

    if not vintage_dirs:
        return _empty_df()

    all_frames: list[pd.DataFrame] = []

    for vdir in vintage_dirs:
        vintage = int(vdir.name)
        date_float, avail_date = _census_dates(vintage)

        frames = [
            _load_census_activity_vintage(vdir, date_float, avail_date),
            _load_census_education_vintage(vdir, date_float, avail_date),
            _load_census_population_vintage(vdir, date_float, avail_date),
            _load_census_housing_vintage(vdir, date_float, avail_date),
            _load_census_families_vintage(vdir, date_float, avail_date),
        ]
        frames = [f for f in frames if len(f) > 0]

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            n_ind = combined["candidate"].nunique()
            n_com = combined["location"].nunique()
            print(f"  Census {vintage}: {len(combined):,} tokens "
                  f"({n_ind} indicators × {n_com:,} communes) "
                  f"[date={date_float:.1f}, avail={avail_date:.1f}]")
            all_frames.append(combined)
        else:
            print(f"  Census {vintage}: no indicators extracted (column mismatch?)")

    if not all_frames:
        return _empty_df()

    return pd.concat(all_frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Geo merge + public entry point
# ═══════════════════════════════════════════════════════════════════════

def _merge_geo_coords(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge latitude/longitude from geo lookup onto a token DataFrame."""
    coords_path = data_dir / "geo" / "location_coords.parquet"
    if coords_path.exists():
        coords = pd.read_parquet(coords_path)
        # Demographic locations are commune codes; BV coords file has BV keys.
        # Build a commune→coords lookup from BV coords (take first BV per commune).
        if "location" in coords.columns:
            # Extract commune code from BV location key  (format: "commune_bv")
            commune_coords = coords[["location", "latitude", "longitude"]].copy()
            commune_coords["commune"] = commune_coords["location"].str.split("_").str[0]
            commune_coords = commune_coords.drop_duplicates("commune", keep="first")
            # Build lookup dict to avoid merge column collisions
            lat_map = commune_coords.set_index("commune")["latitude"]
            lon_map = commune_coords.set_index("commune")["longitude"]
            df["latitude"] = df["location"].map(lat_map).astype(np.float32)
            df["longitude"] = df["location"].map(lon_map).astype(np.float32)
    if "latitude" not in df.columns:
        df["latitude"] = np.float32(np.nan)
    if "longitude" not in df.columns:
        df["longitude"] = np.float32(np.nan)
    df["latitude"] = df["latitude"].fillna(46.2276).astype(np.float32)
    df["longitude"] = df["longitude"].fillna(2.2137).astype(np.float32)
    return df


def load_demographic_tokens(data_dir: Path) -> pd.DataFrame:
    """Load all Census demographic data as universal tokens.

    Scans all vintage directories under data/demographics/census/.
    Each token carries ``availability_date`` (publication date) so
    the router can enforce temporal causality.
    """
    demo_dir = data_dir / "demographics"
    if not demo_dir.exists():
        print("  No demographics directory found, skipping.", flush=True)
        return _empty_df()

    print("  Loading multi-vintage Census data...", flush=True)
    census_df = _load_all_census(demo_dir)

    if len(census_df) == 0:
        print("  No demographic data loaded.", flush=True)
        return _empty_df()

    combined = _merge_geo_coords(census_df, data_dir)
    combined.sort_values("availability_date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    n_indicators = combined["candidate"].nunique()
    n_communes = combined["location"].nunique()
    n_vintages = combined.groupby("candidate")["date_float"].nunique().max()
    print(f"  ✓ Loaded {len(combined):,} demographic tokens "
          f"({n_indicators} indicators × {n_communes:,} communes × ~{n_vintages} vintages)",
          flush=True)

    return combined
