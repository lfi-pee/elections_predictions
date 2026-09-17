"""Sondages 1er tour des présidentielles PASSÉES (2012, 2017) → CSV « tidy », même schéma que
`scrape_pres_2027` — pour mesurer la décote candidat → parti à HORIZON ÉGAL (`lfi_pres_discount`).

Les fichiers Wikipédia déjà présents dans data/polls/presidentielle/{2012,2017}/ ne couvrent que
les derniers mois (janvier → avril) : impossible d'y lire ce que Mélenchon pesait dans la gauche
sept ou neuf mois avant le scrutin, là où se trouve la prévision 2027 aujourd'hui. Cette page
Wikipédia « Liste de sondages sur l'élection présidentielle française de YYYY » remonte, elle, à
l'année précédant le scrutin (et au-delà).

Particularités gérées : mois et année dans les titres de section (2012 : « Avril 2012 »,
« Quatrième trimestre 2011 ») ou dans les légendes de tableau (2017 : « Janvier 2017 »,
« Sondages de 2015 ») ; cellules de date sans mois (« du 30 au 31 ») ; valeurs « 27 % »,
« <0,5 % », « 24 % Aubry » (hypothèse par ligne, le nom après la valeur) ; lignes-événements
(toutes les cellules identiques) ; « Résultats », « Arrêt de publication ».

Sortie : data/polls/presidentielle/YYYY/presidentielle_YYYY_t1_tidy.csv

    python3 -u -m src.scrape_pres_history 2012 2017            # télécharge
    python3 -u -m src.scrape_pres_history 2012=page2012.html   # depuis un HTML local
"""

from __future__ import annotations

import datetime as dt
import html as _html
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

from src.cross_type_ridge import _poll_token_to_block
from src.scrape_pres_2027 import BLOC, _download

URL = "https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle_fran%C3%A7aise_de_{year}"
OUT = "data/polls/presidentielle/{year}/presidentielle_{year}_t1_tidy.csv"

MONTHS = {"janvier": 1, "janv": 1, "février": 2, "fevrier": 2, "févr": 2, "fév": 2, "mars": 3,
          "avril": 4, "avr": 4, "mai": 5, "juin": 6, "juillet": 7, "juil": 7, "août": 8, "aout": 8,
          "septembre": 9, "sept": 9, "octobre": 10, "oct": 10, "novembre": 11, "nov": 11,
          "décembre": 12, "decembre": 12, "déc": 12}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
META = {"sondeur", "date", "dates", "échantillon", "abstention", "indécis", "autres", "abstention, blancs et nuls"}
SKIP_INST = ("Sondeur", "Résultats", "Arrêt de publ", "nan")


def _context(page: str) -> list[tuple[int, str]]:
    """(position, texte) des titres h2–h4 et des légendes <caption>, dans l'ordre du document."""
    out = []
    for m in re.finditer(r"<h([2-4])[^>]*>(.*?)</h\1>", page, re.S):
        out.append((m.start(), _html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()))
    for m in re.finditer(r"<caption[^>]*>(.*?)</caption>", page, re.S):
        out.append((m.start(), _html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()))
    return sorted(out)


def _month_year(texts: list[str]) -> tuple[int | None, int | None]:
    """Année (et mois s'il y est) lus dans les titres/légendes qui précèdent un tableau : le plus
    proche qui porte une année l'emporte ; le mois n'est retenu que s'il vient du même texte."""
    for txt in reversed(texts):
        y = re.search(r"(20\d\d)", txt)
        if y:
            m = re.search(rf"\b({_MONTH_RE})\b", txt.lower())
            return (MONTHS[m.group(1)] if m else None), int(y.group(1))
    return None, None


def _end_date(cell: str, month_ctx: int | None, year_ctx: int | None) -> dt.date | None:
    s = re.sub(r"\[.*?\]", "", str(cell)).lower().replace("\xa0", " ").replace("–", "-").replace("—", "-")
    s = s.replace("1er", "1")
    y = re.search(r"(20\d\d)", s)
    year = int(y.group(1)) if y else year_ctx
    ms = list(re.finditer(rf"({_MONTH_RE})\.?", s))
    if ms:
        month = MONTHS[ms[-1].group(1)]
        days = re.findall(r"(\d{1,2})\s*$", s[:ms[-1].start()].strip())
    else:
        month = month_ctx
        days = re.findall(r"(\d{1,2})", s)
        days = days[-1:] if days else []
    if not (year and month and days):
        return None
    try:
        return dt.date(year, month, int(days[0]))
    except ValueError:
        return None


def _value_name(cell) -> tuple[float | None, str | None]:
    """« 24 % Aubry » → (24, AUBRY) ; « <0,5 % » → (0.25, None) ; « – » → (None, None)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None, None
    if isinstance(cell, (int, float)):
        return float(cell), None
    s = re.sub(r"\[.*?\]", "", str(cell)).replace("\xa0", " ").strip()
    if s in ("", "—", "-", "–", "nan"):
        return None, None
    m = re.match(r"^(<)?\s*(\d+(?:[.,]\d+)?)\s*%?\s*([^\d%]*)", s)
    if not m:
        return None, None
    v = float(m.group(2).replace(",", "."))
    if m.group(1):
        v /= 2  # « <0,5 % » → 0,25
    name = m.group(3).strip().upper() or None
    return v, name


def _header(col: str) -> tuple[str | None, str | None]:
    h = re.sub(r"\[.*?\]", "", str(col)).strip()
    m = re.match(r"^(.*?)\s*\(([^)]+)\)", h)
    if m:
        return m.group(1).strip().upper() or None, m.group(2).strip()
    m = re.match(r"^candidat[e·]*\s+(.+)$", h, re.I)
    if m:
        return None, re.split(r"[/\s]", m.group(1).strip())[0]
    return h.upper() or None, None


def tidy(page: str) -> pd.DataFrame:
    ctx = _context(page)
    tables = [m.start() for m in re.finditer(r'<table[^>]*class="[^"]*wikitable', page)]
    frames = pd.read_html(StringIO(page), attrs={"class": "wikitable"}, decimal=",", thousands="\xa0")
    rows = []
    for pos, t in zip(tables, frames):
        texts = [txt for p, txt in ctx if p < pos]
        # Section : seulement les tableaux du 1er tour (titre h2 le plus récent).
        h2 = [txt for p, txt in ctx if p < pos and re.search(r"<h2", page[p:p + 4])]
        if not h2 or "premier tour" not in h2[-1].lower():
            continue
        month_ctx, year_ctx = _month_year(texts[-3:])
        t = t.copy()
        t.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in t.columns]
        names, seen = [], {}
        for c in t.columns:
            seen[c] = seen.get(c, 0) + 1
            names.append(c if seen[c] == 1 else f"{c} #{seen[c]}")
        t.columns = names
        cand_cols = [c for c in t.columns if re.sub(r"\[.*?\]", "", c).lower().strip() not in META
                     and not c.startswith("Unnamed")]
        date_col = next((c for c in t.columns if c.lower().startswith("date")), None)
        if date_col is None or len(cand_cols) < 4:
            continue
        for k, (_, r) in enumerate(t.iterrows()):
            inst = re.sub(r"\[.*?\]", "", str(r.iloc[0])).strip()
            vals = [str(v) for v in r.tolist()]
            if any(inst.startswith(x) for x in SKIP_INST) or len(set(vals)) == 1:
                continue
            d = _end_date(r[date_col], month_ctx, year_ctx)
            if d is None:
                continue
            ech, _ = _value_name(r.get("Échantillon"))
            for c in cand_cols:
                v, cname = _value_name(r[c])
                if v is None:
                    continue
                name, party = _header(re.sub(r" #\d+$", "", c))
                name = cname or name
                if not name and not party:
                    continue
                bloc = BLOC[_poll_token_to_block(party or "", name or "")]
                rows.append({"institut": inst, "date_fin": d.isoformat(), "echantillon": int(ech) if ech else None,
                             "hypothese": k, "candidat": name, "parti": party, "bloc": bloc, "valeur": v})
    df = pd.DataFrame(rows)
    return df.sort_values(["date_fin", "institut", "hypothese"], ascending=[False, True, True]).reset_index(drop=True)


def main(argv: list[str]) -> None:
    for arg in argv[1:]:
        year, _, local = arg.partition("=")
        page = Path(local).read_text() if local else _download(URL.format(year=year))
        df = tidy(page)
        out = Path(OUT.format(year=year))
        out.parent.mkdir(parents=True, exist_ok=True)
        head = (f"# Sondages 1er tour présidentielle {year} — {URL.format(year=year)}\n"
                f"# Relevé le {dt.date.today().isoformat()} par src/scrape_pres_history.py ; une ligne par candidat·e et hypothèse.\n"
                f"# bloc : nomenclature du modèle (G gauche, CD centre+droite, ED extrême droite, AU hors axe).\n")
        out.write_text(head + df.to_csv(index=False))
        n_polls = df[["institut", "date_fin"]].drop_duplicates().shape[0]
        au = df[df.bloc == "AU"][["candidat", "parti"]].drop_duplicates()
        print(f"{year}: {len(df)} lignes, {n_polls} sondages, {df.date_fin.min()} → {df.date_fin.max()} → {out}")
        print(f"   hors axe (AU) : {au.to_dict('records') if len(au) else 'aucun'}")


if __name__ == "__main__":
    main(sys.argv)
