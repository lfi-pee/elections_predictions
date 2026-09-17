"""Sondages 1er tour de la présidentielle 2027 → CSV « tidy » (une ligne par candidat·e testé·e).

Source : page Wikipédia FR « Liste de sondages sur l'élection présidentielle française de 2027 »,
section « Sondages concernant le premier tour » (un tableau par période, l'année dans le titre
de section car les cellules de date n'en portent pas). Chaque ligne d'un tableau est une
**hypothèse** (jeu de candidats) testée par un institut à une date ; un même sondage en teste
souvent plusieurs.

Sortie : `data/polls/presidentielle/2027/presidentielle_2027_t1_tidy.csv`
    institut, date_fin (ISO), echantillon, hypothese (rang dans le tableau), candidat, parti,
    bloc (G / CD / ED / AU, nomenclature du modèle via cross_type_ridge), valeur (% exprimés).

C'est la seule source de l'ancre nationale 2027 (`scenarios_2027.anchor_from_polls`) : la
présidentielle précède les législatives de six semaines et, sur tous les scrutins d'apprentissage
(2007→2022), la fenêtre d'un an de l'estimateur validé (`cross_type_ridge._build_national_poll_
features`) ne contenait pratiquement QUE des sondages présidentiels.

    python3 -u -m src.scrape_pres_2027              # télécharge la page et régénère le CSV
    python3 -u -m src.scrape_pres_2027 page.html    # depuis un HTML déjà téléchargé
"""

from __future__ import annotations

import datetime as dt
import html as _html
import re
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

from src.cross_type_ridge import _poll_token_to_block

URL = ("https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle"
       "_fran%C3%A7aise_de_2027")
OUT = Path("data/polls/presidentielle/2027/presidentielle_2027_t1_tidy.csv")

MONTHS = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
          "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
          "décembre": 12, "decembre": 12}
BLOC = {"Gauche": "G", "Centre+Droite": "CD", "Extreme_Droite": "ED", "Other": "AU"}
META = {"sondeur", "date", "dates", "échantillon", "autres"}


def _download(url: str = URL) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "elections_predictions (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def _first_round_tables(page: str) -> list[tuple[int, pd.DataFrame]]:
    """(année, tableau) pour chaque wikitable de la section 1er tour, l'année lue dans le titre
    « Année YYYY » de la sous-section (les cellules de date n'ont pas d'année)."""
    events = []
    for m in re.finditer(r"<h([2-4])[^>]*>(.*?)</h\1>", page, re.S):
        txt = _html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        events.append((m.start(), int(m.group(1)), txt))
    tables = [m.start() for m in re.finditer(r'<table[^>]*class="[^"]*wikitable', page)]
    frames = pd.read_html(StringIO(page), attrs={"class": "wikitable"}, decimal=",", thousands="\xa0")
    out, year, in_t1 = [], None, False
    for pos, frame in zip(tables, frames):
        for epos, lvl, txt in events:
            if epos > pos:
                break
            if lvl == 2:
                in_t1 = txt.startswith("Sondages concernant le premier tour")
                year = None
            elif lvl == 3:
                m = re.search(r"(20\d\d)", txt)
                year = int(m.group(1)) if m else None
        if in_t1 and year:
            out.append((year, frame))
    return out


def _end_date(cell: str, year: int) -> dt.date | None:
    """« 9-10 septembre », « 30 avril-2 mai », « 2 - 5 avril 2024 » → date de FIN de terrain."""
    s = str(cell).lower().replace("\xa0", " ").replace("–", "-").replace("—", "-")
    m_year = re.search(r"(20\d\d)", s)
    if m_year:
        year = int(m_year.group(1))
    months = [(m.start(), MONTHS[m.group(0)]) for m in re.finditer("|".join(MONTHS), s)]
    if not months:
        return None
    pos, month = months[-1]
    days = re.findall(r"(\d{1,2})\s*$", s[:pos].strip())
    if not days:
        return None
    try:
        return dt.date(year, month, int(days[0]))
    except ValueError:
        return None


def _num(cell) -> float | None:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    s = re.sub(r"\[.*?\]", "", str(cell)).strip().replace("\xa0", "")
    if s in ("", "—", "-", "–", "nan"):
        return None
    if s.startswith("<"):
        return 0.5
    m = re.match(r"^(\d+(?:[.,]\d+)?)", s)
    return float(m.group(1).replace(",", ".")) if m else None


def _header(col: str) -> tuple[str | None, str | None]:
    """« Glucksmann[c] (PP)[d] » → (GLUCKSMANN, PP) ; « Candidat PS / PP » → (None, PS)."""
    h = re.sub(r"\[.*?\]", "", str(col)).strip()
    m = re.match(r"^(.*?)\s*\(([^)]+)\)", h)
    if m:
        return m.group(1).strip().upper() or None, m.group(2).strip()
    m = re.match(r"^candidat[e·]*\s+(.+)$", h, re.I)
    if m:
        return None, re.split(r"[/\s]", m.group(1).strip())[0]
    return h.upper() or None, None


def _cell_name(cell) -> tuple[str | None, str | None]:
    """« 7 Hollande (PS) » → (HOLLANDE, PS) ; « 36 Bardella » → (BARDELLA, None)."""
    if not isinstance(cell, str):
        return None, None
    s = re.sub(r"\[.*?\]", "", cell.replace("\xa0", " "))
    s = re.sub(r"^[<\d.,\s]+", "", s).strip()
    if not s:
        return None, None
    m = re.match(r"^([^(]+?)\s*\(([^)]+)\)", s)
    if m:
        return m.group(1).strip().upper(), m.group(2).strip()
    return s.upper(), None  # nom complet (« Le Pen », « Bardella »)


def tidy(page: str) -> pd.DataFrame:
    rows = []
    for year, t in _first_round_tables(page):
        t = t.copy()
        t.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in t.columns]
        # Colonnes dupliquées par un colspan d'en-tête (mêmes nom et valeurs) → une seule.
        keep, prev = [], None
        for i, c in enumerate(t.columns):
            if prev is not None and c == t.columns[prev] and t.iloc[:, i].equals(t.iloc[:, prev]):
                continue
            keep.append(i)
            prev = i
        t = t.iloc[:, keep]
        # Noms encore dupliqués (colspan aux valeurs non identiques) → suffixés, sinon `r[c]`
        # renverrait une Series ; la répétition par ligne est traitée par `seen_text` ci-dessous.
        names, seen = [], {}
        for c in t.columns:
            seen[c] = seen.get(c, 0) + 1
            names.append(c if seen[c] == 1 else f"{c} #{seen[c]}")
        t.columns = names
        cand_cols = [c for c in t.columns if c.lower().split("[")[0].strip() not in META]
        date_col = next(c for c in t.columns if c.lower().startswith("date"))
        for k, (_, r) in enumerate(t.iterrows()):
            inst = str(r.iloc[0]).strip()
            vals = [str(v) for v in r.tolist()]
            if inst in ("Sondeur", "Résultats", "nan") or len(set(vals)) == 1:
                continue
            d = _end_date(r[date_col], year)
            if d is None:
                continue
            ech = _num(r.get("Échantillon"))
            # Hypothèse d'union : la cellule d'un·e candidat·e unique s'étend (colspan) sur
            # plusieurs colonnes de parti → pandas la répète ; on ne la compte qu'une fois.
            seen_text: set[str] = set()
            for c in cand_cols:
                raw = r[c]
                if isinstance(raw, str) and re.search(r"[A-Za-zÀ-ÿ]", raw):
                    if raw in seen_text:
                        continue
                    seen_text.add(raw)
                v = _num(raw)
                if v is None:
                    continue
                name, party = _header(re.sub(r" #\d+$", "", c))
                cname, cparty = _cell_name(r[c])
                name = cname or name
                party = cparty or party
                if not name and not party:
                    continue
                bloc = BLOC[_poll_token_to_block(party or "", name or "")]
                rows.append({"institut": re.sub(r"\[.*?\]", "", inst).strip(), "date_fin": d.isoformat(),
                             "echantillon": int(ech) if ech else None, "hypothese": k,
                             "candidat": name, "parti": party, "bloc": bloc, "valeur": v})
    df = pd.DataFrame(rows)
    return df.sort_values(["date_fin", "institut", "hypothese"], ascending=[False, True, True]).reset_index(drop=True)


def main(argv: list[str]) -> None:
    page = Path(argv[1]).read_text() if len(argv) > 1 else _download()
    df = tidy(page)
    unmapped = df[df.bloc == "AU"][["candidat", "parti"]].drop_duplicates()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = (f"# Sondages 1er tour présidentielle 2027 — {URL}\n"
            f"# Relevé le {dt.date.today().isoformat()} par src/scrape_pres_2027.py ; une ligne par candidat·e et hypothèse.\n"
            f"# bloc : nomenclature du modèle (G gauche, CD centre+droite, ED extrême droite, AU hors axe).\n")
    OUT.write_text(head + df.to_csv(index=False))
    n_polls = df[["institut", "date_fin"]].drop_duplicates().shape[0]
    print(f"{len(df)} lignes, {n_polls} sondages (institut × date), {df.date_fin.min()} → {df.date_fin.max()} → {OUT}")
    print("hors axe (AU) :", unmapped.to_dict("records") if len(unmapped) else "aucun")


if __name__ == "__main__":
    main(sys.argv)
