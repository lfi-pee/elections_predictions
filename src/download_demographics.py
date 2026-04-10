#!/usr/bin/env python3
"""Download all available INSEE Census demographic data.

Downloads ZIP files from INSEE, extracts CSVs, and organises them into:
  data/demographics/census/{vintage}/

Strategy: For each known stats page ID, scrape the page HTML for download
links (href containing 'fichier') and download ZIPs matching our patterns.

Run: python3 -m src.download_demographics [--data-dir data/]
"""
from __future__ import annotations

import argparse
import io
import os
import re
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
BASE = "https://www.insee.fr"

# ── Census: known stats page IDs per vintage per category ─────────────
# Format: {vintage: {"pop": [page_ids], "dipl": [page_ids], "emp": [page_ids]}}
# Multiple IDs are tried in order; the page is scraped for ZIP links.

# Modern vintages (2014+): separate page IDs per category
CENSUS_PAGES: dict[int, dict[str, list[int]]] = {
    2022: {"pop": [8582452], "dipl": [8581488], "emp": [8581444],
           "log": [8581474], "fam": [8582452]},
    2021: {"pop": [8201904], "dipl": [8202319], "emp": [8202916],
           "log": [8202349], "fam": [8205182]},
    2020: {"pop": [7632446], "dipl": [7631070], "emp": [7632867],
           "log": [7631186], "fam": [7633206]},
    2019: {"pop": [6456153], "dipl": [6454124], "emp": [6454652],
           "log": [6454155], "fam": [6454116]},
    2018: {"pop": [5395875], "dipl": [5395831], "emp": [5395838],
           "log": [5395856], "fam": [5395819]},
    2017: {"pop": [4515565], "dipl": [4516086], "emp": [4515500],
           "log": [4515532], "fam": [4515503]},
    2016: {"pop": [4171334], "dipl": [4171395], "emp": [4171446],
           "log": [4171415], "fam": [4171359]},
    2015: {"pop": [3564100], "dipl": [3564182], "emp": [3564231],
           "log": [3564300], "fam": [3565598]},
    2014: {"pop": [2862200], "dipl": [2862015], "emp": [2862207],
           "log": [2862034], "fam": [2862009]},
}

# Older vintages (2006-2013): page IDs from the archive system
# Some pages bundle multiple categories together
CENSUS_PAGES.update({
    2013: {"pop": [2044751], "dipl": [2044692], "emp": [2044661],
           "log": [2044711], "fam": [2044615]},
    2012: {"pop": [2044748], "dipl": [2044707], "emp": [2128672],
           "log": [2044713], "fam": [2044618]},
    2011: {"pop": [2044754], "dipl": [2044710], "emp": [2044677],
           "log": [2044715], "fam": [2044612]},
    2010: {"pop": [2044743], "dipl": [2044702], "emp": [2044658],
           "log": [2044717], "fam": [2044610]},
    2009: {"pop": [2044733], "dipl": [2044687], "emp": [2044654],
           "log": [2044719], "fam": [2044608]},
    2008: {"pop": [2044714], "dipl": [2044671], "emp": [2044668],
           "log": [2044721], "fam": [2044606]},
    2007: {"pop": [2044719], "dipl": [2044675], "emp": [2044666],
           "log": [2044723], "fam": [2044603]},
    2006: {"pop": [2044723], "dipl": [2044678], "emp": [2044664],
           "log": [2044725], "fam": [2044600]},
})

# Keywords to match relevant ZIP files per category
CENSUS_ZIP_KEYWORDS = {
    "pop": ["evol-struct-pop", "evol_struct_pop", "pop-age", "pop_age",
            "str-pop", "struct-pop"],
    "dipl": ["diplomes-formation", "diplomes_formation", "dipl", "for-"],
    "emp": ["emploi-pop-active", "emploi_pop_active", "activite-residents",
            "activite_residents", "act-", "pop-act"],
    "log": ["logement", "log-"],
    "fam": ["coupl-fam-men", "coupl_fam_men", "couples-familles",
            "couples_familles", "fam-men"],
}



def _fetch(url: str, max_retries: int = 3) -> bytes | None:
    """Fetch URL with retries, return bytes or None."""
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=60) as resp:
                return resp.read()
        except HTTPError as e:
            if e.code == 404:
                return None
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return None
        except (URLError, TimeoutError):
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return None
    return None


def _scrape_zip_links(page_id: int) -> list[str]:
    """Scrape a statistics page for all ZIP download links."""
    url = f"{BASE}/fr/statistiques/{page_id}"
    html = _fetch(url)
    if not html:
        return []
    text = html.decode("utf-8", errors="replace")
    # Find all href links containing 'fichier' and '.zip'
    links = re.findall(r'href="([^"]*fichier[^"]*\.zip[^"]*)"', text, re.IGNORECASE)
    # Also look for direct download links
    links += re.findall(r'href="([^"]*\.zip)"', text, re.IGNORECASE)
    # Normalize: make absolute
    result = []
    seen = set()
    for link in links:
        if link.startswith("/"):
            link = BASE + link
        elif not link.startswith("http"):
            continue
        if link not in seen:
            seen.add(link)
            result.append(link)
    return result


def _extract_zip(data: bytes, dest_dir: Path) -> list[str]:
    """Extract ZIP bytes into dest_dir, return extracted filenames."""
    extracted = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if not basename or basename.startswith("."):
                    continue
                ext = basename.lower().rsplit(".", 1)[-1] if "." in basename else ""
                if ext not in ("csv", "xls", "xlsx", "txt"):
                    continue
                dest_path = dest_dir / basename
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(basename)
    except zipfile.BadZipFile:
        pass
    return extracted


def _match_zip(url: str, keywords: list[str], vintage: int) -> bool:
    """Check if a ZIP URL matches the category keywords and vintage."""
    lower = url.lower()
    yr2 = str(vintage)[-2:]
    yr4 = str(vintage)

    # Must contain a keyword for the category
    has_keyword = any(kw in lower for kw in keywords)
    # Must reference the vintage year
    has_year = yr4 in lower or f"-{yr2}" in lower or f"_{yr2}" in lower
    # Must be a CSV version if available (prefer _csv suffix)
    return has_keyword and has_year


def download_census(data_dir: Path) -> None:
    """Download all Census vintages."""
    census_dir = data_dir / "demographics" / "census"
    print(f"=== Downloading Census data (vintages {min(CENSUS_PAGES)}–{max(CENSUS_PAGES)}) ===")

    for vintage in sorted(CENSUS_PAGES.keys()):
        vintage_dir = census_dir / str(vintage)
        vintage_dir.mkdir(parents=True, exist_ok=True)

        existing = list(vintage_dir.glob("*.csv")) + list(vintage_dir.glob("*.xls*"))
        # Exclude meta_ files from the count
        existing = [f for f in existing if not f.name.startswith("meta_")]
        if len(existing) >= 5:
            print(f"  {vintage}: {len(existing)} files exist, skipping")
            continue

        cats = CENSUS_PAGES[vintage]
        total_files = 0

        for cat, page_ids in cats.items():
            for page_id in page_ids:
                zip_links = _scrape_zip_links(page_id)
                if not zip_links:
                    print(f"  {vintage} {cat}: no ZIP links on page {page_id}")
                    continue

                # Try to find the best match
                # Prefer CSV versions over XLS
                matched = [u for u in zip_links
                           if _match_zip(u, CENSUS_ZIP_KEYWORDS[cat], vintage)]

                # If no keyword match, try all ZIPs on the page
                if not matched:
                    # For old pages there might be only one ZIP
                    matched = [u for u in zip_links
                               if str(vintage) in u or str(vintage)[-2:] in u.split("/")[-1]]

                # Prefer _csv versions
                csv_matched = [u for u in matched if "_csv" in u.lower()]
                if csv_matched:
                    matched = csv_matched

                if not matched:
                    # Just try the first ZIP on the page
                    matched = zip_links[:1]

                for url in matched[:1]:  # Download first match only
                    data = _fetch(url)
                    if data:
                        files = _extract_zip(data, vintage_dir)
                        total_files += len(files)
                        fname = url.split("/")[-1]
                        print(f"  {vintage} {cat}: {len(files)} files ← {fname}")
                        break

            time.sleep(0.3)

        if total_files == 0:
            print(f"  {vintage}: ⚠ no files downloaded")
        time.sleep(0.2)



def main() -> None:
    parser = argparse.ArgumentParser(description="Download INSEE Census demographic data")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    download_census(args.data_dir)

    # Summary
    census_dir = args.data_dir / "demographics" / "census"

    if census_dir.exists():
        cvs = sorted(d.name for d in census_dir.iterdir() if d.is_dir())
        n = sum(len(list((census_dir / v).glob("*.*"))) for v in cvs)
        print(f"\nCensus: {len(cvs)} vintages ({', '.join(cvs)}), {n} files total")

    total = 0
    if census_dir.exists():
        for f in census_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    print(f"Total disk usage: {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
