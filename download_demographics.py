"""Download INSEE demographic data for the election prediction model.

Downloads latest vintage of each source:
  - Census 2021: Activity, Education, Population (commune-level)
  - BPE 2024: Equipment density (commune-level)

Files are saved to data/demographics/{census,bpe}/.

Usage:
    python download_demographics.py

If automatic downloads fail, the script prints manual download URLs.
"""
from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests

DATA_DIR = Path("data")
DEMO_DIR = DATA_DIR / "demographics"

# INSEE bulk download URLs for commune-level "bases communales comparatives"
# These are the 2021 vintage endpoints. If URLs change, update here or
# download manually from https://www.insee.fr/fr/statistiques?categorie=4
DOWNLOADS: list[dict[str, str]] = [
    {
        "name": "Census Activity (ACT) — unemployment, CSP",
        "url": "https://www.insee.fr/fr/statistiques/fichier/7632513/base-cc-activite-residents-2021_csv.zip",
        "dest_dir": "census",
        "fallback_page": "https://www.insee.fr/fr/statistiques/7632513",
    },
    {
        "name": "Census Education (FOR) — diplomas",
        "url": "https://www.insee.fr/fr/statistiques/fichier/7632529/base-cc-diplomes-formation-2021_csv.zip",
        "dest_dir": "census",
        "fallback_page": "https://www.insee.fr/fr/statistiques/7632529",
    },
    {
        "name": "Census Population (POP) — age structure, immigration",
        "url": "https://www.insee.fr/fr/statistiques/fichier/7632446/base-cc-evol-struct-pop-2021_csv.zip",
        "dest_dir": "census",
        "fallback_page": "https://www.insee.fr/fr/statistiques/7632446",
    },
    {
        "name": "BPE 2024 — equipment counts (health, commerce, services)",
        "url": "https://www.insee.fr/fr/statistiques/fichier/8217527/DS_BPE_CSV_FR.zip",
        "dest_dir": "bpe",
        "fallback_page": "https://www.insee.fr/fr/statistiques/8217527",
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (election-prediction-research) Python/requests",
}


def _download_and_extract(entry: dict[str, str]) -> bool:
    """Download a ZIP from INSEE and extract CSV/XLSX files."""
    dest = DEMO_DIR / entry["dest_dir"]
    dest.mkdir(parents=True, exist_ok=True)

    # Check if we already have files
    existing = list(dest.glob("*.csv")) + list(dest.glob("*.xlsx"))
    if existing:
        print(f"  ✓ {entry['name']}: already have {len(existing)} file(s), skipping")
        return True

    print(f"  ↓ Downloading: {entry['name']}...")
    try:
        resp = requests.get(entry["url"], headers=HEADERS, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ Download failed: {e}")
        print(f"    → Manual download: {entry['fallback_page']}")
        print(f"    → Place CSV/XLSX files in: {dest}/")
        return False

    content_type = resp.headers.get("Content-Type", "")

    if "zip" in content_type or entry["url"].endswith(".zip"):
        try:
            zf = zipfile.ZipFile(BytesIO(resp.content))
            # Extract only CSV/XLSX files
            extracted = 0
            for name in zf.namelist():
                if name.endswith((".csv", ".xlsx", ".xls")) and not name.startswith("__"):
                    zf.extract(name, dest)
                    extracted += 1
                    print(f"    Extracted: {name}")
            if extracted == 0:
                # Extract everything
                zf.extractall(dest)
                print(f"    Extracted all {len(zf.namelist())} files")
            return True
        except zipfile.BadZipFile:
            # Not actually a ZIP — save as-is
            pass

    # Save raw file
    suffix = ".csv" if "csv" in content_type else ".xlsx"
    fname = entry["url"].split("/")[-1].replace(".zip", suffix)
    out_path = dest / fname
    out_path.write_bytes(resp.content)
    print(f"    Saved: {out_path}")
    return True


def verify_files() -> None:
    """Check what demographic files are available."""
    print("\n--- Verification ---")
    for subdir in ["census", "bpe"]:
        d = DEMO_DIR / subdir
        if not d.exists():
            print(f"  {subdir}/: MISSING")
            continue
        files = sorted(d.rglob("*"))
        data_files = [f for f in files if f.suffix in (".csv", ".xlsx", ".xls")]
        print(f"  {subdir}/: {len(data_files)} data file(s)")
        for f in data_files:
            print(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    print("=== INSEE Demographic Data Download ===\n")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    successes = 0
    for entry in DOWNLOADS:
        if _download_and_extract(entry):
            successes += 1

    print(f"\n{successes}/{len(DOWNLOADS)} downloads completed.")

    if successes < len(DOWNLOADS):
        print("\nFor failed downloads, visit the fallback pages above and")
        print("download the commune-level CSV or Excel file manually.")
        print(f"Place files in the appropriate subdirectory under {DEMO_DIR}/")

    verify_files()


if __name__ == "__main__":
    main()
