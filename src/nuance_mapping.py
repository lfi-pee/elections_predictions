from __future__ import annotations

CITY_TO_CODE_COMMUNE: dict[str, str] = {
    "Paris": "75056",
    "Marseille": "13055",
    "Lyon": "69123",
    "Toulouse": "31555",
    "Nice": "06088",
    "Nantes": "44109",
    "Montpellier": "34172",
    "Strasbourg": "67482",
    "Bordeaux": "33063",
    "Lille": "59350",
    "Rennes": "35238",
    "Reims": "51454",
    "Saint Etienne": "42218",
    "Grenoble": "38185",
    "Dijon": "21231",
    "Angers": "49007",
    "Clermont Ferrand": "63113",
    "Brest": "29019",
    "Limoges": "87085",
    "Amiens": "80021",
    "Besancon": "25056",
    "Le Havre": "76351",
    "Tours": "37261",
    "Mulhouse": "68224",
    "Perpignan": "66136",
    "Rouen": "76540",
    "Toulon": "83137",
}

PARTY_TO_NUANCE: dict[str, str] = {
    "RN": "LRN",
    "UDR": "LRN",
    "LR": "LLR",
    "DVD": "LDVD",
    "RE": "LREM",
    "ENS": "LREM",
    "MODEM": "LMDM",
    "MoDem": "LMDM",
    "HOR": "LHOR",
    "PS": "LDVG",
    "DVG": "LDVG",
    "LFI": "LFI",
    "PCF": "LCOM",
    "LÉ": "LECO",
    "EELV": "LECO",
    "REC": "LREC",
    "LO": "LEXG",
    "EXG": "LEXG",
    "NPA": "LEXG",
    "NPA-A": "LEXG",
    "NPA-B": "LEXG",
    "DVC": "LDVC",
    "CC": "LDVC",
    "UDI": "LUDI",
    "LC-LNC": "LNC",
    "EXD": "LEXD",
    "DIV": "LDIV",
    "PRG": "LRDG",
    "ECO": "LECO",
    "DA": "LDIV",
    "EQX": "LECO",
    "LC": "LNC",
    "LNC": "LNC",
    "SE": "LDIV",
}

# Nuance equivalence groups: poll party codes that can map to coalition-level
# result nuance codes in municipales elections.
NUANCE_EQUIVALENCES: dict[str, set[str]] = {
    # Right-wing coalition codes
    "LUD": {"LLR", "LREM", "LMDM", "LHOR", "LUDI", "LDVD", "LNC"},
    "LMAJ": {"LLR", "LREM", "LMDM", "LHOR", "LUDI", "LDVD", "LNC"},
    "LUXD": {"LRN", "LREC", "LEXD"},
    # Left-wing coalition codes
    "LUG": {"LDVG", "LECO", "LFI", "LCOM", "LRDG"},
    "LVEC": {"LECO", "LDVG"},
    "LUC": {"LDVG", "LFI", "LCOM", "LECO", "LEXG"},
    "LFG": {"LFI", "LCOM", "LEXG"},
}


def expand_nuance_group(nuance: str) -> set[str]:
    """Return the set of poll-level nuances that could match a result nuance."""
    result = {nuance}
    if nuance in NUANCE_EQUIVALENCES:
        result |= NUANCE_EQUIVALENCES[nuance]
    return result


def map_coalition_to_nuance(coalition: str) -> str:
    """Map a poll coalition string to a primary nuance code.

    e.g. 'PS - PCF - LÉ' -> 'LDVG' (based on primary party PS)
    """
    if not coalition or coalition == "nan":
        return "UNKNOWN"

    # Take the first party in the coalition as the primary
    primary = coalition.split("-")[0].strip()

    # Handle specific common names
    if "LÉ" in primary:
        primary = "LÉ"
    if "MODEM" in primary:
        primary = "MODEM"

    return PARTY_TO_NUANCE.get(primary, "UNKNOWN")
