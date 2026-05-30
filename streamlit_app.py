"""Enveloppe Streamlit du site « L'élection au bureau de vote près ».

Le livrable reste le site statique de `report_app/` (MapLibre + JS, aucune
dépendance serveur). Streamlit n'ajoute qu'un hôte : il sert `report_app/` via
son partage de fichiers statiques (symlink `static/ -> report_app/`, exposé sous
`app/static/`) et l'affiche plein cadre dans une iframe. On gagne ainsi
l'hébergement et l'accès protégé de Streamlit Cloud sans rien réécrire du site.

    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

SITE_URL = "app/static/index.html"
EMBED_HEIGHT = 5200  # hauteur de la descente d'écran (hero + ~12 panneaux + méthode)

st.set_page_config(
    page_title="L'élection au bureau de vote près",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Efface le chrome Streamlit : le médium doit être le site, pas l'application hôte.
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], #MainMenu, footer { display: none !important; }
      [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
      [data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
      .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
      [data-testid="stIFrame"] { width: 100% !important; }
      html, body { margin: 0; padding: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

components.iframe(SITE_URL, height=EMBED_HEIGHT, scrolling=True)
