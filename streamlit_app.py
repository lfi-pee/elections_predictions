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
# On laisse toutefois respirer un mince encart de prise en main au-dessus de l'iframe.
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], #MainMenu, footer { display: none !important; }
      [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
      [data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
      .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
      [data-testid="stIFrame"] { width: 100% !important; }
      html, body { margin: 0; padding: 0; }
      /* Encart de prise en main : discret, jamais au détriment de la carte. */
      [data-testid="stExpander"] { margin: 6px 12px 0 12px; border: none; }
      [data-testid="stExpander"] summary { font-size: 0.9rem; font-weight: 600; }
      [data-testid="stExpander"] p { font-size: 0.88rem; line-height: 1.5; margin: 0.25rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Prise en main — replié par défaut : n'empiète pas sur la carte, s'ouvre au clic.
with st.expander("ℹ️  Prise en main en 30 secondes — à quoi sert cet outil, et comment le lire"):
    st.markdown(
        """
**Ce que montre cet outil.** Pour chacun des **69 358 bureaux de vote**, la prévision du
1ᵉʳ tour des législatives 2024 — et surtout : **où se trouvent les abstentionnistes de
gauche à aller mobiliser** en fin de campagne.

**Se repérer en 15 secondes :**

- 🔎 **Cherchez une commune** (barre de recherche) ou **zoomez** → la carte descend jusqu'au bureau de vote.
- 🖱️ **Survolez** un bureau = lecture rapide · **cliquez** = fiche détaillée du bureau.
- 🎨 **Couleur par défaut** = gisement à mobiliser (pâle = peu, saturé = beaucoup d'électeurs de gauche à ramener).
- ☑️ Cochez **« bloc en tête »** = le duel sondage national *vs* notre carte, bureau par bureau.
- ⬇️ **Faites défiler** → la méthode, le plan de déploiement (où tracter), la courbe de mobilisation, la comparaison aux sondages.
        """
    )

components.iframe(SITE_URL, height=EMBED_HEIGHT, scrolling=True)
