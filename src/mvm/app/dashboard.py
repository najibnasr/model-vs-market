"""Dashboard « The model vs the bookmakers ».

Lancement : make app   (ou : PYTHONPATH=src streamlit run src/mvm/app/dashboard.py)
Pré-requis : make data, make predict, make backtest, make test-backtest ; make live chaque semaine
pour la saison en cours. Variable d'environnement optionnelle MVM_CONFIG (autre config).

Pages grand public (cette semaine, la saison en direct, ton club, l'historique) et « Backstage »
avec les résultats techniques, pour les entretiens. Textes affichés en anglais.
"""

from __future__ import annotations

import streamlit as st

from mvm.app import data, fan, lab, ui

st.set_page_config(
    page_title="The model vs the bookmakers",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject_css()

live = ui.season_label(data.live_season())
PAGES = {
    "This week": fan.page_week,
    f"Season {live}": fan.page_season,
    "Your club": fan.page_club,
    "Track record": fan.page_track,
    "Backstage": lab.page_lab,
}

ui.brand(f"12 European football leagues, live through the {live} season")
with st.container(key="nav"):
    page = st.radio("Page", list(PAGES), horizontal=True, label_visibility="collapsed", key="page")
PAGES[page]()
