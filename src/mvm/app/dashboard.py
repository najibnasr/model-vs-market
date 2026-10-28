"""Dashboard « Le modèle contre les bookmakers ».

Lancement : make app   (ou : PYTHONPATH=src streamlit run src/mvm/app/dashboard.py)
Pré-requis : make data, make predict, make backtest (et make test-backtest pour la période de test).
Variable d'environnement optionnelle MVM_CONFIG pour pointer vers une autre config.

Quatre pages grand public (le duel, la journée, ton club, et si tu avais parié ?) et une page
« Coulisses » avec les résultats techniques, pour les entretiens.
"""

from __future__ import annotations

import streamlit as st

from mvm.app import fan, lab, ui

st.set_page_config(
    page_title="Le modèle contre les bookmakers",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject_css()

PAGES = {
    "Le duel": fan.page_duel,
    "La journée": fan.page_matchday,
    "Ton club": fan.page_club,
    "Et si tu avais parié ?": fan.page_whatif,
    "Coulisses": lab.page_lab,
}

ui.brand("12 championnats européens, 2012 à 2026")
with st.container(key="nav"):
    page = st.radio("Page", list(PAGES), horizontal=True, label_visibility="collapsed", key="page")
PAGES[page]()
