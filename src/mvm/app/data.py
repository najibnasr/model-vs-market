"""Chargement et préparation des données du dashboard (mis en cache par Streamlit)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mvm.core import backtest, metrics
from mvm.core.io import load_config, read_table
from mvm.core.market import market_probabilities, odds_matrix

OUTCOMES = ("H", "D", "A")
PCOLS = [f"p_{o}" for o in OUTCOMES]
MODEL_NAMES = {
    "ens": "Combinaison Elo + DC",
    "elo": "Elo",
    "dc": "Dixon-Coles",
    "base": "Fréquences de base",
}
# Écart minimal (en probabilité sur le bon résultat) pour désigner un vainqueur du duel.
TIE = 0.01


@st.cache_resource
def config() -> dict:
    return load_config(os.environ.get("MVM_CONFIG", "configs/foot.yaml"))


def league_name(code: str) -> str:
    return config()["leagues"].get(code, {}).get("name", code)


def split_range(split: str) -> tuple[str, str]:
    ev = config()["evaluation"]
    return {
        "tuning": (ev["eval_first"], ev["tuning_last"]),
        "test": (ev["test_first"], ev["test_last"]),
    }[split]


@st.cache_data(show_spinner="Chargement des matchs et des cotes…")
def load_data() -> dict[str, pd.DataFrame]:
    cfg = config()
    proc = Path(cfg["paths"]["processed"])
    events = read_table(proc / "events.parquet")
    odds = read_table(proc / "odds.parquet")
    preds = read_table(proc / "predictions.parquet")
    tpath = proc / "teams.parquet"
    if tpath.exists():
        names = read_table(tpath).drop_duplicates("team_id").set_index("team_id")["canonical_name"]
    else:
        names = pd.Series(dtype="string")
    ev = events.assign(
        home=events.home_team_id.map(names).fillna(events.home_team_id).astype(str),
        away=events.away_team_id.map(names).fillna(events.away_team_id).astype(str),
    )
    ref = cfg["evaluation"]["reference"]
    close = market_probabilities(odds, ref["bookmaker"], ref["timing"], ref["margin_method"])
    close = close[PCOLS].add_prefix("mkt_")
    pre = odds_matrix(odds, cfg["betting"]["bet_bookmaker"], cfg["betting"]["bet_timing"])
    pre = pre.add_prefix("odds_")
    return {"events": ev.set_index("event_id"), "preds": preds, "close": close, "pre": pre}


@st.cache_data(show_spinner=False)
def model_frame(model: str) -> pd.DataFrame:
    """Un match par ligne : probabilités du modèle et du marché (clôture), cotes pré-match,
    probabilité donnée au résultat réel par chacun, vainqueur du « duel » et pari éventuel."""
    d = load_data()
    p = d["preds"][d["preds"].model == model].set_index("event_id")[[*PCOLS, "outcome_idx"]]
    df = d["events"].join(p, how="inner").join(d["close"], how="inner").join(d["pre"])
    y = df["outcome_idx"].to_numpy().astype(int)
    pm = df[PCOLS].to_numpy()
    pk = df[[f"mkt_{c}" for c in PCOLS]].to_numpy()
    rows = np.arange(len(df))
    df["ll_model"] = metrics.log_loss(pm, y)
    df["ll_market"] = metrics.log_loss(pk, y)
    df["pt_model"] = pm[rows, y]
    df["pt_market"] = pk[rows, y]
    gap = df["pt_model"] - df["pt_market"]
    df["duel"] = np.where(gap > TIE, "model", np.where(gap < -TIE, "book", "tie"))

    # Pari de la stratégie principale (sans la mise) : meilleur edge au-dessus du seuil.
    strat = config()["betting"]["strategy"]
    o = df[[f"odds_{k}" for k in OUTCOMES]].to_numpy(float)
    ev = np.where((o <= strat["max_odds"]) & ~np.isnan(o), pm * o - 1.0, -np.inf)
    best = ev.argmax(axis=1)
    has = ev[rows, best] > strat["min_edge"]
    df["bet_idx"] = np.where(has, best, -1)
    df["bet_odds"] = np.where(has, o[rows, best], np.nan)
    df["bet_edge"] = np.where(has, ev[rows, best], np.nan)
    return df.sort_values(["kickoff_date", "home"])


def available_seasons() -> list[str]:
    return sorted(model_frame("ens").season.unique())


# --- backtest interactif ----------------------------------------------------------------------


@st.cache_data(show_spinner="Rejoue tous les paris…")
def betting_frame(split: str):
    from mvm.sports.foot import betting

    return betting.build_frame(config(), split)


@st.cache_data(show_spinner=False)
def simulate(split: str, who: str, book: str, min_edge: float):
    """who : 'model', 'oracle' (probabilités de clôture) ou 'base' (fréquences de base)."""
    from mvm.sports.foot import betting

    cfg = config()
    base, parts = betting_frame(split)
    if who == "oracle":
        df = betting.oracle_frame(base, parts, book)
    else:
        df = betting.frame_for(
            base, parts, "base" if who == "base" else cfg["betting"]["model"], book
        )
    strat = backtest.Strategy(**{**cfg["betting"]["strategy"], "min_edge": min_edge})
    boot = {**cfg["evaluation"]["bootstrap"], "n_boot": 200}
    summary, bets, weekly = backtest.run(
        df, OUTCOMES, strat, cfg["betting"]["initial_bankroll"], **boot
    )
    return summary, bets, weekly


def reports_dir() -> Path:
    return Path(config()["paths"]["reports"])
