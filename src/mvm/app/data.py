"""Chargement et préparation des données du dashboard (mis en cache par Streamlit).

Saisons montrées : l'historique 2012-13 à 2024-25 (référence Pinnacle) et la saison en direct
(référence : moyenne du marché, Pinnacle n'étant plus publié). Le holdout 2025-26 n'est JAMAIS
affiché : il reste réservé à l'évaluation de la v2 (docs/live_2026_27.md).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mvm.core import backtest, metrics
from mvm.core.io import load_config, read_table
from mvm.core.market import market_probabilities, odds_matrix

OUTCOMES = ("H", "D", "A")
PCOLS = [f"p_{o}" for o in OUTCOMES]
MKT = [f"mkt_{c}" for c in PCOLS]
ODDS = [f"odds_{o}" for o in OUTCOMES]
MODEL_NAMES = {
    "ens": "Elo + Dixon-Coles blend",
    "elo": "Elo",
    "dc": "Dixon-Coles",
    "base": "League base rates",
}
# Écart minimal (en probabilité sur le bon résultat) pour désigner un vainqueur du duel.
TIE = 0.01


@st.cache_resource
def config() -> dict:
    return load_config(os.environ.get("MVM_CONFIG", "configs/foot.yaml"))


def league_name(code: str) -> str:
    return config()["leagues"].get(code, {}).get("name", code)


def league_order(codes) -> list[str]:
    lg = config()["leagues"]
    return sorted(codes, key=lambda c: (lg.get(c, {}).get("tier", 9), list(lg).index(c)))


def split_range(split: str) -> tuple[str, str]:
    ev = config()["evaluation"]
    return {
        "tuning": (ev["eval_first"], ev["tuning_last"]),
        "test": (ev["test_first"], ev["test_last"]),
        "archive": (ev["eval_first"], ev["test_last"]),
        "live": (ev.get("forward", "9999"), ev.get("forward", "9999")),
    }[split]


def live_season() -> str:
    return config()["evaluation"].get("forward", "")


def _ref(cfg: dict) -> dict:
    return cfg.get("live", {}).get("reference", cfg["evaluation"]["fallback_reference"])


@st.cache_data(show_spinner="Loading matches and odds…")
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
        league_name=events.league.map(league_name),
    )
    ref, live = cfg["evaluation"]["reference"], _ref(cfg)
    bet_live = cfg.get("live", {}).get("bet_bookmaker", "market_avg")
    return {
        "events": ev.set_index("event_id"),
        "names": names,
        "preds": preds,
        "close_ref": market_probabilities(
            odds, ref["bookmaker"], ref["timing"], ref["margin_method"]
        )[PCOLS],
        "close_live": market_probabilities(
            odds, live["bookmaker"], live["timing"], live["margin_method"]
        )[PCOLS],
        "pre_ref": odds_matrix(odds, cfg["betting"]["bet_bookmaker"], cfg["betting"]["bet_timing"]),
        "pre_live": odds_matrix(odds, bet_live, "pre_match"),
        "pre_max": odds_matrix(odds, "market_max", "pre_match"),
    }


def _bet_columns(df: pd.DataFrame, p: np.ndarray, o: np.ndarray) -> pd.DataFrame:
    """Pari de la stratégie principale (sans la mise) : meilleur edge au-dessus du seuil."""
    strat = config()["betting"]["strategy"]
    rows = np.arange(len(df))
    ev = np.where((o <= strat["max_odds"]) & ~np.isnan(o), p * o - 1.0, -np.inf)
    best = ev.argmax(axis=1) if len(df) else np.array([], int)
    has = ev[rows, best] > strat["min_edge"] if len(df) else np.array([], bool)
    df["bet_idx"] = np.where(has, best, -1)
    df["bet_odds"] = np.where(has, o[rows, best], np.nan)
    df["bet_edge"] = np.where(has, ev[rows, best], np.nan)
    return df


@st.cache_data(show_spinner=False)
def model_frame(model: str) -> pd.DataFrame:
    """Un match joué par ligne : probabilités du modèle et du marché (clôture), cotes pré-match,
    probabilité donnée au résultat réel par chacun, vainqueur du « duel » et pari éventuel."""
    d = load_data()
    a0, a1 = split_range("archive")
    live = live_season()
    p = d["preds"][d["preds"].model == model].set_index("event_id")
    df = d["events"].join(p[[*PCOLS, "outcome_idx", "week"]], how="inner")
    df = df[((df.season >= a0) & (df.season <= a1)) | (df.season == live)].copy()
    is_live = (df.season == live).to_numpy()[:, None]

    def pick(archive: pd.DataFrame, current: pd.DataFrame) -> np.ndarray:
        a = archive.reindex(df.index).to_numpy(float)
        c = current.reindex(df.index).to_numpy(float)
        return np.where(is_live, c, a)

    df[MKT] = pick(d["close_ref"], d["close_live"])
    df[ODDS] = pick(d["pre_ref"][list(OUTCOMES)], d["pre_live"][list(OUTCOMES)])
    df[[f"max_{o}" for o in OUTCOMES]] = d["pre_max"][list(OUTCOMES)].reindex(df.index).to_numpy()
    df = df.dropna(subset=MKT)
    y = df["outcome_idx"].to_numpy().astype(int)
    pm, pk = df[PCOLS].to_numpy(), df[MKT].to_numpy()
    rows = np.arange(len(df))
    df["ll_model"] = metrics.log_loss(pm, y)
    df["ll_market"] = metrics.log_loss(pk, y)
    df["pt_model"] = pm[rows, y]
    df["pt_market"] = pk[rows, y]
    gap = df["pt_model"] - df["pt_market"]
    df["duel"] = np.where(gap > TIE, "model", np.where(gap < -TIE, "book", "tie"))
    df = _bet_columns(df, pm, df[ODDS].to_numpy(float))
    return df.sort_values(["kickoff_date", "home"])


def archive_frame(model: str = "ens") -> pd.DataFrame:
    a0, a1 = split_range("archive")
    df = model_frame(model)
    return df[(df.season >= a0) & (df.season <= a1)]


def live_frame(model: str = "ens") -> pd.DataFrame:
    df = model_frame(model)
    return df[df.season == live_season()]


# --- matchs à venir -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def upcoming() -> pd.DataFrame | None:
    from mvm.sports.foot import live

    cfg = config()
    up = live.load_upcoming(cfg)
    if up is None or up["fixtures"].empty:
        return None
    fx = up["fixtures"].set_index("event_id")
    names = load_data()["names"]
    fx["home"] = fx.home_team_id.map(names).fillna(fx.home_name).astype(str)
    fx["away"] = fx.away_team_id.map(names).fillna(fx.away_name).astype(str)
    fx["league_name"] = fx.league.map(league_name)
    pr = up["predictions"]
    fx = fx.join(pr[pr.model == "ens"].set_index("event_id")[PCOLS], how="inner")
    ref = cfg["live"]["upcoming_reference"]
    mkt = market_probabilities(up["odds"], ref["bookmaker"], ref["timing"], ref["margin_method"])
    fx = fx.join(mkt[PCOLS].add_prefix("mkt_"), how="inner")
    o = odds_matrix(up["odds"], cfg["live"]["bet_bookmaker"], "pre_match")[list(OUTCOMES)]
    fx[ODDS] = o.reindex(fx.index).to_numpy(float)
    fx = _bet_columns(fx, fx[PCOLS].to_numpy(), fx[ODDS].to_numpy(float))
    return fx.sort_values(["kickoff_date", "kickoff_time", "league"])


@st.cache_data(show_spinner=False)
def live_log() -> pd.DataFrame:
    from mvm.sports.foot import live

    return live.read_log(config())


def logged_ids() -> set[str]:
    from mvm.sports.foot import live

    return set(live.logged_before_kickoff(live_log()).index)


def last_update() -> datetime | None:
    log = live_log()
    return None if log.empty else log.logged_at.max().to_pydatetime()


# --- pronostics de l'utilisateur (fichier local) -----------------------------------------------


def picks_path() -> Path:
    return Path(config()["paths"]["reports"]) / "live" / "my_picks.csv"


def load_picks() -> pd.DataFrame:
    p = picks_path()
    if not p.exists():
        return pd.DataFrame(columns=["event_id", "pick", "picked_at"])
    df = pd.read_csv(p, dtype={"event_id": str})
    return df.sort_values("picked_at").groupby("event_id").tail(1)


def kicked_off(date, time: str | None, now: datetime | None = None) -> bool:
    """Heures de football-data.co.uk = heure de Londres. Sans heure, on ferme à minuit."""
    now = now or datetime.now(UTC)
    t = time if isinstance(time, str) and ":" in time else "00:00"
    ko = pd.Timestamp(f"{pd.Timestamp(date):%Y-%m-%d} {t}").tz_localize("Europe/London")
    return ko <= pd.Timestamp(now)


def save_pick(event_id: str, pick: int | None) -> None:
    p = picks_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "pick": -1 if pick is None else int(pick),
                "picked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        ]
    )
    row.to_csv(p, mode="a", header=not p.exists(), index=False)


# --- simulations de paris -----------------------------------------------------------------------


def _boot() -> dict:
    return {**config()["evaluation"]["bootstrap"], "n_boot": 200}


def _strategy(min_edge: float) -> backtest.Strategy:
    return backtest.Strategy(**{**config()["betting"]["strategy"], "min_edge": min_edge})


@st.cache_data(show_spinner="Replaying every bet…")
def betting_frame(split: str):
    from mvm.sports.foot import betting

    return betting.build_frame(config(), split)


@st.cache_data(show_spinner=False)
def simulate(split: str, who: str, book: str, min_edge: float):
    """Historique (Pinnacle). who : 'model', 'oracle' (clôture) ou 'base' (fréquences)."""
    from mvm.sports.foot import betting

    cfg = config()
    base, parts = betting_frame(split)
    if who == "oracle":
        df = betting.oracle_frame(base, parts, book)
    else:
        model = "base" if who == "base" else cfg["betting"]["model"]
        df = betting.frame_for(base, parts, model, book)
    return backtest.run(
        df, OUTCOMES, _strategy(min_edge), cfg["betting"]["initial_bankroll"], **_boot()
    )


@st.cache_data(show_spinner=False)
def simulate_live(who: str, book: str, min_edge: float):
    """Saison en direct : prix moyen (ou meilleur prix), CLV contre la clôture moyenne."""
    df = live_frame("base" if who == "base" else "ens").copy()
    if df.empty:
        return {"n_bets": 0}, pd.DataFrame(), pd.DataFrame()
    for o in OUTCOMES:
        df[f"pc_{o}"] = df[f"mkt_p_{o}"]
        if who == "oracle":
            df[f"p_{o}"] = df[f"mkt_p_{o}"]
        if book == "market_max":
            df[f"odds_{o}"] = df[f"max_{o}"]
    df = df.dropna(subset=ODDS).reset_index()
    return backtest.run(
        df, OUTCOMES, _strategy(min_edge), config()["betting"]["initial_bankroll"], **_boot()
    )


def reports_dir() -> Path:
    return Path(config()["paths"]["reports"])
