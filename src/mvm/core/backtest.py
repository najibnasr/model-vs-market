"""Backtest de value betting, agnostique au sport (K issues).

Le principe. Pour chaque match, le modèle donne des probabilités p_k ; le bookmaker propose des
cotes o_k. L'espérance de gain d'une mise de 1 sur l'issue k vaut

    EV_k = p_k * o_k - 1        (l'"edge" selon le modèle)

On ne parie que si l'edge dépasse un seuil, sur l'issue au plus grand edge (au plus un pari par
match). Si le modèle est moins bon que le marché, les edges qu'il "voit" sont surtout ses propres
erreurs : c'est le piège central du value betting, que le backtest doit rendre visible.

La mise : Kelly fractionnaire. Le critère de Kelly maximise la croissance espérée du LOG du
capital. Pour une cote o et une probabilité p, la fraction optimale du capital à miser est

    f* = (p * o - 1) / (o - 1)     (edge divisé par le gain net en cas de victoire)

On en mise seulement une fraction (1/4 par défaut) : notre p est une ESTIMATION, et Kelly plein
avec une probabilité surestimée mène à la ruine. Kelly fractionnaire sacrifie un peu de
croissance contre beaucoup moins de variance.

La CLV (closing line value). On compare la cote prise à la "vraie" probabilité de clôture
(Pinnacle, marge retirée) : CLV = o * p_close - 1. Si on prend régulièrement des cotes meilleures
que la clôture, on a probablement un vrai edge, même si les résultats (très bruités) ne le
montrent pas encore. C'est l'indicateur que les parieurs professionnels surveillent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mvm.core.metrics import paired_difference


def kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Fraction de Kelly (plein) pour une probabilité p et une cote décimale.

    À IMPLÉMENTER PAR NAJIB (voir le message de Claude pour la ligne exacte).
    """
    raise NotImplementedError("kelly_fraction() : à implémenter")


@dataclass(frozen=True)
class Strategy:
    min_edge: float = 0.05  # EV minimale (selon le modèle) pour parier
    kelly_fraction: float = 0.25  # part de Kelly misée
    max_stake: float = 0.02  # mise maximale par pari, en part du capital
    max_week_exposure: float = 0.25  # somme des mises d'une semaine, en part du capital
    max_odds: float = 10.0  # on ignore les très gros outsiders (cotes peu fiables)


def select_bets(df: pd.DataFrame, outcomes: tuple[str, ...], strat: Strategy) -> pd.DataFrame:
    """df : une ligne par match avec p_<k> (modèle), odds_<k> (prix pris), pc_<k> (proba de
    clôture sans marge), outcome_idx, week. Renvoie une ligne par pari."""
    p = df[[f"p_{k}" for k in outcomes]].to_numpy()
    o = df[[f"odds_{k}" for k in outcomes]].to_numpy()
    pc = df[[f"pc_{k}" for k in outcomes]].to_numpy()
    ev = np.where(o <= strat.max_odds, p * o - 1.0, -np.inf)
    best = ev.argmax(axis=1)
    rows = np.arange(len(df))
    keep = ev[rows, best] > strat.min_edge
    r, b = rows[keep], best[keep]
    f = np.clip(kelly_fraction(p[r, b], o[r, b]), 0.0, None)
    bets = pd.DataFrame(
        {
            "event_id": df["event_id"].to_numpy()[r],
            "week": df["week"].to_numpy()[r],
            "outcome": np.asarray(outcomes)[b],
            "p_model": p[r, b],
            "odds": o[r, b],
            "p_close": pc[r, b],
            "edge": ev[r, b],
            "won": (df["outcome_idx"].to_numpy()[r] == b),
            "stake_frac": np.minimum(strat.kelly_fraction * f, strat.max_stake),
        }
    )
    bets["flat_return"] = np.where(bets["won"], bets["odds"] - 1.0, -1.0)  # pour 1 unité misée
    bets["clv"] = bets["odds"] * bets["p_close"] - 1.0
    for c in ("league", "season", "tier"):
        if c in df.columns:
            bets[c] = df[c].to_numpy()[r]
    return bets


def simulate_bankroll(
    bets: pd.DataFrame, strat: Strategy, initial: float = 1000.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rejoue les paris semaine par semaine. Les paris d'une même semaine sont misés sur le
    capital du début de semaine (ils sont simultanés), avec un plafond d'exposition hebdomadaire."""
    bank = initial
    weekly, staked = [], []
    for wk, g in bets.sort_values("week").groupby("week", sort=True):
        frac = g["stake_frac"].to_numpy()
        total = frac.sum()
        if total > strat.max_week_exposure:
            frac = frac * strat.max_week_exposure / total
        stakes = frac * bank
        pnl = np.where(g["won"], stakes * (g["odds"] - 1.0), -stakes)
        staked.append(g.assign(stake=stakes, pnl=pnl, bank_before=bank))
        bank += pnl.sum()
        weekly.append(
            {
                "week": wk,
                "n_bets": len(g),
                "staked": stakes.sum(),
                "pnl": pnl.sum(),
                "bankroll": bank,
            }
        )
    w = pd.DataFrame(weekly)
    if len(w):
        w["drawdown"] = w["bankroll"] / w["bankroll"].cummax() - 1.0
    return (pd.concat(staked, ignore_index=True) if staked else bets.assign(stake=[], pnl=[]), w)


def cluster_mean(values, clusters, **boot) -> dict:
    """Moyenne et IC bootstrap par cluster (semaine) d'une série, via le test apparié contre 0."""
    v = np.asarray(values, float)
    return paired_difference(v, np.zeros_like(v), clusters, **boot)


def summarize(bets: pd.DataFrame, weekly: pd.DataFrame, initial: float, **boot) -> dict:
    if len(bets) == 0:
        return {"n_bets": 0}
    roi = cluster_mean(bets["flat_return"], bets["week"], **boot)
    clv = cluster_mean(bets["clv"], bets["week"], **boot)
    final = weekly["bankroll"].iloc[-1]
    return {
        "n_bets": len(bets),
        "hit_rate": bets["won"].mean(),
        "avg_odds": bets["odds"].mean(),
        "avg_edge_model": bets["edge"].mean(),
        "roi_flat": roi["mean_diff"],
        "roi_ci_low": roi["ci_low"],
        "roi_ci_high": roi["ci_high"],
        "clv": clv["mean_diff"],
        "clv_ci_low": clv["ci_low"],
        "clv_ci_high": clv["ci_high"],
        "kelly_final_bankroll": final,
        "kelly_return": final / initial - 1.0,
        "max_drawdown": weekly["drawdown"].min(),
    }


def run(df, outcomes, strat: Strategy, initial: float = 1000.0, **boot):
    bets = select_bets(df, outcomes, strat)
    staked, weekly = simulate_bankroll(bets, strat, initial)
    return summarize(staked, weekly, initial, **boot), staked, weekly
