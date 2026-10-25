"""Probabilités du marché, à partir de la table canonique `odds`."""

from __future__ import annotations

import pandas as pd

from mvm.core.odds import implied_probabilities, overround, shin_z
from mvm.core.schema import MARKET_OUTCOMES


def odds_matrix(
    odds: pd.DataFrame, bookmaker: str, timing: str, market: str = "1x2"
) -> pd.DataFrame:
    """Une ligne par match ayant un jeu COMPLET de cotes pour (book, moment) ; colonnes = issues."""
    outcomes = list(MARKET_OUTCOMES[market])
    sub = odds[(odds.bookmaker == bookmaker) & (odds.timing == timing) & (odds.market == market)]
    wide = sub.pivot_table(
        index="event_id", columns="outcome", values="decimal_odds", aggfunc="first"
    )
    return wide.reindex(columns=outcomes).dropna()


def market_probabilities(
    odds: pd.DataFrame, bookmaker: str, timing: str, method: str = "shin", market: str = "1x2"
) -> pd.DataFrame:
    """Probabilités implicites sans marge, plus deux diagnostics (overround, z de Shin)."""
    m = odds_matrix(odds, bookmaker, timing, market)
    p = implied_probabilities(m.to_numpy(), method)
    out = pd.DataFrame(p, index=m.index, columns=[f"p_{c}" for c in m.columns])
    out["overround"] = overround(m.to_numpy())
    out["shin_z"] = shin_z(m.to_numpy())
    return out
