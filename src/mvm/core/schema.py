"""Schéma canonique, agnostique au sport.

Trois tables :
- events : une ligne par match (qui joue, quand, résultat).
- odds   : format long, une ligne par (match, bookmaker, marché, moment, issue).
- teams  : correspondance nom brut -> identifiant stable.

Le cœur (calibration, backtest, CLV) ne lit que ces tables. Un nouveau sport n'a qu'à
produire ces trois tables pour être branché sur tout le reste.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Issues possibles par marché. Le foot a le nul, la NBA non : c'est la seule différence
# structurelle entre les deux sports du point de vue du cœur.
MARKET_OUTCOMES: dict[str, tuple[str, ...]] = {
    "1x2": ("H", "D", "A"),
    "moneyline": ("H", "A"),
}

# Moment de relevé de la cote. Ne jamais mélanger les deux dans une même analyse sans le dire :
# une cote pré-match contient moins d'information qu'une cote de clôture.
TIMINGS: tuple[str, ...] = ("pre_match", "closing")


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    nullable: bool = False


EVENTS_SCHEMA: tuple[Column, ...] = (
    Column("event_id", "string"),
    Column("sport", "string"),
    Column("league", "string"),
    Column("tier", "int64"),
    Column("season", "string"),
    Column("kickoff_date", "datetime64[ns]"),
    Column("kickoff_time", "string", nullable=True),
    Column("home_team_id", "string"),
    Column("away_team_id", "string"),
    Column("home_score", "int64"),
    Column("away_score", "int64"),
    Column("result", "string"),
)

ODDS_SCHEMA: tuple[Column, ...] = (
    Column("event_id", "string"),
    Column("bookmaker", "string"),
    Column("market", "string"),
    Column("timing", "string"),
    Column("outcome", "string"),
    Column("decimal_odds", "float64"),
    Column("source_column", "string"),
)

TEAMS_SCHEMA: tuple[Column, ...] = (
    Column("team_id", "string"),
    Column("country", "string"),
    Column("canonical_name", "string"),
    Column("raw_name", "string"),
)


class SchemaError(ValueError):
    """Levée quand une table ne respecte pas son schéma."""


def coerce(df: pd.DataFrame, schema: tuple[Column, ...]) -> pd.DataFrame:
    """Réordonne et type les colonnes selon le schéma (sans rien inventer)."""
    missing = [c.name for c in schema if c.name not in df.columns]
    if missing:
        raise SchemaError(f"Colonnes manquantes : {missing}")
    out = df[[c.name for c in schema]].copy()
    for c in schema:
        out[c.name] = out[c.name].astype(c.dtype)
    return out


def validate(df: pd.DataFrame, schema: tuple[Column, ...], table: str) -> None:
    """Vérifie colonnes, types et valeurs manquantes. Lève SchemaError si violation."""
    expected = [c.name for c in schema]
    if list(df.columns) != expected:
        raise SchemaError(f"[{table}] colonnes {list(df.columns)} != attendu {expected}")
    for c in schema:
        if str(df[c.name].dtype) != c.dtype:
            raise SchemaError(f"[{table}] {c.name}: type {df[c.name].dtype} != {c.dtype}")
        if not c.nullable and df[c.name].isna().any():
            n = int(df[c.name].isna().sum())
            raise SchemaError(f"[{table}] {c.name}: {n} valeurs manquantes interdites")

    if table == "odds":
        bad_market = set(df["market"]) - set(MARKET_OUTCOMES)
        if bad_market:
            raise SchemaError(f"[odds] marchés inconnus : {bad_market}")
        bad_timing = set(df["timing"]) - set(TIMINGS)
        if bad_timing:
            raise SchemaError(f"[odds] timings inconnus : {bad_timing}")
        for market, outcomes in MARKET_OUTCOMES.items():
            sub = df.loc[df["market"] == market, "outcome"]
            bad = set(sub) - set(outcomes)
            if bad:
                raise SchemaError(f"[odds] issues {bad} invalides pour le marché {market}")
