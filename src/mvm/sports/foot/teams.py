"""Normalisation des noms d'équipes en identifiants stables.

Choix clé : l'identifiant est rattaché au PAYS, pas à la ligue. Un club promu de Championship
en Premier League garde le même team_id, donc son rating (Elo, Dixon-Coles) le suit d'une
division à l'autre. Avec un id par ligue, chaque promu repartirait de zéro et le modèle
perdrait l'information la plus utile sur lui : son niveau de la saison précédente.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def canonical_name(raw: str, country: str, aliases: dict) -> str:
    return (aliases.get(country) or {}).get(raw, raw)


def team_id(raw: str, country: str, aliases: dict) -> str:
    return f"{country}:{slugify(canonical_name(raw, country, aliases))}"


def build_teams_table(pairs: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    """pairs : colonnes (country, raw_name). Retourne la table teams dédoublonnée."""
    t = pairs.drop_duplicates().copy()
    t["canonical_name"] = [
        canonical_name(r, c, aliases) for r, c in zip(t.raw_name, t.country, strict=True)
    ]
    t["team_id"] = [team_id(r, c, aliases) for r, c in zip(t.raw_name, t.country, strict=True)]
    return (
        t[["team_id", "country", "canonical_name", "raw_name"]]
        .sort_values(["country", "team_id", "raw_name"])
        .reset_index(drop=True)
    )


def slug_collisions(teams: pd.DataFrame) -> pd.DataFrame:
    """Noms bruts différents qui tombent sur le même id SANS alias explicite : à vérifier à la main
    (deux clubs distincts fusionnés par erreur, ou bien le même club : alors ajouter l'alias)."""
    implicit = teams[teams.canonical_name == teams.raw_name]
    counts = implicit.groupby("team_id")["raw_name"].nunique()
    return implicit[implicit.team_id.isin(counts[counts > 1].index)]
