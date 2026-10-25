"""Évaluation walk-forward, avec une garantie structurelle contre le leakage.

Principe : on rejoue l'histoire semaine par semaine. Avant chaque semaine, le modèle reçoit
UNIQUEMENT les matchs déjà joués (résultats compris), puis doit prédire les matchs de la semaine
SANS voir leurs scores : les colonnes de résultat sont physiquement retirées du DataFrame passé
à predict(). Un modèle ne peut donc pas tricher, même par erreur.

Pourquoi la semaine et pas le jour : avant 2019-20, on n'a pas l'heure des matchs. Deux matchs
du même samedi ne doivent pas pouvoir s'informer l'un l'autre. Grouper par semaine (lundi au
dimanche) est conservateur : un match du dimanche ne profite pas du résultat du samedi. On perd
un peu d'information, on ne risque jamais d'en voler.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np
import pandas as pd

RESULT_COLUMNS = ("home_score", "away_score", "result")


class Forecaster(Protocol):
    """Interface commune à tous les modèles (Elo, Dixon-Coles, et plus tard NBA)."""

    def update(self, played: pd.DataFrame) -> None:
        """Intègre des matchs joués (avec résultats), en ordre chronologique."""

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Renvoie un tableau (n, K) de probabilités pour des matchs SANS résultats."""


def week_key(dates: pd.Series) -> pd.Series:
    """Lundi de la semaine de chaque date : identifiant de bloc temporel."""
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


def hide_results(fixtures: pd.DataFrame) -> pd.DataFrame:
    return fixtures.drop(columns=[c for c in RESULT_COLUMNS if c in fixtures.columns])


def iter_weeks(events: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    ev = events.sort_values(["kickoff_date", "event_id"])
    yield from ev.groupby(week_key(ev["kickoff_date"]), sort=True)


def walk_forward(
    model: Forecaster, events: pd.DataFrame, eval_seasons: list[str], outcomes=("H", "D", "A")
) -> pd.DataFrame:
    """Rejoue tous les événements dans l'ordre. Prédit seulement les matchs des saisons
    d'évaluation, mais met à jour le modèle sur TOUT (burn-in compris).

    Renvoie une ligne par match prédit : event_id, week, p_<issue>..., outcome_idx.
    """
    eval_set = set(eval_seasons)
    rows = []
    for wk, block in iter_weeks(events):
        to_predict = block[block["season"].isin(eval_set)]
        if len(to_predict):
            p = np.asarray(model.predict(hide_results(to_predict)), dtype=float)
            expected = (len(to_predict), len(outcomes))
            if p.shape != expected:
                raise ValueError(f"predict() a renvoyé {p.shape}, attendu {expected}")
            out = pd.DataFrame(p, columns=[f"p_{o}" for o in outcomes])
            out.insert(0, "event_id", to_predict["event_id"].to_numpy())
            out.insert(1, "week", wk)
            out["outcome_idx"] = (
                to_predict["result"].map({o: i for i, o in enumerate(outcomes)}).to_numpy()
            )
            rows.append(out)
        model.update(block)  # les résultats de la semaine ne servent qu'APRÈS sa prédiction
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def seasons_between(first: str, last: str, available: list[str]) -> list[str]:
    return [s for s in sorted(available) if first <= s <= last]
