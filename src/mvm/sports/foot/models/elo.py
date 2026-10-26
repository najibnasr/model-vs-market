"""Modèle Elo pour le 1X2 : ratings Elo + couche de calibration "logit ordonné".

Problème : Elo donne un score attendu E (une seule valeur), alors qu'on veut trois
probabilités (H, D, A). Le nul n'existe pas dans Elo. Solution en deux couches :

1. Ratings Elo mis à jour match après match (core/elo.py).
2. Logit ordonné : on suppose une "performance relative" latente y* = beta * d + bruit logistique,
   où d est l'écart de rating (avantage du terrain compris). Le résultat est
       A si y* < c1,   D si c1 <= y* < c2,   H si y* >= c2.
   Donc  P(A) = s(c1 - beta d),  P(A ou D) = s(c2 - beta d),  avec s la sigmoïde.
   Les seuils c1 < c2 règlent le taux de nuls de la ligue ; beta règle la confiance qu'on
   accorde à l'écart de ratings. On ajuste (beta, c1, c2) par maximum de vraisemblance, PAR LIGUE,
   sur les saisons passées uniquement (fenêtre glissante), au début de chaque saison.

Pourquoi le logit ordonné et pas un modèle à 3 classes libre : l'ordre H > D > A est une vraie
structure (plus l'écart est grand, plus H est probable et A improbable). L'imposer coûte
3 paramètres par ligue au lieu de 4 et rend le modèle plus stable.

Détails qui comptent :
- Promus : l'identifiant d'équipe est par pays, donc un promu garde son rating de D2.
- Équipe jamais vue : rating = moyenne de sa ligue moins `new_team_offset` (un nouveau venu est en
  moyenne plus faible que ceux qui sont déjà là).
- Intersaison : chaque rating est rapproché de la moyenne de sa ligue d'une fraction
  `season_regression` (transferts, entraîneurs : l'information de la saison passée s'use).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from mvm.core.elo import EloRatings, goal_margin_multiplier

RESULT_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}
RESULT_IDX = {"H": 0, "D": 1, "A": 2}
D_SCALE = 100.0  # écart de rating exprimé en centaines de points : paramètres d'ordre 1


def ordered_logit_probs(d: np.ndarray, beta: float, c1: float, c2: float) -> np.ndarray:
    """Renvoie (n, 3) dans l'ordre (H, D, A)."""
    p_a = expit(c1 - beta * d)
    p_ad = expit(c2 - beta * d)
    p = np.column_stack([1.0 - p_ad, p_ad - p_a, p_a])
    return np.clip(p, 1e-9, 1.0) / np.clip(p, 1e-9, 1.0).sum(axis=1, keepdims=True)


def fit_ordered_logit(d: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """MLE de (beta, c1, c2). Paramétrage c2 = c1 + exp(g) pour garantir c1 < c2."""

    def nll(theta):
        beta, c1, g = theta
        p = ordered_logit_probs(d, beta, c1, c1 + np.exp(g))
        return -np.log(p[np.arange(len(y)), y]).mean()

    # départ : taux de nuls ~ 26 %, avantage domicile ~ 45 % H
    res = minimize(nll, x0=np.array([0.5, -0.8, np.log(1.0)]), method="L-BFGS-B")
    beta, c1, g = res.x
    return float(beta), float(c1), float(c1 + np.exp(g))


class FootElo:
    """Forecaster compatible avec mvm.core.splits.walk_forward."""

    def __init__(
        self,
        k: float = 20.0,
        home_advantage: float = 60.0,
        season_regression: float = 0.2,
        new_team_offset: float = 100.0,
        use_margin: bool = True,
        calib_seasons: int = 6,
    ):
        self.elo = EloRatings(k=k, home_advantage=home_advantage)
        self.season_regression = season_regression
        self.new_team_offset = new_team_offset
        self.use_margin = use_margin
        self.calib_seasons = calib_seasons

        self.current_season: dict[str, str] = {}  # ligue -> saison en cours
        self.last_teams: dict[str, set[str]] = defaultdict(set)  # ligue -> équipes de la saison
        # ligue -> [(saison, d, y)] : écart de rating PRÉ-match et résultat, pour la calibration
        self.history: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
        self.calib: dict[tuple[str, str], tuple[float, float, float]] = {}

    # --- gestion des équipes et des saisons (aucune information de résultat utilisée) ---------

    def _league_mean(self, league: str) -> float:
        teams = self.last_teams[league]
        return float(np.mean([self.elo.get(t) for t in teams])) if teams else self.elo.base

    def _ensure_team(self, team: str, league: str) -> None:
        if team not in self.elo.ratings:
            known = bool(self.last_teams[league])
            offset = self.new_team_offset if known else 0.0
            self.elo.set(team, self._league_mean(league) - offset)

    def _start_season_if_needed(self, league: str, season: str) -> None:
        if self.current_season.get(league) == season:
            return
        if league in self.current_season:  # pas pour la toute première saison
            m = self._league_mean(league)
            for t in self.last_teams[league]:
                r = self.elo.get(t)
                self.elo.set(t, m + (1.0 - self.season_regression) * (r - m))
        self.current_season[league] = season
        self.last_teams[league] = set()

    def _diff(self, home: str, away: str) -> float:
        return (self.elo.get(home) + self.elo.home_advantage - self.elo.get(away)) / D_SCALE

    def _calibration(self, league: str, season: str) -> tuple[float, float, float]:
        key = (league, season)
        if key not in self.calib:
            hist = self.history[league]
            past = sorted({s for s, _, _ in hist if s < season})
            window = set(past[-self.calib_seasons :])
            rows = [(d, y) for s, d, y in hist if s in window]
            if len(rows) < 200:  # trop peu d'historique dans la ligue : on mutualise
                rows = [(d, y) for h in self.history.values() for s, d, y in h if s < season]
            if len(rows) < 50:
                self.calib[key] = (0.5, -0.8, 0.3)  # valeurs neutres raisonnables
            else:
                d, y = map(np.asarray, zip(*rows, strict=True))
                self.calib[key] = fit_ordered_logit(d.astype(float), y.astype(int))
        return self.calib[key]

    # --- interface Forecaster -----------------------------------------------------------------

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(fixtures), 3))
        for i, row in enumerate(fixtures.itertuples(index=False)):
            self._start_season_if_needed(row.league, row.season)
            self._ensure_team(row.home_team_id, row.league)
            self._ensure_team(row.away_team_id, row.league)
            beta, c1, c2 = self._calibration(row.league, row.season)
            d = np.array([self._diff(row.home_team_id, row.away_team_id)])
            out[i] = ordered_logit_probs(d, beta, c1, c2)[0]
        return out

    def update(self, played: pd.DataFrame) -> None:
        for row in played.sort_values(["kickoff_date", "event_id"]).itertuples(index=False):
            lg, h, a = row.league, row.home_team_id, row.away_team_id
            self._start_season_if_needed(lg, row.season)
            self._ensure_team(h, lg)
            self._ensure_team(a, lg)
            self.last_teams[lg].update((h, a))
            self.history[lg].append((row.season, self._diff(h, a), RESULT_IDX[row.result]))
            mult = goal_margin_multiplier(row.home_score - row.away_score) if self.use_margin else 1
            self.elo.update(h, a, RESULT_SCORE[row.result], mult)

    def ratings_table(self) -> pd.DataFrame:
        return (
            pd.Series(self.elo.ratings, name="rating")
            .rename_axis("team_id")
            .sort_values(ascending=False)
            .reset_index()
        )
