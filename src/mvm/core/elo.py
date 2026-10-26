"""Moteur Elo générique (foot, NBA...).

Idée : chaque équipe a un rating R. Avant un match, on calcule le score attendu de l'équipe à
domicile E (entre 0 et 1). Après le match, on corrige les ratings proportionnellement à la
"surprise" S - E, où S vaut 1 (victoire), 0.5 (nul) ou 0 (défaite) :

    R_home <- R_home + K * (S - E)
    R_away <- R_away - K * (S - E)

C'est une descente de gradient en ligne : K est le pas d'apprentissage. Grand K = le modèle
réagit vite (bon si les niveaux changent vite, mauvais car il sur-réagit au bruit). Petit K =
modèle stable mais lent à voir un changement réel (nouvel entraîneur, mercato). Le bon K se
règle sur la période de tuning, jamais sur le test.

Le système est à somme nulle : ce que gagne l'un, l'autre le perd. La moyenne des ratings reste
donc constante, ce qui donne une échelle stable dans le temps.
"""

from __future__ import annotations


def expected_home_score(r_home: float, r_away: float, home_advantage: float) -> float:
    """Score attendu de l'équipe à domicile, entre 0 et 1.

    À IMPLÉMENTER PAR NAJIB. Formule Elo classique (échelle 400) :

        E = 1 / (1 + 10 ** (-(r_home + home_advantage - r_away) / 400))

    Vérifie mentalement : à ratings égaux et sans avantage du terrain, E doit valoir 0.5 ;
    avec 400 points d'écart, E = 10/11, environ 0.91.
    Les tests sont dans tests/core/test_elo.py.
    """
    return 1 / (1 + 10 ** (-(r_home + home_advantage - r_away) / 400))


def goal_margin_multiplier(goal_diff: int) -> float:
    """Multiplicateur de K selon l'écart de buts (convention du World Football Elo).

    Une victoire 4-0 contient plus d'information qu'un 1-0, mais pas 4 fois plus : la croissance
    est volontairement sous-linéaire pour ne pas sur-récompenser les cartons contre des faibles.
    """
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


class EloRatings:
    """Stocke et met à jour les ratings. Ne connaît ni les ligues ni les saisons."""

    def __init__(self, k: float, home_advantage: float, base: float = 1500.0):
        self.k = k
        self.home_advantage = home_advantage
        self.base = base
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def set(self, team: str, value: float) -> None:
        self.ratings[team] = value

    def expected(self, home: str, away: str) -> float:
        return expected_home_score(self.get(home), self.get(away), self.home_advantage)

    def update(self, home: str, away: str, score: float, multiplier: float = 1.0) -> float:
        """Met à jour après un match ; renvoie la variation du rating domicile."""
        delta = self.k * multiplier * (score - self.expected(home, away))
        self.ratings[home] = self.get(home) + delta
        self.ratings[away] = self.get(away) - delta
        return delta
