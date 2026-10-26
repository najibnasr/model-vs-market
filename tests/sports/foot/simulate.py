"""Simulateur de championnats, pour tester les modèles sur des données dont on CONNAÎT la vérité.

Chaque équipe a une force latente (attaque) qui dérive lentement ; les buts suivent des lois de
Poisson. On simule deux divisions d'un même pays avec montées et descentes, et un "marché" qui
connaît les vraies probabilités (plus du bruit et une marge). On sait donc ce qu'on doit
observer : Elo doit battre les fréquences de base, et le marché (qui voit la vérité) doit
battre Elo.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import poisson

HOME_ADV = 0.25
BASE_GOALS = 0.25


def true_probs(lam_h: float, lam_a: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]) / m.sum()


def simulate(n_seasons: int = 10, n_teams: int = 12, first_year: int = 2005, seed: int = 0):
    rng = np.random.default_rng(seed)
    teams = [f"eng:t{i:02d}" for i in range(2 * n_teams)]
    strength = {t: rng.normal(0.3 if i < n_teams else -0.3, 0.3) for i, t in enumerate(teams)}
    divisions = {"E0": teams[:n_teams], "E1": teams[n_teams:]}
    events, odds = [], []
    for s in range(n_seasons):
        year = first_year + s
        season = f"{year}-{year + 1}"
        points = {t: 0 for t in teams}
        for lg, tier in (("E0", 1), ("E1", 2)):
            fixtures = list(itertools.permutations(divisions[lg], 2))
            rng.shuffle(fixtures)
            for j, (h, a) in enumerate(fixtures):
                date = pd.Timestamp(f"{year}-08-10") + pd.Timedelta(days=7 * (j // (n_teams // 2)))
                lam_h = np.exp(BASE_GOALS + HOME_ADV + strength[h] - strength[a])
                lam_a = np.exp(BASE_GOALS + strength[a] - strength[h])
                hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
                res = "H" if hg > ag else "A" if hg < ag else "D"
                points[h] += 3 if res == "H" else 1 if res == "D" else 0
                points[a] += 3 if res == "A" else 1 if res == "D" else 0
                eid = f"{season}-{lg}-{j}"
                events.append(
                    dict(
                        event_id=eid,
                        sport="foot",
                        league=lg,
                        tier=tier,
                        season=season,
                        kickoff_date=date,
                        kickoff_time=None,
                        home_team_id=h,
                        away_team_id=a,
                        home_score=hg,
                        away_score=ag,
                        result=res,
                    )
                )
                p = true_probs(lam_h, lam_a)
                noisy = np.clip(p * np.exp(rng.normal(0, 0.03, 3)), 0.01, None)
                noisy = np.clip(noisy / noisy.sum(), 0.01, 0.95)  # cote > 1 garantie
                price = 1 / (noisy * 1.03)
                for o, price_o in zip("HDA", price, strict=True):
                    odds.append(
                        dict(
                            event_id=eid,
                            bookmaker="pinnacle",
                            market="1x2",
                            timing="closing",
                            outcome=o,
                            decimal_odds=price_o,
                            source_column="PSC" + o,
                        )
                    )
            for t in divisions[lg]:
                strength[t] += rng.normal(0, 0.08)  # dérive entre saisons
        # montées / descentes : 2 derniers de E0 <-> 2 premiers de E1
        down = sorted(divisions["E0"], key=points.get)[:2]
        up = sorted(divisions["E1"], key=points.get)[-2:]
        divisions["E0"] = [t for t in divisions["E0"] if t not in down] + up
        divisions["E1"] = [t for t in divisions["E1"] if t not in up] + down
    ev = pd.DataFrame(events).sort_values(["kickoff_date", "event_id"]).reset_index(drop=True)
    return ev, pd.DataFrame(odds), strength
