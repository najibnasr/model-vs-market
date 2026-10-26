"""Benchmark naïf : les fréquences historiques H/D/A de la ligue.

Ce "modèle" ne sait rien des équipes. Il dit seulement : en Ligue 1, depuis le début des
données, il y a eu 45 % de victoires à domicile, 27 % de nuls, 28 % de défaites. Si Elo ne bat
pas ça nettement, c'est qu'il y a un bug (section 7 du protocole).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

IDX = {"H": 0, "D": 1, "A": 2}


class LeagueBaseRates:
    def __init__(self, prior: tuple[float, float, float] = (0.45, 0.27, 0.28), strength: int = 30):
        self.counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3))
        self.prior = np.asarray(prior) * strength  # lissage : pseudo-observations

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        rows = [self.counts[lg] + self.prior for lg in fixtures["league"]]
        p = np.vstack(rows)
        return p / p.sum(axis=1, keepdims=True)

    def update(self, played: pd.DataFrame) -> None:
        for lg, res in zip(played["league"], played["result"], strict=True):
            self.counts[lg][IDX[res]] += 1
