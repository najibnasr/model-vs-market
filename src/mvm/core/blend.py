"""Test de mélange (forecast encompassing) : le modèle apporte-t-il une information que le
marché n'a pas ?

On combine p = (1 - w) * p_marché + w * p_modèle et on cherche le w qui minimise la log-loss.
- w = 0 : le prix du marché "contient" déjà tout ce que sait le modèle (il l'englobe).
- w > 0 : le modèle porte un signal ORTHOGONAL au prix, même s'il est moins bon seul.

C'est l'analogue exact de la recherche d'alpha en finance : un signal peut être faible seul mais
utile s'il est peu corrélé à ce que le marché intègre déjà. Le poids est ajusté sur la période
de réglage, puis appliqué tel quel sur le test.
"""

from __future__ import annotations

import numpy as np

from mvm.core.metrics import log_loss

GRID = np.round(np.arange(0.0, 0.501, 0.01), 2)


def mix(p_market: np.ndarray, p_model: np.ndarray, w: float) -> np.ndarray:
    return (1.0 - w) * p_market + w * p_model


def fit_weight(p_market, p_model, y, grid=GRID) -> tuple[float, np.ndarray]:
    losses = np.array([log_loss(mix(p_market, p_model, w), y).mean() for w in grid])
    return float(grid[losses.argmin()]), losses
