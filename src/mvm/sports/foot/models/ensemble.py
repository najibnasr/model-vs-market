"""Combinaison de modèles : moyenne pondérée des probabilités (linear opinion pool).

Pourquoi ça peut marcher : sur la période de réglage, Elo et Dixon-Coles se trompent
DIFFÉREMMENT. Elo surestime le domicile et il est trop confiant au milieu de l'échelle ;
Dixon-Coles corrige le domicile mais sous-estime les nuls et il est trop prudent aux extrêmes
(son prior L2 comprime les forces vers la moyenne). Quand deux erreurs sont peu corrélées, leur
moyenne en annule une partie.

Poids fixés à 50/50 SANS réglage : un poids optimisé sur la période de réglage serait un
hyperparamètre de plus, donc une occasion de plus de sur-apprendre. La moyenne simple est un
choix défendable a priori, et la littérature sur la combinaison de prévisions montre qu'elle
est souvent difficile à battre ("forecast combination puzzle").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PCOLS = ["p_H", "p_D", "p_A"]


def linear_pool(preds: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Moyenne pondérée de sorties de walk_forward, sur les matchs prédits par tous les membres."""
    total = sum(weights.values())
    names = list(weights)
    base = preds[names[0]].set_index("event_id")
    common = base.index
    for n in names[1:]:
        common = common.intersection(preds[n].set_index("event_id").index)
    p = sum(
        weights[n] / total * preds[n].set_index("event_id").loc[common, PCOLS].to_numpy()
        for n in names
    )
    out = base.loc[common, ["week", "outcome_idx"]].copy()
    out[PCOLS] = p / np.asarray(p).sum(axis=1, keepdims=True)
    return out.reset_index()[["event_id", "week", *PCOLS, "outcome_idx"]]


class Ensemble:
    """Version Forecaster (pour un usage en direct, ex. le dashboard)."""

    def __init__(self, members: dict[str, object], weights: dict[str, float]):
        self.members, self.weights = members, weights

    def update(self, played: pd.DataFrame) -> None:
        for m in self.members.values():
            m.update(played)

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        total = sum(self.weights.values())
        return sum(self.weights[n] / total * m.predict(fixtures) for n, m in self.members.items())
