"""Des cotes aux probabilités : retrait de la marge du bookmaker.

Une cote décimale o "contient" une probabilité brute 1/o. Mais la somme des 1/o sur toutes les
issues vaut plus que 1 (l'overround) : c'est la marge du book. Pour obtenir les probabilités
que le marché attribue réellement aux issues, il faut répartir cette marge. Toute la question
est : COMMENT la répartir ? Chaque méthode encode une hypothèse sur la façon dont le book
construit ses prix.

- proportionnelle : la marge est répartie au prorata. Simple, mais elle ignore le biais
  favori/outsider (le book charge plus de marge sur les outsiders).
- Shin (1993) : le book se protège contre des parieurs initiés ("insiders", proportion z).
  Il gonfle davantage le prix des outsiders, où un initié gagne le plus. Retirer la marge à la
  Shin rend donc plus de probabilité au favori qu'au prorata.
- power : p_i = (1/o_i)^k, avec k choisi pour que la somme fasse 1. Comme 1/o < 1, élever à une
  puissance k > 1 écrase davantage les petites probabilités (outsiders) : même direction que
  Shin, autre forme fonctionnelle.

Toutes les fonctions prennent un tableau (n_matchs, n_issues) de cotes décimales et renvoient
un tableau de même forme dont chaque ligne somme à 1. Elles marchent pour 2 issues (NBA) ou 3.
"""

from __future__ import annotations

import numpy as np

_BISECTION_STEPS = 80  # précision ~1e-24 sur l'intervalle : largement suffisant


def _as_2d(odds) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    if o.ndim == 1:
        o = o[None, :]
    if np.any(~np.isfinite(o)) or np.any(o <= 1.0):
        raise ValueError("Cotes invalides : toutes doivent être finies et > 1")
    return o


def overround(odds) -> np.ndarray:
    """Somme des probabilités brutes 1/o par ligne. 1.05 = marge d'environ 5 %."""
    return (1.0 / _as_2d(odds)).sum(axis=1)


def proportional(odds) -> np.ndarray:
    """Retrait de marge proportionnel (normalisation) : p_i = (1/o_i) / somme_j (1/o_j).

    À IMPLÉMENTER PAR NAJIB. Indications :
      1. convertir avec _as_2d(odds) ;
      2. calculer les probabilités brutes 1/o ;
      3. diviser chaque ligne par sa somme (attention à l'axe et au broadcasting numpy).
    Les tests sont dans tests/core/test_odds.py : `make test` doit passer sans "skipped".
    """
    raise NotImplementedError("proportional() : à implémenter (voir docstring)")


def _shin_solve(odds) -> tuple[np.ndarray, np.ndarray]:
    q = 1.0 / _as_2d(odds)
    Q = q.sum(axis=1, keepdims=True)

    def probs(z: np.ndarray) -> np.ndarray:
        z = z[:, None]
        return (np.sqrt(z**2 + 4.0 * (1.0 - z) * q**2 / Q) - z) / (2.0 * (1.0 - z))

    lo = np.zeros(q.shape[0])
    hi = np.full(q.shape[0], 0.999)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        too_big = probs(mid).sum(axis=1) > 1.0
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    z = np.where(Q[:, 0] > 1.0, 0.5 * (lo + hi), 0.0)
    p = probs(z)
    return p / p.sum(axis=1, keepdims=True), z  # renormalise l'erreur résiduelle de bissection


def shin(odds) -> np.ndarray:
    """Méthode de Shin. Pour une proportion d'initiés z, la probabilité de l'issue i vaut

        p_i(z) = ( sqrt(z^2 + 4 (1 - z) * q_i^2 / Q) - z ) / ( 2 (1 - z) ),

    avec q_i = 1/o_i et Q = somme des q_i. On cherche z dans [0, 1) tel que somme p_i(z) = 1.
    La somme est décroissante en z, ce qui permet une bissection vectorisée sur toutes les lignes.
    Si l'overround est <= 1 (possible pour un best-price multi-books), z = 0 : on revient au
    proportionnel, Shin n'a pas de sens sans marge.
    """
    return _shin_solve(odds)[0]


def shin_z(odds) -> np.ndarray:
    """Proportion d'initiés implicite z de Shin, par ligne : un diagnostic de la "méfiance"
    du book. On s'attend à un z plus élevé dans les ligues moins liquides."""
    return _shin_solve(odds)[1]


def power(odds) -> np.ndarray:
    """Méthode power : p_i = q_i^k avec k tel que somme p_i = 1 (k > 1 si overround > 1)."""
    q = 1.0 / _as_2d(odds)
    lo = np.full(q.shape[0], 0.2)
    hi = np.full(q.shape[0], 5.0)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        too_big = (q ** mid[:, None]).sum(axis=1) > 1.0  # somme décroît avec k
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    p = q ** (0.5 * (lo + hi))[:, None]
    return p / p.sum(axis=1, keepdims=True)


METHODS = {"proportional": proportional, "shin": shin, "power": power}


def implied_probabilities(odds, method: str = "shin") -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"Méthode inconnue {method!r}, choisir parmi {list(METHODS)}")
    return METHODS[method](odds)
