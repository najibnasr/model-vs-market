"""Métriques de qualité des probabilités, agnostiques au sport.

Conventions : `probs` est un tableau (n, K) dont chaque ligne somme à 1, `outcome` un tableau
d'entiers dans [0, K) donnant l'indice de l'issue réalisée. Pour le 1X2, l'ordre est (H, D, A) :
il est ORDONNÉ (le nul est "entre" les deux victoires), ce qui compte pour la RPS.

Pourquoi trois métriques :
- log-loss : métrique principale. Strictement propre (on minimise son espérance en annonçant ses
  vraies croyances) et directement liée à la finance : la croissance espérée du log de la
  bankroll d'un parieur Kelly est la différence de log-loss entre le marché et lui.
  Défaut : très sévère avec une proba quasi nulle sur une issue qui se réalise.
- Brier : erreur quadratique, bornée, plus tolérante aux probabilités extrêmes.
- RPS (ranked probability score) : tient compte de l'ordre des issues. Prédire 60 % domicile
  quand le résultat est nul est moins grave que quand l'extérieur gagne. Standard en foot
  (Constantinou & Fenton, 2012).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def _check(probs, outcome) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcome, dtype=int)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("probs doit être (n, K) et outcome de longueur n")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Chaque ligne de probs doit sommer à 1")
    if y.min(initial=0) < 0 or y.max(initial=0) >= p.shape[1]:
        raise ValueError("outcome hors de [0, K)")
    return p, y


def one_hot(outcome, k: int) -> np.ndarray:
    y = np.asarray(outcome, dtype=int)
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def log_loss(probs, outcome) -> np.ndarray:
    """Log-loss PAR MATCH (vecteur). La moyenne se fait à l'extérieur, pour pouvoir comparer
    deux modèles match par match (tests appariés)."""
    p, y = _check(probs, outcome)
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))


def brier(probs, outcome) -> np.ndarray:
    p, y = _check(probs, outcome)
    return ((p - one_hot(y, p.shape[1])) ** 2).sum(axis=1)


def rps(probs, outcome) -> np.ndarray:
    p, y = _check(probs, outcome)
    k = p.shape[1]
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_o = np.cumsum(one_hot(y, k), axis=1)[:, :-1]
    return ((cum_p - cum_o) ** 2).sum(axis=1) / (k - 1)


SCORES = {"log_loss": log_loss, "brier": brier, "rps": rps}


def reliability_table(p_outcome, happened, n_bins: int = 10) -> pd.DataFrame:
    """Données du reliability diagram pour UNE issue (ex. victoire à domicile).

    Bins à effectifs égaux (quantiles) plutôt qu'à largeur fixe : les probabilités de nul sont
    concentrées entre 0.2 et 0.33, des bins fixes de 0.1 en mettraient presque tout dans 2 bins.
    """
    df = pd.DataFrame({"p": np.asarray(p_outcome, float), "y": np.asarray(happened, float)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    t = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), freq=("y", "mean"), n=("y", "size")
    )
    # IC binomial à 95 % sur la fréquence observée : un écart dans l'IC n'est pas significatif
    se = np.sqrt(t["freq"] * (1 - t["freq"]) / t["n"])
    t["ci_low"], t["ci_high"] = t["freq"] - 1.96 * se, t["freq"] + 1.96 * se
    return t.reset_index(drop=True)


def ece(p_outcome, happened, n_bins: int = 10) -> float:
    """Expected calibration error : écart moyen |prédit - observé|, pondéré par les effectifs."""
    t = reliability_table(p_outcome, happened, n_bins)
    return float(np.average((t["mean_pred"] - t["freq"]).abs(), weights=t["n"]))


def paired_difference(
    loss_a, loss_b, clusters, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Compare deux prévisions sur LES MÊMES matchs : d_i = loss_a_i - loss_b_i.

    d < 0 en moyenne signifie que A fait mieux que B. L'IC est obtenu par bootstrap PAR CLUSTER
    (typiquement la semaine) : les matchs d'un même week-end ne sont pas indépendants (même
    information disponible, mêmes chocs). Un bootstrap match par match sous-estimerait la
    variance et donnerait des "edges significatifs" trop facilement.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    c = pd.Series(np.asarray(clusters)).astype("category").cat.codes.to_numpy()
    sums = np.bincount(c, weights=d)
    counts = np.bincount(c).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(len(d)),
        "n_clusters": int(len(sums)),
        "a_better": bool(hi < 0),  # règle de décision figée : tout l'IC sous zéro
    }
