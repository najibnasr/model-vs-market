"""Comparaison d'une prévision au marché et au benchmark naïf, selon le protocole figé.

Entrée : un DataFrame de prédictions (sortie de walk_forward) par source, indexé par event_id,
avec p_H, p_D, p_A et outcome_idx. Tout est évalué sur l'ÉCHANTILLON COMMUN : les matchs pour
lesquels toutes les sources ont une prédiction. Comparer un modèle sur 10 000 matchs au marché
sur 9 000 autres ne veut rien dire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mvm.core import metrics

OUTCOMES = ("H", "D", "A")
PCOLS = [f"p_{o}" for o in OUTCOMES]


def common_sample(preds: dict[str, pd.DataFrame], meta: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par match présent partout ; colonnes <source>_p_H... + méta (ligue, semaine...)."""
    frames = []
    for name, df in preds.items():
        frames.append(df.set_index("event_id")[PCOLS].add_prefix(f"{name}_"))
    joined = pd.concat(frames, axis=1, join="inner")
    first = next(iter(preds.values())).set_index("event_id")["outcome_idx"]
    joined["outcome_idx"] = first.reindex(joined.index).astype(int)
    return joined.join(meta, how="inner")


def score_sources(df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    """Ajoute <metric>_<source> pour chaque source et chaque métrique."""
    out = df.copy()
    y = df["outcome_idx"].to_numpy()
    for s in sources:
        p = df[[f"{s}_{c}" for c in PCOLS]].to_numpy()
        for m, fn in metrics.SCORES.items():
            out[f"{m}_{s}"] = fn(p, y)
    return out


def summary(scored: pd.DataFrame, sources: list[str], by: list[str] | None = None) -> pd.DataFrame:
    cols = [f"{m}_{s}" for s in sources for m in metrics.SCORES]
    g = scored.groupby(by) if by else scored.assign(_all="all").groupby("_all")
    t = g[cols].mean()
    t["n"] = g.size()
    return t.round(4)


def paired_table(
    scored: pd.DataFrame, pairs: list[tuple[str, str]], metric: str = "log_loss", **boot
) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        r = metrics.paired_difference(
            scored[f"{metric}_{a}"], scored[f"{metric}_{b}"], scored["week"], **boot
        )
        rows.append({"a": a, "b": b, "metric": metric, **r})
    return pd.DataFrame(rows)


def calibration_table(scored: pd.DataFrame, source: str, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for k, o in enumerate(OUTCOMES):
        p = scored[f"{source}_p_{o}"].to_numpy()
        hit = (scored["outcome_idx"].to_numpy() == k).astype(float)
        rows.append(
            {
                "source": source,
                "outcome": o,
                "ece": metrics.ece(p, hit, n_bins),
                "mean_pred": p.mean(),
                "freq": hit.mean(),
            }
        )
    return pd.DataFrame(rows).round(4)


def reliability_data(scored: pd.DataFrame, source: str, n_bins: int = 10) -> pd.DataFrame:
    """Points des reliability diagrams, pour les figures du write-up et du dashboard."""
    parts = []
    for k, o in enumerate(OUTCOMES):
        t = metrics.reliability_table(
            scored[f"{source}_p_{o}"], (scored["outcome_idx"] == k).astype(float), n_bins
        )
        parts.append(t.assign(source=source, outcome=o))
    return pd.concat(parts, ignore_index=True)


def favourite_longshot(scored: pd.DataFrame, source: str) -> pd.DataFrame:
    """Écart moyen (prédit - observé) selon la probabilité annoncée : un modèle qui sur-estime
    les outsiders aura un écart positif dans les petites probabilités."""
    p = scored[[f"{source}_p_{o}" for o in OUTCOMES]].to_numpy().ravel()
    hit = np.eye(3)[scored["outcome_idx"].to_numpy()].ravel()
    bins = pd.cut(p, [0, 0.15, 0.3, 0.45, 0.6, 0.75, 1.0])
    t = pd.DataFrame({"bin": bins, "p": p, "hit": hit}).groupby("bin", observed=True)
    return t.agg(mean_pred=("p", "mean"), freq=("hit", "mean"), n=("p", "size")).round(4)
