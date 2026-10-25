"""Étape (b) : mesurer le marché AVANT de construire le moindre modèle.

Produit reports/market_baseline_foot.csv et un résumé console. Quatre questions :
  1. Quel est le niveau à battre ? (log-loss / RPS de la clôture Pinnacle, par ligue et saison)
  2. La clôture est-elle plus informative que le pré-match ? (elle devrait l'être)
  3. La méthode de retrait de marge change-t-elle le score du marché ? (choix de la référence)
  4. La clôture moyenne (market_avg) peut-elle remplacer Pinnacle quand Pinnacle disparaît ?

Usage : python -m mvm.sports.foot.market_baseline --config configs/foot.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mvm.core import metrics
from mvm.core.io import load_config, read_table
from mvm.core.market import market_probabilities
from mvm.core.odds import METHODS
from mvm.core.splits import week_key

OUTCOMES = ("H", "D", "A")
SOURCES = [  # (nom, bookmaker, timing)
    ("pin_close", "pinnacle", "closing"),
    ("pin_pre", "pinnacle", "pre_match"),
    ("avg_close", "market_avg", "closing"),
    ("avg_pre", "market_avg", "pre_match"),
]


def _available_methods() -> list[str]:
    ok = []
    for name, fn in METHODS.items():
        try:
            fn([[2.0, 3.4, 3.9]])
            ok.append(name)
        except NotImplementedError:
            print(f"[info] méthode '{name}' pas encore implémentée : ignorée")
    return ok


def score_frame(events: pd.DataFrame, probs: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = events.merge(probs, left_on="event_id", right_index=True)
    p = df[[f"p_{o}" for o in OUTCOMES]].to_numpy()
    y = df["result"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    out = df[["event_id", "league", "tier", "season", "kickoff_date"]].copy()
    for name, fn in metrics.SCORES.items():
        out[f"{name}_{tag}"] = fn(p, y)
    out[f"overround_{tag}"] = df["overround"].to_numpy()
    out[f"shin_z_{tag}"] = df["shin_z"].to_numpy()
    return out.set_index("event_id")


def run(cfg: dict) -> pd.DataFrame:
    proc = Path(cfg["paths"]["processed"])
    events = read_table(proc / "events.parquet")
    odds = read_table(proc / "odds.parquet")
    ev_cfg = cfg["evaluation"]
    events = events[events.season >= ev_cfg["eval_first"]]

    methods = _available_methods()
    frames = []
    for tag, book, timing in SOURCES:
        frames.append(score_frame(events, market_probabilities(odds, book, timing, "shin"), tag))
    for m in methods:  # sensibilité au retrait de marge, sur la référence uniquement
        if m != "shin":
            pr = market_probabilities(odds, "pinnacle", "closing", m)
            frames.append(score_frame(events, pr, f"pin_close_{m}").filter(like="log_loss"))

    base = frames[0]
    for f in frames[1:]:
        base = base.join(f.drop(columns=base.columns, errors="ignore"), how="outer")
    meta = events.set_index("event_id")[["league", "tier", "season", "kickoff_date"]]
    base = meta.join(base.drop(columns=meta.columns, errors="ignore"), how="inner")
    base["week"] = week_key(base["kickoff_date"])
    return base


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(("log_loss", "rps", "overround", "shin_z"))]
    g = df.groupby(["league", "season"])[cols].mean()
    g["n"] = df.groupby(["league", "season"]).size()
    return g.round(4).reset_index()


def paired(df: pd.DataFrame, a: str, b: str, metric: str = "log_loss") -> dict:
    sub = df[[f"{metric}_{a}", f"{metric}_{b}", "week"]].dropna()
    r = metrics.paired_difference(sub[f"{metric}_{a}"], sub[f"{metric}_{b}"], sub["week"])
    return {"a": a, "b": b, "metric": metric, **r}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/foot.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    df = run(cfg)
    rep = Path(cfg["paths"]["reports"])
    rep.mkdir(parents=True, exist_ok=True)
    summary = summarize(df)
    summary.to_csv(rep / "market_baseline_foot.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== Clôture Pinnacle (Shin), moyenne par tier ===")
    print(
        df.groupby("tier")[
            ["log_loss_pin_close", "rps_pin_close", "overround_pin_close", "shin_z_pin_close"]
        ]
        .mean()
        .round(4)
    )
    print("\n=== Comparaisons appariées (d < 0 : la première source fait mieux) ===")
    tests = [
        paired(df, "pin_close", "pin_pre"),
        paired(df, "pin_close", "avg_close"),
        paired(df, "avg_close", "avg_pre"),
    ]
    tests += [
        paired(df, "pin_close", f"pin_close_{m}")
        for m in ("proportional", "power")
        if f"log_loss_pin_close_{m}" in df.columns
    ]
    print(pd.DataFrame(tests).round(5).to_string(index=False))
    print(f"\nDétail par ligue et saison : {rep / 'market_baseline_foot.csv'}")


if __name__ == "__main__":
    main()
