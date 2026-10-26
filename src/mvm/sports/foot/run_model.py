"""Réglage et évaluation des modèles foot, selon docs/protocol.md.

  Réglage (tuning, 2012-13 à 2016-17) :
      python -m mvm.sports.foot.run_model tune --model elo
  Évaluation détaillée sur la période de réglage (autorisé autant qu'on veut) :
      python -m mvm.sports.foot.run_model eval --model elo --split tuning
  Évaluation sur le TEST (une seule fois par modèle final, journalisée) :
      python -m mvm.sports.foot.run_model eval --model elo --split test --confirm-test

Le test est verrouillé derrière --confirm-test, et chaque passage est inscrit dans
reports/test_log.csv avec les paramètres utilisés. Si on le regarde dix fois en changeant les
paramètres entre chaque, ça se verra.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mvm.core import evaluate
from mvm.core.io import load_config, read_table
from mvm.core.market import market_probabilities
from mvm.core.splits import walk_forward, week_key
from mvm.sports.foot.models.baseline import LeagueBaseRates
from mvm.sports.foot.models.elo import FootElo

MODELS = {"elo": FootElo}


def season_range(ev_cfg: dict, split: str) -> tuple[str, str]:
    return {
        "tuning": (ev_cfg["eval_first"], ev_cfg["tuning_last"]),
        "test": (ev_cfg["test_first"], ev_cfg["test_last"]),
        "holdout": (ev_cfg["holdout"], ev_cfg["holdout"]),
    }[split]


def load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc = Path(cfg["paths"]["processed"])
    return read_table(proc / "events.parquet"), read_table(proc / "odds.parquet")


def eval_seasons(events: pd.DataFrame, first: str, last: str) -> list[str]:
    return sorted(s for s in events.season.unique() if first <= s <= last)


def run_walk_forward(model, events: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    # on ne rejoue que jusqu'à la dernière saison utile : rien après n'est jamais vu
    return walk_forward(model, events[events.season <= max(seasons)], seasons)


def market_preds(odds: pd.DataFrame, events: pd.DataFrame, ref: dict) -> pd.DataFrame:
    mp = market_probabilities(odds, ref["bookmaker"], ref["timing"], ref["margin_method"])
    mp = mp[evaluate.PCOLS].reset_index()
    y = events.set_index("event_id")["result"].map({"H": 0, "D": 1, "A": 2})
    mp["outcome_idx"] = mp["event_id"].map(y)
    return mp.dropna(subset=["outcome_idx"])


def tune(cfg: dict, model_name: str) -> pd.DataFrame:
    events, _ = load(cfg)
    ev_cfg = cfg["evaluation"]
    seasons = eval_seasons(events, *season_range(ev_cfg, "tuning"))
    grid = cfg["models"][model_name]["grid"]
    base = cfg["models"][model_name]["params"]
    keys = list(grid)
    rows = []
    combos = list(itertools.product(*grid.values()))
    for i, values in enumerate(combos, 1):
        params = {**base, **dict(zip(keys, values, strict=True))}
        t0 = time.time()
        preds = run_walk_forward(MODELS[model_name](**params), events, seasons)
        sc = evaluate.score_sources(
            preds.rename(columns={c: f"m_{c}" for c in evaluate.PCOLS}), ["m"]
        )
        rows.append(
            {**params, "log_loss": sc["log_loss_m"].mean(), "rps": sc["rps_m"].mean(), "n": len(sc)}
        )
        print(
            f"[{i}/{len(combos)}] {params} -> log_loss={rows[-1]['log_loss']:.5f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
    res = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    out = Path(cfg["paths"]["reports"]) / f"tuning_{model_name}.csv"
    res.to_csv(out, index=False)
    print(f"\nMeilleurs réglages (période de réglage {seasons[0]} à {seasons[-1]}) :")
    print(res.head(10).to_string(index=False))
    print(f"\nTableau complet : {out}")
    return res


def evaluate_model(cfg: dict, model_name: str, split: str) -> dict[str, pd.DataFrame]:
    events, odds = load(cfg)
    ev_cfg = cfg["evaluation"]
    seasons = eval_seasons(events, *season_range(ev_cfg, split))
    params = cfg["models"][model_name]["params"]
    ref = ev_cfg["fallback_reference"] if split == "holdout" else ev_cfg["reference"]

    preds = {
        model_name: run_walk_forward(MODELS[model_name](**params), events, seasons),
        "base": run_walk_forward(LeagueBaseRates(), events, seasons),
        "market": market_preds(odds, events, ref),
    }
    meta = events.set_index("event_id")[["league", "tier", "season", "kickoff_date"]]
    meta = meta.assign(week=week_key(meta["kickoff_date"]))
    sc = evaluate.score_sources(evaluate.common_sample(preds, meta), list(preds))
    boot = {k: ev_cfg["bootstrap"][k] for k in ("n_boot", "seed", "alpha")}
    srcs = list(preds)
    return {
        "scored": sc,
        "overall": evaluate.summary(sc, srcs),
        "by_league": evaluate.summary(sc, srcs, ["tier", "league"]),
        "by_season": evaluate.summary(sc, srcs, ["season"]),
        "paired": pd.concat(
            [
                evaluate.paired_table(sc, [(model_name, "market"), (model_name, "base")], m, **boot)
                for m in ("log_loss", "rps")
            ]
        ),
        "paired_by_league": pd.concat(
            [
                evaluate.paired_table(g, [(model_name, "market")], "log_loss", **boot).assign(
                    league=lg
                )
                for lg, g in sc.groupby("league")
            ]
        ),
        "calibration": pd.concat(
            [evaluate.calibration_table(sc, s) for s in (model_name, "market")]
        ),
        "longshot": pd.concat(
            {s: evaluate.favourite_longshot(sc, s) for s in (model_name, "market")}
        ),
        "reliability": pd.concat(
            [evaluate.reliability_data(sc, s) for s in (model_name, "market")]
        ),
    }


def log_test_run(cfg: dict, model_name: str, split: str, overall: pd.DataFrame) -> None:
    path = Path(cfg["paths"]["reports"]) / "test_log.csv"
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "model": model_name,
                "split": split,
                "params": json.dumps(cfg["models"][model_name]["params"], sort_keys=True),
                "log_loss_model": overall[f"log_loss_{model_name}"].iloc[0],
                "log_loss_market": overall["log_loss_market"].iloc[0],
            }
        ]
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("action", choices=["tune", "eval"])
    ap.add_argument("--model", default="elo", choices=list(MODELS))
    ap.add_argument("--split", default="tuning", choices=["tuning", "test", "holdout"])
    ap.add_argument("--config", default="configs/foot.yaml")
    ap.add_argument("--confirm-test", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.action == "tune":
        tune(cfg, args.model)
        return
    if args.split != "tuning" and not args.confirm_test:
        raise SystemExit(
            "Le test et le holdout ne se regardent qu'une fois par modèle final. "
            "Relancer avec --confirm-test si c'est bien le cas."
        )

    res = evaluate_model(cfg, args.model, args.split)
    rep = Path(cfg["paths"]["reports"])
    stem = f"eval_{args.model}_{args.split}"
    for name in ("by_league", "by_season", "paired_by_league", "reliability"):
        res[name].to_csv(rep / f"{stem}_{name}.csv")
    if args.split != "tuning":
        log_test_run(cfg, args.model, args.split, res["overall"])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== {args.model} | {args.split} | n = {int(res['overall']['n'].iloc[0])} matchs ===")
    print(res["overall"].T.to_string(header=False))
    print("\n=== Comparaisons appariées (d < 0 : le modèle fait mieux) ===")
    print(res["paired"].round(5).to_string(index=False))
    print("\n=== Par ligue : modèle vs marché (log-loss) ===")
    print(
        res["paired_by_league"][["league", "mean_diff", "ci_low", "ci_high", "n"]]
        .round(5)
        .to_string(index=False)
    )
    print("\n=== Calibration (ECE par issue) ===")
    print(res["calibration"].to_string(index=False))
    print("\n=== Biais favori/outsider : prédit vs observé par tranche de probabilité ===")
    print(res["longshot"].to_string())
    print(f"\nDétails : {rep}/{stem}_*.csv")


if __name__ == "__main__":
    main()
