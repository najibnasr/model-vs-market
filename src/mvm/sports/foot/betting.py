"""Étape (e) : backtest de value betting, CLV et test de mélange, pour le foot.

  1. Calculer et mettre en cache les prédictions de tous les modèles (2012-13 à 2025-26) :
         python -m mvm.sports.foot.betting predict
  2. Backtest sur la période de réglage (exploration autorisée) :
         python -m mvm.sports.foot.betting backtest --split tuning
  3. Backtest sur le test (une seule fois, journalisé) :
         python -m mvm.sports.foot.betting backtest --split test --confirm-test

Choix de mise en oeuvre :
- On parie au prix PRÉ-MATCH (relevé des jours avant le match) : c'est le seul prix réellement
  disponible au moment où le modèle a son information. Parier à la clôture serait possible, mais
  on n'aurait alors plus de CLV à mesurer.
- Trois prix : Pinnacle pré-match (principal, prudent), moyenne du marché, meilleure cote
  (borne optimiste : suppose qu'on a des comptes partout et qu'aucun book ne nous limite).
- Deux contrôles pour vérifier que le backtest lui-même est juste :
  * contrôle POSITIF ("oracle") : parier avec les probabilités de clôture, qui contiennent de
    l'information future. Le backtest DOIT montrer un gain et une CLV positive, sinon il est faux.
  * contrôle NÉGATIF : parier avec les fréquences de base (aucune information sur les équipes).
    On doit perdre à peu près la marge du bookmaker.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mvm.core import backtest, blend, metrics
from mvm.core.io import load_config, read_table, write_table
from mvm.core.market import market_probabilities, odds_matrix
from mvm.core.splits import hide_results
from mvm.sports.foot.models.baseline import LeagueBaseRates
from mvm.sports.foot.models.ensemble import linear_pool
from mvm.sports.foot.run_model import MODELS, eval_seasons, load, run_walk_forward, season_range

OUTCOMES = ("H", "D", "A")
PCOLS = [f"p_{o}" for o in OUTCOMES]


def predictions_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed"]) / "predictions.parquet"


def predict_all(
    cfg: dict, upcoming: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward de tous les modèles de 2012-13 jusqu'à la dernière saison disponible (saison en
    cours comprise), puis, si `upcoming` est fourni, prédiction des matchs à venir par les modèles
    dans leur état final (ils ont vu tous les matchs joués, rien d'autre)."""
    events, _ = load(cfg)
    ev = cfg["evaluation"]
    seasons = eval_seasons(events, ev["eval_first"], ev.get("forward", ev["holdout"]))
    members = {m: cls(**cfg["models"][m]["params"]) for m, cls in MODELS.items()}
    members["base"] = LeagueBaseRates()
    preds, ahead = {}, {}
    for m, model in members.items():
        print(f"walk-forward {m}...", flush=True)
        preds[m] = run_walk_forward(model, events, seasons)
        if upcoming is not None and len(upcoming):
            ahead[m] = pd.DataFrame(model.predict(hide_results(upcoming)), columns=PCOLS).assign(
                event_id=upcoming["event_id"].to_numpy()
            )
    weights = cfg["models"]["ens"]["weights"]
    preds["ens"] = linear_pool({m: preds[m] for m in weights}, weights)
    long = pd.concat([df.assign(model=m) for m, df in preds.items()], ignore_index=True)
    write_table(long, predictions_path(cfg))
    print(f"{len(long)} prédictions écrites dans {predictions_path(cfg)}")
    if not ahead:
        return long, pd.DataFrame()
    total = sum(weights.values())
    ens = sum(weights[m] / total * ahead[m][PCOLS].to_numpy() for m in weights)
    ahead["ens"] = pd.DataFrame(ens / ens.sum(axis=1, keepdims=True), columns=PCOLS).assign(
        event_id=upcoming["event_id"].to_numpy()
    )
    up = pd.concat([df.assign(model=m) for m, df in ahead.items()], ignore_index=True)
    return long, up[["event_id", "model", *PCOLS]]


def _wide(mat: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return mat.rename(columns={o: f"{prefix}_{o}" for o in OUTCOMES})


def build_frame(cfg: dict, split: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Échantillon commun : matchs du split avec prédictions, cotes pré-match des 3 sources et
    probabilités de clôture Pinnacle."""
    events, odds = load(cfg)
    first, last = season_range(cfg["evaluation"], split)
    preds = read_table(predictions_path(cfg))
    meta = events.set_index("event_id")[["league", "season", "tier"]]
    meta = meta[(meta.season >= first) & (meta.season <= last)]

    bet = cfg["betting"]
    ref = bet["clv_reference"]
    close = market_probabilities(odds, ref["bookmaker"], ref["timing"], ref["margin_method"])
    close = close[PCOLS].rename(columns={c: "pc_" + c[2:] for c in PCOLS})
    prices = {
        b: _wide(odds_matrix(odds, b, bet["bet_timing"]), "odds")
        for b in [bet["bet_bookmaker"], *bet["price_sensitivity"]]
    }
    pre_shin = market_probabilities(odds, bet["bet_bookmaker"], bet["bet_timing"], "shin")[PCOLS]

    common = meta.index.intersection(close.index).intersection(pre_shin.index)
    for p in prices.values():
        common = common.intersection(p.index)
    by_model = {}
    for m, g in preds.groupby("model"):
        g = g.set_index("event_id")
        by_model[m] = g.loc[g.index.intersection(common)]
    common = common.intersection(by_model["ens"].index)
    base = meta.loc[common].join(close.loc[common])
    base = base.join(pre_shin.loc[common].add_prefix("pre_"))
    return base, {"prices": prices, "preds": by_model}


def frame_for(base, parts, model: str, book: str) -> pd.DataFrame:
    p = parts["preds"][model].loc[base.index, [*PCOLS, "week", "outcome_idx"]]
    return base.join(p).join(parts["prices"][book].loc[base.index]).reset_index()


def oracle_frame(base, parts, book: str) -> pd.DataFrame:
    df = frame_for(base, parts, "ens", book)
    for o in OUTCOMES:
        df[f"p_{o}"] = df[f"pc_{o}"]  # probabilités de CLÔTURE : information future, exprès
    return df


def blend_test(cfg: dict, split: str, boot: dict) -> pd.DataFrame:
    """Poids ajusté sur la période de réglage, évalué sur le split demandé."""
    rows = []
    base_t, parts_t = build_frame(cfg, "tuning")
    base_s, parts_s = (base_t, parts_t) if split == "tuning" else build_frame(cfg, split)
    model = cfg["betting"]["model"]
    for name, cols in (
        ("clôture", [f"pc_{o}" for o in OUTCOMES]),
        ("pré-match", [f"pre_p_{o}" for o in OUTCOMES]),
    ):
        dt = frame_for(base_t, parts_t, model, cfg["betting"]["bet_bookmaker"])
        w, _ = blend.fit_weight(dt[cols].to_numpy(), dt[PCOLS].to_numpy(), dt["outcome_idx"])
        ds = frame_for(base_s, parts_s, model, cfg["betting"]["bet_bookmaker"])
        pm, pmod, y = ds[cols].to_numpy(), ds[PCOLS].to_numpy(), ds["outcome_idx"].to_numpy()
        ll_mkt = metrics.log_loss(pm, y)
        ll_mix = metrics.log_loss(blend.mix(pm, pmod, w), y)
        r = metrics.paired_difference(ll_mix, ll_mkt, ds["week"], **boot)
        rows.append(
            {
                "market": f"Pinnacle {name}",
                "w_from_tuning": w,
                "log_loss_market": ll_mkt.mean(),
                "log_loss_blend": ll_mix.mean(),
                "diff": r["mean_diff"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "blend_better": r["a_better"],
            }
        )
    return pd.DataFrame(rows)


def run_backtest(cfg: dict, split: str) -> dict:
    bet = cfg["betting"]
    boot = {k: cfg["evaluation"]["bootstrap"][k] for k in ("n_boot", "seed", "alpha")}
    init = bet["initial_bankroll"]
    main = backtest.Strategy(**bet["strategy"])
    base, parts = build_frame(cfg, split)
    model, book = bet["model"], bet["bet_bookmaker"]

    rows, main_bets, main_weekly = [], None, None

    def add(label, df, strat):
        s, staked, weekly = backtest.run(df, OUTCOMES, strat, init, **boot)
        rows.append({"strategy": label, **s})
        return staked, weekly

    main_bets, main_weekly = add(
        f"PRINCIPALE {model} @ {book}", frame_for(base, parts, model, book), main
    )
    for b in bet["price_sensitivity"]:
        add(f"{model} @ {b}", frame_for(base, parts, model, b), main)
    for e in bet["edge_grid"]:
        if e != main.min_edge:
            strat = backtest.Strategy(**{**bet["strategy"], "min_edge": e})
            add(f"{model} @ {book}, edge > {e:.2f}", frame_for(base, parts, model, book), strat)
    for m in ("elo", "dc"):
        add(f"{m} @ {book}", frame_for(base, parts, m, book), main)
    add("CONTRÔLE + oracle (clôture)", oracle_frame(base, parts, book), main)
    add("CONTRÔLE - fréquences de base", frame_for(base, parts, "base", book), main)

    by_league = (
        main_bets.groupby("league")
        .agg(n=("won", "size"), roi_flat=("flat_return", "mean"), clv=("clv", "mean"))
        .round(4)
    )
    by_odds = (
        main_bets.assign(odds_bin=pd.cut(main_bets["odds"], [1, 2, 3, 5, 10]))
        .groupby("odds_bin", observed=True)
        .agg(n=("won", "size"), roi_flat=("flat_return", "mean"), clv=("clv", "mean"))
        .round(4)
    )
    return {
        "summary": pd.DataFrame(rows),
        "blend": blend_test(cfg, split, boot),
        "by_league": by_league,
        "by_odds": by_odds,
        "bets": main_bets,
        "weekly": main_weekly,
        "n_matches": len(base),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("action", choices=["predict", "backtest"])
    ap.add_argument("--split", default="tuning", choices=["tuning", "test"])
    ap.add_argument("--config", default="configs/foot.yaml")
    ap.add_argument("--confirm-test", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.action == "predict":
        predict_all(cfg)
        return
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("Le test ne se regarde qu'une fois. Relancer avec --confirm-test.")
    if not predictions_path(cfg).exists():
        raise SystemExit("Prédictions absentes : lancer d'abord `make predict`.")

    res = run_backtest(cfg, args.split)
    rep = Path(cfg["paths"]["reports"])
    rep.mkdir(parents=True, exist_ok=True)
    stem = f"backtest_{args.split}"
    res["summary"].to_csv(rep / f"{stem}_summary.csv", index=False)
    res["blend"].to_csv(rep / f"{stem}_blend.csv", index=False)
    res["bets"].to_csv(rep / f"{stem}_bets.csv", index=False)
    res["weekly"].to_csv(rep / f"{stem}_bankroll.csv", index=False)
    if args.split == "test":
        log = rep / "test_log.csv"
        pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "model": "backtest",
                    "split": args.split,
                    "params": json.dumps(cfg["betting"], sort_keys=True),
                    "log_loss_model": np.nan,
                    "log_loss_market": np.nan,
                }
            ]
        ).to_csv(log, mode="a", header=not log.exists(), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cols = [
        "strategy",
        "n_bets",
        "hit_rate",
        "avg_odds",
        "roi_flat",
        "roi_ci_low",
        "roi_ci_high",
        "clv",
        "clv_ci_low",
        "clv_ci_high",
        "kelly_return",
        "max_drawdown",
    ]
    print(f"\n=== Backtest | {args.split} | {res['n_matches']} matchs ===")
    print("ROI et CLV par unité misée (mise fixe), IC à 95 % par semaine ; Kelly = capital final.")
    print(res["summary"][cols].round(4).to_string(index=False))
    print("\n=== Test de mélange : le modèle ajoute-t-il de l'information au prix ? ===")
    print(res["blend"].round(5).to_string(index=False))
    print("\n=== Stratégie principale par ligue (exploratoire) ===")
    print(res["by_league"].to_string())
    print("\n=== Stratégie principale par tranche de cote (exploratoire) ===")
    print(res["by_odds"].to_string())
    print(f"\nDétails : {rep}/{stem}_*.csv")


if __name__ == "__main__":
    main()
