"""Orchestration de l'étape (a) : raw -> schéma canonique + rapports.

Usage :
    python -m mvm.sports.foot.build --config configs/foot.yaml [--download] [--force]

Le build est déterministe : à fichiers raw identiques (même SHA-256), sorties identiques.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from mvm.core import schema
from mvm.core.io import load_config, write_json, write_table
from mvm.sports.foot import ingest, parse, quality, teams

log = logging.getLogger("mvm.foot.build")
N_OUTCOMES = len(schema.MARKET_OUTCOMES["1x2"])


def _load_aliases(path: str | Path) -> dict:
    p = Path(path)
    if not path or not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build(cfg: dict) -> dict[str, pd.DataFrame]:
    raw_dir = Path(cfg["paths"]["raw"])
    aliases = _load_aliases(cfg["paths"].get("team_aliases", ""))

    event_frames, odds_frames, team_pairs, file_log = [], [], [], []
    for year in range(cfg["seasons"]["first"], cfg["seasons"]["last"] + 1):
        season = ingest.season_label(year)
        for league, meta in cfg["leagues"].items():
            path = ingest.raw_path(raw_dir, league, year)
            if not path.exists():
                file_log.append({"league": league, "season": season, "status": "absent"})
                continue
            p = parse.parse_file(path, league, season)
            m = p.matches
            country = meta["country"]
            m["home_team_id"] = [teams.team_id(t, country, aliases) for t in m.home_team_raw]
            m["away_team_id"] = [teams.team_id(t, country, aliases) for t in m.away_team_raw]
            m["event_id"] = [
                parse.make_event_id(league, d, h, a)
                for d, h, a in zip(m.kickoff_date, m.home_team_id, m.away_team_id, strict=True)
            ]
            m["result"] = parse.derive_result(m.home_score, m.away_score)
            m["sport"] = "foot"
            m["tier"] = meta["tier"]

            event_frames.append(m)
            odds_frames.append(parse.odds_to_long(p, m["event_id"]))
            team_pairs.append(
                pd.DataFrame(
                    {"country": country, "raw_name": pd.concat([m.home_team_raw, m.away_team_raw])}
                )
            )
            file_log.append(
                {
                    "league": league,
                    "season": season,
                    "status": "ok",
                    "rows_raw": p.n_rows_raw,
                    "rows_unplayed": p.n_rows_unplayed,
                    "n_odds_columns": len(p.odds_columns),
                    "unrecognized": p.unrecognized_columns,
                }
            )

    if not event_frames:
        raise SystemExit(f"Aucun fichier dans {raw_dir}. Lancer d'abord `make data`.")

    events_all = (
        pd.concat(event_frames, ignore_index=True)
        .sort_values(["kickoff_date", "league", "event_id"])
        .reset_index(drop=True)
    )
    raw_res = events_all["result_raw"].astype("string")
    events = schema.coerce(events_all, schema.EVENTS_SCHEMA)
    quality.hard_checks(events)

    odds_raw = pd.concat(odds_frames, ignore_index=True)
    odds_clean, quarantined = quality.quarantine_odds(
        odds_raw,
        N_OUTCOMES,
        cfg["quality"]["overround_min"],
        cfg["quality"]["overround_max"],
        cfg["quality"].get("overround_min_best_price", 0.90),
    )
    odds = schema.coerce(odds_clean, schema.ODDS_SCHEMA)
    odds = odds.sort_values(["event_id", "bookmaker", "timing", "outcome"]).reset_index(drop=True)

    teams_tbl = schema.coerce(
        teams.build_teams_table(pd.concat(team_pairs), aliases), schema.TEAMS_SCHEMA
    )

    schema.validate(events, schema.EVENTS_SCHEMA, "events")
    schema.validate(odds, schema.ODDS_SCHEMA, "odds")
    schema.validate(teams_tbl, schema.TEAMS_SCHEMA, "teams")

    return {
        "events": events,
        "odds": odds,
        "teams": teams_tbl,
        "quarantine": quarantined,
        "coverage": quality.coverage_report(events, odds, N_OUTCOMES),
        "issues": pd.DataFrame(quality.soft_checks(events, raw_res)),
        "collisions": teams.slug_collisions(teams_tbl),
        "files": pd.DataFrame(file_log),
    }


def write_outputs(out: dict[str, pd.DataFrame], cfg: dict) -> None:
    proc = Path(cfg["paths"]["processed"])
    rep = Path(cfg["paths"]["reports"])
    for name in ("events", "odds", "teams"):
        write_table(out[name], proc / f"{name}.parquet")
    rep.mkdir(parents=True, exist_ok=True)
    out["coverage"].to_csv(rep / "coverage_foot.csv", index=False)
    out["teams"].to_csv(rep / "teams_foot.csv", index=False)
    out["quarantine"].to_csv(rep / "quarantine_odds_foot.csv", index=False)
    write_json(
        {
            "n_events": len(out["events"]),
            "n_odds_rows": len(out["odds"]),
            "n_teams": out["teams"].team_id.nunique(),
            "n_quarantined_rows": len(out["quarantine"]),
            "quarantine_reasons": out["quarantine"]["reason"].value_counts().to_dict()
            if len(out["quarantine"])
            else {},
            "issues": out["issues"].to_dict("records"),
            "slug_collisions": out["collisions"].to_dict("records"),
            "files": out["files"].to_dict("records"),
        },
        rep / "quality_foot.json",
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/foot.yaml")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-télécharge même si présent")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.download:
        ingest.download_all(cfg, force=args.force)
    out = build(cfg)
    write_outputs(out, cfg)
    log.info(
        "events=%d odds=%d teams=%d quarantaine=%d anomalies=%d",
        len(out["events"]),
        len(out["odds"]),
        out["teams"].team_id.nunique(),
        len(out["quarantine"]),
        len(out["issues"]),
    )


if __name__ == "__main__":
    main()
