"""Saison 2026-27 en direct : le vrai test hors échantillon.

    make live      (à relancer chaque semaine, idéalement le jeudi ou le vendredi)

1. Re-télécharge les CSV de la saison en cours. C'est la seule saison qu'on réécrit : chaque
   version est tracée (URL, date, SHA-256) dans data/raw/foot/live_manifest.json.
2. Télécharge fixtures.csv (matchs à venir et leurs cotes) dans un snapshot horodaté, jamais
   écrasé : c'est notre propre relevé des prix au moment où le modèle prédit.
3. Reconstruit les tables canoniques (saison en cours comprise).
4. Rejoue les modèles, paramètres FIGÉS avant la saison, jusqu'au dernier match joué, puis prédit
   les matchs à venir.
5. Ajoute ces prédictions au journal reports/live/predictions_log.csv (on n'y efface jamais rien).
   Versionné dans git, il prouve que chaque prédiction existait avant le coup d'envoi.

Pinnacle n'est plus publié par football-data depuis 2025-26. En direct, la référence est donc la
moyenne du marché (référence de repli du protocole), moins exigeante que Pinnacle : battre la
moyenne est une barre plus basse, et les résultats doivent être lus avec cette réserve.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from mvm.core.io import load_config, read_json, read_table, sha256_file, write_json, write_table
from mvm.sports.foot import build, ingest, parse, quality, teams

log = logging.getLogger("mvm.foot.live")
OUTCOMES = ("H", "D", "A")


def forward_season(cfg: dict) -> str:
    return cfg["evaluation"]["forward"]


def fixtures_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["raw"]) / "fixtures"


def live_reports(cfg: dict) -> Path:
    return Path(cfg["paths"]["reports"]) / "live"


def upcoming_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed"]) / "upcoming.parquet"


def upcoming_odds_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed"]) / "upcoming_odds.parquet"


# --- 1 et 2 : téléchargements ------------------------------------------------------------------


def _fetch(session: requests.Session, url: str, timeout: float) -> bytes | None:
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException as exc:
        log.warning("Échec réseau %s : %s", url, exc)
        return None
    if resp.status_code != 200 or len(resp.content) < 100:
        log.info("Indisponible (%s) : %s", resp.status_code, url)
        return None
    return resp.content


def download_live(cfg: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    raw = Path(cfg["paths"]["raw"])
    year = int(forward_season(cfg)[:4])
    src = cfg["source"]
    session = requests.Session()
    manifest_path = raw / "live_manifest.json"
    history = read_json(manifest_path) if manifest_path.exists() else []
    stamp = now.isoformat(timespec="seconds")

    def record(path: Path, url: str, kind: str):
        history.append(
            {
                "path": str(path),
                "url": url,
                "kind": kind,
                "fetched_at": stamp,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    n_ok = 0
    for league in cfg["leagues"]:
        url = f"{src['base_url']}/{ingest.season_code(year)}/{league}.csv"
        content = _fetch(session, url, src["timeout_s"])
        if content is None:
            continue
        dest = ingest.raw_path(raw, league, year)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        record(dest, url, "season")
        n_ok += 1

    snap = None
    content = _fetch(session, cfg["live"]["fixtures_url"], src["timeout_s"])
    if content is not None:
        snap = fixtures_dir(cfg) / f"fixtures_{now:%Y%m%dT%H%M%SZ}.csv"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(content)
        record(snap, cfg["live"]["fixtures_url"], "fixtures")
    write_json(history, manifest_path)
    return {"season_files": n_ok, "fixtures_snapshot": str(snap) if snap else None}


def latest_fixtures(cfg: dict) -> Path | None:
    snaps = sorted(fixtures_dir(cfg).glob("fixtures_*.csv"))
    return snaps[-1] if snaps else None


# --- matchs à venir -----------------------------------------------------------------------------


def parse_fixtures(
    path: Path, cfg: dict, played_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fixtures.csv (toutes ligues mélangées) -> (matchs à venir, cotes au format long)."""
    try:
        df = parse.read_csv_robust(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), pd.DataFrame()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "Div" not in df.columns:
        raise ValueError(f"{path}: colonne Div absente ({list(df.columns)[:10]})")
    aliases = build._load_aliases(cfg["paths"].get("team_aliases", ""))
    season = forward_season(cfg)
    rows, odds = [], []
    for league, g in df.groupby(df["Div"].str.strip()):
        if league not in cfg["leagues"]:
            continue
        meta = cfg["leagues"][league]
        home = parse._pick(g, "home_team").str.strip()
        away = parse._pick(g, "away_team").str.strip()
        time = parse._pick(g, "time")
        m = pd.DataFrame(
            {
                "league": league,
                "tier": meta["tier"],
                "season": season,
                "sport": "foot",
                "kickoff_date": parse.parse_dates(parse._pick(g, "date")),
                "kickoff_time": time.str.strip() if time is not None else None,
                "home_name": [teams.canonical_name(t, meta["country"], aliases) for t in home],
                "away_name": [teams.canonical_name(t, meta["country"], aliases) for t in away],
                "home_team_id": [teams.team_id(t, meta["country"], aliases) for t in home],
                "away_team_id": [teams.team_id(t, meta["country"], aliases) for t in away],
            },
            index=g.index,
        ).dropna(subset=["kickoff_date", "home_name", "away_name"])
        m["event_id"] = [
            parse.make_event_id(league, d, h, a)
            for d, h, a in zip(m.kickoff_date, m.home_team_id, m.away_team_id, strict=True)
        ]
        mapping, _ = parse.classify_odds_columns(list(g.columns))
        p = parse.ParsedFile(
            league=league,
            season=season,
            matches=m,
            odds_wide=g.loc[m.index, list(mapping)].apply(pd.to_numeric, errors="coerce"),
            odds_columns=mapping,
        )
        rows.append(m)
        odds.append(parse.odds_to_long(p, m["event_id"]))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    fx = pd.concat(rows, ignore_index=True)
    fx = fx[~fx.event_id.isin(played_ids)].drop_duplicates("event_id")
    od = pd.concat(odds, ignore_index=True)
    od = od[od.event_id.isin(set(fx.event_id))]
    od, _ = quality.quarantine_odds(
        od,
        len(OUTCOMES),
        cfg["quality"]["overround_min"],
        cfg["quality"]["overround_max"],
        cfg["quality"].get("overround_min_best_price", 0.90),
    )
    return fx.sort_values(["kickoff_date", "league", "event_id"]).reset_index(drop=True), od


# --- 4 et 5 : prédictions et journal ------------------------------------------------------------


def append_log(cfg: dict, upcoming: pd.DataFrame, ahead: pd.DataFrame, snapshot: str, now) -> Path:
    path = live_reports(cfg) / "predictions_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = upcoming.set_index("event_id")[
        ["league", "kickoff_date", "kickoff_time", "home_name", "away_name"]
    ]
    rows = ahead.join(meta, on="event_id").assign(
        logged_at=now.isoformat(timespec="seconds"), fixtures_snapshot=Path(snapshot).name
    )
    rows["kickoff_date"] = pd.to_datetime(rows["kickoff_date"]).dt.strftime("%Y-%m-%d")
    cols = [
        "logged_at",
        "fixtures_snapshot",
        "event_id",
        "league",
        "kickoff_date",
        "kickoff_time",
        "home_name",
        "away_name",
        "model",
        "p_H",
        "p_D",
        "p_A",
    ]
    rows[cols].round(5).to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def refresh(cfg: dict, played: pd.DataFrame, now: datetime | None = None) -> dict:
    """Étapes 4 et 5, à partir des matchs joués (table events) et du dernier snapshot."""
    from mvm.sports.foot import betting

    now = now or datetime.now(UTC)
    snap = latest_fixtures(cfg)
    upcoming, up_odds = (
        parse_fixtures(snap, cfg, set(played.event_id)) if snap else (pd.DataFrame(), None)
    )
    if len(upcoming):
        # fixtures.csv peut contenir des matchs déjà joués mais pas encore saisis dans le fichier
        # de la saison : on ne prédit que ce qui est postérieur au dernier match connu.
        upcoming = upcoming[upcoming.kickoff_date >= played.kickoff_date.max()]
        upcoming = upcoming.reset_index(drop=True)
    _, ahead = betting.predict_all(cfg, upcoming if len(upcoming) else None)
    info = {"n_upcoming": len(upcoming)}
    info["n_played_forward"] = int((played.season == forward_season(cfg)).sum())
    if len(upcoming):
        proc = Path(cfg["paths"]["processed"])
        write_table(upcoming, upcoming_path(cfg))
        write_table(up_odds, upcoming_odds_path(cfg))
        write_table(ahead, proc / "upcoming_predictions.parquet")
        info["logged"] = str(append_log(cfg, upcoming, ahead, str(snap), now))
    return info


def run(cfg: dict, download: bool = True, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    info = download_live(cfg, now) if download else {}
    out = build.build(cfg)
    build.write_outputs(out, cfg)
    return {**info, **refresh(cfg, out["events"], now)}


def read_log(cfg: dict) -> pd.DataFrame:
    path = live_reports(cfg) / "predictions_log.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["kickoff_date"])
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    return df


def logged_before_kickoff(log_df: pd.DataFrame, model: str = "ens") -> pd.DataFrame:
    """Dernière prédiction journalisée la VEILLE du match au plus tard (une par match)."""
    if log_df.empty:
        return log_df
    d = log_df[log_df.model == model]
    d = d[d.logged_at.dt.tz_convert(None).dt.normalize() < d.kickoff_date]
    return d.sort_values("logged_at").groupby("event_id").tail(1).set_index("event_id")


def load_upcoming(cfg: dict) -> dict[str, pd.DataFrame] | None:
    proc = Path(cfg["paths"]["processed"])
    paths = [upcoming_path(cfg), upcoming_odds_path(cfg), proc / "upcoming_predictions.parquet"]
    if not all(p.exists() for p in paths):
        return None
    up, od, pr = (read_table(p) for p in paths)
    return {"fixtures": up, "odds": od, "predictions": pr}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/foot.yaml")
    ap.add_argument("--offline", action="store_true", help="sans téléchargement")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    info = run(load_config(args.config), download=not args.offline)
    print(
        f"\nSaison {forward_season(load_config(args.config))} : {info['n_played_forward']} matchs "
        f"joués, {info['n_upcoming']} matchs à venir prédits."
    )
    if info.get("logged"):
        print(f"Journal : {info['logged']} (à committer pour horodater les prédictions)")
    print("Ensuite : make app")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
