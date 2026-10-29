"""Saison en direct : matchs à venir prédits par les modèles figés, puis journalisés."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from mvm.core.io import load_config, read_table, write_table
from mvm.sports.foot import live

from .simulate import simulate, write_fixtures_csv

CUTOFF = pd.Timestamp("2026-10-26")
NOW = datetime(2026, 10, 29, 9, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("live")
    ev, od, _ = simulate(n_seasons=22, n_teams=8, seed=11)  # 2005-06 à 2026-27
    future = ev[(ev.season == "2026-2027") & (ev.kickoff_date >= CUTOFF)]
    played = ev.drop(future.index)
    write_table(played, d / "events.parquet")
    write_table(od[od.event_id.isin(played.event_id)], d / "odds.parquet")
    snap = d / "raw" / "fixtures" / "fixtures_20261029T090000Z.csv"
    snap.parent.mkdir(parents=True)
    write_fixtures_csv(future, od, snap)
    cfg = load_config("configs/foot.yaml")
    cfg["paths"] = {"processed": str(d), "reports": str(d), "raw": str(d / "raw")}
    info = live.refresh(cfg, played, NOW)
    return cfg, info, played, future


def test_every_future_match_is_predicted(live_run):
    cfg, info, _, future = live_run
    assert info["n_upcoming"] == len(future)
    pr = read_table(cfg["paths"]["processed"] + "/upcoming_predictions.parquet")
    assert set(pr.model) == {"elo", "dc", "base", "ens"}
    p = pr[["p_H", "p_D", "p_A"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()


def test_fixture_teams_map_to_known_ids(live_run):
    cfg, _, played, _ = live_run
    up = read_table(cfg["paths"]["processed"] + "/upcoming.parquet")
    known = set(played.home_team_id) | set(played.away_team_id)
    assert set(up.home_team_id) <= known and set(up.away_team_id) <= known
    od = read_table(cfg["paths"]["processed"] + "/upcoming_odds.parquet")
    assert {"market_avg", "market_max"} <= set(od.bookmaker)


def test_log_is_append_only_and_dated_before_kickoff(live_run):
    cfg, info, _, future = live_run
    log1 = live.read_log(cfg)
    assert len(log1) == 4 * len(future)
    live.refresh(cfg, live_run[2], NOW)  # une seconde mise à jour ajoute, n'efface rien
    log2 = live.read_log(cfg)
    assert len(log2) == 2 * len(log1)
    assert log2.iloc[: len(log1)].equals(log1)
    before = live.logged_before_kickoff(log2)
    # seuls comptent les matchs joués au moins le lendemain de la prédiction
    assert len(before) == (future.kickoff_date > pd.Timestamp(NOW.date())).sum() > 0


def test_walk_forward_covers_the_live_season(live_run):
    cfg, _, played, _ = live_run
    preds = read_table(cfg["paths"]["processed"] + "/predictions.parquet")
    live_ids = set(played.loc[played.season == "2026-2027", "event_id"])
    assert live_ids <= set(preds.loc[preds.model == "ens", "event_id"])
