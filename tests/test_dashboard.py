"""Test de fumée du dashboard : chaque page s'exécute sans erreur sur des données simulées,
avec et sans saison en cours."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mvm.core import backtest
from mvm.core.io import load_config, write_table
from mvm.sports.foot import betting, live

from .sports.foot.simulate import simulate, write_fixtures_csv

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "mvm" / "app" / "dashboard.py"
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
CUTOFF = pd.Timestamp("2026-10-26")


def _setup(d: Path, with_live: bool) -> str:
    if with_live:
        ev, od, _ = simulate(n_seasons=22, n_teams=10, seed=7)
        future = ev[(ev.season == "2026-2027") & (ev.kickoff_date >= CUTOFF)]
        played = ev.drop(future.index)
    else:
        ev, od, _ = simulate(n_seasons=13, n_teams=10, seed=7)
        played, future = ev, ev.iloc[:0]
    write_table(played, d / "events.parquet")
    write_table(od[od.event_id.isin(played.event_id)], d / "odds.parquet")
    cfg = load_config("configs/foot.yaml")
    cfg["paths"] = {"processed": str(d), "reports": str(d), "raw": str(d / "raw")}
    if with_live:
        snap = d / "raw" / "fixtures" / "fixtures_20261029T090000Z.csv"
        snap.parent.mkdir(parents=True)
        write_fixtures_csv(future, od, snap)
        live.refresh(cfg, played, datetime(2026, 10, 29, 9, tzinfo=UTC))
    else:
        betting.predict_all(cfg)
    res = betting.run_backtest(cfg, "tuning")
    res["summary"].to_csv(d / "backtest_tuning_summary.csv", index=False)
    res["bets"].to_csv(d / "backtest_tuning_bets.csv", index=False)
    res["weekly"].to_csv(d / "backtest_tuning_bankroll.csv", index=False)
    path = d / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module", params=["live", "no_live"])
def sim_config(request, tmp_path_factory):
    try:
        backtest.kelly_fraction(0.5, 2.2)
    except NotImplementedError:
        pytest.skip("kelly_fraction() pas encore implémentée")
    return _setup(tmp_path_factory.mktemp(request.param), request.param == "live")


PAGES = ["This week", "Season 2026-27", "Your club", "Track record", "Backstage"]
LAB_VIEWS = ["Log-loss match by match", "Calibration", "Detailed backtest", "Method"]


def _app(sim_config):
    os.environ["MVM_CONFIG"] = sim_config
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(sim_config, page):
    at = _app(sim_config)
    at.radio(key="page").set_value(page).run()
    assert not at.exception, at.exception


def test_track_record_what_if_renders(sim_config):
    at = _app(sim_config)
    at.radio(key="page").set_value("Track record").run()
    at.radio(key="track_view").set_value("What if you had bet?").run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("view", LAB_VIEWS)
def test_lab_view_renders_without_exception(sim_config, view):
    at = _app(sim_config)
    at.radio(key="page").set_value("Backstage").run()
    at.radio(key="lab_view").set_value(view).run()
    assert not at.exception, at.exception


def test_holdout_season_is_never_shown(sim_config):
    os.environ["MVM_CONFIG"] = sim_config
    from mvm.app import data

    data.config.clear()
    data.load_data.clear()
    data.model_frame.clear()
    assert "2025-2026" not in set(data.model_frame("ens").season)


def test_picks_are_saved_and_last_one_wins(tmp_path, monkeypatch):
    from mvm.app import data

    monkeypatch.setattr(data, "picks_path", lambda: tmp_path / "my_picks.csv")
    data.save_pick("abc", 0)
    data.save_pick("abc", 2)
    data.save_pick("xyz", None)
    picks = data.load_picks().set_index("event_id")["pick"]
    assert picks["abc"] == 2 and picks["xyz"] == -1


def test_team_codes_and_kits():
    from mvm.app import ui

    assert ui.team_code("Man United") == "MUN"
    assert ui.team_code("Man City") == "MCI"
    assert ui.team_code("FC Koln") == "KOL"
    assert ui.team_code("Bayern Munich") == "BAY"
    assert ui.team_code("Real Madrid") == "RMA"
    (h, _), (a, _) = ui.kits("Arsenal", "Arsenal")
    assert h != a  # deux équipes d'un même match n'ont jamais la même couleur


def test_calls_close_at_kick_off_london_time():
    from mvm.app import data

    now = datetime(2026, 10, 29, 19, 50, tzinfo=UTC)  # 19:50 à Londres (heure d'hiver)
    assert data.kicked_off("2026-10-29", "19:45", now)
    assert not data.kicked_off("2026-10-29", "20:00", now)
    assert data.kicked_off("2026-10-29", None, now)  # sans heure : fermé dès le jour J
    summer = datetime(2026, 8, 15, 14, 30, tzinfo=UTC)  # 15:30 à Londres (heure d'été)
    assert data.kicked_off("2026-08-15", "15:00", summer)
