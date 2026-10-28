"""Test de fumée du dashboard : chaque page s'exécute sans erreur sur des données simulées."""

import os
from pathlib import Path

import pytest
import yaml

from mvm.core import backtest
from mvm.core.io import load_config, write_table
from mvm.sports.foot import betting

from .sports.foot.simulate import simulate

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "mvm" / "app" / "dashboard.py"
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def sim_config(tmp_path_factory):
    try:
        backtest.kelly_fraction(0.5, 2.2)
    except NotImplementedError:
        pytest.skip("kelly_fraction() pas encore implémentée")
    d = tmp_path_factory.mktemp("dash")
    ev, od, _ = simulate(n_seasons=13, n_teams=10, seed=7)
    write_table(ev, d / "events.parquet")
    write_table(od, d / "odds.parquet")
    cfg = load_config("configs/foot.yaml")
    cfg["paths"] = {"processed": str(d), "reports": str(d)}
    betting.predict_all(cfg)
    res = betting.run_backtest(cfg, "tuning")
    res["summary"].to_csv(d / "backtest_tuning_summary.csv", index=False)
    res["bets"].to_csv(d / "backtest_tuning_bets.csv", index=False)
    res["weekly"].to_csv(d / "backtest_tuning_bankroll.csv", index=False)
    path = d / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


PAGES = ["Le duel", "La journée", "Ton club", "Et si tu avais parié ?", "Coulisses"]
LAB_VIEWS = ["Log-loss match par match", "Calibration", "Backtest détaillé", "Méthodologie"]


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


@pytest.mark.parametrize("view", LAB_VIEWS)
def test_lab_view_renders_without_exception(sim_config, view):
    at = _app(sim_config)
    at.radio(key="page").set_value("Coulisses").run()
    at.radio(key="lab_view").set_value(view).run()
    assert not at.exception, at.exception


def test_team_codes_and_kits():
    from mvm.app import ui

    assert ui.team_code("Man United") == "MUN"
    assert ui.team_code("Man City") == "MCI"
    assert ui.team_code("FC Koln") == "KOL"
    assert ui.team_code("Bayern Munich") == "BAY"
    assert ui.team_code("Real Madrid") == "RMA"
    (h, _), (a, _) = ui.kits("Arsenal", "Arsenal")
    assert h != a  # deux équipes d'un même match n'ont jamais la même couleur
