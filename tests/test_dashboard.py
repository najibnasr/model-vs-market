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


@pytest.mark.parametrize("page", ["Match par match", "Calibration", "Backtest", "Méthodologie"])
def test_page_renders_without_exception(sim_config, page):
    os.environ["MVM_CONFIG"] = sim_config
    at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
