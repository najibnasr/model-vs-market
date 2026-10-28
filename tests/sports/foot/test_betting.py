"""Contrôles positif et négatif du backtest, sur un championnat simulé."""

import pytest

from mvm.core import backtest
from mvm.core.io import load_config, write_table
from mvm.sports.foot import betting

from .simulate import simulate


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    try:
        backtest.kelly_fraction(0.5, 2.2)
    except NotImplementedError:
        pytest.skip("kelly_fraction() pas encore implémentée")
    d = tmp_path_factory.mktemp("sim")
    ev, od, _ = simulate(n_seasons=13, n_teams=12, seed=5)
    write_table(ev, d / "events.parquet")
    write_table(od, d / "odds.parquet")
    cfg = load_config("configs/foot.yaml")
    cfg["paths"] = {"processed": str(d), "reports": str(d)}
    cfg["models"]["dc"]["params"]["max_age_days"] = 1460
    betting.predict_all(cfg)
    return betting.run_backtest(cfg, "tuning")["summary"].set_index("strategy")


def test_oracle_control_shows_positive_clv(result):
    oracle = result.loc["CONTRÔLE + oracle (clôture)"]
    assert oracle.clv_ci_low > 0  # le backtest sait détecter un vrai edge


def test_base_rate_control_shows_negative_clv(result):
    base = result.loc["CONTRÔLE - fréquences de base"]
    assert base.clv_ci_high < 0  # aucune information : on paie la marge
