"""Tests de Dixon-Coles : formules, récupération des vrais paramètres, comparaison à Elo."""

import numpy as np
import pandas as pd
import pytest

from mvm.core import elo as core_elo
from mvm.core.splits import walk_forward
from mvm.sports.foot.models import dixon_coles as dcm
from mvm.sports.foot.models.baseline import LeagueBaseRates

from .simulate import HOME_ADV, simulate


def _needs_weights():
    try:
        dcm.time_decay_weights(np.array([0.0]), 0.01)
    except NotImplementedError:
        pytest.skip("time_decay_weights() pas encore implémentée : c'est l'exercice de Najib")


def test_time_decay_weights():
    _needs_weights()
    w = dcm.time_decay_weights(np.array([0.0, 100.0, 200.0]), 0.0069315)  # demi-vie 100 jours
    np.testing.assert_allclose(w, [1.0, 0.5, 0.25], rtol=1e-4)


def test_tau_only_touches_low_scores():
    x = np.array([0, 0, 1, 1, 2, 3])
    y = np.array([0, 1, 0, 1, 2, 0])
    tau = dcm.dc_tau(x, y, 1.5, 1.2, -0.1)
    np.testing.assert_allclose(tau, [1 + 0.18, 1 - 0.15, 1 - 0.12, 1.1, 1.0, 1.0])


def test_outcome_probs_are_valid_and_match_plain_poisson_when_rho_is_zero():
    lam, mu = np.array([1.6, 0.8]), np.array([1.1, 2.0])
    p = dcm.outcome_probs(lam, mu, 0.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] > p[0, 2] and p[1, 2] > p[1, 0]
    rng = np.random.default_rng(0)
    x, y = rng.poisson(1.6, 400_000), rng.poisson(1.1, 400_000)
    assert p[0, 1] == pytest.approx((x == y).mean(), abs=0.003)


def test_negative_rho_increases_draws():
    lam = mu = np.array([1.3])
    assert dcm.outcome_probs(lam, mu, -0.15)[0, 1] > dcm.outcome_probs(lam, mu, 0.0)[0, 1]


@pytest.fixture(scope="module")
def sim():
    return simulate(n_seasons=8, n_teams=16, seed=4)


def test_recovers_home_advantage_and_team_ranking(sim):
    _needs_weights()
    ev, _, strength = sim
    m = dcm.DixonColes(xi=0.0001, l2=0.5, max_age_days=10_000)
    m.update(ev)
    fx = ev.tail(1).drop(columns=["home_score", "away_score", "result"])
    m.predict(fx.assign(kickoff_date=ev.kickoff_date.max() + pd.Timedelta(days=1)))
    cf = m.countries["eng"]
    assert np.mean(list(cf.home.values())) == pytest.approx(HOME_ADV, abs=0.07)
    t = m.team_table("eng")
    truth = pd.Series(strength).reindex(t.index)
    # attaque et défense jouent symétriquement dans le simulateur
    assert t.attack.corr(truth, method="spearman") > 0.7
    assert t.defence.corr(truth, method="spearman") > 0.7


def test_dixon_coles_beats_base_rates_on_simulated_league(sim):
    _needs_weights()
    ev, _, _ = sim
    seasons = ["2010-2011", "2011-2012"]
    dc = walk_forward(dcm.DixonColes(xi=0.003, l2=2.0), ev, seasons)
    base = walk_forward(LeagueBaseRates(), ev, seasons)
    ll = {
        k: -np.log(v[["p_H", "p_D", "p_A"]].to_numpy()[np.arange(len(v)), v.outcome_idx]).mean()
        for k, v in {"dc": dc, "base": base}.items()
    }
    assert ll["dc"] < ll["base"] - 0.02
    try:
        core_elo.expected_home_score(1500, 1500, 0)
    except NotImplementedError:
        return


def test_linear_pool_averages_on_common_matches():
    from mvm.sports.foot.models.ensemble import linear_pool

    a = pd.DataFrame(
        {
            "event_id": ["x", "y"],
            "week": [1, 1],
            "p_H": [0.6, 0.2],
            "p_D": [0.2, 0.3],
            "p_A": [0.2, 0.5],
            "outcome_idx": [0, 2],
        }
    )
    b = pd.DataFrame(
        {
            "event_id": ["y", "z"],
            "week": [1, 2],
            "p_H": [0.4, 0.3],
            "p_D": [0.3, 0.3],
            "p_A": [0.3, 0.4],
            "outcome_idx": [2, 1],
        }
    )
    out = linear_pool({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    assert list(out.event_id) == ["y"]
    np.testing.assert_allclose(out[["p_H", "p_D", "p_A"]].to_numpy(), [[0.3, 0.3, 0.4]])
