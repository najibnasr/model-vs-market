"""Tests des modèles sur un championnat simulé dont on connaît la vérité."""

import numpy as np
import pandas as pd
import pytest

from mvm.core import elo as core_elo
from mvm.core import evaluate
from mvm.core.splits import walk_forward
from mvm.sports.foot.models.baseline import LeagueBaseRates
from mvm.sports.foot.models.elo import FootElo, fit_ordered_logit, ordered_logit_probs

from .simulate import simulate

EVAL = ["2011-2012", "2012-2013", "2013-2014", "2014-2015"]


def _needs_elo():
    try:
        core_elo.expected_home_score(1500, 1500, 0)
    except NotImplementedError:
        pytest.skip("expected_home_score() pas encore implémentée")


@pytest.fixture(scope="module")
def sim():
    return simulate(n_seasons=10, seed=1)


def test_ordered_logit_recovers_true_parameters():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1.5, 20_000)
    p = ordered_logit_probs(d, 0.6, -0.9, 0.4)
    y = np.array([rng.choice(3, p=row) for row in p])
    beta, c1, c2 = fit_ordered_logit(d, y)
    assert beta == pytest.approx(0.6, abs=0.03)
    assert c1 == pytest.approx(-0.9, abs=0.05) and c2 == pytest.approx(0.4, abs=0.05)


def test_ordered_logit_probabilities_are_valid_and_monotone():
    d = np.linspace(-5, 5, 11)
    p = ordered_logit_probs(d, 0.5, -0.8, 0.3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all(np.diff(p[:, 0]) > 0) and np.all(np.diff(p[:, 2]) < 0)


def test_base_rates_match_league_frequencies(sim):
    ev, _, _ = sim
    m = LeagueBaseRates(strength=0)
    m.update(ev)
    p = m.predict(ev.head(1))[0]
    freq = ev[ev.league == ev.league.iloc[0]].result.value_counts(normalize=True)
    assert p[0] == pytest.approx(freq["H"]) and p[1] == pytest.approx(freq["D"])


def test_elo_beats_base_rates_and_market_beats_elo(sim):
    _needs_elo()
    ev, odds, _ = sim
    from mvm.sports.foot.run_model import market_preds

    preds = {
        "elo": walk_forward(FootElo(k=20, home_advantage=60), ev, EVAL),
        "base": walk_forward(LeagueBaseRates(), ev, EVAL),
        "market": market_preds(
            odds, ev, {"bookmaker": "pinnacle", "timing": "closing", "margin_method": "shin"}
        ),
    }
    meta = ev.set_index("event_id")[["league", "season"]].assign(
        week=ev.set_index("event_id")["kickoff_date"]
    )
    sc = evaluate.score_sources(evaluate.common_sample(preds, meta), list(preds))
    ll = sc[[f"log_loss_{s}" for s in preds]].mean()
    assert ll["log_loss_elo"] < ll["log_loss_base"] - 0.02
    assert ll["log_loss_market"] < ll["log_loss_elo"]


def test_elo_ranks_teams_like_their_true_strength(sim):
    _needs_elo()
    ev, _, strength = sim
    m = FootElo(k=20, home_advantage=60)
    m.update(ev)
    r = m.ratings_table().set_index("team_id")["rating"]
    truth = pd.Series(strength).reindex(r.index)
    assert r.corr(truth, method="spearman") > 0.7


def test_promoted_team_keeps_its_rating(sim):
    _needs_elo()
    ev, _, _ = sim
    m = FootElo(k=20, home_advantage=60, season_regression=0.0)
    first = ev[ev.season == "2005-2006"]
    m.update(first)
    promoted = set(ev[(ev.season == "2006-2007") & (ev.league == "E0")].home_team_id) - set(
        first[first.league == "E0"].home_team_id
    )
    t = next(iter(promoted))
    before = m.elo.get(t)
    m.predict(ev[(ev.season == "2006-2007") & (ev.home_team_id == t)].head(1))
    assert m.elo.get(t) == pytest.approx(before)  # pas remis à zéro en changeant de division


def test_new_team_starts_below_league_average(sim):
    _needs_elo()
    ev, _, _ = sim
    m = FootElo(k=20, home_advantage=60, new_team_offset=100)
    m.update(ev[ev.season == "2005-2006"])
    fx = ev.head(1).assign(home_team_id="eng:newcomer", season="2005-2006")
    m.predict(fx.drop(columns=["home_score", "away_score", "result"]))
    assert m.elo.get("eng:newcomer") == pytest.approx(m._league_mean(fx.league.iloc[0]) - 100)
