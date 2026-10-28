import numpy as np
import pandas as pd
import pytest

from mvm.core import backtest, blend

OUT = ("H", "D", "A")


def _needs_kelly():
    try:
        backtest.kelly_fraction(np.array([0.5]), np.array([2.2]))
    except NotImplementedError:
        pytest.skip("kelly_fraction() pas encore implémentée : c'est l'exercice de Najib")


def _frame():
    # match 1 : edge sur H (0.55 * 2.0 - 1 = 0.10) ; match 2 : aucun edge ; match 3 : edge énorme
    # mais sur une cote de 15, au-dessus de max_odds.
    return pd.DataFrame(
        {
            "event_id": ["m1", "m2", "m3"],
            "week": [1, 1, 2],
            "outcome_idx": [0, 1, 2],
            "p_H": [0.55, 0.40, 0.30],
            "p_D": [0.25, 0.30, 0.30],
            "p_A": [0.20, 0.30, 0.40],
            "odds_H": [2.0, 2.3, 2.0],
            "odds_D": [3.4, 3.1, 3.3],
            "odds_A": [4.5, 3.0, 15.0],
            "pc_H": [0.52, 0.42, 0.45],
            "pc_D": [0.27, 0.30, 0.30],
            "pc_A": [0.21, 0.28, 0.25],
        }
    )


def test_kelly_fraction_known_values():
    _needs_kelly()
    # p = 0.5 à cote 2.2 : edge 0.10, gain net 1.2 -> f = 0.10 / 1.2
    assert backtest.kelly_fraction(np.array([0.5]), np.array([2.2]))[0] == pytest.approx(0.1 / 1.2)
    assert backtest.kelly_fraction(np.array([0.5]), np.array([2.0]))[0] == pytest.approx(0.0)
    assert backtest.kelly_fraction(np.array([0.4]), np.array([2.0]))[0] < 0  # pari défavorable


def test_selects_best_edge_above_threshold_and_below_max_odds():
    _needs_kelly()
    bets = backtest.select_bets(_frame(), OUT, backtest.Strategy(min_edge=0.05, max_odds=10))
    assert list(bets.event_id) == ["m1"]
    b = bets.iloc[0]
    assert b.outcome == "H" and b.edge == pytest.approx(0.10) and bool(b.won)
    assert b.flat_return == pytest.approx(1.0)
    assert b.clv == pytest.approx(2.0 * 0.52 - 1)  # cote prise x proba de clôture - 1


def test_stake_is_fractional_kelly_and_capped():
    _needs_kelly()
    s = backtest.Strategy(min_edge=0.0, kelly_fraction=0.25, max_stake=0.02)
    bets = backtest.select_bets(_frame(), OUT, s)
    full = backtest.kelly_fraction(np.array([0.55]), np.array([2.0]))[0]  # 0.10
    assert bets.set_index("event_id").loc["m1", "stake_frac"] == pytest.approx(
        min(0.25 * full, 0.02)
    )


def test_weekly_exposure_cap_and_bankroll_accounting():
    _needs_kelly()
    bets = pd.DataFrame(
        {
            "week": [1, 1, 1],
            "stake_frac": [0.2, 0.2, 0.2],
            "won": [True, False, False],
            "odds": [2.0, 3.0, 3.0],
        }
    )
    staked, weekly = backtest.simulate_bankroll(
        bets, backtest.Strategy(max_week_exposure=0.3), 1000
    )
    np.testing.assert_allclose(staked["stake"], [100, 100, 100])  # 60 % ramenés à 30 %
    assert weekly["bankroll"].iloc[-1] == pytest.approx(1000 + 100 - 200)


def test_blend_weight_is_zero_when_model_is_noise_and_positive_when_informative():
    rng = np.random.default_rng(0)
    n = 30_000
    truth = rng.dirichlet([4, 3, 3], n)
    y = np.array([rng.choice(3, p=p) for p in truth])
    noise = rng.dirichlet([4, 3, 3], n)
    market = 0.7 * truth + 0.3 * noise  # marché imparfait
    w_noise, _ = blend.fit_weight(market, rng.dirichlet([4, 3, 3], n), y)
    w_info, _ = blend.fit_weight(market, truth, y)
    assert w_noise <= 0.03
    assert w_info >= 0.2
