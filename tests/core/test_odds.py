import numpy as np
import pytest

from mvm.core import odds

# Match type : favori à domicile, marge ~5 %
ODDS = np.array([[1.80, 3.60, 4.75], [1.25, 6.50, 12.0], [2.90, 3.20, 2.60]])


def _skip_if_todo(fn):
    try:
        fn(ODDS)
    except NotImplementedError:
        pytest.skip("proportional() pas encore implémentée : c'est l'exercice de Najib")


@pytest.mark.parametrize("method", ["shin", "power", "proportional"])
def test_rows_sum_to_one_and_are_valid(method):
    fn = odds.METHODS[method]
    _skip_if_todo(fn)
    p = fn(ODDS)
    assert p.shape == ODDS.shape
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert (p > 0).all() and (p < 1).all()


def test_proportional_exact_values():
    _skip_if_todo(odds.proportional)
    q = 1 / ODDS
    np.testing.assert_allclose(odds.proportional(ODDS), q / q.sum(axis=1, keepdims=True))


def test_proportional_accepts_a_single_match_and_two_outcomes():
    _skip_if_todo(odds.proportional)
    p = odds.proportional([1.5, 2.6])  # moneyline NBA : 2 issues, une seule ligne
    assert p.shape == (1, 2)
    np.testing.assert_allclose(p.sum(), 1.0)


@pytest.mark.parametrize("method", ["shin", "power"])
def test_shin_and_power_favour_the_favourite_vs_proportional(method):
    # Biais favori/outsider : ces méthodes retirent plus de marge à l'outsider, donc rendent
    # plus de probabilité au favori que la normalisation proportionnelle.
    q = 1 / ODDS
    prop = q / q.sum(axis=1, keepdims=True)
    p = odds.METHODS[method](ODDS)
    fav = prop.argmax(axis=1)
    longshot = prop.argmin(axis=1)
    rows = np.arange(len(ODDS))
    assert (p[rows, fav] > prop[rows, fav]).all()
    assert (p[rows, longshot] < prop[rows, longshot]).all()


def test_fair_odds_are_left_unchanged():
    fair = 1 / np.array([[0.5, 0.3, 0.2]])
    for m in ("shin", "power"):
        np.testing.assert_allclose(odds.METHODS[m](fair), [[0.5, 0.3, 0.2]], atol=1e-9)
    assert odds.shin_z(fair)[0] == pytest.approx(0.0, abs=1e-9)


def test_shin_z_positive_with_margin_and_grows_with_margin():
    low = odds.shin_z([[1.95, 3.50, 4.20]])[0]
    high = odds.shin_z([[1.80, 3.20, 3.90]])[0]
    assert 0 < low < high < 0.2


def test_invalid_odds_rejected():
    with pytest.raises(ValueError):
        odds.shin([[1.0, 3.0, 4.0]])
    with pytest.raises(ValueError):
        odds.implied_probabilities(ODDS, method="magic")
