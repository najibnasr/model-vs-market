import pytest

from mvm.core import elo


def _skip_if_todo():
    try:
        elo.expected_home_score(1500, 1500, 0)
    except NotImplementedError:
        pytest.skip("expected_home_score() pas encore implémentée : c'est l'exercice de Najib")


def test_equal_ratings_without_home_advantage_is_a_coin_flip():
    _skip_if_todo()
    assert elo.expected_home_score(1500, 1500, 0) == pytest.approx(0.5)


def test_400_points_means_ten_to_one():
    _skip_if_todo()
    assert elo.expected_home_score(1900, 1500, 0) == pytest.approx(10 / 11)
    assert elo.expected_home_score(1500, 1900, 0) == pytest.approx(1 / 11)


def test_home_advantage_acts_like_extra_rating_points():
    _skip_if_todo()
    assert elo.expected_home_score(1500, 1500, 60) == pytest.approx(
        elo.expected_home_score(1560, 1500, 0)
    )


def test_update_is_zero_sum_and_moves_towards_the_result():
    _skip_if_todo()
    r = elo.EloRatings(k=20, home_advantage=0)
    r.update("a", "b", score=1.0)  # victoire de a à 50/50 : +10 / -10
    assert r.get("a") == pytest.approx(1510) and r.get("b") == pytest.approx(1490)
    assert r.get("a") + r.get("b") == pytest.approx(3000)


def test_goal_margin_multiplier_is_sublinear():
    m = [elo.goal_margin_multiplier(g) for g in range(0, 6)]
    assert m[0] == m[1] == 1.0 and m[2] == 1.5
    assert m == sorted(m) and m[5] < 5 * m[1]
