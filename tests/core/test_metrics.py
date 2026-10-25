import numpy as np
import pytest

from mvm.core import metrics


def test_perfect_and_uniform_forecasts():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert metrics.log_loss(perfect, y) == pytest.approx([0, 0, 0], abs=1e-12)
    assert metrics.brier(perfect, y).sum() == 0
    uni = np.full((3, 3), 1 / 3)
    assert metrics.log_loss(uni, y) == pytest.approx(np.log(3))


def test_rps_respects_outcome_order():
    # Résultat = victoire extérieure. Mettre la masse sur le nul est moins grave que sur
    # la victoire à domicile : seule la RPS fait cette différence.
    y = np.array([2])
    on_draw = np.array([[0.1, 0.8, 0.1]])
    on_home = np.array([[0.8, 0.1, 0.1]])
    assert metrics.rps(on_draw, y)[0] < metrics.rps(on_home, y)[0]
    assert metrics.brier(on_draw, y)[0] == pytest.approx(metrics.brier(on_home, y)[0])


def test_rps_known_value():
    # p = (0.5, 0.3, 0.2), résultat H : ((0.5-1)^2 + (0.8-1)^2) / 2 = 0.145
    assert metrics.rps([[0.5, 0.3, 0.2]], [0])[0] == pytest.approx(0.145)


def test_rejects_non_normalised_probabilities():
    with pytest.raises(ValueError):
        metrics.log_loss([[0.5, 0.5, 0.5]], [0])


def test_calibrated_forecast_has_small_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 50_000)
    y = rng.uniform(size=p.size) < p
    assert metrics.ece(p, y) < 0.01
    assert metrics.ece(np.clip(p + 0.1, 0, 1), y) > 0.05  # biaisé : détecté


def test_paired_difference_detects_real_edge_but_not_noise():
    rng = np.random.default_rng(1)
    n = 4000
    weeks = np.repeat(np.arange(200), n // 200)
    base = rng.normal(1.0, 0.3, n)
    same = metrics.paired_difference(base + rng.normal(0, 0.05, n), base, weeks)
    assert not same["a_better"]
    better = metrics.paired_difference(base - 0.02, base, weeks)
    assert better["a_better"] and better["mean_diff"] == pytest.approx(-0.02)
