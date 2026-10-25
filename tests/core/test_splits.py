import numpy as np
import pandas as pd
import pytest

from mvm.core import splits


def _events():
    dates = pd.to_datetime(
        ["2011-05-07", "2011-05-08", "2012-08-18", "2012-08-19", "2012-08-25", "2012-08-26"]
    )
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(6)],
            "season": ["2010-2011"] * 2 + ["2012-2013"] * 4,
            "kickoff_date": dates,
            "home_team_id": list("ABABAB"),
            "away_team_id": list("CDCDCD"),
            "home_score": [1, 0, 2, 1, 0, 3],
            "away_score": [0, 0, 1, 1, 2, 0],
            "result": ["H", "D", "H", "D", "A", "H"],
        }
    )


class Spy:
    """Modèle espion : enregistre ce qu'il voit pour vérifier la causalité."""

    def __init__(self):
        self.last_seen = pd.Timestamp.min
        self.violations = 0

    def update(self, played):
        self.last_seen = max(self.last_seen, played.kickoff_date.max())

    def predict(self, fixtures):
        assert not set(splits.RESULT_COLUMNS) & set(fixtures.columns)
        # tout ce qui a été vu doit être dans une semaine STRICTEMENT antérieure
        if (
            splits.week_key(pd.Series([self.last_seen]))[0]
            >= splits.week_key(fixtures.kickoff_date).min()
        ):
            self.violations += 1
        return np.full((len(fixtures), 3), 1 / 3)


def test_walk_forward_is_causal_and_hides_results():
    spy = Spy()
    out = splits.walk_forward(spy, _events(), ["2012-2013"])
    assert spy.violations == 0
    assert len(out) == 4  # burn-in non noté
    assert list(out.outcome_idx) == [0, 1, 2, 0]


def test_same_weekend_matches_cannot_see_each_other():
    # Samedi 18/08 et dimanche 19/08 : même semaine, donc prédits ensemble, avant toute mise à jour
    out = splits.walk_forward(Spy(), _events(), ["2012-2013"])
    assert out.groupby("week").size().tolist() == [2, 2]


class Cheater:
    def update(self, played):
        pass

    def predict(self, fixtures):
        return np.eye(3)[fixtures["result"].map({"H": 0, "D": 1, "A": 2})]


def test_a_model_reading_results_crashes():
    with pytest.raises(KeyError):
        splits.walk_forward(Cheater(), _events(), ["2012-2013"])
