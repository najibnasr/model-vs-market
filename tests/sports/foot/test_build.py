"""Test de bout en bout : raw (fixtures) -> tables canoniques + rapports."""

import pandas as pd
import pytest

from mvm.core import schema
from mvm.core.io import read_table
from mvm.sports.foot import build, quality


@pytest.fixture
def out(cfg):
    return build.build(cfg)


def test_tables_respect_schema(out):
    schema.validate(out["events"], schema.EVENTS_SCHEMA, "events")
    schema.validate(out["odds"], schema.ODDS_SCHEMA, "odds")


def test_event_counts(out):
    assert len(out["events"]) == 24
    assert out["events"].event_id.is_unique


def test_events_sorted_chronologically(out):
    # Pré-requis anti-leakage pour les étapes suivantes : l'ordre temporel est garanti en sortie.
    assert out["events"].kickoff_date.is_monotonic_increasing


def test_promoted_team_keeps_same_id_across_seasons(out):
    ev = out["events"]
    ids_0506 = set(ev[ev.season == "2005-2006"].home_team_id)
    ids_2021 = set(ev[ev.season == "2020-2021"].home_team_id)
    assert "eng:arsenal" in ids_0506 & ids_2021


def test_closing_odds_only_in_recent_season(out):
    od = out["odds"].merge(out["events"][["event_id", "season"]], on="event_id")
    closing_seasons = set(od.loc[od.timing == "closing", "season"])
    assert closing_seasons == {"2020-2021"}


def test_bad_odds_quarantined_not_corrected(out):
    q = out["quarantine"]
    assert set(q.reason) >= {"cote <= 1", "overround hors plage"}
    # Le triplet bet365 entier part en quarantaine, les autres books du même match restent.
    od = out["odds"]
    bad_events = set(q.loc[q.bookmaker == "bet365", "event_id"])
    assert len(bad_events) == 2
    kept = od[od.event_id.isin(bad_events)]
    assert "bet365" not in set(kept.bookmaker)
    assert "market_avg" in set(kept.bookmaker)
    assert (od.decimal_odds > 1).all()


def test_inconsistent_ftr_is_reported(out):
    checks = set(out["issues"].check)
    assert "FTR incohérent avec le score" in checks


def test_coverage_report(out):
    cov = out["coverage"].set_index("season")
    assert cov.loc["2020-2021", "cov_pinnacle_closing"] == 1.0
    assert cov.loc["2005-2006", "cov_pinnacle_closing"] == 0.0
    assert cov.loc["2005-2006", "share_with_time"] == 0.0
    assert 1.0 < cov.loc["2020-2021", "overround_pinnacle_pre_match"] < 1.1


def test_hard_check_catches_duplicates(out):
    ev = pd.concat([out["events"], out["events"].iloc[:1]], ignore_index=True)
    with pytest.raises(quality.DataQualityError):
        quality.hard_checks(ev)


def test_outputs_written_and_reloadable(out, cfg):
    build.write_outputs(out, cfg)
    events = read_table(f"{cfg['paths']['processed']}/events.parquet")
    pd.testing.assert_frame_equal(events, out["events"])


def test_build_is_deterministic(cfg):
    a, b = build.build(cfg), build.build(cfg)
    pd.testing.assert_frame_equal(a["events"], b["events"])
    pd.testing.assert_frame_equal(a["odds"], b["odds"])


def test_best_price_aggregate_below_one_is_not_quarantined(out):
    # Fixture 2005-06 : market_max a un overround de ~0.997 (arbitrage entre books), c'est réel.
    assert "market_max" not in set(out["quarantine"].bookmaker)
    assert "market_max" in set(out["odds"].bookmaker)
