import pandas as pd
import pytest

from mvm.sports.foot import ingest, parse, teams


def test_season_codes():
    assert ingest.season_code(2005) == "0506"
    assert ingest.season_code(1999) == "9900"
    assert ingest.season_label(2020) == "2020-2021"


def test_dates_both_formats():
    s = pd.Series(["13/08/05", "12/09/2020", "n/a"])
    out = parse.parse_dates(s)
    assert out[0] == pd.Timestamp("2005-08-13")
    assert out[1] == pd.Timestamp("2020-09-12")
    assert pd.isna(out[2])


@pytest.mark.parametrize(
    "col, expected",
    [
        ("PSH", ("pinnacle", "pre_match", "H")),
        ("PSCH", ("pinnacle", "closing", "H")),
        ("AvgCD", ("market_avg", "closing", "D")),
        ("BbMxA", ("market_max", "pre_match", "A")),
        ("WHH", ("williamhill", "pre_match", "H")),
        ("WHCH", ("williamhill", "closing", "H")),
        ("BFECA", ("betfair_exchange", "closing", "A")),
        ("1XBH", ("1xbet", "pre_match", "H")),
    ],
)
def test_odds_column_mapping(col, expected):
    mapping, _ = parse.classify_odds_columns([col])
    assert mapping[col] == expected


def test_non_odds_columns_ignored():
    mapping, unrecognized = parse.classify_odds_columns(
        ["FTR", "HTR", "BbAH", "AHh", "FTHG", "PAHH", "MaxAHA", "PCAHH"]
    )
    assert mapping == {}
    assert unrecognized == []


def test_unknown_book_is_reported_not_guessed():
    mapping, unrecognized = parse.classify_odds_columns(["XYZH"])
    assert mapping == {}
    assert "XYZH" in unrecognized


def test_aggregator_duplicate_prefers_new_source():
    mapping, unrecognized = parse.classify_odds_columns(["BbAvH", "AvgH"])
    assert "AvgH" in mapping and "BbAvH" not in mapping
    assert any("BbAvH" in u for u in unrecognized)


def test_new_format_file(cfg):
    p = parse.parse_file(f"{cfg['paths']['raw']}/2020-2021/E0.csv", "E0", "2020-2021")
    assert len(p.matches) == 12  # 4 équipes aller-retour, sans le non joué ni la ligne vide
    assert p.n_rows_unplayed == 1
    assert "Münster Town" in set(p.matches.home_team_raw)  # latin-1 décodé correctement
    assert p.matches.kickoff_time.notna().all()


def test_result_derived_from_scores():
    r = parse.derive_result(pd.Series([2, 0, 1]), pd.Series([1, 0, 3]))
    assert list(r) == ["H", "D", "A"]


def test_event_id_deterministic():
    d = pd.Timestamp("2020-09-12")
    a = parse.make_event_id("E0", d, "eng:arsenal", "eng:chelsea")
    b = parse.make_event_id("E0", d, "eng:arsenal", "eng:chelsea")
    c = parse.make_event_id("E0", d, "eng:chelsea", "eng:arsenal")
    assert a == b and a != c


def test_team_id_is_country_scoped_not_league_scoped():
    # Un promu garde son identité entre divisions : même pays, même id.
    assert teams.team_id("Fulham", "eng", {}) == "eng:fulham"
    assert teams.team_id("Münster Town", "eng", {}) == "eng:munster-town"
    aliases = {"eng": {"Man United": "Manchester United"}}
    assert teams.team_id("Man United", "eng", aliases) == teams.team_id(
        "Manchester United", "eng", aliases
    )
