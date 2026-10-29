# Data card: football module

## Source

[football-data.co.uk](https://www.football-data.co.uk), one CSV per league and season
(`mmz4281/{season}/{league}.csv`). Free, widely used in the sports modelling literature, includes
results and odds from several bookmakers. Notes on the columns: `football-data.co.uk/notes.txt`.

## Scope

- Seasons 2005-2006 to 2025-2026 (21 complete seasons), plus the current season 2026-2027,
  loaded only after the protocol and the models were frozen and refreshed weekly by `make live`
  (forward test). Upcoming fixtures and their prices come from `fixtures.csv`, saved as
  timestamped snapshots that are never overwritten.
- From 2025-2026 on, football-data.co.uk no longer publishes Pinnacle prices: the live season is
  compared with the market average (see `docs/live_2026_27.md`).
- 12 leagues: top two tiers of England, Spain, Italy, Germany, France, plus the top tier of the
  Netherlands and Portugal. Second tiers are included to test whether markets are less efficient
  where liquidity is lower.

## Known properties of the source (to be confirmed by `reports/coverage_foot.csv`)

| issue | consequence |
|---|---|
| Closing odds (`PSCH`, `AvgCH`...) only exist on recent seasons (from 2019-2020) | CLV is computable on recent seasons only |
| Older "pre-match" odds are collected days before kick-off (Friday afternoon for weekend games, Tuesday for midweek) | an edge against them is weaker evidence than an edge against the close |
| Pinnacle columns appear around 2012-2013 | earlier seasons need another market reference |
| Market average/max came from Betbrain until 2018-2019, from another aggregator after | structural break in `market_avg` / `market_max` |
| `market_max` is a best price across many books, reconstructed after the fact | optimistic for a backtest, used only as a sensitivity bound |
| Kick-off time missing on older seasons | ordering within a day is not guaranteed on those seasons |
| Some seasons stopped early (2019-2020, Covid) | reported as incomplete seasons, kept |

## Processing rules

1. `data/raw/` is immutable. Every file is logged in `manifest.json` with its URL, download time
   and SHA-256.
2. Match results are derived from the scores. The source `FTR` column is only used as a check.
3. Team ids are country-scoped (`eng:fulham`), so a promoted team keeps its id and its rating
   across divisions. Spelling variants are merged only through the explicit alias file
   `configs/team_aliases_foot.yaml`.
4. Odds are never corrected. A (match, bookmaker, timing) triplet goes to quarantine if any price
   is <= 1, if an outcome is missing, or if the overround is outside [1.00, 1.20].
5. Unknown odds-like columns are reported, never guessed.

## Audit results

_To be filled after the first full `make data` run: number of matches, coverage table summary,
quarantine counts by reason, incomplete seasons, team spelling variants found._
