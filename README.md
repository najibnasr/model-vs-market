# Model vs Market

Probabilistic models of match outcomes benchmarked against bookmaker prices, with an honest
value-betting backtest (fractional Kelly, transaction costs, closing line value).

**Status:** football steps (a) data, (b) frozen protocol and (c) Elo baseline done.

The goal is not to find a money machine. Beating closing odds after costs is very hard; the
deliverable is a rigorous, reproducible comparison between a model and an efficient market.

## Architecture

```
src/mvm/
├── core/            sport-agnostic: schema, odds/margin removal, calibration, backtest, CLV
└── sports/
    ├── foot/        football-data.co.uk, Elo then Dixon-Coles
    └── nba/         (planned) nba_api / Basketball-Reference, Elo then four factors
```

`core/` never imports from `sports/`. A sport plugs in by producing three canonical tables:

| table    | grain                                              |
|----------|----------------------------------------------------|
| `events` | one row per match                                  |
| `odds`   | one row per (match, bookmaker, market, timing, outcome) |
| `teams`  | raw name to stable, country-scoped team id         |

## Quickstart

```bash
make install        # uv sync
make data           # download football-data.co.uk CSVs + build canonical tables
make test
make market-baseline   # how good is the closing market? (the bar to clear)
make tune-elo          # grid search on the tuning seasons only
make eval-elo          # Elo vs market vs naive base rates, tuning seasons
```

Outputs: `data/processed/foot/*.parquet`, and audit reports in `reports/`
(`coverage_foot.csv`, `quality_foot.json`, `quarantine_odds_foot.csv`, `teams_foot.csv`).

See [`data/README.md`](data/README.md) for the data card.

## Roadmap (football)

- [x] (a) Data pipeline and storage schema
- [x] (b) Frozen evaluation protocol ([`docs/protocol.md`](docs/protocol.md)): walk-forward splits, margin removal, metrics
- [x] (c) Elo baseline ([`docs/results_elo.md`](docs/results_elo.md)): closes 76% of the gap between base rates and the closing market
- [ ] (d) Dixon-Coles and comparison
- [ ] (e) Value-betting backtest: fractional Kelly, CLV, breakdown by league and period
- [ ] (f) Streamlit dashboard
- [ ] (g) Research note and CV line
