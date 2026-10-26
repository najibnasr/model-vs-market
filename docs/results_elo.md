# Elo baseline: results

## Tuning period (2012-13 to 2016-17, 22,828 matches, common sample with the market)

Frozen parameters: K = 15, home advantage = 70, season regression = 0.1, new-team offset = -50,
goal-margin multiplier on, ordered-logit calibration on the last 15 seasons per league.

| Source | Log-loss | RPS | Brier |
|---|---|---|---|
| League base rates | 1.0693 | 0.2242 | 0.6465 |
| **Elo** | **1.0140** | **0.2056** | **0.6078** |
| Pinnacle close (Shin) | 0.9973 | 0.2006 | 0.5967 |

- Elo vs market: +0.0166 log-loss, 95% CI [+0.0143, +0.0191]. Worse than the market in all 12
  leagues (every league CI above zero).
- Elo vs base rates: -0.0554, CI [-0.0598, -0.0508].
- **Elo closes 77% of the gap between naive base rates and the closing market**
  ((1.0693 - 1.0140) / (1.0693 - 0.9973)).

### Diagnostics

- **Home bias.** Elo predicts 45.8% home wins against 44.8% observed, and under-predicts away
  wins (27.2% vs 27.9%). The market is almost exact (44.9%). Likely cause: the calibration layer
  learns home advantage from up to 15 past seasons, while home advantage has been declining. A
  model with time-decayed weights (Dixon-Coles, step d) should reduce this.
- **Residual favourite-longshot bias in the market (exploratory).** Even after Shin, outcomes the
  market prices above 75% happen 84.3% of the time against 81.8% predicted (n = 1,270, about
  2.4 standard errors), and outcomes below 15% happen slightly less than predicted. Found by
  looking at the tuning data, so it is a hypothesis to test, not a finding.

## Expectations written BEFORE running the test period (2017-18 to 2024-25)

1. Elo stays behind the market, by a gap of the same order (+0.012 to +0.022).
2. Elo is relatively worse in **2020-21**: matches without crowds cut home advantage sharply,
   and a calibration learned on past seasons cannot adapt within the season.
3. Elo still beats base rates by about 0.05.

## Test period (2017-18 to 2024-25), run once on 2026-10-26

35,647 matches, common sample with the market.

| Source | Log-loss | RPS | Brier |
|---|---|---|---|
| League base rates | 1.0756 | 0.2280 | 0.6509 |
| **Elo** | **1.0121** | **0.2064** | **0.6061** |
| Pinnacle close (Shin) | 0.9925 | 0.2003 | 0.5926 |

- Elo vs market: **+0.0197**, 95% CI [+0.0175, +0.0218]. Behind the market in all 12 leagues.
- Elo vs base rates: -0.0635, CI [-0.0671, -0.0598].
- Elo closes **76%** of the base-rate-to-market gap (77% on the tuning period): stable.

| Season | Elo | Market | Gap |
|---|---|---|---|
| 2017-18 | 1.0016 | 0.9795 | 0.0221 |
| 2018-19 | 1.0050 | 0.9880 | 0.0170 |
| 2019-20 | 1.0287 | 1.0129 | 0.0158 |
| 2020-21 | 1.0239 | 0.9998 | **0.0241** |
| 2021-22 | 1.0166 | 0.9946 | 0.0220 |
| 2022-23 | 1.0070 | 0.9934 | 0.0136 |
| 2023-24 | 1.0062 | 0.9840 | 0.0222 |
| 2024-25 | 1.0090 | 0.9885 | 0.0205 |

### Checking the pre-registered expectations

1. **Confirmed.** Gap of +0.0197, inside the expected range [+0.012, +0.022], though larger
   than on the tuning period (+0.0166).
2. **Weakly supported only.** 2020-21 is the worst season for Elo (+0.0241), but barely above
   2017-18 and 2023-24 (+0.022). Season-level differences of this size are within noise. We do
   not claim a Covid effect from this table.
3. **Confirmed.** Elo beats base rates by 0.064.

### What the test period teaches

- **The home-advantage diagnosis is confirmed, and stronger.** Elo predicts 45.2% home wins
  against 43.4% observed (+1.8 points, versus +1.0 on the tuning period) and under-predicts away
  wins (28.3% vs 30.0%). The market stays within 0.2 points. Home advantage kept falling over the
  test period and a calibration learned on up to 15 past seasons lags behind. This is the main,
  identifiable weakness to fix in step (d).
- **The market favourite-longshot hypothesis did not replicate.** On the test period, outcomes
  priced above 75% happen 81.8% of the time against 81.6% predicted: no bias. A deviation appears
  in a different bucket (60-75%: 68.4% observed vs 66.4% predicted), but chasing a new bucket
  after the pre-specified one failed is exactly the forking-paths error. Conclusion: no evidence
  of an exploitable bias in Pinnacle closing prices.
