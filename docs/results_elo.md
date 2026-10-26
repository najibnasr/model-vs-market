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

## Test period (2017-18 to 2024-25)

_Run once with `make test-elo`, filled in after the run._
