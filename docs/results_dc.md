# Dixon-Coles and model combination: results

## Dixon-Coles, tuning period (2012-13 to 2016-17, 22,828 matches)

Frozen parameters: xi = 0.002 per day (half-life about 350 days), L2 prior = 3, window 6 years.
The value from the original paper (xi = 0.0065, half-life about 3.5 months) is clearly worse
on these data (1.0227 vs 1.0150).

| Source | Log-loss | RPS |
|---|---|---|
| League base rates | 1.0693 | 0.2242 |
| Elo | 1.0140 | 0.2056 |
| **Dixon-Coles** | **1.0150** | **0.2060** |
| Pinnacle close (Shin) | 0.9973 | 0.2006 |

- **Dixon-Coles vs Elo: +0.0010, 95% CI [-0.0007, +0.0027]. No significant difference.** The
  "more sophisticated" model does not beat the simpler one.
- Dixon-Coles vs market: +0.0177, CI [+0.0156, +0.0198].

### The two models make different mistakes

| | Elo | Dixon-Coles | Observed |
|---|---|---|---|
| Home win (mean predicted) | 45.8% | 44.5% | 44.8% |
| Draw | 27.0% | 26.7% | 27.3% |
| Away win | 27.2% | 28.8% | 27.9% |
| Outcomes predicted above 75% happen... | 81.6% (pred. 81.8%) | 86.3% (pred. 81.5%) | |
| Outcomes predicted below 15% happen... | 10.4% (pred. 10.5%) | 9.3% (pred. 10.6%) | |

- Elo over-predicts home wins (stale home advantage). Dixon-Coles fixes this.
- Dixon-Coles is **under-confident at the extremes**: heavy favourites win more often than it
  says, longshots less often. Consistent with the L2 prior shrinking team strengths towards the
  league mean. It also under-predicts draws slightly (a known limit of Poisson models).

Errors that differ are the textbook case for combining forecasts.

## Combination (Elo + Dixon-Coles, 50/50)

Defined after seeing the tuning results above but **before** any test-period run, with weights
fixed a priori (no tuning).

### Tuning period

| Source | Log-loss | RPS |
|---|---|---|
| Elo | 1.0140 | 0.2056 |
| Dixon-Coles | 1.0150 | 0.2060 |
| **Elo + DC (50/50)** | **1.0125** | **0.2053** |
| Pinnacle close | 0.9973 | 0.2006 |

- Combination vs Elo: -0.0015, CI [-0.0023, -0.0006]. Vs Dixon-Coles: -0.0025,
  CI [-0.0033, -0.0016]. **Significantly better than both members** on log-loss and RPS.
- Closes 79% of the base-rate-to-market gap (Elo: 77%).
- Home calibration in between the two members (45.1% predicted vs 44.8% observed).
- Still under-confident on heavy favourites (81.5% predicted, 84.7% observed), inherited from
  Dixon-Coles.

**Caveat.** The combination was motivated by patterns seen on these same tuning seasons, so its
tuning-period advantage is partly circular. Only the test period counts.

## Expectations written BEFORE running the test period (2017-18 to 2024-25)

1. **Dixon-Coles improves relative to Elo** compared with the tuning period (DC minus Elo below
   +0.0010), because Elo's home bias grew on the test period and time decay should handle it.
   Whether the difference becomes significant is left open.
2. **The combination beats Elo** (whole CI below zero).
3. **The combination beats Dixon-Coles** (whole CI below zero).
4. All three models stay behind the market overall, by +0.012 to +0.022.
5. No model beats the market in any league (no league CI entirely below zero).

## Test period (2017-18 to 2024-25), run once each on 2026-10-28

35,647 matches, common sample.

| Source | Log-loss | RPS | Gap closed (base to market) |
|---|---|---|---|
| League base rates | 1.0756 | 0.2280 | 0% |
| Elo | 1.0121 | 0.2064 | 76% |
| Dixon-Coles | 1.0122 | 0.2064 | 76% |
| **Elo + DC (50/50)** | **1.0099** | **0.2058** | **79%** |
| Pinnacle close | 0.9925 | 0.2003 | 100% |

Paired log-loss differences (95% CI, clustered by week):

| Comparison | Mean diff | CI |
|---|---|---|
| DC vs Elo | +0.0001 | [-0.0013, +0.0014] |
| Combination vs Elo | **-0.0022** | [-0.0029, -0.0015] |
| Combination vs DC | **-0.0023** | [-0.0030, -0.0016] |
| Combination vs market | +0.0175 | [+0.0157, +0.0194] |

| Season | Elo | DC | Combination | Market |
|---|---|---|---|---|
| 2017-18 | 1.0016 | 1.0006 | **0.9990** | 0.9795 |
| 2018-19 | 1.0050 | 1.0053 | **1.0032** | 0.9880 |
| 2019-20 | 1.0287 | **1.0237** | 1.0238 | 1.0129 |
| 2020-21 | 1.0239 | 1.0219 | **1.0204** | 0.9998 |
| 2021-22 | 1.0166 | 1.0151 | **1.0135** | 0.9946 |
| 2022-23 | **1.0070** | 1.0118 | 1.0073 | 0.9934 |
| 2023-24 | 1.0062 | 1.0079 | **1.0047** | 0.9840 |
| 2024-25 | 1.0090 | 1.0122 | **1.0084** | 0.9885 |

### Checking the pre-registered expectations

1. **Confirmed, weakly.** DC minus Elo goes from +0.0010 (tuning) to +0.0001 (test). Still not
   significant: Elo and Dixon-Coles remain statistically indistinguishable.
2. **Confirmed.** The combination beats Elo (-0.0022, whole CI below zero).
3. **Confirmed.** The combination beats Dixon-Coles (-0.0023, whole CI below zero).
4. **Confirmed.** All three models trail the market by +0.0175 to +0.0198.
5. **Confirmed.** No model beats the market in any of the 12 leagues.

### What the test period teaches

- **The combination is robust**: best or within 0.0003 of the best model in all 8 test seasons.
  The gain did not come from one lucky season.
- **Dixon-Coles fixes the home bias.** Home wins predicted 43.5% vs 43.4% observed (Elo: 45.2%).
  Its relative advantage over Elo is largest in 2019-20 and 2020-21 (-0.0050 and -0.0020), the
  seasons with matches behind closed doors, consistent with time decay adapting to a drop in home
  advantage. Exploratory: not pre-registered at the season level.
- **More complex is not better, but different is.** Two models of equal accuracy, with
  different errors, combine into a significantly better one.

**Model carried forward to the betting backtest (step e): the 50/50 combination.**
