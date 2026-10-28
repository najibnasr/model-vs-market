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

## Test period

_Run once each with `make test-dc` and `make test-ens`, filled in after the runs._
