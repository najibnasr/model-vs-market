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

_Tuning-period results and test expectations to be added._
