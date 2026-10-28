# Value-betting backtest: results

Model: 50/50 Elo + Dixon-Coles combination. Bets placed at the **pre-match** price (the only
price available when the model has its information). Main strategy frozen in `configs/foot.yaml`
before any backtest: bet the outcome with the largest model edge if it exceeds 5%, odds at most
10, quarter Kelly capped at 2% of bankroll per bet and 25% per week.

ROI and CLV are per unit staked (flat stakes), with 95% CIs from a bootstrap clustered by week.
CLV = odds taken x Pinnacle closing probability (Shin) - 1.

## Tuning period (2012-13 to 2016-17, 22,659 matches)

| Strategy | Bets | ROI | ROI 95% CI | CLV | CLV 95% CI | Quarter-Kelly bankroll |
|---|---|---|---|---|---|---|
| **Main: combination @ Pinnacle pre-match** | 14,767 | **-5.7%** | [-8.3%, -3.1%] | **-2.24%** | [-2.41%, -2.06%] | -97% |
| Combination @ market average | 9,802 | -10.6% | [-13.9%, -7.5%] | -8.27% | [-8.51%, -8.04%] | -100% |
| Combination @ best price (max) | 17,080 | -2.5% | [-5.0%, +0.1%] | +0.01% | [-0.16%, +0.17%] | -87% |
| Edge > 0% | 20,759 | -5.1% | [-7.2%, -3.0%] | -2.28% | | -96% |
| Edge > 15% | 7,168 | -5.5% | [-9.6%, -1.6%] | -2.30% | | -97% |
| Elo alone | 15,079 | -5.9% | | -2.42% | | -99% |
| Dixon-Coles alone | 16,572 | -5.5% | | -2.22% | | -95% |
| **Positive control: oracle (closing probabilities)** | 7,102 | **+9.1%** | [+5.3%, +13.0%] | **+10.6%** | [+10.4%, +10.8%] | x104 |
| **Negative control: league base rates** | 19,482 | **-4.9%** | [-7.7%, -2.1%] | **-2.79%** | [-2.96%, -2.61%] | -97% |

Blend test (weight fitted on these seasons): **w = 0 against both the Pinnacle close and the
Pinnacle pre-match price.** Mixing in the model never improves on the market alone.

### Reading

1. **The backtest works.** The oracle, which cheats by using closing information, makes money
   with a large positive CLV; the no-information control loses roughly the bookmaker margin.
   Whatever the backtest says about the model can therefore be trusted.
2. **The model's "edges" are mostly its own errors.** It finds a 5% edge in 65% of matches, and
   raising the threshold from 0% to 15% does not improve ROI or CLV at all. If the model's edge
   carried information, bigger edges would mean better bets.
3. **The model barely beats no information at all.** Its CLV (-2.24%) is only about half a point
   better than the base-rate control (-2.79%), and the blend weight is exactly zero: the
   pre-match Pinnacle price already contains everything the model knows.
4. **CLV is the statistic to watch.** Its confidence interval is about 15 times narrower than
   ROI's. Several leagues show a positive ROI (Bundesliga +5.2%, La Liga +3.7%) while their CLV is
   negative: that is luck, and CLV says so.
5. **Kelly amplifies a negative edge into ruin.** Quarter Kelly on a model that overestimates its
   edge, betting about 60 times a week, loses 97% of the bankroll. Kelly is optimal only if the
   probabilities are right.
6. **Best-price CLV around zero is a price effect, not a model effect** (hypothesis). Taking the
   best odds across many books after the fact recovers roughly the closing fair value. This bound
   is unrealistic in practice: stale lines and account limits.

## Expectations written BEFORE running the test period (2017-18 to 2024-25)

1. Main strategy: **CLV negative, whole CI below zero**; ROI negative.
2. Both controls behave as on the tuning period: oracle CLV positive (CI above zero), base-rate
   control CLV negative (CI below zero).
3. Raising the edge threshold does not improve CLV (all thresholds within 0.5 point of each
   other).
4. Best-price CLV within half a point of zero.
5. No league with a CLV confidence interval entirely above zero. (Per-league CIs are not
   computed by the script; checked on point estimates: no league with CLV above +0.5%.)

The blend weight was fixed at w = 0 on the tuning period, so the blend equals the market by
construction on the test period: no separate test is needed.

## Test period (2017-18 to 2024-25, 35,525 matches), run once on 2026-10-28

| Strategy | Bets | ROI | ROI 95% CI | CLV | CLV 95% CI | Quarter-Kelly bankroll |
|---|---|---|---|---|---|---|
| **Main: combination @ Pinnacle pre-match** | 22,875 | **-8.6%** | [-10.8%, -6.5%] | **-4.02%** | [-4.22%, -3.84%] | -99.9% |
| Combination @ market average | 17,602 | -11.1% | [-13.5%, -8.6%] | -7.87% | [-8.05%, -7.69%] | -100% |
| Combination @ best price (max) | 26,770 | -5.0% | [-7.0%, -2.9%] | -1.56% | [-1.74%, -1.38%] | -98% |
| Edge > 0% | 31,833 | -6.4% | | -3.75% | | -99.8% |
| Edge > 2% | 28,125 | -7.4% | | -3.86% | | -99.9% |
| Edge > 10% | 15,892 | -8.8% | | -4.24% | | -99.9% |
| Edge > 15% | 11,075 | -9.0% | | -4.36% | | -99.9% |
| Elo alone | 24,230 | -6.5% | | -3.66% | | -99.7% |
| Dixon-Coles alone | 25,240 | -8.1% | | -4.16% | | -99.9% |
| **Positive control: oracle** | 10,966 | **+9.1%** | [+6.0%, +12.1%] | **+10.76%** | [+10.57%, +10.94%] | x9,100 |
| **Negative control: base rates** | 31,454 | **-7.5%** | [-9.4%, -5.5%] | **-4.52%** | [-4.72%, -4.33%] | -99.8% |

Every league has a negative CLV (best: Segunda Division, -2.8%).

### Checking the pre-registered expectations

1. **Confirmed.** Main strategy CLV -4.02%, whole CI below zero; ROI -8.6%.
2. **Confirmed.** Oracle CLV +10.8% (CI above zero); base-rate control CLV -4.5% (CI below zero).
3. **Failed on the number, confirmed on the substance, and more strongly than expected.** CLV
   spans 0.61 point across thresholds (expected below 0.5), but in the "wrong" direction: it
   gets **monotonically worse** as the threshold rises (-3.75% at 0%, -4.36% at 15%). The
   larger the model's disagreement with the market, the worse the bet.
4. **Failed.** Best-price CLV is -1.56%, not within half a point of zero. The tuning-period
   result did not replicate. A plausible cause, not tested: the source of the market max changed
   in 2019-20 (Betbrain then another aggregator), possibly with fewer books.
5. **Confirmed.** No league with a CLV above +0.5%.

### What the test period teaches

- **The loss deepens, but because of the price, not the model.** Main CLV goes from -2.2%
  (tuning) to -4.0% (test), and the no-information control moves by the same amount (-2.8% to
  -4.5%). The model stays about half a point better than knowing nothing, in both periods. The
  common shift matches the rise in Pinnacle's pre-match margin over time measured in step (b).
- **When the model disagrees most with the market, the market is right.** CLV worsens with the
  edge threshold and is worst on long odds (-5.8% between odds 5 and 10).
- **Conclusion of the betting study.** A results-only model, even a well-calibrated combination
  closing 79% of the information gap, has no exploitable edge against Pinnacle, before or at the
  close. The blend weight is zero: the pre-match price already contains everything the model
  knows. Any real edge would require information the market does not yet price (team news,
  player-level data, speed), not a better treatment of past scores.
