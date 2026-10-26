# Evaluation protocol (football)

**Frozen on 2026-10-25, before any model was fitted.** The git commit that introduces this file
is the timestamp. Any later change goes in the "Amendments" section with its reason, so that a
reader can tell a pre-registered choice from a choice made after seeing results.

Machine-readable version: the `evaluation` block of `configs/foot.yaml`.

## 1. Question

Do our model probabilities for 1X2 outcomes contain information that the closing market price
does not? Secondary: if so, does a value-betting strategy built on them survive the margin, and
does it beat the closing line (CLV)?

## 2. Time splits

| Period | Seasons | Role |
|---|---|---|
| Burn-in | 2005-06 to 2011-12 | Models learn team strengths. Nothing is scored. No Pinnacle odds exist. |
| Tuning | 2012-13 to 2016-17 | Hyperparameters (Elo K, home advantage, Dixon-Coles time decay...) are chosen here. |
| Test | 2017-18 to 2024-25 | Looked at **once per final model**. No tuning on these seasons. |
| Holdout | 2025-26 | Pinnacle only partially available: scored against the fallback reference. |
| Forward | 2026-27 | Live, truly out-of-sample, once the season is over. |

Splitting by whole seasons rather than random folds: matches are not exchangeable. Random
k-fold would let the model learn from April to predict the previous August.

## 3. Information set (no leakage)

- Evaluation is walk-forward by **calendar week** (Monday to Sunday). Before week *w*, a model
  receives every match played before *w*, with results. It then predicts all matches of week *w*.
- Results columns are physically removed from the frame passed to `predict()`
  (`mvm.core.splits.hide_results`). A model that tries to read them crashes. Tested.
- Weekly blocks are deliberately conservative: kick-off times are missing before 2019-20, so a
  Sunday match must not learn from the Saturday results of the same weekend.
- Odds are *not* model inputs. A model that uses market prices as features answers a different
  question and would be reported separately.

## 4. Market reference

- **Reference:** Pinnacle closing odds, margin removed with **Shin's method**.
  Pinnacle is the sharpest widely recorded book (low margin, accepts large stakes, does not limit
  winners), so its closing price is the best available estimate of the true probability.
- **Why Shin:** bookmakers load more margin on longshots (favourite-longshot bias). Proportional
  normalisation ignores this; Shin models it explicitly. Proportional and power are reported as
  robustness checks: conclusions must not depend on the margin-removal method.
- **Fallback** (2025-26 onwards, Pinnacle missing): market average closing odds, Shin. It is only
  accepted if, on 2019-20 to 2024-25 where both exist, its log-loss is within the bootstrap CI of
  Pinnacle's (checked by `make market-baseline`).
- **Common sample:** a match is scored only if the reference triplet is complete. Model and
  market are always compared on exactly the same matches.

## 5. Metrics

- **Primary: log-loss** (mean negative log-probability of the realised outcome). Strictly
  proper, and the right currency for a bettor: the expected log-growth of a Kelly bettor against
  the market equals the log-loss difference between market and model.
- **Secondary:** ranked probability score (RPS, respects the H < D < A ordering, standard in the
  football literature) and Brier score.
- **Calibration:** reliability tables per outcome (H, D, A), equal-count bins, binomial 95%
  intervals, expected calibration error.

## 6. Decision rule (pre-registered)

For two forecasts A and B on the same matches, compute the per-match loss difference
d = loss_A - loss_B. Its mean gets a 95% interval from a **bootstrap clustered by week**
(2,000 resamples, seed 20261025): matches of the same weekend share information and shocks, so
treating them as independent would understate the variance.

- "A beats B" only if the **whole interval is below zero**.
- Results are reported for the full test period, then by league, tier and season. A sub-group
  edge found after looking at the breakdown is labelled **exploratory**, not a finding: with 12
  leagues and 8 seasons, some cells will look significant by chance.

## 7. What would make us stop

- The model fails to beat a naive benchmark (home/draw/away base rates per league) on the tuning
  period: bug or useless model, fix before going further.
- The model "beats" the closing market by a large margin (say more than 0.01 in log-loss) on the
  whole test period: assume leakage until proven otherwise.

## Results of the pre-registered checks (run on 2026-10-25)

`make market-baseline`, eval seasons 2012-13 onwards. Paired log-loss differences, 95% CI
clustered by week.

| Comparison | Mean diff | 95% CI | Reading |
|---|---|---|---|
| Pinnacle close vs Pinnacle pre-match | -0.00267 | [-0.00326, -0.00210] | The close is sharper: information arrives in the last days (team news). |
| Pinnacle close vs market-average close | -0.00031 | [-0.00058, -0.00005] | Pinnacle is sharper, by a small but significant margin. |
| Average close vs average pre-match | -0.00298 | [-0.00381, -0.00216] | Same effect as for Pinnacle. |
| Shin vs proportional (Pinnacle close) | -0.00017 | [-0.00026, -0.00009] | Shin beats proportional: favourite-longshot bias is real. |
| Shin vs power (Pinnacle close) | +0.00004 | [+0.00001, +0.00008] | Power is marginally better than Shin. |

Market level (Pinnacle close, Shin): log-loss 0.954 in top tiers, 1.041 in second tiers
(uniform forecast: ln 3 = 1.099).

**Consequences, applied without changing the rules above:**

1. **Fallback check failed.** The market average is significantly less sharp than Pinnacle.
   It remains the only reference for 2025-26 onwards, so holdout and forward results against it
   are reported as *optimistic*: an edge against the average overstates the edge against a sharp
   close. Moreover, the average's overround jumps by about 1.5 points in 2025-26 in every
   league, which suggests its composition changed (possibly without Pinnacle): the 2019-2025
   validation may not even transfer. The holdout is therefore secondary evidence only.
2. **Reference stays Shin.** Power is better by 0.00004, about 70 times less than the
   close-vs-pre-match gap: practically irrelevant. Switching after seeing results is exactly
   what pre-registration forbids. Power stays as a robustness check.
3. **Scale for later results:** the whole informational value of the last days before kick-off
   is about 0.003 in log-loss. A model claiming to beat the close by more than that should be
   treated as suspect (section 7).
4. **Cross-league comparisons:** second tiers have higher log-loss because their matches are more
   balanced, not necessarily because their markets are less efficient. Model and market are only
   ever compared within the same league.

## Amendments

_None yet._
