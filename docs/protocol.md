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

## Amendments

_None yet._
