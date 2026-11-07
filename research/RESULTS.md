# Measured research results

Source: `research/results/main`. Input SHA-256: `f1e28b9c697769a1a05cd20148241b60b1a73fee4907611163efc6fd9d12770a`.

Cleaning retains 35,116 observations after removing duplicate keys and invalid measurements.

The following model was selected separately at each horizon using the lowest pooled CV MAE. Final-week metrics did not select the model. MAE and RMSE are occupancy percentage points.

| Horizon | CV-selected method | CV MAE | Holdout MAE | Persistence MAE | Holdout RMSE | Holdout n |
|---|---|---|---|---|---|---|
| 30 min | gradient_boosting | 1.418 | 1.483 | 3.611 | 2.096 | 2,907 |
| 60 min | random_forest | 2.407 | 2.431 | 6.706 | 3.486 | 2,244 |
| 120 min | random_forest | 3.953 | 4.092 | 12.252 | 5.649 | 1,886 |

## Baseline comparison on the final week

| Horizon | Method | MAE (pp) | RMSE (pp) |
|---|---|---|---|
| 30 | daily_seasonal_fallback | 9.285 | 15.116 |
| 30 | persistence | 3.611 | 5.128 |
| 30 | weekly_seasonal_fallback | 4.977 | 7.752 |
| 60 | daily_seasonal_fallback | 9.629 | 15.217 |
| 60 | persistence | 6.706 | 9.530 |
| 60 | weekly_seasonal_fallback | 5.490 | 8.409 |
| 120 | daily_seasonal_fallback | 10.834 | 16.694 |
| 120 | persistence | 12.252 | 17.018 |
| 120 | weekly_seasonal_fallback | 6.152 | 9.787 |

## Coverage and uncertainty

- 30 minutes: 31,268 eligible examples across the full period; actual leads 25.917–34.433 minutes (median 29.983).
- 60 minutes: 23,884 eligible examples across the full period; actual leads 55.933–64.100 minutes (median 60.000).
- 120 minutes: 20,477 eligible examples across the full period; actual leads 115.933–124.117 minutes (median 120.017).
- 30 minutes, gradient_boosting: paired daily bootstrap MAE difference against persistence -2.127 pp, 95% interval [-2.511, -1.797] pp; daily/weekly seasonal coverage 85.8%/80.0%.
- 60 minutes, random_forest: paired daily bootstrap MAE difference against persistence -4.274 pp, 95% interval [-4.833, -3.829] pp; daily/weekly seasonal coverage 85.1%/77.3%.
- 120 minutes, random_forest: paired daily bootstrap MAE difference against persistence -8.160 pp, 95% interval [-9.279, -7.154] pp; daily/weekly seasonal coverage 82.5%/84.1%.

Intervals are descriptive, based on seven holdout origin dates; they do not account fully for dependence across days, model selection, or multiple comparisons. Seasonal fallback methods use persistence where historical matches are absent; the complete-history comparison is saved separately. Cohorts vary by horizon. The final week overlaps the legacy evaluation period, so it is not previously unseen external data. See `research/README.md` for the full protocol and limitations.

No historical reported metrics were modified. All values above are calculated from saved predictions, not inferred or fabricated.
