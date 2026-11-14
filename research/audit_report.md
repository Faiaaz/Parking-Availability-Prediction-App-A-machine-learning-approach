# Repository audit

The local project at the start of this study tracked main. The application reads the serialized legacy models and reports. The independent research runner uses the raw Birmingham CSV. Neither pipeline previously fitted a quality gate or evaluated prediction intervals under corrupted observations.

| Evidence | Finding |
|---|---|
| scripts/download_data.py, data/raw/ | Birmingham ZIP and extracted CSV plus one Melbourne snapshot. SFpark is absent. |
| src/config.py, src/data_prep.py | Legacy preprocessing clips counts, divides by capacity and labels ratios below 0.85 available. Calendar, capacity, site identity, two row lags, lag difference and prior rolling mean predict the current row. Actual lead varies. |
| src/train.py, reports/metrics.json | Logistic regression, forest, histogram boosting and calibrated linear SVM classify availability. Ridge, forest and boosting regress occupancy. Calendar-only and persistence comparisons. Last 20% of sorted rows also selects winners. Inner SVM calibration is nonchronological. |
| research/legacy_audit.json, data/processed/ | 35,657 processed rows. 210 excess duplicate keys can expose same-timestamp occupancy through lag features. Train/test share one timestamp, zero keys. Test has 10 excess duplicate keys. |
| research/run_experiments.py, results/main/ | Direct measured targets within five minutes of each nominal lead, earlier tie preferred. Site-isolated histories, four expanding weekly validation folds, purged labels and subsequent week for evaluation. Baselines, Ridge, forests, boosting, point errors and daily bootstrap intervals for differences in mean error. Those intervals describe estimator variability. |
| tests/test_research.py | Original tests cover timestamp alignment, chronology, duplicates, gaps and paired bootstrap alignment. |
| app/streamlit_app.py, requirements.txt | Streamlit dashboard uses historical metrics and what-if inputs. Melbourne map reads a local snapshot. Dependency ranges support the application; research/environment-freeze.txt records installed versions. |

Raw Birmingham spans 2016-10-04 07:46:28 through 2016-12-19 16:30:35. Of 35,717 rows at 30 sites, 216 repeat exact keys and 385 violate occupancy bounds. 35,116 valid unique measurements remain. No conflicting duplicate keys or null fields. Clean within-site median interval 32.93 minutes, 95th percentile 926.05 minutes. Gaps include overnight closures or absence of sampling. Cause is unrecorded. Full-period matched examples number 31,268 / 23,884 / 20,477 for the three horizons. See results/main/audit.json and horizon_coverage.csv.

The original dashboard estimates same-row status with preceding readings. The multihorizon evaluation issues forecasts at an observed origin and uses subsequent occupancy only for matching and scoring. No independently measured sensor fault labels or calibration cohort appear in the original benchmark.
