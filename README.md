# Parking occupancy forecasting and observation reliability

This project includes a Streamlit parking dashboard, an earlier multi-horizon benchmark, and an exploratory reliability experiment. The reliability experiment evaluates whether a fixed quality gate improves predictions beyond giving the same quality indicators to a conventional learner.

## Verified exploratory result

Only Birmingham supplied training, validation, calibration and test observations in the 378 reported evaluations. Melbourne is a stored bay-status snapshot used by the dashboard. SFpark is a candidate for future external validation.

M1 uses occupancy history and calendar/site inputs. M2 adds quality features including missingness, validity, observation age and completeness. M3 blends the fitted M2 forecast with a calendar model using a fixed heuristic quality weight.

| Horizon | M2 clean MAE | M3 clean MAE | M2 degraded MAE | M3 degraded MAE |
|---|---:|---:|---:|---:|
| 30 minutes | 2.15 | 2.28 | 3.15 | 4.07 |
| 60 minutes | 3.31 | 3.42 | 4.26 | 4.94 |
| 120 minutes | 5.18 | 5.25 | 5.94 | 6.37 |

Errors are occupancy percentage points. Degraded MAE averages 20 settings equally. M3 increased mean error at every horizon under both clean and degraded inputs. This experiment provides no evidence of added value from this particular gate beyond M2.

The 378 rows comprise 3 horizons × 21 conditions × 6 methods. Conditions include clean observations and five simulated failure types at two severity levels and two seeds. The six methods are M1, M2, M3, persistence, daily and weekly seasonal baselines. Genuine measured future occupancy supplies every reference label.

## Reproduce

The repository includes the Birmingham input. Use Python 3.13 and the recorded environment for the research experiment.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r research/environment-freeze.txt
python -m unittest discover -s tests -v
python -m research.reliability --output research/results/reliability_reproduction
```

Choose a new output directory. Compare scores.csv and validation.csv to research/results/reliability_v1/ using floating-point tolerances. The recorded input and code hashes are in hashes.txt. Seed 42 governs model fitting, seed 41 training corruption, and seeds 17/29 evaluation corruption. The experiment uses one numerical training thread.

Training ends before November 29, 2016. Validation covers November 29–December 5, calibration December 6–12 and evaluation December 13–19. Actual target timestamps must precede each partition end. Predictions use only history available at the origin. The target is the nearest future reading within ±5 minutes of the nominal 30/60/120-minute lead.

## Interpretation and limitations

Birmingham has no independent sensor-health or manual accuracy labels. Injected failures assess controlled robustness. Completeness is calculated over observed readings, so an unknown reporting schedule limits missed-reading detection. The final week overlaps earlier project evaluations. Repeated scenarios share labels and cannot be treated as independent datasets. New dates, held-out sites and compatible external data remain to be evaluated.

Clean-week residuals calibrate nominal 90% intervals. M2 coverage averaged 88.0%, 88.4% and 90.0% on clean test observations, declining to 78.3%, 81.1% and 86.0% across degraded settings. Temporal dependence and distribution shift limit coverage claims. No algorithmic novelty or general superiority is established.

See [provenance and publication audit](research/publication_audit.md), [methodology](research/methodology.md), [experiment protocol](research/experiment_protocol.md), [limitations](research/limitations.md) and [SFpark requirements](research/sfpark_external_validation.md).

## Dashboard and earlier benchmark

The dashboard uses legacy classifiers/regressors and saved artifacts. Its what-if predictor uses prior readings with an unspecified elapsed horizon. The Melbourne view shows a downloaded snapshot. The earlier fixed-horizon benchmark is documented in research/README.md and research/RESULTS.md.

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/train.py
streamlit run app/streamlit_app.py
```

Legacy training regenerates reports and models. Original scores are historical results under that protocol. The audit identified duplicate-derived same-timestamp lags and holdout-based winner selection, which limit interpreting those scores.

## Attribution and reuse

Birmingham data are credited to Daniel Stolfi, Parking Birmingham (2017), UCI Machine Learning Repository, [doi:10.24432/C51K5Z](https://doi.org/10.24432/C51K5Z). UCI distributes the data under CC BY 4.0 and cites UK Open Government Licence council provenance. Research preprocessing deduplicates rows and excludes invalid counts.

The Melbourne snapshot is City of Melbourne data under [CC BY 4.0](https://discover.data.vic.gov.au/en_AU/dataset/on-street-parking-bay-sensors). Its identifiers refer to public parking bays. SFpark files are absent. A software license has not been selected for this repository; dataset licenses apply to their respective data.
