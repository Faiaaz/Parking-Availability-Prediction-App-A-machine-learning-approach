# Multi-horizon parking occupancy research

This directory contains a separate evaluation pipeline. The original `src/`, application,
raw data, processed data, fitted models, and reported metrics are preserved.

## Project review

- `scripts/download_data.py` downloads the Birmingham ZIP and a Melbourne CSV snapshot.
  Re-downloading the Melbourne file changes its contents; the local snapshot is the research reference.
- `src/config.py` centralizes paths, the 0.85 nearly-full threshold, a 20% test fraction,
  and seed 42.
- `src/data_prep.py` parses dates, clips occupancy to [0, capacity], derives occupancy
  rate and availability, and adds calendar features, two previous-row lags, a shifted
  three-observation mean, and a lag difference per car park. It does not deduplicate,
  enforce elapsed-time lags, validate capacity, or create future-horizon labels.
- `src/train.py` fits logistic regression, random forest, histogram gradient boosting,
  and calibrated linear SVM classifiers; Ridge, forest, and boosting regressors;
  calendar-only variants; and a previous-reading persistence baseline. Scaling and
  encoding are fitted inside pipelines. Regression predictions are clipped to [0,1].
  A row-count chronological split supplies the last 20% for evaluation. The same
  evaluation set chooses the winning models and is therefore not an untouched test.
- `app/streamlit_app.py` displays saved evaluations, occupancy exploration, a what-if
  predictor, and the Melbourne map. The predictor takes manually supplied history
  and calendar inputs; it has no explicit horizon. Melbourne is a stored snapshot,
  although some dashboard labels say “live” or “now.” It is not training data.
- `reports/` contains the existing metrics, holdout observations, and curve arrays;
  `models/` contains serialized estimators and the winning-model bundle. These were
  inspected as existing artifacts and were not retrained or replaced.
- `requirements.txt` uses version ranges rather than an exact environment lock.
  `Makefile` exposes data, features, training, and application commands. The notebook
  directory has no analysis notebooks.

## Dataset and integrity audit

The Birmingham CSV has 35,717 observations from 30 car parks between 4 October and
19 December 2016, with capacity, occupancy count, timestamp, and car-park identity.
There are no missing values in these four fields, and capacity is positive and
constant within each park. There are 216 excess exact duplicate rows, with no
conflicting measurements for the same park and timestamp. There are 385 raw rows
outside [0, capacity]. The existing loader clips them; this experiment excludes
them after deduplication and reports the resulting count in `audit.json`.

The existing processed frame contains 35,657 rows and retains 210 excess duplicate
keys. With row-based shifting, a duplicate can supply the target occupancy as the
“lag” at the same timestamp. This is a concrete leakage route. Deduplicating before
feature construction closes it. Adjacent-row lags also span irregular intervals
and overnight gaps; they cannot be called 30-minute lags.

The old train and test sets share the boundary timestamp 5 December 2016 09:29:08.
The read-only audit checks whether they share actual park/timestamp keys; a shared
timestamp alone does not establish a shared observation. Both issues are reported
separately in `legacy_audit.json`.

Additional limitations in the existing evaluation:

- The target is the same-row occupancy, not a constructed 30/60/120-minute future
  target. The previous reading is the implicit forecast origin, but calendar
  features describe the target row, and actual elapsed horizon is unspecified.
- The SVM uses default three-fold classification calibration inside training,
  rather than chronological calibration. The outer training-only scaling is safe
  from outer-test leakage, but is fitted before those inner calibration splits.
- Boosting's default automatic early stopping may use a randomly selected internal
  validation subset. This is not outer-test leakage, but is not a temporal
  validation protocol.
- Previous observed test occupancies feed subsequent predictions. This is valid for
  an online forecasting protocol, but not a forecast of the whole test block made
  once at its start.
- `1 - occupancy` is a ranking score for availability, not a calibrated probability.
  The legacy persistence hard decision uses 0.85, not a 0.5 cutoff on that score.
- High majority-class F1/AUC alone does not establish useful performance when a car
  park is nearly full. Report both class F1 scores and occupancy errors.

Existing reported scores are retained as historical results under their original
protocol. They must not be relabeled as fixed-horizon forecasting results.

## Declared research plan and protocol

The plan was explained in the chat before implementation. The primary task is
occupancy-rate regression, with availability decisions derived using the existing
0.85 threshold. For each observed origin t and each horizon h in {30,60,120}, fit a
separate direct model to an observed occupancy nearest to t+h within ±5 minutes.
Ties choose the earlier measurement. No interpolation or future filling is used.
Nominal and actual target timestamps are distinct; actual lead times and matching
coverage are saved. This estimates approximate-horizon forecasts, not exactly
30/60/120-minute labels. A tolerance sensitivity experiment can be run separately.

Inputs are current observed occupancy, two preceding observations, a trailing
three-observation mean including current occupancy, change since the preceding
observation, elapsed ages of the two lags, capacity, park identity, and known
calendar features for the nominal target. Current occupancy is legitimate because
the target is strictly in the future. Features never use future occupancy. Capacity
is assumed known at the origin. Observations are treated as immediately available;
collection delays are unknown. Long gaps are represented by lag-age features,
not silently assumed to be half-hour steps.

Four expanding training windows precede nonoverlapping seven-day validation
windows: 15–22 November, 22–29 November, 29 November–6 December, and 6–13 December
2016. The final holdout is 13–20 December (end exclusive), with observed data ending
19 December. All parks use the same calendar boundaries. Training examples require
actual target_time < validation start. Evaluation origins and targets must both lie
inside the evaluation window. This purges boundary-crossing labels. Each window
fits fresh models and preprocessing using its training rows only. Training expands
through previous validation windows; the holdout model may use all earlier labels.

Models use fixed declared hyperparameters, seed 42, and one training thread:
Ridge, random forest, gradient boosting, and calendar-only gradient boosting.
Boosting random internal early stopping is disabled. No hyperparameter search is
performed. The lowest pooled CV MAE selects a model separately for each horizon;
final-week errors do not choose that model. All model holdout scores are supplied
for transparency, not as a license to choose a different winner after seeing them.

Baselines are current-value persistence, previous-day occupancy at the nominal
future time of day, and previous-week occupancy at that time. Seasonal matches use
the same tolerance and must have source_time <= origin. Missing seasonal history
falls back to persistence, explicitly named in results. Seasonal coverage is saved;
`complete_seasonal_history_metrics.csv` compares every method on the identical
subset where both seasonal measurements exist. All methods otherwise share the
same examples within each horizon and fold. Cohorts differ across horizons because
future measurements and boundary exclusions differ.

## Outputs and interpretation

`results/main/` contains measured outputs only:

- `audit.json`, `horizon_coverage.csv`, and `splits.csv`: cleaning counts, temporal
  alignment, sample sizes, seasonal coverage, and training-label boundaries.
- `predictions.csv.gz`: all out-of-fold and final-holdout predictions with park,
  origin, actual target timestamp, lead time, truth, method, and horizon.
- `fold_metrics.csv`, `pooled_metrics.csv`: MAE, RMSE, R², both class F1 scores,
  balanced accuracy, and availability ranking AUC. Errors use occupancy fractions;
  multiply by 100 for occupancy percentage points. AUC uses 1 minus the forecast,
  without claiming calibrated probabilities. Undefined single-class AUC is blank.
- `evaluation_table.csv` and `.tex`: publication tables of pooled CV and final-week
  MAE/RMSE. Pooled errors are calculated from predictions, not averaged fold RMSEs.
- `paired_daily_bootstrap.csv`: seeded 1,000-replicate paired daily-block bootstrap
  intervals for MAE(method) − MAE(persistence), separately for CV and holdout.
  Negative differences favor the method. The same sampled days include all parks.
  These are descriptive uncertainty estimates: just seven holdout days, dependence
  across days, weekly seasonality, model selection, and multiple comparisons limit
  inferential strength. They are not significance-test claims.
- Four figures in vector PDF and 300-dpi PNG: error against horizon, fold stability,
  per-car-park final-week errors for the CV-selected model, and actual lead times.
- `manifest.json`: input/script hashes, versions, seed, command, fixed protocol,
  selected models, legacy artifact hashes, and hashes of generated outputs.

Sampling is predominantly daytime and irregular. Results apply to observed parks
and this short 2016 period, not unseen cities, all-day continuous occupancy, or
unseen parks. No causal interpretation or universal superiority claim is supported.
Invalid-value exclusion and target matching may change which times are evaluated.
The final week was reserved by this new protocol, but overlaps the historical
project's previously inspected holdout period, so it is not a previously unseen
external dataset. Fresh data would strengthen a final confirmatory paper.

## Reproduce

From the repository root, use the existing environment or create one from the saved
full environment snapshot `environment-freeze.txt`. The experiment also writes a
smaller numerical-package lock beside its results. Python/platform details are in
the manifest. Bitwise identity across different library versions or platforms is
not guaranteed; the saved predictions are the authoritative run record.

```bash
.venv/bin/python research/audit_legacy.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python research/run_experiments.py --output research/results/reproduction
```

Output directories must be new: the runner refuses to overwrite any prior run.
Compare prediction and metric hashes with `results/main/manifest.json`.
A failed initial attempt (`results/failed_attempt_01`) contains only an audit file;
no fitted results were produced. Timestamp units were subsequently made explicit
for compatibility with pandas 3, and protocol tests were run before the main run.
The raw input hash must match before comparing runs. Run `audit_legacy.py` again
to verify all original artifact hashes are unchanged.

Optional sensitivity run (separate output, without choosing tolerance from test scores):

```bash
.venv/bin/python research/run_experiments.py --tolerance 2 --output research/results/tolerance_2
```

## Source attribution

UCI's [Parking Birmingham record](https://archive.ics.uci.edu/dataset/482/parking%2Bbirmingham)
identifies 35,717 instances, four variables, daytime coverage for October–December
2016, and the dataset citation: Stolfi, D. (2017), *Parking Birmingham*, UCI Machine
Learning Repository, [doi:10.24432/C51K5Z](https://doi.org/10.24432/C51K5Z).
The record lists CC BY 4.0 and describes the original council data as UK OGL.
It flags missing values in its metadata; the local audit specifically finds no
null fields in the supplied four-column CSV. Missing scheduled times and long gaps
are a separate issue from null cells. Source metadata checked 7 November 2026.

## Fixed model settings

| Method | Settings |
|---|---|
| Ridge | alpha=1; standardized numeric inputs and one-hot park identity |
| Random forest | 150 trees; max_depth=16; min_samples_leaf=3; seed=42; n_jobs=1 |
| Gradient boosting | 150 iterations; max_leaf_nodes=15; learning_rate=0.08; early_stopping=False; seed=42 |
| Calendar boosting | Same boosting settings; capacity, target calendar and park only |

All unlisted estimator defaults are those of the locked scikit-learn version.
The numeric input columns and pipeline configuration are explicit in
`run_experiments.py`. These are deliberately fixed comparisons rather than a
hyperparameter-optimized leaderboard. Classification metrics describe thresholded
regression forecasts; this experiment does not train separate classifiers or
estimate calibrated availability probabilities.

After a run, verify and regenerate its readable results note with:

```bash
.venv/bin/python research/verify_results.py research/results/main
.venv/bin/python research/summarize_results.py research/results/main
```

The LaTeX table uses the standard `booktabs` table rules and `multirow` package. PDFs retain vector text
and lines; PNGs are supplied for review. In manuscripts, cite the protocol, actual
lead-time tolerance, error units, fold dates, and the complete-history seasonal
comparison together with the primary performance table.

Pooled metrics weight each eligible park/origin equally, so parks with more readings
have more weight. Pooled R² includes variation between parks and should be read
alongside per-park MAE. Observed examples have unique origin and target keys within
each horizon; the verifier checks this for the saved evaluation predictions.
Reproduce the three selected final-week ML fits as well as verifying stored metrics:

```bash
.venv/bin/python research/verify_results.py research/results/main --refit-selected
```

This refit checks each selected model against its recorded predictions, with
absolute tolerance 1e-12 and relative tolerance 1e-10 to allow CSV round-tripping.
It is separate from the five deterministic protocol tests. Refit verification
results are recorded in `research/validation.txt`.

The first bootstrap postprocessing pass exposed an index-order mismatch in truth
alignment. This was corrected before reporting results, with a regression test
that checks finite intervals and a known paired difference. The manifest records
the correction and initial source hash; model fits and saved predictions were
unchanged. The final source hash corresponds to the corrected reproducible runner.
