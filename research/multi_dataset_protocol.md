# Multi-dataset pilot protocol v2

Configuration is research/multidata/config.json. Commands from the repository root are:

    python -m research.multidata.acquire
    python -m unittest discover -s tests -v
    # Preserve the published results before recreating the fixed output path.
    mv research/results/v2/pilot research/results/v2/pilot_reference
    python -m research.multidata.pilot
    python -m research.multidata.summarize

Use the existing environment-freeze.txt plus pypdf==7.1.0 for documentation extraction. Pilot output directories must be new. The original experiment and result directories are retained. Raw downloads and per-origin interval predictions are ignored by Git. Source hashes, parameters, input hashes and exclusions are saved in the pilot manifest.

Training/validation/calibration/test dates are fixed in config. Harvard uses July 4/11/18/25 boundaries; Birmingham retains its original boundaries for exploratory diagnosis. The first lexicographic site is reserved for the held-site design before fitting and excluded from training, validation and calibration. Its observed online history is available at prediction time. Models omit the site-ID feature in both designs, allowing a meaningful unseen-site prediction.

M1/M2/calendar use equal 60-iteration boosting budgets, seed 42 and six training copies (clean plus five mechanisms, severity .2, seed 41). The original fixed M3 gate is retained. Validation counterfactuals compare original, no gate, constant-half, freshness and validity gates. None is selected on test performance.

U0 uses a pooled calibration absolute-residual quantile. U1 fits a residual-scale model on validation using ordinary features. U2 uses the same model budget with quality features added. Calibration normalizes residuals by predicted scale, floored at .001. The M2 point forecast remains frozen. All three calibrators receive the identical clean-plus-degraded calibration mixture, with independent corruption seed 53. This differs from v1 clean-only calibration and changes the scientific question to in-mixture calibration transport.

Evaluate clean and five corruptions at severity .2, seeds 17/29. Save point errors, coverage, width, interval score and sample sizes. This bounded pilot has one severity and one model seed; the old Birmingham experiment retains two severities. Paired daily bootstrap resamples whole dates, pooling sites and all scenario replicates within each date. With only seven test dates and shared dependence across days, intervals are descriptive. Held-site calibration relies on transport from other sites, which can fail.

The reference-directory rename requires that pilot_reference does not already exist. Compare newly generated tables to that preserved reference. Raw data remain governed by the source licenses recorded in external_data_audit.md.
