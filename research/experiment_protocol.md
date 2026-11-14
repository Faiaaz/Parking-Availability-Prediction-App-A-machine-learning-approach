# Executable ablation protocol

From the project root use the Python version in results/main/manifest.json and pinned packages in research/environment-freeze.txt. No new packages are required.

    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m research.reliability --output research/results/reliability_reproduction

Use a new output directory. hashes.txt records input and computational source digests. Fixed site ordering, seeded corruption and one numerical training thread support reproduction. scores.csv contains one row per method, horizon and test condition. validation.csv records diagnostic validation MAE and model-specific clean calibration radii.

1. Construct direct site-level labels using the original ±5 minute matching rule. Forecast only at known valid reference observation times. Alter the input stream while retaining the same origin/label pairs for every method.
2. Use expanding historical information with fixed date partitions. Train before Nov 29, validate Nov 29–Dec 5, calibrate Dec 6–12 and evaluate Dec 13–19 (2016). Require actual target timestamps before each partition's exclusive end. Models are frozen after training. Validation is diagnostic; hyperparameters were fixed. Only observed history can cross boundaries.
3. Train all learned estimators on clean plus five separately corrupted copies of training examples (severity 0.2, seed 41). Copies share targets and do not increase independent sample size. Boosting uses seed 42, 60 iterations, 15 leaves and disabled early stopping. Site coding comes from training. Model training uses no scenario identifiers or injection labels.
4. At test, compare clean to independent missing readings, four-reading missing blocks, previous reading replayed with original timestamp, impossible ratio 1.2, and plausible ±0.15 changes clipped to physical limits. Use probabilities 0.1/0.3 and seeds 17/29. Block starts have probability severity/4 with overlapping runs allowed. Missingness acts on the known observation schedule. Carry usable values forward per site; unavailable initial history remains missing for boosting.
5. Compare M0/M1/M2/M3 on each identical cohort. Compare M1 to M2 for feature access and M2 to M3 for the gate. Equal-weight the 20 degraded conditions per horizon, report clean separately. Lower MAE/RMSE and interval score favor a method; coverage must be read with width. Seeds perturb the same test outcomes and cannot serve as independent temporal replicates.
6. A confirmatory study would add calendar alone, constant fusion weights and one-factor gate ablations, freeze choices on validation, and apply paired date-block uncertainty on a fresh evaluation period. These extensions were not used to select or revise the current gate.
