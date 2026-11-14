# Quality validation

quality_detection.csv compares 1−q with the simple missing-or-invalid rule using AUROC and average precision against known synthetic assignment. Clean single-class conditions have undefined AUROC. Assignment is not guaranteed to change a value after clipping, so these statistics measure recovery of interventions. Their rows repeat labels across horizons and designs.

Also report correlation of each prediction-time indicator with subsequent M2 absolute forecast error. That association uses future outcomes only for evaluation. It does not enter feature construction. No real-world measurement-error detection accuracy is claimed. Harvard taxi simulations and injected corruption are distinct sources. Official SFMTA unknown-time metadata were not available as a loaded reference for this pilot.
