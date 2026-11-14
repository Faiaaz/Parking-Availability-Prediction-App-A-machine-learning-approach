# Limitations v2

Sensor references may contain plausible errors. Harvard upstream plausibility filtering and byte-prefix selection alter the population. Capacity-zero exclusion conditions the task on usable space counts. API availability timestamps and timezone definitions are incomplete. The source's API status semantics can include unknown-state handling that must be reconciled against the official SFMTA guide before interpreting ratios as exact occupancy truth.

Melbourne's partial historical events do not establish state continuity. Departure timestamps can reveal future information if misused, so the current adapter is for retrospective labels only. No continuous Melbourne predictor is evaluated. Official hourly SFMTA data were not acquired. Simulated taxi coverage is excluded from this measured-column pilot.

The v2 model removes park-ID input and uses a calibration corruption mixture. Its results must not be pooled with v1 as if the protocol were identical. The bounded run has one severity and one fitting seed. Independent sensor-error validation, representative external sampling, multiple held-site folds, full spatial features and true cross-city model transfer remain incomplete. Date bootstrap uncertainty uses only seven days and does not remove all temporal dependence.
