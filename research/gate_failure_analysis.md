# Fixed gate diagnosis

Executed validation results are in results/v2/pilot/gate_validation.csv. The table compares M2, original gating, constant-half mixing, freshness-only and validity-only mixing at each horizon, dataset and design. Calendar error, mean gate weight and fraction below one-half quantify the competing fallback and gate response.

The pilot can establish predictive counterfactual differences under specified mixing rules. It cannot establish a causal explanation from quality/error associations. Point-level switching trajectories and stratification by every site/time are still pending. The model receives q in M2, so forced interpolation may override a learned quality response. A weak calendar fallback is another testable explanation. Neither explanation is assumed true from the historical negative result alone.

At 30 minutes on future-date validation, Birmingham no-gate MAE was 2.667 pp, original-gate 3.294 pp, freshness-only 2.755 pp and calendar-alone 8.276 pp. Harvard corresponding values were 5.636, 7.020, 5.747 and 15.344 pp. This supports the narrower empirical conclusion that forcing more weight onto this fitted calendar predictor reduced validation accuracy in these settings. It does not isolate an underlying sensor-fault mechanism.
