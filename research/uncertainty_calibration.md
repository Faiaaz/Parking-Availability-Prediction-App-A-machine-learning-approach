# Quality-conditioned interval study

Executed scores.csv and summary.csv compare U0/U1/U2 on the same frozen M2 predictions and identical calibration mixtures. Paired date-block differences in interval score appear in paired_daily_intervals.csv. Lower score is preferable; coverage and width jointly describe calibration and sharpness. coverage_width.pdf/png plot those trade-offs.

U1 is a heteroscedastic residual-scale baseline using ordinary inputs. U2 adds quality features with identical fitting capacity. This establishes an equal-budget conditioning comparison. It does not establish calibrated reliability probabilities. Nominal 90% intervals have no asserted coverage guarantee under time dependence, unseen sites, truncation and distribution shift. Birmingham outcomes have been inspected previously. External convenience-sample results require replication on additional sites and prespecified test periods.

Measured future-date degraded SFpark coverage at 30/60/120 minutes was U0 88.81/89.26/88.00%, U1 88.80/88.89/88.63%, U2 88.50/87.61/88.46%. Corresponding interval scores were U0 .3635/.4290/.5281, U1 .3498/.4057/.4884 and U2 .3411/.3964/.4849.

The held-out SFpark segment reverses that favorable score pattern. U2 coverage was 81.6/70.9/62.6%, versus U0 86.4/81.1/71.7%. U2 interval scores .422/.672/1.199 exceeded U0 .399/.566/1.033 at all horizons. This is a documented generalization failure. U2 is not supported as a universally superior calibrator.
