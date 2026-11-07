"""Generate a readable results note exclusively from saved evaluation outputs."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    out=ap.parse_args().run
    m=pd.read_csv(out/'pooled_metrics.csv')
    manifest=json.loads((out/'manifest.json').read_text())
    coverage=pd.read_csv(out/'horizon_coverage.csv')
    splits=pd.read_csv(out/'splits.csv')
    intervals=pd.read_csv(out/'paired_daily_bootstrap.csv')
    audit=json.loads((out/'audit.json').read_text())
    lines=['# Measured research results','',f"Source: `{out.as_posix()}`. Input SHA-256: `{manifest['input_sha256']}`.",'',f"Cleaning retains {audit['clean_rows']:,} observations after removing duplicate keys and invalid measurements.",'','The following model was selected separately at each horizon using the lowest pooled CV MAE. Final-week metrics did not select the model. MAE and RMSE are occupancy percentage points.','', '| Horizon | CV-selected method | CV MAE | Holdout MAE | Persistence MAE | Holdout RMSE | Holdout n |','|---|---|---|---|---|---|---|']
    for h in [30,60,120]:
        name=manifest['selected_by_cv'][str(h)]
        cv=m[(m.split=='cv')&(m.horizon==h)&(m.model==name)].iloc[0]
        test=m[(m.split=='holdout')&(m.horizon==h)&(m.model==name)].iloc[0]
        base=m[(m.split=='holdout')&(m.horizon==h)&(m.model=='persistence')].iloc[0]
        lines.append(f'| {h} min | {name} | {100*cv.mae:.3f} | {100*test.mae:.3f} | {100*base.mae:.3f} | {100*test.rmse:.3f} | {int(test.n):,} |')
    lines+=['','## Baseline comparison on the final week','','| Horizon | Method | MAE (pp) | RMSE (pp) |','|---|---|---|---|']
    for _,r in m[(m.split=='holdout')&m.model.isin(['persistence','daily_seasonal_fallback','weekly_seasonal_fallback'])].iterrows():
        lines.append(f'| {r.horizon} | {r.model} | {r.mae*100:.3f} | {r.rmse*100:.3f} |')
    lines+=['','## Coverage and uncertainty','']
    for _,r in coverage.iterrows():
        lines.append(f"- {int(r.horizon)} minutes: {int(r.eligible_examples):,} eligible examples across the full period; actual leads {r.lead_min:.3f}–{r.lead_max:.3f} minutes (median {r.lead_median:.3f}).")
    for h in [30,60,120]:
        name=manifest['selected_by_cv'][str(h)]
        r=intervals[(intervals.phase=='holdout')&(intervals.horizon==h)&(intervals.model==name)].iloc[0]
        s=splits[(splits.split=='holdout')&(splits.horizon==h)].iloc[0]
        lines.append(f"- {h} minutes, {name}: paired daily bootstrap MAE difference against persistence {r.mae_difference_vs_persistence*100:.3f} pp, 95% interval [{r.ci_low*100:.3f}, {r.ci_high*100:.3f}] pp; daily/weekly seasonal coverage {s.daily_coverage:.1%}/{s.weekly_coverage:.1%}.")
    lines+=['','Intervals are descriptive, based on seven holdout origin dates; they do not account fully for dependence across days, model selection, or multiple comparisons. Seasonal fallback methods use persistence where historical matches are absent; the complete-history comparison is saved separately. Cohorts vary by horizon. The final week overlaps the legacy evaluation period, so it is not previously unseen external data. See `research/README.md` for the full protocol and limitations.','', 'No historical reported metrics were modified. All values above are calculated from saved predictions, not inferred or fabricated.','']
    path=out.parent.parent/'RESULTS.md'
    path.write_text('\n'.join(lines));print(path)

if __name__=='__main__':main()
