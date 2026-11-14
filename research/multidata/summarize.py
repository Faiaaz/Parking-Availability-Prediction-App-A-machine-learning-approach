"""Summaries and paired date-block uncertainty from executed pilot predictions."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/parking-matplotlib')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('research/results/v2/pilot')
    scores=pd.read_csv(root/'scores.csv')
    scores['condition']=np.where(scores['mode']=='clean','clean','degraded')
    summary=scores.groupby(['dataset','design','horizon','condition','model','interval'])[['mae','coverage','width','interval_score']].mean()
    summary.to_csv(root/'summary.csv')
    predictions=pd.read_csv(root/'interval_predictions.csv.gz',parse_dates=['time'],dtype={'zone':str})
    predictions['interval_score']=predictions.upper-predictions.lower+20*np.maximum(predictions.lower-predictions.target,0)+20*np.maximum(predictions.target-predictions.upper,0)
    predictions['day']=predictions.time.dt.normalize()
    rows=[]
    rng=np.random.default_rng(42)
    for keys,group in predictions.groupby(['dataset','design','horizon']):
        daily=group.groupby(['day','interval']).interval_score.mean().unstack()
        for comparator in ['U0','U1']:
            differences=(daily.U2-daily[comparator]).to_numpy()
            samples=rng.choice(differences,(1000,len(differences)),replace=True).mean(axis=1)
            rows.append(dict(dataset=keys[0],design=keys[1],horizon=keys[2],comparison='U2-'+comparator,
                mean_difference=differences.mean(),lower=np.quantile(samples,.025),upper=np.quantile(samples,.975),days=len(differences)))
    pd.DataFrame(rows).to_csv(root/'paired_daily_intervals.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,dataset in zip(axes,['birmingham','harvard_sfpark']):
        subset=scores[(scores.dataset==dataset)&(scores.design=='future_date')&(scores.model=='M2')&(scores.condition=='degraded')]
        for method,group in subset.groupby('interval'):
            points=group.groupby('horizon')[['width','coverage']].mean()
            ax.plot(points.width*100,points.coverage,'o-',label=method)
        ax.axhline(.9,color='gray',linestyle='--')
        ax.set(title=dataset,xlabel='Mean width (occupancy percentage points)',ylabel='Empirical coverage',ylim=(0,1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(root/'coverage_width.png',dpi=300)
    fig.savefig(root/'coverage_width.pdf')
    print(summary.loc[(slice(None),'future_date',slice(None),'degraded','M2',slice(None)),:].round(4).to_string())


if __name__=='__main__':main()
