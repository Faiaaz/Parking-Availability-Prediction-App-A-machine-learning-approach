"""Reproducible direct multi-horizon evaluation; never writes legacy artifacts."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/parking-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
KEY = ['zone', 'time']
SEED = 42
THRESHOLD = .85
NUM = ['capacity', 'current', 'lag1', 'lag2', 'rolling3', 'delta', 'age1', 'age2', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'weekend']
CAL = ['capacity', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'weekend']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_data(path):
    raw = pd.read_csv(path).rename(columns={'SystemCodeNumber':'zone', 'LastUpdated':'time', 'Capacity':'capacity', 'Occupancy':'occupancy'})
    raw['time'] = pd.to_datetime(raw.time, errors='raise')
    required = raw[['zone', 'time', 'capacity', 'occupancy']]
    if required.isna().any().any() or (raw.capacity <= 0).any():
        raise ValueError('Missing observations or nonpositive capacity require explicit resolution')
    conflicts = raw.groupby(KEY)[['capacity', 'occupancy']].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError('Conflicting duplicate keys: no automatic resolution permitted')
    invalid = (raw.occupancy < 0) | (raw.occupancy > raw.capacity)
    audit = dict(raw_rows=len(raw), zones=raw.zone.nunique(), exact_duplicate_excess=int(raw.duplicated().sum()), duplicate_key_excess=int(raw.duplicated(KEY).sum()), conflicting_keys=int(conflicts.sum()), invalid_occupancy_rows=int(invalid.sum()), missing=required.isna().sum().to_dict(), start=str(raw.time.min()), end=str(raw.time.max()))
    # Exclude invalid sensor values rather than silently changing measured labels.
    clean = raw.drop_duplicates(KEY).loc[lambda x: (x.occupancy >= 0) & (x.occupancy <= x.capacity)].sort_values(KEY).reset_index(drop=True)
    clean['rate'] = clean.occupancy / clean.capacity
    audit['clean_rows'] = len(clean)
    audit['clean_interval_minutes_quantiles'] = clean.groupby('zone').time.diff().dt.total_seconds().div(60).quantile([0,.25,.5,.75,.95,1]).to_dict()
    return clean, audit


def nearest(times, values, wanted, tolerance):
    """Nearest recorded observation, ties resolved toward earlier timestamp."""
    idx = np.searchsorted(times, wanted)
    left = np.clip(idx - 1, 0, len(times)-1)
    right = np.clip(idx, 0, len(times)-1)
    use = np.where(np.abs(times[left]-wanted) <= np.abs(times[right]-wanted), left, right)
    valid = np.abs(times[use]-wanted) <= tolerance
    return np.where(valid, values[use], np.nan), times[use], valid


def make_examples(clean, horizon, tolerance=5):
    frames = []
    minute = 60 * 10**9
    for zone, z in clean.groupby('zone', sort=True):
        z = z.sort_values('time').copy()
        t = z.time.astype('datetime64[ns]').astype('int64').to_numpy()
        y = z.rate.to_numpy()
        desired = t + horizon * minute
        target, target_t, valid = nearest(t, y, desired, tolerance * minute)
        z['target'] = target
        z['target_time'] = pd.to_datetime(target_t)
        z['nominal_target_time'] = pd.to_datetime(desired)
        z['lead_minutes'] = (target_t - t) / minute
        z['current'] = y
        z['lag1'] = z.rate.shift(1)
        z['lag2'] = z.rate.shift(2)
        z['rolling3'] = z.rate.rolling(3, min_periods=3).mean()
        z['delta'] = z.current - z.lag1
        z['age1'] = z.time.diff().dt.total_seconds() / 60
        z['age2'] = (z.time-z.time.shift(2)).dt.total_seconds() / 60
        for days, name in [(1, 'daily'), (7, 'weekly')]:
            season, season_t, ok = nearest(t, y, desired - days*24*60*minute, tolerance*minute)
            ok &= season_t <= t
            z[name] = np.where(ok, season, np.nan)
            z[name+'_source_time'] = pd.to_datetime(np.where(ok, season_t, np.iinfo(np.int64).min))
        hour = z.nominal_target_time.dt.hour + z.nominal_target_time.dt.minute / 60
        dow = z.nominal_target_time.dt.day_of_week
        z['hour_sin'], z['hour_cos'] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
        z['dow_sin'], z['dow_cos'] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
        z['weekend'] = (dow >= 5).astype(int)
        frames.append(z.loc[valid & (target_t > t)].dropna(subset=NUM))
    out = pd.concat(frames, ignore_index=True).sort_values(['time','zone']).reset_index(drop=True)
    assert not out.duplicated(KEY).any()
    assert (out.target_time > out.time).all()
    return out


def windows(clean, n_folds=4, days=7):
    end = clean.time.max().normalize() + pd.Timedelta(days=1)
    holdout = end - pd.Timedelta(days=days)
    for i in range(n_folds):
        start = holdout - pd.Timedelta(days=days*(n_folds-i))
        yield f'cv_{i+1}', start, start+pd.Timedelta(days=days)
    yield 'holdout', holdout, end


def split_examples(examples, start, end):
    train = examples.loc[(examples.time < start) & (examples.target_time < start)].copy()
    test = examples.loc[(examples.time >= start) & (examples.time < end) & (examples.target_time < end)].copy()
    if train.empty or test.empty:
        raise ValueError('Empty chronological window')
    assert train.target_time.max() < test.time.min()
    assert not set(map(tuple, train[KEY].to_numpy())) & set(map(tuple, test[KEY].to_numpy()))
    return train, test


def models():
    def prep(cols):
        return ColumnTransformer([('num', StandardScaler(), cols), ('zone', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['zone'])])
    return {
        'ridge': make_pipeline(prep(NUM), Ridge(alpha=1)),
        'random_forest': make_pipeline(prep(NUM), RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=3, random_state=SEED, n_jobs=1)),
        'gradient_boosting': make_pipeline(prep(NUM), HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=.08, early_stopping=False, random_state=SEED)),
        'calendar_gradient_boosting': make_pipeline(prep(CAL), HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=.08, early_stopping=False, random_state=SEED)),
    }


def score(g):
    y, p = g.actual.to_numpy(), g.prediction.to_numpy()
    a, b = y < THRESHOLD, p < THRESHOLD
    return dict(n=len(g), mae=mean_absolute_error(y,p), rmse=np.sqrt(mean_squared_error(y,p)), r2=r2_score(y,p), f1_available=f1_score(a,b,zero_division=0), f1_full=f1_score(~a,~b,zero_division=0), balanced_accuracy=balanced_accuracy_score(a,b), roc_auc=roc_auc_score(a,1-p) if len(np.unique(a))==2 else None)


def summarize(pred):
    rows = []
    for (split,h,m), g in pred.groupby(['split','horizon','model'], sort=True):
        rows.append(dict(split=split,horizon=h,model=m,**score(g)))
    return pd.DataFrame(rows)


def uncertainty(pred, repetitions=1000):
    """Paired daily block bootstrap of MAE difference; days are resampled jointly."""
    rng = np.random.default_rng(SEED)
    rows = []
    for (split,h), g in pred.assign(phase=np.where(pred.split=='holdout','holdout','cv')).groupby(['phase','horizon']):
        p = g.pivot(index=['time','zone'], columns='model', values='prediction')
        actual = g.drop_duplicates(KEY).set_index(['time','zone']).actual.reindex(p.index)
        errors = p.sub(actual, axis=0).abs()
        dates = pd.to_datetime(p.index.get_level_values('time')).normalize()
        sums = errors.groupby(dates).sum()
        counts = errors.groupby(dates).count()
        draws = rng.integers(0,len(sums),(repetitions,len(sums)))
        for model in errors.columns:
            diff = sums[model]-sums.persistence
            boot = diff.to_numpy()[draws].sum(axis=1)/counts[model].to_numpy()[draws].sum(axis=1)
            rows.append(dict(phase=split,horizon=h,model=model,days=len(sums),mae_difference_vs_persistence=float(errors[model].mean()-errors.persistence.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    return pd.DataFrame(rows)


def render(pred, metrics, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig, name):
        fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
        fig.savefig(out/f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    pooled = summarize(pred.assign(split=np.where(pred.split=='holdout','holdout','cv')))
    pooled.to_csv(out/'pooled_metrics.csv', index=False)
    pivot = pooled.pivot(index=['split','model'], columns='horizon', values=['mae','rmse'])
    pivot.to_latex(out/'evaluation_table.tex', float_format='%.4f', escape=True, caption='Occupancy-rate errors in fractions of capacity (multiply by 100 for percentage points). Horizons are minutes. CV pools out-of-fold predictions; holdout is a separate final week.', label='tab:parking_errors')
    pivot.to_csv(out/'evaluation_table.csv')
    fig, axes = plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    colors = plt.get_cmap('tab10')
    order = sorted(pred.model.unique())
    for ax, phase in zip(axes,['cv','holdout']):
        for i,m in enumerate(order):
            q = pooled[(pooled.split==phase)&(pooled.model==m)].sort_values('horizon')
            ax.plot(q.horizon,q.mae*100,'o-',label=m.replace('_',' '),color=colors(i))
        ax.set(title='Rolling-origin CV' if phase=='cv' else 'Final seven-day holdout',xlabel='Nominal horizon (minutes)',xticks=[30,60,120])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('MAE (occupancy percentage points)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)
    save(fig,'horizon_errors')
    fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=True)
    cv = metrics[metrics.split!='holdout']
    for ax,h in zip(axes,[30,60,120]):
        for i,m in enumerate(order):
            q=cv[(cv.horizon==h)&(cv.model==m)].sort_values('split')
            ax.plot(range(1,len(q)+1),q.mae*100,'o-',color=colors(i),label=m.replace('_',' '))
        ax.set(title=f'{h}-minute forecast',xlabel='Validation fold',xticks=[1,2,3,4]);ax.grid(alpha=.2)
    axes[0].set_ylabel('MAE (percentage points)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False)
    save(fig,'fold_stability')
    # Per-zone holdout error for the model selected using CV only.
    selection=pooled[pooled.split=='cv'].sort_values('mae').groupby('horizon').first().model
    zone_rows=[]
    fig,ax=plt.subplots(figsize=(9,5))
    for h in [30,60,120]:
        q=pred[(pred.split=='holdout')&(pred.horizon==h)&(pred.model==selection[h])].copy()
        q['error']=(q.actual-q.prediction).abs()*100
        v=q.groupby('zone').error.mean().sort_index()
        ax.plot(v.values,range(len(v)),'o',label=f'{h} min: {selection[h].replace("_"," ")}')
        zone_rows.extend(dict(horizon=h,model=selection[h],zone=z,mae_percentage_points=e) for z,e in v.items())
    ax.set(yticks=range(len(v)),yticklabels=v.index,xlabel='Holdout MAE (percentage points)',title='Per-car-park error: model selected using CV only')
    ax.tick_params(axis='y',labelsize=7);ax.legend(fontsize=7);ax.grid(axis='x',alpha=.2)
    save(fig,'zone_errors')
    pd.DataFrame(zone_rows).to_csv(out/'zone_errors.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=True)
    base=pred[(pred.split=='holdout')&(pred.model=='persistence')]
    for ax,h in zip(axes,[30,60,120]):
        q=base[base.horizon==h]
        ax.hist(q.lead_minutes,bins=20,color='#0072B2',edgecolor='white')
        ax.axvline(h,color='#D55E00',linestyle='--');ax.set(title=f'Nominal {h} minutes',xlabel='Actual lead (minutes)')
    axes[0].set_ylabel('Matched observations')
    save(fig,'actual_lead_times')
    return pooled, selection.to_dict()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'data/raw/dataset.csv')
    ap.add_argument('--output',type=Path,default=ROOT/'research/results/main')
    ap.add_argument('--tolerance',type=float,default=5)
    args=ap.parse_args()
    if not 0 <= args.tolerance < 30:
        ap.error('Tolerance must be nonnegative and less than the shortest horizon')
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    clean,audit=clean_data(args.input)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    predictions=[]; split_rows=[]; coverage=[]
    for h in [30,60,120]:
        examples=make_examples(clean,h,args.tolerance)
        coverage.append(dict(horizon=h,clean_observations=len(clean),eligible_examples=len(examples),lead_min=float(examples.lead_minutes.min()),lead_max=float(examples.lead_minutes.max()),lead_median=float(examples.lead_minutes.median())))
        for split,start,end in windows(clean):
            train,test=split_examples(examples,start,end)
            split_rows.append(dict(split=split,horizon=h,start=str(start),end_exclusive=str(end),n_train=len(train),n_test=len(test),last_training_label=str(train.target_time.max()),first_test_origin=str(test.time.min()),daily_coverage=float(test.daily.notna().mean()),weekly_coverage=float(test.weekly.notna().mean()),both_seasons_coverage=float(test[['daily','weekly']].notna().all(axis=1).mean())))
            preds={'persistence':test.current.to_numpy(),'daily_seasonal_fallback':test.daily.fillna(test.current).to_numpy(),'weekly_seasonal_fallback':test.weekly.fillna(test.current).to_numpy()}
            for name,model in models().items():
                model.fit(train[NUM+['zone']],train.target)
                preds[name]=np.clip(model.predict(test[NUM+['zone']]),0,1)
            for name,p in preds.items():
                q=test[['zone','time','target_time','lead_minutes']].copy()
                q['actual']=test.target;q['prediction']=p;q['model']=name;q['split']=split;q['horizon']=h
                q['daily_observed']=test.daily.notna();q['weekly_observed']=test.weekly.notna()
                predictions.append(q)
            print(f'{h} min {split}: train={len(train)}, test={len(test)}',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(out/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(split_rows).to_csv(out/'splits.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'horizon_coverage.csv',index=False)
    metrics=summarize(pred);metrics.to_csv(out/'fold_metrics.csv',index=False)
    # Seasonal comparison without persistence fallback, identical complete-history subset.
    strict=pred[pred.daily_observed & pred.weekly_observed]
    summarize(strict).to_csv(out/'complete_seasonal_history_metrics.csv',index=False)
    uncertainty(pred).to_csv(out/'paired_daily_bootstrap.csv',index=False)
    pooled,selection=render(pred,metrics,out)
    package_names=['numpy','pandas','scikit-learn','scipy','matplotlib','joblib','threadpoolctl','pyarrow']
    versions={p:importlib.metadata.version(p) for p in package_names if importlib.util.find_spec(p.replace('-','_')) or p in ['scikit-learn']}
    (out/'environment-lock.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    protected=[ROOT/'reports/metrics.json',ROOT/'reports/holdout_sample.csv',*sorted((ROOT/'models').glob('*.joblib')),*sorted((ROOT/'reports').glob('*.joblib')),ROOT/'data/processed/birmingham_features.parquet']
    manifest=dict(seed=SEED,python=sys.version,platform=platform.platform(),packages=versions,input_sha256=sha(args.input),script_sha256=sha(__file__),command=sys.argv,selected_by_cv=selection,tolerance_minutes=args.tolerance,threshold=THRESHOLD,protocol='Direct per-horizon regression; expanding training; target_time < evaluation start; online observed history; invalid values excluded; seasonal missing values fall back to persistence; fixed hyperparameters; no tuning on holdout',legacy_hashes={str(p.relative_to(ROOT)):sha(p) for p in protected if p.exists()})
    manifest['output_hashes']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pooled.to_string(index=False),flush=True)

if __name__=='__main__':
    main()
