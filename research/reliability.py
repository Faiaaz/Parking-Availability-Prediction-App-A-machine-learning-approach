"""Causal observation-quality stress test on measured Birmingham targets."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from research.run_experiments import ROOT, clean_data, make_examples, nearest, sha

MODES = ('missing', 'block', 'stale', 'invalid', 'plausible')
BASE = ['current_input', 'previous_input', 'older_input', 'capacity', 'hour_sin',
        'hour_cos', 'dow_sin', 'dow_cos', 'park']
QUALITY = ['missing', 'invalid', 'age', 'jump', 'completeness', 'past_validity', 'quality']
CALENDAR = BASE[3:]


def observed_stream(clean, mode, severity, seed):
    """Corrupt only inputs. Each site's stream is shared across its forecast horizons."""
    rng = np.random.default_rng(seed)
    groups = []
    for _, site in clean.groupby('zone', sort=True):
        site = site.sort_values('time')
        measured = site.rate.reset_index(drop=True)
        time = site.time.reset_index(drop=True)
        selected = pd.Series(rng.random(len(site)) < severity)
        values = measured.copy()
        source = time.copy()
        if mode == 'block':
            selected = pd.Series(rng.random(len(site)) < severity / 4).rolling(4, min_periods=1).max().astype(bool)
        if mode in ('missing', 'block'):
            values[selected] = np.nan
        elif mode == 'stale':
            values[selected] = measured.shift()[selected]
            source[selected] = time.shift()[selected]
        elif mode == 'invalid':
            values[selected] = 1.2
        elif mode == 'plausible':
            values[selected] = (values[selected] + .15 * rng.choice([-1, 1], selected.sum())).clip(0, 1)
        elif mode != 'clean':
            raise ValueError(mode)
        valid = values.between(0, 1)
        current = values.where(valid).ffill()
        age = (time - source.where(valid).ffill()).dt.total_seconds().div(60).fillna(1e6)
        complete = valid.rolling(3, min_periods=1).mean()
        jump = current.diff().abs().fillna(0)
        quality = valid.astype(float) * np.exp(-age / 60) * (1 - jump) * complete
        groups.append(pd.DataFrame(dict(zone=site.zone.iloc[0], time=time, current_input=current,
            previous_input=current.shift(), older_input=current.shift(2), missing=values.isna(),
            invalid=values.notna() & ~valid, age=age, jump=jump, completeness=complete,
            past_validity=valid.shift().rolling(12, min_periods=1).mean(), quality=quality,
            injected=selected)))
    return pd.concat(groups, ignore_index=True)


def partition(frame):
    cuts = pd.to_datetime(['2016-11-29', '2016-12-06', '2016-12-13','2016-12-20'])
    return [frame[(frame.time >= start) & (frame.target_time < end)] for start, end in
            zip([pd.Timestamp.min, *cuts[:-1]], cuts)]


def history(frame, stream, horizon):
    joined = frame.merge(stream, on=['zone','time'], validate='one_to_one')
    for days in (1, 7):
        estimates = pd.Series(np.nan, index=joined.index)
        for zone, group in joined.groupby('zone'):
            available = stream[(stream.zone == zone) & ~stream.missing & ~stream.invalid & (stream.age == 0)]
            if available.empty:
                continue
            times = available.time.to_numpy(dtype='datetime64[ns]').astype('int64')
            wanted = (group.time + pd.Timedelta(minutes=horizon) - pd.Timedelta(days=days)).to_numpy(dtype='datetime64[ns]').astype('int64')
            estimates.loc[group.index] = nearest(times, available.current_input.to_numpy(), wanted, 300_000_000_000)[0]
        joined[f'season{days}'] = estimates.fillna(joined.current_input)
    return joined


def radius(errors, alpha=.1):
    errors = np.asarray(errors)
    rank = int(np.ceil((len(errors)+1)*(1-alpha)))
    if not len(errors) or rank > len(errors) or not np.isfinite(errors).all():
        raise ValueError('Insufficient finite calibration residuals')
    return np.sort(errors)[rank-1]


def evaluate(actual, pred, width):
    lo, hi = np.maximum(0, pred-width), np.minimum(1, pred+width)
    error = np.asarray(actual) - pred
    return dict(n=len(error), mae=np.abs(error).mean(), rmse=np.sqrt(np.mean(error**2)),
                coverage=np.mean((actual>=lo)&(actual<=hi)), width=np.mean(hi-lo),
                interval_score=np.mean(hi-lo+20*np.maximum(lo-actual,0)+20*np.maximum(actual-hi,0)))


def run(out):
    path = ROOT/'data/raw/dataset.csv'
    clean, _ = clean_data(path)
    scenarios = [('clean',0,0)] + [(mode,level,seed) for mode in MODES for level in (.1,.3) for seed in (17,29)]
    rows = []
    validation = []
    out.mkdir(parents=True, exist_ok=False)
    for h in (30,60,120):
        pairs = make_examples(clean,h)
        train = partition(pairs)[0]
        parks = {z:i for i,z in enumerate(sorted(train.zone.unique()))}
        pairs['park'] = pairs.zone.map(parks).fillna(-1)
        def build(mode, level, seed):
            return history(pairs, observed_stream(clean,mode,level,seed),h)
        reference = build('clean',0,0)
        training = pd.concat([partition(reference)[0]] + [partition(build(m,.2,41))[0] for m in MODES])
        forecasters = {}
        for name,cols in [('M1',BASE),('M2',BASE+QUALITY),('calendar',CALENDAR)]:
            estimator = HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=15,early_stopping=False,random_state=42)
            estimator.fit(training[cols],training.target)
            forecasters[name] = estimator,cols
        fallback = train.target.median()
        def predict(frame):
            preds = {name:model.predict(frame[cols]).clip(0,1) for name,(model,cols) in forecasters.items()}
            preds['M3'] = frame.quality.to_numpy()*preds['M2'] + (1-frame.quality.to_numpy())*preds.pop('calendar')
            for name,col in [('persistence','current_input'),('daily','season1'),('weekly','season7')]:
                preds[name] = frame[col].fillna(fallback).to_numpy()
            return preds
        _, val, cal, _ = partition(reference)
        radii = {name:radius(np.abs(cal.target-p)) for name,p in predict(cal).items()}
        for name,p in predict(val).items():
            validation.append(dict(horizon=h,model=name,n=len(val),mae=np.mean(np.abs(val.target-p)),n_cal=len(cal),radius=radii[name]))
        for mode,level,seed in scenarios:
            test = partition(reference if mode=='clean' else build(mode,level,seed))[3]
            auc = roc_auc_score(test.injected,1-test.quality) if test.injected.nunique()==2 else np.nan
            for name,p in predict(test).items():
                rows.append(dict(horizon=h,mode=mode,severity=level,seed=seed,model=name,
                                 quality_auc=auc, injected_share=test.injected.mean(), **evaluate(test.target,p,radii[name])))
        print('Finished horizon',h,flush=True)
    pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
    pd.DataFrame(validation).to_csv(out/'validation.csv',index=False)
    (out/'hashes.txt').write_text(f'input {sha(path)}\nrunner {sha(__file__)}\ntarget_builder {sha(ROOT/"research/run_experiments.py")}\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
