"""Versioned city-specific pilots, validation gate diagnosis and interval comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from research.reliability import observed_stream, history, radius, evaluate, BASE, QUALITY, MODES
from research.run_experiments import ROOT, make_examples, sha
from research.multidata.adapters import birmingham, harvard, melbourne_intervals

CONFIG = json.loads(Path(__file__).with_name('config.json').read_text())
FEATURES = [c for c in BASE if c != 'park']
CALENDAR = FEATURES[3:]


def partitions(frame, cuts):
    bounds = [pd.Timestamp.min, *pd.to_datetime(cuts)]
    return [frame[(frame.time>=a)&(frame.target_time<b)] for a,b in zip(bounds,bounds[1:])]


def estimator():
    return HistGradientBoostingRegressor(max_iter=CONFIG['model_iterations'],
        max_leaf_nodes=15, early_stopping=False, random_state=CONFIG['seed'])


def main():
    out = ROOT/'research/results/v2/pilot'
    out.mkdir(parents=True,exist_ok=False)
    all_scores, diagnoses, detections, predictions, split_counts = [],[],[],[],[]
    audits = {}
    sources = {'birmingham':ROOT/'data/raw/dataset.csv',
               'harvard_sfpark':ROOT/'data/external/v2/harvard_prefix.tab'}
    for dataset, adapter in [('birmingham',birmingham),('harvard_sfpark',harvard)]:
        clean,audits[dataset] = adapter(sources[dataset])
        for horizon in CONFIG['horizons']:
            pairs = make_examples(clean,horizon)
            def build(mode='clean',seed=0):
                return history(pairs,observed_stream(clean,mode,CONFIG['severity'] if mode!='clean' else 0,seed),horizon)
            frames = [build()] + [build(mode,41) for mode in MODES]
            cuts = CONFIG[dataset+'_cuts']
            for design in ['future_date','held_site']:
                # Fixed lexicographically first site, chosen without outcome inspection.
                held = sorted(clean.zone.unique())[0]
                def select(frame, test=False):
                    if design=='future_date':
                        return frame
                    return frame[frame.zone.eq(held) if test else frame.zone.ne(held)]
                train = pd.concat([select(partitions(f,cuts)[0]) for f in frames])
                val = pd.concat([select(partitions(f,cuts)[1]) for f in frames])
                cal = pd.concat([select(partitions(build(mode,53),cuts)[2]) for mode in ['clean',*MODES]])
                fitted = {}
                for name,cols in [('M1',FEATURES),('M2',FEATURES+QUALITY),('calendar',CALENDAR)]:
                    model = estimator().fit(train[cols],train.target)
                    fitted[name] = (model,cols)
                def predict(frame):
                    p = {name:m.predict(frame[cols]).clip(0,1) for name,(m,cols) in fitted.items()}
                    q = frame.quality.to_numpy()
                    p['M3'] = q*p['M2']+(1-q)*p['calendar']
                    for name,col in [('persistence','current_input'),('daily','season1'),('weekly','season7')]:
                        p[name]=frame[col].fillna(train.target.median()).to_numpy()
                    return p
                vp = predict(val)
                gates = {'original':val.quality.to_numpy(),'constant_half':np.full(len(val),.5),
                         'freshness':np.exp(-val.age.to_numpy()/60),
                         'validity':(~val.missing & ~val.invalid).to_numpy().astype(float),
                         'no_gate':np.ones(len(val))}
                for gate,weights in gates.items():
                    p=weights*vp['M2']+(1-weights)*vp['calendar']
                    diagnoses.append(dict(dataset=dataset,horizon=horizon,design=design,gate=gate,
                        n=len(val),mae=np.mean(np.abs(val.target-p)),calendar_mae=np.mean(np.abs(val.target-vp['calendar'])),
                        mean_weight=np.mean(weights),weight_below_half=np.mean(weights<.5)))
                # Residual-scale predictors use validation residuals only.
                scales={}
                residual = np.abs(val.target-vp['M2']).clip(.001)
                for name,cols in [('U1',FEATURES),('U2',FEATURES+QUALITY)]:
                    scales[name]=(estimator().fit(val[cols],residual),cols)
                cp=predict(cal)
                radii={name:radius(np.abs(cal.target-p)) for name,p in cp.items()}
                factors={}
                for name,(model,cols) in scales.items():
                    factors[name]=radius(np.abs(cal.target-cp['M2'])/np.maximum(model.predict(cal[cols]),.001))
                split_counts.append(dict(dataset=dataset,horizon=horizon,design=design,
                    train=len(train),validation=len(val),calibration=len(cal),held_site=held if design=='held_site' else ''))
                for mode in ['clean',*MODES]:
                    for seed in ([0] if mode=='clean' else CONFIG['test_seeds']):
                        test=select(partitions(build(mode,seed),cuts)[3],test=True)
                        if test.empty:
                            raise ValueError(f'Empty test {dataset} {design}')
                        pred=predict(test)
                        for score_name,score in [('heuristic',1-test.quality),('simple_rule',(test.missing|test.invalid).astype(float))]:
                            detections.append(dict(dataset=dataset,design=design,horizon=horizon,mode=mode,seed=seed,
                                indicator=score_name,n=len(test),auc=roc_auc_score(test.injected,score) if test.injected.nunique()==2 else np.nan,
                                average_precision=average_precision_score(test.injected,score) if test.injected.any() else np.nan,
                                error_correlation=pd.Series(np.asarray(score)).corr(pd.Series(np.abs(test.target.to_numpy()-pred['M2'])))))
                        for model,p in pred.items():
                            all_scores.append(dict(dataset=dataset,design=design,horizon=horizon,mode=mode,seed=seed,
                                model=model,interval='U0',**evaluate(test.target,p,radii[model])))
                        for method in ['U0','U1','U2']:
                            width=np.full(len(test),radii['M2'])
                            if method!='U0':
                                model,cols=scales[method]
                                width=factors[method]*np.maximum(model.predict(test[cols]),.001)
                                all_scores.append(dict(dataset=dataset,design=design,horizon=horizon,mode=mode,seed=seed,
                                    model='M2',interval=method,**evaluate(test.target,pred['M2'],width)))
                            saved=test[['zone','time','target_time','target']].copy()
                            saved['prediction']=pred['M2'];saved['lower']=np.maximum(0,pred['M2']-width)
                            saved['upper']=np.minimum(1,pred['M2']+width)
                            for key,value in dict(dataset=dataset,design=design,horizon=horizon,mode=mode,seed=seed,interval=method).items():
                                saved[key]=value
                            predictions.append(saved)
                print(dataset,horizon,design,flush=True)
    for name,rows in [('scores',all_scores),('gate_validation',diagnoses),('quality_detection',detections),('splits',split_counts)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    pd.concat(predictions).to_csv(out/'interval_predictions.csv.gz',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(config=CONFIG,audits=audits,
        inputs={k:sha(v) for k,v in sources.items()},code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}),indent=2))
    events=melbourne_intervals(ROOT/'data/external/v2/melbourne_sample.csv')
    events.groupby('site_id').size().to_csv(out/'melbourne_sample_counts.csv')


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
