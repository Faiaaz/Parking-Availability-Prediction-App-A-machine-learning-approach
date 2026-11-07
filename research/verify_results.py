"""Verify recorded hashes, temporal splits, cohorts, and recomputed metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .run_experiments import ROOT, sha, summarize, clean_data, make_examples, split_examples, windows, models, NUM
except ImportError:
    from run_experiments import ROOT, sha, summarize, clean_data, make_examples, split_examples, windows, models, NUM


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--refit-selected',action='store_true',help='Refit each CV-selected ML method on final-week training data and compare saved predictions')
    args=ap.parse_args();out=args.run
    manifest=json.loads((out/'manifest.json').read_text())
    assert sha(ROOT/'data/raw/dataset.csv')==manifest['input_sha256']
    assert sha(ROOT/'research/run_experiments.py')==manifest['script_sha256']
    for name,digest in manifest['output_hashes'].items():
        assert sha(out/name)==digest,name
    for name,digest in manifest['legacy_hashes'].items():
        assert sha(ROOT/name)==digest,name
    p=pd.read_csv(out/'predictions.csv.gz',parse_dates=['time','target_time'])
    assert (p.target_time>p.time).all()
    np.testing.assert_allclose((p.target_time-p.time).dt.total_seconds()/60,p.lead_minutes)
    assert ((p.lead_minutes-p.horizon).abs()<=manifest['tolerance_minutes']+1e-9).all()
    assert not p.duplicated(['split','horizon','model','time','zone']).any()
    assert not p.duplicated(['horizon','model','target_time','zone']).any()
    assert p.prediction.between(0,1).all() and p.actual.between(0,1).all()
    counts=p.groupby(['split','horizon','model']).size().unstack()
    assert counts.eq(counts.iloc[:,0],axis=0).all().all()
    identical=p.groupby(['split','horizon','time','zone']).agg(n_actual=('actual','nunique'),n_target=('target_time','nunique'),n_methods=('model','nunique'))
    assert (identical.n_actual==1).all() and (identical.n_target==1).all()
    assert (identical.n_methods==p.model.nunique()).all()
    recorded=pd.read_csv(out/'fold_metrics.csv').sort_values(['split','horizon','model']).reset_index(drop=True)
    recomputed=summarize(p).sort_values(['split','horizon','model']).reset_index(drop=True)
    pd.testing.assert_frame_equal(recorded,recomputed,check_dtype=False,atol=1e-12,rtol=1e-10)
    clean,_=clean_data(ROOT/'data/raw/dataset.csv')
    for h in [30,60,120]:
        examples=make_examples(clean,h,manifest['tolerance_minutes'])
        for split,start,end in windows(clean):
            train,test=split_examples(examples,start,end)
            q=p[(p.split==split)&(p.horizon==h)&(p.model=='persistence')]
            assert len(q)==len(test)
            assert train.target_time.max()<q.time.min()
            assert (q.target_time<end).all()
            if args.refit_selected and split=='holdout':
                name=manifest['selected_by_cv'][str(h)]
                available=models()
                if name in available:
                    model=available[name]
                    model.fit(train[NUM+['zone']],train.target)
                    repeated=np.clip(model.predict(test[NUM+['zone']]),0,1)
                    saved=p[(p.split==split)&(p.horizon==h)&(p.model==name)].set_index(['time','zone']).reindex(pd.MultiIndex.from_frame(test[['time','zone']])).prediction.to_numpy()
                    np.testing.assert_allclose(repeated,saved,atol=1e-12,rtol=1e-10)
                    print(f'Refit reproduced {h}-minute {name} holdout predictions within 1e-12 absolute / 1e-10 relative tolerance.')
    print(f'Verified {len(p):,} saved predictions, all output and legacy hashes, identical model cohorts, recomputed metrics, and purged temporal boundaries.')

if __name__=='__main__':main()
