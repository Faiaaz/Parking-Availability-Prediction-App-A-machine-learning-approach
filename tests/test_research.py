"""Protocol tests using small deterministic fixtures, not reported experiment results."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from research.run_experiments import clean_data, make_examples, split_examples, nearest, uncertainty


class ProtocolTests(unittest.TestCase):
    def fixture(self):
        t=pd.date_range('2020-01-01',periods=12*48,freq='30min')
        return pd.DataFrame({'zone':'A','time':t,'capacity':100,'occupancy':np.arange(len(t))%100,'rate':(np.arange(len(t))%100)/100})

    def test_elapsed_horizon_and_future_history(self):
        raw=self.fixture()
        for h in [30,60,120]:
            e=make_examples(raw,h)
            self.assertTrue((e.target_time-e.time==pd.Timedelta(minutes=h)).all())
            self.assertTrue((e.daily_source_time.dropna()<=e.loc[e.daily_source_time.notna(),'time']).all())
            self.assertTrue((e.weekly_source_time.dropna()<=e.loc[e.weekly_source_time.notna(),'time']).all())
            cut=pd.Timestamp('2020-01-10')
            changed=raw.copy();changed.loc[changed.time>cut,'rate']=.123
            a=e.loc[e.time<=cut].set_index('time')
            b=make_examples(changed,h).loc[lambda x:x.time<=cut].set_index('time')
            pd.testing.assert_frame_equal(a[['current','lag1','lag2','rolling3','daily','weekly']],b[['current','lag1','lag2','rolling3','daily','weekly']])

    def test_purge_and_group_boundaries(self):
        e=make_examples(self.fixture(),120)
        start=pd.Timestamp('2020-01-10');end=pd.Timestamp('2020-01-12')
        train,test=split_examples(e,start,end)
        self.assertLess(train.target_time.max(),start)
        self.assertGreaterEqual(test.time.min(),start)
        self.assertLess(test.target_time.max(),end)
        self.assertTrue((train.time<start-pd.Timedelta(minutes=120)).all())

    def test_duplicate_policy_and_invalid_values(self):
        x=pd.DataFrame({'SystemCodeNumber':['A','A','A'],'LastUpdated':['2020-01-01','2020-01-01','2020-01-02'],'Capacity':[100]*3,'Occupancy':[20,20,101]})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.csv';x.to_csv(p,index=False)
            clean,audit=clean_data(p)
            self.assertEqual(len(clean),1);self.assertEqual(audit['exact_duplicate_excess'],1)
            self.assertEqual(audit['invalid_occupancy_rows'],1)
            x.loc[1,'Occupancy']=21;x.to_csv(p,index=False)
            with self.assertRaises(ValueError):clean_data(p)

    def test_paired_bootstrap_preserves_index_alignment(self):
        rows=[]
        for t in pd.date_range('2020-01-01',periods=7,freq='D'):
            for zone in ['A','B']:
                for model,pred in [('persistence',.5),('perfect',.2)]:
                    rows.append(dict(time=t,zone=zone,model=model,prediction=pred,actual=.2,split='holdout',horizon=30))
        ci=uncertainty(pd.DataFrame(rows),repetitions=50).set_index('model')
        self.assertTrue(np.isfinite(ci[['ci_low','ci_high']]).all().all())
        self.assertAlmostEqual(ci.loc['perfect','mae_difference_vs_persistence'],-.3)
        self.assertAlmostEqual(ci.loc['perfect','ci_low'],-.3)
        self.assertAlmostEqual(ci.loc['persistence','ci_high'],0)

    def test_no_match_across_long_gap(self):
        times=np.array([0,30,120],dtype=np.int64)
        values=np.array([.1,.2,.3])
        result,_,valid=nearest(times,values,np.array([15,60]),5)
        self.assertFalse(valid.any());self.assertTrue(np.isnan(result).all())
        result,matched,valid=nearest(times,values,np.array([15]),15)
        self.assertEqual(matched[0],0)  # deterministic earlier tie

if __name__=='__main__':unittest.main()
