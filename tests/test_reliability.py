import unittest
import numpy as np
import pandas as pd
from research.reliability import observed_stream, partition, radius, evaluate, MODES


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.site = pd.DataFrame(dict(zone='A', time=pd.date_range('2016-12-01', periods=20, freq='30min'), rate=np.linspace(.1,.9,20)))

    def test_invalid_missing_and_age(self):
        raw = self.site.copy()
        raw.loc[1,'rate'] = np.nan
        raw.loc[2,'rate'] = 1.1
        stream = observed_stream(raw,'clean',0,0)
        self.assertTrue(stream.loc[1,'missing'] and stream.loc[2,'invalid'])
        self.assertEqual(stream.loc[2,'age'],60)
        self.assertEqual(stream.loc[2,'quality'],0)
        self.assertTrue(stream.quality.between(0,1).all())

    def test_seed_and_causality(self):
        changed = self.site.copy()
        changed.loc[10:,'rate'] = .01
        for mode in MODES:
            stream = observed_stream(self.site,mode,.4,19)
            pd.testing.assert_frame_equal(stream,observed_stream(self.site,mode,.4,19))
            pd.testing.assert_frame_equal(stream.iloc[:10],observed_stream(changed,mode,.4,19).iloc[:10])

    def test_calibration_and_score(self):
        with self.assertRaises(ValueError):
            radius([])
        self.assertEqual(radius(np.arange(10)),9)
        result=evaluate(np.array([.8]),np.array([.5]),.1)
        self.assertEqual(result['coverage'],0)
        self.assertAlmostEqual(result['interval_score'],4.2)

    def test_partition_purge(self):
        times=pd.to_datetime(['2016-11-28 23:45','2016-11-29 00:00','2016-12-06 00:00','2016-12-13 00:00'])
        parts=partition(pd.DataFrame(dict(time=times,target_time=times+pd.Timedelta(minutes=30))))
        self.assertEqual([len(p) for p in parts],[0,1,1,1])
