import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from research.multidata.adapters import harvard, event_state
from research.multidata.pilot import partitions


class AdapterTests(unittest.TestCase):
    def test_capacity_and_simulation_separation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample'
            path.write_text('timestamp;segmentid;capacity;occupied;observed1;diff1\n2020-01-01;A;0;0;1;99\n2020-01-02;A;10;3;0;99\npartial')
            frame,audit=harvard(path)
            self.assertEqual(len(frame),1)
            self.assertAlmostEqual(frame.rate.iloc[0],.3)
            self.assertNotIn('diff1',frame)
            self.assertEqual(audit['invalid_rows'],1)
            self.assertTrue(frame.available_at.eq(frame.observed_at).all())

    def test_unknown_gap_and_conflict(self):
        start=pd.Timestamp('2020-01-01')
        intervals=pd.DataFrame(dict(site_id=['A','A'],observed_at=[start,start],end=[start+pd.Timedelta(hours=1)]*2,occupied_count=[0,1]))
        self.assertTrue(np.isnan(event_state(intervals,'A',start)))
        self.assertTrue(np.isnan(event_state(intervals,'B',start)))
        self.assertTrue(np.isnan(event_state(intervals,'A',start+pd.Timedelta(hours=2))))
        self.assertEqual(event_state(intervals.iloc[:1],'A',start),0)

    def test_partition_label_boundary(self):
        origin=pd.Timestamp('2020-01-01 23:30')
        frame=pd.DataFrame(dict(time=[origin],target_time=[origin+pd.Timedelta(hours=1)]))
        parts=partitions(frame,['2020-01-02','2020-01-03','2020-01-04','2020-01-05'])
        self.assertTrue(all(p.empty for p in parts))
