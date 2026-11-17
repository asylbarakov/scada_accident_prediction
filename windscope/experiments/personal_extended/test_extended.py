import importlib.util,unittest
import numpy as np
import pandas as pd

class ExtendedTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('extended'),'Extended sensor experiment is not implemented')
        import extended
        return extended
    def test_temperature_delay_never_uses_future_and_keeps_observed_extremes(self):
        d=self.module();raw=pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01',periods=20,freq='10min',tz='UTC'),temperature=np.arange(20,dtype=float)))
        raw.loc[5,'temperature']=205
        result=d.delayed_targets(raw,{'extra':'temperature'})
        self.assertTrue(np.isnan(result.extra.iloc[0]));self.assertEqual(result.extra.iloc[6],205)
        changed=raw.copy();changed.loc[12:,'temperature']=1e9
        pd.testing.assert_frame_equal(result.iloc[:12],d.delayed_targets(changed,{'extra':'temperature'}).iloc[:12])
    def test_gap_cannot_reuse_stale_temperature(self):
        d=self.module();raw=pd.DataFrame(dict(timestamp=pd.to_datetime(['2024-01-01 00:00','2024-01-01 00:10','2024-01-01 01:00'],utc=True),temperature=[20,21,22]))
        result=d.delayed_targets(raw,{'extra':'temperature'})
        self.assertEqual(result.extra.iloc[1],20);self.assertTrue(np.isnan(result.extra.iloc[2]))
    def test_invalid_values_become_missing_without_imputation(self):
        d=self.module();raw=pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01',periods=4,freq='10min',tz='UTC'),temperature=[20,np.inf,np.nan,21]))
        result=d.delayed_targets(raw,{'extra':'temperature'})
        self.assertTrue(result.extra.iloc[2:].isna().all())

if __name__=='__main__':unittest.main()
