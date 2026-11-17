import importlib
import importlib.util
import unittest
import numpy as np
import pandas as pd


class DynamicsTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('dynamics'), 'Causal residual dynamics have not been implemented')
        return importlib.import_module('dynamics')

    def frame(self,n=24):
        d=self.module()
        frame=pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01',periods=n,freq='h',tz='UTC'),source_event=1,turbine_id='a',history_ready=True))
        for channel in d.CHANNELS:frame[channel+'__normalized_residual']=np.arange(n,dtype=float)
        return frame

    def test_known_slope_persistence_and_warmup(self):
        d=self.module();result=d.dynamic_features(self.frame())
        self.assertFalse(result.history_ready.iloc[:5].any())
        self.assertTrue(result.history_ready.iloc[5:].all())
        self.assertAlmostEqual(result.iloc[5][d.CHANNELS[0]+'__slope6'],1.)
        self.assertAlmostEqual(result.iloc[5][d.CHANNELS[0]+'__mean6'],2.5)
        self.assertAlmostEqual(result.iloc[5][d.CHANNELS[0]+'__persistence6'],5/6)

    def test_future_values_and_annotations_cannot_change_past_features(self):
        d=self.module();frame=self.frame();first=d.dynamic_features(frame)
        changed=frame.copy()
        for c in d.CHANNELS:changed.loc[12:,c+'__normalized_residual']=1e9
        changed['label']=1;changed['status_type_id']=4;changed['failure_at']='2024-01-02'
        pd.testing.assert_frame_equal(d.dynamic_features(changed).iloc[:12],first.iloc[:12])
        pd.testing.assert_frame_equal(d.dynamic_features(frame.iloc[:12]),first.iloc[:12])
        self.assertFalse(any(c in first for c in ['label','failure_at','status_type_id']))

    def test_gaps_source_changes_and_unavailable_rows_reset_context(self):
        d=self.module();frame=self.frame();frame.loc[10:,'timestamp']+=pd.Timedelta(hours=1)
        frame.loc[18:,'source_event']=2
        result=d.dynamic_features(frame)
        self.assertFalse(result.history_ready.iloc[10:15].any())
        self.assertFalse(result.history_ready.iloc[18:23].any())
        frame=self.frame()
        for c in d.CHANNELS:frame.loc[10,c+'__normalized_residual']=np.nan
        result=d.dynamic_features(frame)
        self.assertFalse(result.history_ready.iloc[10:16].any())
        self.assertTrue(result.history_ready.iloc[16])

    def test_head_fitting_ignores_heldout_values_and_labels(self):
        d=self.module();frame=self.frame(80);features=['feature']
        frame['feature']=np.tile(np.arange(8),10).astype(float)
        frame['label']=(frame.feature>3).astype(int)
        frame.loc[40:,'turbine_id']='b'
        config=dict(C=.1,class_weight='balanced',max_iter=1000,random_state=42,solver='lbfgs')
        model,keys=d.fit_head(frame,features,['a'],config)
        changed=frame.copy();changed.loc[40:,'feature']=1e9;changed.loc[40:,'label']=1-changed.loc[40:,'label']
        other,other_keys=d.fit_head(changed,features,['a'],config)
        np.testing.assert_array_equal(model.predict_proba(frame[features]),other.predict_proba(frame[features]))
        pd.testing.assert_frame_equal(keys,other_keys)
        self.assertEqual(set(keys.turbine_id),{'a'})


if __name__=='__main__':unittest.main()
