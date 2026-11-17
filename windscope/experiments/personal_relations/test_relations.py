import importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd

class RelationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('relations'),'Physical relation experiment is not implemented')
        import relations
        return relations
    def raw(self,n=80):
        return pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01',periods=n,freq='10min',tz='UTC'),turbine_id='a',sensor_13_avg=60.,sensor_14_avg=50.,sensor_38_avg=50.,sensor_39_avg=55.,sensor_40_avg=53.,power_30_avg=np.arange(n)/100))
    def test_relations_and_power_windows_use_only_delayed_past(self):
        d=self.module();raw=self.raw();result,ready=d.physical_relations(raw)
        self.assertEqual(result.generator_bearing_delta__now.iloc[42],10)
        self.assertEqual(result.transformer_phase_spread__now.iloc[42],5)
        self.assertAlmostEqual(result.grid_active_power__now.iloc[42],.41,places=6)
        self.assertAlmostEqual(result.grid_active_power__1h_mean.iloc[42],.385,places=6)
        self.assertAlmostEqual(result.grid_active_power__6h_mean.iloc[42],.235,places=6)
        self.assertTrue(ready.iloc[42])
        changed=raw.copy();changed.loc[45:,'sensor_14_avg']=1e9;changed.loc[45:,'power_30_avg']=1e9;changed['failure']=1
        pd.testing.assert_frame_equal(d.physical_relations(changed)[0].iloc[:45],result.iloc[:45])
        pd.testing.assert_frame_equal(d.physical_relations(raw.iloc[:45])[0],result.iloc[:45])
    def test_gap_resets_power_context_and_incomplete_phases_do_not_hide_imbalance(self):
        d=self.module();raw=self.raw();raw.loc[40,'sensor_39_avg']=np.nan
        result,_=d.physical_relations(raw)
        self.assertTrue(np.isnan(result.transformer_phase_spread__now.iloc[41]))
        raw=self.raw().drop(index=40).reset_index(drop=True);result,ready=d.physical_relations(raw)
        self.assertTrue(np.isnan(result.grid_active_power__now.iloc[40]));self.assertFalse(ready.iloc[40:60].any())
    def frame(self):
        n=120
        return pd.DataFrame(dict(turbine_id='a',source_event=1,timestamp=pd.date_range('2024-01-01',periods=n,freq='h',tz='UTC'),history_ready=True,normal=[True]*100+[False]*20,role=['train']*60+['calibration']*40+['test']*20,x=np.tile(np.arange(6),20).astype(float),generator_bearing_delta__now=10.,transformer_phase_spread__now=5.,label=[0]*100+[1]*20))
    def config(self):
        return dict(regime_features=['x'],required_current_inputs=['x'],targets=['generator_bearing_delta__now','transformer_phase_spread__now'],two_sided_targets=['generator_bearing_delta__now'],minimum_target_count=2,minimum_training_normal_hours=10,minimum_calibration_normal_hours=5,score_smoothing_hours=1,regressor=dict(n_estimators=5,max_depth=2,n_jobs=1,random_state=42))
    def test_negative_relation_deviation_warns_and_future_labels_cannot_affect_fitting(self):
        d=self.module();frame=self.frame();model=d.fit_model(frame,self.config());changed=frame.copy()
        changed.loc[100:,'generator_bearing_delta__now']=-1000;changed.loc[100:,'label']=0
        other=d.fit_model(changed,self.config())
        first=d.score_model(model,frame);np.testing.assert_array_equal(first.score,d.score_model(other,frame).score)
        self.assertGreater(d.score_model(model,changed).score.iloc[-1],100)
        with tempfile.TemporaryDirectory() as path:
            model.save(path);loaded=d.PersonalModel.load(path)
            np.testing.assert_array_equal(d.score_model(loaded,frame).score,first.score)

if __name__=='__main__':unittest.main()
