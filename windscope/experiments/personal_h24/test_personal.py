import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from personal import historical_roles,normal_threshold,PersonalModel


class PersonalTests(unittest.TestCase):
    def config(self):
        return dict(temporal_training_fraction=.6,normal_past_hours=1,normal_followup_hours=2,train_embargo_hours=2,calibration_context_embargo_hours=1,
            minimum_training_normal_hours=10,minimum_calibration_normal_hours=5,
            regime_features=['x'],required_current_inputs=['x'],targets=['temperature','oil'],minimum_target_count=2,score_smoothing_hours=3,
            regressor=dict(n_estimators=5,max_depth=2,learning_rate=.1,tree_method='hist',n_jobs=1,random_state=42),
            false_alarm_budget_per_720_normal_hours=20.,max_normal_warning_fraction=.05)

    def raw(self):
        t=pd.date_range('2024-01-01',periods=1441,freq='10min',tz='UTC')
        return pd.DataFrame(dict(timestamp=t,partition=['train']*1200+['prediction']*241,status_type_id=[0]*len(t),observed=[True]*len(t)))

    def test_normal_future_and_temporal_roles_never_reach_prediction(self):
        raw=self.raw();config=self.config()
        result,info=historical_roles(raw,config)
        train=raw.loc[result.role.eq('train')&result.normal]
        calibration=raw.loc[result.role.eq('calibration')&result.normal]
        boundary=pd.Timestamp(info['boundary']);start=raw.loc[raw.partition.eq('prediction'),'timestamp'].min()
        self.assertTrue((train.timestamp+pd.Timedelta(hours=2)).le(boundary).all())
        self.assertTrue(calibration.timestamp.ge(boundary+pd.Timedelta(hours=1)).all())
        self.assertTrue((calibration.timestamp+pd.Timedelta(hours=2)).lt(start).all())
        self.assertFalse(result.normal.loc[raw.partition.eq('prediction')].any())
        changed=raw.copy();changed.loc[1200:,'status_type_id']=4
        pd.testing.assert_frame_equal(historical_roles(changed,config)[0],result)

    def test_status_break_and_timestamp_gap_censor_certified_normal(self):
        raw=self.raw();raw.loc[100,'status_type_id']=4
        result,_=historical_roles(raw,self.config())
        self.assertFalse(result.normal.iloc[88:107].any())
        raw=self.raw().drop(index=100).reset_index(drop=True)
        result,_=historical_roles(raw,self.config())
        self.assertFalse(result.normal.iloc[88:106].any())

    def test_threshold_uses_calibration_normal_only_and_has_zero_alarm_fallback(self):
        points=pd.DataFrame(dict(source_event=[1]*101,timestamp=pd.date_range('2024-01-01',periods=101,freq='h',tz='UTC'),score=[1.]*100+[1e6],normal=[True]*100+[False]))
        threshold,metrics=normal_threshold(points,dict(false_alarm_budget_per_720_normal_hours=0.,max_normal_warning_fraction=0.))
        self.assertGreater(threshold,1.);self.assertLess(threshold,1e6)
        self.assertEqual(metrics['normal_warning_fraction'],0.)

    def features(self):
        n=120
        return pd.DataFrame(dict(turbine_id=['a']*n,source_event=[1]*n,timestamp=pd.date_range('2024-01-01',periods=n,freq='h',tz='UTC'),
            history_ready=[True]*n,normal=[True]*100+[False]*20,role=['train']*60+['calibration']*40+['test']*20,
            x=np.tile(np.arange(6),20).astype(float),temperature=np.tile(np.arange(6),20)+20.,oil=np.tile(np.arange(6),20)+30.,label=[0]*100+[1]*20))

    def test_fit_and_calibration_ignore_all_future_rows_and_model_is_prefix_invariant(self):
        frame=self.features();model=PersonalModel(self.config()).fit(frame)
        changed=frame.copy();changed.loc[100:,'temperature']=1e9;changed.loc[100:,'label']=0
        other=PersonalModel(self.config()).fit(changed)
        first=model.predict(frame);second=other.predict(frame)
        np.testing.assert_array_equal(first.score,second.score)
        np.testing.assert_array_equal(model.predict(frame.drop(columns=['label'])).score,first.score)
        np.testing.assert_array_equal(model.predict(frame.iloc[:110]).score,first.score.iloc[:110])
        self.assertTrue(model.fit_keys.timestamp.max()<model.calibration_keys.timestamp.min())

    def test_native_export_round_trip_and_reject_mixed_sources(self):
        frame=self.features();model=PersonalModel(self.config()).fit(frame)
        with tempfile.TemporaryDirectory() as path:
            model.save(Path(path));loaded=PersonalModel.load(Path(path))
            np.testing.assert_array_equal(model.predict(frame).score,loaded.predict(frame).score)
        mixed=frame.copy();mixed.loc[110:,'source_event']=2
        with self.assertRaises(ValueError):PersonalModel(self.config()).fit(mixed)


if __name__=='__main__':unittest.main()
