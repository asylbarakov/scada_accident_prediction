import unittest
import numpy as np
import pandas as pd
import hashlib
import tempfile
from pathlib import Path
from run import verify_protocol_copy
from nbm import smooth_scores,operational_metrics,select_threshold,NormalBehaviourModel


class WarningTests(unittest.TestCase):
    def test_protocol_copy_allows_formatting_but_rejects_changed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.json';copy=Path(directory)/'copy.json'
            source.write_text('{"horizon":24}');copy.write_text('{\n  "horizon": 24\n}\n')
            expected=hashlib.sha256(source.read_bytes()).hexdigest()
            verify_protocol_copy(source,copy,expected)
            copy.write_text('{"horizon":48}')
            with self.assertRaises(ValueError):verify_protocol_copy(source,copy,expected)

    def frame(self,scores,labels=None):
        n=len(scores)
        return pd.DataFrame(dict(turbine_id=['a']*n,source_event=[1]*n,timestamp=pd.date_range('2024-01-01',periods=n,freq='h',tz='UTC'),
            label=pd.array(labels if labels is not None else [0]*n,dtype='Int8'),episode=['e' if labels and v==1 else None for v in (labels or [0]*n)],history_ready=[True]*n,score=scores))

    def test_smoothing_is_causal_and_resets_on_missing_gap_and_source(self):
        frame=self.frame([1.,2.,3.,100.,5.,6.,7.,8.,9.])
        frame.loc[3,'score']=np.nan
        full=smooth_scores(frame,3)
        self.assertEqual(full.iloc[2],2.)
        self.assertTrue(full.iloc[3:6].isna().all())
        self.assertEqual(full.iloc[6],6.)
        pd.testing.assert_series_equal(smooth_scores(frame.iloc[:7],3),full.iloc[:7])
        changed=frame.copy();changed.loc[7:,'source_event']=2
        self.assertTrue(smooth_scores(changed,3).iloc[7:].isna().all())
        changed=frame.copy();changed.loc[6:,'timestamp']+=pd.Timedelta(hours=1)
        self.assertTrue(smooth_scores(changed,3).iloc[6:8].isna().all())

    def test_always_active_is_coverage_but_not_new_forecast_and_missing_event_stays(self):
        frame=self.frame([2.]*6,[None,None,1,1,1,1])
        episodes=[dict(id='e',source_event=1,turbine_id='a',failure_at='2024-01-02T02:00Z'),dict(id='missing',source_event=2,turbine_id='b',failure_at='2024-01-04T00:00Z')]
        result=operational_metrics(frame,1.,episodes)
        self.assertEqual(result['window_hit_events'],1)
        self.assertEqual(result['onset_hit_events'],0)
        self.assertEqual(result['total_events'],2)
        self.assertEqual(result['left_censored_alarm_starts'],1)

    def test_crossing_before_failure_has_actual_onset_lead(self):
        frame=self.frame([0.,0.,2.,2.],[None,None,1,1])
        episodes=[dict(id='e',source_event=1,turbine_id='a',failure_at='2024-01-02T00:00Z')]
        result=operational_metrics(frame,1.,episodes)
        self.assertEqual(result['onset_hit_events'],1)
        self.assertEqual(result['events'][0]['first_onset_lead_hours'],22.)

    def test_duration_budget_rejects_long_alarm_even_with_one_start(self):
        frame=self.frame([.8]*100+[.9,.9],[0]*100+[1,1])
        episodes=[dict(id='e',source_event=1,turbine_id='a',failure_at='2024-01-05T06:00Z')]
        threshold,result=select_threshold(frame,episodes,dict(false_alarm_budget_per_720_normal_hours=10.,max_normal_warning_fraction=.05))
        self.assertEqual(threshold,.9);self.assertEqual(result['normal_warning_fraction'],0.)
        self.assertEqual(result['onset_hit_events'],1)

    def test_disabled_mode_works_for_unbounded_residual_scores(self):
        frame=self.frame([100.,200.,300.,1.,1.],[0,0,0,1,1])
        episodes=[dict(id='e',source_event=1,turbine_id='a',failure_at='2024-01-02T00:00Z')]
        threshold,result=select_threshold(frame,episodes,dict(false_alarm_budget_per_720_normal_hours=0.,max_normal_warning_fraction=0.))
        self.assertIsNone(threshold);self.assertEqual(result['window_hit_events'],0)

    def test_fit_ignores_other_turbines_and_anomaly_values_and_labels_at_inference(self):
        rows=[]
        for asset in ['a','b','c']:
            for i in range(24):rows.append(dict(turbine_id=asset,source_event=ord(asset),timestamp=pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(hours=i),history_ready=True,label=0,x=float(i%4),temperature=float(i%4+20),other_temperature=float(i%4+25)))
        data=pd.DataFrame(rows)
        config=dict(seed=42,regime_features=['x'],required_current_inputs=['x'],targets=['temperature','other_temperature'],minimum_target_count=2,minimum_normal_rows_per_target=10,score_smoothing_hours=3,
            regressor=dict(max_iter=10,max_leaf_nodes=3,min_samples_leaf=3,early_stopping=False))
        model=NormalBehaviourModel(config).fit(data,['a','b'])
        changed=data.copy();changed.loc[changed.turbine_id.eq('c'),'temperature']=1e8
        changed=pd.concat([changed,changed.loc[changed.turbine_id.eq('a')].assign(label=1,temperature=-1e8)],ignore_index=True)
        second=NormalBehaviourModel(config).fit(changed,['a','b'])
        np.testing.assert_array_equal(model.predict(data).score,second.predict(data).score)
        np.testing.assert_array_equal(model.predict(data.drop(columns=['label'])).score,model.predict(data).score)
        full=model.predict(data)
        prefix=model.predict(data.iloc[:18])
        np.testing.assert_array_equal(full.score.iloc[:18],prefix.score)
        future=data.copy();future.loc[18:,'temperature']=1e10
        np.testing.assert_array_equal(full.score.iloc[:18],model.predict(future).score.iloc[:18])
        with self.assertRaises(ValueError):NormalBehaviourModel(config).fit(data,['a','absent'])


if __name__=='__main__':unittest.main()
