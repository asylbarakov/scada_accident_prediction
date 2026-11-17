import unittest
import numpy as np
import pandas as pd
from core import label_event, reconcile, fold_assets, evaluate, choose_threshold


class ProtocolTests(unittest.TestCase):
    def test_future_boundaries_and_no_post_failure_label(self):
        fault = pd.Timestamp('2024-01-03T00:00Z')
        t = pd.Series([fault-pd.Timedelta(hours=24,minutes=10), fault-pd.Timedelta(hours=24), fault-pd.Timedelta(minutes=10),fault])
        labels = label_event(t, pd.Series(True,index=t.index),'anomaly',t.min(),fault,'e',True,[],[])
        self.assertTrue(pd.isna(labels.label.iloc[0]))
        self.assertEqual(labels.label.iloc[1:3].tolist(),[1,1])
        self.assertTrue(pd.isna(labels.label.iloc[3]))

    def test_normal_requires_full_unbroken_future_and_no_competing_fault(self):
        t = pd.Series(pd.date_range('2024-01-01',periods=433,freq='10min',tz='UTC'))
        observed=pd.Series(True,index=t.index);observed.iloc[12]=False
        out=label_event(t,observed,'normal',t.min(),t.max(),None,False,[],[])
        self.assertTrue(pd.isna(out.label.iloc[0]))
        self.assertEqual(out.label.iloc[13],0)
        self.assertTrue(pd.isna(out.label.iloc[-2]))
        conflict=label_event(t,pd.Series(True,index=t.index),'normal',t.min(),t.max(),None,False,[t[60]],[])
        self.assertTrue(pd.isna(conflict.label.iloc[0]))

    def test_ambiguous_and_competing_faults_censored(self):
        t=pd.Series(pd.date_range('2024-01-02',periods=6,freq='h',tz='UTC'));fault=t[0]+pd.Timedelta(hours=24)
        for canonical,others in [(False,[]),(True,[fault-pd.Timedelta(hours=2)])]:
            out=label_event(t,pd.Series(True,index=t.index),'anomaly',t.min(),fault,'e',canonical,[],others)
            self.assertTrue(out.label.isna().all())

    def test_duplicate_preference_and_conflict_censoring(self):
        frame=pd.DataFrame(dict(turbine_id=['a']*4,timestamp=pd.to_datetime(['2024-01-01','2024-01-01','2024-01-02','2024-01-02'],utc=True),source_event=[1,2,1,2],avg_fingerprint=['x','x','y','y'],label=pd.array([1,1,1,0],dtype='Int8'),reason=['positive']*4,episode=['e','e','e',None],failure_at=[pd.NaT]*4,history_ready=[False,True,True,True]))
        result=reconcile(frame)
        self.assertTrue(pd.isna(result.label.iloc[0]));self.assertEqual(result.label.iloc[1],1)
        self.assertTrue(result.label.iloc[2:].isna().all())

    def test_whole_turbine_folds(self):
        assets=['0','10','11','13','21']
        for i,test in enumerate(assets):
            split=fold_assets(assets,test)
            self.assertEqual(split['test'],[test]);self.assertEqual(split['validation'],[assets[(i+1)%5]])
            self.assertEqual(len(set(sum(split.values(),[]))),5)
            self.assertEqual(len(split['train']),3)

    def test_unavailable_fault_counts_as_missed(self):
        points=pd.DataFrame(dict(source_event=[1,1,2],timestamp=pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC'),label=pd.array([1,1,0],dtype='Int8'),episode=['e','e',None],score=[.8,.9,.7],history_ready=[True,True,True]))
        episodes=[dict(id='e',failure_at='2024-01-02T00:00Z'),dict(id='unavailable',failure_at='2024-01-03T00:00Z')]
        metrics=evaluate(points,.75,episodes)
        self.assertEqual(metrics['detected_events'],1);self.assertEqual(metrics['total_events'],2)
        self.assertEqual(metrics['events'][1]['available_positive_hours'],0)
        self.assertEqual(metrics['event_recall'],.5)

    def test_threshold_respects_validation_budget_and_ignores_unlabelled_scores(self):
        points=pd.DataFrame(dict(source_event=[1,1,2,2,3],timestamp=pd.to_datetime(['2024-01-01','2024-01-01 01:00','2024-01-03','2024-01-03 01:00','2024-01-04'],utc=True,format='mixed'),label=pd.array([1,1,0,0,None],dtype='Int8'),episode=['e','e',None,None,None],score=[.8,.9,.85,.1,1.],history_ready=[True]*5))
        episodes=[dict(id='e',failure_at='2024-01-02T00:00Z')]
        threshold,metrics=choose_threshold(points,episodes,2.)
        self.assertGreater(threshold,.85);self.assertLessEqual(threshold,.9)
        self.assertEqual(metrics['false_warning_episodes'],0)
        self.assertEqual(metrics['event_recall'],1.)


if __name__=='__main__':unittest.main()
