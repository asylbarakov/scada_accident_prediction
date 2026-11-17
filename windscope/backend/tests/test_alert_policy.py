import importlib.util
import unittest
from datetime import datetime, timedelta


class AlertPolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('app.alert_engine'), 'Independent alert engine is missing')
        from app.alert_engine import AlertPolicy, advance
        self.policy=AlertPolicy()
        self.advance=advance
        self.start=datetime(2023,8,4,6,10)

    def step(self,state,index,label,**kwargs):
        result={'status':'ok','predicted_class':label,'class_1_score':0.99 if label==1 else 0.01}
        return self.advance(state,self.start+timedelta(minutes=index*10),result,self.policy,**kwargs)

    def test_single_positive_is_not_an_alert_three_consecutive_confirm_once(self):
        state={};events=[]
        for i in range(6):
            state,event=self.step(state,i,1);events.append(event)
        self.assertEqual(events,['pending','pending','activated','updated','updated','updated'])
        self.assertEqual(state['first_signal'],'2023-08-04 06:10:00')
        self.assertEqual(state['activated_at'],'2023-08-04 06:30:00')

    def test_hysteresis_requires_three_negative_points_and_cooldown_uses_source_time(self):
        state={}
        for i in range(3):state,event=self.step(state,i,1)
        state,event=self.step(state,3,0);self.assertEqual(event,'updated')
        state,event=self.step(state,4,0);self.assertEqual(event,'updated')
        state,event=self.step(state,5,0);self.assertEqual(event,'resolved')
        for i in range(6,11):
            state,event=self.step(state,i,1);self.assertNotEqual(event,'activated')
        state,event=self.step(state,11,1);self.assertEqual(event,'activated')

    def test_gap_and_bad_quality_break_confirmation_without_fabricating_normal(self):
        state,event=self.step({},0,1);state,event=self.step(state,1,1)
        state,event=self.step(state,4,1);self.assertEqual(event,'pending')
        self.assertEqual(state['positive_count'],1)
        state,event=self.step(state,5,1,quality_ok=False)
        self.assertEqual(event,'quality_blocked');self.assertEqual(state['positive_count'],0)

    def test_invalid_measurement_suspends_active_alert_explicitly(self):
        state={}
        for i in range(3):state,event=self.step(state,i,1)
        state,event=self.step(state,3,1,quality_ok=False)
        self.assertEqual(event,'interrupted');self.assertEqual(state['resolution_reason'],'data_quality')

    def test_policy_configuration_is_validated_and_changes_confirmation_delay(self):
        from app.alert_engine import AlertPolicy,advance
        with self.assertRaises(ValueError):AlertPolicy(activate_points=0)
        with self.assertRaises(ValueError):AlertPolicy(cooldown_minutes=-1)
        state,event=advance({},self.start,{'status':'ok','predicted_class':1},AlertPolicy(activate_points=1))
        self.assertEqual(event,'activated')


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('app.diagnostics'), 'Explainable diagnostics are missing')
        from app.diagnostics import diagnose
        self.diagnose=diagnose
        self.t=datetime(2023,8,4,12)

    def point(self,**values):
        return {'asset_id':'0','source_time':str(self.t),'values':values}

    def test_binary_signal_does_not_invent_component_fault_or_critical(self):
        d=self.diagnose(self.point(sensor_13_avg=60),{'status':'ok','predicted_class':1,'class_1_score':1},[])
        self.assertIsNone(d['suspected_subsystem']);self.assertEqual(d['category'],'unclassified')
        self.assertEqual(d['severity'],'observation');self.assertIsNone(d['prediction_horizon'])
        self.assertTrue(d['evidence']);self.assertIn('Причина не определена',d['explanation'])

    def test_statistical_temperature_deviation_has_causal_evidence_not_failure_claim(self):
        past=[{'source_time':str(self.t-timedelta(minutes=10*(i+1))),'values':{'sensor_13_avg':20+(i%3)}} for i in range(36)]
        d=self.diagnose(self.point(sensor_13_avg=60),{'status':'ok','predicted_class':1},past)
        self.assertEqual(d['suspected_subsystem'],'generator')
        self.assertEqual(d['detection_source'],'rule')
        self.assertEqual(d['evidence'][0]['reference_count'],36)
        self.assertIn('не подтверждает',d['confidence_limitations'])
        future=[{'source_time':str(self.t+timedelta(minutes=i+1)),'values':{'sensor_13_avg':20+(i%3)}} for i in range(36)]
        self.assertIsNone(self.diagnose(self.point(sensor_13_avg=60),{'status':'ok','predicted_class':1},future)['suspected_subsystem'])

    def test_class_zero_does_not_mean_physically_healthy(self):
        d=self.diagnose(self.point(sensor_13_avg=60),{'status':'ok','predicted_class':0},[])
        self.assertEqual(d['condition'],'not_assessed');self.assertEqual(d['severity'],'none')

    def test_unavailable_inference_is_data_quality_observation(self):
        d=self.diagnose(self.point(sensor_13_avg=60),{'status':'unavailable','predicted_class':None},[])
        self.assertEqual(d['category'],'data_quality');self.assertEqual(d['condition'],'unknown')


if __name__=='__main__':unittest.main()
