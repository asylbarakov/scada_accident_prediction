import unittest
from datetime import datetime,timedelta,timezone
from sqlalchemy import create_engine,select
from sqlalchemy.orm import Session
from app.models import Base,Scenario,Measurement,OperationalAlert
from app import service
from app.personal_policy import advance_watch

PERSONAL='care-personal-normal-h24-v1'


class WatchPolicyTests(unittest.TestCase):
    def test_signal_is_retained_through_negative_and_unavailable_then_expires(self):
        at=datetime(2023,10,19,8)
        state,transition=advance_watch({},at,{'status':'ok','predicted_class':1})
        self.assertEqual(transition,'activated')
        for hours,result in [(1,{'status':'ok','predicted_class':0}),(2,{'status':'unavailable'}),(3,{'status':'data_gap'})]:
            state,transition=advance_watch(state,at+timedelta(hours=hours),result)
            self.assertTrue(state['active']);self.assertNotIn(transition,['resolved','interrupted'])
        state,transition=advance_watch(state,at+timedelta(hours=24),{'status':'ok','predicted_class':0})
        self.assertEqual(transition,'resolved');self.assertEqual(state['resolution_reason'],'watch_window_elapsed')


class PersonalReplayTests(unittest.TestCase):
    def setUp(self):
        self.engine=create_engine('sqlite://');Base.metadata.create_all(self.engine);self.db=Session(self.engine,expire_on_commit=False)
        self.at=datetime(2023,10,19,8);self.now=datetime(2026,1,1,tzinfo=timezone.utc)
        self.db.add(Scenario(id='26',asset_id='0',source_hash='x',rows=70,start=self.at-timedelta(hours=8),end=self.at+timedelta(hours=3),prediction_start=self.at,prediction_end=self.at+timedelta(hours=3),prediction_points=19,annotation={'start':str(self.at-timedelta(days=7)),'end':str(self.at+timedelta(hours=2)),'label':'anomaly','description':'SECRET_FAULT'}))
        for i in range(67):self.db.add(Measurement(scenario_id='26',row_id=i,source_at=self.at+timedelta(minutes=(i-48)*10),partition='train' if i<48 else 'prediction',status_type_id=4,values={'sensor_0_avg':20.,'sensor_41_avg':45.,'DO_NOT_SEND':999}))
        self.db.commit()

    def tearDown(self):self.db.close();self.engine.dispose()

    def test_personal_payload_is_only_eight_hours_past_and_alert_survives_next_negative(self):
        s=service.create_session(self.db,'26',36000,model_id=PERSONAL)
        service.command(self.db,s.id,'start',0,'personal-start',now=self.now)
        job=service.claim_tick(self.db,self.now);payload=service.forecast_payload(self.db,job)
        self.assertEqual(payload['source_id'],'26');self.assertEqual(len(payload['history']),49)
        self.assertNotIn('SECRET_FAULT',str(payload));self.assertNotIn('status_type_id',str(payload));self.assertNotIn('DO_NOT_SEND',str(payload))
        result={'status':'ok','predicted_class':1,'class_1_score':2.6,'deviation_score':2.6,'warning_index':1.3,'threshold':2.,'model_id':PERSONAL,'prediction_horizon':'P1D','prediction_horizon_hours':24,'evidence':[]}
        service.commit_tick(self.db,job,result,self.now)
        for i in range(1,7):
            job=service.claim_tick(self.db,self.now+timedelta(seconds=i))
            service.commit_tick(self.db,job,dict(result,predicted_class=0) if i==6 else {'status':'not_scheduled','model_id':PERSONAL},self.now+timedelta(seconds=i))
        alert=self.db.scalar(select(OperationalAlert))
        self.assertIsNone(alert.end);self.assertEqual(alert.detail['prediction_horizon'],'P1D')

    def test_future_failure_is_hidden_until_replay_reaches_it_or_user_reveals(self):
        from app.demo import demo_check
        s=service.create_session(self.db,'26',300,model_id=PERSONAL)
        hidden=demo_check(self.db,s.id,False)
        self.assertIsNone(hidden['failure_at']);self.assertEqual(hidden['state'],'not_reached')
        revealed=demo_check(self.db,s.id,True)
        self.assertEqual(revealed['failure_at'],str(self.at+timedelta(hours=2)))
        self.assertTrue(revealed['future_revealed'])


if __name__=='__main__':unittest.main()
