import unittest
from datetime import datetime,timedelta,timezone
from sqlalchemy import create_engine,select,func
from sqlalchemy.orm import Session
from app import service
from app.models import Base,Scenario,Measurement,ReplaySession,Prediction,OperationalAlert
from app.forecast_protocol import H7_MODEL


class ContinuousReplayTests(unittest.TestCase):
    def setUp(self):
        self.engine=create_engine('sqlite://');Base.metadata.create_all(self.engine)
        self.db=Session(self.engine,expire_on_commit=False)
        self.now=datetime(2026,1,1,tzinfo=timezone.utc);self.at=datetime(2023,8,12,10)
        self.db.add(Scenario(id='22',asset_id='21',source_hash='x',rows=7,start=self.at,end=self.at+timedelta(hours=1),
            prediction_start=self.at,prediction_end=self.at+timedelta(hours=1),prediction_points=7,annotation={}))
        for i in range(7):self.db.add(Measurement(scenario_id='22',row_id=i,source_at=self.at+timedelta(minutes=i*10),
            partition='prediction',status_type_id=1,values={}))
        self.db.commit()

    def tearDown(self):self.db.close();self.engine.dispose()

    def result(self,positive=True):
        return dict(status='ok',model_id=H7_MODEL,predicted_class=int(positive),class_1_score=1.0 if positive else .01,prediction_horizon='P7D')

    def test_slow_speed_does_not_emit_backlog_in_a_burst_after_delayed_inference(self):
        s=service.create_session(self.db,'22',600,model_id=H7_MODEL)
        service.command(self.db,s.id,'start',0,'start',now=self.now)
        job=service.claim_tick(self.db,self.now)
        delayed=self.now+timedelta(seconds=30)
        service.commit_tick(self.db,job,self.result(),delayed)
        self.assertIsNone(service.claim_tick(self.db,delayed))
        self.assertIsNone(service.claim_tick(self.db,delayed+timedelta(seconds=.9)))
        self.assertIsNotNone(service.claim_tick(self.db,delayed+timedelta(seconds=1)))

    def test_continuous_replay_starts_a_clean_cycle_and_preserves_the_previous_audit(self):
        self.assertIn('continuous',ReplaySession.__table__.columns,'Continuous monitoring is missing')
        s=service.create_session(self.db,'22',600,model_id=H7_MODEL,continuous=True)
        service.command(self.db,s.id,'start',0,'start',now=self.now)
        for i in range(7):
            job=service.claim_tick(self.db,self.now+timedelta(seconds=i))
            self.assertIsNotNone(job)
            result=self.result() if i in (0,6) else {'status':'not_scheduled','model_id':H7_MODEL}
            service.commit_tick(self.db,job,result,self.now+timedelta(seconds=i))
        self.assertEqual((s.status,s.cycle,s.generation,s.position),('running',2,1,0))
        self.assertIsNone(s.virtual_time)
        self.assertIsNone(service.snapshot(self.db,s)['latest_prediction'])
        self.assertEqual(service.snapshot(self.db,s)['prediction_count'],0)
        self.assertEqual(self.db.scalar(select(func.count()).select_from(Prediction)),2)
        old=self.db.scalar(select(OperationalAlert))
        self.assertEqual(old.status,'resolved');self.assertEqual(old.resolution_reason,'dataset_replay')
        self.assertIsNone(service.claim_tick(self.db,self.now+timedelta(seconds=6.9)))
        next_job=service.claim_tick(self.db,self.now+timedelta(seconds=7))
        self.assertEqual((next_job['row_id'],next_job['generation']),(0,1))

    def test_measurement_counter_and_prediction_counter_have_different_cadence(self):
        s=service.create_session(self.db,'22',600,model_id=H7_MODEL)
        service.command(self.db,s.id,'start',0,'start',now=self.now)
        for i in range(3):
            job=service.claim_tick(self.db,self.now+timedelta(seconds=i))
            service.commit_tick(self.db,job,self.result() if i==0 else {'status':'not_scheduled'},self.now+timedelta(seconds=i))
        snap=service.snapshot(self.db,s)
        self.assertEqual(snap['position'],3)
        self.assertEqual(snap.get('prediction_count'),1)

    def test_monitoring_uses_the_whole_source_and_h7_instead_of_the_latest_short_test(self):
        self.assertTrue(callable(getattr(service,'start_monitoring',None)),'Monitoring start is missing')
        old=service.create_session(self.db,'22',36000,start=self.at,end=self.at+timedelta(minutes=10))
        new=service.start_monitoring(self.db,'22',300)
        self.assertEqual((new.model_id,new.total,new.speed,new.continuous),(H7_MODEL,7,300,True))
        self.assertEqual(new.status,'running')
        self.assertEqual(new.start,self.at);self.assertEqual(new.end,self.at+timedelta(hours=1))
        reused=service.start_monitoring(self.db,'22',60)
        self.assertEqual(reused.id,new.id)
        self.assertEqual(reused.speed,60)
        self.assertNotEqual(new.id,old.id)

    def test_newer_short_test_cannot_hide_monitoring_turbine_or_its_warning(self):
        from app.main import turbines
        self.assertIn('continuous',ReplaySession.__table__.columns)
        full=service.create_session(self.db,'22',600,model_id=H7_MODEL,continuous=True)
        service.command(self.db,full.id,'start',0,'full-start',now=self.now)
        service.commit_tick(self.db,service.claim_tick(self.db,self.now),self.result(),self.now)
        short=service.create_session(self.db,'22',600,start=self.at,end=self.at+timedelta(minutes=10),model_id=H7_MODEL,continuous=True)
        self.assertEqual(turbines(db=self.db)[0]['session']['id'],full.id)
        self.assertEqual(service.list_alerts(self.db)[0]['session_id'],full.id)
        self.assertTrue(service.snapshot(self.db,full).get('monitoring'))
        self.assertFalse(service.snapshot(self.db,short).get('monitoring'))

    def test_continuous_mode_skips_empty_waits_but_keeps_the_real_gap_and_no_fake_samples(self):
        self.assertIn('continuous',ReplaySession.__table__.columns)
        for i in range(1,7):self.db.get(Measurement,('22',i)).source_at+=timedelta(hours=4)
        scenario=self.db.get(Scenario,'22');scenario.end+=timedelta(hours=4);scenario.prediction_end=scenario.end
        self.db.commit()
        s=service.create_session(self.db,'22',300,model_id=H7_MODEL,continuous=True)
        service.command(self.db,s.id,'start',0,'gap-start',now=self.now)
        service.commit_tick(self.db,service.claim_tick(self.db,self.now),self.result(),self.now)
        next_job=service.claim_tick(self.db,self.now+timedelta(seconds=2))
        self.assertIsNotNone(next_job)
        self.assertEqual(next_job['source_time'],'2023-08-12 14:10:00')
        service.commit_tick(self.db,next_job,{'status':'not_scheduled'},self.now+timedelta(seconds=2))
        snap=service.snapshot(self.db,s)
        self.assertEqual((snap['position'],snap['model_state']),(2,'unknown'))
        history=service.history(self.db,s.id)
        self.assertEqual(len(history),2)
        self.assertEqual(history[-1]['gap_before_seconds'],15000)
