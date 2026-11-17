import unittest
from datetime import datetime,timedelta,timezone
from sqlalchemy import create_engine,select,func
from sqlalchemy.orm import Session
from app.models import Base,Scenario,Measurement
from app import service


class PersistentAlertsTests(unittest.TestCase):
    def setUp(self):
        self.engine=create_engine('sqlite://');Base.metadata.create_all(self.engine)
        self.db=Session(self.engine,expire_on_commit=False)
        self.source=datetime(2023,8,4,6,10);self.now=datetime(2026,1,1,tzinfo=timezone.utc)
        self.db.add(Scenario(id='0',asset_id='0',source_hash='x',rows=12,start=self.source,end=self.source+timedelta(minutes=110),
            prediction_start=self.source,prediction_end=self.source+timedelta(minutes=110),prediction_points=12,
            annotation={'start':'2023-08-06 06:10:00','end':'2023-08-20 06:10:00','label':'anomaly','description':'DO NOT USE'}))
        from app.diagnostics import REQUIRED
        for i in range(12):self.db.add(Measurement(scenario_id='0',row_id=i,source_at=self.source+timedelta(minutes=i*10),partition='prediction',values={k:20 for k in REQUIRED},status_type_id=1))
        self.db.commit();self.session=service.create_session(self.db,'0',600)
        service.command(self.db,self.session.id,'start',0,'start',now=self.now)

    def tearDown(self):self.db.close();self.engine.dispose()

    def tick(self,i,label):
        job=service.claim_tick(self.db,self.now+timedelta(seconds=i))
        self.assertIsNotNone(job)
        service.commit_tick(self.db,job,{'status':'ok','predicted_class':label,'class_1_score':.99 if label else .01,'model_id':'test'},self.now+timedelta(seconds=i))

    def test_alert_is_grouped_acknowledged_and_resolved_without_changing_raw_predictions(self):
        for i in range(3):self.tick(i,1)
        self.assertTrue(hasattr(service,'list_alerts'),'Persistent alert center is missing')
        alerts=service.list_alerts(self.db,session_id=self.session.id)
        self.assertEqual(len(alerts),1);a=alerts[0]
        self.assertEqual(a['start'],'2023-08-04 06:30:00')
        self.assertEqual(a['first_signal'],'2023-08-04 06:10:00')
        self.assertEqual(a['status'],'active');self.assertIsNone(a['subsystem'])
        self.assertNotIn('DO NOT USE',str(a))
        service.acknowledge(self.db,a['id'])
        self.db.expire_all()
        self.assertEqual(service.list_alerts(self.db,session_id=self.session.id)[0]['status'],'acknowledged')
        for i in range(3,6):self.tick(i,0)
        a=service.list_alerts(self.db,session_id=self.session.id)[0]
        self.assertEqual(a['status'],'resolved');self.assertEqual(a['end'],'2023-08-04 07:00:00')
        from app.models import Prediction,ReplayEvent
        self.assertEqual(self.db.scalar(select(func.count()).select_from(Prediction)),6)
        self.assertEqual(len(self.db.scalars(select(ReplayEvent).where(ReplayEvent.type=='alert_activated')).all()),1)
        self.assertTrue(a['acknowledged_at'])

    def test_reset_hides_old_alerts_and_diagnostics_but_preserves_audit(self):
        for i in range(3):self.tick(i,1)
        self.assertTrue(hasattr(service,'list_alerts'),'Persistent alert center is missing')
        self.assertEqual(len(service.list_alerts(self.db,session_id=self.session.id)),1)
        service.command(self.db,self.session.id,'reset',self.session.revision,'reset',now=self.now)
        self.assertEqual(service.list_alerts(self.db,session_id=self.session.id),[])
        self.assertIsNone(service.snapshot(self.db,self.session)['diagnostic'])
        audit=service.list_alerts(self.db,archive=True)
        self.assertEqual(len(audit),1)
        self.assertEqual(audit[0]['status'],'resolved')
        self.assertEqual(audit[0]['resolution_reason'],'replay_reset')

    def test_alert_evidence_stays_at_activation_timestamp_while_live_diagnostics_advance(self):
        for i in range(4):self.tick(i,1)
        a=service.list_alerts(self.db,session_id=self.session.id)[0]
        self.assertEqual(a['diagnostic']['timestamp'],'2023-08-04 06:30:00')
        self.assertEqual(service.snapshot(self.db,self.session)['diagnostic']['timestamp'],'2023-08-04 06:40:00')

    def test_api_processing_timestamps_are_canonical_utc_despite_database_session_offset(self):
        for i in range(3):self.tick(i,1)
        from app.models import OperationalAlert,ReplayEvent
        from app.main import event_dict
        a=self.db.scalar(select(OperationalAlert))
        a.acknowledged_at=datetime(2026,1,1,5,tzinfo=timezone(timedelta(hours=5)))
        self.assertEqual(service.alert_dict(a)['acknowledged_at'],'2026-01-01T00:00:00+00:00')
        e=self.db.scalar(select(ReplayEvent).limit(1));e.emitted_at=a.acknowledged_at
        self.assertEqual(event_dict(e)['emitted_at'],'2026-01-01T00:00:00+00:00')


if __name__=='__main__':unittest.main()
