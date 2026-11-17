import unittest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from app.models import Base, Scenario, Measurement, Prediction, ReplaySession
from app import service

H7 = 'care-h7-logistic-20261116'


class ForecastReplayTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://')
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine, expire_on_commit=False)
        self.at = datetime(2023, 8, 12, 10)
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.db.add(Scenario(id='22', asset_id='21', source_hash='x', rows=160,
            start=self.at-timedelta(days=1), end=self.at+timedelta(minutes=140),
            prediction_start=self.at, prediction_end=self.at+timedelta(minutes=140), prediction_points=15,
            annotation={'start': '2023-08-12 00:00:00', 'end': '2023-08-19 10:00:00',
                        'label': 'anomaly', 'description': 'SECRET LABEL'}))
        for i in range(160):
            self.db.add(Measurement(scenario_id='22', row_id=i, source_at=self.at+timedelta(minutes=(i-144)*10),
                partition='train' if i < 144 else 'prediction', status_type_id=999,
                values={'sensor_0_avg': 20, 'sensor_13_avg': 40, 'DO_NOT_SEND': 999}))
        self.db.commit()
        self.assertIn('model_id', ReplaySession.__table__.columns, 'Replay does not preserve the selected model')
        self.s = service.create_session(self.db, '22', 36000, model_id=H7)
        service.command(self.db, self.s.id, 'start', 0, 'start', now=self.now)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def tick(self, i, result):
        job = service.claim_tick(self.db, self.now+timedelta(seconds=i))
        self.assertIsNotNone(job)
        self.assertTrue(service.commit_tick(self.db, job, result, self.now+timedelta(seconds=i)))

    def result(self, label):
        return {'status': 'ok', 'predicted_class': label, 'class_1_score': 1.0 if label else 0.1,
                'model_id': H7, 'prediction_horizon': 'P7D', 'prediction_horizon_hours': 168,
                'score_semantics': 'uncalibrated_failure_score', 'threshold': 0.9999999614086562}

    def test_payload_has_only_past_physical_readings_and_never_labels(self):
        job = service.claim_tick(self.db, self.now)
        payload = service.forecast_payload(self.db, job)
        self.assertEqual(payload['source_time'], '2023-08-12 10:00:00')
        self.assertEqual(len(payload['history']), 145)
        self.assertLessEqual(max(p['source_time'] for p in payload['history']), payload['source_time'])
        self.assertEqual(set(payload), {'asset_id', 'source_time', 'history'})
        self.assertNotIn('SECRET LABEL', str(payload))
        self.assertNotIn('status_type_id', str(payload))
        self.assertNotIn('DO_NOT_SEND', str(payload))
        self.assertNotIn('partition', str(payload))

    def test_hourly_alert_survives_unscheduled_ticks_and_resolves_on_next_negative(self):
        self.tick(0, self.result(1))
        self.assertEqual(len(service.list_alerts(self.db, session_id=self.s.id)), 1)
        for i in range(1, 6):
            self.tick(i, {'status': 'not_scheduled', 'model_id': H7})
        a = service.list_alerts(self.db, session_id=self.s.id)[0]
        self.assertEqual(a['status'], 'active')
        self.assertEqual(a['points'],1)
        self.assertEqual(a['last'],'2023-08-12 10:00:00')
        self.assertIsNone(a['subsystem'])
        self.assertEqual(a['diagnostic']['category'], 'future_failure')
        self.assertNotIn('SECRET LABEL', str(a))
        self.assertEqual(self.db.scalar(select(func.count()).select_from(Prediction)), 1)
        self.assertEqual(service.snapshot(self.db, self.s)['latest_prediction']['source_time'], '2023-08-12 10:00:00')
        self.tick(6, self.result(0))
        self.assertEqual(service.list_alerts(self.db, session_id=self.s.id)[0]['status'], 'resolved')
        self.assertEqual(self.db.scalar(select(func.count()).select_from(Prediction)), 2)

    def test_unavailable_prediction_interrupts_warning_instead_of_marking_normal(self):
        self.tick(0, self.result(1))
        for i in range(1, 6):
            self.tick(i, {'status': 'not_scheduled', 'model_id': H7})
        self.tick(6, {'status': 'unavailable', 'model_id': H7, 'predicted_class': None, 'class_1_score': None})
        a = service.list_alerts(self.db, session_id=self.s.id)[0]
        self.assertEqual(a['resolution_reason'], 'data_quality')
        self.assertEqual(service.snapshot(self.db, self.s)['model_state'], 'unknown')

    def test_schedule_is_hourly_and_model_is_preserved_after_reset(self):
        from app.forecast_protocol import forecast_due
        self.assertTrue(forecast_due('2023-08-12 10:00:00'))
        self.assertFalse(forecast_due('2023-08-12 10:10:00'))
        self.assertTrue(forecast_due('2023-08-12 11:00:00'))
        service.command(self.db, self.s.id, 'reset', self.s.revision, 'reset', now=self.now)
        self.assertEqual(service.snapshot(self.db, self.s)['model_id'], H7)

    def test_unknown_model_cannot_create_a_session(self):
        with self.assertRaises(ValueError):
            service.create_session(self.db, '22', model_id='made-up-model')

    def test_multiple_events_do_not_duplicate_the_same_physical_turbine(self):
        from app.main import turbines
        original=self.db.get(Scenario,'22')
        self.db.add(Scenario(id='13',asset_id='21',source_hash='y',rows=0,start=original.start,end=original.end,
            prediction_start=original.prediction_start,prediction_end=original.prediction_end,prediction_points=0,annotation={}))
        self.db.commit()
        fleet=turbines(db=self.db)
        self.assertEqual(len(fleet),1)
        self.assertEqual(fleet[0]['scenario_id'],'22')
        self.assertEqual(set(fleet[0]['scenario_ids']),{'13','22'})

    def test_measurement_gap_invalidates_a_previous_forecast_between_hourly_runs(self):
        self.tick(0,self.result(1))
        self.db.delete(self.db.get(Measurement,('22',145)))
        self.db.commit()
        self.tick(1,{'status':'not_scheduled','model_id':H7})
        snap=service.snapshot(self.db,self.s)
        self.assertEqual(snap['model_state'],'unknown')
        self.assertEqual(snap['latest_prediction']['status'],'data_gap')
        self.assertEqual(service.list_alerts(self.db,session_id=self.s.id)[0]['resolution_reason'],'data_gap')
