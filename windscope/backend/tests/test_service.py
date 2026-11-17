import importlib.util
import unittest
from datetime import datetime, timedelta, timezone


class PersistentReplayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('app.service'),'Persistent replay service not implemented')
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from app.models import Base,Scenario,Measurement
        self.engine=create_engine('sqlite://')
        Base.metadata.create_all(self.engine)
        self.db=Session(self.engine,expire_on_commit=False)
        self.now=datetime(2026,1,1,tzinfo=timezone.utc)
        self.db.add(Scenario(id='0',asset_id='0',source_hash='x',rows=4,
            start=datetime(2023,8,4,5,50),end=datetime(2023,8,6,6,10),
            prediction_start=datetime(2023,8,4,6),prediction_end=datetime(2023,8,6,6,10),prediction_points=3,
            annotation={'start':'2023-08-06 06:10:00','end':'2023-08-20 06:10:00','label':'anomaly','description':'bearing'}))
        for i,t in enumerate([datetime(2023,8,4,5,50),datetime(2023,8,4,6),datetime(2023,8,4,6,10),datetime(2023,8,6,6,10)]):
            self.db.add(Measurement(scenario_id='0',row_id=i,source_at=t,partition='train' if i==0 else 'prediction',values={'sensor_0_avg':20+i},status_type_id=0))
        self.db.commit()

    def tearDown(self):
        if hasattr(self,'db'):self.db.close();self.engine.dispose()

    def new(self):
        from app.service import create_session
        return create_session(self.db,'0',600)

    def test_pause_fences_inflight_result_and_state_is_persistent(self):
        from app.service import command,claim_tick,commit_tick,snapshot
        from app.models import ReplaySession,Prediction
        from sqlalchemy import select,func
        s=self.new();command(self.db,s.id,'start',0,'one',now=self.now)
        job=claim_tick(self.db,self.now)
        command(self.db,s.id,'pause',1,'two',now=self.now)
        self.assertFalse(commit_tick(self.db,job,{'status':'ok','predicted_class':1},self.now))
        self.db.expire_all();restored=self.db.get(ReplaySession,s.id)
        self.assertEqual((restored.status,restored.position),('paused',0))
        self.assertEqual(self.db.scalar(select(func.count()).select_from(Prediction)),0)

    def test_live_queries_clip_future_and_reset_hides_old_generation(self):
        from app.service import command,claim_tick,commit_tick,history,annotations
        s=self.new();command(self.db,s.id,'start',0,'start',now=self.now)
        self.assertEqual([r['row_id'] for r in history(self.db,s.id)], [0])
        job=claim_tick(self.db,self.now)
        self.assertTrue(commit_tick(self.db,job,{'status':'ok','predicted_class':1,'model_id':'test'},self.now))
        self.assertEqual([r['row_id'] for r in history(self.db,s.id)], [0,1])
        self.assertEqual(annotations(self.db,s.id),[])
        self.assertEqual(annotations(self.db,s.id,analysis=True)[0]['end'],'2023-08-20 06:10:00')
        old_sequence=s.sequence
        command(self.db,s.id,'reset',s.revision,'reset',now=self.now)
        self.assertEqual((s.generation,s.position,s.status),(1,0,'created'))
        self.assertGreater(s.sequence,old_sequence)
        self.assertEqual([r['row_id'] for r in history(self.db,s.id)], [0])

    def test_duplicate_command_is_idempotent_but_stale_revision_conflicts(self):
        from app.service import command,Conflict
        s=self.new();command(self.db,s.id,'start',0,'same',now=self.now)
        seq=s.sequence
        command(self.db,s.id,'start',0,'same',now=self.now)
        self.assertEqual(s.sequence,seq)
        with self.assertRaises(Conflict):command(self.db,s.id,'pause',0,'different',now=self.now)

    def test_reset_event_backlog_does_not_expose_previous_generation_telemetry(self):
        from app.service import command,claim_tick,commit_tick
        from app.main import events
        s=self.new();command(self.db,s.id,'start',0,'start',now=self.now)
        commit_tick(self.db,claim_tick(self.db,self.now),{'status':'ok','predicted_class':0},self.now)
        command(self.db,s.id,'reset',s.revision,'reset',now=self.now)
        visible=events(s.id,after_seq=0,limit=200,db=self.db)
        self.assertTrue(all(e['generation']==1 for e in visible))
        self.assertFalse(any(e['type']=='measurement' for e in visible))

    def test_analysis_downsampling_retains_real_gap_boundaries(self):
        from app.models import Measurement
        from app.service import history
        for i in range(4,24):
            self.db.add(Measurement(scenario_id='0',row_id=i,source_at=datetime(2023,8,6,6,10)+timedelta(minutes=(i-3)*10),partition='prediction',values={},status_type_id=0))
        self.db.commit();s=self.new()
        points=history(self.db,s.id,max_points=10,analysis=True)
        self.assertIn(2,[p['row_id'] for p in points])
        self.assertEqual(next(p for p in points if p['row_id']==3)['gap_before_seconds'],172800)
        self.assertEqual(next(p for p in points if p['row_id']==6)['gap_before_seconds'],600)


if __name__=='__main__':unittest.main()
