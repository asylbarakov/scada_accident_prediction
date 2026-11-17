"""Live API -> PostgreSQL -> replay worker -> isolated H7 -> WebSocket verification."""
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import httpx
from websockets.sync.client import connect

H7='care-h7-logistic-20261116'
client=httpx.Client(base_url='http://api:8000/api/v1',timeout=20)


def get(path):
    r=client.get(path);r.raise_for_status();return r.json()


def post(path,body):
    r=client.post(path,json=body);r.raise_for_status();return r.json()


def command(session,action):
    return post(f"/replay-sessions/{session['id']}/commands",dict(action=action,revision=session['revision'],command_id=str(uuid.uuid4())))


def completed(identifier):
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        s=get('/replay-sessions/'+identifier)
        if s['status']=='completed':return s
        time.sleep(.1)
    raise AssertionError('Replay did not finish: '+identifier)


def main():
    cases=json.loads(Path('/verification/archived_cases.json').read_text(encoding='utf-8-sig'))
    scenarios={s['id']:s for s in get('/scenarios')}
    assert {'0','25','13','22','72'} <= set(scenarios)
    fleet=get('/turbines')
    assert len(fleet)==3 and len({t['asset_id'] for t in fleet})==3
    assert {m['id'] for m in get('/models')} >= {H7}
    results=[]
    for event,points in cases.items():
        for point in points:
            at=datetime.fromisoformat(point['timestamp']).replace(tzinfo=None)
            body=dict(scenario_id=event,speed=36000,start=str(at),end=str(at+timedelta(minutes=20)))
            s=post('/replay-sessions',body)
            assert s['model_id']==H7 and s['alert_policy']['activate_points']==1
            assert s['latest_prediction'] is None
            live=get(f"/replay-sessions/{s['id']}/telemetry")['points']
            assert all(datetime.fromisoformat(p['source_time'])<at for p in live)
            stream_events=[]
            with connect(f"ws://api:8000/api/v1/replay-sessions/{s['id']}/stream?after_seq={s['sequence']}") as ws:
                assert json.loads(ws.recv(timeout=5))['type']=='snapshot'
                command(s,'start')
                while True:
                    message=json.loads(ws.recv(timeout=20))
                    if message['type']=='measurement':stream_events.append(message)
                    if message['type']=='completed':break
            done=completed(s['id'])
            assert done['position']==3
            assert len(stream_events)==3
            assert stream_events[0]['payload']['prediction']['model_id']==H7
            assert all(m['payload']['prediction'] is None for m in stream_events[1:])
            predictions=get(f"/replay-sessions/{s['id']}/predictions")
            assert len(predictions)==1
            p=predictions[0]
            assert p['status']=='ok' and p['prediction_horizon_hours']==168
            assert abs(p['class_1_score']-point['score'])<=1e-10,(event,at,p,point)
            assert p['predicted_class']==int(point['alert'])
            assert p['feature_cutoff']==str(at-timedelta(minutes=10))
            assert p['score_semantics']=='uncalibrated_failure_score'
            assert done['latest_prediction']['source_time']==str(at)
            alerts=get(f"/replay-sessions/{s['id']}/alerts")
            assert len(alerts)==int(point['alert'])
            assert all(a['subsystem'] is None and a['diagnostic']['category']=='future_failure' for a in alerts)
            reset=command(done,'reset')
            assert reset['latest_prediction'] is None and reset['model_id']==H7
            assert get(f"/replay-sessions/{s['id']}/predictions")==[]
            results.append(dict(event=event,at=str(at),session_id=s['id'],score=p['class_1_score'],
                                archived_score=point['score'],alert=bool(p['predicted_class']),
                                websocket_measurements=len(stream_events)))
    # The previous XGBoost is still available and remains pinned to legacy sessions.
    at=datetime.fromisoformat(scenarios['25']['prediction_start'])
    s=post('/replay-sessions',dict(scenario_id='25',model_id='swtfps-scada-xgboost-v1.0.0',
           speed=36000,start=str(at),end=str(at+timedelta(minutes=10))))
    command(s,'start');done=completed(s['id'])
    legacy=get(f"/replay-sessions/{s['id']}/predictions")
    assert len(legacy)==2 and all(p['status']=='ok' and p['prediction_horizon'] is None for p in legacy)
    report=dict(status='passed',cases=results,legacy_compatibility=True,
                real_inference=True,future_readings_used=False,classification_implemented=False)
    Path('/verification/e2e_h7.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
