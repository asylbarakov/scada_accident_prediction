"""Real continuous replay, pacing changes and newly computed H7 warnings."""
import json,time,uuid
from pathlib import Path
from datetime import datetime,timedelta
import httpx
from websockets.sync.client import connect

client=httpx.Client(base_url='http://api:8000/api/v1',timeout=15)


def get(path):
    r=client.get(path);r.raise_for_status();return r.json()


def post(path,data):
    r=client.post(path,json=data);r.raise_for_status();return r.json()


def command(s,action,**extra):
    return post(f"/replay-sessions/{s['id']}/commands",dict(action=action,revision=s['revision'],command_id=str(uuid.uuid4()),**extra))


def wait_for(identifier,predicate,timeout=25):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        s=get('/replay-sessions/'+identifier)
        if predicate(s):return s
        time.sleep(.15)
    raise AssertionError('Monitoring timed out: '+identifier)


def main():
    scenario=next(s for s in get('/scenarios') if s['id']=='22')
    s=post('/monitoring',{'scenario_id':'22','speed':300})
    assert s['continuous'] and s['model_id']=='care-h7-logistic-20261116'
    assert s['total']==scenario['prediction_points'] and s['total']>1000
    assert s['start']==scenario['prediction_start'] and s['end']==scenario['prediction_end']
    s=command(s,'reset');s=command(s,'start')
    s=wait_for(s['id'],lambda s:s['position']>=2)
    slow=command(s,'speed',speed=60)
    position=slow['position'];started=time.monotonic()
    time.sleep(2)
    later=get('/replay-sessions/'+s['id'])
    slow_elapsed=time.monotonic()-started
    assert later['speed']==60 and later['position']==position,'Slow speed was not respected'
    fast=command(later,'speed',speed=300)
    s=wait_for(fast['id'],lambda s:s['prediction_count']>=2)
    predictions=get(f"/replay-sessions/{s['id']}/predictions")
    assert s['status']=='running' and len(predictions)>=2
    assert all(p['prediction_horizon_hours']==168 for p in predictions)
    assert abs(predictions[0]['class_1_score']-0.9999999998243527)<1e-10
    assert predictions[0]['source_time']=='2023-08-12 10:00:00'
    assert any(p['predicted_class']==1 for p in predictions)
    assert get(f"/replay-sessions/{s['id']}/alerts")
    command(s,'pause')
    # An intentionally short repeated range verifies the boundary without
    # waiting days. User monitoring always requests the entire source instead.
    at=datetime.fromisoformat(predictions[0]['source_time'])
    loop=post('/replay-sessions',dict(scenario_id='22',speed=600,continuous=True,start=str(at),end=str(at+timedelta(minutes=10))))
    with connect(f"ws://api:8000/api/v1/replay-sessions/{loop['id']}/stream?after_seq={loop['sequence']}") as ws:
        assert json.loads(ws.recv(timeout=5))['type']=='snapshot'
        command(loop,'start')
        while True:
            event=json.loads(ws.recv(timeout=12))
            if event['type']=='cycle_started':break
        assert event['generation']==1 and event['payload']['cycle']==2
        assert 'measurement' not in event['payload']
    loop=get('/replay-sessions/'+loop['id'])
    assert loop['status']=='running' and loop['cycle']>=2
    command(loop,'pause')
    archive=get('/alerts?archive=true&session_id='+loop['id'])
    assert any(a['resolution_reason']=='dataset_replay' for a in archive)
    report=dict(status='passed',whole_source_points=s['total'],monitor_session=s['id'],
        slow_speed_respected=True,slow_check_wall_seconds=slow_elapsed,
        new_prediction_count=s['prediction_count'],first_warning=predictions[0]['source_time'],
        continuous_boundary_websocket=True,loop_session=loop['id'],loop_cycle=loop['cycle'],
        old_cycles_archived=True,future_readings_used=False)
    Path('/verification/monitoring_e2e.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
