"""Actual HTTP + WebSocket + PostgreSQL + worker + original XGBoost checks."""
import csv,json,time,uuid
from datetime import datetime,timedelta
from pathlib import Path
import httpx
from websockets.sync.client import connect

BASE='http://api:8000/api/v1'
client=httpx.Client(base_url=BASE,timeout=10)
checks=[]


def get(path):
    r=client.get(path);r.raise_for_status();return r.json()


def post(path,body):
    if path=='/replay-sessions':body={'model_id':'swtfps-scada-xgboost-v1.0.0',**body}
    r=client.post(path,json=body);r.raise_for_status();return r.json()


def cmd(s,action,**extra):
    return post('/replay-sessions/'+s['id']+'/commands',dict(action=action,revision=s['revision'],command_id=str(uuid.uuid4()),**extra))


def wait_state(identifier,predicate,timeout=8):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        state=get('/replay-sessions/'+identifier)
        if predicate(state):return state
        time.sleep(.08)
    raise AssertionError('Timed out waiting for replay state')


def warning_baseline(scenario):
    with Path('/baseline_predictions/event_'+scenario+'_predictions.csv').open(newline='') as stream:
        return {int(row['row_id']):row for row in csv.DictReader(stream) if row['partition']=='prediction'}


def main():
    for _ in range(30):
        try:get('/health');break
        except httpx.HTTPError:time.sleep(.3)
    scenarios=[s for s in get('/scenarios') if s['id'] in ('0','25')];assert len(scenarios)==2
    assert sum(s['rows'] for s in scenarios)==109698
    assert len(get('/sensors'))==81
    for scenario in scenarios:
        identifier=scenario['id'];start=datetime.fromisoformat(scenario['prediction_start'])
        s=post('/replay-sessions',dict(scenario_id=identifier,speed=600,start=str(start),end=str(start+timedelta(minutes=30))))
        assert s['position']==0 and s['total']==4
        history=get('/replay-sessions/'+s['id']+'/telemetry')['points']
        assert all(datetime.fromisoformat(p['source_time'])<start for p in history)
        if identifier=='0':assert get('/replay-sessions/'+s['id']+'/annotations')==[]
        with connect('ws://api:8000/api/v1/replay-sessions/'+s['id']+'/stream?after_seq='+str(s['sequence'])) as ws:
            assert json.loads(ws.recv(timeout=5))['type']=='snapshot'
            s=cmd(s,'start');last=0
            while True:
                event=json.loads(ws.recv(timeout=5))
                if 'sequence' in event:assert event['sequence']>last;last=event['sequence']
                if event['type']=='measurement':break
            assert event['payload']['prediction']['status']=='ok'
            assert event['payload']['prediction']['model_id']=='swtfps-scada-xgboost-v1.0.0'
            s=cmd(get('/replay-sessions/'+s['id']),'pause');position=s['position']
            time.sleep(1.15)
            assert get('/replay-sessions/'+s['id'])['position']==position
        # Reconnect from a saved cursor: the pause state event must be recoverable.
        with connect('ws://api:8000/api/v1/replay-sessions/'+s['id']+'/stream?after_seq='+str(last)) as ws:
            assert json.loads(ws.recv(timeout=5))['type']=='snapshot'
            recovered=json.loads(ws.recv(timeout=5));assert recovered['sequence']>last
        cutoff=s['virtual_time']
        assert all(p['source_time']<=cutoff for p in get('/replay-sessions/'+s['id']+'/telemetry')['points'])
        s=cmd(s,'speed',speed=3600);s=cmd(s,'resume')
        s=wait_state(s['id'],lambda x:x['status']=='completed')
        predictions=get('/replay-sessions/'+s['id']+'/predictions')
        assert len(predictions)==4
        expected=warning_baseline(identifier)
        for p in predictions:
            reference=expected[p['row_id']]
            assert p['source_time']==reference['source_time']
            assert p['predicted_class']==int(reference['predicted_class'])
            assert abs(p['class_1_score']-float(reference['class_1_score']))<1e-7
            assert p['prediction_horizon'] is None
        # v2: one grouped alert after the third real positive point; raw outputs untouched.
        alerts=get('/replay-sessions/'+s['id']+'/alerts')
        assert len(alerts)==1 and alerts[0]['status']=='active'
        assert alerts[0]['start']==predictions[2]['source_time']
        assert alerts[0]['diagnostic']['timestamp']==predictions[2]['source_time']
        assert alerts[0]['first_signal']==predictions[0]['source_time']
        assert alerts[0]['severity']!='critical'
        assert 'Generator bearing failure' not in str(alerts[0]['diagnostic'])
        acknowledged=post('/alerts/'+str(alerts[0]['id'])+'/acknowledge',{})
        assert acknowledged['status']=='acknowledged' and acknowledged['acknowledged_at']
        assert post('/alerts/'+str(alerts[0]['id'])+'/acknowledge',{})['acknowledged_at']==acknowledged['acknowledged_at']
        assert get('/alerts?session_id='+s['id']+'&status=acknowledged')[0]['id']==alerts[0]['id']
        diagnostics=get('/replay-sessions/'+s['id']+'/diagnostics')
        assert len(diagnostics)==4 and all(d['prediction_horizon'] is None for d in diagnostics)
        s=cmd(s,'reset')
        assert s['position']==0 and s['generation']==1
        assert get('/replay-sessions/'+s['id']+'/predictions')==[]
        assert get('/replay-sessions/'+s['id']+'/alerts')==[]
        assert get('/replay-sessions/'+s['id']+'/diagnostics')==[]
        archived=get('/alerts?archive=true&session_id='+s['id'])
        assert archived[0]['status']=='resolved' and archived[0]['resolution_reason']=='replay_reset'
        assert all(e['generation']==1 for e in get('/replay-sessions/'+s['id']+'/events'))
        checks.append(dict(scenario=identifier,rows=4,real_model_matches_baseline=True,
            pause_resume_reset=True,reconnect_backlog=True,refresh_persistence=True,no_future_telemetry=True,
            grouped_alert_confirmation=True,acknowledgement_persistent=True,no_annotation_in_diagnostics=True))
        print('PASS scenario',identifier,flush=True)
    fault=next(s for s in scenarios if s['id']=='0')
    sample=post('/replay-sessions',dict(scenario_id='0',speed=36000))
    all_points=get('/replay-sessions/'+sample['id']+'/telemetry?mode=analysis&max_points=3000&start='+fault['prediction_start'].replace(' ','T'))['points']
    pairs=[(a,b) for a,b in zip(all_points,all_points[1:]) if (datetime.fromisoformat(b['source_time'])-datetime.fromisoformat(a['source_time'])).total_seconds()>600]
    assert pairs
    before,after=pairs[0]
    gap=post('/replay-sessions',dict(scenario_id='0',speed=36000,start=before['source_time'],end=after['source_time']))
    gap=cmd(gap,'start');gap=wait_state(gap['id'],lambda x:x['status']=='completed')
    predictions=get('/replay-sessions/'+gap['id']+'/predictions')
    assert [p['source_time'] for p in predictions]==[before['source_time'],after['source_time']]
    events=get('/replay-sessions/'+gap['id']+'/events')
    assert any(e['type']=='gap' and e['payload']['synthetic'] is False for e in events)
    checks.append(dict(actual_missing_timestamp_gap=True,stored_predictions=2,synthetic_points=0))
    result=dict(status='passed',checks=checks)
    Path('/evidence/full_stack_e2e.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
