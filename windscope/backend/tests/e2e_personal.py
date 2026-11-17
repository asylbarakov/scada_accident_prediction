"""Live local demo acceptance: genuine inference, future visibility and pacing."""
import json,time,sys,uuid
from urllib.request import Request,urlopen

BASE=sys.argv[1] if len(sys.argv)>1 else 'http://host.docker.internal:8100/api/v1'


def call(path,body=None):
    request=Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=20) as response:return json.load(response)


def complete(case):
    session=call('/demo/start',dict(scenario_id=case,speed=36000,repeat=False))
    identifier=session['id'];deadline=time.monotonic()+50
    hidden=call(f'/demo/{identifier}/check')
    if case!='25':assert hidden['failure_at'] is None
    while time.monotonic()<deadline:
        session=call(f'/replay-sessions/{identifier}')
        if session['status']=='completed':break
        time.sleep(.15)
    assert session['status']=='completed',session
    predictions=call(f'/replay-sessions/{identifier}/predictions?limit=1000')
    good=[p for p in predictions if p['status']=='ok']
    assert good and all(p['model_id']=='care-personal-normal-h24-v1' for p in good)
    assert all(p['feature_cutoff']<p['source_time'] for p in good)
    result=call(f'/demo/{identifier}/check')
    alerts=call(f'/replay-sessions/{identifier}/alerts')
    if case=='26':
        assert result['state']=='warned_before_failure',result
        assert abs(result['lead_hours']-7/3)<1e-10,result
    if case=='25':assert not any(p['predicted_class']==1 for p in good)
    if case=='0':assert result['state']=='missed_failure',result
    if case=='42':
        assert result['state']=='warned_before_failure' and alerts
        assert any(a['status']!='resolved' for a in alerts),'Watch was lost on a lower score'
    return dict(case=case,session_id=identifier,measurements=session['position'],successful_model_calls=len(good),evaluation=result['state'],lead_hours=result['lead_hours'],alerts=len(alerts))


results=[complete(case) for case in ('26','25','0','42')]
session=call('/demo/start',dict(scenario_id='26',speed=300,repeat=False))
identifier=session['id'];began=time.monotonic();deadline=began+20
while time.monotonic()<deadline:
    session=call(f'/replay-sessions/{identifier}')
    if session['position']>=4:break
    time.sleep(.2)
elapsed=time.monotonic()-began
assert elapsed>=5.5 and session['position']<=5,(elapsed,session['position'])
call(f'/replay-sessions/{identifier}/commands',dict(action='pause',revision=session['revision'],command_id=str(uuid.uuid4())))
loop=call('/demo/start',dict(scenario_id='26',speed=36000,repeat=True));deadline=time.monotonic()+50
while time.monotonic()<deadline:
    loop=call('/replay-sessions/'+loop['id'])
    if loop['cycle']>=2 and loop['position']>=2:break
    time.sleep(.15)
assert loop['cycle']>=2 and loop['status']=='running',loop
call('/replay-sessions/'+loop['id']+'/commands',dict(action='pause',revision=loop['revision'],command_id=str(uuid.uuid4())))
print(json.dumps(dict(status='passed',cases=results,pacing=dict(speed=300,positions=session['position'],elapsed_seconds=round(elapsed,2)),continuous_cycle=loop['cycle']),indent=2))
