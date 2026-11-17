import json,sys,time,uuid
from pathlib import Path
import httpx
client=httpx.Client(base_url='http://api:8000/api/v1',timeout=5)
path=Path('/evidence/restart_checkpoint.json')


def get(id):
    response=client.get('/replay-sessions/'+id);response.raise_for_status();return response.json()


def command(s,action,**extra):
    response=client.post('/replay-sessions/'+s['id']+'/commands',json=dict(action=action,revision=s['revision'],command_id=str(uuid.uuid4()),**extra));response.raise_for_status();return response.json()


if sys.argv[1]=='prepare':
    response=client.post('/replay-sessions',json={'scenario_id':'0','speed':1});response.raise_for_status();s=command(response.json(),'start')
    for _ in range(50):
        s=get(s['id'])
        if s['position']:break
        time.sleep(.1)
    assert s['position']==1
    path.write_text(json.dumps(s),encoding='utf-8');print('Prepared running session',s['id'],s['position'])
else:
    saved=json.loads(path.read_text(encoding='utf-8'))
    for _ in range(50):
        try:s=get(saved['id']);break
        except httpx.HTTPError:time.sleep(.2)
    assert s['status']=='paused' and s['position']==saved['position'] and s['generation']==saved['generation']
    s=command(s,'speed',speed=3600);s=command(s,'resume')
    for _ in range(50):
        s=get(s['id'])
        if s['position']>saved['position']:break
        time.sleep(.1)
    assert s['position']>saved['position'];s=command(s,'pause')
    Path('/evidence/restart_verified.json').write_text(json.dumps({'status':'passed','restored_cursor':saved['position'],'resumed_cursor':s['position'],'session_id':s['id']},indent=2),encoding='utf-8')
    print('PASS worker/API restart and persistent resume')
