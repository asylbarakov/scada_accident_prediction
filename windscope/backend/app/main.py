import asyncio,logging
from datetime import datetime
from typing import Literal
from fastapi import FastAPI,Depends,HTTPException,WebSocket,WebSocketDisconnect,Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,ConfigDict,field_validator
from sqlalchemy import select,text
from sqlalchemy.orm import Session
from app.db import db_session,SessionLocal
from app.models import Scenario,Sensor,ReplaySession,ReplayEvent,Prediction,Alert,Diagnostic
from app import service
from app.domain import display_unit,utc_iso
from app.alert_engine import AlertPolicy
from app.sensor_catalog import describe
from app.forecast_protocol import H7_MODEL,LEGACY_MODEL,PERSONAL_MODEL,model_info

logging.basicConfig(level=logging.INFO)
app=FastAPI(title='WindScope SCADA API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:3100','http://127.0.0.1:3100'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])


@app.exception_handler(service.Missing)
async def missing(_,exc):return JSONResponse(status_code=404,content={'detail':str(exc)})


@app.exception_handler(service.Conflict)
async def conflict(_,exc):return JSONResponse(status_code=409,content={'detail':str(exc)})


@app.exception_handler(ValueError)
async def invalid(_,exc):return JSONResponse(status_code=422,content={'detail':str(exc)})


class PolicyInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    activate_points:int=Field(default=3,ge=1,le=100)
    resolve_points:int=Field(default=3,ge=1,le=100)
    cooldown_minutes:int=Field(default=60,ge=0,le=10080)


class Create(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scenario_id:str
    model_id:Literal['care-h7-logistic-20261116','swtfps-scada-xgboost-v1.0.0','care-personal-normal-h24-v1']=H7_MODEL
    speed:float=Field(default=600,ge=1,le=36000)
    start:datetime|None=None
    end:datetime|None=None
    continuous:bool=False
    alert_policy:PolicyInput|None=None

    @field_validator('start','end')
    @classmethod
    def naive(cls,value):
        if value and value.tzinfo is not None:raise ValueError('Используйте исходное время набора без timezone')
        return value


class Command(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action:Literal['start','pause','resume','reset','speed']
    revision:int=Field(ge=0)
    command_id:str=Field(min_length=8,max_length=80)
    speed:float|None=Field(default=None,ge=1,le=36000)


@app.get('/api/v1/health')
def health(db:Session=Depends(db_session)):
    db.execute(text('SELECT 1'));return {'status':'ok','dataset_version':2}


@app.get('/api/v1/scenarios')
def scenarios(db:Session=Depends(db_session)):
    return [dict(id=s.id,asset_id=s.asset_id,rows=s.rows,start=str(s.start),end=str(s.end),
        prediction_start=str(s.prediction_start),prediction_end=str(s.prediction_end),prediction_points=s.prediction_points,
        dataset='Wind Farm A',dataset_version=2,timezone='unknown') for s in db.scalars(select(Scenario).order_by(Scenario.id.desc())).all()]


@app.get('/api/v1/sensors')
def sensors(db:Session=Depends(db_session)):
    return [dict(**describe(s.key),description=s.description,unit=display_unit(s.unit),source_unit=s.unit) for s in db.scalars(select(Sensor).order_by(Sensor.key)).all()]


@app.get('/api/v1/turbines')
def turbines(db:Session=Depends(db_session)):
    result=[]
    grouped={}
    for scenario in db.scalars(select(Scenario).order_by(Scenario.asset_id,Scenario.id)).all():
        grouped.setdefault(scenario.asset_id,[]).append(scenario)
    for asset,items in grouped.items():
        ids=[s.id for s in items]
        session=db.scalar(select(ReplaySession).join(Scenario,ReplaySession.scenario_id==Scenario.id)
            .where(ReplaySession.scenario_id.in_(ids)).order_by(service.monitoring_priority().desc(),ReplaySession.created_at.desc()).limit(1))
        s=next((item for item in items if session and item.id==session.scenario_id),items[0])
        result.append(dict(asset_id=s.asset_id,name='WT-'+s.asset_id.zfill(2),scenario_id=s.id,
                           scenario_ids=ids,
                           session=service.snapshot(db,session) if session else None))
    return result


@app.get('/api/v1/replay-sessions')
def sessions(scenario_id:str|None=None,db:Session=Depends(db_session)):
    q=select(ReplaySession)
    if scenario_id:q=q.where(ReplaySession.scenario_id==scenario_id)
    return [service.snapshot(db,s) for s in db.scalars(q.order_by(ReplaySession.created_at.desc()).limit(50)).all()]


@app.post('/api/v1/replay-sessions',status_code=201)
def create(body:Create,db:Session=Depends(db_session)):
    return service.snapshot(db,service.create_session(db,body.scenario_id,body.speed,body.start,body.end,body.alert_policy.model_dump() if body.alert_policy else None,body.model_id,body.continuous))


class MonitorInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scenario_id:str
    speed:float=Field(default=300,ge=1,le=36000)


@app.post('/api/v1/monitoring')
def start_monitoring(body:MonitorInput,db:Session=Depends(db_session)):
    return service.snapshot(db,service.start_monitoring(db,body.scenario_id,body.speed))


@app.get('/api/v1/models')
def models():return [model_info(PERSONAL_MODEL),model_info(H7_MODEL),model_info(LEGACY_MODEL)]


class DemoInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scenario_id:Literal['0','25','26','42']='26'
    speed:float=Field(default=300,ge=1,le=36000)
    autoplay:bool=True
    repeat:bool=True
    replace_session_id:str|None=None


@app.get('/api/v1/demo/catalog')
def demo_catalog(db:Session=Depends(db_session)):
    from app.demo import catalog
    return catalog(db)


@app.post('/api/v1/demo/start')
def demo_start(body:DemoInput,db:Session=Depends(db_session)):
    from app.demo import start
    return start(db,**body.model_dump())


@app.get('/api/v1/demo/{identifier}/check')
def demo_result(identifier:str,reveal:bool=False,db:Session=Depends(db_session)):
    from app.demo import demo_check
    return demo_check(db,identifier,reveal)


@app.get('/api/v1/replay-sessions/{identifier}')
def session(identifier:str,db:Session=Depends(db_session)):return service.snapshot(db,identifier)


@app.post('/api/v1/replay-sessions/{identifier}/commands')
def command(identifier:str,body:Command,db:Session=Depends(db_session)):
    return service.command(db,identifier,body.action,body.revision,body.command_id,body.speed)


@app.get('/api/v1/replay-sessions/{identifier}/telemetry')
def history(identifier:str,mode:Literal['replay','analysis']='replay',max_points:int=Query(1000,ge=10,le=3000),
            start:datetime|None=None,end:datetime|None=None,db:Session=Depends(db_session)):
    if (start and start.tzinfo) or (end and end.tzinfo):raise ValueError('Используйте время набора без timezone')
    return {'mode':mode,'synthetic':False,'points':service.history(db,identifier,max_points,start,end,analysis=mode=='analysis')}


@app.get('/api/v1/replay-sessions/{identifier}/annotations')
def annotations(identifier:str,mode:Literal['replay','analysis']='replay',db:Session=Depends(db_session)):
    return service.annotations(db,identifier,analysis=mode=='analysis')


@app.get('/api/v1/replay-sessions/{identifier}/predictions')
def predictions(identifier:str,limit:int=Query(1000,ge=1,le=3000),before_id:int|None=None,db:Session=Depends(db_session)):
    s=service.required(db,ReplaySession,identifier)
    q=select(Prediction).where(Prediction.session_id==s.id,Prediction.generation==s.generation)
    if before_id:q=q.where(Prediction.id<before_id)
    rows=db.scalars(q.order_by(Prediction.id.desc()).limit(limit)).all()
    return [dict(id=p.id,**p.result) for p in reversed(rows)]


@app.get('/api/v1/replay-sessions/{identifier}/alerts')
def alerts(identifier:str,db:Session=Depends(db_session)):
    service.required(db,ReplaySession,identifier)
    return service.list_alerts(db,session_id=identifier)


@app.get('/api/v1/alerts')
def alert_center(turbine_id:str|None=None,severity:Literal['observation','warning','critical']|None=None,
    status:Literal['active','acknowledged','resolved']|None=None,session_id:str|None=None,
    limit:int=Query(200,ge=1,le=1000),archive:bool=False,db:Session=Depends(db_session)):
    return service.list_alerts(db,session_id,turbine_id,severity,status,limit,archive)


@app.post('/api/v1/alerts/{identifier}/acknowledge')
def acknowledge(identifier:int,db:Session=Depends(db_session)):return service.acknowledge(db,identifier)


@app.get('/api/v1/replay-sessions/{identifier}/diagnostics')
def diagnostics(identifier:str,limit:int=Query(200,ge=1,le=3000),db:Session=Depends(db_session)):
    s=service.required(db,ReplaySession,identifier)
    return [d.result for d in db.scalars(select(Diagnostic).where(Diagnostic.session_id==s.id,Diagnostic.generation==s.generation).order_by(Diagnostic.row_id.desc()).limit(limit)).all()]


@app.get('/api/v1/alert-policy')
def alert_policy():return dict(default=AlertPolicy().to_dict(),critical_enabled=False,diagnostics_version='causal-observations-v2.0')


def event_dict(e):
    return dict(schema_version='1.0',session_id=e.session_id,sequence=e.sequence,generation=e.generation,
                type=e.type,payload=e.payload,emitted_at=utc_iso(e.emitted_at))


@app.get('/api/v1/replay-sessions/{identifier}/events')
def events(identifier:str,after_seq:int=Query(0,ge=0),limit:int=Query(200,ge=1,le=1000),db:Session=Depends(db_session)):
    s=service.required(db,ReplaySession,identifier)
    return [event_dict(e) for e in db.scalars(select(ReplayEvent).where(ReplayEvent.session_id==identifier,
        ReplayEvent.generation==s.generation,ReplayEvent.sequence>after_seq).order_by(ReplayEvent.sequence).limit(limit)).all()]


@app.websocket('/api/v1/replay-sessions/{identifier}/stream')
async def stream(websocket:WebSocket,identifier:str,after_seq:int=0):
    origin=websocket.headers.get('origin')
    if origin and origin not in ('http://localhost:3100','http://127.0.0.1:3100'):
        await websocket.close(code=1008);return
    with SessionLocal() as db:
        s=db.get(ReplaySession,identifier)
        if not s:await websocket.close(code=1008);return
        snap=service.snapshot(db,s)
    await websocket.accept();cursor=max(0,after_seq)
    await websocket.send_json({'type':'snapshot','session':snap})
    idle=0
    try:
        while True:
            with SessionLocal() as db:
                current=db.get(ReplaySession,identifier)
                if not current:break
                batch=db.scalars(select(ReplayEvent).where(ReplayEvent.session_id==identifier,
                    ReplayEvent.generation==current.generation,ReplayEvent.sequence>cursor).order_by(ReplayEvent.sequence).limit(200)).all()
                payload=[event_dict(e) for e in batch]
            for event in payload:
                await asyncio.wait_for(websocket.send_json(event),timeout=10);cursor=event['sequence']
            if payload:idle=0
            else:
                idle+=1
                if idle%50==0:await websocket.send_json({'type':'heartbeat','after_seq':cursor})
                await asyncio.sleep(.2)
    except (WebSocketDisconnect,RuntimeError,asyncio.TimeoutError):
        logging.info('WebSocket disconnected session=%s after_seq=%s',identifier,cursor)
