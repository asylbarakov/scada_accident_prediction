"""Persistent replay use cases. No model loading and no browser-owned clocks."""
import math
from datetime import timedelta
from uuid import uuid4
from sqlalchemy import select,func,or_,and_
from app.models import Scenario,Measurement,ReplaySession,Prediction,Alert,ReplayEvent,Diagnostic,OperationalAlert,AlertRuntime
from app.domain import utc_now,utc_iso,aware,remaining_source,next_due,schedule_after,model_state,observable_annotation
from app.alert_engine import AlertPolicy,advance
from app.diagnostics import diagnose,quality,forecast_observation
from app.forecast_protocol import H7_MODEL,LEGACY_MODEL,PERSONAL_MODEL,PERSONAL_SOURCES,H7_SENSORS,model_info
from app.personal_policy import advance_watch


class Conflict(ValueError):pass
class Missing(ValueError):pass


def monitoring_priority():
    return and_(ReplaySession.continuous==True,ReplaySession.model_id==H7_MODEL,
                ReplaySession.start==Scenario.prediction_start,ReplaySession.end==Scenario.prediction_end)


def required(db, cls, identifier):
    item=db.get(cls,identifier)
    if item is None:raise Missing('Объект не найден')
    return item


def measurement_dict(m):
    return {'row_id':m.row_id,'source_time':str(m.source_at),'values':m.values,'partition':m.partition} if m else None


def prediction_count(db,s):
    return db.scalar(select(func.count()).select_from(Prediction).where(Prediction.session_id==s.id,
        Prediction.generation==s.generation,Prediction.result['status'].as_string()=='ok'))


def snapshot(db,s):
    if isinstance(s,str):s=required(db,ReplaySession,s)
    scenario=required(db,Scenario,s.scenario_id)
    pred=db.scalar(select(Prediction).where(Prediction.session_id==s.id,Prediction.generation==s.generation).order_by(Prediction.id.desc()).limit(1))
    q=select(Measurement).where(Measurement.scenario_id==s.scenario_id)
    q=q.where(Measurement.source_at<=s.virtual_time) if s.virtual_time else q.where(Measurement.source_at<s.start)
    latest=db.scalar(q.order_by(Measurement.row_id.desc()).limit(1))
    diagnostic=db.scalar(select(Diagnostic).where(Diagnostic.session_id==s.id,Diagnostic.generation==s.generation).order_by(Diagnostic.row_id.desc()).limit(1))
    runtime=db.get(AlertRuntime,(s.id,s.generation))
    active_count=db.scalar(select(func.count()).select_from(OperationalAlert).where(OperationalAlert.session_id==s.id,OperationalAlert.generation==s.generation,OperationalAlert.end==None))
    return dict(id=s.id,scenario_id=s.scenario_id,asset_id=scenario.asset_id,status=s.status,speed=s.speed,
        generation=s.generation,revision=s.revision,sequence=s.sequence,position=s.position,total=s.total,
        continuous=s.continuous,cycle=s.cycle,prediction_count=prediction_count(db,s),
        monitoring=bool(s.continuous and s.model_id==H7_MODEL and s.start==scenario.prediction_start and s.end==scenario.prediction_end),
        progress=round(s.position/max(s.total,1)*100,3),start=str(s.start),end=str(s.end),
        virtual_time=str(s.virtual_time) if s.virtual_time else None,latest=measurement_dict(latest),
        latest_prediction=pred.result if pred else None,model_state=model_state(pred.result if pred else None),
        dataset_timezone='unknown',model_id=s.model_id,model=model_info(s.model_id),condition='not_assessed',
        diagnostic=diagnostic.result if diagnostic else None,active_alerts=active_count,
        alert_policy=runtime.policy if runtime else AlertPolicy().to_dict())


def emit(db,s,type_,payload):
    s.sequence+=1
    db.add(ReplayEvent(session_id=s.id,sequence=s.sequence,generation=s.generation,type=type_,payload=payload))


def create_session(db,scenario_id,speed=600,start=None,end=None,alert_policy=None,model_id=LEGACY_MODEL,continuous=False):
    model_info(model_id)
    if model_id==PERSONAL_MODEL and scenario_id not in PERSONAL_SOURCES:raise ValueError('Нет проверенного персонального baseline для этого сценария')
    scenario=required(db,Scenario,scenario_id)
    if not 1<=speed<=36000:raise ValueError('Скорость: от 1 до 36000')
    start=start or scenario.prediction_start;end=end or scenario.prediction_end
    if start<scenario.prediction_start or end>scenario.prediction_end or end<start:
        raise ValueError('Replay range должен находиться внутри prediction-части')
    total=db.scalar(select(func.count()).select_from(Measurement).where(Measurement.scenario_id==scenario_id,Measurement.source_at>=start,Measurement.source_at<=end))
    if not total:raise ValueError('В выбранном интервале нет измерений')
    s=ReplaySession(id=str(uuid4()),scenario_id=scenario_id,speed=speed,total=total,start=start,end=end,model_id=model_id,continuous=continuous)
    settings=dict(alert_policy or {})
    if model_id in (H7_MODEL,PERSONAL_MODEL):
        settings={'activate_points':1,'resolve_points':1,'cooldown_minutes':0,**settings,'max_gap_seconds':3600}
    policy=AlertPolicy(**settings)
    db.add(s);db.flush();db.add(AlertRuntime(session_id=s.id,generation=s.generation,policy=policy.to_dict(),state={}))
    emit(db,s,'state_changed',{'reason':'created'});db.commit()
    return s


def start_monitoring(db,scenario_id,speed=300):
    scenario=required(db,Scenario,scenario_id)
    s=db.scalar(select(ReplaySession).where(ReplaySession.scenario_id==scenario_id,ReplaySession.continuous==True,
        ReplaySession.model_id==H7_MODEL,ReplaySession.start==scenario.prediction_start,ReplaySession.end==scenario.prediction_end)
        .order_by(ReplaySession.created_at.desc()).with_for_update().limit(1))
    if s is None:s=create_session(db,scenario_id,speed,model_id=H7_MODEL,continuous=True)
    if s.speed!=speed:command(db,s.id,'speed',s.revision,str(uuid4()),speed=speed)
    if s.status=='completed':command(db,s.id,'reset',s.revision,str(uuid4()))
    if s.status in ('created','paused'):command(db,s.id,'start' if s.status=='created' else 'resume',s.revision,str(uuid4()))
    return s


def command(db,identifier,action,revision,command_id,speed=None,now=None):
    now=now or utc_now()
    s=db.scalar(select(ReplaySession).where(ReplaySession.id==identifier).with_for_update())
    if not s:raise Missing('Сессия не найдена')
    duplicate=db.scalar(select(ReplayEvent.sequence).where(ReplayEvent.session_id==identifier,
        ReplayEvent.payload['command_id'].as_string()==command_id).limit(1))
    if duplicate is not None:return snapshot(db,s)
    if s.revision!=revision:raise Conflict('Состояние изменилось. Обновите сессию и повторите команду')
    if action in ('start','resume'):
        if s.status not in ('created','paused'):raise Conflict('Запуск разрешён из состояния ожидания или паузы')
        s.due_at=next_due(now,s.remaining,s.speed);s.status='running'
    elif action=='pause':
        if s.status!='running':raise Conflict('Поставить на паузу можно запущенную сессию')
        s.remaining=remaining_source(s.due_at,now,s.speed);s.due_at=None;s.status='paused'
    elif action=='reset':
        runtime=db.get(AlertRuntime,(s.id,s.generation))
        saved_policy=runtime.policy if runtime else AlertPolicy().to_dict()
        for alert in db.scalars(select(OperationalAlert).where(OperationalAlert.session_id==s.id,OperationalAlert.generation==s.generation,OperationalAlert.end==None)).all():
            alert.status='resolved';alert.end=alert.last;alert.resolved_at=now;alert.resolution_reason='replay_reset'
        s.status='created';s.position=0;s.cursor=-1;s.virtual_time=None;s.due_at=None;s.remaining=0;s.generation+=1;s.cycle=1
        db.add(AlertRuntime(session_id=s.id,generation=s.generation,policy=saved_policy,state={}))
    elif action=='speed':
        if speed is None or not 1<=speed<=36000:raise ValueError('Скорость: от 1 до 36000')
        s.remaining=remaining_source(s.due_at,now,s.speed) if s.status=='running' else s.remaining
        s.speed=speed
        if s.status=='running':s.due_at=next_due(now,s.remaining,speed)
    else:raise ValueError('Неизвестная команда')
    s.revision+=1;s.lease=None;s.lease_until=None
    emit(db,s,'state_changed',{'reason':action,'command_id':command_id})
    db.commit();return snapshot(db,s)


def claim_tick(db,now=None):
    now=now or utc_now()
    s=db.scalar(select(ReplaySession).where(ReplaySession.status=='running',ReplaySession.due_at<=now,
        or_(ReplaySession.lease_until==None,ReplaySession.lease_until<now)).order_by(ReplaySession.due_at).with_for_update(skip_locked=True).limit(1))
    if not s:return None
    m=db.scalar(select(Measurement).where(Measurement.scenario_id==s.scenario_id,Measurement.row_id>s.cursor,
        Measurement.source_at>=s.start,Measurement.source_at<=s.end).order_by(Measurement.row_id).limit(1))
    if not m:
        s.status='completed';s.due_at=None;emit(db,s,'completed',{});db.commit();return None
    scenario=required(db,Scenario,s.scenario_id)
    s.lease=str(uuid4());s.lease_until=now+timedelta(seconds=30)
    job=dict(session_id=s.id,revision=s.revision,generation=s.generation,lease=s.lease,
        model_id=s.model_id,
        scenario_id=s.scenario_id,asset_id=scenario.asset_id,row_id=m.row_id,source_time=str(m.source_at),
        partition=m.partition,values=dict(m.values),due_at=s.due_at,previous_time=s.virtual_time)
    db.commit();return job


def forecast_payload(db,job):
    from datetime import datetime
    at=datetime.fromisoformat(job['source_time'])
    personal=job['model_id']==PERSONAL_MODEL
    rows=db.scalars(select(Measurement).where(Measurement.scenario_id==job['scenario_id'],
        Measurement.source_at>=at-timedelta(hours=8 if personal else 168),Measurement.source_at<=at)
        .order_by(Measurement.source_at)).all()
    payload=dict(asset_id=job['asset_id'],source_time=job['source_time'],history=[
        dict(source_time=str(m.source_at),values={k:m.values.get(k) for k in H7_SENSORS}) for m in rows])
    if personal:payload['source_id']=job['scenario_id']
    return payload


def commit_tick(db,job,result,now=None):
    now=now or utc_now()
    s=db.scalar(select(ReplaySession).where(ReplaySession.id==job['session_id']).with_for_update())
    if not s or s.status!='running' or s.revision!=job['revision'] or s.generation!=job['generation'] or s.lease!=job['lease']:
        db.rollback();return False
    from datetime import datetime
    source=datetime.fromisoformat(job['source_time'])
    personal=s.model_id==PERSONAL_MODEL
    h7=s.model_id in (H7_MODEL,PERSONAL_MODEL)
    scheduled=result.get('status')!='not_scheduled'
    gap=(source-job['previous_time']).total_seconds() if job['previous_time'] else None
    if h7 and not scheduled and gap and gap>600:
        scheduled=True
        result=dict(status='data_gap',predicted_class=None,class_1_score=None,model_id=s.model_id,
                    prediction_horizon='P1D' if personal else 'P7D',prediction_horizon_hours=24 if personal else 168)
    result=dict(result,source_time=job['source_time'],processed_at=now.isoformat(),row_id=job['row_id'],kind='model')
    diagnostic=None
    if scheduled:
        db.add(Prediction(session_id=s.id,generation=s.generation,row_id=job['row_id'],source_at=source,processed_at=now,result=result))
        measurement=dict(asset_id=job['asset_id'],source_time=job['source_time'],values=job['values'])
        if h7:diagnostic=forecast_observation(measurement,result)
        else:
            past=db.scalars(select(Measurement).where(Measurement.scenario_id==s.scenario_id,Measurement.source_at<source).order_by(Measurement.row_id.desc()).limit(72)).all()
            diagnostic=diagnose(measurement,result,[measurement_dict(m) for m in past])
        db.add(Diagnostic(session_id=s.id,generation=s.generation,row_id=job['row_id'],source_at=source,result=diagnostic))
    runtime=db.get(AlertRuntime,(s.id,s.generation))
    if runtime is None:
        runtime=AlertRuntime(session_id=s.id,generation=s.generation,policy=AlertPolicy().to_dict(),state={});db.add(runtime)
    policy=AlertPolicy(**{k:v for k,v in runtime.policy.items() if k!='version'})
    if personal and (scheduled or (gap and gap>600)):
        state,transition=advance_watch(runtime.state,source,result)
    elif scheduled or (gap and gap>600):
        policy_result={'status':'data_gap'} if gap and gap>600 else result
        state,transition=advance(runtime.state,source,policy_result,policy,quality_ok=True if h7 else quality(job['values']))
    else:state,transition=runtime.state,'not_scheduled'
    runtime.state=state
    active=db.scalar(select(OperationalAlert).where(OperationalAlert.session_id==s.id,OperationalAlert.generation==s.generation,OperationalAlert.end==None).limit(1))
    if active:
        if scheduled:active.last=source;active.points+=1
        if transition in ('resolved','interrupted'):
            active.end=source;active.status='resolved';active.resolved_at=now;active.resolution_reason=state['resolution_reason']
            emit(db,s,'alert_resolved',{'alert_id':active.id,'reason':active.resolution_reason})
        # Keep activation evidence immutable. Live diagnostics are stored separately.
    elif transition=='activated':
        active=OperationalAlert(session_id=s.id,generation=s.generation,asset_id=job['asset_id'],
            first_signal=datetime.fromisoformat(state['first_signal']),start=source,last=source,severity='warning',
            subsystem=diagnostic['suspected_subsystem'],detail=diagnostic,policy=runtime.policy,created_at=now)
        db.add(active);db.flush();emit(db,s,'alert_activated',{'alert_id':active.id,'severity':active.severity})
    if gap and gap>600:emit(db,s,'gap',{'source_time':job['source_time'],'gap_seconds':gap,'synthetic':False})
    s.cursor=job['row_id'];s.position+=1;s.virtual_time=source;s.lease=None;s.lease_until=None
    following=db.scalar(select(Measurement.source_at).where(Measurement.scenario_id==s.scenario_id,
        Measurement.row_id>s.cursor,Measurement.source_at<=s.end).order_by(Measurement.row_id).limit(1))
    if following is None:
        s.status='completed';s.due_at=None;s.remaining=0
    else:
        s.remaining=(following-source).total_seconds()
        if s.continuous:s.remaining=min(s.remaining,600)
        # A slow inference call must not trigger a burst of overdue measurements.
        s.due_at=next_due(max(now,aware(job['due_at'])),s.remaining,s.speed)
    emit(db,s,'measurement',{'measurement':{'row_id':job['row_id'],'source_time':job['source_time'],
        'values':job['values'],'partition':job['partition'],'gap_before_seconds':gap},'prediction':result if scheduled else None,'position':s.position,
        'total':s.total,'status':s.status,'virtual_time':str(source),'diagnostic':diagnostic,'alert_transition':transition,
        'prediction_count':prediction_count(db,s)})
    if s.status=='completed' and s.continuous:
        for alert in db.scalars(select(OperationalAlert).where(OperationalAlert.session_id==s.id,
                OperationalAlert.generation==s.generation,OperationalAlert.end==None)).all():
            alert.status='resolved';alert.end=alert.last;alert.resolved_at=now;alert.resolution_reason='dataset_replay'
        s.status='running';s.generation+=1;s.cycle+=1;s.position=0;s.cursor=-1;s.virtual_time=None
        s.remaining=600;s.due_at=next_due(now,600,s.speed)
        db.add(AlertRuntime(session_id=s.id,generation=s.generation,policy=runtime.policy,state={}))
        emit(db,s,'cycle_started',{'cycle':s.cycle,'reason':'historical_data_repeated'})
    elif s.status=='completed':emit(db,s,'completed',{})
    db.commit();return True


def history(db,identifier,max_points=1000,start=None,end=None,analysis=False):
    s=required(db,ReplaySession,identifier)
    q=select(Measurement).where(Measurement.scenario_id==s.scenario_id)
    if not analysis:
        q=q.where(Measurement.source_at<=s.virtual_time) if s.virtual_time else q.where(Measurement.source_at<s.start)
    if start:q=q.where(Measurement.source_at>=start)
    if end:q=q.where(Measurement.source_at<=end)
    max_points=max(10,min(max_points,3000))
    gaps={}
    if analysis:
        timeline=db.execute(select(Measurement.row_id,Measurement.source_at).where(Measurement.scenario_id==s.scenario_id,
            *([Measurement.source_at>=start] if start else []),*([Measurement.source_at<=end] if end else [])).order_by(Measurement.row_id)).all()
        if not timeline:return []
        anchors={timeline[0][0],timeline[-1][0]}
        for a,b in zip(timeline,timeline[1:]):
            delta=(b[1]-a[1]).total_seconds();gaps[b[0]]=delta
            if delta>600:anchors.update((a[0],b[0]))
        stride=max(1,math.ceil(len(timeline)/max_points))
        q=q.where(or_(Measurement.row_id%stride==0,Measurement.row_id.in_(anchors)))
        rows=db.scalars(q.order_by(Measurement.row_id).limit(max_points+len(anchors)+2)).all()
    else:
        rows=list(reversed(db.scalars(q.order_by(Measurement.row_id.desc()).limit(max_points)).all()))
        gaps={b.row_id:(b.source_at-a.source_at).total_seconds() for a,b in zip(rows,rows[1:])}
    return [dict(measurement_dict(m),gap_before_seconds=gaps.get(m.row_id)) for m in rows]


def annotations(db,identifier,analysis=False):
    s=required(db,ReplaySession,identifier);a=required(db,Scenario,s.scenario_id).annotation
    if analysis:return [dict(a,visible_until=a['end'])]
    visible=observable_annotation(a,s.virtual_time)
    return [visible] if visible else []


def alert_dict(a):
    return dict(id=a.id,session_id=a.session_id,generation=a.generation,turbine_id=a.asset_id,
        first_signal=str(a.first_signal),start=str(a.start),last=str(a.last),end=str(a.end) if a.end else None,
        status=a.status,severity=a.severity,subsystem=a.subsystem,points=a.points,diagnostic=a.detail,policy=a.policy,
        created_at=utc_iso(a.created_at),acknowledged_at=utc_iso(a.acknowledged_at),
        resolved_at=utc_iso(a.resolved_at),resolution_reason=a.resolution_reason)


def list_alerts(db,session_id=None,turbine_id=None,severity=None,status=None,limit=200,archive=False):
    q=select(OperationalAlert).join(ReplaySession,OperationalAlert.session_id==ReplaySession.id)
    if not archive:q=q.where(OperationalAlert.generation==ReplaySession.generation)
    if session_id:q=q.where(OperationalAlert.session_id==session_id)
    elif not archive:
        ranked=select(ReplaySession.id.label('session_id'),func.row_number().over(partition_by=ReplaySession.scenario_id,
            order_by=(monitoring_priority().desc(),ReplaySession.created_at.desc())).label('rank'))\
            .join(Scenario,ReplaySession.scenario_id==Scenario.id).subquery()
        q=q.where(OperationalAlert.session_id.in_(select(ranked.c.session_id).where(ranked.c.rank==1)))
    if turbine_id:q=q.where(OperationalAlert.asset_id==turbine_id)
    if severity:q=q.where(OperationalAlert.severity==severity)
    if status:q=q.where(OperationalAlert.status==status)
    return [alert_dict(a) for a in db.scalars(q.order_by(OperationalAlert.id.desc()).limit(limit)).all()]


def acknowledge(db,identifier):
    a=required(db,OperationalAlert,identifier)
    s=db.scalar(select(ReplaySession).where(ReplaySession.id==a.session_id).with_for_update())
    db.refresh(a)
    if a.generation!=s.generation:raise Conflict('Предупреждение относится к предыдущему запуску')
    if a.status=='resolved':raise Conflict('Предупреждение уже закрыто')
    if a.status!='acknowledged':
        a.status='acknowledged';a.acknowledged_at=utc_now();emit(db,s,'alert_acknowledged',{'alert_id':a.id});db.commit()
    return alert_dict(a)
