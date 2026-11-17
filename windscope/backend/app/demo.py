"""Curated historical demonstration. Outcome labels never enter inference."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import select
from app import service
from app.models import Scenario,ReplaySession,Prediction
from app.forecast_protocol import PERSONAL_MODEL

CASES={
    '26':dict(title='Раннее предупреждение',subtitle='Турбина 00 · короткий демонстрационный проход',start='2023-10-19 04:00:00',end='2023-10-19 14:00:00',kind='warning'),
    '25':dict(title='Нормальная работа',subtitle='Турбина 11 · контрольный пример без предупреждений',start='2023-05-24 06:00:00',end='2023-05-24 18:00:00',kind='normal'),
    '42':dict(title='Длинное упреждение',subtitle='Турбина 10 · сигнал и последующее снижение score',start='2023-09-15 23:00:00',end='2023-09-16 18:00:00',kind='warning'),
    '0':dict(title='Пропущенный отказ',subtitle='Турбина 00 · ограничение модели на подшипнике',start='2023-08-19 22:00:00',end='2023-08-20 09:00:00',kind='missed'),
}


def catalog(db):
    return dict(model_id=PERSONAL_MODEL,dataset='CARE v6 · Wind Farm A',historical=True,
        evaluation=dict(detected_events=4,total_events=11,false_alarm_episodes=17,normal_hours=2983,normal_warning_fraction=.0248072410325176),
        cases=[dict(id=identifier,**case,available=db.get(Scenario,identifier) is not None) for identifier,case in CASES.items()],
        disclaimer='Сценарии подобраны для демонстрации. Полная историческая проверка: 4 из 11 отказов; независимая внешняя проверка не выполнена.')


def start(db,scenario_id,speed=300,autoplay=True,repeat=True,replace_session_id=None):
    if scenario_id not in CASES:raise ValueError('Неизвестный демонстрационный сценарий')
    case=CASES[scenario_id];begin=datetime.fromisoformat(case['start']);end=datetime.fromisoformat(case['end'])
    previous=db.get(ReplaySession,replace_session_id) if replace_session_id else None
    if previous and previous.model_id!=PERSONAL_MODEL:previous=None
    reusable=previous and previous.scenario_id==scenario_id and previous.start==begin and previous.end==end and previous.continuous==repeat
    if reusable:
        session=previous
        if session.speed!=speed:service.command(db,session.id,'speed',session.revision,str(uuid4()),speed=speed)
    else:
        if previous and previous.status=='running':service.command(db,previous.id,'pause',previous.revision,str(uuid4()))
        session=service.create_session(db,scenario_id,speed,begin,end,model_id=PERSONAL_MODEL,continuous=repeat)
    if autoplay:
        if session.status=='completed':service.command(db,session.id,'reset',session.revision,str(uuid4()))
        if session.status in ('created','paused'):service.command(db,session.id,'start' if session.status=='created' else 'resume',session.revision,str(uuid4()))
    return service.snapshot(db,session)


def demo_check(db,identifier,reveal=False):
    session=service.required(db,ReplaySession,identifier)
    if session.model_id!=PERSONAL_MODEL:raise ValueError('Проверка относится к персональному demo replay')
    annotation=service.required(db,Scenario,session.scenario_id).annotation
    if annotation['label']!='anomaly':
        return dict(state='normal_scenario',failure_at=None,future_revealed=bool(reveal),first_warning=None,lead_hours=None,
            note='Контрольный исторический сценарий с normal-разметкой. Это не диагноз модели.')
    fault=datetime.fromisoformat(annotation['end'])
    reached=session.virtual_time is not None and session.virtual_time>=fault
    if not reached and not reveal:return dict(state='not_reached',failure_at=None,future_revealed=False,first_warning=None,lead_hours=None,note='Будущая метка скрыта. Модель получает только уже доступные измерения.')
    from datetime import timedelta
    warning=db.scalar(select(Prediction).where(Prediction.session_id==session.id,Prediction.generation==session.generation,
        Prediction.source_at>=fault-timedelta(hours=24),Prediction.source_at<fault,
        Prediction.result['status'].as_string()=='ok',Prediction.result['predicted_class'].as_integer()==1)
        .order_by(Prediction.source_at).limit(1))
    return dict(state='warned_before_failure' if reached and warning else 'missed_failure' if reached else 'future_reference',failure_at=str(fault),future_revealed=bool(reveal and not reached),
        first_warning=str(warning.source_at) if warning else None,lead_hours=(fault-warning.source_at).total_seconds()/3600 if warning else None,
        note='Отказ по CARE-разметке: конец аннотированного аномального интервала. Метка используется только для проверки, не для inference.')
