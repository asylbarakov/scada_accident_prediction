"""Versioned, source-clock alert policy; independent of ML and event annotations."""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

POLICY_VERSION='consecutive-hysteresis-v2.0'


@dataclass(frozen=True)
class AlertPolicy:
    activate_points:int=3
    resolve_points:int=3
    cooldown_minutes:int=60
    max_gap_seconds:int=600

    def __post_init__(self):
        if not 1<=self.activate_points<=100 or not 1<=self.resolve_points<=100:
            raise ValueError('Confirmation counts must be 1..100')
        if not 0<=self.cooldown_minutes<=10080 or self.max_gap_seconds not in (600,3600):
            raise ValueError('Cooldown must be 0..10080 minutes; cadence must match the model')

    def to_dict(self):return dict(asdict(self),version=POLICY_VERSION)


def advance(previous,source_time,result,policy,quality_ok=True):
    state=dict(previous)
    active=bool(state.get('active'))
    last=datetime.fromisoformat(state['last_time']) if state.get('last_time') else None
    if last and source_time<=last:raise ValueError('Alert clock must be strictly chronological')
    gap=result.get('status')=='data_gap' or bool(last and (source_time-last).total_seconds()>policy.max_gap_seconds)
    valid=quality_ok and result.get('status')=='ok' and result.get('predicted_class') in (0,1)
    state['last_time']=str(source_time)
    if gap or not valid:
        state.update(active=False,positive_count=0,negative_count=0,first_signal=None)
        if active:
            state.update(resolution_reason='data_gap' if gap else 'data_quality',cooldown_until=str(source_time+timedelta(minutes=policy.cooldown_minutes)))
            return state,'interrupted'
        if not valid:return state,'quality_blocked'
    if result['predicted_class']==1:
        state['negative_count']=0
        state['positive_count']=state.get('positive_count',0)+1
        state['first_signal']=state.get('first_signal') or str(source_time)
        if state.get('active'):return state,'updated'
        cooldown=datetime.fromisoformat(state['cooldown_until']) if state.get('cooldown_until') else None
        if state['positive_count']>=policy.activate_points and (cooldown is None or source_time>=cooldown):
            state.update(active=True,activated_at=str(source_time),resolution_reason=None)
            return state,'activated'
        return state,'pending'
    state.update(positive_count=0,first_signal=None)
    state['negative_count']=state.get('negative_count',0)+1
    if state.get('active'):
        if state['negative_count']>=policy.resolve_points:
            state.update(active=False,resolution_reason='negative_confirmation',cooldown_until=str(source_time+timedelta(minutes=policy.cooldown_minutes)))
            return state,'resolved'
        return state,'updated'
    return state,'quiet'
