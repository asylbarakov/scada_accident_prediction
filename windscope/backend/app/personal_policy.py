"""A watch is an operational incident, not a claim that risk remains high."""
from datetime import datetime,timedelta


def advance_watch(previous,source_time,result):
    state=dict(previous)
    if state.get('last_time') and source_time<=datetime.fromisoformat(state['last_time']):raise ValueError('Watch clock must be chronological')
    state['last_time']=str(source_time)
    positive=result.get('status')=='ok' and result.get('predicted_class')==1
    if positive:
        active=bool(state.get('active'))
        state.update(active=True,first_signal=state.get('first_signal') if active else str(source_time),
            activated_at=state.get('activated_at') if active else str(source_time),last_signal=str(source_time),watch_until=str(source_time+timedelta(hours=24)),resolution_reason=None)
        return state,'updated' if active else 'activated'
    if state.get('active') and source_time>=datetime.fromisoformat(state['watch_until']):
        state.update(active=False,resolution_reason='watch_window_elapsed')
        return state,'resolved'
    return state,'updated' if state.get('active') else 'quiet'
