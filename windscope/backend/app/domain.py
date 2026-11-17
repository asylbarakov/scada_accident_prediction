"""Replay time and visibility policies, independent of database and HTTP."""
from datetime import datetime, timedelta, timezone


def utc_now():
    return datetime.now(timezone.utc)


def aware(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def utc_iso(value):
    return aware(value).astimezone(timezone.utc).isoformat() if value is not None else None


def remaining_source(due, now, speed):
    return max(0, (aware(due)-aware(now)).total_seconds()*speed) if due else 0


def next_due(now, remaining, speed):
    if not 1 <= speed <= 36000:
        raise ValueError('Speed must be between 1 and 36000')
    return now+timedelta(seconds=max(0,remaining)/speed)


def schedule_after(previous_due, previous_source, next_source, speed):
    return next_due(previous_due,(next_source-previous_source).total_seconds(),speed)


def observable_annotation(annotation, cutoff):
    if cutoff is None or datetime.fromisoformat(annotation['start']) > cutoff:
        return None
    result=dict(annotation)
    if datetime.fromisoformat(annotation['end']) > cutoff:
        result['end']=None
        result['visible_until']=cutoff.isoformat(sep=' ')
    else:
        result['visible_until']=annotation['end']
    return result


def model_state(result):
    if not result or result.get('status') != 'ok':
        return 'unknown'
    return 'warning' if result.get('predicted_class') == 1 else 'normal'


def display_unit(source_unit):
    # Preserve source metadata; repair only the two unambiguous degree units.
    return {'\ufffdC':'°C','\ufffd':'°'}.get(source_unit,source_unit)
