"""Offline annotation comparison only. This module is not imported by inference."""
from datetime import datetime


def summarize(records, annotation):
    records = list(records)
    valid = [r for r in records if r['status'] == 'ok']
    start = datetime.fromisoformat(annotation['event_start'])
    end = datetime.fromisoformat(annotation['event_end'])
    anomaly = annotation['event_label'] == 'anomaly'
    warnings = []
    episodes = []
    confusion = dict(tn=0, fp=0, fn=0, tp=0)
    previous = None
    previous_warning = False
    before = during = after = 0
    for r in records:
        timestamp = datetime.fromisoformat(r['source_time'])
        positive = r['status'] == 'ok' and r['predicted_class'] == 1
        if positive:
            warnings.append(timestamp)
            if anomaly:
                if timestamp < start:
                    before += 1
                elif timestamp <= end:
                    during += 1
                else:
                    after += 1
            if not previous_warning or (timestamp - previous).total_seconds() != 600:
                episodes.append(dict(start=r['source_time'], end=r['source_time'], points=1))
            else:
                episodes[-1]['end'] = r['source_time']
                episodes[-1]['points'] += 1
        if r['status'] == 'ok':
            actual = anomaly and start <= timestamp <= end
            confusion[('t' if positive == actual else 'f') + ('p' if positive else 'n')] += 1
        previous, previous_warning = timestamp, positive
    first = min(warnings) if warnings else None
    lead = (start-first).total_seconds()/60 if anomaly and first is not None and first < start else None
    tp, fp, fn, tn = [confusion[k] for k in ('tp', 'fp', 'fn', 'tn')]
    return dict(rows=len(records), valid_rows=len(valid), invalid_rows=len(records)-len(valid),
                source_start=records[0]['source_time'] if records else None,
                source_end=records[-1]['source_time'] if records else None,
                warning_points=len(warnings), warning_fraction=len(warnings)/len(valid) if valid else None,
                warning_episodes=len(episodes), episodes=episodes,
                first_warning_time=first.isoformat(sep=' ') if first else None,
                lead_to_annotation_start_minutes=lead,
                warnings_before_interval=before if anomaly else None,
                warnings_during_interval=during if anomaly else None,
                warnings_after_interval=after if anomaly else None,
                normal_false_alert_points_proxy=len(warnings) if not anomaly else None,
                confusion=confusion,
                annotation_precision=tp/(tp+fp) if tp+fp else None,
                annotation_recall=tp/(tp+fn) if tp+fn else None,
                annotation_accuracy=(tp+tn)/len(valid) if valid else None,
                metric_semantics='pointwise annotation agreement; not independent model validation')
