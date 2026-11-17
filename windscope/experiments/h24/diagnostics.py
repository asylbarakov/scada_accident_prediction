"""Post-evaluation diagnostics only. Never changes models, selection or thresholds."""
import json
from pathlib import Path
import pandas as pd


def summarize(points,episodes):
    points=points.sort_values(['source_event','timestamp']).copy()
    points['active']=points.history_ready&points.score.notna()&points.score.ge(points.threshold)
    rows=[]
    for asset,g in points.groupby('turbine_id'):
        negative=g.label.eq(0).fillna(False)&g.history_ready&g.score.notna()
        positive=g.label.eq(1).fillna(False)
        rows.append(dict(asset=str(asset),known_negative_hours=int(negative.sum()),normal_warning_hours=int((negative&g.active).sum()),
            normal_warning_fraction=float((negative&g.active).sum()/negative.sum()) if negative.sum() else None,
            target_scheduled_hours=int(positive.sum()),target_available_hours=int((positive&g.history_ready&g.score.notna()).sum()),
            unknown_warning_hours=int((g.label.isna()&g.active).sum())))
    events=[]
    for episode in episodes:
        fault=pd.Timestamp(episode['failure_at'])
        g=points.loc[points.source_event.eq(episode['source_event'])&points.timestamp.lt(fault)]
        hit=g.loc[g.active]
        target_hit=hit.loc[hit.timestamp.ge(fault-pd.Timedelta(hours=24))]
        first=hit.timestamp.min() if len(hit) else None
        target_first=target_hit.timestamp.min() if len(target_hit) else None
        events.append(dict(**episode,first_source_warning=first.isoformat() if first is not None else None,
            first_source_warning_lead_hours=float((fault-first).total_seconds()/3600) if first is not None else None,
            warning_before_target_window=bool(hit.timestamp.lt(fault-pd.Timedelta(hours=24)).any()),
            first_target_warning=target_first.isoformat() if target_first is not None else None,
            first_target_warning_lead_hours=float((fault-target_first).total_seconds()/3600) if target_first is not None else None))
    return dict(scope='diagnostics only; target-window coverage is not proof of a new timely alarm',per_asset=rows,events=events)


if __name__=='__main__':
    out=Path('/output');metrics=json.loads((out/'metrics.json').read_text())
    points=pd.read_csv(out/'oof_predictions.csv');points['timestamp']=pd.to_datetime(points.timestamp,utc=True)
    result=summarize(points,metrics['selected_strategy']['events'])
    (out/'operational_diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
