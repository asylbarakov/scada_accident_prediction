"""Explain frozen source warnings offline; never alters models or thresholds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

out=Path('/output');report=json.loads((out/'metrics.json').read_text());gate=json.loads((out/'data_gate.json').read_text())
metadata=pd.read_csv('/research/data/CARE_To_Compare/Wind Farm A/event_info.csv',sep=';')
data=pd.read_csv(out/'prediction_scores.csv');data.timestamp=pd.to_datetime(data.timestamp,utc=True)
channels=['gearbox_bearing_temperature','gearbox_oil_temperature','generator_bearing_temperature','hydraulic_oil_temperature']
events=[]
for event in report['metrics']['events']:
    description=metadata.loc[metadata.event_id.eq(event['source_event']),'event_description'].iloc[0]
    entry=dict(event=event['source_event'],asset=event['turbine_id'],historical_description=description,onset_hit=event['onset_hit'],lead_hours=event['first_onset_lead_hours'])
    fault=pd.Timestamp(event['failure_at'])
    target=data.loc[data.source_event.eq(event['source_event'])&data.timestamp.ge(fault-pd.Timedelta(hours=24))&data.timestamp.lt(fault)].sort_values('timestamp')
    active=target.loc[target.warning.eq(True)]
    available=target.loc[target.score.notna()]
    entry.update(last_warning_lead_hours=float((fault-active.timestamp.max()).total_seconds()/3600) if len(active) else None,
        warning_hours_last6=int((target.timestamp.ge(fault-pd.Timedelta(hours=6))&target.warning.eq(True)).sum()),
        available_hours_last6=int((target.timestamp.ge(fault-pd.Timedelta(hours=6))&target.score.notna()).sum()),
        last_available_warning=bool(available.warning.iloc[-1]) if len(available) else None,
        last_available_lead_hours=float((fault-available.timestamp.max()).total_seconds()/3600) if len(available) else None)
    if event['onset_hit']:
        at=pd.Timestamp(event['first_observed_onset'])
        window=data.loc[data.source_event.eq(event['source_event'])&data.timestamp.between(at-pd.Timedelta(hours=2),at)].sort_values('timestamp')
        residuals=window[[c+'__normalized_residual' for c in channels]].to_numpy()
        positive=np.maximum(np.nan_to_num(residuals,nan=-np.inf),0)
        winners=positive.argmax(axis=1);maxima=positive.max(axis=1)
        contributions={channel:float(maxima[winners==i].sum()/3) for i,channel in enumerate(channels)}
        actual_score=float(window.score.iloc[-1])
        if len(window)==3:np.testing.assert_allclose(sum(contributions.values()),actual_score,rtol=0,atol=1e-12)
        entry.update(first_onset=at.isoformat(),trigger_score=actual_score,threshold=float(window.threshold.iloc[-1]),
            observed_score_window_hours=len(window),channel_contributions=contributions,largest_signal_channel=max(contributions,key=contributions.get))
    events.append(entry)
normal_sources=[dict(event=s['event'],asset=s['asset'],normal_hours=s['test']['known_normal_hours'],warning_hours=s['test']['normal_warning_hours'],
    warning_fraction=s['test']['normal_warning_fraction'],false_alarm_episodes=s['test']['false_alarm_episodes'],
    false_alarms_per_720_normal_hours=s['test']['false_alarms_per_720_normal_hours']) for s in report['sources'] if s['test']['known_normal_hours']]
components=[]
for description,g in pd.DataFrame(events).groupby('historical_description'):
    components.append(dict(historical_description=description,episodes=len(g),detected=int(g.onset_hit.sum())))
result=dict(scope='post-evaluation diagnostics; historical description is not model fault classification',events=events,by_historical_component=components,
    normal_sources=normal_sources,total_training_row_occurrences=sum(s['training_normal_hours'] for s in gate['sources']),
    total_calibration_row_occurrences=sum(s['calibration_normal_hours'] for s in gate['sources']),
    independent_turbines=5,models_are_source_specific=True)
(out/'diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(by_historical_component=components,detected_events=[e for e in events if e['onset_hit']]),indent=2))
