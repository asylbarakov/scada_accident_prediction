"""Post-evaluation signal attribution and false-episode analysis, no tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

out=Path('/output');report=json.loads((out/'metrics.json').read_text());results={}
for name,entry in report['variants'].items():
    data=pd.read_csv(out/f'{name}_predictions.csv');data.timestamp=pd.to_datetime(data.timestamp,utc=True)
    data=data.sort_values(['source_event','timestamp'])
    continuous=data.source_event.eq(data.source_event.shift())&data.timestamp.diff().eq(pd.Timedelta(hours=1))
    known_normal=data.label.eq(0)&data.score.notna()&data.history_ready
    false=known_normal&data.warning
    starts=false&~(false.shift(fill_value=False)&continuous)
    normal=[]
    for event,points in data.groupby('source_event'):
        mask=data.source_event.eq(event);hours=int((known_normal&mask).sum())
        if hours:normal.append(dict(event=int(event),normal_hours=hours,false_episodes=int((starts&mask).sum()),warning_hours=int((false&mask).sum())))
    hits=[]
    for event in entry['metrics']['events']:
        if not event['onset_hit']:continue
        at=pd.Timestamp(event['first_observed_onset']);fault=pd.Timestamp(event['failure_at'])
        group=data.loc[data.source_event.eq(event['source_event'])]
        window=group.loc[group.timestamp.between(at-pd.Timedelta(hours=2),at)].copy()
        manifest=json.loads((out/name/f'event_{event["source_event"]}'/'normal_model.json').read_text());config=manifest['config']
        channels=[t.removesuffix('__now') for t in config['targets']]
        values=window[[c+'__normalized_residual' for c in channels]].to_numpy()
        for i,target in enumerate(config['targets']):
            if target in config.get('two_sided_targets',[]):values[:,i]=np.abs(values[:,i])
        positive=np.maximum(np.nan_to_num(values,nan=-np.inf),0);winners=positive.argmax(axis=1);maxima=positive.max(axis=1)
        contributions={c:float(maxima[winners==i].sum()/3) for i,c in enumerate(channels)}
        np.testing.assert_allclose(sum(contributions.values()),window.score.iloc[-1],rtol=0,atol=1e-12)
        latest=window.iloc[-1];evidence=[dict(channel=c,actual=float(latest[c+'__actual']),expected=float(latest[c+'__expected']),normalized_residual=float(latest[c+'__normalized_residual'])) for c in channels]
        target=group.loc[group.timestamp.ge(fault-pd.Timedelta(hours=24))&group.timestamp.lt(fault)]
        active=target.loc[target.warning]
        hits.append(dict(event=event['source_event'],first_h24_onset=str(at),lead_hours=event['first_onset_lead_hours'],threshold=float(latest.threshold),score=float(latest.score),
            score_contributions=contributions,current_channel_evidence=evidence,
            warning_hours_last6=int((target.timestamp.ge(fault-pd.Timedelta(hours=6))&target.warning).sum()),
            available_hours_last6=int((target.timestamp.ge(fault-pd.Timedelta(hours=6))&target.score.notna()).sum()),
            last_warning_lead_hours=float((fault-active.timestamp.max()).total_seconds()/3600)))
    results[name]=dict(normal_sources=sorted(normal,key=lambda r:r['false_episodes'],reverse=True),hits=hits)
(out/'diagnostics.json').write_text(json.dumps(dict(scope='post-evaluation attribution; channel contribution is not fault diagnosis; no threshold changes',variants=results),indent=2,allow_nan=False)+'\n')
rows=[]
for event in report['baseline_metrics']['events']:
    row=dict(event=event['source_event'],baseline_lead_hours=event['first_onset_lead_hours'])
    for name,entry in report['variants'].items():row[name+'_lead_hours']=next(e['first_onset_lead_hours'] for e in entry['metrics']['events'] if e['id']==event['id'])
    rows.append(row)
pd.DataFrame(rows).to_csv(out/'event_comparison.csv',index=False)
print('Saved fixed-result attribution and event comparison.')
