"""Read-only feasibility probe; no fitting or changes to research artifacts."""
import json
from pathlib import Path
import pandas as pd

root=Path('/research/cache/care_h7_v6')
coverage=json.loads((root/'coverage.json').read_text())
labels=pd.read_parquet(root/'labels.parquet')
times=pd.to_datetime(labels.timestamp,utc=True)
faults=pd.to_datetime(labels.h7_failure_at,utc=True)
hourly=times.eq(times.dt.floor('h'))
positive=labels.h7_label.eq(1)&times.lt(faults)&faults.le(times+pd.Timedelta(hours=24))
print('DEPENDENCY_GROUPS',json.dumps(coverage['dependency_groups']))
print('CACHED_H24_CANDIDATES')
for asset,rows in labels.groupby('turbine_id'):
    selected=rows.index
    raw=labels.loc[selected].loc[(positive&hourly).loc[selected]]
    good=raw.loc[raw.history_ready]
    negatives=labels.loc[selected].loc[(labels.h7_label.eq(0)&hourly&labels.history_ready).loc[selected]]
    print(json.dumps(dict(asset=str(asset),positive_hours=len(raw),ready_24h_positive_hours=len(good),
        episodes_raw=raw.h7_episode.dropna().nunique(),episodes_ready=good.h7_episode.dropna().nunique(),
        conservative_negative_hours=len(negatives))))
print('CANONICAL_EPISODES')
events={e['event']:e for e in coverage['event_summaries']}
for group in coverage['fault_groups']:
    anchor=group['anchor_event']
    print(json.dumps(dict(id=group['id'],events=group['events'],anchor=anchor,
        asset=events[anchor]['asset'] if anchor is not None else None,status=group['status'])))
