import json,sys
from pathlib import Path
import pandas as pd
import numpy as np
root=Path('/baseline')
m=json.loads((root/'metrics.json').read_text());d=json.loads((root/'diagnostics.json').read_text())
p=pd.read_csv(root/'prediction_scores.csv');p['timestamp']=pd.to_datetime(p.timestamp,utc=True)
channels=['gearbox_bearing_temperature','gearbox_oil_temperature','generator_bearing_temperature','hydraulic_oil_temperature']
result=[]
for e in d['events']:
    source=e['event'];f=pd.Timestamp(next(x['failure_at'] for x in m['metrics']['events'] if x['source_event']==source))
    w=p.loc[p.source_event.eq(source)&p.timestamp.ge(f-pd.Timedelta(hours=24))&p.timestamp.lt(f)].copy()
    row=dict(event=source,description=e['historical_description'],hit=e['onset_hit'],scheduled_hours=len(w),available=int(w.score.notna().sum()),max_score_ratio=float((w.score/w.threshold).max()),channels={})
    for c in channels:
        z=w[c+'__normalized_residual'];a=w[c+'__actual']
        row['channels'][c]=dict(z_median=float(z.median()),z_max=float(z.max()),z_min=float(z.min()),actual_min=float(a.min()),actual_max=float(a.max()),max_1h_rise=float(z.diff().max()))
    result.append(row)
normal=sorted(d['normal_sources'],key=lambda x:x['false_alarm_episodes'],reverse=True)
drift=[]
for n in normal[:4]:
    source=n['event'];cal=pd.read_csv(root/f'event_{source}/historical_calibration_scores.csv');test=p.loc[p.source_event.eq(source)&p.label.eq(0)]
    ftr=pd.read_parquet(root/f'features/{source}.parquet');calf=ftr.loc[ftr.role.eq('calibration')&ftr.normal];testf=ftr.loc[ftr.role.eq('test')&ftr.label.eq(0)]
    stats={}
    for c in channels:
        col=c+'__normalized_residual';stats[c]=dict(cal_median=float(cal.loc[cal.normal,col].median()),cal_q95=float(cal.loc[cal.normal,col].quantile(.95)),normal_test_median=float(test[col].median()),normal_test_q95=float(test[col].quantile(.95)))
    regime={col:dict(cal_median=float(calf[col].median()),test_median=float(testf[col].median()),cal_min=float(calf[col].min()),cal_max=float(calf[col].max()),test_min=float(testf[col].min()),test_max=float(testf[col].max())) for col in ['grid_active_power__now','ambient_temperature__now','wind_speed__now']}
    drift.append(dict(**n,channels=stats,regime=regime))
output=dict(scope='retrospective error analysis, does not tune or alter baseline',events=result,normal_drift=drift)
Path('/output/baseline_error_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(events=[{k:v for k,v in r.items() if k!='channels'} for r in result],largest_normal_source={k:v for k,v in drift[0].items() if k not in ('channels','regime')}),indent=2))
