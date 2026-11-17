"""Post-evaluation comparison on common available decisions; never selects a model."""
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.append('/evaluation')
from nbm import operational_metrics

out=Path('/output');baseline=Path('/baseline')
report=json.loads((out/'metrics.json').read_text())
episodes=json.loads((out/'data_gate.json').read_text())['canonical_faults']
old=pd.read_csv(baseline/'prediction_scores.csv',dtype={'turbine_id':str})
new=pd.read_csv(out/'oof_predictions.csv',dtype={'turbine_id':str})
keys=['turbine_id','source_event','timestamp']
for data in (old,new):data.timestamp=pd.to_datetime(data.timestamp,utc=True)
joined=old[keys+['score','history_ready']].merge(new[keys+['score','history_ready']],on=keys,suffixes=('_base','_candidate'),validate='one_to_one')
joined['common']=joined.score_base.notna()&joined.score_candidate.notna()&joined.history_ready_base&joined.history_ready_candidate
results={}
for name,data in [('baseline',old),('candidate',new)]:
    data=data.merge(joined[keys+['common']],on=keys,validate='one_to_one')
    data['history_ready']&=data.common
    data['score']=data.score.where(data.common)
    results[name]=operational_metrics(data,None,episodes,decisions=data.warning)
save=dict(scope='post-evaluation diagnostic on identical available timestamps; no threshold changes or promotion decision tuning',common_hours=int(joined.common.sum()),results=results)
(out/'common_exposure.json').write_text(json.dumps(save,indent=2,allow_nan=False)+'\n')
rows=[]
audit=json.loads((out/'baseline_error_audit.json').read_text())
for event in audit['events']:
    candidate=next(e for e in report['metrics']['events'] if e['source_event']==event['event'])
    base=next(e for e in report['baseline_metrics']['events'] if e['source_event']==event['event'])
    rows.append(dict(event=event['event'],historical_description=event['description'],baseline_available_h24=event['available'],baseline_max_score_ratio=event['max_score_ratio'],baseline_hit=base['onset_hit'],baseline_lead_hours=base['first_onset_lead_hours'],candidate_hit=candidate['onset_hit'],candidate_lead_hours=candidate['first_onset_lead_hours'],candidate_available_h24=candidate['available_target_hours']))
pd.DataFrame(rows).to_csv(out/'event_comparison.csv',index=False)
print(json.dumps(dict(common_hours=save['common_hours'],results={n:{k:r[k] for k in ['onset_hit_events','false_alarm_episodes','known_normal_hours','false_alarms_per_720_normal_hours']} for n,r in results.items()}),indent=2))
