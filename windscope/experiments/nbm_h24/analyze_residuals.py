"""Descriptive post-evaluation tradeoffs. Does not select or change operating points."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nbm import operational_metrics

out=Path('/output');report=json.loads((out/'metrics.json').read_text());curves=[];summaries=[]
for fold in report['folds']:
    asset=fold['test_asset'];validation=pd.read_csv(out/f'fold_{asset}/validation_scores.csv')
    validation.timestamp=pd.to_datetime(validation.timestamp,utc=True)
    episodes=[{key:e[key] for key in ('id','source_event','turbine_id','failure_at','group_status')} for e in fold['validation']['events']]
    for threshold in np.unique(validation.loc[validation.label.notna(),'score'].dropna()):
        metrics=operational_metrics(validation,float(threshold),episodes)
        curves.append(dict(test_asset=asset,threshold=float(threshold),onset_event_recall=metrics['onset_event_recall'],window_event_recall=metrics['window_event_recall'],
            normal_warning_fraction=metrics['normal_warning_fraction'],false_alarms_per_720_normal_hours=metrics['false_alarms_per_720_normal_hours']))
    test=pd.read_csv(out/f'fold_{asset}/test_predictions.csv')
    for label in (0,1):
        sample=test.loc[test.label.eq(label)]
        for target in fold['calibration']:
            name=target.removesuffix('__now')
            z=sample[name+'__normalized_residual'].dropna()
            residual=sample[name+'__residual'].dropna()
            summaries.append(dict(asset=asset,label=label,target=name,rows=len(z),residual_median_celsius=float(residual.median()),normalized_median=float(z.median()),normalized_q95=float(z.quantile(.95))))
curves=pd.DataFrame(curves);curves.to_csv(out/'validation_threshold_tradeoffs.csv',index=False)
pd.DataFrame(summaries).to_csv(out/'residual_distributions.csv',index=False)
summary=[]
for asset,g in curves.groupby('test_asset'):
    onset=g.loc[g.onset_event_recall.gt(0)]
    duration=g.loc[g.normal_warning_fraction.le(.05)]
    frequency=g.loc[g.false_alarms_per_720_normal_hours.le(2.)]
    both=duration.loc[duration.false_alarms_per_720_normal_hours.le(2.)]
    summary.append(dict(test_asset=str(asset),max_validation_onset_recall_without_budgets=float(g.onset_event_recall.max()),
        min_normal_warning_fraction_for_any_onset=float(onset.normal_warning_fraction.min()) if len(onset) else None,
        max_recall_with_duration_only=float(duration.onset_event_recall.max()) if len(duration) else 0.,
        max_recall_with_frequency_only=float(frequency.onset_event_recall.max()) if len(frequency) else 0.,
        max_recall_with_both=float(both.onset_event_recall.max()) if len(both) else 0.))
(out/'threshold_diagnosis.json').write_text(json.dumps(dict(scope='validation-only descriptive tradeoffs; not retuning',folds=summary),indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
