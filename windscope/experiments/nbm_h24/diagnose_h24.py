"""Read-only diagnosis of the frozen H24 experiment; no fitting or tuning."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

old=Path('/previous');out=Path('/output')
metrics=json.loads((old/'metrics.json').read_text())
data=old/'data.parquet'
assert hashlib.sha256(data.read_bytes()).hexdigest()==metrics['data_sha256']
frame=pd.read_parquet(data)
oof=pd.read_csv(old/'oof_predictions.csv',dtype={'turbine_id':'string'})
channels=[c for c in frame if c.endswith('__now')]
stats=[];shifts=[];attributions=[]
for asset,g in frame.loc[frame.history_ready].groupby('turbine_id'):
    for label in (0,1):
        sample=g.loc[g.label.eq(label)]
        for channel in channels:
            x=sample[channel]
            stats.append(dict(asset=str(asset),label=label,feature=channel,rows=len(x),missing_fraction=float(x.isna().mean()),
                median=float(x.median()),q05=float(x.quantile(.05)),q95=float(x.quantile(.95))))
for fold in metrics['folds']:
    test=fold['test_asset'];name=fold['selected']
    normal_train=frame.loc[frame.turbine_id.isin(fold['split']['train'])&frame.history_ready&frame.label.eq(0)]
    normal_test=frame.loc[frame.turbine_id.eq(test)&frame.history_ready&frame.label.eq(0)]
    for channel in channels:
        a=normal_train[channel].dropna();b=normal_test[channel].dropna()
        low,high=a.quantile([.01,.99]);scale=max(float(a.quantile(.75)-a.quantile(.25)),1e-6)
        shifts.append(dict(test_asset=test,feature=channel,train_normal_median=float(a.median()),test_normal_median=float(b.median()),
            median_shift_in_train_iqr=float((b.median()-a.median())/scale),outside_train_01_99=float(((b<low)|(b>high)).mean())))
    destination=old/f'fold_{test}'/name
    manifest=json.loads((destination/'inference.json').read_text())
    assert hashlib.sha256((destination/'pipeline.joblib').read_bytes()).hexdigest()==manifest['pipeline_sha256']
    model=joblib.load(destination/'pipeline.joblib')
    train=frame.loc[frame.turbine_id.isin(fold['split']['train'])&frame.history_ready&frame.label.notna()]
    test_rows=frame.loc[frame.turbine_id.eq(test)&frame.history_ready&frame.label.eq(0)]
    def mean_contribution(rows):
        transformed=model.named_steps['scaler'].transform(model.named_steps['imputer'].transform(rows[manifest['features']]))
        return (transformed*model.named_steps['model'].coef_[0]).mean(axis=0)
    difference=mean_contribution(test_rows)-mean_contribution(train.loc[train.label.eq(0)])
    top=np.argsort(np.abs(difference))[-8:][::-1]
    attributions.append(dict(test_asset=test,threshold=manifest['threshold'],alerts_disabled=manifest['alerts_disabled'],
        largest_normal_logit_contribution_changes=[dict(feature=manifest['features'][i],delta=float(difference[i])) for i in top]))
pd.DataFrame(stats).to_csv(out/'h24_sensor_distributions.csv',index=False)
pd.DataFrame(shifts).to_csv(out/'h24_normal_distribution_shift.csv',index=False)
result=dict(scope='descriptive diagnosis, not causal proof',source_data_sha256=metrics['data_sha256'],
    top_normal_median_shifts=pd.DataFrame(shifts).assign(magnitude=lambda f:f.median_shift_in_train_iqr.abs()).sort_values('magnitude',ascending=False).head(12).drop(columns='magnitude').to_dict('records'),
    logistic_attribution_changes=attributions,
    validation_always_warn=[dict(test_asset=f['test_asset'],validation_asset=f['split']['validation'][0],
        positive_hours=f['candidates']['logistic']['validation']['tp']+f['candidates']['logistic']['validation']['fn'],
        normal_hours=f['candidates']['logistic']['validation']['known_negative_hours'],
        chosen_normal_warning_hours=f['candidates']['logistic']['validation']['fp'],
        chosen_false_alarm_episodes=f['candidates']['logistic']['validation']['false_warning_episodes']) for f in metrics['folds']],
    known_limitations=['Previous threshold rule limits alarm starts but not warning duration.',
        'No external unseen turbines are present in this cached experiment.',
        'Distribution shifts and logit changes do not establish sensor-unit errors or physical failure causes.'])
(out/'h24_error_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result['top_normal_median_shifts'][:5],indent=2))
