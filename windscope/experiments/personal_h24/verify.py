"""Verify historical boundaries and native source models without refitting."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append('/evaluation')
from nbm import operational_metrics
from personal import PersonalModel,normal_threshold,digest
from run import verify_sources

out=Path('/output');code=Path('/experiment');report=json.loads((out/'metrics.json').read_text());gate=json.loads((out/'data_gate.json').read_text())
verify_sources();assert digest(code/'protocol.json')==report['protocol_sha256']==gate['protocol_sha256']
for name in ('personal.py','run.py'):assert digest(code/name)==report['code_sha256'][name]
assert digest('/evaluation/nbm.py')==gate['evaluator_sha256']
config=json.loads((code/'protocol.json').read_text())
checked=0;prefix_checks=0;fit_rows=0;calibration_rows=0;maximum_error=0.
for source in gate['sources']:
    event=source['event'];features=out/f'features/{event}.parquet';assert digest(features)==source['features_sha256']
    frame=pd.read_parquet(features)
    if not source['eligible']:continue
    directory=out/f'event_{event}';manifest=json.loads((directory/'inference.json').read_text())
    assert digest(directory/'normal_model.json')==manifest['normal_model_manifest_sha256']
    model=PersonalModel.load(directory)
    for name,role in [('fit_rows.csv','train'),('calibration_rows.csv','calibration')]:
        keys=pd.read_csv(directory/name,dtype={'turbine_id':'string'});keys.timestamp=pd.to_datetime(keys.timestamp,utc=True)
        assert keys.source_event.eq(event).all() and keys.turbine_id.eq(source['asset']).all()
        matched=keys.merge(frame,on=['turbine_id','source_event','timestamp'],validate='one_to_one')
        assert len(matched)==len(keys) and matched.normal.all() and matched.role.eq(role).all()
        assert matched.source_partition.eq('train').all() and matched.history_ready.all()
        followup=matched.timestamp+pd.Timedelta(hours=config['normal_followup_hours'])
        assert followup.lt(pd.Timestamp(source['bounds']['prediction_start'])).all()
        if role=='train':assert followup.le(pd.Timestamp(source['bounds']['boundary'])).all();fit_rows+=len(keys)
        else:calibration_rows+=len(keys)
    scored=model.predict(frame)
    calibration=scored.join(frame[['normal','role']]).loc[frame.role.eq('calibration')]
    threshold,normal_metrics=normal_threshold(calibration,config)
    assert threshold==manifest['threshold']
    assert normal_metrics['normal_warning_fraction']<=config['max_normal_warning_fraction']
    assert normal_metrics['false_alarms_per_720_normal_hours']<=config['false_alarm_budget_per_720_normal_hours']
    test=scored.loc[frame.role.eq('test')]
    saved=pd.read_csv(directory/'prediction_scores.csv');saved.timestamp=pd.to_datetime(saved.timestamp,utc=True)
    assert test.timestamp.reset_index(drop=True).equals(saved.timestamp.reset_index(drop=True))
    np.testing.assert_allclose(test.score,saved.score,rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_array_equal(test.score.ge(threshold),saved.warning)
    error=np.abs(test.score.to_numpy()-saved.score.to_numpy())
    if test.score.notna().any():maximum_error=max(maximum_error,float(np.nanmax(error)))
    checked+=int(test.score.notna().sum())
    available=test.index[test.score.notna()]
    if len(available):
        for index in [available[0],available[-1]]:
            past=model.predict(frame.loc[:index])
            assert past.loc[index,'score']==scored.loc[index,'score'];prefix_checks+=1
saved=pd.read_csv(out/'prediction_scores.csv');saved.timestamp=pd.to_datetime(saved.timestamp,utc=True)
recomputed=operational_metrics(saved,None,gate['canonical_faults'],decisions=saved.warning)
for key in ('onset_hit_events','window_hit_events','normal_warning_hours','available_hours','unavailable_hours','tp','fn','fp','tn'):
    assert recomputed[key]==report['metrics'][key],key
result=dict(status='passed',source_models_verified=sum(s['eligible'] for s in gate['sources']),native_regressors_verified=4*sum(s['eligible'] for s in gate['sources']),
    prediction_scores_reproduced=checked,max_absolute_error=maximum_error,training_rows_checked=fit_rows,calibration_rows_checked=calibration_rows,
    raw_feature_prefix_checks=len(gate['raw_prefix_checks']),model_test_prefix_checks=prefix_checks,
    only_same_source_past_used=True,prediction_partition_used_for_fitting=False,threshold_uses_fault_labels=False,fitting_performed=False)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
