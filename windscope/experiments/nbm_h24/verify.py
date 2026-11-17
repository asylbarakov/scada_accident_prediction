"""Read-only model verification plus a small evidence JSON, no fitting."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from nbm import operational_metrics,select_threshold
from run import digest,input_gate

out=Path('/output');code=Path('/experiment');previous=Path('/previous')
report=json.loads((out/'metrics.json').read_text());config=json.loads((code/'protocol.json').read_text())
assert digest(code/'protocol.json')==report['protocol_sha256']
for name in ('run.py','nbm.py'):
    assert digest(code/name)==report['code_sha256'][name]
# input_gate checks all original raw/cache hashes and previous protocol values.
frame,_,gate=input_gate(config,out)
saved=pd.read_csv(out/'oof_predictions.csv',dtype={'turbine_id':'string'})
saved.timestamp=pd.to_datetime(saved.timestamp,utc=True)
checked=0;maximum_error=0.;training_rows=0
for fold in report['folds']:
    test=fold['test_asset'];directory=out/f'fold_{test}'
    manifest=json.loads((directory/'inference.json').read_text())
    assert digest(directory/'normal_model.joblib')==manifest['model_sha256']
    assert test not in manifest['split']['train']+manifest['split']['validation']
    model=joblib.load(directory/'normal_model.joblib')
    keys=model.fitted_keys
    assert set(keys.turbine_id)==set(fold['split']['train'])
    fitted=keys.merge(frame[['turbine_id','source_event','timestamp','label']],on=['turbine_id','source_event','timestamp'],validate='one_to_one')
    assert fitted.label.eq(0).all();training_rows+=len(fitted)
    subset=frame.loc[frame.turbine_id.eq(test)].sort_values(['source_event','timestamp'])
    result=model.predict(subset)
    archived=saved.loc[saved.turbine_id.eq(test)].sort_values(['source_event','timestamp'])
    assert subset[['source_event','timestamp']].reset_index(drop=True).equals(archived[['source_event','timestamp']].reset_index(drop=True))
    np.testing.assert_allclose(result.score.to_numpy(),archived.score.to_numpy(),rtol=0,atol=1e-12,equal_nan=True)
    maximum_error=max(maximum_error,float(np.abs(result.score.to_numpy()-archived.score.to_numpy())[result.score.notna()].max(initial=0)))
    checked+=int(result.score.notna().sum())
    # Validation recomputation verifies the frozen operating point and both budgets.
    val=pd.read_csv(directory/'validation_scores.csv')
    val.timestamp=pd.to_datetime(val.timestamp,utc=True)
    episodes=[e for e in gate['canonical_faults'] if e['turbine_id'] in fold['split']['validation']]
    threshold,metrics=select_threshold(val,episodes,config)
    assert threshold==fold['threshold']==manifest['threshold']
    assert metrics['normal_warning_fraction']<=config['max_normal_warning_fraction']
    assert metrics['false_alarms_per_720_normal_hours']<=config['false_alarm_budget_per_720_normal_hours']
aggregate=operational_metrics(saved,None,gate['canonical_faults'],decisions=saved.warning)
for key in ('onset_hit_events','window_hit_events','normal_warning_hours','available_hours','tp','fn','fp','tn'):
    assert aggregate[key]==report['nbm'][key],key
external=out/'external_metadata/manifest.json'
metadata_files=0
if external.exists():
    for file in json.loads(external.read_text())['files']:
        path=out/'external_metadata'/Path(file['member']).relative_to('CARE_To_Compare')
        assert digest(path)==file['sha256'];metadata_files+=1
result=dict(status='passed',reproduced_available_oof_scores=checked,max_absolute_score_error=maximum_error,
    training_rows_checked_across_folds=training_rows,training_labels_all_normal=True,
    physical_prefix_checks=len(report['prefix_checks']),upstream_raw_feature_prefix_checks=gate.get('causal_prefix_checks') and len(gate['causal_prefix_checks']),
    raw_and_cache_hashes_checked=len(gate['inputs_sha256']),external_metadata_files_verified=metadata_files,
    fitting_performed=False,thresholds_changed=False)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
