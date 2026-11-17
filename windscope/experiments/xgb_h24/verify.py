"""Independent native-bundle replay, training-row and protocol checks; no fitting."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append('/evaluation')
from nbm import operational_metrics,select_threshold
from model import NativeAdapter,digest
from run import verify_sources,candidate_key,predict_points

out=Path('/output');code=Path('/experiment')
metrics=json.loads((out/'metrics.json').read_text());config=json.loads((code/'protocol.json').read_text());gate=json.loads((out/'data_gate.json').read_text())
verify_sources()
assert metrics['protocol_sha256']==gate['protocol_sha256']==digest(code/'protocol.json')
assert metrics['data_sha256']==gate['data_sha256']==digest(out/'data.parquet')
assert metrics['evaluator_sha256']==digest('/evaluation/nbm.py')
for name in ('model.py','run.py'):assert digest(code/name)==metrics['code_sha256'][name]
frame=pd.read_parquet(out/'data.parquet')
selected=pd.read_csv(out/'oof_predictions.csv',dtype={'turbine_id':'string'});selected.timestamp=pd.to_datetime(selected.timestamp,utc=True)
checked=0;single_checks=0;maximum_error=0.;fit_occurrences=0
for fold in metrics['folds']:
    asset=fold['test_asset'];validation_metrics={}
    for candidate in config['candidates']:
        directory=out/f'fold_{asset}'/candidate
        adapter=NativeAdapter(directory);manifest=adapter.metadata
        assert asset not in manifest['split']['train']+manifest['split']['validation']
        train=frame.loc[frame.turbine_id.isin(manifest['split']['train'])&frame['ready_'+candidate]&frame.label.notna()]
        fit_keys=pd.read_csv(directory/'fit_rows.csv',dtype={'turbine_id':'string'});fit_keys.timestamp=pd.to_datetime(fit_keys.timestamp,utc=True)
        assert len(train)==len(fit_keys)==manifest['fit_info']['training_rows']
        assert set(fit_keys.turbine_id)==set(manifest['split']['train'])
        joined=fit_keys.merge(train[['turbine_id','source_event','timestamp','label']],on=['turbine_id','source_event','timestamp'],validate='one_to_one')
        assert len(joined)==len(fit_keys) and joined.label.isin([0,1]).all()
        np.testing.assert_array_equal(train[manifest['features']].astype('float32').min().to_numpy(),json.loads((directory/'minmax.json').read_text())['data_min'])
        np.testing.assert_array_equal(train[manifest['features']].astype('float32').max().to_numpy(),json.loads((directory/'minmax.json').read_text())['data_max'])
        assert manifest['fit_info']['scale_pos_weight']==train.label.eq(0).sum()/train.label.eq(1).sum()
        fit_occurrences+=len(train)
        val=frame.loc[frame.turbine_id.isin(manifest['split']['validation'])]
        val_points=predict_points(val,candidate,adapter)
        episodes=[e for e in gate['canonical_faults'] if e['turbine_id'] in manifest['split']['validation']]
        threshold,val_metrics=select_threshold(val_points,episodes,config)
        assert threshold==manifest['threshold']
        assert val_metrics['normal_warning_fraction']<=config['max_normal_warning_fraction']
        assert val_metrics['false_alarms_per_720_normal_hours']<=config['false_alarm_budget_per_720_normal_hours']
        validation_metrics[candidate]=val_metrics
        rows=frame.loc[frame.turbine_id.eq(asset)]
        reproduced=predict_points(rows,candidate,adapter)
        saved=pd.read_csv(directory/'test_predictions.csv',dtype={'turbine_id':'string'});saved.timestamp=pd.to_datetime(saved.timestamp,utc=True)
        assert reproduced[['source_event','timestamp']].reset_index(drop=True).equals(saved[['source_event','timestamp']].reset_index(drop=True))
        np.testing.assert_allclose(reproduced.score,saved.score,rtol=0,atol=1e-12,equal_nan=True)
        assert np.array_equal(reproduced.warning,saved.warning)
        good=rows.index[rows['ready_'+candidate]]
        if len(good):
            errors=np.abs(reproduced.score.to_numpy()-saved.score.to_numpy())
            maximum_error=max(maximum_error,float(np.nanmax(errors)))
            for index in [good[0],good[len(good)//2],good[-1]]:
                score=adapter.predict(rows.loc[[index]])['score'].iloc[0]
                assert score==reproduced.loc[index,'score'];single_checks+=1
        checked+=len(good)
    assert max(config['candidates'],key=lambda name:candidate_key(validation_metrics[name]))==fold['selected']
aggregate=operational_metrics(selected,None,gate['canonical_faults'],decisions=selected.warning)
for key in ('onset_hit_events','window_hit_events','normal_warning_hours','available_hours','tp','fn','fp','tn'):
    assert aggregate[key]==metrics['selected'][key],key
result=dict(status='passed',native_bundles_checked=10,available_test_scores_reproduced_across_candidates=checked,
    single_measurement_calls=single_checks,max_absolute_score_error=maximum_error,training_occurrences_checked=fit_occurrences,
    raw_seven_feature_prefix_checks=len(gate['seven_feature_prefix_checks']),upstream_history_prefix_checks=gate['upstream_raw_prefix_checks'],
    original_artifact_unchanged=True,original_pickle_loaded=False,fitting_performed=False,thresholds_changed=False)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
