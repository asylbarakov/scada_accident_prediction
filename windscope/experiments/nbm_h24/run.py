"""Frozen exploratory NBM experiment; never alters H24 or research inputs."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from nbm import NormalBehaviourModel,operational_metrics,select_threshold

CODE=Path(__file__).parent
PREVIOUS=Path('/previous')
RESEARCH=Path('/research')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save(value,path):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def verify_protocol_copy(source,copy,expected_sha256):
    if digest(source)!=expected_sha256 or json.loads(Path(source).read_text())!=json.loads(Path(copy).read_text()):
        raise ValueError('Previous protocol source or archived values changed')


def input_gate(config,out):
    prior=json.loads((PREVIOUS/'metrics.json').read_text())
    gate=json.loads((PREVIOUS/'data_gate.json').read_text())
    assert digest(PREVIOUS/'data.parquet')==prior['data_sha256']
    assert prior['protocol_sha256']==gate['protocol_sha256']
    verify_protocol_copy(Path('/previous-code/protocol.json'),PREVIOUS/'protocol.json',prior['protocol_sha256'])
    for key,expected in gate['inputs_sha256'].items():
        category,name=key.split('/',1)
        path=RESEARCH/('data/CARE_To_Compare/Wind Farm A' if category=='raw' else 'cache/care_h7_v6')/name
        if digest(path)!=expected:raise ValueError(f'Source changed: {key}')
    frame=pd.read_parquet(PREVIOUS/'data.parquet')
    if sorted(frame.turbine_id.unique())!=config['assets']:raise ValueError('Unexpected turbines')
    if frame.duplicated(['source_event','timestamp']).any():raise ValueError('Duplicate source timestamps')
    assert set(config['regime_features']).isdisjoint(config['targets'])
    counts=[]
    for asset,g in frame.groupby('turbine_id'):
        normal=g.loc[g.history_ready&g.label.eq(0)&g[config['required_current_inputs']].notna().all(axis=1)]
        counts.append(dict(asset=str(asset),normal_rows=len(normal),targets={target:int(normal[target].notna().sum()) for target in config['targets']}))
    passed=all(min(row['targets'].values())>=config['minimum_normal_rows_per_target'] for row in counts)
    result=dict(status='passed' if passed else 'failed',prior_experiment='care-h24-loto-v1',
        prior_data_sha256=prior['data_sha256'],protocol_sha256=digest(CODE/'protocol.json'),
        previous_metrics_sha256=digest(PREVIOUS/'metrics.json'),previous_gate_sha256=digest(PREVIOUS/'data_gate.json'),
        original_inputs_sha256=gate['inputs_sha256'],per_asset_normal_counts=counts,
        canonical_faults=gate['canonical_faults'],physical_config=gate['feature_config'],features=gate['features'],
        prior_causal_prefix_checks=len(gate['causal_prefix_checks']),
        external_holdout_available=False,source_lineage_cross_turbine_reuse=gate['cross_turbine_physical_reuse'])
    save(result,out/'data_gate.json')
    if not passed:raise ValueError('NBM data gate failed')
    return frame,prior,gate


def run(out):
    config=json.loads((CODE/'protocol.json').read_text())
    frame,prior,gate=input_gate(config,out)
    save(config,out/'protocol.json')
    folds=[];oof=[];prefix_checks=[];reload_errors=[]
    for fold in prior['folds']:
        test=fold['test_asset'];split=fold['split']
        expected_validation=config['assets'][(config['assets'].index(test)+1)%len(config['assets'])]
        assert split['test']==[test] and split['validation']==[expected_validation]
        assert set(sum(split.values(),[]))==set(config['assets']) and len(sum(split.values(),[]))==5
        print(f'Fitting normal models for outer asset {test}',flush=True)
        model=NormalBehaviourModel(config).fit(frame,split['train'])
        subset=frame.loc[frame.turbine_id.isin(split['validation']+split['test'])]
        calculated=model.predict(subset)
        # Evaluation labels are attached only after physical-only inference.
        points=calculated.join(subset[['label','reason','episode','failure_at']])
        val=points.loc[points.turbine_id.isin(split['validation'])]
        val_episodes=[e for e in gate['canonical_faults'] if e['turbine_id'] in split['validation']]
        threshold,val_metrics=select_threshold(val,val_episodes,config)
        # Operating point frozen before computing the outer test metrics.
        test_points=points.loc[points.turbine_id.eq(test)].copy()
        test_episodes=[e for e in gate['canonical_faults'] if e['turbine_id']==test]
        test_metrics=operational_metrics(test_points,threshold,test_episodes)
        test_points['threshold']=threshold
        test_points['warning']=False if threshold is None else test_points.score.ge(threshold)&test_points.history_ready
        test_points['outer_test_asset']=test
        destination=out/f'fold_{test}';destination.mkdir()
        joblib.dump(model,destination/'normal_model.joblib')
        manifest=dict(experiment_id=config['experiment_id'],test_asset=test,split=split,threshold=threshold,alerts_disabled=threshold is None,
            score_semantics='uncalibrated_smoothed_positive_temperature_residual',warning_horizon_under_evaluation_hours=24,
            calibration=model.calibration,training_normal_counts=model.normal_counts,config=config,
            model_sha256=digest(destination/'normal_model.joblib'))
        save(manifest,destination/'inference.json')
        model.fitted_keys.to_csv(destination/'fit_rows.csv',index=False)
        val.to_csv(destination/'validation_scores.csv',index=False)
        test_points.to_csv(destination/'test_predictions.csv',index=False)
        loaded=joblib.load(destination/'normal_model.joblib')
        regenerated=loaded.predict(subset)
        np.testing.assert_allclose(regenerated.score,points.score,rtol=0,atol=0,equal_nan=True)
        reload_errors.append(dict(test_asset=test,max_error=0.))
        for event,event_rows in subset.loc[subset.turbine_id.eq(test)].groupby('source_event'):
            valid=points.loc[event_rows.index].score.dropna()
            if not len(valid):continue
            for index in [valid.index[0],valid.index[-1]]:
                past=event_rows.loc[event_rows.timestamp.le(subset.loc[index,'timestamp'])]
                sequential=model.predict(past)
                np.testing.assert_allclose(sequential.loc[index,'score'],points.loc[index,'score'],rtol=0,atol=0)
                prefix_checks.append(dict(test_asset=test,event=int(event),timestamp=subset.loc[index,'timestamp'].isoformat()))
        folds.append(dict(test_asset=test,split=split,threshold=threshold,alerts_disabled=threshold is None,
            validation=val_metrics,test=test_metrics,calibration=model.calibration,training_normal_counts=model.normal_counts))
        oof.append(test_points)
        print(f'fold={test} threshold={threshold} validation_onsets={val_metrics["onset_hit_events"]}/{val_metrics["total_events"]} test_onsets={test_metrics["onset_hit_events"]}/{test_metrics["total_events"]}',flush=True)
    combined=pd.concat(oof,ignore_index=True)
    result=operational_metrics(combined,None,gate['canonical_faults'],decisions=combined.warning)
    old=pd.read_csv(PREVIOUS/'oof_predictions.csv',dtype={'turbine_id':'string'})
    old.timestamp=pd.to_datetime(old.timestamp,utc=True)
    comparison=operational_metrics(old,None,gate['canonical_faults'],decisions=old.warning)
    baselines={name:operational_metrics(combined,None,gate['canonical_faults'],decisions=pd.Series(bool(value),index=combined.index)) for name,value in [('never_warn',0),('always_warn',1)]}
    combined.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(result['events']).to_csv(out/'fault_results.csv',index=False)
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scikit_learn=sklearn.__version__,joblib=joblib.__version__)
    report=dict(experiment_id=config['experiment_id'],evaluation_status=config['evaluation_status'],protocol_sha256=digest(CODE/'protocol.json'),
        data_sha256=digest(PREVIOUS/'data.parquet'),code_sha256={p.name:digest(p) for p in CODE.glob('*.py')},
        nbm=result,previous_lr_same_operational_metrics=comparison,trivial_baselines=baselines,folds=folds,
        prefix_checks=prefix_checks,reload_checks=reload_errors,versions=versions,
        production_integrated=False,calibrated_failure_probability=False,future_sensor_trajectory=False)
    save(report,out/'metrics.json')
    save(dict(status='not_available_locally',present_farms=['Wind Farm A'],new_independent_test_turbines=0,
        exclusions=['Same CARE files or replay cycles are not new holdout data.','EDP failure-log CSV alone is not SCADA and is not established as independent of CARE Farm A.'],
        candidate='CARE Wind Farms B/C or a separately sourced farm, after sensor schema and lineage review',
        future_gate=['new turbine lineage','documented sensor mapping and units','confirmed fault anchors','normal follow-up coverage','new frozen protocol before examining outcomes']),out/'external_validation_status.json')
    print(json.dumps({key:value for key,value in result.items() if key!='events'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='/output');args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    if (out/'metrics.json').exists() or any(out.glob('fold_*')):raise FileExistsError('Choose fresh output; partial/full fitting outputs are never overwritten')
    run(out)
