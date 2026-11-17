"""One frozen H24 forecast-head experiment; reads baseline bundles, writes a new run."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit

sys.path.extend(['/personal-code','/feature-code','/evaluation'])
from personal import PersonalModel,digest
from core import fold_assets
from pipeline import assert_lineage_separation
from nbm import operational_metrics,select_threshold
from dynamics import dynamic_features,feature_names,fit_head

CODE=Path(__file__).parent
BASE=Path('/baseline')
OUT=Path('/output')


def save(value,path):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def prepare(config):
    if (OUT/'data_gate.json').exists():raise FileExistsError('Choose a fresh output directory')
    source_gate=json.loads((BASE/'data_gate.json').read_text())
    report=json.loads((BASE/'metrics.json').read_text())
    if digest('/personal-code/personal.py')!=report['code_sha256']['personal.py']:raise ValueError('Frozen regressor implementation changed')
    prior=Path('/previous/data.parquet')
    if digest(prior)!=source_gate['prior_label_data_sha256']:raise ValueError('Reviewed labels changed')
    original=pd.read_parquet(prior)
    original['turbine_id']=original.turbine_id.astype(str)
    lineage_path=Path('/research/cache/care_h7_v6/observation_lineage.parquet')
    if digest(lineage_path)!=source_gate['input_sha256']['cache/observation_lineage.parquet']:raise ValueError('Physical lineage changed')
    lineage=pd.read_parquet(lineage_path,columns=['turbine_id','avg_fingerprint','full_fingerprint'])
    assert_lineage_separation(lineage)
    del lineage
    keys=['turbine_id','source_event','timestamp']
    labels=['label','reason','episode','failure_at']
    frames=[];checks=[];sources=[]
    for source in source_gate['sources']:
        event=source['event'];path=BASE/f'features/{event}.parquet'
        if digest(path)!=source['features_sha256']:raise ValueError('Historical feature cache changed')
        frame=pd.read_parquet(path);frame['turbine_id']=frame.turbine_id.astype(str)
        bundle=BASE/f'event_{event}'
        inference=json.loads((bundle/'inference.json').read_text())
        if digest(bundle/'normal_model.json')!=inference['normal_model_manifest_sha256']:raise ValueError('Baseline manifest changed')
        model=PersonalModel.load(bundle)
        residuals=model.predict(frame)
        physical=residuals.join(frame[[n+'__now' for n in ['wind_speed','ambient_temperature','grid_active_power','generator_speed','rotor_speed']]])
        dynamics=dynamic_features(physical)
        names=feature_names(dynamics)
        test_indices=frame.index[frame.role.eq('test')]
        for stop in [test_indices[0]+1,test_indices[-1]+1]:
            prefix=dynamic_features(physical.iloc[:stop])
            pd.testing.assert_frame_equal(prefix,dynamics.iloc[:stop],check_exact=False,rtol=0,atol=1e-12)
            checks.append(dict(event=event,stop_at=str(physical.timestamp.iloc[stop-1]),passed=True))
        test=dynamics.loc[frame.role.eq('test')].copy()
        truth=original.loc[original.source_event.eq(event),keys+labels]
        test=test.merge(truth,on=keys,how='left',validate='one_to_one',indicator=True)
        if len(test)!=len(truth) or not test['_merge'].eq('both').all():raise ValueError('Canonical label keys differ')
        test=test.drop(columns='_merge')
        frames.append(test)
        sources.append(dict(event=event,feature_cache_sha256=digest(path),normal_manifest_sha256=digest(bundle/'normal_model.json'),scheduled_hours=len(test),available_hours=int(test.history_ready.sum())))
        print(f'prepared dynamics source={event} ready={int(test.history_ready.sum())}/{len(test)}',flush=True)
    combined=pd.concat(frames,ignore_index=True)
    known=combined.loc[combined.label.notna()]
    if known.duplicated(['turbine_id','timestamp']).any():raise ValueError('Duplicate labelled physical decisions')
    combined.to_parquet(OUT/'dynamic_features.parquet',index=False)
    gate=dict(status='passed',protocol_sha256=digest(CODE/'protocol.json'),baseline_metrics_sha256=digest(BASE/'metrics.json'),
        baseline_gate_sha256=digest(BASE/'data_gate.json'),canonical_label_data_sha256=digest(prior),
        features=names,canonical_faults=source_gate['canonical_faults'],sources=sources,real_prefix_checks=checks,
        cached_raw_prefix_checks_inherited=len(source_gate['raw_prefix_checks']),cross_turbine_physical_reuse=0,
        data_sha256=digest(OUT/'dynamic_features.parquet'),test_labels_used_for_feature_construction=False)
    save(gate,OUT/'data_gate.json');save(config,OUT/'protocol.json')
    return gate


def native_head(model,features):
    return dict(features=features,imputer=model['imputer'].statistics_.tolist(),mean=model['scaler'].mean_.tolist(),
        scale=model['scaler'].scale_.tolist(),coefficients=model['model'].coef_[0].tolist(),intercept=float(model['model'].intercept_[0]),classes=model['model'].classes_.tolist())


def predict_native(head,frame):
    values=frame[head['features']].to_numpy(dtype=float)
    values=np.where(np.isnan(values),np.array(head['imputer']),values)
    return expit(((values-np.array(head['mean']))/np.array(head['scale']))@np.array(head['coefficients'])+head['intercept'])


def train(config):
    if (OUT/'metrics.json').exists():raise FileExistsError('Do not overwrite evaluated results')
    gate=json.loads((OUT/'data_gate.json').read_text())
    if digest(CODE/'protocol.json')!=gate['protocol_sha256'] or digest(OUT/'dynamic_features.parquet')!=gate['data_sha256']:raise ValueError('Protocol or prepared data changed')
    frame=pd.read_parquet(OUT/'dynamic_features.parquet');features=gate['features'];episodes=gate['canonical_faults']
    folds=[];predictions=[];fit_audit=[]
    for test_asset in config['assets']:
        split=fold_assets(config['assets'],test_asset)
        model,keys=fit_head(frame,features,split['train'],config['head'])
        if set(keys.turbine_id)&set(split['validation']+split['test']):raise ValueError('Asset leakage')
        validation=frame.loc[frame.turbine_id.isin(split['validation'])].copy()
        test=frame.loc[frame.turbine_id.isin(split['test'])].copy()
        val_events=[e for e in episodes if e['turbine_id'] in split['validation']]
        test_events=[e for e in episodes if e['turbine_id'] in split['test']]
        if validation.loc[validation.label.eq(1),'episode'].nunique()<1 or (validation.history_ready&validation.label.eq(0)).sum()<24:raise ValueError('Insufficient validation events/exposure')
        if min(frame.loc[frame.turbine_id.isin(split['train'])&frame.history_ready,'label'].value_counts().get(c,0) for c in (0,1))<20:raise ValueError('Insufficient training class counts')
        head=native_head(model,features)
        for subset in (validation,test):
            subset['score']=np.nan;ready=subset.history_ready
            values=model.predict_proba(subset.loc[ready,features])[:,1]
            subset.loc[ready,'score']=values
            np.testing.assert_allclose(predict_native(head,subset.loc[ready]),values,atol=1e-12,rtol=0)
        threshold,val_metrics=select_threshold(validation,val_events,config)
        test['threshold']=threshold
        test['warning']=False if threshold is None else test.score.ge(threshold)&test.history_ready
        test['outer_test_asset']=test_asset
        metrics=operational_metrics(test,None,test_events,decisions=test.warning)
        destination=OUT/f'fold_{test_asset}';destination.mkdir()
        save(head,destination/'head.json')
        save(dict(split=split,threshold=threshold,alerts_disabled=threshold is None,horizon_hours=24,score_semantics=config['score_semantics']),destination/'inference.json')
        keys.to_csv(destination/'fit_rows.csv',index=False)
        public=[c for c in test if c not in features]
        test[public].to_csv(destination/'predictions.csv',index=False)
        predictions.append(test[public])
        fit_audit.append(dict(test_asset=test_asset,training_assets=sorted(keys.turbine_id.unique().tolist()),training_rows=len(keys),validation_assets=split['validation'],test_assets=split['test'],native_reload_max_error_bound=1e-12))
        folds.append(dict(test_asset=test_asset,split=split,threshold=threshold,alerts_disabled=threshold is None,validation=val_metrics,test=metrics))
        print(f'fold={test_asset} threshold={threshold} validation_hits={val_metrics["onset_hit_events"]}/{len(val_events)} test_hits={metrics["onset_hit_events"]}/{len(test_events)}',flush=True)
    combined=pd.concat(predictions,ignore_index=True)
    metrics=operational_metrics(combined,None,episodes,decisions=combined.warning)
    baseline=json.loads((BASE/'metrics.json').read_text())['metrics']
    promoted=(metrics['onset_hit_events']>baseline['onset_hit_events'] and metrics['false_alarms_per_720_normal_hours']<=config['false_alarm_budget_per_720_normal_hours'] and metrics['normal_warning_fraction']<=config['max_normal_warning_fraction'] and metrics['availability']>=baseline['availability']-.01)
    combined.to_csv(OUT/'oof_predictions.csv',index=False)
    pd.DataFrame(metrics['events']).to_csv(OUT/'fault_results.csv',index=False)
    result=dict(experiment_id=config['experiment_id'],scope=config['scope'],metrics=metrics,baseline_metrics=baseline,folds=folds,fit_audit=fit_audit,
        protocol_sha256=digest(CODE/'protocol.json'),data_sha256=gate['data_sha256'],code_sha256={p.name:digest(p) for p in CODE.glob('*.py')},
        versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,sklearn=sklearn.__version__),
        promotion_gate_passed=bool(promoted),production_integrated=False,independent_external_validation=False,
        probability_calibrated=False,future_sensor_trajectory=False,
        comparison_caveat='head supervision, feature representation and threshold selection all change; this is not an isolated feature ablation; pooled ranking scores come from distinct outer folds')
    save(result,OUT/'metrics.json')
    print(json.dumps(dict(candidate={k:v for k,v in metrics.items() if k!='events'},promotion_gate_passed=bool(promoted)),indent=2),flush=True)


def verify():
    report=json.loads((OUT/'metrics.json').read_text());gate=json.loads((OUT/'data_gate.json').read_text())
    if digest(CODE/'protocol.json')!=report['protocol_sha256']:raise ValueError('Protocol changed after evaluation')
    for name,expected in report['code_sha256'].items():
        if digest(CODE/name)!=expected:raise ValueError('Code changed after evaluation')
    frame=pd.read_parquet(OUT/'dynamic_features.parquet');stored=pd.read_csv(OUT/'oof_predictions.csv',dtype={'turbine_id':str,'outer_test_asset':str})
    stored['timestamp']=pd.to_datetime(stored.timestamp,utc=True)
    comparisons=[]
    for fold in report['folds']:
        asset=fold['test_asset'];path=OUT/f'fold_{asset}'
        head=json.loads((path/'head.json').read_text());test=frame.loc[frame.turbine_id.eq(asset)].copy()
        test['score']=np.nan;ready=test.history_ready
        test.loc[ready,'score']=predict_native(head,test.loc[ready])
        original=stored.loc[stored.outer_test_asset.eq(asset)].sort_values(['source_event','timestamp'])
        test=test.sort_values(['source_event','timestamp'])
        np.testing.assert_allclose(test.score,original.score,atol=1e-12,rtol=0,equal_nan=True)
        keys=pd.read_csv(path/'fit_rows.csv',dtype={'turbine_id':str})
        assert set(keys.turbine_id)==set(fold['split']['train'])
        warning=pd.Series(False,index=test.index) if fold['threshold'] is None else test.score.ge(fold['threshold'])&test.history_ready
        assert np.array_equal(warning.to_numpy(),original.warning.to_numpy())
        comparisons.append(dict(test_asset=asset,rows=len(test),scored_rows=int(ready.sum()),native_replay_passed=True))
    save(dict(status='passed',folds=comparisons,real_prefix_checks=len(gate['real_prefix_checks']),
        no_supervised_test_asset_in_fitting=True,unknown_labels_excluded_from_fitting=True,protocol_unchanged=True,
        checks_do_not_establish_predictive_accuracy=True),OUT/'verification.json')
    print('Independent native-head replay verified for all five folds.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','verify']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);config=json.loads((CODE/'protocol.json').read_text())
    if args.stage=='prepare':prepare(config)
    elif args.stage=='train':train(config)
    else:verify()
