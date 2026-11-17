"""Prepare and fit new H24 XGBoost models; never load or modify research pickle."""
import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost
sys.path.append('/evaluation')
from nbm import operational_metrics,select_threshold
from model import RESEARCH_FEATURES,PHASES,digest,seven_features,fit_xgb,export_bundle,NativeAdapter

CODE=Path(__file__).parent
PREVIOUS=Path('/previous')
RESEARCH=Path('/research')
ORIGINAL_SHA='98584786a6a4421f74e599743378c67aadaeaa67ffbfdd582a14f5d235617fd8'


def save(value,path):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def verify_sources():
    previous=json.loads((PREVIOUS/'metrics.json').read_text())
    gate=json.loads((PREVIOUS/'data_gate.json').read_text())
    assert digest(PREVIOUS/'data.parquet')==previous['data_sha256']
    assert digest(Path('/previous-code/protocol.json'))==previous['protocol_sha256']==gate['protocol_sha256']
    assert json.loads((PREVIOUS/'protocol.json').read_text())==json.loads(Path('/previous-code/protocol.json').read_text())
    for key,expected in gate['inputs_sha256'].items():
        category,name=key.split('/',1)
        source=RESEARCH/('data/CARE_To_Compare/Wind Farm A' if category=='raw' else 'cache/care_h7_v6')/name
        assert digest(source)==expected,key
    assert digest('/original/model.pkl')==ORIGINAL_SHA
    return previous,gate


def prepare(out,config):
    previous,gate=verify_sources()
    if config['research_features']!=RESEARCH_FEATURES or config['voltage_sources']!=PHASES:raise ValueError('Research schema changed')
    frame=pd.read_parquet(PREVIOUS/'data.parquet')
    raw_dir=RESEARCH/'data/CARE_To_Compare/Wind Farm A/datasets'
    appended=[];prefix_checks=[]
    for event in sorted(frame.source_event.unique()):
        raw=pd.read_csv(raw_dir/f'{event}.csv',sep=';',float_precision='round_trip',dtype={'asset_id':'string'},
            usecols=['time_stamp','asset_id','train_test',*[n for n in RESEARCH_FEATURES if n!='voltage'],*PHASES])
        raw['timestamp']=pd.to_datetime(raw.time_stamp,utc=True)
        features=seven_features(raw)
        scheduled=raw.train_test.eq('prediction')&raw.timestamp.eq(raw.timestamp.dt.floor('h'))
        indices=raw.index[scheduled]
        for index in [indices[0],indices[-1]]:
            prefix=seven_features(raw.loc[:index])
            np.testing.assert_allclose(prefix.loc[index],features.loc[index],rtol=0,atol=0,equal_nan=True)
            prefix_checks.append(dict(event=int(event),timestamp=raw.timestamp[index].isoformat()))
        selected=features.loc[indices].copy()
        selected['source_event']=event;selected['timestamp']=raw.loc[indices,'timestamp'];selected['turbine_id']=raw.loc[indices,'asset_id']
        appended.append(selected)
    new=pd.concat(appended,ignore_index=True)
    keys=['turbine_id','source_event','timestamp']
    frame=frame.merge(new,on=keys,how='left',validate='one_to_one',indicator=True)
    assert frame['_merge'].eq('both').all() and len(frame)==len(new)
    frame=frame.drop(columns='_merge')
    assert len(gate['features'])==160
    frame['ready_research7']=frame.history_ready&frame[RESEARCH_FEATURES].notna().all(axis=1)
    frame['ready_history160']=frame.history_ready
    features=dict(research7=RESEARCH_FEATURES,history160=gate['features'])
    checks=[]
    for fold in previous['folds']:
        for candidate in config['candidates']:
            train=frame.loc[frame.turbine_id.isin(fold['split']['train'])&frame['ready_'+candidate]&frame.label.notna()]
            val=frame.loc[frame.turbine_id.isin(fold['split']['validation'])&frame['ready_'+candidate]]
            counts=train.label.value_counts()
            passed=all(int(counts.get(label,0))>=config['gate']['minimum_train_rows_per_class'] for label in (0,1)) and int(val.label.eq(0).sum())>=config['gate']['minimum_validation_negative_hours']
            checks.append(dict(test_asset=fold['test_asset'],candidate=candidate,training_counts={str(k):int(v) for k,v in counts.items()},
                validation_negative_hours=int(val.label.eq(0).sum()),passed=bool(passed)))
    frame.to_parquet(out/'data.parquet',index=False)
    data_gate=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',protocol_sha256=digest(CODE/'protocol.json'),
        evaluator_sha256=digest('/evaluation/nbm.py'),input_data_sha256=previous['data_sha256'],data_sha256=digest(out/'data.parquet'),
        original_artifact_sha256=ORIGINAL_SHA,original_inputs_sha256=gate['inputs_sha256'],features=features,folds=previous['folds'],
        canonical_faults=gate['canonical_faults'],fold_checks=checks,seven_feature_prefix_checks=prefix_checks,
        upstream_raw_prefix_checks=len(gate['causal_prefix_checks']),physical_feature_config=gate['feature_config'],
        availability=[dict(candidate=c,scheduled=len(frame),ready=int(frame['ready_'+c].sum()),known_positive=int((frame['ready_'+c]&frame.label.eq(1)).sum()),known_normal=int((frame['ready_'+c]&frame.label.eq(0)).sum())) for c in config['candidates']])
    save(data_gate,out/'data_gate.json');save(config,out/'protocol.json')
    if data_gate['status']!='passed':raise ValueError('Data gate failed; no fitting')
    print(json.dumps(dict(status=data_gate['status'],availability=data_gate['availability']),indent=2),flush=True)


def candidate_key(metrics):
    return (metrics['onset_event_recall'],-metrics['normal_warning_fraction'],-metrics['false_alarms_per_720_normal_hours'],metrics['point_precision'] or 0.)


def predict_points(frame,candidate,adapter):
    names=['turbine_id','source_event','timestamp','label','reason','episode','failure_at']
    points=frame[names].copy()
    points['history_ready']=frame['ready_'+candidate]
    points['score']=np.nan
    valid=points.index[points.history_ready]
    if len(valid):points.loc[valid,'score']=adapter.predict(frame.loc[valid])['score']
    threshold=adapter.metadata['threshold']
    points['threshold']=threshold
    points['warning']=False if threshold is None else points.score.ge(threshold)&points.history_ready
    return points


def train(out,config):
    previous,source_gate=verify_sources()
    gate=json.loads((out/'data_gate.json').read_text())
    assert gate['status']=='passed' and gate['data_sha256']==digest(out/'data.parquet')
    assert gate['protocol_sha256']==digest(CODE/'protocol.json') and gate['evaluator_sha256']==digest('/evaluation/nbm.py')
    frame=pd.read_parquet(out/'data.parquet');episodes=gate['canonical_faults']
    folds=[];selected_frames=[];candidate_frames={name:[] for name in config['candidates']};reload_checks=[]
    for fold in gate['folds']:
        test=fold['test_asset'];split=fold['split'];validation={};adapters={}
        assert split['test']==[test] and not set(split['train'])&set(split['test']+split['validation'])
        val_episodes=[e for e in episodes if e['turbine_id'] in split['validation']]
        for candidate in config['candidates']:
            names=gate['features'][candidate]
            training=frame.copy();training['history_ready']=frame['ready_'+candidate]
            model,scaler,fit_info=fit_xgb(training,names,split['train'],config['xgboost'])
            val=frame.loc[frame.turbine_id.isin(split['validation'])]
            temporary=val[['turbine_id','source_event','timestamp','label','reason','episode','failure_at']].copy()
            temporary['history_ready']=val['ready_'+candidate];temporary['score']=np.nan
            good=temporary.index[temporary.history_ready]
            temporary.loc[good,'score']=model.predict_proba(scaler.transform(val.loc[good,names].astype('float32')))[:,1]
            threshold,val_metrics=select_threshold(temporary,val_episodes,config)
            destination=out/f'fold_{test}'/candidate
            export_bundle(model,scaler,names,threshold,dict(candidate=candidate,split=split,fit_info=fit_info,
                prediction_horizon_hours=24,measurement_delay_steps=1,score_semantics=config['score_semantics'],
                missing_policy='all_required' if candidate=='research7' else 'xgboost_nan'),destination)
            adapter=NativeAdapter(destination)
            regenerated=adapter.predict(val.loc[good])['score']
            np.testing.assert_array_equal(regenerated.to_numpy(),temporary.loc[good,'score'].to_numpy())
            reload_checks.append(dict(test_asset=test,candidate=candidate,rows=len(good),max_error=0.))
            temporary.to_csv(destination/'validation_scores.csv',index=False)
            fitted=training.loc[training.turbine_id.isin(split['train'])&training.history_ready&training.label.notna(),['turbine_id','source_event','timestamp']]
            fitted.to_csv(destination/'fit_rows.csv',index=False)
            adapters[candidate]=adapter;validation[candidate]=dict(threshold=threshold,alerts_disabled=threshold is None,metrics=val_metrics,fit_info=fit_info)
        # No outer scores were used to choose the feature set or thresholds.
        selected=max(config['candidates'],key=lambda name:candidate_key(validation[name]['metrics']))
        test_rows=frame.loc[frame.turbine_id.eq(test)]
        test_episodes=[e for e in episodes if e['turbine_id']==test]
        results={}
        for candidate in config['candidates']:
            adapter=adapters[candidate];points=predict_points(test_rows,candidate,adapter)
            points['candidate']=candidate;points['outer_test_asset']=test
            results[candidate]=dict(validation=validation[candidate],test=operational_metrics(points,adapter.metadata['threshold'],test_episodes))
            points.to_csv(out/f'fold_{test}'/candidate/'test_predictions.csv',index=False)
            candidate_frames[candidate].append(points)
            if candidate==selected:selected_frames.append(points)
            print(f'fold={test} candidate={candidate} threshold={adapter.metadata["threshold"]} val_onsets={validation[candidate]["metrics"]["onset_hit_events"]}/{len(val_episodes)} test_onsets={results[candidate]["test"]["onset_hit_events"]}/{len(test_episodes)}',flush=True)
        folds.append(dict(test_asset=test,split=split,selected=selected,candidates=results))
    selected=pd.concat(selected_frames,ignore_index=True)
    combined=operational_metrics(selected,None,episodes,decisions=selected.warning)
    candidates={}
    for name,parts in candidate_frames.items():
        data=pd.concat(parts,ignore_index=True)
        candidates[name]=operational_metrics(data,None,episodes,decisions=data.warning)
        data.to_csv(out/f'{name}_oof_predictions.csv',index=False)
    old=pd.read_csv(PREVIOUS/'oof_predictions.csv',dtype={'turbine_id':'string'});old.timestamp=pd.to_datetime(old.timestamp,utc=True)
    old_metrics=operational_metrics(old,None,episodes,decisions=old.warning)
    selected.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(combined['events']).to_csv(out/'fault_results.csv',index=False)
    report=dict(experiment_id=config['experiment_id'],evaluation_status=config['evaluation_status'],protocol_sha256=digest(CODE/'protocol.json'),
        data_sha256=digest(out/'data.parquet'),evaluator_sha256=gate['evaluator_sha256'],code_sha256={p.name:digest(p) for p in CODE.glob('*.py')},
        selected=combined,candidates=candidates,folds=folds,previous_lr_same_metrics=old_metrics,reload_checks=reload_checks,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__),
        original_artifact_unchanged=digest('/original/model.pkl')==ORIGINAL_SHA,original_artifact_loaded=False,
        production_integrated=False,calibrated_failure_probability=False,future_sensor_forecasting=False)
    save(report,out/'metrics.json')
    print(json.dumps({key:value for key,value in combined.items() if key!='events'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train']);parser.add_argument('--output',default='/output');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((CODE/'protocol.json').read_text())
    if args.stage=='prepare':
        if (out/'data.parquet').exists():raise FileExistsError('Use a fresh output directory')
        prepare(out,config)
    else:
        if (out/'metrics.json').exists() or any(out.glob('fold_*')):raise FileExistsError('Do not overwrite fitting outputs')
        train(out,config)
