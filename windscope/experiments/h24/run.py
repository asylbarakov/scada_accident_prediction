"""Offline H24 experiment. All research inputs must be mounted read-only."""
import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from scada_pdm.preparation import feature_metadata
from core import label_event,reconcile,fold_assets,evaluate,choose_threshold,selection_key
from pipeline import physical_features,assert_lineage_separation,fit_candidate

RESEARCH=Path('/research')
RAW=RESEARCH/'data/CARE_To_Compare/Wind Farm A'
CODE=Path(__file__).parent


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save(value,path):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare(out,protocol):
    old=RESEARCH/'results/care_h7_windows_20261116/preflight.json'
    verified=json.loads(old.read_text())
    cache=RESEARCH/'cache/care_h7_v6'
    integrity={}
    for name,expected in verified['raw_source_sha256'].items():
        actual=digest(RAW/name)
        if actual!=expected:raise ValueError(f'Raw source changed: {name}')
        integrity['raw/'+name]=actual
    for name in ('coverage.json','observation_lineage.parquet'):
        actual=digest(cache/name)
        if actual!=verified['cache_sha256'][name]:raise ValueError(f'Cache changed: {name}')
        integrity['cache/'+name]=actual
    coverage=json.loads((cache/'coverage.json').read_text())
    lineage=pd.read_parquet(cache/'observation_lineage.parquet')
    lineage['turbine_id']=lineage.turbine_id.astype(str)
    assert_lineage_separation(lineage)
    metadata=pd.read_csv(RAW/'event_info.csv',sep=';',dtype={'asset':'string'})
    for key in ('event_start','event_end'):metadata[key]=pd.to_datetime(metadata[key],utc=True)
    inherited=json.loads((RESEARCH/'configs/care_h7.json').read_text())
    feature_config={key:protocol[key] for key in ('cadence_minutes','measurement_delay_steps','windows_hours','required_history_hours')}
    feature_config.update(sensors=inherited['sensors'],operation=None)
    features=list(feature_metadata(feature_config))
    groups=coverage['fault_groups']
    group_for={e:g for g in groups for e in g['events']}
    episodes=[]
    for group in groups:
        if group['anchor_event'] is None:raise ValueError('Unresolved canonical episode')
        row=metadata.loc[metadata.event_id.eq(group['anchor_event'])].iloc[0]
        episodes.append(dict(id=group['id'],source_event=int(row.event_id),turbine_id=str(row.asset),
            failure_at=row.event_end.isoformat(),group_status=group['status']))
    all_frames=[];causal_checks=[]
    for row in metadata.sort_values('event_id').itertuples():
        event=int(row.event_id)
        columns={name:info['column'] for name,info in feature_config['sensors'].items()}
        raw=pd.read_csv(RAW/f'datasets/{event}.csv',sep=';',float_precision='round_trip',
            usecols=['time_stamp','asset_id','train_test',*columns.values()],dtype={'asset_id':'string'})
        raw['timestamp']=pd.to_datetime(raw.time_stamp,utc=True)
        if not raw.timestamp.is_monotonic_increasing or raw.timestamp.duplicated().any():raise ValueError('Invalid chronology')
        # Only 24 hours of past context are needed; each source remains a separate timeline.
        prediction_start=raw.loc[raw.train_test.eq('prediction'),'timestamp'].min()
        raw=raw.loc[raw.timestamp.ge(prediction_start-pd.Timedelta(hours=24))].reset_index(drop=True)
        physical=raw[list(columns.values())].rename(columns={v:k for k,v in columns.items()}).astype('float32').replace([np.inf,-np.inf],np.nan)
        physical.insert(0,'timestamp',raw.timestamp);physical.insert(0,'turbine_id',raw.asset_id.astype(str))
        feature_frame,ready=physical_features(physical,feature_config)
        scheduled=raw.train_test.eq('prediction')&raw.timestamp.eq(raw.timestamp.dt.floor('h'))
        candidate=raw.index[scheduled&ready]
        if len(candidate):
            # Real-data prefix checks: future telemetry is omitted, not masked after inference.
            for index in (candidate[0],candidate[-1]):
                prefix,prefix_ready=physical_features(physical.iloc[:index+1],feature_config)
                np.testing.assert_allclose(prefix.iloc[-1],feature_frame.loc[index],rtol=0,atol=0,equal_nan=True)
                assert prefix_ready.iloc[-1]==ready.loc[index]
                causal_checks.append(dict(event=event,timestamp=raw.timestamp[index].isoformat()))
        pred=raw.index[raw.train_test.eq('prediction')]
        identity=lineage.loc[lineage.source_event.eq(event)].set_index('timestamp')
        observed=pd.Series(identity.loc[pd.DatetimeIndex(raw.loc[pred,'timestamp']),'avg_fingerprint'].notna().to_numpy(),index=pred)
        fault=group_for.get(event)
        aliases=[a for a in coverage['fault_anchor_aliases'] if a['in_event']==event]
        anchors=metadata.loc[metadata.asset.eq(row.asset)&metadata.event_label.eq('anomaly'),'event_end'].tolist()
        anchors.extend(pd.Timestamp(a['timestamp']) for a in aliases)
        competing=[pd.Timestamp(a['timestamp']) for a in aliases if a['partition']=='prediction' and group_for.get(a['anchor_event'])!=fault]
        labels=label_event(raw.loc[pred,'timestamp'],observed,row.event_label,row.event_start,row.event_end,
            fault['id'] if fault else None,bool(fault and fault['anchor_event']==event),anchors,competing)
        frame=pd.concat([physical.loc[pred,['turbine_id','timestamp']],labels,feature_frame.loc[pred]],axis=1)
        frame['source_event']=event;frame['history_ready']=ready.loc[pred]
        frame['avg_fingerprint']=identity.loc[pd.DatetimeIndex(frame.timestamp),'avg_fingerprint'].to_numpy()
        all_frames.append(frame)
        print(f'prepared event={event} scheduled={int(scheduled.sum())}',flush=True)
        del feature_frame,physical,raw;gc.collect()
    frame=reconcile(pd.concat(all_frames,ignore_index=True))
    # Reconcile at native 10-minute cadence before selecting the hourly decision schedule.
    frame=frame.loc[frame.timestamp.eq(frame.timestamp.dt.floor('h'))].reset_index(drop=True)
    known=frame.loc[frame.label.notna()]
    if known.duplicated(['turbine_id','timestamp']).any():raise ValueError('Duplicate labelled keys')
    gates=[]
    for test in protocol['assets']:
        split=fold_assets(protocol['assets'],test)
        train=frame.loc[frame.turbine_id.isin(split['train'])&frame.history_ready&frame.label.notna()]
        val=frame.loc[frame.turbine_id.isin(split['validation'])&frame.history_ready]
        counts=train.label.value_counts()
        negative=int(val.label.eq(0).sum());positive=val.loc[val.label.eq(1),'episode'].nunique()
        passed=all(int(counts.get(i,0))>=protocol['gate']['min_train_rows_per_class'] for i in (0,1)) and negative>=protocol['gate']['min_validation_negative_hours'] and positive>=protocol['gate']['min_validation_positive_episodes']
        gates.append(dict(test=test,split=split,training_counts={str(k):int(v) for k,v in counts.items()},validation_negative_hours=negative,validation_episodes=positive,passed=bool(passed)))
    gate=dict(status='passed' if all(g['passed'] for g in gates) else 'failed',protocol_sha256=digest(CODE/'protocol.json'),
        inputs_sha256=integrity,raw_rows=len(lineage),cross_turbine_physical_reuse=0,canonical_faults=episodes,
        features=features,feature_config=feature_config,causal_prefix_checks=causal_checks,
        folds=gates,reason_counts={str(k):int(v) for k,v in frame.reason.value_counts().items()},
        per_asset=[dict(asset=str(asset),scheduled_hours=len(g),available_hours=int(g.history_ready.sum()),
            ready_positive_hours=int((g.history_ready&g.label.eq(1)).sum()),ready_negative_hours=int((g.history_ready&g.label.eq(0)).sum())) for asset,g in frame.groupby('turbine_id')])
    frame.to_parquet(out/'data.parquet',index=False)
    save(gate,out/'data_gate.json');save(protocol,out/'protocol.json')
    if gate['status']!='passed':raise ValueError('Data gate failed; no fitting permitted')
    return gate


def train(out,protocol):
    gate=json.loads((out/'data_gate.json').read_text())
    if gate['status']!='passed' or gate['protocol_sha256']!=digest(CODE/'protocol.json'):raise ValueError('Gate/protocol mismatch')
    frame=pd.read_parquet(out/'data.parquet');features=gate['features'];episodes=gate['canonical_faults']
    folds=[];selected_oof=[];candidate_oof={name:[] for name in protocol['models']};reload_checks=[]
    for test in protocol['assets']:
        split=fold_assets(protocol['assets'],test)
        val_episodes=[e for e in episodes if e['turbine_id'] in split['validation']]
        test_episodes=[e for e in episodes if e['turbine_id']==test]
        results={};predictions={}
        for name in protocol['models']:
            model=fit_candidate(name,protocol,frame,features,split['train'])
            subset=frame.loc[frame.turbine_id.isin(split['validation']+split['test'])].copy()
            subset['score']=np.nan;good=subset.index[subset.history_ready]
            subset.loc[good,'score']=model.predict_proba(subset.loc[good,features])[:,1]
            val=subset.loc[subset.turbine_id.isin(split['validation'])]
            threshold,val_metrics=choose_threshold(val,val_episodes,protocol['false_alarm_budget_per_720_negative_hours'])
            destination=out/f'fold_{test}'/name;destination.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,destination/'pipeline.joblib')
            reloaded=joblib.load(destination/'pipeline.joblib')
            np.testing.assert_allclose(reloaded.predict_proba(subset.loc[good[:10],features])[:,1],subset.loc[good[:10],'score'],rtol=0,atol=1e-14)
            reload_checks.append(dict(test_asset=test,model=name,passed=True))
            scored=subset.loc[subset.turbine_id.eq(test)].copy()
            metrics=evaluate(scored,threshold,test_episodes)
            inference=dict(experiment_id=protocol['experiment_id'],model=name,test_asset=test,training_assets=split['train'],validation_assets=split['validation'],
                threshold=threshold,alerts_disabled=threshold>1,features=features,feature_config=gate['feature_config'],
                horizon_hours=24,score_semantics='uncalibrated_binary_score_for_annotated_failure_within_24h',pipeline_sha256=digest(destination/'pipeline.joblib'))
            save(inference,destination/'inference.json')
            results[name]=dict(threshold=threshold,alerts_disabled=threshold>1,validation=val_metrics,test=metrics)
            # All prediction rows, including unavailable/unknown, remain inspectable.
            public=[c for c in scored if c not in features+['avg_fingerprint']]
            predictions[name]=scored[public].copy()
            predictions[name]['threshold']=threshold;predictions[name]['warning']=predictions[name].score.ge(threshold)&predictions[name].history_ready
            predictions[name]['model']=name;predictions[name]['outer_test_asset']=test
            predictions[name].to_csv(destination/'test_predictions.csv',index=False)
            candidate_oof[name].append(predictions[name])
            print(f'fold={test} model={name} val_hits={val_metrics["detected_events"]}/{val_metrics["total_events"]} test_hits={metrics["detected_events"]}/{metrics["total_events"]}',flush=True)
        # Selection depends exclusively on inner validation metrics, before comparing outer scores.
        chosen=max(results,key=lambda name:selection_key(results[name]['validation']))
        selected_oof.append(predictions[chosen]);folds.append(dict(test_asset=test,split=split,selected=chosen,candidates=results))
    selected=pd.concat(selected_oof,ignore_index=True)
    # A fixed .5 operating point on already selected decisions combines fold-specific thresholds.
    combined=evaluate(selected.assign(score=selected.warning.astype(float).where(selected.history_ready)),.5,episodes)
    # Ranking metrics must use original continuous OOF scores; scores are from different folds.
    rank=evaluate(selected,.5,episodes)
    combined['average_precision']=rank['average_precision'];combined['roc_auc']=rank['roc_auc']
    combined['ranking_metric_caveat']='pooled scores from different fitted models; operating decisions use per-fold validation thresholds'
    baselines={name:evaluate(selected.assign(score=float(value)),.5,episodes) for name,value in [('never_warn',0),('always_warn',1)]}
    candidate_summary={}
    for name,frames in candidate_oof.items():
        data=pd.concat(frames,ignore_index=True)
        candidate_summary[name]=evaluate(data.assign(score=data.warning.astype(float).where(data.history_ready)),.5,episodes)
    selected.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(combined['events']).to_csv(out/'fault_results.csv',index=False)
    report=dict(experiment_id=protocol['experiment_id'],scope='nested whole-turbine cross-validation; exploratory internal validation, not external prospective validation',
        selected_strategy=combined,folds=folds,candidates=candidate_summary,trivial_baselines=baselines,reload_checks=reload_checks,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scikit_learn=sklearn.__version__,joblib=joblib.__version__),
        code_sha256={p.name:digest(p) for p in CODE.glob('*.py')},protocol_sha256=digest(CODE/'protocol.json'),data_sha256=digest(out/'data.parquet'),
        production_integration=False,calibrated_probability=False,future_sensor_forecasting=False)
    save(report,out/'metrics.json')
    print(json.dumps({k:v for k,v in combined.items() if k!='events'},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train']);parser.add_argument('--output',default='/output');args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    protocol=json.loads((CODE/'protocol.json').read_text())
    if args.stage=='prepare':
        if (out/'data.parquet').exists():raise FileExistsError('Choose a new output directory')
        gate=prepare(out,protocol)
        print(json.dumps({k:gate[k] for k in ('status','per_asset','folds')},indent=2),flush=True)
    else:
        if (out/'metrics.json').exists():raise FileExistsError('Training results already exist; choose a fresh run')
        train(out,protocol)


if __name__=='__main__':main()
