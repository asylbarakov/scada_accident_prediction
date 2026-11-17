"""Single-source personal history experiment: preparation and frozen training."""
import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost
sys.path.append('/feature-code');sys.path.append('/evaluation')
from pipeline import physical_features
from nbm import operational_metrics
from personal import PersonalModel,historical_roles,normal_threshold,digest

CODE=Path(__file__).parent;PREVIOUS=Path('/previous');RESEARCH=Path('/research')


def save(value,path):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def verify_sources():
    gate=json.loads((PREVIOUS/'data_gate.json').read_text());metrics=json.loads((PREVIOUS/'metrics.json').read_text())
    assert digest(PREVIOUS/'data.parquet')==metrics['data_sha256']
    assert digest('/feature-code/pipeline.py')==metrics['code_sha256']['pipeline.py']
    for key,expected in gate['inputs_sha256'].items():
        category,name=key.split('/',1)
        source=RESEARCH/('data/CARE_To_Compare/Wind Farm A' if category=='raw' else 'cache/care_h7_v6')/name
        assert digest(source)==expected,key
    return gate


def prepare(out,config):
    gate=verify_sources();old=pd.read_parquet(PREVIOUS/'data.parquet')
    physical_config=gate['feature_config'].copy();physical_config['windows_hours']=config['feature_windows_hours'];physical_config['required_history_hours']=config['required_history_hours']
    features_config=physical_config['sensors'];mapping={n:v['column'] for n,v in features_config.items()}
    sources=[];prefix_checks=[];total_rows=0
    (out/'features').mkdir()
    for event in sorted(old.source_event.unique()):
        raw=pd.read_csv(RESEARCH/f'data/CARE_To_Compare/Wind Farm A/datasets/{event}.csv',sep=';',float_precision='round_trip',
            usecols=['time_stamp','asset_id','train_test','status_type_id',*mapping.values()],dtype={'asset_id':'string'})
        raw['timestamp']=pd.to_datetime(raw.time_stamp,utc=True)
        scada=raw[list(mapping.values())].rename(columns={v:k for k,v in mapping.items()}).astype('float32').replace([np.inf,-np.inf],np.nan)
        scada.insert(0,'timestamp',raw.timestamp);scada.insert(0,'turbine_id',raw.asset_id)
        observed=scada[['wind_speed','ambient_temperature','grid_active_power']].notna().all(axis=1)&scada[['gearbox_bearing_temperature','gearbox_oil_temperature','generator_bearing_temperature','hydraulic_oil_temperature']].notna().sum(axis=1).ge(config['minimum_target_count'])
        roles,bounds=historical_roles(pd.DataFrame(dict(timestamp=raw.timestamp,partition=raw.train_test,status_type_id=raw.status_type_id,observed=observed)),config)
        features,ready=physical_features(scada,physical_config)
        indices=raw.index[raw.timestamp.eq(raw.timestamp.dt.floor('h'))]
        needed=list(dict.fromkeys(config['regime_features']+config['targets']))
        points=features.loc[indices,needed].copy()
        points['timestamp']=raw.loc[indices,'timestamp'];points['turbine_id']=raw.loc[indices,'asset_id'];points['source_event']=int(event)
        points['history_ready']=ready.loc[indices];points['normal']=roles.loc[indices,'normal'];points['role']=roles.loc[indices,'role'];points['source_partition']=raw.loc[indices,'train_test']
        labels=old.loc[old.source_event.eq(event),['timestamp','label','reason','episode','failure_at']]
        points=points.merge(labels,on='timestamp',how='left',validate='one_to_one')
        test=points.loc[points.role.eq('test')]
        assert len(test)==len(labels)
        # Real raw prefixes, not just cached-feature truncation.
        for index in [indices[raw.loc[indices,'train_test'].eq('prediction')][0],indices[-1]]:
            past,past_ready=physical_features(scada.loc[:index],physical_config)
            np.testing.assert_allclose(past.loc[index,needed],features.loc[index,needed],rtol=0,atol=0,equal_nan=True)
            prefix_checks.append(dict(event=int(event),timestamp=raw.timestamp[index].isoformat()))
        eligible=points.normal&points.history_ready&points[config['required_current_inputs']].notna().all(axis=1)
        train=points.loc[eligible&points.role.eq('train')];calibration=points.loc[eligible&points.role.eq('calibration')]
        train_counts={t:int(train[t].notna().sum()) for t in config['targets']};cal_counts={t:int(calibration[t].notna().sum()) for t in config['targets']}
        passed=min(train_counts.values())>=config['minimum_training_normal_hours'] and min(cal_counts.values())>=config['minimum_calibration_normal_hours']
        boundary=pd.Timestamp(bounds['boundary']);prediction_start=pd.Timestamp(bounds['prediction_start'])
        assert train.timestamp.add(pd.Timedelta(hours=config['normal_followup_hours'])).le(boundary).all()
        assert calibration.timestamp.add(pd.Timedelta(hours=config['normal_followup_hours'])).lt(prediction_start).all()
        assert not points.loc[points.role.eq('test'),'normal'].any()
        path=out/f'features/{event}.parquet';points.to_parquet(path,index=False);total_rows+=len(points)
        sources.append(dict(event=int(event),asset=str(raw.asset_id.iloc[0]),eligible=bool(passed),bounds=bounds,
            training_normal_hours=len(train),calibration_normal_hours=len(calibration),training_target_counts=train_counts,calibration_target_counts=cal_counts,
            train_end=train.timestamp.max().isoformat() if len(train) else None,calibration_start=calibration.timestamp.min().isoformat() if len(calibration) else None,
            calibration_end=calibration.timestamp.max().isoformat() if len(calibration) else None,prediction_hours=len(test),features_sha256=digest(path)))
        print(f'prepared source={event} train_normal={len(train)} calibration_normal={len(calibration)} eligible={passed}',flush=True)
    status='passed' if all(s['eligible'] for s in sources) else 'partial_unavailable'
    result=dict(status=status,sources=sources,protocol_sha256=digest(CODE/'protocol.json'),input_sha256=gate['inputs_sha256'],
        prior_label_data_sha256=digest(PREVIOUS/'data.parquet'),feature_code_sha256=digest('/feature-code/pipeline.py'),evaluator_sha256=digest('/evaluation/nbm.py'),
        canonical_faults=gate['canonical_faults'],physical_feature_config=physical_config,raw_prefix_checks=prefix_checks,total_hourly_rows=total_rows,
        no_cross_source_training=True,normal_status_retrospectively_available_assumed=True)
    save(result,out/'data_gate.json');save(config,out/'protocol.json')
    print(json.dumps(dict(status=status,eligible_sources=sum(s['eligible'] for s in sources),total_sources=len(sources)),indent=2))


def train(out,config):
    verify_sources();gate=json.loads((out/'data_gate.json').read_text())
    assert gate['protocol_sha256']==digest(CODE/'protocol.json') and gate['evaluator_sha256']==digest('/evaluation/nbm.py')
    reports=[];test_frames=[];reload_checks=[]
    for source in gate['sources']:
        event=source['event'];path=out/f'features/{event}.parquet'
        assert digest(path)==source['features_sha256']
        frame=pd.read_parquet(path);destination=out/f'event_{event}';destination.mkdir()
        if not source['eligible']:
            points=frame.loc[frame.role.eq('test'),['turbine_id','source_event','timestamp','history_ready','label','reason','episode','failure_at']].copy()
            points['score']=np.nan;points['warning']=False;points['threshold']=np.nan;points['model_status']='insufficient_historical_normal_data'
            report=dict(event=event,asset=source['asset'],trained=False,threshold=None)
        else:
            model=PersonalModel(config).fit(frame)
            calculated=model.predict(frame)
            calibration=calculated.join(frame[['role','normal']]).loc[frame.role.eq('calibration')]
            threshold,normal_metrics=normal_threshold(calibration,config)
            model.save(destination)
            loaded=PersonalModel.load(destination)
            replay=loaded.predict(frame)
            np.testing.assert_array_equal(replay.score.to_numpy(),calculated.score.to_numpy())
            reload_checks.append(dict(event=event,max_error=0.))
            model.fit_keys.to_csv(destination/'fit_rows.csv',index=False);model.calibration_keys.to_csv(destination/'calibration_rows.csv',index=False)
            calibration.to_csv(destination/'historical_calibration_scores.csv',index=False)
            points=calculated.join(frame[['label','reason','episode','failure_at','role']]).loc[frame.role.eq('test')].copy()
            points['warning']=points.score.ge(threshold)&points.history_ready;points['threshold']=threshold;points['model_status']='ready'
            inference=dict(source_event=event,asset=source['asset'],threshold=threshold,score_semantics='personal_normal_temperature_deviation_not_failure_probability',
                future_failure_window_under_evaluation_hours=24,normal_metrics=normal_metrics,bounds=source['bounds'],normal_model_manifest_sha256=digest(destination/'normal_model.json'))
            save(inference,destination/'inference.json')
            report=dict(event=event,asset=source['asset'],trained=True,threshold=threshold,historical_calibration=normal_metrics,residual_calibration=model.calibration)
        points.to_csv(destination/'prediction_scores.csv',index=False);test_frames.append(points)
        event_episodes=[e for e in gate['canonical_faults'] if e['source_event']==event]
        report['test']=operational_metrics(points,None,event_episodes,decisions=points.warning)
        reports.append(report)
        print(f'source={event} trained={source["eligible"]} onset_hits={report["test"]["onset_hit_events"]}/{len(event_episodes)} normal_warning_hours={report["test"]["normal_warning_hours"]}',flush=True)
    combined=pd.concat(test_frames,ignore_index=True)
    metrics=operational_metrics(combined,None,gate['canonical_faults'],decisions=combined.warning)
    combined.to_csv(out/'prediction_scores.csv',index=False);pd.DataFrame(metrics['events']).to_csv(out/'fault_results.csv',index=False)
    report=dict(experiment_id=config['experiment_id'],scope=config['scope'],protocol_sha256=digest(CODE/'protocol.json'),sources=reports,metrics=metrics,
        code_sha256={p.name:digest(p) for p in CODE.glob('*.py')},reload_checks=reload_checks,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__),
        prediction_partition_used_for_fitting=False,prediction_fault_labels_used_for_threshold=False,independent_external_validation=False,
        production_integrated=False,calibrated_failure_probability=False,future_sensor_trajectory=False)
    save(report,out/'metrics.json')
    print(json.dumps({k:v for k,v in metrics.items() if k!='events'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train']);args=parser.parse_args()
    out=Path('/output');out.mkdir(exist_ok=True,parents=True);config=json.loads((CODE/'protocol.json').read_text())
    if args.stage=='prepare':
        if (out/'data_gate.json').exists():raise FileExistsError('Use a fresh output directory')
        prepare(out,config)
    else:
        if any(out.glob('event_*')):raise FileExistsError('Do not overwrite fitted source models')
        train(out,config)
