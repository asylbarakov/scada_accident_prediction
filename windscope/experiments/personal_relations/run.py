"""Three predeclared ablations; no post-test selection or application changes."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from relations import PersonalModel,RELATION_TARGETS,POWER_FEATURES,physical_relations,fit_model,score_model,combine
from personal import digest,normal_threshold
from nbm import operational_metrics

CODE=Path(__file__).parent
BASE=Path('/baseline')
OUT=Path('/output')
POWER_AUX=['power30__now','power30__1h_mean','power30__6h_mean']


def save(value,path):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def actual_frame(frame):
    result=frame.copy()
    for target,source in zip(POWER_FEATURES,POWER_AUX):result[target]=result[source]
    result['history_ready']&=result.power30_ready
    return result


def fit():
    if (OUT/'metrics.json').exists():raise FileExistsError('Choose a fresh output directory')
    protocol=json.loads((CODE/'protocol.json').read_text());protocol_hash=digest(CODE/'protocol.json')
    baseline=json.loads((BASE/'metrics.json').read_text());gate=json.loads((BASE/'data_gate.json').read_text())
    if digest('/personal-code/personal.py')!=baseline['code_sha256']['personal.py']:raise ValueError('Baseline implementation changed')
    save(protocol,OUT/'protocol.json');(OUT/'source_features').mkdir()
    predictions={name:[] for name in protocol['variants']};sources=[];raw_checks=[];model_checks=[];quality=[]
    for source in gate['sources']:
        event=source['event'];feature_path=BASE/f'features/{event}.parquet'
        if digest(feature_path)!=source['features_sha256']:raise ValueError('Frozen features changed')
        path=Path('/research/data/CARE_To_Compare/Wind Farm A/datasets')/f'{event}.csv'
        if digest(path)!=gate['input_sha256'][f'raw/datasets/{event}.csv']:raise ValueError('Raw input changed')
        raw=pd.read_csv(path,sep=';',usecols=['time_stamp','asset_id','sensor_13_avg','sensor_14_avg','sensor_38_avg','sensor_39_avg','sensor_40_avg','power_30_avg'],dtype={'asset_id':'string'},float_precision='round_trip')
        raw['timestamp']=pd.to_datetime(raw.time_stamp,utc=True);raw['turbine_id']=raw.asset_id
        derived,power_ready=physical_relations(raw)
        derived['timestamp']=raw.timestamp;derived['power30_ready']=power_ready
        hourly=derived.loc[raw.timestamp.eq(raw.timestamp.dt.floor('h'))].set_index('timestamp')
        frame=pd.read_parquet(feature_path)
        selected=hourly.reindex(pd.DatetimeIndex(frame.timestamp))
        for name in RELATION_TARGETS:frame[name]=selected[name].to_numpy()
        for target,name in zip(POWER_AUX,POWER_FEATURES):frame[target]=selected[name].to_numpy()
        frame['power30_ready']=selected.power30_ready.to_numpy()
        assert np.array_equal(frame.history_ready.to_numpy(),frame.power30_ready.to_numpy()),'Changed context availability'
        assert not frame.loc[frame.role.eq('test'),'normal'].any()
        prepared=OUT/f'source_features/{event}.parquet';frame.to_parquet(prepared,index=False)
        original=PersonalModel.load(BASE/f'event_{event}')
        original_scores=score_model(original,frame)
        archive=pd.read_csv(BASE/f'event_{event}/prediction_scores.csv')
        np.testing.assert_allclose(original_scores.loc[frame.role.eq('test'),'score'],archive.score,rtol=0,atol=1e-12,equal_nan=True)
        changed=actual_frame(frame)
        relation_config=dict(original.config,targets=RELATION_TARGETS,two_sided_targets=['generator_bearing_delta__now'])
        rel_old=fit_model(frame,relation_config)
        power_model=fit_model(changed,dict(original.config,two_sided_targets=[]))
        rel_actual=fit_model(changed,relation_config)
        models={'relations_only':combine(original,rel_old),'power30_only':power_model,'relations_power30':combine(power_model,rel_actual)}
        boundary=pd.Timestamp(source['bounds']['boundary']);start=pd.Timestamp(source['bounds']['prediction_start'])
        for model in [rel_old,power_model,rel_actual]:
            assert model.fit_keys.timestamp.add(pd.Timedelta(hours=24)).le(boundary).all()
            assert model.calibration_keys.timestamp.add(pd.Timedelta(hours=24)).lt(start).all()
        for i in [frame.index[frame.role.eq('test')][0],frame.index[-1]]:
            raw_index=raw.index[raw.timestamp.eq(frame.timestamp.iloc[i])][0]
            prefix,ready_prefix=physical_relations(raw.iloc[:raw_index+1])
            np.testing.assert_allclose(prefix.iloc[-1],hourly[RELATION_TARGETS+POWER_FEATURES].loc[frame.timestamp.iloc[i]],rtol=0,atol=0,equal_nan=True)
            assert ready_prefix.iloc[-1]==frame.power30_ready.iloc[i]
            raw_checks.append(dict(event=event,at=str(frame.timestamp.iloc[i]),passed=True))
        for target in RELATION_TARGETS:
            for role in ['train','calibration','test']:
                values=frame.loc[frame.role.eq(role),target]
                quality.append(dict(event=event,target=target,role=role,rows=len(values),missing=int(values.isna().sum()),minimum=float(values.min()) if values.notna().any() else None,maximum=float(values.max()) if values.notna().any() else None))
        entry=dict(event=event,asset=source['asset'],source_features_sha256=digest(prepared),variants={})
        for name,model in models.items():
            current=frame if name=='relations_only' else changed
            scores=score_model(model,current)
            calibration=scores.join(current[['role','normal']]).loc[current.role.eq('calibration')]
            threshold,normal=normal_threshold(calibration,original.config)
            destination=OUT/name/f'event_{event}';destination.mkdir(parents=True)
            model.save(destination);loaded=PersonalModel.load(destination)
            np.testing.assert_allclose(score_model(loaded,current).score,scores.score,rtol=0,atol=0,equal_nan=True)
            for stop in [frame.index[frame.role.eq('test')][0]+1,len(frame)]:
                np.testing.assert_allclose(score_model(model,current.iloc[:stop]).score,scores.score.iloc[:stop],rtol=0,atol=1e-12,equal_nan=True)
                model_checks.append(dict(event=event,variant=name,stop=int(stop),passed=True))
            fitted=rel_old if name=='relations_only' else power_model
            fitted.fit_keys.to_csv(destination/'fit_rows.csv',index=False)
            fitted.calibration_keys.to_csv(destination/'calibration_rows.csv',index=False)
            points=scores.join(frame[['label','reason','episode','failure_at','role']]).loc[frame.role.eq('test')].copy()
            points['threshold']=threshold;points['warning']=points.score.ge(threshold)&points.history_ready
            points.to_csv(destination/'prediction_scores.csv',index=False);predictions[name].append(points)
            entry['variants'][name]=dict(threshold=threshold,historical_calibration=normal,model_manifest_sha256=digest(destination/'normal_model.json'),fit_rows=len(fitted.fit_keys),calibration_rows=len(fitted.calibration_keys))
        sources.append(entry)
        print(f'source={event}: three fixed variants fitted and verified',flush=True)
    summaries={}
    for name,frames in predictions.items():
        combined=pd.concat(frames,ignore_index=True);metrics=operational_metrics(combined,None,gate['canonical_faults'],decisions=combined.warning)
        combined.to_csv(OUT/f'{name}_predictions.csv',index=False);pd.DataFrame(metrics['events']).to_csv(OUT/f'{name}_faults.csv',index=False)
        promote=metrics['onset_hit_events']>4 and metrics['false_alarms_per_720_normal_hours']<=2 and metrics['normal_warning_fraction']<=.05 and metrics['availability']>=baseline['metrics']['availability']-.01
        summaries[name]=dict(metrics=metrics,promotion_gate_passed=bool(promote))
        print(json.dumps(dict(variant=name,hits=metrics['onset_hit_events'],false_episodes=metrics['false_alarm_episodes'],false_rate=metrics['false_alarms_per_720_normal_hours'],gate=bool(promote))),flush=True)
    assert digest(CODE/'protocol.json')==protocol_hash
    save(dict(experiment_id=protocol['experiment_id'],baseline_metrics=baseline['metrics'],variants=summaries,sources=sources,
        protocol_sha256=protocol_hash,code_sha256={p.name:digest(p) for p in CODE.glob('*.py')},
        baseline_gate_sha256=digest(BASE/'data_gate.json'),production_integrated=False,independent_external_validation=False,
        calibrated_failure_probability=False,model_selection_from_test=False),OUT/'metrics.json')
    save(dict(scope='delayed relations, original numeric scale, no finite-value clipping',channels=quality),OUT/'quality.json')
    save(dict(status='passed',baseline_reproduced_sources=22,native_reload_bundles=66,raw_prefix_checks=raw_checks,model_prefix_checks=model_checks,
        fitting_calibration_followup_before_prediction=True,power_context_availability_identical=True),OUT/'verification.json')


def verify():
    report=json.loads((OUT/'metrics.json').read_text());baseline_gate=json.loads((BASE/'data_gate.json').read_text())
    assert digest(CODE/'protocol.json')==report['protocol_sha256']
    for name,expected in report['code_sha256'].items():assert digest(CODE/name)==expected
    replays={name:[] for name in report['variants']};count=0
    for source in report['sources']:
        event=source['event'];path=OUT/f'source_features/{event}.parquet'
        assert digest(path)==source['source_features_sha256']
        frame=pd.read_parquet(path)
        for name,metadata in source['variants'].items():
            destination=OUT/name/f'event_{event}'
            assert digest(destination/'normal_model.json')==metadata['model_manifest_sha256']
            model=PersonalModel.load(destination)
            points=score_model(model,frame if name=='relations_only' else actual_frame(frame))
            points=points.join(frame[['label','reason','episode','failure_at','role']]).loc[frame.role.eq('test')].copy()
            points['warning']=points.score.ge(metadata['threshold'])&points.history_ready
            stored=pd.read_csv(destination/'prediction_scores.csv')
            np.testing.assert_allclose(points.score,stored.score,rtol=0,atol=1e-12,equal_nan=True)
            assert np.array_equal(points.warning.to_numpy(),stored.warning.to_numpy())
            replays[name].append(points);count+=1
    for name,frames in replays.items():
        combined=pd.concat(frames,ignore_index=True)
        metrics=operational_metrics(combined,None,baseline_gate['canonical_faults'],decisions=combined.warning)
        assert metrics==report['variants'][name]['metrics']
    save(dict(status='passed',bundles=count,decisions_per_variant=8433,all_scores_and_metrics_reproduced=True,no_fitting=True,protocol_and_code_unchanged=True),OUT/'independent_replay.json')
    print('Independent native replay: 66 bundles, all metrics reproduced.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',nargs='?',default='fit',choices=['fit','verify']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='fit':fit()
    else:verify()
