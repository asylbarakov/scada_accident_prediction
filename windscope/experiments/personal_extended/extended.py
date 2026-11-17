"""Controlled extension of frozen normal behaviour with four temperature targets."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.extend(['/personal-code','/evaluation'])
from personal import PersonalModel,normal_threshold,digest
from nbm import operational_metrics


def delayed_targets(raw,mapping):
    times=pd.to_datetime(raw.timestamp,utc=True)
    if not times.is_monotonic_increasing or times.duplicated().any():raise ValueError('Invalid raw chronology')
    adjacent=times.diff().eq(pd.Timedelta(minutes=10))
    return pd.DataFrame({name:pd.to_numeric(raw[column],errors='raise').replace([np.inf,-np.inf],np.nan).shift(1).where(adjacent).astype('float32') for name,column in mapping.items()},index=raw.index)


def save(value,path):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(out):
    code=Path(__file__).parent;config=json.loads((code/'protocol.json').read_text())
    base=Path('/baseline');gate=json.loads((base/'data_gate.json').read_text());baseline=json.loads((base/'metrics.json').read_text())
    if (out/'metrics.json').exists():raise FileExistsError('Choose a fresh run')
    if digest('/personal-code/personal.py')!=baseline['code_sha256']['personal.py']:raise ValueError('Baseline implementation changed')
    protocol_hash=digest(code/'protocol.json');save(config,out/'protocol.json')
    frames=[];sources=[];quality=[];checks=[]
    for source in gate['sources']:
        event=source['event'];feature_path=base/f'features/{event}.parquet'
        if digest(feature_path)!=source['features_sha256']:raise ValueError('Baseline features changed')
        raw_path=Path('/research/data/CARE_To_Compare/Wind Farm A/datasets')/f'{event}.csv'
        if digest(raw_path)!=gate['input_sha256'][f'raw/datasets/{event}.csv']:raise ValueError('Raw source changed')
        raw=pd.read_csv(raw_path,sep=';',usecols=['time_stamp',*config['added_targets'].values()],float_precision='round_trip')
        raw['timestamp']=pd.to_datetime(raw.time_stamp,utc=True)
        extra=delayed_targets(raw,config['added_targets']);extra['timestamp']=raw.timestamp
        extra=extra.loc[raw.timestamp.eq(raw.timestamp.dt.floor('h'))].set_index('timestamp')
        frame=pd.read_parquet(feature_path)
        for name in config['added_targets']:frame[name]=extra[name].reindex(pd.DatetimeIndex(frame.timestamp)).to_numpy()
        original=PersonalModel.load(base/f'event_{event}')
        original_scores=original.predict(frame)
        old=pd.read_csv(base/f'event_{event}/prediction_scores.csv')
        np.testing.assert_allclose(original_scores.loc[frame.role.eq('test'),'score'],old.score,rtol=0,atol=1e-12,equal_nan=True)
        additional_config=original.config.copy();additional_config['targets']=list(config['added_targets'])
        added=PersonalModel(additional_config).fit(frame)
        model=PersonalModel(dict(original.config,targets=original.config['targets']+list(config['added_targets'])))
        model.regressors={**original.regressors,**added.regressors};model.calibration={**original.calibration,**added.calibration}
        scores=model.predict(frame)
        calibration=scores.join(frame[['role','normal']]).loc[frame.role.eq('calibration')]
        threshold,normal_metrics=normal_threshold(calibration,original.config)
        destination=out/f'event_{event}';destination.mkdir()
        model.save(destination);loaded=PersonalModel.load(destination)
        np.testing.assert_allclose(loaded.predict(frame).score,scores.score,rtol=0,atol=0,equal_nan=True)
        added.fit_keys.to_csv(destination/'fit_rows.csv',index=False);added.calibration_keys.to_csv(destination/'calibration_rows.csv',index=False)
        for index in [frame.index[frame.role.eq('test')][0],frame.index[-1]]:
            np.testing.assert_allclose(model.predict(frame.iloc[:index+1]).score,scores.score.iloc[:index+1],rtol=0,atol=1e-12,equal_nan=True)
            raw_index=raw.index[raw.timestamp.eq(frame.timestamp.iloc[index])][0]
            prefix=delayed_targets(raw.iloc[:raw_index+1],config['added_targets'])
            np.testing.assert_allclose(prefix.iloc[-1],extra.loc[frame.timestamp.iloc[index]],rtol=0,atol=0,equal_nan=True)
            checks.append(dict(event=event,at=str(frame.timestamp.iloc[index]),raw_and_model_prefix=True))
        boundary=pd.Timestamp(source['bounds']['boundary']);start=pd.Timestamp(source['bounds']['prediction_start'])
        assert added.fit_keys.timestamp.add(pd.Timedelta(hours=24)).le(boundary).all()
        assert added.calibration_keys.timestamp.add(pd.Timedelta(hours=24)).lt(start).all()
        for name in config['added_targets']:
            for role in ['train','calibration','test']:
                values=frame.loc[frame.role.eq(role),name]
                certified=frame.loc[frame.role.eq(role)&frame.normal,name]
                quality.append(dict(event=event,target=name,role=role,rows=len(values),missing=int(values.isna().sum()),
                    at_205=int(values.eq(205).sum()),certified_normal_at_205=int(certified.eq(205).sum()),
                    minimum=float(values.min()) if values.notna().any() else None,maximum=float(values.max()) if values.notna().any() else None))
        points=scores.join(frame[['label','reason','episode','failure_at','role']]).loc[frame.role.eq('test')].copy()
        points['threshold']=threshold;points['warning']=points.score.ge(threshold)&points.history_ready
        points.to_csv(destination/'prediction_scores.csv',index=False);frames.append(points)
        sources.append(dict(event=event,asset=source['asset'],threshold=threshold,historical_calibration=normal_metrics,
            added_target_calibration=added.calibration,fit_rows=len(added.fit_keys),calibration_rows=len(added.calibration_keys),
            reused_baseline_manifest_sha256=digest(base/f'event_{event}/normal_model.json'),new_manifest_sha256=digest(destination/'normal_model.json')))
        print(f'source={event} eight targets fitted; threshold={threshold:.3f}',flush=True)
    combined=pd.concat(frames,ignore_index=True);metrics=operational_metrics(combined,None,gate['canonical_faults'],decisions=combined.warning)
    old_metrics=baseline['metrics']
    promote=metrics['onset_hit_events']>old_metrics['onset_hit_events'] and metrics['false_alarms_per_720_normal_hours']<=2 and metrics['normal_warning_fraction']<=.05 and metrics['availability']>=old_metrics['availability']-.01
    combined.to_csv(out/'prediction_scores.csv',index=False);pd.DataFrame(metrics['events']).to_csv(out/'fault_results.csv',index=False)
    save(dict(scope='quality audit, no finite-value clipping or label-dependent filtering',channels=quality),out/'quality.json')
    save(dict(status='passed',baseline_reproduced_for_all_sources=True,native_reload_sources=len(sources),raw_and_model_prefix_checks=checks,
        all_added_fitting_and_calibration_followup_before_prediction=True,protocol_unchanged=digest(code/'protocol.json')==protocol_hash),out/'verification.json')
    result=dict(experiment_id=config['experiment_id'],protocol_sha256=protocol_hash,metrics=metrics,baseline_metrics=old_metrics,sources=sources,
        promotion_gate_passed=bool(promote),production_integrated=False,independent_external_validation=False,
        prior_gate_sha256=digest(base/'data_gate.json'),code_sha256={p.name:digest(p) for p in code.glob('*.py')})
    save(result,out/'metrics.json')
    print(json.dumps(dict(metrics={k:v for k,v in metrics.items() if k!='events'},promotion_gate_passed=bool(promote)),indent=2),flush=True)


if __name__=='__main__':
    out=Path('/output');out.mkdir(exist_ok=True,parents=True);run(out)
