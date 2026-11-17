"""Read-only model audit. No fitting, UI changes, threshold tuning, or DB access."""
import gc,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix
from forecast.inference import H7Adapter,care_state
from scada_pdm.preparation import make_features

ROOT=Path('/research')
OUT=Path('/audit-output')
RUN=ROOT/'results/care_h7_windows_20261116/training'
RAW=ROOT/'data/CARE_To_Compare/Wind Farm A'
KEY=['turbine_id','source_event','timestamp']


def series_alarms(frame,threshold):
    frame=frame.sort_values('timestamp').copy()
    active=frame['score'].ge(threshold).fillna(False)
    continued=frame['timestamp'].diff().eq(pd.Timedelta(hours=1))
    start=active & ~(active.shift(fill_value=False)&continued)
    frame['warning_episode']=start.cumsum().where(active)
    return frame


def evaluate(frame,threshold):
    frame=frame.sort_values(['source_event','timestamp']).copy()
    frame['decision']=frame['score'].ge(threshold)
    y=frame['label'].astype(int)
    tn,fp,fn,tp=confusion_matrix(y,frame['decision'],labels=[0,1]).ravel()
    detected=[];alarms=0;false_alarms=0
    for _,group in frame.groupby('source_event'):
        grouped=series_alarms(group,threshold)
        for _,alarm in grouped.dropna(subset=['warning_episode']).groupby('warning_episode'):
            alarms+=1
            false_alarms+=int(not alarm['label'].eq(1).any())
    for episode,group in frame.loc[y.eq(1)].groupby('episode_id'):
        positive=group.loc[group['decision']]
        fault=group['failure_at'].iloc[0]
        first=positive['timestamp'].min() if len(positive) else None
        detected.append(dict(episode=str(episode),failure_at=str(fault),detected=first is not None,
            first_warning=str(first) if first is not None else None,
            lead_hours=(fault-first).total_seconds()/3600 if first is not None else None))
    return dict(rows=len(frame),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        point_recall=float(tp/(tp+fn)),point_precision=float(tp/(tp+fp)) if tp+fp else None,
        average_precision=float(average_precision_score(y,frame['score'])),
        roc_auc=float(roc_auc_score(y,frame['score'])),
        detected_events=sum(e['detected'] for e in detected),eligible_events=len(detected),
        warning_episodes=alarms,false_warning_episodes=false_alarms,
        known_negative_hours=int(y.eq(0).sum()),events=detected)


def main():
    OUT.mkdir(exist_ok=True)
    adapter=H7Adapter(RUN/'h7/logistic',RUN/'config.json')
    metadata=pd.read_csv(RAW/'event_info.csv',sep=';',dtype={'asset':'string'})
    labels=pd.read_parquet(ROOT/'cache/care_h7_v6/labels.parquet')
    labels['timestamp']=pd.to_datetime(labels['timestamp'],utc=True).dt.tz_localize(None)
    labels['turbine_id']=labels['turbine_id'].astype(str)
    archived=pd.read_parquet(RUN/'h7/logistic/test_predictions.parquet')
    archived['timestamp']=pd.to_datetime(archived['timestamp'],utc=True).dt.tz_localize(None)
    archived['failure_at']=pd.to_datetime(archived['failure_at'],utc=True).dt.tz_localize(None)
    archived['turbine_id']=archived['turbine_id'].astype(str)
    preflight=json.loads((ROOT/'results/care_h7_windows_20261116/preflight.json').read_text())
    for filename in ('event_info.csv','feature_description.csv'):
        assert hashlib.sha256((RAW/filename).read_bytes()).hexdigest()==preflight['raw_source_sha256'][filename]
    for filename in ('labels.parquet','coverage.json'):
        assert hashlib.sha256((ROOT/'cache/care_h7_v6'/filename).read_bytes()).hexdigest()==preflight['cache_sha256'][filename]
    config=adapter.config
    sensors={name:info['column'] for name,info in config['sensors'].items()}
    reports=[];all_points=[];online_errors=[]
    for event in (13,22,72):
        info=metadata.loc[metadata.event_id.eq(event)].iloc[0]
        source=RAW/f'datasets/{event}.csv'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==preflight['raw_source_sha256'][f'datasets/{event}.csv']
        raw=pd.read_csv(source,sep=';',float_precision='round_trip',
            usecols=['time_stamp','asset_id','train_test',*sensors.values()],dtype={'asset_id':'string'})
        times=pd.to_datetime(raw['time_stamp'])
        scada=raw[list(sensors.values())].astype('float32').replace([np.inf,-np.inf],np.nan)
        scada=scada.rename(columns={v:k for k,v in sensors.items()})
        scada.insert(0,'timestamp',times);scada.insert(0,'turbine_id',raw.asset_id.astype(str))
        # Feature construction is causal. Labels/annotations are joined only below.
        features,_,ready=make_features(scada,config,state_policy=care_state)
        scheduled=raw['train_test'].eq('prediction')&times.eq(times.dt.floor('h'))
        indices=raw.index[scheduled]
        good=indices[ready.loc[indices]]
        points=pd.DataFrame(dict(turbine_id=scada.loc[indices,'turbine_id'],source_event=event,
            timestamp=times.loc[indices],status=np.where(ready.loc[indices],'ok','warming_up'),score=np.nan))
        points.loc[good,'score']=adapter.pipeline.predict_proba(features.loc[good,adapter.features])[:,1]
        points['decision']=points['score'].ge(adapter.threshold).where(points.status.eq('ok'))
        points['sensor_0_avg']=raw.loc[indices,'sensor_0_avg']
        points['sensor_11_avg']=raw.loc[indices,'sensor_11_avg']
        # Verify actual sequential adapter calls against vectorized causal inference.
        sampled=good[[0,len(good)//2,-1]]
        for i in sampled:
            at=times[i]
            past=raw.loc[times.between(at-pd.Timedelta(hours=168),at)]
            history=[dict(source_time=r['time_stamp'],values={k:None if pd.isna(r[k]) else float(r[k]) for k in sensors.values()})
                     for r in past.to_dict('records')]
            result=adapter.predict(str(info['asset']),str(at),history)
            assert result['status']=='ok'
            error=abs(result['class_1_score']-float(points.loc[i,'score']))
            assert error<1e-10,(event,str(at),error)
            online_errors.append(error)
        # Evaluation-only metadata: never supplied to the pipeline.
        points=points.merge(labels[KEY+['h7_label','h7_reason','h7_episode','h7_failure_at']],on=KEY,how='left',validate='one_to_one')
        points.to_csv(OUT/f'event_{event}_hourly.csv',index=False)
        scored=points.loc[points.status.eq('ok')]
        report=dict(event=event,asset=str(info['asset']),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_label=str(info['event_label']),description=None if pd.isna(info['event_description']) else str(info['event_description']),
            annotation_start=str(info['event_start']),annotation_end=str(info['event_end']),
            prediction_measurements=int(raw.train_test.eq('prediction').sum()),scheduled_hours=len(points),
            successful_hours=len(scored),unavailable_hours=int(points.status.ne('ok').sum()),
            readiness_percent=100*len(scored)/len(points),warning_points=int(scored['decision'].eq(True).sum()),
            full_warning_episodes=int(series_alarms(points,adapter.threshold)['warning_episode'].nunique()),
            score_min=float(scored.score.min()),score_max=float(scored.score.max()),
            first_valid_forecast=str(scored.timestamp.min()))
        if info['event_label']=='anomaly':
            fault=pd.Timestamp(info['event_end']);target=points.timestamp.lt(fault)&points.timestamp.ge(fault-pd.Timedelta(days=7))
            hit=points.loc[target&points.decision.eq(True)]
            report.update(first_h7_warning=str(hit.timestamp.min()) if len(hit) else None,
                last_h7_warning=str(hit.timestamp.max()) if len(hit) else None,
                first_lead_hours=float((fault-hit.timestamp.min()).total_seconds()/3600) if len(hit) else None,
                last_lead_hours=float((fault-hit.timestamp.max()).total_seconds()/3600) if len(hit) else None,
                last_hours={str(h):dict(scheduled=int((points.timestamp.lt(fault)&points.timestamp.ge(fault-pd.Timedelta(hours=h))).sum()),
                    available=int(((points.timestamp.lt(fault)&points.timestamp.ge(fault-pd.Timedelta(hours=h)))&points.status.eq('ok')).sum()),
                    warning=int(((points.timestamp.lt(fault)&points.timestamp.ge(fault-pd.Timedelta(hours=h)))&points.decision.eq(True)).sum())) for h in (6,24,48,168)})
        reports.append(report);all_points.append(points)
        del features,raw,scada;gc.collect()
    combined=pd.concat(all_points,ignore_index=True)
    compared=archived.merge(combined[KEY+['score','status']],on=KEY,suffixes=('_archived','_new'),validate='one_to_one')
    assert len(compared)==644
    max_error=float((compared.score_archived-compared.score_new).abs().max())
    assert compared.status.eq('ok').all() and max_error<1e-10
    evaluated=compared.rename(columns={'score_archived':'archive_score','score_new':'score'})
    real=evaluate(evaluated,adapter.threshold)
    baselines={name:evaluate(evaluated.assign(score=constant),.5) for name,constant in [('never_warn',0.),('always_warn',1.)]}
    stored_lgb=pd.read_parquet(RUN/'h7/lightgbm/test_predictions.parquet')
    stored_lgb['timestamp']=pd.to_datetime(stored_lgb['timestamp'],utc=True).dt.tz_localize(None)
    stored_lgb['failure_at']=pd.to_datetime(stored_lgb['failure_at'],utc=True).dt.tz_localize(None)
    lgb_meta=json.loads((RUN/'h7/lightgbm/inference.json').read_text())
    # Artifact-only check: LightGBM is not loaded or installed in the LR service.
    lgb=evaluate(stored_lgb,lgb_meta['threshold'])
    result=dict(status='passed',scope='no retraining; unchanged thresholds; held-out asset 21; one historical pass',
        model_id='care-h7-logistic-20261116',threshold=adapter.threshold,forecast_horizon_hours=168,
        source_events=reports,archive_rows_reproduced=len(compared),max_archive_score_error=max_error,
        sequential_raw_checks=len(online_errors),max_sequential_score_error=max(online_errors),
        held_out_metrics=real,trivial_baselines=baselines,
        lightgbm_archived_metrics=lgb,lightgbm_threshold=lgb_meta['threshold'],
        future_sensor_forecasting_implemented=False,calibrated_failure_probability=False,
        independent_test_turbines=1,independent_test_episodes=2,source_and_label_integrity_verified=True)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=True))


if __name__=='__main__':main()
