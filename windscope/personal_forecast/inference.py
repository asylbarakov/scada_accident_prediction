"""Online personal deviations from at most eight hours of physical history."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from personal import PersonalModel
from scada_pdm.preparation import make_features

MODEL_ID='care-personal-normal-h24-v1'
CHANNELS={
    'gearbox_bearing_temperature':('sensor_11_avg','Подшипник редуктора'),
    'gearbox_oil_temperature':('sensor_12_avg','Масло редуктора'),
    'generator_bearing_temperature':('sensor_13_avg','Подшипник генератора'),
    'hydraulic_oil_temperature':('sensor_41_avg','Масло гидравлической группы'),
}


def physical_state(values,_):
    observed=values.notna().any(axis=1)
    return pd.DataFrame(dict(observed=observed,at_risk=observed))


class PersonalAdapter:
    def __init__(self,root):
        self.root=Path(root);self.registry=json.loads((self.root/'registry.json').read_text(encoding='utf-8'))
        if self.registry['model_id']!=MODEL_ID:raise ValueError('Unsupported personal registry')
        self.config=self.registry['feature_config'];self.sensors=[v['column'] for v in self.config['sensors'].values()]
        self.models={}
        for identifier,entry in self.registry['sources'].items():
            path=self.root/identifier/'normal_model.json'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['normal_model_sha256']:raise ValueError('Historical baseline manifest checksum mismatch')
            self.models[identifier]=PersonalModel.load(path.parent)

    def predict(self,source_id,asset_id,source_time,history):
        started=time.perf_counter()
        if source_id not in self.models:raise ValueError('Historical baseline is unavailable for this source')
        entry=self.registry['sources'][source_id];at=pd.Timestamp(source_time)
        if at.tzinfo is not None or pd.isna(at) or at!=at.floor('h'):raise ValueError('Personal decisions are hourly on the original dataset clock')
        if asset_id!=entry['asset_id']:raise ValueError('Asset does not match historical baseline')
        if at<pd.Timestamp(entry['prediction_start']).tz_localize(None):raise ValueError('Decision precedes historical baseline deployment')
        if not 1<=len(history)<=49:raise ValueError('Expected at most eight hours of measurements')
        times=pd.to_datetime([row['source_time'] for row in history],errors='raise')
        if times.tz is not None or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing or times.max()!=at or times.min()<at-pd.Timedelta(hours=8) or not times.equals(times.floor('10min')):raise ValueError('History must be chronological and end at the decision, with no future rows')
        for row in history:
            if not set(row['values']).issubset(self.sensors):raise ValueError('Only physical model sensors are allowed')
            if any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)) for v in row['values'].values()):raise ValueError('Invalid sensor value')
        values=pd.DataFrame([{name:row['values'].get(info['column']) for name,info in self.config['sensors'].items()} for row in history],dtype='float32')
        scada=values.copy();scada.insert(0,'timestamp',times);scada.insert(0,'turbine_id',asset_id)
        features,_,ready=make_features(scada,self.config,state_policy=physical_state)
        features['timestamp']=times;features['turbine_id']=asset_id;features['source_event']=int(source_id);features['history_ready']=ready
        hourly=features.loc[features.timestamp.eq(features.timestamp.dt.floor('h'))].copy()
        scored=self.models[source_id].predict(hourly)
        latest=scored.iloc[-1];score=None if pd.isna(latest.score) else float(latest.score)
        threshold=entry['threshold'];valid=score is not None
        evidence=[]
        for name,(sensor,label) in CHANNELS.items():
            actual=latest[name+'__actual'];expected=latest[name+'__expected'];residual=latest[name+'__residual'];normalized=latest[name+'__normalized_residual']
            if all(pd.notna(v) for v in (actual,expected,residual,normalized)):
                evidence.append(dict(sensor=sensor,label=label,value=float(actual),expected=float(expected),delta=float(residual),unit='°C',normalized_residual=float(normalized),
                    observation_time=str(at-pd.Timedelta(minutes=10))))
        evidence.sort(key=lambda row:row['normalized_residual'],reverse=True)
        return dict(status='ok' if valid else 'warming_up',model_id=MODEL_ID,predicted_class=int(score>=threshold) if valid else None,
            class_1_score=score,deviation_score=score,threshold=threshold,warning_index=score/threshold if valid and threshold>0 else None,
            prediction_horizon='P1D',prediction_horizon_hours=24,prediction_interval_start=str(at),prediction_interval_end=str(at+pd.Timedelta(hours=24)),
            feature_cutoff=str(at-pd.Timedelta(minutes=10)),cadence_minutes=60,score_semantics='personal_normal_deviation_not_probability',
            evidence=evidence,inference_ms=round((time.perf_counter()-started)*1000,2),history_points=len(history),
            baseline=dict(training_normal_hours=entry['training_normal_hours'],calibration_normal_hours=entry['calibration_normal_hours'],available_before=entry['baseline_end']))
