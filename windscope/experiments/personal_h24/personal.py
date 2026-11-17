"""Personal normal-only fitting with explicit historical time boundaries."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_roles(raw,config):
    t=pd.to_datetime(raw.timestamp,utc=True)
    if not t.is_monotonic_increasing or t.duplicated().any():raise ValueError('Invalid source chronology')
    if set(raw.partition)!={'train','prediction'}:raise ValueError('Missing prefix or test partition')
    start=t.loc[raw.partition.eq('prediction')].min()
    if t.loc[raw.partition.eq('train')].max()>=start:raise ValueError('Partitions overlap')
    first=t.loc[raw.partition.eq('train')].min()
    boundary=(first+(start-first)*config['temporal_training_fraction']).floor('h')
    base=raw.partition.eq('train')&raw.status_type_id.eq(0)&raw.observed
    continuous=t.diff().eq(pd.Timedelta(minutes=10))
    segment=(~continuous|~base|~base.shift(fill_value=False)).cumsum()
    run_start=t.groupby(segment).transform('min');run_end=t.groupby(segment).transform('max')
    normal=base&t.sub(run_start).ge(pd.Timedelta(hours=config['normal_past_hours']))&t.add(pd.Timedelta(hours=config['normal_followup_hours'])).le(run_end)
    role=pd.Series('embargo',index=raw.index,dtype='string')
    role.loc[raw.partition.eq('train')&t.le(boundary-pd.Timedelta(hours=config['train_embargo_hours']))]='train'
    role.loc[raw.partition.eq('train')&t.ge(boundary+pd.Timedelta(hours=config['calibration_context_embargo_hours']))]='calibration'
    role.loc[raw.partition.eq('prediction')]='test'
    normal&=~role.eq('train')|t.add(pd.Timedelta(hours=config['normal_followup_hours'])).le(boundary)
    return pd.DataFrame(dict(role=role,normal=normal)),dict(boundary=boundary.isoformat(),prediction_start=start.isoformat(),prefix_start=first.isoformat())


def smooth(points,hours):
    ordered=points.sort_values(['source_event','timestamp'])
    valid=ordered.score.notna()&np.isfinite(ordered.score)
    contiguous=ordered.source_event.eq(ordered.source_event.shift())&ordered.timestamp.diff().eq(pd.Timedelta(hours=1))
    segment=(~contiguous|~valid|~valid.shift(fill_value=False)).cumsum()
    return ordered.score.where(valid).groupby(segment,sort=False).rolling(hours,min_periods=hours).mean().droplevel(0).reindex(points.index)


def normal_threshold(points,config):
    p=points.sort_values(['source_event','timestamp'])
    normal=p.normal&p.score.notna()&np.isfinite(p.score)
    if not normal.any():raise ValueError('No historical normal calibration score')
    scores=p.score.fillna(-np.inf).to_numpy();mask=normal.to_numpy()
    continuous=(p.source_event.eq(p.source_event.shift())&p.timestamp.diff().eq(pd.Timedelta(hours=1))).to_numpy()
    candidates=np.unique(np.r_[scores[mask],np.nextafter(scores[mask].max(),np.inf)])
    for threshold in candidates:
        alarm=(scores>=threshold)&mask
        fraction=float(alarm.sum()/mask.sum())
        if fraction>config['max_normal_warning_fraction']:continue
        episodes=int((alarm&~(np.r_[False,alarm[:-1]]&continuous)).sum())
        rate=episodes*720/int(mask.sum())
        if rate<=config['false_alarm_budget_per_720_normal_hours']:
            return float(threshold),dict(normal_hours=int(mask.sum()),normal_warning_hours=int(alarm.sum()),normal_warning_fraction=fraction,false_alarm_episodes=episodes,false_alarms_per_720_normal_hours=rate)
    raise AssertionError('Threshold above calibration max must be feasible')


class PersonalModel:
    def __init__(self,config):
        self.config=config

    def fit(self,frame):
        if frame.source_event.nunique()!=1 or frame.turbine_id.nunique()!=1:raise ValueError('Personal fitting requires one source and asset')
        config=self.config
        eligible=frame.normal&frame.history_ready&frame[config['required_current_inputs']].notna().all(axis=1)
        train=frame.loc[eligible&frame.role.eq('train')]
        calibration=frame.loc[eligible&frame.role.eq('calibration')]
        if train.empty or calibration.empty or train.timestamp.max()>=calibration.timestamp.min():raise ValueError('Invalid training/calibration chronology')
        self.regressors={};self.calibration={}
        self.fit_keys=train[['turbine_id','source_event','timestamp']].copy()
        self.calibration_keys=calibration[['turbine_id','source_event','timestamp']].copy()
        for target in config['targets']:
            known=train.loc[train[target].notna()];cal=calibration.loc[calibration[target].notna()]
            if len(known)<config['minimum_training_normal_hours'] or len(cal)<config['minimum_calibration_normal_hours']:raise ValueError(f'Insufficient past normal rows: {target}')
            model=XGBRegressor(**config['regressor'])
            model.fit(known[config['regime_features']].astype('float32'),known[target].astype('float32'))
            residual=cal[target]-model.predict(cal[config['regime_features']].astype('float32'))
            center=float(residual.median());scale=max(float(residual.quantile(.95)-center),1.)
            self.calibration[target]=dict(center_celsius=center,scale_celsius=scale,train_rows=len(known),calibration_rows=len(cal))
            self.regressors[target]=model
        return self

    def predict(self,frame):
        config=self.config;result=frame[['turbine_id','source_event','timestamp','history_ready']].copy()
        valid=frame.history_ready&frame[config['required_current_inputs']].notna().all(axis=1)
        residuals=[]
        for target,model in self.regressors.items():
            expected=pd.Series(np.nan,index=frame.index)
            if valid.any():expected.loc[valid]=model.predict(frame.loc[valid,config['regime_features']].astype('float32'))
            residual=frame[target]-expected;cal=self.calibration[target]
            z=(residual-cal['center_celsius'])/cal['scale_celsius'];residuals.append(z)
            channel=target.removesuffix('__now')
            result[channel+'__actual']=frame[target];result[channel+'__expected']=expected
            result[channel+'__residual']=residual;result[channel+'__normalized_residual']=z
        normalized=pd.concat(residuals,axis=1)
        result['raw_score']=normalized.max(axis=1).clip(lower=0).where(valid&normalized.notna().sum(axis=1).ge(config['minimum_target_count']))
        result['score']=result.raw_score
        result['score']=smooth(result,config['score_smoothing_hours'])
        return result

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);files={}
        for i,(target,model) in enumerate(self.regressors.items()):
            name=f'temperature_{i}.ubj';model.save_model(directory/name)
            files[target]=dict(file=name,sha256=digest(directory/name))
        manifest=dict(config=self.config,calibration=self.calibration,models=files)
        (directory/'normal_model.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,directory):
        directory=Path(directory);manifest=json.loads((directory/'normal_model.json').read_text())
        result=cls(manifest['config']);result.calibration=manifest['calibration'];result.regressors={}
        for target,entry in manifest['models'].items():
            path=directory/entry['file']
            if digest(path)!=entry['sha256']:raise ValueError('Native model checksum mismatch')
            model=XGBRegressor();model.load_model(path);result.regressors[target]=model
        return result
