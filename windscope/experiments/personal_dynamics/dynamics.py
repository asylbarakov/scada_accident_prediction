"""Causal residual features; no labels or future observations enter this module."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CHANNELS=['gearbox_bearing_temperature','gearbox_oil_temperature','generator_bearing_temperature','hydraulic_oil_temperature']
REGIME=['wind_speed','ambient_temperature','grid_active_power','generator_speed','rotor_speed']


def dynamic_features(frame):
    if not frame.index.is_unique:raise ValueError('Feature indices must be unique')
    times=pd.to_datetime(frame.timestamp,utc=True)
    for _,g in frame.assign(_time=times).groupby(['turbine_id','source_event'],sort=False):
        if not g._time.is_monotonic_increasing or g._time.duplicated().any():raise ValueError('Invalid source chronology')
    z=frame[[c+'__normalized_residual' for c in CHANNELS]].replace([np.inf,-np.inf],np.nan)
    valid=frame.history_ready.fillna(False)&z.notna().sum(axis=1).ge(2)
    continuous=frame.source_event.eq(frame.source_event.shift())&frame.turbine_id.eq(frame.turbine_id.shift())&times.diff().eq(pd.Timedelta(hours=1))
    segment=(~continuous|~valid|~valid.shift(fill_value=False)).cumsum()
    result=frame[['turbine_id','source_event','timestamp']].copy()

    def rolling(series,hours,operation):
        windows=series.groupby(segment,sort=False).rolling(hours,min_periods=hours)
        if operation=='slope':values=windows.apply(lambda x:float(np.dot(x,np.arange(6)-2.5)/17.5),raw=True)
        elif operation=='std':values=windows.std(ddof=0)
        else:values=windows.mean()
        return values.droplevel(0).reindex(frame.index)

    for c in CHANNELS:
        values=z[c+'__normalized_residual'].where(valid)
        result[c+'__z']=values
        for hours in (3,6):result[c+f'__mean{hours}']=rolling(values,hours,'mean')
        result[c+'__std6']=rolling(values,6,'std')
        result[c+'__slope6']=rolling(values,6,'slope')
        result[c+'__persistence6']=rolling(values.ge(1).astype(float).where(values.notna()),6,'mean')
        result[c+'__delta1']=values.groupby(segment,sort=False).diff()
    positive=z.clip(lower=0)
    result['joint__max']=positive.max(axis=1).where(valid)
    ordered=np.sort(positive.fillna(-np.inf).to_numpy(),axis=1)
    result['joint__second']=pd.Series(ordered[:,-2],index=frame.index).where(valid)
    result['joint__count']=z.ge(1).sum(axis=1).astype(float).where(valid)
    for name in REGIME:
        column=name+'__now'
        if column in frame:result['regime__'+name]=frame[column].where(valid)
    result['history_ready']=valid&rolling(valid.astype(float).where(valid),6,'mean').eq(1)
    return result


def feature_names(frame):
    return [c for c in frame if c not in ('turbine_id','source_event','timestamp','history_ready')]


def fit_head(frame,features,assets,config):
    forbidden={'label','failure_at','episode','reason','status_type_id','role','source_partition','turbine_id','source_event','timestamp','normal'}
    if forbidden.intersection(features):raise ValueError('Nonphysical or identity feature requested')
    train=frame.loc[frame.turbine_id.isin(assets)&frame.history_ready&frame.label.notna()]
    if train.label.nunique()!=2:raise ValueError('Training needs both known classes')
    model=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler()),('model',LogisticRegression(**config))])
    model.fit(train[features],train.label.astype(int))
    return model,train[['turbine_id','source_event','timestamp']].copy()
