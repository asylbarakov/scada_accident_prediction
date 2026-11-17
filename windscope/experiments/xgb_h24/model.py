"""Causal seven-feature extraction, train-only scaling and native XGBoost bundles."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

RESEARCH_FEATURES=['sensor_0_avg','sensor_1_avg','sensor_43_avg','sensor_11_avg','wind_speed_3_avg','voltage','sensor_50']
PHASES=['sensor_32_avg','sensor_33_avg','sensor_34_avg']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seven_features(raw):
    times=pd.to_datetime(raw.timestamp,utc=True)
    if times.duplicated().any() or not times.is_monotonic_increasing:raise ValueError('Expected one chronological source timeline')
    names=[n for n in RESEARCH_FEATURES if n!='voltage']
    physical=raw[names+PHASES].astype('float32').replace([np.inf,-np.inf],np.nan)
    physical['voltage']=physical[PHASES].sum(axis=1,min_count=3)/3
    delayed=physical[RESEARCH_FEATURES].shift(1)
    continuous=times.diff().eq(pd.Timedelta(minutes=10))&raw.asset_id.eq(raw.asset_id.shift())
    return delayed.where(continuous,np.nan,axis=0).astype('float32')


def fit_xgb(frame,features,train_assets,parameters):
    train=frame.loc[frame.turbine_id.isin(train_assets)&frame.history_ready&frame.label.isin([0,1])]
    if train.empty or set(train.turbine_id)!=set(train_assets):raise ValueError('Missing training turbine rows')
    negative=int(train.label.eq(0).sum());positive=int(train.label.eq(1).sum())
    if min(negative,positive)==0:raise ValueError('Training needs both classes')
    matrix=train[features].astype('float32')
    if np.isinf(matrix.to_numpy()).any():raise ValueError('Infinite feature value')
    if matrix.notna().sum().eq(0).any():raise ValueError('Feature has no observed training values')
    scaler=MinMaxScaler(clip=False).fit(matrix)
    weight=negative/positive
    model=XGBClassifier(**parameters,scale_pos_weight=weight)
    model.fit(scaler.transform(matrix),train.label.astype(int))
    info=dict(training_rows=len(train),negative=negative,positive=positive,scale_pos_weight=weight,train_assets=list(train_assets))
    return model,scaler,info


def export_bundle(model,scaler,features,threshold,metadata,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model.save_model(directory/'booster.ubj')
    scaling=dict(features=features,dtype='float32',clip=False,scale=scaler.scale_.tolist(),offset=scaler.min_.tolist(),
        data_min=scaler.data_min_.tolist(),data_max=scaler.data_max_.tolist())
    (directory/'minmax.json').write_text(json.dumps(scaling,indent=2,allow_nan=False)+'\n')
    manifest=dict(metadata,features=features,threshold=threshold,alerts_disabled=threshold is None,
        scaler_sha256=digest(directory/'minmax.json'),booster_sha256=digest(directory/'booster.ubj'))
    (directory/'inference.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


class NativeAdapter:
    """Native UBJSON + numeric JSON. No pickle, fitting or label access."""
    def __init__(self,directory):
        directory=Path(directory)
        self.metadata=json.loads((directory/'inference.json').read_text())
        for name,key in [('booster.ubj','booster_sha256'),('minmax.json','scaler_sha256')]:
            if digest(directory/name)!=self.metadata[key]:raise ValueError(f'Bundle checksum mismatch: {name}')
        self.features=self.metadata['features']
        scaling=json.loads((directory/'minmax.json').read_text())
        if scaling['features']!=self.features or scaling['clip']:raise ValueError('Invalid scaler schema')
        self.scale=np.asarray(scaling['scale'],dtype='float32');self.offset=np.asarray(scaling['offset'],dtype='float32')
        if len(self.scale)!=len(self.features) or not np.isfinite(self.scale).all() or not np.isfinite(self.offset).all():raise ValueError('Invalid scaler values')
        self.model=XGBClassifier();self.model.load_model(directory/'booster.ubj')
        if self.model.n_features_in_!=len(self.features):raise ValueError('Booster feature count mismatch')

    def predict(self,frame):
        if not set(self.features).issubset(frame):raise ValueError('Missing model features')
        matrix=frame[self.features].to_numpy(dtype='float32',copy=True)
        if np.isinf(matrix).any():raise ValueError('Infinite feature value')
        valid=(~np.isnan(matrix)).any(axis=1) if self.metadata.get('missing_policy')=='xgboost_nan' else (~np.isnan(matrix)).all(axis=1)
        matrix*=self.scale;matrix+=self.offset
        scores=np.full(len(frame),np.nan,dtype='float64')
        if valid.any():scores[valid]=self.model.predict_proba(matrix[valid])[:,1]
        threshold=self.metadata['threshold']
        warning=np.zeros(len(frame),dtype=bool) if threshold is None else scores>=threshold
        return pd.DataFrame(dict(score=scores,warning=warning),index=frame.index)
