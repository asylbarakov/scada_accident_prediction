"""Past-only physical relations and normal-only conditional regression."""
import sys
import numpy as np
import pandas as pd
sys.path.extend(['/personal-code','/feature-code','/extended-code'])
from personal import PersonalModel,smooth
from pipeline import physical_features
from extended import delayed_targets

RELATION_TARGETS=['generator_bearing_delta__now','transformer_phase_spread__now']
POWER_FEATURES=['grid_active_power__now','grid_active_power__1h_mean','grid_active_power__6h_mean']


def physical_relations(raw):
    mapping={'de':'sensor_13_avg','nde':'sensor_14_avg','l1':'sensor_38_avg','l2':'sensor_39_avg','l3':'sensor_40_avg'}
    delayed=delayed_targets(raw,mapping)
    phase=delayed[['l1','l2','l3']]
    result=pd.DataFrame({'generator_bearing_delta__now':delayed.de-delayed.nde,
        'transformer_phase_spread__now':(phase.max(axis=1)-phase.min(axis=1)).where(phase.notna().all(axis=1))},index=raw.index)
    scada=raw[['turbine_id','timestamp']].copy()
    scada['grid_active_power']=pd.to_numeric(raw.power_30_avg,errors='raise').replace([np.inf,-np.inf],np.nan).astype('float32')
    config=dict(cadence_minutes=10,measurement_delay_steps=1,windows_hours=[1,6],required_history_hours=6,operation=None,
        sensors={'grid_active_power':dict(column='power_30_avg',measurement='Grid power',unit='anonymized scale')})
    power,ready=physical_features(scada,config)
    return result.join(power[POWER_FEATURES]).astype('float32'),ready


def fit_model(frame,config):
    model=PersonalModel(config).fit(frame)
    eligible=frame.normal&frame.history_ready&frame[config['required_current_inputs']].notna().all(axis=1)
    for target in config.get('two_sided_targets',[]):
        cal=frame.loc[eligible&frame.role.eq('calibration')&frame[target].notna()]
        residual=cal[target]-model.regressors[target].predict(cal[config['regime_features']].astype('float32'))
        center=model.calibration[target]['center_celsius']
        model.calibration[target]['scale_celsius']=max(float((residual-center).abs().quantile(.95)),1.)
        model.calibration[target]['two_sided']=True
    return model


def score_model(model,frame):
    result=model.predict(frame)
    channels=[]
    for target in model.config['targets']:
        values=result[target.removesuffix('__now')+'__normalized_residual']
        channels.append(values.abs() if target in model.config.get('two_sided_targets',[]) else values)
    z=pd.concat(channels,axis=1)
    valid=frame.history_ready&frame[model.config['required_current_inputs']].notna().all(axis=1)
    result['raw_score']=z.max(axis=1).clip(lower=0).where(valid&z.notna().sum(axis=1).ge(model.config['minimum_target_count']))
    result['score']=result.raw_score
    result['score']=smooth(result,model.config['score_smoothing_hours'])
    return result


def combine(base,relations):
    config=dict(base.config,targets=base.config['targets']+RELATION_TARGETS,two_sided_targets=['generator_bearing_delta__now'])
    model=PersonalModel(config)
    model.regressors={**base.regressors,**relations.regressors};model.calibration={**base.calibration,**relations.calibration}
    return model
