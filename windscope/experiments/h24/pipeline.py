"""Physical-only causal features and train-only preprocessing for the experiment."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from scada_pdm.preparation import make_features


def physical_state(values, _operation):
    observed=values.notna().any(axis=1)
    return pd.DataFrame(dict(observed=observed,at_risk=observed))


def physical_features(scada,config):
    # Whitelist physical measurements: annotations/partitions never enter features.
    physical=scada[['turbine_id','timestamp',*config['sensors']]].copy()
    features,_,ready=make_features(physical,config,state_policy=physical_state)
    return features,ready


def assert_lineage_separation(lineage):
    for column in ('avg_fingerprint','full_fingerprint'):
        identity=lineage.dropna(subset=[column])
        if identity.groupby(column,sort=False).turbine_id.nunique().gt(1).any():
            raise ValueError(f'cross-turbine physical reuse: {column}')


def fit_candidate(name,config,frame,features,train_assets):
    train=frame.loc[frame.turbine_id.isin(train_assets)&frame.label.notna()&frame.history_ready]
    if name=='logistic':
        pipeline=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('scaler',StandardScaler()),('model',LogisticRegression(random_state=config['seed'],**config['models'][name]))])
    elif name=='hist_gradient_boosting':
        pipeline=Pipeline([('model',HistGradientBoostingClassifier(random_state=config['seed'],**config['models'][name]))])
    else:raise ValueError(name)
    pipeline.fit(train[features],train.label.astype(int))
    return pipeline
