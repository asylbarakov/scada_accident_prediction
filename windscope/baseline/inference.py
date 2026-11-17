"""Causal, single-measurement adapter for the reviewed SWTFPS SCADA artifact.

Never import the authors' training scripts. This module never fits a scaler/model,
reads event annotations, or constructs a future window.
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Protocol

FEATURE_NAMES = (
    'sensor_0_avg', 'sensor_1_avg', 'sensor_43_avg', 'sensor_11_avg',
    'wind_speed_3_avg', 'voltage', 'sensor_50',
)
PHASE_NAMES = ('sensor_32_avg', 'sensor_33_avg', 'sensor_34_avg')
MODEL_SHA256 = '98584786a6a4421f74e599743378c67aadaeaa67ffbfdd582a14f5d235617fd8'
MODEL_ID = 'swtfps-scada-xgboost-v1.0.0'
PREPROCESSING_VERSION = 'scada-seven-features-v1'


@dataclass(frozen=True)
class Measurement:
    source_time: str
    asset_id: str
    row_id: int
    partition: str
    scenario_id: str
    values: Mapping[str, object]


@dataclass(frozen=True)
class Prediction:
    source_time: str
    asset_id: str
    row_id: int
    partition: str
    scenario_id: str
    status: str
    predicted_class: int | None
    class_0_score: float | None
    class_1_score: float | None
    features: Mapping[str, float] = field(default_factory=dict)
    error: str | None = None
    kind: str = 'model'
    model_id: str = MODEL_ID
    model_sha256: str = MODEL_SHA256
    preprocessing_version: str = PREPROCESSING_VERSION
    schema_version: str = '1.0'
    score_semantics: str = 'uncalibrated_binary_class_score_no_forecast_horizon'
    prediction_horizon: str | None = None

    def to_dict(self):
        return asdict(self)


class PredictionAdapter(Protocol):
    def predict(self, measurement: Measurement) -> Prediction: ...


def _number(values, name):
    raw = values.get(name)
    if raw is None or isinstance(raw, bool):
        raise ValueError(f'Missing or invalid feature: {name}')
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Non-numeric feature: {name}') from exc
    if not math.isfinite(value):
        raise ValueError(f'Non-finite feature: {name}')
    return value


def feature_frame(values: Mapping[str, object]):
    """Select raw SCADA units and derive voltage; no clipping/scaling/fitting."""
    import pandas as pd
    features = {}
    for name in FEATURE_NAMES:
        if name == 'voltage':
            features[name] = sum(_number(values, phase) for phase in PHASE_NAMES) / 3.0
        else:
            features[name] = _number(values, name)
    return pd.DataFrame([features], columns=list(FEATURE_NAMES), dtype='float64')


def iter_measurements(path: Path, partition: str = 'all'):
    """Lazily read primary CSVs. Reject ordering errors instead of sorting future rows."""
    if partition not in ('all', 'train', 'prediction'):
        raise ValueError('partition must be all, train, or prediction')
    path = Path(path)
    if path.name.startswith('comma_'):
        raise ValueError('Only the primary Kaggle v2 CSV group is accepted')
    previous_time = previous_id = asset = None
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        required = {'time_stamp', 'asset_id', 'id', 'train_test'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('Invalid primary CSV schema or delimiter')
        for row in reader:
            timestamp = datetime.fromisoformat(row['time_stamp'])
            row_id = int(row['id'])
            if timestamp.tzinfo is not None:
                raise ValueError('Expected naive source timestamps, not an invented timezone')
            if previous_time is not None and (timestamp <= previous_time or row_id <= previous_id):
                raise ValueError('Measurements must have increasing source times and row IDs')
            if asset is not None and row['asset_id'] != asset:
                raise ValueError('An event scenario must contain one turbine')
            if row['train_test'] not in ('train', 'prediction'):
                raise ValueError('Unknown source partition')
            previous_time, previous_id, asset = timestamp, row_id, row['asset_id']
            if partition != 'all' and row['train_test'] != partition:
                continue
            sensors = {k: v for k, v in row.items()
                       if k.startswith(('sensor_', 'wind_speed_', 'power_', 'reactive_power_'))}
            yield Measurement(row['time_stamp'], row['asset_id'], row_id,
                              row['train_test'], path.stem, sensors)


class ScadaXGBoostAdapter:
    """Load only the reviewed digest; caller must supply OS/container isolation."""
    def __init__(self, model_path: Path):
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from xgboost import XGBClassifier

        model_path = Path(model_path)
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != MODEL_SHA256:
            raise ValueError('Artifact digest differs from the reviewed release; loading refused')
        # This is an explicit executable-format trust boundary. Run only using
        # scripts/run_baseline.ps1 (network none, non-root, read-only mounts).
        model = joblib.load(model_path)
        if not isinstance(model, Pipeline) or list(model.named_steps) != ['scaler', 'xgb']:
            raise ValueError('Expected the published scaler + xgb Pipeline')
        if not isinstance(model.named_steps['scaler'], MinMaxScaler):
            raise ValueError('Unexpected preprocessing')
        if not isinstance(model.named_steps['xgb'], XGBClassifier):
            raise ValueError('Unexpected estimator')
        if tuple(model.feature_names_in_) != FEATURE_NAMES or model.n_features_in_ != 7:
            raise ValueError('The saved feature contract does not match the mapper')
        if model.classes_.tolist() != [0, 1]:
            raise ValueError('Unexpected class mapping')
        # Inference-only resource selection; no fitting or weight changes.
        model.named_steps['xgb'].set_params(n_jobs=1, device='cpu')
        self.pipeline = model

    def predict(self, measurement: Measurement) -> Prediction:
        common = dict(source_time=measurement.source_time, asset_id=measurement.asset_id,
                      row_id=measurement.row_id, partition=measurement.partition,
                      scenario_id=measurement.scenario_id)
        try:
            frame = feature_frame(measurement.values)
        except ValueError as exc:
            return Prediction(**common, status='invalid_input', predicted_class=None,
                              class_0_score=None, class_1_score=None, error=str(exc))
        features = {name: float(frame.iloc[0][name]) for name in FEATURE_NAMES}
        # Use the original estimator's class decision, rather than tuning a threshold.
        label = int(self.pipeline.predict(frame)[0])
        scores = self.pipeline.predict_proba(frame)[0]
        if label not in (0, 1) or not all(math.isfinite(float(s)) for s in scores):
            raise RuntimeError('Invalid model output')
        return Prediction(**common, status='ok', predicted_class=label,
                          class_0_score=float(scores[0]), class_1_score=float(scores[1]),
                          features=features)

    def inspection(self):
        import json
        scaler = self.pipeline.named_steps['scaler']
        estimator = self.pipeline.named_steps['xgb']
        config = json.loads(estimator.get_booster().save_config())
        return dict(model_id=MODEL_ID, artifact_sha256=MODEL_SHA256,
                    feature_order=list(self.pipeline.feature_names_in_),
                    classes=self.pipeline.classes_.tolist(),
                    pipeline_steps=list(self.pipeline.named_steps),
                    scaler_data_min=scaler.data_min_.tolist(), scaler_data_max=scaler.data_max_.tolist(),
                    scaler_scale=scaler.scale_.tolist(), scaler_min=scaler.min_.tolist(),
                    scaler_clip=bool(scaler.clip), scaler_n_samples_seen=int(scaler.n_samples_seen_),
                    objective=estimator.objective, booster_config=config)
