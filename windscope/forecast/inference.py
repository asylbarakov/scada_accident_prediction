"""Causal H7 inference. No labels, event files, fitting, or extra scaling."""
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from scada_pdm.preparation import make_features, feature_metadata, validate_config

MODEL_ID = 'care-h7-logistic-20261116'
MODEL_SHA256 = '60836ffb2881ab639d8507483a0a2be6a5dd0c5b9067f2c94943ca56f97bb892'


def care_state(values, _operation):
    # Matches scada_pdm.care.care_state; selects only physical channels upstream.
    observed = values.notna().any(axis=1)
    return pd.DataFrame({'observed': observed, 'at_risk': observed})


class H7Adapter:
    def __init__(self, model_dir, config_path):
        model_dir = Path(model_dir)
        self.config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        validate_config(self.config)
        if self.config['mode'] != 'care_h7' or self.config['measurement_delay_steps'] != 1:
            raise ValueError('Expected the archived CARE H7 configuration')
        self.metadata = json.loads((model_dir/'inference.json').read_text(encoding='utf-8'))
        self.features = self.metadata['feature_names']
        if self.features != list(feature_metadata(self.config)) or len(self.features) != 210:
            raise ValueError('Feature schema does not match CARE H7')
        self.threshold = self.metadata['threshold']
        if not 0 < self.threshold <= 1 or self.metadata['score_semantics'] != 'uncalibrated_failure_score':
            raise ValueError('Unsupported threshold or score semantics')
        path = model_dir/'pipeline.joblib'
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('Model artifact checksum mismatch')
        # The checksum establishes identity, not pickle safety. Only run in the
        # isolated, unprivileged container with no credentials and no external network.
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            self.pipeline = joblib.load(path)
        if list(self.pipeline.feature_names_in_) != self.features or list(self.pipeline.classes_) != [0, 1]:
            raise ValueError('Pipeline input order or classes do not match the manifest')

    def predict(self, asset_id, source_time, history):
        at = pd.Timestamp(source_time)
        if at.tzinfo is not None or pd.isna(at) or at != at.floor('10min'):
            raise ValueError('Use the original naive, 10-minute dataset timeline')
        if not history or len(history) > 1009:
            raise ValueError('Expected at most 168 hours of physical measurements')
        times = pd.to_datetime([r['source_time'] for r in history], errors='raise')
        if (times.tz is not None or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing
                or times.max() > at or times.min() < at-pd.Timedelta(hours=168)
                or not times.equals(times.floor('10min'))):
            raise ValueError('History must be chronological, unique and no later than the prediction time')
        result = dict(status='warming_up', predicted_class=None, class_1_score=None,
            model_id=MODEL_ID, prediction_horizon='P7D', prediction_horizon_hours=168,
            prediction_interval_start=str(at), prediction_interval_end=str(at+pd.Timedelta(days=7)),
            feature_cutoff=str(at-pd.Timedelta(minutes=10)), threshold=self.threshold,
            score_semantics='uncalibrated_failure_score', cadence_minutes=60,
            required_history_hours=24, dataset_timezone='unknown', experimental=True)
        if times[-1] != at:
            return result
        columns = self.config['sensors']
        raw = pd.DataFrame({name: [r['values'].get(info['column']) for r in history]
                            for name, info in columns.items()}, dtype='float32')
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw.insert(0, 'timestamp', times)
        raw.insert(0, 'turbine_id', str(asset_id))
        features, _, ready = make_features(raw, self.config, state_policy=care_state)
        if not bool(ready.iloc[-1]):
            return result
        score = float(self.pipeline.predict_proba(features[self.features].tail(1))[0, 1])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid model score')
        return dict(result, status='ok', class_1_score=score, predicted_class=int(score >= self.threshold))
