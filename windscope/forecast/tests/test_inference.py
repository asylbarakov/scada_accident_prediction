"""Real artifact and raw-history tests; run only in the restricted ML container."""
import importlib.util
import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class H7InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ.get('RESEARCH_ROOT', '/research'))
        cls.run_dir = cls.root / 'results/care_h7_windows_20261116/training'

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('forecast.inference'), 'H7 adapter is missing')
        from forecast.inference import H7Adapter
        self.adapter = H7Adapter(self.run_dir / 'h7/logistic', self.run_dir / 'config.json')

    def history(self, event, at):
        config = self.adapter.config
        channels = [v['column'] for v in config['sensors'].values()]
        raw = pd.read_csv(self.root / f'data/CARE_To_Compare/Wind Farm A/datasets/{event}.csv',
                          sep=';', usecols=['time_stamp', *channels])
        times = pd.to_datetime(raw['time_stamp'])
        at = pd.Timestamp(at).tz_localize(None)
        raw = raw.loc[times.between(at - pd.Timedelta(hours=168), at)]
        return [{'source_time': r['time_stamp'], 'values': {k: None if pd.isna(r[k]) else float(r[k]) for k in channels}}
                for r in raw.to_dict('records')]

    def test_raw_history_reproduces_archived_held_out_predictions(self):
        # Removing the lag, changing a unit/column, or changing rolling statistics must fail this.
        saved = pd.read_parquet(self.run_dir / 'h7/logistic/test_predictions.parquet')
        for event in (13, 22, 72):
            points = saved.loc[saved['source_event'].eq(event)]
            self.assertGreater(len(points), 0)
            for index in (0, len(points)//2, len(points)-1):
                row = points.iloc[index]
                with self.subTest(event=event, at=str(row['timestamp'])):
                    at = str(row['timestamp'].tz_localize(None))
                    result = self.adapter.predict('21', at, self.history(event, at))
                    self.assertEqual(result['status'], 'ok')
                    self.assertAlmostEqual(result['class_1_score'], float(row['score']), delta=1e-10)
                    self.assertEqual(result['predicted_class'], int(row['alert']))
                    self.assertEqual(result['prediction_horizon_hours'], 168)
                    self.assertEqual(result['score_semantics'], 'uncalibrated_failure_score')

    def test_current_values_and_unknown_metadata_cannot_change_lagged_prediction(self):
        at = '2023-08-12 10:00:00'
        history = self.history(22, at)
        first = self.adapter.predict('21', at, history)
        history[-1]['values'] = {k: 999999.0 for k in history[-1]['values']}
        second = self.adapter.predict('21', at, history)
        self.assertEqual(first['class_1_score'], second['class_1_score'])
        self.assertEqual(first['feature_cutoff'], '2023-08-12 09:50:00')

    def test_future_and_duplicate_measurements_are_rejected(self):
        history = self.history(22, '2023-08-12 10:00:00')
        with self.assertRaises(ValueError):
            self.adapter.predict('21', '2023-08-12 09:50:00', history)
        with self.assertRaises(ValueError):
            self.adapter.predict('21', '2023-08-12 10:00:00', history + [history[-1]])

    def test_gap_requires_new_history_and_never_implies_normal(self):
        history = self.history(22, '2023-08-12 10:00:00')
        # Preserve only the last hour after a gap: a 24-hour window cannot be ready.
        result = self.adapter.predict('21', '2023-08-12 10:00:00', history[-7:])
        self.assertEqual(result['status'], 'warming_up')
        self.assertIsNone(result['predicted_class'])
        self.assertIsNone(result['class_1_score'])

    def test_pipeline_reproduces_every_archived_test_score_without_refitting(self):
        saved = pd.read_parquet(self.run_dir / 'h7/logistic/test_predictions.parquet')
        cache = pd.read_parquet(self.root / 'cache/care_h7_v6/features.parquet')
        cache = cache.reset_index(drop=True)
        cache['timestamp'] = pd.to_datetime(cache['timestamp'], utc=True)
        merged = saved.merge(cache, on=['turbine_id', 'source_event', 'timestamp'], validate='one_to_one')
        self.assertEqual(len(merged), 644)
        actual = self.adapter.pipeline.predict_proba(merged[self.adapter.features])[:, 1]
        np.testing.assert_allclose(actual, merged['score'], rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
