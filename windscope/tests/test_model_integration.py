"""Run only in the offline restricted container with the reviewed model mounted."""
import os
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('BASELINE_TEST_MODEL'), 'No isolated artifact mount')
class PublishedModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from baseline.inference import ScadaXGBoostAdapter
        cls.adapter = ScadaXGBoostAdapter(Path(os.environ['BASELINE_TEST_MODEL']))

    def measurement(self, **extra):
        from baseline.inference import Measurement
        values = dict(sensor_0_avg=22, sensor_1_avg=302.9, sensor_43_avg=30,
                      sensor_11_avg=39, wind_speed_3_avg=1.7,
                      sensor_32_avg=398.9, sensor_33_avg=397.9, sensor_34_avg=394.3,
                      sensor_50=-1280)
        values.update(extra)
        return Measurement('2022-08-04 06:10:00','0',0,'train','0',values)

    def test_inference_ignores_labels_and_future_metadata(self):
        first = self.adapter.predict(self.measurement(failure=0, future_sensor_0_avg=-999))
        second = self.adapter.predict(self.measurement(failure=1, future_sensor_0_avg=999))
        self.assertEqual(first.status, 'ok')
        self.assertEqual(first.predicted_class, second.predicted_class)
        self.assertEqual(first.class_1_score, second.class_1_score)

    def test_sequential_and_batch_predictions_match_the_original_pipeline(self):
        import pandas as pd
        from baseline.inference import feature_frame
        measurements = [self.measurement(), self.measurement(sensor_43_avg=39,sensor_11_avg=65)]
        frames = pd.concat([feature_frame(m.values) for m in measurements], ignore_index=True)
        expected_classes = self.adapter.pipeline.predict(frames)
        expected_scores = self.adapter.pipeline.predict_proba(frames)
        for i, measurement in enumerate(measurements):
            result = self.adapter.predict(measurement)
            self.assertEqual(result.predicted_class, int(expected_classes[i]))
            self.assertAlmostEqual(result.class_1_score, float(expected_scores[i,1]), places=7)
        self.assertEqual(self.adapter.inspection()['scaler_n_samples_seen'],16000)

    def test_invalid_input_is_not_a_normal_prediction(self):
        result = self.adapter.predict(self.measurement(sensor_32_avg='NaN'))
        self.assertEqual(result.status, 'invalid_input')
        self.assertIsNone(result.predicted_class)


if __name__ == '__main__':
    unittest.main()
