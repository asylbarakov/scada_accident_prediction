import importlib.util
import tempfile
import unittest
from pathlib import Path


class OutputContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('baseline.run'),
                             'The sequential runner has not been implemented')

    def test_invalid_point_is_recorded_without_fabricated_normal_class(self):
        from baseline.inference import Prediction
        from baseline.run import csv_record
        result = csv_record(Prediction('2023-01-01 00:00:00', '0', 1, 'prediction',
            '0', 'invalid_input', None, None, None, error='missing feature'))
        self.assertIsNone(result['predicted_class'])
        self.assertIsNone(result['class_1_score'])
        self.assertEqual(result['status'], 'invalid_input')

    def test_csv_record_has_structured_provenance_and_unscaled_features(self):
        from baseline.inference import Prediction
        from baseline.run import csv_record
        result = csv_record(Prediction('2023-01-01 00:00:00', '11', 4, 'prediction',
            '25', 'ok', 1, .2, .8, features={'sensor_50':125000, 'voltage':400}))
        self.assertEqual(result['sensor_50'], 125000)
        self.assertEqual(result['voltage'], 400)
        self.assertEqual(result['scenario_id'], '25')
        self.assertEqual(result['kind'], 'model')
        self.assertEqual(result['prediction_horizon'], None)


if __name__ == '__main__':
    unittest.main()
