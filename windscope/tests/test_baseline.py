"""Contracts: no extra scaling, no future rows/labels, honest offline metrics."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


class BaselineContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('baseline.inference'),
                             'The inference module has not been implemented')

    def test_feature_mapper_preserves_units_order_and_uses_all_three_phases(self):
        from baseline.inference import feature_frame
        record = dict(sensor_0_avg=12, sensor_1_avg=270, sensor_43_avg=35,
                      sensor_11_avg=50, wind_speed_3_avg=6,
                      sensor_32_avg=390, sensor_33_avg=400, sensor_34_avg=410,
                      sensor_50=125000, failure=1, event_start='2099-01-01')
        frame = feature_frame(record)
        self.assertEqual(list(frame.columns), ['sensor_0_avg', 'sensor_1_avg',
            'sensor_43_avg', 'sensor_11_avg', 'wind_speed_3_avg', 'voltage', 'sensor_50'])
        self.assertEqual(frame.iloc[0].tolist(), [12, 270, 35, 50, 6, 400, 125000])

    def test_mapper_rejects_missing_nonfinite_and_boolean_values(self):
        from baseline.inference import feature_frame
        base = dict(sensor_0_avg=12, sensor_1_avg=270, sensor_43_avg=35,
                    sensor_11_avg=50, wind_speed_3_avg=6,
                    sensor_32_avg=390, sensor_33_avg=400, sensor_34_avg=410,
                    sensor_50=125000)
        for value in (None, 'NaN', 'inf', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                feature_frame(dict(base, sensor_32_avg=value))
        del base['sensor_11_avg']
        with self.assertRaises(ValueError):
            feature_frame(base)

    def test_reader_is_lazy_and_rejects_nonchronological_rows(self):
        from baseline.inference import iter_measurements
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / '0.csv'
            p.write_text('time_stamp;asset_id;id;train_test;status_type_id;'
                         'sensor_0_avg\n'
                         '2023-01-01 00:10:00;0;0;train;0;12\n'
                         '2023-01-01 00:00:00;0;1;prediction;4;99\n', encoding='utf-8')
            records = iter_measurements(p)
            self.assertEqual(next(records).source_time, '2023-01-01 00:10:00')
            with self.assertRaises(ValueError):
                next(records)

    def test_reader_preserves_split_ids_and_sensor_values_without_labels(self):
        from baseline.inference import iter_measurements
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / '25.csv'
            p.write_text('time_stamp;asset_id;id;train_test;status_type_id;'
                         'sensor_50;failure\n'
                         '2023-01-01 00:00:00;11;3;train;0;125000;1\n'
                         '2023-01-01 00:10:00;11;4;prediction;4;125001;0\n', encoding='utf-8')
            record = next(iter_measurements(p, partition='prediction'))
            self.assertEqual((record.asset_id, record.row_id, record.partition), ('11', 4, 'prediction'))
            self.assertEqual(record.values, {'sensor_50': '125001'})


class OfflineEvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('baseline.evaluation'),
                             'The offline evaluation module has not been implemented')

    def records(self):
        return [dict(source_time='2023-01-01 '+t+':00', predicted_class=c,
                     status='ok', partition='prediction') for t,c in
                [('00:00',0),('00:10',1),('00:20',1),('00:40',1),('00:50',0)]]

    def test_fault_comparison_counts_gap_separated_episodes_and_annotation_lead(self):
        from baseline.evaluation import summarize
        result = summarize(self.records(), dict(event_label='anomaly',
            event_start='2023-01-01 00:30:00', event_end='2023-01-01 00:40:00'))
        self.assertEqual(result['warning_points'], 3)
        self.assertEqual(result['warning_episodes'], 2)
        self.assertEqual(result['warnings_before_interval'], 2)
        self.assertEqual(result['warnings_during_interval'], 1)
        self.assertEqual(result['first_warning_time'], '2023-01-01 00:10:00')
        self.assertEqual(result['lead_to_annotation_start_minutes'], 20)
        self.assertEqual(result['confusion'], {'tn': 2, 'fp': 2, 'fn': 0, 'tp': 1})

    def test_normal_scenario_reports_false_alert_proxy_without_fault_lead(self):
        from baseline.evaluation import summarize
        result = summarize(self.records(), dict(event_label='normal',
            event_start='2023-01-01 00:00:00', event_end='2023-01-01 00:50:00'))
        self.assertEqual(result['normal_false_alert_points_proxy'], 3)
        self.assertIsNone(result['lead_to_annotation_start_minutes'])
        self.assertEqual(result['confusion'], {'tn': 2, 'fp': 3, 'fn': 0, 'tp': 0})

    def test_no_warning_does_not_invent_lead_time(self):
        from baseline.evaluation import summarize
        records = self.records()
        for row in records:
            row['predicted_class'] = 0
        result = summarize(records, dict(event_label='anomaly',
            event_start='2023-01-01 00:30:00', event_end='2023-01-01 00:40:00'))
        self.assertIsNone(result['first_warning_time'])
        self.assertIsNone(result['lead_to_annotation_start_minutes'])
        self.assertEqual(result['confusion']['fn'], 1)


if __name__ == '__main__':
    unittest.main()
