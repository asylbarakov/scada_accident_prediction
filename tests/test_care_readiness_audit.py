"""Synthetic-unit tests: no CARE download required."""
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from care_readiness_audit import (horizon_label, load_event_info,
                                  parse_timestamp, scan_prediction_timestamps, summarize_event)

UTC = timezone.utc


class TestCareReadiness(unittest.TestCase):
    def test_failure_anchor_and_horizon(self):
        fault = datetime(2022, 2, 1, tzinfo=UTC)
        nine_days_before = fault - timedelta(days=9)
        self.assertEqual(horizon_label(nine_days_before, 7, failure_at=fault), 0)
        self.assertEqual(horizon_label(nine_days_before, 30, failure_at=fault), 1)
        self.assertEqual(horizon_label(fault - timedelta(days=7), 7, failure_at=fault), 1)
        self.assertEqual(horizon_label(fault - timedelta(days=7, seconds=1), 7, failure_at=fault), 0)
        self.assertIsNone(horizon_label(fault, 7, failure_at=fault))
        self.assertEqual(horizon_label(fault - timedelta(days=30), 30, failure_at=fault), 1)
        self.assertEqual(horizon_label(fault - timedelta(days=30, seconds=1), 30, failure_at=fault), 0)

    def test_right_censoring_on_normal_event(self):
        t = datetime(2022, 1, 1, tzinfo=UTC)
        end = t + timedelta(days=14)
        self.assertEqual(horizon_label(t, 7, normal_observed_until=end), 0)
        self.assertEqual(horizon_label(t, 7, normal_observed_until=t + timedelta(days=7)), 0)
        self.assertIsNone(horizon_label(t, 30, normal_observed_until=end))
        self.assertIsNone(horizon_label(t, 7))

    def test_timestamp_parsing(self):
        a = parse_timestamp('2022-01-01 00:00:00')
        b = parse_timestamp('2022-01-01T00:00:00Z')
        self.assertEqual(a, b)

    def test_csv_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            farm = Path(temp) / 'Wind Farm A'
            (farm / 'datasets').mkdir(parents=True)
            with (farm / 'event_info.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f, delimiter=';')
                w.writerow(['event_id', 'event_label', 'event_start', 'event_end', 'asset'])
                w.writerow([42, 'anomaly', '2022-01-02 00:00:00', '2022-01-03 00:00:00', 'T01'])
            info = load_event_info(Path(temp), 'A')
            self.assertEqual(len(info), 1)
            self.assertEqual(info[0]['event_id'], 42)
            with (farm / 'datasets' / '42.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f, delimiter=';')
                w.writerow(['time_stamp', 'train_test', 'sensor_1_avg'])
                w.writerow(['2022-01-01 00:00:00', 'train', 1])
                w.writerow(['2022-01-02 00:00:00', 'prediction', 2])
                w.writerow(['2022-01-02 00:10:00', 'prediction', 3])
            timestamps, counts = scan_prediction_timestamps(farm / 'datasets' / '42.csv')
            self.assertEqual(counts['train'], 1)
            self.assertEqual(counts['prediction'], 2)
            self.assertEqual(len(timestamps), 2)


    def test_failure_boundary_and_status_invariance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'datasets'
            folder.mkdir()
            start = datetime(2022, 2, 1, tzinfo=UTC)
            event = {'event_id': 1, 'event_label': 'anomaly', 'asset': 'T01',
                     'event_start': start + timedelta(minutes=10), 'event_start_id': '1',
                     'event_end': start + timedelta(minutes=30), 'event_end_id': '3'}
            for status in (0, 3, 4):
                with self.subTest(status=status):
                    with (folder / '1.csv').open('w', newline='', encoding='utf-8') as f:
                        w = csv.writer(f, delimiter=';')
                        w.writerow(['time_stamp', 'id', 'train_test', 'status_type_id', 'sensor_1_avg'])
                        for index in range(5):
                            t = start + timedelta(minutes=10 * index)
                            w.writerow([t.isoformat(), index, 'prediction', status, index])
                    result = summarize_event(event, root, (7, 30), True)
                    self.assertEqual(result['fault_anchor'], 'event_end')
                    self.assertEqual(result['boundary_checks']['event_end'], 'verified')
                    for h in (7, 30):
                        self.assertEqual(result[f'h{h}d'],
                                         {'positive': 3, 'negative': 0, 'censored': 0, 'excluded': 2})
            event['event_end_id'] = '4'
            result = summarize_event(event, root, (7, 30), True)
            self.assertEqual(result['boundary_checks']['event_end'], 'row_id_mismatch')
            self.assertEqual(result['h7d']['excluded'], 5)
            del event['event_end_id']
            result = summarize_event(event, root, (7, 30), True)
            self.assertEqual(result['boundary_checks']['event_end'], 'timestamp_only')
            self.assertEqual(result['h7d']['excluded'], 5)
            event.update(event_end=start + timedelta(minutes=50), event_end_id='5')
            result = summarize_event(event, root, (7, 30), True)
            self.assertEqual(result['boundary_checks']['event_end'], 'timestamp_absent_from_prediction')
            self.assertEqual(result['h30d']['positive'], 0)

    def test_normal_padding_and_single_missing_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'datasets'
            folder.mkdir()
            start = datetime(2022, 2, 1, tzinfo=UTC)
            week = 7 * 24 * 6
            with (folder / '1.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f, delimiter=';')
                w.writerow(['time_stamp', 'id', 'train_test'])
                for index in range(-1, week + 5):
                    if index != 1:
                        w.writerow([start + timedelta(minutes=10 * index), index, 'prediction'])
            event = {'event_id': 1, 'event_label': 'normal',
                     'event_start': start, 'event_start_id': '0',
                     'event_end': start + timedelta(minutes=10 * (week + 2)),
                     'event_end_id': str(week + 2)}
            result = summarize_event(event, root, (7, 30), True)
            self.assertEqual(result['cadence_breaks'], 1)
            self.assertEqual(result['h7d'],
                             {'positive': 0, 'negative': 1, 'censored': week + 1, 'excluded': 3})
            self.assertEqual(result['h30d']['negative'], 0)
            event['event_label'] = 'anomaly'
            event['event_end'] += timedelta(minutes=10)
            event['event_end_id'] = str(week + 3)
            result = summarize_event(event, root, (7, 30), True)
            self.assertEqual(result['h7d']['negative'], 0)
            self.assertEqual(result['h7d']['censored'], 3)
            self.assertEqual(result['unconfirmed_negative_candidates_in_anomaly_events']['h7d'], 1)


if __name__ == '__main__':
    unittest.main()
