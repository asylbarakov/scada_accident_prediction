import importlib.util
import unittest
from datetime import datetime, timezone, timedelta


class ReplayClockTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('app.domain'), 'Replay domain is not implemented')

    def test_pause_and_speed_preserve_remaining_source_interval(self):
        from app.domain import remaining_source, next_due
        now = datetime(2026,1,1,tzinfo=timezone.utc)
        remaining = remaining_source(now+timedelta(seconds=3),now,600)
        self.assertEqual(remaining,1800)
        self.assertEqual(next_due(now,remaining,3600),now+timedelta(seconds=.5))

    def test_due_schedule_uses_actual_gap_without_inventing_samples(self):
        from app.domain import schedule_after
        wall = datetime(2026,1,1,tzinfo=timezone.utc)
        source = datetime(2023,8,4,6,10)
        self.assertEqual(schedule_after(wall,source,source+timedelta(minutes=390),600),
                         wall+timedelta(seconds=39))

    def test_future_annotation_is_hidden_and_future_end_is_redacted(self):
        from app.domain import observable_annotation
        annotation={'start':'2023-08-06 06:10:00','end':'2023-08-20 06:10:00','description':'bearing'}
        self.assertIsNone(observable_annotation(annotation,datetime(2023,8,5)))
        result=observable_annotation(annotation,datetime(2023,8,7))
        self.assertIsNone(result['end'])
        self.assertEqual(result['visible_until'],'2023-08-07 00:00:00')

    def test_class_one_never_becomes_critical_or_calibrated_probability(self):
        from app.domain import model_state
        self.assertEqual(model_state({'status':'ok','predicted_class':1}),'warning')
        self.assertEqual(model_state({'status':'invalid_input','predicted_class':None}),'unknown')

    def test_damaged_degree_symbol_is_repaired_only_for_known_unit_shapes(self):
        import app.domain as domain
        self.assertTrue(callable(getattr(domain,'display_unit',None)))
        self.assertEqual(domain.display_unit('\ufffdC'),'°C')
        self.assertEqual(domain.display_unit('Wh'),'Wh')
        self.assertEqual(domain.display_unit('unknown\ufffd'),'unknown\ufffd')


if __name__=='__main__':unittest.main()
