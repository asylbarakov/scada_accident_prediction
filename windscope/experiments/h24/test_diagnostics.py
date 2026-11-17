import unittest
import pandas as pd
from diagnostics import summarize


class OperationalTests(unittest.TestCase):
    def test_prior_alarm_is_not_reported_as_new_24h_warning(self):
        fault=pd.Timestamp('2024-01-03T00:00Z')
        points=pd.DataFrame(dict(turbine_id=['a']*4,source_event=[1]*4,timestamp=[fault-pd.Timedelta(hours=h) for h in [30,25,24,23]],score=[.9]*4,threshold=[.8]*4,history_ready=[True]*4,label=pd.array([None,0,1,1],dtype='Int8')))
        report=summarize(points,[dict(id='e',source_event=1,turbine_id='a',failure_at=fault.isoformat())])
        event=report['events'][0]
        self.assertTrue(event['warning_before_target_window'])
        self.assertEqual(event['first_source_warning_lead_hours'],30.)
        self.assertEqual(report['per_asset'][0]['normal_warning_fraction'],1.)
        self.assertEqual(event['first_target_warning_lead_hours'],24.)


if __name__=='__main__':unittest.main()
