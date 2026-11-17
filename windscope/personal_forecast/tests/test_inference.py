import json
import unittest
from pathlib import Path
import pandas as pd
from personal_forecast.inference import PersonalAdapter,MODEL_ID


class PersonalInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter=PersonalAdapter('/registry')
        cls.raw=pd.read_csv('/research/data/CARE_To_Compare/Wind Farm A/datasets/26.csv',sep=';')
        cls.times=pd.to_datetime(cls.raw.time_stamp)

    def payload(self,at):
        at=pd.Timestamp(at);rows=self.raw.loc[self.times.between(at-pd.Timedelta(hours=8),at)]
        return dict(source_id='26',asset_id='0',source_time=str(at),history=[dict(source_time=r['time_stamp'],values={key:None if pd.isna(r[key]) else float(r[key]) for key in self.adapter.sensors}) for r in rows.to_dict('records')])

    def test_real_online_prefix_matches_frozen_experiment_and_warns(self):
        archive=pd.read_csv('/research-output/event_26/prediction_scores.csv')
        archive['at']=pd.to_datetime(archive.timestamp,utc=True).dt.tz_localize(None)
        for at in ['2023-10-19 06:00:00','2023-10-19 08:00:00','2023-10-19 10:00:00']:
            result=self.adapter.predict(**self.payload(at))
            expected=archive.loc[archive['at'].eq(pd.Timestamp(at))].iloc[0]
            self.assertEqual(result['status'],'ok');self.assertEqual(result['model_id'],MODEL_ID)
            self.assertAlmostEqual(result['deviation_score'],expected.score,places=10)
            self.assertEqual(result['predicted_class'],int(expected.warning))
            self.assertEqual(result['prediction_horizon_hours'],24)
            self.assertAlmostEqual(result['warning_index'],result['deviation_score']/result['threshold'])
            self.assertTrue(result['evidence'])

    def test_future_samples_wrong_asset_and_labels_are_rejected(self):
        payload=self.payload('2023-10-19 08:00:00')
        payload['history'][-1]['source_time']='2023-10-19 08:10:00'
        with self.assertRaises(ValueError):self.adapter.predict(**payload)
        payload=self.payload('2023-10-19 08:00:00');payload['asset_id']='21'
        with self.assertRaises(ValueError):self.adapter.predict(**payload)
        payload=self.payload('2023-10-19 08:00:00');payload['history'][0]['values']['failure']=1
        with self.assertRaises(ValueError):self.adapter.predict(**payload)

    def test_missing_history_returns_unknown_and_history_cannot_precede_deployment(self):
        payload=self.payload('2023-10-19 08:00:00');payload['history']=payload['history'][-4:]
        result=self.adapter.predict(**payload)
        self.assertEqual(result['status'],'warming_up');self.assertIsNone(result['predicted_class'])
        payload=self.payload('2023-01-01 08:00:00')
        with self.assertRaises(ValueError):self.adapter.predict(**payload)


if __name__=='__main__':unittest.main()
