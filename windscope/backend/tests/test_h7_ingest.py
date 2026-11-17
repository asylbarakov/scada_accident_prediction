import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from app.models import Base, Scenario, Measurement, ReplaySession
from app import ingest


class H7ImportTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://')
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'datasets').mkdir()
        (self.root/'event_info.csv').write_text('event_id;asset;event_start_id;event_end_id;event_start;event_end;event_label;event_description\n22;21;1;2;2023-08-12 10:00:00;2023-08-12 10:10:00;anomaly;source label\n')
        (self.root/'feature_description.csv').write_text('sensor_name;description;unit\nsensor_0;Ambient;C\n')
        self.csv_path = self.root/'datasets/22.csv'
        with self.csv_path.open('w', newline='') as f:
            columns = ['id', 'asset_id', 'time_stamp', 'train_test', 'status_type_id'] + [f'sensor_{i}_avg' for i in range(81)]
            writer = csv.DictWriter(f, columns, delimiter=';')
            writer.writeheader()
            for i, at in enumerate(['2023-08-12 09:50:00', '2023-08-12 10:00:00', '2023-08-12 10:10:00']):
                writer.writerow(dict(id=i, asset_id='21', time_stamp=at, train_test='train' if i==0 else 'prediction', status_type_id=1,
                                     **{f'sensor_{j}_avg': j+i for j in range(81)}))
        self.manifest = {'files': [{'local_path': p, 'sha256': hashlib.sha256((self.root/p).read_bytes()).hexdigest()}
                         for p in ['event_info.csv', 'feature_description.csv', 'datasets/22.csv']]}

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        self.temp.cleanup()

    def test_import_is_idempotent_and_preserves_source_timestamps_and_labels(self):
        self.assertTrue(hasattr(ingest, 'import_scenarios'), 'Additional H7 source groups cannot be imported')
        for _ in range(2):
            ingest.import_scenarios(self.db, self.root, self.manifest, ['22'], model_id='care-h7-logistic-20261116', dataset_subdir='datasets')
        self.assertEqual(self.db.scalar(select(func.count()).select_from(Measurement)), 3)
        self.assertEqual(self.db.scalar(select(func.count()).select_from(ReplaySession)), 1)
        self.assertEqual(self.db.get(Scenario, '22').annotation['description'], 'source label')
        self.assertEqual(str(self.db.get(Measurement, ('22', 0)).source_at), '2023-08-12 09:50:00')

    def test_changed_raw_data_is_rejected_before_inserting_rows(self):
        self.assertTrue(hasattr(ingest, 'import_scenarios'), 'H7 source checks are missing')
        with self.csv_path.open('a') as stream:stream.write('\n')
        with self.assertRaises(ValueError):
            ingest.import_scenarios(self.db, self.root, self.manifest, ['22'], model_id='care-h7-logistic-20261116', dataset_subdir='datasets')
        self.assertEqual(self.db.scalar(select(func.count()).select_from(Scenario)), 0)
