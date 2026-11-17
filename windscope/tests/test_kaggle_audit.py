import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


class KaggleAuditTests(unittest.TestCase):
    def module(self):
        path=Path(__file__).resolve().parents[1]/'scripts/audit_kaggle_v2.py'
        self.assertTrue(path.exists(),'Kaggle inventory and integrity audit is not implemented')
        spec=importlib.util.spec_from_file_location('kaggle_audit',str(path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_inventory_separates_primary_and_comma_and_rejects_duplicate_names(self):
        d=self.module();files=[dict(name=n,totalBytes=3) for n in ['Wind Farm A/datasets/0.csv','Wind Farm A/datasets/comma_0.csv','Wind Farm B/datasets/1.csv']]
        self.assertEqual([f['name'] for f in d.primary_files(files,'A')],['Wind Farm A/datasets/0.csv'])
        with self.assertRaises(ValueError):d.validate_inventory(files+[files[0]])

    def test_matching_remote_bytes_never_overwrite_existing_data(self):
        d=self.module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);local=root/'original.csv';local.write_bytes(b'original')
            result=d.verify_stream(io.BytesIO(b'original'),local,root/'remote.csv',8)
            self.assertTrue(result['matches_local']);self.assertEqual(local.read_bytes(),b'original')
            self.assertFalse((root/'remote.csv').exists())

    def test_different_remote_bytes_are_saved_separately(self):
        d=self.module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);local=root/'original.csv';local.write_bytes(b'original')
            result=d.verify_stream(io.BytesIO(b'remote'),local,root/'remote.csv',6)
            self.assertFalse(result['matches_local']);self.assertEqual(local.read_bytes(),b'original')
            self.assertEqual((root/'remote.csv').read_bytes(),b'remote')

    def test_truncated_download_cannot_become_verified_file(self):
        d=self.module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):d.verify_stream(io.BytesIO(b'short'),root/'absent',root/'remote.csv',99)
            self.assertFalse((root/'remote.csv').exists())

    def test_profile_checks_sensor_missingness_time_gaps_and_annotation_row_ids(self):
        d=self.module()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'0.csv'
            path.write_text('time_stamp;asset_id;id;train_test;status_type_id;sensor_14_avg;power_30_avg\n'
                '2023-01-01 00:00:00;0;0;train;0;20;0.3\n'
                '2023-01-01 00:10:00;0;1;prediction;0;;0.4\n'
                '2023-01-01 00:40:00;0;2;prediction;4;24;0.5\n')
            annotation=dict(asset='0',event_start_id='1',event_end_id='2',event_start='2023-01-01 00:10:00',event_end='2023-01-01 00:40:00')
            result=d.profile_csv(path,annotation,['sensor_14_avg','power_30_avg'])
            self.assertEqual(result['rows'],3);self.assertEqual(result['non_10min_intervals'],1)
            self.assertEqual(result['sensors']['sensor_14_avg']['missing'],1)
            self.assertTrue(result['annotation_ids_match_timestamps'])
            self.assertEqual(result['partitions'],{'train':1,'prediction':2})
            with self.assertRaises(ValueError):d.profile_csv(path,annotation,['absent_sensor'])


if __name__=='__main__':unittest.main()
