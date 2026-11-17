import importlib.util,unittest


class SensorCatalogTests(unittest.TestCase):
    def test_statistics_are_grouped_under_physical_measurement_with_source_key_preserved(self):
        self.assertIsNotNone(importlib.util.find_spec('app.sensor_catalog'),'Human sensor catalog is missing')
        from app.sensor_catalog import describe
        a=describe('power_30_avg');b=describe('power_30_std')
        self.assertEqual(a['physical_key'],b['physical_key'])
        self.assertEqual(a['key'],'power_30_avg');self.assertEqual(b['statistic'],'std')
        self.assertEqual(a['group'],'electrical');self.assertTrue(a['display_name'])

    def test_generator_bearing_is_distinct_from_gearbox_and_unknown_units_are_not_invented(self):
        self.assertIsNotNone(importlib.util.find_spec('app.sensor_catalog'),'Human sensor catalog is missing')
        from app.sensor_catalog import describe
        self.assertEqual(describe('sensor_13_avg')['group'],'generator')
        self.assertEqual(describe('sensor_11_avg')['group'],'gearbox')
        self.assertEqual(describe('sensor_22_avg')['group'],'electrical')
        self.assertNotIn('unit',describe('sensor_22_avg'))


if __name__=='__main__':unittest.main()
