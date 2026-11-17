"""Model identities and the physical-only H7 wire contract (no ML dependencies)."""
from datetime import datetime

LEGACY_MODEL = 'swtfps-scada-xgboost-v1.0.0'
H7_MODEL = 'care-h7-logistic-20261116'
PERSONAL_MODEL = 'care-personal-normal-h24-v1'
PERSONAL_SOURCES = frozenset(('0','25','26','42'))
H7_SENSORS = ('wind_speed_3_avg', 'sensor_0_avg', 'power_29_avg', 'sensor_11_avg',
              'sensor_12_avg', 'sensor_13_avg', 'sensor_18_avg', 'sensor_38_avg',
              'sensor_41_avg', 'sensor_52_avg')


def forecast_due(source_time):
    at = datetime.fromisoformat(source_time)
    return at.minute == 0 and at.second == 0 and at.microsecond == 0


def model_info(identifier):
    if identifier == PERSONAL_MODEL:
        return dict(id=identifier,name='Персональная модель · ранний сигнал',horizon_hours=24,
                    cadence_minutes=60,score_semantics='personal_normal_deviation_not_probability',experimental=True)
    if identifier == H7_MODEL:
        return dict(id=identifier, name='Logistic Regression · H7', horizon_hours=168,
                    cadence_minutes=60, score_semantics='uncalibrated_failure_score', experimental=True)
    if identifier == LEGACY_MODEL:
        return dict(id=identifier, name='XGBoost · исходный baseline', horizon_hours=None,
                    cadence_minutes=10, score_semantics='uncalibrated_class_score', experimental=True)
    raise ValueError('Неизвестная модель')
