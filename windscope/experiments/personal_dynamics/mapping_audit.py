"""Read-only sensor semantics audit; preserves the metadata's original unit text."""
import csv
import hashlib
import json
from pathlib import Path

source=Path('/research/data/CARE_To_Compare/Wind Farm A/feature_description.csv')
with source.open(encoding='utf-8-sig',newline='') as file:
    metadata={r['sensor_name']:r for r in csv.DictReader(file,delimiter=';')}
config=json.loads(Path('/research/configs/care_h7.json').read_text())
targets=['sensor_11','sensor_12','sensor_13','sensor_41']
channels=['power_29','power_30','sensor_13','sensor_14','sensor_38','sensor_39','sensor_40']
report=dict(scope='verified metadata and current configuration; selection hypotheses require a new experiment',metadata_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    channels=[dict(sensor=s,description=metadata[s]['description'],declared_unit=metadata[s]['unit'],personal_temperature_target=s in targets) for s in channels],
    current_power_mapping=config['sensors']['grid_active_power'],
    power_mapping_note='power_29 is Possible grid active power, power_30 is Grid power; verify availability and numeric scaling before interpreting either as actual load. Current config unit p.u. differs from metadata kW; no conversion introduced in this experiment.',
    unit_encoding_note='Original temperature unit cells contain a replacement character; preserved as supplied, not repaired by guessing.')
Path('/output/sensor_mapping_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Sensor mapping audit saved; no sensor mapping or units were changed.')
