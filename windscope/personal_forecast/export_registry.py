"""Export reviewed native models only; no annotations or prediction CSV are copied."""
import hashlib,json,shutil
from pathlib import Path

source=Path('/research-output');target=Path('/registry');target.mkdir(parents=True,exist_ok=True)
gate=json.loads((source/'data_gate.json').read_text());entries={}
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
for identifier in ('0','25','26','42'):
    original=source/f'event_{identifier}';metadata=json.loads((original/'inference.json').read_text())
    manifest_path=original/'normal_model.json'
    assert sha(manifest_path)==metadata['normal_model_manifest_sha256']
    manifest=json.loads(manifest_path.read_text());destination=target/identifier;destination.mkdir(exist_ok=True)
    shutil.copyfile(manifest_path,destination/'normal_model.json')
    for entry in manifest['models'].values():
        assert sha(original/entry['file'])==entry['sha256'];shutil.copyfile(original/entry['file'],destination/entry['file'])
    source_info=next(row for row in gate['sources'] if str(row['event'])==identifier)
    entries[identifier]=dict(asset_id=metadata['asset'],threshold=metadata['threshold'],prediction_start=metadata['bounds']['prediction_start'],
        training_normal_hours=source_info['training_normal_hours'],calibration_normal_hours=source_info['calibration_normal_hours'],
        baseline_end=source_info['calibration_end'],source_sha256=gate['input_sha256'][f'raw/datasets/{identifier}.csv'],normal_model_sha256=sha(manifest_path))
registry=dict(model_id='care-personal-normal-h24-v1',feature_config=gate['physical_feature_config'],history_hours=8,sources=entries)
(target/'registry.json').write_text(json.dumps(registry,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Exported native historical baselines:',', '.join(entries))
