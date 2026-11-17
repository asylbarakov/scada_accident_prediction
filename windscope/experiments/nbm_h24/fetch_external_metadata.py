"""Public-data discovery only. Download four metadata files, no models or SCADA."""
import hashlib
import json
import sys
from pathlib import Path,PurePosixPath
from zipfile import ZipFile
import pandas as pd

sys.path.insert(0,'/source-tools')
import fetch_care_a

fetch_care_a.TRANSFER_LIMIT=1024*1024
out=Path('/output')/'external_metadata'
members={f'CARE_To_Compare/Wind Farm {farm}/{name}' for farm in ('B','C') for name in ('event_info.csv','feature_description.csv')}
files=[];farms=[]
with fetch_care_a.RemoteZip() as remote,ZipFile(remote) as archive:
    existing={entry.filename:entry for entry in archive.infolist() if entry.filename in members}
    if set(existing)!=members:raise ValueError('Unexpected external metadata layout')
    for name,entry in sorted(existing.items()):
        data=archive.read(entry)
        relative=PurePosixPath(name).relative_to('CARE_To_Compare')
        path=out.joinpath(*relative.parts);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=data:raise ValueError('Existing metadata differs')
        path.write_bytes(data)
        files.append(dict(member=name,bytes=len(data),crc32=f'{entry.CRC:08x}',sha256=hashlib.sha256(data).hexdigest()))
    transferred=remote.transferred
for farm in ('B','C'):
    events=pd.read_csv(out/f'Wind Farm {farm}/event_info.csv',sep=';')
    features=pd.read_csv(out/f'Wind Farm {farm}/feature_description.csv',sep=';')
    asset='asset' if 'asset' in events else 'asset_id'
    farms.append(dict(farm=farm,event_files=len(events),turbines=int(events[asset].nunique()),
        event_label_counts={str(k):int(v) for k,v in events.event_label.value_counts().items()},
        described_sensors=len(features),feature_metadata_columns=features.columns.tolist(),
        raw_scada_downloaded=False,normal_model_compatible=False,sensor_mapping_verified=False))
manifest=dict(status='metadata_prepared_scada_not_downloaded',source='https://zenodo.org/records/15846963',
    archive_md5_declared_by_source='2547b58c21ac8c242d13232860cf500c',archive_size_bytes=fetch_care_a.ARCHIVE_SIZE,
    integrity='individual ZIP CRC32 plus local SHA256; entire archive MD5 not recomputed',
    network_bytes=transferred,files=files,farms=farms,
    source_notice='CARE v6, Gück and Roelofs, CC BY-SA 4.0; preserve attribution for dataset derivatives.',
    limitations=['Metadata is not a prepared external test set.','Sensor names/units and raw lineage require review before transfer.','No models were loaded and no predictions were generated during this network-enabled discovery.'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(manifest,indent=2))
