"""Fetch only demo CSVs from CARE v6 and verify every byte against manifests.

Standard library only; does not load models or overwrite mismatched input files.
"""
import hashlib,json,sys
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'project/scripts'))
from fetch_care_a import RemoteZip


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure(path,payload,expected):
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Source checksum mismatch: '+str(path))
    if path.exists():
        if sha(path)!=expected:raise ValueError('Existing source differs; not overwritten: '+str(path))
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)


def main():
    required={}
    for name in ('h7_scenarios_manifest.json','personal_scenarios_manifest.json'):
        for entry in json.loads((ROOT/'data'/name).read_text(encoding='utf-8-sig'))['files']:
            path=entry['local_path'];expected=entry['sha256']
            if path in required and required[path]!=expected:raise ValueError('Conflicting source manifests')
            required[path]=expected
    farm=ROOT/'project/data/CARE_To_Compare/Wind Farm A'
    missing=[]
    for relative,expected in required.items():
        path=farm/relative
        if path.exists():
            if sha(path)!=expected:raise ValueError('Existing source checksum mismatch: '+str(path))
        else:missing.append(relative)
    if missing:
        with RemoteZip() as remote,ZipFile(remote) as archive:
            for relative in missing:
                payload=archive.read('CARE_To_Compare/Wind Farm A/'+relative)
                ensure(farm/relative,payload,required[relative]);print('Verified CARE v6 '+relative,flush=True)
    legacy=json.loads((ROOT/'data/scenarios_manifest.json').read_text(encoding='utf-8-sig'))
    for entry in legacy['files']:
        name=entry['local_path'];destination=ROOT/'data/raw/wind_farm_a_v2'/name
        if destination.exists():
            if sha(destination)!=entry['sha256']:raise ValueError('Existing legacy source differs')
            continue
        original=ROOT/'data/source_readme.md' if name=='README.md' else farm/('datasets/'+name if name.endswith('.csv') and name[:-4].isdigit() else name)
        ensure(destination,original.read_bytes(),entry['sha256'])
    print('Demo inputs verified: '+', '.join(sorted(required)),flush=True)


if __name__=='__main__':main()
