"""Download pinned Linux CPython 3.12 wheels from PyPI; verify published SHA-256.

This preparation utility uses only the standard library and never loads a model.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch_one(requirement):
    name, version = requirement.split('==')
    with urllib.request.urlopen('https://pypi.org/pypi/{}/{}/json'.format(name,version), timeout=40) as response:
        metadata = json.load(response)
    candidates = []
    for file in metadata['urls']:
        filename = file['filename']
        if not filename.endswith('.whl'):
            continue
        _, py, abi, platform = filename[:-4].rsplit('-',3)
        if not (py in ('py3','py2.py3','cp312') and abi in ('none','cp312')):
            continue
        if platform != 'any' and not ('manylinux' in platform and platform.endswith('x86_64')):
            continue
        candidates.append(file)
    if not candidates:
        raise RuntimeError('No Linux CPython 3.12 wheel: '+requirement)
    file = sorted(candidates,key=lambda f: f['filename'])[0]
    destination = ROOT/'.wheelhouse'/file['filename']
    expected = file['digests']['sha256']
    if not destination.exists():
        with urllib.request.urlopen(file['url'], timeout=90) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError('Download checksum mismatch: '+file['filename'])
        destination.write_bytes(payload)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Cached checksum mismatch: '+file['filename'])
    print('Verified '+file['filename'],flush=True)
    return dict(requirement=requirement,filename=file['filename'],sha256=expected,url=file['url'])


def main():
    requirements = [line.strip() for line in (ROOT/'requirements-baseline.lock.txt').read_text(encoding='utf-8-sig').splitlines()
                    if line.strip() and not line.startswith('#')]
    (ROOT/'.wheelhouse').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(fetch_one,requirements))
    (ROOT/'artifacts/ml_baseline').mkdir(parents=True,exist_ok=True)
    (ROOT/'artifacts/ml_baseline/wheel_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
