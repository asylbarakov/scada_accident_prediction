"""Audit official Kaggle v2 inventory and verify every primary Farm A byte.

Standard library only; no model loading, credentials, training or source overwrite.
"""
import argparse
import concurrent.futures
import csv
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import quote,urlencode
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
SLUG='azizkasimov/wind-turbine-scada-data-for-early-fault-detection'
API='https://www.kaggle.com/api/v1/datasets/'
SERVICE=['time_stamp','asset_id','id','train_test','status_type_id']
SENSORS=['sensor_0_avg','sensor_1_avg','sensor_11_avg','sensor_12_avg','sensor_13_avg','sensor_14_avg','sensor_18_avg',
    'sensor_32_avg','sensor_33_avg','sensor_34_avg','sensor_38_avg','sensor_39_avg','sensor_40_avg','sensor_41_avg','sensor_43_avg',
    'sensor_50','sensor_52_avg','wind_speed_3_avg','power_29_avg','power_30_avg']


def save(value,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def request(url):
    return urlopen(Request(url,headers={'User-Agent':'WindScope-Data-Audit/1.0'}),timeout=90)


def validate_inventory(files):
    names=[f['name'] for f in files]
    if len(set(names))!=len(names):raise ValueError('Duplicate remote file names')
    if any(name.startswith(('/','\\')) or '..' in name.split('/') or '\\' in name for name in names):raise ValueError('Unsafe remote file name')
    return files


def primary_files(files,farm):
    if farm not in ('A','B','C'):raise ValueError('Unknown farm')
    return [f for f in validate_inventory(files) if re.fullmatch(r'Wind Farm '+farm+r'/datasets/[0-9]+\.csv',f['name'])]


def inventory(out):
    with request(API+'view/'+SLUG) as response:view=json.load(response)
    save(view,out/'view.json')
    files=[];token=None;seen=set();page=0
    while True:
        params=dict(datasetVersionNumber=2,pageSize=100)
        if token:params['pageToken']=token
        with request(API+'list/'+SLUG+'?'+urlencode(params)) as response:data=json.load(response)
        if data.get('hasErrorMessage'):raise ValueError(data['errorMessage'])
        page+=1;save(data,out/('files_page_'+str(page)+'.json'));files.extend(data['datasetFiles'])
        token=data.get('nextPageToken')
        if not token:break
        if token in seen:raise ValueError('Repeated remote pagination token')
        seen.add(token)
    validate_inventory(files)
    groups=[]
    for farm in 'ABC':
        primary=primary_files(files,farm)
        comma=[f for f in files if re.fullmatch(r'Wind Farm '+farm+r'/datasets/comma_[0-9]+\.csv',f['name'])]
        groups.append(dict(farm=farm,primary_csv=len(primary),primary_bytes=sum(f['totalBytes'] for f in primary),comma_csv=len(comma),comma_bytes=sum(f['totalBytes'] for f in comma)))
    result=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),version_requested=2,current_version=view['currentVersionNumber'],
        dataset_id=view['id'],last_updated=view['lastUpdated'],pages=page,file_count=len(files),total_listed_bytes=sum(f['totalBytes'] for f in files),groups=groups,files=files)
    save(result,out/'inventory.json');return result


def verify_stream(stream,existing,destination,expected_bytes):
    existing=Path(existing);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_name(destination.name+'.part')
    digest=hashlib.sha256();count=0
    with partial.open('wb') as target:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            count+=len(chunk)
            if count>expected_bytes:raise ValueError('Remote response exceeds inventory size')
            digest.update(chunk);target.write(chunk)
    if count!=expected_bytes:raise ValueError('Truncated response: expected {}, got {}'.format(expected_bytes,count))
    remote_sha=digest.hexdigest();local_sha=sha(existing) if existing.is_file() else None
    match=local_sha==remote_sha
    if match:
        partial.unlink()
        preferred=existing
    else:
        if destination.exists():
            if sha(destination)!=remote_sha:raise ValueError('Separate verified copy already differs; refusing overwrite')
            partial.unlink()
        else:partial.replace(destination)
        preferred=destination
    return dict(bytes=count,remote_sha256=remote_sha,local_sha256=local_sha,local_present=local_sha is not None,
        matches_local=match,preferred_file=str(preferred.relative_to(ROOT)) if ROOT in preferred.parents else str(preferred),source_unchanged=True)


def local_source(name):
    if name.startswith('Wind Farm A/'):return ROOT/'project/data/CARE_To_Compare'/name
    if name.startswith(('Wind Farm B/','Wind Farm C/')):return ROOT/'artifacts/nbm_h24_v1/external_metadata'/name
    if name=='README.md':return ROOT/'data/source_readme.md'
    return ROOT/'data/raw/kaggle_v2_verified'/name


def verify_remote(entry):
    name=entry['name'];url=API+'download/'+SLUG+'/'+quote(name,safe='')+'?datasetVersionNumber=2'
    destination=ROOT/'data/raw/kaggle_v2_verified'/name
    for attempt in range(3):
        try:
            with request(url) as response:result=verify_stream(response,local_source(name),destination,entry['totalBytes'])
            result.update(name=name,url=url,version=2)
            print('Verified '+name+'; matches local='+str(result['matches_local']),flush=True)
            return result
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))


def profile_csv(path,annotation,sensors):
    counts=Counter();assets=set();status=Counter();missing=Counter();sensor_stats={name:dict(available=0,missing=0,nonfinite=0,invalid_numeric=0,minimum=None,maximum=None,partitions={}) for name in sensors}
    previous=None;first=None;last=None;gaps=0;unordered=0;max_gap=0;rows=0;annotation_rows={}
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream,delimiter=';');header=next(reader)
        if len(header)!=len(set(header)):raise ValueError('Duplicate CSV columns')
        absent=set(SERVICE+sensors)-set(header)
        if absent:raise ValueError('Missing required columns: '+str(sorted(absent)))
        idx={c:i for i,c in enumerate(header)};physical=[c for c in header if c not in SERVICE]
        starts={int(float(annotation[k])):k for k in ('event_start_id','event_end_id')}
        for row in reader:
            if len(row)!=len(header):raise ValueError('Malformed CSV row')
            rows+=1;at=datetime.fromisoformat(row[idx['time_stamp']]);part=row[idx['train_test']]
            if first is None:first=at
            if previous is not None:
                delta=(at-previous).total_seconds();gaps+=int(delta!=600);unordered+=int(delta<=0);max_gap=max(max_gap,delta)
            previous=at;last=at;counts[part]+=1;assets.add(row[idx['asset_id']]);status[row[idx['status_type_id']]]+=1
            row_id=int(row[idx['id']])
            if row_id in starts:annotation_rows[starts[row_id]]=at.isoformat(sep=' ')
            for column in physical:
                if row[idx[column]].strip().lower() in ('','nan','na','null'):missing[column]+=1
            for column in sensors:
                text=row[idx[column]].strip();s=sensor_stats[column]
                if text.lower() in ('','nan','na','null'):s['missing']+=1;continue
                try:value=float(text)
                except ValueError:s['invalid_numeric']+=1;continue
                if not math.isfinite(value):s['nonfinite']+=1;continue
                s['available']+=1;s['partitions'][part]=s['partitions'].get(part,0)+1
                s['minimum']=value if s['minimum'] is None else min(value,s['minimum'])
                s['maximum']=value if s['maximum'] is None else max(value,s['maximum'])
    aligned=all(annotation_rows.get(key)==datetime.fromisoformat(annotation[key[:-3]]).isoformat(sep=' ') for key in ('event_start_id','event_end_id'))
    return dict(rows=rows,columns=len(header),measurement_columns=len(physical),column_names=header,assets=sorted(assets),
        partitions=dict(counts),status_counts=dict(status),first_timestamp=str(first),last_timestamp=str(last),
        non_10min_intervals=gaps,non_increasing_intervals=unordered,max_interval_seconds=max_gap,
        missing_sensor_cells=sum(missing.values()),missing_by_column=dict(missing),sensors=sensor_stats,
        annotation_ids_match_timestamps=aligned,annotation_asset_matches=assets=={str(annotation['asset'])})


def read_metadata(path,delimiter=';'):
    with Path(path).open(encoding='utf-8-sig',newline='') as file:return list(csv.DictReader(file,delimiter=delimiter))


def profile(out,listing,verification):
    by_name={r['name']:r for r in verification['files']}
    def path(name):return ROOT/by_name[name]['preferred_file']
    metadata={farm:read_metadata(path('Wind Farm '+farm+'/event_info.csv')) for farm in 'ABC'}
    feature_meta={farm:read_metadata(path('Wind Farm '+farm+'/feature_description.csv')) for farm in 'ABC'}
    primary_a=primary_files(listing['files'],'A');meta_a={str(r['event_id']):r for r in metadata['A']}
    ids={Path(e['name']).stem for e in primary_a}
    if ids!=set(meta_a):raise ValueError('Farm A event metadata and file inventory disagree')
    profiles=[]
    for entry in sorted(primary_a,key=lambda e:int(Path(e['name']).stem)):
        event=Path(entry['name']).stem
        current=profile_csv(path(entry['name']),meta_a[event],SENSORS);current['event']=int(event);current['label']=meta_a[event]['event_label']
        profiles.append(current);print('Profiled event '+event+'; rows='+str(current['rows']),flush=True)
    comma_a=read_metadata(path('Wind Farm A/comma_event_info.csv'),',')
    comma_51=next((r for r in comma_a if str(r['event_id'])=='51'),None)
    metadata_summary=[]
    for farm in 'ABC':
        metadata_summary.append(dict(farm=farm,events=len(metadata[farm]),labels=dict(Counter(r['event_label'] for r in metadata[farm])),
            described_sensors=len(feature_meta[farm]),features=feature_meta[farm],
            raw_files_present=sum(local_source(e['name']).exists() for e in primary_files(listing['files'],farm)),
            raw_files_remotely_verified=len(primary_a) if farm=='A' else 0))
    ready=all(p['columns']==86 and p['annotation_ids_match_timestamps'] and p['annotation_asset_matches'] and not p['non_increasing_intervals'] for p in profiles)
    ready=ready and all(by_name[e['name']]['matches_local'] for e in primary_a)
    result=dict(status='farm_a_inputs_verified' if ready else 'issues_found',version=2,checked_at_utc=datetime.now(timezone.utc).isoformat(),
        farm_a=dict(files=len(profiles),rows=sum(p['rows'] for p in profiles),events=profiles,
            annotation_labels=dict(Counter(p['label'] for p in profiles)),required_sensors=SENSORS,
            all_primary_bytes_match_existing=all(by_name[e['name']]['matches_local'] for e in primary_a)),
        metadata_summary=metadata_summary,comma_warning=dict(primary_event_51=meta_a['51'],comma_event_51=comma_51,groups_must_not_be_mixed=True),
        full_dataset_locally_present=all(m['raw_files_present']==m['events'] for m in metadata_summary),
        no_training_or_app_changes=True,
        limitations=['Farm B/C SCADA measurements were not downloaded or profiled.','Byte identity does not establish label accuracy or predictive model quality.','The new sensor experiment has not been trained.','No new global deduplication performed; use the reviewed canonical episode protocol.'])
    save(result,out/'readiness.json')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');parser.add_argument('--output',default='artifacts/kaggle_v2_inventory_audit');args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    if ROOT not in out.parents:raise ValueError('Audit output must stay inside workspace')
    out.mkdir(parents=True,exist_ok=True)
    if args.offline:
        listing=json.loads((out/'inventory.json').read_text(encoding='utf-8'))
        verified=json.loads((out/'remote_verification.json').read_text(encoding='utf-8'))
        for row in verified['files']:
            if sha(ROOT/row['preferred_file'])!=row['remote_sha256']:raise ValueError('Verified local file changed')
    else:
        listing=inventory(out)
        wanted={f['name'] for f in primary_files(listing['files'],'A')}
        wanted.update('Wind Farm '+farm+'/'+name for farm in 'ABC' for name in ['event_info.csv','feature_description.csv','comma_event_info.csv'])
        wanted.update(['README.md','README.txt'])
        entries=[f for f in listing['files'] if f['name'] in wanted]
        if len(entries)!=len(wanted):raise ValueError('Required remote entries missing')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(verify_remote,entries))
        verified=dict(version=2,source=API,all_bytes_downloaded_and_hashed=True,network_payload_bytes=sum(r['bytes'] for r in results),files=results)
        save(verified,out/'remote_verification.json')
    result=profile(out,listing,verified)
    print(json.dumps(dict(status=result['status'],farm_a_csv=result['farm_a']['files'],rows=result['farm_a']['rows'],all_primary_bytes_match_existing=result['farm_a']['all_primary_bytes_match_existing'],full_dataset_locally_present=result['full_dataset_locally_present']),indent=2))


if __name__=='__main__':main()
