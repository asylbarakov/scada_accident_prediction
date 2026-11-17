"""Idempotent, hash-verified import of primary Kaggle v2; never loads the model."""
import csv,json,hashlib,math,logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import select,func
from app.db import SessionLocal
from app.models import Scenario,Sensor,Measurement,ReplaySession
from app.service import create_session
from app.forecast_protocol import LEGACY_MODEL,H7_MODEL,PERSONAL_MODEL

logging.basicConfig(level=logging.INFO)
log=logging.getLogger('windscope.ingest')


def source_text(path):
    payload=path.read_bytes()
    try:return payload.decode('utf-8-sig')
    except UnicodeDecodeError:return payload.decode('cp1252')


def import_scenarios(db,root,manifest,identifiers,*,model_id=LEGACY_MODEL,dataset_subdir=''):
    root=Path(root)
    for f in manifest['files']:
        if hashlib.sha256((root/f['local_path']).read_bytes()).hexdigest()!=f['sha256']:
            raise ValueError('Source checksum mismatch: '+f['local_path'])
    events={r['event_id']:r for r in csv.DictReader(source_text(root/'event_info.csv').splitlines(),delimiter=';')}
    descriptions={r['sensor_name']:r for r in csv.DictReader(source_text(root/'feature_description.csv').splitlines(),delimiter=';')}
    for identifier in identifiers:
        expected=next(f['sha256'] for f in manifest['files'] if f['local_path']==(Path(dataset_subdir)/(identifier+'.csv')).as_posix())
        existing=db.get(Scenario,identifier)
        if existing:
            if existing.source_hash!=expected:raise ValueError('Scenario version conflict')
            if db.scalar(select(func.count()).select_from(Measurement).where(Measurement.scenario_id==identifier))!=existing.rows:
                raise ValueError('Incomplete existing import')
            log.info('Already imported scenario %s',identifier);continue
        with (root/dataset_subdir/(identifier+'.csv')).open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream,delimiter=';');rows=[];previous=None;assets=set();metadata=events[identifier]
            for row in reader:
                time=datetime.fromisoformat(row['time_stamp']);row_id=int(row['id']);assets.add(row['asset_id'])
                if previous and (time<=previous[0] or row_id<=previous[1]):raise ValueError('Nonchronological input')
                previous=(time,row_id)
                if row_id==int(metadata['event_start_id']) and str(time)!=metadata['event_start']:raise ValueError('Event boundary mismatch')
                if row_id==int(metadata['event_end_id']) and str(time)!=metadata['event_end']:raise ValueError('Event boundary mismatch')
                values={}
                for key,value in row.items():
                    if not key.startswith(('sensor_','wind_speed_','power_','reactive_power_')):continue
                    value=float(value) if value.strip() else float('nan')
                    values[key]=value if math.isfinite(value) else None
                if len(values)!=81:raise ValueError('Expected 81 measurement fields')
                rows.append(dict(scenario_id=identifier,row_id=row_id,source_at=time,partition=row['train_test'],values=values,status_type_id=int(row['status_type_id'])))
        if assets!={metadata['asset']}:raise ValueError('Asset mismatch')
        pred=[r for r in rows if r['partition']=='prediction']
        annotation=dict(start=metadata['event_start'],end=metadata['event_end'],label=metadata['event_label'],description=metadata['event_description'])
        db.add(Scenario(id=identifier,asset_id=metadata['asset'],source_hash=expected,rows=len(rows),start=rows[0]['source_at'],end=rows[-1]['source_at'],
            prediction_start=pred[0]['source_at'],prediction_end=pred[-1]['source_at'],prediction_points=len(pred),annotation=annotation))
        db.flush()
        for offset in range(0,len(rows),1000):db.execute(Measurement.__table__.insert(),rows[offset:offset+1000])
        if not db.scalar(select(Sensor.key).limit(1)):
            for key in rows[0]['values']:
                base=key
                for suffix in ('_avg','_min','_max','_std'):
                    if base.endswith(suffix):base=base[:-len(suffix)];break
                desc=descriptions.get(base,{})
                db.add(Sensor(key=key,description=desc.get('description',key),unit=desc.get('unit','')))
        db.commit();log.info('Imported %s: %s records',identifier,len(rows))
    for identifier in identifiers:
        if not db.scalar(select(ReplaySession.id).where(ReplaySession.scenario_id==identifier).limit(1)):
            create_session(db,identifier,model_id=model_id)


def main():
    with SessionLocal() as db:
        import_scenarios(db,Path('/input'),json.loads(Path('/manifest.json').read_text(encoding='utf-8')),('0','25'))
        h7_manifest=Path('/h7-manifest.json')
        if h7_manifest.exists():
            import_scenarios(db,Path('/h7-input'),json.loads(h7_manifest.read_text(encoding='utf-8')),
                             ('13','22','72'),model_id=H7_MODEL,dataset_subdir='datasets')
        personal_manifest=Path('/personal-manifest.json')
        if personal_manifest.exists():
            import_scenarios(db,Path('/h7-input'),json.loads(personal_manifest.read_text(encoding='utf-8')),('0','25','26','42'),model_id=PERSONAL_MODEL,dataset_subdir='datasets')



if __name__=='__main__':main()
