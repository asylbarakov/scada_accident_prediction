import os,time,logging
import httpx
from sqlalchemy import select,text
from app.db import engine,SessionLocal
from app.models import ReplaySession
from app.domain import utc_now,remaining_source
from app.service import claim_tick,commit_tick,emit
from app.service import forecast_payload
from app.forecast_protocol import H7_MODEL,PERSONAL_MODEL,forecast_due

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
log=logging.getLogger('windscope.worker')
MODEL_INPUTS=('sensor_0_avg','sensor_1_avg','sensor_43_avg','sensor_11_avg','wind_speed_3_avg','sensor_32_avg','sensor_33_avg','sensor_34_avg','sensor_50')


def main():
    # A single active worker is intentional for this local MVP. Persistent leases
    # and revision fencing still protect an in-flight tick from commands.
    leader=engine.connect()
    if not leader.scalar(text('SELECT pg_try_advisory_lock(8307261)')):
        raise RuntimeError('Another replay worker is active')
    leader.commit()
    with SessionLocal() as db:
        for s in db.scalars(select(ReplaySession).where(ReplaySession.status=='running')).all():
            s.remaining=remaining_source(s.due_at,utc_now(),s.speed);s.status='paused';s.due_at=None
            s.revision+=1;s.lease=None;s.lease_until=None;emit(db,s,'state_changed',{'reason':'worker_restarted'})
        db.commit()
    log.info('Worker ready; interrupted sessions restored paused')
    with httpx.Client(base_url=os.getenv('ML_URL','http://ml:8001'),timeout=5) as client, httpx.Client(base_url=os.getenv('H7_URL','http://forecast:8002'),timeout=15) as forecast_client, httpx.Client(base_url=os.getenv('PERSONAL_URL','http://personal:8003'),timeout=15) as personal_client:
        while True:
            try:
                with SessionLocal() as db:job=claim_tick(db)
                if not job:time.sleep(.05);continue
                if job['model_id'] in (H7_MODEL,PERSONAL_MODEL):
                    model=job['model_id'];personal=model==PERSONAL_MODEL
                    if not forecast_due(job['source_time']):result={'status':'not_scheduled','model_id':model}
                    else:
                        try:
                            with SessionLocal() as db:payload=forecast_payload(db,job)
                            response=(personal_client if personal else forecast_client).post('/predict',json=payload);response.raise_for_status();result=response.json()
                            if result.get('model_id')!=model:raise ValueError('Unexpected forecast model')
                        except (httpx.HTTPError,ValueError):
                            log.exception('H7 inference unavailable for session %s',job['session_id'])
                            result={'status':'unavailable','predicted_class':None,'class_1_score':None,
                                'model_id':model,'prediction_horizon':'P1D' if personal else 'P7D','prediction_horizon_hours':24 if personal else 168,
                                'error':'H7 inference service unavailable'}
                    with SessionLocal() as db:commit_tick(db,job,result)
                    continue
                payload={key:job[key] for key in ('source_time','asset_id','row_id','partition','scenario_id')}
                payload['values']={k:job['values'].get(k) for k in MODEL_INPUTS}
                try:
                    response=client.post('/predict',json=payload);response.raise_for_status();result=response.json()
                except (httpx.HTTPError,ValueError):
                    log.exception('Inference unavailable for session %s',job['session_id'])
                    result={'status':'unavailable','predicted_class':None,'class_1_score':None,
                        'model_id':'swtfps-scada-xgboost-v1.0.0','error':'Inference service unavailable','prediction_horizon':None}
                with SessionLocal() as db:commit_tick(db,job,result)
            except Exception:
                log.exception('Replay tick failed');time.sleep(1)


if __name__=='__main__':main()
