"""Read-only operational readiness; these development assets are not new test data."""
import json
from pathlib import Path
import pandas as pd
from forecast.inference import care_state
from scada_pdm.preparation import make_features

root=Path('/research')
config=json.loads((root/'configs/care_h7.json').read_text())
columns={name:info['column'] for name,info in config['sensors'].items()}
out=[]
for event in (0,25):
    raw=pd.read_csv(root/f'data/CARE_To_Compare/Wind Farm A/datasets/{event}.csv',sep=';',
                    usecols=['time_stamp','asset_id','train_test',*columns.values()],dtype={'asset_id':str})
    times=pd.to_datetime(raw.time_stamp)
    scada=raw[list(columns.values())].astype('float32').rename(columns={v:k for k,v in columns.items()})
    scada.insert(0,'timestamp',times);scada.insert(0,'turbine_id',raw.asset_id)
    features,_,ready=make_features(scada,config,state_policy=care_state)
    selected=raw.train_test.eq('prediction')&times.eq(times.dt.floor('h'))
    out.append(dict(event=event,quality_evaluation=False,
        first_prediction_measurement=str(times[raw.train_test.eq('prediction')].min()),
        first_ready_hour=str(times[selected&ready].min()),
        scheduled_hours=int(selected.sum()),not_ready_hours=int((selected&~ready).sum())))
Path('/audit-output/development_readiness.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
