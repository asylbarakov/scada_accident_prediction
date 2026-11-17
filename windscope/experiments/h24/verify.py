"""Verify saved evidence without refitting or threshold selection."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from core import evaluate
from run import digest

out=Path('/output');research=Path('/research');code=Path('/experiment')
metrics=json.loads((out/'metrics.json').read_text());gate=json.loads((out/'data_gate.json').read_text())
assert digest(out/'data.parquet')==metrics['data_sha256']
assert digest(code/'protocol.json')==metrics['protocol_sha256']==gate['protocol_sha256']
for key in ('core.py','pipeline.py','run.py'):
    assert digest(code/key)==metrics['code_sha256'][key],key
for key,expected in gate['inputs_sha256'].items():
    category,name=key.split('/',1)
    path=research/('data/CARE_To_Compare/Wind Farm A' if category=='raw' else 'cache/care_h7_v6')/name
    assert digest(path)==expected,key
runtime_sources={}
for name in ('preparation.py','edp.py'):
    runtime_path=Path('/work/reference/scada_pdm')/name
    assert digest(runtime_path)==digest(research/'scada_pdm'/name)
    runtime_sources[name]=digest(runtime_path)
frame=pd.read_parquet(out/'data.parquet');oof=pd.read_csv(out/'oof_predictions.csv')
oof['timestamp']=pd.to_datetime(oof.timestamp,utc=True);oof['turbine_id']=oof.turbine_id.astype(str)
maximum_error=0.;checked=0
for fold in metrics['folds']:
    test=fold['test_asset'];name=fold['selected'];directory=out/f'fold_{test}'/name
    metadata=json.loads((directory/'inference.json').read_text())
    assert digest(directory/'pipeline.joblib')==metadata['pipeline_sha256']
    assert test not in metadata['training_assets']+metadata['validation_assets']
    assert not set(metadata['training_assets'])&set(metadata['validation_assets'])
    pipeline=joblib.load(directory/'pipeline.joblib')
    rows=frame.loc[frame.turbine_id.eq(test)&frame.history_ready].sort_values(['source_event','timestamp'])
    saved=oof.loc[oof.turbine_id.eq(test)&oof.history_ready].sort_values(['source_event','timestamp'])
    assert len(rows)==len(saved)
    assert rows[['source_event','timestamp']].reset_index(drop=True).equals(saved[['source_event','timestamp']].reset_index(drop=True))
    scores=pipeline.predict_proba(rows[gate['features']])[:,1]
    error=float(np.max(np.abs(scores-saved.score.to_numpy())))
    assert error<1e-12,error
    maximum_error=max(maximum_error,error);checked+=len(rows)
    assert (saved.warning.to_numpy()==(scores>=metadata['threshold'])).all()
recomputed=evaluate(oof.assign(score=oof.warning.astype(float).where(oof.history_ready)),.5,gate['canonical_faults'])
for key in ('tp','fp','tn','fn','detected_events','total_events','false_warning_episodes','known_negative_hours','unavailable_hours'):
    assert recomputed[key]==metrics['selected_strategy'][key],key
result=dict(status='passed',reproduced_oof_predictions=checked,max_score_error=maximum_error,
    source_hashes_checked=len(gate['inputs_sha256']),runtime_feature_sources=runtime_sources,
    canonical_faults=len(gate['canonical_faults']),causal_prefix_checks=len(gate['causal_prefix_checks']),
    fitting_performed=False,thresholds_changed=False)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
