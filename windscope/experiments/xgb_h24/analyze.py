"""Post-evaluation distributions and tree importance; never retunes the experiment."""
import json
from pathlib import Path
import pandas as pd
from model import NativeAdapter

out=Path('/output');report=json.loads((out/'metrics.json').read_text());importance=[];score_stats=[]
for fold in report['folds']:
    for name in ('research7','history160'):
        directory=out/f'fold_{fold["test_asset"]}'/name
        adapter=NativeAdapter(directory)
        gains=adapter.model.get_booster().get_score(importance_type='gain')
        total=sum(gains.values())
        for key,value in gains.items():
            index=int(key[1:])
            importance.append(dict(test_asset=fold['test_asset'],candidate=name,feature=adapter.features[index],gain=value,gain_fraction=value/total if total else 0))
        for subset in ('validation','test'):
            data=pd.read_csv(directory/('validation_scores.csv' if subset=='validation' else 'test_predictions.csv'))
            for label in (0,1):
                x=data.loc[data.label.eq(label),'score'].dropna()
                score_stats.append(dict(test_asset=fold['test_asset'],candidate=name,subset=subset,label=label,rows=len(x),
                    q05=float(x.quantile(.05)),median=float(x.median()),q95=float(x.quantile(.95)),maximum=float(x.max())))
pd.DataFrame(importance).sort_values(['test_asset','candidate','gain_fraction'],ascending=[True,True,False]).to_csv(out/'feature_importance.csv',index=False)
pd.DataFrame(score_stats).to_csv(out/'score_distributions.csv',index=False)
top=pd.DataFrame(importance).sort_values('gain_fraction',ascending=False).groupby(['test_asset','candidate']).head(3)
(out/'diagnostics.json').write_text(json.dumps(dict(scope='descriptive post-evaluation analysis, no causal interpretation',top_features=top.to_dict('records'),score_distributions=score_stats),indent=2,allow_nan=False)+'\n')
print(top.loc[top.candidate.eq('research7')].to_string(index=False))
