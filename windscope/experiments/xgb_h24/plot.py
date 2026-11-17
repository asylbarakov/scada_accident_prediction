"""Plot saved CSV without deserializing any model."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/output');metrics=json.loads((out/'metrics.json').read_text())
data=pd.read_csv(out/'oof_predictions.csv');data.timestamp=pd.to_datetime(data.timestamp,utc=True)
fig,axes=plt.subplots(6,2,figsize=(15,19),layout='constrained')
for ax,event in zip(axes.flat,metrics['selected']['events']):
    failure=pd.Timestamp(event['failure_at'])
    g=data.loc[data.source_event.eq(event['source_event'])&data.timestamp.lt(failure)&data.timestamp.ge(failure-pd.Timedelta(hours=48))].sort_values('timestamp')
    x=(failure-g.timestamp).dt.total_seconds()/3600
    ax.axvspan(0,24,color='#fff0c5')
    ax.plot(x,g.score,color='#246bb2',marker='.',markersize=3)
    threshold=g.threshold.dropna()
    if len(threshold):ax.axhline(threshold.iloc[0],color='#bd3434',ls='--')
    else:ax.text(.02,.88,'Предупреждения отключены по validation',transform=ax.transAxes,fontsize=9,color='#9c3030')
    warning=g.warning.eq(True)
    ax.scatter(x[warning],g.loc[warning,'score'],color='#bd3434',s=22)
    ax.set(title=f"Event {event['source_event']} · {g.candidate.iloc[0]} · пропущен",xlabel='Часов до отказа',ylabel='Некалиброванный score',xlim=(48,0),ylim=(-.03,1.04))
    ax.grid(alpha=.2)
axes.flat[-1].axis('off')
axes.flat[-1].text(0,.95,'XGBoost H24: проверка двух наборов признаков\n\nresearch7 — семь признаков исследования.\nhistory160 — причинные признаки истории.\n\nВыбор признаков и порогов — только validation.\nЖёлтая зона — последние 24 часа до отказа.\n\n0/11 новых предупреждений в целевом окне.\nScore сам по себе не является\nкалиброванной вероятностью поломки.',va='top',fontsize=11)
fig.suptitle('Новые XGBoost H24: исключённые из обучения турбины',fontsize=17)
fig.savefig(out/'fault_scores.png',dpi=145);plt.close(fig)

stats=pd.read_csv(out/'score_distributions.csv')
fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
for ax,candidate in zip(axes,('research7','history160')):
    for label,color,shift in [(0,'#2871a8',-.12),(1,'#c8483d',.12)]:
        g=stats.loc[stats.candidate.eq(candidate)&stats.subset.eq('test')&stats.label.eq(label)].sort_values('test_asset')
        ax.errorbar(pd.RangeIndex(len(g))+shift,g['median'],yerr=[g['median']-g.q05,g.q95-g['median']],fmt='o',capsize=4,color=color,label='Normal' if label==0 else 'Последние 24 часа до отказа')
        ax.set_xticks(range(len(g)),g.test_asset.astype(str))
    ax.set(title=f'{candidate}: test score, median и q05–q95',xlabel='Исключённая турбина',ylabel='Некалиброванный score',ylim=(-.03,1.04));ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'score_distributions.png',dpi=150);plt.close(fig)
print('Created 2 XGBoost figures')
