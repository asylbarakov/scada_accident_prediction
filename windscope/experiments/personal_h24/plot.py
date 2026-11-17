"""Historical-personalization diagnostics from exported CSV only."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/output');report=json.loads((out/'metrics.json').read_text())
data=pd.read_csv(out/'prediction_scores.csv');data.timestamp=pd.to_datetime(data.timestamp,utc=True)
fig,axes=plt.subplots(6,2,figsize=(15,19),layout='constrained')
for ax,event in zip(axes.flat,report['metrics']['events']):
    fault=pd.Timestamp(event['failure_at'])
    g=data.loc[data.source_event.eq(event['source_event'])&data.timestamp.lt(fault)&data.timestamp.ge(fault-pd.Timedelta(hours=72))].sort_values('timestamp')
    x=(fault-g.timestamp).dt.total_seconds()/3600
    ax.axvspan(0,24,color='#fff0c5')
    ax.plot(x,g.score,color='#286faa',marker='.',markersize=3)
    threshold=g.threshold.dropna()
    if len(threshold):ax.axhline(threshold.iloc[0],color='#b83333',ls='--')
    warning=g.warning.eq(True);ax.scatter(x[warning],g.loc[warning,'score'],color='#b83333',s=16)
    status=f"новая тревога: {event['first_onset_lead_hours']:.1f} ч" if event['onset_hit'] else 'новой тревоги нет'
    ax.set(title=f"Event {event['source_event']} · {status}",xlabel='Часов до отказа',ylabel='Score отклонения',xlim=(72,0));ax.grid(alpha=.2)
axes.flat[-1].axis('off')
axes.flat[-1].text(0,.95,'Персональная модель по собственной прошлой истории\n\nСиний — score температурных отклонений.\nКрасный пунктир — порог из прошлой normal-истории.\nКрасные точки — активные предупреждения.\nЖёлтый — последние 24 часа перед отказом.\n\nВысокий score не является вероятностью отказа.\nПредупреждение задолго до окна не получает\nновый lead time из-за обрезки рисунка.',va='top',fontsize=11)
fig.suptitle('Персональные нормальные модели: временная проверка',fontsize=17)
fig.savefig(out/'fault_scores.png',dpi=145);plt.close(fig)

rows=[]
for source in report['sources']:
    if source['trained']:
        rows.append(dict(event=source['event'],calibration=source['historical_calibration']['normal_warning_fraction'],prediction=source['test']['normal_warning_fraction'],hours=source['test']['known_normal_hours']))
fractions=pd.DataFrame(rows).dropna(subset=['prediction']).sort_values('event')
fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
x=range(len(fractions));ax.bar([v-.18 for v in x],fractions.calibration,width=.36,label='Прошлая normal calibration')
ax.bar([v+.18 for v in x],fractions.prediction,width=.36,label='Будущие подтверждённые normal-часы')
ax.axhline(.05,color='#b83333',ls='--',label='Calibration-лимит 5%');ax.set_xticks(list(x),fractions.event.astype(str))
ax.set(xlabel='Event (только сценарии с известным normal follow-up)',ylabel='Доля normal-времени под тревогой',title='Перенос порога из прошлой истории в prediction-часть');ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(out/'normal_warning_duration.png',dpi=150);plt.close(fig)
print('Created 2 personal-model figures')
