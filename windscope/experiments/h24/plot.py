"""Static figures from CSV only; never deserialize models in the plotting image."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/output')
metrics=json.loads((out/'metrics.json').read_text())
data=pd.read_csv(out/'oof_predictions.csv')
data['timestamp']=pd.to_datetime(data.timestamp,utc=True)
events=metrics['selected_strategy']['events']
fig,axes=plt.subplots(6,2,figsize=(15,20),layout='constrained')
for ax,event in zip(axes.flat,events):
    failure=pd.Timestamp(event['failure_at'])
    part=data.loc[data.source_event.eq(event['source_event'])&data.timestamp.lt(failure)&data.timestamp.ge(failure-pd.Timedelta(hours=48))].sort_values('timestamp')
    x=(failure-part.timestamp).dt.total_seconds()/3600
    ax.axvspan(0,24,color='#fff0c5',label='Целевое окно до отказа')
    ax.plot(x,part.score,color='#246bb2',marker='.',markersize=3,label='Score на исключённой турбине')
    threshold=float(part.threshold.iloc[0]) if len(part) else np.nan
    if threshold<=1:ax.axhline(threshold,color='#c53734',ls='--',label='Порог из validation')
    warned=part.loc[part.warning.eq(True)]
    ax.scatter((failure-warned.timestamp).dt.total_seconds()/3600,warned.score,color='#c53734',s=20,zorder=3)
    lead=f"тревога в окне: {event['lead_hours']:.1f} ч" if event['detected'] else 'ПРОПУЩЕН'
    ax.set(title=f"Event {event['source_event']} · турбина {event['turbine_id']} · {lead}",xlabel='Часов до отказа (0 — отказ)',ylabel='Некалиброванный score',xlim=(48,0),ylim=(-.03,1.06))
    ax.grid(alpha=.2)
    if threshold>1:
        ax.text(.02,.06,'Тревоги отключены: validation не дал допустимого порога',transform=ax.transAxes,fontsize=9,color='#a42727')
axes.flat[-1].axis('off')
axes.flat[-1].text(0,.95,'Проверка H24: 5 внешних folds\n\nКаждая турбина исключена из своего обучения.\nЖёлтая зона: (t, t + 24 ч] содержит отказ.\nКрасные точки: предупреждения.\nПробелы: нет доступного прогноза.\n\nEvent 45 и 84: тревоги начались намного раньше\nцелевого окна. Это не доказательство\nсвоевременного прогноза на 24 часа.\n\nПорог выбран на другой турбине.\nScore не является калиброванной вероятностью.\nEvent 51/72 учтён как один отказ.',va='top',fontsize=12)
fig.suptitle('Прогноз отказа в следующие 24 часа — независимые для обучения турбины',fontsize=18)
fig.savefig(out/'fault_predictions.png',dpi=150);plt.close(fig)

fig,axes=plt.subplots(5,1,figsize=(15,15),layout='constrained')
for ax,(asset,part) in zip(axes,data.groupby('turbine_id')):
    normal=part.loc[part.label.eq(0)].sort_values(['source_event','timestamp']).reset_index(drop=True)
    ax.plot(np.arange(len(normal)),normal.score,color='#246bb2',lw=1)
    if len(normal) and normal.threshold.iloc[0]<=1:ax.axhline(normal.threshold.iloc[0],color='#c53734',ls='--')
    alarm=normal.warning.eq(True)
    ax.scatter(normal.index[alarm],normal.loc[alarm,'score'],color='#c53734',s=15)
    ax.set(title=f'Турбина {asset}: подтверждённые нормальные часы, только доступные/известные точки',xlabel='Индекс наблюдения (разные файлы склеены только на рисунке)',ylabel='Score',ylim=(-.03,1.06))
    ax.grid(alpha=.2)
fig.savefig(out/'normal_predictions.png',dpi=150);plt.close(fig)
print('Created fault_predictions.png and normal_predictions.png')
