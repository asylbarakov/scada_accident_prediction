"""Figures from exported CSV; no models loaded."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/output');metrics=json.loads((out/'metrics.json').read_text())
data=pd.read_csv(out/'oof_predictions.csv');data.timestamp=pd.to_datetime(data.timestamp,utc=True)
fig,axes=plt.subplots(6,2,figsize=(15,19),layout='constrained')
for ax,event in zip(axes.flat,metrics['nbm']['events']):
    fault=pd.Timestamp(event['failure_at'])
    g=data.loc[data.source_event.eq(event['source_event'])&data.timestamp.lt(fault)&data.timestamp.ge(fault-pd.Timedelta(hours=72))].sort_values('timestamp')
    x=(fault-g.timestamp).dt.total_seconds()/3600
    ax.axvspan(0,24,color='#fff0c5')
    ax.plot(x,g.score,color='#246bb2',lw=1.4,marker='.',markersize=3)
    threshold=g.threshold.dropna()
    if len(threshold):ax.axhline(threshold.iloc[0],color='#ba3434',ls='--')
    else:ax.text(.02,.91,'Предупреждения отключены по validation',transform=ax.transAxes,fontsize=9,color='#a32d2d')
    ax.set(title=f"Event {event['source_event']} · турбина {event['turbine_id']}",xlabel='Часов до отказа',ylabel='Score отклонения, не вероятность',xlim=(72,0))
    ax.grid(alpha=.2)
axes.flat[-1].axis('off')
axes.flat[-1].text(0,.95,'NBM: температуры относительно нормального режима\n\nСиний — сглаженный score отклонений.\nЖёлтый — последние 24 часа перед отказом.\n\nНи один validation-fold не дал нового\nпредупреждения в целевом окне при лимитах:\n• не более 5% нормального времени под тревогой;\n• не более 2 ложных эпизодов на 720 часов.\n\nРост score сам по себе не означает\nподтверждённый прогноз скорой поломки.',va='top',fontsize=11)
fig.suptitle('Нормальное поведение: зафиксированный эксперимент H24',fontsize=17)
fig.savefig(out/'nbm_fault_scores.png',dpi=145);plt.close(fig)

event=metrics['nbm']['events'][0];fault=pd.Timestamp(event['failure_at'])
g=data.loc[data.source_event.eq(event['source_event'])&data.timestamp.lt(fault)&data.timestamp.ge(fault-pd.Timedelta(hours=168))].sort_values('timestamp')
x=(fault-g.timestamp).dt.total_seconds()/3600
target='generator_bearing_temperature'
fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
axes[0].plot(x,g[target+'__actual'],label='Измеренная температура (t − 10 мин)',color='#246bb2')
axes[0].plot(x,g[target+'__expected'],label='Ожидаемая при нормальном режиме',color='#d58918')
axes[0].set(ylabel='°C',title='Event 0: температура подшипника генератора и нормальное ожидание');axes[0].legend()
axes[1].plot(x,g[target+'__residual'],color='#7754a8');axes[1].axhline(0,color='gray',ls='--');axes[1].set(ylabel='Отклонение, °C')
axes[2].plot(x,g.score,color='#246bb2');axes[2].set(ylabel='Score четырёх каналов',xlabel='Часов до отказа',xlim=(168,0))
for ax in axes:ax.axvspan(0,24,color='#fff0c5');ax.grid(alpha=.2)
fig.savefig(out/'temperature_residual_example.png',dpi=150);plt.close(fig)

curves=pd.read_csv(out/'validation_threshold_tradeoffs.csv')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for asset,g in curves.groupby('test_asset'):
    axes[0].scatter(g.normal_warning_fraction,g.onset_event_recall,s=10,alpha=.45,label=f'Outer {asset}')
    axes[1].scatter(g.false_alarms_per_720_normal_hours,g.onset_event_recall,s=10,alpha=.45,label=f'Outer {asset}')
axes[0].axvline(.05,color='#ba3434',ls='--',label='Лимит 5%');axes[1].axvline(2,color='#ba3434',ls='--',label='Лимит 2/720 ч')
axes[0].set(xlabel='Доля нормального времени под тревогой',ylabel='Доля отказов с новым предупреждением',title='Validation: ограничение длительности')
axes[1].set(xlabel='Ложные эпизоды / 720 нормальных часов',ylabel='Доля отказов с новым предупреждением',title='Validation: ограничение частоты')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_ylim(-.03,1.03)
fig.savefig(out/'validation_tradeoffs.png',dpi=150);plt.close(fig)
print('Created 3 NBM figures')
