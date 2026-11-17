"""Export a static comparison; scoring and thresholds remain frozen."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/output');report=json.loads((out/'metrics.json').read_text());events=pd.read_csv(out/'event_comparison.csv')
baseline=report['baseline_metrics'];candidate=report['metrics']
fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained',gridspec_kw={'width_ratios':[1.8,1,1]})
y=np.arange(len(events));ax=axes[0]
ax.barh(y-.17,events.baseline_lead_hours.fillna(0),height=.34,label='Baseline',color='#327ea8')
ax.barh(y+.17,events.candidate_lead_hours.fillna(0),height=.34,label='H24 + динамика',color='#d88825')
for i,row in events.iterrows():
    for offset,key in [(-.17,'baseline_lead_hours'),(.17,'candidate_lead_hours')]:
        if pd.isna(row[key]):ax.plot(.15,i+offset,marker='x',color='#777',markersize=5)
ax.set_yticks(y,['№'+str(v) for v in events.event]);ax.invert_yaxis()
ax.set(xlabel='Упреждение нового сигнала, ч',ylabel='Историческое событие',title='Попадания и пропуски (×)')
ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2)
axes[1].bar(['Baseline','H24 +\nдинамика'],[baseline['onset_hit_events'],candidate['onset_hit_events']],color=['#327ea8','#d88825'])
axes[1].set(ylim=(0,11),ylabel='Обнаружено из 11 отказов',title='Ранние предупреждения')
for i,v in enumerate([baseline['onset_hit_events'],candidate['onset_hit_events']]):axes[1].text(i,v+.2,str(v),ha='center')
rates=[baseline['false_alarms_per_720_normal_hours'],candidate['false_alarms_per_720_normal_hours']]
axes[2].bar(['Baseline','H24 +\nдинамика'],rates,color=['#327ea8','#d88825'])
axes[2].axhline(2,color='#ba3737',ls='--',label='Лимит 2 / 720 ч')
axes[2].set(ylabel='Ложных эпизодов / 720 normal-часов',title='Нагрузка ложных тревог')
for i,v in enumerate(rates):axes[2].text(i,v+.08,f'{v:.2f}',ha='center')
axes[2].legend()
fig.suptitle('Новый прогноз H24: меньше ложных тревог, больше пропущенных отказов',fontsize=15)
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
print('Saved comparison.png')
