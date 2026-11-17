"""Static scientific comparison of the three predeclared variants."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/output');report=json.loads((out/'metrics.json').read_text());events=pd.read_csv(out/'event_comparison.csv')
names=['baseline','relations_only','power30_only','relations_power30']
labels=['Baseline','Связи','Power30','Связи +\nPower30'];colors=['#637d94','#3e9ba0','#b18a48','#2869ad']
metrics=[report['baseline_metrics']]+[report['variants'][n]['metrics'] for n in names[1:]]
fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained',gridspec_kw={'width_ratios':[1.6,1,1]})
y=np.arange(len(events))
for i,(name,label,color) in enumerate(zip(names,labels,colors)):
    key='baseline_lead_hours' if name=='baseline' else name+'_lead_hours'
    values=events[key]
    axes[0].barh(y+(i-1.5)*.19,values.fillna(0),height=.18,label=label.replace('\n',' '),color=color)
axes[0].set_yticks(y,['№'+str(v) for v in events.event]);axes[0].invert_yaxis()
axes[0].set(xlabel='Часов до аннотированного отказа',title='Новые предупреждения в H24')
axes[0].legend(loc='lower right',fontsize=8);axes[0].grid(axis='x',alpha=.2)
axes[1].bar(labels,[m['onset_hit_events'] for m in metrics],color=colors)
axes[1].set(ylim=(0,11),ylabel='Обнаружено из 11 событий',title='Обнаруженные отказы')
axes[2].bar(labels,[m['false_alarms_per_720_normal_hours'] for m in metrics],color=colors)
axes[2].axhline(2,color='#ba3737',ls='--',label='Лимит 2 / 720 ч')
axes[2].set(ylabel='Ложных эпизодов / 720 normal-часов',title='Ложные предупреждения');axes[2].legend(fontsize=8)
for ax,key in [(axes[1],'onset_hit_events'),(axes[2],'false_alarms_per_720_normal_hours')]:
    for i,m in enumerate(metrics):ax.text(i,m[key]+.1,str(m[key]) if isinstance(m[key],int) else f'{m[key]:.2f}',ha='center')
fig.suptitle('Физические связи и фактическая мощность: улучшение на истории, лимит тревог не пройден',fontsize=14)
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
print('Saved comparison.png')
