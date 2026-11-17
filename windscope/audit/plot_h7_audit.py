"""Static audit figure from evaluation CSVs; no model loading or synthetic data."""
import csv,json
from datetime import datetime,timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter,AutoDateLocator

root=Path('/audit-output')
report=json.loads((root/'metrics.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained')
for ax,info in zip(axes,report['source_events']):
    rows=list(csv.DictReader((root/f"event_{info['event']}_hourly.csv").open()))
    times=[datetime.fromisoformat(r['timestamp']) for r in rows]
    decisions=[int(r['decision']=='True') if r['status']=='ok' else float('nan') for r in rows]
    ax.step(times,decisions,where='post',color='#c87800',linewidth=1.8,label='Решение модели: 1 — предупреждение, 0 — ниже порога')
    missing=[t for t,r in zip(times,rows) if r['status']!='ok']
    if missing:ax.scatter(missing,[-.1]*len(missing),marker='x',s=18,color='#a02035',label='Нет прогноза: не готова история')
    if info['source_label']=='anomaly':
        fault=datetime.fromisoformat(info['annotation_end'])
        ax.axvspan(fault-timedelta(days=7),fault,color='#3273b8',alpha=.09,label='Истинное окно: отказ в следующие 7 суток')
        ax.axvline(fault,color='#a02035',linewidth=1.5,linestyle='--',label='Аннотированный момент отказа')
        if info['last_h7_warning']:
            last=datetime.fromisoformat(info['last_h7_warning'])
            ax.annotate(f"Последний сигнал: за {info['last_lead_hours']:.0f} ч до отказа",xy=(last,1),
                        xytext=(last+timedelta(hours=9),.64),arrowprops={'arrowstyle':'->','color':'#404040'},fontsize=9)
    title=f"Event {info['event']} · турбина {info['asset']} · "
    title+= 'нормальный сценарий' if info['source_label']=='normal' else ('гидравлика, по исторической разметке' if info['event']==22 else 'редуктор, по исторической разметке')
    ax.set_title(title,loc='left',fontweight='bold')
    ax.set_ylim(-.2,1.2);ax.set_yticks([0,1]);ax.set_ylabel('Сигнал H7')
    ax.grid(axis='y',alpha=.25)
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4,maxticks=8))
    ax.xaxis.set_major_formatter(DateFormatter('%d.%m\n%H:%M'))
unique={}
for ax in axes:
    handles,labels=ax.get_legend_handles_labels()
    unique.update(zip(labels,handles))
fig.legend(list(unique.values()),list(unique),loc='outside lower center',ncol=2,fontsize=9,frameon=False)
fig.suptitle('Аудит H7: заново рассчитанные сигналы против фактических будущих событий\nОтложенная турбина 21; один проход истории, без обучения и изменения порога',fontsize=13,fontweight='bold')
fig.savefig(root/'heldout_warning_audit.png',dpi=150)
print(root/'heldout_warning_audit.png')
