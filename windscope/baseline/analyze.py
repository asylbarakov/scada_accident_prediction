"""Offline comparison and plots. No model artifact is mounted for this stage."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from baseline.evaluation import summarize


def read_predictions(path):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['predicted_class'] = int(row['predicted_class']) if row['predicted_class'] else None
        for key in ('class_0_score','class_1_score','sensor_0_avg','sensor_43_avg',
                    'sensor_11_avg','wind_speed_3_avg','voltage','sensor_50'):
            row[key] = float(row[key]) if row[key] else None
    return rows


def plot_scenario(rows, annotation, destination, scenario, partition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    selected = rows if partition == 'all' else [r for r in rows if r['partition'] == partition]
    times = [datetime.fromisoformat(r['source_time']) for r in selected]
    if not times:
        raise ValueError('No observations to plot')
    start = datetime.fromisoformat(annotation['event_start'])
    end = datetime.fromisoformat(annotation['event_end'])
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={'height_ratios':[1.5,1,1,1]}, layout='constrained')
    fig.set_facecolor('#f6f8fc')
    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(alpha=.16)
        ax.spines[['top','right']].set_visible(False)
        if annotation['event_label'] == 'anomaly':
            ax.axvspan(start, end, color='#e89b70', alpha=.23)
            ax.axvline(start, color='#ba5d2a', linewidth=1, linestyle='--')
    # Break lines at missing timesteps instead of inventing continuity.
    def series(key):
        tx, vals = [], []
        for i, (t,row) in enumerate(zip(times,selected)):
            if i and (t-times[i-1]).total_seconds() > 600:
                tx.append(t); vals.append(np.nan)
            tx.append(t); vals.append(row.get(key) if row.get(key) is not None else np.nan)
        return tx, vals

    axes[0].plot(*series('class_1_score'), color='#225bbb', linewidth=.8, label='Score класса 1')
    alarms = [r for r in selected if r['predicted_class'] == 1]
    axes[0].scatter([datetime.fromisoformat(r['source_time']) for r in alarms],
                    [1.055]*len(alarms), color='#b63245', marker='|', s=14, label='Предсказанный класс 1')
    axes[0].axhline(.5, color='#657286', linewidth=.8, linestyle=':', label='Порог binary classifier: 0,5')
    axes[0].set_ylim(-.035,1.11)
    axes[0].set_ylabel('Некалиброванный score')
    axes[0].legend(loc='upper right', ncols=3, fontsize=8)
    if annotation['event_label'] == 'anomaly':
        axes[0].text(.01,.06,'Оранжевый фон: аннотированный интервал аномалии',
                     transform=axes[0].transAxes, fontsize=9, color='#82451f')
    else:
        axes[0].text(.01,.06,'Normal-сценарий: аннотированных аномалий нет',
                     transform=axes[0].transAxes, fontsize=9, color='#334155')
    for key,label,color in [('sensor_0_avg','Ambient','#2a75b9'),
                            ('sensor_43_avg','Nacelle','#388969'),
                            ('sensor_11_avg','Gearbox bearing','#ab6441')]:
        axes[1].plot(*series(key), label=label, color=color, linewidth=.65)
    axes[1].set_ylabel('Температура, °C')
    axes[1].legend(loc='upper right', ncols=3, fontsize=8)
    axes[2].plot(*series('wind_speed_3_avg'), color='#177f7d', linewidth=.65)
    axes[2].set_ylabel('Скорость ветра, m/s')
    axes[3].plot(*series('voltage'), color='#7351a8', linewidth=.65, label='Среднее напряжение')
    axes[3].set_ylabel('Напряжение, V', color='#7351a8')
    right = axes[3].twinx()
    right.plot(*series('sensor_50'), color='#657286', linewidth=.55, alpha=.7)
    right.set_ylabel('sensor_50, Wh по metadata', color='#657286')
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].set_xlabel('Время исходного набора — timezone не установлена')
    label = 'вся история' if partition == 'all' else 'исходная prediction-часть'
    fig.suptitle(f'SWTFPS XGBoost · event {scenario} · {annotation["event_label"]} · {label}\n'
                 'Offline-сравнение; score не задаёт вероятность отказа в будущем',
                 fontsize=13, color='#17243c')
    fig.savefig(destination, dpi=150, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    args = parser.parse_args()
    with args.annotations.open(encoding='utf-8-sig', newline='') as stream:
        annotations = {row['event_id']:row for row in csv.DictReader(stream, delimiter=';')}
    output = dict(annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                  evaluation_only=True, scenarios={})
    for scenario in ('0','25'):
        rows = read_predictions(args.predictions/f'event_{scenario}_predictions.csv')
        annotation = annotations[scenario]
        summaries = {}
        for part in ('all','train','prediction'):
            subset = rows if part == 'all' else [r for r in rows if r['partition'] == part]
            summaries[part] = summarize(subset, annotation)
        test_start = summaries['prediction']['source_start']
        previous = [r for r in rows if test_start and r['source_time'] < test_start]
        first_test = next((r for r in rows if r['partition']=='prediction'), None)
        summaries['prediction']['warning_continues_from_train'] = bool(
            previous and first_test and previous[-1]['predicted_class']==1 and first_test['predicted_class']==1)
        output['scenarios'][scenario] = dict(annotation=annotation, summaries=summaries)
        for part in ('all','prediction'):
            plot_scenario(rows, annotation, args.predictions/f'event_{scenario}_{part}.png', scenario, part)
        episodes = summaries['prediction']['episodes']
        with (args.predictions/f'event_{scenario}_prediction_warning_episodes.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['start','end','points'])
            writer.writeheader(); writer.writerows(episodes)
        compact = {k:v for k,v in summaries['prediction'].items() if k != 'episodes'}
        print(json.dumps(dict(scenario=scenario, prediction_summary=compact), ensure_ascii=False), flush=True)
    (args.predictions/'evaluation.json').write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
