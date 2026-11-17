"""Offline alert-policy evaluation from preserved real baseline predictions, no model load."""
import argparse,csv,json,math
from collections import deque,Counter
from datetime import datetime
from pathlib import Path
from app.alert_engine import AlertPolicy,advance
from app.diagnostics import diagnose,quality


def evaluate(identifier,raw_root,predictions_root,policy):
    state={};alerts=[];active=None;raw_positive=0;total=0;invalid=0;diagnostic_counts=Counter();history=deque(maxlen=72);timeline=[]
    with (raw_root/(identifier+'.csv')).open(encoding='utf-8-sig',newline='') as r,(predictions_root/('event_'+identifier+'_predictions.csv')).open(newline='') as p:
        for measurement,prediction in zip(csv.DictReader(r,delimiter=';'),csv.DictReader(p)):
            if measurement['id']!=prediction['row_id'] or measurement['time_stamp']!=prediction['source_time']:raise ValueError('Raw/golden prediction alignment mismatch')
            values={k:float(v) if v and math.isfinite(float(v)) else None for k,v in measurement.items() if k.startswith(('sensor_','wind_speed_','power_','reactive_power_'))}
            now=datetime.fromisoformat(measurement['time_stamp']);point=dict(source_time=str(now),asset_id=measurement['asset_id'],values=values)
            if measurement['train_test']=='prediction':
                result=dict(status=prediction['status'],predicted_class=int(prediction['predicted_class']) if prediction['predicted_class'] else None,
                    class_1_score=float(prediction['class_1_score']) if prediction['class_1_score'] else None,model_id=prediction['model_id'])
                total+=1;raw_positive+=result['predicted_class']==1;invalid+=not quality(values)
                diagnostic=diagnose(point,result,list(history));diagnostic_counts[diagnostic['category']]+=1
                state,event=advance(state,now,result,policy,quality_ok=quality(values))
                if event=='activated':
                    active=dict(first_signal=state['first_signal'],activated_at=str(now),last=str(now),resolved_at=None,
                        added_delay_minutes=(now-datetime.fromisoformat(state['first_signal'])).total_seconds()/60)
                    alerts.append(active)
                elif active:active['last']=str(now)
                if event in ('resolved','interrupted') and active:
                    active.update(resolved_at=str(now),resolution_reason=state['resolution_reason']);active=None
                timeline.append((now,result['class_1_score']))
            history.append(point)
    # Ground truth is read only after causal diagnostics/policy processing completes.
    with (raw_root/'event_info.csv').open(encoding='utf-8-sig') as stream:
        annotation=next(row for row in csv.DictReader(stream,delimiter=';') if row['event_id']==identifier)
    onset=datetime.fromisoformat(annotation['event_start']) if annotation['event_label']=='anomaly' else None
    for a in alerts:
        a['duration_minutes']=(datetime.fromisoformat(a['resolved_at'] or a['last'])-datetime.fromisoformat(a['activated_at'])).total_seconds()/60
        a['right_censored']=a['resolved_at'] is None
    first=alerts[0]['activated_at'] if alerts else None
    return dict(scenario=identifier,policy=policy.to_dict(),points=total,raw_positive=raw_positive,
        raw_positive_percent=round(raw_positive/max(1,total)*100,2),invalid_feature_points=invalid,
        generated_alerts=len(alerts),total_alert_minutes=sum(a['duration_minutes'] for a in alerts),
        alerts_on_normal_scenario=len(alerts) if onset is None else None,
        alerts_before_annotation=sum(datetime.fromisoformat(a['activated_at'])<onset for a in alerts) if onset else None,
        first_alert=first,first_alert_to_annotation_minutes=(onset-datetime.fromisoformat(first)).total_seconds()/60 if onset and first and datetime.fromisoformat(first)<onset else None,
        added_delay_minutes=[a['added_delay_minutes'] for a in alerts],diagnostic_categories=dict(diagnostic_counts),alerts=alerts),timeline,annotation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=Path('/input'))
    parser.add_argument('--predictions',type=Path,default=Path('/predictions'));parser.add_argument('--output',type=Path,default=Path('/output'))
    parser.add_argument('--activate',type=int,default=3);parser.add_argument('--resolve',type=int,default=3);parser.add_argument('--cooldown',type=int,default=60)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);policy=AlertPolicy(args.activate,args.resolve,args.cooldown)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(15,7),layout='constrained');results=[]
    for identifier,ax in zip(('0','25'),axes):
        result,timeline,annotation=evaluate(identifier,args.raw,args.predictions,policy);results.append(result)
        ax.plot([t[0] for t in timeline],[t[1] for t in timeline],linewidth=.6,color='#5b7cfa',label='Raw class 1 score (uncalibrated)')
        for i,a in enumerate(result['alerts']):ax.axvspan(datetime.fromisoformat(a['activated_at']),datetime.fromisoformat(a['resolved_at'] or a['last']),color='#f59e0b',alpha=.25,label='Post-processed alert' if i==0 else None)
        if annotation['event_label']=='anomaly':ax.axvspan(datetime.fromisoformat(annotation['event_start']),datetime.fromisoformat(annotation['event_end']),color='#ef4444',alpha=.1,label='Dataset annotation (offline only)')
        ax.set_title(f"Event {identifier}: {result['raw_positive']}/{result['points']} raw positives, {result['generated_alerts']} alerts")
        ax.set_ylim(-.05,1.1);ax.grid(alpha=.15);ax.legend(loc='lower left',fontsize=8);ax.set_ylabel('Class score')
    fig.savefig(args.output/'alert_policy_comparison.png',dpi=150);plt.close(fig)
    (args.output/'alert_evaluation.json').write_text(json.dumps(dict(policy=policy.to_dict(),scenarios=results,
        limitations=['Two non-independent scenarios; no retraining or threshold tuning','Fewer notifications do not establish improved ML accuracy','Alert durations use source time; end-of-range alerts are right-censored','No known prediction horizon; lead is relative to annotation only']),indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('alerts','added_delay_minutes')} for r in results],ensure_ascii=False))


if __name__=='__main__':main()
