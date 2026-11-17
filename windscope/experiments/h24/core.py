"""H24 labels and evaluation. This module never consumes model features."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

HORIZON = pd.Timedelta(hours=24)
STEP = pd.Timedelta(minutes=10)


def label_event(t, observed, kind, start, end, episode, canonical, other_anchors, competing):
    label=pd.Series(pd.NA,index=t.index,dtype='Int8')
    reason=pd.Series('unknown_failure_free_future',index=t.index,dtype='string')
    if kind=='anomaly':
        positive=t.lt(end)&t.add(HORIZON).ge(end)
        if not canonical:
            reason[:]='ambiguous_fault_episode'
        elif any(end-HORIZON<=anchor<end for anchor in competing):
            reason.loc[positive]='potential_competing_fault'
        else:
            label.loc[positive]=1;reason.loc[positive]='positive'
        reason.loc[t.ge(end)]='at_or_after_fault'
    elif kind=='normal':
        inside=t.between(start,end)
        run_end=t.shift(-1).sub(t).ne(STEP)|~observed.shift(-1,fill_value=False)
        last=t.where(run_end).bfill()
        full=observed&inside&t.add(HORIZON).le(end)&t.add(HORIZON).le(last)
        conflict=pd.Series(False,index=t.index)
        for anchor in other_anchors:
            conflict|=t.le(anchor)&t.add(HORIZON).ge(anchor)
        label.loc[full&~conflict]=0
        reason.loc[~inside]='outside_annotated_normal_interval'
        reason.loc[inside]='incomplete_normal_followup'
        reason.loc[full&conflict]='potential_competing_fault'
        reason.loc[full&~conflict]='negative'
    else:
        raise ValueError('Unknown annotation type')
    positive=label.eq(1).fillna(False)
    return pd.DataFrame(dict(label=label,reason=reason,
        episode=pd.Series(episode,index=t.index,dtype='string').where(positive),
        failure_at=pd.Series(end,index=t.index).where(positive)))


def reconcile(labels):
    """Conservative cross-file reconciliation matching the verified CARE policy."""
    result=labels.copy()
    missing=result.label.notna()&result.avg_fingerprint.isna()
    result.loc[missing,['label','reason']]=[pd.NA,'no_physical_readings']
    for keys in (['turbine_id','timestamp'],['avg_fingerprint']):
        candidates=result.loc[result.label.notna()].sort_values(
            ['history_ready','source_event','timestamp'],ascending=[False,True,True])
        repeated=candidates.loc[candidates.groupby(keys).source_event.transform('nunique').gt(1)]
        for _,group in repeated.groupby(keys,sort=False):
            if group.label.nunique()>1 or group.episode.nunique()>1 or group.avg_fingerprint.nunique()>1:
                result.loc[group.index,['label','reason']]=[pd.NA,'conflicting_repeated_observation']
            else:
                drop=group.index[group.source_event.ne(group.iloc[0].source_event)]
                result.loc[drop,['label','reason']]=[pd.NA,'deduplicated_observation']
    result['episode']=result.episode.where(result.label.eq(1).fillna(False))
    result['failure_at']=result.failure_at.where(result.label.eq(1).fillna(False))
    return result


def fold_assets(assets,test):
    validation=assets[(assets.index(test)+1)%len(assets)]
    return dict(train=[a for a in assets if a not in (test,validation)],validation=[validation],test=[test])


def evaluate(points,threshold,episodes):
    points=points.sort_values(['source_event','timestamp']).copy()
    available=points.history_ready&points.score.notna()
    active=available&points.score.ge(threshold)
    known=available&points.label.notna()
    y=points.loc[known,'label'].astype(int)
    tn,fp,fn,tp=confusion_matrix(y,active.loc[known],labels=[0,1]).ravel()
    # Count starts only on certified normal exposure. Unknown hours break runs;
    # this is not an estimate of false alarms throughout unlabelled operation.
    negative=known&points.label.eq(0).fillna(False)
    false_active=active&negative
    continued=points.source_event.eq(points.source_event.shift())&points.timestamp.diff().eq(pd.Timedelta(hours=1))
    false_starts=false_active&~(false_active.shift(fill_value=False)&continued)
    negative_hours=int(negative.sum())
    events=[]
    for episode in episodes:
        target=points.episode.eq(episode['id']).fillna(False)&points.label.eq(1).fillna(False)
        hit=points.loc[target&active]
        fault=pd.Timestamp(episode['failure_at'])
        first=hit.timestamp.min() if len(hit) else None
        events.append(dict(**episode,scheduled_positive_hours=int(target.sum()),
            available_positive_hours=int((target&available).sum()),warning_hours=len(hit),
            detected=first is not None,first_warning=first.isoformat() if first is not None else None,
            lead_hours=float((fault-first).total_seconds()/3600) if first is not None else None))
    return dict(scheduled_hours=len(points),available_hours=int(available.sum()),known_scored_hours=int(known.sum()),
        unavailable_hours=int((~available).sum()),availability=float(available.mean()) if len(points) else 0.,
        unknown_warning_hours=int((active&points.label.isna()).sum()),
        tp=int(tp),fn=int(fn),tn=int(tn),fp=int(fp),
        point_recall=float(tp/(tp+fn)) if tp+fn else None,
        point_precision=float(tp/(tp+fp)) if tp+fp else None,
        average_precision=float(average_precision_score(y,points.loc[known,'score'])) if y.nunique()==2 else None,
        roc_auc=float(roc_auc_score(y,points.loc[known,'score'])) if y.nunique()==2 else None,
        detected_events=sum(e['detected'] for e in events),total_events=len(events),
        event_recall=sum(e['detected'] for e in events)/len(events) if events else 0.,
        known_negative_hours=negative_hours,false_warning_episodes=int(false_starts.sum()),
        false_alarms_per_720_negative_hours=float(false_starts.sum()*720/negative_hours) if negative_hours else None,
        events=events)


def selection_key(metrics):
    return (metrics['event_recall'],-metrics['false_alarms_per_720_negative_hours'],metrics['point_precision'] or 0.)


def choose_threshold(validation,episodes,budget):
    """Exact observed-score search, using validation only; >1 explicitly disables alerts."""
    p=validation.sort_values(['source_event','timestamp'])
    available=(p.history_ready&p.score.notna()).to_numpy()
    negative=(p.label.eq(0).fillna(False)&available).to_numpy()
    known=p.label.notna().to_numpy()&available
    positive=(p.label.eq(1).fillna(False)&available).to_numpy()
    scores=p.score.fillna(-np.inf).to_numpy()
    candidates=np.unique(np.r_[scores[known],np.nextafter(1.,np.inf)])
    continued=(p.source_event.eq(p.source_event.shift())&p.timestamp.diff().eq(pd.Timedelta(hours=1))).to_numpy()
    masks=[(p.episode.eq(e['id']).fillna(False).to_numpy()&positive) for e in episodes]
    best=None
    for threshold in candidates:
        active=(scores>=threshold)&available
        false=active&negative
        starts=false&~(np.r_[False,false[:-1]]&continued)
        rate=float(starts.sum()*720/negative.sum())
        if rate>budget:continue
        recall=sum(bool((active&m).any()) for m in masks)/len(masks)
        precision=float((active&positive).sum()/(active&known).sum()) if (active&known).any() else 0.
        key=(recall,-rate,precision,float(threshold))
        if best is None or key>best[0]:best=(key,float(threshold))
    threshold=best[1]
    return threshold,evaluate(validation,threshold,episodes)
