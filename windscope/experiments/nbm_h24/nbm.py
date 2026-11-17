"""Train-normal-only temperature residuals and causal warning evaluation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score,roc_auc_score


def smooth_scores(points,hours):
    p=points.sort_values(['source_event','timestamp'])
    available=p.score.notna()&np.isfinite(p.score)
    continuous=p.source_event.eq(p.source_event.shift())&p.timestamp.diff().eq(pd.Timedelta(hours=1))
    segment=(~continuous|~available|~available.shift(fill_value=False)).cumsum()
    result=p.score.where(available).groupby(segment,sort=False).rolling(hours,min_periods=hours).mean().droplevel(0)
    return result.reindex(points.index)


def operational_metrics(points,threshold,episodes,*,decisions=None):
    p=points.sort_values(['source_event','timestamp']).copy()
    available=p.history_ready&p.score.notna()&np.isfinite(p.score)
    if decisions is not None:active=decisions.reindex(p.index).fillna(False)&available
    else:active=pd.Series(False,index=p.index) if threshold is None else p.score.ge(threshold)&available
    continuous=p.source_event.eq(p.source_event.shift())&p.timestamp.diff().eq(pd.Timedelta(hours=1))
    previous_observed=available.shift(fill_value=False)&continuous
    previous_active=active.shift(fill_value=False)
    onset=active&previous_observed&~previous_active
    left_censored=active&~previous_observed
    negative=p.label.eq(0).fillna(False)&available
    positive=p.label.eq(1).fillna(False)&available
    false=negative&active
    false_starts=false&~(false.shift(fill_value=False)&continuous)
    tp=int((positive&active).sum());fn=int((positive&~active).sum());fp=int(false.sum());tn=int((negative&~active).sum())
    events=[]
    for e in episodes:
        at=pd.Timestamp(e['failure_at'])
        target=p.source_event.eq(e['source_event'])&p.timestamp.ge(at-pd.Timedelta(hours=24))&p.timestamp.lt(at)
        eligible=target&p.episode.eq(e['id']).fillna(False)&p.label.eq(1).fillna(False)
        hits=p.loc[eligible&active];new=p.loc[eligible&onset]
        past=p.loc[p.source_event.eq(e['source_event'])&p.timestamp.lt(at)&active]
        events.append(dict(**e,scheduled_target_hours=int(target.sum()),eligible_target_hours=int(eligible.sum()),
            available_target_hours=int((target&available).sum()),window_hit=bool(len(hits)),onset_hit=bool(len(new)),
            first_target_warning=hits.timestamp.min().isoformat() if len(hits) else None,
            first_observed_onset=new.timestamp.min().isoformat() if len(new) else None,
            first_onset_lead_hours=float((at-new.timestamp.min()).total_seconds()/3600) if len(new) else None,
            first_source_warning=past.timestamp.min().isoformat() if len(past) else None,
            first_source_warning_lead_hours=float((at-past.timestamp.min()).total_seconds()/3600) if len(past) else None))
    known=negative|positive;y=p.loc[known,'label'].astype(int)
    total=len(events);normal_hours=int(negative.sum())
    return dict(scheduled_hours=len(p),available_hours=int(available.sum()),unavailable_hours=int((~available).sum()),
        availability=float(available.mean()) if len(p) else 0.,known_scored_hours=int(known.sum()),
        tp=tp,fn=fn,fp=fp,tn=tn,point_precision=tp/(tp+fp) if tp+fp else None,point_recall=tp/(tp+fn) if tp+fn else None,
        known_normal_hours=normal_hours,normal_warning_hours=fp,normal_warning_fraction=fp/normal_hours if normal_hours else None,
        false_alarm_episodes=int(false_starts.sum()),false_alarms_per_720_normal_hours=float(false_starts.sum()*720/normal_hours) if normal_hours else None,
        unknown_warning_hours=int((active&p.label.isna()).sum()),left_censored_alarm_starts=int(left_censored.sum()),
        observed_onsets=int(onset.sum()),window_hit_events=sum(e['window_hit'] for e in events),
        onset_hit_events=sum(e['onset_hit'] for e in events),total_events=total,
        onset_event_recall=sum(e['onset_hit'] for e in events)/total if total else 0.,
        window_event_recall=sum(e['window_hit'] for e in events)/total if total else 0.,
        average_precision=float(average_precision_score(y,p.loc[known,'score'])) if y.nunique()==2 else None,
        roc_auc=float(roc_auc_score(y,p.loc[known,'score'])) if y.nunique()==2 else None,events=events)


def select_threshold(validation,episodes,config):
    p=validation.sort_values(['source_event','timestamp'])
    available=(p.history_ready&p.score.notna()&np.isfinite(p.score)).to_numpy()
    scores=p.score.fillna(-np.inf).to_numpy()
    negative=(p.label.eq(0).fillna(False).to_numpy()&available)
    positive=(p.label.eq(1).fillna(False).to_numpy()&available)
    known=negative|positive
    if negative.sum()==0:raise ValueError('Validation has no known normal exposure')
    continuous=(p.source_event.eq(p.source_event.shift())&p.timestamp.diff().eq(pd.Timedelta(hours=1))).to_numpy()
    previous_observed=np.r_[False,available[:-1]]&continuous
    masks=[p.source_event.eq(e['source_event']).to_numpy()&p.episode.eq(e['id']).fillna(False).to_numpy()&positive for e in episodes]
    best=None
    for threshold in np.unique(scores[known]):
        active=(scores>=threshold)&available
        false=active&negative
        fraction=float(false.sum()/negative.sum())
        if fraction>config['max_normal_warning_fraction']:continue
        starts=false&~(np.r_[False,false[:-1]]&continuous)
        rate=float(starts.sum()*720/negative.sum())
        if rate>config['false_alarm_budget_per_720_normal_hours']:continue
        onset=active&previous_observed&~np.r_[False,active[:-1]]
        recall=sum(bool((onset&m).any()) for m in masks)/len(masks) if masks else 0.
        if recall==0:continue
        precision=float((active&positive).sum()/(active&known).sum()) if (active&known).any() else 0.
        key=(recall,-fraction,-rate,precision,float(threshold))
        if best is None or key>best[0]:best=(key,float(threshold))
    threshold=best[1] if best else None
    return threshold,operational_metrics(validation,threshold,episodes)


class NormalBehaviourModel:
    """No IDs, annotations or future measurements enter the regressors."""
    def __init__(self,config):
        self.config=config

    def _regressor(self):
        return HistGradientBoostingRegressor(random_state=self.config['seed'],**self.config['regressor'])

    def fit(self,frame,train_assets):
        config=self.config;features=config['regime_features']
        allowed=frame[config['required_current_inputs']].notna().all(axis=1)
        train=frame.loc[frame.turbine_id.isin(train_assets)&frame.history_ready&frame.label.eq(0)&allowed].copy()
        if set(train.turbine_id)!=set(train_assets) or len(train_assets)<2:raise ValueError('Missing training turbine normal data')
        self.train_assets=list(train_assets);self.regressors={};self.calibration={};self.normal_counts={}
        self.fitted_keys=train[['turbine_id','source_event','timestamp']].reset_index(drop=True)
        for target in config['targets']:
            sample=train.loc[train[target].notna()]
            residuals=[]
            for heldout in train_assets:
                inner=sample.loc[sample.turbine_id.ne(heldout)];outer=sample.loc[sample.turbine_id.eq(heldout)]
                if min(len(inner),len(outer))<config['minimum_normal_rows_per_target']:raise ValueError(f'Insufficient normal data: {target}/{heldout}')
                inner_model=self._regressor().fit(inner[features],inner[target])
                residuals.extend((outer[target]-inner_model.predict(outer[features])).tolist())
            residuals=np.asarray(residuals)
            center=float(np.median(residuals));scale=max(float(np.quantile(residuals,.95)-center),1.)
            self.calibration[target]=dict(center_celsius=center,scale_celsius=scale,crossfit_rows=len(residuals))
            self.normal_counts[target]=len(sample)
            self.regressors[target]=self._regressor().fit(sample[features],sample[target])
        return self

    def predict(self,frame):
        config=self.config
        result=frame[['turbine_id','source_event','timestamp','history_ready']].copy()
        valid=frame.history_ready&frame[config['required_current_inputs']].notna().all(axis=1)
        normalized=[]
        for target in config['targets']:
            expected=pd.Series(np.nan,index=frame.index)
            if valid.any():expected.loc[valid]=self.regressors[target].predict(frame.loc[valid,config['regime_features']])
            actual=frame[target]
            residual=actual-expected
            calibration=self.calibration[target]
            z=(residual-calibration['center_celsius'])/calibration['scale_celsius']
            name=target.removesuffix('__now')
            result[name+'__actual']=actual
            result[name+'__expected']=expected
            result[name+'__residual']=residual
            result[name+'__normalized_residual']=z
            normalized.append(z.rename(name))
        z=pd.concat(normalized,axis=1)
        measured=z.notna().sum(axis=1)
        result['measured_target_count']=measured
        result['score']=z.max(axis=1).clip(lower=0).where(valid&measured.ge(config['minimum_target_count']))
        result['raw_residual_score']=result.score
        result['score']=smooth_scores(result,config['score_smoothing_hours'])
        return result
