export const H7_MODEL='care-h7-logistic-20261116';
export function forecastState(session){
 const p=session?.latest_prediction;
 if(!p||p.status!=='ok'||p.predicted_class==null)return 'unknown';
 const time=v=>Date.parse(v.replace(' ','T')+'Z');
 if(session.virtual_time&&time(session.virtual_time)-time(p.source_time)>=3600000)return 'stale';
 return p.predicted_class===1?'warning':'below_threshold';
}
