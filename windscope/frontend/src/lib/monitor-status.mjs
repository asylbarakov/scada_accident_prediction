export function preferredSession(sessions){
 return sessions.find(s=>s.monitoring&&s.model_id==='care-h7-logistic-20261116')||sessions[0];
}
export function monitorStatus(session){
 const speed=session?.speed||300;
 const clock=session?.virtual_time||session?.start;
 let nextForecast=null;
 if(clock){
  const at=new Date(clock.replace(' ','T')+'Z');
  at.setUTCMinutes(0,0,0);at.setUTCHours(at.getUTCHours()+1);
  nextForecast=at.toISOString().slice(0,19).replace('T',' ');
 }
 return {measurementSeconds:600/speed,predictionSeconds:3600/speed,nextForecast,predictionCount:session?.prediction_count||0};
}
