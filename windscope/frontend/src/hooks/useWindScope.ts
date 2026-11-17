'use client';
import {useCallback,useEffect,useRef,useState} from 'react';
import type {Point,Prediction,Annotation} from '../components/TelemetryChart';
import type {Sensor} from '../components/SensorPicker';
import type {Alert,Snapshot,Turbine,Policy} from '../lib/types';
import {retryBootstrap} from '../lib/bootstrap-recovery.mjs';
import {preferredSession} from '../lib/monitor-status.mjs';
type Scenario={id:string;asset_id:string;prediction_start:string;prediction_end:string;prediction_points:number;rows:number};
export async function api<T>(path:string,options?:RequestInit):Promise<T>{const r=await fetch('/api/v1'+path,{...options,headers:{'Content-Type':'application/json',...options?.headers},cache:'no-store'});if(!r.ok){const b=await r.json().catch(()=>({}));throw new Error(typeof b.detail==='string'?b.detail:'Запрос не выполнен. Проверьте подключение и параметры.');}return r.json();}
const merge=<T extends {row_id:number}>(a:T[],b:T[])=>[...new Map([...a,...b].map(p=>[p.row_id,p])).values()].sort((a,b)=>a.row_id-b.row_id).slice(-1400);

export function useWindScope(){
 const [fleet,setFleet]=useState<Turbine[]>([]),[scenarios,setScenarios]=useState<Scenario[]>([]),[catalog,setCatalog]=useState<Sensor[]>([]);
 const [sid,setSid]=useState(''),[session,setSession]=useState<Snapshot|null>(null),[points,setPoints]=useState<Point[]>([]),[predictions,setPredictions]=useState<Prediction[]>([]),[annotations,setAnnotations]=useState<Annotation[]>([]),[alerts,setAlerts]=useState<Alert[]>([]),[center,setCenter]=useState<Alert[]>([]);
 const [mode,setMode]=useState<'replay'|'analysis'>('replay'),[connection,setConnection]=useState('connecting'),[error,setError]=useState(''),[busy,setBusy]=useState(false),[now,setNow]=useState(''),[rangeStart,setRangeStart]=useState(''),[rangeEnd,setRangeEnd]=useState(''),[notice,setNotice]=useState(''),[focusTime,setFocusTime]=useState('');
 const lastSeq=useRef(0),generation=useRef(0),activeId=useRef(sid),modeRef=useRef(mode),sessionRef=useRef(session),snapshotSeq=useRef(0),focusRef=useRef(focusTime),seen=useRef(new Set<number>());
 activeId.current=sid;modeRef.current=mode;sessionRef.current=session;focusRef.current=focusTime;
 const refreshFleet=useCallback(async()=>{const [f,a]=await Promise.all([api<Turbine[]>('/turbines'),api<Alert[]>('/alerts?limit=1000')]);setFleet(f);setCenter(a);},[]);
 const loadData=useCallback(async(id:string,m:string)=>{
  const requestedGeneration=generation.current,requestedFocus=focusRef.current;
  let query='';if(requestedFocus){const t=Date.parse(requestedFocus.replace(' ','T')+'Z');const iso=(n:number)=>new Date(n).toISOString().slice(0,19);query=`&start=${iso(t-10800000)}&end=${iso(t+10800000)}`;}
  const [h,p,a,als]=await Promise.all([api<{points:Point[]}>(`/replay-sessions/${id}/telemetry?mode=${m}&max_points=1200${query}`),api<Prediction[]>(`/replay-sessions/${id}/predictions?limit=1200`),api<Annotation[]>(`/replay-sessions/${id}/annotations?mode=${m}`),api<Alert[]>(`/replay-sessions/${id}/alerts`)]);
  if(activeId.current!==id||modeRef.current!==m||generation.current!==requestedGeneration||focusRef.current!==requestedFocus)return;
  setPoints(old=>m==='replay'&&!requestedFocus?merge(h.points,old):h.points);setPredictions(old=>merge(p,old));setAnnotations(a);setAlerts(als);
 },[]);
 useEffect(()=>{const controller=new AbortController();void retryBootstrap(async()=>{const [f,s,c,a,d]=await Promise.all([api<Turbine[]>('/turbines'),api<Scenario[]>('/scenarios'),api<Sensor[]>('/sensors'),api<Alert[]>('/alerts?limit=1000'),api<Snapshot[]>('/replay-sessions?scenario_id=22')]);if(controller.signal.aborted)return;const remembered=localStorage.getItem('windscope-session-v3');let restored=remembered;if(remembered)try{const rememberedSession=await api<Snapshot>(`/replay-sessions/${remembered}`);if(!rememberedSession.monitoring&&preferredSession(d)?.monitoring)restored=null;}catch{restored=null;}if(controller.signal.aborted)return;setFleet(f);setScenarios(s);setCatalog(c);setCenter(a);setSid(restored||preferredSession(d)?.id||f.find(t=>t.scenario_id==='22')?.session?.id||f[0]?.session?.id||'');setError('');},{signal:controller.signal,onError:()=>setError('Сервисы пока недоступны. Повторяем подключение…')});return()=>controller.abort();},[]);
 useEffect(()=>{const update=()=>setNow(new Date().toISOString().slice(11,19));update();const timer=setInterval(update,1000);return()=>clearInterval(timer);},[]);
 useEffect(()=>{let disposed=false;const timer=setInterval(()=>{refreshFleet().catch(()=>{});if(!sid)return;const m=modeRef.current,g=generation.current;const current=()=>!disposed&&activeId.current===sid&&generation.current===g;
  api<Alert[]>(`/replay-sessions/${sid}/alerts`).then(a=>{if(current())setAlerts(a);}).catch(()=>{});
  api<Annotation[]>(`/replay-sessions/${sid}/annotations?mode=${m}`).then(a=>{if(current()&&modeRef.current===m)setAnnotations(a);}).catch(()=>{});
 },3000);return()=>{disposed=true;clearInterval(timer);};},[sid,refreshFleet]);
 useEffect(()=>{
  if(!sid)return;setSession(null);setPoints([]);setPredictions([]);setAnnotations([]);setAlerts([]);let disposed=false;const controller=new AbortController();let socket:WebSocket|undefined,timer:ReturnType<typeof setTimeout>|undefined,retry=0;
  const snapshot=(s:Snapshot)=>{if(disposed||activeId.current!==s.id)return;if(s.generation>generation.current){generation.current=s.generation;clear();loadData(sid,modeRef.current).catch(()=>{});}setSession(old=>old?.id===s.id&&(old.generation>s.generation||(old.generation===s.generation&&old.sequence>s.sequence))?old:s);};
  const clear=()=>{setPoints([]);setPredictions([]);setAnnotations([]);setAlerts([]);};
  const connect=()=>{if(disposed)return;setConnection(retry?'reconnecting':'connecting');socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.hostname}:8100/api/v1/replay-sessions/${sid}/stream?after_seq=${lastSeq.current}`);
   socket.onopen=()=>{retry=0;setConnection('connected');};socket.onmessage=e=>{if(disposed)return;const msg=JSON.parse(e.data);if(msg.type==='heartbeat')return;
    if(msg.type==='snapshot'){snapshotSeq.current=msg.session.sequence;snapshot(msg.session);return;}
    if(msg.sequence<=lastSeq.current)return;lastSeq.current=msg.sequence;if(msg.generation<generation.current)return;
    if(msg.generation>generation.current){generation.current=msg.generation;clear();setSession(old=>old?{...old,generation:msg.generation,latest_prediction:null,diagnostic:null,prediction_count:0,position:0,virtual_time:null}:old);loadData(sid,modeRef.current).catch(()=>{});}
    if(msg.type==='measurement'){
     const p=msg.payload;if(modeRef.current==='replay'&&!focusRef.current)setPoints(old=>merge(old,[p.measurement]));if(p.prediction)setPredictions(old=>merge(old,[p.prediction]));
     setSession(old=>old?{...old,latest:p.measurement,latest_prediction:p.prediction||old.latest_prediction,diagnostic:p.diagnostic||old.diagnostic,prediction_count:p.prediction_count??old.prediction_count,position:p.position,progress:p.position/p.total*100,virtual_time:p.virtual_time,status:p.status,sequence:msg.sequence,model_state:p.prediction?(p.prediction.status==='ok'?(p.prediction.predicted_class===1?'warning':'normal'):'unknown'):old.model_state}:old);
    }else if(msg.type.startsWith('alert_')){
     if(msg.type==='alert_activated'&&msg.sequence>snapshotSeq.current&&!seen.current.has(msg.payload.alert_id)){seen.current.add(msg.payload.alert_id);setNotice(sessionRef.current?.model_id==='care-h7-logistic-20261116'?`Прогноз H7: возможен отказ в следующие 7 суток. Предупреждение #${msg.payload.alert_id}.`:`Новое предупреждение #${msg.payload.alert_id}. Откройте центр предупреждений, чтобы изучить измерения.`);}
     refreshFleet().catch(()=>{});api<Alert[]>(`/replay-sessions/${sid}/alerts`).then(a=>{if(!disposed)setAlerts(a);}).catch(()=>{});
    }else if(msg.type==='state_changed'||msg.type==='completed'||msg.type==='cycle_started'){api<Snapshot>(`/replay-sessions/${sid}`).then(snapshot).catch(()=>{});if(msg.type==='cycle_started')refreshFleet().catch(()=>{});}
   };socket.onerror=()=>socket?.close();socket.onclose=()=>{if(!disposed){setConnection('reconnecting');timer=setTimeout(connect,Math.min(5000,500*2**retry++));}};
  };
  void retryBootstrap(async()=>{const s=await api<Snapshot>(`/replay-sessions/${sid}`);if(disposed)return;generation.current=s.generation;lastSeq.current=s.sequence;snapshotSeq.current=s.sequence;snapshot(s);setRangeStart(s.start.replace(' ','T').slice(0,16));setRangeEnd(s.end.replace(' ','T').slice(0,16));await loadData(sid,modeRef.current);if(!disposed){localStorage.setItem('windscope-session-v3',sid);setError('');connect();}},{signal:controller.signal,onError:()=>{if(!disposed)setError('Не удалось загрузить сессию. Повторяем подключение…');}});
  return()=>{disposed=true;controller.abort();if(timer)clearTimeout(timer);socket?.close();};
 },[sid,loadData,refreshFleet]);
 const initialized=useRef(false);
 useEffect(()=>{if(!initialized.current){initialized.current=true;return;}if(sid){setPoints([]);setAnnotations([]);loadData(sid,mode).catch(e=>setError(e.message));}},[mode,focusTime,loadData]); // session bootstrap owns sid changes
 const switchMode=(m:'replay'|'analysis')=>{if(m===mode)return;setPoints([]);setAnnotations([]);setMode(m);};
 const send=async(action:string,speed?:number)=>{const s=sessionRef.current;if(!s)return;setBusy(true);setError('');try{const next=await api<Snapshot>(`/replay-sessions/${s.id}/commands`,{method:'POST',body:JSON.stringify({action,revision:s.revision,command_id:crypto.randomUUID(),...(speed?{speed}:{})})});if(activeId.current!==s.id)return;setSession(next);if(next.generation!==generation.current){generation.current=next.generation;lastSeq.current=next.sequence;setPoints([]);setPredictions([]);setAnnotations([]);setAlerts([]);setFocusTime('');await loadData(next.id,modeRef.current);}await refreshFleet();}catch(e){setError((e as Error).message);api<Snapshot>(`/replay-sessions/${s.id}`).then(n=>{if(activeId.current===s.id)setSession(n);}).catch(()=>{});}finally{setBusy(false);}};
 const newRange=async(policy:Policy,full=false,model_id='care-h7-logistic-20261116')=>{if(!session)return;setBusy(true);setError('');try{const s=await api<Snapshot>('/replay-sessions',{method:'POST',body:JSON.stringify({scenario_id:session.scenario_id,model_id,speed:session.speed,...(!full?{start:rangeStart.replace('T',' '),end:rangeEnd.replace('T',' ')}:{}),alert_policy:{activate_points:policy.activate_points,resolve_points:policy.resolve_points,cooldown_minutes:policy.cooldown_minutes}})});setFocusTime('');setSid(s.id);await refreshFleet();}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
 const startMonitoring=async(scenario=sessionRef.current?.scenario_id||'22')=>{setBusy(true);setError('');try{const current=sessionRef.current;const next=await api<Snapshot>('/monitoring',{method:'POST',body:JSON.stringify({scenario_id:scenario,speed:current?.continuous?current.speed:300})});setFocusTime('');setMode('replay');setSid(next.id);setSession(next);await refreshFleet();}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
 const choose=async(scenario:string)=>{await startMonitoring(scenario);};
 const acknowledge=async(id:number)=>{setBusy(true);try{const a=await api<Alert>(`/alerts/${id}/acknowledge`,{method:'POST'});setAlerts(old=>old.map(v=>v.id===id?a:v));await refreshFleet();}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
 return {fleet,scenarios,catalog,sid,setSid,session,points,predictions,annotations,alerts,center,mode,switchMode,connection,error,setError,busy,now,rangeStart,setRangeStart,rangeEnd,setRangeEnd,notice,setNotice,focusTime,setFocusTime,send,newRange,choose,acknowledge,startMonitoring};
}
