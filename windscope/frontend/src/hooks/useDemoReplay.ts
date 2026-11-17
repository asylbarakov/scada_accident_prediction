'use client';
import {useCallback,useEffect,useRef,useState} from 'react';
import {api} from './useWindScope';
import type {DemoSession,DemoCatalog,DemoPrediction,DemoPoint,DemoAlert,DemoCheck} from '../lib/demo-types';
import {PERSONAL_MODEL,sourceTime} from '../lib/demo-view.mjs';

const merge=<T extends {row_id:number}>(a:T[],b:T[])=>[...new Map([...a,...b].map(p=>[p.row_id,p])).values()].sort((a,b)=>a.row_id-b.row_id).slice(-800);

export function useDemoReplay(){
 const [catalog,setCatalog]=useState<DemoCatalog|null>(null),[session,setSession]=useState<DemoSession|null>(null),[sid,setSid]=useState('');
 const [points,setPoints]=useState<DemoPoint[]>([]),[predictions,setPredictions]=useState<DemoPrediction[]>([]),[alerts,setAlerts]=useState<DemoAlert[]>([]),[check,setCheck]=useState<DemoCheck|null>(null),[future,setFuture]=useState<DemoPoint[]>([]);
 const [reveal,setReveal]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState(''),[connection,setConnection]=useState('connecting'),[lastReceived,setLastReceived]=useState(0),[now,setNow]=useState(0),[bootstrap,setBootstrap]=useState(0);
 const current=useRef<DemoSession|null>(null),activeId=useRef(''),revealRef=useRef(false),generation=useRef(0),sequence=useRef(0);
 revealRef.current=reveal;
 const clear=useCallback(()=>{setPoints([]);setPredictions([]);setAlerts([]);setCheck(null);},[]);
 const accept=useCallback((next:DemoSession)=>{
  if(next.id!==activeId.current)return;
  const old=current.current;
  if(old?.id===next.id&&(old.generation>next.generation||(old.generation===next.generation&&old.sequence>next.sequence)))return;
  if(next.generation!==generation.current){generation.current=next.generation;clear();}
  sequence.current=Math.max(sequence.current,next.sequence);current.current=next;setSession(next);
 },[clear]);
 const load=useCallback(async(id:string)=>{
  const snap=await api<DemoSession>(`/replay-sessions/${id}`);
  if(activeId.current!==id)return;
  accept(snap);
  const gen=snap.generation;
  const requestedReveal=revealRef.current;
  const start=new Date(sourceTime(snap.start)-8*3600000).toISOString().slice(0,19).replace('T',' ');
  const [history,pred,items,outcome]=await Promise.all([
   api<{points:DemoPoint[]}>(`/replay-sessions/${id}/telemetry?max_points=800&start=${encodeURIComponent(start)}`),
   api<DemoPrediction[]>(`/replay-sessions/${id}/predictions?limit=800`),api<DemoAlert[]>(`/replay-sessions/${id}/alerts`),
   api<DemoCheck>(`/demo/${id}/check?reveal=${requestedReveal}`),
  ]);
  if(activeId.current!==id||generation.current!==gen)return;
  setPoints(old=>merge(old,history.points));setPredictions(old=>merge(old,pred));setAlerts(items);if(requestedReveal===revealRef.current)setCheck(outcome);setError('');
 },[accept]);
 useEffect(()=>{setNow(Date.now());const timer=setInterval(()=>setNow(Date.now()),1000);return()=>clearInterval(timer);},[]);
 useEffect(()=>{
  let disposed=false;
  (async()=>{
   try{
    const result=await api<DemoCatalog>('/demo/catalog');if(disposed)return;setCatalog(result);
    const saved=localStorage.getItem('windscope-personal-demo');let restored:DemoSession|null=null;
    if(saved)try{const next=await api<DemoSession>(`/replay-sessions/${saved}`);if(next.model_id===PERSONAL_MODEL&&result.cases.some(c=>c.id===next.scenario_id))restored=next;}catch{}
    if(disposed)return;
    const initial=restored||await api<DemoSession>('/demo/start',{method:'POST',body:JSON.stringify({scenario_id:'26',autoplay:false,speed:300,repeat:true})});
    if(disposed)return;activeId.current=initial.id;current.current=initial;generation.current=initial.generation;setSession(initial);setSid(initial.id);localStorage.setItem('windscope-personal-demo',initial.id);
   }catch(e){if(!disposed)setError((e as Error).message);}
  })();return()=>{disposed=true;};
 },[bootstrap]);
 useEffect(()=>{
  if(!sid)return;
  let disposed=false,socket:WebSocket|undefined,retry:ReturnType<typeof setTimeout>|undefined;
  const connect=()=>{
   if(disposed)return;setConnection('connecting');
   socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.hostname}:8100/api/v1/replay-sessions/${sid}/stream?after_seq=${sequence.current}`);
   socket.onopen=()=>{if(!disposed)setConnection('connected');};
   socket.onmessage=event=>{
    if(disposed||activeId.current!==sid)return;
    const message=JSON.parse(event.data);
    if(message.type==='heartbeat')return;
    if(message.type==='snapshot'){accept(message.session);return;}
    if(message.sequence<=sequence.current)return;sequence.current=message.sequence;
    if(message.generation<generation.current)return;
    if(message.generation!==generation.current){generation.current=message.generation;clear();void load(sid).catch(()=>{});}
    if(message.type==='measurement'){
     const p=message.payload;setLastReceived(Date.now());setPoints(old=>merge(old,[p.measurement]));if(p.prediction)setPredictions(old=>merge(old,[p.prediction]));
     const old=current.current;if(old)accept({...old,latest:p.measurement,latest_prediction:p.prediction||old.latest_prediction,diagnostic:p.diagnostic||old.diagnostic,position:p.position,total:p.total,progress:p.position/p.total*100,virtual_time:p.virtual_time,status:p.status,sequence:message.sequence,generation:message.generation,prediction_count:p.prediction_count??old.prediction_count});
    }else if(message.type.startsWith('alert_')||message.type==='cycle_started'||message.type==='state_changed'||message.type==='completed')void load(sid).catch(e=>setError(e.message));
   };
   socket.onerror=()=>socket?.close();socket.onclose=()=>{if(!disposed){setConnection('reconnecting');retry=setTimeout(connect,1500);}};
  };
  void load(sid).then(connect).catch(e=>{if(!disposed){setError(e.message);retry=setTimeout(connect,1500);}});
  const poll=setInterval(()=>{void load(sid).catch(e=>{if(!disposed)setError(e.message);});},2500);
  return()=>{disposed=true;clearInterval(poll);if(retry)clearTimeout(retry);socket?.close();};
 },[sid,accept,clear,load]);
 useEffect(()=>{
  setFuture([]);if(!sid)return;
  let alive=true;
  void api<DemoCheck>(`/demo/${sid}/check?reveal=${reveal}`).then(result=>{if(alive&&activeId.current===sid)setCheck(result);}).catch(()=>{});
  if(reveal&&current.current){
   const s=current.current;
   void api<{points:DemoPoint[]}>(`/replay-sessions/${sid}/telemetry?mode=analysis&max_points=1000&start=${encodeURIComponent(s.start)}&end=${encodeURIComponent(s.end)}`).then(result=>{if(alive&&revealRef.current&&activeId.current===sid)setFuture(result.points);}).catch(e=>setError(e.message));
  }
  return()=>{alive=false;};
 },[sid,reveal]);
 const choose=async(caseId:string)=>{
  setBusy(true);setError('');
  try{
   const next=await api<DemoSession>('/demo/start',{method:'POST',body:JSON.stringify({scenario_id:caseId,speed:current.current?.speed||300,autoplay:true,repeat:true,replace_session_id:current.current?.id})});
   clear();setReveal(false);setFuture([]);sequence.current=next.sequence;generation.current=next.generation;activeId.current=next.id;current.current=next;setSession(next);setSid(next.id);localStorage.setItem('windscope-personal-demo',next.id);await load(next.id);
  }catch(e){setError((e as Error).message);}finally{setBusy(false);}
 };
 const command=async(action:string,speed?:number)=>{
  const s=current.current;if(!s)return;
  const next=await api<DemoSession>(`/replay-sessions/${s.id}/commands`,{method:'POST',body:JSON.stringify({action,revision:s.revision,command_id:crypto.randomUUID(),...(speed?{speed}:{})})});
  if(action==='reset'){clear();setReveal(false);setFuture([]);sequence.current=next.sequence;generation.current=next.generation;}
  accept(next);return next;
 };
 const act=async(action:string,speed?:number)=>{setBusy(true);setError('');try{await command(action,speed);if(current.current)await load(current.current.id);}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
 const restart=async()=>{setBusy(true);try{await command('reset');await command('start');if(current.current)await load(current.current.id);}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
 const acknowledge=async(id:number)=>{setBusy(true);try{await api(`/alerts/${id}/acknowledge`,{method:'POST'});if(current.current)await load(current.current.id);}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
 return {catalog,session,points,predictions,alerts,check,future,reveal,setReveal,busy,error,connection,lastReceived,now,choose,act,restart,acknowledge,retry:()=>setBootstrap(value=>value+1)};
}
