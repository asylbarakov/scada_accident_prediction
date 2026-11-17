'use client';
import {useEffect,useRef} from 'react';
import * as echarts from 'echarts';
import type {DemoPoint,DemoPrediction} from '../lib/demo-types';
import {sourceTime,warningIndex} from '../lib/demo-view.mjs';

type Props={kind:'risk'|'temperature';points:DemoPoint[];predictions:DemoPrediction[];future:DemoPoint[];sensor:string;virtualTime:string|null;start:string;end:string;failureAt:string|null};
export default function DemoChart({kind,points,predictions,future,sensor,virtualTime,start,end,failureAt}:Props){
 const host=useRef<HTMLDivElement>(null),chart=useRef<echarts.ECharts|null>(null);
 useEffect(()=>{if(!host.current)return;chart.current=echarts.init(host.current,undefined,{renderer:'canvas'});const observer=new ResizeObserver(()=>chart.current?.resize());observer.observe(host.current);return()=>{observer.disconnect();chart.current?.dispose();chart.current=null;};},[]);
 useEffect(()=>{
  if(!chart.current)return;
  const cutoff=sourceTime(virtualTime||start),series:echarts.SeriesOption[]=[];
  const visible=predictions.filter(p=>sourceTime(p.source_time)<=cutoff);
  if(kind==='risk'){
   const data:Array<[number,number|null]>=[];let previous:number|undefined;
   for(const p of visible){const at=sourceTime(p.source_time);if(previous&&at-previous>3600000)data.push([at-1,null]);data.push([at,p.status==='ok'?warningIndex(p):null]);previous=at;}
   series.push({name:'Отклонение относительно порога',type:'line',showSymbol:true,symbolSize:6,connectNulls:false,lineStyle:{width:3,color:'#53c6e8'},itemStyle:{color:'#53c6e8'},areaStyle:{color:'rgba(83,198,232,.08)'},data,
    markLine:{silent:true,symbol:'none',data:[{yAxis:1,label:{formatter:'Порог предупреждения · 1,0',position:'insideEndTop',color:'#f0be5a'},lineStyle:{color:'#f0be5a',type:'dashed',width:1.5}}]}});
   series.push({name:'Ранний сигнал',type:'scatter',symbolSize:10,itemStyle:{color:'#f0be5a'},data:visible.filter(p=>p.status==='ok'&&p.predicted_class===1).map(p=>[sourceTime(p.source_time),warningIndex(p)])});
  }else{
   const measured:Array<[number,number|null]>=[];let previous:number|undefined;
   for(const p of points.filter(p=>sourceTime(p.source_time)<=cutoff)){const at=sourceTime(p.source_time);if(previous&&at-previous>600000)measured.push([at-1,null]);measured.push([at,p.values[sensor]??null]);previous=at;}
   series.push({name:'Измеренная температура',type:'line',showSymbol:false,connectNulls:false,lineStyle:{width:2.3,color:'#53c6e8'},data:measured});
   series.push({name:'Ожидаемая при этой нагрузке',type:'line',showSymbol:false,connectNulls:false,lineStyle:{width:2,color:'#bb9bdd',type:'dashed'},data:visible.filter(p=>p.status==='ok').map(p=>{const evidence=p.evidence?.find(e=>e.sensor===sensor);return [sourceTime(p.feature_cutoff||p.source_time),evidence?.expected??null];})});
   if(future.length)series.push({name:'Известное будущее из истории',type:'line',showSymbol:false,connectNulls:false,lineStyle:{width:1.7,color:'#8796a7',type:'dotted'},data:future.filter(p=>sourceTime(p.source_time)>=cutoff).map(p=>[sourceTime(p.source_time),p.values[sensor]??null])});
  }
  const markers:Record<string,unknown>[]=[{xAxis:cutoff,lineStyle:{color:'#7f91a5',type:'dashed',width:1},label:{formatter:'Сейчас',position:'insideEndBottom',color:'#aebaca'}}];
  if(failureAt)markers.push({xAxis:sourceTime(failureAt),lineStyle:{color:'#f17d76',width:1.5,type:'dotted'},label:{formatter:'Отказ по разметке',position:'insideEndTop',color:'#f17d76'}});
  // Give the clock its own silent series, preserving the risk threshold marker.
  series.push({type:'line',data:[],silent:true,markLine:{symbol:'none',silent:true,data:markers as never}});
  chart.current.setOption({animation:false,useUTC:true,backgroundColor:'transparent',textStyle:{fontFamily:'Inter,Segoe UI,sans-serif',color:'#a7b5c5'},
   tooltip:{trigger:'axis',confine:true,backgroundColor:'#132232',borderColor:'#2c3d50',textStyle:{color:'#e8eff5'},valueFormatter:(value:unknown)=>typeof value==='number'?value.toLocaleString('ru-RU',{maximumFractionDigits:2}):'нет данных'},
   legend:{top:8,left:18,right:18,textStyle:{color:'#aebdcb',fontSize:11},itemWidth:16,itemHeight:6},
   grid:{left:58,right:30,top:56,bottom:40},
   xAxis:{type:'time',min:sourceTime(start)-(kind==='temperature'?2:1)*3600000,max:sourceTime(end),splitNumber:(host.current?.clientWidth||800)<500?4:8,axisLabel:{color:'#899bad',fontSize:11,hideOverlap:true},axisLine:{lineStyle:{color:'#2a394b'}},splitLine:{show:false}},
   yAxis:{type:'value',scale:kind==='temperature',min:kind==='risk'?0:undefined,max:kind==='risk'?(extent:{max:number})=>Math.max(1.4,Math.ceil(extent.max*1.15*10)/10):undefined,name:kind==='risk'?'× порога':'°C',nameTextStyle:{color:'#899bad'},axisLabel:{color:'#899bad',fontSize:11},splitLine:{lineStyle:{color:'#203044',type:'dashed'}}},
   series},{notMerge:true});
 },[kind,points,predictions,future,sensor,virtualTime,start,end,failureAt]);
 return <div className={'demo-chart '+kind} ref={host} role="img" aria-label={kind==='risk'?'Ранний сигнал относительно порога: превышение единицы вызывает предупреждение':'Измеренная и ожидаемая температура; будущее из истории показано только в режиме проверки'}/>;
}
