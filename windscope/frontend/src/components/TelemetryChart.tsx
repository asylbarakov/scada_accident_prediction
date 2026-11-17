'use client';
import {useEffect,useRef} from 'react';
import * as echarts from 'echarts';

export type Point={row_id:number;source_time:string;values:Record<string,number|null>;partition:string;gap_before_seconds?:number|null};
export type Prediction={id?:number;row_id:number;source_time:string;processed_at?:string;status:string;predicted_class:number|null;class_1_score:number|null;model_id?:string;prediction_horizon?:string|null;prediction_horizon_hours?:number;prediction_interval_start?:string;prediction_interval_end?:string;feature_cutoff?:string;threshold?:number;score_semantics?:string;cadence_minutes?:number};
export type Annotation={start:string;end:string|null;visible_until:string;label:string;description:string};
export const labels:Record<string,string>={sensor_0_avg:'Окружающая среда',sensor_43_avg:'Гондола',sensor_11_avg:'Подшипник редуктора',wind_speed_3_avg:'Скорость ветра',voltage:'Среднее напряжение',sensor_50:'sensor_50 · исходный сигнал'};
const time=(value:string)=>Date.parse(value.replace(' ','T')+'Z');
export function value(p:Point,key:string){if(key==='voltage'){const v=['sensor_32_avg','sensor_33_avg','sensor_34_avg'].map(k=>p.values[k]);return v.every(n=>n!=null)?v.reduce<number>((a,n)=>a+(n??0),0)/3:null;}return p.values[key];}

export default function TelemetryChart({points,predictions,annotations,sensors,unit,model=false,names={},virtualTime,focusTime}:{points:Point[];predictions:Prediction[];annotations:Annotation[];sensors:string[];unit:string;model?:boolean;names?:Record<string,string>;virtualTime?:string|null;focusTime?:string}){
 const host=useRef<HTMLDivElement>(null);const chart=useRef<echarts.ECharts|null>(null);
 useEffect(()=>{if(!host.current)return;chart.current=echarts.init(host.current,undefined,{renderer:'canvas'});const resize=new ResizeObserver(()=>chart.current?.resize());resize.observe(host.current);return()=>{resize.disconnect();chart.current?.dispose();chart.current=null;};},[]);
 useEffect(()=>{
   if(!chart.current)return;
   const colors=['#38bdf8','#34d399','#5b7cfa'];
   const scores:Array<[number,number|null]>=[];let last:number|undefined;
   const h7=predictions.some(p=>p.prediction_horizon==='P7D');
   const cadence=h7?3600000:600000;
   for(const p of predictions){const t=time(p.source_time);if(last&&t-last>cadence)scores.push([t-1,null]);scores.push([t,p.class_1_score]);last=t;}
   const series:Record<string,unknown>[]=model?[{name:'Score класса 1',type:'line',showSymbol:false,lineStyle:{width:1.7,color:'#fbbf24'},itemStyle:{color:'#fbbf24'},data:scores},
    {name:'Класс 1',type:'scatter',symbolSize:5,itemStyle:{color:'#f59e0b'},data:predictions.filter(p=>p.predicted_class===1).map(p=>[time(p.source_time),1.04])}]:sensors.map((key,i)=>{
     const data:Array<[number,number|null]>=[];let previous:number|undefined;
     for(const p of points){const t=time(p.source_time);if(previous&&(p.gap_before_seconds||0)>600)data.push([t-1,null]);data.push([t,value(p,key)]);previous=t;}
     return {name:names[key]||labels[key]||key,type:'line',showSymbol:false,connectNulls:false,lineStyle:{width:1.8,color:colors[i%3]},itemStyle:{color:colors[i%3]},data};
   });
   if(model&&h7){
    const decisions:Array<[number,number|null]>=[];let previous:number|undefined;
    for(const p of predictions){const t=time(p.source_time);if(previous&&t-previous>cadence)decisions.push([t-1,null]);decisions.push([t,p.status==='ok'?p.predicted_class:null]);previous=t;}
    series.splice(0,series.length,
     {name:'Предупреждение H7',type:'line',step:'end',connectNulls:false,showSymbol:true,symbolSize:5,lineStyle:{width:2.2,color:'#fbbf24'},itemStyle:{color:'#fbbf24'},data:decisions},
     {name:'Score · не вероятность',type:'line',showSymbol:false,lineStyle:{width:1,color:'#8a96a3',type:'dashed'},itemStyle:{color:'#8a96a3'},data:scores});
   }
   if(series.length&&annotations.some(a=>a.label==='anomaly'))series[0].markArea={silent:true,itemStyle:{color:'rgba(251,191,36,.08)'},data:annotations.filter(a=>a.label==='anomaly').map(a=>[{xAxis:time(a.start),name:'Аннотация набора'},{xAxis:time(a.end||a.visible_until)}])};
   const marker=focusTime||virtualTime;
   if(series.length&&marker&&points.length&&time(marker)>=time(points[0].source_time)&&time(marker)<=time(points[points.length-1].source_time))series[0].markLine={silent:true,symbol:'none',lineStyle:{color:'#38bdf8',type:'dashed',width:1},label:{formatter:focusTime?'Предупреждение':'Время replay',color:'#8a96a3',fontSize:10,position:'insideEndTop'},data:[{xAxis:time(marker)}]};
   chart.current.setOption({animation:false,useUTC:true,backgroundColor:'transparent',textStyle:{fontFamily:'Inter,system-ui,sans-serif',color:'#8a96a3'},
    color:colors,grid:{left:55,right:24,top:60,bottom:62},legend:{type:'scroll',top:0,left:12,right:12,textStyle:{color:'#c3ccd6',fontSize:11},itemWidth:14,itemHeight:7,pageIconColor:'#38bdf8',pageTextStyle:{color:'#8a96a3'}},
    tooltip:{trigger:'axis',backgroundColor:'#121922',borderColor:'#252f3b',textStyle:{color:'#e6edf3'},axisPointer:{type:'line'},confine:true},
    xAxis:{type:'time',axisLine:{lineStyle:{color:'#252f3b'}},axisTick:{show:false},splitLine:{show:false},axisLabel:{color:'#8a96a3',fontSize:10}},
    yAxis:{type:'value',scale:!model,min:model?0:undefined,max:model?1.1:undefined,name:unit,nameTextStyle:{color:'#8a96a3'},axisLabel:{color:'#8a96a3',fontSize:10},splitLine:{lineStyle:{color:'#1b232d'}}},
    dataZoom:[{type:'inside'},{type:'slider',height:16,bottom:8,borderColor:'#252f3b',backgroundColor:'#0c1117',fillerColor:'rgba(56,189,248,.08)',handleStyle:{color:'#38bdf8'},textStyle:{color:'#8a96a3'},dataBackground:{lineStyle:{color:'#5b6673'},areaStyle:{color:'#1b232d'}}}],series},{notMerge:false,replaceMerge:['series'],lazyUpdate:true});
 },[points,predictions,annotations,sensors,unit,model,names,virtualTime,focusTime]);
 return <div ref={host} className={model?'chart model-chart':'chart'} role="img" aria-label={model?'Динамика score модели и аннотации':'Интерактивный график SCADA; масштабируйте колёсиком или ползунком'}/>;
}
