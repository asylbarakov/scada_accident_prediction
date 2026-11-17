'use client';
import {Activity,Clock3,ShieldAlert} from 'lucide-react';
import type {Snapshot} from '../lib/types';
import {date,fmt} from '../lib/types';
import {forecastState} from '../lib/forecast-view.mjs';
import {monitorStatus} from '../lib/monitor-status.mjs';

const duration=(seconds:number)=>seconds>=60?`${fmt(seconds/60,1)} мин`:`${fmt(seconds,2)} с`;

export default function ForecastStatus({session,connected}:{session:Snapshot;connected:boolean}){
 const state=forecastState(session),info=monitorStatus(session),p=session.latest_prediction;
 const title=state==='warning'?'Возможен отказ в следующие 7 суток':state==='below_threshold'?'Порог предупреждения H7 не превышен':state==='stale'?'Ожидаем свежий прогноз H7':p?.status==='warming_up'?'Накопление истории для прогноза':p?.status==='data_gap'?'Прогноз недоступен: разрыв измерений':p?.status==='unavailable'?'Сервис прогноза недоступен':'Ожидаем первый расчёт H7';
 return <section className={'forecast-status '+state} aria-label="Прогноз и работа потока">
  <div className="forecast-status-heading"><ShieldAlert size={20}/><div><strong>{title}</strong><p>{p?.status==='ok'?`Расчёт по данным на ${date(p.source_time)} · горизонт до ${date(p.prediction_interval_end)}`:'Модель использует прошлые измерения. Минимальная история — 24 часа.'}</p></div><span className="source-label"><Activity size={13}/>{!connected?'Переподключение':session.status==='running'?'Поток идёт':session.status==='paused'?'Поток на паузе':session.status==='completed'?'Отрезок завершён':'Готов к запуску'}</span></div>
  <div className="forecast-status-facts"><div><span>Измерения в текущем проходе</span><strong>{session.position.toLocaleString('ru-RU')}</strong><small>обычно раз в {duration(info.measurementSeconds)}</small></div><div><span>Рассчитано прогнозов H7</span><strong>{info.predictionCount}</strong><small>примерно раз в {duration(info.predictionSeconds)} при этой скорости</small></div><div><span><Clock3 size={12}/> Следующий расчёт по шкале SCADA</span><strong>{date(info.nextForecast)}</strong><small>один час истории между прогнозами</small></div></div>
  <p className="forecast-status-note">H7 оценивает возможность отказа в течение 7 суток. Точное время поломки не определяется; score не является калиброванной вероятностью.</p>
 </section>;
}
