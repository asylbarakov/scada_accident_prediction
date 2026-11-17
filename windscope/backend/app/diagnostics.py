"""Causal observations, not a component-failure classifier or safety thresholds."""
import math
from datetime import datetime,timedelta
from statistics import median

VERSION='causal-observations-v2.0'
TEMPERATURES={
    'sensor_13_avg':('generator','Подшипник генератора DE'),
    'sensor_14_avg':('generator','Подшипник генератора NDE'),
    'sensor_15_avg':('generator','Обмотка генератора L1'),
    'sensor_16_avg':('generator','Обмотка генератора L2'),
    'sensor_17_avg':('generator','Обмотка генератора L3'),
    'sensor_11_avg':('gearbox','Подшипник редуктора'),
    'sensor_12_avg':('gearbox','Масло редуктора'),
}
EVIDENCE={**{k:(v[1],'°C') for k,v in TEMPERATURES.items()},
    'sensor_43_avg':('Гондола','°C'),'wind_speed_3_avg':('Скорость ветра','m/s'),
    'sensor_32_avg':('Напряжение L1','V'),'sensor_18_avg':('Обороты генератора','rpm')}
REQUIRED=('sensor_0_avg','sensor_1_avg','sensor_43_avg','sensor_11_avg','wind_speed_3_avg','sensor_32_avg','sensor_33_avg','sensor_34_avg','sensor_50')


def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def quality(values):return all(finite(values.get(k)) for k in REQUIRED)


def forecast_observation(measurement,result):
    """Describe a future-horizon signal without inferring a component or cause."""
    valid=result.get('status')=='ok'
    positive=valid and result.get('predicted_class')==1
    if result.get('model_id')=='care-personal-normal-h24-v1':
        return dict(turbine_id=measurement['asset_id'],timestamp=measurement['source_time'],detection_source='ML',model_output=dict(result),
            suspected_subsystem=None,category='early_warning' if valid else 'data_quality',condition='not_assessed',
            explanation='Температурные отклонения превысили персональный порог. Это возможный предвестник отказа.' if positive else 'Порог раннего сигнала сейчас не превышен.' if valid else 'Недостаточно непрерывной истории для расчёта раннего сигнала.',
            evidence=result.get('evidence',[]),severity='warning' if positive else 'none' if valid else 'observation',prediction_horizon='P1D',version='personal-normal-v1',
            confidence_limitations='Историческая проверка: 4 из 11 отказов, 17 ложных эпизодов. Score не является вероятностью; причина и время поломки не определяются.',
            suggested_action='Проверить отклонения датчиков и журнал эксплуатации. Снижение score не подтверждает устранение неисправности.')
    explanation=('Модель предупреждает о возможном отказе в следующие 7 суток.' if positive else
                 'Порог предупреждения об отказе в следующие 7 суток не превышен.' if valid else
                 'Прогноз H7 недоступен: недостаточно истории или сервис модели не ответил.')
    return dict(turbine_id=measurement['asset_id'],timestamp=measurement['source_time'],detection_source='ML',
        model_output=dict(result),suspected_subsystem=None,category='future_failure' if valid else 'data_quality',
        condition='not_assessed',explanation=explanation,evidence=[],severity='warning' if positive else 'none' if valid else 'observation',
        prediction_horizon='P7D',version='care-h7-forecast-v1',
        confidence_limitations='Экспериментальная модель H7: на отложенном тесте обнаружен 1 из 2 отказов. Score не калиброван. Тип, причина и точное время отказа не определяются.',
        suggested_action='Проверить историю SCADA и журнал эксплуатации; запланировать инженерную проверку по регламенту. Отсутствие предупреждения не подтверждает исправность.')


def diagnose(measurement,result,past):
    now=datetime.fromisoformat(measurement['source_time']);values=measurement['values']
    evidence=[dict(sensor=k,label=label,value=values[k],unit=unit) for k,(label,unit) in EVIDENCE.items() if finite(values.get(k))]
    d=dict(turbine_id=measurement['asset_id'],timestamp=str(now),detection_source='ML',
        model_output={k:result.get(k) for k in ('status','predicted_class','class_1_score','model_id')},
        suspected_subsystem=None,category='unclassified',condition='not_assessed',
        explanation='Сигнал модели не обнаружен. Физическая исправность оборудования не оценена.',
        evidence=evidence[:6],severity='none',prediction_horizon=None,version=VERSION,
        confidence_limitations='Бинарная модель не определяет причину и не подтверждает отказ. Score не является вероятностью будущего отказа.',
        suggested_action='Продолжить наблюдение и плановые проверки по регламенту производителя.')
    if result.get('status')!='ok':
        d.update(category='data_quality',condition='unknown',detection_source='rule',severity='observation',
            explanation='Результат модели недоступен или входные данные неполны. Состояние оборудования неизвестно.',
            suggested_action='Проверить полноту входных датчиков и доступность сервиса модели; не считать отсутствие результата нормой.')
        return d
    if result.get('predicted_class')!=1:return d
    d.update(severity='observation',explanation='Обнаружено отклонение в работе оборудования. Причина не определена.',
        suggested_action='Сопоставить температуры генератора и редуктора, нагрузку и ветер с предыдущими измерениями; проверить журнал эксплуатации. Не останавливать оборудование только по этому сигналу.')
    # Explicit statistical heuristic: last 12 source hours, >=24 prior finite points,
    # upward robust deviation >=6 MAD scales. No OEM/safety threshold is claimed.
    causal=[p for p in past if now-timedelta(hours=12)<=datetime.fromisoformat(p['source_time'])<now]
    deviations=[]
    for key,(subsystem,label) in TEMPERATURES.items():
        if not finite(values.get(key)):continue
        samples=[p['values'][key] for p in causal if finite(p['values'].get(key))]
        if len(samples)<24:continue
        center=median(samples);mad=median([abs(x-center) for x in samples]);scale=1.4826*mad
        if scale<=1e-9:continue  # Flat reference cannot support a stable relative score.
        z=(values[key]-center)/scale
        if z>=6:deviations.append((z,key,subsystem,label,center,len(samples)))
    if deviations:
        z,key,subsystem,label,center,count=max(deviations)
        specific=dict(sensor=key,label=label,value=values[key],unit='°C',reference_median=center,
            reference_count=count,reference_hours=12,robust_deviation=round(z,2),method='upward_6_MAD')
        d.update(detection_source='rule',suspected_subsystem=subsystem,category='relative_temperature',
            explanation=f'{label}: необычное повышение относительно собственной предыстории за 12 часов. Это статистическое наблюдение, причина не установлена.',
            evidence=[specific]+[e for e in evidence[:5] if e['sensor']!=key],
            suggested_action=f'Проверить тренд «{label}», нагрузку, охлаждение и исправность датчика по регламенту производителя. Правило не подтверждает отказ узла.')
    return d
