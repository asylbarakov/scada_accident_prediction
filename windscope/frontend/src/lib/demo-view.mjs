export const PERSONAL_MODEL='care-personal-normal-h24-v1';
export const sourceTime=value=>value?Date.parse(/[zZ]$|[+-]\d\d:\d\d$/.test(value)?value:value.replace(' ','T')+'Z'):NaN;
export const sourceLabel=value=>value?value.slice(0,16).replace('T',' '):'—';
export const shortTime=value=>value?value.slice(11,16):'—';
export const stepSeconds=speed=>Number.isFinite(speed)&&speed>0?600/speed:null;
export function warningIndex(prediction){
 const value=prediction?.deviation_score,threshold=prediction?.threshold;
 return typeof value==='number'&&Number.isFinite(value)&&typeof threshold==='number'&&threshold>0?value/threshold:null;
}
export function leadLabel(hours){
 if(typeof hours!=='number'||!Number.isFinite(hours))return '—';
 const minutes=Math.round(hours*60),h=Math.floor(minutes/60),m=minutes%60;
 return `${h?h+' ч ':''}${m||!h?m+' мин':''}`.trim();
}
/**
 * @param {import('./demo-types').DemoSession|null|undefined} session
 * @param {import('./demo-types').DemoAlert[]} alerts
 * @param {import('./demo-types').DemoCheck|null} check
 */
export function demoState(session,alerts=[],check=null){
 if(check?.state==='missed_failure')return {kind:'missed',title:'Отказ произошёл без раннего сигнала',note:'В этом историческом примере модель не предупредила о поломке. Отсутствие сигнала не означает исправность.'};
 const prediction=session?.latest_prediction;
 const age=sourceTime(session?.virtual_time)-sourceTime(prediction?.source_time);
 const valid=prediction?.status==='ok'&&age>=0&&age<3600000;
 if(valid&&prediction.predicted_class===1)return {kind:'warning',title:'Обнаружен ранний сигнал',note:'Отклонения от обычного режима превысили персональный порог. Требуется инженерная проверка в следующие сутки.'};
 if(alerts.some(alert=>alert.status!=='resolved'))return {kind:'watch',title:'Ранний сигнал остаётся на контроле',note:valid?'Текущий score снизился. Это не подтверждает устранение неисправности; наблюдение сохраняется.':'Новый расчёт недоступен. Предупреждение сохраняется, состояние оборудования не подтверждено.'};
 if(valid)return {kind:'below',title:'Сигнал ниже порога предупреждения',note:'В текущих измерениях модель не выделила сильного отклонения. Это не подтверждение исправности турбины.'};
 if(!session||session.status==='created'||(session.position===0&&!prediction))return {kind:'ready',title:'Готово к исторической проверке',note:'Нажмите «Запустить». Модель будет получать измерения по порядку и пересчитывать сигнал каждый час данных.'};
 return {kind:'unknown',title:'Ожидаем доступный расчёт модели',note:'Для расчёта нужны непрерывная история и доступные датчики. Измерения продолжают поступать.'};
}
