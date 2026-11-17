import test from 'node:test';
import assert from 'node:assert/strict';
let monitor;try{monitor=await import('../src/lib/monitor-status.mjs');}catch{}
test('monitor cadence explains wall time and the next real hourly forecast',()=>{
 assert.ok(monitor,'Monitoring status is missing');
 const info=monitor.monitorStatus({speed:300,virtual_time:'2023-08-12 10:20:00',prediction_count:2});
 assert.equal(info.measurementSeconds,2);
 assert.equal(info.predictionSeconds,12);
 assert.equal(info.nextForecast,'2023-08-12 11:00:00');
 assert.equal(info.predictionCount,2);
 const slow=monitor.monitorStatus({speed:60,virtual_time:'2023-08-12 10:00:00'});
 assert.equal(slow.measurementSeconds,10);assert.equal(slow.predictionSeconds,60);
});
test('full-scenario monitoring is preferred over a short test session',()=>{
 assert.ok(monitor,'Monitoring selection is missing');
 assert.equal(monitor.preferredSession([{id:'test',continuous:true,monitoring:false,model_id:'care-h7-logistic-20261116'},{id:'monitor',continuous:true,monitoring:true,model_id:'care-h7-logistic-20261116'}]).id,'monitor');
});
