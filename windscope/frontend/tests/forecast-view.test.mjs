import test from 'node:test';
import assert from 'node:assert/strict';
let view;try{view=await import('../src/lib/forecast-view.mjs');}catch{}
const model_id='care-h7-logistic-20261116';
test('a held hourly prediction becomes stale and missing results never imply no warning',()=>{
 assert.ok(view,'Forecast freshness handling is missing');
 const s={model_id,virtual_time:'2023-08-12 10:50:00',latest_prediction:{status:'ok',source_time:'2023-08-12 10:00:00',predicted_class:1,class_1_score:0.99999999}};
 assert.equal(view.forecastState(s),'warning');
 assert.equal(view.forecastState({...s,virtual_time:'2023-08-12 11:00:00'}),'stale');
 assert.equal(view.forecastState({...s,latest_prediction:null}),'unknown');
 assert.equal(view.forecastState({...s,latest_prediction:{...s.latest_prediction,status:'unavailable'}}),'unknown');
 assert.equal(view.forecastState({...s,latest_prediction:{...s.latest_prediction,predicted_class:0}}),'below_threshold');
});
