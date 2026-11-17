import test from 'node:test';
import assert from 'node:assert/strict';
let selection;
try {selection=await import('../src/lib/sensor-selection.mjs');} catch {}
const catalog=[
 {key:'sensor_13_avg',physical_key:'sensor_13',statistic:'avg',display_name:'Подшипник генератора DE',group:'generator',unit:'°C',recommended:true},
 {key:'power_30_avg',physical_key:'power_30',statistic:'avg',display_name:'Активная мощность сети',group:'electrical',unit:'kW'},
 {key:'power_30_max',physical_key:'power_30',statistic:'max',display_name:'Активная мощность сети',group:'electrical',unit:'kW'},
];
test('search finds physical measurement by Russian name or CSV key and retains variants',()=>{
 assert.ok(selection,'Grouped searchable sensor selection is missing');
 assert.equal(selection.groupMeasurements(catalog,'мощность','all',true)[0].variants.length,2);
 assert.equal(selection.groupMeasurements(catalog,'power_30_max','all',true)[0].physical_key,'power_30');
 assert.equal(selection.groupMeasurements(catalog,'нет датчика','all',true).length,0);
});
test('recommended selection and advanced statistics preserve raw units and keys',()=>{
 assert.ok(selection,'Grouped searchable sensor selection is missing');
 assert.equal(selection.groupMeasurements(catalog,'','recommended',false).length,1);
 assert.equal(selection.groupMeasurements(catalog,'','electrical',false)[0].variants.length,1);
 assert.equal(selection.groupMeasurements(catalog,'','electrical',true)[0].variants[1].key,'power_30_max');
 assert.equal(selection.groupMeasurements(catalog,'','electrical',true)[0].unit,'kW');
});
