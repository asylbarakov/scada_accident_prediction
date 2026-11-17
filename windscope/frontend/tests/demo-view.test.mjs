import test from 'node:test';
import assert from 'node:assert/strict';
let view;try{view=await import('../src/lib/demo-view.mjs');}catch{}

test('a lowered score keeps an operational watch and never implies repaired equipment',()=>{
 assert.ok(view,'Demo view is missing');
 const s={status:'running',virtual_time:'2023-10-19 09:10:00',latest_prediction:{status:'ok',source_time:'2023-10-19 09:00:00',predicted_class:0}};
 assert.equal(view.demoState(s,[{status:'active'}],null).kind,'watch');
 assert.equal(view.demoState(s,[],null).kind,'below');
 assert.equal(view.demoState({...s,latest_prediction:{status:'warming_up'}},[],null).kind,'unknown');
 assert.equal(view.demoState(s,[],{state:'missed_failure'}).kind,'missed');
});

test('deviation index is a ratio to the threshold and source pacing is explicit',()=>{
 assert.ok(view,'Demo view is missing');
 assert.equal(view.warningIndex({deviation_score:2.6,threshold:2}),1.3);
 assert.equal(view.warningIndex({deviation_score:null,threshold:2}),null);
 assert.equal(view.stepSeconds(300),2);assert.equal(view.stepSeconds(75),8);
 assert.equal(view.leadLabel(2+1/3),'2 ч 20 мин');
});

test('stale or future model output does not become a current successful prediction',()=>{
 assert.ok(view,'Demo view is missing');
 const s={status:'running',virtual_time:'2023-10-19 10:00:00',latest_prediction:{status:'ok',source_time:'2023-10-19 08:00:00',predicted_class:1}};
 assert.equal(view.demoState(s,[],null).kind,'unknown');
 assert.equal(view.demoState({...s,latest_prediction:{...s.latest_prediction,source_time:'2023-10-19 11:00:00'}},[],null).kind,'unknown');
});
