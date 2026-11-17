import test from 'node:test';
import assert from 'node:assert/strict';
let view;try{view=await import('../src/lib/alert-view.mjs');}catch{}
const alerts=[{id:1,turbine_id:'0',severity:'warning',status:'active'},{id:2,turbine_id:'11',severity:'observation',status:'resolved'}];
test('all alert filters apply together and hidden selection cannot keep stale details visible',()=>{
 assert.ok(view,'Filtered alert view is missing');
 const visible=view.filterAlerts(alerts,'0','warning','open');assert.deepEqual(visible.map(a=>a.id),[1]);
 assert.equal(view.selectedDetail(visible,2).id,1);
 const empty=view.filterAlerts(alerts,'0','critical','open');assert.equal(view.selectedDetail(empty,1),undefined);
 assert.deepEqual(view.filterAlerts(alerts,'11','','resolved').map(a=>a.id),[2]);
});
