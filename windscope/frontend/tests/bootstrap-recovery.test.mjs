import test from 'node:test';
import assert from 'node:assert/strict';
let recovery;try{recovery=await import('../src/lib/bootstrap-recovery.mjs');}catch{}

test('loading recovers when the API becomes available after the page opened',async()=>{
 assert.ok(recovery,'Initial API loading cannot recover');
 let attempts=0;
 const result=await recovery.retryBootstrap(async()=>{
  if(++attempts<3)throw new TypeError('Failed to fetch');
  return {scenario:'22',session:'restored',sensors:81};
 },{signal:new AbortController().signal,retryMs:0});
 assert.deepEqual(result,{scenario:'22',session:'restored',sensors:81});
 assert.equal(attempts,3);
});

test('unmount during an outage stops further loading attempts',async()=>{
 assert.ok(recovery,'Cancellable recovery is missing');
 const controller=new AbortController();let attempts=0;
 const pending=recovery.retryBootstrap(async()=>{attempts++;throw new Error('offline');},
  {signal:controller.signal,retryMs:10000,onError:()=>controller.abort()});
 assert.equal(await pending,undefined);
 assert.equal(attempts,1);
});

test('a late response from an abandoned session cannot restore stale state',async()=>{
 assert.ok(recovery,'Abandoned loading attempts are not cancelled');
 const controller=new AbortController();let resolve;
 const pending=recovery.retryBootstrap(()=>new Promise(r=>{resolve=r;}),{signal:controller.signal});
 controller.abort();resolve({session:'old'});
 assert.equal(await pending,undefined);
});
