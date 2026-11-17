/**
 * Retry read-only startup loading until the API is ready or the view is abandoned.
 * @template T
 * @param {()=>Promise<T>} load
 * @param {{signal:AbortSignal,onError?:(error:unknown)=>void,retryMs?:number}} options
 * @returns {Promise<T|undefined>}
 */
export async function retryBootstrap(load,{signal,onError,retryMs=1000}){
 let failures=0;
 while(!signal.aborted){
  try{
   const result=await load();
   return signal.aborted?undefined:result;
  }catch(error){
   if(signal.aborted)return;
   onError?.(error);
   if(signal.aborted)return;
   await new Promise(resolve=>{
    const done=()=>{clearTimeout(timer);signal.removeEventListener('abort',done);resolve();};
    const timer=setTimeout(done,Math.min(5000,retryMs*2**Math.min(failures++,3)));
    signal.addEventListener('abort',done,{once:true});
    if(signal.aborted)done();
   });
  }
 }
}
