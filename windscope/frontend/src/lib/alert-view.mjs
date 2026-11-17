export function filterAlerts(alerts,turbine,severity,status){
 return alerts.filter(a=>(!turbine||a.turbine_id===turbine)&&(!severity||a.severity===severity)&&(status==='open'?a.status!=='resolved':!status||a.status===status));
}
export function selectedDetail(visible,id){return visible.find(a=>a.id===id)||visible[0];}
