export function groupMeasurements(catalog,query,category,advanced){
 const search=query.trim().toLocaleLowerCase('ru');const groups=new Map();
 for(const sensor of catalog){
  if(category==='recommended'&&!sensor.recommended)continue;
  if(category!=='all'&&category!=='recommended'&&sensor.group!==category)continue;
  const id=sensor.physical_key||sensor.key;
  if(!groups.has(id))groups.set(id,{...sensor,physical_key:id,variants:[]});
  groups.get(id).variants.push(sensor);
 }
 return [...groups.values()].filter(g=>!search||[g.display_name,g.physical_key,...g.variants.map(s=>s.key)].join(' ').toLocaleLowerCase('ru').includes(search))
  .map(g=>({...g,variants:g.variants.filter(s=>advanced||s.statistic==='avg')})).filter(g=>g.variants.length);
}
