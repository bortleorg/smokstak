// Source-level offline report smoke test. This is not a browser layout test.
// Uses only Node's standard library; no network and no rendering engine.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const page=fs.readFileSync(process.argv[2],'utf8');
const json=page.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(json);
const context=new Proxy({}, {get:()=>()=>{}});
function element(tag){return {tag,children:[],value:'',style:{},checked:false,textContent:'',attributes:{},
  setAttribute(name,value){this.attributes[name]=value},focus(){this.focused=true},
  append(...nodes){this.children.push(...nodes);if(this.tag==='select'&&this.children.length===1)this.value=nodes[0].value;},
  replaceChildren(...nodes){this.children=[...nodes]},
  addEventListener(name,callback){this[name]=callback},scrollIntoView(){},
  getBoundingClientRect(){return {width:620,height:270,left:0,top:0}},getContext(){return context}};}
const ids={};for(const match of page.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))ids[match[2]]=element(match[1]);
ids.data.textContent=json;ids['depth-axis'].value='hours';ids['time-axis'].value='index';ids.log.checked=false;
for(const [id,value] of Object.entries({'review-scope':'jumps','jump-threshold':'1','collection-goal':'review','noise-units':'relative','night-sort':'latest','night-boundary':'12','extra-hours':'','noise-goal':'','minimum-gain':'','pick-region':'signal','preview-stretch':'1','snr-goal':''}))if(ids[id])ids[id].value=value;
const scope={document:{getElementById:id=>{assert.ok(ids[id],id);return ids[id]},createElement:element},
  window:{devicePixelRatio:1,addEventListener(){}},innerWidth:1440,innerHeight:1000};
const code=page.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
vm.runInNewContext(code,scope,{timeout:5000});
assert.equal(vm.runInNewContext('excludedFrames.size',scope),0);
assert.equal(ids['review-export'].disabled,true);
ids['review-mark-visible'].onclick();
const marked=vm.runInNewContext('excludedFrames.size',scope);
assert.equal(marked,vm.runInNewContext('reviewRows(data.filters[0]).rows.length',scope));
if(data.filters.length>1){ids.filter.value='1';ids.filter.change();assert.equal(vm.runInNewContext('excludedFrames.size',scope),marked);ids.filter.value='0';ids.filter.change()}
ids['review-clear'].onclick();assert.equal(vm.runInNewContext('excludedFrames.size',scope),0);
ids['night-review-frames'].onclick();assert.equal(ids['review-scope'].value,'session');assert.equal(ids['review-details'].open,true);
ids['review-scope'].value='jumps';ids['review-scope'].change();
assert.equal(ids.version.textContent,data.schema_version);
if(ids['collection-summary']){
 assert.match(ids['collection-badge'].textContent,/no stopping target/i);
 assert.equal(ids['noise-goal'].value,'');assert.equal(ids['snr-goal'].value,'');
 assert.equal(vm.runInNewContext('state(data.filters[0]).target',scope),null);
 assert.equal(vm.runInNewContext('state(data.filters[0]).reduction',scope),null);
 assert.ok(!ids['collection-answer'].textContent.startsWith('About '));
 if(ids['collection-review-night'].style.display!=='none'){ids['collection-review-night'].onclick();assert.ok(vm.runInNewContext('state(data.filters[0]).night',scope));assert.equal(ids['night-details'].open,true)}
 assert.equal(ids['improvement-options'].children.length,data.filters[0].integration_depth.some(d=>d.noise>0)?3:0);
}
for(const id of ['health-details','noise-details','region-details'])if(ids[id]){assert.equal(typeof ids[id].toggle,'function');ids[id].toggle();}
for(let i=0;i<data.filters.length;i++){
  ids.filter.value=String(i);ids.filter.change();
  assert.equal(ids.cards.children.length,4);
  assert.equal(ids['health-cards'].children.length,4);
  assert.equal(ids.frames.children.length,data.frames.filter(f=>f.filter===data.filters[i].filter).length);
  ids['time-axis'].value='time';ids['time-axis'].change();
  ids['depth-axis'].value='frame_count';ids['depth-axis'].change();
  ids.log.checked=true;ids.log.change();
  if(data.filters[i].snapshots?.length){
    const patches=data.filters[i].snapshots.at(-1).patches;
    if(patches.length>1){
      ids['signal-patch'].value=String(patches[0].id);ids['signal-patch'].change();
      ids['background-patch'].value=String(patches[1].id);ids['background-patch'].change();
      assert.match(ids['region-result'].textContent,data.filters[i].snapshots.some(s=>s.frame_count%2===0)?/contrast SNR/:/Select signal/);
      ids['background-patch'].value=String(patches[0].id);ids['background-patch'].change();
      assert.match(ids['region-result'].textContent,/different patches/);
      ids['background-patch'].value=String(patches[1].id);ids['background-patch'].change();
      ids['snr-goal'].value='10';ids['snr-goal'].change();
      ids['preview-stretch'].value='4';ids['preview-stretch'].change();
      assert.equal(vm.runInNewContext('state(data.filters[Number($("filter").value)]).target',scope),10);
    }
  }
}
// Section shortcuts disclose their content; goals remain explicit.
for(const [button,target] of [['nav-night','night-details'],['nav-plan','forecast-details'],['nav-region','region-details'],['nav-health','health-details']]){
 ids[target].open=false;ids[button].onclick();assert.equal(ids[target].open,true);
}
ids['region-set-goal'].onclick();assert.equal(ids['collection-goal'].value,'snr');
assert.equal(ids['goal-controls'].open,true);assert.equal(ids['snr-goal'].focused,true);
// Sorting changes the inspection order, never the measurement accumulation order.
scope.nightOrder=vm.runInNewContext('nightsFor(data.filters[Number($("filter").value)]).map(n=>n.id).join()',scope);
for(const sort of ['latest','increase','order']){
 ids['night-sort'].value=sort;ids['night-sort'].change();
 const dates=vm.runInNewContext(`(()=>{const ns=nightsFor(data.filters[Number($('filter').value)]).slice();if($('night-sort').value==='latest')ns.reverse();if($('night-sort').value==='increase')ns.sort((a,b)=>(a.change??Infinity)-(b.change??Infinity));return ns.map(n=>n.date).join()})()`,scope);
 assert.equal(ids.nights.children[0].children.slice(1).map(row=>row.children[0].children[0].textContent).join(),dates);
 assert.equal(vm.runInNewContext('nightsFor(data.filters[Number($("filter").value)]).map(n=>n.id).join()',scope),scope.nightOrder);
}
assert.equal(vm.runInNewContext('noiseChange(-12).textContent',scope),'12% higher');
assert.equal(vm.runInNewContext('noiseChange(12).textContent',scope),'12% lower');
assert.equal(vm.runInNewContext('noiseChange(null).textContent',scope),'Unavailable');
for(const log of [false,true]){
 scope.axisLog=log;
 const ticks=vm.runInNewContext('numericTicks(2,346,axisLog,400,true)',scope);
 assert.ok(ticks.length>=2&&ticks.every(n=>Number.isInteger(n)&&n>=2&&n<=346));
}
if(ids['collection-summary']){
 const synthetic={filter:'T',integration_depth:[{frame_count:100,integration_seconds:90000,unknown_exposures:0,noise:1}],snapshots:[{frame_count:100,integration_seconds:90000,patches:[{id:1,median:10,noise:3},{id:2,median:0,noise:4}]}]};
 scope.adviceFixture=synthetic;
 const advice=expression=>vm.runInNewContext(expression,scope);
 assert.equal(advice('collectionAdvice(adviceFixture,[],{mode:"snr",region:{signal:1,background:2,target:1}}).status'),'met');
 assert.match(advice('collectionAdvice(adviceFixture,[],{mode:"snr",region:{signal:1,background:2,target:1}}).answer'),/^0 more/);
 assert.match(advice('collectionAdvice(adviceFixture,[],{mode:"snr",region:{signal:1,background:2,target:4}}).answer'),/75.*hours/);
 assert.match(advice('collectionAdvice(adviceFixture,[],{mode:"noise",reduction:.1}).answer'),/6.*hours/);
 assert.match(advice('collectionAdvice(adviceFixture,[],{mode:"noise",reduction:.1}).next'),/too weak/);
 assert.match(advice('collectionAdvice(adviceFixture,[],{mode:"review"}).answer'),/Review the trend/);
 assert.equal(advice('collectionAdvice(adviceFixture,[],{mode:"snr",region:{signal:1,background:2,target:null}}).status'),'missing');
 assert.match(advice('collectionAdvice(adviceFixture,[{date:"2026-01-01",rows:[],pairs:[],hours:0,start:10,before:{frame_count:10,noise:10},after:{frame_count:20,noise:12}}],{mode:"review"}).answer'),/Investigate the recent noise increase/);
 assert.match(advice('collectionAdvice(adviceFixture,[{date:"2026-01-01",rows:[],pairs:[],hours:0,start:10,before:{frame_count:10,noise:10},after:{frame_count:20,noise:8}}],{mode:"review"}).answer'),/Noise is falling/);
 assert.match(advice('collectionAdvice(adviceFixture,[],{mode:"noise",reduction:null}).answer'),/Choose an improvement/);
 // An explicit choice activates a scenario; merely opening the report does not.
 if(ids['improvement-options'].children.length){
  ids['improvement-options'].children[0].onclick();
  assert.equal(ids['collection-goal'].value,'noise');assert.equal(Number(ids['noise-goal'].value),5);
  ids['noise-goal'].value='';ids['noise-goal'].change();
  assert.match(ids['collection-answer'].textContent,/Choose an improvement/);
  ids['collection-goal'].value='review';ids['collection-goal'].change();
 }
 // Recent progress is net change between endpoints, not a sum of percentages.
 assert.equal(advice('recentProgress([{start:0,after:{frame_count:10,noise:10}},{start:10,before:{frame_count:10,noise:10},after:{frame_count:20,noise:9}},{start:20,before:{frame_count:20,noise:9},after:{frame_count:30,noise:8}},{start:30,before:{frame_count:30,noise:8},after:{frame_count:40,noise:7}}]).change'),30.000000000000004);

 assert.equal(advice('collectionAdvice(adviceFixture,[],{mode:"snr",region:{signal:1,background:1,target:4}}).status'),'missing');
 assert.equal(advice('collectionAdvice(adviceFixture,[],{mode:"snr",region:{signal:2,background:1,target:4}}).status'),'missing');
 advice('adviceFixture.integration_depth[0].unknown_exposures=1');
 assert.match(advice('collectionAdvice(adviceFixture,[],{mode:"noise",reduction:.1}).answer'),/24.*frames/);
 for(const mode of ['snr','noise','review']){ids['collection-goal'].value=mode;ids['collection-goal'].change();assert.ok(ids['collection-answer'].textContent.length>0)}
}
if(ids['extra-hours']){
 assert.equal(vm.runInNewContext('requiredFrames(100,.1,1)',scope),24);
 assert.ok(Math.abs(vm.runInNewContext('noiseRatio(100,100,1)',scope)-Math.SQRT1_2)<1e-12);
 assert.equal(vm.runInNewContext('requiredFrames(100,0,1)',scope),null);
 assert.equal(vm.runInNewContext('forecastScenarios({},[{date:"Unknown date",pairs:[1,1,1]},{date:"Unknown date",pairs:[1,1,1]},{date:"Unknown date",pairs:[1,1,1]}],{noise:1,frame_count:10}).length',scope),0);
 ids['night-boundary'].value='12';
 assert.equal(vm.runInNewContext('nightKey({capture_time:"2026-09-24T02:00:00Z"})',scope),'2026-09-23');
 assert.equal(vm.runInNewContext('nightKey({capture_time:null})',scope),'Unknown date');
 const accounting=vm.runInNewContext(`(()=>{const g=data.filters[0],ns=nightsFor(g);return {accepted:ns.reduce((s,n)=>s+n.end-n.start,0),expected:g.accepted_frames,invalidPairs:ns.some(n=>n.pairs.some(x=>!Number.isFinite(x)))}})()`,scope);
 assert.equal(accounting.accepted,accounting.expected);assert.equal(accounting.invalidPairs,false);
 const backup=vm.runInNewContext('data.filters[0].integration_depth.map(d=>d.unknown_exposures)',scope);
 if(backup.length){
  vm.runInNewContext('data.filters[0].integration_depth.forEach(d=>d.unknown_exposures=1);renderForecast(data.filters[0],nightsFor(data.filters[0]))',scope);
  assert.match(ids['goal-result'].textContent,/incomplete/);
  scope.savedUnknown=Array.from(backup);vm.runInNewContext('data.filters[0].integration_depth.forEach((d,i)=>d.unknown_exposures=savedUnknown[i])',scope);
 }
 // Contrast definition includes noise in both regions; no implicit aperture sqrt(Npix).
 assert.equal(vm.runInNewContext('regionSeries({snapshots:[{frame_count:2,integration_seconds:60,patches:[{id:1,median:15,noise:3},{id:2,median:5,noise:4}]}]},{signal:1,background:2})[0].snr',scope),2);
}
assert.ok(!page.includes('https://')&&!page.includes('http://'));
if(process.argv[3])assert.deepEqual(data,JSON.parse(fs.readFileSync(process.argv[3],'utf8')));
console.log(`Report source smoke test passed: ${data.filters.length} filters, ${data.frames.length} frames; embedded data matches JSON.`);
