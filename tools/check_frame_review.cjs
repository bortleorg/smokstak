// Offline source/interaction smoke test; not a browser layout or CSP test.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const file=path.resolve(process.argv[2]),page=fs.readFileSync(file,'utf8');
const raw=page.match(/<script id="review-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const report=JSON.parse(raw),code=page.match(/<script>([\s\S]*?)<\/script>/)[1];
const helper=require('../crates/sr-cli/src/frame_review.js');
assert.equal(helper.metricValue({suppressed:.25},'suppressed'),25);
assert.equal(helper.metricValue({hfd:null},'hfd'),null);
assert.equal(helper.metricValue({},'obstruction'),null,'old audit has unknown mask, not zero');
const maskFrame={obstruction_mask:[[false,true],[false,false]]};
assert.equal(helper.metricValue(maskFrame,'obstruction'),25);
assert.equal(helper.maskedAt(maskFrame,1,0),true);
assert.equal(helper.maskedAt(maskFrame,0,1),false);
assert.equal(helper.maskFraction({obstruction_mask:[[null]]}),null);
assert.equal(helper.metricValue({photometric_gain:[18]},'photometric_gain'),18);
assert.equal(helper.metricValue({photometric_gain:[.9,1.1,1]},'photometric_gain'),1);
assert.equal(helper.metricValue({photometric_gain:[null]},'photometric_gain'),null);
const range=helper.stretchRange([{pixels:[-2,0,1]},{pixels:[2,3,4]}]);
assert.ok(range.low<0&&range.high>3);
assert.equal(helper.displayValue(null,range,8),null);
const queueFixture=[
  {path:'late-H.fit',filter:'H',status:'used',capture_time:'2026-09-25T04:00:00',exposure_seconds:180,weight:.7},
  {path:'early-O.fit',filter:'O',status:'used',capture_time:'2026-09-25T01:00:00Z',exposure_seconds:180,weight:1},
  {path:'cloud-H.fit',filter:'H',status:'not-used',reason:'Cloud obstruction',capture_time:'2026-09-25T02:00:00.123456',exposure_seconds:180,weight:0},
  {path:'near-H.fit',filter:'H',status:'used',capture_time:'2026-09-24T19:05:00-07:00',exposure_seconds:null,weight:1},
  {path:'unknown-H.fit',filter:'H',status:'used',weight:null}
];
assert.deepEqual(helper.filteredIndices(queueFixture,'H','*','','capture'),[2,3,0,4]);
assert.deepEqual(helper.filteredIndices(queueFixture,'H','*','cloud 2026','capture'),[2]);
assert.deepEqual(helper.filteredIndices(queueFixture,'H','used','','weight'),[0,3,4]);
assert.equal(helper.comparisonIndex(queueFixture,2),3,'nearest used exposure in same filter, not nearest build index');
assert.deepEqual(helper.integration(queueFixture),{seconds:540,unknown:2});
assert.equal(helper.captureStamp({capture_time:'2026-09-25T02:00:00'}),Date.parse('2026-09-25T02:00:00Z'));
assert.equal(helper.captureStamp({capture_time:'nonsense'}),null);
assert.equal(helper.externalPath(String.raw`\\?\G:\images\one.fit`),String.raw`G:\images\one.fit`);
assert.equal(helper.externalPath(String.raw`\\?\UNC\server\astro\one.fit`),String.raw`\\server\astro\one.fit`);
assert.equal(helper.externalPath('/astro/one.fit'),'/astro/one.fit');
const gainFixture=queueFixture.map((f,i)=>({...f,iso_or_gain:i===0||i===2?300:50}));
assert.equal(helper.comparisonIndex(gainFixture,2),0,'same-gain used peer preferred over closer capture at different gain');
assert.deepEqual(helper.filteredIndices(gainFixture,'H','*','','capture','300'),[2,0]);
const loaded=[],downloads=[];let scope;
function element(tag){return {tag,children:[],value:'',style:{},attributes:{},clientWidth:900,
  append(...nodes){this.children.push(...nodes);if(this.tag==='head')for(const node of nodes){
    assert.match(node.src,/^review-assets\/[a-f0-9]{64}\.js$/);loaded.push(node.src);
    try{vm.runInContext(fs.readFileSync(path.join(path.dirname(file),node.src),'utf8'),scope);node.onload?.();}catch(error){node.onerror?.();throw error;}
  }},replaceChildren(...nodes){this.children=[...nodes]},setAttribute(k,v){this.attributes[k]=v},
  addEventListener(k,fn){this[k]=fn},remove(){},focus(){},click(){},
  getContext(){const canvas=this;return {createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData(image){canvas.imageData=image.data}}}};}
const ids={};for(const m of page.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))ids[m[2]]=element(m[1]);
ids['review-data'].textContent=raw;
for(const [k,v]of Object.entries({filter:'*',status:'*',metric:'weight',zoom:'2',gain:'8',search:'',order:'capture','sensor-gain':'*'}))if(ids[k])ids[k].value=v;
scope=vm.createContext({document:{head:element('head'),body:element('body'),getElementById:id=>{assert.ok(ids[id],id);return ids[id]},createElement:element,createElementNS:(_,tag)=>element(tag),addEventListener(){}},window:{addEventListener(){}},setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){},Blob,URL:{createObjectURL:blob=> {downloads.push(blob);return 'blob:test'},revokeObjectURL(){}},navigator:{clipboard:{writeText:async()=>{}}}});
vm.runInContext(code,scope,{timeout:5000});
(async()=>{
  await new Promise(setImmediate);
  assert.ok(loaded.length<=2,'only selected and comparison assets should load');
  assert.equal(Number(ids['count-excluded'].textContent),report.frames.filter(f=>f.status==='excluded').length);
  for(const mode of ['aligned','native','thumb']){ids['mode-'+mode].click();await new Promise(setImmediate);assert.equal(ids['mode-'+mode].attributes['aria-pressed'],'true');}
  if(ids['mode-sky']){
    ids['mode-sky'].click();await new Promise(setImmediate);
    assert.equal(ids['mode-sky'].attributes['aria-pressed'],'true');
    assert.match(ids['mode-note'].textContent,/Red shows the production obstruction mask/);
    ids.metric.value='obstruction';ids.metric.change();assert.ok(ids.graph.children.length);
    const masked=report.frames.find(f=>helper.maskFraction(f)>0&&f.preview_asset);
    if(masked&&ids.search){
      ids.search.value=helper.basename(masked.path);ids.search.input();await new Promise(setImmediate);
      ids.rows.children[0].click();await new Promise(setImmediate);
      const canvas=ids['selected-image'].children.find(c=>c.tag==='canvas');
      assert.ok(canvas&&canvas.imageData,'registered sky preview loaded');
      assert.ok(canvas.imageData.some((v,i)=>i%4===0&&v>canvas.imageData[i+1]),'production mask appears red on registered sky');
      assert.match(ids['selected-reason'].textContent,/Obstruction mask:/);
      ids.search.value='';ids.search.input();await new Promise(setImmediate);
    }
  }
  ids.gain.value='20';ids.gain.input();assert.equal(ids['gain-value'].textContent,'20');
  ids['reset-stretch'].click();assert.equal(ids.gain.value,'8');
  for(const metric of ['hfd','weight','residual','suppressed','eccentricity']){ids.metric.value=metric;ids.metric.change();assert.ok(ids.graph.children.length);}
  if(ids['normalization-note']){
    ids.metric.value='photometric_gain';ids.metric.change();assert.ok(ids.graph.children.length);
    const fallback=report.frames.filter(f=>f.status==='used'&&['level-only','exposure'].includes(f.photometry_source)).length;
    assert.equal(ids['normalization-note'].className,fallback?'readout':'hidden');
  }
  for(const status of ['used','not-used','excluded']){ids.status.value=status;ids.status.change();await new Promise(setImmediate);assert.equal(ids.rows.children.length,Math.min(40,report.frames.filter(f=>f.status===status).length));}
  if(ids.search){
    const exportedPath = code.includes('const externalPath') ? helper.externalPath : p=>p;
    ids.status.value='*';ids.status.change();
    ids.filter.value=report.frames[0].filter || '';ids.filter.change();
    ids.order.value='capture';ids.order.change();
    const expected=helper.filteredIndices(report.frames,ids.filter.value,'*','','capture');
    ids['download-matching'].click();
    assert.equal(await downloads.at(-1).text(),expected.map(i=>exportedPath(report.frames[i].path)).join('\n')+'\n','export all matching pages, not comparison or other filters');
    ids.rows.children[0].click();await new Promise(setImmediate);
    if(expected.length>1){ids['frame-next'].click();await new Promise(setImmediate);assert.equal(ids['selected-name'].textContent,helper.basename(report.frames[expected[1]].path));ids['frame-previous'].click();}
    ids.search.value='missing-evidence-search-012345';ids.search.input();
    assert.equal(ids.rows.children.length,0);assert.equal(ids['download-matching'].disabled,true);assert.equal(ids['frame-next'].disabled,true);
    ids.search.value='';ids.search.input();
    ids.order.value='suppressed';ids.order.change();
    const sorted=helper.filteredIndices(report.frames,ids.filter.value,'*','','suppressed');
    ids['download-matching'].click();assert.equal(await downloads.at(-1).text(),sorted.map(i=>exportedPath(report.frames[i].path)).join('\n')+'\n');
  }
  ids.filter.value='no-such-filter';ids.filter.change();
  assert.equal(ids.rows.children.length,0);assert.equal(ids['selected-name'].textContent,'No frame selected');
  assert.equal(ids['comparison-name'].textContent,'No frame selected');
  for(const status of ['used','not-used','excluded'])assert.equal(helper.frameList(report.frames,status).trim().split('\n').filter(Boolean).length,report.frames.filter(f=>f.status===status).length);
  assert.ok(!page.includes('__REVIEW_JSON__')&&!page.includes('__REVIEW_JS__'));
  console.log(`Frame review passed: ${report.frames.length} rows; real lazy assets and controls exercised. No browser layout/CSP verification.`);
})().catch(error=>{console.error(error);process.exitCode=1});
