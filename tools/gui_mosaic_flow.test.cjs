// Race regression tests for the inline GUI controller, without a browser/runtime dependency.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const html = readFileSync(require('node:path').join(__dirname,'../crates/sr-cli/src/gui.html'),'utf8');
const invalidation = html.slice(html.indexOf('function invalidateMosaic()'),html.indexOf("for (const id of ['mosaic-plan'"));
const review = html.slice(html.indexOf('async function reviewMosaic()'),html.indexOf("$('mosaic-review').addEventListener('click', reviewMosaic)"));
function controller() {
  const nodes = new Map(); const pending = []; const rendered = [];
  const $ = id => { if (!nodes.has(id)) nodes.set(id,{value:'',disabled:false,hidden:false}); return nodes.get(id); };
  const context = vm.createContext({$,Number,JSON,Error,Set,
    post:(_url,request)=>new Promise(resolve=>pending.push({request,resolve})),
    showErr:()=>{}, remember:()=>{}, esc:s=>s,
    mosaicGroup:f=>f.group, drawMosaicSources:()=>rendered.push($('mosaic-plan').value)});
  vm.runInContext(`let mosaicReview=null,mosaicReviewKey='',mosaicReviewSerial=0,mosaicReviewBusy=false;
    const mosaicRequest=()=>({plan:$('mosaic-plan').value,tile:256,memory_mb:2048});
    const mosaicKey=()=>JSON.stringify(mosaicRequest());
    function mosaicCanBuild(){ $('mosaic-build').disabled=mosaicReviewBusy||!mosaicReview||mosaicReviewKey!==mosaicKey(); }
    ${invalidation}\n${review}`,context);
  const run = code => vm.runInContext(code,context);
  const response = plan => ({ok:true,plan,plan_sha256:plan,frames:[],width:10,height:10,scale:1,filter:'H',calibration:'raw',notes:[]});
  return {$,pending,rendered,run,response};
}
test('new history selection wins even when the previous plan review is in flight',async()=>{
  const c=controller(); c.$('mosaic-plan').value='A';const first=c.run('reviewMosaic()');
  c.$('mosaic-plan').value='B';const second=c.run('reviewMosaic()');
  c.pending[0].resolve(c.response('A'));await first;
  assert.equal(c.$('mosaic-plan').value,'B'); assert.equal(c.$('mosaic-build').disabled,true);
  assert.deepEqual(c.rendered,[]);
  c.pending[1].resolve(c.response('B'));await second;
  assert.deepEqual(c.rendered,['B']); assert.equal(c.$('mosaic-build').disabled,false);
});
test('changing selected frames invalidates an outstanding review without leaving controls busy',async()=>{
  const c=controller();c.$('mosaic-plan').value='old';const request=c.run('reviewMosaic()');
  c.$('mosaic-plan').value='';c.run('invalidateMosaic()');
  c.pending[0].resolve(c.response('old'));await request;
  assert.equal(c.$('mosaic-plan').value,'');assert.equal(c.$('mosaic-review').disabled,false);
  assert.equal(c.$('mosaic-build').disabled,true);assert.deepEqual(c.rendered,[]);
});

test('Stack handoff freezes precisely the inspected frames and explicit exclusions',()=>{
  const nodes=new Map();const handlers=new Map();
  const $=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',addEventListener:(_event,fn)=>handlers.set(id,fn)});return nodes.get(id);};
  const context=vm.createContext({$,found:{mosaic_selection:[['a.fits','G:/frames/a.fits'],['b.fits','G:/frames/b.fits']]},
    excluded:new Set(['b.fits']),showErr:()=>assert.fail('Unexpected input error'),invalidateMosaic:()=>{}});
  const start=html.indexOf("$('mosaic-use-loaded').addEventListener('click', () => {");
  const end=html.indexOf("$('start-mosaic').addEventListener",start);
  vm.runInContext(`let mosaicLoadedExclusions=[];${html.slice(start,end)}`,context);
  handlers.get('mosaic-use-loaded')();
  assert.equal($('mosaic-input').value,'');
  assert.equal($('mosaic-input-text').value,'G:/frames/a.fits');
  assert.match($('mosaic-input-note').textContent,/1 frames selected/);
  assert.equal($('mosaic-plan').value,'');
  assert.ok(html.indexOf('id="mosaic-input-note"')>html.indexOf('</details>',html.indexOf('Other ways to choose frames')));
});
