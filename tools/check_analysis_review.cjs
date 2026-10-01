const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {jumpCandidates,reviewBundle,reviewZip}=require('../crates/sr-cli/src/analyze/review.js');
const filters=['H','O','H','H','O','H','O','H','O','H'];
const report={project:'test project',frames:filters.map((filter,index)=>({index,filter,accepted:index!==2,path:`C:\\images\\${index}.fits`,file:`${index}.fits`,flags:[]})),filters:[
 {filter:'H',integration_depth:[{frame_count:2,frame_index:3,noise:10},{frame_count:3,frame_index:5,noise:null},{frame_count:4,frame_index:7,noise:12},{frame_count:5,frame_index:9,noise:null}]},
 {filter:'O',integration_depth:[{frame_count:2,frame_index:4,noise:4},{frame_count:4,frame_index:8,noise:3}]}]};
const events=jumpCandidates(report,1);assert.equal(events.length,1);assert.deepEqual(events[0].frame_indices,[5,7]);assert.equal(jumpCandidates(report,21).length,0);
const selected=new Set([5,7]),files=reviewBundle(report,selected);
const lines=name=>files[name].split('\n').filter(s=>s&&!s.startsWith('#'));
assert.deepEqual(lines('questionable-frames.txt'),['C:\\images\\5.fits','C:\\images\\7.fits']);
assert.equal(lines('retained-frames.txt').length,8);assert.ok(lines('retained-frames.txt').includes('C:\\images\\2.fits')); // Do not silently drop previously rejected inputs.
assert.equal(new Set([...lines('retained-frames.txt'),...lines('questionable-frames.txt')]).size,report.frames.length);
assert.equal(JSON.parse(files['questionable-frames.json']).frames[0].jumps[0].from,2);
assert.throws(()=>reviewBundle(report,new Set()),/Select at least/);
assert.throws(()=>reviewBundle(report,new Set([999])),/absent/);
assert.throws(()=>reviewBundle(report,new Set(report.frames.map(f=>f.index))),/empty/);
assert.throws(()=>reviewBundle({...report,frames:report.frames.map(f=>({...f,path:'relative.fits'}))},selected),/absolute/);
const manual=reviewBundle(report,new Set([1]));assert.equal(JSON.parse(manual['questionable-frames.json']).frames[0].jumps.length,0);
const unicode={...report,frames:report.frames.map(f=>({...f,path:f.path.replace('images','nebula α [test]')}))};
const unicodeFiles=reviewBundle(unicode,selected);
if(process.argv[2]){fs.mkdirSync(process.argv[2],{recursive:true});fs.writeFileSync(path.join(process.argv[2],'review-fixture.zip'),Buffer.concat(reviewZip(unicodeFiles)));fs.writeFileSync(path.join(process.argv[2],'copy-questionable.ps1'),files['copy-questionable.ps1']);}
console.log('Review attribution, partitions, manual selection, export guards and ZIP fixture passed.');
