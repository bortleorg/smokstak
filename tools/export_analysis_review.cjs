// Export a provisional experiment from the same attribution logic as the report.
// No original images are read, copied, moved, or rejected by this tool.
const fs=require('node:fs'),path=require('node:path');
const {jumpCandidates,reviewBundle,reviewZip}=require('../crates/sr-cli/src/analyze/review.js');
const [input,output,thresholdText='1']=process.argv.slice(2),threshold=Number(thresholdText);
if(!input||!output||!Number.isFinite(threshold)||threshold<0||threshold>1000)throw Error('Usage: node tools/export_analysis_review.cjs REPORT.json NEW_OUTPUT_DIRECTORY [minimum_jump_percent=1]');
if(fs.existsSync(output))throw Error('Choose a new output directory; existing exports are preserved.');
const report=JSON.parse(fs.readFileSync(input,'utf8')),events=jumpCandidates(report,threshold);
const selected=new Set(events.flatMap(e=>e.frame_indices));
const files=reviewBundle(report,selected,threshold,'Provisional experiment: all frames added across upward steps meeting the threshold. Inspect originals before treating these candidates as bad frames.');
fs.mkdirSync(output,{recursive:true});
for(const [name,text] of Object.entries(files))fs.writeFileSync(path.join(output,name),text,{encoding:'utf8',flag:'wx'});
fs.writeFileSync(path.join(output,'smokstak-frame-review.zip'),Buffer.concat(reviewZip(files)),{flag:'wx'});
console.log(`${events.length} upward intervals; ${selected.size} questionable / ${report.frames.length-selected.size} retained. Export: ${path.resolve(output)}`);
for(const group of report.filters)console.log(`${group.filter}: ${report.frames.filter(f=>f.filter===group.filter&&selected.has(f.index)).length} questionable`);
