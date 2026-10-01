// Candidate attribution is an interval, not a causal test of an individual frame.
function jumpCandidates(report, minimumPercent=1) {
 const events=[];
 for(const group of report.filters){
  const accepted=report.frames.filter(f=>f.filter===group.filter&&f.accepted);
  const measured=group.integration_depth.filter(d=>Number.isFinite(d.noise)&&d.noise>0);
  for(let i=1;i<measured.length;i++){
   const before=measured[i-1],after=measured[i],rise=100*(after.noise/before.noise-1);
   if(!(rise>0&&rise>=minimumPercent)||after.frame_count<=before.frame_count)continue;
   const frames=accepted.slice(before.frame_count,after.frame_count);
   if(!frames.length)continue;
   events.push({filter:group.filter,from:before.frame_count,to:after.frame_count,rise_percent:rise,
    before_noise:before.noise,after_noise:after.noise,pair_noise:after.pair_noise??null,
    frame_indices:frames.map(f=>f.index)});
  }
 }
 return events.sort((a,b)=>b.rise_percent-a.rise_percent);
}
function portableFramePath(path){return typeof path==='string'&&path.trim()===path&&!/[\r\n\0"]/u.test(path)&&(/^[A-Za-z]:[\\/]/.test(path)||/^\\\\[^\\]+\\[^\\]+/.test(path)||path.startsWith('/'))}
function reviewBundle(report, selected, minimumPercent=1, source='Manual selection in the report'){
 const indices=new Set(report.frames.map(f=>f.index));
 if([...selected].some(i=>!indices.has(i)))throw Error('Selection contains frames absent from this report.');
 if(report.frames.some(f=>!portableFramePath(f.path)))throw Error('Export requires absolute frame paths without newlines or quotes. Regenerate the analysis with absolute input paths.');
 if(!selected.size)throw Error('Select at least one questionable frame first.');
 const questionable=report.frames.filter(f=>selected.has(f.index)),retained=report.frames.filter(f=>!selected.has(f.index));
 if(!retained.length)throw Error('The retained list would be empty. Keep at least one frame.');
 const events=jumpCandidates(report,minimumPercent);
 const manifest={schema_version:1,source_project:report.project,selection_source:source,minimum_jump_percent:minimumPercent,
  caution:'Frames were added across upward steps in a balanced split-noise estimate, or selected manually. Association does not establish causation. Rerunning changes split pairing and may change the reference and common footprint.',
  retained_count:retained.length,questionable_count:questionable.length,
  frames:questionable.map(f=>({index:f.index,path:f.path,filter:f.filter,capture_time:f.capture_time,exposure_seconds:f.exposure_seconds,
   accepted:f.accepted,hfd:f.hfd,eccentricity:f.eccentricity,background_median:f.background_median,
   registration_residual_sensor_px:f.registration_residual_sensor_px,flags:f.flags,
   jumps:events.filter(e=>e.frame_indices.includes(f.index))}))};
 const list=(rows,note)=>'# Smokstak review export: '+note+'\n# Provisional selection; originals are unchanged.\n'+rows.map(f=>f.path).join('\n')+'\n';
 const csvCell=v=>'"'+String(v??'').replaceAll('"','""')+'"';
 const csv=[['Input index (1-based)','Filter','Capture UTC','Path','HFD px','Eccentricity','Raw sky','Registration residual px','Jump evidence'],
  ...manifest.frames.map(f=>[f.index+1,f.filter,f.capture_time,f.path,f.hfd,f.eccentricity,f.background_median,f.registration_residual_sensor_px,
   f.jumps.map(e=>`N=${e.from} to ${e.to}: +${e.rise_percent.toFixed(3)}%`).join('; ')||'Manual selection'])].map(row=>row.map(csvCell).join(',')).join('\r\n')+'\r\n';
 return {'retained-frames.txt':list(retained,`${retained.length} retained inputs across all filters in this report; only selected frames removed`),
  'questionable-frames.txt':list(questionable,`${questionable.length} questionable inputs for inspection; not proven bad`),
  'questionable-frames.json':JSON.stringify(manifest,null,2)+'\n','questionable-frames.csv':csv,
  'copy-questionable.ps1':copyQuestionableScript(),
  'README.txt':`SMOKSTAK FRAME REVIEW\n\n${source}\nScreening threshold: ${minimumPercent}% rise between balanced measurements (not a significance threshold).\n${retained.length} retained / ${questionable.length} questionable frames. Lists cover every filter in this report and preserve all unselected inputs, including previously rejected frames.\n\nInspect questionable-frames.csv for the full paths and evidence. Open the original FITS/XISF files in your astro software, or copy them with the helper below. No image pixels are embedded in this bundle.\n\nWindows PowerShell, from the extracted bundle directory:\n  .\\copy-questionable.ps1 -Destination 'G:\\astro-review' -WhatIf\n  .\\copy-questionable.ps1 -Destination 'G:\\astro-review'\nUse a new destination directory. The helper copies original bytes with unique index prefixes and writes copied-frames.txt. It never deletes or moves inputs. Source drives must be mounted.\n\nRerun with your usual options, selecting filters when stacking a multi-filter list:\n  smokstak analyze retained-frames.txt --json reviewed-analysis.json --html reviewed-analysis.html\n  smokstak stack retained-frames.txt --filter H\n\nCompare the result to the original, at matched scale and stretch. Removing frames changes integration time, split pairing, and possibly the reference/common footprint. A lower split-noise estimate alone does not establish better final-image quality. Cache reuse depends on compatible reference and settings. The exported lists are the saved selection; changing checkboxes does not recompute the displayed curves.\n`};
}
function copyQuestionableScript(){return String.raw`[CmdletBinding(SupportsShouldProcess=$true)]
param([string]$Destination = (Join-Path $PSScriptRoot 'questionable-originals'),
      [string]$Manifest = (Join-Path $PSScriptRoot 'questionable-frames.json'))
$ErrorActionPreference = 'Stop'
$review = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
if ($review.schema_version -ne 1 -or @($review.frames).Count -eq 0) { throw 'Invalid or empty review manifest.' }
$destinationPath = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $destinationPath) { throw 'Choose a new destination directory; existing files will not be overwritten.' }
$jobs = @()
foreach ($frame in $review.frames) {
    $sourcePath = [string]$frame.path
    if ($sourcePath -notmatch '^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\[^\\]+\\)') { throw "Expected an absolute Windows source path: $sourcePath" }
    $sourceFile = Get-Item -LiteralPath $sourcePath
    if ($sourceFile.PSIsContainer) { throw "Expected a file: $sourcePath" }
    $name = ('{0:D6}_' -f ([int]$frame.index + 1)) + $sourceFile.Name
    $jobs += [PSCustomObject]@{ Source = $sourceFile.FullName; Target = (Join-Path $destinationPath $name); Bytes = $sourceFile.Length }
}
if (@($jobs.Target | Select-Object -Unique).Count -ne $jobs.Count) { throw 'Duplicate copy destinations in manifest.' }
$totalBytes = ($jobs | Measure-Object -Property Bytes -Sum).Sum
if ($PSCmdlet.ShouldProcess($destinationPath, "Copy $($jobs.Count) original frames ($totalBytes bytes)")) {
    [IO.Directory]::CreateDirectory($destinationPath) | Out-Null
    foreach ($job in $jobs) { [IO.File]::Copy($job.Source, $job.Target, $false) }
    [IO.File]::WriteAllLines((Join-Path $destinationPath 'copied-frames.txt'), [string[]]$jobs.Target, [Text.UTF8Encoding]::new($false))
    Write-Output "Copied $($jobs.Count) original frames to $destinationPath"
}
`}
// Small, uncompressed ZIP writer: only text manifests/scripts, never image data.
function reviewZip(files){
 const encoder=new TextEncoder(),parts=[],central=[];let offset=0;
 const crc32=bytes=>{let crc=0xffffffff;for(const byte of bytes){crc^=byte;for(let j=0;j<8;j++)crc=(crc>>>1)^((crc&1)?0xedb88320:0)}return(crc^0xffffffff)>>>0};
 const header=(size,fields)=>{const bytes=new Uint8Array(size),view=new DataView(bytes.buffer);for(const [at,value,width] of fields)width===2?view.setUint16(at,value,true):view.setUint32(at,value,true);return bytes};
 for(const [name,value] of Object.entries(files)){
  const filename=encoder.encode(name),body=encoder.encode(value),crc=crc32(body);
  const local=header(30,[[0,0x04034b50,4],[4,20,2],[6,0x800,2],[12,33,2],[14,crc,4],[18,body.length,4],[22,body.length,4],[26,filename.length,2]]);
  const entry=header(46,[[0,0x02014b50,4],[4,20,2],[6,20,2],[8,0x800,2],[14,33,2],[16,crc,4],[20,body.length,4],[24,body.length,4],[28,filename.length,2],[42,offset,4]]);
  parts.push(local,filename,body);central.push(entry,filename);offset+=local.length+filename.length+body.length;
 }
 const size=central.reduce((n,p)=>n+p.length,0),count=Object.keys(files).length;
 return [...parts,...central,header(22,[[0,0x06054b50,4],[8,count,2],[10,count,2],[12,size,4],[16,offset,4]])];
}
const excludedFrames=new Set();
function reviewThreshold(){return optionalNumber('jump-threshold',0,1000)??1}
function reviewRows(g){
 const candidates=jumpCandidates(data,reviewThreshold()).filter(e=>e.filter===g.filter),ids=new Set(candidates.flatMap(e=>e.frame_indices));
 let rows=data.frames.filter(f=>f.filter===g.filter);
 if($('review-scope').value==='jumps')rows=rows.filter(f=>ids.has(f.index));
 if($('review-scope').value==='session'){const nights=nightsFor(g),night=nights.find(n=>n.id===state(g).night)||nights.at(-1);rows=night?.rows||[]}
 return {rows,candidates};
}
function renderReview(g){
 const {rows,candidates}=reviewRows(g),selected=data.frames.filter(f=>excludedFrames.has(f.index));
 $('review-summary').textContent=`${selected.length} marked across all filters · ${data.frames.length-selected.length} retained. ${fmt(selected.reduce((n,f)=>n+(f.exposure_seconds||0),0)/3600,2)} known hours marked. Curves still show the original data; rerun to measure the effect.`;
 $('review-selection').textContent=data.filters.map(f=>`${f.filter||'Unknown'}: ${selected.filter(row=>row.filter===f.filter).length} marked`).join(' · ');
 $('review-visible').textContent=`${rows.length} frames shown · ${candidates.length} upward intervals ≥ ${fmt(reviewThreshold(),2)}% in this filter. The threshold is a screening choice, not statistical significance.`;
 const rendered=rows.map(f=>{
  const label=text('label',''),check=document.createElement('input');check.type='checkbox';check.checked=excludedFrames.has(f.index);check.setAttribute('aria-label',`Mark ${f.file} questionable`);check.onchange=()=>{check.checked?excludedFrames.add(f.index):excludedFrames.delete(f.index);renderReview(g)};label.append(check,text('span',String(f.index+1)));
  const name=text('div',f.file),path=text('small',f.path);path.className='frame-path';name.append(path,text('small',f.capture_time||'Capture time unavailable'));
  const events=candidates.filter(e=>e.frame_indices.includes(f.index));
  return [label,name,events.map(e=>`N=${e.from}→${e.to}: +${fmt(e.rise_percent,2)}% (${e.frame_indices.length} frames)`).join('; ')||'Manual / session review',fmt(f.hfd,2),fmt(f.eccentricity,3),fmt(f.background_median,5),f.rejection||f.flags.map(x=>x.text||x.kind).join('; ')||'No advisory flag'];
 });
 $('review-frames').replaceChildren(table(['Mark / input #','Original frame / path','Interval evidence','HFD px','Eccentricity','Raw sky','Existing warnings'],rendered));
 $('review-export').disabled=!selected.length||selected.length===data.frames.length;
}
function initReview(){
 for(const id of ['jump-threshold','review-scope'])$(id).addEventListener('change',()=>renderReview(data.filters[Number($('filter').value)]));
 $('review-mark-visible').onclick=()=>{reviewRows(data.filters[Number($('filter').value)]).rows.forEach(f=>excludedFrames.add(f.index));renderReview(data.filters[Number($('filter').value)])};
 $('review-clear').onclick=()=>{excludedFrames.clear();renderReview(data.filters[Number($('filter').value)])};
 $('review-export').onclick=()=>{try{const files=reviewBundle(data,excludedFrames,reviewThreshold()),blob=new Blob(reviewZip(files),{type:'application/zip'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='smokstak-frame-review.zip';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);$('review-error').textContent='Saved bundle requested. Extract it to use the lists and copy helper.'}catch(error){$('review-error').textContent=error.message}};
 $('night-review-frames').onclick=()=>{$('review-scope').value='session';renderReview(data.filters[Number($('filter').value)]);reveal('review-details')};
}
if(typeof module!=='undefined')module.exports={jumpCandidates,reviewBundle,reviewZip};
