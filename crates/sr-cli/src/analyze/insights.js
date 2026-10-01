// Derived report insights. No source files, network requests or hidden rejection.
const viewState=new Map();
const med=v=>{const a=v.filter(Number.isFinite).sort((a,b)=>a-b), n=a.length;return n?(a[Math.floor((n-1)/2)]+a[Math.floor(n/2)])/2:null};
const quantile=(values,q)=>{const a=values.filter(Number.isFinite).sort((a,b)=>a-b);if(!a.length)return null;const i=(a.length-1)*q,k=Math.floor(i);return a[k]+(a[Math.ceil(i)]-a[k])*(i-k)};
function state(g){if(!viewState.has(g.filter))viewState.set(g.filter,{signal:null,background:null,preview:null,baseline:2,night:null,target:null,mode:'review',reduction:null});return viewState.get(g.filter)}
function inputNumber(id,lo,hi,fallback){const value=$(id).value;const n=value===''?NaN:Number(value);return Number.isFinite(n)?Math.max(lo,Math.min(hi,n)):fallback}
function optionalNumber(id,lo,hi){const raw=String($(id).value??'').trim(),value=Number(raw);return raw!==''&&Number.isFinite(value)&&value>=lo&&value<=hi?value:null}
function sessionBudget(g,nights,latest){
 const complete=nights.filter(n=>n.date!=='Unknown date'&&n.rows.some(f=>f.accepted)&&n.rows.filter(f=>f.accepted).every(f=>f.exposure_seconds>0));
 const typicalHours=med(complete.map(n=>n.hours));
 const specified=optionalNumber('extra-hours',0,10000),hours=specified??typicalHours;
 const seconds=latest?.unknown_exposures===0&&latest.integration_seconds>0?latest.integration_seconds/latest.frame_count:null;
 const extra=hours!=null&&seconds?Math.ceil(hours*3600/seconds):null;
 const observed=forecastScenarios(g,nights,latest);
 const ratios=observed.length?observed.map(([,r])=>r):[1];
 const gains=extra!=null&&latest?.noise>0?ratios.map(r=>100*(1-noiseRatio(latest.frame_count,extra,r))):[];
 return {hours,seconds,extra,gains,observed,specified:specified!=null};
}
function recentProgress(nights){
 const batches=nights.filter(n=>n.after&&n.after.frame_count>n.start).slice(-3),last=batches.at(-1),before=batches[0]?.before;
 return {batches,last,change:before?.noise>0&&last?.after?100*(1-last.after.noise/before.noise):null};
}
function saveSelections(id){const g=data.filters[Number($('filter').value)];if(!g||id==='filter')return;const s=state(g);for(const [key,name] of [['signal','signal-patch'],['background','background-patch'],['preview','preview-depth'],['baseline','baseline']])if(id===name)s[key]=$(id).value===''?null:Number($(id).value);if(id==='snr-goal')s.target=optionalNumber(id,.1,10000);if(id==='noise-goal'){const n=optionalNumber(id,1,90);s.reduction=n==null?null:n/100}if(id==='collection-goal')s.mode=$('collection-goal').value}
function options(id,items,value){const el=$(id);el.replaceChildren();for(const [v,label] of items){const o=text('option',label);o.value=String(v);el.append(o)}el.value=value==null?'':String(value)}
function table(headers,rows){const t=document.createElement('table'),h=document.createElement('tr');headers.forEach(x=>h.append(text('th',x)));t.append(h);for(const row of rows){const tr=document.createElement('tr');for(const value of row){const td=document.createElement('td');td.append(value&&typeof value==='object'?value:text('span',String(value)));tr.append(td)}t.append(tr)}return t}
function nightKey(frame){const t=captureTime(frame.capture_time);return Number.isFinite(t)?new Date(t-inputNumber('night-boundary',0,23,12)*3600000).toISOString().slice(0,10):'Unknown date'}
function nightsFor(g){
 const rows=data.frames.filter(f=>f.filter===g.filter), groups=[];let count=0,last=null;const accepted=[];
 for(const f of rows){const key=nightKey(f);if(!last||last.date!==key){last={id:key+':'+f.index,date:key,rows:[],start:count,end:count,pairs:[]};groups.push(last)}last.rows.push(f);if(f.accepted){accepted.push(f);count++;last.end=count}}
 const measured=g.integration_depth.filter(d=>d.noise!=null);
 for(const n of groups){n.before=measured.filter(d=>d.frame_count<=n.start).at(-1);n.after=measured.filter(d=>d.frame_count<=n.end).at(-1);n.change=n.before?.noise>0&&n.after&&n.after.frame_count>n.before.frame_count?100*(1-n.after.noise/n.before.noise):null;n.hours=n.rows.filter(f=>f.accepted).reduce((sum,f)=>sum+(f.exposure_seconds||0),0)/3600;n.hfd=med(n.rows.filter(f=>f.accepted).map(f=>f.hfd));
  n.pairs=g.integration_depth.filter(d=>d.frame_count>n.start&&d.frame_count<=n.end&&d.pair_noise>0&&d.frame_count>=2&&nightKey(accepted[d.frame_count-2])===n.date&&nightKey(accepted[d.frame_count-1])===n.date).map(d=>d.pair_noise);
 }
 return groups;
}
function selectNightForFrame(g,index){const n=nightsFor(g).find(n=>n.rows.some(f=>f.index===index));if(n){state(g).night=n.id;redraw();reveal('night-details')}}
function nightMarkers(g,axis){return nightsFor(g).slice(0,-1).map(n=>{const d=g.integration_depth.find(d=>d.frame_count===n.end);return{x:axis==='hours'?(d?.integration_seconds||0)/3600:n.end}})}
function noiseChange(change){const el=text('span',change==null?'Unavailable':`${fmt(Math.abs(change),1)}% ${change>=0?'lower':'higher'}`);el.className=change==null?'neutral':change>=0?'good':'bad';return el}
function renderNights(g,nights){
 const s=state(g),selected=nights.find(n=>n.id===s.night)||nights.at(-1);
 const ordered=nights.slice();if($('night-sort').value==='latest')ordered.reverse();if($('night-sort').value==='increase')ordered.sort((a,b)=>(a.change??Infinity)-(b.change??Infinity));
 const rows=ordered.map(n=>{const b=text('button',n.date);b.className=n.id===selected?.id?'selected':'';b.setAttribute('aria-pressed',String(n.id===selected?.id));b.onclick=()=>{s.night=n.id;redraw()};return [b,`${n.rows.filter(f=>f.accepted).length} / ${n.rows.length}`,fmt(n.hours,2),noiseChange(n.change),n.before&&n.after?`${n.before.frame_count} → ${n.after.frame_count}`:'No baseline',fmt(n.hfd)+' px',fmt(med(n.rows.filter(f=>f.accepted).map(f=>f.background_median)),5),n.rows.filter(f=>f.flags.length||!f.accepted).length]});
 $('nights').replaceChildren(table(['Night start (UTC boundary)','Accepted / total','Known hours','Measured noise change','Measured N before → after','Median HFD','Median raw sky','Flagged / rejected'],rows));
 const n=nights.find(n=>n.id===s.night)||nights.at(-1);if(!n){$('night-detail').textContent='No frames.';return}
 const previous=nights[nights.indexOf(n)-1],hfdChange=previous?.hfd>0&&n.hfd!=null?100*(n.hfd/previous.hfd-1):null;
 const gain=med(n.rows.filter(f=>f.accepted).map(f=>f.photometry?.map.gain[0])),ecc=med(n.rows.filter(f=>f.accepted).map(f=>f.eccentricity));
 $('night-detail').textContent=`${n.date}: ${fmt(n.hours,2)} known hours; median HFD ${fmt(n.hfd)} px, eccentricity ${fmt(ecc)}, gain ${fmt(gain)}. ${hfdChange==null?'':`HFD ${hfdChange>=0?'increased':'decreased'} ${fmt(Math.abs(hfdChange),1)}% versus the previous batch.`} ${n.change==null?'No bracketing balanced measurements available.':`Cumulative noise ${n.change>=0?'decreased':'increased'} ${fmt(Math.abs(n.change),1)}% across N=${n.before.frame_count}–${n.after.frame_count}.`} ${n.rows.some(f=>f.accepted&&f.exposure_seconds==null)?'Exposure total is incomplete.':''}`;
}
function noiseRatio(n,extra,relativeFrameNoise){return Math.sqrt(n*(n+extra*relativeFrameNoise**2))/(n+extra)}
function requiredFrames(n,reduction,r=1){if(!(n>0&&reduction>0&&reduction<1&&r>0))return null;const t=(1-reduction)**2,b=2*t-r*r,x=(-b+Math.sqrt(b*b+4*t*(1-t)))/(2*t);const extra=Math.ceil(n*x);return Number.isFinite(extra)&&extra<2e9?extra:null}
function forecastScenarios(g,nights,latest){const samples=nights.filter(n=>n.date!=='Unknown date'&&n.pairs.length>=3).map(n=>med(n.pairs));const scale=latest?.noise*Math.sqrt(latest?.frame_count);return samples.length>=3&&scale>0?[['Quieter observed nights',quantile(samples,.25)/scale],['Typical observed nights',quantile(samples,.5)/scale],['Noisier observed nights',quantile(samples,.75)/scale]]:[]}
// One decision summary, using the same goals and scenario equations as the details.
function collectionAdvice(g,nights,settings){
 const latest=g.integration_depth.filter(d=>d.noise!=null).at(-1);
 const missing={status:'missing',answer:'Not enough evidence yet',goal:'No usable noise estimate',detail:'More-frame estimates require a positive balanced measurement.',next:'Review the measurement warnings before planning more exposures.',reason:'No count of additional frames can be justified from the available measurements.'};
 if(!latest||!(latest.noise>0))return missing;
 const seconds=latest.unknown_exposures===0&&latest.integration_seconds>0?latest.integration_seconds/latest.frame_count:null;
 if(settings.mode==='review'||!settings.mode){
  const progress=recentProgress(nights),budget=sessionBudget(g,nights,latest),fit=g.recent_fit,usable=fit&&fit.exponent<-.05&&fit.r_squared>=.8;
  const trend=progress.change==null?'Too few bracketing measurements to summarize recent progress.':`Measured noise ${progress.change>=0?'fell':'rose'} ${fmt(Math.abs(progress.change),1)}% across the last ${progress.batches.length} acquisition batches (N=${progress.batches[0].before.frame_count}–${progress.last.after.frame_count}).`;
  const low=Math.round(Math.min(...budget.gains)),high=Math.round(Math.max(...budget.gains));
  const gainText=low===high?(low===0?'a noise change smaller than 1%':'about '+Math.abs(low)+'% '+(low>0?'lower':'higher')+' noise'):low>=0?'roughly '+low+'–'+high+'% lower noise':high<=0?'roughly '+Math.abs(high)+'–'+Math.abs(low)+'% higher noise':'between '+Math.abs(low)+'% higher and '+high+'% lower noise';
  const expectation=budget.gains.length?`${budget.specified?'Your proposed session':'A typical session'} of ${fmt(budget.hours,1)} h (${budget.extra} frames) gives ${gainText} in a conditional ${budget.observed.length?'observed-night':'equal-noise'} scenario.`:'Session planning needs complete exposure durations and dated batches, or a supplied session length.';
  const worst=nights.filter(n=>n.change<0).sort((a,b)=>a.change-b.change)[0];
  return {status:'review',answer:progress.change<0?'Investigate the recent noise increase':progress.change>0?'Noise is falling; remaining time is uncertain':usable?'Data is improving — no stopping target set':'Review the trend before a large collection run',goal:`${g.filter||'Unknown filter'} · No stopping target set`,detail:trend+' '+expectation,next:usable?'Compare the expected gain with the time you want to spend, or set a goal for a specific faint feature.':`Remaining time is uncertain: the recent fit is too weak.${worst?` Review ${worst.date}, where cumulative noise rose ${fmt(-worst.change,1)}%.`:''} Reassess after the next session.`,reason:'Recent changes are observed differences between cumulative measurements, not proof that a particular night caused them. Session scenarios assume independent future noise and are not forecasts or confidence intervals. No automatic noise or SNR target is assigned.',reviewNight:!usable?worst?.id:null};
 }
 const amount=frames=>seconds?`About ${fmt(frames*seconds/3600,1)} more hours`:`About ${fmt(frames,0)} more frames`;
 const count=frames=>`${fmt(frames,0)} additional frames${seconds?` at ${fmt(seconds/60,1)} min per frame`:''}`;
 const base=`Based on the last balanced measurement, N=${latest.frame_count}. Hours count usable exposure time, excluding overheads and rejected frames. Shared systematic errors are not measured by the split-noise method.`;
 if(settings.mode==='snr'){
  const sample=regionSeries(g,settings.region).at(-1),target=settings.region.target;
  if(!(target>0&&Number.isFinite(target)))return {...missing,answer:'Set your contrast-SNR target',goal:'No stopping target set',detail:'Choose a meaningful target for the selected feature; the report does not assume SNR 5 or any other universal threshold.',next:'Inspect the matched patches and enter your own target.',selectRegions:!sample};
  if(!sample)return {...missing,answer:'Choose the feature you want to improve',goal:`Selected-region contrast-SNR goal: ${fmt(target,1)}`,detail:'Select different signal and background patches to get a collect-or-stop answer for that feature.',next:'A project-wide definition of “good enough” has not been selected.',selectRegions:true};
  if(!(sample.snr>0))return {...missing,goal:`Current contrast SNR ${fmt(sample.snr,2)}; target ${fmt(target,1)}`,detail:'The selected patches do not show positive contrast.',next:'Review your signal and background selections.',selectRegions:true};
  if(sample.snr>=target)return {status:'met',answer:'0 more — your selected goal is met',goal:`${g.filter||'Unknown filter'}: contrast SNR ${fmt(sample.snr,2)} exceeds or equals your target of ${fmt(target,1)}`,detail:'No additional collection is needed for this measured patch-contrast goal.',next:'You can stop collecting for this goal and assess the image, or choose a fainter feature or higher target.',reason:`This applies only to the chosen patches at N=${sample.n}. It does not certify final-image quality or remove shared systematic errors.`};
  const frames=Math.ceil(sample.n*((target/sample.snr)**2-1));
  if(!Number.isFinite(frames)||frames>=2e9)return {...missing,goal:'Selected contrast-SNR target is beyond a useful extrapolation range',next:'Review the regions and target before committing observing time.'};
  return {status:'scenario',answer:amount(frames),goal:`${g.filter||'Unknown filter'}: contrast SNR ${fmt(sample.snr,2)} → ${fmt(target,1)}`,detail:count(frames)+'. Equal-quality, independent-noise scenario.',next:'Use this as an observing budget, then reassess the selected feature.',reason:`Assumes stable signal, equal future frame noise and no shared error floor. The contrast proxy is not aperture SNR or detection significance. ${seconds?'':'Exposure durations are incomplete, so hours are unavailable. '}${base}`};
 }
 if(!(settings.reduction>0&&settings.reduction<1))return {...missing,answer:'Choose an improvement to explore',goal:'No stopping target set',detail:'Use the 5%, 10% and 20% scenarios below for comparison, or enter a custom reduction.',next:'A noise-reduction percentage is an optional improvement budget, not a finished-image criterion.'};
 const goal=settings.reduction,observed=forecastScenarios(g,nights,latest),typical=observed[1],frames=requiredFrames(latest.frame_count,goal,typical?typical[1]:1);
 if(frames==null)return missing;
 const range=observed.map(([,r])=>requiredFrames(latest.frame_count,goal,r));
 const rangeText=range.length&&range.every(v=>v!=null)?` Observed-night scenarios span ${seconds?fmt(Math.min(...range)*seconds/3600,1)+'–'+fmt(Math.max(...range)*seconds/3600,1)+' h':fmt(Math.min(...range),0)+'–'+fmt(Math.max(...range),0)+' frames'}.`:'';
 const fit=g.recent_fit,reliable=fit&&fit.exponent<-.05&&fit.r_squared>=.8;
 return {status:'scenario',answer:amount(frames),goal:`${g.filter||'Unknown filter'}: for another ${fmt(goal*100,0)}% reduction in difference noise`,detail:count(frames)+`. ${typical?'Typical observed-night':'Equal-noise reference'} scenario.`+rangeText,next:reliable?'Treat this as a collection budget and reassess after the next batch.':'Treat this as a planning scenario; the recent trend is too weak for a firm collect/stop recommendation.',reason:`The selected percentage is an improvement goal, not an automatic threshold for a finished image. ${typical?'The range comes from observed night-quality quartiles, not a confidence interval.':'There are too few eligible nights to estimate a night-quality range.'} Assumes independent future noise and continued photometric consistency. ${seconds?'':'Exposure durations are incomplete, so hours are unavailable. '}${base}`};
}
function renderCollection(g,nights){
 const selected=state(g),mode=selected.mode; $('collection-goal').value=mode;$('noise-goal').value=selected.reduction==null?'':Number((selected.reduction*100).toFixed(6));
 $('collection-noise-control').style.display=mode==='noise'?'':'none';$('collection-snr-control').style.display=mode==='snr'?'':'none';
 const advice=collectionAdvice(g,nights,{mode,reduction:selected.reduction,region:selected});
 $('collection-summary').className=advice.status+(recentProgress(nights).change<0?' attention':'');
 $('collection-badge').textContent=advice.status==='met'?'Selected goal met':advice.status==='review'?'Progress review — no stopping target set':advice.status==='scenario'?'Conditional planning estimate':'Decision unavailable';
 $('collection-answer').textContent=advice.answer;$('collection-goal-context').textContent=advice.goal;$('collection-detail').textContent=advice.detail;$('collection-next').textContent=advice.next;$('collection-reason').textContent=advice.reason;
 $('collection-select-regions').style.display=advice.selectRegions?'':'none';
 $('collection-review-night').style.display=advice.reviewNight?'':'none';$('collection-review-night').onclick=()=>{selected.night=advice.reviewNight;redraw();reveal('night-details')};
 const latest=g.integration_depth.filter(d=>d.noise!=null).at(-1),budget=sessionBudget(g,nights,latest);
 $('review-filters').replaceChildren(table(['Filter / time','Recent measured noise'],data.filters.map(filter=>{
  const p=recentProgress(nightsFor(filter)),cell=text('div',''),button=text('button',filter.filter||'Unknown');button.className=filter===g?'selected':'';button.setAttribute('aria-pressed',String(filter===g));button.onclick=()=>{$('filter').value=String(data.filters.indexOf(filter));redraw()};
  cell.append(button,text('span',fmt(filter.known_integration_seconds/3600,1)+' h'));cell.children[1].className='trend-context';
  const trend=text('div',''),value=noiseChange(p.change);value.className+=' trend-label';trend.append(value);const context=text('span',p.change==null?'More history needed':`${p.batches.length} batches · N=${p.batches[0].before.frame_count}–${p.last.after.frame_count}`);context.className='trend-context';trend.append(context);return [cell,trend];
 })));
 const rising=data.filters.filter(f=>recentProgress(nightsFor(f)).change<0).map(f=>f.filter||'Unknown');
 $('project-attention').textContent=rising.length?`Inspect first: ${rising.join(', ')}. Measured noise rose recently; review those sessions before committing a large collection run.`:'Choose a filter above to compare its measured progress and next-session budget.';
 $('project-attention').className=rising.length?'bad':'';
 $('improvement-options').replaceChildren();
 if(latest?.noise>0){const r=budget.observed[1]?.[1]??1;
  for(const fraction of [.05,.10,.20]){const frames=requiredFrames(latest.frame_count,fraction,r),button=text('button',`${fraction*100}% less noise · ${frames==null?'Unavailable':budget.seconds?fmt(frames*budget.seconds/3600,1)+' h':fmt(frames,0)+' frames'}`);button.onclick=()=>{selected.mode='noise';selected.reduction=fraction;redraw()};$('improvement-options').append(button)}
 }
}
function renderForecast(g,nights){
 const latest=g.integration_depth.filter(d=>d.noise!=null).at(-1),fit=g.recent_fit,reliable=fit&&fit.exponent<-.05&&fit.r_squared>=.8;
 $('forecast-status').textContent=reliable?`Recent empirical fit is usable for conditional extrapolation (R² ${fmt(fit.r_squared)}), not a prediction of future weather or a systematic floor.`:'The recent fit does not support a reliable frame-count forecast. The scenarios below are conditional calculations, not fitted forecasts.';
 if(!latest||!(latest.noise>0)){$('scenarios').textContent='No positive balanced noise measurement.';$('goal-result').textContent='';return}
 const seconds=latest.unknown_exposures===0&&latest.integration_seconds>0?latest.integration_seconds/latest.frame_count:null;
 const budget=sessionBudget(g,nights,latest),{hours,extra,observed}=budget,scenarios=[['Equal-noise reference',1],...observed],selected=state(g).reduction,minGain=optionalNumber('minimum-gain',0,90);
 $('extra-hours').placeholder='Typical session';
 $('scenarios').replaceChildren(table(['Assumed future quality',`Noise reduction from ${hours==null?'unspecified':fmt(hours,1)} added hours`,'5% less noise','10% less noise','20% less noise'],scenarios.map(([name,r])=>[name,extra==null?'Unknown duration':fmt(100*(1-noiseRatio(latest.frame_count,extra,r)),1)+'%',...[.05,.1,.2].map(goal=>{const needed=requiredFrames(latest.frame_count,goal,r);return needed==null?'Unavailable':fmt(needed,0)+' frames'+(seconds?' / '+fmt(needed*seconds/3600,1)+' h':'')})])));
 const typical=observed[1],improvement=extra==null?null:100*(1-noiseRatio(latest.frame_count,extra,typical?typical[1]:1));
 $('goal-result').textContent=`${selected!=null?'Selected improvement scenario: '+fmt(selected*100,0)+'% lower noise.':'No stopping target set. The percentages are optional scenarios.'} ${extra==null?'Hours cannot be converted to frames because session length or exposure information is incomplete.':`${budget.specified?'Proposed':'Typical recorded'} session: ${fmt(hours,1)} h ≈ ${extra} frames; ${typical?'typical observed-night':'equal-noise reference'} improvement ${fmt(improvement,1)}%. ${minGain==null?'':improvement<minGain?'Below your chosen minimum worthwhile improvement.':'Meets your chosen minimum under this assumption.'}`} ${observed.length?'The range describes observed night variation, not statistical confidence.':'Insufficient dated pairs for observed-night scenarios.'}`;

}
function regionSeries(g,s){return(g.snapshots||[]).flatMap(snap=>{const a=snap.patches.find(p=>p.id===s.signal),b=snap.patches.find(p=>p.id===s.background);if(!a||!b||a.id===b.id||a.noise==null||b.noise==null)return[];const noise=Math.hypot(a.noise,b.noise),signal=a.median-b.median;return noise>0?[{n:snap.frame_count,hours:snap.integration_seconds/3600,signal,noise,snr:signal/noise,a,b}]:[]})}
function renderPatch(canvasId,patch,low,high){const c=$(canvasId);c.width=256;c.height=256;const ctx=c.getContext('2d');ctx.fillStyle='#101923';ctx.fillRect(0,0,256,256);if(!patch){ctx.fillStyle='#a6b9c9';ctx.font='13px system-ui';ctx.textAlign='center';ctx.fillText(canvasId==='patch-background'?'Choose a background patch':'Choose a signal patch',128,128);return}for(let i=0;i<256;i++){const value=patch.low+parseInt(patch.pixels.slice(i*4,i*4+4),16)/65535*(patch.high-patch.low);const gray=Math.round(255*Math.asinh(5*Math.max(0,Math.min(1,(value-low)/(high-low))))/Math.asinh(5));ctx.fillStyle=`rgb(${gray},${gray},${gray})`;ctx.fillRect(i%16*16,Math.floor(i/16)*16,16,16)}}
function renderMap(g,s,snap){const c=$('patch-map'),r=c.getBoundingClientRect();c.width=Math.max(200,r.width);c.height=300;c.onclick=null;const ctx=c.getContext('2d');ctx.fillStyle='#101923';ctx.fillRect(0,0,c.width,c.height);const positions=g.patch_positions||[];if(!snap||!positions.length)return;const xmax=Math.max(...positions.map(p=>p[0]))+32,ymax=Math.max(...positions.map(p=>p[1]))+32;const low=quantile(snap.patches.map(p=>p.median),.1),high=quantile(snap.patches.map(p=>p.median),.95),pts=[],scale=Math.min((c.width-24)/xmax,276/ymax),left=(c.width-xmax*scale)/2,top=(300-ymax*scale)/2;
 for(const p of snap.patches){const pos=positions[p.id],x=left+pos[0]*scale,y=top+pos[1]*scale,t=(p.median-low)/Math.max(high-low,1e-12);ctx.fillStyle=p.id===s.signal?'#59d5c7':p.id===s.background?'#e8b668':`rgb(${80+Math.round(160*Math.max(0,Math.min(1,t)))},${80+Math.round(160*Math.max(0,Math.min(1,t)))},${80+Math.round(160*Math.max(0,Math.min(1,t)))})`;const size=p.id===s.signal||p.id===s.background?8:3;ctx.fillRect(x-size/2,y-size/2,size,size);pts.push({x,y,id:p.id})}
 for(const p of pts.filter(p=>p.id===s.signal||p.id===s.background)){ctx.fillStyle=p.id===s.signal?'#59d5c7':'#e8b668';ctx.font='bold 14px system-ui';ctx.fillText(p.id===s.signal?'S':'B',p.x+7,p.y-6)}
 c.onclick=e=>{const rect=c.getBoundingClientRect(),x=(e.clientX-rect.left)*c.width/rect.width,y=(e.clientY-rect.top)*c.height/rect.height;let best=null,dist=225;for(const p of pts){const d=(p.x-x)**2+(p.y-y)**2;if(d<dist){best=p;dist=d}}if(best){s[$('pick-region').value==='background'?'background':'signal']=best.id;redraw()}};
}
function renderRegions(g){const s=state(g),snaps=g.snapshots||[],latest=snaps.filter(x=>x.frame_count%2===0).at(-1)||snaps.at(-1);const positions=g.patch_positions||[];const opts=[['','Choose a patch'],...(latest?.patches||[]).map(p=>[p.id,`#${p.id} · (${positions[p.id]?.[0]}, ${positions[p.id]?.[1]})`])];options('signal-patch',opts,s.signal);options('background-patch',opts,s.background);$('snr-goal').value=s.target??'';
 if(s.preview==null||!snaps.some(x=>x.frame_count===s.preview))s.preview=snaps.filter(x=>x.frame_count<=(latest?.frame_count||1)/4).at(-1)?.frame_count||snaps[0]?.frame_count;
 options('preview-depth',snaps.map(x=>[x.frame_count,`N=${x.frame_count} · ${fmt(x.integration_seconds/3600,2)} h`]),s.preview);renderMap(g,s,latest);
 const before=snaps.find(x=>x.frame_count===s.preview),a=latest?.patches.find(p=>p.id===s.signal),b=before?.patches.find(p=>p.id===s.signal),bg=latest?.patches.find(p=>p.id===s.background),stretch=Number($('preview-stretch').value)||1;
 // One physical black/white range for both depths and the background patch.
 const spread=a?(a.high-a.low)/(2*stretch):1,low=a?a.median-spread:0,high=a?a.median+spread:1;
 renderPatch('patch-before',b,low,high);renderPatch('patch-after',a,low,high);renderPatch('patch-background',bg,low,high);
 $('patch-before-label').textContent=`Earlier signal · N=${before?.frame_count??'—'} · ${fmt((before?.integration_seconds||0)/3600,2)} h`;
 $('patch-after-label').textContent=`Latest balanced signal · N=${latest?.frame_count??'—'} · shared asinh stretch; display clipping only`;
 $('patch-background-label').textContent='Background · same depth, stretch and scale';
 const series=regionSeries(g,s),last=series.at(-1);chart('region-snr',[{color:'#59d5c7',line:true,values:series.map(p=>({x:p.hours,y:p.snr,label:`N=${p.n} · ${fmt(p.hours,2)} h`}))},{color:'#e8b668',line:true,values:series.map(p=>({x:p.hours,y:s.target,label:'Your contrast-SNR target'}))}],{xlabel:'Known integration (h)',ylabel:'Contrast SNR'});
 if(!last){$('region-result').textContent=s.signal===s.background&&s.signal!=null?'Signal and background must be different patches.':'Select signal and background patches to measure their contrast. No target is selected automatically.';$('region-goal').textContent='A positive, measured contrast is needed for an SNR goal.';renderPriority();return}
 $('region-result').textContent=`N=${last.n}: signal above background ${fmt(last.signal,7)}; combined split noise ${fmt(last.noise,7)}; per-sample contrast SNR ${fmt(last.snr,2)}. This is not an aperture or final-stack SNR.`;
 if(!(s.target>0)){$('region-goal').textContent='Contrast is measured, but no stopping target is set. Enter a target only after deciding what this feature needs.';renderPriority();return}
 const required=last.snr>0&&last.snr<s.target?Math.ceil(last.n*((s.target/last.snr)**2-1)):0;
 $('region-goal').textContent=last.snr<=0?'No positive contrast in these regions; an exposure forecast is unavailable.':last.snr>=s.target?`Your selected contrast-SNR goal of ${fmt(s.target,1)} is met at this sampling scale.`:required<2e9?`Target ${fmt(s.target,1)}: equal-quality, independent-noise scenario needs about ${required} more frames${g.unknown_exposures===0&&g.accepted_frames>0?' ('+fmt(required*g.known_integration_seconds/g.accepted_frames/3600,1)+' h)':''}. This assumes stable signal and no shared error floor; it is not a fitted forecast.`:'Target is beyond a useful extrapolation range.';
 renderPriority();
}
function renderPriority(){const entries=data.filters.map(g=>{const s=state(g),last=regionSeries(g,s).at(-1);return{g,s,last,completion:last?.snr>0&&s.target>0?last.snr/s.target:null}});const configured=entries.filter(e=>e.completion!=null),pending=configured.filter(e=>e.completion<1).sort((a,b)=>a.completion-b.completion);const note=text('p',configured.length!==entries.length?'Set comparable feature goals for every filter before deciding acquisition priority.':pending.length?`${pending[0].g.filter||'Unknown'} has the largest relative shortfall against your selected goals. This is a goal comparison, not a recommendation based solely on exposure totals.`:'All selected contrast goals are met; this does not certify overall image quality.');$('filter-priority').replaceChildren(text('h2','Filter goals'),note,table(['Filter','Known accepted hours','Selected contrast SNR','Goal','Goal reached'],entries.map(e=>[e.g.filter||'Unknown',fmt(e.g.known_integration_seconds/3600,1),fmt(e.last?.snr,2),e.s.target>0?fmt(e.s.target,1):'Not configured',e.completion==null?'Unavailable':fmt(e.completion*100,0)+'%'])))}
function renderInsights(g){const s=state(g),measured=g.integration_depth.filter(d=>d.noise!=null),last=measured.at(-1);if(!measured.some(d=>d.frame_count===s.baseline))s.baseline=measured[0]?.frame_count;options('baseline',measured.filter(d=>d.fit_checkpoint||d===measured[0]||d===last).map(d=>[d.frame_count,`N=${d.frame_count} · ${fmt(d.integration_seconds/3600,2)} h`]),s.baseline);const nights=nightsFor(g),fit=g.recent_fit;const reliable=fit&&fit.exponent<-.05&&fit.r_squared>=.8;
 $('decision').textContent=`${g.filter||'Unknown filter'} · ${fmt(g.known_integration_seconds/3600,1)} known accepted hours across ${nights.length} date batches. ${reliable?'Recent scaling supports conditional extrapolation.':'More-frame forecast: insufficient evidence from the recent trend.'} ${last?`Latest balanced noise measurement: N=${last.frame_count}.`:''}`;
 const gains=nights.filter(n=>n.change!=null).map(n=>({x:$('depth-axis').value==='hours'?(n.after?.integration_seconds||0)/3600:n.after.frame_count,y:n.change,label:`${n.date} · ${fmt(n.hours,2)} h added · N=${n.before.frame_count}–${n.after.frame_count}`,pick:()=>{state(g).night=n.id;redraw();reveal('night-details')}}));
 chart('night-gain',[{color:'#59d5c7',values:gains.filter(p=>p.y>=0)},{color:'#edb276',values:gains.filter(p=>p.y<0)},{color:'#8194a5',line:true,values:gains.map(p=>({...p,y:0,label:'No change'}))}],{xlabel:$('depth-axis').value==='hours'?'Cumulative known integration (h)':'Cumulative accepted frames',integerX:$('depth-axis').value==='frame_count',ylabel:'Noise improvement (%)'});
 renderNights(g,nights);renderForecast(g,nights);renderRegions(g);renderCollection(g,nights);
}
