(function () {
  'use strict';
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const basename = p => String(p || '').split(/[\\/]/).pop();
  // Canonical Rust paths may use Windows' extended prefix; export ordinary paths
  // for astronomy applications, preserving UNC server/share paths.
  const externalPath = p => String(p).replace(/^\\\\\?\\UNC\\/i, '\\\\').replace(/^\\\\\?\\(?=[a-z]:\\)/i, '');
  const label = status => ({used: 'Used', 'not-used': 'Not used', excluded: 'Excluded'})[status] || 'Unknown';
  function percentile(sorted, fraction) {
    if (!sorted.length) return null;
    const at = (sorted.length - 1) * Math.max(0, Math.min(1, fraction));
    const low = Math.floor(at), high = Math.ceil(at);
    return sorted[low] * (1 - (at - low)) + sorted[high] * (at - low);
  }
  function stretchRange(patches) {
    const samples = [];
    for (const patch of patches) if (patch && Array.isArray(patch.pixels)) {
      for (const value of patch.pixels) if (finite(value)) samples.push(value);
    }
    samples.sort((a, b) => a - b);
    if (!samples.length) return null;
    const low = percentile(samples, .005), high = percentile(samples, .998);
    return {low, high: high > low ? high : low + Math.max(Math.abs(low) * 1e-6, 1e-8)};
  }
  function displayValue(value, range, gain) {
    if (!finite(value) || !range) return null;
    const amount = Math.min(1, Math.max(0, (value - range.low) / (range.high - range.low)));
    const g = Math.max(1, finite(gain) ? gain : 1);
    return Math.round(255 * Math.asinh(g * amount) / Math.asinh(g));
  }
  function captureStamp(frame) {
    const value = frame.capture_time;
    if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$/i.test(value)) return null;
    const stamp = Date.parse(/(Z|[+-]\d{2}:?\d{2})$/i.test(value) ? value : value + 'Z');
    return Number.isFinite(stamp) ? stamp : null;
  }
  function filteredIndices(frames, filter, status, query = '', order = 'input', gain = '*') {
    const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const indices = frames.flatMap((f, i) => (filter === '*' || String(f.filter || '') === filter) && (status === '*' || f.status === status) && (gain === '*' || String(finite(f.iso_or_gain) ? f.iso_or_gain : 'unknown') === gain) && words.every(word => `${f.path} ${f.reason} ${f.capture_time || ''} ${f.photometry_source || ''}`.toLowerCase().includes(word)) ? [i] : []);
    const key = f => order === 'capture' ? captureStamp(f) : metricValue(f, order);
    if (order !== 'input') indices.sort((a, b) => {
      const av = key(frames[a]), bv = key(frames[b]);
      return (av === null) - (bv === null) || (av !== null && bv !== null ? (av - bv) * (order === 'suppressed' ? -1 : 1) : 0) || a - b;
    });
    return indices;
  }
  function comparisonIndex(frames, selected) {
    const source = frames[selected];
    if (!source) return -1;
    const peers = frames.map((f, i) => ({f, i})).filter(({f, i}) => i !== selected && (f.filter || '') === (source.filter || ''));
    const stamp = captureStamp(source);
    const distance = ({f, i}) => stamp !== null ? (captureStamp(f) === null ? Infinity : Math.abs(captureStamp(f) - stamp)) : Math.abs(i - selected);
    const gainMismatch = f => finite(source.iso_or_gain) && f.iso_or_gain !== source.iso_or_gain ? 1 : 0;
    peers.sort((a, b) => (a.f.status === 'used' ? 0 : 1) - (b.f.status === 'used' ? 0 : 1) || gainMismatch(a.f) - gainMismatch(b.f) || distance(a) - distance(b) || Math.abs(a.i - selected) - Math.abs(b.i - selected));
    return peers.length ? peers[0].i : -1;
  }
  function frameList(frames, status) { return frames.filter(f => f.status === status).map(f => externalPath(f.path)).join('\n') + (frames.some(f => f.status === status) ? '\n' : ''); }
  function metricValue(frame, metric) {
    if (metric === 'obstruction') { const fraction = maskFraction(frame); return fraction === null ? null : fraction * 100; }
    if (metric === 'photometric_gain') {
      const gains = frame.photometric_gain;
      return Array.isArray(gains) && gains.length && gains.every(finite) ? percentile([...gains].sort((a, b) => a - b), .5) : null;
    }
    return finite(frame[metric]) ? frame[metric] * (metric === 'suppressed' ? 100 : 1) : null;
  }
  function maskFraction(frame) {
    const mask = frame && frame.obstruction_mask;
    if (!Array.isArray(mask) || !mask.length || !Array.isArray(mask[0]) || !mask[0].length ||
      !mask.every(row => Array.isArray(row) && row.length === mask[0].length && row.every(v => typeof v === 'boolean'))) return null;
    return mask.flat().filter(Boolean).length / (mask.length * mask[0].length);
  }
  function maskedAt(frame, x, y) {
    if (maskFraction(frame) === null) return false;
    const mask = frame.obstruction_mask;
    return mask[Math.round(Math.max(0, Math.min(1, y)) * (mask.length - 1))][Math.round(Math.max(0, Math.min(1, x)) * (mask[0].length - 1))];
  }
  function integration(frames) {
    return frames.reduce((out, f) => { if (finite(f.exposure_seconds) && f.exposure_seconds > 0) out.seconds += f.exposure_seconds; else out.unknown++; return out; }, {seconds: 0, unknown: 0});
  }
  const helpers = {percentile, stretchRange, displayValue, filteredIndices, comparisonIndex, frameList, metricValue, basename, captureStamp, integration, externalPath, maskFraction, maskedAt};
  if (typeof module !== 'undefined' && module.exports) module.exports = helpers;
  if (typeof document === 'undefined') return;
  const $ = id => document.getElementById(id);
  const data = JSON.parse($('review-data').textContent);
  const frames = Array.isArray(data.frames) ? data.frames : [];
  const colors = {used: '#63ddcb', 'not-used': '#edbb72', excluded: '#b9a0ec'};
  const fmt = (x, digits = 3) => finite(x) ? x.toLocaleString(undefined, {maximumFractionDigits: digits}) : '—';
  const el = (name, value, className) => { const n = document.createElement(name); if (value !== undefined) n.textContent = value; if (className) n.className = className; return n; };
  let selected = frames.findIndex(f => f.status !== 'used');
  if (selected < 0 && frames.length) selected = 0;
  let comparison = comparisonIndex(frames, selected), mode = 'native', page = 0, blinking = false, blinkOther = false, timer = null;
  const pageSize = 40, previewCache = new Map(), previewPending = new Map(), previewFailed = new Set();
  let previewEpoch = 0;
  const assetPattern = /^review-assets\/([a-f0-9]{64})\.js$/;
  function cachedImages(frame) {
    if (!frame) return null;
    if (frame.images) return frame.images;
    const match = assetPattern.exec(frame.preview_asset || '');
    return match ? previewCache.get(match[1]) || null : null;
  }
  function finishPreview(id, images) {
    const pending = previewPending.get(id);
    if (!pending) return;
    previewPending.delete(id); clearTimeout(pending.timeout); pending.script.remove();
    if (images && typeof images === 'object') {
      previewCache.delete(id); previewCache.set(id, images);
      while (previewCache.size > 12) previewCache.delete(previewCache.keys().next().value);
    } else previewFailed.add(id);
    pending.resolve(images || null);
  }
  globalThis.smokstakReviewPatch = (id, images) => { if (typeof id === 'string' && /^[a-f0-9]{64}$/.test(id)) finishPreview(id, images); };
  function loadPreview(frame) {
    if (!frame || frame.images) return Promise.resolve(frame && frame.images);
    const match = assetPattern.exec(frame.preview_asset || '');
    if (!match) return Promise.resolve(null);
    const id = match[1];
    if (previewCache.has(id)) { const images = previewCache.get(id); previewCache.delete(id); previewCache.set(id, images); return Promise.resolve(images); }
    if (previewFailed.has(id)) return Promise.resolve(null);
    if (previewPending.has(id)) return previewPending.get(id).promise;
    const script = el('script'); script.src = frame.preview_asset;
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    previewPending.set(id, {promise, resolve, script, timeout: setTimeout(() => finishPreview(id, null), 15000)});
    script.onerror = () => finishPreview(id, null);
    script.onload = () => { if (previewPending.has(id)) finishPreview(id, null); };
    document.head.append(script);
    return promise;
  }
  const filterNames = [...new Set(frames.map(f => String(f.filter || '')))].sort();
  const mixed = filterNames.filter(filter => new Set(frames.filter(f => String(f.filter || '') === filter && finite(f.iso_or_gain)).map(f => f.iso_or_gain)).size > 1);
  $('configuration-note').textContent = mixed.length ? `Mixed ISO/gain settings in ${mixed.map(f => f || 'unknown filter').join(', ')}. Use the ISO/gain selector to inspect each setting. Production estimates a noise model per filter, not per gain setting; this audit does not establish optimal mixed-gain weighting.` : '';
  $('configuration-note').className = mixed.length ? 'readout' : 'hidden';
  const fallbackCount = frames.filter(f => f.status === 'used' && ['level-only', 'exposure'].includes(f.photometry_source)).length;
  $('normalization-note').textContent = fallbackCount ? `${fallbackCount} used frame(s) have level-only or exposure-metadata brightness matching. Retained does not mean gain normalization was measured. Search “level-only” or “exposure” to inspect these frames and check their scaling before accepting the stack.` : '';
  $('normalization-note').className = fallbackCount ? 'readout' : 'hidden';
  for (const filter of filterNames) { const option = el('option', filter || 'Unknown filter'); option.value = filter; $('filter').append(option); }
  for (const gain of [...new Set(frames.map(f => finite(f.iso_or_gain) ? f.iso_or_gain : 'unknown'))].sort((a, b) => (a === 'unknown') - (b === 'unknown') || a - b)) { const option = el('option', gain === 'unknown' ? 'Unknown setting' : String(gain)); option.value = String(gain); $('sensor-gain').append(option); }
  for (const status of ['used', 'not-used', 'excluded']) $('count-' + status).textContent = frames.filter(f => f.status === status).length.toLocaleString();
  for (const note of data.notes || []) $('notes').append(el('li', note));
  function scoped() { return filteredIndices(frames, $('filter').value, $('status').value, $('search').value, $('order').value, $('sensor-gain').value); }
  function patch(frame) { const images = cachedImages(frame); return images ? images[mode] : null; }
  function renderPatch(container, image, range, zoom, frame) {
    container.replaceChildren();
    if (!image || !image.width || !image.height || !Array.isArray(image.pixels)) { container.append(el('span', 'Preview unavailable for this frame.', 'unavailable')); return; }
    const canvas = el('canvas'); canvas.width = image.width; canvas.height = image.height;
    canvas.style.width = image.width * zoom + 'px'; canvas.style.height = image.height * zoom + 'px';
    canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${mode === 'sky' ? 'Registered overview with obstruction mask' : mode === 'thumb' ? 'Overview' : mode === 'aligned' ? 'Registered crop' : 'Native sensor crop'}, ${image.width} by ${image.height} pixels`);
    const ctx = canvas.getContext('2d'), out = ctx.createImageData(image.width, image.height);
    for (let i = 0; i < image.width * image.height; i++) {
      const v = displayValue(image.pixels[i], range, Number($('gain').value));
      const missing = ((i % image.width >> 3) + (Math.floor(i / image.width) >> 3)) % 2 ? 24 : 42;
      out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = v === null ? missing : v;
      out.data[i * 4 + 3] = 255;
      if (mode === 'sky' && v !== null && maskedAt(frame, (i % image.width) / Math.max(1, image.width - 1), Math.floor(i / image.width) / Math.max(1, image.height - 1))) {
        out.data[i * 4] = Math.round(v * .65 + 255 * .35);
        out.data[i * 4 + 1] = out.data[i * 4 + 2] = Math.round(v * .65);
      }
    }
    ctx.putImageData(out, 0, 0); container.append(canvas);
  }
  function meta(frame, image) {
    if (!frame) return '';
    const center = image && image.center ? ` · center (${fmt(image.center[0], 1)}, ${fmt(image.center[1], 1)})` : '';
    const gains = Array.isArray(frame.photometric_gain) ? frame.photometric_gain : [];
    const normalization = gains.length ? ` · photometric gain ${gains.map(g => fmt(g) + '×').join(' / ')}${gains.length === 3 ? ' RGB' : ''} (${frame.photometry_source || 'source unknown'})` : '';
    return `${frame.filter || 'Unknown filter'} · ${frame.capture_time || 'Capture time unknown'} · ${finite(frame.exposure_seconds) ? fmt(frame.exposure_seconds, 1) + ' s' : 'Exposure unknown'} · ISO/gain ${fmt(frame.iso_or_gain, 1)}${normalization} · HFD ${fmt(frame.hfd)} px · eccentricity ${fmt(frame.eccentricity)} · residual ${fmt(frame.residual)} px${center}`;
  }
  function populateComparison() {
    $('comparison').replaceChildren();
    if (selected >= 0) frames.forEach((f, i) => {
      if (i === selected || (f.filter || '') !== (frames[selected].filter || '')) return;
      const option = el('option', `#${i + 1} · ${label(f.status)} · ${basename(f.path)}`); option.value = i; $('comparison').append(option);
    });
    if (comparison < 0) { const option = el('option', 'No same-filter comparison'); option.value = '-1'; $('comparison').append(option); }
    $('comparison').value = String(comparison); $('comparison').disabled = comparison < 0;
    $('blink').disabled = comparison < 0;
  }
  function renderInspector() {
    const epoch = ++previewEpoch;
    const needsLoad = [frames[selected], frames[comparison]].filter(frame => {
      if (!frame || frame.images) return false;
      const match = assetPattern.exec(frame.preview_asset || '');
      return match && !previewCache.has(match[1]) && !previewFailed.has(match[1]);
    });
    if (needsLoad.length) Promise.all(needsLoad.map(loadPreview)).then(() => { if (epoch === previewEpoch) renderInspector(); });
    const a = frames[selected], b = frames[comparison], pa = patch(a), pb = patch(b), range = stretchRange([pa, pb]), zoom = Number($('zoom').value);
    const shown = blinkOther ? b : a, shownPatch = blinkOther ? pb : pa;
    renderPatch($('selected-image'), shownPatch, range, zoom, shown); renderPatch($('comparison-image'), pb, range, zoom, b);
    for (const [prefix, frame, image] of [['selected', shown, shownPatch], ['comparison', b, pb]]) {
      $(prefix + '-name').textContent = frame ? basename(frame.path) : 'No frame selected';
      $(prefix + '-name').title = frame ? frame.path : '';
      $(prefix + '-state').textContent = frame ? `${label(frame.status)} · frame factor ${fmt(frame.weight)} · guide rejection ${fmt(metricValue(frame, 'suppressed'), 1)}%` : '';
      $(prefix + '-state').className = 'detail ' + (frame ? frame.status : '');
      $(prefix + '-reason').textContent = frame && frame.reason ? frame.reason : '';
      if (frame && maskFraction(frame) !== null) $(prefix + '-reason').textContent += ` Obstruction mask: ${fmt(maskFraction(frame) * 100, 1)}% of sky cells.${maskFraction(frame) > 0 && frame.status === 'used' ? ' Only unmasked regions are eligible to contribute.' : ''}`;
      $(prefix + '-meta').textContent = meta(frame, image);
    }
    $('selected-heading').textContent = blinkOther ? 'Comparison (blinking)' : 'Selected frame';
    $('copy-selected').disabled = !shown;
    $('stretch-note').textContent = range ? `Both previews share limits ${range.low.toExponential(3)} to ${range.high.toExponential(3)} in raw normalized detector units (joint 0.5–99.8 percentiles). Missing samples appear as checks.` : 'No image samples are available for a shared stretch.';
    $('mode-note').textContent = mode === 'sky' ? 'Whole field in common reference orientation. Red shows the production obstruction mask; pixels underneath are still displayed for inspection. Missing coverage is checkered. Mask percentage counts sky cells, not rejected photons. Older reports may lack this view or mask evidence.' : mode === 'native' ? 'Native sensor samples, without interpolation. Crops center on the common reference area when registration is available; orientation and subpixel alignment can differ. Unregistered frames use sensor center.' : mode === 'aligned' ? 'Common reference-area crops, bilinearly resampled at one sensor pixel per pixel. Alignment allows direct comparison; interpolation affects apparent sharpness and noise.' : 'Downsampled whole-frame overview. Use native crops or the original file to judge fine structure.';
    const missingAsset = [a, b].some(frame => { const match = frame && assetPattern.exec(frame.preview_asset || ''); return match && previewFailed.has(match[1]); });
    $('preview-note').textContent = [needsLoad.length ? 'Loading selected previews…' : '', missingAsset ? 'Preview asset could not be loaded. Keep the review-assets folder beside this report.' : '', a && a.preview_note, b && b.preview_note].filter(Boolean).filter((v, i, all) => all.indexOf(v) === i).join(' ');
    $('gain-value').textContent = $('gain').value;
    const queue = scoped(), position = queue.indexOf(selected);
    $('frame-previous').disabled = position <= 0;
    $('frame-next').disabled = position < 0 || position >= queue.length - 1;
    $('queue-position').textContent = position < 0 ? 'No matching frame' : `Frame ${position + 1} of ${queue.length} in this review`;
  }
  function stopBlink() { blinking = false; blinkOther = false; if (timer !== null) clearInterval(timer); timer = null; $('blink').setAttribute('aria-pressed', 'false'); $('blink').textContent = 'Blink'; $('blink-indicator').textContent = ''; }
  function selectFrame(index) {
    if (!frames[index]) return;
    stopBlink(); selected = index; comparison = comparisonIndex(frames, selected);
    const indices = scoped(), position = indices.indexOf(index); if (position >= 0) page = Math.floor(position / pageSize);
    populateComparison(); renderInspector(); renderTable(); renderGraph();
  }
  function svgNode(tag, attrs, value) { const node = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v); if (value !== undefined) node.textContent = value; return node; }
  function renderGraph() {
    const graph = $('graph'); graph.replaceChildren(); const width = Math.max(420, graph.clientWidth || 1000), height = 245, left = 65, right = width - 20, top = 18, bottom = 203;
    graph.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const metric = $('metric').value, indices = scoped(), values = indices.map(i => metricValue(frames[i], metric)).filter(finite);
    let high = values.length ? Math.max(...values) : 1, low = Math.min(0, values.length ? Math.min(...values) : 0);
    if (high <= low) high = low + 1;
    const xmin = 1, xmax = Math.max(1, indices.length);
    const xp = x => left + (right - left) * (xmax === xmin ? .5 : (x - xmin) / (xmax - xmin));
    const yp = y => bottom - (bottom - top) * (y - low) / (high - low);
    for (let t = 0; t <= 4; t++) {
      const v = low + (high - low) * t / 4, y = yp(v);
      graph.append(svgNode('line', {x1:left, x2:right, y1:y, y2:y, stroke:'#344b5c'}));
      graph.append(svgNode('text', {x:left - 9, y:y + 4, 'text-anchor':'end'}, fmt(v, 3)));
    }
    const tickSet = new Set(); for (let t = 0; t <= 4; t++) tickSet.add(Math.round(xmin + (xmax - xmin) * t / 4));
    for (const value of tickSet) graph.append(svgNode('text', {x:xp(value), y:bottom + 23, 'text-anchor':'middle'}, String(value)));
    for (const [position, i] of indices.entries()) {
      const frame = frames[i], value = metricValue(frame, metric); if (!finite(value)) continue;
      const dot = svgNode('circle', {cx:xp(position + 1), cy:yp(value), r:i === selected ? 6 : 4, fill:colors[frame.status] || '#abc0cf', class:'dot', tabindex:'0', role:'button', 'aria-label':`Frame ${i + 1}, ${label(frame.status)}, ${metric} ${fmt(value)}, ${basename(frame.path)}`});
      if (i === selected) { dot.style.stroke = '#ffffff'; dot.style.strokeWidth = '2'; }
      dot.append(svgNode('title', {}, `#${i + 1} · ${basename(frame.path)}\n${label(frame.status)} · ${fmt(value)}\n${frame.reason || ''}`));
      dot.addEventListener('click', () => selectFrame(i)); dot.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectFrame(i); } }); graph.append(dot);
    }
    const missing = indices.length - values.length;
    const notes = {obstruction:'Fraction of sky-mask cells blocked. Used frames can have partial masks; no mask evidence is different from zero obstruction.', weight:'Weight before local suppression; this is not effective exposure.', photometric_gain:'Multiplicative brightness correction: mono gain or median RGB channel gain. A level-only factor of 1 means the gain was not measured. Inspect the recorded source.', suppressed:'Fraction of guide locations suppressed, not discarded exposures or exact output-pixel rejection.', hfd:'Smaller stellar HFD generally indicates tighter stars; inspect eccentricity and the crop too.', eccentricity:'Star shape measurement. Compare like filters and sample enough stars.', residual:'Registration fit residual in sensor pixels.'};
    $('graph-note').textContent = `${notes[metric]} ${missing ? `${missing} shown frame(s) have no measurement and are omitted from the plot.` : ''}${!indices.length ? ' No frames match these controls.' : ''}`;
    const unknownTimes = indices.filter(i => captureStamp(frames[i]) === null).length;
    $('order-note').textContent = `Position in current review order · ${$('order').value === 'capture' ? 'acquisition time; unknown times last' : $('order').value === 'input' ? 'build input order' : 'sorted by ' + $('order').value}. Spacing does not represent elapsed time.${$('order').value === 'capture' && unknownTimes ? ` ${unknownTimes} capture times unknown.` : ''}`;
  }
  function renderTable() {
    const indices = scoped(), pages = Math.max(1, Math.ceil(indices.length / pageSize)); page = Math.min(page, pages - 1);
    $('rows').replaceChildren(); $('table-count').textContent = `${indices.length.toLocaleString()} matching frame(s) · page ${page + 1} of ${pages}`;
    const total = integration(indices.map(i => frames[i]));
    const decisions = ['used', 'not-used', 'excluded'].map(s => `${indices.filter(i => frames[i].status === s).length} ${label(s).toLowerCase()}`).join(' · ');
    $('scope-summary').textContent = `${decisions} · ${fmt(total.seconds / 3600, 2)} known exposure hours${total.unknown ? ` · ${total.unknown} unknown durations` : ''}. Exposure totals are not effective integration.`;
    $('download-matching').disabled = !indices.length;
    $('previous').disabled = page === 0; $('next').disabled = page >= pages - 1;
    for (const index of indices.slice(page * pageSize, (page + 1) * pageSize)) {
      const f = frames[index], row = el('tr'); if (index === selected) row.className = 'selected';
      const frameCell = el('td'), button = el('button', `#${index + 1} · ${f.filter || 'Unknown filter'}`); button.type = 'button'; button.setAttribute('aria-label', `Inspect ${f.path}`); frameCell.append(button);
      frameCell.append(el('span', basename(f.path), 'path')); frameCell.title = f.path; row.append(frameCell);
      const decision = el('td'); decision.append(el('span', label(f.status), f.status)); decision.append(el('span', f.reason || 'No additional reason recorded.', 'reason')); row.append(decision);
      row.append(el('td', fmt(f.weight), 'num'), el('td', fmt(f.hfd, 2), 'num'), el('td', finite(f.suppressed) ? fmt(f.suppressed * 100, 1) + '%' : '—', 'num'));
      row.addEventListener('click', () => selectFrame(index)); $('rows').append(row);
    }
  }
  function updateScope() { stopBlink(); page = 0; const choices = scoped(); if (!choices.length) { selected = -1; comparison = -1; populateComparison(); } if (choices.length) selectFrame(choices.includes(selected) ? selected : choices[0]); else { renderTable(); renderGraph(); renderInspector(); } }
  for (const id of ['filter', 'status', 'order', 'sensor-gain']) $(id).addEventListener('change', updateScope);
  $('search').addEventListener('input', updateScope);
  for (const [id, direction] of [['frame-previous', -1], ['frame-next', 1]]) $(id).addEventListener('click', () => { const queue = scoped(), position = queue.indexOf(selected); if (position >= 0) selectFrame(queue[position + direction]); });
  $('metric').addEventListener('change', renderGraph);
  $('comparison').addEventListener('change', () => { stopBlink(); comparison = Number($('comparison').value); renderInspector(); });
  for (const nextMode of ['native', 'aligned', 'thumb', 'sky']) $('mode-' + nextMode).addEventListener('click', () => { stopBlink(); mode = nextMode; for (const item of ['native', 'aligned', 'thumb', 'sky']) $('mode-' + item).setAttribute('aria-pressed', String(item === mode)); renderInspector(); });
  $('zoom').addEventListener('change', renderInspector);
  $('gain').addEventListener('input', renderInspector);
  $('reset-stretch').addEventListener('click', () => { $('gain').value = '8'; renderInspector(); });
  $('blink').addEventListener('click', () => { if (blinking) { stopBlink(); renderInspector(); return; } if (comparison < 0) return; blinking = true; $('blink').setAttribute('aria-pressed', 'true'); $('blink').textContent = 'Stop blink'; $('blink-indicator').textContent = 'Blinking left panel'; timer = setInterval(() => { blinkOther = !blinkOther; renderInspector(); }, 800); });
  document.addEventListener('visibilitychange', () => { if (document.hidden && blinking) { stopBlink(); renderInspector(); } });
  $('copy-selected').addEventListener('click', async () => { const frame = frames[blinkOther ? comparison : selected]; if (!frame) return; const path = externalPath(frame.path); try { await navigator.clipboard.writeText(path); $('status-message').textContent = 'Full path copied.'; } catch (_) { $('status-message').textContent = `Copy this path: ${path}`; } });
  for (const status of ['used', 'not-used', 'excluded']) $('download-' + status).addEventListener('click', () => { const blob = new Blob([frameList(frames, status)], {type:'text/plain;charset=utf-8'}), url = URL.createObjectURL(blob), a = el('a'); a.href = url; a.download = `smokstak-${status}-frames.txt`; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); $('status-message').textContent = `Saved ${label(status).toLowerCase()} frame list for the whole build.`; });
  $('download-matching').addEventListener('click', () => { const indices = scoped(); if (!indices.length) return; const blob = new Blob([indices.map(i => externalPath(frames[i].path)).join('\n') + '\n'], {type:'text/plain;charset=utf-8'}), url = URL.createObjectURL(blob), a = el('a'); a.href = url; a.download = 'smokstak-review-selection.txt'; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); $('status-message').textContent = `Saved ${indices.length} matching frames across every page, in the current review order.`; });
  $('previous').addEventListener('click', () => { page = Math.max(0, page - 1); renderTable(); });
  $('next').addEventListener('click', () => { page++; renderTable(); });
  let resizeTimer; window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderGraph, 120); });
  if (selected >= 0) selectFrame(selected);
  else { populateComparison(); renderInspector(); renderTable(); renderGraph(); }
}());
