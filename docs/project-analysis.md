# Project analysis

```sh
smokstak analyze ./lights
smokstak analyze ./frames.txt --filter O
smokstak analyze ./lights --json out/project-analysis.json --html out/project-analysis.html
```

`analyze` measures monochrome FITS/XISF projects without reconstruction. It
writes authoritative, versioned JSON and a self-contained HTML dashboard with
filter selection, noise/integration axes, capture-time plots, hover details,
frame warnings and conditional improvement estimates. Open the HTML directly;
it needs no server, network, CDN or installed plotting library. Copy that file
to share the report. The command does not open a browser automatically.

`survey` measures individual frames and flags unusual star shape/count, without
registration or integration-depth analysis. `stack` reconstructs an image.
`analyze` does neither a full output raster nor colour processing, TIFF output,
restoration, burst defect detection or repeated stacks. It uses the same input
discovery, FITS/XISF decoders, star metrics, registration, photometric matching,
survey flags and flat-field metric as the existing commands.

## Inputs, ordering and acceptance

Directory, single-frame and text-list semantics are those of `sr-raw`:
filename sorting, relative list paths, comments, mixed-format restrictions,
`--pattern`, header-based `--filter`, and duplicate removal all apply. Filter
selection is case-insensitive; group identities preserve the actual header
labels, as in the stack. `--max-frames` takes the first N after selection.
`--fits-row-order` uses the existing decoder setting.

Each filter has its own first readable, sufficiently large mono reference,
registration, photometric anchor, sample footprint and cumulative curve.
Changing filters never adds their pixel measurements together. The project
summary adds their exposure durations only. Missing filters form an explicitly
warned unknown group: the program cannot determine whether they share a
passband. Bayer/colour input and incompatible dimensions are reported as
rejected. Unreadable files are retained with their errors, like `survey`.

The default curve answers how the project accumulated in **filename order**,
not quality rank or guaranteed timestamp order. A list is also sorted by the
existing discovery code. Capture starts come from FITS `DATE-OBS`, including
preserved XISF keywords; native XISF `Observation:Time:Start` is a fallback.
Missing timestamps are null. They are never invented from file modification
times. Health plots can switch to capture-time coordinates, omitting missing or
unparseable dates; that switch does not reorder the accumulation experiment.

“Accepted” means mono data of compatible dimensions with registration passing
the stack's existing usability gate. Correlation confidence is normalized
over the completed filter group using the same function as `stack`. Survey
warnings remain advisory: a soft or cloudy exposure can still contribute,
which lets its effect on project depth be measured. Inspect the warnings
and use a revised input list to study an explicitly culled project. The first
frame fixes the grid and has identity registration; its zero residual is not
an independent validation of its quality.

## Samples and noise

The default requests **262,144 detector samples**: 1,024 deterministic,
nonoverlapping candidate patches, stratified across the central 80% of each
reference's width and height. Selection depends only on dimensions and the
requested count, never on unusually quiet pixels in the first noisy frame.
Small frames use fewer patches. Each patch is a 16 × 16 measurement grid at
two-sensor-pixel pitch. `--samples` accepts 8,192–2,097,152 samples, rounded down
to whole patches; it is mainly useful for validation and profiling.

The standard WCS/asterism seeds, correlation registration, affine stellar
refinement and held-out stellar distortion fit place each target. Correlation
local-warp fitting is omitted. The forward warp maps detector coordinates to
the reference; **its inverse** gathers the detector sample nearest each fixed
reference coordinate. No full intermediate image is resampled. Nearest-site
selection is within 0.71 target sensor pixels of the inverse coordinate and
does not introduce a phase-dependent interpolation smoothing kernel. Two-pixel
pitch reduces duplicate neighbours under near-unit warps; any patch containing
duplicates is invalidated. This is approximate scene correspondence at the
detector sampling limit, not exact continuous-scene reconstruction.

Clipped, saturated, nonfinite, defective, duplicate or out-of-bounds samples
invalidate their whole patch. Only patches valid in **every accepted frame of
that filter** enter the cumulative experiment. A cheap compact-sample replay
establishes a common footprint at every N, avoiding a false noise decline from
changing coverage. Fewer than 32 common patches suppresses the curve and model,
with an explicit warning. Adding data can shrink this footprint and change
earlier points. These common-coverage values are not permanently cached.

Frames are normalized onto the reference scale using the existing stellar
photometric gain and block-median pedestal. Stellar gain availability is
recorded; the standard block/level/exposure fallbacks are explicitly warned
about, since they may confound transparency and changing sky. Additive sky
fields are off. Pairwise matching cannot apply the full burst's
obstruction consensus; `analyze` does not claim to reproduce production
rejection or weighting.

Since schema 2, the accumulator keeps two f64 sums at each common sample: the ordinary sum and a
signed sum with alternating +1/-1 frame coefficients. At **even N**:

1. Form `D = (sum_odd - sum_even) / N` at each scene location.
2. Measure `1.4826 * median(abs(D - median(D)))` within each patch using the
   existing MAD helper.
3. Report the median of patch sigmas as `noise`.

For independent samples with individual variances `v_i`, both D and the actual
full equal-weight mean have variance `sum(v_i) / N^2`. Stable scene structure
cancels from D. This identity holds with unequal variances, but does not make
MAD an exact RMS estimator for non-Gaussian, contaminated distributions.
No interpolation, clipping, or monotonic smoothing is applied to the mean.
At odd N there is no balanced measurement: `noise`, `ideal_noise`, the ratio,
and `local_fit` are null. Integration accounting still includes those frames.

The previous estimator is retained as `spatial_residual`: plane-detrended RMS
of the mean, averaged over the lowest decile of patches. This measures spatial
variation including real astronomical structure and outliers; it is neither
random noise nor an identified systematic-error floor. Its chosen decile can
change with N. The split statistic uses every common patch, with a robust
median instead of selecting a new low decile.

Difference noise also cancels shared systematic errors. Changing seeing,
registration, transparency, and residual illumination can remain in it.
Alternating observing conditions can systematically differ between the halves.
Sparse outliers have less influence on MAD, but extensive contamination can
still bias it. It measures the distribution core, not the full outlier energy.
Do not interpret the new curve as total image error or a production stack's
weighted/rejected/resampled noise. These limitations appear above the charts.

`ideal_noise = noise_at_N_2 * sqrt(2/N)` is explicitly an **equal-noise guide**,
not a prediction for varying normalized frame quality. A newly added noisy
pair can increase the measured value. No generic signal or SNR is reported.
Per-frame background median/MAD and `spatial_noise` retain their original
raw-detector definitions; registration residual is before stellar refinement.

## Fits and projections

Fit `noise(N) = a * N^b` by ordinary least squares on `ln(noise)` versus `ln(N)`.
Checkpoints start at N=4; each next checkpoint is `ceil(1.25 * previous_N)` rounded up to an even count.
The current endpoint is included once if it is not already a checkpoint.
This avoids weighting the dense late portion of a long project overwhelmingly.
At least five positive finite-noise points spanning a factor of two are
required (the first possible fit is at N=14). The global fit uses all these
checkpoints; a local fit uses only those in `[N/4, N]` plus the current endpoint.
The chart shows local fits at checkpoints plus the latest measured endpoint,
so changing window membership is not presented as a dense staircase.

Amplitude, exponent, R², point count and first/last N are serialized. R² is
descriptive fit quality, **not** a confidence interval: cumulative means share
data, so ordinary IID regression standard errors would mislead. Short windows
and small sample populations remain noisy. Per-frame quality trends should be
read alongside a changing exponent.

For desired fractional noise reduction `p`, a conditional projection is
`N_required = ceil(N_current * (1-p)^(1/b_recent))`. Report the additional
frames for 5%, 10% and 20% reductions only when the recent fit has
`b < -0.05`, `R² >= 0.8`, and a finite projected total below two billion frames.
Hours use the mean accepted exposure only when every accepted duration is
known. Flat or weak fits have null projections. These estimates assume the
recent power law persists; they do not extrapolate a noise floor correctly and
are not forecasts, exposure recommendations or promises of image improvement.

## Integration accounting and JSON

Positive finite exposure seconds are summed; missing or invalid durations are
counted separately, never silently assumed to be one second. Known time is a
lower bound when durations are missing. Equal-weight noise means still count
each frame once, even when durations differ.

Schema version 3 has these top-level fields:

| Field | Contents |
|---|---|
| `schema_version`, `program_version`, `build_sha256` | Format and exact executable identity |
| `project`, `order`, `method`, `limitations` | Input and measurement definition |
| `requested_samples` | Requested candidate sample count |
| `summary` | Input/duplicate/accepted/rejected counts, known and accepted integration, unknown exposures, cache hits |
| `frames` | Discovery index/path/filter, timestamps/exposures, cumulative known and accepted project time, survey metrics/flags, background, normalization, registration, acceptance/error/cache state |
| `filters` | Per-filter anchor, common sample count, integration accounting, medians, depth points, fits, projections and warnings |
| `timings` | Discovery, full hashing, decode, quality, registration, normalization, sampling, cache, cumulative analysis, report rendering and processing seconds |

Depth points contain N, the input frame index, cumulative known seconds and
unknown-duration count, measured noise, ideal noise, ratio and local fit.
Unavailable scalar measurements/models are JSON null, never NaN or Infinity.
The HTML embeds this same result and only plots/formats it; regressions,
ratios, medians and projections are not recalculated in JavaScript.
`report_render_seconds` measures a serialization/render dry pass; filesystem
write time is included in the terminal's total wall time, not that field.

## Cache and resource bounds

Caching is on by default in `./cache/analysis`; `--cache-dir` changes it and
`--no-cache` uses a temporary sample spool removed when the command finishes.
JSON scalar records accompany little-endian f32 sample files (about 1 MiB per
frame at the default size). Samples have checked lengths and SHA-256 digests;
missing, truncated or corrupt entries are recomputed. There are no decoded-frame
cache files. Old entries are not automatically evicted.

The existing fingerprint builder includes the cache format, program version and
build revision. Analysis adds a method-version tag, **full input SHA-256**, full
reference SHA-256, sample count, row-order setting, registration configuration,
and the **executable's SHA-256**. Thus a source rebuild, input edit anywhere in
the file, changed reference, or relevant option invalidates the entry. Different
filenames alone do not define identity. Full hashing deliberately costs a read
of every input on warm runs. Existing duplicate detection retains its bounded
head/tail convention; the analysis cache uses stronger full-file identity.

Appending files after the existing anchor reuses old measurements and only
decodes/registers new frames (plus one reference per filter). Final confidence,
survey flags, common coverage, means and fits are recomputed from compact
records so project-dependent results cannot become stale. Adding an earlier
anchor changes dependent keys. No change to stack-cache behavior is implied.

Decoded pixels occupy a reference and one current frame, plus their registration
proxies and bounded detector/photometry working storage. Scalar records grow
with N; sample accumulators grow with S; disk samples grow with N×S. No code
path stores N full images or computes stacks of lengths 1, 2, …, N. Cumulative
work is O(N×S) for fixed S (tile sorting adds a factor log(number_of_tiles),
independent of N); global checkpoint fitting is O(log N) once per filter and
local fits have bounded size.

## Recompute reports from existing compact samples

When a statistics change invalidates the executable cache key, the explicit
replay utility can reuse a preserved measurement set without reading FITS:

```powershell
cargo run --release -p sr-cli --example replay_analysis -- ORIGINAL_REPORT.json CACHE_DIRECTORY NEW_OUTPUT_DIRECTORY
```

This preserves original acceptance, order, registration, photometry, and
measurement provenance. It checks sample SHA-256, layout, photometry and common
coverage, and refuses duplicate records or an existing output directory. It
adds statistics replay provenance and regenerates schema-3 JSON/HTML using the
same Rust accumulator and HTML template as normal analyze. It does not verify
that source FITS files still match the historical report. Normal analyze cache
invalidation remains unchanged. Keep the source report with its original cache.

## Progress report (schema 3)

The default noise chart uses known integration hours on a linear axis and noise
as a percentage of a selectable measured baseline. Absolute units, frame count,
and log axes remain available. Dotted boundaries mark acquisition batches;
click a noise point or a night row to inspect cumulative changes and quality.

Night groups use capture timestamps and a configurable UTC day boundary
(default noon). They are contiguous batches in accumulation order, not a new
sort or rejection rule. Unknown dates remain explicit and do not train night
scenarios. A noise-change percentage uses the latest even measurement before
and after each batch, with the actual frame counts shown. It can straddle a
boundary by one frame and is not an independent estimate of that night's
causal contribution. HFD, background, gain, eccentricity and flags help explain
changes. Unknown exposure totals remain lower bounds.

`pair_noise` at even N is the median patch MAD sigma of the newest frame pair's
difference divided by sqrt(2). This approximates a typical single-frame noise
level for that pair, including scene mismatch. Scenarios require at least three
dated batches, each with at least three within-night pairs. Their quartiles
are variation between observed nights, not confidence intervals. Current
mean noise is converted to an equivalent frame noise by multiplying by sqrt(N).
If a future frame's noise is r times that equivalent value, the independent
variance model gives `noise_new/noise_now = sqrt(N*(N+m*r*r))/(N+m)` for m new
frames. The goal solver inverts this expression. Exposure conversion requires
complete known durations. No shared-floor or future-weather prediction is made.
The equal-noise reference remains available when the empirical fit is weak.

### Matched patch previews and region goals

At powers of two, the final count and the last even count, Rust captures all
common patches. Each patch includes its full-precision median and balanced
MAD sigma, plus 256 display samples encoded as 16-bit hex values with explicit
physical low/high endpoints. Display encoding clips at median ± 8 spatial MAD;
it does not alter the numerical statistics. The snapshots retain sparse
16x16 samples at two-pixel spacing, not contiguous full-resolution crops.

The report map shows actual reference coordinates. Users explicitly select a
signal and background patch. Earlier/latest previews and the background use
one physical black/white range and the same asinh stretch, anchored to the
latest signal patch. Changing the shared stretch affects all previews together.
These previews inspect representative local samples, not a final reconstruction.
Snapshots add O(samples * log(frame_count)) report storage; a large project's
offline HTML can reach tens of megabytes, dominated by those previews. No FITS decode is
needed for historical cache replay.

The selected-region metric is explicitly **per-sample contrast SNR**:
`(median_signal - median_background) / hypot(split_sigma_signal, split_sigma_background)`.
It uses noise in both regions and does not divide by sqrt(number of pixels).
It is a contrast proxy, not aperture SNR, detection significance, or a claim
about all nebular structure. Shared errors and spatial covariance remain
unmeasured. A positive contrast is required for an additional-frame scenario;
selecting the same patch twice is refused. Frame estimates assume stable
contrast and equal independent noise. They are not extrapolations of the global
filter slope. Goals and patch selections are retained per filter during the
open report session, without writing files or sending data elsewhere.

Filter goals compare measured contrast to user-selected targets. Acquisition
priority is withheld until every filter has a positive, configured comparison;
then the largest relative goal shortfall is identified, not automatically the
filter with the shortest exposure time. Meeting a selected goal never certifies
overall image quality.

### Default decision: review progress, without an invented finish line

The report starts in **Review progress** mode. No noise-reduction percentage,
contrast-SNR target, or minimum worthwhile gain is assigned. The headline
reports falling difference noise, an increase worth inspecting, or insufficient
trend evidence. It summarizes the net change over the last three acquisition
batches using bracketing balanced measurements, not a sum of per-night
percentages. It also shows a compact comparison of all filters.

The initial session budget is the median accepted integration duration of
complete dated batches. A user-entered duration overrides it. The projected
benefit is a rounded conditional scenario from observed night-quality quartiles,
not a forecast, confidence interval or universal recommendation. No hours are
invented when exposure information is incomplete. Weak fits remain explicit;
a button opens the batch with the largest observed noise increase for review.
This is an inspection aid, never automatic frame rejection or proof of cause.

The 5%, 10% and 20% buttons compare optional improvement budgets. Clicking one
explicitly activates that scenario. A custom percentage must be entered;
blank or invalid input does not silently become 10%. Selected-region SNR goals
also start blank, rather than assuming SNR 5. Only an explicit, measured region
goal can yield a zero-additional-frames result. Meeting that contrast goal does
not certify final-image quality. Per-filter selections are retained while the
report is open; the review and detailed panels share the same settings.

### Reading the report

The default view puts the next action beside a comparison of all filters.
The comparison explicitly says whether measured noise is lower or higher over
each filter's last three batches; rising noise in another filter remains visible
while exploring the selected one. It does not rank filters by exposure totals.
Four primary readouts cover accepted time, accepted frames, recent noise change
and star size. The two primary charts show integration progress and batch changes.

Session inspection, conditional collection budgets, selected-feature contrast
SNR, frame quality and model diagnostics are expandable. Section shortcuts open
and scroll to the corresponding panel. Selecting a noise or batch point opens
its acquisition session. The session table can display latest batches first,
largest noise increases first, or original accumulation order; sorting never
changes the accumulation or statistics. Explicit lower/higher labels accompany
color, and the ideal guide uses a dashed line distinct from measured noise.

Both numeric axes use round tick values; the noise-chart readout identifies its
baseline and latest balanced frame count. The patch map preserves coordinate
aspect ratio and labels signal/background selections S/B. Matched previews
remain sparse samples with a shared stretch, not full-resolution image crops.
Method cautions, the guide ratio, common sample count, source path and cache
timings remain available in the method sections. No measurements or automatic
stopping thresholds are changed by the presentation.

### Reviewing upward steps and exporting inputs

**Review / export frames** lists the accepted frames added between successive
balanced measurements whose noise increased by at least the selected percentage.
The default 1% threshold is an adjustable screening choice, not a significance
test or rejection rule. Usually two frames belong to a step. Both are candidates;
the cumulative estimate cannot establish which member is responsible. Attribution
uses the accepted sequence within each filter, not adjacent global input indices.

Checkboxes start empty. Review jump candidates, the selected acquisition session,
or every input in the selected filter. Mark individual frames or all shown.
Marks survive filter changes during the open report session. The original curves
remain unchanged. The save button downloads a self-contained ZIP containing:

- `retained-frames.txt`: all report inputs except explicitly marked frames.
- `questionable-frames.txt`: only marked frames, for inspection or another run.
- `questionable-frames.csv` and `.json`: original paths, quality measurements,
  capture times and associated upward intervals.
- `copy-questionable.ps1`: optional original-byte copying into a new directory.
- `README.txt`: copy instructions and rerun examples.

The export includes all filters represented in the report, preserves unselected
technical rejects, and never modifies originals. Select a filter when stacking a
multi-filter list. The regular input reader applies its usual filename sorting.
New analyses record absolute paths, including when invoked with a relative input;
old reports with relative paths refuse a portable export rather than guessing.

The copy helper requires Windows and mounted source drives. `-WhatIf` checks
sources and reports the byte count without copying. Actual copies use unique
input-index prefixes and produce `copied-frames.txt`; existing destination
directories are refused. The report contains no full-resolution image pixels.
Inspect the originals or their byte-identical copies in your chosen astro software.

For a reproducible **provisional** experiment without browser interaction:

```text
node tools/export_analysis_review.cjs project-analysis.json new-review-directory 1
```

This marks every interval candidate meeting that threshold across all report
filters, writes the same files and ZIP, and refuses existing output directories.
It does not establish that the selected frames are bad. Removing frames changes
integration time and split pairing, and may change the reference and common
footprint. Compare reruns using the same settings and judge the final integration
at matched scale and stretch; optimizing this sampled curve alone is insufficient.

The `replay_analysis` example also accepts an optional fourth argument containing
an exact retained-path list. This recomputes the depth curves and snapshots from
checksum-verified saved samples, reindexes retained inputs, and recalculates
counts, integration totals and quality medians. It fixes the original registration,
photometry and common footprint for comparison; advisory flags and original
measurement timings are retained. The output is explicitly labeled a subset
replay, not fresh source-image measurement. The destination must not exist.

```text
replay_analysis original-report.json original-cache new-report-directory retained-frames.txt
```
