# smokstak

Multi-frame super-resolution for camera bursts and astronomical sub-frames, in
Rust.

Takes a burst of frames of the same scene — camera RAW, FITS or XISF — and
reconstructs one higher-resolution image by merging the **original CFA sensor
samples** on a common high-resolution grid. No per-frame demosaic, no upscaling
filter, no learned prior — the extra resolution comes from the sub-pixel
diversity that is already in the burst.

```
smokstak stack ./burst --scale 2 --output result.tif --diagnostics ./diag
```

## Status

### Windows portable app

Download the artifact from a successful run of the **Windows portable** workflow
(`.github/workflows/windows-portable.yml`) in this repository's Actions tab.
Extract the artifact, then extract `smokstak-windows-x64.zip` and double-click
`Start-smokstak.cmd`. It opens the embedded web UI in your browser; keep the
launcher window open while processing. No development tools are required.
The app runs locally on Windows x64, including an RTX 4090 workstation, but
currently computes on the CPU: GPU acceleration is not implemented.

Builds run on pushes to `main`, `v*` tags, pull requests, and manual workflow
dispatch. Each ZIP includes its source revision and has a SHA256 checksum.
These are development builds.

The engine runs end to end on four real bursts — a 100-frame tripod test chart
and a 200-frame hand-held roof scene, both Nikon NEF; a 36-frame dithered
one-shot-colour deep-sky sequence; and a monochrome narrowband set through three
filters — plus synthetic ground truth. GPU acceleration is not started.

An experimental CLI path, `smokstak mosaic build`, reconstructs mono H-alpha
mosaics and mixed-instrument inputs from a validated geometry/photometry plan.
It reads native windows and writes tiled scientific FITS products with bounded
working buffers; `smokstak mosaic prepare` builds the plan from raw inputs. See
[the mosaic guide](docs/mosaics.md) for inputs, checks and limitations.
The GUI's **Mosaics & mixed scopes** section imports prepared plans, reviews
source footprints and registration, and runs the same tiled engine with progress,
cancellation and saved results. Ordinary stacking defaults remain unchanged.

## What it actually delivers

Measured on a sample burst — 100 × Nikon Z 7II NEF, 45 MP, 400 mm
f/8, ISO 800, 1/15 s, tripod, ISO-12233 resolution chart. Slanted-edge MTF50 in
**cycles per sensor pixel**, which is the only figure comparable across output
scales; an upscaler improves cycles-per-*output*-pixel while resolving nothing.
Reproduce with `smokstak measure` — see below.

| method | MTF50 (cy/sensor px) | flat-field noise |
|---|---|---|
| 1 frame, no merge | 0.1475 | 0.00339 |
| 100 frames, demosaic → align → average | 0.1175 | 0.00125 |
| 100 frames, CFA drizzle (isotropic kernel) | 0.1661 | 0.00271 |
| 100 frames, RAW-domain burst SR | **0.1713** | **0.00243** |

Three things worth reading off that table.

* **+16% resolution over a single frame**, with 28% less noise. Real detail,
  not contrast: MTF50 is measured from a slanted edge, and the overshoot column
  (`smokstak measure`) stays at 0.006, so this is not sharpening.
* **+46% resolution over demosaic-then-average.** This is the number that
  justifies the whole raw-domain architecture. Demosaicing each frame first
  destroys the sub-pixel information the merge needs, and averaging afterwards
  cannot bring it back — it produces the *lowest* noise in the table and the
  worst resolution by a wide margin.
* **The structure-aware kernel beats isotropic drizzle on both axes**, +3%
  resolution at 10% lower noise. That only became true once the detail
  threshold was made noise-relative; before that, the kernel was mishandling
  low-contrast texture and the two were resolution-neutral.

The gain over one frame is modest because this burst is *optically* limited: at
400 mm f/8 the lens and diffraction, not the sensor grid, set MTF50 near
0.15 cy/px. Super-resolution removes the sampling and demosaic contribution to
the blur. It cannot remove the optics, and nothing here pretends otherwise.

Against synthetic ground truth, where the optical PSF is known and the burst has
full sub-pixel diversity, the same code delivers **+3 to +5 dB PSNR** and **+24%
MTF50** over one interpolated frame, and **+4.5 dB on a low-contrast
fine-texture panel** — the content a slanted-edge measurement cannot see.

A 1:1 crop of the chart's ring pattern, one frame against the reconstruction,
shows rings staying separated considerably further toward the centre, and the
colour fringing on them disappearing.

### How noise falls with frame count

Same scale, same kernel, uniform out-of-focus region, `--local-warp off`:

| frames | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 100 |
|---|---|---|---|---|---|---|---|---|
| noise sigma | 0.0254 | 0.0199 | 0.0153 | 0.0115 | 0.0090 | 0.0072 | 0.0062 | 0.0058 |
| ideal `1/sqrt(N)` | 0.0254 | 0.0179 | 0.0127 | 0.0090 | 0.0063 | 0.0045 | 0.0032 | 0.0025 |

Noise falls monotonically and substantially — 4.4x by 100 frames — but lands
short of the ideal `sqrt(N)`, flattening against a floor near 0.005. That floor
is a *correlated* component that averaging cannot remove: fixed-pattern sensor
response, plus real low-contrast structure in the region being measured, which
is a photograph of a wall and not a flat field. Separating the two needs a dark
frame and a flat field, which this dataset does not include. The honest summary
is that the temporal averaging works as expected and the measurement floor is
not purely sensor noise.

### A second, harder burst

A roof burst — 200 frames, hand-held, ISO 100, 1/160 s,
400 mm, of a vent on an asphalt-shingle roof. Everything about it is harder than
the chart: a repetitive texture that makes patch matching ambiguous, 63 px of median inter-frame motion with a ~117 px step where the camera
was moved between frames 99 and 100, and enough real deformation that local warp
earns its place (+41.9% alignment improvement, against −9.6% on the chart).

The pipeline handles it: both halves of the bimodal burst register correctly,
with no ghosting or doubling in the output, and every output pixel is backed by
real samples. Full frame in 236 s, 3.85 billion samples merged, 182.3 effective
frames per pixel.

There is no clean slanted edge anywhere in this scene, and `smokstak measure`
says so — it reports edge straightness of 0.41–0.47 and the accompanying warning
rather than a confident MTF number from a granular shingle boundary. Two other
measurements do work:

| measurement | 1 frame | 200 frames |
|---|---|---|
| noise on the smooth vent pipe | 0.00430 | **0.00074** (5.8x lower) |
| edge straightness on the pipe rim | 0.224 | **0.887** |

The straightness figure is worth a second look: on a single frame the pipe's rim
is too noisy for the edge finder to identify a coherent edge at all, and on the
merge it is clean.

For resolution on a scene with no measurable edge, compare radially averaged
power spectra against the same frame interpolated up, in cycles per sensor pixel:

| band | 0.15–0.20 | 0.20–0.25 | 0.25–0.30 | 0.30–0.40 |
|---|---|---|---|---|
| merged / single | 1.00 | 1.28 | **1.53** | **1.38** |

More energy where the optics still carry signal. That is the signature of
resolution rather than of a filter: this path contains no sharpening, and a
merge cannot amplify a frequency — it can only combine samples.

A caution on reading these ratios: a spectrum must be anchored at a low
frequency, where both images see the same real scene content. Comparing raw
power confounds the two images' different exposure normalisation, and
normalising by total variance is worse — the single frame's noise inflates its
own variance and deflates its whole normalised spectrum, flattering the merge.

This burst also found a kernel defect. Compared side by side against a
single frame, both auto-levelled, the reconstruction looked *softer* on the
shingle granules. It was: the detail-versus-flat threshold was an absolute
gradient value carried over from the literature, and low-contrast texture that
sits well above the noise was being classified as flat and blurred with a kernel
of sigma 1.2 sensor pixels. The chart never showed it, because a printed edge is
far above any absolute threshold, and neither did the MTF measurement, because a
slanted edge is precisely the case the kernel handles well.

Ground truth then settled what the eye could not. The merge does look smoother
than a single frame on that texture, and it is nonetheless *more accurate* there
— 47.2 dB against 42.7 dB on the synthetic texture panel. Most of the extra
"detail" in a single frame is sensor noise and demosaic speckle that auto-levels
amplify. Both facts are true, and only ground truth separates them.

It also found two real defects: a
registration confidence metric calibrated against an absolute residual that did
not transfer from tripod to hand-held (it labelled 82 good frames as failures
and down-weighted them threefold), and a reference-selection bias that, once
corrected, exposed a worse latent bug — the scorer choosing a frame containing a
transient object, which then becomes the definition of the scene.

### Lateral chromatic aberration

A lens images red and blue at slightly different magnifications from green, so
colour fringes appear toward the frame corners. This is a *geometric* defect,
which makes it a natural fit here: the merge already deposits individual sensor
samples and already knows the colour of each one, so the correction is one extra
transform on a sample's position before it is deposited — no resampling, no
separate pass, and applied *before* frames are combined rather than to an
already-interpolated image.

It is measured from the burst, not from a lens profile. The same 400 mm lens,
three unrelated scenes:

| burst | red magnification | blue |
|---|---|---|
| brick chimney, 25 frames | −0.03855% | −0.03231% |
| test chart, 100 frames | −0.03854% | −0.03487% |
| shingle roof, 200 frames | −0.04696% | −0.02606% |

Brick and chart agree on red to four decimal places, independently measured on
completely different subjects. Both channels come out *smaller* than green, so
this lens fringes green/magenta rather than the textbook red/cyan.

Correcting it on the brick burst removes **82% of the channel misregistration**
(residual 2.75 → 0.48 output pixels at the corner, about 1.37 → 0.24 sensor
pixels). What remains is longitudinal aberration — channels focused at different
distances rather than imaged at different sizes — which no geometric correction
can touch.

On by default; `--no-ca` disables it, and `smokstak inspect` reports the
measurement whether or not you correct.

## Fast project analysis

```sh
smokstak analyze ./lights
smokstak analyze ./frames.txt --filter O
```

For large monochrome FITS/XISF projects, `analyze` writes
`project-analysis.json` and a standalone interactive `project-analysis.html`.
It reports frame health, integration time, per-filter robust half-stack difference noise, separate spatial residuals,
equal-noise `1/sqrt(N)` guides, global/recent exponents and conditional improvement
estimates. It retains a reference and one current decoded frame, samples fixed
registered scene locations, and caches compact measurements for incremental
reruns. It does not reconstruct an image or claim generic SNR.

Unlike `survey`, it registers and measures cumulative depth; unlike `stack`, it
produces no full raster. Filename order is explicit, filters stay separate, and
missing exposure durations and uncertain fits stay visible. See
[definitions, options, cache behavior and limitations](docs/project-analysis.md).
The report also compares observing nights, shows matched sampled-patch previews,
and supports selected-region contrast-SNR goals and conditional extra-hour scenarios.

## Persistent projects and later data

```sh
smokstak project init ./my-project ./frames.txt --reference-file ./reference.fit
smokstak project build ./my-project
smokstak project add ./my-project ./new-night
smokstak project build ./my-project
```

For compatible calibrated monochrome FITS/XISF, projects save full-content
identities, a pinned grid, processing settings and reversible review decisions.
Every build runs the production stacker over the complete selected population
and publishes new masters without overwriting earlier runs. Disk-backed retained
buffers reduce heap growth; updates currently remerge all selected data.
See [project commands, validation and limitations](docs/projects.md).

## The page

`smokstak gui` serves one page on `127.0.0.1` and opens a browser at it. It is
a front end and nothing else: it runs `smokstak stack` as a child process and
reads its output to say which stage the run has reached, so the page and the
command line cannot disagree about what a reconstruction is. It adds no
dependencies -- the server is a few hundred lines against `std::net`, which is
less code than a toolkit's "hello window" and a great deal less to trust.

It takes a list of frames dropped on it, pasted into it, or chosen through the
file picker, and it takes a typed path to a list, a folder or a single frame. A
dropped file arrives without its folder, so a list of *relative* paths cannot be
resolved and is refused with that explanation rather than guessed at.

Between reading the burst and running it, the page asks the two questions worth
asking -- the output scale, and whether to favour depth or texture -- and offers
the background and per-filter switches with a sentence each on what they mean.
Ordinary stacks use private disk-backed working buffers, with a selectable
scratch folder, so large sets retain global photometry and rejection decisions
instead of independently processing batches. This is not a hard RAM limit and
needs ample scratch disk space. The default worker count is two.

Frame review and diagnostics are enabled by default. Finished runs offer
**Inspect frames and masks**, opening the saved report locally with its lazy
preview assets. Reports remain accessible from run history. The mono
**Noise and integration analysis** control measures the loaded inputs minus
survey exclusions, independently of a trial stack's frame limit. Each analysis
writes a new report folder and reuses compact measurement caches; it does not
measure the finished production master or automatically exclude frames.

**Saved mono projects** exposes create/build, add-and-rebuild, exclude/restore,
relink, and status/latest-result actions. Project updates reconsider all selected
exposures and preserve earlier masters. New projects use the default production
1× recipe; existing projects retain their saved recipe, independently of the
ordinary stack controls. Project ingestion and noise analysis currently require
monochrome data; ordinary stacking and its frame audit also support OSC.

It can also measure every frame first (`smokstak survey`, below) and leave out
the ones that are wrong rather than soft: trailed stars, a clouded sky, a file
that will not decode. Soft frames are named but kept, because the stack already
weights them down. Finished runs, their stats and the colour images made from
them are kept between sessions in `%APPDATA%\smokstak\history.json` (or
`~/Library/Application Support/smokstak/` on macOS, `$XDG_DATA_HOME/smokstak/`
elsewhere); `SMOKSTAK_HISTORY` names another file, and set empty keeps none.

Frames can also come from the catalog server rather than a folder. The page
asks it (`GET /frames` with a target, filter, dates, a `where` expression, or a
query saved in the catalog itself), shows what that found — frames, integration
per filter, dates, a few thumbnails — and finds each frame on this computer:
under the path the catalog rewrote for stacking machines, under its own path,
or under a "from => to" rewrite given on the page. The ones found are written
as a list and read like any other. Where the catalog screens its frames, the
page can leave out the ones it judged trailed, partly blocked, crossed by a
satellite, or standing out from the rest of their session (a passing cloud, a
veil, dew); a frame it has not judged yet is kept, and the page says how many
of those there were. Parameters are checked against the catalog's own
specification first, because the server ignores one it does not know and
would otherwise match every frame a mistyped filter was meant to leave out.
A query can be bookmarked; the bookmark
keeps the question, so asking it again picks up every frame indexed since. The
server is reached through the system `curl` (credentials on its standard input,
never its arguments) and its address, credentials, rewrites and bookmarks are
kept in `catalog.json` beside the history; `SMOKSTAK_CATALOG` names another file.

## Honesty features

These exist because a stacker that cannot be caught inventing detail is not
worth trusting.

* **Sub-pixel diversity is measured and reported before merging.** A burst with
  no inter-frame motion is told it supports 1.0x, not 2.0x, however many frames
  it has.
* **The local warp field must prove it improves alignment**, or it is discarded.
  On the chart burst the fitted fields made alignment 9.6% *worse* — they were
  fitting correlation noise — so they are refused.
* **Super-resolution and sharpening are separate.** `--postprocess none` is the
  default; when restoration is used, the unrestored result is written alongside.
* **Every run emits a manifest** (`run.json`) with source hashes, the reference
  frame and why it was chosen, every parameter, the noise model, the colour
  transform, timings and warnings.
* **Coverage, weight, rejection and effective-frame-count maps** are written per
  run, so "is this part of the output supported by real samples?" is answerable.

## Commands

```bash
# What is in this burst? Metadata consistency, quality, motion, diversity.
smokstak inspect ./burst

# Measure every frame, one per core and none held, and name the ones that look
# worse than the rest of their filter: elongated stars, few or no stars, soft,
# unreadable. Works on a night too large to stack at once.
smokstak survey ./burst --json survey.json

# Register only, with diagnostics; no resampled intermediates are produced.
smokstak register ./burst --diagnostics ./diag

# Reconstruct.
smokstak stack ./burst --scale 2 --output result.tif --diagnostics ./diag

# The input can also be a text file naming the frames, one per line, for a
# selection that is not a directory: the result of a catalogue query, or the
# frames that survived a review. Relative paths are taken against the list.
smokstak stack ./frames.txt --split-by-filter --output ngc6871.tif

# A page, for a burst you would rather not type flags for. Drop the list of
# frames on it, or paste it, or point it at a folder; it reads the burst, asks
# what you want, then runs this same program and shows how far it has got.
smokstak gui

# Synthetic ground-truth validation: eight scenarios, every claim checked.
smokstak selftest

# Measure resolution, noise and residual chromatic aberration on finished
# images. Without --edge-box only noise and fringing are reported. On a crop,
# pass --ca-centre, since the aberration is radial about the optical axis and
# that axis may be outside the crop.
smokstak measure result.tif@2 single.tif@1 --edge-box 2700,1550,40,120
```

Useful options for `stack`: `--backend rgb-mean|cfa-drizzle|burst-sr`,
`--roi X,Y,W,H` (iterate on a crop), `--max-frames N --select first|spread|sharpest`,
`--local-warp off|on|auto` (`on` forces the field and warns if it is not helping),
`--lucky off|auto|0.5`, `--postprocess none|mild`, `--float-tiff`, `--tile N`.

Kernel tuning, for measuring the resolution/noise trade rather than arguing
about it: `--k-detail`, `--k-denoise`, `--detail-snr`, `--kernel-radius`,
`--no-denoise-scaling`. Chromatic aberration correction is on by default;
`--no-ca` turns it off.

### Using only part of a burst

`--max-frames N` keeps `N` frames; `--select` decides which, and the choice is
not cosmetic.

* `first` (default) — filename order. Cheapest, and *not* a quality judgement.
* `spread` — `N` evenly spaced across the burst. Free, and keeps the burst's
  motion envelope intact.
* `sharpest` — survey the whole burst, keep the sharpest `N`. Costs an extra
  decode pass, done frame-by-frame so peak memory stays at one frame per thread
  rather than the whole burst.

Why it matters: the roof burst's camera position steps by ~117 px between frames
99 and 100, so its first 120 frames are 100 from one position and 20 from the
other. Taking 120 of 200 three ways:

| `--select` | median inter-frame motion | effective phases (h / v) |
|---|---|---|
| `first` | 4.00 px | 20.2 / 17.6 |
| `spread` | 62.92 px | 23.0 / 25.4 |
| `sharpest` | 5.83 px | **26.6 / 26.6** |

All three still support 2x on this burst, which has diversity to spare. On a
tighter burst the difference between 17.6 and 26.6 effective phases is the
difference between 2x being supported and not.

## Deep sky and FITS

FITS files are read directly; `--pattern` is optional, so a directory of `.fit`
or `.fits` is picked up the same way a directory of NEF is.

XISF is read too, which is how a set that has already been through another
calibration pipeline often gets in, and there is no calibration here yet to do
that job.
Single-channel files stack like any other frame. A file that has already been
debayered is not a frame — there is no mosaic left in it to reconstruct from —
but `measure` reads one, so a master from other software can be measured by
the same code that measures ours, with no export in between.

```
smokstak stack ./lights --scale 2 --output veil.tif --diagnostics ./diag
```

An astronomical burst is a different problem from a hand-held daytime one in
three ways that each needed work, and none of them were visible in the metrics
the project already had.

**The header cannot be trusted for the things that matter.** `BAYERPAT`
describes the image as the capture program displays it, and whether that matches
the array as stored depends on `ROWORDER`, which the files here do not carry.
Getting it wrong swaps red for blue. The white level is not 65535 on a 14-bit
sensor writing 16-bit files. Both are taken from the pixels instead — the green
sites are found from the cell means, the converter's step from the low bits that
no sample ever sets.

**The sky brightens during the session.** Over 71 minutes the background of one
test burst rises 12-14%, which is five times the tolerance motion rejection
allows, at every pixel. Uncorrected, the merge discarded a third of its samples
for no reason but the time of night. Each frame is now carried onto the
reference's photometric scale by an affine map: gain from matched stellar fluxes
where supported, pedestal from block medians, with block/level fallbacks.
Per-frame additive sky fields are off by default.

**Monochrome sensors and narrowband filters.** A mono camera is handled as a
mosaic whose four positions carry one colour, so all the 2x2 geometry keeps
working, while the channel count drops to one: single-channel output, no colour
transform, no chromatic aberration, and no kernel widening for channels that are
not sparsely sampled. Absent channels are allocated at zero size rather than
filled with zeros, which at 2x on a 26 MP sensor is two and a half gigabytes
saved.

A narrowband set holds several filters in one directory. They register against
each other perfectly well, which is why merging them is a *fatal* error rather
than something left to look plausible — `--filter H` selects one, reading each
file's header rather than its name.

**Iterating.** `--cache` reuses the alignment and the defect scan between runs.

Tuning means changing the scale, the kernel or the region, and none of those
change how the frames line up. On a 400-pixel crop of the 100-frame chart burst
the merge — the only stage that depends on what is being tuned — is 1.5 s of a
69 s run. With the cache warm that run is 33 s, and its output is byte-identical
to the uncached one.

Off by default: the fingerprint covers the input files and the configuration
those stages read, but it cannot cover a change to the code, so clear `./cache`
after changing registration or defect detection.

**Looking at the result.** `--preview` writes `<output>.preview.png` beside it.

A deep-sky result is not dark, it is *flat*: the median sits near mid-grey and
the robust spread is under a percent, so on screen it is one shade with a few
white dots. A daytime result from the same pipeline has fifty times the spread.
The preview applies the standard astronomical screen transfer — shadows clipped
a few deviations below the background, then a midtone transfer putting the
background at a quarter — and applies it **only when it would widen the spread**,
so a chart comes out unchanged rather than darkened. Nothing touches the result
itself.

**All three filters onto one grid.**

```
smokstak stack ./lights --split-by-filter --scale 2 --output out/m.tif
smokstak composite H=out/m_H.tif O=out/m_O.tif S=out/m_S.tif --palette sho -o sho.tif
```

`--split-by-filter` registers every frame in the directory to one reference and
then reconstructs each filter separately onto the grid that fixes. The masters
share a grid by construction — compositing them finds **0.04 pixels** left to
correct, against 16.0 for separately stacked masters — so there is no second
resampling to do.

What makes it work is separating two things that had been one word: the
*geometry* reference fixes the output grid and is global, while the *comparison*
reference, which motion rejection and the photometric match measure against, has
to stay inside a filter. An H-alpha frame and an OIII frame register against
each other perfectly well and disagree about the entire nebula.

**Compositing the channels.**

```
smokstak composite H=h.tif O=o.tif S=s.tif --palette sho --output sho.tif
```

Masters stacked one filter at a time do *not* share a grid — each stack chose a
reference from inside its own filter, and on one narrowband set that puts oxygen
16.0 pixels from hydrogen. Combined without alignment every star splits into
three coloured dots, so alignment is on by default; on masters from
`--split-by-filter` it finds nothing to do.

`--fit linear` matches gain and offset between channels, which is the usual
choice for a palette and also the one that throws away the most: afterwards the
channels are equal by construction and what survives is where they differ
*spatially*. `--fit offset` matches only the background and keeps the relative
brightness the filters actually measured. Palettes are `sho`, `hso`, `hoo` and
`rgb`, or `--map R=H,G=S,B=O`. None of them is a colour anything really is —
sulphur and hydrogen emit 16 nm apart in the deep red — and the choice is
yours.

Registration on the monochrome set agrees with its plate solves to **0.0003
degrees and 0.19 px at the worst corner**, better than the colour burst on every
measure, which is what sampling every site rather than one in four should give.

**Background flattening, if you ask for it.** `--flatten-background` fits
vignetting and a sky gradient and removes them — dividing the first and
subtracting the second, because one scales the light and the other adds to it,
and treating both the same way silently rescales every real brightness in the
frame by a different factor. On one test burst the corner-to-centre
background spread falls from 2.04% to 0.37%.

Off by default, and it stays that way: nothing in one image distinguishes a
corner that is dim because of the optics from one that is dim because the object
is not there. With `--diagnostics` the fitted model is written out as an image,
which is how you check it did not start fitting your nebula. It is not a
substitute for a flat frame — dust motes and per-pixel sensitivity are far too
small for any model this smooth to see.

**Frame quality is measured on the stars.** Gradient energy ranks a star field
mostly by how many stars are in it. Sharpness is now a half-flux diameter where
a burst has point sources, reported in arcseconds when the header gives a plate
scale, with eccentricity alongside to say *why* a frame is soft — seeing is
round, a guiding error is not. One test burst reports four frames elongated
within a few degrees of horizontal, which is a tracking problem and not weather.
Daytime bursts fall back to gradient energy automatically.

**Hot pixels become streaks, not dots.** A defect is fixed to the sensor and the
merge aligns the scene, so a dithered burst drags each bad site across the
output. One test burst had about sixteen hundred of them, single-channel and
radial, invisible to every metric in the suite and obvious in a 1:1 crop. They
are found from the burst — a site whose excess over its same-colour neighbours
stays put while the scene does not — and masked.

On the 36-frame IC 1340 sequence (ASI294MC Pro, 4144x2822, 120 s subframes,
dithered, 1.3 degrees of field rotation end to end):

| | |
|---|---|
| full 2x reconstruction | 8288 x 5644 in 44 s |
| registration residual | 0.213 sensor px, median |
| agreement with the plate solves | 0.0020 deg rotation, 0.42 px worst corner |
| sub-pixel diversity | 8.8 / 10.0 effective phases, ample for 2x |
| samples rejected | 0.76% |
| effective frames | 33.9 of 36 |
| stars measured per frame | 626, half-flux diameter 3.65 to 7.77 px |

And on the monochrome narrowband set (ASI2600MM Pro, 6248x4176, 10 x 180 s
through H):

| | |
|---|---|
| full 2x reconstruction | 12496 x 8352 in 26 s |
| registration residual | 0.126 sensor px, median |
| agreement with the plate solves | 0.0003 deg rotation, 0.19 px worst corner |
| samples rejected | 0.08% |
| background noise | 0.163 of the background in one frame, 0.025 in the stack |

Registration was graded against a source that knows nothing about this program:
every frame carries its own plate solve, and the WCS gives an independent
frame-to-frame transform to compare against. That is the strongest validation in
the project — stronger than the synthetic harness, because the data is real.

What is *not* handled: calibration frames and colorimetric
output. The result is linear and needs stretching elsewhere.

## Documentation

* **`docs/conventions.md`** — the four coordinate grids, which way transforms
  point, and where the units change. Read this before touching geometry; nearly
  every bug in this codebase has been a coordinate or a sign.
* **`docs/diagnostics.md`** — what each file in a `--diagnostics` directory
  means, and what to check when a result looks wrong.
* **`docs/project-analysis.md`** — the `analyze` report: definitions, options,
  cache behavior and limitations.
* **`docs/projects.md`** — persistent project commands and outputs.
* **`docs/mosaics.md`** — mono mosaics and mixed instruments: preparation, checks, build.
* **`docs/scientific-exports.md`** — FITS and XISF master exports.

## How it works

```
NEF burst
  ↓  sr-raw        decode, black/white normalise, mask bad samples
  ↓  sr-noise      fit var(x) = alpha*x + beta from the burst itself
  ↓  sr-quality    per-frame and per-region quality vectors
  ↓  sr-register   phase-correlation probes → robust affine ladder;
  ↓                 lateral chromatic aberration measured the same way
  ↓  sr-warp       optional smooth local field, gated on measured improvement
  ↓  sr-reconstruct  measure diversity, build kernels, reject motion, merge
  ↓  sr-color      white balance, camera matrix, optional sRGB encode
  ↓  sr-output / sr-diagnostics   TIFF, maps, manifest
```

The rule the design exists to protect: **detector samples stay detector
samples.** Registration results, quality, noise and confidence travel *beside*
the mosaic as metadata. No stage produces a resampled RGB intermediate, and the
merge reads the original sensor values.

Some pieces worth knowing about:

* **Registration** probes the reference with a grid of patches, pulls the
  matching patches out of the target *through the current estimate*, and fits an
  update. Working on residuals keeps the correlator in the small-shift regime
  where it is accurate to under 0.02 px. Sub-pixel refinement evaluates the
  inverse transform where the peak actually is rather than fitting a parabola to
  it, which removes a systematic bias of several percent of a pixel.
* **Plate-solve seeding** reads the WCS most capture programs write into a FITS
  header — or, for an XISF file, the astrometric solution properties stored
  there instead — and offers it as a starting estimate. Correlation refines and does not
  search, so a burst spanning a meridian flip contains frames a half turn apart
  that it cannot find from the identity — on a 68-hour NGC 6871 set that was 28%
  of the integration, weighted out of the merge with a confidence of 0.003. The
  seed is only ever offered: where it matters the frame is registered from both
  starting points and the pixels decide.
* **The transform ladder** goes translation → Euclidean → similarity → affine
  and stops at the simplest model the residuals justify, because extra degrees
  of freedom absorb real local deformation into a global warp and invent detail.
* **The merge kernel** is oriented by the local structure tensor: elongated
  along an edge, narrowed across it, widened in flat regions. Following Wronski
  et al., *Handheld Multi-Frame Super-Resolution* (SIGGRAPH 2019) §5.2.
* **Robustness** compares each frame against the reference through the noise
  model, with a tolerance that grows with the local gradient times the
  registration uncertainty — so textured regions are not rejected merely for
  being textured. On the synthetic motion scenario it catches 19% of samples.
* **Tiling** is over the output; for each tile and frame the source rectangle is
  found by inverting the transform. Reconstruction working buffers are bounded
  by tile size, but `stack` still retains the decoded burst and its proxies.
  The result is bit-identical across tile sizes (tested). `analyze` instead
  streams decoded frames and retains only compact measurements.

## Performance

Full chart burst, 100 × 45 MP → 10816 × 7200, AMD Ryzen 9 5950X (16 cores):

| stage | time |
|---|---|
| decode 100 NEFs | 9.1 s |
| registration | 19.7 s |
| robustness maps | 13.4 s |
| merge (1.96 billion samples) | 33.9 s |
| colour, write, diagnostics | 37 s |
| **total** | **~2 minutes** |

The 200-frame roof burst takes about 210 s to the same output size, with the
merge at 71 s — linear in frame count, as expected.

Samples are held as decoded, 16-bit, and normalised on read, so a frame costs
**37 MB rather than 97 MB**: 7.3 GiB for a 200-frame 45 MP burst instead of
18.1 GiB. Saturation and clipping are derived from the value rather than stored,
which is where the rest of the saving comes from. Decoding got 4x faster as a
side effect, because it no longer converts every sample and writes four bytes
where two will do.

The projected figure is logged before decoding begins, and warned about above
24 GB, because the alternative way to discover it is an allocation failure two
minutes into a run.

## Testing

```bash
cargo test --workspace                        # 200 unit tests
cargo clippy --workspace --all-targets        # clean, and expected to stay so
smokstak selftest                              # 8 end-to-end scenarios against ground truth
```

A change that alters the merge, registration or colour path should also be
checked against its own output: run a reconstruction before and after and
compare the files. Several changes that were meant to be behaviour-preserving
have been confirmed that way, and it is a stronger statement than any test in
the suite.

The synthetic harness runs the *same* registration and merge code as real data
over bursts whose geometry, blur, noise and content are known, and checks each
claim the program makes: registration accuracy against planted shifts,
resolution against a single interpolated frame, preservation of low-contrast
fine texture, raw-domain merge against demosaic-then-average, motion rejection,
honest reporting when a burst has no diversity, and local warp on a deformed
burst.

Several real defects were found by that harness rather than by inspection,
including a registration failure that silently returned identity transforms on
small proxies. Others were found by running a second real dataset and by a
side-by-side crop that a metric had missed — a reminder that a measurement only
covers what it was pointed at.

## Not done

* GPU backend. The CPU path is the numerical reference and no
  data structure assumes CPU-only.
* Longitudinal chromatic aberration — the purple fringing that comes from
  channels focusing at different distances. Lateral aberration is corrected;
  this one needs per-channel deconvolution and is not attempted.
* Fitting the optical centre rather than assuming the frame centre. The
  estimator already reports the per-channel centre offset, so this is a small
  step from where it is.
* Decoded-frame caching for reconstruction: `stack --cache` reuses registration
  and defects but still decodes inputs. `analyze` separately caches compact
  measurements and samples, avoiding decode on hits except for filter anchors.
* Iterative forward-model MFSR and explicit PSF estimation.
* X-Trans and already-debayered input are rejected explicitly rather than
  misinterpreted. Bayer and monochrome sensors are supported; narrowband
  channels can share a reconstruction grid with `--split-by-filter`.
* Colour balance beyond matching the channels is not attempted. An SHO composite
  of a hydrogen-rich target comes out green, because it is.
* Deep-sky output is linear and not colorimetric. There is no camera matrix for
  a cooled astronomy sensor behind an arbitrary filter, so the run warns and the
  result needs stretching and colour-balancing elsewhere.
* No calibration frames. Darks, flats and bias are not read, and hot pixels are
  found from the burst instead. A flat would still correct
  vignetting and dust, which nothing here does.
* `--lucky` is implemented and tested but has not been validated on a real
  atmospheric burst, because we do not have one. It is off by default.

## Layout

```
crates/
  sr-core          data model: Plane, RawFrame, samples, CFA, geometry, math
  sr-raw           RAW, FITS and XISF decode, burst validation
  sr-noise         heteroscedastic noise estimation, fixed-pattern defects
  sr-quality       frame quality, point sources, photometric matching
  sr-register      phase correlation, transform fitting, reference selection
  sr-warp          local deformation fields
  sr-reconstruct   coverage, kernels, robustness, lucky regions, the merge
  sr-color         white balance, camera colour matrix, preview stretch
  sr-composite     aligning, scaling and combining stacked channels
  sr-output        TIFF read/write
  sr-diagnostics   manifests, CSV tables, diagnostic rasters
  sr-synth         synthetic ground truth and metrics (PSNR, SSIM, MTF)
  sr-cli           smokstak
docs/                     conventions, diagnostics, analysis and project guides
tools/                    report checkers, benchmarks and export verification
```
