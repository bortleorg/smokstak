# Mono mosaics and mixed instruments

`smokstak mosaic` reconstructs one canvas from monochrome FITS exposures whose
footprints and optical scales differ: panels of a mosaic, or the same target
through two telescopes. It is an opt-in, experimental path; ordinary `stack`
commands and defaults do not use it.

## Supported input

- 2–512 uncompressed, primary-image, 16-bit integer mono FITS exposures.
- At most 120 distinct pointings.
- Every exposure carries the same filter label.

Floating-point or already-calibrated FITS, OSC/CFA data and absolute WCS output
are outside this path. No dark, flat or bias calibration is applied, and the
original files are never modified.

## Command line

```bash
smokstak mosaic prepare EXPOSURES --output NEW_PREPARATION_DIRECTORY
smokstak mosaic build NEW_PREPARATION_DIRECTORY/plan.json --output NEW_RESULT_DIRECTORY --experimental
```

`EXPOSURES` is a folder (panel subfolders included) or a text list of paths.
`prepare` takes `--memory-mb` (default 2048), `--cache-dir` for a reusable,
source-fingerprinted star cache, and `--stage-dir` to copy every source to local
disk first (see below). `build` takes `--tile` (default 256) and
`--memory-mb` (default 512); `--plan-sha256` binds the build to one reviewed plan.

Both commands refuse an existing output directory, work in a staging directory
and publish only on success. A failure writes `FAILED.txt` with the reason. A
process that is killed can leave staging files behind; they are never a result.

### Frames on a network drive

Both commands read many small windows of each source. Over a network share every
request is a round trip, so they read in bulk where they can: preparation keeps
full-width bands of rows, and the build keeps a cache of full-width bands within
its memory budget, reused along each row of output tiles. A tile that would need
more bands than the cache can hold reads its windows directly instead. Output is
identical either way.

For the fastest runs, `--stage-dir DIR` (or **Copy frames to this computer
first** on the page) copies each source into `DIR` as `<sha256>.fits` during the
fingerprinting pass that already reads every byte. Preparation and the build then
use the local copies; labels keep the original file names, and a copy already
present is verified, not trusted by name. It needs free space for every source.

## The page

In `smokstak gui`, choose **Mosaics & mixed scopes**, pick the exposures and a new
result folder, and click **Find panels and prepare mosaic**. Preparation runs as a
cancellable child process with per-frame progress, and the page reattaches to it
after a refresh. From the ordinary stacking flow, **Make a mosaic from these
frames** carries over the loaded input and any explicit exclusions.

When preparation finishes, the detected layout opens: projected footprints, the
output canvas, panel and optical-group filters, and per-exposure registration
residuals, validation-star counts and stellar HFD. The filters affect the review
only; they never drop exposures from the build. **Build experimental mosaic** runs
the same engine as the CLI, bound to the exact plan that was reviewed.

A previously prepared `plan.json` can also be imported and reviewed.

## What preparation does

No panel numbering or reference exposure has to be chosen.

1. **Stars.** Detected in bounded native-resolution windows, so memory does not
   grow with sensor size.
2. **Layout.** Repeated pointings are grouped into panels, and exposures are
   grouped by optical train from their measured scale.
3. **Geometry.** A projective transform per exposure and a shared relative optical
   model per optical group, fitted over the connected overlap graph. Each
   exposure's stars are split into training and withheld validation stars; it
   must pass at least 30 validation stars with median residual at most 0.75 and
   p90 at most 1.5 reference pixels. These are relative centroid checks, not
   absolute astrometry.
4. **Stellar flux.** Native aperture photometry with a common angular aperture,
   corrected for local projected pixel area. Two relative models are fitted:
   - one gain per exposure, from the median log-flux ratio of each overlap;
   - a smooth relative response per exposure: that gain times the exponential of a
     quadratic across the field, fitted to individual matched stars. This absorbs
     vignetting that differs between exposures or instruments.

   Every overlap withholds a fifth of its matched stars; they never train the
   fit. Withheld stars with SNR ≥ 50 in both exposures are used, because fainter
   ones mostly test their own aperture measurement; an overlap with fewer than
   five of those is topped up with its next brightest withheld stars. A model is
   applicable only if withheld stars agree to a median of 0.05 and a p90 of 0.20
   in log flux: per overlap where it has at least ten withheld stars, and pooled
   across the sparser overlaps otherwise.

   The spatial response is fitted in coordinates centred on each footprint, with
   a weak prior (about ten percent across a frame) that only decides what no
   overlap measures, such as a shape shared by every frame of one instrument
   outside the other's field. It must stay within 0.25–4× over each footprint; the
   build weights every sample by its own corrected noise, so a strongly boosted
   corner counts for correspondingly less. It replaces the single gain only when
   it lowers the pooled withheld median by at least 5%. `plan.json` notes record
   both candidates.
5. **Background.** Additive offset-and-plane corrections fitted only to
   differences between overlapping patches of the same sky, validated on an
   independent checkerboard of patches: an overlap's withheld p90 may not grow
   by more than 15%, unless it stays within the p90 that sampling noise alone
   produces (from each exposure's measured sky noise and patch size). The anchor exposure keeps its full sky;
   nothing flattens a single image.

Any exposure that cannot be placed or photometrically tied to the rest stops
preparation with an explanation. Nothing is silently dropped.

## What building does

The canvas is independent of any single input. Source windows and output tiles
bound the image buffers; `--memory-mb` limits estimated application buffers and is
not an operating-system working-set limit. `tile / scale` must be an even integer.

Exposures are grouped by common-scale stellar HFD (10% tolerance). Each group
computes its own rejection consensus, and the finest group with native coverage
owns each location, so wide stars from one instrument do not blur fine ones from
another. The cost is a smaller noise benefit where instruments overlap and
possible resolution transitions at group boundaries. Weights taper over 10% of the
true detector edge.

The result directory contains:

- `image.fits`: normalized mono samples; uncovered pixels are NaN.
- `weight.fits`: accumulated reconstruction weights, not inverse variance.
- `samples.fits`: deposited sample count, not independent exposures.
- `preview-linear.fits` and `preview.png`: reduced display previews; the PNG is
  asinh-stretched for viewing only.
- `plan.json` and `result.json`: inputs, geometry, policies and execution record.

## Limitations

- Relative photometry only. The result is not photometrically calibrated, and
  corrections outside measured overlaps are extrapolation.
- The spatial response is relative to the anchor exposure; the anchor's own
  vignetting remains. A flat field is still the right correction for that.
- A plan that passes every check can still contain a poor individual overlap.
  Inspect the notes and the seams in the result; an attractive stretch is not
  evidence that faint nebulosity was preserved.
