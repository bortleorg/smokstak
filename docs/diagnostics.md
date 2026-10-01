# Reading the diagnostics

`smokstak stack --diagnostics ./diag` writes a directory whose purpose is to let
you answer, after the fact: which frames were used, where the output is
supported by real samples, whether the requested scale was ever justified, and
whether this run would reproduce.

Every raster is written as 16-bit grey, normalised over a percentile range so it
is viewable. Where the absolute values matter, an unnormalised 32-bit float copy
is written alongside as `<name>.f32.tif`.

## Start here

**`run.json`** — the run manifest. Program version and build profile, every
input file with a content hash **in processing order**, the reference frame and
why it was chosen, the complete configuration, the fitted noise model, the
colour transform, per-stage timings, output dimensions, and a warnings list.

If you read one file, read this one. The warnings list is where the program
tells you it did something you might not want:

* `requested 2.00x but the burst's sub-pixel diversity supports about 1.10x`
* `exposure/ISO differs by up to N% across the burst`
* `N% of sensor sites are saturated on average`
* `no usable camera colour matrix; output colour is not colorimetric`
* `reconstructed despite failed burst validation (--force)`

**`frames.csv`** — one row per frame, and the place to look when a result is
worse than expected. Columns: quality (`sharpness`, `contrast`,
`estimated_blur`, `blur_anisotropy`, `saturation_fraction`), point sources
(`star_hfd`, `star_eccentricity`, `star_count`), photometry
(`exposure_scale`, `photometric_gain`, `photometric_offset`,
`photometric_source`), geometry (`transform_model`, `shift_x`, `shift_y`,
`rotation_deg`, `scale`), fit quality (`residual_rms`, `residual_p90`,
`inliers`, `probes`, `overlap`, `confidence`), local warp (`local_warp_mean`,
`local_warp_max`) and `robustness_rejected`.

Geometry and residuals are in **sensor pixels**, converted from the proxy grid
the registration works in.

Three columns are easy to misread:

* `shift_x` and `shift_y` are the displacement of the frame's **centre**, not
  the transform's translation component. On a burst that rotates, the
  translation measured at the origin is a lever arm and reports tens of pixels
  of "motion" for a scene that barely moved.
* `transform_model` describes what the accumulated transform **does**, to within
  a tenth of a sensor pixel over the frame — not which model the last refinement
  step fitted. Those are different questions and the second one is nearly
  useless: registration is iterative and each step fits the residual, so a
  well-converged rotating frame ends on a translation of a few hundredths of a
  pixel.
* `sharpness` is a half-flux diameter, inverted and normalised to the burst
  median, on a burst with point sources; gradient energy, likewise normalised,
  on one without. Larger is sharper either way, one is the burst median, and
  the quality table's header says which was used. `star_count` of zero means
  this frame had none and the two columns beside it say nothing.
* `photometric_gain` and `photometric_offset` are the **green** channel of the
  map that carries this frame onto the reference. `photometric_source` says how
  it was arrived at: `measured` (gain and pedestal both fitted), `level-only`
  (pedestal only, because the scene could not identify a gain or the fitted one
  was implausible), `exposure` (too few usable blocks; the metadata figure was
  used), or `disabled`.

## The maps

| file | what it shows | what to look for |
|---|---|---|
| `reference-preview.tif` | the reference frame, demosaiced, white-balanced, at sensor resolution, cropped to `--roi` when one was given, grey for a monochrome sensor | framing and focus, without waiting for a reconstruction |
| `r/g/b-coverage.tif` | contributing samples per output pixel, per channel; a monochrome reconstruction writes one `coverage.tif` instead | dark patches mean thin support; red and blue are always a quarter of green's density |
| `weight-map.tif` | accumulated kernel weight, green | large-scale variation means uneven frame contribution |
| `effective-frame-count.tif` | `(sum w)^2 / sum w^2` over per-frame totals, on an 8-output-pixel grid | how many frames the merge *behaved* like. Far below the frame count means something is discarding frames |
| `rejection-count.tif` | samples suppressed by the robustness model | should trace moving objects. Broad diffuse rejection means the model is firing on registration error, not motion |
| `motion-mask-summary.tif` | mean robustness across the burst, 0–1 | dark = distrusted. A dark region that is not a moving object is a warning; a *uniformly* half-dark map means the frames disagree photometrically, not spatially |
| `background-model.tif` | the fitted vignetting and sky gradient, relative to each channel's centre | only written with `--flatten-background`. Should look like optics and sky: a smooth radial falloff and a plane. If it looks like your subject, the model is eating it — turn the flattening off |
| `defect-map.tif` | count of masked sensor sites per 8x8 block | only written when defects were found. Should be sparse and unstructured; a bright row or column is a bad column, and a cluster is worth investigating |
| `sampling-phase-map.tif` | effective distinct phases per region | low values mean that region cannot be super-resolved, whatever the global verdict said |
| `local-quality-summary.tif` | per-region sharpness of the first frame | structure here means the burst is a candidate for `--lucky` |
| `warp-magnitude-map.tif` | mean local-warp displacement, sensor px | only written when a local field was accepted |
| `warp-confidence-map.tif` | mean local-warp confidence, 0–1 | dark = the field was inpainted from neighbours there, not measured |
| `registration-residual-map.tif` | 6 columns x one row per frame | columns are `rms`, `p50`, `p90`, `p99`, `confidence`, `overlap`. A strip, not a picture — read it in the CSV instead |
| `sampling-phase.csv` | the sub-pixel phase histogram | for plotting the phase distribution directly |
| `global-registration.csv` | as `frames.csv`, written by `smokstak register` | registration without reconstructing |
| `frame-quality.csv` | per-frame quality vector | quality without registration |

## Common situations

**"The output is soft."** Check `effective-frame-count`: if it is far below the
frame count, frames are being discarded — look at `confidence` in `frames.csv`
and at `motion-mask-summary`. If effective frames is healthy, check the
diversity verdict in `run.json`; a burst with no sub-pixel motion cannot be
super-resolved and the program will have said so. Only then suspect the kernel.

**"Colours look wrong at the edges of the frame."** Chromatic aberration. Run
`smokstak inspect` and read the aberration report; if the corner shift is above
about a third of a pixel it is worth correcting, and correction is on by
default. Residual colour after correction is longitudinal aberration, which is
not geometric and is not corrected.

**"Some frames were rejected."** `robustness_rejected` in `frames.csv` is the
fraction of each frame suppressed. A few percent is normal. A frame at 100% did
not overlap the reference. A burst where *every* frame shows heavy rejection
usually means the reference is unrepresentative rather than that the burst is
bad.

**"It says 1.0x is supported but I have 200 frames."** Frame count does not
create sub-pixel diversity; camera motion does. A burst locked to a rigid tripod
with no drift samples the same phase every time. The verdict in `run.json` and
the `sampling-phase-map` will agree. More frames will still reduce noise.

**"Almost every frame is half-distrusted, everywhere."** Look at
`motion-mask-summary`: if it is uniformly grey rather than dark in places, the
frames disagree about *brightness*, not about content. Check `photometric_gain`
and `photometric_offset` in `frames.csv` — a burst whose illumination drifted
and could not be matched will show it there, and the run prints how far the
brightness varied. This is what an uncorrected night sky looks like: the
background rises through the session and motion rejection reads it as the scene
changing.

**"One frame is much softer than the rest."** Check `star_eccentricity` in
`frames.csv` against the burst median, which the run prints. A soft *and* round
frame is seeing, focus drift or thin cloud — check `star_count`, which drops
when cloud arrives before the stars soften. A soft *and* elongated frame is
mechanical: wind, a guiding error, a snagged cable, or a mount that needs
aligning. The run names those frames and gives the angle they are drawn out
along, which usually identifies the axis at fault.

**"There are short coloured dashes over the image."** Fixed-pattern sensor
defects, dragged across the scene by the merge. Check whether the run printed a
line about them: if it says the burst moved too little to tell a hot pixel from
a point source, the dither was under one sensor pixel and detection declined on
purpose. If it reported some and dashes remain, they are the ones it could not
separate from real content — the miss rate depends on how much the scene itself
persists between frames.

**"The reconstruction is worse than one frame."** Check edge `straightness` in
`smokstak measure` before trusting any MTF figure from it. A single frame
legitimately looks sharper on fine low-contrast texture while being further
from the truth.

## The preview

`--preview` writes `<output>.preview.png`, downsampled to 2000 px on the long
edge. It is a way of looking at a result, never a result: the transfer it
applies is not invertible and destroys the linear relationship everything else
preserves.

The run says which of two things happened. Either it names the shadow clip and
midtone it chose and what the background became, or it says no stretch was
needed. The second is not a failure — it means the robust spread was already
wide enough that a background-targeting transfer would have darkened the image
rather than opened it up, which is the normal answer for anything shot in
daylight.

## Reproducing a run

`run.json` holds every parameter, the seed, and a hash per input file in
processing order. The same inputs and configuration produce the same output
within floating-point tolerance; the merge is deterministic and is tested to be
independent of tile size.

What is *not* captured: the exact dependency versions come from `Cargo.lock`,
and the source revision is recorded only if the build set `SRSTACK_REVISION`.
