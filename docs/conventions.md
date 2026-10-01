# Coordinate systems, units and sign conventions

Almost every non-obvious bug in this codebase has been a coordinate or a sign.
A registration that silently returned identity, a chromatic correction applied
backwards, a diversity measure that read a half-pixel shift as diversity when it
is a whole output pixel — each was a unit error, not an algorithm error, and
each was found by a test rather than by reading the code.

This is the single page to read before touching geometry.

## The four grids

| grid | one pixel is | where it comes from |
|---|---|---|
| **sensor** | one photosite | the decoded RAW, after cropping to the active area |
| **guide / proxy** | 2 x 2 photosites | `RawFrame::guide_rgb`, one RGB triple per mosaic cell |
| **output** | `1 / scale` photosites | the reconstruction, `scale` times finer than the sensor |
| **pyramid level `L`** | `2^L` guide pixels | `RegistrationImage::level(L)` |

Conversions, all of which appear in the code:

```text
guide  -> sensor      multiply by 2
sensor -> guide       multiply by 0.5
sensor -> output      o = (r + 0.5) * scale - 0.5
output -> sensor      r = (o + 0.5) / scale - 0.5
level L -> level 0    multiply by 2^L
```

The `+0.5 / -0.5` in the sensor-to-output conversion is not decoration. Both
grids are **pixel-centre based**: integer `(x, y)` addresses the centre of a
pixel, not its corner. Dropping the half-pixel puts the output grid off by half
an output pixel, which is exactly the kind of error that still produces a
plausible image.

`GlobalTransform::rescale(factor)` moves a transform between grids. It scales
only the translation, because the linear part of an affine is
scale-invariant — a rotation is the same rotation whatever the pixel pitch.

## Which way transforms point

**A frame's `WarpField` maps that frame's own sensor coordinates into reference
coordinates.** Forward, always. The merge only ever pushes samples forward, so
it never needs an inverse.

```rust
let (ref_x, ref_y) = warp.map(sensor_x, sensor_y);
```

`WarpField::inverse_map` exists for *gathering* — pulling a patch out of a
target frame to compare against the reference. It is exact for the global part
and first-order for the local field. Never use it to deposit samples.

`GlobalTransform::compose`: `a.compose(&b)` applies `b` first, then `a`. A
registration update is an increment in reference coordinates, so it composes on
the **left**: `t = update.compose(&t)`.

## Phase correlation

`Correlator::shift(reference, target)` returns `(dx, dy)` such that

> a point at `p` in the **target** corresponds to `p + (dx, dy)` in the
> **reference**.

So if the target's content has moved by `+3` px, the returned `dx` is `-3`. This
is fixed by `recovers_integer_shift_with_documented_sign`; if you change the
cross-power spectrum, that test is what tells you which way round you ended up.

Two properties worth knowing before trusting a result:

* Accuracy is sub-0.02 px only for **small residual** shifts. A large
  uncorrected shift is biased by a few percent, because the analysis window is
  fixed while the content moves through it. The registration loop always
  re-extracts the target through the current estimate, so it stays in the good
  regime; nothing else should rely on a single large-shift measurement.
* The whitening is **gated**. Full whitening gives every frequency bin equal
  authority, including bins holding nothing but leakage, which destroys the
  estimate on narrowband content. Bins below a fraction of the peak
  cross-magnitude are dropped and the top of the band is rolled off.

## The mosaic

A monochrome sensor is represented as a mosaic whose four positions carry the
same colour, so that every piece of 2x2 geometry below still applies. The number
of channels is a separate question, asked through `RawFrame::channels()`, and
`channel_at(x, y)` — not `color_at` — is what gives an output channel index. A
monochrome frame's sites are labelled green so the geometry works and all go to
channel zero, because there is only one. Using `color_at` as a channel index is
correct for a mosaic and silently wrong for mono, which is how several stages
were found to be failing.

`CfaPattern::codes` is in raster order within the 2 x 2 cell: `(0,0)`, `(1,0)`,
`(0,1)`, `(1,1)`. `Levels::cell(x, y)` is `(y & 1) * 2 + (x & 1)` and indexes
both the pattern and the per-cell black and white levels.

Cropping shifts the mosaic phase. `sr-raw` forces the crop origin even so the
cropped array's phase is a clean shift of the sensor's, and applies the same
shift to the black levels.

For an RGGB sensor, relative to the green sample positions:

```text
red   at -0.5, -0.5 sensor px      (top-left of the cell)
green at  0.0,  0.0                (the two greens average to the cell centre)
blue  at +0.5, +0.5                (bottom-right)
```

That offset is **geometry, not aberration**. The merge handles it exactly by
depositing every sample at its own true sensor coordinate, so it must be removed
before attributing a channel offset to the lens — see `chroma::bayer_lattice_offset`.

## Where the frame's own array comes from

Camera RAW arrives in storage order and is cropped to the active area, with the
crop origin forced even so the mosaic phase of the cropped array is a clean shift
of the sensor's.

FITS has one extra degree of freedom: the standard puts the origin at the bottom
left, and capture programs mostly write readout order without saying so.
`--fits-row-order auto` believes `ROWORDER` and, without it, takes the array as
stored — the same convention every other reader here uses. The setting decides
one thing only, whether the result comes out mirrored top to bottom, because the
mosaic pattern is checked against the pixels afterwards either way. A frame whose
`BAYERPAT` disagrees with its own green diagonal is read with the pattern shifted
by a row, and the run says so once.

## Units in reported numbers

The same quantity is measured in different grids in different places, so
everything user-facing is converted to sensor pixels first.

* `GlobalRegistration` fields are in **proxy** pixels. The CLI multiplies by 2
  before printing or writing them to `frames.csv`.
* `WarpConfig` distances are in **proxy** pixels.
* `KernelConfig::radius` is in **output** pixels; the kernel variances
  `k_detail` and `k_denoise` are in **sensor** pixels squared. They are
  different grids on purpose: the radius bounds a loop over output pixels, and
  the kernel shape is a property of the optics.
* `GlobalRegistration::centre_shift`, and the `shift_x`/`shift_y` columns it
  feeds, are the displacement of the frame's **centre**, not the transform's
  translation component. Those differ by the rotation's lever arm: half a degree
  about the centre of a 4144 x 2822 frame moves the scene not at all and gives a
  translation at the origin of 28 pixels. `GlobalTransform::shift` still returns
  the raw translation, and is the wrong thing to report.
* `MTF50` from `smokstak measure` is printed in both cycles per output pixel and
  cycles per sensor pixel. **Only the sensor-referred figure is comparable
  across scales.** Upscaling an image lowers its cycles-per-output-pixel while
  resolving nothing, so quoting that number flatters every upscaler ever
  written.

## Sub-pixel phase

For an output scale `s`, a sensor shift of `1/s` is one whole output pixel and
therefore adds **no** diversity. At 2x, shifts of 0.0 and 0.5 sensor pixels are
the same phase.

Diversity is measured as the largest gap between frame phases around the circle,
not by binning: binning and clustering measures both fail on bursts whose
phases bunch together.

## Rules of thumb

* Converting between grids is where the bugs are. If a number looks wrong by a
  factor of 2, 4, or 0.5, look here first.
* Radial quantities — chromatic aberration especially — are radial **about the
  optical axis**. On a crop that axis is usually outside the image, so a
  measurement that assumes the image centre is meaningless. `smokstak measure`
  takes `--ca-centre` for exactly this reason.
* When a convention is not obvious from the code, pin it with a test that would
  fail if the sign flipped. Several already exist and all of them earned their
  place.
* A quantity named after a transform is not the same as what the transform does.
  `TransformModel` on a `GlobalRegistration` is the *effective* model — the
  simplest one that reproduces the accumulated transform over the frame to
  within a tenth of a sensor pixel — because the model that was last fitted is a
  statement about the final residual and not about the geometry.
