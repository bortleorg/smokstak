# Persistent monochrome projects

`smokstak project` keeps a reproducible set of calibrated monochrome FITS/XISF
exposures and publishes a new master for each successful build. It supports the
same directory, file and text-list discovery as the other CLI commands.

```powershell
smokstak project init ./my-target ./approved-frames.txt --reference-file ./lights/reference.fit
smokstak project build ./my-target

smokstak project add ./my-target ./new-night
smokstak project build ./my-target
smokstak project status ./my-target
```

The initial recipe uses production reconstruction at **1x**, with no postprocess.
Use `init --recipe recipe.json` to supply a full serialized
`ReconstructionConfig`; the saved revision contains the complete effective recipe.
The named reference fixes the grid. Filters are stacked separately. Inputs must
have matching sensor dimensions and filter metadata. Calibration remains the
caller's responsibility: hashing records exactly which calibrated files were
used, but does not verify their calibration.

## Updates and review

The GUI's **Saved mono projects** panel runs these same commands. Choose a
project folder and an action. Create requires an original reference frame;
add/relink takes a source directory or list; exclude/restore takes a saved review
list, with a reason required for exclusion. Mutations are followed by a build.
**Show project / latest result** displays status and the most recent successfully
published masters and frame review without rebuilding. The GUI never presents
pending project outputs as completed masters.

An update rebuilds **all selected exposures together** through the production
pipeline. Registration, per-filter noise, photometry, defects, rejection and
weights are reconsidered for the combined population. It does not combine old
finished masters or retain obsolete batch decisions. This first implementation
prioritizes correctness over the speed of partial updates.

Full SHA256 file identities deduplicate renamed copies. Input order is canonical
by identity, independent of when files were added. Sources and the reference are
verified before and after a build. Modified source content fails the build;
recalibrated inputs should be ingested as new files, with obsolete versions
explicitly excluded.

Review lists accept one original path or full content identity per line. Relative
paths are resolved against the list's directory. The questionable-frame list
exported by the analysis report can be used directly:

```powershell
smokstak project exclude ./my-target ./questionable-frames.txt --reason "Inspected: cloud or tracking damage"
smokstak project build ./my-target
smokstak project include ./my-target ./questionable-frames.txt
```

Exclusion is a reversible, versioned decision. A rise in the sampled noise curve
does **not** automatically exclude a frame. The existing production weighting and
pixel rejection still operate on selected inputs. Excluding the geometry reference
keeps its grid, without contributing its exposure to the reconstruction.

If originals move, `project relink ./my-target ./new-location` matches full content
hashes and records their new locations while preserving review decisions and
reference identity. Unknown content is rejected; use `add` for new exposures.

### Visual build audit

Every new project build writes `frame-review.html` beside its masters. Open it
locally, keeping its `review-assets` folder beside it. It includes every project
member and distinguishes explicit review exclusions, production registration
failures/zero frame factors, and inputs eligible for weighted integration.
Positive weight does not guarantee contribution at every pixel.

Click a point or row to inspect its reason and compare it with another exposure
through the same filter. Graphs show frame factors, stellar HFD/eccentricity,
registration residual and guide suppression fractions. These are production
measurements; guide suppression is not a count of discarded exposures.

The review defaults to acquisition order from the recorded capture times; unknown
times sort last. Plot spacing is frame position, not elapsed time. Sorting the
view never changes the build's canonical input order or reconstruction. Search
paths, capture dates or decision reasons, then step through matching frames with
Previous/Next frame. The default comparison is the nearest used same-filter
exposure in capture time, preferring the same ISO/camera gain setting when known,
with input position as a fallback when times are absent. The ISO/gain selector
helps inspect mixed-setting sessions separately. This is the header setting, not
electrons per ADU; production still estimates one noise model per filter, so the
audit does not establish optimal weighting across gain settings.
Known exposure totals describe the current review selection and keep missing
durations explicit; they are not effective integration or an SNR estimate.

**Save current review selection** exports all matching pages in the displayed
order. The existing used/not-used/excluded exports still cover the whole build.
Export and clipboard paths omit Windows' extended-path prefix for compatibility
with astronomy applications, preserving network server/share paths.
Capture time and exposure metadata are optional, so old audit JSON remains readable.

The inspector includes a sparse whole-frame overview, a 96×96 native sensor crop,
and a registered crop at sensor pitch. A star near the reference field/ROI centre
is preferred as the shared location. Native crops preserve the decoded samples
and orientation; registered crops use bilinear resampling. Both previews use the
same raw-level stretch, with adjustable display strength and blinking. An
excluded or unreliably registered frame uses an explicitly unaligned sensor-centre
crop. Missing/modified excluded originals retain their reasons with no invented
preview or quality measurements.

Used/not-used/excluded path lists can be saved for inspection in astronomy
software. The viewer does not alter project decisions. JSON evidence and lazy
preview assets are included in the build's artifact hashes. Assets load for the
selected pair, with a small browser cache, so the page does not load every crop
for a thousand-frame project at once. Ordinary mono and OSC `stack` runs with
diagnostics also write a per-filter audit; they cannot list inputs absent from
their input list. Earlier completed builds are not modified retroactively.

OSC audits label native crops as raw Bayer mosaics. Registered crops and sparse
overviews use one green CFA phase; interpolation never mixes Bayer colours.
These grayscale inspection views do not replace colour assessment of the RGB
master. The ordinary stacker accepts `--scratch-dir PATH` for the same disk-backed
buffer storage used by projects. Persistent project ingestion remains mono-only.

Visual-audit validation: native/registered crop coordinate and sample tests,
HTML escaping, complete membership/status/asset checks in the production project
integration test, and `node tools/check_frame_review.cjs path/to/frame-review.html`
for lazy assets and viewer controls. The Node check is a DOM simulation, not a
browser rendering test. The synthetic example under `out/frame-review-demo`
contains a failed-registration frame and an explicit tracking-smear exclusion.

## Outputs and failures

Projects contain:

- `revisions/00000001.json`, etc.: immutable membership, recipe, reference and review snapshots.
- `runs/00000001/`, etc.: preserved float TIFF/FITS masters, preview, production diagnostics,
  exact selected input list, revision snapshot and `project-build.json` with artifact hashes.
- `cache/`: registration and defect caches scoped to the executable and revision.

Outputs are built in a new `.pending` directory and published only after the stack
succeeds and sources pass a second identity check. Failures preserve previous
masters and leave partial diagnostics in the pending directory. A process crash
can leave `project.lock`; verify no project command is running before removing
that lock. Never modify committed revisions or run directories in place.

## Large projects

Builds default to two workers and disk-backed decoded samples, guide images,
registration pyramids and robustness maps. Original input files are never mapped
for mutation. Scratch files are private, preserve sample bits, and are released
when their last user finishes. Defect scanning uses spatial tiles and shared masks.

This removes the requirement to retain every full image on the heap. It is **not
a hard RAM limit**: the operating system controls mapped-page residency, and
active processing still needs image and output buffers. Fast local scratch and
ample free disk space matter. Scratch usage grows with total decoded data and
can exceed source size for compressed inputs. `--workers` controls concurrent
work; `--in-memory` selects the resident comparison path; `--no-cache` disables
cache reuse. Keep workers fixed when comparing reproducibility.

## Validation and current limits

The end-to-end test creates actual FITS exposures, builds A, adds B, and compares
against a fresh A+B project, including scientific image bytes and processing
decisions. It also compares resident versus disk-backed execution, cold versus
warm caches, multiple filters and an excluded geometry reference. Unit tests
cover exact spill values, copy-on-write isolation, cleanup, tile boundaries,
map restoration and failure/publication guards.

These are synthetic correctness tests, not a 1000-frame/62 MP performance claim.
The equivalence test requires byte-identical TIFF/FITS output at the same worker
count, not merely a visually similar preview. Run them with:

```powershell
cargo test -p sr-cli --test project --release --offline
cargo test -p sr-cli --bin smokstak --offline
cargo test --workspace --lib --offline
cargo clippy --workspace --all-targets --offline -- -D warnings
```

The sampled `analyze` report remains a separate diagnostic. Automatic quality
optimization, held-out validation, comparisons measured on successive full
masters, calibrated collection forecasts, hard memory budgets and selective
contribution reuse are not implemented here.
