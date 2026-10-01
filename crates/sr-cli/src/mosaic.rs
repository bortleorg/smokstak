//! Explicit mono mosaic build from a validated, reproducible geometry plan.
//! Original samples are read in windows and passed to the production estimator.
#[cfg(test)]
#[path = "mosaic_tests.rs"]
mod tests;
use crate::mosaic_source::PreparedSource;
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sr_core::geometry::RadialChroma;
use sr_core::projection::FrameProjection;
use sr_core::{
    CfaPattern, DefectMask, FrameMetadata, NoiseModel, NoiseSource, Plane, RawFrame,
    ReconstructionConfig, Rect, SamplePlane, WarpField,
};
use sr_quality::photometry::PhotometricMatch;
use sr_reconstruct::{KernelField, MergeInputs};
use std::{
    collections::HashSet,
    fs,
    io::{BufWriter, Read},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const PSF_RELATIVE_TOLERANCE: f32 = 0.1;
const DETECTOR_EDGE_FEATHER: f32 = 0.1;
// Full 240-frame evaluation plans are below 300 KiB. Bound both encoded input
// and decoded metadata before source I/O; this remains comfortably below the
// smallest supported 64 MiB reconstruction budget.
const MAX_PLAN_BYTES: usize = 2 * 1024 * 1024;
const MAX_PLAN_METADATA_BYTES: usize = 1024 * 1024;
const MAX_PLAN_STRING_BYTES: usize = 64 * 1024;
const MAX_PLAN_NOTES: usize = 1024;

fn mosaic_reconstruction_config(scale: f32) -> ReconstructionConfig {
    ReconstructionConfig {
        scale,
        roi: None,
        ..ReconstructionConfig::default()
    }
}

/// Freeze minimum-based PSF cohorts over the whole validated plan. Passing the
/// representatives to tile-local selection preserves membership when a cohort's
/// actual sharpest frame does not overlap that tile. Neighboring representatives
/// exceed the same f32 threshold used by the reconstruction cohort builder.
pub(crate) fn global_psf_cohort_representatives(hfd: &[f32]) -> Vec<f32> {
    let mut ordered: Vec<_> = (0..hfd.len()).collect();
    ordered.sort_by(|&a, &b| hfd[a].total_cmp(&hfd[b]).then(a.cmp(&b)));
    let mut representatives = vec![0.0; hfd.len()];
    let mut minimum = None;
    for index in ordered {
        if minimum.is_none_or(|value| hfd[index] > value * (1.0 + PSF_RELATIVE_TOLERANCE)) {
            minimum = Some(hfd[index]);
        }
        representatives[index] = minimum.expect("nonempty current cohort");
    }
    representatives
}

#[derive(Parser, Debug)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand, Debug)]
enum Command {
    /// Automatically register mono exposures and prepare a reviewable mosaic plan.
    Prepare {
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        memory_mb: usize,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Copy every source into this local folder first (verified by SHA-256)
        /// and prepare and build from the copies. Worth it for sources on a
        /// network share; needs free space for all of them.
        #[arg(long)]
        stage_dir: Option<PathBuf>,
    },
    /// Build a mono mosaic from an independently validated geometry plan.
    Build {
        plan: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 256)]
        tile: usize,
        #[arg(long, default_value_t = 512)]
        memory_mb: usize,
        /// Explicit geometry/photometry experiment; result is not a certified master.
        #[arg(long)]
        experimental: bool,
        /// Require the exact prepared plan reviewed by the GUI.
        #[arg(long)]
        plan_sha256: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grid {
    pub origin: [f32; 2],
    pub width: usize,
    pub height: usize,
    pub scale: f32,
}
/// Largest factor a relative stellar response may apply, up or down, anywhere
/// in a source footprint. Uncalibrated instruments measured here differ by up
/// to about 2.5x at the corners. The merge weights every sample by its own
/// corrected variance, so a boosted corner counts for correspondingly less.
pub(crate) const MAX_RELATIVE_RESPONSE: f64 = 4.;
pub(crate) const RESPONSE_RANGE_TEXT: &str = "0.25–4";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelativeLogGain {
    pub center: [f64; 2],
    pub normalization_scale: f64,
    /// x, y, x², xy, y² in normalized common reference coordinates.
    pub coefficients: [f64; 5],
}

impl RelativeLogGain {
    pub(crate) fn factor_at(&self, x: f64, y: f64) -> f64 {
        let u = (x - self.center[0]) / self.normalization_scale;
        let v = (y - self.center[1]) / self.normalization_scale;
        let [a, b, c, d, e] = self.coefficients;
        (a * u + b * v + c * u * u + d * u * v + e * v * v).exp()
    }

    /// Conservative interval bound, including polynomial extrema inside a box.
    pub(crate) fn log_bounds(&self, bounds: [f64; 4]) -> Result<[f64; 2]> {
        ensure!(
            self.center
                .iter()
                .chain(self.coefficients.iter())
                .all(|v| v.is_finite())
                && self.normalization_scale.is_finite()
                && self.normalization_scale > 0.,
            "Invalid relative stellar-response model"
        );
        ensure!(
            bounds.iter().all(|v| v.is_finite())
                && bounds[0] <= bounds[2]
                && bounds[1] <= bounds[3],
            "Invalid stellar-response footprint"
        );
        let u = [
            (bounds[0] - self.center[0]) / self.normalization_scale,
            (bounds[2] - self.center[0]) / self.normalization_scale,
        ];
        let v = [
            (bounds[1] - self.center[1]) / self.normalization_scale,
            (bounds[3] - self.center[1]) / self.normalization_scale,
        ];
        let square = |r: [f64; 2]| {
            [
                if r[0] <= 0. && r[1] >= 0. {
                    0.
                } else {
                    r[0].powi(2).min(r[1].powi(2))
                },
                r[0].powi(2).max(r[1].powi(2)),
            ]
        };
        let products = [u[0] * v[0], u[0] * v[1], u[1] * v[0], u[1] * v[1]];
        let uv = [
            products.iter().copied().fold(f64::INFINITY, f64::min),
            products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ];
        let mut result = [0.; 2];
        for (coefficient, range) in
            self.coefficients
                .into_iter()
                .zip([u, v, square(u), uv, square(v)])
        {
            let a = coefficient * range[0];
            let b = coefficient * range[1];
            result[0] += a.min(b);
            result[1] += a.max(b);
        }
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "Relative stellar-response model overflows"
        );
        Ok(result)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameSpec {
    pub path: PathBuf,
    /// Display provenance only; neither label changes reconstruction membership.
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub group: String,
    pub width: usize,
    pub height: usize,
    pub bytes: u64,
    pub sha256: String,
    pub projection: FrameProjection,
    pub noise: NoiseModel,
    pub sky: f32,
    pub gain: f32,
    /// Relative stellar response only; preserves the anchor frame's response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative_log_gain: Option<RelativeLogGain>,
    pub offset: f32,
    /// Additive slopes per global reference pixel; offset is the value at (0,0).
    #[serde(default)]
    pub background_plane: [f32; 2],
    /// Optional additive coefficients of global x², xy and y².
    #[serde(default)]
    pub background_quadratic: [f32; 3],
    pub weight: f32,
    /// Measured bright-star HFD in the common reference-pixel scale.
    pub psf_hfd: f32,
    pub registration_p50: f32,
    pub registration_p90: f32,
    pub validation_stars: usize,
}

impl FrameSpec {
    /// Scalar gain times the relative stellar response, in common coordinates.
    pub(crate) fn matched_gain(&self, x: f64, y: f64) -> f64 {
        f64::from(self.gain)
            * self
                .relative_log_gain
                .as_ref()
                .map_or(1., |g| g.factor_at(x, y))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub version: u32,
    pub filter: String,
    pub calibration: String,
    pub grid: Grid,
    pub frames: Vec<FrameSpec>,
    pub notes: Vec<String>,
}

fn plan_memory_bytes(plan: &Plan) -> Result<usize> {
    ensure!(
        (2..=512).contains(&plan.frames.len()),
        "mosaic requires 2..512 frames"
    );
    ensure!(
        plan.notes.len() <= MAX_PLAN_NOTES,
        "mosaic plan has too many notes"
    );
    let mut allocated = std::mem::size_of::<Plan>()
        .checked_add(
            plan.frames
                .capacity()
                .checked_mul(std::mem::size_of::<FrameSpec>())
                .context("plan allocation overflow")?,
        )
        .and_then(|n| {
            n.checked_add(
                plan.notes
                    .capacity()
                    .checked_mul(std::mem::size_of::<String>())?,
            )
        })
        .context("plan allocation overflow")?;
    let mut metadata = 0usize;
    let mut include = |len: usize, capacity: usize| -> Result<()> {
        ensure!(
            len <= MAX_PLAN_STRING_BYTES,
            "mosaic plan metadata string exceeds 64 KiB"
        );
        metadata = metadata
            .checked_add(len)
            .context("plan metadata overflow")?;
        allocated = allocated
            .checked_add(capacity)
            .context("plan allocation overflow")?;
        Ok(())
    };
    include(plan.filter.len(), plan.filter.capacity())?;
    include(plan.calibration.len(), plan.calibration.capacity())?;
    for note in &plan.notes {
        include(note.len(), note.capacity())?;
    }
    for frame in &plan.frames {
        include(frame.path.as_os_str().len(), frame.path.capacity())?;
        include(frame.sha256.len(), frame.sha256.capacity())?;
        include(frame.label.len(), frame.label.capacity())?;
        include(frame.group.len(), frame.group.capacity())?;
    }
    ensure!(
        metadata <= MAX_PLAN_METADATA_BYTES,
        "mosaic plan metadata exceeds 1 MiB"
    );
    ensure!(
        allocated <= MAX_PLAN_BYTES,
        "decoded mosaic plan allocation exceeds 2 MiB"
    );
    Ok(allocated)
}

#[cfg(test)]
fn read_plan(path: &Path) -> Result<Plan> {
    Ok(read_plan_fingerprinted(path)?.0)
}

fn read_plan_fingerprinted(path: &Path) -> Result<(Plan, String)> {
    let input = fs::File::open(path).with_context(|| format!("opening plan {}", path.display()))?;
    ensure!(
        input.metadata()?.len() <= MAX_PLAN_BYTES as u64,
        "encoded mosaic plan exceeds 2 MiB"
    );
    let mut bytes = Vec::new();
    // The extra byte detects growth between the metadata check and the read;
    // Take also bounds special streams and concurrent producers independently.
    input
        .take(MAX_PLAN_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_PLAN_BYTES,
        "encoded mosaic plan exceeds 2 MiB"
    );
    let plan: Plan = serde_json::from_slice(&bytes)?;
    plan_memory_bytes(&plan)?;
    Ok((plan, format!("{:x}", Sha256::digest(&bytes))))
}

fn validate_geometry(plan: &Plan, tile: usize, memory_mb: usize) -> Result<()> {
    plan_memory_bytes(plan)?;
    ensure!(plan.version == 1, "unsupported mosaic plan version");
    ensure!(!plan.filter.trim().is_empty(), "filter must be explicit");
    ensure!(
        (64..=1024).contains(&tile) && tile.is_multiple_of(2),
        "tile must be even and in 64..1024"
    );
    ensure!(
        (64..=65536).contains(&memory_mb),
        "memory budget must be in 64..65536 MiB"
    );
    let g = &plan.grid;
    ensure!(
        g.width > 0 && g.height > 0 && g.width < 1 << 23 && g.height < 1 << 23,
        "invalid output dimensions"
    );
    ensure!(
        g.origin.iter().all(|v| v.is_finite()) && (1.0..=8.0).contains(&g.scale),
        "invalid output origin or scale"
    );
    // Rejection and PSF ownership use a two-reference-pixel guide lattice.
    // Compute from the exact represented f32 scale in f64: f32 division could
    // round a near-even quotient to an integer and hide a changing guide phase.
    let reference_tile_step = tile as f64 / f64::from(g.scale);
    ensure!(
        reference_tile_step % 2.0 == 0.0,
        "tile / output scale must be an even integer to preserve the reference guide lattice"
    );
    ensure!(
        g.width
            .checked_mul(g.height)
            .and_then(|n| n.checked_mul(12))
            .is_some(),
        "output byte count overflow"
    );
    for f in &plan.frames {
        f.projection.validate()?;
        let maximum_gain = if let Some(model) = &f.relative_log_gain {
            let bounds =
                crate::mosaic_prepare_photometry::bounds(&f.projection, f.width, f.height)?;
            let [lo, hi] = model.log_bounds(bounds)?;
            ensure!(
                lo >= -MAX_RELATIVE_RESPONSE.ln() && hi <= MAX_RELATIVE_RESPONSE.ln(),
                "Relative stellar-response correction exceeds {RESPONSE_RANGE_TEXT} over source footprint: {}",
                f.path.display()
            );
            f.gain * hi.exp() as f32
        } else {
            f.gain
        };
        ensure!(
            f.width > 0
                && f.height > 0
                && f.gain.is_finite()
                && f.gain > 0.
                && f.offset.is_finite()
                && f.background_plane.iter().all(|v| v.is_finite())
                && f.background_quadratic.iter().all(|v| v.is_finite())
                && f.weight.is_finite()
                && f.weight > 0.
                && f.psf_hfd.is_finite()
                && f.psf_hfd > 0.
                && f.sky.is_finite(),
            "invalid frame photometry: {}",
            f.path.display()
        );
        // Check the model itself rather than NoiseModel::variance's defensive
        // floor. Integer sources use normalized [0,1] values; include a supplied
        // higher sky estimate and its photometric gain in the finite-range gate.
        ensure!(
            f.noise.alpha.is_finite()
                && f.noise.alpha >= 0.
                && f.noise.beta.is_finite()
                && f.noise.beta >= 0.
                && [0., 1., f.sky.max(0.)].iter().all(|&level| {
                    let variance = f.noise.alpha * level + f.noise.beta;
                    let matched_variance = variance * maximum_gain * maximum_gain;
                    variance.is_finite()
                        && variance > 0.
                        && matched_variance.is_finite()
                        && matched_variance > 0.
                }),
            "invalid frame noise model: {}",
            f.path.display()
        );
        ensure!(
            f.validation_stars >= 30
                && f.registration_p50.is_finite()
                && f.registration_p50 >= 0.
                && f.registration_p50 <= 0.75
                && f.registration_p90.is_finite()
                && f.registration_p90 >= f.registration_p50
                && f.registration_p90 <= 1.5,
            "registration gate failed: {}",
            f.path.display()
        );
    }
    Ok(())
}

/// Review the same schema and geometry gates without reading entire sources.
/// The build still verifies source sizes, hashes and filters before publication.
pub(crate) fn read_review_plan_fingerprinted(
    path: &Path,
    tile: usize,
    memory_mb: usize,
) -> Result<(Plan, String)> {
    let (plan, digest) = read_plan_fingerprinted(path)?;
    validate_geometry(&plan, tile, memory_mb)?;
    Ok((plan, digest))
}

fn validate(plan: &Plan, tile: usize, memory_mb: usize) -> Result<()> {
    validate_geometry(plan, tile, memory_mb)?;
    let mut source_paths = HashSet::new();
    let mut source_fingerprints = HashSet::new();
    for f in &plan.frames {
        let canonical = fs::canonicalize(&f.path)
            .with_context(|| format!("resolving source {}", f.path.display()))?;
        #[cfg(windows)]
        let key = canonical.to_string_lossy().to_lowercase();
        #[cfg(not(windows))]
        let key = canonical;
        ensure!(
            source_paths.insert(key),
            "duplicate source file in mosaic plan: {}",
            f.path.display()
        );
        ensure!(
            fs::metadata(&f.path)?.len() == f.bytes,
            "source size changed: {}",
            f.path.display()
        );
        ensure!(
            f.sha256.len() == 64 && source_hash(&f.path)? == f.sha256,
            "source fingerprint differs from plan: {}",
            f.path.display()
        );
        ensure!(
            source_fingerprints.insert(&f.sha256),
            "duplicate source content in mosaic plan: {}",
            f.path.display()
        );
        ensure!(
            sr_raw::peek_filter(&f.path).as_deref() == Some(plan.filter.as_str()),
            "filter differs from plan: {}",
            f.path.display()
        );
    }
    Ok(())
}

fn source_hash(path: &Path) -> Result<String> {
    let mut source = fs::File::open(path)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hash = Sha256::new();
    loop {
        let n = source.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn run(args: &Args) -> Result<()> {
    match &args.command {
        Command::Prepare {
            input,
            output,
            memory_mb,
            cache_dir,
            stage_dir,
        } => crate::mosaic_prepare::run(
            input,
            output,
            *memory_mb,
            cache_dir.as_deref(),
            stage_dir.as_deref(),
        ),
        Command::Build {
            plan,
            output,
            tile,
            memory_mb,
            experimental,
            plan_sha256,
        } => {
            ensure!(
                *experimental,
                "mosaic integration is experimental; pass --experimental to produce a labeled evaluation output"
            );
            let (plan, digest) = read_plan_fingerprinted(plan)?;
            ensure!(
                plan_sha256
                    .as_ref()
                    .is_none_or(|expected| expected == &digest),
                "Prepared plan changed after review; review it again before building"
            );
            build(&plan, output, *tile, *memory_mb)
        }
    }
}

fn empty_frame(width: usize, height: usize, noise: NoiseModel) -> RawFrame {
    RawFrame {
        width,
        height,
        samples: SamplePlane::from_normalised(width, height, vec![0.; width * height]),
        cfa: CfaPattern::MONO,
        defects: DefectMask::none(width, height),
        noise,
        metadata: FrameMetadata::default(),
    }
}

fn tile_photometry(
    f: &FrameSpec,
    origin: (f32, f32),
    size: (usize, usize),
) -> Result<PhotometricMatch> {
    let mut photo = PhotometricMatch {
        gain: [f.gain; 3],
        ..PhotometricMatch::IDENTITY
    };
    let [sx, sy] = f.background_plane;
    let center = (
        origin.0 + size.0 as f32 * 0.5,
        origin.1 + size.1 as f32 * 0.5,
    );
    if let Some(model) = &f.relative_log_gain {
        let u = (f64::from(center.0) - model.center[0]) / model.normalization_scale;
        let v = (f64::from(center.1) - model.center[1]) / model.normalization_scale;
        let du = size.0 as f64 / (2. * model.normalization_scale);
        let dv = size.1 as f64 / (2. * model.normalization_scale);
        let [a, b, c, d, e] = model.coefficients;
        let coefficients = [
            a * u + b * v + c * u * u + d * u * v + e * v * v,
            du * (a + 2. * c * u + d * v),
            dv * (b + d * u + 2. * e * v),
            c * du * du,
            d * du * dv,
            e * dv * dv,
        ]
        .map(|v| v as f32);
        ensure!(
            coefficients.iter().all(|v| v.is_finite()),
            "Relative stellar-response tile conversion overflows"
        );
        photo.log_gain = Some([coefficients; 3]);
    }
    photo.offset = [f.offset + sx * center.0 + sy * center.1; 3];
    let last = (sr_quality::photometry::FIELD - 1) as f32;
    for channel in &mut photo.field {
        for (y, row) in channel.iter_mut().enumerate() {
            for (x, value) in row.iter_mut().enumerate() {
                *value = sx * size.0 as f32 * (x as f32 / last - 0.5)
                    + sy * size.1 as f32 * (y as f32 / last - 0.5);
            }
        }
    }
    if f.background_quadratic != [0.; 3] {
        // Evaluate in f64 to avoid cancellation on large global canvases. The
        // existing bilinear field reproduces xy exactly; x²/y² interpolation
        // error is bounded independently before tiles are reconstructed.
        let [xx, xy, yy] = f.background_quadratic.map(f64::from);
        let quadratic = |x: f64, y: f64| xx * x * x + xy * x * y + yy * y * y;
        let center_value = quadratic(f64::from(center.0), f64::from(center.1));
        photo.offset = [photo.offset[0] + center_value as f32; 3];
        for channel in &mut photo.field {
            for (y, row) in channel.iter_mut().enumerate() {
                for (x, value) in row.iter_mut().enumerate() {
                    let gx = f64::from(origin.0) + size.0 as f64 * x as f64 / f64::from(last);
                    let gy = f64::from(origin.1) + size.1 as f64 * y as f64 / f64::from(last);
                    *value += (quadratic(gx, gy) - center_value) as f32;
                }
            }
        }
    }
    ensure!(
        photo
            .offset
            .iter()
            .chain(photo.field.iter().flatten().flatten())
            .all(|v| v.is_finite()),
        "background correction overflows on the requested output grid"
    );
    Ok(photo)
}

/// Caller owns the final destination. A failed build leaves only its uniquely
/// named staging directory; it never replaces an earlier completed result.
pub fn build(plan: &Plan, output: &Path, tile: usize, memory_mb: usize) -> Result<()> {
    validate(plan, tile, memory_mb)?;
    ensure!(
        !output.exists(),
        "output already exists: {}",
        output.display()
    );
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let name = output
        .file_name()
        .context("output needs a directory name")?
        .to_string_lossy();
    let staging = parent.join(format!(
        ".{name}.partial-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&staging)?;
    let result = (|| -> Result<()> {
        let mut record = BufWriter::new(
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(staging.join("plan.json"))?,
        );
        serde_json::to_writer_pretty(&mut record, plan)?;
        std::io::Write::flush(&mut record)?;
        drop(record);
        // Keep guide/statistics temporaries independent of host-wide Rayon settings.
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
        pool.install(|| build_staged(plan, &staging, tile, memory_mb))
    })();
    if let Err(ref e) = result {
        let _ = fs::write(staging.join("FAILED.txt"), format!("{e:#}\n"));
    }
    result?;
    ensure!(
        !output.exists(),
        "destination appeared while building; completed staging retained"
    );
    fs::rename(&staging, output)?;
    println!("Mosaic evaluation output: {}", output.display());
    Ok(())
}

/// Lets tests prove output does not depend on the source band cache. Any test
/// may observe it set: by that same claim, no output can change.
#[cfg(test)]
pub(crate) static READ_DIRECTLY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
fn test_reads_directly() -> bool {
    #[cfg(test)]
    {
        READ_DIRECTLY.load(std::sync::atomic::Ordering::SeqCst)
    }
    #[cfg(not(test))]
    {
        false
    }
}

fn build_staged(plan: &Plan, out: &Path, tile: usize, memory_mb: usize) -> Result<()> {
    let started = Instant::now();
    let g = &plan.grid;
    let psf_cohorts = global_psf_cohort_representatives(
        &plan.frames.iter().map(|f| f.psf_hfd).collect::<Vec<_>>(),
    );
    let budget = memory_mb
        .checked_mul(1024 * 1024)
        .context("memory budget overflow")?;
    let preview_factor = g.width.max(g.height).div_ceil(1600).max(1);
    let pw = g.width.div_ceil(preview_factor);
    let ph = g.height.div_ceil(preview_factor);
    let total = g
        .width
        .div_ceil(tile)
        .checked_mul(g.height.div_ceil(tile))
        .context("tile count overflow")?;
    let plan_bytes = plan_memory_bytes(plan)?;
    // Persistent arrays, output bookkeeping, plan and bounded miscellaneous
    // buffers are reserved before opening sources or allocating any mesh.
    let mut resident = pw
        .checked_mul(ph)
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| n.checked_add(total.checked_mul(3)?))
        .and_then(|n| n.checked_add(plan_bytes))
        .and_then(|n| n.checked_add(8 * 1024 * 1024))
        .context("resident working set overflow")?;
    ensure!(
        resident < budget,
        "persistent mosaic buffers exceed memory budget"
    );
    let mut sources = Vec::with_capacity(plan.frames.len());
    for f in &plan.frames {
        let nodes = ((budget - resident) / 12).min(1_000_000);
        let source = PreparedSource::open_with_node_budget(&f.path, &f.projection, f.noise, nodes)?;
        resident = resident
            .checked_add(source.mesh_bytes())
            .context("mesh bytes overflow")?;
        ensure!(resident < budget, "source meshes exceed memory budget");
        sources.push(source);
    }
    for (s, f) in sources.iter().zip(&plan.frames) {
        ensure!(
            (s.width, s.height) == (f.width, f.height),
            "source dimensions changed"
        );
    }
    let writer = |name: &str| -> Result<_> {
        sr_output::scientific::MonoFitsTileWriter::new(
            BufWriter::new(
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(out.join(name))?,
            ),
            g.width,
            g.height,
            tile,
        )
    };
    let mut image = writer("image.fits")?;
    let mut weights_out = writer("weight.fits")?;
    let mut counts_out = writer("samples.fits")?;
    let cfg = mosaic_reconstruction_config(g.scale);
    let variance = cfg.kernel.mono_kernel_variance.unwrap_or(0.16);
    let mut betas: Vec<_> = plan
        .frames
        .iter()
        .map(|f| f.noise.variance(f.sky) * f.gain * f.gain)
        .collect();
    betas.sort_by(f32::total_cmp);
    let noise = NoiseModel::new(0., betas[betas.len() / 2], NoiseSource::Measured);
    // Rejection statistics live in reference/guide pixels. Keep their context
    // fixed in that coordinate system when the output is supersampled.
    let halo = (32. * g.scale).ceil() as usize;
    let mut preview_sum = vec![0f64; pw * ph];
    let mut preview_count = vec![0u32; pw * ph];
    let mut done = 0usize;
    let mut peak_estimated = resident;
    let mut band_cache = crate::mosaic_source::BandCache::default();
    let mut bypassed_tiles = 0usize;
    let mut supported_pixels = 0u64;
    let mut minimum_effective_frames: Option<f32> = None;
    let mut tile_mean_effective_range: Option<[f32; 2]> = None;
    for y in (0..g.height).step_by(tile) {
        for x in (0..g.width).step_by(tile) {
            let w = tile.min(g.width - x);
            let h = tile.min(g.height - y);
            // Fixed halo and global tile lattice prevent rejection neighborhoods
            // from ending at the written tile boundary.
            let origin = (
                g.origin[0] + (x as f32 - halo as f32) / g.scale,
                g.origin[1] + (y as f32 - halo as f32) / g.scale,
            );
            let ew = w + 2 * halo;
            let eh = h + 2 * halo;
            let bounds = Rect::new(0, 0, ew, eh);
            let rectangles: Vec<_> = sources
                .iter()
                .map(|s| s.source_rect(origin, bounds, g.scale, cfg.kernel.radius))
                .collect::<Result<_>>()?;
            let native_pixels = rectangles
                .iter()
                .flatten()
                .try_fold(0usize, |sum, r| {
                    sum.checked_add(r.width.checked_mul(r.height)?)
                })
                .context("source window size overflow")?;
            let rw = ((ew as f32 / g.scale).ceil() as usize + 2).div_ceil(2) * 2;
            let rh = ((eh as f32 / g.scale).ceil() as usize + 2).div_ceil(2) * 2;
            let active_count = rectangles.iter().flatten().count();
            let estimate = native_pixels
                .checked_mul(64)
                .and_then(|n| {
                    n.checked_add(rw.checked_mul(rh)?.checked_mul(8 * active_count + 128)?)
                })
                .and_then(|n| n.checked_add(w.checked_mul(h)?.checked_mul(256)?))
                .and_then(|n| n.checked_add(resident))
                // Encoded read buffer of the one source window being read.
                .and_then(|n| n.checked_add(sr_raw::window::MAX_READ_SPAN_BYTES))
                .context("working set overflow")?;
            ensure!(
                estimate <= budget,
                "tile working-set estimate {} MiB exceeds {} MiB; choose a smaller tile",
                estimate.div_ceil(1024 * 1024),
                memory_mb
            );
            // Source bands may use half of what this tile leaves of the budget;
            // the other half stays free for its crop meshes. A tile needing
            // more bands than that reads its windows directly rather than
            // evicting bands it is about to need again.
            band_cache.set_capacity((budget - estimate) / 2);
            let band_bytes: usize = rectangles
                .iter()
                .zip(&sources)
                .filter_map(|(r, s)| r.map(|r| s.band_bytes(r)))
                .sum();
            let cached = band_bytes <= (budget - estimate) / 2 && !test_reads_directly();
            if !cached {
                bypassed_tiles += 1;
            }
            peak_estimated = peak_estimated.max(estimate + band_cache.bytes());
            let mut crop_mesh_bytes = 0usize;
            let mut frames = vec![empty_frame(rw, rh, noise)];
            let mut warps = vec![WarpField::identity()];
            let mut photo = vec![PhotometricMatch::IDENTITY];
            let mut rejection_photo = vec![PhotometricMatch::IDENTITY];
            let mut weights = vec![0.];
            let mut psf = vec![0.];
            let mut sky = vec![[0.; 3]];
            for (i, rect) in rectangles.into_iter().enumerate() {
                if let Some(rect) = rect {
                    let crop = if cached {
                        sources[i].read_rect_cached(i, rect, &mut band_cache)?
                    } else {
                        sources[i].read_rect(rect)?
                    };
                    let mut p = sources[i].projection.clone();
                    p.center[0] -= rect.x as f64;
                    p.center[1] -= rect.y as f64;
                    p.output_center[0] -= origin.0 as f64;
                    p.output_center[1] -= origin.1 as f64;
                    let old = sources[i].warp.map(
                        rect.x as f32 + rect.width as f32 * 0.5,
                        rect.y as f32 + rect.height as f32 * 0.5,
                    );
                    let exact = p
                        .map(rect.width as f64 * 0.5, rect.height as f64 * 0.5)
                        .context("invalid crop projection")?;
                    ensure!(
                        (old.0 as f64 - exact.0 - origin.0 as f64)
                            .hypot(old.1 as f64 - exact.1 - origin.1 as f64)
                            < 0.1,
                        "crop geometry differs from validated source mesh"
                    );
                    // Mesh resolution must not depend on what happens to be
                    // cached, or output would too: meshes get the budget they
                    // would have without the cache, and the cache gives way.
                    let nodes = ((budget - estimate - crop_mesh_bytes) / 12).min(1_000_000);
                    let warp = p.to_warp_with_node_budget(rect.width, rect.height, 0.01, nodes)?;
                    if let Some(local) = &warp.local {
                        crop_mesh_bytes += local.u.capacity() * 8 + local.conf.capacity() * 4;
                    }
                    ensure!(
                        estimate + crop_mesh_bytes <= budget,
                        "crop meshes exceed memory budget"
                    );
                    band_cache.set_capacity(
                        (budget - estimate - crop_mesh_bytes).min((budget - estimate) / 2),
                    );
                    peak_estimated =
                        peak_estimated.max(estimate + crop_mesh_bytes + band_cache.bytes());
                    warps.push(warp);
                    frames.push(crop.frame);
                    let [xx, _, yy] = plan.frames[i].background_quadratic;
                    let cells = (sr_quality::photometry::FIELD - 1) as f64;
                    let interpolation_bound = (f64::from(xx).abs() * (rw as f64 / cells).powi(2)
                        + f64::from(yy).abs() * (rh as f64 / cells).powi(2))
                        / 4.;
                    let minimum_gain = if let Some(model) = &plan.frames[i].relative_log_gain {
                        model.log_bounds([
                            f64::from(origin.0),
                            f64::from(origin.1),
                            f64::from(origin.0) + rw as f64,
                            f64::from(origin.1) + rh as f64,
                        ])?[0]
                            .exp() as f32
                            * plan.frames[i].gain
                    } else {
                        plan.frames[i].gain
                    };
                    let corrected_sigma =
                        plan.frames[i].noise.variance(plan.frames[i].sky).sqrt() * minimum_gain;
                    ensure!(
                        interpolation_bound <= f64::from(corrected_sigma) * 0.01,
                        "quadratic background interpolation exceeds 1% of source noise; choose a smaller tile"
                    );
                    photo.push(tile_photometry(&plan.frames[i], origin, (rw, rh))?);
                    // Guide cell coordinates represent native centers 2*g+0.5;
                    // its photometry API normalizes 2*g, so include that shift.
                    rejection_photo.push(tile_photometry(
                        &plan.frames[i],
                        (origin.0 + 0.5, origin.1 + 0.5),
                        (rw, rh),
                    )?);
                    weights.push(plan.frames[i].weight);
                    psf.push(psf_cohorts[i]);
                    sky.push([plan.frames[i].sky; 3]);
                }
            }
            let (values, weight, count) = if frames.len() > 1 {
                let active: Vec<_> = weights.iter().map(|v| *v > 0.).collect();
                let robustness = sr_reconstruct::mosaic_quality::build_maps_with_psf_cohorts(
                    &frames,
                    &warps,
                    0,
                    &active,
                    &rejection_photo,
                    &noise,
                    0.5,
                    &cfg.robustness,
                    &vec![[0.; 3]; frames.len()],
                    &psf,
                    PSF_RELATIVE_TOLERANCE,
                )?;
                let kernels =
                    KernelField::isotropic(rw / 2, rh / 2, variance, cfg.kernel.radius, 2);
                let inputs = MergeInputs {
                    frames: &frames,
                    warps: &warps,
                    reference: 0,
                    photometry: &photo,
                    noise,
                    robustness: &robustness,
                    kernels: &kernels,
                    frame_weight: &weights,
                    lucky: None,
                    chroma: RadialChroma::identity(),
                };
                let result = sr_reconstruct::merge::reconstruct_mono_tile_feathered(
                    &inputs,
                    &cfg,
                    (0., 0.),
                    Rect::new(halo, halo, w, h),
                    &sky,
                    DETECTOR_EDGE_FEATHER,
                )?;
                if result.stats.min_effective_frames > 0. {
                    let low = result.stats.min_effective_frames;
                    minimum_effective_frames =
                        Some(minimum_effective_frames.map_or(low, |v| v.min(low)));
                    let mean = result.stats.mean_effective_frames;
                    tile_mean_effective_range = Some(
                        tile_mean_effective_range
                            .map_or([mean, mean], |v| [v[0].min(mean), v[1].max(mean)]),
                    );
                }
                (result.values, result.weight, result.count)
            } else {
                (vec![f32::NAN; w * h], vec![0.; w * h], vec![0.; w * h])
            };
            image.write_tile(x, y, w, h, &values)?;
            weights_out.write_tile(x, y, w, h, &weight)?;
            counts_out.write_tile(x, y, w, h, &count)?;
            for dy in 0..h {
                for dx in 0..w {
                    let v = values[dy * w + dx];
                    if v.is_finite() {
                        supported_pixels += 1;
                        let i = ((y + dy) / preview_factor) * pw + (x + dx) / preview_factor;
                        preview_sum[i] += v as f64;
                        preview_count[i] += 1;
                    }
                }
            }
            done += 1;
            if done == 1 || done.is_multiple_of(25) || done == total {
                println!(
                    "mosaic tile {done}/{total}, {:.1}s",
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }
    image.finish()?;
    weights_out.finish()?;
    counts_out.finish()?;
    let preview = Plane::from_vec(
        pw,
        ph,
        preview_sum
            .into_iter()
            .zip(&preview_count)
            .map(|(s, n)| if *n > 0 { (s / *n as f64) as f32 } else { 0. })
            .collect(),
    );
    let mut preview_planes = [preview, Plane::new(0, 0), Plane::new(0, 0)];
    sr_output::scientific::write_fits(&out.join("preview-linear.fits"), &preview_planes, 1)?;
    // Display only, after writing the linear preview. Original output tiles
    // have already been finalized; stretching cannot change scientific pixels.
    let display_range = stretch_mosaic_preview(&mut preview_planes[0], &preview_count);
    drop(preview_count);
    sr_output::write_preview_png(&out.join("preview.png"), &preview_planes[..1], 1600)?;
    fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
        "status":"experimental evaluation","experimental":true,"width":g.width,"height":g.height,"frames":plan.frames.len(),
        "elapsed_seconds":started.elapsed().as_secs_f64(),
        "source_band_cache":{"hits":band_cache.hits,"misses":band_cache.misses,"tiles_read_directly":bypassed_tiles},"peak_working_set_estimate_bytes":peak_estimated,
        "memory_budget_bytes":budget,"tile":tile,"calibration":plan.calibration,
        "preview":{"black":display_range[0],"white":display_range[1],"stretch":"display-only asinh, strength 20, supported 0.5/99.8 percentiles"},
        "coverage":{"supported_pixels":supported_pixels,"unsupported_pixels":g.width as u64*g.height as u64-supported_pixels,
            "minimum_effective_frames":minimum_effective_frames,"range_of_tile_mean_effective_frames":tile_mean_effective_range,
            "note":"Effective frames are measured on the merger's decimated weight grid. The tile-mean range is not a global mean or an exposure-time map."},
        "policies":{"rejection":"per-frame photometrically scaled noise with independent PSF-cohort consensus and peer uncertainty",
            "psf_selection":"finest cohort with native geometric guide coverage; rejected fine samples do not admit broader substitutes",
            "background":"explicit global additive coefficients; quadratic field interpolation bounded to 1% of corrected source sky-noise sigma",
            "psf_hfd_relative_tolerance":PSF_RELATIVE_TOLERANCE,
            "psf_cohort_membership":"fixed across the complete plan before tile selection",
            "rejection_extra_normalized_sigma_floor":cfg.robustness.sigma_floor,
            "psf_cohort_representative_hfd_by_plan_frame":psf_cohorts,
            "detector_edge_feather_fraction":DETECTOR_EDGE_FEATHER},
        "limitations":["Geometry and source units require plan validation","Memory is an application buffer estimate, not OS peak RSS","No absolute astrometric WCS or certified photometric calibration",
            "Global stellar HFD is a resolution proxy and does not certify spatially varying PSF, ellipticity or undersampling",
            "PSF selection trades broader-frame noise reduction for finer resolution",
            "Uncalibrated detector inputs cannot establish calibrated radiometry or distinguish detector gradients from celestial structure"]}))?,
    )?;
    Ok(())
}

fn stretch_mosaic_preview(preview: &mut Plane<f32>, counts: &[u32]) -> [f32; 2] {
    let mut values: Vec<_> = preview
        .data
        .iter()
        .zip(counts)
        .filter_map(|(&v, &n)| (n > 0 && v.is_finite()).then_some(v))
        .collect();
    values.sort_by(f32::total_cmp);
    let range = if values.is_empty() {
        [0., 1.]
    } else {
        let black = values[((values.len() - 1) as f64 * 0.005) as usize];
        let white = values[((values.len() - 1) as f64 * 0.998) as usize];
        [black, white.max(black + 1e-9)]
    };
    for (v, &n) in preview.data.iter_mut().zip(counts) {
        *v = if n > 0 && v.is_finite() {
            let normalized = ((*v as f64 - range[0] as f64)
                / (range[1] as f64 - range[0] as f64).max(1e-12))
            .clamp(0., 1.);
            ((20. * normalized).asinh() / 20f64.asinh()) as f32
        } else {
            0.
        };
    }
    range
}
