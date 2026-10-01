//! Streaming monochrome project telemetry; never invokes reconstruction.
mod cache;
mod report;
mod stats;

use crate::pipeline::{flag_frames, registration_usable, SurveyRow};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sr_core::frame::{FrameMetadata, RawFrame};
use sr_core::geometry::{GlobalTransform, WarpField};
use sr_core::star::Star;
use sr_quality::photometry::{FramePhotometry, PhotometricMatch, PhotometrySource};
use sr_quality::stars::StarMetrics;
use sr_register::global::GlobalRegistration;
use sr_register::pyramid::RegistrationImage;
use stats::{Depth, Fit, Projection, PITCH, PIXELS, TILE};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(clap::Args, Debug)]
pub struct Args {
    /// Directory, single frame, or existing Smokstak text-list format.
    pub input: PathBuf,
    #[arg(long)]
    pub pattern: Option<String>,
    /// Select a filter by its header, case-insensitively.
    #[arg(long)]
    pub filter: Option<String>,
    #[arg(long, default_value = "project-analysis.json")]
    pub json: PathBuf,
    #[arg(long, default_value = "project-analysis.html")]
    pub html: PathBuf,
    /// Compact per-frame cache; enabled by default for incremental analysis.
    #[arg(long, default_value = "./cache/analysis")]
    pub cache_dir: PathBuf,
    /// Use a temporary sample spool and do not reuse previous results.
    #[arg(long)]
    pub no_cache: bool,
    #[arg(long, default_value = "auto")]
    pub fits_row_order: String,
    /// Requested detector samples, rounded down to whole 16x16 patches.
    #[arg(long, default_value_t = stats::DEFAULT_SAMPLES)]
    pub samples: usize,
    /// Limit inputs in existing filename order, after filter selection.
    #[arg(long)]
    pub max_frames: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Record {
    metadata: FrameMetadata,
    width: usize,
    height: usize,
    stars: Option<StarMetrics>,
    sharpness: f32,
    saturation: f32,
    background_median: Option<f32>,
    background_mad_sigma: Option<f32>,
    spatial_noise: Option<f32>,
    registration: GlobalRegistration,
    warp: WarpField,
    photometry: FramePhotometry,
    stellar_gain: bool,
    valid_tiles: Vec<bool>,
    samples_digest: String,
}

#[derive(Serialize)]
struct Frame {
    index: usize,
    #[serde(flatten)]
    survey: SurveyRow,
    capture_time: Option<String>,
    exposure_seconds: Option<f32>,
    known_integration_seconds: f64,
    accepted_integration_seconds: f64,
    accepted: bool,
    rejection: Option<String>,
    cache_hit: bool,
    background_median: Option<f32>,
    background_mad_sigma: Option<f32>,
    spatial_noise: Option<f32>,
    photometry: Option<FramePhotometry>,
    stellar_gain: Option<bool>,
    registration_residual_sensor_px: Option<f32>,
    registration_confidence: Option<f32>,
}

impl Frame {
    fn empty(index: usize, path: &Path, filter: &str) -> Self {
        Self {
            index,
            survey: SurveyRow {
                path: path.to_string_lossy().into_owned(),
                file: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                filter: filter.to_string(),
                ..Default::default()
            },
            capture_time: None,
            exposure_seconds: None,
            known_integration_seconds: 0.0,
            accepted_integration_seconds: 0.0,
            accepted: false,
            rejection: None,
            cache_hit: false,
            background_median: None,
            background_mad_sigma: None,
            spatial_noise: None,
            photometry: None,
            stellar_gain: None,
            registration_residual_sensor_px: None,
            registration_confidence: None,
        }
    }
    fn reject(&mut self, reason: String) {
        self.rejection = Some(reason);
        self.accepted = false;
    }
    fn set_record(&mut self, r: &Record, hit: bool) {
        self.capture_time = r.metadata.capture_time.clone();
        self.exposure_seconds = r
            .metadata
            .exposure_time
            .filter(|t| t.is_finite() && *t > 0.0);
        self.survey.sharpness = Some(r.sharpness);
        self.survey.saturation = r.saturation;
        self.survey.hfd = r.stars.map(|s| s.hfd);
        self.survey.eccentricity = r.stars.map(|s| s.eccentricity);
        self.survey.stars = r.stars.map(|s| s.count);
        self.background_median = r.background_median;
        self.background_mad_sigma = r.background_mad_sigma;
        self.spatial_noise = r.spatial_noise;
        self.photometry = Some(r.photometry);
        self.stellar_gain = Some(r.stellar_gain);
        self.registration_residual_sensor_px = Some(2.0 * r.registration.residual_p50);
        self.registration_confidence = Some(r.registration.confidence);
        self.cache_hit = hit;
        self.accepted = registration_usable(&r.registration);
        if !self.accepted {
            self.reject("registration did not pass the stack's usability gate".into());
        }
    }
}

#[derive(Default, Serialize)]
struct Timings {
    discovery_seconds: f64,
    fingerprint_seconds: f64,
    decode_seconds: f64,
    quality_seconds: f64,
    registration_seconds: f64,
    normalization_seconds: f64,
    sampling_seconds: f64,
    cache_seconds: f64,
    analysis_seconds: f64,
    report_render_seconds: f64,
    processing_seconds: f64,
}

#[derive(Serialize)]
struct Group {
    filter: String,
    reference_file: Option<String>,
    reference_sha256: Option<String>,
    candidate_tiles: usize,
    common_tiles: usize,
    sample_count: usize,
    accepted_frames: usize,
    rejected_frames: usize,
    known_integration_seconds: f64,
    unknown_exposures: usize,
    median_hfd_sensor_px: Option<f32>,
    median_eccentricity: Option<f32>,
    median_registration_residual_sensor_px: Option<f32>,
    global_fit: Option<Fit>,
    recent_fit: Option<Fit>,
    projections: Vec<Projection>,
    integration_depth: Vec<Depth>,
    patch_positions: Vec<(usize, usize)>,
    snapshots: Vec<stats::Snapshot>,
    warnings: Vec<String>,
}

#[derive(Serialize)]
struct Summary {
    input_frames: usize,
    duplicate_files_dropped: usize,
    accepted_frames: usize,
    rejected_frames: usize,
    known_integration_seconds: f64,
    accepted_integration_seconds: f64,
    unknown_exposures: usize,
    cache_hits: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    program_version: &'static str,
    build_sha256: String,
    project: String,
    order: &'static str,
    method: &'static str,
    limitations: Vec<&'static str>,
    requested_samples: usize,
    summary: Summary,
    frames: Vec<Frame>,
    filters: Vec<Group>,
    timings: Timings,
}

struct Reference {
    frame: RawFrame,
    proxy: RegistrationImage,
    stars: Vec<Star>,
}

fn decode(path: &Path, opts: &sr_raw::ReadOptions, timings: &mut Timings) -> Result<RawFrame> {
    let t = Instant::now();
    let result = sr_raw::decode_with(path, opts);
    timings.decode_seconds += t.elapsed().as_secs_f64();
    Ok(result?)
}

fn make_reference(frame: RawFrame, timings: &mut Timings) -> Reference {
    let t = Instant::now();
    let cfg = sr_core::config::RegistrationConfig::default();
    let proxy = RegistrationImage::build(&frame.registration_luma(), cfg.pyramid_levels);
    let stars = sr_quality::stars::positions(&frame, 3000);
    timings.registration_seconds += t.elapsed().as_secs_f64();
    Reference {
        frame,
        proxy,
        stars,
    }
}

fn seed(reference: &RawFrame, frame: &RawFrame, stars: &[Vec<Star>]) -> Option<GlobalTransform> {
    let solved = reference
        .metadata
        .wcs
        .zip(frame.metadata.wcs)
        .and_then(|(r, t)| {
            sr_core::wcs::relative_affine(&r, &t, reference.width, reference.height)
        });
    if let Some(a) = solved {
        return Some(
            GlobalTransform {
                m: a.map(|v| v as f32),
            }
            .rescale(0.5),
        );
    }
    sr_register::asterism::match_stars(
        &stars[0][..stars[0].len().min(200)],
        &stars[1][..stars[1].len().min(200)],
    )
    .map(|m| m.transform.rescale(0.5))
}

/// Gather the nearest original detector sample to each fixed reference-grid
/// location. No interpolation kernel changes noise with fractional dither.
/// Two-pixel spacing avoids repeated neighbours under normal near-unit warps.
fn extract(
    frame: &RawFrame,
    warp: &WarpField,
    tiles: &[(usize, usize)],
    map: PhotometricMatch,
) -> (Vec<f32>, Vec<bool>) {
    let mut out = vec![f32::NAN; tiles.len() * PIXELS];
    let mut valid = vec![true; tiles.len()];
    for (i, &(x, y)) in tiles.iter().enumerate() {
        let mut used = std::collections::HashSet::with_capacity(PIXELS);
        for j in 0..TILE {
            for k in 0..TILE {
                let (rx, ry) = ((x + k * PITCH) as f32, (y + j * PITCH) as f32);
                let Some((sx, sy)) = warp.inverse_map(rx, ry) else {
                    valid[i] = false;
                    continue;
                };
                let (sx, sy) = (sx.round(), sy.round());
                if !sx.is_finite()
                    || !sy.is_finite()
                    || sx < 0.0
                    || sy < 0.0
                    || sx >= frame.width as f32
                    || sy >= frame.height as f32
                {
                    valid[i] = false;
                    continue;
                }
                let (sx, sy) = (sx as usize, sy as usize);
                let index = sy * frame.width + sx;
                let value = frame.value(sx, sy);
                if !frame.usable_value(index, value) || !used.insert(index) {
                    valid[i] = false;
                    continue;
                }
                out[i * PIXELS + j * TILE + k] = map.apply(0, value);
            }
        }
    }
    (out, valid)
}

fn measure(
    frame: &RawFrame,
    reference: &Reference,
    is_reference: bool,
    tiles: &[(usize, usize)],
    timings: &mut Timings,
) -> (Record, Vec<f32>) {
    let t = Instant::now();
    let surveyed = crate::pipeline::survey_frame(frame, true, 0);
    let stars = surveyed.stars;
    let sharpness = surveyed.sharpness;
    let saturation = surveyed.saturation_fraction;
    timings.quality_seconds += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let star_lists = vec![
        reference.stars.clone(),
        if is_reference {
            reference.stars.clone()
        } else {
            sr_quality::stars::positions(frame, 3000)
        },
    ];
    let (registration, warp) = if is_reference {
        (GlobalRegistration::identity(0), WarpField::identity())
    } else {
        let cfg = sr_core::config::RegistrationConfig::default();
        let proxy = RegistrationImage::build(&frame.registration_luma(), cfg.pyramid_levels);
        let r = sr_register::global::register_pair_seeded(
            1,
            &reference.proxy,
            &proxy,
            &cfg,
            &mut sr_register::correlate::CorrelatorCache::new(),
            seed(&reference.frame, frame, &star_lists),
        );
        let mut transforms = [GlobalTransform::IDENTITY, r.transform.rescale(2.0)];
        let polished = sr_register::refine::refine_against_stars(0, &star_lists, &transforms);
        if polished[1].1.applied {
            transforms[1] = polished[1].0.compose(&transforms[1]);
        }
        let mut warp = WarpField::global_only(transforms[1]);
        let mut fields = sr_register::distortion::refine_stars(
            0,
            &star_lists,
            &transforms,
            frame.width,
            frame.height,
        );
        warp.local = fields.pop().flatten();
        (r, warp)
    };
    timings.registration_seconds += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (photometry, stellar_gain) = if is_reference {
        (
            FramePhotometry {
                map: PhotometricMatch::IDENTITY,
                source: PhotometrySource::Measured,
            },
            true,
        )
    } else {
        let phot_stars: Vec<_> = [&reference.frame, frame].iter().zip(&star_lists)
            .map(|(f, s)| sr_quality::stars::photometric_catalog(f, &s[..s.len().min(600)]))
            .collect();
        let stellar_gain = sr_quality::photometry::star_gains(
            &[WarpField::identity(), warp.clone()],
            0,
            &phot_stars,
            2,
        )[1];
        let photometry = sr_quality::photometry::match_with_star_refs(
            &[&reference.frame, frame],
            &[WarpField::identity(), warp.clone()],
            0,
            &[
                1.0,
                sr_raw::exposure_level(&reference.frame.metadata)
                    / sr_raw::exposure_level(&frame.metadata),
            ],
            &[],
            &phot_stars,
            false,
        )[1];
        // A stellar estimate can be declined by the photometry gain limits;
        // report the gain actually applied, not merely the availability of stars.
        let stellar_gain = stellar_gain.is_some_and(|g| {
            photometry.source == PhotometrySource::Measured && g[0] == photometry.map.gain[0]
        });
        (photometry, stellar_gain)
    };
    timings.normalization_seconds += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (raw, valid_tiles) = extract(frame, &warp, tiles, PhotometricMatch::IDENTITY);
    let values: Vec<_> = raw.iter().copied().filter(|v| v.is_finite()).collect();
    let background_median = (!values.is_empty()).then(|| sr_core::math::median(&values));
    let background_mad_sigma = (!values.is_empty()).then(|| sr_core::math::mad_sigma(&values));
    let mut sigmas: Vec<_> = raw
        .chunks_exact(PIXELS)
        .zip(&valid_tiles)
        .filter(|(_, valid)| **valid)
        .map(|(v, _)| sr_noise::spatial::detrended_tile_sigma(v, TILE))
        .collect();
    let spatial_noise =
        (!sigmas.is_empty()).then(|| sr_noise::spatial::lower_decile_noise(&mut sigmas));
    let samples = raw.iter().map(|&v| photometry.map.apply(0, v)).collect();
    timings.sampling_seconds += t.elapsed().as_secs_f64();
    (
        Record {
            metadata: frame.metadata.clone(),
            width: frame.width,
            height: frame.height,
            stars,
            sharpness,
            saturation,
            background_median,
            background_mad_sigma,
            spatial_noise,
            registration,
            warp,
            photometry,
            stellar_gain,
            valid_tiles,
            samples_digest: String::new(),
        },
        samples,
    )
}

fn median(values: impl Iterator<Item = f32>) -> Option<f32> {
    let values: Vec<_> = values.filter(|v| v.is_finite()).collect();
    (!values.is_empty()).then(|| sr_core::math::median(&values))
}

pub fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    anyhow::ensure!(
        (8192..=2_097_152).contains(&args.samples),
        "--samples must be between 8192 and 2097152"
    );
    anyhow::ensure!(args.max_frames != Some(0), "--max-frames must be positive");
    anyhow::ensure!(
        report::output_identity(&args.json)? != report::output_identity(&args.html)?,
        "JSON and HTML must have different paths"
    );
    let opts = sr_raw::ReadOptions {
        fits_row_order: sr_raw::RowOrder::parse(&args.fits_row_order)
            .context("--fits-row-order must be auto, bottom-up or top-down")?,
    };
    let mut timings = Timings::default();
    let t = Instant::now();
    let mut paths = sr_raw::collect_files(&args.input, args.pattern.as_deref())?;
    // Reports and exported review lists must remain usable from another directory.
    // Keep discovery order; do not resolve symlinks or choose a different reference.
    for path in &mut paths {
        *path = std::path::absolute(&*path)
            .with_context(|| format!("resolving input path {}", path.display()))?;
    }
    if let Some(want) = &args.filter {
        paths.retain(|p| sr_raw::peek_filter(p).is_some_and(|f| f.eq_ignore_ascii_case(want)));
    }
    let (mut paths, duplicates) = sr_raw::deduplicate(paths);
    if let Some(n) = args.max_frames {
        paths.truncate(n);
    }
    anyhow::ensure!(!paths.is_empty(), "no frames to analyze");
    // Never let a chosen output overwrite an input file or the input list.
    for output in [&args.json, &args.html] {
        if let Ok(canonical) = output.canonicalize() {
            anyhow::ensure!(
                !paths.iter().chain(std::iter::once(&args.input)).any(|p| p
                    .canonicalize()
                    .ok()
                    .as_ref()
                    == Some(&canonical)),
                "output names an input file"
            );
        }
    }
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut frames = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        let filter = sr_raw::peek_filter(p).unwrap_or_default();
        groups.entry(filter.clone()).or_default().push(i);
        frames.push(Frame::empty(i, p, &filter));
    }
    timings.discovery_seconds = t.elapsed().as_secs_f64();
    println!(
        "Found {} frames ({} duplicate files dropped)",
        paths.len(),
        duplicates.len()
    );
    let store = cache::Store::new(&args.cache_dir, args.no_cache)?;
    let t = Instant::now();
    let build = cache::digest_file(&std::env::current_exe()?)?;
    let options = format!(
        "{}:{:?}:{}",
        args.samples,
        opts.fits_row_order,
        serde_json::to_string(&sr_core::config::RegistrationConfig::default())?
    );
    timings.fingerprint_seconds += t.elapsed().as_secs_f64();
    let mut output_groups = Vec::new();
    for (filter, indices) in groups {
        println!(
            "Measuring and registering {} frames: filter {}",
            indices.len(),
            if filter.is_empty() {
                "(unknown)"
            } else {
                &filter
            }
        );
        let mut reference: Option<Reference> = None;
        let mut ref_hash = String::new();
        let mut ref_path = None;
        let mut tiles = Vec::new();
        let mut records: Vec<(usize, String, Record)> = Vec::new();
        for &i in &indices {
            let t = Instant::now();
            let digest = cache::digest_file(&paths[i]);
            timings.fingerprint_seconds += t.elapsed().as_secs_f64();
            let digest = match digest {
                Ok(d) => d,
                Err(e) => {
                    frames[i].survey.error = e.to_string();
                    frames[i].reject(e.to_string());
                    continue;
                }
            };
            let mut decoded = None;
            if reference.is_none() {
                match decode(&paths[i], &opts, &mut timings) {
                    Ok(f)
                        if f.is_mono()
                            && stats::tiles(f.width, f.height, args.samples).len() >= 32 =>
                    {
                        tiles = stats::tiles(f.width, f.height, args.samples);
                        ref_hash = digest.clone();
                        ref_path = Some(paths[i].display().to_string());
                        reference = Some(make_reference(f, &mut timings));
                    }
                    Ok(f) => {
                        frames[i].capture_time = f.metadata.capture_time;
                        frames[i].exposure_seconds = f
                            .metadata
                            .exposure_time
                            .filter(|t| t.is_finite() && *t > 0.0);
                        frames[i].reject("analysis requires monochrome frames large enough for at least 32 sample patches".into());
                        continue;
                    }
                    Err(e) => {
                        frames[i].survey.error = e.to_string();
                        frames[i].reject(e.to_string());
                        continue;
                    }
                }
            }
            let reference = reference.as_ref().unwrap();
            let is_reference = ref_path.as_deref() == Some(paths[i].to_string_lossy().as_ref());
            let key = cache::key(&build, &digest, &ref_hash, &options);
            let t = Instant::now();
            let cached = store.load(&key).filter(|r| {
                r.valid_tiles.len() == tiles.len()
                    && r.width == reference.frame.width
                    && r.height == reference.frame.height
            });
            timings.cache_seconds += t.elapsed().as_secs_f64();
            let (record, hit) = match cached {
                Some(r) => (r, true),
                None => {
                    if !is_reference {
                        match decode(&paths[i], &opts, &mut timings) {
                            Ok(f) => decoded = Some(f),
                            Err(e) => {
                                frames[i].survey.error = e.to_string();
                                frames[i].reject(e.to_string());
                                continue;
                            }
                        }
                    }
                    let frame = decoded.as_ref().unwrap_or(&reference.frame);
                    if !frame.is_mono()
                        || frame.width != reference.frame.width
                        || frame.height != reference.frame.height
                    {
                        frames[i].capture_time = frame.metadata.capture_time.clone();
                        frames[i].exposure_seconds = frame
                            .metadata
                            .exposure_time
                            .filter(|t| t.is_finite() && *t > 0.0);
                        frames[i].reject("sensor dimensions or mono/CFA type differ from this filter's reference".into());
                        continue;
                    }
                    let (mut r, samples) =
                        measure(frame, reference, is_reference, &tiles, &mut timings);
                    let t = Instant::now();
                    store.save(&key, &mut r, &samples)?;
                    timings.cache_seconds += t.elapsed().as_secs_f64();
                    (r, false)
                }
            };
            frames[i].cache_hit = hit;
            records.push((i, key, record));
            println!(
                "Measured {} of {}{}",
                i + 1,
                paths.len(),
                if hit { " (cached)" } else { "" }
            );
        }
        drop(reference); // release full-resolution data before the sample replay
        let t = Instant::now();
        let mut registrations: Vec<_> = records
            .iter()
            .map(|(_, _, r)| r.registration.clone())
            .collect();
        sr_register::global::normalise_confidence(&mut registrations);
        let mut valid = vec![true; tiles.len()];
        for ((i, _, r), registration) in records.iter_mut().zip(registrations) {
            r.registration = registration;
            // Cache hit is an observation of this run, not cached metadata.
            // Filled below from the counters maintained during measurement.
            let hit = frames[*i].cache_hit;
            frames[*i].set_record(r, hit);
            if frames[*i].accepted {
                for (v, &present) in valid.iter_mut().zip(&r.valid_tiles) {
                    *v &= present;
                }
            }
        }
        let common_tiles = valid.iter().filter(|&&v| v).count();
        let accepted = indices.iter().filter(|&&i| frames[i].accepted).count();
        let mut group = Group {
            filter,
            reference_file: ref_path,
            reference_sha256: (!ref_hash.is_empty()).then_some(ref_hash),
            candidate_tiles: tiles.len(),
            common_tiles,
            sample_count: common_tiles * PIXELS,
            accepted_frames: accepted,
            rejected_frames: indices.len() - accepted,
            known_integration_seconds: 0.0,
            unknown_exposures: 0,
            median_hfd_sensor_px: median(indices.iter().filter_map(|&i| frames[i].survey.hfd)),
            median_eccentricity: median(
                indices
                    .iter()
                    .filter_map(|&i| frames[i].survey.eccentricity),
            ),
            median_registration_residual_sensor_px: median(
                indices
                    .iter()
                    .filter_map(|&i| frames[i].registration_residual_sensor_px),
            ),
            global_fit: None,
            recent_fit: None,
            projections: Vec::new(),
            integration_depth: Vec::new(),
            patch_positions: tiles.clone(),
            snapshots: Vec::new(),
            warnings: Vec::new(),
        };
        if common_tiles < 32 || accepted == 0 {
            group.warnings.push("Insufficient common valid coverage (32 patches required); no noise curve or fitted model.".into());
        } else {
            println!(
                "Analyzing integration depth: {accepted} frames, {} common samples",
                group.sample_count
            );
            let mut acc = stats::Accumulator::new(valid);
            for (i, key, r) in &records {
                if !frames[*i].accepted {
                    continue;
                }
                let values = store.samples(key, r)?;
                group
                    .integration_depth
                    .push(acc.push(&values, frames[*i].exposure_seconds, *i));
                if stats::snapshot_due(group.integration_depth.len(), accepted) {
                    group.snapshots.push(acc.snapshot());
                }
            }
            group.global_fit = acc.global_fit();
            group.recent_fit = group
                .integration_depth
                .iter()
                .rev()
                .find(|d| d.noise.is_some())
                .and_then(|d| d.local_fit.clone());
        }
        for &i in &indices {
            if frames[i].accepted {
                match frames[i].exposure_seconds {
                    Some(t) => group.known_integration_seconds += t as f64,
                    None => group.unknown_exposures += 1,
                }
            }
        }
        group.projections = stats::projections(
            group.recent_fit.as_ref(),
            accepted,
            group.known_integration_seconds,
            group.unknown_exposures,
        );
        if group.unknown_exposures > 0 {
            group
                .warnings
                .push("Integration totals exclude unknown exposure durations.".into());
        }
        let fallback = indices
            .iter()
            .filter(|&&i| frames[i].accepted && frames[i].stellar_gain == Some(false))
            .count();
        if fallback > 0 {
            group.warnings.push(format!("{fallback} accepted frames lack a stellar gain; the standard block/level/exposure fallback may confound sky changes with transparency. Treat noise scaling cautiously."));
        }
        if group.filter.is_empty() {
            group.warnings.push("Filter is missing: verify that these frames share one passband before interpreting their curve.".into());
        }
        if group.global_fit.is_none() {
            group.warnings.push(
                "Not enough positive-noise checkpoints spanning a factor of two for a fit.".into(),
            );
        }
        timings.analysis_seconds += t.elapsed().as_secs_f64();
        output_groups.push(group);
    }
    let mut survey: Vec<_> = frames.iter().map(|f| f.survey.clone()).collect();
    flag_frames(&mut survey);
    let (mut known, mut accepted_time) = (0.0, 0.0);
    for (frame, row) in frames.iter_mut().zip(survey) {
        frame.survey = row;
        if let Some(t) = frame.exposure_seconds {
            known += t as f64;
            if frame.accepted {
                accepted_time += t as f64;
            }
        }
        frame.known_integration_seconds = known;
        frame.accepted_integration_seconds = accepted_time;
    }
    let accepted = frames.iter().filter(|f| f.accepted).count();
    let summary = Summary {
        input_frames: frames.len(),
        duplicate_files_dropped: duplicates.len(),
        accepted_frames: accepted,
        rejected_frames: frames.len() - accepted,
        known_integration_seconds: known,
        accepted_integration_seconds: accepted_time,
        unknown_exposures: frames
            .iter()
            .filter(|f| f.exposure_seconds.is_none())
            .count(),
        cache_hits: frames.iter().filter(|f| f.cache_hit).count(),
    };
    timings.processing_seconds = started.elapsed().as_secs_f64();
    let mut report=Report {schema_version:3,program_version:env!("CARGO_PKG_VERSION"),build_sha256:build,
        project:args.input.display().to_string(),order:"existing input discovery filename order, independently within each filter",
        method:stats::METHOD,
        limitations:vec![stats::LIMITATION,
            "Nearest-detector sampling approximates scene positions within 0.71 target pixels. No interpolation smoothing; two-pixel pitch and duplicate-site rejection reduce spatial covariance.",
            "Spatial residuals include astronomical structure and are not a measurement of random noise or a proven systematic noise floor.",
            "Fits and R-squared are descriptive. Cumulative points are correlated; no IID confidence intervals are claimed. Projections assume recent scaling persists and are not forecasts.",
            "Survey warnings do not automatically reject exposures. Accepted means decoded mono data with usable registration, not a guarantee of photometric quality.",
            "Pairwise standard star/block photometry uses a fixed per-filter anchor; sky fields are off. Burst obstruction masking and burst defect detection are not run.",
            "Adding frames can change common coverage and relative confidence, so earlier curve values may change. Adding an earlier reference invalidates dependent cached measurements."],
        requested_samples:args.samples,summary,frames,filters:output_groups,timings};
    println!("Writing report...");
    let t = Instant::now();
    let _ = report::html(&report)?;
    report.timings.report_render_seconds = t.elapsed().as_secs_f64();
    report::write(&report, &args.json, &args.html)?;
    for group in &report.filters {
        println!(
            "Filter {}: {} accepted, {:.2} h; noise exponent {}, recent {}",
            group.filter,
            group.accepted_frames,
            group.known_integration_seconds / 3600.0,
            group
                .global_fit
                .as_ref()
                .map(|f| format!("{:.3}", f.exponent))
                .unwrap_or_else(|| "unavailable".into()),
            group
                .recent_fit
                .as_ref()
                .map(|f| format!("{:.3}", f.exponent))
                .unwrap_or_else(|| "unavailable".into())
        );
    }
    println!(
        "Report: {}\nData: {}\nTotal: {:.2?}",
        args.html.display(),
        args.json.display(),
        started.elapsed()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::samples::{DefectMask, SamplePlane};
    fn frame(warp: GlobalTransform) -> RawFrame {
        let (w, h) = (512, 512);
        let values = (0..w * h)
            .map(|i| {
                let (x, y) = warp.apply((i % w) as f32, (i / w) as f32);
                0.1 + x * 0.0002
                    + y * 0.0001
                    + ((x as i32 + 3 * y as i32).rem_euclid(7)) as f32 * 0.001
            })
            .collect();
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_normalised(w, h, values),
            cfa: sr_core::cfa::CfaPattern::MONO,
            defects: DefectMask::none(w, h),
            noise: sr_core::frame::NoiseModel::nominal(100.0, 65535.0),
            metadata: FrameMetadata::default(),
        }
    }
    #[test]
    fn inverse_sampling_follows_scene_through_translation_and_flip() {
        let tiles = stats::tiles(512, 512, 8192);
        let reference = frame(GlobalTransform::IDENTITY);
        let (expected, valid) = extract(
            &reference,
            &WarpField::identity(),
            &tiles,
            PhotometricMatch::IDENTITY,
        );
        assert!(valid.iter().all(|&v| v));
        for transform in [
            GlobalTransform::translation(13.0, -7.0),
            GlobalTransform {
                m: [-1.0, 0.0, 511.0, 0.0, -1.0, 511.0],
            },
        ] {
            let target = frame(transform);
            let (actual, valid) = extract(
                &target,
                &WarpField::global_only(transform),
                &tiles,
                PhotometricMatch::IDENTITY,
            );
            assert!(valid.iter().all(|&v| v));
            assert_eq!(actual, expected);
            let (wrong, _) = extract(
                &target,
                &WarpField::identity(),
                &tiles,
                PhotometricMatch::IDENTITY,
            );
            assert_ne!(
                wrong, expected,
                "negative control must sample different scene positions"
            );
        }
    }
    #[test]
    fn gain_scales_noise_and_offset_does_not() {
        let frame = frame(GlobalTransform::IDENTITY);
        let tiles = stats::tiles(512, 512, 8192);
        let (original, valid) = extract(
            &frame,
            &WarpField::identity(),
            &tiles,
            PhotometricMatch::IDENTITY,
        );
        let map = PhotometricMatch {
            gain: [2.0; 3],
            offset: [0.15; 3],
            ..PhotometricMatch::IDENTITY
        };
        let (scaled, _) = extract(&frame, &WarpField::identity(), &tiles, map);
        let a = stats::Accumulator::new(valid.clone()).push(&original, Some(10.0), 0);
        let b = stats::Accumulator::new(valid).push(&scaled, Some(10.0), 0);
        assert!((b.spatial_residual / a.spatial_residual - 2.0).abs() < 1e-4);
    }
    #[test]
    fn sample_boundary_and_defects_are_not_silently_filled() {
        let mut f = frame(GlobalTransform::IDENTITY);
        let tiles = vec![(0, 0), (64, 64)];
        f.defects.set(64 * 512 + 64);
        let (_, v) = extract(
            &f,
            &WarpField::global_only(GlobalTransform::translation(1.0, 0.0)),
            &tiles,
            PhotometricMatch::IDENTITY,
        );
        assert!(!v[0]);
        let (_, v) = extract(
            &f,
            &WarpField::identity(),
            &tiles,
            PhotometricMatch::IDENTITY,
        );
        assert!(!v[1]);
    }
}
