//! Wiring between the CLI surface and the processing crates.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rayon::prelude::*;

use sr_core::config::{Backend, LuckyMode, PostProcess, ReconstructionConfig};
use sr_core::frame::{FrameQuality, LocalQualityMap, RawFrame};
use sr_core::geometry::WarpField;
use sr_core::plane::Plane;
use sr_core::product::{ReconstructionProduct, SamplingCoverage};
use sr_diagnostics::{FrameRow, RunManifest, SourceFile, Timings};
use sr_quality::photometry::{FramePhotometry, PhotometricMatch, PhotometrySource};
use sr_quality::stars::StarMetrics;
use sr_reconstruct::kernel::KernelField;
use sr_register::global::GlobalRegistration;
use sr_register::pyramid::RegistrationImage;
use sr_register::reference::ReferenceChoice;

use crate::cache::{Cache, CachedDefects, Fingerprint};

/// Guide pixels per side of a local-quality region.
const QUALITY_REGION: usize = 64;

/// How much of the integer file's range must remain above the background
/// before the encoding is worth remarking on.
///
/// Astronomical practice puts the sky near a quarter of full scale, so three
/// quarters of headroom. This is not that bar — it is the bar below which the
/// file has become one flat shade with a few dots on it.
const MIN_HEADROOM: f32 = 0.6;

/// The share of the frame that may clip in some channels and not others before
/// the false colour it produces is worth remarking on.
const MAX_UNEVEN_CLIP: f32 = 1e-5;

/// A frame obstructed over more than this share of itself is dropped whole.
const UNUSABLE_OBSTRUCTION: f32 = 0.30;

/// A frame whose additive sky field spans more than this fraction of the sky
/// level is dropped whole: the sky rose or fell by more than itself across the
/// frame, which is dawn or cloud, not a night sky with a gradient.
const UNUSABLE_SWING: f32 = 1.0;

/// Estimate the lens's lateral chromatic aberration from the burst.
///
/// The aberration belongs to the lens, not to a frame, so a handful of frames
/// is enough and the median across them is steadier than any one of them. The
/// guides are rebuilt here rather than kept from loading, because only a few
/// frames are needed and a full set of RGB guides for a long burst is several
/// gigabytes.
fn estimate_chromatic_aberration(
    burst: &LoadedBurst,
    reference: usize,
    weights: &[f32],
) -> sr_register::chroma::ChromaticAberration {
    debug_assert_eq!(weights.len(), burst.frames.len());
    let sample = chroma_sample_indices(weights, reference);
    if burst.frames[reference].is_mono() {
        // One channel cannot be misregistered against another.
        return sr_register::chroma::ChromaticAberration::none();
    }
    let rggb = burst.frames[reference].cfa == sr_core::cfa::CfaPattern::RGGB;
    let estimates: Vec<sr_register::chroma::ChromaticAberration> = sample
        .par_iter()
        .map(|&i| {
            let guide = burst.frames[i].guide_rgb();
            sr_register::chroma::estimate(&guide, 96, 14, 1.10, rggb)
        })
        .collect();
    sr_register::chroma::combine(&estimates)
}

fn chroma_sample_indices(weights: &[f32], reference: usize) -> Vec<usize> {
    // Rejected exposures must not change the optical correction applied to
    // accepted ones. Space probes through contributors, not the loaded burst.
    let active: Vec<_> = weights.iter().enumerate()
        .filter_map(|(i, &w)| (w.is_finite() && w > 0.0).then_some(i)).collect();
    let n = active.len();
    if n <= 6 { return active; }
    let mut sample: Vec<_> = (0..6).map(|k| active[k * (n - 1) / 5]).collect();
    if active.contains(&reference) { sample.push(reference); }
    sample.sort_unstable();
    sample.dedup();
    sample
}

/// What a survey pass learned about one frame.
///
/// Lives here rather than in the ingestion crate because ranking frames is a
/// judgement about quality, and the crate that decodes files has no business
/// holding an opinion about which of them is good.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameSurvey {
    pub index: usize,
    /// Larger is sharper. A half-flux diameter, inverted, where the frame had
    /// point sources to measure; gradient energy of the half-resolution guide
    /// otherwise.
    pub sharpness: f32,
    pub saturation_fraction: f32,
    /// Set when the sharpness came from point sources.
    pub stars: Option<StarMetrics>,
}

/// Measure every frame without holding any of them.
///
/// Selecting the best `N` of a burst has an ordering problem: ranking frames
/// requires decoding them, and the reason for selecting a subset is usually
/// that decoding all of them at once does not fit in memory. So this decodes
/// each frame, measures it, and drops it, keeping only a handful of scalars.
/// Peak memory is one frame per worker thread rather than the whole burst.
///
/// The cost is a second decode of the chosen frames afterwards.
fn survey_all(
    paths: &[PathBuf],
    opts: &sr_raw::ReadOptions,
    star_metrics: bool,
) -> Result<Vec<FrameSurvey>> {
    // The decoder's own error type; `?` below widens it when the survey ends.
    let mut results: Vec<(usize, sr_core::Result<FrameSurvey>)> = paths
        .par_iter()
        .enumerate()
        .map(|(i, p)| (i, survey_one(p, opts, star_metrics, i)))
        .collect();
    results.sort_by_key(|(i, _)| *i);
    let mut out = Vec::with_capacity(results.len());
    for (_, r) in results {
        out.push(r?);
    }
    Ok(out)
}

/// Decode one frame, measure it, and drop it.
fn survey_one(
    p: &Path,
    opts: &sr_raw::ReadOptions,
    star_metrics: bool,
    index: usize,
) -> sr_core::Result<FrameSurvey> {
    sr_raw::decode_with(p, opts).map(|f| survey_frame(&f, star_metrics, index))
}

/// Survey an already decoded frame, shared with streaming project analysis.
pub(crate) fn survey_frame(f: &RawFrame, star_metrics: bool, index: usize) -> FrameSurvey {
    let stars = if star_metrics {
        sr_quality::stars::measure(f)
    } else {
        None
    };
    let sharpness = match &stars {
        // Inverted so that larger is sharper either way. The scale differs
        // wildly between the two, which does not matter: this is only ever
        // used to order one burst against itself, and a burst is measured
        // one way or the other.
        Some(m) if m.hfd > 0.0 => 1.0 / m.hfd,
        _ => sr_quality::tenengrad(&f.guide_rgb().luma()),
    };
    FrameSurvey {
        index,
        sharpness,
        saturation_fraction: f.saturation_fraction(),
        stars,
    }
}

/// Eccentricity above which a frame's stars are elongated rather than merely
/// as elongated as the optics and mount make every frame.
///
/// Judged against the burst: every frame is slightly elongated by the same
/// optics and the same mount, and what matters is the frame that is worse than
/// its neighbours.
fn elongation_threshold(median: f32) -> f32 {
    (median + 0.15).max(0.35)
}

/// Below this share of its filter's median star count, a frame has lost the sky
/// to something in front of it.
const FEW_STARS: f32 = 0.5;

/// Below this share of its filter's median sharpness, a frame is named as soft.
const SOFT: f32 = 0.75;

/// A filter with fewer frames than this has no typical frame to hold one
/// against.
const SURVEY_MIN_GROUP: usize = 5;

/// Why a frame looks worse than the rest of its filter.
#[derive(Clone, Debug, serde::Serialize)]
pub struct SurveyFlag {
    /// `elongated`, `few-stars`, `no-stars`, `soft` or `unreadable`.
    pub kind: &'static str,
    pub text: String,
}

/// One frame as the survey found it, for a reader deciding what to leave out.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SurveyRow {
    pub path: String,
    pub file: String,
    pub filter: String,
    /// Against the median of frames through the same filter: 1.0 is typical
    /// and larger is sharper. Absent for a frame measured on a different scale
    /// from the rest of its filter, which has no number comparable to theirs.
    pub sharpness: Option<f32>,
    pub hfd: Option<f32>,
    pub eccentricity: Option<f32>,
    pub stars: Option<usize>,
    pub saturation: f32,
    pub flags: Vec<SurveyFlag>,
    /// Why the frame could not be decoded, if it could not.
    pub error: String,
}

/// Measure every frame of a burst and say which look worse than the rest.
///
/// Leaving frames out is a decision the stack mostly makes for itself: every
/// frame is weighted by how sharp it is, so a soft one costs little. What it
/// cannot tell from sharpness is a frame that is wrong — stars trailed by wind
/// or a snagged cable, a sky thinned by cloud — and finding those means
/// measuring every frame, which is the same pass `--select sharpest` makes and
/// holds no more than one frame per worker.
pub fn survey(spec: &InputSpec, json: &Path) -> Result<()> {
    let mut paths = sr_raw::collect_files(spec.path, spec.pattern)?;
    if let Some(want) = spec.filter {
        paths.retain(|p| sr_raw::peek_filter(p).is_some_and(|f| f.eq_ignore_ascii_case(want)));
    }
    anyhow::ensure!(!paths.is_empty(), "no frames to survey");
    let total = paths.len();
    // Both lines are read by the page to say how far it has got.
    println!("surveying {total} frames");
    let t = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let mut rows: Vec<SurveyRow> = paths
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let measured = survey_one(p, &spec.read, spec.star_metrics, i);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            println!("surveyed {n} of {total}");
            let mut row = SurveyRow {
                path: p.to_string_lossy().into_owned(),
                file: p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                filter: sr_raw::peek_filter(p).unwrap_or_default(),
                ..Default::default()
            };
            match measured {
                Ok(s) => {
                    row.sharpness = Some(s.sharpness);
                    row.saturation = s.saturation_fraction;
                    if let Some(m) = s.stars {
                        row.hfd = Some(m.hfd);
                        row.eccentricity = Some(m.eccentricity);
                        row.stars = Some(m.count);
                    }
                }
                Err(e) => row.error = e.to_string(),
            }
            row
        })
        .collect();
    flag_frames(&mut rows);
    let flagged = rows.iter().filter(|r| !r.flags.is_empty()).count();
    println!("survey in {:.2?}: {flagged} of {total} frames flagged", t.elapsed());
    std::fs::write(json, serde_json::to_string(&serde_json::json!({ "frames": rows }))?)
        .with_context(|| format!("writing {}", json.display()))?;
    Ok(())
}

/// Put each frame's sharpness against its filter's median, and name the frames
/// that look worse than the rest of their filter.
///
/// Per filter, because the filters are not alike: an oxygen frame has fewer
/// and fainter stars than a hydrogen frame of the same sky, and held against
/// hydrogen every one of them would look clouded.
pub(crate) fn flag_frames(rows: &mut [SurveyRow]) {
    let mut filters: Vec<String> = rows.iter().map(|r| r.filter.clone()).collect();
    filters.sort();
    filters.dedup();
    for filter in filters {
        let group: Vec<usize> = (0..rows.len())
            .filter(|&i| rows[i].filter == filter && rows[i].error.is_empty())
            .collect();
        let starred: Vec<usize> = group.iter().copied().filter(|&i| rows[i].stars.is_some()).collect();
        // The rule the stack uses to decide what its own sharpness came from.
        let by_stars = !group.is_empty() && starred.len() * 4 >= group.len() * 3;
        let raw: Vec<f32> = group
            .iter()
            .filter(|&&i| rows[i].stars.is_some() == by_stars)
            .filter_map(|&i| rows[i].sharpness)
            .collect();
        let median = if raw.is_empty() { 0.0 } else { sr_core::math::median(&raw) };
        for &i in &group {
            let same_scale = rows[i].stars.is_some() == by_stars;
            rows[i].sharpness = match rows[i].sharpness {
                Some(s) if same_scale && median > 0.0 => Some(s / median),
                _ => None,
            };
        }
        if group.len() < SURVEY_MIN_GROUP || raw.is_empty() {
            continue;
        }
        let of = if filter.is_empty() { "the set".to_string() } else { format!("filter {filter}") };

        if by_stars {
            let ecc: Vec<f32> = starred.iter().filter_map(|&i| rows[i].eccentricity).collect();
            let med_ecc = sr_core::math::median(&ecc);
            let threshold = elongation_threshold(med_ecc);
            let counts: Vec<f32> =
                starred.iter().filter_map(|&i| rows[i].stars).map(|c| c as f32).collect();
            let med_count = sr_core::math::median(&counts);
            for &i in &group {
                let r = &mut rows[i];
                match (r.eccentricity, r.stars) {
                    (Some(e), Some(c)) => {
                        if e > threshold {
                            r.flags.push(SurveyFlag {
                                kind: "elongated",
                                text: format!(
                                    "elongated stars, eccentricity {e:.2} against {med_ecc:.2} \
                                     typical of {of}: wind, guiding or a snag"
                                ),
                            });
                        }
                        if med_count >= 10.0 && (c as f32) < FEW_STARS * med_count {
                            r.flags.push(SurveyFlag {
                                kind: "few-stars",
                                text: format!(
                                    "{c} stars against {med_count:.0} typical of {of}: cloud, \
                                     haze or dew"
                                ),
                            });
                        }
                    }
                    _ => r.flags.push(SurveyFlag {
                        kind: "no-stars",
                        text: format!(
                            "no stars found, where {of} has about {med_count:.0}: cloud, or \
                             pointed somewhere else"
                        ),
                    }),
                }
            }
        }
        for &i in &group {
            if let Some(s) = rows[i].sharpness.filter(|s| *s < SOFT) {
                rows[i].flags.push(SurveyFlag {
                    kind: "soft",
                    text: format!("soft, {s:.2}x the sharpness typical of {of}"),
                });
            }
        }
    }
    for r in rows.iter_mut().filter(|r| !r.error.is_empty()) {
        let text = format!("could not be read: {}", r.error);
        r.flags.push(SurveyFlag { kind: "unreadable", text });
    }
}

/// Which frames `--max-frames` keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameSelect {
    /// Filename order. Cheapest, and arbitrary with respect to quality.
    First,
    /// The sharpest N, found by surveying the burst first.
    Sharpest,
    /// N evenly spaced across the burst.
    Spread,
}

/// Choose which frames to keep.
///
/// Worth being explicit about what each mode costs. `First` is a truncation and
/// is not a quality judgement: on a burst whose camera position changes
/// part-way through, it can take everything from one half and almost nothing
/// from the other. `Sharpest` decodes the whole burst once to rank it and then
/// decodes the survivors again. `Spread` is free and keeps the burst's motion
/// envelope intact, which matters because that motion is the sub-pixel
/// diversity super-resolution runs on.
fn choose_frames(
    paths: &[PathBuf],
    keep: usize,
    select: FrameSelect,
    opts: &sr_raw::ReadOptions,
    stars: bool,
) -> Result<Vec<PathBuf>> {
    if keep >= paths.len() {
        return Ok(paths.to_vec());
    }
    let picked: Vec<usize> = match select {
        FrameSelect::First => (0..keep).collect(),
        FrameSelect::Spread => (0..keep)
            .map(|i| i * (paths.len() - 1) / (keep - 1).max(1))
            .collect(),
        FrameSelect::Sharpest => {
            log::info!(
                "surveying {} frames to find the sharpest {keep} (this decodes the \
                 burst an extra time)",
                paths.len()
            );
            let t = Instant::now();
            let survey = survey_all(paths, opts, stars)?;
            let med = sr_core::math::median(
                &survey.iter().map(|s| s.sharpness).collect::<Vec<_>>(),
            )
            .max(1e-20);
            let measured = survey.iter().filter(|s| s.stars.is_some()).count();
            let mut order: Vec<&FrameSurvey> = survey.iter().collect();
            order.sort_by(|a, b| {
                let sa = a.sharpness * (1.0 - a.saturation_fraction.min(1.0));
                let sb = b.sharpness * (1.0 - b.saturation_fraction.min(1.0));
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut idx: Vec<usize> = order.iter().take(keep).map(|s| s.index).collect();
            idx.sort_unstable();
            log::info!(
                "survey in {:.2?}: sharpness from {}; keeping frames from {:.2}x to {:.2}x \
                 the burst median",
                t.elapsed(),
                if measured * 4 >= survey.len() * 3 {
                    "point sources"
                } else {
                    "gradient energy"
                },
                survey[*idx.iter().min().unwrap()].sharpness / med,
                order[0].sharpness / med
            );
            idx
        }
    };

    // Temporal order is preserved whatever the selection, so that the reference
    // survey and every diagnostic still read in capture order.
    Ok(picked.into_iter().map(|i| paths[i].clone()).collect())
}

/// Everything the pipeline learns about a burst before reconstruction.
pub struct LoadedBurst {
    pub spill_dir: Option<PathBuf>,
    pub paths: Vec<PathBuf>,
    pub frames: Vec<RawFrame>,
    pub validation: sr_raw::BurstValidation,
    pub guide_luma: Vec<Plane<f32>>,
    pub qualities: Vec<FrameQuality>,
    pub noise: sr_core::frame::NoiseModel,
    pub noise_source: String,
    /// Point-source shape per frame, where the frame had point sources.
    pub stars: Vec<Option<StarMetrics>>,
    /// Whether `qualities[..].sharpness` is a half-flux diameter rather than a
    /// gradient energy. Only affects how the number should be described.
    pub sharpness_from_stars: bool,
    /// Frame index the caller insisted on as the reference, if any.
    pub forced_reference: Option<usize>,
    /// Whether that frame was added only to fix the grid, and so must not
    /// contribute its own photons to a merge it is not part of.
    pub reference_is_extra: bool,
}

fn human_bytes(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", U[i])
}

/// Where the frames come from and how to read them.
///
/// One struct rather than five parameters because every entry point needs all
/// of them and none of them alone.
#[derive(Clone, Copy, Debug)]
pub struct InputSpec<'a> {
    /// Verified, content-deduplicated project order, when supplied by a project.
    pub ordered_paths: Option<&'a [PathBuf]>,
    /// Private scratch storage for retained image buffers.
    pub spill_dir: Option<&'a Path>,
    /// A directory of frames, or a single file.
    pub path: &'a Path,
    /// Extension to match inside a directory. `None` takes every format we read.
    pub pattern: Option<&'a str>,
    pub max_frames: Option<usize>,
    pub select: FrameSelect,
    pub read: sr_raw::ReadOptions,
    /// Measure the shape of point sources and rank frames by it. Off falls
    /// back to gradient energy, which is what this did before.
    pub star_metrics: bool,
    /// Keep only frames taken through this filter. A narrowband set holds
    /// several, and they are separate images of the sky.
    pub filter: Option<&'a str>,
    /// Where to reuse alignment and defect scans between runs, if anywhere.
    pub cache_dir: Option<&'a Path>,
    /// Frame that fixes the output grid, whether or not it is one of the
    /// inputs.
    ///
    /// Without this every run picks its own reference, so two runs over
    /// different halves of a set produce images that cannot be added together.
    /// Naming one makes the grid a property of the *project* rather than of
    /// the batch, which is what lets a set too large for memory be
    /// reconstructed in pieces -- and what lets two people's data land on the
    /// same pixels.
    pub reference_file: Option<&'a Path>,
}

pub fn load_burst(spec: &InputSpec) -> Result<LoadedBurst> {
    let InputSpec { path: input, pattern, max_frames, select, read, star_metrics, filter, .. } =
        *spec;
    let read = &read;
    let mut all = match spec.ordered_paths {
        Some(paths) => paths.to_vec(),
        None => sr_raw::collect_files(input, pattern)?,
    };
    anyhow::ensure!(!all.is_empty(), "no input frames");
    if let Some(want) = filter {
        // Sorted by header rather than by filename: a capture program is free to
        // name files however it likes, and the filter is recorded in the frame.
        let before = all.len();
        all.retain(|p| {
            sr_raw::peek_filter(p)
                .map(|f| f.eq_ignore_ascii_case(want))
                .unwrap_or(false)
        });
        anyhow::ensure!(
            !all.is_empty(),
            "no frames in {} were taken through filter {want:?}",
            input.display()
        );
        log::info!("filter {want:?}: {} of {before} frames", all.len());
    }
    // Before anything is decoded, so a duplicated set costs nothing to notice
    // and the frame selection below chooses among real exposures.
    let (all, duplicates) = if spec.ordered_paths.is_some() {
        (all, Vec::new())
    } else {
        sr_raw::deduplicate(all)
    };
    if !duplicates.is_empty() {
        log::warn!(
            "{} of the frames listed are byte-for-byte copies of others and have been \
             dropped; {} exposures remain",
            duplicates.len(),
            all.len()
        );
        for d in duplicates.iter().take(3) {
            log::warn!(
                "  {} is the same file as {}",
                d.dropped.display(),
                d.same_as.display()
            );
        }
        if duplicates.len() > 3 {
            log::warn!("  ... and {} more", duplicates.len() - 3);
        }
    }

    let paths = match max_frames {
        Some(n) if n.max(1) < all.len() => {
            let keep = n.max(1);
            let chosen = choose_frames(&all, keep, select, read, star_metrics)?;
            log::info!(
                "using {keep} of {} frames, selected by {}",
                all.len(),
                match select {
                    FrameSelect::First => "filename order",
                    FrameSelect::Sharpest => "sharpness",
                    FrameSelect::Spread => "even spacing across the burst",
                }
            );
            chosen
        }
        _ => all,
    };

    // The frame that fixes the grid goes in front, so that its index is known
    // before anything is decoded. If it is already one of the inputs it keeps
    // its place in the merge; if it is not, it is carried for its geometry
    // alone and weighted out later.
    let (paths, forced_reference, reference_is_extra) = match spec.reference_file {
        Some(r) => {
            let canonical = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
            let want = canonical(r);
            match paths.iter().position(|p| canonical(p) == want) {
                Some(i) => {
                    let mut v = paths;
                    v.swap(0, i);
                    (v, Some(0), false)
                }
                None => {
                    anyhow::ensure!(
                        r.is_file(),
                        "the reference frame {} is not a file",
                        r.display()
                    );
                    let mut v = vec![r.to_path_buf()];
                    v.extend(paths);
                    log::info!(
                        "reference frame {} is not among the inputs; it fixes the output \
                         grid and contributes nothing to the merge",
                        r.display()
                    );
                    (v, Some(0), true)
                }
            }
        }
        None => (paths, None, false),
    };

    // Project the resident cost before committing to it. A burst is held in
    // full for the whole run, so the requirement scales with frame count and
    // can reach tens of gigabytes; an allocation failure part-way through is an
    // abort with no explanation, which is a poor way to learn this.
    let probe = sr_raw::decode_with(&paths[0], read)?;
    // The samples are the smaller half of it. Each frame also carries a guide
    // image at half pitch in f32 -- a quarter the pixels at twice the width, so
    // half the samples again -- and registration builds a pyramid over that,
    // which is another third on top. At 61 megapixels that is 122 MB of
    // samples and 140 MB of everything else, and projecting only the first
    // number is how a 294-frame batch that was reported as "31.3 GiB" aborted
    // on an allocation failure part way through.
    let samples = probe.bytes() as u64;
    let guide = (probe.width as u64 / 2) * (probe.height as u64 / 2) * 4;
    let per_frame = samples + guide + guide * 4 / 3;
    let projected = per_frame * paths.len() as u64;
    drop(probe);
    log::info!(
        "decoding {} frames from {} (about {} resident, {} per frame: {} of samples and \
         {} of guide and pyramid)",
        paths.len(),
        input.display(),
        human_bytes(projected),
        human_bytes(per_frame),
        human_bytes(samples),
        human_bytes(per_frame - samples)
    );
    if let Some(dir) = spec.spill_dir {
        std::fs::create_dir_all(dir)?;
        log::info!("retained sample, guide and pyramid buffers use private disk-backed storage at {}; OS paging still controls resident memory", dir.display());
    }
    if spec.spill_dir.is_none() && projected > 24 * 1024 * 1024 * 1024 {
        log::warn!(
            "this burst needs roughly {} of memory for the frames, their guides and their \
             pyramids, before any reconstruction buffers. If the run is killed or aborts on \
             an allocation failure, reduce --max-frames, reconstruct a region with --roi, or \
             stack it in batches with --accumulate and add them with `smokstak combine`.",
            human_bytes(projected)
        );
    }
    let t = Instant::now();
    let frames = if let Some(dir) = spec.spill_dir {
        paths.iter().map(|path| {
            let mut frame = sr_raw::decode_with(path, read)?;
            frame.samples.spill(dir)?;
            Ok(frame)
        }).collect::<Result<Vec<_>>>()?
    } else {
        sr_raw::decode_all(&paths, read)?
    };
    let bytes: u64 = frames.iter().map(|f| f.bytes() as u64).sum();
    log::info!(
        "decoded {} frames in {:.2?} ({} resident)",
        frames.len(),
        t.elapsed(),
        human_bytes(bytes)
    );

    // A frame carried only to fix the grid is not part of the burst and must
    // not be judged as if it were: it can be from another filter, another
    // night or another exposure length, and none of that makes the data
    // inconsistent. It is always first, so the rest validate on their own and
    // its entry is spliced back in as neutral.
    let validation = if reference_is_extra {
        let mut v = sr_raw::validate_burst(&frames[1..]);
        v.frame_count += 1;
        v.exposure_scale.insert(0, 1.0);
        v
    } else {
        sr_raw::validate_burst(&frames)
    };

    let t = Instant::now();
    let guide_luma: Vec<Plane<f32>> = frames.par_iter().map(|f| {
        let mut guide = f.guide_rgb().luma();
        if let Some(dir) = spec.spill_dir { guide.spill(dir)?; }
        Ok(guide)
    }).collect::<Result<_>>()?;
    log::info!("built registration proxies in {:.2?}", t.elapsed());

    let mut qualities: Vec<FrameQuality> = guide_luma
        .par_iter()
        .zip(frames.par_iter())
        .map(|(l, f)| sr_quality::analyse(l, f.saturation_fraction()))
        .collect();

    // Point sources, if this burst has any. Measured on the mosaic rather than
    // the guide, because a guide pixel is two sensor pixels and a star is not
    // much more than that.
    let t = Instant::now();
    let stars: Vec<Option<StarMetrics>> = if star_metrics {
        sr_quality::stars::measure_burst(&frames)
    } else {
        vec![None; frames.len()]
    };
    let sharpness_from_stars = sr_quality::apply_star_sharpness(&mut qualities, &stars);
    if sharpness_from_stars {
        let counts: Vec<f32> = stars.iter().flatten().map(|s| s.count as f32).collect();
        log::info!(
            "measured point sources in {:.2?}: {} of {} frames, {:.0} stars each (median); \
             frame sharpness is now a half-flux diameter",
            t.elapsed(),
            counts.len(),
            frames.len(),
            sr_core::math::median(&counts)
        );
    }
    sr_quality::normalise_against_median(&mut qualities);

    // This is the initial inspect/header summary. Reconstruction measures each
    // filter's usable members separately below. A grid-only frame contributes
    // no data and must not become even the summary's fallback noise model.
    let noise_frames = if reference_is_extra { &frames[1..] } else { &frames[..] };
    let (noise, noise_source) = sr_noise::estimate_burst(noise_frames);
    log::info!(
        "noise model: var = {:.3e} * x + {:.3e} (from {})",
        noise.alpha,
        noise.beta,
        noise_source
    );

    Ok(LoadedBurst {
        spill_dir: spec.spill_dir.map(Path::to_path_buf),
        paths,
        frames,
        validation,
        guide_luma,
        qualities,
        noise,
        noise_source,
        stars,
        sharpness_from_stars,
        forced_reference,
        reference_is_extra,
    })
}

/// Registration products shared by `register` and `stack`.
pub struct RegisteredBurst {
    pub choice: ReferenceChoice,
    pub registrations: Vec<GlobalRegistration>,
    /// Warps in full sensor coordinates.
    pub warps: Vec<WarpField>,
    pub local_warp_applied: bool,
    /// Median registration residual in sensor pixels.
    pub residual_sigma: f32,
}

/// What a cached registration holds.
///
/// The whole of [`RegisteredBurst`], which is a pure function of the frames and
/// the configuration fields named in [`registration_fingerprint`].
#[derive(serde::Serialize, serde::Deserialize)]
struct CachedRegistration {
    choice: ReferenceChoice,
    registrations: Vec<GlobalRegistration>,
    warps: Vec<WarpField>,
    local_warp_applied: bool,
    residual_sigma: f32,
}

/// Everything registration reads, and nothing else.
///
/// Deliberately not the whole configuration: the point of the cache is that
/// changing the scale or the kernel does not invalidate the alignment. The
/// configuration structs go in whole, so a field added to one of them changes
/// the fingerprint without anyone having to remember this function exists.
fn registration_fingerprint(burst: &LoadedBurst, cfg: &ReconstructionConfig) -> String {
    registration_fingerprint_of(&burst.frames, cfg, burst.sharpness_from_stars, burst.forced_reference)
}

fn registration_fingerprint_of(
    frames: &[RawFrame],
    cfg: &ReconstructionConfig,
    sharpness_from_stars: bool,
    forced_reference: Option<usize>,
) -> String {
    Fingerprint::new()
        .text("registration")
        .text("validated-stellar-distortion-v2")
        .text("isolated-mono-registration-v2")
        .frames(frames)
        .config(&cfg.registration)
        .config(&cfg.warp)
        .config(&cfg.local_warp)
        .config(&cfg.reference)
        // --reference-file is resolved during loading, outside cfg.reference.
        // Omitting it lets an automatic-reference cache silently override the named grid.
        .config(&forced_reference)
        // Reference selection ranks frames by quality, and which metric that is
        // depends on whether the burst had point sources to measure.
        .config(&sharpness_from_stars)
        .finish()
}

pub fn register_burst(
    burst: &LoadedBurst,
    cfg: &ReconstructionConfig,
    cache: &Cache,
) -> Result<RegisteredBurst> {
    let key = registration_fingerprint(burst, cfg);
    if let Some(c) = cache.load::<CachedRegistration>("registration", &key) {
        if c.warps.len() == burst.frames.len() && c.registrations.len() == burst.frames.len() {
            println!("Registration: reused from cache");
            return Ok(RegisteredBurst {
                choice: c.choice,
                registrations: c.registrations,
                warps: c.warps,
                local_warp_applied: c.local_warp_applied,
                residual_sigma: c.residual_sigma,
            });
        }
        log::warn!("cached registration is for a different frame count; recomputing");
    }
    let t = Instant::now();
    let proxies: Vec<RegistrationImage> = burst
        .frames
        .par_iter()
        .map(|f| {
            let mut proxy = RegistrationImage::build(&f.registration_luma(), cfg.registration.pyramid_levels);
            if let Some(dir) = burst.spill_dir.as_deref() {
                for level in &mut proxy.levels { level.spill(dir)?; }
            }
            Ok(proxy)
        })
        .collect::<Result<_>>()?;
    log::info!("built {} pyramids in {:.2?}", proxies.len(), t.elapsed());

    let choice = match cfg.reference.or(burst.forced_reference) {
        Some(i) => {
            anyhow::ensure!(
                i < burst.frames.len(),
                "--reference {i} is out of range for a {}-frame burst",
                burst.frames.len()
            );
            ReferenceChoice {
                index: i,
                reason: if burst.forced_reference == Some(i) && cfg.reference.is_none() {
                    "named with --reference-file, so that every batch of this project \
                     lands on the same grid"
                        .to_string()
                } else {
                    "specified on the command line".to_string()
                },
                positions: vec![(0.0, 0.0); burst.frames.len()],
                envelope_radius: 0.0,
                scores: vec![0.0; burst.frames.len()],
            }
        }
        None => {
            let t = Instant::now();
            let c = sr_register::reference::select_reference(
                &proxies,
                &burst.qualities,
                &cfg.registration,
            );
            log::info!("reference survey in {:.2?}", t.elapsed());
            c
        }
    };
    log::info!(
        "reference frame {} ({}): {}",
        choice.index,
        burst.frames[choice.index].metadata.file_name,
        choice.reason
    );

    let t = Instant::now();
    let seeds = registration_seeds(burst, choice.index);
    let registrations = sr_register::global::register_burst_seeded(
        &proxies,
        choice.index,
        &cfg.registration,
        &seeds,
    );
    log::info!("global registration in {:.2?}", t.elapsed());

    // Proxy coordinates are half the sensor pitch in each axis.
    let mut globals: Vec<sr_core::geometry::GlobalTransform> =
        registrations.iter().map(|r| r.transform.rescale(2.0)).collect();

    // Polish each placement against the stars themselves.
    //
    // Correlation finds the frame -- through a meridian flip, through a
    // hundred pixels of drift -- and leaves it about a pixel out. A pixel of
    // scatter in where each frame is put is a pixel of blur in the stack: on a
    // 96-frame burst the stacked star came out half again as wide as a single
    // frame's.
    let t_stars = Instant::now();
    let star_lists: Vec<Vec<sr_core::star::Star>> = (0..burst.frames.len())
        .into_par_iter()
        .map(|i| sr_quality::stars::positions(&burst.frames[i], STAR_REFINE_LIMIT))
        .collect();
    let refinements =
        sr_register::refine::refine_against_stars(choice.index, &star_lists, &globals);
    let mut refined = 0usize;
    let mut before = Vec::new();
    let mut after = Vec::new();
    for (i, (correction, report)) in refinements.iter().enumerate() {
        if report.applied {
            globals[i] = correction.compose(&globals[i]);
            refined += 1;
            before.push(report.before);
            after.push(report.after);
        }
    }
    if refined > 0 {
        log::info!(
            "star refinement: {refined} of {} frames polished in {:.2?}, median pair \
             separation {:.2} px before and {:.2} after",
            burst.frames.len(),
            t_stars.elapsed(),
            sr_core::math::median(&before),
            sr_core::math::median(&after)
        );
    } else if !star_lists[choice.index].is_empty() {
        log::info!("star refinement: nothing to polish");
    }
    {
        // Why a frame was left alone matters as much as the median. A frame
        // declined because it was already placed is a success; one declined
        // because the pairs disagreed is a frame still carrying its error into
        // the stack.
        let (mut few, mut big, mut no_gain, mut already) = (0, 0, 0, 0);
        for (i, (_, r)) in refinements.iter().enumerate() {
            if r.applied || i == choice.index {
                continue;
            }
            if r.pairs < 25 {
                few += 1;
            } else if r.worst > 6.0 {
                big += 1;
            } else if r.before < 0.3 {
                already += 1;
            } else {
                no_gain += 1;
            }
            log::debug!(
                "  frame {i}: {} pairs, {:.2} px before, {:.2} after, moved at most {:.2}",
                r.pairs,
                r.before,
                r.after,
                r.worst
            );
        }
        if few + big + no_gain + already > 0 {
            log::info!(
                "star refinement declined: {few} with too few pairs, {big} asking too \
                 large a move, {no_gain} no closer after fitting, {already} already placed"
            );
        }
    }

    let mut warps: Vec<WarpField> = globals
        .iter()
        .map(|g| WarpField::global_only(*g))
        .collect();

    if !matches!(cfg.local_warp, sr_core::config::LocalWarpMode::Off) {
        let (w, h) = proxies[choice.index].dims();
        let fields = sr_register::distortion::refine_stars(
            choice.index, &star_lists, &globals, w*2, h*2,
        );
        for (warp, field) in warps.iter_mut().zip(fields) {
            warp.local = field;
        }
    }

    let t = Instant::now();
    let stellar_fields: Vec<_> = warps.iter().map(|w| w.local.is_some()).collect();
    sr_warp::maybe_refine(&proxies, choice.index, &registrations, cfg, &mut warps);
    // Auto correlation acceptance must also agree with point-source geometry.
    // Explicit `on` keeps its documented force behavior for diagnostic trials.
    if matches!(cfg.local_warp, sr_core::config::LocalWarpMode::Auto) {
        let (w,h) = proxies[choice.index].dims();
        let mut vetoed = 0;
        for (i,warp) in warps.iter_mut().enumerate() {
            if stellar_fields[i] { continue; }
            if let Some(field) = &warp.local
                && sr_register::distortion::validates_correlation(
                    &star_lists[choice.index],&star_lists[i],&warp.global,field,w*2,h*2,
                ) == Some(false) {
                    warp.local = None;
                    vetoed += 1;
                }
        }
        if vetoed > 0 { log::info!("stellar validation rejected {vetoed} correlation fields"); }
    }
    let local_warp_applied = warps.iter().any(|w| w.local.is_some());
    if local_warp_applied {
        log::info!("local warp refinement in {:.2?}", t.elapsed());
    }

    let rms: Vec<f32> = registrations
        .iter()
        .filter(|r| r.probes > 0)
        .map(|r| r.residual_rms)
        .collect();
    // Proxy pixels are half a sensor pixel.
    let residual_sigma = if rms.is_empty() { 0.5 } else { sr_core::math::median(&rms) * 2.0 };

    cache.store(
        "registration",
        &key,
        &CachedRegistration {
            choice: choice.clone(),
            registrations: registrations.clone(),
            warps: warps.clone(),
            local_warp_applied,
            residual_sigma,
        },
    );
    Ok(RegisteredBurst { choice, registrations, warps, local_warp_applied, residual_sigma })
}

/// Whether a scale above 1.0 has anything to recover, as opposed to enough
/// sampling diversity to attempt it.
///
/// These are two different questions and only one of them was being asked. The
/// coverage analysis measures how many distinct sub-pixel phases the burst
/// visited, which says whether a finer grid can be *filled*. It says nothing
/// about whether there is detail to put in it. Super-resolution recovers
/// information that aliasing folded down, and aliasing only happens when the
/// point-spread function is narrower than two pixels. Above that the frames
/// are already sampled above Nyquist, nothing is folded, and a finer grid
/// resamples what is there rather than resolving more.
///
/// Measured on a 68-hour narrowband burst whose stars are 2.85 px across:
/// reconstructing at 2x raised the raw power in the 0.2 to 0.45 cycles per
/// sensor pixel band by 1.4 to 1.9 times and left the signal-to-noise in that
/// band flat, +10% at the low end and -11% at the high. Four times the output
/// and four times the merge for nothing. Its coverage analysis said "2.00x
/// supported" throughout, and was right about the question it answers.
fn report_sampling_headroom(burst: &LoadedBurst, scale: f32, warnings: &mut Vec<String>) {
    if scale <= 1.05 {
        return;
    }
    let hfd: Vec<f32> = burst.stars.iter().flatten().map(|m| m.hfd).collect();
    if hfd.len() * 2 < burst.frames.len() {
        // Too few frames with measurable point sources to say anything. A
        // daytime scene has no PSF to measure and this test does not apply.
        return;
    }
    let median = sr_core::math::median(&hfd);
    // A Gaussian's half-flux diameter is its full width at half maximum, and
    // Nyquist wants two samples across that.
    let headroom = (2.0 / median.max(1e-3)).clamp(1.0, 4.0);
    if headroom >= scale - 0.05 {
        println!(
            "Sampling headroom: point sources are {median:.2} px across, so the optics \
             support about {headroom:.2}x."
        );
        return;
    }
    let note = format!(
        "point sources are {median:.2} px across (half-flux diameter), which is already \
         sampled above Nyquist: there is no aliased detail for {scale:.2}x to unfold, so a \
         finer grid will resample rather than resolve. Sub-pixel diversity is a different \
         question, and a burst can have plenty of it and nothing to recover"
    );
    println!("Sampling headroom: {note}");
    warnings.push(note);
}

/// A starting estimate for each frame, from the plate solves in the headers.
///
/// Correlation refines; it does not search. A burst whose frames were taken
/// either side of a meridian flip contains two groups 180 degrees apart, and
/// from a standing start the second group registers to nothing: on the
/// 407-frame NGC 6871 set that was 28% of the integration, silently weighted
/// out of the merge with a confidence of 0.003.
///
/// The solve says where each frame was pointed, so the relative map is
/// arithmetic. It is only ever offered — `register_burst_seeded` keeps it only
/// if it beats registering from the identity — because a header is a claim,
/// not a measurement.
/// A starting estimate for every frame that needs one, from whatever can
/// supply it.
///
/// The plate solve first, because it is free. Star patterns for the rest,
/// because plenty of data has no solve: on a two-season set of NGC 7023, 657
/// frames of the first season carry none at all and sit 96 degrees rotated
/// from the second, with a 1% difference in plate scale. Neither is something
/// correlation can find, and the header cannot help with a frame that has no
/// header to help with.
///
/// Matching stars costs a detection pass over the burst, so it is only done
/// when the solves have left frames unaccounted for.
fn registration_seeds(
    burst: &LoadedBurst,
    reference: usize,
) -> Vec<Option<sr_core::geometry::GlobalTransform>> {
    let mut seeds = plate_solve_seeds(burst, reference);
    let missing: Vec<usize> = (0..burst.frames.len())
        .filter(|&i| i != reference && seeds[i].is_none())
        .collect();
    if missing.is_empty() {
        return seeds;
    }

    // Which stars are the brightest depends on the filter, because stars have
    // colours: a red star leads the list through R and a blue one through B.
    // Matching a pattern across filters therefore compares two different
    // selections of the sky and mostly fails -- on the Iris set, 28 of 151
    // frames placed against a luminance reference where 6 of 6 placed against
    // one of their own filter, and the run that followed was worthless.
    let ref_filter = burst.frames[reference].metadata.filter.clone();
    let mismatched = missing
        .iter()
        .filter(|&&i| {
            let f = &burst.frames[i].metadata.filter;
            f.is_some() && ref_filter.is_some() && f != &ref_filter
        })
        .count();
    if mismatched * 2 > missing.len() {
        log::warn!(
            "{mismatched} frames need placing by their stars against a reference taken \
             through a different filter ({} rather than theirs). Which stars are brightest \
             depends on the filter, so most will not match: name a reference of their own \
             filter with --reference-file.",
            ref_filter.as_deref().unwrap_or("none")
        );
    }

    let t = Instant::now();
    // Sensor coordinates, where the star detector works; registration works on
    // the guide, at half that pitch, so only the translation changes.
    let ref_stars = sr_quality::stars::positions(&burst.frames[reference], STAR_MATCH_LIMIT);
    if ref_stars.len() < 12 {
        log::info!("no star pattern in the reference frame to match against");
        return seeds;
    }
    let found: Vec<(usize, Option<sr_register::asterism::Match>)> = missing
        .par_iter()
        .map(|&i| {
            let stars = sr_quality::stars::positions(&burst.frames[i], STAR_MATCH_LIMIT);
            let m = sr_register::asterism::match_stars(&ref_stars, &stars);
            if let Some(m) = m {
                log::debug!(
                    "frame {i}: placed by {} stars, turned {:.2} deg, residual {:.2} px",
                    m.pairs, m.rotation_deg, m.residual
                );
            }
            (i, m)
        })
        .collect();

    let mut matched = 0usize;
    let mut worst_rotation = 0.0f32;
    for (i, m) in found {
        if let Some(m) = m {
            seeds[i] = Some(m.transform.rescale(0.5));
            matched += 1;
            let r = m.rotation_deg.abs().min(360.0 - m.rotation_deg.abs());
            worst_rotation = worst_rotation.max(r);
        }
    }
    log::info!(
        "star patterns: {matched} of {} frames without a plate solve were placed by their \
         stars in {:.2?}, the furthest turned by {worst_rotation:.1} degrees",
        missing.len(),
        t.elapsed()
    );
    seeds
}

/// Stars taken from each frame for the photometric gain.
///
/// Fewer than the refinement uses: this is a median of flux ratios and a few
/// hundred settle it, while every extra star is an aperture measured.
const STAR_PHOTOMETRY_LIMIT: usize = 600;

/// Stars taken from each frame for the refinement fit.
///
/// Far more than the pattern matcher uses, because this is not a search: the
/// frame is already placed and every star that pairs is another constraint on
/// six numbers. The cost is a nearest-neighbour lookup each, which is nothing.
const STAR_REFINE_LIMIT: usize = 3000;

/// Stars taken from each frame for pattern matching. The bright end is what two
/// frames of one field reliably share, and matching cost grows with the count.
const STAR_MATCH_LIMIT: usize = 200;

fn plate_solve_seeds(
    burst: &LoadedBurst,
    reference: usize,
) -> Vec<Option<sr_core::geometry::GlobalTransform>> {
    let Some(ref_wcs) = burst.frames[reference].metadata.wcs else {
        return vec![None; burst.frames.len()];
    };
    let (w, h) = (burst.frames[reference].width, burst.frames[reference].height);
    let mut solved = 0usize;
    let seeds: Vec<Option<sr_core::geometry::GlobalTransform>> = burst
        .frames
        .iter()
        .map(|f| {
            let wcs = f.metadata.wcs?;
            let a = sr_core::wcs::relative_affine(&ref_wcs, &wcs, w, h)?;
            solved += 1;
            // The affine is in sensor pixels; registration works on the guide,
            // which is half that pitch, so only the translation changes.
            Some(
                sr_core::geometry::GlobalTransform {
                    m: [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32, a[4] as f32, a[5] as f32],
                }
                .rescale(0.5),
            )
        })
        .collect();
    if solved > 1 {
        // Frames facing the other way are the ones this exists for, so say how
        // many there are rather than leaving it to be inferred from a residual.
        let flipped = burst
            .frames
            .iter()
            .filter_map(|f| f.metadata.wcs)
            .filter(|w| {
                let d = (w.orientation_deg() - ref_wcs.orientation_deg()).abs();
                let d = if d > 180.0 { 360.0 - d } else { d };
                d > 90.0
            })
            .count();
        log::info!(
            "plate solves: {solved} of {} frames carry one; {flipped} are pointed the other \
             way up from the reference",
            burst.frames.len()
        );
    }
    seeds
}

fn report_header(burst: &LoadedBurst) -> String {
    let mut s = burst.validation.report();
    s.push_str(&format!(
        "\nNoise model:     var = {:.3e} * x + {:.3e}  ({})\n",
        burst.noise.alpha, burst.noise.beta, burst.noise_source
    ));
    let f = &burst.frames[0];
    s.push_str(&format!(
        "Sensor:          {}x{} active, CFA {}, black {:.0}, white {:.0}\n",
        f.width,
        f.height,
        f.cfa.name(),
        f.metadata.black_levels[0],
        f.metadata.white_level
    ));
    s
}

fn quality_table(burst: &LoadedBurst) -> String {
    let mut out = String::from("\nFrame quality (sharpness relative to burst median");
    out.push_str(if burst.sharpness_from_stars {
        ", from point sources):\n"
    } else {
        ", from gradient energy):\n"
    });
    let mut order: Vec<usize> = (0..burst.frames.len()).collect();
    order.sort_by(|&a, &b| {
        burst.qualities[b]
            .sharpness
            .partial_cmp(&burst.qualities[a].sharpness)
            .unwrap()
    });
    let meta = &burst.frames[0].metadata;
    let show = |i: usize| {
        let q = &burst.qualities[i];
        match &burst.stars[i] {
            Some(m) => {
                let arcsec = match m.hfd_arcsec(meta.pixel_pitch_um, meta.focal_length) {
                    Some(a) => format!("{a:>5.2}\""),
                    None => "      ".to_string(),
                };
                format!(
                    "  {:>4}  {:<16} sharpness {:>5.2}x  hfd {:>5.2} px {}  ecc {:>4.2}  \
                     {:>4} stars  sat {:>6.3}%\n",
                    i,
                    burst.frames[i].metadata.file_name,
                    q.sharpness,
                    m.hfd,
                    arcsec,
                    m.eccentricity,
                    m.count,
                    q.saturation_fraction * 100.0
                )
            }
            None => format!(
                "  {:>4}  {:<16} sharpness {:>5.2}x  contrast {:>7.4}  blur {:>5.2}  \
                 sat {:>6.3}%\n",
                i,
                burst.frames[i].metadata.file_name,
                q.sharpness,
                q.contrast,
                q.estimated_blur,
                q.saturation_fraction * 100.0
            ),
        }
    };
    for &i in order.iter().take(3) {
        out.push_str(&show(i));
    }
    if order.len() > 6 {
        out.push_str("   ...\n");
    }
    for &i in order.iter().rev().take(3).collect::<Vec<_>>().iter().rev() {
        out.push_str(&show(*i));
    }
    out
}

/// Fit and remove vignetting and a sky gradient, reporting what was taken out.
///
/// Returns the model when one was applied, so that it can be written out for
/// inspection. The check a reader should make is not on the coefficients: it is
/// whether the rendered model looks like optics or like their subject.
fn flatten_background(
    rgb: &mut [Plane<f32>; 3],
    channels: usize,
    warnings: &mut Vec<String>,
) -> Option<sr_reconstruct::background::BackgroundModel> {
    let Some(m) = sr_reconstruct::background::fit(rgb, channels) else {
        log::warn!("no background model could be fitted; the result is unchanged");
        return None;
    };
    println!("\n{}", m.describe());
    // Half a percent over the whole frame is not worth a correction, and
    // claiming to have flattened something already flat is worse than saying
    // nothing.
    let magnitude = sr_reconstruct::background::magnitude(&m);
    if magnitude < 0.005 {
        println!("  already flat to within half a percent; left alone");
        return None;
    }
    sr_reconstruct::background::apply(rgb, &m);
    warnings.push(format!(
        "background flattened: {:.1}% of the background level was modelled as vignetting and \
         sky gradient and removed. A model cannot tell those from large-scale structure in \
         the subject; check background-model.tif",
        magnitude * 100.0
    ));
    Some(m)
}

/// Say what the point sources looked like, and name the frames that are soft
/// for a reason worth knowing about.
///
/// A soft frame is not interesting on its own — seeing varies and a stacker
/// weights frames by quality anyway. A frame whose stars are *elongated* is
/// interesting, because the cause is mechanical: wind, a guiding error, a cable
/// snagging, field rotation from a mount that is not aligned. That is something
/// an operator can go and fix before the next session, and it is invisible in
/// any single number about sharpness.
fn report_stars(burst: &LoadedBurst, warnings: &mut Vec<String>) {
    let measured: Vec<&StarMetrics> = burst.stars.iter().flatten().collect();
    if measured.len() * 4 < burst.frames.len() * 3 {
        return;
    }
    let meta = &burst.frames[0].metadata;
    let hfd: Vec<f32> = measured.iter().map(|m| m.hfd).collect();
    let ecc: Vec<f32> = measured.iter().map(|m| m.eccentricity).collect();
    let med_hfd = sr_core::math::median(&hfd);
    let med_ecc = sr_core::math::median(&ecc);
    let arcsec = StarMetrics { hfd: med_hfd, ..Default::default() }
        .hfd_arcsec(meta.pixel_pitch_um, meta.focal_length)
        .map(|a| format!(" ({a:.2} arcsec)"))
        .unwrap_or_default();

    println!(
        "\nPoint sources: {:.0} stars per frame, half-flux diameter {med_hfd:.2} px{arcsec} \
         median, {:.2} to {:.2} across the burst",
        sr_core::math::median(&measured.iter().map(|m| m.count as f32).collect::<Vec<_>>()),
        hfd.iter().cloned().fold(f32::MAX, f32::min),
        hfd.iter().cloned().fold(0.0f32, f32::max),
    );

    // Elongation is judged against the burst, not against a fixed number.
    let threshold = elongation_threshold(med_ecc);
    let bad: Vec<usize> = burst
        .stars
        .iter()
        .enumerate()
        .filter(|(_, s)| s.map(|m| m.eccentricity > threshold).unwrap_or(false))
        .map(|(i, _)| i)
        .collect();
    if !bad.is_empty() {
        let angles: Vec<f32> = bad
            .iter()
            .filter_map(|&i| burst.stars[i].map(|m| m.angle_deg))
            .collect();
        println!(
            "  frames {bad:?} have elongated stars (eccentricity above {threshold:.2}, \
             burst median {med_ecc:.2}, elongated near {:.0} deg)",
            sr_core::math::median(&angles)
        );
        warnings.push(format!(
            "{} frame(s) show elongated stars against a burst median eccentricity of \
             {med_ecc:.2}; that is a mount or wind problem rather than seeing",
            bad.len()
        ));
    }
}

/// Say what the photometric match did, and warn when it did a lot.
///
/// A burst that needed a large correction is telling the operator something
/// about the shoot — cloud, moonrise, a lens cap of dew — and burying that in a
/// gain the merge quietly applied would be the wrong kind of automatic.
fn report_photometry(
    matches: &[FramePhotometry],
    burst: &LoadedBurst,
    warnings: &mut Vec<String>,
) {
    let level = sr_core::math::median(
        &burst.qualities.iter().map(|q| q.mean_level).collect::<Vec<_>>(),
    )
    .max(1e-4);
    let changes: Vec<f32> = matches.iter().map(|m| m.map.relative_change(level)).collect();
    let worst = changes.iter().cloned().fold(0.0f32, f32::max);
    let fell_back: Vec<usize> = matches
        .iter()
        .enumerate()
        .filter(|(_, m)| m.source == PhotometrySource::Exposure)
        .map(|(i, _)| i)
        .collect();

    println!(
        "\nPhotometric match: brightness varies by up to {:.1}% across the burst, \
         normalised onto the reference",
        worst * 100.0
    );
    if !fell_back.is_empty() {
        println!(
            "  no fit from the pixels for frame(s) {fell_back:?}; the exposure metadata was \
             used instead"
        );
    }
    // What the flat part of the match could not do. Reported separately
    // because it answers a different question: not "was the sky brighter"
    // but "was it brighter at one end of the frame than the other".
    let fields: Vec<f32> = matches
        .iter()
        .filter(|m| m.map.varies_across_frame())
        .map(|m| {
            (0..3)
                .map(|c| m.map.field_amplitude(c))
                .fold(0.0f32, f32::max)
        })
        .collect();
    if !fields.is_empty() {
        let worst_field = fields.iter().cloned().fold(0.0f32, f32::max);
        println!(
            "  {} frame(s) needed a sky gradient corrected across the frame as well, the \
             largest {:.1}% of the background between its lightest place and its darkest",
            fields.len(),
            100.0 * worst_field / level.max(1e-9)
        );
    }
    // Parts of frames that were not sky at all.
    let obstructed: Vec<(usize, f32)> = matches
        .iter()
        .enumerate()
        .filter(|(_, m)| m.map.blocked_fraction() > 0.0)
        .map(|(i, m)| (i, m.map.blocked_fraction()))
        .collect();
    if !obstructed.is_empty() {
        let worst = obstructed.iter().map(|&(_, f)| f).fold(0.0f32, f32::max);
        let frames: Vec<usize> = obstructed.iter().map(|&(i, _)| i).collect();
        println!(
            "  {} frame(s) had something between the sensor and the sky -- a tree, a roof, \
             thick cloud -- over up to {:.0}% of the frame; those regions were left out: \
             frames {:?}",
            obstructed.len(),
            worst * 100.0,
            frames
        );
        if worst > 0.5 {
            warnings.push(format!(
                "{} frame(s) were more than half obstructed and contributed only what was \
                 clear",
                obstructed.iter().filter(|&&(_, f)| f > 0.5).count()
            ));
        }
    }

    if worst > 0.10 {
        warnings.push(format!(
            "frame brightness varies by up to {:.1}% across the burst; the frames were \
             normalised onto the reference, but a change that large usually means cloud, \
             moonlight or dew",
            worst * 100.0
        ));
    }
}

/// Find the sensor's fixed-pattern defects and mask them in every frame.
///
/// Gated on the burst having moved. The scan separates a defect from a star by
/// asking whether the excess stays put while the scene does not, and on a burst
/// with no motion that question has no answer: every star is fixed to the
/// sensor too. Rather than quietly delete the sky, this declines and says so.
fn reset_decode_masks(frames: &mut [RawFrame], original: &[sr_core::samples::DefectMask]) {
    assert_eq!(frames.len(), original.len());
    for (frame, mask) in frames.iter_mut().zip(original) {
        frame.defects = mask.clone();
    }
}

/// Add this filter's sensor evidence to original decode masks, preserving both
/// independent decode failures and masks belonging to other filter members.
fn install_group_mask(frames: &mut [RawFrame], members: &[usize], mask: &sr_core::samples::DefectMask) {
    let needs_union = members.iter().any(|&i| !frames[i].defects.is_empty());
    let sites: Vec<usize> = if needs_union {
        (0..mask.width * mask.height).filter(|&i| mask.get(i)).collect()
    } else { Vec::new() };
    for &i in members {
        if frames[i].defects.is_empty() {
            frames[i].defects = mask.clone();
        } else {
            for &site in &sites { frames[i].defects.set(site); }
        }
    }
}

fn apply_defect_mask(
    burst: &mut LoadedBurst,
    reg: &RegisteredBurst,
    group: &FilterGroup,
    cache: &Cache,
    warnings: &mut Vec<String>,
) {
    if group.members.len() < sr_noise::defects::MIN_FRAMES {
        return;
    }
    // The scan reads the group's frames and the noise model, and is gated on
    // the motion the registration found — so the registration's own
    // fingerprint goes in rather than being restated.
    let key = Fingerprint::new()
        .text("defects-persistent-mono-four-v5-group-original-masks")
        .text(&registration_fingerprint(burst, &ReconstructionConfig {
            reference: Some(reg.choice.index),
            ..Default::default()
        }))
        .frames(&burst.frames)
        .indices(&group.members)
        .config(&burst.noise)
        .finish();
    if let Some(c) = cache.load::<CachedDefects>("defects", &key) {
        if c.width == burst.frames[0].width && c.height == burst.frames[0].height {
            println!(
                "Fixed-pattern defects: {} sites masked ({} hot, {} cold), reused from cache",
                c.sites.len(),
                c.hot,
                c.cold
            );
            let mask = c.to_mask();
            install_group_mask(&mut burst.frames, &group.members, &mask);
            return;
        }
        log::warn!("cached defect map is for a different sensor size; rescanning");
    }
    // Motion in sensor pixels, measured where the frames actually differ: the
    // spread of frame centres, not any one frame's displacement.
    let xs: Vec<f32> = group
        .members
        .iter()
        .map(|&i| reg.registrations[i].centre_shift.0 * 2.0)
        .collect();
    let ys: Vec<f32> = group
        .members
        .iter()
        .map(|&i| reg.registrations[i].centre_shift.1 * 2.0)
        .collect();
    let (cx, cy) = (sr_core::math::median(&xs), sr_core::math::median(&ys));
    let spread: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .collect();
    let motion = sr_core::math::median(&spread);
    // One sensor pixel. Below that a point source stays on the same site from
    // frame to frame and is indistinguishable from a defect.
    if motion < 1.0 {
        warnings.push(format!(
            "the burst moved only {motion:.2} px between frames, too little to tell a hot \
             pixel from a point source, so fixed-pattern defects were not masked"
        ));
        return;
    }

    let members: Vec<&RawFrame> = group.members.iter().map(|&i| &burst.frames[i]).collect();
    let (mask, report) = sr_noise::defects::find_fixed_pattern(&members, &burst.noise);
    println!("{}", report.describe());
    // A sensor with more than a few tenths of a percent bad is not a sensor
    // with bad pixels, it is a detector that has misfired.
    if report.fraction > 0.005 {
        warnings.push(format!(
            "{:.2}% of sites were called defective, which is far more than a sensor has; \
             the mask was discarded",
            report.fraction * 100.0
        ));
        return;
    }
    if report.total() == 0 {
        return;
    }
    cache.store(
        "defects",
        &key,
        &CachedDefects::from_mask(&mask, report.hot, report.cold),
    );
    install_group_mask(&mut burst.frames, &group.members, &mask);
}

fn registration_summary(burst: &LoadedBurst, reg: &RegisteredBurst) -> String {
    let mut out = String::new();
    let shifts: Vec<f32> = reg
        .registrations
        .iter()
        .map(|r| (r.centre_shift.0.powi(2) + r.centre_shift.1.powi(2)).sqrt() * 2.0)
        .collect();
    let failed: Vec<usize> = reg
        .registrations
        .iter()
        .filter(|r| r.confidence < 0.25)
        .map(|r| r.frame)
        .collect();

    out.push_str(&format!(
        "\nReference frame: {} ({})\n  {}\n",
        reg.choice.index,
        burst.frames[reg.choice.index].metadata.file_name,
        reg.choice.reason
    ));
    out.push_str(&format!(
        "\nBurst motion (sensor pixels):\n  median {:.2}, max {:.2}, envelope radius {:.2}\n",
        sr_core::math::median(&shifts),
        shifts.iter().cloned().fold(0.0f32, f32::max),
        reg.choice.envelope_radius * 2.0
    ));
    let mut models: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &reg.registrations {
        *models.entry(r.model.name()).or_insert(0) += 1;
    }
    out.push_str(&format!(
        "  transform models chosen: {}\n",
        models
            .into_iter()
            .map(|(k, v)| format!("{k} ({v})"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let rms: Vec<f32> = reg.registrations.iter().map(|r| r.residual_rms * 2.0).collect();
    out.push_str(&format!(
        "  registration residual: median {:.3} px, worst {:.3} px (sensor scale)\n",
        sr_core::math::median(&rms),
        rms.iter().cloned().fold(0.0f32, f32::max)
    ));
    if reg.local_warp_applied {
        let mean: f32 = reg.warps.iter().map(|w| w.max_local()).sum::<f32>()
            / reg.warps.len().max(1) as f32;
        out.push_str(&format!(
            "  local warp: applied, mean peak displacement {mean:.2} sensor px\n"
        ));
    } else {
        out.push_str("  local warp: not applied\n");
    }
    if !failed.is_empty() {
        out.push_str(&format!(
            "  {} frame(s) registered poorly: {:?}\n",
            failed.len(),
            failed
        ));
    }
    out
}

pub fn inspect(
    spec: &InputSpec,
    quick: bool,
    json: Option<&Path>,
) -> Result<()> {
    let burst = load_burst(spec)?;
    let mut out = report_header(&burst);
    out.push_str(&quality_table(&burst));

    if quick {
        print!("{out}");
        return Ok(());
    }

    let cfg = ReconstructionConfig::default();
    let reg = register_burst(&burst, &cfg, &Cache::new(spec.cache_dir))?;
    out.push_str(&registration_summary(&burst, &reg));

    let coverage = sr_reconstruct::coverage::analyse_coverage(&burst.frames, &reg.warps, 2.0, 8);
    out.push('\n');
    out.push_str(&coverage.describe());
    out.push('\n');

    let ca = estimate_chromatic_aberration(&burst, reg.choice.index, &frame_weights(&burst, &reg));
    out.push('\n');
    out.push_str(&ca.describe());

    print!("{out}");

    if let Some(p) = json {
        let doc = serde_json::json!({
            "input": spec.path.display().to_string(),
            "frames": burst.frames.len(),
            "validation": burst.validation,
            "noise": {
                "alpha": burst.noise.alpha,
                "beta": burst.noise.beta,
                "source": burst.noise_source,
            },
            "reference": { "index": reg.choice.index, "reason": reg.choice.reason },
            "registrations": reg.registrations,
            "coverage": coverage,
            "chromatic_aberration": ca,
        });
        std::fs::write(p, serde_json::to_string_pretty(&doc)?)
            .with_context(|| format!("writing {}", p.display()))?;
        log::info!("wrote {}", p.display());
    }
    Ok(())
}

fn frame_rows(
    burst: &LoadedBurst,
    reg: &RegisteredBurst,
    rejected: Option<&[f32]>,
    used: &[bool],
    photometry: &[FramePhotometry],
) -> Vec<FrameRow> {
    // A monochrome burst's photometric match lives in channel 0; a mosaic's is
    // reported from green. Reading green either way meant every monochrome run
    // reported a gain of 1.0 and an offset of 0.0 no matter what the match
    // actually did — the merge used the right channel, but the CSV that exists
    // to check the merge did not.
    let pc = if burst.frames.first().map(|f| f.is_mono()).unwrap_or(false) { 0 } else { 1 };
    (0..burst.frames.len())
        .map(|i| {
            let r = &reg.registrations[i];
            let q = &burst.qualities[i];
            let (lw_mean, lw_max) = match &reg.warps[i].local {
                Some(d) => (d.mean_magnitude(), d.max_magnitude()),
                None => (0.0, 0.0),
            };
            FrameRow {
                index: i,
                file: burst.frames[i].metadata.file_name.clone(),
                used: used[i],
                sharpness: q.sharpness,
                contrast: q.contrast,
                estimated_blur: q.estimated_blur,
                blur_anisotropy: q.blur_anisotropy,
                saturation_fraction: q.saturation_fraction,
                star_hfd: burst.stars[i].map(|m| m.hfd).unwrap_or(0.0),
                star_eccentricity: burst.stars[i].map(|m| m.eccentricity).unwrap_or(0.0),
                star_count: burst.stars[i].map(|m| m.count).unwrap_or(0),
                exposure_scale: burst.validation.exposure_scale[i],
                photometric_gain: photometry.get(i).map(|p| p.map.gain[pc]).unwrap_or(1.0),
                photometric_offset: photometry.get(i).map(|p| p.map.offset[pc]).unwrap_or(0.0),
                photometric_source: photometry
                    .get(i)
                    .map(|p| p.source.name())
                    .unwrap_or("unknown")
                    .to_string(),
                transform_model: r.model.name().to_string(),
                // Reported in sensor pixels, which is what the user thinks in.
                shift_x: r.centre_shift.0 * 2.0,
                shift_y: r.centre_shift.1 * 2.0,
                rotation_deg: r.rotation_deg,
                scale: r.scale,
                residual_rms: r.residual_rms * 2.0,
                residual_p90: r.residual_p90 * 2.0,
                inliers: r.inliers,
                probes: r.probes,
                overlap: r.overlap,
                confidence: r.confidence,
                local_warp_mean: lw_mean,
                local_warp_max: lw_max,
                robustness_rejected: rejected.map(|v| v[i]).unwrap_or(0.0),
            }
        })
        .collect()
}

fn write_registration_diagnostics(
    dir: &Path,
    burst: &LoadedBurst,
    reg: &RegisteredBurst,
    roi: Option<(usize, usize, usize, usize)>,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;

    let used = vec![true; burst.frames.len()];
    let reference = &burst.frames[reg.choice.index];
    let reference_stars = sr_quality::stars::positions(reference, STAR_REFINE_LIMIT);
    let alignment: Vec<_> = burst
        .frames
        .par_iter()
        .enumerate()
        .map(|(i, frame)| {
            let stars = sr_quality::stars::positions(frame, STAR_REFINE_LIMIT);
            serde_json::json!({
                "index": i, "file": frame.metadata.file_name,
                "is_reference": i == reg.choice.index,
                "evidence": sr_register::validation::applied_alignment(
                    &reference_stars, &stars, &reg.warps[i], reference.width, reference.height),
            })
        })
        .collect();
    std::fs::write(
        dir.join("applied-star-alignment.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "coordinates": "full reference sensor pixels; cells row-major 4x4",
            "selection": "reciprocal nearest stars within 4 sensor pixels under the applied global-plus-local warp; no new fit",
            "limitations": "conditional residuals, not calibrated confidence; inspect match fractions and cell coverage; catalogues may contain defects/blends and overlap fitting stars; reference self-match is not independent evidence",
            "changes_merge_weights": false,
            "frames": alignment,
        }))?)?;
    sr_diagnostics::write_frames_csv(
        &dir.join("global-registration.csv"),
        &frame_rows(burst, reg, None, &used, &photometry_of(burst, reg, true)),
    )?;
    sr_diagnostics::write_quality_csv(
        &dir.join("frame-quality.csv"),
        &burst.frames,
        &burst.qualities,
    )?;

    // Reference preview at sensor resolution, so the operator can see what the
    // output grid is anchored to without waiting for a full reconstruction.
    //
    // White balance, colour matrix and exposure normalisation are applied. The
    // raw linear mosaic is green-dominated and sits in the bottom few percent
    // of the range, so an unrendered preview is a near-black green rectangle:
    // technically faithful and useless for checking framing and focus.
    // Only the region being reconstructed: a full sensor is 150 MB of preview
    // for a run that produced a 300-pixel crop, and the part it shows is not
    // the part being asked about.
    let ref_frame = &burst.frames[reg.choice.index];
    let mut preview = sr_raw::demosaic_bilinear(ref_frame);
    let channels = ref_frame.channels();
    if let Some((x, y, w, h)) = roi {
        let (x, y) = (x.min(ref_frame.width), y.min(ref_frame.height));
        let w = w.min(ref_frame.width - x);
        let h = h.min(ref_frame.height - y);
        if w > 0 && h > 0 {
            for p in preview.iter_mut().take(channels) {
                *p = p.crop(x, y, w, h);
            }
        }
    }
    let color = sr_color::ColorTransform::from_metadata(&ref_frame.metadata);
    color.apply(&mut preview);
    sr_color::normalise_exposure(&mut preview, 0.9995, 1.0);
    sr_output::write_product16(
        &dir.join("reference-preview.tif"),
        &sr_color::to_rendered(&preview),
        channels,
    )?;

    // Residual and warp maps, averaged over the burst so one image answers
    // "where did registration struggle?".
    let (gw, gh) = (ref_frame.width / 2, ref_frame.height / 2);
    let mut warp_mag = Plane::<f32>::new(gw, gh);
    let mut warp_conf = Plane::<f32>::new(gw, gh);
    let mut n = 0.0f32;
    for w in &reg.warps {
        if let Some(d) = &w.local {
            for y in 0..gh {
                for x in 0..gw {
                    let (ux, uy) = d.sample(x as f32 * 2.0, y as f32 * 2.0);
                    warp_mag.data[y * gw + x] += (ux * ux + uy * uy).sqrt();
                    warp_conf.data[y * gw + x] += d.sample_conf(x as f32 * 2.0, y as f32 * 2.0);
                }
            }
            n += 1.0;
        }
    }
    if n > 0.0 {
        for v in warp_mag.data.iter_mut() {
            *v /= n;
        }
        for v in warp_conf.data.iter_mut() {
            *v /= n;
        }
        sr_output::write_diagnostic(&dir.join("warp-magnitude-map.tif"), &warp_mag, None, true)?;
        sr_output::write_diagnostic(
            &dir.join("warp-confidence-map.tif"),
            &warp_conf,
            Some((0.0, 1.0)),
            false,
        )?;
    }

    // Per-frame registration residual as a small raster: rows are frames.
    let mut residual = Plane::<f32>::new(6, burst.frames.len());
    for (i, r) in reg.registrations.iter().enumerate() {
        residual.data[i * 6] = r.residual_rms * 2.0;
        residual.data[i * 6 + 1] = r.residual_p50 * 2.0;
        residual.data[i * 6 + 2] = r.residual_p90 * 2.0;
        residual.data[i * 6 + 3] = r.residual_p99 * 2.0;
        residual.data[i * 6 + 4] = r.confidence;
        residual.data[i * 6 + 5] = r.overlap;
    }
    sr_output::write_gray32f(&dir.join("registration-residual-map.tif"), &residual)?;
    Ok(())
}

pub fn register(
    spec: &InputSpec,
    cfg: &ReconstructionConfig,
    diagnostics: &Path,
) -> Result<()> {
    let burst = load_burst(spec)?;
    print!("{}", report_header(&burst));
    for w in &burst.validation.fatal {
        log::error!("{w}");
    }
    let cache = Cache::new(spec.cache_dir);
    let reg = register_burst(&burst, cfg, &cache)?;
    print!("{}", registration_summary(&burst, &reg));

    write_registration_diagnostics(diagnostics, &burst, &reg, cfg.roi)?;
    println!("\nRegistration diagnostics written to {}", diagnostics.display());
    Ok(())
}

/// Per-frame merge weights.
///
/// Registration confidence dominates: a frame that could not be placed reliably
/// should not be averaged in as if it had been. Sharpness contributes mildly and
/// is clamped, because aggressive quality weighting is really lucky imaging and
/// belongs behind `--lucky` where the user can see it.
fn frame_weights(burst: &LoadedBurst, reg: &RegisteredBurst) -> Vec<f32> {
    let mut w: Vec<f32> = (0..burst.frames.len())
        .map(|i| {
            if !registration_usable(&reg.registrations[i]) { return 0.0; }
            let conf = reg.registrations[i].confidence.clamp(0.0, 1.0);
            let sharp = burst.qualities[i].sharpness.clamp(0.5, 2.0);
            conf * sharp
        })
        .collect();
    // Deliberately not normalised against this burst's mean. The merge divides
    // by its own accumulated weight, so a common factor across the frames of
    // one run changes nothing within it — and dividing by the batch's mean
    // makes two batches incomparable, which matters the moment a set is
    // reconstructed in pieces and added back together with `smokstak combine`.
    // What remains is per-frame confidence times sharpness, on a scale that
    // means the same thing in every batch of the project.
    // A frame named only to fix the grid is not part of this batch's data. It
    // would otherwise be counted once per batch, which across twenty batches
    // is twenty copies of one exposure.
    if burst.reference_is_extra
        && let Some(i) = burst.forced_reference {
            w[i] = 0.0;
        }
    w
}

/// A tiny weight cannot make a failed alignment safe for photometry or an
/// unweighted consensus. Preserve low-overlap but accurately placed frames;
/// exclude catastrophic residuals when the registration also reports low trust.
pub(crate) fn registration_usable(r: &sr_register::global::GlobalRegistration) -> bool {
    r.confidence.is_finite()
        && r.confidence > 0.0
        && r.residual_p50.is_finite()
        && (r.confidence >= 0.05 || r.residual_p50 <= 1.0)
}

fn registered_group(group: &FilterGroup, regs: &[sr_register::global::GlobalRegistration]) -> Result<FilterGroup> {
    let active: Vec<bool> = group.active.iter().enumerate()
        .map(|(i, &live)| live && regs.get(i).is_some_and(registration_usable))
        .collect();
    let members: Vec<usize> = group.members.iter().copied().filter(|&i| active[i]).collect();
    anyhow::ensure!(!members.is_empty(), "no reliably registered exposures remain in filter group {}", group.name);
    let reference = if active[group.reference] { group.reference } else {
        *members.iter().max_by(|&&a, &&b| regs[a].confidence.total_cmp(&regs[b].confidence)).unwrap()
    };
    Ok(FilterGroup { name: group.name.clone(), active, members, reference })
}

fn local_quality_maps(burst: &LoadedBurst) -> Vec<LocalQualityMap> {
    burst
        .guide_luma
        .par_iter()
        .map(|l| sr_quality::local_quality(l, QUALITY_REGION))
        .collect()
}

/// Where a run's files go.
///
/// Carried as one value because every writer needs most of it and a filter
/// group needs its own resolved copy.
#[derive(Clone, Copy)]
struct OutputSpec<'a> {
    output: &'a Path,
    diagnostics: Option<&'a Path>,
    preview: Option<&'a Path>,
    preview_size: usize,
    float_tiff: bool,
    fits: bool,
    xisf: bool,
    /// Where this batch's contribution goes, for `smokstak combine`.
    accumulate: Option<&'a Path>,
}

/// The parts of a `stack` invocation that are the same for every filter.
struct StackRun<'a> {
    spec: &'a InputSpec<'a>,
    cfg: &'a ReconstructionConfig,
    out: OutputSpec<'a>,
    correct_ca: bool,
    started: std::time::SystemTime,
}

/// Everything decided inside one filter group, and borrowed by the merge.
///
/// These are computed together because they are computed *per filter*: a frame
/// through one filter and a frame through another are different pictures of the
/// sky, however well they register.
struct GroupInputs {
    frame_exclusions: Vec<Option<String>>,
    photometric_reference: Option<usize>,
    fits: Vec<FramePhotometry>,
    photometry: Vec<PhotometricMatch>,
    coverage: SamplingCoverage,
    robustness: sr_reconstruct::robustness::RobustnessMaps,
    kernels: KernelField,
    lucky: Option<sr_reconstruct::lucky::LuckySelection>,
    weights: Vec<f32>,
    chroma: sr_core::geometry::RadialChroma,
}

fn group_noise(frames: &[RawFrame], members: &[usize]) -> (sr_core::frame::NoiseModel, String) {
    let selected: Vec<_> = members.iter().map(|&i| &frames[i]).collect();
    sr_noise::estimate_burst_refs(&selected)
}

fn photometric_exclusion(obstructed: f32, swing: f32) -> Option<String> {
    (obstructed > UNUSABLE_OBSTRUCTION || swing > UNUSABLE_SWING).then(|| format!(
        "Photometric gate rejected this frame: {:.2}% of sky cells marked obstructed, {:.2}% sky variation. Limits: >{:.0}% obstruction or >{:.0}% variation.",
        100.0*obstructed,100.0*swing,100.0*UNUSABLE_OBSTRUCTION,100.0*UNUSABLE_SWING))
}

/// Defects, photometry, coverage, robustness, kernels, lucky regions, weights
/// and chromatic aberration — in that order, because each one's report reads as
/// an answer to the last.
#[allow(clippy::too_many_arguments)]
fn prepare_group(
    burst: &mut LoadedBurst,
    reg: &RegisteredBurst,
    group: &FilterGroup,
    cfg: &ReconstructionConfig,
    cache: &Cache,
    correct_ca: bool,
    timings: &mut Timings,
    warnings: &mut Vec<String>,
) -> Result<GroupInputs> {
    let reference = reg.choice.index;
    let registered = registered_group(group, &reg.registrations)?;
    let excluded: Vec<usize> = group.active.iter().zip(&registered.active).enumerate()
        .filter_map(|(i, (&was, &now))| (was && !now).then_some(i)).collect();
    if !excluded.is_empty() {
        let note = format!("{} frame(s) excluded before photometry: unreliable registration", excluded.len());
        println!("{note}");
        warnings.push(note);
        for i in excluded {
            let r = &reg.registrations[i];
            println!("    frame {i} ({}): registration confidence {:.4}, median residual {:.2} guide pixels",
                burst.frames[i].metadata.file_name, r.confidence, r.residual_p50);
        }
    }
    let group = &registered;

    // Geometry was prepared globally and does not read burst.noise. Every
    // subsequent noise consumer (defects, rejection, kernels, merge and the
    // manifest) must see only this filter's usable selected measurements.
    // Group membership already excludes any geometry-only reference.
    let (noise, source) = group_noise(&burst.frames, &group.members);
    burst.noise = noise;
    burst.noise_source = source;
    log::info!("filter {:?} noise model: var = {:.3e} * x + {:.3e} (from {})",
        group.name, burst.noise.alpha, burst.noise.beta, burst.noise_source);

    // Fixed-pattern defects, once the burst's motion is known. Order matters
    // both ways: the scan needs the burst to have moved before it can tell a
    // dead site from a star, and the merge needs the mask before it drags one
    // bad site across the scene into a streak.
    //
    // Inside the group, because the steadiness test that separates a defect
    // from a star asks whether a site's excess varies between frames, and
    // frames through different filters vary for reasons that are nothing to do
    // with the sensor. Run across all thirty frames of the narrowband set it
    // finds 796 sites; run inside each filter it finds around 1200, and those
    // are the ones that are actually there.
    if cfg.detect_defects {
        let t = Instant::now();
        apply_defect_mask(burst, reg, group, cache, warnings);
        timings.record("defects", t.elapsed());
    }

    // Measured inside the group and against a frame inside the group. The
    // global reference may well be from another filter, which for these two
    // stages would be a comparison between two different subjects.
    let fits = if cfg.photometric_match {
        // Point sources, for the transparency. A gain fitted to block medians
        // is a statement about the sky, and on a night whose sky changed it is
        // the sky's ratio and not the air's; the stars are the only thing in
        // the frame with no sky in them. Measured here rather than carried
        // from registration because registration is cached and this is not.
        let t = Instant::now();
        let star_lists: Vec<Vec<sr_core::star::Star>> = (0..burst.frames.len())
            .into_par_iter()
            .map(|i| {
                if group.active.get(i).copied().unwrap_or(false) || i == group.reference {
                    sr_quality::stars::positions_for_photometry(&burst.frames[i], STAR_PHOTOMETRY_LIMIT)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let measured = star_lists.iter().filter(|l| !l.is_empty()).count();
        log::info!(
            "photometry: point sources for the transparency measured on {measured} of {}              frames in {:.2?}",
            burst.frames.len(),
            t.elapsed()
        );
        sr_quality::photometry::match_with_stars(
            &burst.frames,
            &reg.warps,
            group.reference,
            &burst.validation.exposure_scale,
            &group.active,
            &star_lists,
            cfg.sky_field,
        )
    } else {
        sr_quality::photometry::from_exposure_only(&burst.validation.exposure_scale)
    };
    if cfg.photometric_match {
        report_photometry(&fits, burst, warnings);
    }
    let photometry: Vec<PhotometricMatch> = fits.iter().map(|p| p.map).collect();

    // Frames the match found to be not of this sky. More than a third of the
    // frame obstructed, or a sky that rose or fell by more than itself across
    // the frame -- dawn, cloud lit by the moon -- is not a frame with a bad
    // region in it, it is a bad frame, and the parts of it that pass every
    // test still carry the photometry of the parts that did not. Frames shot
    // into trees at dawn are like this, and the trees they leave in the stack
    // are the first thing anyone sees.
    let level = sr_core::math::median(
        &burst.qualities.iter().map(|q| q.mean_level).collect::<Vec<_>>(),
    )
    .max(1e-4);
    let mut active = group.active.clone();
    let mut frame_exclusions = vec![None; burst.frames.len()];
    let mut dropped = Vec::new();
    for (i, f) in fits.iter().enumerate() {
        if !active.get(i).copied().unwrap_or(false) || i == group.reference {
            continue;
        }
        let obstructed = f.map.blocked_fraction();
        let swing = (0..3).map(|c| f.map.field_amplitude(c)).fold(0.0f32, f32::max) / level;
        if let Some(reason) = photometric_exclusion(obstructed, swing) {
            active[i] = false;
            frame_exclusions[i] = Some(reason);
            dropped.push((i, obstructed, swing));
        }
    }
    if !dropped.is_empty() {
        println!(
            "  {} frame(s) dropped as not of this sky:",
            dropped.len()
        );
        for &(i, obstructed, swing) in &dropped {
            println!(
                "    frame {i} ({}): obstructed over {:.0}% of the frame, sky varying {:.0}% \
                 across it",
                burst.frames[i].metadata.file_name,
                obstructed * 100.0,
                swing * 100.0
            );
        }
        warnings.push(format!(
            "{} frame(s) dropped as unusable: obstructed over more than {:.0}% of the frame or \
             with a sky varying by more than {:.0}% across it",
            dropped.len(),
            UNUSABLE_OBSTRUCTION * 100.0,
            UNUSABLE_SWING * 100.0
        ));
    }

    // Sampling diversity, before anything is merged: this is the number that
    // says whether the requested scale is real.
    let t = Instant::now();
    let coverage =
        sr_reconstruct::coverage::analyse_coverage(&burst.frames, &reg.warps, cfg.scale, 8);
    timings.record("coverage", t.elapsed());
    println!("\n{}", coverage.describe());
    report_sampling_headroom(burst, cfg.scale, warnings);
    if coverage.recommended_scale < cfg.scale - 0.1 {
        warnings.push(format!(
            "requested {:.2}x but the burst's sub-pixel diversity supports about {:.2}x",
            cfg.scale, coverage.recommended_scale
        ));
    }

    // Robustness maps.
    let t = Instant::now();
    let robustness = sr_reconstruct::robustness::build_maps_spooled(
        &burst.frames,
        &reg.warps,
        group.reference,
        &active,
        &photometry,
        &burst.noise,
        reg.residual_sigma,
        &cfg.robustness,
        burst.spill_dir.as_deref(),
    )?;
    timings.record("robustness", t.elapsed());
    if cfg.robustness.enabled {
        let mean = robustness.rejected_fraction.iter().sum::<f32>()
            / robustness.rejected_fraction.len().max(1) as f32;
        log::info!("robustness: {:.2}% of the burst suppressed on average", mean * 100.0);
    }

    let mut weights = frame_weights(burst, reg);
    // Frames outside this group carry no weight, which is what actually keeps
    // them out of the merge. Everything else in the group machinery — the
    // photometric match, the robustness maps — assumes that is true, and until
    // it was, every filter's master quietly contained all three.
    for (w, live) in weights.iter_mut().zip(&active) {
        if !live {
            *w = 0.0;
        }
    }

    // Excluded exposures (including other filters and a grid-only reference)
    // supply no samples and must not shrink the reconstruction kernel.
    let contributing = weights.iter().filter(|w| w.is_finite() && **w > 0.0).count();
    log::info!("kernel: {contributing} contributing frames out of {} loaded", burst.frames.len());

    // Structure-aware kernels, from the reference guide.
    let t = Instant::now();
    // Keep clipped star cores bright for structural decisions. The
    // registration guide deliberately omits them when fitting alignment.
    let kernel_guide = burst.frames[reference].structure_guide_rgb().luma();
    let ref_luma = &kernel_guide;
    // Measured on the guide rather than derived from the sensor model. The
    // guide is a 2x2 cell reduction, so its scatter is smaller than the
    // per-sample figure by a factor that depends on the mosaic -- and every
    // threshold the kernel uses is expressed in units of this, so overstating
    // it widens the kernel everywhere. The model's value is the floor, for a
    // guide so flat that differencing it finds nothing.
    let noise_sigma = sr_reconstruct::kernel::guide_noise_sigma(ref_luma)
        .max(0.05 * burst.noise.std_dev(ref_luma.mean().max(0.0)));
    log::info!(
        "kernel: guide noise {:.3e}, against {:.3e} from the sensor model",
        noise_sigma,
        burst.noise.std_dev(ref_luma.mean().max(0.0))
    );
    let kernels = match cfg.backend {
        Backend::HandheldBurstSr => KernelField::for_sensor(
            ref_luma,
            noise_sigma,
            contributing,
            burst.frames[reference].cfa,
            &cfg.kernel,
        ),
        // The baselines get a fixed circular kernel of the same nominal size, so
        // a comparison between backends isolates the kernel, not its scale.
        _ => KernelField::isotropic(
            ref_luma.width,
            ref_luma.height,
            cfg.kernel.k_detail.max(0.25),
            cfg.kernel.radius,
            2,
        ),
    };
    timings.record("kernels", t.elapsed());

    let t = Instant::now();
    let lucky = choose_lucky(burst, reg, cfg);
    timings.record("lucky", t.elapsed());

    let t = Instant::now();
    let chroma = estimate_chroma(burst, reference, correct_ca, &weights);
    timings.record("chromatic-aberration", t.elapsed());

    Ok(GroupInputs {
        frame_exclusions,
        photometric_reference: cfg.photometric_match.then_some(group.reference),
        fits,
        photometry,
        coverage,
        robustness,
        kernels,
        lucky,
        weights,
        chroma,
    })
}

/// Which frames contribute where, when the seeing was not the same all night.
///
/// Reports for itself, because "off" and "not needed" are different answers and
/// an operator who asked for lucky imaging deserves to be told which one they
/// got.
fn choose_lucky(
    burst: &LoadedBurst,
    reg: &RegisteredBurst,
    cfg: &ReconstructionConfig,
) -> Option<sr_reconstruct::lucky::LuckySelection> {
    let reference = reg.choice.index;
    let build = |maps: &[LocalQualityMap], fraction: f32| {
        sr_reconstruct::lucky::build(
            maps,
            &reg.warps,
            burst.frames[reference].width,
            burst.frames[reference].height,
            cfg.scale,
            fraction,
        )
    };
    let (lucky, note) = match cfg.lucky {
        LuckyMode::Off => (None, "off".to_string()),
        LuckyMode::Fraction(f) => {
            let sel = build(&local_quality_maps(burst), f);
            let note =
                format!("keeping the best {:.0}% of frames per region", sel.fraction * 100.0);
            (Some(sel), note)
        }
        LuckyMode::Auto => {
            let maps = local_quality_maps(burst);
            let v = sr_reconstruct::lucky::seeing_variability(&maps);
            if v > 1.15 {
                let sel = build(&maps, 0.6);
                let note = format!(
                    "enabled automatically (local sharpness varies by {:.0}% across the \
                     burst); keeping the best {:.0}% per region",
                    (v - 1.0) * 100.0,
                    sel.fraction * 100.0
                );
                (Some(sel), note)
            } else {
                (
                    None,
                    format!(
                        "not needed (local sharpness varies by only {:.0}% across the burst)",
                        (v - 1.0) * 100.0
                    ),
                )
            }
        }
    };
    println!("Lucky-region selection: {note}");
    lucky
}

/// Lateral chromatic aberration, measured and then judged worth correcting.
fn estimate_chroma(
    burst: &LoadedBurst,
    reference: usize,
    correct_ca: bool,
    weights: &[f32],
) -> sr_core::geometry::RadialChroma {
    let identity = sr_core::geometry::RadialChroma::identity();
    if burst.frames[reference].is_mono() {
        // Lateral chromatic aberration is one channel landing in a different
        // place from another. With one channel there is no such thing.
        return identity;
    }
    if !correct_ca {
        println!("Chromatic aberration: disabled");
        return identity;
    }
    let ca = estimate_chromatic_aberration(burst, reference, weights);
    if ca.is_significant() {
        println!(
            "Chromatic aberration: correcting: red {:+.4}%, blue {:+.4}% magnification at \
             the corner ({:.2} and {:.2} px)",
            (ca.magnification[0] - 1.0) * 100.0,
            (ca.magnification[2] - 1.0) * 100.0,
            ca.corner_shift[0],
            ca.corner_shift[2]
        );
        ca.correction()
    } else {
        println!(
            "Chromatic aberration: measured but below the threshold worth correcting \
             ({:.2} px at the corner)",
            ca.corner_shift[0].max(ca.corner_shift[2])
        );
        identity
    }
}

/// The product after colour, background and restoration, with the linear
/// reference kept beside it.
struct FinishedProduct {
    /// Scene-linear, captured before any restoration so the two can always be
    /// compared.
    linear: [Plane<f32>; 3],
    color: sr_color::ColorTransform,
    gain: f32,
    background: Option<sr_reconstruct::background::BackgroundModel>,
    restoration: Option<sr_reconstruct::restore::RestorationReport>,
}

fn finish_product(
    product: &mut ReconstructionProduct,
    burst: &LoadedBurst,
    reference: usize,
    cfg: &ReconstructionConfig,
    timings: &mut Timings,
    warnings: &mut Vec<String>,
) -> FinishedProduct {
    let t = Instant::now();
    let color = sr_color::ColorTransform::from_metadata(&burst.frames[reference].metadata);
    if color.fallback {
        warnings.push(
            "no usable camera colour matrix; output colour is not colorimetric".into(),
        );
    }
    // Background, before the colour transform: vignetting is a property of the
    // optics and the sensor, so it is removed in the domain it happened in.
    let background = if cfg.flatten_background {
        flatten_background(&mut product.rgb, product.channels, warnings)
    } else {
        None
    };

    color.apply(&mut product.rgb);
    let gain = sr_color::normalise_exposure(&mut product.rgb, 0.9995, 1.0);
    log::info!("exposure normalisation gain {gain:.3}");
    timings.record("color", t.elapsed());

    let linear = product.rgb.clone();

    let mut restoration = None;
    if matches!(cfg.postprocess, PostProcess::Mild) {
        let t = Instant::now();
        let sigma = burst.noise.std_dev(0.2) * gain / (burst.frames.len() as f32).sqrt();
        let r = sr_reconstruct::restore::mild_sharpen(
            &mut product.rgb,
            product.channels,
            0.6,
            2,
            sigma.max(1e-4),
        );
        timings.record("postprocess", t.elapsed());
        log::info!(
            "mild restoration applied: mean change {:.5}, max {:.5}",
            r.mean_change,
            r.max_change
        );
        restoration = Some(r);
    }

    FinishedProduct { linear, color, gain, background, restoration }
}

/// What a batch contributes, before it is normalised into an image.
///
/// A set too large to hold in memory is reconstructed in pieces, and pieces
/// have to be added rather than averaged: a frame's contribution to an output
/// pixel is a weight and a weighted value, and the merge already carries both.
/// `product.rgb` is the ratio and `product.weight` is the denominator, so the
/// numerator is their product and a batch is summed by summing each.
///
/// Written before the colour transform and before exposure normalisation,
/// because those are display decisions and both would have to be undone to add
/// two batches together.
fn write_accumulator(
    dir: &Path,
    product: &ReconstructionProduct,
    reference: &RawFrame,
    frames: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for c in 0..product.channels {
        let mut num = product.rgb[c].clone();
        for (n, w) in num.data.iter_mut().zip(&product.weight[c].data) {
            *n *= *w;
        }
        sr_output::write_gray32f(&dir.join(format!("sum-{c}.tif")), &num)?;
        sr_output::write_gray32f(&dir.join(format!("weight-{c}.tif")), &product.weight[c])?;
    }
    let meta = serde_json::json!({
        "format": 1,
        "width": product.width,
        "height": product.height,
        "channels": product.channels,
        "frames": frames,
        // The grid this batch was reconstructed on. Adding two batches only
        // means anything if they agree, so `combine` checks.
        "reference_file": reference.metadata.file_name,
        "reference_sha256_prefix": reference.metadata.sha256_prefix,
    });
    std::fs::write(dir.join("accumulator.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "Accumulator written to {} ({} frames, {} x {}); add it to others with `smokstak combine`",
        dir.display(),
        frames,
        product.width,
        product.height
    );
    Ok(())
}

/// Every image file a run produces, in the order it announces them.
fn write_stack_outputs(
    out: &OutputSpec,
    product: &ReconstructionProduct,
    finished: &FinishedProduct,
) -> Result<Vec<String>> {
    if let Some(parent) = out.output.parent()
        && !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    let mut output_files = Vec::new();

    if let (Some(m), Some(dir)) = (&finished.background, out.diagnostics) {
        // Rendered small: the model is five coefficients per channel, so it
        // carries no detail, and a full-size copy of it would be as large as
        // the reconstruction for no information at all.
        let (w, h) = (product.width / 8, product.height / 8);
        std::fs::create_dir_all(dir)?;
        let field = sr_reconstruct::background::render_for_display(m, w.max(64), h.max(64));
        // No tone curve: this is a correction field, not a photograph, and a
        // gamma applied to it would misrepresent the amounts.
        sr_output::write_product16(&dir.join("background-model.tif"), &field, m.channels)?;
        println!(
            "Background model written to {}",
            dir.join("background-model.tif").display()
        );
    }

    // The 16-bit file is the picture, and is rendered as one. The background
    // is flattened for it -- vignetting and sky gradient, the model that is
    // off by default for the linear product because it cannot tell those from
    // large-scale structure -- because a screen transfer that puts the sky at a
    // quarter of the range turns a seven percent gradient into a band across
    // the frame that is the first thing anyone sees. Then each channel is
    // divided by its own white point so that a saturated star is white, and
    // the transfer applied. The linear file beside it is the measurement, and
    // none of this touches it.
    let mut display = product.rgb.clone();
    let flattened_for_display = if finished.background.is_none() {
        flatten_for_display(&mut display, product.channels)
    } else {
        None
    };
    let white = sr_color::stretch::ceiling_from_data(&display, product.channels);
    let (rendered, how) = sr_color::stretch::render(&display, product.channels, &white);
    drop(display);
    let rendered = as_product(&rendered, product.channels);
    sr_output::write_product16(out.output, &rendered, product.channels)?;
    output_files.push(out.output.display().to_string());
    println!("Wrote {} ({} x {})", out.output.display(), product.width, product.height);
    match &how {
        sr_color::stretch::Rendering::Stretched { per_channel } => {
            let s = per_channel.get(per_channel.len() / 2).unwrap_or(&per_channel[0]);
            println!(
                "Rendered:          sky placed at a quarter of the range in each channel; \
                 shadows clipped at {:.4}, midtone {:.4}; white at {:.3}/{:.3}/{:.3}",
                s.shadows, s.midtone, white[0], white[1], white[2]
            );
            if let Some(pct) = flattened_for_display {
                println!(
                    "  background flattened for the picture only ({pct:.1}% of the sky level \
                     as vignetting and gradient); the linear file keeps it"
                );
            }
        }
        sr_color::stretch::Rendering::Encoded => {
            println!("Rendered:          already legible; sRGB transfer only");
        }
    }

    // What the file we just wrote looks like as a picture, which is a separate
    // question from whether the reconstruction is right and was for a long time
    // the only question nothing here asked.
    if product.channels >= 3 {
        let e = sr_color::encoding_of_rendered(&rendered);
        println!("Encoding:          {}", e.describe());
        if e.headroom < MIN_HEADROOM {
            println!(
                "  the background alone occupies {:.0}% of the file's range, leaving \
                 {:.0}% for everything above it",
                e.sky * 100.0,
                e.headroom * 100.0
            );
        }
        if e.uneven_fraction() > MAX_UNEVEN_CLIP {
            println!(
                "  {} pixels clip in some channels and not others and will read as \
                 false colour, most of them star cores",
                e.uneven
            );
        }
    }

    if let Some(p) = out.preview {
        write_preview(p, &finished.linear, product.channels, out.preview_size)?;
        output_files.push(p.display().to_string());
    }

    for p in sr_output::write_scientific_copies(out.output, &finished.linear, product.channels, out.fits, out.xisf)? {
        println!("Wrote {} (32-bit float master)", p.display());
        output_files.push(p.display().to_string());
    }
    if out.float_tiff {
        let p = out.output.with_extension("linear.tif");
        sr_output::write_product32f(&p, &finished.linear, product.channels)?;
        output_files.push(p.display().to_string());
        println!("Wrote {} (32-bit float, linear, unrestored)", p.display());
    }
    if finished.restoration.is_some() {
        let p = out.output.with_extension("unrestored.tif");
        let mut display = finished.linear.clone();
        if finished.background.is_none() {
            flatten_for_display(&mut display, product.channels);
        }
        let white = sr_color::stretch::ceiling_from_data(&display, product.channels);
        let (r, _) = sr_color::stretch::render(&display, product.channels, &white);
        sr_output::write_product16(&p, &as_product(&r, product.channels), product.channels)?;
        output_files.push(p.display().to_string());
        println!("Wrote {} (no restoration, for comparison)", p.display());
    }
    Ok(output_files)
}

/// Flatten the background of a copy that is only going to be looked at.
///
/// The same model `--flatten-background` fits, with the same half-percent
/// floor below which nothing is claimed; here it is applied without a warning,
/// because the file it goes into is the picture and the measurement beside it
/// is untouched. Returns the magnitude removed, as a percentage of the sky.
fn flatten_for_display(rgb: &mut [Plane<f32>; 3], channels: usize) -> Option<f32> {
    let m = sr_reconstruct::background::fit(rgb, channels)?;
    let magnitude = sr_reconstruct::background::magnitude(&m);
    if magnitude < 0.005 {
        return None;
    }
    sr_reconstruct::background::apply(rgb, &m);
    Some(magnitude * 100.0)
}

/// Three planes from however many the render produced, the spare ones empty,
/// which is the shape every product writer expects.
fn as_product(planes: &[Plane<f32>], channels: usize) -> [Plane<f32>; 3] {
    let empty = Plane::<f32>::new(0, 0);
    [
        planes.first().cloned().unwrap_or_else(|| empty.clone()),
        if channels > 1 { planes[1].clone() } else { empty.clone() },
        if channels > 2 { planes[2].clone() } else { empty },
    ]
}

fn write_stack_diagnostics(
    dir: &Path,
    burst: &LoadedBurst,
    reg: &RegisteredBurst,
    cfg: &ReconstructionConfig,
    product: &ReconstructionProduct,
    inputs: &GroupInputs,
) -> Result<()> {
    let reference = reg.choice.index;
    std::fs::create_dir_all(dir)?;
    write_registration_diagnostics(dir, burst, reg, cfg.roi)?;
    sr_diagnostics::write_product_diagnostics(dir, product)?;
    write_weight_probes(dir, burst, reg, cfg, inputs)?;
    sr_diagnostics::write_phase_histogram_csv(&dir.join("sampling-phase.csv"), &inputs.coverage)?;

    let phase_map =
        sr_reconstruct::coverage::phase_coverage_map(&burst.frames, &reg.warps, cfg.scale, 64);
    sr_output::write_diagnostic(&dir.join("sampling-phase-map.tif"), &phase_map, None, true)?;

    if !burst.frames[reference].defects.is_empty() {
        // Downsampled by counting rather than by sampling: at a hundredth
        // of the sensor's size an isolated site would vanish from a
        // point-sampled map, and isolated sites are the whole subject.
        let d = &burst.frames[reference].defects;
        let cell = 8usize;
        let (gw, gh) = (d.width.div_ceil(cell), d.height.div_ceil(cell));
        let mut m = Plane::<f32>::new(gw, gh);
        for y in 0..d.height {
            for x in 0..d.width {
                if d.get(y * d.width + x) {
                    m.data[(y / cell) * gw + x / cell] += 1.0;
                }
            }
        }
        sr_output::write_diagnostic(&dir.join("defect-map.tif"), &m, None, true)?;
    }
    if cfg.robustness.enabled {
        let m = inputs.robustness.mean_plane();
        sr_output::write_diagnostic(
            &dir.join("motion-mask-summary.tif"),
            &m,
            Some((0.0, 1.0)),
            false,
        )?;
    }
    if let Some(sel) = &inputs.lucky {
        let c = sel.contributor_map();
        sr_output::write_diagnostic(&dir.join("lucky-contributors.tif"), &c, None, true)?;
    }
    let lq = local_quality_maps(burst);
    if let Some(first) = lq.first() {
        sr_output::write_diagnostic(
            &dir.join("local-quality-summary.tif"),
            &first.plane(),
            None,
            true,
        )?;
    }

    let used: Vec<bool> = inputs.weights.iter().map(|w| w.is_finite() && *w > 0.0).collect();
    // Preserve the actual spatial corrections and exclusions, not just the
    // central green offset in frames.csv. Coordinates span the full reference
    // sensor even when this run reconstructs an ROI.
    let photometric_rows: Vec<_> = inputs
        .fits
        .iter()
        .enumerate()
        .map(|(i, fit)| {
            serde_json::json!({
                "index": i,
                "file": burst.frames[i].metadata.file_name,
                "used": used[i],
                "weight": inputs.weights[i],
                "fit": fit,
            })
        })
        .collect();
    std::fs::write(
        dir.join("photometry.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "coordinates": "normalised full reference sensor, -1 to 1 on each axis",
            "photometric_reference": inputs.photometric_reference.map(|i| serde_json::json!({
                "index": i,
                "file": burst.frames[i].metadata.file_name,
                "filter": burst.frames[i].metadata.filter,
            })),
            "grid_reference_index": reg.choice.index,
            "frames": photometric_rows,
        }))?,
    )?;
    let rejected = if cfg.robustness.enabled {
        Some(inputs.robustness.rejected_fraction.as_slice())
    } else {
        None
    };
    sr_diagnostics::write_frames_csv(
        &dir.join("frames.csv"),
        &frame_rows(burst, reg, rejected, &used, &inputs.fits),
    )?;
    Ok(())
}

/// Sample pre-kernel weight factors for a fixed full-reference 6x4 patch grid.
/// These are diagnostic probes, not measured deposited-weight fractions.
fn write_weight_probes(
    dir: &Path,
    burst: &LoadedBurst,
    reg: &RegisteredBurst,
    cfg: &ReconstructionConfig,
    inputs: &GroupInputs,
) -> Result<()> {
    let reference = &burst.frames[reg.choice.index];
    let norm = burst.noise.variance(0.18);
    let mut frames = Vec::with_capacity(burst.frames.len());
    for (i, frame) in burst.frames.iter().enumerate() {
        let photo = inputs.photometry[i];
        let sky = sr_reconstruct::merge::sky_of(frame);
        let noise_weight: [f32; 3] = std::array::from_fn(|c| {
            // The demosaiced baseline shares green's variance; mono deposits
            // into channel zero only. Mirror those merge conventions here.
            let c = if matches!(cfg.backend, Backend::RgbMeanBaseline) {
                1
            } else if frame.is_mono() {
                0
            } else { c };
            norm / (burst.noise.variance(sky[c]) * photo.gain_of(c).powi(2)).max(1e-12)
        });
        let mut probes = Vec::with_capacity(216);
        for py in 0..4 {
            for px in 0..6 {
                for dy in [-32.0f32, 0.0, 32.0] {
                    for dx in [-32.0f32, 0.0, 32.0] {
                        let x = (px as f32 + 0.5) * reference.width as f32 / 6.0 + dx;
                        let y = (py as f32 + 0.5) * reference.height as f32 / 4.0 + dy;
                        let covered = reg.warps[i].inverse_map(x, y).is_some_and(|(sx, sy)| {
                            sx >= 0.0 && sy >= 0.0 && sx < frame.width as f32 && sy < frame.height as f32
                        });
                        let blocked = photo.blocked_at(2.0*x/frame.width as f32-1.0, 2.0*y/frame.height as f32-1.0);
                        let robustness = if matches!(cfg.backend, Backend::RgbMeanBaseline) {
                            1.0
                        } else {
                            inputs.robustness.at(i, x, y)
                        };
                        let local = reg.warps[i].local_confidence_at_ref(x, y).max(0.05);
                        let lucky = inputs.lucky.as_ref().map_or(1.0, |l| l.weight(i, x, y));
                        let base: [f32; 3] = std::array::from_fn(|c| {
                            if covered && !blocked && inputs.weights[i] > 0.0 {
                                noise_weight[c] * robustness * local * lucky * inputs.weights[i]
                            } else { 0.0 }
                        });
                        probes.push(serde_json::json!({
                            "patch": [px, py], "reference_xy": [x, y],
                            "covered": covered, "blocked": blocked,
                            "robustness": robustness, "local_confidence": local,
                            "lucky": lucky, "base_weight_rgb": base,
                        }));
                    }
                }
            }
        }
        frames.push(serde_json::json!({
            "index": i, "file": Path::new(&frame.metadata.path).file_name().map(|p| p.to_string_lossy()),
            "frame_weight": inputs.weights[i], "sky_rgb": sky,
            "noise_weight_rgb": noise_weight, "probes": probes,
        }));
    }
    let evidence = serde_json::json!({
        "method": "6x4 full-reference patch centers; nine probes at offsets -32/0/32 sensor pixels per axis; same sky estimator and factor accessors as merge",
        "limitations": "Pre-kernel coordinate probes, not deposited exposure fractions. Excludes sample-specific defects/clipping, chromatic displacement, kernel support, and profile-fit response; mask/coverage checked at the probe coordinate. Use green probes for raw-sky traces.",
        "frames": frames,
    });
    std::fs::write(dir.join("weight-probes.json"), serde_json::to_vec(&evidence)?)?;
    Ok(())
}

/// One filter's reconstruction, onto the grid the global reference fixed.
///
/// The burst is `&mut` because the defect scan writes its mask into the frames,
/// and it is per-filter: see `prepare_group`.
fn stack_group(
    run: &StackRun,
    burst: &mut LoadedBurst,
    reg: &RegisteredBurst,
    group: &FilterGroup,
    cache: &Cache,
    mut timings: Timings,
    mut warnings: Vec<String>,
) -> Result<()> {
    let cfg = run.cfg;
    let reference = reg.choice.index;

    // Every path this group writes to, resolved once so nothing below has to
    // know that filter groups exist.
    let output = per_filter_path(run.out.output, &group.name);
    let diagnostics = run.out.diagnostics.map(|d| {
        if group.name.is_empty() { d.to_path_buf() } else { d.join(&group.name) }
    });
    let preview = run.out.preview.map(|p| per_filter_path(p, &group.name));
    let accumulate = run.out.accumulate.map(|d| {
        if group.name.is_empty() { d.to_path_buf() } else { d.join(&group.name) }
    });
    let out = OutputSpec {
        output: &output,
        diagnostics: diagnostics.as_deref(),
        preview: preview.as_deref(),
        accumulate: accumulate.as_deref(),
        ..run.out
    };

    let inputs =
        prepare_group(burst, reg, group, cfg, cache, run.correct_ca, &mut timings, &mut warnings)?;

    let merge = sr_reconstruct::MergeInputs {
        frames: &burst.frames,
        warps: &reg.warps,
        reference,
        photometry: &inputs.photometry,
        noise: burst.noise,
        robustness: &inputs.robustness,
        kernels: &inputs.kernels,
        frame_weight: &inputs.weights,
        lucky: inputs.lucky.as_ref(),
        chroma: inputs.chroma,
    };

    log::info!(
        "merging {} frames with the {} backend at {:.2}x",
        inputs.weights.iter().filter(|w| w.is_finite() && **w > 0.0).count(),
        cfg.backend.name(),
        cfg.scale
    );
    let t = Instant::now();
    let mut product = sr_reconstruct::reconstruct(&merge, cfg)?;
    timings.record("merge", t.elapsed());
    log::info!("merge finished in {:.2?}", t.elapsed());

    println!("\n{}", sr_diagnostics::summarise(&product, None));

    if let Some(dir) = out.accumulate {
        let contributing = inputs.weights.iter().filter(|w| **w > 0.0).count();
        write_accumulator(dir, &product, &burst.frames[reference], contributing)?;
    }

    let finished =
        finish_product(&mut product, burst, reference, cfg, &mut timings, &mut warnings);

    let t = Instant::now();
    let output_files = write_stack_outputs(&out, &product, &finished)?;
    timings.record("write", t.elapsed());

    if let Some(dir) = out.diagnostics {
        let t = Instant::now();
        write_stack_diagnostics(dir, burst, reg, cfg, &product, &inputs)?;
        // Audit only members of this filter, including failed registrations.
        // Other filters and a geometry-only reference are not rejected frames.
        {
            let center = crate::frame_review::center(&burst.frames[reference], cfg.roi);
            let mut review = Vec::new();
            for &i in &group.members {
                let used = inputs.weights[i].is_finite() && inputs.weights[i] > 0.0;
                let registered = registration_usable(&reg.registrations[i]);
                let mut row = crate::frame_review::ReviewFrame {
                    path: burst.paths[i].clone(), filter: burst.frames[i].metadata.filter.clone().unwrap_or_default(),
                    capture_time: None, exposure_seconds: None, iso_or_gain: None,
                    photometric_gain: Some(inputs.fits[i].map.gain[..burst.frames[i].channels()].to_vec()),
                    photometry_source: Some(inputs.fits[i].source.name().to_string()),
                    obstruction_mask: Some(inputs.fits[i].map.blocked.iter().map(|r| r.to_vec()).collect()),
                    status: if used { "used" } else { "not-used" }.into(),
                    reason: if used { format!("Eligible: confidence {:.4} × bounded sharpness {:.4}. Local coverage and rejection still apply.",
                            reg.registrations[i].confidence.clamp(0.0,1.0), burst.qualities[i].sharpness.clamp(0.5,2.0)) }
                        else if !registered { format!("Registration rejected: confidence {:.4}, median residual {:.3} sensor px. Requires finite positive confidence and finite residual, with confidence ≥0.05 or residual ≤2 sensor px.",
                            reg.registrations[i].confidence, reg.registrations[i].residual_p50*2.0) }
                        else { inputs.frame_exclusions[i].clone().unwrap_or_else(|| "Zero or invalid production frame factor".into()) },
                    weight: inputs.weights[i].is_finite().then_some(inputs.weights[i]),
                    hfd: burst.stars[i].map(|s|s.hfd), eccentricity: burst.stars[i].map(|s|s.eccentricity),
                    residual: reg.registrations[i].residual_p50.is_finite().then_some(reg.registrations[i].residual_p50*2.0),
                    suppressed: (used && cfg.robustness.enabled && !matches!(cfg.backend, Backend::RgbMeanBaseline))
                        .then(|| inputs.robustness.rejected_fraction[i]),
                    preview_asset: None, preview_note: String::new(),
                };
                crate::frame_review::capture(dir, &mut row, &burst.frames[i], registered.then_some(&reg.warps[i]), center)?;
                review.push(row);
            }
            crate::frame_review::write(dir, &review)?;
        }
        timings.record("diagnostics", t.elapsed());
        println!("Diagnostics written to {}", dir.display());
    }

    let mut manifest = build_manifest(
        burst,
        reg,
        cfg,
        run.spec.path,
        &product,
        Some(inputs.coverage),
        &finished.color,
        finished.gain,
        output_files,
        timings.into_map(),
        warnings,
        run.started,
    );
    manifest.photometric_reference_index = inputs.photometric_reference;
    let manifest_path = match out.diagnostics {
        Some(d) => d.join("run.json"),
        None => out.output.with_extension("run.json"),
    };
    sr_diagnostics::write_manifest(&manifest_path, &manifest)?;
    println!("Run manifest written to {}", manifest_path.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn stack(
    spec: &InputSpec,
    split_by_filter: bool,
    cfg: &ReconstructionConfig,
    output: &Path,
    diagnostics: Option<&Path>,
    preview: Option<&Path>,
    preview_size: usize,
    float_tiff: bool,
    fits: bool,
    xisf: bool,
    accumulate: Option<&Path>,
    force: bool,
    correct_ca: bool,
) -> Result<()> {
    let started = std::time::SystemTime::now();
    let mut timings = Timings::default();
    let mut warnings: Vec<String> = Vec::new();

    let t = Instant::now();
    let mut burst = load_burst(spec)?;
    timings.record("decode", t.elapsed());
    // Before the report is printed, so a run that is doing exactly the right
    // thing is not described as fatally wrong on its way to doing it.
    let splitting = split_by_filter && burst.validation.filter_conflict;
    if splitting {
        // The whole point of the mode: several filters, reconstructed apart but
        // onto one grid.
        burst
            .validation
            .fatal
            .retain(|m| !m.starts_with("the burst holds more than one filter"));
        if let Some(f) = burst.validation.fields.iter_mut().find(|f| f.field == "Filter") {
            f.severity = sr_raw::Severity::Note;
        }
    }
    print!("{}", report_header(&burst));
    if splitting {
        println!("note: one reconstruction per filter, all onto one shared grid");
    }

    if !burst.validation.is_usable() {
        // The report above went to stdout and the errors below go to the log.
        // Redirected into one file the two buffer differently, and without this
        // the errors land in the middle of the report.
        use std::io::Write;
        let _ = std::io::stdout().flush();
        for m in &burst.validation.fatal {
            log::error!("{m}");
        }
        // --force merges the burst as it stands. For a filter conflict that
        // means merging light that was never the same light, so point at the
        // mode that keeps the filters apart instead.
        anyhow::ensure!(
            force,
            "burst validation failed; {}",
            if burst.validation.filter_conflict {
                "pass --split-by-filter to stack each filter separately, or --force to merge them anyway"
            } else {
                "pass --force to reconstruct anyway"
            }
        );
        warnings.push("reconstructed despite failed burst validation (--force)".into());
    }
    warnings.extend(burst.validation.warnings.iter().cloned());

    let t = Instant::now();
    let cache = Cache::new(spec.cache_dir);
    let reg = register_burst(&burst, cfg, &cache)?;
    timings.record("register", t.elapsed());
    print!("{}", registration_summary(&burst, &reg));

    report_stars(&burst, &mut warnings);

    // One reconstruction per filter, all onto the grid the global reference
    // fixed. Everything above this point is shared — the geometry, the frame
    // quality — and everything below it has to stay inside a filter.
    let mut groups = if split_by_filter {
        filter_groups(&burst, &burst.qualities)
    } else {
        vec![FilterGroup {
            name: String::new(),
            active: vec![true; burst.frames.len()],
            members: (0..burst.frames.len()).collect(),
            reference: reg.choice.index,
        }]
    };
    if burst.reference_is_extra {
        exclude_grid_reference(&mut groups, reg.choice.index, &burst.qualities);
    }
    let split = groups.len() > 1;
    if split {
        println!(
            "\nReconstructing {} filters onto one grid: {}",
            groups.len(),
            groups
                .iter()
                .map(|g| format!("{} ({} frames)", g.name, g.members.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let run = StackRun {
        spec,
        cfg,
        out: OutputSpec {
            output,
            diagnostics,
            preview,
            preview_size,
            float_tiff,
            fits,
            xisf,
            accumulate,
        },
        correct_ca,
        started,
    };
    let decode_masks: Vec<_> = burst.frames.iter().map(|f| f.defects.clone()).collect();
    for group in &groups {
        if split {
            println!("\n=== filter {} ===", group.name);
        }
        // Each group starts from the shared state and adds its own, so one
        // filter's warnings and timings do not leak into the next one's
        // manifest.
        // Arc-backed masks make this reset cheap, and a group's early exits
        // (no motion/no defects/disabled scan) cannot retain a previous scan.
        reset_decode_masks(&mut burst.frames, &decode_masks);
        stack_group(&run, &mut burst, &reg, group, &cache, timings.clone(), warnings.clone())?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_manifest(
    burst: &LoadedBurst,
    reg: &RegisteredBurst,
    cfg: &ReconstructionConfig,
    input: &Path,
    product: &ReconstructionProduct,
    coverage: Option<SamplingCoverage>,
    color: &sr_color::ColorTransform,
    gain: f32,
    output_files: Vec<String>,
    timings: BTreeMap<String, u128>,
    warnings: Vec<String>,
    started: std::time::SystemTime,
) -> RunManifest {
    let started_utc = started
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_else(|_| "unknown".into());

    RunManifest {
        program: "smokstak".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        build_profile: if cfg!(debug_assertions) { "debug".into() } else { "release".into() },
        revision: option_env!("SRSTACK_REVISION").unwrap_or("unrecorded").into(),
        started_utc,
        host: sr_diagnostics::HostInfo::default(),
        dependencies: sr_diagnostics::dependency_versions(),
        input: input.display().to_string(),
        sources: burst
            .paths
            .iter()
            .enumerate()
            .map(|(i, p)| SourceFile {
                index: i,
                path: p.display().to_string(),
                hash: burst.frames[i].metadata.sha256_prefix.clone(),
            })
            .collect(),
        reference_index: reg.choice.index,
        reference_file: burst.frames[reg.choice.index].metadata.file_name.clone(),
        reference_reason: reg.choice.reason.clone(),
        photometric_reference_index: None,
        config: cfg.clone(),
        seed: cfg.seed,
        noise: sr_diagnostics::NoiseRecord {
            alpha: burst.noise.alpha,
            beta: burst.noise.beta,
            source: burst.noise_source.clone(),
        },
        color: sr_diagnostics::ColorRecord {
            wb: color.wb,
            cam_to_srgb: color.cam_to_srgb,
            matrix_fallback: color.fallback,
            exposure_gain: gain,
        },
        output_width: product.width,
        output_height: product.height,
        output_files,
        coverage,
        stats: product.stats.clone(),
        timings_ms: timings,
        warnings,
    }
}

pub fn selftest(
    out: &Path,
    frames: usize,
    size: usize,
    keep: bool,
    photometric_match: bool,
    detect_defects: bool,
) -> Result<()> {
    crate::selftest::run(out, frames, size, keep, photometric_match, detect_defects)
}

/// The photometric match for a whole burst against its reference.
///
/// `stack` measures this per filter group instead; this is for the paths that
/// report on a burst as a whole.
fn photometry_of(
    burst: &LoadedBurst,
    reg: &RegisteredBurst,
    enabled: bool,
) -> Vec<FramePhotometry> {
    if enabled {
        // Through the stars, as the stack itself does, so that the reported
        // gain is the one the merge used and not one from a path nothing runs.
        let stars: Vec<Vec<sr_core::star::Star>> = (0..burst.frames.len())
            .into_par_iter()
            .map(|i| sr_quality::stars::positions_for_photometry(&burst.frames[i], STAR_PHOTOMETRY_LIMIT))
            .collect();
        sr_quality::photometry::match_with_stars(
            &burst.frames,
            &reg.warps,
            reg.choice.index,
            &burst.validation.exposure_scale,
            &[],
            &stars,
            true,
        )
    } else {
        sr_quality::photometry::from_exposure_only(&burst.validation.exposure_scale)
    }
}

/// Write a viewable copy of a linear result.
///
/// Astronomical output is legible only after a stretch, and the reason is worth
/// stating precisely: it is not dark, it is flat. On the reconstructions here
/// the median sits near mid-grey and the robust spread is under a percent, so
/// the whole frame reads as one shade with a few white dots on it. A daytime
/// result from the same pipeline has fifty times the spread and needs nothing.
///
/// So the stretch is chosen from the image and then *measured*: if it would not
/// widen the spread it is not applied, and the preview is simply the result at
/// a size that opens. Nothing here touches the output file.
fn write_preview(
    path: &Path,
    linear: &[Plane<f32>],
    channels: usize,
    max_edge: usize,
) -> Result<()> {
    write_preview_of(path, linear, channels, max_edge, false)
}

/// `display_referred` says the image has already been stretched, so the
/// preview must neither stretch it again nor encode it: both would be a second
/// display transform on top of the first, and the result is a washed-out image
/// that looks like the stack lost its contrast.
fn write_preview_of(
    path: &Path,
    linear: &[Plane<f32>],
    channels: usize,
    max_edge: usize,
    display_referred: bool,
) -> Result<()> {
    if display_referred {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        let view: Vec<Plane<f32>> = linear.iter().take(channels).cloned().collect();
        sr_output::write_preview_png(path, &view, max_edge)?;
        println!(
            "Preview written to {} (already stretched; shown as it is)",
            path.display()
        );
        return Ok(());
    }
    // The same rendering the 16-bit picture gets, so that the preview is a
    // preview of it. It used to take one stretch from the three channels
    // together and apply it to each, which leaves whatever colour cast the
    // linear data has: a one-shot-colour mosaic carries no white balance -- an
    // astronomy camera behind an arbitrary filter has no colorimetry to give
    // one -- so its raw sky is green, and the preview was green while the
    // picture beside it was neutral. Dividing each channel by its own ceiling,
    // which is what the picture does, is what puts the sky at the same level in
    // all three.
    let white = sr_color::stretch::ceiling_from_data(linear, channels);
    let (view, how) = sr_color::stretch::render(linear, channels, &white);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    sr_output::write_preview_png(path, &view, max_edge)?;
    match how {
        sr_color::stretch::Rendering::Stretched { per_channel } => {
            let s = per_channel.get(per_channel.len() / 2).unwrap_or(&per_channel[0]);
            let median = sr_core::math::median(
                &linear[0].data.iter().step_by(101).cloned().collect::<Vec<_>>(),
            );
            println!("Preview written to {} ({})", path.display(), s.describe(median));
        }
        sr_color::stretch::Rendering::Encoded => println!(
            "Preview written to {} (no stretch needed; the result is already legible)",
            path.display()
        ),
    }
    Ok(())
}

/// One filter's frames, and the frame inside it that others are compared to.
///
/// The *geometry* reference is global and lives on `RegisteredBurst`; this is
/// the separate question of which frame the robustness model and the
/// photometric match measure against, and it has to stay inside the group. An
/// H-alpha frame and an OIII one disagree about the entire nebula, so a model
/// that reads disagreement as motion would reject one of them wholesale.
pub struct FilterGroup {
    pub name: String,
    pub active: Vec<bool>,
    pub members: Vec<usize>,
    pub reference: usize,
}

/// A grid-only frame must not become the content reference or take part in
/// defect, photometry and rejection estimates, even in a single-filter run.
/// Its global warp remains available to define output coordinates.
fn exclude_grid_reference(groups: &mut Vec<FilterGroup>, grid: usize, qualities: &[FrameQuality]) {
    for group in groups.iter_mut() {
        group.active[grid] = false;
        group.members.retain(|&i| i != grid);
        if group.reference == grid
            && let Some(&reference) = group.members.iter().max_by(|&&a, &&b| {
                qualities[a].composite().total_cmp(&qualities[b].composite())
            }) {
                group.reference = reference;
            }
    }
    // --split-by-filter must not emit an empty master for the external filter.
    groups.retain(|group| !group.members.is_empty());
}

/// Split a burst by the filter each frame was taken through.
///
/// A burst with no filters recorded, or all the same, comes back as one group
/// named for it, so the split path and the ordinary path are the same code.
fn filter_groups(burst: &LoadedBurst, qualities: &[FrameQuality]) -> Vec<FilterGroup> {
    let mut names: Vec<String> = Vec::new();
    for f in &burst.frames {
        let n = f.metadata.filter.clone().unwrap_or_default();
        if !names.contains(&n) {
            names.push(n);
        }
    }
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let members: Vec<usize> = burst
                .frames
                .iter()
                .enumerate()
                .filter(|(_, f)| f.metadata.filter.clone().unwrap_or_default() == name)
                .map(|(i, _)| i)
                .collect();
            let mut active = vec![false; burst.frames.len()];
            for &i in &members {
                active[i] = true;
            }
            // The best frame in the group, by the same composite score the
            // global reference is chosen with. Only its *content* matters here,
            // not where it sits, because the grid is already fixed.
            let reference = *members
                .iter()
                .max_by(|&&a, &&b| {
                    qualities[a]
                        .composite()
                        .partial_cmp(&qualities[b].composite())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(&0);
            FilterGroup { name, active, members, reference }
        })
        .collect()
}

/// Insert a group name before a path's extension: `m.tif` becomes `m_H.tif`.
fn per_filter_path(base: &Path, name: &str) -> PathBuf {
    if name.is_empty() {
        return base.to_path_buf();
    }
    let stem = base.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let ext = base.extension().map(|s| s.to_string_lossy().to_string());
    let file = match ext {
        Some(e) => format!("{stem}_{name}.{e}"),
        None => format!("{stem}_{name}"),
    };
    base.with_file_name(file)
}

/// Combine separately stacked channels into one colour image.
///
/// Three problems, in this order: the masters do not share a grid, they do not
/// share a scale, and which filter goes to which primary is a choice rather
/// than a fact. Each is reported, because each is a decision a reader of the
/// result should be able to see was made.
#[allow(clippy::too_many_arguments)]
pub fn composite(
    channels: &[String],
    palette: &str,
    map: Option<&str>,
    fit: sr_composite::Fit,
    reference: Option<&str>,
    align: bool,
    assume_linear: bool,
    stretch_channels: bool,
    luminance: Option<&str>,
    luminance_strength: f32,
    output: &Path,
    preview: Option<&Path>,
    preview_size: usize,
    float_tiff: bool,
    fits: bool,
    xisf: bool,
) -> Result<()> {
    // NAME=path, so that the palette can refer to filters by name rather than
    // by the order they happened to be typed in.
    let mut loaded: Vec<sr_composite::Channel> = Vec::new();
    for spec in channels {
        let (name, path) = spec
            .split_once('=')
            .with_context(|| format!("{spec:?} is not NAME=path; for example H=h.tif"))?;
        let (mut image, floating) = sr_output::read_plane(Path::new(path))?;
        let linear = floating || assume_linear;
        if !linear {
            // What this program writes at 16 bits is sRGB-encoded, and a fit
            // through a transfer curve fits the curve as much as the sky.
            for v in image.data.iter_mut() {
                *v = sr_color::srgb_decode(*v);
            }
        }
        log::info!(
            "{name}: {}x{} from {path} ({})",
            image.width,
            image.height,
            if linear { "linear" } else { "sRGB-encoded, decoded on read" }
        );
        loaded.push(sr_composite::Channel { name: name.trim().to_string(), image });
    }
    anyhow::ensure!(!loaded.is_empty(), "no channels given");

    // Which filter goes to which primary.
    let wanted: [String; 3] = match map {
        Some(spec) => {
            let mut out = [String::new(), String::new(), String::new()];
            for part in spec.split(',') {
                let (k, v) = part
                    .split_once('=')
                    .with_context(|| format!("{part:?} is not PRIMARY=NAME; for example R=H"))?;
                let slot = match k.trim().to_ascii_uppercase().as_str() {
                    "R" => 0,
                    "G" => 1,
                    "B" => 2,
                    other => anyhow::bail!("{other:?} is not one of R, G, B"),
                };
                out[slot] = v.trim().to_string();
            }
            anyhow::ensure!(
                out.iter().all(|v| !v.is_empty()),
                "--map must name all three of R, G and B"
            );
            out
        }
        None => {
            let p = sr_composite::palette(palette)
                .with_context(|| format!("unknown palette {palette:?}"))?;
            [p[0].to_string(), p[1].to_string(), p[2].to_string()]
        }
    };
    let index_of = |name: &str| -> Result<usize> {
        loaded
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
            .with_context(|| {
                format!(
                    "no channel named {name:?}; loaded {:?}",
                    loaded.iter().map(|c| &c.name).collect::<Vec<_>>()
                )
            })
    };
    let mapping = [index_of(&wanted[0])?, index_of(&wanted[1])?, index_of(&wanted[2])?];
    let luminance = match luminance {
        Some(name) => Some(index_of(name)?),
        None => None,
    };
    println!(
        "Palette: red from {}, green from {}, blue from {}",
        wanted[0], wanted[1], wanted[2]
    );

    // The green channel by default: it carries most of the luminance a reader
    // will judge the result by, so matching the others to it keeps the frame
    // looking like what was actually measured.
    let reference = match reference {
        Some(name) => index_of(name)?,
        None => mapping[1],
    };
    println!("Measured against: {}", loaded[reference].name);

    if align {
        let cfg = sr_core::config::RegistrationConfig::default();
        let t = Instant::now();
        let a = sr_composite::align_channels(&mut loaded, reference, &cfg);
        println!("\nAlignment (output pixels):");
        for r in &a {
            if r.confidence <= 0.0 {
                println!("  {:<4} could not be aligned; left as it was", r.name);
                continue;
            }
            println!(
                "  {:<4} shift {:+7.2}, {:+7.2}   rotation {:+.4} deg   residual {:.3}",
                r.name, r.shift.0, r.shift.1, r.rotation_deg, r.residual
            );
        }
        log::info!("alignment in {:.2?}", t.elapsed());
    }

    let f = sr_composite::fit_channels(&mut loaded, reference, fit);
    if fit != sr_composite::Fit::None {
        println!("\nScale match ({fit:?}):");
        for r in &f {
            if r.blocks == 0 {
                continue;
            }
            println!(
                "  {:<4} gain {:.4}  offset {:+.5}   from {} of {} blocks",
                r.name, r.gain, r.offset, r.inliers, r.blocks
            );
        }
    }

    // Each channel stretched on its own histogram, before they are combined.
    //
    // In linear light a narrowband palette shows which filter is brightest,
    // and for SHO that is hydrogen by a wide margin: the result is green with
    // the other two filters buried in it. Stretching each channel separately
    // is what makes a palette show *where* the filters differ instead — it is
    // a display transform, applied per channel, and it is not a measurement.
    // The fit above still runs first, so the backgrounds start together.
    if stretch_channels {
        println!("
Per-channel stretch (display transform, before combining):");
        for c in loaded.iter_mut() {
            let one = std::slice::from_ref(&c.image);
            let median = sr_color::stretch::background(one, 1).map(|(m, _)| m);
            match (sr_color::stretch::choose(one, 1), median) {
                (Some(st), Some(m)) => {
                    c.image = sr_color::stretch::apply(one, 1, &st).remove(0);
                    println!("  {:<4} {}", c.name, st.describe(m));
                }
                _ => println!("  {:<4} left as it was; no usable spread to stretch", c.name),
            }
        }
    }

    let mut rgb = sr_composite::combine(&loaded, mapping);

    // Luminance last, after the colour channels have been matched to each
    // other and stretched: it replaces their brightness, so anything that
    // changes their brightness has to have happened already.
    if let Some(li) = luminance {
        let mut plane = loaded[li].image.clone();
        if stretch_channels {
            let one = std::slice::from_ref(&plane);
            if let Some(st) = sr_color::stretch::choose(one, 1) {
                plane = sr_color::stretch::apply(one, 1, &st).remove(0);
            }
        }
        anyhow::ensure!(
            plane.width == rgb[0].width && plane.height == rgb[0].height,
            "the luminance is {}x{} and the colour is {}x{}; they have to share a grid",
            plane.width,
            plane.height,
            rgb[0].width,
            rgb[0].height
        );
        sr_composite::apply_luminance(&mut rgb, &plane, luminance_strength);
        println!(
            "Luminance from {} at strength {:.2}: it carries the detail, the colour \
             channels carry the hue",
            loaded[li].name, luminance_strength
        );
    }
    let rgb = rgb;
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    // A stretched channel is already display-referred; putting it through the
    // transfer curve as well would apply two display transforms.
    if stretch_channels {
        sr_output::write_rgb16(output, &rgb)?;
    } else {
        sr_output::write_rgb16(output, &sr_color::to_rendered(&rgb))?;
    }
    println!(
        "\nWrote {} ({} x {})",
        output.display(),
        rgb[0].width,
        rgb[0].height
    );
    if let Some(p) = preview {
        write_preview_of(p, &rgb, 3, preview_size, stretch_channels)?;
    }
    for p in sr_output::write_scientific_copies(output, &rgb, 3, fits, xisf)? {
        println!("Wrote {} (32-bit float master)", p.display());
    }
    if float_tiff {
        let p = output.with_extension("linear.tif");
        sr_output::write_rgb32f(&p, &rgb)?;
        println!("Wrote {}", p.display());
    }
    Ok(())
}

/// A finished image to be measured, whoever wrote it.
///
/// TIFF is what this program writes. XISF is what other software writes, and
/// being able to measure such a master without exporting it again is the point
/// of reading the format: a comparison is only worth anything if both sides are
/// measured by the same code on the same numbers.
/// A single-channel image is replicated across the three, as the TIFF path
/// already does, so a mono master measures as a mono master rather than failing.
fn read_finished(path: &Path) -> Result<[Plane<f32>; 3]> {
    if sr_raw::format_of(path) != Some(sr_raw::Format::Xisf) {
        return sr_output::read_rgb(path);
    }
    let (image, planes) = sr_raw::xisf::read_planes(path)?;
    anyhow::ensure!(!planes.is_empty(), "{}: no channels", path.display());
    let pick = |c: usize| planes[c.min(planes.len() - 1)].clone();
    log::info!(
        "{}: {}x{} {} in {:?}, {}",
        path.display(),
        image.width,
        image.height,
        image.colour_space,
        image.format,
        if image.image_type.is_empty() { "no image type".into() } else { image.image_type.clone() }
    );
    Ok([pick(0), pick(1), pick(2)])
}

/// Measure resolution and noise on finished images.
///
/// Results are reported in cycles per *sensor* pixel as well as per output
/// pixel. Only the former can be compared between a 1x and a 2x result: an
/// upscaled image trivially has a lower MTF50 per output pixel without
/// resolving anything more, and quoting that number would flatter every
/// upscaler ever written.
/// Add the accumulators of several batches into one image.
///
/// This is what makes a set larger than memory, or larger than one person's
/// hard drive, reconstructable: each batch is stacked on the same grid with
/// `--reference-file` and writes what it contributed with `--accumulate`, and
/// this sums those contributions. Summing weights and weighted values is
/// exactly what the merge does within a batch, so a set split into ten pieces
/// and recombined is the same arithmetic as one run over all of it -- not an
/// average of averages, which would weight a thin batch like a thick one.
pub fn combine(
    dirs: &[PathBuf],
    output: &Path,
    preview: Option<&Path>,
    preview_size: usize,
    float_tiff: bool,
    fits: bool,
    xisf: bool,
) -> Result<()> {
    anyhow::ensure!(!dirs.is_empty(), "no accumulators to combine");

    let mut sum: Vec<Plane<f32>> = Vec::new();
    let mut weight: Vec<Plane<f32>> = Vec::new();
    let mut channels = 0usize;
    let mut frames = 0u64;
    let mut grid: Option<(usize, usize, String)> = None;

    for dir in dirs {
        let text = std::fs::read_to_string(dir.join("accumulator.json"))
            .with_context(|| format!("{} is not an accumulator directory", dir.display()))?;
        let meta: serde_json::Value = serde_json::from_str(&text)?;
        let w = meta["width"].as_u64().unwrap_or(0) as usize;
        let h = meta["height"].as_u64().unwrap_or(0) as usize;
        let c = meta["channels"].as_u64().unwrap_or(0) as usize;
        let reference = meta["reference_sha256_prefix"].as_str().unwrap_or("").to_string();
        frames += meta["frames"].as_u64().unwrap_or(0);

        match &grid {
            None => {
                grid = Some((w, h, reference));
                channels = c;
                for k in 0..c {
                    sum.push(sr_output::read_plane(&dir.join(format!("sum-{k}.tif")))?.0);
                    weight.push(sr_output::read_plane(&dir.join(format!("weight-{k}.tif")))?.0);
                }
            }
            Some((gw, gh, gref)) => {
                anyhow::ensure!(
                    w == *gw && h == *gh && c == channels,
                    "{} is {w}x{h} in {c} channels and the first is {gw}x{gh} in \
                     {channels}; accumulators can only be added on a common grid",
                    dir.display()
                );
                anyhow::ensure!(
                    &reference == gref,
                    "{} was reconstructed against a different reference frame; pass the \
                     same --reference-file to every batch or the pixels do not correspond",
                    dir.display()
                );
                for k in 0..channels {
                    let (n, _) = sr_output::read_plane(&dir.join(format!("sum-{k}.tif")))?;
                    let (d, _) = sr_output::read_plane(&dir.join(format!("weight-{k}.tif")))?;
                    for (a, b) in sum[k].data.iter_mut().zip(&n.data) {
                        *a += *b;
                    }
                    for (a, b) in weight[k].data.iter_mut().zip(&d.data) {
                        *a += *b;
                    }
                }
            }
        }
    }
    let (w, h, _) = grid.expect("at least one accumulator");

    let mut rgb = [Plane::new(0, 0), Plane::new(0, 0), Plane::new(0, 0)];
    let mut unsupported = 0usize;
    for k in 0..channels {
        let mut p = Plane::<f32>::new(w, h);
        for i in 0..w * h {
            let d = weight[k].data[i];
            if d > 0.0 {
                p.data[i] = sum[k].data[i] / d;
            } else {
                unsupported += 1;
            }
        }
        rgb[k] = p;
    }

    println!(
        "Combined {} accumulators: {frames} frames, {w} x {h}, {channels} channel(s)",
        dirs.len()
    );
    if unsupported > 0 {
        println!(
            "  {unsupported} of {} channel-pixels had no coverage in any batch and are zero",
            w * h * channels
        );
    }
    let gain = sr_color::normalise_exposure(&mut rgb, 0.9995, 1.0);
    log::info!("exposure normalisation gain {gain:.3}");

    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    sr_output::write_product16(output, &sr_color::to_rendered(&rgb), channels)?;
    println!("Wrote {} ({w} x {h})", output.display());
    if let Some(p) = preview {
        write_preview(p, &rgb, channels, preview_size)?;
    }
    for p in sr_output::write_scientific_copies(output, &rgb, channels, fits, xisf)? {
        println!("Wrote {} (32-bit float master)", p.display());
    }
    if float_tiff {
        let p = output.with_extension("linear.tif");
        sr_output::write_product32f(&p, &rgb, channels)?;
        println!("Wrote {} (32-bit float, linear)", p.display());
    }
    Ok(())
}

pub fn measure(
    inputs: &[String],
    edge_box: Option<(usize, usize, usize, usize)>,
    edge: Option<(f32, f32, f32, f32)>,
    flat_box: Option<(usize, usize, usize, usize)>,
    ca_centre: Option<(f32, f32)>,
) -> Result<()> {
    println!(
        "{:<28} {:>9} {:>7} {:>8} {:>9} {:>11} {:>11} {:>9} {:>10} {:>9} {:>8}",
        "image", "size", "scale", "angle", "straight", "MTF50/out", "MTF50/sensor", "overshoot",
        "noise", "fringing", "residCA"
    );

    for token in inputs {
        let (path_str, scale) = match token.rsplit_once('@') {
            Some((p, s)) => {
                let v: f32 = s
                    .parse()
                    .map_err(|_| anyhow::anyhow!("bad scale in {token:?}"))?;
                anyhow::ensure!(v > 0.0, "scale in {token:?} must be positive");
                (p, v)
            }
            None => (token.as_str(), 1.0f32),
        };
        let path = Path::new(path_str);
        let rgb = read_finished(path)?;
        let luma = sr_output::luma(&rgb);
        let fringing = sr_synth::metrics::edge_chroma_fringing(&rgb);

        // Residual channel misregistration, measured on the finished image with
        // the same estimator used on the input. This closes the loop on a
        // chromatic-aberration correction: it should leave nothing behind.
        // `rggb` is false because an RGB image has no mosaic lattice offset.
        let guide = sr_core::frame::GuideImage {
            width: rgb[0].width,
            height: rgb[0].height,
            r: rgb[0].clone(),
            g: rgb[1].clone(),
            b: rgb[2].clone(),
        };
        let residual_ca = sr_register::chroma::estimate_about(
            &guide, 96, 12, 1.10, false, ca_centre,
        );
        // Reported in the image's own pixels, and this guide is full resolution
        // rather than half, so the estimator's doubling is undone.
        let ca_corner = residual_ca.corner_shift[0].max(residual_ca.corner_shift[2]) * 0.5;

        let found = match edge {
            Some((x0, y0, x1, y1)) => Some(sr_synth::metrics::FoundEdge {
                line: (x0 * scale, y0 * scale, x1 * scale, y1 * scale),
                angle_deg: f32::NAN,
                contrast: f32::NAN,
                straightness: f32::NAN,
            }),
            // No edge asked for, and none guessed at: the noise, fringing and
            // residual CA columns do not need one, and a deep-sky frame has no
            // slanted edge to find. The row prints with "no edge" in the
            // resolution columns.
            None => edge_box.and_then(|(bx, by, bw, bh)| {
                sr_synth::metrics::fit_edge_in_box(
                    &luma,
                    (bx as f32 * scale) as usize,
                    (by as f32 * scale) as usize,
                    (bw as f32 * scale) as usize,
                    (bh as f32 * scale) as usize,
                )
            }),
        };

        let noise_plane = match flat_box {
            Some((x, y, w, h)) => {
                let (x, y) = ((x as f32 * scale) as usize, (y as f32 * scale) as usize);
                let (w, h) = ((w as f32 * scale) as usize, (h as f32 * scale) as usize);
                if x + w <= luma.width && y + h <= luma.height {
                    luma.crop(x, y, w, h)
                } else {
                    luma.clone()
                }
            }
            None => luma.clone(),
        };
        let noise = sr_synth::metrics::flat_field_noise(&noise_plane, (16.0 * scale) as usize);

        let Some(found) = found else {
            println!(
                "{:<28} {:>9} {:>7.2} {:>8} {:>9} {:>11} {:>11} {:>9} {:>10.6} {:>9.6} {:>8.3}",
                path_str,
                format!("{}x{}", luma.width, luma.height),
                scale,
                "-",
                "-",
                "no edge",
                "-",
                "-",
                noise,
                fringing,
                ca_corner
            );
            continue;
        };

        // The analysis window scales with the image so the same amount of
        // scene is used whatever the magnification.
        let half_width = 12.0 * scale;
        let half_length = 40.0 * scale;
        let mtf = sr_synth::metrics::slanted_edge_mtf(&luma, found.line, half_width, half_length);

        match mtf {
            Some(m) => println!(
                "{:<28} {:>9} {:>7.2} {:>8.2} {:>9.3} {:>11.4} {:>11.4} {:>9.3} {:>10.6} {:>9.6} {:>8.3}",
                path_str,
                format!("{}x{}", luma.width, luma.height),
                scale,
                found.angle_deg,
                found.straightness,
                m.mtf50,
                m.mtf50 * scale,
                m.overshoot,
                noise,
                fringing,
                ca_corner
            ),
            None => println!(
                "{:<28} {:>9} {:>7.2} {:>8.2} {:>9.3} {:>11} {:>11} {:>9} {:>10.6} {:>9.6} {:>8.3}",
                path_str,
                format!("{}x{}", luma.width, luma.height),
                scale,
                found.angle_deg,
                found.straightness,
                "unmeasurable",
                "-",
                "-",
                noise,
                fringing,
                ca_corner
            ),
        }
    }

    println!(
        "\nMTF50 is in cycles per pixel. Compare the sensor-referred column: it is the\n\
         only one that says whether real detail was recovered rather than resampled.\n\
         An edge with straightness below about 0.8 is not a clean single edge and its\n\
         MTF should not be trusted. `residCA` is the channel misregistration still\n\
         present in the image, in pixels at the furthest corner from the optical\n\
         centre. On a crop, pass --ca-centre or that column is meaningless."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn photometric_rejection_records_the_gate_that_actually_excluded_a_frame() {
        assert!(super::photometric_exclusion(0.30,1.0).is_none());
        let obstructed=super::photometric_exclusion(0.359375,0.62).unwrap();
        assert!(obstructed.contains("35.94%") && obstructed.contains(">30%"));
        let gradient=super::photometric_exclusion(0.0,1.01).unwrap();
        assert!(gradient.contains("101.00%") && gradient.contains(">100%"));
    }
    use super::*;

    fn small_group_frame(filter: &str, beta: f32) -> RawFrame {
        RawFrame {
            width: 8, height: 8,
            samples: sr_core::samples::SamplePlane::from_normalised(8, 8, vec![0.2; 64]),
            cfa: sr_core::cfa::CfaPattern::MONO,
            defects: sr_core::samples::DefectMask::none(8, 8),
            noise: sr_core::frame::NoiseModel::new(0.0, beta, sr_core::frame::NoiseSource::Nominal),
            metadata: sr_core::frame::FrameMetadata { filter: Some(filter.into()), ..Default::default() },
        }
    }

    #[test]
    fn filter_noise_excludes_other_filters_failed_geometry_and_grid_reference() {
        let mut frames = vec![small_group_frame("O", 0.5), small_group_frame("H", 1e-6),
            small_group_frame("O", 1e-4), small_group_frame("H", 0.1)];
        let qualities = vec![FrameQuality::default(); 4];
        let mut groups = vec![
            FilterGroup { name: "H".into(), active: vec![false,true,false,true], members: vec![1,3], reference: 1 },
            FilterGroup { name: "O".into(), active: vec![true,false,true,false], members: vec![0,2], reference: 0 },
        ];
        exclude_grid_reference(&mut groups, 0, &qualities);
        let mut registrations: Vec<_> = (0..4).map(sr_register::global::GlobalRegistration::identity).collect();
        registrations[3].confidence = 0.004;
        registrations[3].residual_p50 = 23.0;
        let h = registered_group(&groups[0], &registrations).unwrap();
        let o = registered_group(&groups[1], &registrations).unwrap();
        assert_eq!(h.members, [1]);
        assert_eq!(o.members, [2]);
        let h_noise = group_noise(&frames, &h.members);
        let o_noise = group_noise(&frames, &o.members);
        assert_eq!(h_noise.0, frames[1].noise);
        assert_eq!(o_noise.0, frames[2].noise);
        assert_ne!(h_noise.0, o_noise.0);
        frames[0].noise.beta = 100.0;
        frames[2].noise.beta = 10.0;
        frames[3].noise.beta = 20.0;
        assert_eq!(group_noise(&frames, &h.members), h_noise);
        assert_eq!(group_noise(&frames, &o.members).0, frames[2].noise);
    }

    #[test]
    fn filter_masks_preserve_decode_defects_without_cross_filter_leakage() {
        let mut frames = vec![small_group_frame("H", 1e-6), small_group_frame("H", 1e-6), small_group_frame("O", 1e-4)];
        frames[0].defects.set(3);
        frames[2].defects.set(4);
        let decode: Vec<_> = frames.iter().map(|f| f.defects.clone()).collect();
        let mut h_mask = sr_core::samples::DefectMask::none(8, 8);
        h_mask.set(7);
        install_group_mask(&mut frames, &[0, 1], &h_mask);
        assert!(frames[0].defects.get(3), "decode defect must survive sensor-mask application");
        assert!(frames[0].defects.get(7));
        assert!(frames[1].defects.get(7));
        assert_eq!(frames[2].defects, decode[2], "H must never mask O");
        // O may return early without any mask (no motion, no defects, disabled
        // detection). Reset alone must restore the exact original state.
        reset_decode_masks(&mut frames, &decode);
        for (frame, original) in frames.iter().zip(&decode) { assert_eq!(&frame.defects, original); }
        let mut o_mask = sr_core::samples::DefectMask::none(8, 8);
        o_mask.set(9);
        install_group_mask(&mut frames, &[2], &o_mask);
        assert!(frames[2].defects.get(4));
        assert!(frames[2].defects.get(9));
        assert!(!frames[2].defects.get(7));
        assert_eq!(frames[0].defects, decode[0]);
        assert_eq!(frames[1].defects, decode[1]);
        assert!(!decode[0].get(7), "original shared mask must remain immutable");
        assert!(!decode[2].get(9));
    }

    #[test]
    fn a_frame_is_named_against_the_rest_of_its_own_filter() {
        let row = |file: &str, filter: &str, hfd: f32, ecc: f32, count: Option<usize>| SurveyRow {
            path: format!("G:/data/{file}"),
            file: file.into(),
            filter: filter.into(),
            // Gradient energy, on a scale of its own, where no stars were found.
            sharpness: Some(if count.is_some() { 1.0 / hfd } else { 5.0 }),
            hfd: count.map(|_| hfd),
            eccentricity: count.map(|_| ecc),
            stars: count,
            ..Default::default()
        };
        let mut rows = vec![
            row("h0", "H", 3.0, 0.20, Some(200)),
            row("h1", "H", 3.0, 0.22, Some(210)),
            row("h2", "H", 3.1, 0.18, Some(190)),
            row("h3", "H", 3.0, 0.20, None),
            row("h4", "H", 3.0, 0.60, Some(200)),
            row("h5", "H", 3.0, 0.20, Some(40)),
            row("h6", "H", 5.0, 0.20, Some(200)),
            row("h7", "H", 2.9, 0.21, Some(205)),
            // A tenth of hydrogen's stars, and not clouded for it.
            row("o0", "O", 3.0, 0.20, Some(20)),
            row("o1", "O", 3.0, 0.20, Some(21)),
            row("o2", "O", 3.0, 0.20, Some(22)),
            row("o3", "O", 3.0, 0.20, Some(20)),
            row("o4", "O", 3.0, 0.20, Some(21)),
            SurveyRow {
                path: "G:/data/bad".into(),
                file: "bad".into(),
                filter: "H".into(),
                error: "truncated".into(),
                ..Default::default()
            },
        ];
        flag_frames(&mut rows);
        let kinds = |f: &str| -> Vec<&str> {
            rows.iter().find(|r| r.file == f).unwrap().flags.iter().map(|f| f.kind).collect()
        };

        assert!(kinds("h0").is_empty());
        assert_eq!(kinds("h3"), ["no-stars"]);
        assert_eq!(kinds("h4"), ["elongated"]);
        assert_eq!(kinds("h5"), ["few-stars"]);
        assert_eq!(kinds("h6"), ["soft"]);
        assert_eq!(kinds("bad"), ["unreadable"]);
        for o in ["o0", "o1", "o2", "o3", "o4"] {
            assert!(kinds(o).is_empty(), "{o}: {:?}", kinds(o));
        }
        assert!((rows[0].sharpness.unwrap() - 1.0).abs() < 0.05, "{:?}", rows[0].sharpness);
        assert_eq!(rows[3].sharpness, None, "no number comparable to the others'");
    }

    #[test]
    fn rejected_frames_cannot_change_chroma_sample_selection() {
        let weights = vec![1.0; 11];
        let selected = chroma_sample_indices(&weights, 3);
        let mut extended = vec![0.0];
        for &w in &weights { extended.extend([w, 0.0]); }
        let extended_selected = chroma_sample_indices(&extended, 7);
        assert!(extended_selected.iter().all(|&i| extended[i] > 0.0));
        assert_eq!(selected, extended_selected.iter().map(|i| (i - 1) / 2).collect::<Vec<_>>());
        assert_eq!(chroma_sample_indices(&[0.0, f32::NAN, 1.0, f32::INFINITY, -1.0], 0), vec![2]);
        assert!(chroma_sample_indices(&[0.0; 4], 0).is_empty());
        assert_eq!(chroma_sample_indices(&[1.0; 3], 1), vec![0, 1, 2]);
    }

    #[test]
    fn grid_only_reference_cannot_define_content_or_emit_a_filter_master() {
        let qualities: Vec<_> = [100., 1., 2.].into_iter().map(|sharpness| FrameQuality {
            sharpness, ..Default::default()
        }).collect();
        let mut ordinary = vec![FilterGroup {
            name: String::new(), active: vec![true; 3], members: vec![0, 1, 2], reference: 0,
        }];
        exclude_grid_reference(&mut ordinary, 0, &qualities);
        assert_eq!(ordinary[0].active, [false, true, true]);
        assert_eq!(ordinary[0].members, [1, 2]);
        assert_eq!(ordinary[0].reference, 2);
        let mut split = vec![
            FilterGroup { name: "O".into(), active: vec![true, false, false], members: vec![0], reference: 0 },
            FilterGroup { name: "H".into(), active: vec![false, true, true], members: vec![1, 2], reference: 2 },
        ];
        exclude_grid_reference(&mut split, 0, &qualities);
        assert_eq!(split.len(), 1);
        assert_eq!(split[0].name, "H");
        assert_eq!(split[0].active, ordinary[0].active);
        assert_eq!(split[0].members, ordinary[0].members);
        assert_eq!(split[0].reference, ordinary[0].reference);
        // A content reference already inside the data remains unchanged.
        ordinary[0].reference = 1;
        exclude_grid_reference(&mut ordinary, 0, &qualities);
        assert_eq!(ordinary[0].reference, 1);
    }

    #[test]
    fn failed_alignment_cannot_enter_photometry_even_with_a_small_positive_weight() {
        let mut regs: Vec<_> = (0..3).map(sr_register::global::GlobalRegistration::identity).collect();
        regs[1].confidence = 0.004;
        regs[1].residual_p50 = 23.0;
        // Low overlap alone is not a reason to discard accurate geometry.
        regs[2].confidence = 0.03;
        regs[2].residual_p50 = 0.1;
        let group = FilterGroup { name: "test".into(), active: vec![true; 3], members: vec![0,1,2], reference: 1 };
        let valid = registered_group(&group, &regs).unwrap();
        assert_eq!(valid.active, vec![true, false, true]);
        assert_eq!(valid.members, vec![0,2]);
        assert_eq!(valid.reference, 0);
        assert!(registered_group(&FilterGroup {
            name: "bad".into(), active: vec![false,true,false], members: vec![1], reference: 1,
        }, &regs).is_err());
    }

    /// The property that makes the cache worth having, and the one that makes
    /// it dangerous, are the same property: a fingerprint has to ignore exactly
    /// the things registration ignores and nothing more.
    #[test]
    fn tuning_parameters_do_not_invalidate_the_alignment() {
        let base = ReconstructionConfig::default();
        let key = |c: &ReconstructionConfig| registration_fingerprint_of(&[], c, true, None);
        let b = key(&base);

        // What an operator changes between runs. None of it reaches
        // registration, so none of it may change the fingerprint — otherwise
        // the cache never hits and the whole thing is dead weight.
        for changed in [
            ReconstructionConfig {
                scale: 3.0,
                ..base.clone()
            },
            ReconstructionConfig {
                tile: 1024,
                ..base.clone()
            },
            ReconstructionConfig {
                roi: Some((1, 2, 3, 4)),
                ..base.clone()
            },
            ReconstructionConfig {
                backend: Backend::CfaDrizzle,
                ..base.clone()
            },
            ReconstructionConfig {
                postprocess: PostProcess::Mild,
                ..base.clone()
            },
            ReconstructionConfig {
                flatten_background: true,
                ..base.clone()
            },
            ReconstructionConfig {
                kernel: sr_core::config::KernelConfig {
                    radius: 4.0,
                    ..base.kernel
                },
                ..base.clone()
            },
        ] {
            assert_eq!(key(&changed), b, "a tuning parameter invalidated the cache");
        }
    }

    #[test]
    fn anything_registration_reads_does_invalidate_it() {
        let base = ReconstructionConfig::default();
        let key = |c: &ReconstructionConfig| registration_fingerprint_of(&[], c, true, None);
        let b = key(&base);

        for changed in [
            ReconstructionConfig {
                registration: sr_core::config::RegistrationConfig {
                    global_patch: 64,
                    ..base.registration
                },
                ..base.clone()
            },
            ReconstructionConfig {
                local_warp: sr_core::config::LocalWarpMode::On,
                ..base.clone()
            },
            ReconstructionConfig {
                reference: Some(3),
                ..base.clone()
            },
            ReconstructionConfig {
                warp: sr_core::config::WarpConfig {
                    spacing: 99,
                    ..base.warp
                },
                ..base.clone()
            },
        ] {
            assert_ne!(key(&changed), b, "a registration parameter was not fingerprinted");
        }

        // Which metric ranked the frames decides which one was chosen as the
        // reference, so it decides the alignment too.
        assert_ne!(
            registration_fingerprint_of(&[], &base, false, None),
            registration_fingerprint_of(&[], &base, true, None)
        );
    }

    #[test]
    fn named_reference_grid_invalidates_automatic_and_other_named_alignment() {
        let cfg = ReconstructionConfig::default();
        let key = |reference| registration_fingerprint_of(&[], &cfg, true, reference);
        assert_ne!(key(None), key(Some(0)));
        assert_ne!(key(Some(0)), key(Some(1)));
        assert_eq!(key(Some(0)), key(Some(0)));
    }
}
