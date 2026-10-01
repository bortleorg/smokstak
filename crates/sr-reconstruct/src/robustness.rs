//! Stage 9: moving-object and occlusion rejection.
//!
//! For every frame we ask, per region: does this frame agree with the reference
//! to within what noise and registration uncertainty can explain? Where it does
//! not, something moved — a leaf, a bird, a car, a specular highlight — or the
//! registration failed, and those samples must not be averaged in.
//!
//! Three things make this work better than sigma-clipping demosaiced RGB:
//!
//! * the comparison is against a *noise model*, so the threshold adapts to
//!   signal level instead of being a magic constant;
//! * the expected disagreement from residual misalignment is added to the
//!   tolerance, in proportion to the local gradient, so textured regions are
//!   not rejected for being textured;
//! * the result is a soft weight, not a binary mask, so a marginal sample
//!   fades out rather than producing a visible seam.

use rayon::prelude::*;
use std::path::Path;

use sr_core::config::RobustnessConfig;
use sr_quality::photometry::PhotometricMatch;
use sr_core::frame::{GuideImage, NoiseModel, RawFrame};
use sr_core::geometry::WarpField;
use sr_core::plane::Plane;

/// Per-frame robustness at guide resolution, quantised to a byte.
///
/// A byte is plenty for a weight in `[0, 1]` and keeps a hundred maps for a
/// 45 MP sensor under half a gigabyte.
pub struct RobustnessMaps {
    pub width: usize,
    pub height: usize,
    pub maps: Vec<Plane<u8>>,
    /// Fraction of each frame suppressed below `reject_below`.
    pub rejected_fraction: Vec<f32>,
    /// Aligned burst consensus, for structural decisions independent of one
    /// reference exposure's noise. Absent when rejection was disabled.
    pub consensus_luma: Option<Plane<f32>>,
}

impl RobustnessMaps {
    /// Weight for a reference *sensor* coordinate.
    ///
    /// Interpolated, not sampled. The map is at guide resolution — one value
    /// per 2x2 block of sensor sites — and reading it by truncation gives every
    /// sensor pixel in a block the same weight, with a step at each block
    /// boundary. That does not merely look blocky: on a burst where a fifth of
    /// the samples are being suppressed, it modulates how many frames reach
    /// each output pixel on a two-pixel grid, and the result carries the grid
    /// as background structure. Interpolating costs three multiplies and turns
    /// the steps into a slope.
    ///
    /// The guide cell `(gx, gy)` averages the sensor sites starting at
    /// `(2gx, 2gy)`, so it stands at sensor coordinate `2gx + 0.5` — hence the
    /// half-pixel in the mapping back.
    #[inline]
    pub fn at(&self, frame: usize, rx: f32, ry: f32) -> f32 {
        if self.maps.is_empty() {
            return 1.0;
        }
        let m = &self.maps[frame];
        // A frame outside the active set has no map. It is not going to be
        // merged either, so the value never reaches an accumulator; returning
        // full trust keeps it from looking like a rejection in a diagnostic.
        if m.data.is_empty() {
            return 1.0;
        }
        let (w, h) = (m.width, m.height);
        if rx < 0.0 || ry < 0.0 || rx >= (2 * w) as f32 || ry >= (2 * h) as f32 {
            return 0.0;
        }
        // Clamped rather than extrapolated: the outer half-cell has nothing
        // beyond it to lean on, and holding the edge value there is what the
        // truncating version did too.
        let u = ((rx - 0.5) * 0.5).clamp(0.0, (w - 1) as f32);
        let v = ((ry - 0.5) * 0.5).clamp(0.0, (h - 1) as f32);
        let (x0, y0) = (u as usize, v as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (fx, fy) = (u - x0 as f32, v - y0 as f32);
        let p = |x: usize, y: usize| m.data[y * w + x] as f32;
        let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * fx;
        let bottom = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * fx;
        (top + (bottom - top) * fy) * (1.0 / 255.0)
    }

    pub fn plane(&self, frame: usize) -> Plane<f32> {
        let m = &self.maps[frame];
        if m.data.is_empty() {
            return Plane::filled(self.width, self.height, 1.0);
        }
        Plane::from_vec(
            m.width,
            m.height,
            m.data.iter().map(|&v| v as f32 / 255.0).collect(),
        )
    }

    /// Mean robustness across frames, as a diagnostic of where the burst
    /// disagreed with itself.
    pub fn mean_plane(&self) -> Plane<f32> {
        let (w, h) = (self.width, self.height);
        let mut out = Plane::<f32>::new(w, h);
        if self.maps.is_empty() {
            return Plane::filled(w, h, 1.0);
        }
        let mut n = 0usize;
        for m in &self.maps {
            if m.data.len() != w * h {
                continue;
            }
            n += 1;
            for i in 0..w * h {
                out.data[i] += m.data[i] as f32;
            }
        }
        let s = 1.0 / (255.0 * n.max(1) as f32);
        for v in out.data.iter_mut() {
            *v *= s;
        }
        out
    }
}

/// The scene as the whole burst agrees on it, at guide resolution, in the
/// reference's coordinates.
///
/// Every frame used to be judged against the reference frame, and that is one
/// exposure with one exposure's noise in it. The comparison therefore carried
/// the reference's own grain into all the others, in the same place every time,
/// and whatever it caused -- a rejection, a fractional weight -- repeated
/// identically in every frame and survived the average as structure. A mean of
/// the burst has a hundredth of the noise of any one frame and no structure of
/// its own to imprint: the integration, not a neighbour.
///
/// Two passes, because a plain mean is not robust enough to be a reference. One
/// satellite in one frame moves a mean of six by a sixth of its brightness,
/// which is enough that the five innocent frames then disagree with the mean
/// and are rejected -- the test for exactly that failed on the one-pass
/// version. The second pass weights each frame's contribution down by how far
/// it stood from the first, so an outlier is crushed while the ordinary
/// spread between frames is left alone.
///
/// Returns sums and counts rather than a mean, because what judges a frame has
/// to be the burst *without* that frame: see [`without`].
#[allow(clippy::too_many_arguments)]
fn consensus_guide(
    frames: &[RawFrame],
    warps: &[WarpField],
    active: &[bool],
    photometry: &[PhotometricMatch],
    noise: &NoiseModel,
    channels: usize,
    gw: usize,
    gh: usize,
) -> Consensus {
    let n = gw * gh;
    /// One sweep's product: the weighted sums, the weights they were built
    /// from, and how many frames looked at each site regardless of weight.
    type Sweep = (Vec<Plane<f32>>, Vec<f32>, Vec<u16>);
    let empty =
        || -> Sweep { (vec![Plane::<f32>::new(gw, gh); channels], vec![0.0f32; n], vec![0u16; n]) };

    // `first` is None on the opening pass and the running mean on the second.
    let sweep = |first: Option<&Sweep>| {
        (0..frames.len())
            .into_par_iter()
            .filter(|&i| active.get(i).copied().unwrap_or(false))
            .fold(empty, |(mut sums, mut counts, mut seen), i| {
                let guide = frames[i].structure_guide_rgb();
                let stats: Vec<Plane<f32>> =
                    (0..channels).map(|c| local_stats(guide.channel(c)).0).collect();
                let photo =
                    photometry.get(i).copied().unwrap_or(PhotometricMatch::IDENTITY);
                let warp = &warps[i];
                for gy in 0..gh {
                    for gx in 0..gw {
                        let Some((sx, sy)) =
                            warp.inverse_map(gx as f32 * 2.0 + 0.5, gy as f32 * 2.0 + 0.5)
                        else {
                            continue;
                        };
                        let (tx, ty) = ((sx - 0.5) * 0.5, (sy - 0.5) * 0.5);
                        if tx < 0.0 || ty < 0.0 || tx > (guide.width - 1) as f32 || ty > (guide.height - 1) as f32
                        {
                            continue;
                        }
                        let idx = gy * gw + gx;
                        let (u, v) =
                            (2.0 * gx as f32 / gw as f32 - 1.0, 2.0 * gy as f32 / gh as f32 - 1.0);
                        // An obstructed frame did not look at this site.
                        if photo.blocked_at(u, v) {
                            continue;
                        }
                        // Counted before any weighting: how many frames looked
                        // at this site at all, which is a different question
                        // from how much they agreed.
                        seen[idx] += 1;
                        // One weight for the whole site, from the channel that
                        // disagrees most, so a frame is not half in and half
                        // out of its own consensus.
                        let mut w = 1.0f32;
                        if let Some((s0, c0, _)) = first {
                            if c0[idx] > 0.0 {
                                for c in 0..channels {
                                    let mine = photo.apply_at(
                                        c,
                                        stats[c].bilinear(tx, ty),
                                        u,
                                        v,
                                    );
                                    let mean = s0[c].data[idx] / c0[idx];
                                    w = w.min(outlier_weight(mine, mean, noise));
                                }
                            }
                        }
                        if w <= 0.0 {
                            continue;
                        }
                        for c in 0..channels {
                            sums[c].data[idx] +=
                                w * photo.apply_at(c, stats[c].bilinear(tx, ty), u, v);
                        }
                        counts[idx] += w;
                    }
                }
                (sums, counts, seen)
            })
            .reduce(empty, |(mut a, mut ac, mut asn), (b, bc, bsn)| {
                for c in 0..a.len() {
                    for i in 0..n {
                        a[c].data[i] += b[c].data[i];
                    }
                }
                for i in 0..n {
                    ac[i] += bc[i];
                    asn[i] += bsn[i];
                }
                (a, ac, asn)
            })
    };

    // Three sweeps, not two. The opening mean is contaminated by whatever it is
    // meant to exclude -- one satellite in six frames drags it a sixth of the
    // way to the satellite -- and against a mean pulled that far off, the five
    // innocent frames look like outliers too and the site collapses. One
    // further sweep is enough: with the outlier already weighted down, the mean
    // returns to the clean value and the innocent frames come back to full
    // weight. Measured on exactly that case, the weights go 0.17 to 0.98 while
    // the outlier goes 0.008 to 0.004.
    let mut prior = sweep(None);
    for _ in 0..CONSENSUS_SWEEPS - 2 {
        prior = sweep(Some(&prior));
    }
    let (sums, counts, seen) = sweep(Some(&prior));
    // The last mean is kept because the main loop has to reproduce the weight
    // this frame was admitted with, or it cannot take its own contribution back
    // out of the sum exactly.
    let (first_sums, first_counts, _) = prior;
    Consensus { sums, counts, seen, first_sums, first_counts, frame_variance: None }
}

// The opt-in mono path keeps two extra f32 arrays per reference guide pixel
// (final and penultimate weighted variance sums). Sweeps run sequentially, so
// they do not multiply guide-sized accumulator allocations by worker count.
struct ConsensusVariance {
    sums: Vec<f32>,
    first_sums: Vec<f32>,
}

fn matched_guide_variance(noise: &NoiseModel, photo: &PhotometricMatch,
    matched: f32, u: f32, v: f32) -> sr_core::Result<f32> {
    let gain = photo.gain_at(0,u,v);
    if !gain.is_finite() || gain<=0. {
        return Err(sr_core::SrError::Input("invalid local matched guide gain".into()));
    }
    let offset = photo.apply_at(0, 0., u, v);
    let raw = ((matched - offset) / gain).max(0.);
    let variance = noise.variance(raw) * gain * gain / BLUR_EFFECTIVE_SAMPLES;
    if !matched.is_finite() || !offset.is_finite() || !variance.is_finite() || variance < 0. {
        return Err(sr_core::SrError::Input("nonfinite matched guide noise variance".into()));
    }
    Ok(variance)
}

fn outlier_weight_with_variance(mine: f32, mean: f32, variance: f32) -> f32 {
    let tolerance = (CONSENSUS_SIGMAS * variance.max(1e-20).sqrt())
        .max(CONSENSUS_FRACTION * mean.max(0.)).max(1e-9);
    let t = (mine - mean) / tolerance;
    1. / (1. + t * t)
}

#[allow(clippy::too_many_arguments)]
fn consensus_guide_frame_noise(frames: &[RawFrame], warps: &[WarpField], active: &[bool],
    photometry: &[PhotometricMatch], gw: usize, gh: usize) -> sr_core::Result<Consensus> {
    struct Sweep {
        sums: Plane<f32>, counts: Vec<f32>, seen: Vec<u16>, variances: Vec<f32>,
    }
    let n = gw * gh;
    let sweep = |prior: Option<&Sweep>| -> sr_core::Result<Sweep> {
        let mut result = Sweep { sums: Plane::new(gw, gh), counts: vec![0.; n],
            seen: vec![0; n], variances: vec![0.; n] };
        for (i, frame) in frames.iter().enumerate() {
            if !active[i] { continue; }
            let guide = frame.structure_guide_rgb();
            let stats = local_stats(guide.channel(0)).0;
            let photo = &photometry[i];
            for gy in 0..gh { for gx in 0..gw {
                let Some((sx, sy)) = warps[i].inverse_map(gx as f32 * 2. + 0.5, gy as f32 * 2. + 0.5) else { continue; };
                let (tx, ty) = ((sx - 0.5) * 0.5, (sy - 0.5) * 0.5);
                if tx < 0. || ty < 0. || tx > (guide.width - 1) as f32 || ty > (guide.height - 1) as f32 { continue; }
                let idx = gy * gw + gx;
                let (u, v) = (2. * gx as f32 / gw as f32 - 1., 2. * gy as f32 / gh as f32 - 1.);
                if photo.blocked_at(u, v) { continue; }
                result.seen[idx] += 1;
                let mine = photo.apply_at(0, stats.bilinear(tx, ty), u, v);
                let stored_variance = matched_guide_variance(&frame.noise, photo, mine, u, v)?;
                let mut weight = 1.;
                if let Some(first) = prior {
                    if first.counts[idx] > 0. {
                        let mean = first.sums.data[idx] / first.counts[idx];
                        let candidate = matched_guide_variance(&frame.noise, photo, mean, u, v)?;
                        let mean_variance = first.variances[idx] / first.counts[idx].powi(2);
                        weight = outlier_weight_with_variance(mine, mean, candidate + mean_variance);
                    }
                }
                result.sums.data[idx] += weight * mine;
                result.counts[idx] += weight;
                result.variances[idx] += weight * weight * stored_variance;
            }}
        }
        Ok(result)
    };
    let mut prior = sweep(None)?;
    for _ in 0..CONSENSUS_SWEEPS - 2 { prior = sweep(Some(&prior))?; }
    let result = sweep(Some(&prior))?;
    Ok(Consensus { sums: vec![result.sums], counts: result.counts, seen: result.seen,
        first_sums: vec![prior.sums], first_counts: prior.counts,
        frame_variance: Some(ConsensusVariance { sums: result.variances, first_sums: prior.variances }) })
}

/// What the burst agrees the scene is, and enough of the working to remove any
/// one frame's contribution to it exactly.
pub struct Consensus {
    sums: Vec<Plane<f32>>,
    counts: Vec<f32>,
    /// How many frames looked at each site, before any weighting.
    seen: Vec<u16>,
    first_sums: Vec<Plane<f32>>,
    first_counts: Vec<f32>,
    frame_variance: Option<ConsensusVariance>,
}

impl Consensus {
    /// Mono leave-one-out mean and variance, using exactly the admission weight
    /// from the final sweep. Pixel noise is independent between source frames;
    /// the established guide blur effective-sample approximation is retained.
    fn without_frame_noise(&self, idx: usize, mine: f32, noise: &NoiseModel,
        photo: &PhotometricMatch, u: f32, v: f32) -> sr_core::Result<(f32, f32)> {
        let variance = self.frame_variance.as_ref().expect("per-frame consensus variance");
        let mut weight = 1.;
        if self.first_counts[idx] > 0. {
            let mean = self.first_sums[0].data[idx] / self.first_counts[idx];
            let candidate = matched_guide_variance(noise, photo, mean, u, v)?;
            let mean_variance = variance.first_sums[idx] / self.first_counts[idx].powi(2);
            weight = outlier_weight_with_variance(mine, mean, candidate + mean_variance);
        }
        let count = self.counts[idx] - weight;
        if count <= 1e-3 { return Ok((mine, 0.)); }
        let mean = ((self.sums[0].data[idx] - weight * mine) / count).max(0.);
        let own_variance = matched_guide_variance(noise, photo, mine, u, v)?;
        let other_variance = (variance.sums[idx] - weight * weight * own_variance).max(0.) / count.powi(2);
        Ok((mean, other_variance))
    }
    /// The burst's opinion of a site in one channel, with this frame's own
    /// contribution taken back out.
    ///
    /// `mine` contains every active channel on the reference's photometric
    /// scale. Admission uses the worst channel's weight for the entire site;
    /// removing a different per-channel weight would leave a colour bias.
    /// Recompute that same shared weight against the saved penultimate sweep.
    #[inline]
    fn without(&self, c: usize, idx: usize, rtx: f32, rty: f32, mine: &[f32], noise: &NoiseModel)
        -> f32 {
        let w = if self.first_counts[idx] > 0.0 {
            mine.iter().enumerate().fold(1.0f32, |w, (k, &value)| {
                w.min(outlier_weight(value,
                    self.first_sums[k].data[idx] / self.first_counts[idx], noise))
            })
        } else {
            1.0
        };
        let sum = self.sums[c].bilinear(rtx, rty) - w * mine[c];
        let count = self.counts[idx] - w;
        if count <= 1e-3 {
            return mine[c];
        }
        (sum / count).max(0.0)
    }

    /// Whether a site has anybody but this frame to be judged against.
    ///
    /// Weights, so the bar is below two: two frames that agree perfectly well
    /// still each contribute a little under one when the burst is small enough
    /// that they pull each other's mean about.
    #[inline]
    fn has_others(&self, idx: usize) -> bool {
        self.counts[idx] >= 1.5
    }

    /// How many frames looked at a site, before any weighting.
    #[inline]
    fn seen(&self, idx: usize) -> u16 {
        self.seen[idx]
    }

    /// Mean luma across channels, for the gradient the tolerance uses.
    fn luma(&self, gw: usize, gh: usize, channels: usize) -> Plane<f32> {
        let mut l = Plane::<f32>::new(gw, gh);
        for i in 0..gw * gh {
            let inv = 1.0 / (channels as f32 * self.counts[i].max(1e-6));
            for c in 0..channels {
                l.data[i] += self.sums[c].data[i] * inv;
            }
        }
        l
    }
}

/// How much a frame's opinion of a site counts, given the burst's first guess.
///
/// Deliberately forgiving. What this has to remove is a satellite or an
/// aircraft, which stands out by tens of percent of the sky level; what it must
/// not remove is the ordinary spread between frames -- a gradient the
/// photometric field followed only approximately, a little misregistration --
/// because a consensus assembled from a subset of the burst is exactly the
/// anchored reference this whole change exists to get away from.
fn outlier_weight(mine: f32, mean: f32, noise: &NoiseModel) -> f32 {
    let level = mean.max(0.0);
    let grain = (noise.variance(level) / BLUR_EFFECTIVE_SAMPLES).max(1e-20).sqrt();
    let tolerance = (CONSENSUS_SIGMAS * grain).max(CONSENSUS_FRACTION * level).max(1e-9);
    let t = (mine - mean) / tolerance;
    1.0 / (1.0 + t * t)
}

/// Passes over the burst. The first is a plain mean, and each after it
/// reweights against what the one before decided.
const CONSENSUS_SWEEPS: usize = 3;

/// How many times the grain on a blurred guide value a frame may stand from the
/// burst before its opinion starts to count for less.
const CONSENSUS_SIGMAS: f32 = 6.0;

/// And never less than this fraction of the sky level, so that a burst whose
/// frames genuinely differ by a few percent still all belong to it.
const CONSENSUS_FRACTION: f32 = 0.10;



/// Turn a graded trust into a decision, with a narrow band of doubt.
///
/// The model produces a number between zero and one, and using it directly as a
/// weight means the merge is modulated *everywhere*: a site the model half
/// trusts contributes half as much, so the effective exposure varies from pixel
/// to pixel by however much the model's opinion varies. The model's opinion
/// varies most around stars, because that is where the local gradient is
/// steepest and a fraction of a pixel of misregistration produces the largest
/// disagreement. The result is a dip in exposure around every star, repeated in
/// the same place in every frame, which comes out of the stack as a ring.
///
/// It is not a small effect. On a 96-frame burst, graded weighting cost more
/// photometric signal-to-noise than switching the robustness stage off
/// entirely. The rejection was worth having; the grading was costing more than
/// the rejection was saving.
///
/// So the weight is one or zero, with a narrow ramp between `reject_below` and
/// twice it so that a marginal site fades rather than flicking. Almost every
/// site is now exactly one, and the merge is no longer modulated by how the
/// model feels about a place it has no complaint about.
///
/// The seam this grading existed to prevent does not appear, for two reasons
/// that were not true of the handheld bursts the model came from: a hundred
/// frames average a hard per-frame decision into a smooth ensemble, and the map
/// is read with interpolation, so the edge of a rejected region is a slope in
/// space rather than a step.
#[inline]
fn decide(r: f32, reject_below: f32) -> f32 {
    let lo = reject_below;
    let hi = (2.0 * reject_below).min(1.0);
    if r <= lo {
        return 0.0;
    }
    if r >= hi {
        return 1.0;
    }
    let t = (r - lo) / (hi - lo).max(1e-6);
    // Smoothstep, so the ramp meets both ends with zero slope and the fade has
    // no corner in it.
    t * t * (3.0 - 2.0 * t)
}

/// Turns a median absolute deviation into the scale of a normal distribution,
/// so that "twice the typical disagreement" means about three sigma.
const MAD_TO_SIGMA: f32 = 1.4826;

/// How far a frame's typical disagreement may widen its tolerance, in units of
/// the disagreement noise and misregistration would produce on their own.
///
/// The systematics the leniency exists for -- a gradient the photometric field
/// could not quite follow, a lens the affine could not quite place -- are a
/// few sigma. A frame under cloud disagrees by many, and on one test burst thirty
/// of ninety-six frames carried more than ten percent of gradient; admitted at
/// their own typical, they left the stack's sky with five times the mid-scale
/// structure of a conventional stack.
const TYPICAL_CAP: f32 = 3.0;

/// One site in this many along each axis is sampled to find the typical
/// disagreement. A sixteenth of a 26-megapixel frame is still four hundred
/// thousand measurements, which is far more than a median needs.
const TYPICAL_STRIDE: usize = 4;

/// The median absolute disagreement between a frame and the reference, per
/// channel, over a sample of the frame.
///
/// This is the same quantity the rejection rule tests site by site, summarised
/// over the whole frame so that the rule has something to be relative to. It is
/// a median, so the moving objects the rule exists to catch do not move it: on
/// a frame where a satellite crosses a hundredth of the sites, the other
/// ninety-nine hundredths set the number.
#[allow(clippy::too_many_arguments)]
fn typical_disagreement(
    channels: usize,
    consensus: &Consensus,
    noise: &NoiseModel,
    stats: &[(Plane<f32>, Plane<f32>)],
    ref_at: &[(f32, f32)],
    ref_ok: &[bool],
    warp: &WarpField,
    photo: &PhotometricMatch,
    gw: usize,
    gh: usize,
) -> Vec<f32> {
    let mut per_channel: Vec<Vec<f32>> = vec![Vec::new(); channels];
    for gy in (0..gh).step_by(TYPICAL_STRIDE) {
        for gx in (0..gw).step_by(TYPICAL_STRIDE) {
            let idx = gy * gw + gx;
            if !ref_ok[idx] {
                continue;
            }
            let Some((sx, sy)) = warp.inverse_map(gx as f32 * 2.0 + 0.5, gy as f32 * 2.0 + 0.5) else {
                continue;
            };
            let (tx, ty) = ((sx - 0.5) * 0.5, (sy - 0.5) * 0.5);
            if tx < 0.0 || ty < 0.0 || tx > (stats[0].0.width - 1) as f32 || ty > (stats[0].0.height - 1) as f32 {
                continue;
            }
            let (rtx, rty) = ref_at[idx];
            let mut mine = [0.0f32; 3];
            for c in 0..channels {
                mine[c] = photo.apply_at(
                    c,
                    stats[c].0.bilinear(tx, ty),
                    2.0 * gx as f32 / gw as f32 - 1.0,
                    2.0 * gy as f32 / gh as f32 - 1.0,
                );
            }
            for c in 0..channels {
                let mu_ref = consensus.without(c, idx, rtx, rty, &mine[..channels], noise);
                per_channel[c].push((mine[c] - mu_ref).abs());
            }
        }
    }
    per_channel
        .iter()
        .map(|v| if v.is_empty() { 0.0 } else { sr_core::math::median(v) })
        .collect()
}

/// Local mean and variance of a guide channel, over a small window.
fn local_stats(p: &Plane<f32>) -> (Plane<f32>, Plane<f32>) {
    let mean = p.blur_n(2);
    let mut sq = Plane::<f32>::new(p.width, p.height);
    for i in 0..p.data.len() {
        sq.data[i] = p.data[i] * p.data[i];
    }
    let msq = sq.blur_n(2);
    let mut var = Plane::<f32>::new(p.width, p.height);
    for i in 0..var.data.len() {
        var.data[i] = (msq.data[i] - mean.data[i] * mean.data[i]).max(0.0);
    }
    (mean, var)
}

/// Effective sample count behind one blurred value, used to convert per-sample
/// noise into noise on a local mean. Two passes of a `(1 2 1)/4` binomial blur
/// in each axis give a footprint of roughly this many independent samples.
const BLUR_EFFECTIVE_SAMPLES: f32 = 9.0;

/// Build robustness maps for every frame against the reference.
///
/// `reference` is the frame others are *compared* against, which is not
/// necessarily the frame that defines the output grid. When several filters are
/// reconstructed onto one grid, the grid comes from whichever frame was chosen
/// globally while the comparison has to stay inside a filter — an H-alpha frame
/// and an OIII one disagree about the whole nebula, and a model that reads that
/// as motion rejects everything.
///
/// So the reference is sampled through its own warp like any other frame. With
/// the usual single-filter burst its warp is the identity and this costs
/// nothing.
///
/// `active` selects the frames to build maps for. Frames outside it get an
/// empty map and are not examined.
/// `registration_sigma` is in sensor pixels, not half-resolution guide pixels.
// Eight arguments: the burst, what to do with it, and how. Grouping them would
// mean a struct that exists only to be unpacked here.
#[allow(clippy::too_many_arguments)]
pub fn build_maps(
    frames: &[RawFrame],
    warps: &[WarpField],
    reference: usize,
    active: &[bool],
    photometry: &[PhotometricMatch],
    noise: &NoiseModel,
    registration_sigma: f32,
    cfg: &RobustnessConfig,
) -> RobustnessMaps {
    build_maps_spooled(frames, warps, reference, active, photometry, noise, registration_sigma, cfg, None)
        .expect("building owned robustness maps performs no scratch I/O")
}

/// The same global consensus and rejection decisions, with completed maps
/// optionally moved to immutable scratch instead of retained on the heap.
#[allow(clippy::too_many_arguments)]
pub fn build_maps_spooled(
    frames: &[RawFrame],
    warps: &[WarpField],
    reference: usize,
    active: &[bool],
    photometry: &[PhotometricMatch],
    noise: &NoiseModel,
    registration_sigma: f32,
    cfg: &RobustnessConfig,
    spill_dir: Option<&Path>,
) -> sr_core::Result<RobustnessMaps> {
    build_maps_impl(frames, warps, reference, active, photometry, noise, registration_sigma, cfg, spill_dir, None, false)
}

/// Build rejection maps with fixed per-frame typical disagreement on the
/// photometrically matched scale. Bounded callers can reuse one global estimate
/// across all windows instead of changing rejection tolerance at tile borders.
/// Zero selects noise/registration-based tolerance without a measured MAD floor.
#[allow(clippy::too_many_arguments)]
pub fn build_maps_with_typical(
    frames: &[RawFrame],
    warps: &[WarpField],
    reference: usize,
    active: &[bool],
    photometry: &[PhotometricMatch],
    noise: &NoiseModel,
    registration_sigma: f32,
    cfg: &RobustnessConfig,
    typical: &[[f32; 3]],
) -> sr_core::Result<RobustnessMaps> {
    validate_fixed_inputs(frames, warps, reference, active, photometry, typical)?;
    build_maps_impl(frames, warps, reference, active, photometry, noise, registration_sigma, cfg, None, Some(typical), false)
}

fn validate_fixed_inputs(frames: &[RawFrame], warps: &[WarpField], reference: usize,
    active: &[bool], photometry: &[PhotometricMatch], typical: &[[f32; 3]]) -> sr_core::Result<()> {
    if reference >= frames.len() || warps.len() != frames.len()
        || active.len() != frames.len() || photometry.len() != frames.len()
        || typical.len() != frames.len()
        || typical.iter().flatten().any(|v| !v.is_finite() || *v < 0.)
        || frames.iter().any(|f| f.width < 2 || f.height < 2)
    {
        return Err(sr_core::SrError::Input("invalid fixed robustness inputs or typical disagreement".into()));
    }
    Ok(())
}

/// Opt-in mono rejection with native per-frame noise transformed by photometry.
/// Includes the weighted leave-one-out consensus uncertainty; fixed `typical`
/// values keep tolerance independent of tile contents. Existing common-noise
/// entry points do not use this policy. Two additional f32 arrays are retained
/// per reference guide pixel (8 bytes); sequential consensus sweeps bound scratch.
#[allow(clippy::too_many_arguments)]
pub fn build_maps_with_frame_noise(
    frames: &[RawFrame], warps: &[WarpField], reference: usize, active: &[bool],
    photometry: &[PhotometricMatch], noise: &NoiseModel, registration_sigma: f32,
    cfg: &RobustnessConfig, typical: &[[f32; 3]],
) -> sr_core::Result<RobustnessMaps> {
    validate_fixed_inputs(frames, warps, reference, active, photometry, typical)?;
    if frames.len() > u16::MAX as usize || !registration_sigma.is_finite() || registration_sigma < 0.
        || frames.iter().any(|f| !f.is_mono() || !f.noise.alpha.is_finite() || f.noise.alpha < 0.
            || !f.noise.beta.is_finite() || f.noise.beta < 0.
            || !f.noise.variance(0.18).is_finite() || f.noise.variance(0.18) <= 0.)
        || photometry.iter().any(|p| !p.gain[0].is_finite() || p.gain[0] <= 0.
            || !p.offset[0].is_finite() || p.field[0].iter().flatten().any(|v| !v.is_finite()))
    {
        return Err(sr_core::SrError::Input("per-frame robustness requires mono frames and finite positive noise/photometry".into()));
    }
    for (frame,photo) in frames.iter().zip(photometry) {
        if photo.log_gain.is_some() {
            let (minimum,maximum)=photo.gain_bounds(0,[-1.,1.],[-1.,1.])
                .ok_or_else(||sr_core::SrError::Input("invalid spatial guide gain".into()))?;
            for raw in [0.,1.] {for gain in [minimum,maximum] {
                let variance=frame.noise.variance(raw)*gain*gain;
                if !variance.is_finite() || variance<=0. {
                    return Err(sr_core::SrError::Input("spatial gain overflows or collapses guide noise".into()));
                }
            }}
        }
    }
    build_maps_impl(frames, warps, reference, active, photometry, noise, registration_sigma, cfg,
        None, Some(typical), true)
}

#[allow(clippy::too_many_arguments)]
fn build_maps_impl(
    frames: &[RawFrame],
    warps: &[WarpField],
    reference: usize,
    active: &[bool],
    photometry: &[PhotometricMatch],
    noise: &NoiseModel,
    registration_sigma: f32,
    cfg: &RobustnessConfig,
    spill_dir: Option<&Path>,
    fixed_typical: Option<&[[f32; 3]]>,
    frame_noise: bool,
) -> sr_core::Result<RobustnessMaps> {
    let ref_guide = frames[reference].structure_guide_rgb();
    let (gw, gh) = (ref_guide.width, ref_guide.height);

    if !cfg.enabled {
        return Ok(RobustnessMaps {
            width: gw,
            height: gh,
            maps: Vec::new(),
            rejected_fraction: vec![0.0; frames.len()],
            consensus_luma: None,
        });
    }

    // A monochrome frame's three guide planes hold the same values, so the
    // statistics behind them are the same statistics computed three times. On
    // a narrowband burst that is two thirds of this stage.
    let channels = frames[reference].channels();

    // What every frame is judged against: the burst's own mean, not one of its
    // members. See `consensus_guide` for why that matters more than it sounds.
    let consensus = if frame_noise {
        consensus_guide_frame_noise(frames, warps, active, photometry, gw, gh)?
    } else {
        consensus_guide(frames, warps, active, photometry, noise, channels, gw, gh)
    };
    // Whether a site has anybody but this frame to be judged against. Where it
    // has not, the frame is admitted rather than rejected: see below.
    let ref_ok: Vec<bool> = (0..gw * gh).map(|i| consensus.has_others(i)).collect();

    // Local gradient magnitude of the consensus luma, in guide pixels. Combined
    // with the registration residual this bounds how much disagreement
    // misalignment alone can produce.
    let ref_luma = consensus.luma(gw, gh, channels);
    let mut grad = Plane::<f32>::new(gw, gh);
    for y in 0..gh {
        for x in 0..gw {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(gw - 1);
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(gh - 1);
            let gx = 0.5 * (ref_luma.data[y * gw + xp] - ref_luma.data[y * gw + xm]);
            let gy = 0.5 * (ref_luma.data[yp * gw + x] - ref_luma.data[ym * gw + x]);
            grad.data[y * gw + x] = (gx * gx + gy * gy).sqrt();
        }
    }

    // The consensus is built in the reference's own guide coordinates, so a
    // site is looked up where it is. The indirection the reference frame used
    // to need is gone with it.
    let ref_at: Vec<(f32, f32)> = (0..gw * gh)
        .map(|i| ((i % gw) as f32, (i / gw) as f32))
        .collect();

    let results: Vec<(Plane<u8>, f32)> = (0..frames.len())
        .into_par_iter()
        .map(|i| {
            let mut map = Plane::<u8>::new(gw, gh);
            if !active.get(i).copied().unwrap_or(true) {
                return Ok((Plane::<u8>::new(0, 0), 0.0));
            }

            let guide = frames[i].structure_guide_rgb();
            let stats: Vec<(Plane<f32>, Plane<f32>)> =
                (0..channels).map(|c| local_stats(guide.channel(c))).collect();
            // Frames are compared on the reference's photometric scale, not
            // their own. Without this a burst whose illumination drifted is
            // distrusted uniformly, everywhere, for having been shot later.
            let photo = photometry.get(i).copied().unwrap_or(PhotometricMatch::IDENTITY);

            // Guide coordinates are half the sensor pitch, so the sensor-space
            // warp is applied at twice the guide coordinate and halved back.
            let warp = &warps[i];
            let mut rejected = 0u64;

            // What this frame typically disagrees with the reference by,
            // measured on a sample of the frame rather than predicted from the
            // sensor.
            //
            // The tolerance below is built from a noise model, and a noise
            // model only describes the part of the disagreement that is noise.
            // The rest -- a sky gradient the photometric match could not quite
            // follow, a lens the affine could not quite place -- is systematic,
            // is present over the whole frame, and is not the frame being
            // wrong about the scene. Judged against noise alone it looks like
            // motion everywhere, and on a 96-frame burst that was a fifth of
            // every sample thrown away.
            //
            // So the tolerance is at least this frame's own typical
            // disagreement, which makes the rule what a robust rule should be:
            // reject what stands out from what this frame usually does, not
            // what stands out from what a perfect frame would have done.
            let typical = fixed_typical.map(|values| values[i][..channels].to_vec())
                .unwrap_or_else(|| typical_disagreement(
                    channels, &consensus, noise, &stats, &ref_at, &ref_ok, warp, &photo, gw, gh,
                ));

            for gy in 0..gh {
                for gx in 0..gw {
                    let idx = gy * gw + gx;
                    let (rx, ry) = (gx as f32 * 2.0 + 0.5, gy as f32 * 2.0 + 0.5);
                    let Some((sx, sy)) = warp.inverse_map(rx, ry) else {
                        map.data[idx] = 0;
                        rejected += 1;
                        continue;
                    };
                    let (tx, ty) = ((sx - 0.5) * 0.5, (sy - 0.5) * 0.5);
                    if tx < 0.0 || ty < 0.0 || tx > (guide.width - 1) as f32 || ty > (guide.height - 1) as f32 {
                        map.data[idx] = 0;
                        rejected += 1;
                        continue;
                    }
                    let (rtx, rty) = ref_at[idx];
                    // Obstructed here: the merge will not use these samples,
                    // and the map says so.
                    if photo.blocked_at(2.0 * gx as f32 / gw as f32 - 1.0, 2.0 * gy as f32 / gh as f32 - 1.0) {
                        map.data[idx] = 0;
                        rejected += 1;
                        continue;
                    }

                    // Through the same map the merge will use, field and all:
                    // comparing a frame against the burst with a gradient still
                    // between them rejects the gradient as if it were the scene
                    // changing.
                    let mut mu_t = [0.0f32; 3];
                    for (c, m) in mu_t.iter_mut().enumerate().take(channels) {
                        *m = photo.apply_at(
                            c,
                            stats[c].0.bilinear(tx, ty),
                            2.0 * gx as f32 / gw as f32 - 1.0,
                            2.0 * gy as f32 / gh as f32 - 1.0,
                        );
                    }

                    // Is there anybody but this frame to be judged against?
                    //
                    // Two different reasons the answer can be no, and they want
                    // opposite outcomes. If nobody else *looked* at this site,
                    // there is nothing to disagree with, and suppressing the
                    // sample throws away the only measurement of that piece of
                    // sky -- on a burst of one frame that was every sample, and
                    // the stack came out black. If others looked and the burst
                    // still cannot agree, the site is genuinely unresolved and
                    // the conservative answer is to take nothing from it.
                    if consensus.seen(idx) < 2 {
                        map.data[idx] = 255;
                        continue;
                    }
                    if !ref_ok[idx] {
                        map.data[idx] = 0;
                        rejected += 1;
                        continue;
                    }

                    // Worst disagreement across the channels: a colour change
                    // in one channel is still a scene change.
                    let mut worst = 0.0f32;
                    for c in 0..channels {
                        let (mu_ref, comparison_variance) = if frame_noise {
                            let (u, v) = (2. * gx as f32 / gw as f32 - 1., 2. * gy as f32 / gh as f32 - 1.);
                            let (mean, mean_variance) = consensus.without_frame_noise(idx, mu_t[0], &frames[i].noise, &photo, u, v)?;
                            let candidate = matched_guide_variance(&frames[i].noise, &photo, mean, u, v)?;
                            (mean, Some(candidate + mean_variance))
                        } else {
                            (consensus.without(c, idx, rtx, rty, &mu_t[..channels], noise), None)
                        };
                        let mu_t = mu_t[c];
                        let d = (mu_t - mu_ref).abs();

                        // Tolerance: noise on a local mean, plus what residual
                        // misalignment can produce given the local gradient,
                        // plus a floor.
                        let level = mu_ref.max(0.0);
                        // One frame's worth of noise, not two: what it is being
                        // compared against is a mean of the whole burst and has
                        // almost none of its own.
                        let noise_mean_var = comparison_variance.unwrap_or_else(|| noise.variance(level) / BLUR_EFFECTIVE_SAMPLES);
                        // grad is per guide pixel (two sensor pixels), while
                        // registration_sigma is expressed in sensor pixels.
                        let mis = grad.bilinear(rtx, rty) * (0.5 * registration_sigma);
                        let expected2 =
                            noise_mean_var + mis * mis + cfg.sigma_floor * cfg.sigma_floor;
                        // The frame's own typical disagreement widens the
                        // tolerance, but only so far. Unbounded, a frame that
                        // disagrees with the burst everywhere -- cloud across
                        // half of it, dawn -- sets its own tolerance from that
                        // disagreement and is admitted in full, and the cloud
                        // is averaged into the sky. The cap keeps the leniency
                        // for the few-sigma systematics it exists for and
                        // withdraws it from a frame that is wrong by more.
                        let sigma2 = expected2.max({
                            let s = (MAD_TO_SIGMA * typical[c]).min(TYPICAL_CAP * expected2.sqrt());
                            s * s
                        });

                        // Flat regions are dominated by noise, so a difference
                        // there means little and should be forgiven; textured
                        // regions get the strict threshold.
                        let s = cfg.s2;

                        let r = (s * (-(d * d) / sigma2).exp() - cfg.t).clamp(0.0, 1.0);
                        worst = worst.max(1.0 - r);
                    }
                    let r = 1.0 - worst;
                    if r < cfg.reject_below {
                        rejected += 1;
                    }
                    map.data[idx] = (decide(r, cfg.reject_below) * 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
            let frac = rejected as f32 / (gw * gh) as f32;
            if let Some(dir) = spill_dir {
                map.spill(dir)?;
            }
            Ok((map, frac))
        })
        .collect::<sr_core::Result<Vec<_>>>()?;

    let mut maps = Vec::with_capacity(results.len());
    let mut rejected_fraction = Vec::with_capacity(results.len());
    for (m, f) in results {
        maps.push(m);
        rejected_fraction.push(f);
    }

    // The mono per-frame policy compares peer consensuses, often on a synthetic
    // inactive reference grid. A zero may mean no native coverage, not a shared
    // disagreement with one reference exposure. The legacy majority restoration
    // would invent support outside footprints and must not run in this mode.
    let restored = if frame_noise { 0 } else { restore_where_the_reference_is_the_outlier(
        &mut maps,
        active,
        reference,
        gw * gh,
        cfg,
        spill_dir,
    )? };
    if restored > 0 {
        log::info!(
            "robustness: {restored} sites where most frames disagreed with the reference \
             rather than with each other; the reference was taken to be the odd one out"
        );
        // The per-frame fractions were counted before this, so they are recomputed
        // rather than left describing rejections that have since been undone.
        for (i, m) in maps.iter().enumerate() {
            if m.data.is_empty() {
                continue;
            }
            let cut = (cfg.reject_below * 255.0) as u8;
            let n = m.data.iter().filter(|v| **v < cut).count();
            rejected_fraction[i] = n as f32 / (gw * gh) as f32;
        }
    }

    Ok(RobustnessMaps { width: gw, height: gh, maps, rejected_fraction, consensus_luma: Some(ref_luma) })
}

/// Frames needed before "most of them" is a statement worth acting on.
///
/// With four frames a majority is three, and three frames agreeing by chance
/// is not rare enough to overrule the reference on.
const CONSENSUS_MIN_FRAMES: usize = 5;

/// Undo rejections that say more about the reference than about the frame.
///
/// Every frame is judged against the reference, which works until the
/// reference is the frame that is wrong. A satellite, a cosmic ray or an
/// aircraft in it makes every other frame disagree exactly where the defect
/// is; each is then rejected there for telling the truth, and the only frame
/// still contributing is the one carrying the artefact.
///
/// Where most of the burst was rejected at one site, they are not all wrong in
/// the same way at the same place -- they are agreeing with each other and
/// disagreeing with the reference. Their weight is given back. What survives
/// rejection is then the majority view, which is what rejecting against a
/// median would have given in the first place, without the pass over every
/// frame that a median needs.
fn restore_where_the_reference_is_the_outlier(
    maps: &mut [Plane<u8>],
    active: &[bool],
    reference: usize,
    sites: usize,
    cfg: &RobustnessConfig,
    spill_dir: Option<&Path>,
) -> sr_core::Result<usize> {
    let judged: Vec<usize> = (0..maps.len())
        .filter(|&i| {
            i != reference && active.get(i).copied().unwrap_or(false) && !maps[i].data.is_empty()
        })
        .collect();
    if judged.len() < CONSENSUS_MIN_FRAMES {
        return Ok(0);
    }
    let cut = (cfg.reject_below * 255.0) as u8;
    let majority = judged.len() / 2 + 1;

    let mut rejected_here = vec![0u16; sites];
    for &i in &judged {
        for (site, v) in maps[i].data.iter().enumerate() {
            if *v < cut {
                rejected_here[site] += 1;
            }
        }
    }

    let restored = rejected_here.iter().filter(|&&count| count as usize >= majority).count();
    if restored == 0 {
        return Ok(0);
    }
    // Decide the same majority sites before touching any map. Then change one
    // frame at a time: a mapped buffer copies on mutation, so a pixel-major
    // loop would inadvertently materialize every frame's map on the heap.
    for &i in &judged {
        for (site, &count) in rejected_here.iter().enumerate() {
            if count as usize >= majority && maps[i].data[site] < cut {
                maps[i].data[site] = 255;
            }
        }
        if let Some(dir) = spill_dir {
            maps[i].spill(dir)?;
        }
    }
    Ok(restored)
}

/// Guide image helper re-exported for callers that already hold one.
pub fn guide_of(frame: &RawFrame) -> GuideImage {
    frame.guide_rgb()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::cfa::CfaPattern;
    use sr_core::geometry::GlobalTransform;
    use sr_core::samples::{DefectMask, SamplePlane};

    /// A mosaiced frame whose scene is a smooth gradient, optionally with a
    /// bright square painted in to stand in for a moving object.
    fn frame_with_blob(w: usize, h: usize, blob: Option<(usize, usize, usize)>) -> RawFrame {
        let mut data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = 0.25 + 0.2 * (x as f32 / w as f32);
            }
        }
        if let Some((bx, by, bs)) = blob {
            for y in by..(by + bs).min(h) {
                for x in bx..(bx + bs).min(w) {
                    data[y * w + x] = 0.9;
                }
            }
        }
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_normalised(w, h, data),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            noise: NoiseModel::nominal(100.0, 16000.0),
            metadata: Default::default(),
        }
    }

    fn noise_model() -> NoiseModel {
        NoiseModel::new(2.0e-5, 4.0e-6, sr_core::frame::NoiseSource::Measured)
    }

    fn unequal_noise_fixture(count: usize, hot_patch: bool, normalized: bool) -> (Vec<RawFrame>, Vec<PhotometricMatch>) {
        let mut frames = Vec::new();
        let mut photos = Vec::new();
        for i in 0..count {
            let high = i == count - 1;
            let gain = if high { 3f32 } else { 1. };
            let offset = if high { 0.02 } else { 0. };
            let sigma = if high { 0.006f32 } else { 0.001 };
            let mut state = 0x32ad_d658_4983_abd9u64 ^ ((i as u64 + 1) * 997);
            let mut uniform = || {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let mut values = Vec::new();
            for y in 0..128 { for x in 0..128 {
                let normal = (-2. * uniform().ln()).sqrt() * (2. * std::f64::consts::PI * uniform()).cos();
                let hot = if high && hot_patch && (52..76).contains(&x) && (52..76).contains(&y) { 0.3 } else { 0. };
                let raw = (0.12 + hot - offset) / gain + sigma * normal as f32;
                values.push(if normalized { raw * gain + offset } else { raw });
            }}
            let mut frame = frame_with_blob(128, 128, None);
            frame.cfa = CfaPattern::MONO;
            frame.samples = SamplePlane::from_normalised(128, 128, values);
            frame.noise = NoiseModel::new(0., sigma * sigma * if normalized { gain * gain } else { 1. }, sr_core::frame::NoiseSource::Measured);
            frames.push(frame);
            photos.push(PhotometricMatch { gain: [if normalized { 1. } else { gain }; 3],
                offset: [if normalized { 0. } else { offset }; 3], ..PhotometricMatch::IDENTITY });
        }
        (frames, photos)
    }

    fn quiet_retention(maps: &RobustnessMaps, frame: usize) -> f32 {
        let mut retained = 0.;
        let mut count = 0;
        for y in 6..58 { for x in 6..58 {
            if (x as f32 - 32.).hypot(y as f32 - 32.) < 16. { continue; }
            retained += maps.maps[frame].data[y * 64 + x] as f32 / 255.;
            count += 1;
        }}
        retained / count as f32
    }

    #[test]
    fn per_frame_noise_preserves_noisy_sky_and_rejects_hot_patch() {
        let (frames, photo) = unequal_noise_fixture(6, true, false);
        let warps = vec![WarpField::identity(); 6];
        let common = NoiseModel::new(0., 1e-6, sr_core::frame::NoiseSource::Measured);
        let cfg = RobustnessConfig::default();
        let legacy = build_maps_with_typical(&frames, &warps, 0, &[true; 6], &photo, &common, 0., &cfg, &[[0.; 3]; 6]).unwrap();
        let maps = build_maps_with_frame_noise(&frames, &warps, 0, &[true; 6], &photo, &common, 0., &cfg, &[[0.; 3]; 6]).unwrap();
        let retained = quiet_retention(&maps, 5);
        let before = quiet_retention(&legacy, 5);
        println!("heterogeneous noise: quiet retention {before} -> {retained}; hot-patch weight {}", maps.at(5, 64.5, 64.5));
        assert!(retained > 0.98, "ordinary noisy sky retained {retained}");
        assert!(retained > before + 0.05, "fixture must expose common-noise bias: {before} -> {retained}");
        assert!(maps.at(5, 64.5, 64.5) < 0.05, "hot patch admitted");
        for i in 0..5 { assert!(quiet_retention(&maps, i) > 0.98); }
    }

    #[test]
    fn per_frame_noise_includes_uncertain_other_frame_in_two_frame_overlap() {
        let (frames, photo) = unequal_noise_fixture(2, false, false);
        let common = NoiseModel::new(0., 1e-6, sr_core::frame::NoiseSource::Measured);
        let cfg = RobustnessConfig::default();
        let warps = vec![WarpField::identity(); 2];
        let maps = build_maps_with_frame_noise(&frames, &warps, 0, &[true; 2],
            &photo, &common, 0., &cfg, &[[0.; 3]; 2]).unwrap();
        for i in 0..2 { assert!(quiet_retention(&maps, i) > 0.98, "frame {i} retained {}", quiet_retention(&maps, i)); }
        // In particular the clean candidate cannot be judged only by its own
        // variance when its sole comparison is a much noisier exposure.
        let consensus = consensus_guide_frame_noise(&frames, &warps, &[true; 2], &photo, 64, 64).unwrap();
        let guide = frames[0].structure_guide_rgb();
        let mine = local_stats(guide.channel(0)).0.data[10 * 64 + 10];
        let (_, variance) = consensus.without_frame_noise(10 * 64 + 10, mine, &frames[0].noise, &photo[0], -0.6875, -0.6875).unwrap();
        assert!((variance - 0.006f32.powi(2) * 9. / BLUR_EFFECTIVE_SAMPLES).abs() < 1e-9);
    }

    #[test]
    fn per_frame_noise_is_equivalent_to_explicit_gain_normalization() {
        let (raw, photo) = unequal_noise_fixture(6, true, false);
        let (normalized, identity) = unequal_noise_fixture(6, true, true);
        let common = noise_model();
        let cfg = RobustnessConfig::default();
        let warps = vec![WarpField::identity(); 6];
        let evaluate = |frames: &[RawFrame], p: &[PhotometricMatch]| build_maps_with_frame_noise(frames,
            &warps, 0, &[true; 6], p, &common, 0., &cfg, &[[0.; 3]; 6]).unwrap();
        let a = evaluate(&raw, &photo);
        let b = evaluate(&normalized, &identity);
        let worst = a.maps.iter().zip(&b.maps).flat_map(|(x, y)| x.data.iter().zip(y.data.iter()))
            .map(|(&x, &y)| x.abs_diff(y)).max().unwrap();
        assert!(worst <= 1, "gain normalization changes rejection by {worst} byte levels");
    }

    #[test]
    fn common_noise_apis_remain_independent_of_native_noise_metadata() {
        let (mut frames, photo) = unequal_noise_fixture(6, true, false);
        let warps = vec![WarpField::identity(); 6];
        let common = noise_model();
        let cfg = RobustnessConfig::default();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let before = pool.install(|| build_maps(&frames, &warps, 0, &[true; 6], &photo, &common, 0., &cfg));
        let fixed_before = pool.install(|| build_maps_with_typical(&frames, &warps, 0, &[true; 6], &photo, &common, 0., &cfg, &[[0.; 3]; 6])).unwrap();
        for frame in &mut frames { frame.noise.beta *= 100.; }
        let after = pool.install(|| build_maps(&frames, &warps, 0, &[true; 6], &photo, &common, 0., &cfg));
        let fixed_after = pool.install(|| build_maps_with_typical(&frames, &warps, 0, &[true; 6], &photo, &common, 0., &cfg, &[[0.; 3]; 6])).unwrap();
        assert_eq!(before.maps, after.maps);
        assert_eq!(before.consensus_luma, after.consensus_luma);
        assert_eq!(fixed_before.maps, fixed_after.maps);
        assert_eq!(fixed_before.consensus_luma, fixed_after.consensus_luma);
    }

    #[test]
    fn matched_noise_recovers_raw_level_before_evaluating_shot_variance() {
        let noise = NoiseModel::new(0.002, 0.0001, sr_core::frame::NoiseSource::Measured);
        let mut photo = PhotometricMatch { gain: [3.; 3], offset: [0.02; 3], ..PhotometricMatch::IDENTITY };
        for value in photo.field[0].iter_mut().flatten() { *value = 0.04; }
        let variance = matched_guide_variance(&noise, &photo, 0.3, 0.1, -0.2).unwrap();
        let expected = noise.variance((0.3 - 0.02 - 0.04) / 3.) * 9. / BLUR_EFFECTIVE_SAMPLES;
        assert!((variance - expected).abs() < 1e-10);
    }

    #[test]
    fn per_frame_noise_rejects_nonmono_and_invalid_noise() {
        let (mut frames, photo) = unequal_noise_fixture(2, false, false);
        let warps = vec![WarpField::identity(); 2];
        frames[0].cfa = CfaPattern::RGGB;
        assert!(build_maps_with_frame_noise(&frames, &warps, 0, &[true; 2], &photo,
            &noise_model(), 0., &RobustnessConfig::default(), &[[0.; 3]; 2]).is_err());
        frames[0].cfa = CfaPattern::MONO;
        for beta in [-1., f32::NAN, f32::INFINITY] {
            frames[0].noise.beta = beta;
            assert!(build_maps_with_frame_noise(&frames, &warps, 0, &[true; 2], &photo,
                &noise_model(), 0., &RobustnessConfig::default(), &[[0.; 3]; 2]).is_err());
        }
    }

    #[test]
    fn six_partial_frames_never_restore_support_outside_native_footprints() {
        let mut frames = vec![frame_with_blob(128, 128, None)];
        let mut warps = vec![WarpField::identity()];
        for i in 0..6 {
            frames.push(frame_with_blob(32, 32, None));
            let origin = if i < 3 { 16. } else { 72. };
            warps.push(WarpField::global_only(sr_core::GlobalTransform::translation(origin, origin)));
        }
        for frame in &mut frames {
            frame.cfa = CfaPattern::MONO;
            frame.samples = SamplePlane::from_normalised(frame.width, frame.height, vec![0.2; frame.width * frame.height]);
        }
        let maps = build_maps_with_frame_noise(&frames, &warps, 0,
            &[false, true, true, true, true, true, true], &[PhotometricMatch::IDENTITY; 7],
            &noise_model(), 0., &RobustnessConfig::default(), &[[0.; 3]; 7]).unwrap();
        assert!(maps.maps[0].data.is_empty());
        for i in 1..7 {
            for (x, y) in [(8.5, 8.5), (60.5, 60.5), (116.5, 116.5)] {
                assert_eq!(maps.at(i, x, y), 0., "frame {i} invented coverage at {x},{y}");
            }
            let inside = if i <= 3 { 32.5 } else { 88.5 };
            let other = if i <= 3 { 88.5 } else { 32.5 };
            assert_eq!(maps.at(i, inside, inside), 1.);
            assert_eq!(maps.at(i, other, other), 0.);
        }
    }

    #[test]
    fn mixed_native_dimensions_use_source_bounds_for_consensus_and_rejection() {
        for (w, h, dx, dy, inside, outside) in [
            (128, 96, -40., -12., (40.5, 30.5), None),
            (24, 20, 16., 18., (26.5, 26.5), Some((50.5, 50.5))),
        ] {
            let mut frames = vec![frame_with_blob(64, 64, None), frame_with_blob(w, h, None)];
            for frame in &mut frames {
                frame.cfa = CfaPattern::MONO;
                frame.samples = SamplePlane::from_normalised(frame.width, frame.height, vec![0.25; frame.width * frame.height]);
            }
            let warps = [WarpField::identity(), WarpField {
                global: sr_core::GlobalTransform::translation(dx, dy),
                ..WarpField::identity()
            }];
            let maps = build_maps_with_typical(&frames, &warps, 0, &[false, true],
                &[PhotometricMatch::IDENTITY; 2], &noise_model(), 0.1,
                &RobustnessConfig::default(), &[[0.; 3]; 2]).unwrap();
            assert_eq!(maps.at(1, inside.0, inside.1), 1.);
            let luma = maps.consensus_luma.as_ref().unwrap();
            let index = ((inside.1 - 0.5) as usize / 2) * 32 + (inside.0 - 0.5) as usize / 2;
            assert!((luma.data[index] - 0.25).abs() < 1e-6);
            if let Some((x, y)) = outside {
                assert_eq!(maps.at(1, x, y), 0.);
                let index = ((y - 0.5) as usize / 2) * 32 + (x - 0.5) as usize / 2;
                assert_eq!(luma.data[index], 0.);
            }
        }
    }

    #[test]
    fn fixed_typical_rejects_invalid_values_and_lengths() {
        let frames = [frame_with_blob(16, 16, None)];
        for typical in [vec![], vec![[-1.; 3]], vec![[f32::NAN; 3]], vec![[f32::INFINITY; 3]]] {
            assert!(build_maps_with_typical(&frames, &[WarpField::identity()], 0, &[true],
                &[PhotometricMatch::IDENTITY], &noise_model(), 0.1,
                &RobustnessConfig::default(), &typical).is_err());
        }
    }

    fn spill_test_dir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("smokstak-robustness-test-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn spooled_maps_match_owned_maps_with_reference_and_frame_outliers() {
        let dir = spill_test_dir();
        // Hold reduction order fixed: this test isolates storage, not Rayon's
        // floating-point reduction scheduling.
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        for outlier in [0, 3] {
            let frames: Vec<_> = (0..7).map(|i|
                frame_with_blob(64, 64, (i == outlier).then_some((20, 20, 12)))
            ).collect();
            let warps = vec![WarpField::identity(); frames.len()];
            let active = [true, true, true, true, true, true, false];
            let photometry = vec![PhotometricMatch::IDENTITY; frames.len()];
            let noise = noise_model();
            let cfg = RobustnessConfig::default();
            let owned = pool.install(|| build_maps(&frames, &warps, 0, &active, &photometry, &noise, 0.05, &cfg));
            let spooled = pool.install(|| build_maps_spooled(&frames, &warps, 0, &active, &photometry, &noise, 0.05, &cfg, Some(&dir))).unwrap();
            assert_eq!(owned.maps, spooled.maps);
            assert_eq!(owned.rejected_fraction, spooled.rejected_fraction);
            assert_eq!(owned.consensus_luma, spooled.consensus_luma);
            assert!(spooled.maps.iter().filter(|m| !m.data.is_empty()).all(|m| m.data.is_mapped()));
            assert!(spooled.maps[6].data.is_empty());
            assert_eq!(spooled.at(outlier, 26.5, 26.5), owned.at(outlier, 26.5, 26.5));
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn restoration_changes_identical_majority_sites_and_respills_each_frame() {
        let dir = spill_test_dir();
        let cfg = RobustnessConfig::default();
        let active = [true, true, true, true, true, true, true, false];
        let mut owned = vec![Plane::<u8>::filled(5, 1, 255); 8];
        owned[0].data.fill(0); // Reference is never restored.
        owned[7].data.fill(0); // Inactive frame is never counted or restored.
        for map in owned.iter_mut().take(5).skip(1) { map.data[0] = 0; }
        for map in owned.iter_mut().take(4).skip(1) { map.data[1] = 0; }
        for map in owned.iter_mut().take(7).skip(1) { map.data[3] = 0; }
        owned[6].data[0] = 200; // Accepted fractional weight stays unchanged.
        let mut spooled = owned.clone();
        for map in &mut spooled { map.spill(&dir).unwrap(); }
        assert_eq!(restore_where_the_reference_is_the_outlier(&mut owned, &active, 0, 5, &cfg, None).unwrap(), 2);
        assert_eq!(restore_where_the_reference_is_the_outlier(&mut spooled, &active, 0, 5, &cfg, Some(&dir)).unwrap(), 2);
        assert_eq!(owned, spooled);
        assert!(spooled.iter().all(|m| m.data.is_mapped()));
        assert_eq!(spooled[0].data.as_slice(), &[0; 5]);
        assert_eq!(spooled[7].data.as_slice(), &[0; 5]);
        assert_eq!(spooled[1].data.as_slice(), &[255, 0, 255, 255, 255]);
        assert_eq!(spooled[6].data[0], 200);
        drop(spooled);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn spooled_map_io_failure_propagates_and_disabled_maps_need_no_scratch() {
        let dir = spill_test_dir();
        let blocker = dir.join("not-a-directory");
        std::fs::write(&blocker, b"existing file").unwrap();
        let frames = vec![frame_with_blob(16, 16, None), frame_with_blob(16, 16, None)];
        let warps = vec![WarpField::identity(); 2];
        let photo = [PhotometricMatch::IDENTITY; 2];
        let noise = noise_model();
        assert!(matches!(build_maps_spooled(&frames, &warps, 0, &[true; 2], &photo, &noise, 0.05,
            &RobustnessConfig::default(), Some(&blocker)), Err(sr_core::SrError::Io(_))));
        let cfg = RobustnessConfig { enabled: false, ..Default::default() };
        let disabled = build_maps_spooled(&frames, &warps, 0, &[true; 2], &photo, &noise, 0.05, &cfg, Some(&blocker)).unwrap();
        assert!(disabled.maps.is_empty());
        assert_eq!(std::fs::read(&blocker).unwrap(), b"existing file");
        std::fs::remove_file(blocker).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn removing_a_colour_outlier_recovers_the_other_frames_in_every_channel() {
        // Five identical clean exposures and a sixth with unequal RGB deviations.
        // Removing the sixth must recover the clean value regardless of which
        // channel limited that exposure's shared consensus contribution.
        let clean = [0.2f32, 0.3, 0.4];
        let outlier = [0.21f32, 0.32, 0.8];
        let frames: Vec<_> = (0..6).map(|i| {
            let mut frame = frame_with_blob(32, 32, None);
            let rgb = if i == 5 { outlier } else { clean };
            frame.samples = SamplePlane::from_normalised(32, 32,
                (0..32*32).map(|p| rgb[frame.cfa.color_at(p%32,p/32).index()]).collect());
            frame
        }).collect();
        let noise = noise_model();
        let consensus = consensus_guide(&frames, &vec![WarpField::identity();6],
            &[true;6], &[PhotometricMatch::IDENTITY;6], &noise, 3, 16, 16);
        for (c, &expected) in clean.iter().enumerate() {
            let actual = consensus.without(c, 8*16+8, 8., 8., &outlier, &noise);
            assert!((actual-expected).abs()<1e-6,
                "channel {c}: removing the outlier leaves {actual}, expected {expected}");
        }
    }

    #[test]
    fn a_shadow_is_not_excused_by_doubling_registration_uncertainty() {
        // The scene rises 0.002 per sensor pixel. One pixel of registration
        // uncertainty cannot explain a local deficit of 0.006. The guide's
        // gradient is twice the sensor gradient because its pixels are 2x2.
        let mut frames = Vec::new();
        for i in 0..8 {
            let mut frame = frame_with_blob(128, 128, None);
            frame.cfa = CfaPattern::MONO;
            let data = (0..128*128).map(|p| {
                let (x, y) = (p % 128, p / 128);
                let shadow = i == 7 && (32..64).contains(&x) && (32..64).contains(&y);
                0.2 + 0.002 * x as f32 - if shadow { 0.006 } else { 0.0 }
            }).collect();
            frame.samples = SamplePlane::from_normalised(128, 128, data);
            frames.push(frame);
        }
        let cfg = RobustnessConfig { sigma_floor: 0.0, ..Default::default() };
        let maps = build_maps(&frames, &vec![WarpField::identity(); 8], 0,
            &[true; 8], &[PhotometricMatch::IDENTITY; 8],
            &NoiseModel::new(0.0, 0.0, sr_core::frame::NoiseSource::Measured), 1.0, &cfg);
        let shadow = maps.at(7, 48.5, 48.5);
        assert!(shadow < 0.1, "the shadow retained weight {shadow}");
        assert!(maps.at(0, 48.5, 48.5) > 0.9, "clean sky was suppressed");
    }

    #[test]
    fn a_burst_of_one_rejects_nothing() {
        // There is nobody to disagree with, so nothing can be an outlier. This
        // used to fail the "somebody other than this frame saw the site" test
        // everywhere and suppress the entire frame, and a stack of one frame
        // came out black.
        let frames = vec![frame_with_blob(128, 128, None)];
        let warps = vec![WarpField::identity()];
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true],
            &[PhotometricMatch::IDENTITY],
            &noise_model(),
            0.05,
            &RobustnessConfig::default(),
        );
        assert!(
            maps.rejected_fraction[0] < 0.01,
            "suppressed {:.1}% of the only frame there is",
            100.0 * maps.rejected_fraction[0]
        );
        let m = maps.plane(0);
        let mut lo = 1.0f32;
        for v in &m.data {
            lo = lo.min(*v);
        }
        assert!(lo > 0.9, "lowest weight {lo}");
    }

    #[test]
    fn a_flipped_frames_photometry_is_read_on_the_reference_grid() {
        let mut frames = vec![frame_with_blob(128, 128, None); 2];
        frames[0].samples = SamplePlane::from_normalised(128, 128, vec![0.3; 128 * 128]);
        frames[1].samples = SamplePlane::from_normalised(128, 128,
            (0..128 * 128).map(|i| {
                let rx = 127.0 - (i % 128) as f32;
                0.3 + 0.1 * (2.0 * rx / 128.0 - 1.0)
            }).collect());
        let warps = vec![WarpField::identity(), WarpField::global_only(GlobalTransform {
            m: [-1.0, 0.0, 127.0, 0.0, -1.0, 127.0],
        })];
        let mut photometry = [PhotometricMatch::IDENTITY; 2];
        let n = photometry[1].field[0].len();
        for plane in &mut photometry[1].field {
            for row in plane {
                for (x, v) in row.iter_mut().enumerate() {
                    *v = -0.1 * (2.0 * x as f32 / (n - 1) as f32 - 1.0);
                }
            }
        }
        for row in &mut photometry[1].blocked {
            for v in row.iter_mut().take(n / 4) { *v = true; }
        }
        let maps = build_maps(&frames, &warps, 0, &[true; 2], &photometry,
            &noise_model(), 0.05, &RobustnessConfig::default());
        let m = maps.plane(1);
        assert_eq!(m.data[32 * m.width + 4], 0.0, "obstructed sky was admitted");
        assert!(m.data[32 * m.width + 48] > 0.9,
            "clear sky with a corrected gradient was rejected: {}", m.data[32 * m.width + 48]);
    }

    #[test]
    fn guide_cell_centres_survive_a_half_turn() {
        let mut upright = frame_with_blob(128, 128, Some((44, 52, 20)));
        upright.cfa = CfaPattern::MONO;
        let mut flipped = upright.clone();
        flipped.samples = SamplePlane::from_normalised(128, 128,
            (0..128 * 128).rev().map(|i| upright.value(i % 128, i / 128)).collect());
        let frames = vec![upright, flipped];
        let warps = vec![WarpField::identity(), WarpField::global_only(GlobalTransform {
            m: [-1.0, 0.0, 127.0, 0.0, -1.0, 127.0],
        })];
        let photo = [PhotometricMatch::IDENTITY; 2];
        let a = consensus_guide(&frames, &warps, &[true, false], &photo, &noise_model(), 1, 64, 64).luma(64,64,1);
        let b = consensus_guide(&frames, &warps, &[false, true], &photo, &noise_model(), 1, 64, 64).luma(64,64,1);
        let mut worst = 0.0f32;
        for y in 8..56 {
            for x in 8..56 { worst = worst.max((a.data[y*64+x]-b.data[y*64+x]).abs()); }
        }
        assert!(worst < 1e-6, "rotating the sensor shifted its guide content: {worst}");
    }

    #[test]
    fn identical_frames_are_fully_trusted() {
        let frames = vec![frame_with_blob(128, 128, None), frame_with_blob(128, 128, None)];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 2],
            &[PhotometricMatch::IDENTITY; 2],
            &noise_model(),
            0.05,
            &RobustnessConfig::default(),
        );
        let m = maps.plane(1);
        // Guide resolution is half the sensor's; ignore the border, where the
        // blur window runs off the edge.
        let mut lo = 1.0f32;
        for y in 8..m.height - 8 {
            for x in 8..m.width - 8 {
                lo = lo.min(m.data[y * m.width + x]);
            }
        }
        assert!(lo > 0.9, "identical frames were distrusted: min weight {lo}");
    }

    #[test]
    fn a_moving_object_is_rejected_where_it_sits() {
        let reference = frame_with_blob(128, 128, None);
        let moved = frame_with_blob(128, 128, Some((40, 40, 24)));
        let frames = vec![reference, moved];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 2],
            &[PhotometricMatch::IDENTITY; 2],
            &noise_model(),
            0.05,
            &RobustnessConfig::default(),
        );
        let m = maps.plane(1);
        // The blob occupies guide pixels 20..32.
        let inside = m.data[26 * m.width + 26];
        let outside = m.data[10 * m.width + 10];
        assert!(inside < 0.1, "moving object not rejected: weight {inside}");
        assert!(outside > 0.9, "static background rejected: weight {outside}");
        assert!(maps.rejected_fraction[1] > 0.0);
    }

    /// The failure mode of judging every frame against one frame.
    ///
    /// A satellite, a cosmic ray or an aircraft in the *reference* makes every
    /// other frame disagree with it exactly where the defect is. Each of them
    /// is then rejected there for telling the truth, and the only frame left
    /// contributing is the one carrying the artefact -- which is how a defect
    /// in one exposure ends up in the result at full strength.
    #[test]
    fn a_defect_in_the_reference_does_not_reject_everyone_else() {
        let marked = frame_with_blob(128, 128, Some((40, 40, 24)));
        let clean = || frame_with_blob(128, 128, None);
        let frames = vec![marked, clean(), clean(), clean(), clean(), clean()];
        let warps = vec![WarpField::identity(); 6];
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 6],
            &[PhotometricMatch::IDENTITY; 6],
            &noise_model(),
            0.05,
            &RobustnessConfig::default(),
        );
        // Guide pixel 26,26 is inside the blob. Five frames agree with each
        // other there and disagree only with the reference, so they are the
        // consensus and must keep their weight.
        for i in 1..6 {
            let m = maps.plane(i);
            let w = m.data[26 * m.width + 26];
            assert!(
                w > 0.5,
                "frame {i} was rejected where it disagreed with a defect in the \
                 reference: weight {w}"
            );
        }
    }

    /// The converse, and the one that must not break: an aircraft in one frame
    /// out of many is a minority of one and stays rejected. Giving weight back
    /// to whatever disagrees would be no rejection at all.
    #[test]
    fn one_frame_with_an_object_is_still_rejected_among_many() {
        let clean = || frame_with_blob(128, 128, None);
        let frames = vec![
            clean(),
            clean(),
            frame_with_blob(128, 128, Some((40, 40, 24))),
            clean(),
            clean(),
            clean(),
        ];
        let warps = vec![WarpField::identity(); 6];
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 6],
            &[PhotometricMatch::IDENTITY; 6],
            &noise_model(),
            0.05,
            &RobustnessConfig::default(),
        );
        let m = maps.plane(2);
        let inside = m.data[26 * m.width + 26];
        assert!(inside < 0.1, "the odd frame out kept its weight: {inside}");
        // And the frames that agree with each other are untouched.
        let other = maps.plane(3);
        assert!(other.data[26 * other.width + 26] > 0.9);
    }

    #[test]
    fn a_pure_exposure_difference_is_normalised_away() {
        let reference = frame_with_blob(128, 128, None);
        let mut brighter = frame_with_blob(128, 128, None);
        {
            let mut vals: Vec<f32> = (0..128 * 128).map(|i| brighter.value_at(i) * 1.10).collect();
            brighter.samples = SamplePlane::from_normalised(128, 128, std::mem::take(&mut vals));
        }
        let frames = vec![reference, brighter];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        // The photometric match supplies a gain of 1/1.10 for this frame.
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 2],
            &[PhotometricMatch::IDENTITY, PhotometricMatch::from_exposure(1.0 / 1.10)],
            &noise_model(),
            0.05,
            &RobustnessConfig::default(),
        );
        let m = maps.plane(1);
        let mut lo = 1.0f32;
        for y in 8..m.height - 8 {
            for x in 8..m.width - 8 {
                lo = lo.min(m.data[y * m.width + x]);
            }
        }
        assert!(lo > 0.9, "exposure normalisation failed: min weight {lo}");
    }

    #[test]
    fn frames_that_do_not_overlap_are_rejected_wholesale() {
        let frames = vec![frame_with_blob(128, 128, None), frame_with_blob(128, 128, None)];
        let warps = vec![
            WarpField::identity(),
            WarpField::global_only(GlobalTransform::translation(4000.0, 0.0)),
        ];
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 2],
            &[PhotometricMatch::IDENTITY; 2],
            &noise_model(),
            0.05,
            &RobustnessConfig::default(),
        );
        assert!(maps.rejected_fraction[1] > 0.99, "{}", maps.rejected_fraction[1]);
    }

    #[test]
    fn disabling_robustness_costs_nothing() {
        let frames = vec![frame_with_blob(64, 64, None), frame_with_blob(64, 64, Some((10, 10, 20)))];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let cfg = RobustnessConfig { enabled: false, ..Default::default() };
        let maps = build_maps(&frames, &warps, 0, &[true; 2], &[PhotometricMatch::IDENTITY; 2], &noise_model(), 0.05, &cfg);
        assert!(maps.maps.is_empty());
        assert_eq!(maps.at(1, 10.0, 10.0), 1.0);
    }

    /// A map with a known ramp across it, at guide resolution.
    fn ramp_map(w: usize, h: usize) -> RobustnessMaps {
        let data: Vec<u8> = (0..w * h).map(|i| ((i % w) * 30).min(255) as u8).collect();
        RobustnessMaps {
            width: w,
            height: h,
            maps: vec![Plane::from_vec(w, h, data)],
            rejected_fraction: vec![0.0],
            consensus_luma: None,
        }
    }

    #[test]
    fn the_weight_field_does_not_arrive_in_two_pixel_steps() {
        // The map is one value per 2x2 block of sensor sites. Read by
        // truncation, both sensor pixels of a block get the same weight and the
        // next block jumps -- so on a burst where a fifth of the samples are
        // suppressed, the number of frames reaching each output pixel varies on
        // a two-pixel grid and that grid is stamped into the background.
        let maps = ramp_map(8, 4);
        // From the first cell centre to the last. Beyond those the field is
        // deliberately held flat, which the edge test covers.
        let row: Vec<f32> = (0..15).map(|i| maps.at(0, i as f32 + 0.5, 3.5)).collect();

        let steps: Vec<f32> = row.windows(2).map(|w| w[1] - w[0]).collect();
        let smallest = steps.iter().cloned().fold(f32::INFINITY, f32::min);
        let largest = steps.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            smallest > 0.0,
            "the field is flat somewhere along a ramp, so it is being read in steps: {row:?}"
        );
        assert!(
            largest - smallest < 0.02 * largest,
            "the steps are uneven ({smallest} to {largest}), which is a staircase not a slope"
        );
    }

    #[test]
    fn a_guide_cell_centre_still_reads_its_own_value() {
        // Interpolation must not move the field: the sensor coordinate at the
        // middle of a guide cell has to give that cell back exactly, or every
        // weight in the burst is quietly shifted half a cell.
        let maps = ramp_map(8, 4);
        for g in 0..8 {
            let got = maps.at(0, 2.0 * g as f32 + 0.5, 2.0 * 1.0 + 0.5);
            let want = (g * 30) as f32 / 255.0;
            assert!((got - want).abs() < 1e-6, "cell {g} read {got}, holds {want}");
        }
    }

    #[test]
    fn a_coordinate_off_the_map_is_still_no_weight_at_all() {
        let maps = ramp_map(8, 4);
        assert_eq!(maps.at(0, -1.0, 4.0), 0.0);
        assert_eq!(maps.at(0, 4.0, -1.0), 0.0);
        assert_eq!(maps.at(0, 16.0, 4.0), 0.0);
        assert_eq!(maps.at(0, 4.0, 8.0), 0.0);
        // And the outer half-cell holds the edge value rather than fading out,
        // which is what the truncating version did and what the merge expects
        // at the very edge of the reconstructed area.
        assert!((maps.at(0, 0.0, 0.0) - 0.0).abs() < 1e-6);
        let last = (7 * 30) as f32 / 255.0;
        assert!((maps.at(0, 15.9, 7.9) - last).abs() < 1e-6);
    }

    /// The same scene lit by a gradient the photometric match was not given.
    ///
    /// This stands in for what really happens: a sky glow the match followed
    /// only approximately, leaving a smooth residual over the whole frame.
    fn frame_with_unmodelled_glow(w: usize, h: usize, amount: f32) -> RawFrame {
        let mut f = frame_with_blob(w, h, None);
        let mut data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let u = x as f32 / w as f32;
                data[y * w + x] = 0.25 + 0.2 * u + amount * (0.4 + 0.6 * u);
            }
        }
        f.samples = SamplePlane::from_normalised(w, h, data);
        f
    }

    #[test]
    fn a_frame_is_not_rejected_everywhere_for_being_lit_differently() {
        // A frame that disagrees with the reference over its whole area is not
        // a frame full of moving objects. It is a frame the photometric match
        // did not quite finish, and the tolerance has to be relative to what
        // this frame usually does or the merge throws away everything it was
        // given. Judged against a noise model alone this frame is rejected
        // wholesale; judged against its own typical disagreement it is kept.
        let (w, h) = (96usize, 96usize);
        let frames = vec![
            frame_with_blob(w, h, None),
            frame_with_unmodelled_glow(w, h, 0.01),
        ];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let cfg = RobustnessConfig::default();
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 2],
            &[PhotometricMatch::IDENTITY; 2],
            &noise_model(),
            0.05,
            &cfg,
        );
        assert!(
            maps.rejected_fraction[1] < 0.05,
            "a uniformly mislit frame lost {:.0}% of itself",
            100.0 * maps.rejected_fraction[1]
        );
    }

    #[test]
    fn a_frame_under_cloud_is_not_forgiven_for_being_wrong_everywhere() {
        // The other side of the leniency. A frame whose sky is a fifth brighter
        // than the burst's over its whole area is not a frame the photometric
        // match did not quite finish; it is a frame shot through cloud, and if
        // it may set its own tolerance from that disagreement it is admitted
        // in full and the cloud is averaged into the sky. Above the cap it is
        // held to the same standard as any other frame, and the part of it
        // that is wrong by more goes.
        let (w, h) = (96usize, 96usize);
        let frames = vec![
            frame_with_blob(w, h, None),
            frame_with_blob(w, h, None),
            frame_with_blob(w, h, None),
            frame_with_unmodelled_glow(w, h, 0.05),
        ];
        let warps = vec![WarpField::identity(); 4];
        let cfg = RobustnessConfig::default();
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 4],
            &[PhotometricMatch::IDENTITY; 4],
            &noise_model(),
            0.05,
            &cfg,
        );
        assert!(
            maps.rejected_fraction[3] > 0.5,
            "a frame a fifth brighter than the burst kept {:.0}% of itself",
            100.0 * (1.0 - maps.rejected_fraction[3])
        );
        assert!(
            maps.rejected_fraction[1] < 0.05,
            "a clean frame lost {:.0}% of itself beside it",
            100.0 * maps.rejected_fraction[1]
        );
    }

    #[test]
    fn a_moving_object_is_still_found_on_a_mislit_frame() {
        // The other half. Being lenient about the whole frame must not make the
        // rule blind to the thing it exists for: a blob that is nowhere near
        // what the rest of the frame is doing still has to go.
        let (w, h) = (96usize, 96usize);
        let mut lit = frame_with_unmodelled_glow(w, h, 0.01);
        let mut data: Vec<f32> = (0..w * h)
            .map(|i| lit.samples.value_in_cell(i, (i / w % 2) * 2 + (i % w % 2)))
            .collect();
        for y in 40..56 {
            for x in 40..56 {
                data[y * w + x] = 0.95;
            }
        }
        lit.samples = SamplePlane::from_normalised(w, h, data);
        let frames = vec![frame_with_blob(w, h, None), lit];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let cfg = RobustnessConfig::default();
        let maps = build_maps(
            &frames,
            &warps,
            0,
            &[true; 2],
            &[PhotometricMatch::IDENTITY; 2],
            &noise_model(),
            0.05,
            &cfg,
        );
        let m = &maps.maps[1];
        let (gw, _gh) = (m.width, m.height);
        let inside = m.data[24 * gw + 24] as f32 / 255.0;
        let outside = m.data[6 * gw + 6] as f32 / 255.0;
        assert!(
            inside < 0.25,
            "the object was trusted at {inside:.2} on a mislit frame"
        );
        assert!(
            outside > 0.75,
            "the rest of the mislit frame was distrusted at {outside:.2}"
        );
    }
}
