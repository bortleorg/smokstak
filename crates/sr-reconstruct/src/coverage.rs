//! Stage 10: measured sub-pixel sampling diversity.
//!
//! Before promising 2x, find out whether the burst actually contains samples at
//! the intermediate positions. A hundred frames from a rigid tripod may agree
//! to better than a twentieth of a pixel: that burst can be denoised
//! magnificently and super-resolved not at all. The software should say so
//! rather than deliver an upsampled average with a confident filename.

use rayon::prelude::*;

use sr_core::cfa::CfaPattern;
use sr_core::frame::RawFrame;
use sr_core::geometry::WarpField;
use sr_core::plane::Plane;
use sr_core::product::SamplingCoverage;

/// Effective number of distinct occupied positions in a histogram, via the
/// perplexity `exp(H)`. Used for the human-readable histogram only.
fn perplexity(hist: &[f64]) -> f32 {
    let total: f64 = hist.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0f64;
    for &c in hist {
        if c > 0.0 {
            let p = c / total;
            h -= p * p.ln();
        }
    }
    h.exp() as f32
}

/// Running circular statistics of sub-pixel phase, per frame.
///
/// Sub-pixel phase is an angle, not a position on a line, and binning it is a
/// trap: a burst whose phase lands on a bin edge has its samples split between
/// two neighbouring bins by nothing more than rounding, and any bin-counting
/// measure then reports diversity that does not exist. Accumulating
/// `exp(i*2*pi*phase)` avoids the question entirely.
#[derive(Clone, Copy, Default)]
struct PhaseMoments {
    cos: f64,
    sin: f64,
    n: f64,
}

impl PhaseMoments {
    #[inline]
    fn add(&mut self, phase: f32) {
        let a = std::f64::consts::TAU * phase as f64;
        self.cos += a.cos();
        self.sin += a.sin();
        self.n += 1.0;
    }

    /// Circular mean phase of this frame, in `[0, 1)`.
    fn mean_phase(&self) -> Option<f32> {
        if self.n <= 0.0 {
            return None;
        }
        let (x, y) = (self.cos / self.n, self.sin / self.n);
        if x.abs() < 1e-12 && y.abs() < 1e-12 {
            return None;
        }
        let a = y.atan2(x) / std::f64::consts::TAU;
        Some((a as f32).rem_euclid(1.0))
    }

    /// How concentrated this frame's own phases are, in `[0, 1]`. Below 1 the
    /// frame spans a range of phases by itself, which rotation and local warp
    /// both produce.
    fn concentration(&self) -> f32 {
        if self.n <= 0.0 {
            return 0.0;
        }
        (((self.cos * self.cos + self.sin * self.sin).sqrt()) / self.n) as f32
    }
}

/// Effective number of distinguishable phases, from the largest uncovered gap.
///
/// The question the reconstruction actually needs answered is not "are the
/// phases clustered?" but "is there a phase region with no samples in it?" —
/// because a gap is where the output grid has nothing to interpolate from.
/// So the measure is the largest gap between consecutive phases around the
/// circle: `N` evenly spread phases leave gaps of `1/N`, a single phase leaves
/// a gap of 1, and two phases half a pixel apart leave gaps of 1/2, which is
/// exactly the diversity 2x requires.
///
/// This also avoids a trap that a clustering measure falls into. `N` phases
/// drawn at random do not cancel exactly; their resultant is about `1/sqrt(N)`
/// by chance alone, which a resultant-based measure reads as a quarter of the
/// diversity being absent for a 16-frame burst when in fact none is.
fn effective_phases(per_frame: &[PhaseMoments], scale: f32) -> f32 {
    let mut phases: Vec<f32> = per_frame.iter().filter_map(|f| f.mean_phase()).collect();
    if phases.is_empty() {
        return 0.0;
    }
    if phases.len() == 1 {
        // A single frame still covers a range of phases if its own samples are
        // spread, which is what a rotated or locally warped frame does.
        let spread = 1.0 - per_frame[0].concentration();
        return 1.0 + spread.clamp(0.0, 1.0) * (scale - 1.0).max(0.0);
    }
    phases.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Wrap-around gap, then every interior gap.
    let mut max_gap = phases[0] + 1.0 - phases[phases.len() - 1];
    for i in 1..phases.len() {
        max_gap = max_gap.max(phases[i] - phases[i - 1]);
    }
    if max_gap <= 1e-6 {
        return phases.len() as f32;
    }
    (1.0 / max_gap).clamp(1.0, phases.len() as f32)
}

/// Accumulate the sub-pixel phase of every contributing sample, per channel.
///
/// `warps` map each frame's sensor coordinates into reference sensor
/// coordinates; `scale` is the output magnification. Sampling is strided
/// because the distribution converges long before every one of two billion
/// sites has been visited.
pub fn analyse_coverage(
    frames: &[RawFrame],
    warps: &[WarpField],
    scale: f32,
    bins: usize,
) -> SamplingCoverage {
    assert_eq!(frames.len(), warps.len());
    let bins = bins.max(2);
    let stride = 8usize;

    type FrameStats = ([Vec<f64>; 3], [PhaseMoments; 2], [[PhaseMoments; 2]; 3]);
    let per_frame: Vec<FrameStats> = frames
        .par_iter()
        .zip(warps.par_iter())
        .map(|(f, w)| {
            let mut hist: [Vec<f64>; 3] = [
                vec![0.0; bins * bins],
                vec![0.0; bins * bins],
                vec![0.0; bins * bins],
            ];
            let mut axis = [PhaseMoments::default(); 2];
            let mut per_channel = [[PhaseMoments::default(); 2]; 3];
            // Visit all four positions of every sampled mosaic cell. Striding
            // by an even step alone would only ever land on one CFA phase and
            // would report the other channels as having no coverage at all.
            let mut cy = 0;
            while cy + 1 < f.height {
                let mut cx = 0;
                while cx + 1 < f.width {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (x, y) = (cx + dx, cy + dy);
                            let i = y * f.width + x;
                            if !f.usable_value(i, f.value(x, y)) {
                                continue;
                            }
                            let (rx, ry) = w.map(x as f32, y as f32);
                            // Reference sensor coordinate -> output grid.
                            let ox = (rx + 0.5) * scale - 0.5;
                            let oy = (ry + 0.5) * scale - 0.5;
                            let px = ox - ox.floor();
                            let py = oy - oy.floor();
                            let bx = ((px * bins as f32) as usize).min(bins - 1);
                            let by = ((py * bins as f32) as usize).min(bins - 1);
                            let c = f.channel_at(x, y);
                            hist[c][by * bins + bx] += 1.0;
                            axis[0].add(px);
                            axis[1].add(py);
                            per_channel[c][0].add(px);
                            per_channel[c][1].add(py);
                        }
                    }
                    cx += stride;
                }
                cy += stride;
            }
            (hist, axis, per_channel)
        })
        .collect();

    let mut hist: [Vec<f64>; 3] = [
        vec![0.0; bins * bins],
        vec![0.0; bins * bins],
        vec![0.0; bins * bins],
    ];
    // Phase evidence stays per frame; only the histogram is pooled.
    let mut axis_x: Vec<PhaseMoments> = Vec::with_capacity(per_frame.len());
    let mut axis_y: Vec<PhaseMoments> = Vec::with_capacity(per_frame.len());
    let mut chan: [[Vec<PhaseMoments>; 2]; 3] = Default::default();
    for (h, a, pc) in &per_frame {
        for c in 0..3 {
            for i in 0..bins * bins {
                hist[c][i] += h[c][i];
            }
            chan[c][0].push(pc[c][0]);
            chan[c][1].push(pc[c][1]);
        }
        axis_x.push(a[0]);
        axis_y.push(a[1]);
    }

    let combined: Vec<f64> = (0..bins * bins)
        .map(|i| hist[0][i] + hist[1][i] + hist[2][i])
        .collect();

    let nx = effective_phases(&axis_x, scale);
    let ny = effective_phases(&axis_y, scale);

    // Grade diversity against what the requested scale *needs*, not against the
    // histogram resolution. Reconstructing at `s` requires `s` distinguishable
    // phases per axis; a burst with exactly that many is fully served, and one
    // with a single phase is not served at all. Note that this is why a
    // half-sensor-pixel shift buys nothing at 2x: it is a whole output pixel.
    let grade1d = |n: f32| {
        if scale <= 1.0 {
            1.0
        } else {
            ((n - 1.0) / (scale - 1.0)).clamp(0.0, 1.0)
        }
    };
    let grade2d = |n: f32| {
        if scale <= 1.0 {
            1.0
        } else {
            ((n - 1.0) / (scale * scale - 1.0)).clamp(0.0, 1.0)
        }
    };

    // Uniformity keeps the histogram view: it is a description of the
    // distribution's shape rather than an input to the verdict.
    let uniformity = grade2d(perplexity(&combined));
    let channel_occupancy = [
        grade1d(effective_phases(&chan[0][0], scale).min(effective_phases(&chan[0][1], scale))),
        grade1d(effective_phases(&chan[1][0], scale).min(effective_phases(&chan[1][1], scale))),
        grade1d(effective_phases(&chan[2][0], scale).min(effective_phases(&chan[2][1], scale))),
    ];

    // A scale of `s` needs `s` distinguishable phases per axis. The effective
    // phase count is the honest measure of that.
    let supported = nx.min(ny);
    let recommended_scale = supported.clamp(1.0, scale.max(1.0));

    let verdict = if recommended_scale >= scale - 0.05 {
        format!(
            "The burst carries enough sub-pixel diversity for {scale:.2}x \
             (effective phases per axis: {nx:.1} horizontal, {ny:.1} vertical)."
        )
    } else if recommended_scale <= 1.05 {
        format!(
            "The frames are almost perfectly co-registered (effective phases per axis: \
             {nx:.1} horizontal, {ny:.1} vertical). This burst supports strong denoising \
             but essentially no true super-resolution; anything above 1.0x would be \
             interpolation."
        )
    } else {
        format!(
            "Sub-pixel diversity supports about {recommended_scale:.2}x, below the requested \
             {scale:.2}x (effective phases per axis: {nx:.1} horizontal, {ny:.1} vertical). \
             Detail beyond the supported scale will be interpolated, not measured."
        )
    };

    SamplingCoverage {
        channels: frames[0].channels(),
        scale,
        bins,
        phase_histogram: combined,
        channel_occupancy,
        uniformity,
        horizontal_diversity: grade1d(nx),
        vertical_diversity: grade1d(ny),
        recommended_scale,
        verdict,
    }
}

/// Spatial map of local phase diversity over the output grid.
///
/// Turbulence and local warp make diversity vary across the frame; a single
/// global number would hide that. Each cell reports the effective number of
/// distinct phases available to it.
pub fn phase_coverage_map(
    frames: &[RawFrame],
    warps: &[WarpField],
    scale: f32,
    cell: usize,
) -> Plane<f32> {
    let (w, h) = (frames[0].width, frames[0].height);
    let gw = w.div_ceil(cell);
    let gh = h.div_ceil(cell);
    let bins = 4usize;

    let mut acc: Vec<Vec<f64>> = vec![vec![0.0; bins * bins]; gw * gh];
    let stride = 8usize;
    for (f, wf) in frames.iter().zip(warps) {
        let mut cy = 0;
        while cy + 1 < f.height {
            let mut cx = 0;
            while cx + 1 < f.width {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (x, y) = (cx + dx, cy + dy);
                        let i = y * f.width + x;
                        if !f.usable_value(i, f.value(x, y)) {
                            continue;
                        }
                        let (rx, ry) = wf.map(x as f32, y as f32);
                        if rx >= 0.0 && ry >= 0.0 && (rx as usize) < w && (ry as usize) < h {
                            let g = (ry as usize / cell) * gw + (rx as usize / cell);
                            let ox = (rx + 0.5) * scale - 0.5;
                            let oy = (ry + 0.5) * scale - 0.5;
                            let bx = (((ox - ox.floor()) * bins as f32) as usize).min(bins - 1);
                            let by = (((oy - oy.floor()) * bins as f32) as usize).min(bins - 1);
                            acc[g][by * bins + bx] += 1.0;
                        }
                    }
                }
                cx += stride;
            }
            cy += stride;
        }
    }

    Plane::from_vec(gw, gh, acc.iter().map(|hgram| perplexity(hgram)).collect())
}

/// Sub-pixel phase of each frame's global translation, in output pixels.
///
/// A compact view of what the burst offers before any per-sample analysis: for
/// a translation-dominated burst this is essentially the whole story.
pub fn frame_phases(warps: &[WarpField], cfa: &CfaPattern, scale: f32) -> Vec<(f32, f32)> {
    let _ = cfa;
    warps
        .iter()
        .map(|w| {
            let (rx, ry) = w.map(0.0, 0.0);
            let ox = (rx + 0.5) * scale - 0.5;
            let oy = (ry + 0.5) * scale - 0.5;
            (ox - ox.floor(), oy - oy.floor())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::cfa::CfaPattern;
    use sr_core::frame::{NoiseModel, RawFrame};
    use sr_core::geometry::GlobalTransform;
    use sr_core::samples::{DefectMask, SamplePlane};

    fn frame(w: usize, h: usize) -> RawFrame {
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_normalised(w, h, vec![0.5; w * h]),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            noise: NoiseModel::nominal(100.0, 16000.0),
            metadata: Default::default(),
        }
    }

    #[test]
    fn identical_frames_report_no_super_resolution_support() {
        let frames: Vec<RawFrame> = (0..16).map(|_| frame(128, 128)).collect();
        let warps: Vec<WarpField> = (0..16).map(|_| WarpField::identity()).collect();
        let c = analyse_coverage(&frames, &warps, 2.0, 4);
        assert!(
            c.recommended_scale < 1.1,
            "recommended {} for a perfectly static burst",
            c.recommended_scale
        );
        assert!(c.verdict.contains("denoising"), "{}", c.verdict);
    }

    #[test]
    fn evenly_spread_shifts_support_the_requested_scale() {
        // Sixteen frames whose shifts tile the half-pixel grid at 2x output.
        let frames: Vec<RawFrame> = (0..16).map(|_| frame(128, 128)).collect();
        let warps: Vec<WarpField> = (0..16)
            .map(|i| {
                let dx = (i % 4) as f32 * 0.25;
                let dy = (i / 4) as f32 * 0.25;
                WarpField::global_only(GlobalTransform::translation(dx, dy))
            })
            .collect();
        let c = analyse_coverage(&frames, &warps, 2.0, 4);
        assert!(
            c.recommended_scale > 1.9,
            "recommended {} with fully diverse sampling",
            c.recommended_scale
        );
        assert!(c.horizontal_diversity > 0.9, "h {}", c.horizontal_diversity);
        assert!(c.vertical_diversity > 0.9, "v {}", c.vertical_diversity);
    }

    #[test]
    fn diversity_in_one_axis_only_is_reported_as_such() {
        let frames: Vec<RawFrame> = (0..8).map(|_| frame(128, 128)).collect();
        let warps: Vec<WarpField> = (0..8)
            .map(|i| {
                WarpField::global_only(GlobalTransform::translation((i % 4) as f32 * 0.25, 0.0))
            })
            .collect();
        let c = analyse_coverage(&frames, &warps, 2.0, 4);
        assert!(c.horizontal_diversity > 0.9, "h {}", c.horizontal_diversity);
        assert!(c.vertical_diversity < 0.1, "v {}", c.vertical_diversity);
        assert!(
            c.recommended_scale < 1.2,
            "recommended {}",
            c.recommended_scale
        );
    }

    #[test]
    fn half_pixel_sensor_shifts_add_nothing_at_2x() {
        // A half-sensor-pixel shift is exactly one output pixel at 2x, so a
        // burst offering only 0 and 0.5 has one phase, not two.
        let frames: Vec<RawFrame> = (0..8).map(|_| frame(128, 128)).collect();
        let warps: Vec<WarpField> = (0..8)
            .map(|i| {
                let d = (i % 2) as f32 * 0.5;
                WarpField::global_only(GlobalTransform::translation(d, d))
            })
            .collect();
        let c = analyse_coverage(&frames, &warps, 2.0, 4);
        assert!(
            c.recommended_scale < 1.1,
            "recommended {}",
            c.recommended_scale
        );
        assert!(c.horizontal_diversity < 0.1, "h {}", c.horizontal_diversity);
    }

    #[test]
    fn every_cfa_channel_is_sampled() {
        // An even stride over a 2x2 mosaic lands on one CFA phase only, which
        // would report the other two channels as having no coverage.
        let frames: Vec<RawFrame> = (0..8).map(|_| frame(128, 128)).collect();
        let warps: Vec<WarpField> = (0..8)
            .map(|i| {
                WarpField::global_only(GlobalTransform::translation(
                    (i % 4) as f32 * 0.25,
                    (i / 4) as f32 * 0.25,
                ))
            })
            .collect();
        let c = analyse_coverage(&frames, &warps, 2.0, 4);
        for (i, &occ) in c.channel_occupancy.iter().enumerate() {
            assert!(occ > 0.5, "channel {i} occupancy {occ}");
        }
    }

    #[test]
    fn randomly_shifted_frames_read_as_fully_diverse() {
        // Phases drawn at random do not cancel exactly; without debiasing their
        // residual resultant would be read as a quarter of the diversity being
        // absent, which is a property of the sample size, not of the burst.
        let frames: Vec<RawFrame> = (0..16).map(|_| frame(128, 128)).collect();
        let mut seed = 987654321u64;
        let warps: Vec<WarpField> = (0..16)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let a = (seed >> 33) as f32 / (1u32 << 31) as f32;
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let b = (seed >> 33) as f32 / (1u32 << 31) as f32;
                WarpField::global_only(GlobalTransform::translation(a * 3.0, b * 3.0))
            })
            .collect();
        let c = analyse_coverage(&frames, &warps, 2.0, 8);
        assert!(
            c.recommended_scale > 1.9,
            "random sub-pixel shifts reported as only {}x",
            c.recommended_scale
        );
    }

    #[test]
    fn phase_on_a_bin_edge_is_not_mistaken_for_diversity() {
        // Every frame at the same phase, jittered by a hair. A histogram-based
        // measure splits these across two bins and reports diversity that is
        // not there; the circular measure does not.
        let frames: Vec<RawFrame> = (0..16).map(|_| frame(128, 128)).collect();
        let warps: Vec<WarpField> = (0..16)
            .map(|i| {
                let d = if i % 2 == 0 { -0.002 } else { 0.002 };
                WarpField::global_only(GlobalTransform::translation(d, d))
            })
            .collect();
        let c = analyse_coverage(&frames, &warps, 2.0, 8);
        assert!(
            c.recommended_scale < 1.05,
            "jitter reported as diversity: {}",
            c.recommended_scale
        );
    }

    #[test]
    fn perplexity_matches_intuition() {
        assert!((perplexity(&[1.0, 0.0, 0.0, 0.0]) - 1.0).abs() < 1e-5);
        assert!((perplexity(&[1.0, 1.0, 1.0, 1.0]) - 4.0).abs() < 1e-4);
    }
}
