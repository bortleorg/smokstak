//! Fixed-pattern sensor defects, found from the burst.
//!
//! A hot pixel is not noise. It is a site that reads high in every frame, by
//! about the same amount, and no amount of averaging removes it. What averaging
//! does instead is worse than leaving it: the frames are aligned before they are
//! combined, so a defect fixed to the *sensor* is dragged across the *scene*,
//! and one bad site becomes a short coloured streak as long as the burst's
//! motion. On the IC 1340 burst that was roughly sixteen hundred streaks, in
//! single channels, over an otherwise clean stack — the most visible flaw in the
//! image and one no metric in the suite noticed.
//!
//! Two properties separate a defect from a star, and both are needed:
//!
//! * **It does not vary.** The excess over its neighbours is the same in every
//!   frame. A real point source lands on different sites as the camera dithers,
//!   so at any one site its contribution comes and goes.
//! * **It stands alone.** The excess is confined to one site. A real source is
//!   spread by the optics across neighbouring sites of the same colour, however
//!   undersampled it is.
//!
//! Requiring both means the detector does not need to know how far the burst
//! moved, only that it moved at all — which the caller checks, because on a
//! burst with no motion a point source is indistinguishable from a defect by
//! any test whatsoever.
//!
//! How completely it finds them depends on the scene, and the difference is
//! large. On a flat field it is essentially complete: 1525 sites on the
//! IC 1340 burst, against roughly 1600 counted by hand, with the streaks gone
//! and the flux of the brightest four hundred stars unchanged to three decimal
//! places. On the synthetic resolution chart, under a dither of a pixel and a
//! half, it finds about three quarters — because there the scene at a site
//! persists between frames almost as well as a defect does. That is the right
//! way round: the miss leaves a streak, and the alternative error deletes a
//! star.
//!
//! Comparison is always against the four nearest sites of the *same* colour,
//! two pixels away, so the mosaic is never crossed and a red site is never
//! judged against a green one.
//!
//! Long monochrome bursts also admit a second proof: a pixel can remain a
//! strong, isolated excess while its amplitude changes between nights. Require
//! that evidence in at least 80% of 16 or more exposures, checking all eight
//! immediate neighbors in each exposure. This path never crosses a CFA mosaic.

use rayon::prelude::*;
use sr_core::frame::{NoiseModel, RawFrame};
use sr_core::math;
use sr_core::samples::DefectMask;

/// How far above the local noise a site's typical excess must sit.
///
/// Eight sigma of the *per-sample* noise, which for the burst this was
/// developed on is a few hundred sensor codes: high enough that ordinary sky
/// never reaches it, low enough to catch the warm pixels that are too faint to
/// see in one frame and perfectly visible once thirty-six are stacked.
const EXCESS_SIGMA: f32 = 8.0;

/// The excess must be this stable across the burst: its median absolute
/// deviation, over the median itself.
///
/// On a flat field the two populations separate by an order of magnitude —
/// defects come in around 0.04 to 0.10, scene content well above 1.0 — so
/// anything between them would do. It is set nearer the defects than the
/// midpoint because falsely deleting a star is a worse error than leaving a
/// streak.
///
/// A low quantile of the excess was tried instead, on the reasoning that what
/// defines a defect is that the excess never goes away. It is a better
/// description and a worse test: it raised recall on the synthetic chart from
/// 76% to 81% and the false positives from 5 to 249, and cost every scenario
/// in the suite half a decibel. Asking whether the excess is *steady* turns
/// out to discriminate far better than asking whether it is always present,
/// because scene content under a small dither is often always present too.
const STABILITY: f32 = 0.5;

/// A defect's excess must exceed its same-colour neighbours' by this factor.
///
/// This is the isolation test. A star two pixels wide raises its neighbours
/// too; a dead or hot site does not raise anything.
const ISOLATION: f32 = 3.0;

const MIN_VARIABLE_FRAMES: usize = 16;

// Unlike the single-frame amplitude gate above, this evidence must repeat in
// at least 80% of a long mono burst and dominate all eight immediate neighbors.
// Eight sigma misses persistent warm sites visible after integration.
const PERSISTENT_MONO_SIGMA: f32 = 4.0;

/// Per-exposure evidence for a changing mono hot pixel. Neighbor checks must
/// happen before temporal aggregation: median neighbors can hide a moving PSF.
fn isolated_mono_hot(frame: &RawFrame, x: usize, y: usize, d: f32, local: f32, noise: &NoiseModel) -> bool {
    if d <= 0.0 || d * d < PERSISTENT_MONO_SIGMA * PERSISTENT_MONO_SIGMA * noise.variance(local) {
        return false;
    }
    for ny in y - 1..=y + 1 {
        for nx in x - 1..=x + 1 {
            if nx == x && ny == y { continue; }
            let i = ny * frame.width + nx;
            let v = frame.value_at(i);
            if !frame.usable_value(i, v) || d < ISOLATION * (v - local).abs() {
                return false;
            }
        }
    }
    true
}

/// Rows processed together. Each band holds one difference image per frame, so
/// this trades memory against how often the burst is traversed.
const BAND: usize = 32;

/// Refuse to run below this many frames: a median and a deviation over fewer
/// samples than this decide nothing.
pub const MIN_FRAMES: usize = 8;

/// Motion, in sensor pixels, the burst must have before this can be believed.
///
/// One pixel. Below it a point source lands on the same site frame after frame
/// and is a fixed pattern by every test there is, including both of the ones
/// here. Callers check this; the scan itself cannot, because it never sees the
/// geometry.
pub const MIN_MOTION_PX: f32 = 1.0;

/// What the scan found.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefectReport {
    pub hot: usize,
    pub cold: usize,
    /// Defective sites as a fraction of the sensor.
    pub fraction: f32,
}

impl DefectReport {
    pub fn total(&self) -> usize {
        self.hot + self.cold
    }

    pub fn describe(&self) -> String {
        if self.total() == 0 {
            return "Fixed-pattern defects: none found".to_string();
        }
        format!(
            "Fixed-pattern defects: {} sites masked ({} hot, {} cold, {:.4}% of the sensor)",
            self.total(),
            self.hot,
            self.cold,
            self.fraction * 100.0
        )
    }
}

/// A site's excess over the four nearest sites of its own colour.
///
/// `None` at the border, and wherever the sample or its neighbours are not
/// usable — a saturated star's neighbours say nothing about it.
#[inline]
fn excess(frame: &RawFrame, x: usize, y: usize) -> Option<(f32, f32)> {
    let w = frame.width;
    let i = y * w + x;
    let v = frame.value_at(i);
    if !frame.usable_value(i, v) {
        return None;
    }
    let mut nb = [0.0f32; 4];
    for (k, (nx, ny)) in [(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)]
        .into_iter()
        .enumerate()
    {
        let j = ny * w + nx;
        let n = frame.value_at(j);
        if !frame.usable_value(j, n) {
            return None;
        }
        nb[k] = n;
    }
    let local = math::median(&nb);
    Some((v - local, local))
}

/// Scan a burst for sites that misread in the same way in every frame.
///
/// The caller is responsible for two things. The burst must have moved — with
/// no motion this will happily report every star in the field, because with no
/// motion every star *is* a fixed pattern. And the frames must be of the same
/// scene: the steadiness test asks whether a site's excess over its neighbours
/// varies, and frames through different filters vary for reasons that have
/// nothing to do with the sensor. Handed a mixed narrowband set, it finds
/// roughly a third fewer of the sites it finds in any one filter of that set.
///
/// Takes references so that a subset of a burst can be scanned without moving
/// any frames.
pub fn find_fixed_pattern(frames: &[&RawFrame], noise: &NoiseModel) -> (DefectMask, DefectReport) {
    find_fixed_pattern_tiled(frames, noise, 128)
}

fn find_fixed_pattern_tiled(frames: &[&RawFrame], noise: &NoiseModel, tile_width: usize) -> (DefectMask, DefectReport) {
    let (w, h) = (frames[0].width, frames[0].height);
    let mut mask = DefectMask::none(w, h);
    if frames.len() < MIN_FRAMES || w < 8 || h < 8 {
        return (mask, DefectReport::default());
    }
    let variable_mono = frames.len() >= MIN_VARIABLE_FRAMES && frames.iter().all(|f| f.cfa.is_mono());

    // One pass per band over the whole burst, so that each frame is read in
    // row order rather than the burst being strided through per site.
    // Bound scratch by a spatial tile, rather than the entire sensor width.
    // Two-pixel halos preserve the original same-colour isolation decision.
    let tiles: Vec<(usize, usize)> = (2..h - 2).step_by(BAND)
        .flat_map(|y| (4..w - 4).step_by(tile_width).map(move |x| (x, y))).collect();
    let found: Vec<Vec<(usize, bool)>> = tiles
        .par_iter()
        .map(|&(x0, y0)| {
            let x1 = (x0 + tile_width).min(w - 4);
            let xa = x0 - 2;
            let xb = x1 + 2;
            let y1 = (y0 + BAND).min(h - 2);
            // Two rows of halo either side, so that the isolation test can look
            // at a site's same-colour neighbours without the band edge hiding
            // defects from it. Without the halo one row in eight goes untested,
            // which is not a rounding error, it is a visible fraction of the
            // streaks left in the picture.
            let ya = y0.saturating_sub(2).max(2);
            let yb = (y1 + 2).min(h - 2);
            let rows = yb - ya;
            let span = xb - xa;
            let n = frames.len();
            // Excess per frame for every site of this band, plus the local
            // level the noise threshold is taken at.
            let mut diff = vec![f32::NAN; n * rows * span];
            let mut level = vec![0.0f32; rows * span];
            let mut level_n = vec![0u32; rows * span];
            let mut isolated_count = vec![0usize; rows * span];
            for (f, frame) in frames.iter().enumerate() {
                for y in ya..yb {
                    for x in xa..xb {
                        let k = (y - ya) * span + (x - xa);
                        if let Some((d, local)) = excess(frame, x, y) {
                            diff[f * rows * span + k] = d;
                            level[k] += local;
                            level_n[k] += 1;
                            if variable_mono && isolated_mono_hot(frame, x, y, d, local, noise) {
                                isolated_count[k] += 1;
                            }
                        }
                    }
                }
            }

            // Median and deviation of each site's excess over the burst.
            let mut med = vec![0.0f32; rows * span];
            let mut stable = vec![false; rows * span];
            let mut buf: Vec<f32> = Vec::with_capacity(n);
            for k in 0..rows * span {
                buf.clear();
                for f in 0..n {
                    let d = diff[f * rows * span + k];
                    if d.is_finite() {
                        buf.push(d);
                    }
                }
                if buf.len() * 2 < n {
                    continue;
                }
                let m = math::median(&buf);
                let mad = math::mad_sigma(&buf);
                med[k] = m;
                stable[k] = mad < STABILITY * m.abs();
            }

            let mut out = Vec::new();
            for y in y0..y1 {
                // The halo is context, not a candidate: a site two rows from
                // the sensor's own edge has no neighbours to be isolated from.
                if y < ya + 2 || y + 2 >= yb {
                    continue;
                }
                for x in x0..x1 {
                    let k = (y - ya) * span + (x - xa);
                    if variable_mono && isolated_count[k] * 5 >= n * 4 {
                        out.push((y * w + x, true));
                        continue;
                    }
                    if !stable[k] || level_n[k] == 0 {
                        continue;
                    }
                    let m = med[k];
                    let local = level[k] / level_n[k] as f32;
                    if m.abs() < EXCESS_SIGMA * noise.std_dev(local) {
                        continue;
                    }
                    // Isolation: a defect is one site, and its same-colour
                    // neighbours are two away in each direction.
                    let worst_neighbour = [
                        med[k - 2],
                        med[k + 2],
                        med[k - 2 * span],
                        med[k + 2 * span],
                    ]
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max);
                    if m.abs() < ISOLATION * worst_neighbour {
                        continue;
                    }
                    out.push((y * w + x, m > 0.0));
                }
            }
            out
        })
        .collect();

    let mut report = DefectReport::default();
    for band in found {
        for (i, hot) in band {
            mask.set(i);
            if hot {
                report.hot += 1;
            } else {
                report.cold += 1;
            }
        }
    }
    report.fraction = report.total() as f32 / (w * h) as f32;
    (mask, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::cfa::CfaPattern;
    use sr_core::frame::{FrameMetadata, NoiseSource};
    use sr_core::samples::{Levels, SamplePlane};

    const N: usize = 96;

    fn refs(frames: &[RawFrame]) -> Vec<&RawFrame> {
        frames.iter().collect()
    }

    fn noise() -> NoiseModel {
        NoiseModel::new(2.0e-5, 4.0e-6, NoiseSource::Manual)
    }

    /// A burst of sky with point sources that move with the dither, plus a set
    /// of sites that read high in every frame however the camera moved.
    fn burst(frames: usize, stars: &[(i32, i32)], hot: &[usize], dither: i32) -> Vec<RawFrame> {
        (0..frames)
            .map(|f| {
                let mut s = (f as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut rnd = move || {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (((s >> 40) & 0x3ff) as f32 - 512.0) * 0.6
                };
                // Whole-pixel dither, so a star lands on a different site in
                // each frame while a defect does not move.
                let (ox, oy) = if f == 0 {
                    (0, 0)
                } else {
                    ((f as i32 % (2 * dither + 1)) - dither, (f as i32 / 3 % (2 * dither + 1)) - dither)
                };
                let mut data = vec![0u16; N * N];
                for y in 0..N {
                    for x in 0..N {
                        data[y * N + x] = (6000.0 + rnd()).clamp(0.0, 65535.0) as u16;
                    }
                }
                for &(sx, sy) in stars {
                    // Two sensor pixels across, which is one site of each
                    // colour: as undersampled as a real point source gets.
                    for dy in 0..2i32 {
                        for dx in 0..2i32 {
                            let x = sx + ox + dx;
                            let y = sy + oy + dy;
                            if x >= 0 && y >= 0 && (x as usize) < N && (y as usize) < N {
                                data[y as usize * N + x as usize] += 9000;
                            }
                        }
                    }
                }
                for &i in hot {
                    data[i] = 20000 + (f as u16 % 200);
                }
                RawFrame {
                    width: N,
                    height: N,
                    samples: SamplePlane::from_u16(
                        N,
                        N,
                        data,
                        Levels::new([0.0; 4], [65535.0; 4]),
                    ),
                    cfa: CfaPattern::RGGB,
                    defects: DefectMask::none(N, N),
                    noise: noise(),
                    metadata: FrameMetadata::default(),
                }
            })
            .collect()
    }

    #[test]
    fn tiled_scan_preserves_band_decisions_at_tile_seams() {
        let hot = [20*N+11, 30*N+12, 40*N+13, 50*N+19, 60*N+20, 70*N+21];
        let mut frames = burst(24, &[(24,24),(44,30)], &hot, 3);
        for mono in [false, true] {
            if mono { for frame in &mut frames { frame.cfa = CfaPattern::MONO; } }
            let references = refs(&frames);
            let (expected, report) = find_fixed_pattern_tiled(&references, &noise(), N);
            let (actual, tiled_report) = find_fixed_pattern_tiled(&references, &noise(), 8);
            for i in 0..N*N { assert_eq!(expected.get(i), actual.get(i), "site {i}"); }
            assert_eq!(report.hot, tiled_report.hot);
            assert_eq!(report.cold, tiled_report.cold);
            assert!(actual.count() >= hot.len());
        }
    }

    #[test]
    fn persistent_mono_warm_pixel_below_single_frame_threshold_is_found() {
        let mut frames = burst(24, &[(20,20),(44,30)], &[], 3);
        let warm = 50*N+15;
        let transient = 75*N+75;
        let broad = 25*N+65;
        for (f,frame) in frames.iter_mut().enumerate() {
            frame.cfa = CfaPattern::MONO;
            let data = match &mut frame.samples.data {
                sr_core::samples::SampleData::U16(v) => v,
                _ => unreachable!(),
            };
            // About six sigma in each exposure, with unambiguous isolation.
            // Reset only these neighborhoods to avoid stochastic boundary cases.
            for center in [warm,transient,broad] {
                for dy in -2isize..=2 { for dx in -2isize..=2 {
                    data[(center as isize+dy*N as isize+dx) as usize]=6000;
                }}
            }
            data[warm]=6950;
            if f<4 { data[transient]=6950; }
            for i in [broad,broad+1,broad+N,broad+N+1] { data[i]=6950; }
        }
        let (mask,report)=find_fixed_pattern(&refs(&frames),&noise());
        assert!(mask.get(warm), "persistent isolated warm signal was missed");
        assert_eq!(report.hot,1, "moving stars, broad structure and transients must survive");
        for frame in &mut frames { frame.cfa=CfaPattern::RGGB; }
        assert!(!find_fixed_pattern(&refs(&frames),&noise()).0.get(warm));
        for frame in &mut frames { frame.cfa=CfaPattern::MONO; }
        assert!(!find_fixed_pattern(&refs(&frames[..15]),&noise()).0.get(warm));
    }

    #[test]
    fn persistent_mono_four_sigma_site_requires_repeated_isolation() {
        let mut frames = burst(24, &[(20,20),(44,30)], &[], 3);
        let warm = 50*N+15;
        let transient = 75*N+75;
        let broad = 25*N+65;
        for (f,frame) in frames.iter_mut().enumerate() {
            frame.cfa = CfaPattern::MONO;
            let data = match &mut frame.samples.data {
                sr_core::samples::SampleData::U16(v) => v,
                _ => unreachable!(),
            };
            // About four-and-a-half sigma in each exposure, with unambiguous isolation.
            // Reset only these neighborhoods to avoid stochastic boundary cases.
            for center in [warm,transient,broad] {
                for dy in -2isize..=2 { for dx in -2isize..=2 {
                    data[(center as isize+dy*N as isize+dx) as usize]=6000;
                }}
            }
            data[warm]=6700;
            if f<4 { data[transient]=6700; }
            for i in [broad,broad+1,broad+N,broad+N+1] { data[i]=6700; }
        }
        let (mask,report)=find_fixed_pattern(&refs(&frames),&noise());
        assert!(mask.get(warm), "persistent isolated warm signal was missed");
        assert_eq!(report.hot,1, "moving stars, broad structure and transients must survive");
        for frame in &mut frames { frame.cfa=CfaPattern::RGGB; }
        assert!(!find_fixed_pattern(&refs(&frames),&noise()).0.get(warm));
        for frame in &mut frames { frame.cfa=CfaPattern::MONO; }
        assert!(!find_fixed_pattern(&refs(&frames[..15]),&noise()).0.get(warm));
    }

    #[test]
    fn variable_mono_hot_pixel_is_found_without_masking_moving_stars_or_transients() {
        let stars = [(20, 20), (44, 30), (60, 62)];
        let mut frames = burst(24, &stars, &[], 3);
        let hot = 50 * N + 15;
        let transient = 75 * N + 75;
        let broad = 25 * N + 65;
        for (f, frame) in frames.iter_mut().enumerate() {
            frame.cfa = CfaPattern::MONO;
            let data = match &mut frame.samples.data {
                sr_core::samples::SampleData::U16(v) => v,
                _ => unreachable!(),
            };
            let amplitude = [3000, 6000, 12000, 30000][f / 6];
            data[hot] = 6000 + amplitude;
            // A PSF occupying adjacent mono pixels is not a sensor defect,
            // even when it varies and its center stays at the same location.
            for i in [broad, broad + 1, broad + N, broad + N + 1] {
                data[i] = 6000 + amplitude;
            }
            if f < 4 { data[transient] = 30000; }
        }
        let (mask, report) = find_fixed_pattern(&refs(&frames), &noise());
        assert!(mask.get(hot));
        assert_eq!(report.hot, 1, "moving stars and the broad source must survive");
        assert!(!mask.get(transient));
        for i in [broad, broad + 1, broad + N, broad + N + 1] { assert!(!mask.get(i)); }

        // This is new mono evidence, not a change to the existing CFA rule.
        for frame in &mut frames { frame.cfa = CfaPattern::RGGB; }
        assert!(!find_fixed_pattern(&refs(&frames), &noise()).0.get(hot));
    }

    #[test]
    fn variable_hot_requires_evidence_in_eighty_percent_of_all_exposures() {
        let hot = 50 * N + 15;
        let mut frames = burst(24, &[], &[], 3);
        for (f, frame) in frames.iter_mut().enumerate() {
            frame.cfa = CfaPattern::MONO;
            let data = match &mut frame.samples.data {
                sr_core::samples::SampleData::U16(v) => v,
                _ => unreachable!(),
            };
            data[hot] = if f < 19 { 6000 + [3000, 6000, 12000, 30000][f % 4] } else { 65535 };
        }
        assert!(!find_fixed_pattern(&refs(&frames), &noise()).0.get(hot), "saturated samples are not positive evidence");
        if let sr_core::samples::SampleData::U16(data) = &mut frames[19].samples.data { data[hot] = 36000; }
        assert!(find_fixed_pattern(&refs(&frames), &noise()).0.get(hot));
        assert!(!find_fixed_pattern(&refs(&frames[..12]), &noise()).0.get(hot), "short bursts keep the conservative rule");
    }

    #[test]
    fn hot_sites_are_found_and_stars_are_not() {
        let stars = [(20, 20), (44, 30), (60, 62), (30, 70), (70, 24)];
        let hot: Vec<usize> = vec![
            25 * N + 51,
            40 * N + 12,
            55 * N + 80,
            66 * N + 40,
            18 * N + 66,
            72 * N + 55,
        ];
        let frames = burst(24, &stars, &hot, 3);
        let (mask, report) = find_fixed_pattern(&refs(&frames), &noise());

        for &i in &hot {
            assert!(mask.get(i), "missed the hot site at {i}");
        }
        assert_eq!(report.hot, hot.len(), "flagged {} sites, planted {}", report.hot, hot.len());
        assert_eq!(report.cold, 0);

        // No star, at any of its dithered positions, may be masked.
        for &(sx, sy) in &stars {
            for dy in -4i32..6 {
                for dx in -4i32..6 {
                    let (x, y) = (sx + dx, sy + dy);
                    if x < 0 || y < 0 || x as usize >= N || y as usize >= N {
                        continue;
                    }
                    assert!(
                        !mask.get(y as usize * N + x as usize),
                        "masked a star site at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn dead_sites_are_found_too() {
        let mut frames = burst(20, &[(30, 30)], &[], 3);
        let dead = 50 * N + 50;
        for f in frames.iter_mut() {
            let data = match &mut f.samples.data {
                sr_core::samples::SampleData::U16(v) => v,
                _ => unreachable!(),
            };
            // Not zero: zero normalises to the black point and is discarded as
            // unusable before this ever sees it, which would hide the defect
            // rather than record it.
            data[dead] = 400;
        }
        let (mask, report) = find_fixed_pattern(&refs(&frames), &noise());
        assert!(mask.get(dead), "missed the dead site");
        assert_eq!(report.cold, 1);
    }

    #[test]
    fn too_short_a_burst_is_declined_rather_than_guessed_at() {
        let hot = vec![25 * N + 51];
        let frames = burst(4, &[], &hot, 3);
        let (mask, report) = find_fixed_pattern(&refs(&frames), &noise());
        assert!(mask.is_empty());
        assert_eq!(report.total(), 0);
    }
}
