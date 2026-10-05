//! Frame quality measured on point sources.
//!
//! Gradient energy answers "how much fine detail is in this frame", which on a
//! landscape is a good proxy for focus and on a star field is mostly a proxy
//! for how bright the sky was. Two frames of the same field through the same
//! optics differ in one thing that matters — the size and shape of the stars —
//! and that is worth measuring directly rather than inferring.
//!
//! Directly is also more useful to read. "Half-flux diameter 2.4 px, 6.8
//! arcseconds" says what is wrong with a frame and roughly why; "estimated blur
//! 0.53" does not.
//!
//! ## What is measured
//!
//! **Half-flux diameter.** The diameter of the circle containing half a star's
//! flux, doubled. For a Gaussian this equals the full width at half maximum
//! exactly, and unlike a profile fit it is an integral, so it survives the
//! undersampling that short focal lengths produce — the burst this was written
//! against samples at 2.8 arcseconds per pixel, where a star is barely two
//! pixels across and no peak fit means anything.
//!
//! Trailing shows up in it for free: an elongated star's flux is spread further
//! from its centre, so its half-flux diameter grows. A frame ruined by a guiding
//! error is soft by this measure without anything being said about direction.
//!
//! **Eccentricity and its angle**, from the second moments, which is what says
//! *why* a frame is soft. Seeing is round; wind, guiding error and field
//! rotation are not.
//!
//! **How many stars were measurable**, which is a transparency proxy. Cloud
//! removes stars from a frame before it softens the ones that remain.
//!
//! ## Working on the mosaic
//!
//! Everything here runs on the undemosaiced frame, because the half-resolution
//! guide the other metrics use samples a star at one pixel and a shape cannot be
//! measured from one sample. The mosaic then intrudes in two places, and both
//! had to be dealt with:
//!
//! * **Finding sources.** A site's brightness carries the mosaic's own
//!   modulation, so detection runs on 2x2 cell sums, which do not. Detecting on
//!   sites works for narrow sources and fails progressively as they broaden: the
//!   core goes flat, the two green sites of a cell tie for brightest, both pass
//!   a local-maximum test, and the crowding rule then discards the pair. That
//!   made the metric fall silent above about four pixels of width — precisely
//!   the frames worth flagging.
//! * **Measuring shape.** Each site is divided by its own mosaic position's
//!   measured response to the burst's stars before any moment is computed.
//!   Without it the two greens lying on one diagonal are read as elongation
//!   along that diagonal, on a perfectly round star, in every frame.
//!
//! ## What it does not measure
//!
//! The half-flux diameter is bounded by the measurement window: a source much
//! wider than it has flux outside, and comes out short. The ordering survives —
//! a wider source still measures wider — but the width itself stops being
//! accurate somewhere past nine pixels.
//!
//! It is also not seeing. It is the width of what the sensor recorded, which
//! includes the optics, the focus, and any trailing. That is the right quantity
//! for ranking frames and the wrong one to quote as a site's atmosphere.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sr_core::frame::RawFrame;
use sr_core::math;
use sr_core::plane::Plane;

/// Detection threshold, in noise sigmas above the local sky.
const DETECT_SIGMA: f32 = 10.0;
/// A neighbour counts as part of the source above this many sigmas.
const NEIGHBOUR_SIGMA: f32 = 3.0;
/// Radius at which a point source must have returned to background.
const ANNULUS: f32 = 9.0;
/// How far above background the annulus may sit, in sigmas or as a fraction of
/// the source's own peak, whichever is more generous.
const ANNULUS_SIGMA: f32 = 4.0;
const ANNULUS_PEAK_FRACTION: f32 = 0.10;
/// Radius of the measurement window, in sensor pixels.
const WINDOW: usize = 11;

/// Annulus the local sky is read from, in pixels from the source.
///
/// Far enough out that a source the window can measure has put almost nothing
/// there -- the widest this code will report is a half-flux diameter near the
/// window, whose wings are three sigma inside the inner radius -- and no
/// further, so it costs no reads beyond the window already taken.
const LOCAL_SKY_INNER: f32 = 7.0;
const LOCAL_SKY_MID: f32 = 9.0;
const LOCAL_SKY_OUTER: f32 = 11.0;

/// How much brighter the inner half of the annulus may be than the outer half,
/// as a fraction of the source's peak, before the annulus is judged to be
/// holding the source's own wings rather than the sky behind it.
const ANNULUS_FLATNESS: f32 = 0.03;
/// Cells across the frame for spatial sampling, and how many stars to take from
/// each. Spread rather than brightest: the brightest stars are the ones nearest
/// saturation, where the profile is flat-topped and says nothing.
const GRID: usize = 8;
const PER_CELL: usize = 12;
/// Below this many measurable stars, the frame does not support the metric and
/// nothing is reported rather than something noisy.
pub const MIN_STARS: usize = 20;

/// How much of the frame has to sit at one level, within the sensor's own
/// noise, before "a source above the sky" means anything.
///
/// This is the test that keeps a daytime scene out. Detection works at the
/// sensor's noise, which on a clean frame is thousands of times below the
/// scene's own contrast, so on a resolution chart most of the frame is
/// hundreds of sigma from the median and every corner of every bar is a local
/// maximum surrounded by pixels darker than itself. No local test rejects
/// those, because locally they look exactly like stars. What separates the two
/// cases is global: a star field is background with a little signal on it, and
/// a photograph is not.
///
/// The astronomical burst this was built against measures 0.977. The margin to
/// 0.75 is wide on purpose, so that a frame with a large nebula or a bright
/// moon in it is still recognised.
const BACKGROUND_FRACTION: f32 = 0.75;

/// Tiles across the frame the background fraction is counted within, and the
/// samples a tile needs before it is counted at all.
const FLATNESS_TILES: usize = 8;
const FLATNESS_MIN_SAMPLES: usize = 32;

/// Point-source shape statistics for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StarMetrics {
    /// Stars that passed every rejection and were measured.
    pub count: usize,
    /// Median half-flux diameter, in sensor pixels. Equal to the full width at
    /// half maximum for a Gaussian source.
    pub hfd: f32,
    /// Median eccentricity: 0 is round, and a value approaching 1 is a line.
    pub eccentricity: f32,
    /// Direction of the elongation, in degrees anticlockwise from the sensor's
    /// x axis. Meaningless when `eccentricity` is small.
    pub angle_deg: f32,
}

impl StarMetrics {
    /// Half-flux diameter in arcseconds, when the frame knows its plate scale.
    pub fn hfd_arcsec(
        &self,
        pixel_pitch_um: Option<f32>,
        focal_length_mm: Option<f32>,
    ) -> Option<f32> {
        let (p, f) = (pixel_pitch_um?, focal_length_mm?);
        if !(p > 0.0 && f > 0.0) {
            return None;
        }
        // 206265 arcseconds per radian. With the pitch in microns and the focal
        // length in millimetres the two unit conversions cancel to a factor of
        // a thousand, leaving 206.265.
        Some(self.hfd * 206.265 * p / f)
    }

    pub fn describe(&self, pixel_pitch_um: Option<f32>, focal_length_mm: Option<f32>) -> String {
        let arcsec = match self.hfd_arcsec(pixel_pitch_um, focal_length_mm) {
            Some(a) => format!(" ({a:.2} arcsec)"),
            None => String::new(),
        };
        format!(
            "{} stars, half-flux diameter {:.2} px{}, eccentricity {:.2} at {:.0} deg",
            self.count, self.hfd, arcsec, self.eccentricity, self.angle_deg
        )
    }
}

/// One accepted detection, in sensor coordinates.
struct Detection {
    x: usize,
    y: usize,
}

/// Sky level and noise sigma for each position in the 2x2 mosaic cell.
struct Sky {
    level: [f32; 4],
    sigma: [f32; 4],
    /// Fraction of sampled sites within three sigma of their cell's level.
    background_fraction: f32,
}

fn sky_of(frame: &RawFrame) -> Sky {
    let mut level = [0.0f32; 4];
    let mut sigma = [0.0f32; 4];
    // A stride: the median of a few tens of thousands of samples locates the
    // sky to far better than the noise on any one of them.
    let step = ((frame.width * frame.height / 40_000).max(1) as f32)
        .sqrt()
        .max(1.0) as usize
        * 2;
    // Samples kept per tile as well as pooled, because how flat the sky is has
    // to be asked of the sky nearby -- see below.
    let mut tiles: Vec<Vec<f32>> = vec![Vec::new(); FLATNESS_TILES * FLATNESS_TILES * 4];
    for cell in 0..4 {
        let (ox, oy) = (cell % 2, cell / 2);
        let mut v = Vec::new();
        let mut y = oy;
        while y < frame.height {
            let mut x = ox;
            while x < frame.width {
                let s = frame.value(x, y);
                if frame.usable_value(y * frame.width + x, s) {
                    v.push(s);
                    let tx = (x * FLATNESS_TILES / frame.width).min(FLATNESS_TILES - 1);
                    let ty = (y * FLATNESS_TILES / frame.height).min(FLATNESS_TILES - 1);
                    tiles[(ty * FLATNESS_TILES + tx) * 4 + cell].push(s);
                }
                x += step;
            }
            y += step;
        }
        level[cell] = if v.is_empty() { 0.0 } else { math::median(&v) };
        sigma[cell] = frame.noise.std_dev(level[cell]).max(1e-9);
    }

    // How much of the frame is background, asked tile by tile.
    //
    // The question this answers is whether the frame is a star field or a
    // photograph: background with a little signal on it, or signal everywhere.
    // Asked of the frame as a whole it answers a different question, because
    // the sky over a wide field is not one level. Twenty of ninety-six frames
    // of one burst -- a whole second night, moonlit and gradient-ridden -- came out
    // at 0.73 to 0.75 against a threshold of 0.75, and lost their star metrics
    // and their registration polish for a gradient that no star measurement
    // here is affected by, since every source is measured against an annulus a
    // few pixels wide. A tile is small enough that the gradient across it is
    // far inside the noise, so what is left in the count is what was meant:
    // sources.
    let (mut flat, mut total) = (0u64, 0u64);
    for v in tiles.iter_mut() {
        if v.len() < FLATNESS_MIN_SAMPLES {
            continue;
        }
        let l = math::median(v);
        let sg = frame.noise.std_dev(l).max(1e-9);
        flat += v.iter().filter(|x| (*x - l).abs() < 3.0 * sg).count() as u64;
        total += v.len() as u64;
    }
    let background_fraction = if total == 0 {
        1.0
    } else {
        flat as f32 / total as f32
    };
    Sky {
        level,
        sigma,
        background_fraction,
    }
}

/// Value at a site in units of its own cell's noise, above its own cell's sky.
#[inline]
fn signal(frame: &RawFrame, sky: &Sky, x: usize, y: usize) -> f32 {
    let cell = (y & 1) * 2 + (x & 1);
    let i = y * frame.width + x;
    let v = frame.samples.value_in_cell(i, cell);
    if frame.defects.get(i) {
        return 0.0;
    }
    (v - sky.level[cell]) / sky.sigma[cell]
}

/// Signal-to-noise of each 2x2 mosaic cell, at half resolution.
///
/// Detection runs here rather than on individual sites, because a site's
/// brightness carries the mosaic's own modulation and a cell's does not. On a
/// narrow source that does not matter, since the true centre outshines the gain
/// difference between colours. On a broad one the core is flat, the two green
/// sites of a cell tie for brightest, *both* pass a local-maximum test, and the
/// crowding rule then discards the pair. The star rejects itself, and it does so
/// more often the softer it is: exactly the frames worth flagging.
///
/// Summing the four sites of a cell removes the modulation by construction,
/// because every cell holds one of each colour. Four samples of noise average to
/// half of one, so the sum halved is in the same units as a single site.
fn cell_snr(frame: &RawFrame, sky: &Sky) -> Plane<f32> {
    let (cw, ch) = (frame.width / 2, frame.height / 2);
    let mut out = Plane::<f32>::new(cw, ch);
    for cy in 0..ch {
        for cx in 0..cw {
            let mut acc = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    acc += signal(frame, sky, 2 * cx + dx, 2 * cy + dy);
                }
            }
            out.data[cy * cw + cx] = 0.5 * acc;
        }
    }
    out
}

/// Find candidate point sources: bright, locally maximal, not alone, not
/// saturated, not crowded.
///
/// The frame is divided into regions and a fixed number taken from each, so that
/// the sample describes the whole field rather than whichever corner happened to
/// be richest. Within a region the candidates are found first and thinned
/// afterwards: stopping at the first dozen would sample the top-left of every
/// region, which is a spatial bias dressed up as a spatial sample.
fn detect(frame: &RawFrame, sky: &Sky) -> Vec<Detection> {
    detect_with(frame, sky, false, PER_CELL)
}

/// `strongest` picks each cell's best sources rather than a spread of them,
/// and `per_cell` caps how many are taken from each.
fn detect_with(frame: &RawFrame, sky: &Sky, strongest: bool, per_cell: usize) -> Vec<Detection> {
    let (w, h) = (frame.width, frame.height);
    let m = WINDOW + 2;
    if w < 4 * m || h < 4 * m {
        return Vec::new();
    }
    let snr = cell_snr(frame, sky);
    let (cw, ch) = (snr.width, snr.height);
    let margin = m / 2 + 1;
    let region_w = cw / GRID;
    let region_h = ch / GRID;
    // In cells, so that a source and a duplicate of it cannot both be taken.
    let reach = WINDOW;

    let regions: Vec<Vec<Detection>> = (0..GRID * GRID)
        .into_par_iter()
        .map(|c| {
            let (gx, gy) = (c % GRID, c / GRID);
            let x0 = (gx * region_w).max(margin);
            let x1 = ((gx + 1) * region_w).min(cw - margin);
            let y0 = (gy * region_h).max(margin);
            let y1 = ((gy + 1) * region_h).min(ch - margin);
            if x1 <= x0 || y1 <= y0 {
                return Vec::new();
            }

            let mut cand: Vec<(usize, usize)> = Vec::new();
            for y in y0..y1 {
                for x in x0..x1 {
                    let s = snr.data[y * cw + x];
                    if s < DETECT_SIGMA {
                        continue;
                    }
                    // A flat-topped source has no measurable profile.
                    if (0..2).any(|dy| (0..2).any(|dx| frame.value(2 * x + dx, 2 * y + dy) >= 1.0))
                    {
                        continue;
                    }
                    // A point source sits on background. This separates a star
                    // from a bright patch of scene: a local maximum on a
                    // resolution chart is a corner of something that stays
                    // bright for many pixels in every direction.
                    if !returns_to_background(&snr, x, y, s) {
                        continue;
                    }
                    let mut peak = true;
                    let mut companions = 0usize;
                    for dy in -2i64..=2 {
                        for dx in -2i64..=2 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let n =
                                snr.data[(y as i64 + dy) as usize * cw + (x as i64 + dx) as usize];
                            if n > s {
                                peak = false;
                            }
                            if dx.abs() <= 1 && dy.abs() <= 1 && n > NEIGHBOUR_SIGMA {
                                companions += 1;
                            }
                        }
                    }
                    // A source confined to one cell is a defective site rather
                    // than a star: no optics put all of a point source into two
                    // microns of silicon.
                    if peak && companions >= 3 {
                        cand.push((x, y));
                    }
                }
            }

            // Crowding, against every candidate rather than only the accepted
            // ones: two sources inside one measurement window corrupt both, and
            // whether the other was sampled is beside the point. The list is in
            // raster order, so the neighbours of a candidate are the entries
            // within a few rows of it.
            let crowded: Vec<bool> = (0..cand.len())
                .map(|i| {
                    let (x, y) = cand[i];
                    let mut j = i;
                    while j > 0 {
                        j -= 1;
                        if y - cand[j].1 > reach {
                            break;
                        }
                        if cand[j].0.abs_diff(x) <= reach {
                            return true;
                        }
                    }
                    let mut j = i + 1;
                    while j < cand.len() {
                        if cand[j].1 - y > reach {
                            break;
                        }
                        if cand[j].0.abs_diff(x) <= reach {
                            return true;
                        }
                        j += 1;
                    }
                    false
                })
                .collect();
            let usable: Vec<(usize, usize)> = cand
                .into_iter()
                .zip(crowded)
                .filter(|(_, c)| !c)
                .map(|(p, _)| p)
                .collect();

            // Back to sensor coordinates at the cell origin. The measurement
            // finds its own centroid, so half a cell of imprecision here costs
            // nothing.
            let site = |(x, y): (usize, usize)| Detection { x: 2 * x, y: 2 * y };
            if strongest {
                // The same stars every time, which is what matching a pattern
                // between two frames needs. Ranked by the detection statistic
                // rather than by anything re-measured afterwards: two frames
                // agree on which sources are strongest far better than they
                // agree on a flux summed from their own pixels.
                let mut v: Vec<(usize, usize)> = usable;
                v.sort_by(|a, b| {
                    let (sa, sb) = (snr.data[a.1 * cw + a.0], snr.data[b.1 * cw + b.0]);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                v.truncate(per_cell);
                v.into_iter().map(site).collect()
            } else {
                // Evenly through the cell instead, so that the shape metrics
                // are not measured on its brightest sources alone.
                let stride = (usable.len() / per_cell).max(1);
                usable
                    .into_iter()
                    .step_by(stride)
                    .take(per_cell)
                    .map(site)
                    .collect()
            }
        })
        .collect();
    regions.into_iter().flatten().collect()
}

/// Whether the neighbourhood of a bright site comes back to the sky.
///
/// Sampled on a ring rather than over a disc: what matters is the level at a
/// radius the source should have faded by, and a dozen points around a circle
/// establish that far more cheaply than a filled annulus.
fn returns_to_background(snr: &Plane<f32>, x: usize, y: usize, peak: f32) -> bool {
    const POINTS: usize = 12;
    let mut ring = [0.0f32; POINTS];
    for (k, slot) in ring.iter_mut().enumerate() {
        let a = std::f32::consts::TAU * k as f32 / POINTS as f32;
        // Rounded before the bounds are checked, not after: a coordinate of
        // 255.6 is inside a 256-wide plane and its rounding is not.
        let nx = (x as f32 + ANNULUS * a.cos()).round();
        let ny = (y as f32 + ANNULUS * a.sin()).round();
        if nx < 0.0 || ny < 0.0 || nx >= snr.width as f32 || ny >= snr.height as f32 {
            return false;
        }
        *slot = snr.data[ny as usize * snr.width + nx as usize];
    }
    let mut v = ring;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = 0.5 * (v[POINTS / 2 - 1] + v[POINTS / 2]);
    median < (ANNULUS_SIGMA).max(ANNULUS_PEAK_FRACTION * peak)
}

/// Median response of each mosaic position to the burst's stars.
///
/// The four sites of a cell see one star through three different filters, so
/// their amplitudes differ by the star's colour and the sensor's sensitivity.
/// Dividing by this makes the four comparable, which is what a shape
/// measurement needs.
fn channel_response(frame: &RawFrame, sky: &Sky, stars: &[Detection]) -> [f32; 4] {
    let mut acc: [Vec<f32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for d in stars {
        // The core only: further out the star has faded into the sky and the
        // ratio would be a ratio of noise.
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let x = (d.x as i64 + dx) as usize;
                let y = (d.y as i64 + dy) as usize;
                let cell = (y & 1) * 2 + (x & 1);
                acc[cell].push(signal(frame, sky, x, y).max(0.0));
            }
        }
    }
    let mut out = [1.0f32; 4];
    let mut med = [0.0f32; 4];
    for c in 0..4 {
        med[c] = if acc[c].is_empty() {
            0.0
        } else {
            math::median(&acc[c])
        };
    }
    let reference = math::median(&med).max(1e-6);
    for c in 0..4 {
        // A cell that saw nothing keeps a response of one rather than becoming
        // a divide by almost zero.
        out[c] = if med[c] > 1e-3 {
            med[c] / reference
        } else {
            1.0
        };
    }
    out
}

/// Half-flux radius and second moments of one source.
///
/// `None` when the window holds no net flux, which happens when a detection was
/// a noise spike that survived the neighbour test.
/// `aperture` is the radius the second moments are taken over. `None` scales it
/// to the source's own size, which is fine for one frame in isolation and wrong
/// for comparing frames: a soft frame would then be measured through a wider
/// window than a sharp one, and the two shapes are no longer the same
/// measurement. [`measure_burst`] passes one aperture for the whole burst.
fn measure_one(
    frame: &RawFrame,
    sky: &Sky,
    response: &[f32; 4],
    d: &Detection,
    aperture: Option<f32>,
) -> Option<(f32, f32, f32)> {
    let r = WINDOW as i64;
    let mut pixels: Vec<(f32, f32, f32)> = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
    for dy in -r..=r {
        for dx in -r..=r {
            let x = (d.x as i64 + dx) as usize;
            let y = (d.y as i64 + dy) as usize;
            let cell = (y & 1) * 2 + (x & 1);
            let w = signal(frame, sky, x, y) / response[cell].max(1e-6);
            pixels.push((dx as f32, dy as f32, w));
        }
    }

    // The sky under *this* star, not the sky over the whole frame.
    //
    // `signal` removes one level per mosaic cell, measured across the entire
    // image. That is the right zero for detection and the wrong one for
    // measuring a source that sits on nebulosity: everything in the window is
    // then above zero, the half-flux integral never closes, and the radius
    // runs out to the window's edge. On a 16 mm frame of NGC 7000 -- emission
    // corner to corner -- that reported stars 10.3 pixels across which the
    // pixels plainly show to be three, and every metric built on it followed.
    //
    // A median over an annulus outside the source is robust to a neighbour
    // falling in it, which a mean would not be.
    let local_sky = {
        let ring = |lo: f32, hi: f32| -> Option<f32> {
            let v: Vec<f32> = pixels
                .iter()
                .filter(|(x, y, _)| {
                    let r = x.hypot(*y);
                    r >= lo && r <= hi
                })
                .map(|(_, _, w)| *w)
                .collect();
            (v.len() >= 8).then(|| math::median(&v))
        };
        let peak = pixels.iter().map(|(_, _, w)| *w).fold(0.0f32, f32::max);
        match (
            ring(LOCAL_SKY_INNER, LOCAL_SKY_MID),
            ring(LOCAL_SKY_MID, LOCAL_SKY_OUTER),
        ) {
            // Flat between the two rings: whatever is out there belongs to the
            // sky, and the source has stopped contributing. Take it away.
            (Some(inner), Some(outer)) if inner - outer <= ANNULUS_FLATNESS * peak.max(1e-6) => {
                outer
            }
            // Still falling: these are the source's own wings, and subtracting
            // them would measure the star as smaller than it is. A source that
            // wide is at the limit of what this window can measure anyway.
            _ => 0.0,
        }
    };
    let pixels: Vec<(f32, f32, f32)> = pixels
        .iter()
        .map(|&(x, y, w)| (x, y, w - local_sky))
        .collect();

    // Centroid from the core, where the source dominates the noise.
    let (mut cx, mut cy, mut cw) = (0.0f32, 0.0f32, 0.0f32);
    for &(x, y, w) in &pixels {
        if w > 0.0 && x.hypot(y) <= 2.5 {
            cx += w * x;
            cy += w * y;
            cw += w;
        }
    }
    if cw <= 0.0 {
        return None;
    }
    cx /= cw;
    cy /= cw;

    // Half-flux radius: the radius containing half the net flux in the window.
    // Deliberately an integral rather than a fit — at this plate scale a star
    // is barely two pixels across and there is nothing to fit a peak to.
    let mut radial: Vec<(f32, f32)> = pixels
        .iter()
        .map(|&(x, y, w)| ((x - cx).hypot(y - cy), w))
        .collect();
    radial.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = radial.iter().map(|p| p.1).sum();
    if total <= 0.0 {
        return None;
    }
    let half = 0.5 * total;
    let mut acc = 0.0f32;
    let mut hfr = radial.last().map(|p| p.0).unwrap_or(0.0);
    for i in 0..radial.len() {
        let next = acc + radial[i].1;
        if next >= half {
            // Linear interpolation between the two enclosing radii.
            let prev_r = if i == 0 { 0.0 } else { radial[i - 1].0 };
            let span = (radial[i].1).max(1e-9);
            hfr = prev_r + (radial[i].0 - prev_r) * ((half - acc) / span).clamp(0.0, 1.0);
            break;
        }
        acc = next;
    }

    // Second moments, over the source rather than the whole window: weighting
    // by the square of the radius makes far-out noise dominate otherwise.
    let limit = aperture.unwrap_or(2.5 * hfr).clamp(2.0, WINDOW as f32);
    let (mut m20, mut m02, mut m11, mut mw) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &(x, y, w) in &pixels {
        let (dx, dy) = (x - cx, y - cy);
        if w <= 0.0 || dx.hypot(dy) > limit {
            continue;
        }
        m20 += w * dx * dx;
        m02 += w * dy * dy;
        m11 += w * dx * dy;
        mw += w;
    }
    if mw <= 0.0 {
        return None;
    }
    let (m20, m02, m11) = (m20 / mw, m02 / mw, m11 / mw);
    let (l1, l2, e) = math::eig_sym2(m20, m11, m02);
    let (major, minor) = (l1.max(l2).max(1e-12), l1.min(l2).max(0.0));
    let ecc = (1.0 - minor / major).clamp(0.0, 1.0).sqrt();
    let angle = e[1].atan2(e[0]).to_degrees();
    Some((2.0 * hfr, ecc, angle))
}

/// Measure the point sources in one frame.
///
/// `None` when the frame does not hold enough of them to say anything, which is
/// every daytime scene and any exposure lost to cloud. Callers fall back to the
/// gradient metrics, which is what they had before.
pub use sr_core::star::Star;

/// The brightest stars in a frame, at sub-pixel positions.
///
/// The same detection and centroiding the quality metrics use, returning where
/// the stars are rather than how big they are. Registration wants this when a
/// frame has nothing else to be placed by: a pattern of stars is invariant to
/// rotation and scale in a way that image correlation is not.
///
/// `limit` caps the list at the brightest, because matching cost grows quickly
/// with it and the bright end is the part two frames reliably share.
pub fn positions(frame: &RawFrame, limit: usize) -> Vec<Star> {
    positions_impl(frame, limit)
}

/// The registration catalogue remeasured in normalized detector units for
/// photometry. Registration's ranking flux is noise-normalized and must never
/// be compared between exposures to derive a brightness scale.
pub fn positions_for_photometry(frame: &RawFrame, limit: usize) -> Vec<Star> {
    photometric_catalog(frame, &positions(frame, limit))
}

/// Preserve astrometric positions/order, replacing only ranking brightness by
/// signed aperture flux above a local, per-CFA-phase annulus median. Missing,
/// saturated or defective apertures are omitted rather than partially summed.
/// Aperture radius is 6 sensor pixels; the sky annulus spans radii 7–11.
pub fn photometric_catalog(frame: &RawFrame, positions: &[Star]) -> Vec<Star> {
    positions
        .iter()
        .filter_map(|s| aperture_flux(frame, s.x, s.y).map(|flux| Star { flux, ..*s }))
        .collect()
}

fn aperture_flux(frame: &RawFrame, cx: f32, cy: f32) -> Option<f32> {
    if !cx.is_finite()
        || !cy.is_finite()
        || cx < 12.0
        || cy < 12.0
        || cx + 12.0 >= frame.width as f32
        || cy + 12.0 >= frame.height as f32
    {
        return None;
    }
    let mut sky: [Vec<f32>; 4] = Default::default();
    let mut aperture = Vec::new();
    let (ix, iy) = (cx.round() as i64, cy.round() as i64);
    for dy in -12i64..=12 {
        for dx in -12i64..=12 {
            let (x, y) = ((ix + dx) as usize, (iy + dy) as usize);
            let r2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
            if r2 > 121.0 {
                continue;
            }
            let value = frame.value(x, y);
            let usable =
                value.is_finite() && value < 0.98 && !frame.defects.get(y * frame.width + x);
            let cell = (y & 1) * 2 + (x & 1);
            if r2 <= 36.0 {
                if !usable {
                    return None;
                }
                aperture.push((cell, value));
            } else if r2 >= 49.0 && usable {
                sky[cell].push(value);
            }
        }
    }
    if sky.iter().any(|s| s.len() < 12) {
        return None;
    }
    let sky = sky.map(|s| math::median(&s));
    // Keep negative sky-subtracted samples; clipping them biases faint sources.
    let flux = aperture
        .iter()
        .map(|&(cell, v)| v as f64 - sky[cell] as f64)
        .sum::<f64>() as f32;
    (flux.is_finite() && flux > 0.0).then_some(flux)
}

fn positions_impl(frame: &RawFrame, limit: usize) -> Vec<Star> {
    let sky = sky_of(frame);
    if sky.background_fraction < BACKGROUND_FRACTION {
        return Vec::new();
    }
    // Four per cell beyond what is asked for, so that the global truncation
    // still leaves the field evenly covered.
    let per_cell = (limit / (GRID * GRID) + 4).max(4);
    let stars = detect_with(frame, &sky, true, per_cell);
    if stars.len() < MIN_STARS {
        return Vec::new();
    }
    let response = channel_response(frame, &sky, &stars);
    let mut out: Vec<Star> = stars
        .par_iter()
        .filter_map(|d| {
            let (dx, dy, flux) = centroid_one(frame, &sky, &response, d)?;
            Some(Star {
                x: d.x as f32 + dx,
                y: d.y as f32 + dy,
                flux,
            })
        })
        .collect();
    // Brightest first, and deterministic where two are equal.
    out.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
    });
    out.truncate(limit);
    out
}

/// Sub-pixel offset from a detection's cell, and the flux it carries.
fn centroid_one(
    frame: &RawFrame,
    sky: &Sky,
    response: &[f32; 4],
    d: &Detection,
) -> Option<(f32, f32, f32)> {
    // Only the core. Summing the whole window would rank a faint star sitting
    // on bright nebulosity above a bright one on empty sky, because most of
    // the window is then nebula: on NGC 7023 that made the "brightest" stars
    // of two consecutive frames almost disjoint sets, and a star pattern that
    // does not survive from one frame to the next matches nothing.
    let r = 3i64;
    let (mut cx, mut cy, mut cw) = (0.0f32, 0.0f32, 0.0f32);
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx as f32).hypot(dy as f32) > 2.5 {
                continue;
            }
            let x = (d.x as i64 + dx) as usize;
            let y = (d.y as i64 + dy) as usize;
            let cell = (y & 1) * 2 + (x & 1);
            let w = signal(frame, sky, x, y) / response[cell].max(1e-6);
            if w > 0.0 {
                cx += w * dx as f32;
                cy += w * dy as f32;
                cw += w;
            }
        }
    }
    if cw <= 0.0 {
        return None;
    }
    Some((cx / cw, cy / cw, cw))
}

pub fn measure(frame: &RawFrame) -> Option<StarMetrics> {
    measure_with(frame, None)
}

/// Measure every frame of a burst on equal terms.
///
/// Two passes. The first finds how large the burst's stars are; the second
/// re-measures their shape through one aperture sized from that, so that a
/// frame is not judged through a window chosen by its own softness. The extra
/// pass costs about as much again as the first, which on a 36-frame burst is a
/// tenth of a second.
pub fn measure_burst(frames: &[RawFrame]) -> Vec<Option<StarMetrics>> {
    let first: Vec<Option<StarMetrics>> = frames.par_iter().map(measure).collect();
    let hfd: Vec<f32> = first
        .iter()
        .flatten()
        .map(|m| m.hfd)
        .filter(|h| *h > 0.0)
        .collect();
    if hfd.len() * 4 < frames.len() * 3 || hfd.len() < 3 {
        return first;
    }
    // Wide enough to hold the burst's stars, narrow enough that the sky beyond
    // them does not dominate a measurement weighted by the square of the
    // radius.
    let aperture = (1.25 * math::median(&hfd)).clamp(2.0, WINDOW as f32);
    frames
        .par_iter()
        .map(|f| measure_with(f, Some(aperture)))
        .collect()
}

/// Measure one frame, optionally through an aperture chosen elsewhere.
pub fn measure_with(frame: &RawFrame, aperture: Option<f32>) -> Option<StarMetrics> {
    let sky = sky_of(frame);
    if sky.background_fraction < BACKGROUND_FRACTION {
        log::debug!(
            "no star metric: only {:.3} of sites sit at the sky level",
            sky.background_fraction
        );
        return None;
    }
    let stars = detect(frame, &sky);
    if stars.len() < MIN_STARS {
        log::debug!("no star metric: {} sources found", stars.len());
        return None;
    }
    let response = channel_response(frame, &sky, &stars);

    let measured: Vec<(f32, f32, f32)> = stars
        .par_iter()
        .filter_map(|d| measure_one(frame, &sky, &response, d, aperture))
        .collect();
    if measured.len() < MIN_STARS {
        return None;
    }

    let hfd: Vec<f32> = measured.iter().map(|m| m.0).collect();
    let ecc: Vec<f32> = measured.iter().map(|m| m.1).collect();
    // Angles are directions, not headings: 179 degrees and -1 degree are the
    // same elongation. Averaging them as numbers would give 89, so they are
    // doubled onto the full circle, averaged as vectors, and halved back.
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    for m in &measured {
        let a = 2.0 * m.2.to_radians();
        sx += m.1 * a.cos();
        sy += m.1 * a.sin();
    }
    let angle = 0.5 * sy.atan2(sx).to_degrees();

    Some(StarMetrics {
        count: measured.len(),
        hfd: math::median(&hfd),
        eccentricity: math::median(&ecc),
        angle_deg: if angle < 0.0 { angle + 180.0 } else { angle },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::cfa::CfaPattern;
    use sr_core::frame::{FrameMetadata, NoiseModel, NoiseSource};
    use sr_core::samples::{DefectMask, Levels, SamplePlane};

    /// A star field with a known Gaussian point spread function.
    ///
    /// `sigma_x` and `sigma_y` are in sensor pixels, so an elongated pair plants
    /// a known eccentricity. Channel amplitudes differ, as a real mosaic's do,
    /// which is what the response normalisation has to remove.
    fn field(sigma_x: f32, sigma_y: f32, theta: f32, n_stars: usize) -> RawFrame {
        field_on(sigma_x, sigma_y, theta, n_stars, 0.0)
    }

    /// The same field on a sky that ramps across the frame.
    ///
    /// `gradient` is the total rise from one edge to the other, in units of the
    /// sky's own noise -- which is how a real gradient is worth stating, and
    /// what decides whether a flatness test survives it.
    fn field_on(sigma_x: f32, sigma_y: f32, theta: f32, n_stars: usize, gradient: f32) -> RawFrame {
        let (w, h) = (512usize, 512usize);
        let sky = [0.06f32, 0.09, 0.09, 0.08];
        let gain = [0.7f32, 1.0, 1.0, 0.85];
        let mut val = vec![0.0f32; w * h];
        // The noise the model gives at the sky, which the gradient is stated in.
        let sky_sigma = (1.0e-6f32 * 0.09 + 1.0e-8).sqrt();
        for y in 0..h {
            for x in 0..w {
                let ramp = gradient * sky_sigma * (x + y) as f32 / (w + h) as f32;
                val[y * w + x] = sky[(y & 1) * 2 + (x & 1)] + ramp;
            }
        }
        // A deterministic spread of stars, inset from the edges.
        let (c, s) = theta.to_radians().sin_cos();
        let (ct, st) = (s, c);
        for k in 0..n_stars {
            let kk = k as f32;
            let px = 40.0 + (kk * 97.0) % (w as f32 - 80.0);
            let py = 40.0 + (kk * 61.0) % (h as f32 - 80.0);
            let amp = 0.35 + 0.25 * ((kk * 0.37).sin().abs());
            let r = 10i64;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (fx, fy) = (dx as f32, dy as f32);
                    // Rotate into the source's own axes.
                    let u = fx * ct + fy * st;
                    let v = -fx * st + fy * ct;
                    let g = (-(u * u) / (2.0 * sigma_x * sigma_x)
                        - (v * v) / (2.0 * sigma_y * sigma_y))
                        .exp();
                    let x = px as i64 + dx;
                    let y = py as i64 + dy;
                    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                        continue;
                    }
                    let cell = ((y as usize) & 1) * 2 + ((x as usize) & 1);
                    val[y as usize * w + x as usize] += amp * gain[cell] * g;
                }
            }
        }
        let data: Vec<u16> = val
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 65535.0) as u16)
            .collect();
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_u16(w, h, data, Levels::new([0.0; 4], [65535.0; 4])),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            // Small enough that the detection threshold is well below the
            // planted stars and well above nothing.
            noise: NoiseModel::new(1.0e-6, 1.0e-8, NoiseSource::Manual),
            metadata: FrameMetadata::default(),
        }
    }

    #[test]
    fn photometric_flux_is_independent_of_detection_noise_units() {
        let clear = field_on(1.2, 1.2, 0.0, 64, 0.0);
        let mut changed_noise = clear.clone();
        changed_noise.noise.alpha *= 100.0;
        changed_noise.noise.beta *= 100.0;
        let catalogs = vec![
            positions_for_photometry(&clear, 512),
            positions_for_photometry(&changed_noise, 512),
        ];
        assert!(catalogs.iter().all(|c| c.len() >= 30));
        let gains = crate::photometry::star_gains(
            &[
                sr_core::geometry::WarpField::identity(),
                sr_core::geometry::WarpField::identity(),
            ],
            0,
            &catalogs,
            2,
        );
        assert!(
            (gains[1].unwrap()[0] - 1.0).abs() < 0.005,
            "changing a detection-noise model changed the inferred brightness"
        );
        let old = [positions(&clear, 512), positions(&changed_noise, 512)];
        let rank_ratio = old[0][0].flux / old[1][0].flux;
        assert!(
            (rank_ratio - 10.0).abs() < 0.01,
            "fixture must expose the ranking/flux units mismatch"
        );
    }

    #[test]
    fn aperture_flux_preserves_large_signal_scales_and_negative_calibrated_skies() {
        let original = field_on(1.2, 1.2, 0.0, 64, 0.0);
        let positions = positions(&original, 512);
        let baseline = photometric_catalog(&original, &positions);
        for scale in [1.0f32 / 18.0, 1.0] {
            let mut transformed = original.clone();
            transformed.samples = SamplePlane::from_normalised(
                original.width,
                original.height,
                (0..original.width * original.height)
                    .map(|i| original.value(i % original.width, i / original.width) * scale - 0.2)
                    .collect(),
            );
            let measured = photometric_catalog(&transformed, &positions);
            assert_eq!(baseline.len(), measured.len());
            for (a, b) in baseline.iter().zip(measured) {
                assert!((b.flux / a.flux / scale - 1.0).abs() < 0.001);
            }
        }
        let star = baseline[0];
        let mut defective = original.clone();
        defective
            .defects
            .set(star.y.round() as usize * original.width + star.x.round() as usize);
        assert!(aperture_flux(&defective, star.x, star.y).is_none());
        assert!(aperture_flux(&original, 1.0, 1.0).is_none());
    }

    #[test]
    fn half_flux_diameter_recovers_the_planted_width() {
        // For a Gaussian the half-flux diameter is the full width at half
        // maximum, which is 2.3548 sigma.
        for sigma in [1.0f32, 1.4, 2.0, 2.6, 3.2, 4.0] {
            let m = measure(&field(sigma, sigma, 0.0, 120)).expect("a star field is measurable");
            let want = 2.3548 * sigma;
            assert!(
                (m.hfd - want).abs() < 0.12 * want,
                "sigma {sigma}: measured {:.3}, wanted {want:.3}",
                m.hfd
            );
            // Every planted source has to survive, not just enough of them.
            // Attrition that grows with softness is the failure mode this
            // guards: it used to reach total silence by sigma 2.6, and a frame
            // that cannot be measured is a frame that cannot be ranked worst.
            assert!(
                m.count >= 100,
                "sigma {sigma}: only {} of 120 measured",
                m.count
            );
        }
    }

    #[test]
    fn the_measurement_stays_ordered_past_the_window() {
        // The window bounds the accuracy, not the ordering. A source wider than
        // the window has flux outside it, so the half-flux radius comes out
        // short — but it still comes out larger than a narrower source, and an
        // ordering is all the ranking needs. Silence would be worse: a frame
        // that cannot be measured cannot be ranked worst either.
        let a = measure(&field(4.0, 4.0, 0.0, 120)).unwrap();
        let b = measure(&field(6.0, 6.0, 0.0, 120)).unwrap();
        assert!(b.hfd > a.hfd * 1.15, "{:.2} then {:.2}", a.hfd, b.hfd);
        // Understated, as the window implies, and reported so in the docs
        // rather than being presented as an accurate width.
        assert!(b.hfd < 2.3548 * 6.0, "{:.2} was not understated", b.hfd);
    }

    #[test]
    fn a_softer_frame_measures_larger_than_a_sharper_one() {
        // Ranking is what the metric is for, and it has to be monotonic even
        // where the absolute figure is not exact.
        let sharp = measure(&field(0.9, 0.9, 0.0, 60)).unwrap();
        let mid = measure(&field(1.4, 1.4, 0.0, 60)).unwrap();
        let soft = measure(&field(2.8, 2.8, 0.0, 60)).unwrap();
        assert!(sharp.hfd < mid.hfd, "{} vs {}", sharp.hfd, mid.hfd);
        assert!(mid.hfd < soft.hfd, "{} vs {}", mid.hfd, soft.hfd);
    }

    #[test]
    fn round_stars_measure_round_despite_the_mosaic() {
        // The two greens of a cell lie on one diagonal and the red and blue
        // amplitudes differ, so without the per-channel response normalisation
        // this reports elongation at 45 degrees on a perfectly round source.
        let m = measure(&field(1.4, 1.4, 0.0, 60)).unwrap();
        assert!(
            m.eccentricity < 0.35,
            "round stars measured {:.3}",
            m.eccentricity
        );
    }

    #[test]
    fn a_sky_gradient_does_not_silence_the_metric() {
        // The flatness test asks whether the frame is background with a little
        // signal on it. Asked of the whole frame it answers a different
        // question, because a wide field's sky is not one level: a gradient of
        // twenty sigma edge to edge puts most of the frame more than three sigma
        // from the frame's median, and the metric falls silent on a frame whose
        // stars are perfectly measurable. Twenty of ninety-six frames of one burst
        // were lost that way, and with them their registration polish.
        let flat = measure(&field(1.4, 1.4, 0.0, 90)).expect("a flat sky is measurable");
        let sloped = measure(&field_on(1.4, 1.4, 0.0, 90, 20.0))
            .expect("a sky gradient is not a reason to give up on the stars");
        assert!(
            (sloped.hfd - flat.hfd).abs() < 0.1 * flat.hfd,
            "gradient changed the width from {:.2} to {:.2}",
            flat.hfd,
            sloped.hfd
        );
        assert!(
            sloped.count >= flat.count * 8 / 10,
            "gradient cost {} of {} sources",
            flat.count - sloped.count.min(flat.count),
            flat.count
        );
    }

    #[test]
    fn a_frame_that_is_signal_everywhere_is_still_refused() {
        // The other side of the same test, which is what it is for. A daytime
        // scene has no background: every part of it is far from every other
        // part, locally as well as globally, so making the flatness test local
        // must not let one through.
        let mut f = field(1.4, 1.4, 0.0, 60);
        let (w, h) = (f.width, f.height);
        let mut raised = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                // Bars: half the frame bright, half dark, at a scale far below
                // a tile, which is what a photograph looks like to this test.
                let bright = ((x / 7) + (y / 7)) % 2 == 0;
                raised[y * w + x] = if bright { 50_000 } else { 8_000 };
            }
        }
        f.samples = SamplePlane::from_u16(w, h, raised, Levels::new([0.0; 4], [65535.0; 4]));
        assert!(
            measure(&f).is_none(),
            "a photograph was measured as a star field"
        );
    }

    #[test]
    fn trailing_is_measured_as_elongation_along_its_own_axis() {
        let m = measure(&field(2.6, 1.0, 30.0, 60)).unwrap();
        assert!(m.eccentricity > 0.6, "eccentricity {:.3}", m.eccentricity);
        // The angle is a direction, so 30 and 210 are the same answer.
        let err = (m.angle_deg - 30.0).rem_euclid(180.0);
        let err = err.min(180.0 - err);
        assert!(err < 20.0, "angle {:.1} deg, planted 30", m.angle_deg);
        // Trailing costs sharpness without anything being said about direction.
        let round = measure(&field(1.0, 1.0, 0.0, 60)).unwrap();
        assert!(m.hfd > round.hfd * 1.3, "{} vs {}", m.hfd, round.hfd);
    }

    #[test]
    fn gradient_energy_ranks_a_softer_frame_higher_when_it_is_richer() {
        // The reason this module exists. Gradient energy measures how much
        // structure a frame contains, and on a star field that is mostly a
        // count of stars: a richer or brighter exposure scores higher whether
        // or not its stars are sharper. Here the softer frame has twice as many
        // sources, and gradient energy prefers it.
        let sharp = field(1.0, 1.0, 0.0, 40);
        let soft_but_rich = field(1.7, 1.7, 0.0, 80);

        let g_sharp = crate::tenengrad(&sharp.guide_rgb().luma());
        let g_soft = crate::tenengrad(&soft_but_rich.guide_rgb().luma());
        assert!(
            g_soft > g_sharp,
            "the fixture does not reproduce the failure: gradient energy {g_soft} vs {g_sharp}"
        );

        let s = measure(&sharp).unwrap();
        let r = measure(&soft_but_rich).unwrap();
        assert!(
            s.hfd < r.hfd,
            "half-flux diameter agreed with gradient energy: {:.3} vs {:.3}",
            s.hfd,
            r.hfd
        );
    }

    /// A scene with structure at every level, as a photograph has: bars and
    /// blocks spanning the whole range rather than background with sources on
    /// it.
    fn chart() -> RawFrame {
        let (w, h) = (512usize, 512usize);
        let mut data = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                // Bars of varying pitch, plus a coarse blocking, so that the
                // frame has bright regions, dark regions and edges everywhere.
                let pitch = 3 + (x / 64);
                let bar = if (x / pitch + y / pitch) % 2 == 0 {
                    0.75
                } else {
                    0.15
                };
                let block = 0.1 * ((x / 128 + y / 128) % 3) as f32;
                data[y * w + x] = ((bar + block).clamp(0.0, 1.0) * 65535.0) as u16;
            }
        }
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_u16(w, h, data, Levels::new([0.0; 4], [65535.0; 4])),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            noise: NoiseModel::new(1.0e-6, 1.0e-8, NoiseSource::Manual),
            metadata: FrameMetadata::default(),
        }
    }

    #[test]
    fn a_photograph_is_refused_rather_than_measured() {
        // The regression this guards. Detection works at the sensor's noise,
        // which on a clean daytime frame is far below the scene's contrast, so
        // every corner of every bar is a bright local maximum surrounded by
        // darker pixels — locally indistinguishable from a star. Measuring
        // them produced a half-flux diameter pinned at the window size and an
        // identical "sharpness" for every frame of the burst, which is worse
        // than no measurement: it replaced a ranking that worked.
        assert!(
            measure(&chart()).is_none(),
            "measured a resolution chart as a star field"
        );
    }

    #[test]
    fn a_scene_without_point_sources_reports_nothing() {
        let m = measure(&field(1.4, 1.4, 0.0, 3));
        assert!(m.is_none(), "measured {m:?} from three sources");
    }

    #[test]
    fn hot_sites_are_not_counted_as_stars() {
        // A hot pixel is brighter than any star here and confined to one site.
        // Counting it would drag the half-flux diameter towards zero, making a
        // frame look sharper the more broken its sensor is.
        let mut f = field(1.4, 1.4, 0.0, 60);
        let clean = measure(&f).unwrap();
        let data = match &mut f.samples.data {
            sr_core::samples::SampleData::U16(v) => v,
            _ => unreachable!(),
        };
        for k in 0..400 {
            let x = 7 + (k * 53) % 500;
            let y = 5 + (k * 31) % 500;
            data[y * 512 + x] = 60000;
        }
        let dirty = measure(&f).unwrap();
        assert!(
            (dirty.hfd - clean.hfd).abs() < 0.2 * clean.hfd,
            "hot sites moved the measurement from {:.3} to {:.3}",
            clean.hfd,
            dirty.hfd
        );
    }

    #[test]
    fn arcseconds_need_both_the_pitch_and_the_focal_length() {
        let m = StarMetrics {
            count: 100,
            hfd: 2.0,
            eccentricity: 0.1,
            angle_deg: 0.0,
        };
        // 4.63 um at 337 mm is 2.834 arcsec per pixel.
        let a = m.hfd_arcsec(Some(4.63), Some(337.0)).unwrap();
        assert!((a - 5.668).abs() < 0.01, "{a}");
        assert!(m.hfd_arcsec(None, Some(337.0)).is_none());
        assert!(m.hfd_arcsec(Some(4.63), None).is_none());
    }
}
