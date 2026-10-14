//! Procedural latent scenes with known content.
//!
//! The scene deliberately contains the structures the reconstruction is
//! supposed to be judged on: slanted edges for MTF, a radial chirp for
//! aliasing and false colour, fine bars near the sensor's Nyquist limit, and
//! broadband texture. Anything a stacker can fake on smooth gradients alone is
//! not evidence.

use rand::Rng;
use rand_pcg::Pcg64Mcg;

use sr_core::plane::Plane;

/// A point source, placed so that a metric knows where to look.
///
/// `peak` is in the same units as the rest of the scene, and the bright ones
/// are deliberately far above 1.0: after the optical blur the sensor clips
/// them, which is the one thing a photograph of a star field always contains
/// and which a scene clamped into range can never reproduce.
#[derive(Clone, Copy, Debug)]
pub struct Star {
    pub x: f32,
    pub y: f32,
    pub peak: f32,
}

/// The colour every planted star is given, as a multiple of green.
///
/// Constant with radius by construction, so a channel ratio that varies with
/// radius in the reconstruction is the reconstruction's doing and not the
/// scene's.
pub const STAR_COLOUR: [f32; 3] = [1.0, 0.9, 0.8];

/// A high-resolution latent scene in linear RGB.
#[derive(Clone, Debug)]
pub struct LatentScene {
    pub width: usize,
    pub height: usize,
    pub rgb: [Plane<f32>; 3],
    /// Low-contrast fine-texture panel, as `(x, y, w, h)` in latent pixels.
    pub texture_panel: (usize, usize, usize, usize),
    /// Slanted edges placed in the scene, as `(x0, y0, x1, y1)` in HR pixels,
    /// so the MTF measurement knows where to look.
    pub edges: Vec<(f32, f32, f32, f32)>,
    /// Point sources, brightest first. The latent grid is the output grid, so
    /// these are also where they land in a reconstruction.
    pub stars: Vec<Star>,
}

/// Signed distance from a point to an infinite line through two points.
#[inline]
fn line_distance(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt().max(1e-9);
    ((px - x0) * dy - (py - y0) * dx) / len
}

/// Build the latent scene at `size x size`, where `hr_per_lr` is how many
/// latent pixels correspond to one sensor pixel.
pub fn build(size: usize, hr_per_lr: f32, seed: u64) -> LatentScene {
    let mut rng = Pcg64Mcg::new(seed as u128 | 1);
    let w = size;
    let h = size;
    let mut r = Plane::filled(w, h, 0.12f32);
    let mut g = Plane::filled(w, h, 0.12f32);
    let mut b = Plane::filled(w, h, 0.12f32);

    // Broadband texture across many octaves: what a real subject looks like to
    // a registration algorithm.
    let comps: Vec<(f32, f32, f32, f32)> = (0..120)
        .map(|i| {
            let octave = (i % 7) as f32;
            let k = 0.004 * 2.0f32.powf(octave) / hr_per_lr.max(1.0) * hr_per_lr;
            let ang: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            (
                k * ang.cos(),
                k * ang.sin(),
                rng.gen_range(0.0..std::f32::consts::TAU),
                1.0 / (1.0 + octave),
            )
        })
        .collect();

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut acc = 0.0f32;
            for &(kx, ky, ph, amp) in &comps {
                acc += 0.035 * amp * (kx * x as f32 + ky * y as f32 + ph).sin();
            }
            r.data[i] += acc;
            g.data[i] += acc * 0.95;
            b.data[i] += acc * 0.9;
        }
    }

    // A radial chirp: spatial frequency rises toward the centre, sweeping
    // through and past the sensor's Nyquist limit. Aliasing and false colour
    // show up here first.
    let (ccx, ccy) = (w as f32 * 0.30, h as f32 * 0.30);
    let radius = size as f32 * 0.18;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - ccx;
            let dy = y as f32 - ccy;
            let d = (dx * dx + dy * dy).sqrt();
            if d > radius || d < 2.0 {
                continue;
            }
            // Angular bars, so frequency increases as radius decreases.
            let theta = dy.atan2(dx);
            let v = (theta * 24.0).sin();
            let t = 0.5 + 0.35 * v.signum() * v.abs().powf(0.4);
            let i = y * w + x;
            r.data[i] = t;
            g.data[i] = t;
            b.data[i] = t;
        }
    }

    // Slanted edges at about 5 degrees, the standard geometry for a
    // slanted-edge MTF measurement.
    let mut edges = Vec::new();
    let edge_specs = [
        (
            w as f32 * 0.70,
            h as f32 * 0.20,
            5.0f32,
            [0.75f32, 0.72, 0.68],
            [0.10f32, 0.10, 0.11],
        ),
        (
            w as f32 * 0.72,
            h as f32 * 0.70,
            95.0,
            [0.70, 0.70, 0.70],
            [0.12, 0.12, 0.12],
        ),
    ];
    for &(ex, ey, angle_deg, bright, dark) in &edge_specs {
        let a = angle_deg.to_radians();
        let (dx, dy) = (a.cos(), a.sin());
        let half = size as f32 * 0.11;
        let x0 = ex - dx * half;
        let y0 = ey - dy * half;
        let x1 = ex + dx * half;
        let y1 = ey + dy * half;
        edges.push((x0, y0, x1, y1));
        let box_half = size as f32 * 0.10;
        for y in 0..h {
            for x in 0..w {
                if (x as f32 - ex).abs() > box_half || (y as f32 - ey).abs() > box_half {
                    continue;
                }
                let s = line_distance(x as f32, y as f32, x0, y0, x1, y1);
                // A hard step; the optical PSF applied later is what band-limits
                // it, which is exactly what the MTF should measure.
                let c = if s > 0.0 { bright } else { dark };
                let i = y * w + x;
                r.data[i] = c[0];
                g.data[i] = c[1];
                b.data[i] = c[2];
            }
        }
    }

    // Fine bar groups straddling the sensor Nyquist limit: below it a good
    // reconstruction resolves them, above it nothing honest can.
    let bar_x = (w as f32 * 0.12) as usize;
    let bar_y = (h as f32 * 0.70) as usize;
    for group in 0..4 {
        let period_lr = 1.4f32 + group as f32 * 0.9; // sensor pixels per cycle
        let period = period_lr * hr_per_lr;
        let x0 = bar_x + group * (size / 22);
        for y in bar_y..(bar_y + size / 8).min(h) {
            for x in x0..(x0 + size / 28).min(w) {
                let phase = (x as f32 / period * std::f32::consts::TAU).sin();
                let v = 0.5 + 0.32 * if phase > 0.0 { 1.0 } else { -1.0 };
                let i = y * w + x;
                r.data[i] = v;
                g.data[i] = v;
                b.data[i] = v;
            }
        }
    }

    // A panel of low-contrast fine texture: granules on a roof shingle, fabric,
    // foliage at distance. This is the content an adaptive kernel is most
    // likely to mistake for a flat region and smooth away, and it is invisible
    // to a slanted-edge MTF measurement, which only ever looks at a high
    // contrast step.
    let tex_x = (w as f32 * 0.06) as usize;
    let tex_y = (h as f32 * 0.06) as usize;
    let tex_w = (w as f32 * 0.22) as usize;
    let tex_h = (h as f32 * 0.22) as usize;
    let mut tseed = seed ^ 0xA5A5_5A5A;
    let mut trnd = move || {
        tseed = tseed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((tseed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    // Generated at the sensor's own sampling pitch, then held constant across
    // the latent pixels of one sensor cell, so the texture sits right at the
    // detector's resolution limit rather than beyond it.
    let cell = hr_per_lr.max(1.0) as usize;
    let cells_w = tex_w / cell + 1;
    let cells_h = tex_h / cell + 1;
    let values: Vec<f32> = (0..cells_w * cells_h).map(|_| trnd() * 0.09).collect();
    for y in tex_y..(tex_y + tex_h).min(h) {
        for x in tex_x..(tex_x + tex_w).min(w) {
            let cx = (x - tex_x) / cell;
            let cy = (y - tex_y) / cell;
            let v = 0.34 + values[cy * cells_w + cx];
            let i = y * w + x;
            r.data[i] = v * 1.02;
            g.data[i] = v;
            b.data[i] = v * 0.97;
        }
    }

    // Saturated colour patches, to exercise the colour path and highlights.
    let patches = [
        (0.80f32, 0.85f32, [0.62f32, 0.16, 0.14]),
        (0.88, 0.85, [0.15, 0.55, 0.20]),
        (0.80, 0.93, [0.13, 0.20, 0.62]),
    ];
    for &(fx, fy, c) in &patches {
        let cx = (w as f32 * fx) as usize;
        let cy = (h as f32 * fy) as usize;
        let s = (size / 24).max(3);
        for y in cy.saturating_sub(s)..(cy + s).min(h) {
            for x in cx.saturating_sub(s)..(cx + s).min(w) {
                let i = y * w + x;
                r.data[i] = c[0];
                g.data[i] = c[1];
                b.data[i] = c[2];
            }
        }
    }

    for p in [&mut r, &mut g, &mut b] {
        for v in p.data.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
    }

    // Point sources, added after the clamp because their whole purpose is to
    // exceed it. A scene held inside the sensor range never saturates, so a
    // suite built on one cannot see anything that goes wrong where the sensor
    // runs out -- which on a real star field is the middle of every bright
    // star.
    //
    // The peaks span the interesting boundary: two that clip across several
    // pixels, one that clips at its very centre, and one that never does and
    // so serves as the control.
    let stars = star_field(size, hr_per_lr);
    let sigma = (hr_per_lr * 0.5).max(0.6);
    let reach = (sigma * 4.0).ceil() as i64;
    for st in &stars {
        let (sx, sy) = (st.x.round() as i64, st.y.round() as i64);
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let (px, py) = (sx + dx, sy + dy);
                if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
                    continue;
                }
                let ex = px as f32 - st.x;
                let ey = py as f32 - st.y;
                let a = st.peak * (-(ex * ex + ey * ey) / (2.0 * sigma * sigma)).exp();
                let i = py as usize * w + px as usize;
                r.data[i] += a * STAR_COLOUR[0];
                g.data[i] += a * STAR_COLOUR[1];
                b.data[i] += a * STAR_COLOUR[2];
            }
        }
    }

    LatentScene {
        width: w,
        height: h,
        rgb: [r, g, b],
        edges,
        texture_panel: (tex_x, tex_y, tex_w.min(w - tex_x), tex_h.min(h - tex_y)),
        stars,
    }
}

/// Where the point sources go, and how bright.
///
/// A clear band of sky on the right of the frame, away from the chirp, the
/// edges, the bar groups and the colour patches, so that a measurement made
/// around a star is measuring the star.
fn star_field(size: usize, hr_per_lr: f32) -> Vec<Star> {
    // Peak as it will be after the optical blur widens the planted Gaussian:
    // a Gaussian of sigma s convolved with one of sigma t keeps
    // s^2 / (s^2 + t^2) of its height, and the suite's blur is about
    // 0.62 sensor pixels. Stated this way the numbers below mean what they
    // say -- 4.0 is four times full scale.
    let s = (hr_per_lr * 0.5).max(0.6);
    let t = hr_per_lr * 0.62;
    let survives = s * s / (s * s + t * t);
    [(0.55f32, 4.0f32), (0.68, 1.6), (0.81, 0.9), (0.92, 0.35)]
        .iter()
        .map(|&(fx, after)| Star {
            x: size as f32 * fx,
            y: size as f32 * 0.45,
            peak: after / survives,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_has_content_at_many_scales() {
        let s = build(256, 2.0, 42);
        assert_eq!(s.rgb[0].dims(), (256, 256));
        assert_eq!(s.edges.len(), 2);
        // Detail should survive several decimations, or the coarse pyramid
        // levels of the registration would have nothing to work with.
        let mut cur = s.rgb[1].clone();
        for _ in 0..3 {
            cur = cur.blur3().downsample2();
            let (lo, hi) = cur.min_max();
            assert!(
                hi - lo > 0.05,
                "scene went flat at {}x{}",
                cur.width,
                cur.height
            );
        }
    }

    #[test]
    fn texture_panel_is_low_contrast_but_present() {
        let s = build(512, 2.0, 3);
        let (x, y, w, h) = s.texture_panel;
        assert!(w > 32 && h > 32);
        let g = &s.rgb[1];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for j in y..y + h {
            for i in x..x + w {
                let v = g.data[j * g.width + i];
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        // Real structure, but an order of magnitude below a printed edge.
        assert!(hi - lo > 0.04, "texture too faint: {}", hi - lo);
        assert!(hi - lo < 0.25, "texture is not low contrast: {}", hi - lo);
    }

    #[test]
    fn the_scene_stays_in_range_away_from_the_stars() {
        let s = build(128, 2.0, 7);
        for p in &s.rgb {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for y in 0..s.height {
                for x in 0..s.width {
                    if s.stars
                        .iter()
                        .any(|st| (st.x - x as f32).hypot(st.y - y as f32) < 12.0)
                    {
                        continue;
                    }
                    let v = p.data[y * s.width + x];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            assert!(lo >= 0.0 && hi <= 1.0, "range {lo}..{hi}");
        }
    }

    #[test]
    fn the_brightest_stars_are_above_full_scale_and_the_faintest_is_not() {
        // The suite is worthless for saturation if nothing in it saturates,
        // and worthless as a comparison if everything does.
        let s = build(256, 2.0, 11);
        assert_eq!(s.stars.len(), 4);
        let g = &s.rgb[1];
        let at = |st: &Star| g.data[st.y.round() as usize * s.width + st.x.round() as usize];
        let peaks: Vec<f32> = s.stars.iter().map(at).collect();
        assert!(
            peaks[0] > 4.0,
            "the brightest star peaks at only {}",
            peaks[0]
        );
        assert!(
            peaks[2] > 1.0,
            "the third star does not reach full scale: {}",
            peaks[2]
        );
        assert!(
            peaks[3] < 1.0,
            "the control star saturates too: {}",
            peaks[3]
        );
        for w in peaks.windows(2) {
            assert!(
                w[0] > w[1],
                "the stars are not in descending order: {peaks:?}"
            );
        }
    }

    #[test]
    fn the_stars_sit_on_empty_sky() {
        // A metric taken around a star must not be measuring the chirp or an
        // edge that happens to be underneath it.
        let s = build(256, 2.0, 5);
        let bare = build_without_stars(256, 2.0, 5);
        let g = &bare.rgb[1];
        for st in &s.stars {
            let (sx, sy) = (st.x.round() as usize, st.y.round() as usize);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for y in sy - 14..=sy + 14 {
                for x in sx - 14..=sx + 14 {
                    let v = g.data[y * bare.width + x];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            assert!(
                hi - lo < 0.12,
                "star at {sx},{sy} sits on structure of {}",
                hi - lo
            );
        }
    }

    /// The scene as it would be with the star field empty, for tests that need
    /// to know what is underneath.
    fn build_without_stars(size: usize, hr_per_lr: f32, seed: u64) -> LatentScene {
        let mut s = build(size, hr_per_lr, seed);
        let sigma = (hr_per_lr * 0.5).max(0.6);
        let reach = (sigma * 4.0).ceil() as i64;
        for st in s.stars.clone() {
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let px = st.x.round() as i64 + dx;
                    let py = st.y.round() as i64 + dy;
                    if px < 0 || py < 0 || px >= s.width as i64 || py >= s.height as i64 {
                        continue;
                    }
                    let ex = px as f32 - st.x;
                    let ey = py as f32 - st.y;
                    let a = st.peak * (-(ex * ex + ey * ey) / (2.0 * sigma * sigma)).exp();
                    let i = py as usize * s.width + px as usize;
                    for (c, p) in s.rgb.iter_mut().enumerate() {
                        p.data[i] -= a * STAR_COLOUR[c];
                    }
                }
            }
        }
        s
    }
}
