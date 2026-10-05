//! Lateral chromatic aberration: measuring it, and expressing it as geometry.
//!
//! A lens focuses different wavelengths at slightly different magnifications, so
//! the red and blue images are very slightly larger or smaller than the green
//! one. The effect is zero on the optical axis and grows with distance from it,
//! which is why colour fringes appear at the corners of a frame and not in the
//! middle.
//!
//! This is a *geometric* defect, and that makes it a natural fit for this
//! pipeline rather than an afterthought. Because the merge deposits each sensor
//! sample individually and already knows which colour every sample is, the
//! correction is one extra transform applied to a sample's position before it
//! is deposited. No resampling, no demosaic, no separate correction pass, and
//! nothing to interpolate â€” the sample simply lands where the lens should have
//! put it.
//!
//! What is measured here is the *magnification* difference between channels.
//! The constant offset between the red, green and blue sample lattices of a
//! Bayer mosaic is not aberration and is not corrected here: the merge already
//! handles it exactly, by depositing every sample at its own true sensor
//! coordinate.

use serde::{Deserialize, Serialize};

use sr_core::frame::GuideImage;
use sr_core::math::{cholesky_solve, mad_sigma, median, tukey_weight};
use sr_core::plane::Plane;

use crate::correlate::CorrelatorCache;
use crate::pyramid::{extract_patch, probe_grid};

/// Measured lateral chromatic aberration for one frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChromaticAberration {
    /// Magnification of each channel relative to green. Green is 1.0 by
    /// definition; a value of 1.0004 means that channel's image is 0.04%
    /// larger.
    pub magnification: [f32; 3],
    /// Residual channel offset at the frame centre, in sensor pixels, after the
    /// known Bayer lattice offset has been removed. Decentring rather than
    /// aberration; reported, not corrected.
    pub centre_offset: [(f32, f32); 3],
    /// Displacement the magnification produces at the far corner of the frame,
    /// in sensor pixels. This is the number that corresponds to visible
    /// fringing.
    pub corner_shift: [f32; 3],
    /// Probes that produced a usable match, per channel.
    pub probes: [usize; 3],
    /// Robust residual of the fit, in sensor pixels.
    pub residual: [f32; 3],
    /// The fitted radial model, in sensor coordinates. This is what the merge
    /// applies; `magnification` is its linear part, reported for readability.
    pub radial: sr_core::geometry::RadialChroma,
}

impl ChromaticAberration {
    pub fn none() -> Self {
        Self {
            magnification: [1.0, 1.0, 1.0],
            centre_offset: [(0.0, 0.0); 3],
            corner_shift: [0.0; 3],
            probes: [0; 3],
            residual: [0.0; 3],
            radial: sr_core::geometry::RadialChroma::identity(),
        }
    }

    /// Whether the measurement found anything worth correcting.
    ///
    /// The threshold is a third of a sensor pixel at the corner: below that the
    /// correction is smaller than the registration's own uncertainty and is not
    /// worth claiming.
    pub fn is_significant(&self) -> bool {
        self.corner_shift.iter().any(|&s| s > 0.33)
            && self
                .probes
                .iter()
                .enumerate()
                .all(|(c, &p)| c == 1 || p >= 12)
    }

    /// The correction to apply during the merge, in sensor coordinates.
    pub fn correction(&self) -> sr_core::geometry::RadialChroma {
        self.radial
    }

    pub fn describe(&self) -> String {
        let names = ["R", "G", "B"];
        let mut s = String::from("Lateral chromatic aberration (relative to green):\n");
        for c in [0usize, 2] {
            s.push_str(&format!(
                "  {}: magnification {:+.5}% at the corner, {:.2} px  \
                 (from {} probes, residual {:.3} px)\n",
                names[c],
                (self.magnification[c] - 1.0) * 100.0,
                self.corner_shift[c],
                self.probes[c],
                self.residual[c]
            ));
        }
        if self.is_significant() {
            s.push_str("  Correcting this is worthwhile.\n");
        } else {
            s.push_str("  Below the threshold worth correcting.\n");
        }
        s
    }
}

/// Offset of each channel's sample lattice from the green lattice, in sensor
/// pixels, for an RGGB mosaic.
///
/// Red sits at the top-left of every 2x2 cell and blue at the bottom-right,
/// while the two greens average to the cell centre. That is geometry, not
/// aberration, and it must be removed before a channel offset is attributed to
/// the lens.
fn bayer_lattice_offset(guide_is_rggb: bool) -> [(f32, f32); 3] {
    if guide_is_rggb {
        [(-0.5, -0.5), (0.0, 0.0), (0.5, 0.5)]
    } else {
        [(0.0, 0.0); 3]
    }
}

/// Sensor-coordinate offsets from the optical centre, measured displacement,
/// and correlation weight. Fit constant displacement as a nuisance term: Bayer
/// lattice offsets and decentring are not radial magnification. Projecting them
/// onto radius before fitting aliases them into the lens model when probes are
/// unevenly distributed around the optical axis.
fn fit_radial_displacements(obs: &[[f32; 5]], radius: f32) -> Option<([f32; 4], f32)> {
    if obs.len() < 8 {
        return None;
    }
    let norm = radius.max(1.0) as f64;
    let mut weights: Vec<f64> = obs.iter().map(|o| o[4].max(1e-3) as f64).collect();
    let mut solution = [0.0f64; 4];
    let mut residuals = Vec::new();
    for pass in 0..6 {
        let mut normal = [0.0f64; 16];
        let mut rhs = [0.0f64; 4];
        for (o, &weight) in obs.iter().zip(&weights) {
            let (x, y) = (o[0] as f64 / norm, o[1] as f64 / norm);
            let r2 = x * x + y * y;
            for (basis, value) in [
                ([1.0, 0.0, -x, -x * r2], o[2]),
                ([0.0, 1.0, -y, -y * r2], o[3]),
            ] {
                for i in 0..4 {
                    rhs[i] += weight * basis[i] * value as f64;
                    for j in 0..4 {
                        normal[i * 4 + j] += weight * basis[i] * basis[j];
                    }
                }
            }
        }
        if !cholesky_solve(&mut normal, &mut rhs, 4) {
            return None;
        }
        solution = rhs;
        residuals = obs
            .iter()
            .map(|o| {
                let (x, y) = (o[0] as f64 / norm, o[1] as f64 / norm);
                let k = solution[2] + solution[3] * (x * x + y * y);
                ((o[2] as f64 - solution[0] + k * x).hypot(o[3] as f64 - solution[1] + k * y))
                    as f32
            })
            .collect();
        if pass < 5 {
            let spread = mad_sigma(&residuals).max(median(&residuals)).max(1e-4);
            for ((w, o), &r) in weights.iter_mut().zip(obs).zip(&residuals) {
                *w = o[4].max(1e-3) as f64 * tukey_weight(r, 4.0 * spread) as f64;
            }
        }
    }
    if solution.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some((
        [
            solution[0] as f32,
            solution[1] as f32,
            (solution[2] / norm) as f32,
            (solution[3] / norm) as f32,
        ],
        median(&residuals),
    ))
}

/// Estimate lateral chromatic aberration from one frame's guide image.
///
/// Correlates the red and blue planes against green on a grid of patches and
/// fits radial magnification and constant displacement separately. Patches without enough
/// texture â€” sky, for instance â€” produce no match and are dropped, which is why
/// the probe counts are reported alongside the result.
pub fn estimate(
    guide: &GuideImage,
    patch: usize,
    probes_across: usize,
    min_peak_ratio: f32,
    rggb: bool,
) -> ChromaticAberration {
    estimate_about(guide, patch, probes_across, min_peak_ratio, rggb, None)
}

/// As [`estimate`], with an explicit optical centre in the guide's own pixels.
///
/// Lateral aberration is radial *about the optical axis*, so measuring it on a
/// crop requires knowing where that axis is. Assuming the image centre is right
/// for a full frame and wrong for anything else â€” on a corner crop the true
/// centre lies outside the image entirely, and a fit that assumes otherwise
/// reports a magnification that means nothing.
pub fn estimate_about(
    guide: &GuideImage,
    patch: usize,
    probes_across: usize,
    min_peak_ratio: f32,
    rggb: bool,
    centre_override: Option<(f32, f32)>,
) -> ChromaticAberration {
    let mut out = ChromaticAberration::none();
    let green = guide.g.clone();
    let (gw, gh) = (green.width, green.height);
    let patch = patch.min(gw / 4).min(gh / 4).max(16) & !1;

    let sites = probe_grid(gw, gh, patch, probes_across);
    let mut cache = CorrelatorCache::new();
    let lattice = bayer_lattice_offset(rggb);

    // Guide pixels are two sensor pixels.
    let centre_guide = centre_override.unwrap_or((gw as f32 * 0.5, gh as f32 * 0.5));
    // Normalising radius: the furthest corner from the optical centre, so that
    // the quadratic term always means "the extra magnification out there".
    let corner_radius_sensor = {
        let mut r: f32 = 0.0;
        for &(x, y) in &[
            (0.0f32, 0.0f32),
            (gw as f32, 0.0),
            (0.0, gh as f32),
            (gw as f32, gh as f32),
        ] {
            let d = ((x - centre_guide.0).powi(2) + (y - centre_guide.1).powi(2)).sqrt();
            r = r.max(d);
        }
        (r * 2.0).max(1.0)
    };

    let mut buf_g = vec![0.0f32; patch * patch];
    let mut buf_c = vec![0.0f32; patch * patch];

    // Sensor-coordinate geometry: the guide is half resolution.
    let centre_sensor = (centre_guide.0 * 2.0, centre_guide.1 * 2.0);
    out.radial = sr_core::geometry::RadialChroma {
        centre: centre_sensor,
        norm: corner_radius_sensor,
        coeff: [[0.0; 2]; 3],
    };

    for c in [0usize, 2] {
        let plane: &Plane<f32> = guide.channel(c);
        // Keep both components until fitting: a constant Bayer displacement
        // must not be projected into a radial lens distortion.
        let mut obs: Vec<[f32; 5]> = Vec::with_capacity(sites.len());

        for &(px, py) in &sites {
            extract_patch(&green, px, py, patch, &mut buf_g);
            extract_patch(plane, px, py, patch, &mut buf_c);
            let correlator = cache.get(patch);
            let Some(sh) = correlator.shift(&buf_g, &buf_c) else {
                continue;
            };
            if sh.peak_ratio < min_peak_ratio {
                continue;
            }
            // Channel misregistration is a fraction of a pixel; anything larger
            // is a failed match on repetitive content.
            if sh.magnitude() > 3.0 {
                continue;
            }
            let weight = (sh.peak_ratio - 1.0).clamp(0.0, 4.0);

            // Radial component, in sensor units.
            let rx = (px - centre_guide.0) * 2.0;
            let ry = (py - centre_guide.1) * 2.0;
            let r = (rx * rx + ry * ry).sqrt();
            if r < corner_radius_sensor * 0.15 {
                // Too close to the axis for the radial direction to be
                // meaningful, and it carries almost no information about
                // magnification anyway.
                continue;
            }
            // The displacement carries this channel's position onto green's, so
            // a channel imaged too large gives a negative radial component.
            obs.push([rx, ry, sh.dx * 2.0, sh.dy * 2.0, weight]);
        }

        let Some(([tx, ty, k1, k2], residual)) =
            fit_radial_displacements(&obs, corner_radius_sensor)
        else {
            continue;
        };
        out.probes[c] = obs.len();
        out.radial.coeff[c] = [k1, k2];
        let mag_corner = 1.0 + k1 + k2;
        out.magnification[c] = mag_corner;
        out.corner_shift[c] = (mag_corner - 1.0).abs() * corner_radius_sensor;
        out.residual[c] = residual;
        out.centre_offset[c] = (tx - lattice[c].0, ty - lattice[c].1);
    }

    out
}

/// Combine per-frame estimates into one for the burst.
///
/// The aberration is a property of the lens, so every frame measures the same
/// quantity and a median across frames is both more accurate than any single
/// frame and immune to one frame's bad probe.
pub fn combine(estimates: &[ChromaticAberration]) -> ChromaticAberration {
    if estimates.is_empty() {
        return ChromaticAberration::none();
    }
    let mut out = ChromaticAberration::none();
    for c in 0..3 {
        let mags: Vec<f32> = estimates
            .iter()
            .filter(|e| e.probes[c] >= 8)
            .map(|e| e.magnification[c])
            .collect();
        if mags.is_empty() {
            continue;
        }
        out.magnification[c] = median(&mags);
        out.probes[c] = estimates.iter().map(|e| e.probes[c]).sum::<usize>() / estimates.len();
        out.residual[c] = median(&estimates.iter().map(|e| e.residual[c]).collect::<Vec<_>>());
        let ox: Vec<f32> = estimates.iter().map(|e| e.centre_offset[c].0).collect();
        let oy: Vec<f32> = estimates.iter().map(|e| e.centre_offset[c].1).collect();
        out.centre_offset[c] = (median(&ox), median(&oy));
        let cs: Vec<f32> = estimates.iter().map(|e| e.corner_shift[c]).collect();
        out.corner_shift[c] = median(&cs);

        let k1: Vec<f32> = estimates
            .iter()
            .filter(|e| e.probes[c] >= 8)
            .map(|e| e.radial.coeff[c][0])
            .collect();
        let k2: Vec<f32> = estimates
            .iter()
            .filter(|e| e.probes[c] >= 8)
            .map(|e| e.radial.coeff[c][1])
            .collect();
        if !k1.is_empty() {
            out.radial.coeff[c] = [median(&k1), median(&k2)];
        }
    }
    // Geometry is shared; take it from the first estimate that has any.
    if let Some(e) = estimates.iter().find(|e| e.radial.norm > 1.0) {
        out.radial.centre = e.radial.centre;
        out.radial.norm = e.radial.norm;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uneven_probe_coverage_does_not_turn_lattice_offsets_into_aberration() {
        for (k1, k2) in [(0.0, 0.0), (0.0004, 0.0008)] {
            let mut obs = Vec::new();
            // Texture exists only in one quadrant. The constant lattice shift
            // then has a strong radial projection, despite not being a lens
            // magnification. Also recover a real radial distortion alongside it.
            for y in [80.0f32, 140.0, 220.0, 310.0] {
                for x in [70.0f32, 160.0, 250.0, 350.0] {
                    let k = k1 + k2 * (x * x + y * y) / (500.0 * 500.0);
                    obs.push([x, y, -0.5 - k * x, -0.5 - k * y, 1.0]);
                }
            }
            let (fit, residual) = fit_radial_displacements(&obs, 500.0).unwrap();
            assert!((fit[0] + 0.5).abs() < 1e-5 && (fit[1] + 0.5).abs() < 1e-5);
            assert!(
                (fit[2] - k1).abs() < 1e-7 && (fit[3] - k2).abs() < 1e-7,
                "lattice offset leaked into radial coefficients: {fit:?}"
            );
            assert!(residual < 1e-5);
        }
        assert!(
            fit_radial_displacements(&[[100.0, 0.0, 0.0, 0.0, 1.0]; 16], 500.0).is_none(),
            "a single radius cannot identify two radial terms"
        );
    }

    /// A guide image whose red and blue planes are magnified copies of green,
    /// which is exactly what lateral aberration produces.
    fn synthetic_guide(n: usize, mag_r: f32, mag_b: f32) -> GuideImage {
        let mut seed = 0x9E37u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let comps: Vec<(f32, f32, f32, f32)> = (0..96)
            .map(|i| {
                let octave = (i % 6) as f32;
                let k = 0.03 * 2.0f32.powf(octave);
                let ang = next() * std::f32::consts::TAU;
                (k * ang.cos(), k * ang.sin(), next(), 1.0 / (1.0 + octave))
            })
            .collect();
        let scene = |x: f32, y: f32| {
            let mut acc = 0.0f32;
            for &(kx, ky, ph, amp) in &comps {
                acc += 0.06 * amp * (kx * x + ky * y + ph * std::f32::consts::TAU).sin();
            }
            0.5 + acc
        };
        let c = n as f32 * 0.5;
        let mut planes = Vec::new();
        for m in [mag_r, 1.0, mag_b] {
            let mut p = Plane::<f32>::new(n, n);
            for y in 0..n {
                for x in 0..n {
                    // This channel images the scene magnified by `m`.
                    let sx = c + (x as f32 - c) / m;
                    let sy = c + (y as f32 - c) / m;
                    p.data[y * n + x] = scene(sx, sy);
                }
            }
            planes.push(p);
        }
        GuideImage {
            width: n,
            height: n,
            r: planes[0].clone(),
            g: planes[1].clone(),
            b: planes[2].clone(),
        }
    }

    #[test]
    fn recovers_a_planted_magnification() {
        // 0.1% magnification: about 2 sensor pixels at the corner of a frame
        // this size, which is a visible fringe.
        let guide = synthetic_guide(512, 1.001, 0.9992);
        let ca = estimate(&guide, 64, 10, 1.10, true);
        assert!(
            ca.probes[0] > 12 && ca.probes[2] > 12,
            "probes {:?}",
            ca.probes
        );
        assert!(
            (ca.magnification[0] - 1.001).abs() < 1.5e-4,
            "red magnification {}",
            ca.magnification[0]
        );
        assert!(
            (ca.magnification[2] - 0.9992).abs() < 1.5e-4,
            "blue magnification {}",
            ca.magnification[2]
        );
        assert_eq!(ca.magnification[1], 1.0, "green is the reference");
        assert!(ca.is_significant(), "{}", ca.describe());
    }

    #[test]
    fn reports_nothing_when_channels_agree() {
        let guide = synthetic_guide(512, 1.0, 1.0);
        let ca = estimate(&guide, 64, 10, 1.10, true);
        assert!(
            ca.corner_shift[0] < 0.33,
            "corner shift {}",
            ca.corner_shift[0]
        );
        assert!(
            ca.corner_shift[2] < 0.33,
            "corner shift {}",
            ca.corner_shift[2]
        );
        assert!(!ca.is_significant(), "{}", ca.describe());
    }

    #[test]
    fn correction_undoes_the_magnification() {
        let guide = synthetic_guide(512, 1.001, 0.9992);
        let ca = estimate(&guide, 64, 10, 1.10, true);
        let rc = ca.correction();
        // Sensor coordinates: the guide is half resolution, so the frame is
        // 1024 across and its centre is at 512.
        let centre = rc.centre;
        let p = (centre.0 + 400.0, centre.1);
        let (cx, _) = rc.apply(0, p.0, p.1);
        let corrected_radius = cx - centre.0;
        // Red was imaged 0.1% too large, so its samples must come back in.
        assert!(
            corrected_radius < 400.0 && corrected_radius > 400.0 * 0.998,
            "corrected radius {corrected_radius}"
        );
        // Green is untouched.
        assert_eq!(rc.apply(1, p.0, p.1), p);
    }

    #[test]
    fn radial_fit_recovers_a_quadratic_term() {
        // A magnification that grows faster than linearly with radius: the
        // linear-only model cannot represent this, and leaves a residual at the
        // corner that is exactly the fringing we are trying to remove.
        let n = 512;
        let mut seed = 0x5150u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let comps: Vec<(f32, f32, f32, f32)> = (0..96)
            .map(|i| {
                let octave = (i % 6) as f32;
                let k = 0.03 * 2.0f32.powf(octave);
                let ang = next() * std::f32::consts::TAU;
                (k * ang.cos(), k * ang.sin(), next(), 1.0 / (1.0 + octave))
            })
            .collect();
        let scene = |x: f32, y: f32| {
            let mut acc = 0.0f32;
            for &(kx, ky, ph, amp) in &comps {
                acc += 0.06 * amp * (kx * x + ky * y + ph * std::f32::consts::TAU).sin();
            }
            0.5 + acc
        };
        let c = n as f32 * 0.5;
        let rmax = (c * c + c * c).sqrt();
        let (k1, k2) = (0.0004f32, 0.0008f32);
        let mut planes = Vec::new();
        for chan in 0..3 {
            let mut p = Plane::<f32>::new(n, n);
            for y in 0..n {
                for x in 0..n {
                    let (dx, dy) = (x as f32 - c, y as f32 - c);
                    let r = (dx * dx + dy * dy).sqrt();
                    let m = if chan == 0 {
                        1.0 + k1 + k2 * (r / rmax) * (r / rmax)
                    } else {
                        1.0
                    };
                    p.data[y * n + x] = scene(c + dx / m, c + dy / m);
                }
            }
            planes.push(p);
        }
        let guide = GuideImage {
            width: n,
            height: n,
            r: planes[0].clone(),
            g: planes[1].clone(),
            b: planes[2].clone(),
        };
        let ca = estimate(&guide, 64, 12, 1.10, true);
        let got = ca.radial.coeff[0];
        assert!(
            (got[1] - k2).abs() < 4e-4,
            "quadratic term {} vs planted {k2} (linear {} vs {k1})",
            got[1],
            got[0]
        );
        // The corner magnification is what matters, and both terms feed it.
        let corner = ca.radial.magnification_at_corner(0);
        assert!(
            (corner - (1.0 + k1 + k2)).abs() < 4e-4,
            "corner magnification {corner} vs {}",
            1.0 + k1 + k2
        );
    }

    #[test]
    fn a_featureless_guide_yields_no_estimate() {
        let flat = Plane::filled(256, 256, 0.5);
        let guide = GuideImage {
            width: 256,
            height: 256,
            r: flat.clone(),
            g: flat.clone(),
            b: flat,
        };
        let ca = estimate(&guide, 64, 8, 1.10, true);
        assert!(!ca.is_significant());
        assert_eq!(ca.magnification, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn combining_frames_takes_the_median() {
        let mut a = ChromaticAberration::none();
        let mut b = ChromaticAberration::none();
        let mut c = ChromaticAberration::none();
        for (e, m) in [(&mut a, 1.0010f32), (&mut b, 1.0012), (&mut c, 1.0050)] {
            e.magnification[0] = m;
            e.probes[0] = 20;
            e.probes[2] = 20;
        }
        let combined = combine(&[a, b, c]);
        // The outlier frame does not move the answer.
        assert!((combined.magnification[0] - 1.0012).abs() < 1e-6);
    }
}
