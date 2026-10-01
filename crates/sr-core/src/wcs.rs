//! Where a frame was pointed, as its capture program recorded it.
//!
//! Plate-solving software writes the solution into the FITS header: a
//! projection type, a reference pixel, the sky coordinate that pixel looks at,
//! and a matrix giving the scale and rotation. Every frame from an ASIAIR, and
//! from most other capture programs, carries one.
//!
//! This is worth having for one reason. Correlation-based registration is
//! seeded from the identity and refines: it can follow a burst that drifts by
//! tens of pixels, and it cannot follow one that flips. A German equatorial
//! mount crossing the meridian rotates the camera through 180 degrees, so
//! frames of the same target taken either side of it are related by a rotation
//! no amount of refinement will find from a standing start. On the burst this
//! was written for — 407 narrowband frames of NGC 6871 over eighteen nights —
//! that silently discarded 28% of the integration: the flipped frames
//! registered to nothing, scored a confidence of 0.003, and were weighted out
//! of the merge.
//!
//! ## What this is not
//!
//! It is not the registration. A header cannot be trusted for everything,
//! and a plate solve is a claim like any other: it can be stale,
//! it can be from a different night's session, and its own accuracy is usually
//! a pixel or so — a tenth of what the pixels themselves will give. So a
//! solution is used as a *seed* and the existing pyramid refines from there,
//! and if the seeded fit is not better than the unseeded one, the seed is
//! discarded. The pixels remain the authority.

use serde::{Deserialize, Serialize};

/// A gnomonic (TAN) plate solution, which is what every capture program writes.
///
/// SIP distortion coefficients are deliberately ignored. They correct the last
/// fraction of a pixel at the frame corners, and what this is for is landing
/// within the correlator's capture range.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Wcs {
    /// Reference pixel, zero-based, in sensor coordinates.
    pub crpix: (f64, f64),
    /// Sky coordinate of that pixel, in degrees.
    pub crval: (f64, f64),
    /// Degrees per pixel, as `[[CD1_1, CD1_2], [CD2_1, CD2_2]]`.
    pub cd: [[f64; 2]; 2],
}

impl Wcs {
    /// Pixel to sky, in degrees. `(x, y)` is zero-based sensor coordinates.
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.crpix.0;
        let dy = y - self.crpix.1;
        // Intermediate world coordinates, degrees, then radians.
        let xi = (self.cd[0][0] * dx + self.cd[0][1] * dy).to_radians();
        let eta = (self.cd[1][0] * dx + self.cd[1][1] * dy).to_radians();

        let (ra0, dec0) = (self.crval.0.to_radians(), self.crval.1.to_radians());
        let (sd, cd0) = (dec0.sin(), dec0.cos());
        let denom = cd0 - eta * sd;
        let ra = ra0 + xi.atan2(denom);
        let dec = ((sd + eta * cd0) / (1.0 + xi * xi + eta * eta).sqrt()).asin();
        (ra.to_degrees().rem_euclid(360.0), dec.to_degrees())
    }

    /// Sky to pixel, the inverse of `pixel_to_sky`.
    ///
    /// `None` when the matrix is singular or the coordinate is on the far side
    /// of the sky from the projection centre, where a gnomonic projection has
    /// no answer.
    pub fn sky_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
        let (ra, dec) = (ra_deg.to_radians(), dec_deg.to_radians());
        let (ra0, dec0) = (self.crval.0.to_radians(), self.crval.1.to_radians());
        let (sd0, cd0) = (dec0.sin(), dec0.cos());
        let (sd, cdd) = (dec.sin(), dec.cos());
        let dra = ra - ra0;
        let cosc = sd0 * sd + cd0 * cdd * dra.cos();
        if cosc <= 1e-6 {
            return None;
        }
        let xi = (cdd * dra.sin() / cosc).to_degrees();
        let eta = ((cd0 * sd - sd0 * cdd * dra.cos()) / cosc).to_degrees();

        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        if det.abs() < 1e-18 {
            return None;
        }
        let dx = (self.cd[1][1] * xi - self.cd[0][1] * eta) / det;
        let dy = (-self.cd[1][0] * xi + self.cd[0][0] * eta) / det;
        Some((dx + self.crpix.0, dy + self.crpix.1))
    }

    /// Scale in arcseconds per pixel, from the determinant of the matrix.
    pub fn scale_arcsec(&self) -> f64 {
        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        det.abs().sqrt() * 3600.0
    }

    /// Position angle of the frame, in degrees.
    ///
    /// Only meaningful as a difference between two frames, which is how the
    /// meridian flip shows up: 180 degrees apart.
    pub fn orientation_deg(&self) -> f64 {
        self.cd[0][1].atan2(self.cd[0][0]).to_degrees()
    }

    /// Whether this solution is usable at all.
    ///
    /// A header can carry the keywords and still say nothing — a zero matrix,
    /// a scale that no telescope produces. Anything that fails here is treated
    /// as a frame with no solution rather than as an error.
    pub fn is_plausible(&self) -> bool {
        let finite = self.crpix.0.is_finite()
            && self.crpix.1.is_finite()
            && self.crval.0.is_finite()
            && self.crval.1.is_finite()
            && self.cd.iter().flatten().all(|v| v.is_finite());
        if !finite {
            return false;
        }
        if self.crval.1.abs() > 90.0 {
            return false;
        }
        // A hundredth of an arcsecond per pixel is a scale no amateur
        // instrument reaches; a degree per pixel is an all-sky camera.
        let s = self.scale_arcsec();
        s > 0.01 && s < 3600.0
    }
}

/// How to map one frame's pixels onto another's, from their two solutions.
///
/// Both are evaluated on a grid rather than composed algebraically: two TAN
/// projections about different centres do not compose to an affine map at all.
/// Over a field this wide the difference reaches a few pixels at the corners,
/// which is measured and guarded by a test — and is the right amount of wrong
/// for something whose job is to land inside the correlator's capture range.
///
/// Returns the affine as `[a, b, tx, c, d, ty]` mapping target pixels to
/// reference pixels: `rx = a*x + b*y + tx`, `ry = c*x + d*y + ty`.
pub fn relative_affine(
    reference: &Wcs,
    target: &Wcs,
    width: usize,
    height: usize,
) -> Option<[f64; 6]> {
    if !reference.is_plausible() || !target.is_plausible() {
        return None;
    }
    // A 5x5 grid over the target frame, mapped through the sky into the
    // reference frame. Twenty-five points for six unknowns.
    let mut ata = [[0.0f64; 3]; 3];
    let mut atx = [0.0f64; 3];
    let mut aty = [0.0f64; 3];
    let mut n = 0;
    for gy in 0..5 {
        for gx in 0..5 {
            let x = (width as f64 - 1.0) * gx as f64 / 4.0;
            let y = (height as f64 - 1.0) * gy as f64 / 4.0;
            let (ra, dec) = target.pixel_to_sky(x, y);
            let Some((rx, ry)) = reference.sky_to_pixel(ra, dec) else {
                continue;
            };
            if !rx.is_finite() || !ry.is_finite() {
                continue;
            }
            let row = [x, y, 1.0];
            for i in 0..3 {
                for j in 0..3 {
                    ata[i][j] += row[i] * row[j];
                }
                atx[i] += row[i] * rx;
                aty[i] += row[i] * ry;
            }
            n += 1;
        }
    }
    if n < 6 {
        return None;
    }
    let sx = solve3(ata, atx)?;
    let sy = solve3(ata, aty)?;
    Some([sx[0], sx[1], sx[2], sy[0], sy[1], sy[2]])
}

/// Gaussian elimination with partial pivoting on a 3x3 system.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in col + 1..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..3 {
            let f = a[r][col] / a[col][col];
            let pivot_row = a[col];
            for (c, v) in a[r].iter_mut().enumerate().skip(col) {
                *v -= f * pivot_row[c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for j in i + 1..3 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solve of the field this was written for: 173 mm at 3.76 um, so about
    /// 4.5 arcseconds per pixel, pointed at NGC 6871.
    fn ngc6871() -> Wcs {
        Wcs {
            crpix: (3943.4, 1069.4),
            crval: (300.0823, 34.7828),
            cd: [
                [-2.03473e-5, 1.24206e-3],
                [-1.23976e-3, -1.79194e-5],
            ],
        }
    }

    #[test]
    fn pixel_and_sky_are_inverses() {
        let w = ngc6871();
        for &(x, y) in &[(0.0, 0.0), (3124.0, 2088.0), (6247.0, 4175.0), (100.5, 4000.25)] {
            let (ra, dec) = w.pixel_to_sky(x, y);
            let (bx, by) = w.sky_to_pixel(ra, dec).expect("on this side of the sky");
            assert!((bx - x).abs() < 1e-6, "x {x} came back as {bx}");
            assert!((by - y).abs() < 1e-6, "y {y} came back as {by}");
        }
    }

    #[test]
    fn the_scale_is_the_one_the_optics_give() {
        // 3.76 um pixels at 173 mm: 206265 * 3.76e-3 / 173 = 4.48 arcsec.
        let s = ngc6871().scale_arcsec();
        assert!((s - 4.48).abs() < 0.05, "scale {s}");
    }

    #[test]
    fn a_frame_against_itself_is_the_identity() {
        let w = ngc6871();
        let a = relative_affine(&w, &w, 6248, 4176).unwrap();
        assert!((a[0] - 1.0).abs() < 1e-9 && (a[4] - 1.0).abs() < 1e-9, "{a:?}");
        assert!(a[1].abs() < 1e-9 && a[3].abs() < 1e-9, "{a:?}");
        assert!(a[2].abs() < 1e-6 && a[5].abs() < 1e-6, "{a:?}");
    }

    /// The case this exists for. A German equatorial mount crossing the
    /// meridian turns the camera through 180 degrees; the solve says so, and
    /// the recovered map is a rotation about the field with the pointing
    /// difference folded in.
    #[test]
    fn a_meridian_flip_is_recovered_as_a_half_turn() {
        let a = ngc6871();
        let mut b = a;
        b.cd = [[-a.cd[0][0], -a.cd[0][1]], [-a.cd[1][0], -a.cd[1][1]]];
        let m = relative_affine(&a, &b, 6248, 4176).unwrap();
        // A half turn: the linear part is -I.
        assert!((m[0] + 1.0).abs() < 1e-6, "{m:?}");
        assert!((m[4] + 1.0).abs() < 1e-6, "{m:?}");
        assert!(m[1].abs() < 1e-6 && m[3].abs() < 1e-6, "{m:?}");
        // And it maps the reference pixel onto itself, because both solutions
        // put the same sky there.
        let (x, y) = (a.crpix.0, a.crpix.1);
        let px = m[0] * x + m[1] * y + m[2];
        let py = m[3] * x + m[4] * y + m[5];
        assert!((px - x).abs() < 0.01 && (py - y).abs() < 0.01, "{px} {py}");
    }

    #[test]
    fn a_pointing_offset_becomes_a_translation() {
        let a = ngc6871();
        let mut b = a;
        // Move the reference pixel by 300 columns' worth of sky.
        let (ra, dec) = a.pixel_to_sky(a.crpix.0 + 300.0, a.crpix.1 + 120.0);
        b.crval = (ra, dec);
        let m = relative_affine(&a, &b, 6248, 4176).unwrap();
        let px = m[0] * a.crpix.0 + m[1] * a.crpix.1 + m[2];
        let py = m[3] * a.crpix.0 + m[4] * a.crpix.1 + m[5];
        assert!((px - (a.crpix.0 + 300.0)).abs() < 3.0, "x {px}");
        assert!((py - (a.crpix.1 + 120.0)).abs() < 3.0, "y {py}");
    }

    /// How wrong the affine is, and why that is the right amount of wrong.
    ///
    /// Two gnomonic projections about different centres do not compose to an
    /// affine map: over a field this wide the difference is a few pixels at
    /// the corners. That is the whole point of a seed. It has to land inside
    /// the correlator's capture range -- tens of pixels -- and the pixels
    /// themselves then supply the other three decimal places. An exact
    /// composition would be more code saying the same thing.
    #[test]
    fn the_affine_is_a_seed_and_not_a_solution() {
        let a = ngc6871();
        let mut b = a;
        let (ra, dec) = a.pixel_to_sky(a.crpix.0 + 300.0, a.crpix.1 + 120.0);
        b.crval = (ra, dec);
        let m = relative_affine(&a, &b, 6248, 4176).unwrap();

        let mut worst = 0.0f64;
        for gy in 0..9 {
            for gx in 0..9 {
                let x = 6247.0 * gx as f64 / 8.0;
                let y = 4175.0 * gy as f64 / 8.0;
                let (ra, dec) = b.pixel_to_sky(x, y);
                let (tx, ty) = a.sky_to_pixel(ra, dec).unwrap();
                let px = m[0] * x + m[1] * y + m[2];
                let py = m[3] * x + m[4] * y + m[5];
                worst = worst.max(((px - tx).powi(2) + (py - ty).powi(2)).sqrt());
            }
        }
        assert!(worst < 10.0, "affine seed is {worst:.2} px out at the corners");
        assert!(worst > 0.01, "this test no longer measures anything");
    }

    #[test]
    fn nonsense_is_declined_rather_than_believed() {
        let mut w = ngc6871();
        w.cd = [[0.0, 0.0], [0.0, 0.0]];
        assert!(!w.is_plausible());
        assert!(relative_affine(&ngc6871(), &w, 6248, 4176).is_none());

        let mut w = ngc6871();
        w.crval.1 = 91.0;
        assert!(!w.is_plausible());
    }
}
