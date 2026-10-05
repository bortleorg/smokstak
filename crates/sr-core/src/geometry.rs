//! Registration geometry.
//!
//! Convention used everywhere in this workspace: a frame's `WarpField` maps
//! **that frame's own sensor coordinates into reference-frame coordinates**.
//! Reconstruction therefore only ever pushes samples forward; it never needs a
//! resampled intermediate image.
//!
//! Coordinates are pixel-centre based: integer `(x, y)` is the centre of the
//! sample at column `x`, row `y`.

use serde::{Deserialize, Serialize};

use crate::plane::Plane;

/// 2x3 affine: `x' = m[0]x + m[1]y + m[2]`, `y' = m[3]x + m[4]y + m[5]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlobalTransform {
    pub m: [f32; 6],
}

impl Default for GlobalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl GlobalTransform {
    pub const IDENTITY: GlobalTransform = GlobalTransform {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };

    pub fn translation(dx: f32, dy: f32) -> Self {
        GlobalTransform {
            m: [1.0, 0.0, dx, 0.0, 1.0, dy],
        }
    }

    /// Rotation by `theta` (radians) and uniform scale about the origin, plus
    /// translation. This is the "similarity" model of the transform hierarchy.
    pub fn similarity(theta: f32, scale: f32, dx: f32, dy: f32) -> Self {
        let (s, c) = theta.sin_cos();
        GlobalTransform {
            m: [scale * c, -scale * s, dx, scale * s, scale * c, dy],
        }
    }

    #[inline]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.m[0] * x + self.m[1] * y + self.m[2],
            self.m[3] * x + self.m[4] * y + self.m[5],
        )
    }

    #[inline]
    pub fn linear_det(&self) -> f32 {
        self.m[0] * self.m[4] - self.m[1] * self.m[3]
    }

    pub fn inverse(&self) -> Option<GlobalTransform> {
        let det = self.linear_det();
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let a = self.m[4] * inv;
        let b = -self.m[1] * inv;
        let d = -self.m[3] * inv;
        let e = self.m[0] * inv;
        let c = -(a * self.m[2] + b * self.m[5]);
        let f = -(d * self.m[2] + e * self.m[5]);
        Some(GlobalTransform {
            m: [a, b, c, d, e, f],
        })
    }

    /// `self` applied after `other`.
    pub fn compose(&self, other: &GlobalTransform) -> GlobalTransform {
        let a = self.m;
        let b = other.m;
        GlobalTransform {
            m: [
                a[0] * b[0] + a[1] * b[3],
                a[0] * b[1] + a[1] * b[4],
                a[0] * b[2] + a[1] * b[5] + a[2],
                a[3] * b[0] + a[4] * b[3],
                a[3] * b[1] + a[4] * b[4],
                a[3] * b[2] + a[4] * b[5] + a[5],
            ],
        }
    }

    /// Re-express a transform estimated at one resolution so it applies at
    /// another. `factor` is `target_scale / source_scale`; registering on a
    /// half-resolution proxy and applying at full sensor resolution uses 2.0.
    pub fn rescale(&self, factor: f32) -> GlobalTransform {
        GlobalTransform {
            m: [
                self.m[0],
                self.m[1],
                self.m[2] * factor,
                self.m[3],
                self.m[4],
                self.m[5] * factor,
            ],
        }
    }

    /// Translation component.
    pub fn shift(&self) -> (f32, f32) {
        (self.m[2], self.m[5])
    }

    /// Rotation implied by the linear part, in radians.
    pub fn rotation(&self) -> f32 {
        self.m[3].atan2(self.m[0])
    }

    /// Mean linear scale implied by the linear part.
    pub fn scale(&self) -> f32 {
        let sx = (self.m[0] * self.m[0] + self.m[3] * self.m[3]).sqrt();
        let sy = (self.m[1] * self.m[1] + self.m[4] * self.m[4]).sqrt();
        0.5 * (sx + sy)
    }

    /// How far this transform moves the centre of a `w` x `h` image.
    ///
    /// This, not [`shift`](Self::shift), is what "how far did the frame move"
    /// means. The translation component is measured at the origin, so for a
    /// frame that rotated about its own centre it reports the lever arm from
    /// the corner — tens of pixels for a rotation of half a degree — while the
    /// scene barely moved at all.
    pub fn centre_offset(&self, w: f32, h: f32) -> (f32, f32) {
        let (cx, cy) = (w * 0.5, h * 0.5);
        let (tx, ty) = self.apply(cx, cy);
        (tx - cx, ty - cy)
    }

    /// Largest distance by which `self` and `other` disagree about where a
    /// point of a `w` x `h` image goes.
    ///
    /// Both are affine, so their difference is affine, and the extreme of an
    /// affine function over a rectangle is at a corner.
    pub fn max_difference(&self, other: &GlobalTransform, w: f32, h: f32) -> f32 {
        [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
            .iter()
            .map(|&(x, y)| {
                let (ax, ay) = self.apply(x, y);
                let (bx, by) = other.apply(x, y);
                ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
            })
            .fold(0.0f32, f32::max)
    }

    /// The simplest model that reproduces this transform over a `w` x `h`
    /// image to within `tol` pixels.
    ///
    /// Not the same question as which model was last fitted. Registration is
    /// iterative and each step fits the *residual*, so the final step of a
    /// well-converged rotating frame is a translation of a few hundredths of a
    /// pixel — while the transform it refined carries the rotation. Asking the
    /// accumulated transform what it does is the only version of this that a
    /// diagnostic can be read literally.
    pub fn effective_model(&self, w: f32, h: f32, tol: f32) -> TransformModel {
        let (dx, dy) = (self.m[2], self.m[5]);
        let theta = self.rotation();
        let candidates = [
            (
                TransformModel::Translation,
                GlobalTransform::translation(dx, dy),
            ),
            (
                TransformModel::Euclidean,
                GlobalTransform::similarity(theta, 1.0, dx, dy),
            ),
            (
                TransformModel::Similarity,
                GlobalTransform::similarity(theta, self.scale(), dx, dy),
            ),
        ];
        for (kind, approx) in candidates {
            if self.max_difference(&approx, w, h) <= tol {
                return kind;
            }
        }
        TransformModel::Affine
    }

    /// How far this transform displaces a point in the worst corner of a
    /// `w` x `h` image. Used to size search windows and tile halos.
    pub fn max_displacement(&self, w: f32, h: f32) -> f32 {
        let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)];
        corners
            .iter()
            .map(|&(x, y)| {
                let (tx, ty) = self.apply(x, y);
                ((tx - x).powi(2) + (ty - y).powi(2)).sqrt()
            })
            .fold(0.0f32, f32::max)
    }
}

/// Which affine sub-model was actually fitted. Recorded so that diagnostics can
/// show we did not hand a telephoto burst more projective freedom than its
/// residuals justified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformModel {
    Translation,
    Euclidean,
    Similarity,
    Affine,
}

impl TransformModel {
    pub fn dof(self) -> usize {
        match self {
            TransformModel::Translation => 2,
            TransformModel::Euclidean => 3,
            TransformModel::Similarity => 4,
            TransformModel::Affine => 6,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TransformModel::Translation => "translation",
            TransformModel::Euclidean => "euclidean",
            TransformModel::Similarity => "similarity",
            TransformModel::Affine => "affine",
        }
    }
}

/// A smooth residual displacement field defined on a regular grid **in
/// reference-frame coordinates**, evaluated after the global transform.
///
/// Storing it as a coarse mesh rather than a dense raster is deliberate: it
/// keeps the field smooth by construction and keeps memory flat in frame count.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeformationField {
    /// Reference coordinate of grid node (0, 0).
    pub origin: (f32, f32),
    /// Node spacing in reference pixels.
    pub spacing: f32,
    pub grid_w: usize,
    pub grid_h: usize,
    /// Displacement per node, added to the globally transformed position.
    pub u: Vec<[f32; 2]>,
    /// Per-node confidence in `[0, 1]`.
    pub conf: Vec<f32>,
}

impl DeformationField {
    pub fn zeros(origin: (f32, f32), spacing: f32, grid_w: usize, grid_h: usize) -> Self {
        Self {
            origin,
            spacing,
            grid_w,
            grid_h,
            u: vec![[0.0, 0.0]; grid_w * grid_h],
            conf: vec![0.0; grid_w * grid_h],
        }
    }

    #[inline]
    pub fn node(&self, gx: usize, gy: usize) -> [f32; 2] {
        self.u[gy * self.grid_w + gx]
    }

    /// Bilinear evaluation at a reference-frame position, clamped at the edges.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> (f32, f32) {
        if self.grid_w == 0 || self.grid_h == 0 {
            return (0.0, 0.0);
        }
        let gx = (x - self.origin.0) / self.spacing;
        let gy = (y - self.origin.1) / self.spacing;
        let gx = gx.clamp(0.0, (self.grid_w - 1) as f32);
        let gy = gy.clamp(0.0, (self.grid_h - 1) as f32);
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(self.grid_w - 1);
        let y1 = (y0 + 1).min(self.grid_h - 1);
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let w = self.grid_w;
        let p00 = self.u[y0 * w + x0];
        let p10 = self.u[y0 * w + x1];
        let p01 = self.u[y1 * w + x0];
        let p11 = self.u[y1 * w + x1];
        let ax = p00[0] + (p10[0] - p00[0]) * fx;
        let bx = p01[0] + (p11[0] - p01[0]) * fx;
        let ay = p00[1] + (p10[1] - p00[1]) * fx;
        let by = p01[1] + (p11[1] - p01[1]) * fx;
        (ax + (bx - ax) * fy, ay + (by - ay) * fy)
    }

    #[inline]
    pub fn sample_conf(&self, x: f32, y: f32) -> f32 {
        if self.grid_w == 0 || self.grid_h == 0 {
            return 0.0;
        }
        let gx = ((x - self.origin.0) / self.spacing).clamp(0.0, (self.grid_w - 1) as f32);
        let gy = ((y - self.origin.1) / self.spacing).clamp(0.0, (self.grid_h - 1) as f32);
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(self.grid_w - 1);
        let y1 = (y0 + 1).min(self.grid_h - 1);
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let w = self.grid_w;
        let a = self.conf[y0 * w + x0] * (1.0 - fx) + self.conf[y0 * w + x1] * fx;
        let b = self.conf[y1 * w + x0] * (1.0 - fx) + self.conf[y1 * w + x1] * fx;
        a * (1.0 - fy) + b * fy
    }

    pub fn max_magnitude(&self) -> f32 {
        self.u
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1]).sqrt())
            .fold(0.0f32, f32::max)
    }

    pub fn mean_magnitude(&self) -> f32 {
        if self.u.is_empty() {
            return 0.0;
        }
        let s: f32 = self
            .u
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1]).sqrt())
            .sum();
        s / self.u.len() as f32
    }

    /// Rescale a field estimated on a proxy so it applies at another resolution.
    pub fn rescale(&self, factor: f32) -> DeformationField {
        DeformationField {
            origin: (self.origin.0 * factor, self.origin.1 * factor),
            spacing: self.spacing * factor,
            grid_w: self.grid_w,
            grid_h: self.grid_h,
            u: self
                .u
                .iter()
                .map(|d| [d[0] * factor, d[1] * factor])
                .collect(),
            conf: self.conf.clone(),
        }
    }

    pub fn magnitude_plane(&self) -> Plane<f32> {
        Plane::from_vec(
            self.grid_w,
            self.grid_h,
            self.u
                .iter()
                .map(|d| (d[0] * d[0] + d[1] * d[1]).sqrt())
                .collect(),
        )
    }

    pub fn confidence_plane(&self) -> Plane<f32> {
        Plane::from_vec(self.grid_w, self.grid_h, self.conf.clone())
    }
}

/// Complete geometric model for one frame: rigid-ish global part plus optional
/// smooth local refinement (atmospheric shimmer, parallax, rolling deformation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WarpField {
    pub global: GlobalTransform,
    pub local: Option<DeformationField>,
}

impl WarpField {
    pub fn identity() -> Self {
        Self {
            global: GlobalTransform::IDENTITY,
            local: None,
        }
    }

    pub fn global_only(global: GlobalTransform) -> Self {
        Self {
            global,
            local: None,
        }
    }

    /// Map a sample position from this frame's sensor grid into reference
    /// coordinates.
    #[inline]
    pub fn map(&self, x: f32, y: f32) -> (f32, f32) {
        let (gx, gy) = self.global.apply(x, y);
        match &self.local {
            Some(d) => {
                let (ux, uy) = d.sample(gx, gy);
                (gx + ux, gy + uy)
            }
            None => (gx, gy),
        }
    }

    /// Map a reference position back onto this frame's sensor grid.
    ///
    /// The local field is defined in reference coordinates, so the inverse is
    /// exact for the global part and first-order for the local part: subtract
    /// the displacement sampled at the reference position, then apply the
    /// global inverse. That approximation is good to second order in the
    /// field's spatial derivative, which the regulariser keeps small by
    /// construction. Used for gathering, never for depositing samples.
    #[inline]
    pub fn inverse_map(&self, rx: f32, ry: f32) -> Option<(f32, f32)> {
        let (qx, qy) = match &self.local {
            Some(d) => {
                let (ux, uy) = d.sample(rx, ry);
                (rx - ux, ry - uy)
            }
            None => (rx, ry),
        };
        self.global.inverse().map(|inv| inv.apply(qx, qy))
    }

    /// Confidence of the local model at a reference position; 1.0 when there is
    /// no local model to doubt.
    #[inline]
    pub fn local_confidence_at_ref(&self, rx: f32, ry: f32) -> f32 {
        match &self.local {
            Some(d) => d.sample_conf(rx, ry),
            None => 1.0,
        }
    }

    pub fn rescale(&self, factor: f32) -> WarpField {
        WarpField {
            global: self.global.rescale(factor),
            local: self.local.as_ref().map(|d| d.rescale(factor)),
        }
    }

    /// Upper bound on local displacement, for halo sizing.
    pub fn max_local(&self) -> f32 {
        self.local
            .as_ref()
            .map(|d| d.max_magnitude())
            .unwrap_or(0.0)
    }
}

/// Per-channel radial geometry, for lateral chromatic aberration.
///
/// The magnification difference between colour channels is not constant across
/// the frame: it grows from zero on the optical axis and, on a real lens, not
/// linearly. A single scale factor — an affine transform — captures only the
/// first term and leaves a systematic residual at the corners, which is exactly
/// where the fringing that motivated the correction lives. So the model is an
/// even polynomial in radius, which is what the physics of a rotationally
/// symmetric lens allows.
///
/// `coeff[c] = [k1, k2]` gives channel `c` a magnification of
/// `1 + k1 + k2 * (r / norm)^2` at radius `r` from `centre`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadialChroma {
    /// Optical centre, in sensor coordinates.
    pub centre: (f32, f32),
    /// Radius the quadratic term is normalised against, usually the frame's
    /// half-diagonal, so that `k2` is the extra magnification at the corner.
    pub norm: f32,
    pub coeff: [[f32; 2]; 3],
}

impl Default for RadialChroma {
    fn default() -> Self {
        Self::identity()
    }
}

impl RadialChroma {
    pub fn identity() -> Self {
        Self {
            centre: (0.0, 0.0),
            norm: 1.0,
            coeff: [[0.0; 2]; 3],
        }
    }

    pub fn is_identity(&self) -> bool {
        self.coeff.iter().all(|k| k[0] == 0.0 && k[1] == 0.0)
    }

    /// Map a sample of channel `c` onto the position green would have imaged
    /// the same scene point at.
    #[inline]
    pub fn apply(&self, c: usize, x: f32, y: f32) -> (f32, f32) {
        let k = self.coeff[c.min(2)];
        if k[0] == 0.0 && k[1] == 0.0 {
            return (x, y);
        }
        let dx = x - self.centre.0;
        let dy = y - self.centre.1;
        let n2 = (self.norm * self.norm).max(1e-6);
        let t = (dx * dx + dy * dy) / n2;
        let m = 1.0 + k[0] + k[1] * t;
        if !(m.is_finite() && m > 0.5) {
            return (x, y);
        }
        let s = 1.0 / m;
        (self.centre.0 + dx * s, self.centre.1 + dy * s)
    }

    /// Largest displacement this correction applies within a `w` x `h` frame.
    pub fn max_displacement(&self, w: f32, h: f32) -> f32 {
        let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)];
        let mut worst = 0.0f32;
        for c in 0..3 {
            for &(x, y) in &corners {
                let (px, py) = self.apply(c, x, y);
                worst = worst.max(((px - x).powi(2) + (py - y).powi(2)).sqrt());
            }
        }
        worst
    }

    /// Effective magnification of channel `c` at the normalising radius.
    pub fn magnification_at_corner(&self, c: usize) -> f32 {
        1.0 + self.coeff[c.min(2)][0] + self.coeff[c.min(2)][1]
    }
}

/// Axis-aligned integer rectangle, used for active areas and tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
    pub fn x1(&self) -> usize {
        self.x + self.width
    }
    pub fn y1(&self) -> usize {
        self.y + self.height
    }
    pub fn area(&self) -> usize {
        self.width * self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_inverse_round_trips() {
        let t = GlobalTransform {
            m: [1.002, 0.01, -3.5, -0.008, 0.999, 7.25],
        };
        let inv = t.inverse().unwrap();
        for &(x, y) in &[(0.0f32, 0.0f32), (100.0, 250.0), (-40.0, 900.0)] {
            let (a, b) = t.apply(x, y);
            let (rx, ry) = inv.apply(a, b);
            assert!((rx - x).abs() < 1e-3, "x {rx} vs {x}");
            assert!((ry - y).abs() < 1e-3, "y {ry} vs {y}");
        }
    }

    #[test]
    fn compose_matches_sequential_application() {
        let a = GlobalTransform::similarity(0.01, 1.001, 2.0, -1.0);
        let b = GlobalTransform::translation(5.0, 3.0);
        let ab = a.compose(&b);
        let (x, y) = (37.0, 91.0);
        let (bx, by) = b.apply(x, y);
        let expect = a.apply(bx, by);
        let got = ab.apply(x, y);
        assert!((got.0 - expect.0).abs() < 1e-3);
        assert!((got.1 - expect.1).abs() < 1e-3);
    }

    #[test]
    fn rescale_scales_translation_only() {
        let t = GlobalTransform::translation(1.5, -2.5);
        let up = t.rescale(2.0);
        assert_eq!(up.shift(), (3.0, -5.0));
        assert_eq!(up.m[0], 1.0);
    }

    #[test]
    fn radial_chroma_is_zero_on_axis_and_grows_outward() {
        let rc = RadialChroma {
            centre: (100.0, 100.0),
            norm: 100.0,
            coeff: [[0.001, 0.0005], [0.0, 0.0], [0.0, 0.0]],
        };
        // Nothing moves at the optical centre.
        let (x, y) = rc.apply(0, 100.0, 100.0);
        assert!((x - 100.0).abs() < 1e-6 && (y - 100.0).abs() < 1e-6);

        // At the normalising radius the magnification is 1 + k1 + k2.
        let (px, _) = rc.apply(0, 200.0, 100.0);
        let want = 100.0 + 100.0 / (1.0 + 0.001 + 0.0005);
        assert!((px - want).abs() < 1e-3, "{px} vs {want}");

        // Green is untouched.
        assert_eq!(rc.apply(1, 200.0, 100.0), (200.0, 100.0));

        // The displacement grows with radius.
        let near = (rc.apply(0, 150.0, 100.0).0 - 150.0).abs();
        let far = (rc.apply(0, 200.0, 100.0).0 - 200.0).abs();
        assert!(far > near * 1.5, "near {near} far {far}");
    }

    #[test]
    fn deformation_field_interpolates() {
        let mut d = DeformationField::zeros((0.0, 0.0), 10.0, 3, 3);
        d.u[0] = [0.0, 0.0];
        d.u[1] = [2.0, 0.0];
        let (ux, _) = d.sample(5.0, 0.0);
        assert!((ux - 1.0).abs() < 1e-5, "got {ux}");
    }

    #[test]
    fn the_centre_offset_is_not_the_translation_component() {
        // Half a degree about the centre of a 4144 x 2822 frame: the scene has
        // not moved, but the translation measured at the origin is enormous.
        let (w, h) = (4144.0f32, 2822.0f32);
        let theta = 0.5f32.to_radians();
        let rot = GlobalTransform::similarity(theta, 1.0, 0.0, 0.0);
        let (cx, cy) = (w * 0.5, h * 0.5);
        let (rx, ry) = rot.apply(cx, cy);
        let about_centre = GlobalTransform::similarity(theta, 1.0, cx - rx, cy - ry);

        let (ox, oy) = about_centre.centre_offset(w, h);
        assert!(ox.hypot(oy) < 1e-3, "centre moved by {}", ox.hypot(oy));
        let (tx, ty) = about_centre.shift();
        assert!(
            tx.hypot(ty) > 20.0,
            "translation at the origin was only {}",
            tx.hypot(ty)
        );
    }

    #[test]
    fn a_rotation_is_not_reported_as_a_translation() {
        // The defect this exists to prevent: a burst rotating by half a degree
        // registered correctly, and the diagnostics called it translation-only
        // because the last refinement step had nothing left to do but shift.
        let (w, h) = (2072.0f32, 1411.0f32);
        let rot = GlobalTransform::similarity(0.5f32.to_radians(), 1.0, 3.0, -2.0);
        assert_eq!(rot.effective_model(w, h, 0.05), TransformModel::Euclidean);

        let pure = GlobalTransform::translation(3.0, -2.0);
        assert_eq!(
            pure.effective_model(w, h, 0.05),
            TransformModel::Translation
        );

        // A rotation small enough to be invisible over the frame is a
        // translation, and saying so is the point of measuring in pixels.
        let tiny = GlobalTransform::similarity(1e-6, 1.0, 3.0, -2.0);
        assert_eq!(
            tiny.effective_model(w, h, 0.05),
            TransformModel::Translation
        );

        let scaled = GlobalTransform::similarity(0.5f32.to_radians(), 1.001, 3.0, -2.0);
        assert_eq!(
            scaled.effective_model(w, h, 0.05),
            TransformModel::Similarity
        );

        // Unequal axis scales are not a similarity at any tolerance this small.
        let sheared = GlobalTransform {
            m: [1.001, 0.0, 3.0, 0.0, 0.999, -2.0],
        };
        assert_eq!(sheared.effective_model(w, h, 0.05), TransformModel::Affine);
    }
}
