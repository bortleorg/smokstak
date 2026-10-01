//! Exact mono sensor-to-grid projection and a checked adapter to legacy meshes.
use crate::{DeformationField, GlobalTransform, Result, SrError, WarpField};
use serde::{Deserialize, Serialize};

/// Brown-Conrady cubic correction in normalized sensor coordinates, followed by
/// a homogeneous projection, output scaling and output translation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameProjection {
    pub center: [f64; 2],
    pub normalization_scale: f64,
    /// Radial k1, tangential p1, tangential p2.
    pub distortion: [f64; 3],
    /// Row-major 3x3 matrix.
    pub homography: [f64; 9],
    pub output_center: [f64; 2],
    pub output_scale: f64,
}

fn error(message: &str) -> SrError {
    SrError::Registration(format!("projection: {message}"))
}

fn inverse3(m: [f64; 9]) -> Option<[f64; 9]> {
    let scale = m.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let m = m.map(|v| v / scale);
    let a = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ];
    let det = m[0] * a[0] + m[1] * a[3] + m[2] * a[6];
    if !det.is_finite() || det.abs() < 1e-14 {
        return None;
    }
    // A homogeneous inverse may retain an arbitrary common scale.
    Some(a.map(|v| v / det))
}

fn homogeneous(m: &[f64; 9], x: f64, y: f64) -> Option<(f64, f64)> {
    let z = m[6] * x + m[7] * y + m[8];
    let scale = (m[6] * x).abs() + (m[7] * y).abs() + m[8].abs();
    if !z.is_finite() || z.abs() <= 1e-12 * scale.max(f64::MIN_POSITIVE) {
        return None;
    }
    let q = (
        (m[0] * x + m[1] * y + m[2]) / z,
        (m[3] * x + m[4] * y + m[5]) / z,
    );
    (q.0.is_finite() && q.1.is_finite()).then_some(q)
}

impl FrameProjection {
    pub fn validate(&self) -> Result<()> {
        if !self
            .center
            .iter()
            .chain(self.output_center.iter())
            .chain(self.distortion.iter())
            .chain(self.homography.iter())
            .all(|v| v.is_finite())
            || !self.normalization_scale.is_finite()
            || self.normalization_scale <= 0.0
            || !self.output_scale.is_finite()
            || self.output_scale <= 0.0
            || inverse3(self.homography).is_none()
        {
            return Err(error(
                "nonfinite, nonpositive scale, or singular parameters",
            ));
        }
        Ok(())
    }

    fn distortion_at(&self, x: f64, y: f64) -> ((f64, f64), [f64; 4]) {
        let [k, p1, p2] = self.distortion;
        let r2 = x * x + y * y;
        (
            (
                x + k * x * r2 + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                y + k * y * r2 + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
            ),
            [
                1.0 + k * (3.0 * x * x + y * y) + 2.0 * p1 * y + 6.0 * p2 * x,
                2.0 * k * x * y + 2.0 * p1 * x + 2.0 * p2 * y,
                2.0 * k * x * y + 2.0 * p1 * x + 2.0 * p2 * y,
                1.0 + k * (x * x + 3.0 * y * y) + 6.0 * p1 * y + 2.0 * p2 * x,
            ],
        )
    }

    /// Exact forward mapping. Invalid parameters and projective horizons fail.
    pub fn map(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.validate().ok()?;
        self.map_unchecked(x, y)
    }

    fn map_unchecked(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let (u, _) = self.distortion_at(
            (x - self.center[0]) / self.normalization_scale,
            (y - self.center[1]) / self.normalization_scale,
        );
        let q = homogeneous(&self.homography, u.0, u.1)?;
        let p = (
            q.0 * self.output_scale + self.output_center[0],
            q.1 * self.output_scale + self.output_center[1],
        );
        (p.0.is_finite() && p.1.is_finite()).then_some(p)
    }

    /// Bounded Newton inverse on the central, locally orientation-preserving
    /// distortion branch. At most 40 iterations and 12 backtracking steps;
    /// convergence requires a forward roundtrip within 1e-6 output pixels.
    /// This is independent of `WarpField`'s approximate inverse.
    pub fn inverse_map(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.validate().ok()?;
        let target = homogeneous(
            &inverse3(self.homography)?,
            (x - self.output_center[0]) / self.output_scale,
            (y - self.output_center[1]) / self.output_scale,
        )?;
        let mut p = target;
        for _ in 0..40 {
            let (q, j) = self.distortion_at(p.0, p.1);
            let det = j[0] * j[3] - j[1] * j[2];
            if !det.is_finite() || det <= 1e-14 {
                return None;
            }
            let r = (q.0 - target.0, q.1 - target.1);
            let sensor = (
                p.0 * self.normalization_scale + self.center[0],
                p.1 * self.normalization_scale + self.center[1],
            );
            let mapped = self.map_unchecked(sensor.0, sensor.1)?;
            if (mapped.0 - x).hypot(mapped.1 - y) <= 1e-6 {
                return Some(sensor);
            }
            let step = (
                (j[3] * r.0 - j[1] * r.1) / det,
                (-j[2] * r.0 + j[0] * r.1) / det,
            );
            let mut accepted = false;
            for b in 0..12 {
                let factor = 0.5f64.powi(b);
                let trial = (p.0 - step.0 * factor, p.1 - step.1 * factor);
                let (t, _) = self.distortion_at(trial.0, trial.1);
                if (t.0 - target.0).hypot(t.1 - target.1) < r.0.hypot(r.1) {
                    p = trial;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                return None;
            }
        }
        None
    }

    /// Build an affine plus bilinear residual approximation, checked against
    /// exact mapping on a quarter-cell source lattice including all edges.
    /// The measured bound is not a mathematical supremum between test points.
    /// Mesh spacing refines 64,32,16,8,4; at most one million nodes are allowed.
    /// Models crossing horizons, mesh budgets or the requested error fail.
    /// Analytic mesh values outside the sensor are limited to a one-cell halo;
    /// this API never samples, interpolates or extrapolates detector data.
    pub fn to_warp(&self, width: usize, height: usize, max_error: f64) -> Result<WarpField> {
        self.to_warp_with_node_budget(width, height, max_error, 1_000_000)
    }

    /// Same checked approximation with a caller-supplied allocation budget.
    /// Rejects zero or budgets above the global million-node limit. Each mesh
    /// uses 12 bytes per node; the limit is checked before allocating either
    /// displacement or confidence arrays, including refinement attempts.
    pub fn to_warp_with_node_budget(
        &self,
        width: usize,
        height: usize,
        max_error: f64,
        max_nodes: usize,
    ) -> Result<WarpField> {
        if !(1..=1_000_000).contains(&max_nodes) {
            return Err(error("mesh node budget must be in 1..=1000000"));
        }
        self.validate()?;
        if width == 0
            || height == 0
            || width > 1_000_000
            || height > 1_000_000
            || width.checked_mul(height).is_none()
            || !max_error.is_finite()
            || max_error <= 0.0
        {
            return Err(error("invalid dimensions or tolerance"));
        }
        // Linearize where this raster actually lives. For a cropped frame the
        // optical normalization centre may be far outside the crop; using it
        // here leaves large residuals and degrades the legacy first-order
        // inverse. Optical normalization in map_unchecked remains unchanged.
        let (cx, cy) = ((width - 1) as f64 * 0.5, (height - 1) as f64 * 0.5);
        let p = self
            .map_unchecked(cx, cy)
            .ok_or_else(|| error("center at horizon"))?;
        let xp = self
            .map_unchecked(cx + 0.5, cy)
            .ok_or_else(|| error("invalid center derivative"))?;
        let xm = self
            .map_unchecked(cx - 0.5, cy)
            .ok_or_else(|| error("invalid center derivative"))?;
        let yp = self
            .map_unchecked(cx, cy + 0.5)
            .ok_or_else(|| error("invalid center derivative"))?;
        let ym = self
            .map_unchecked(cx, cy - 0.5)
            .ok_or_else(|| error("invalid center derivative"))?;
        let (a, b, d, e) = (xp.0 - xm.0, yp.0 - ym.0, xp.1 - xm.1, yp.1 - ym.1);
        let global = GlobalTransform {
            m: [
                a as f32,
                b as f32,
                (p.0 - a * cx - b * cy) as f32,
                d as f32,
                e as f32,
                (p.1 - d * cx - e * cy) as f32,
            ],
        };
        if !global.m.iter().all(|v| v.is_finite()) {
            return Err(error("affine overflow"));
        }
        let inverse = global
            .inverse()
            .filter(|g| g.m.iter().all(|v| v.is_finite()))
            .ok_or_else(|| error("singular center derivative"))?;
        let (w, h) = ((width - 1) as f32, (height - 1) as f32);
        let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)].map(|(x, y)| global.apply(x, y));
        let minx = corners.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let miny = corners.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let maxx = corners
            .iter()
            .map(|p| p.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let maxy = corners
            .iter()
            .map(|p| p.1)
            .fold(f32::NEG_INFINITY, f32::max);
        for spacing in [64.0f32, 32.0, 16.0, 8.0, 4.0] {
            let origin = (
                (minx / spacing).floor() * spacing - spacing,
                (miny / spacing).floor() * spacing - spacing,
            );
            let gw = ((maxx - origin.0) / spacing).ceil() as usize;
            let gh = ((maxy - origin.1) / spacing).ceil() as usize;
            let gw = gw
                .checked_add(2)
                .ok_or_else(|| error("mesh size overflow"))?;
            let gh = gh
                .checked_add(2)
                .ok_or_else(|| error("mesh size overflow"))?;
            let count = gw
                .checked_mul(gh)
                .filter(|n| *n <= max_nodes)
                .ok_or_else(|| error("mesh exceeds node budget"))?;
            let mut u = Vec::new();
            u.try_reserve_exact(count)
                .map_err(|_| error("mesh allocation failed"))?;
            let halo = spacing as f64
                * ((inverse.m[0].abs() + inverse.m[1].abs())
                    .max(inverse.m[3].abs() + inverse.m[4].abs()) as f64)
                * 2.0;
            for iy in 0..gh {
                for ix in 0..gw {
                    let node = (
                        origin.0 + ix as f32 * spacing,
                        origin.1 + iy as f32 * spacing,
                    );
                    let s = inverse.apply(node.0, node.1);
                    let sensor = (
                        (s.0 as f64).clamp(-halo, w as f64 + halo),
                        (s.1 as f64).clamp(-halo, h as f64 + halo),
                    );
                    let exact = self
                        .map_unchecked(sensor.0, sensor.1)
                        .ok_or_else(|| error("mesh reaches projection horizon"))?;
                    let g = global.apply(sensor.0 as f32, sensor.1 as f32);
                    let delta = [(exact.0 - g.0 as f64) as f32, (exact.1 - g.1 as f64) as f32];
                    if !delta.iter().all(|v| v.is_finite()) {
                        return Err(error("mesh displacement overflow"));
                    }
                    u.push(delta);
                }
            }
            let warp = WarpField {
                global,
                local: Some(DeformationField {
                    origin,
                    spacing,
                    grid_w: gw,
                    grid_h: gh,
                    u,
                    conf: vec![1.0; count],
                }),
            };
            let scale = (a.abs() + b.abs()).max(d.abs() + e.abs());
            let source_step = ((spacing as f64 / (4.0 * scale.max(1.0))).floor() as usize).max(1);
            let nx = (width - 1).div_ceil(source_step) + 1;
            let ny = (height - 1).div_ceil(source_step) + 1;
            if nx.checked_mul(ny).is_none_or(|n| n > 50_000_000) {
                return Err(error("validation lattice exceeds budget"));
            }
            let mut passed = true;
            let mut horizon_sign = None;
            'check: for iy in 0..ny {
                for ix in 0..nx {
                    let x = (ix * source_step).min(width - 1) as f64;
                    let y = (iy * source_step).min(height - 1) as f64;
                    let (u, j) = self.distortion_at(
                        (x - self.center[0]) / self.normalization_scale,
                        (y - self.center[1]) / self.normalization_scale,
                    );
                    if j[0] * j[3] - j[1] * j[2] <= 1e-14 {
                        return Err(error("source distortion folds or is singular"));
                    }
                    let z =
                        self.homography[6] * u.0 + self.homography[7] * u.1 + self.homography[8];
                    let sign = z.is_sign_positive();
                    if horizon_sign.is_some_and(|s| s != sign) {
                        return Err(error("source crosses projection horizon"));
                    }
                    horizon_sign = Some(sign);
                    let exact = self
                        .map_unchecked(x, y)
                        .ok_or_else(|| error("source reaches projection horizon"))?;
                    let approx = warp.map(x as f32, y as f32);
                    if (exact.0 - approx.0 as f64).hypot(exact.1 - approx.1 as f64) > max_error {
                        passed = false;
                        break 'check;
                    }
                }
            }
            if passed {
                return Ok(warp);
            }
        }
        Err(error("mesh cannot meet requested mapping tolerance"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> FrameProjection {
        FrameProjection {
            center: [0.0; 2],
            normalization_scale: 1.0,
            distortion: [0.0; 3],
            homography: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            output_center: [0.0; 2],
            output_scale: 1.0,
        }
    }

    #[test]
    fn identity_and_similarity_roundtrip() {
        let mut p = identity();
        assert_eq!(p.map(3.0, 9.0), Some((3.0, 9.0)));
        assert_eq!(p.inverse_map(3.0, 9.0), Some((3.0, 9.0)));
        p.center = [100.0, 80.0];
        p.normalization_scale = 100.0;
        p.output_center = [500.0, -200.0];
        p.output_scale = 200.0;
        p.homography = [0.0, -1.0, 0.1, 1.0, 0.0, -0.2, 0.0, 0.0, 1.0];
        assert_eq!(p.map(110.0, 100.0), Some((480.0, -220.0)));
        let sensor = p.inverse_map(480.0, -220.0).unwrap();
        assert!((sensor.0 - 110.0).abs() < 1e-8 && (sensor.1 - 100.0).abs() < 1e-8);
        let warp = p.to_warp(200, 160, 0.001).unwrap();
        let q = warp.map(110.0, 100.0);
        assert!((q.0 - 480.0).abs() < 0.001 && (q.1 + 220.0).abs() < 0.001);
    }

    #[test]
    fn distorted_projective_mesh_matches_full_sensor_including_edges() {
        let p = FrameProjection {
            center: [3123.5, 2087.5],
            normalization_scale: 3124.0,
            distortion: [0.012, 0.001, -0.0015],
            homography: [1.12, -0.17, 0.3, 0.17, 1.12, -0.1, 0.012, -0.008, 1.0],
            output_center: [4000.0, 5000.0],
            output_scale: 3124.0,
        };
        let warp = p.to_warp(6248, 4176, 0.025).unwrap();
        for iy in 0..=40 {
            for ix in 0..=60 {
                let x = 6247.0 * ix as f64 / 60.0;
                let y = 4175.0 * iy as f64 / 40.0;
                let q = p.map(x, y).unwrap();
                let back = p.inverse_map(q.0, q.1).unwrap();
                assert!((x - back.0).hypot(y - back.1) < 1e-5);
                let approx = warp.map(x as f32, y as f32);
                assert!((q.0 - approx.0 as f64).hypot(q.1 - approx.1 as f64) <= 0.025);
            }
        }
    }

    #[test]
    fn cropped_mesh_uses_raster_center_without_moving_optical_normalization() {
        let full = FrameProjection {
            center: [3123.5, 2087.5],
            normalization_scale: 3124.0,
            distortion: [0.012, 0.001, -0.0015],
            homography: [1.12, -0.17, 0.3, 0.17, 1.12, -0.1, 0.012, -0.008, 1.0],
            output_center: [4000., 5000.],
            output_scale: 3124.0,
        };
        let mut crop = full.clone();
        let offset = [5600., 3700.];
        crop.center = [full.center[0] - offset[0], full.center[1] - offset[1]];
        let warp = crop.to_warp(320, 256, 0.025).unwrap();
        for iy in 0..=23 {
            for ix in 0..=31 {
                let x = 319. * ix as f64 / 31.;
                let y = 255. * iy as f64 / 23.;
                let exact = crop.map(x, y).unwrap();
                let original = full.map(x + offset[0], y + offset[1]).unwrap();
                assert!((exact.0 - original.0).hypot(exact.1 - original.1) < 1e-10);
                let approx = warp.map(x as f32, y as f32);
                assert!((exact.0 - approx.0 as f64).hypot(exact.1 - approx.1 as f64) <= 0.025);
                let back = warp.inverse_map(exact.0 as f32, exact.1 as f32).unwrap();
                assert!(
                    (x - back.0 as f64).hypot(y - back.1 as f64) < 0.025,
                    "cropped approximate inverse displaced {x},{y} to {back:?}"
                );
                let exact_back = crop.inverse_map(exact.0, exact.1).unwrap();
                assert!((x - exact_back.0).hypot(y - exact_back.1) < 1e-5);
            }
        }
    }

    #[test]
    fn invalid_singular_horizon_and_budget_fail_explicitly() {
        let mut p = identity();
        p.homography = [0.0; 9];
        assert!(p.validate().is_err());
        assert!(p.map(1.0, 1.0).is_none());
        assert!(p.inverse_map(1.0, 1.0).is_none());
        p = identity();
        p.normalization_scale = f64::NAN;
        assert!(p.validate().is_err());
        p = identity();
        p.homography[6] = 1.0;
        p.homography[8] = -2.0;
        assert!(p.map(2.0, 0.0).is_none());
        assert!(p.to_warp(5, 5, 0.01).is_err());
        p = identity();
        assert!(p
            .to_warp_with_node_budget(6248, 4176, 0.025, 1)
            .unwrap_err()
            .to_string()
            .contains("node budget"));
        assert!(p.to_warp_with_node_budget(8, 8, 0.025, 0).is_err());
        assert!(p.to_warp_with_node_budget(8, 8, 0.025, 1_000_001).is_err());
        let bounded = p.to_warp_with_node_budget(8, 8, 0.025, 100).unwrap();
        assert!(bounded.local.unwrap().u.len() <= 100);
        assert!(p.to_warp(usize::MAX, 2, 0.01).is_err());
        assert!(p.to_warp(0, 2, 0.01).is_err());
        assert!(p.to_warp(100, 100, 0.0).is_err());
        assert!(p.to_warp(1_000_000, 1_000_000, 0.01).is_err());
        p.output_center = [1e12, 1e12];
        assert!(p.to_warp(10, 10, 1e-9).is_err());
        assert!(p.map(f64::NAN, 0.0).is_none());
    }
}
