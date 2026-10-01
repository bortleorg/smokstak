//! Explicit projective geometry for overlapping sky projections.
//! Separate from the burst transform ladder: callers must validate on stars
//! withheld from both correspondence fitting and model selection before use.
use serde::{Deserialize, Serialize};
use sr_core::math::{cholesky_solve, median};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ProjectiveTransform {
    /// Row-major homogeneous mapping, source sensor pixels to destination pixels.
    pub m: [f64; 9],
}

impl ProjectiveTransform {
    pub fn apply(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let m = self.m;
        let z = m[6] * x + m[7] * y + m[8];
        if !z.is_finite() || z.abs() < 1e-12 {
            return None;
        }
        let p = (
            (m[0] * x + m[1] * y + m[2]) / z,
            (m[3] * x + m[4] * y + m[5]) / z,
        );
        (p.0.is_finite() && p.1.is_finite()).then_some(p)
    }

    pub fn compose(&self, other: &Self) -> Self {
        let mut m = [0.; 9];
        for r in 0..3 {
            for c in 0..3 {
                m[3 * r + c] = (0..3).map(|k| self.m[3 * r + k] * other.m[3 * k + c]).sum();
            }
        }
        Self { m }
    }

    /// Bounds of a complete rectangular source footprint, in destination pixels.
    /// Reject a projective horizon crossing the rectangle instead of producing
    /// finite-looking corner bounds for an unbounded interior.
    pub fn bounds(&self, width: usize, height: usize) -> Option<[f64; 4]> {
        if width == 0 || height == 0 {
            return None;
        }
        let m = self.m;
        let determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
        if !determinant.is_finite() || determinant == 0.0 {
            return None;
        }
        let corners = [
            (-0.5, -0.5),
            (width as f64 - 0.5, -0.5),
            (-0.5, height as f64 - 0.5),
            (width as f64 - 0.5, height as f64 - 0.5),
        ];
        let mut sign = 0.;
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (x, y) in corners {
            let z = self.m[6] * x + self.m[7] * y + self.m[8];
            if !z.is_finite() || z.abs() < 1e-8 || (sign != 0. && z.signum() != sign) {
                return None;
            }
            sign = z.signum();
            let (x, y) = self.apply(x, y)?;
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x);
            bounds[3] = bounds[3].max(y);
        }
        Some(bounds)
    }
}

/// Fit a robust homography to fixed, independently established star pairs.
/// Each pair is (source, destination). Normalize coordinates before solving;
/// direct sensor-coordinate normal equations are badly conditioned on 60 MP data.
/// Returns a candidate only, never a declaration that the registration is safe.
pub fn fit(pairs: &[([f64; 2], [f64; 2])]) -> Option<ProjectiveTransform> {
    if pairs.len() < 12
        || pairs
            .iter()
            .any(|(a, b)| a.iter().chain(b).any(|v| !v.is_finite()))
    {
        return None;
    }
    let normalization = |source: bool| {
        let mut center = [0.; 2];
        for (a, b) in pairs {
            let p = if source { a } else { b };
            for i in 0..2 {
                center[i] += p[i] / pairs.len() as f64;
            }
        }
        let rms = (pairs
            .iter()
            .map(|(a, b)| {
                let p = if source { a } else { b };
                (p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2)
            })
            .sum::<f64>()
            / pairs.len() as f64)
            .sqrt();
        (center, rms)
    };
    let (src, ss) = normalization(true);
    let (dst, ds) = normalization(false);
    if ss < 1e-6 || ds < 1e-6 {
        return None;
    }
    let n: Vec<_> = pairs
        .iter()
        .map(|(a, b)| {
            (
                [(a[0] - src[0]) / ss, (a[1] - src[1]) / ss],
                [(b[0] - dst[0]) / ds, (b[1] - dst[1]) / ds],
            )
        })
        .collect();
    let mut weights = vec![1.; n.len()];
    let mut result = None;
    for _ in 0..8 {
        let mut ata = vec![0.; 64];
        let mut atb = vec![0.; 8];
        for (([x, y], [u, v]), weight) in n.iter().zip(&weights) {
            for (row, value) in [
                ([*x, *y, 1., 0., 0., 0., -u * x, -u * y], *u),
                ([0., 0., 0., *x, *y, 1., -v * x, -v * y], *v),
            ] {
                for i in 0..8 {
                    atb[i] += weight * row[i] * value;
                    for j in 0..8 {
                        ata[i * 8 + j] += weight * row[i] * row[j];
                    }
                }
            }
        }
        if !cholesky_solve(&mut ata, &mut atb, 8) {
            return None;
        }
        let h = ProjectiveTransform {
            m: [
                atb[0], atb[1], atb[2], atb[3], atb[4], atb[5], atb[6], atb[7], 1.,
            ],
        };
        let errors: Vec<f32> = n
            .iter()
            .map(|(a, b)| {
                h.apply(a[0], a[1]).map_or(f32::INFINITY, |p| {
                    ((p.0 - b[0]).hypot(p.1 - b[1]) * ds) as f32
                })
            })
            .collect();
        if errors.iter().any(|e| !e.is_finite()) {
            return None;
        }
        let cutoff = (median(&errors) * 2.).max(0.1);
        for (w, e) in weights.iter_mut().zip(errors) {
            *w = if e <= cutoff { 1. } else { (cutoff / e) as f64 };
        }
        result = Some(h);
    }
    let src_norm = ProjectiveTransform {
        m: [
            1. / ss,
            0.,
            -src[0] / ss,
            0.,
            1. / ss,
            -src[1] / ss,
            0.,
            0.,
            1.,
        ],
    };
    let dst_inverse = ProjectiveTransform {
        m: [ds, 0., dst[0], 0., ds, dst[1], 0., 0., 1.],
    };
    Some(dst_inverse.compose(&result?).compose(&src_norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recovers_sky_projection_on_withheld_points_at_large_coordinates() {
        let truth = ProjectiveTransform {
            m: [0.98, 0.04, 450., -0.03, 1.01, 3500., 2e-6, -3e-6, 1.],
        };
        let pairs: Vec<_> = (0..120)
            .map(|i| {
                let a = [(i % 12) as f64 * 700., (i / 12) as f64 * 100. + 100.];
                let b = truth.apply(a[0], a[1]).unwrap();
                (a, [b.0, b.1])
            })
            .collect();
        let fitted = fit(&pairs).unwrap();
        for a in [[23., 458.], [9400., 6300.], [0., 0.]] {
            let b = truth.apply(a[0], a[1]).unwrap();
            let p = fitted.apply(a[0], a[1]).unwrap();
            assert!((b.0 - p.0).hypot(b.1 - p.1) < 1e-5);
        }
    }
    #[test]
    fn refuses_degenerate_pairs_and_horizon_crossings() {
        assert!(fit(&vec![([2., 3.], [8., 4.]); 20]).is_none());
        let line: Vec<_> = (0..20).map(|i| ([i as f64, 0.], [i as f64, 2.])).collect();
        assert!(fit(&line).is_none());
        let horizon = ProjectiveTransform {
            m: [1., 0., 0., 0., 1., 0., 0.01, 0., -1.],
        };
        assert!(horizon.bounds(200, 100).is_none());
    }

    #[test]
    fn robust_fit_limits_influence_of_bad_correspondences() {
        let truth = ProjectiveTransform {
            m: [1., 0.02, 120., -0.01, 1., -30., 1e-6, -2e-6, 1.],
        };
        let pairs: Vec<_> = (0..200)
            .map(|i| {
                let a = [(i % 20) as f64 * 400., (i / 20) as f64 * 500.];
                let mut b = truth.apply(a[0], a[1]).unwrap();
                if i % 11 == 0 {
                    b.0 += 20.;
                    b.1 -= 15.;
                }
                (a, [b.0, b.1])
            })
            .collect();
        let fitted = fit(&pairs).unwrap();
        for p in [[300., 400.], [5000., 3000.], [7500., 4000.]] {
            let a = truth.apply(p[0], p[1]).unwrap();
            let b = fitted.apply(p[0], p[1]).unwrap();
            assert!((a.0 - b.0).hypot(a.1 - b.1) < 0.1);
        }
    }
}
