//! Noise-guarded local quadratic reconstruction, without a cross-channel prior.
//! Clipped footprints use the merge's separate shared estimator, not curvature.

const PRODUCT: [[usize; 6]; 6] = [
    [0, 1, 2, 3, 4, 5],
    [1, 3, 4, 6, 7, 8],
    [2, 4, 5, 7, 8, 9],
    [3, 6, 7, 10, 11, 12],
    [4, 7, 8, 11, 12, 13],
    [5, 8, 9, 12, 13, 14],
];

#[derive(Clone, Copy, Default)]
pub(crate) struct Curvature {
    normal: [f32; 15],
    noise: [f32; 15],
    rhs: [f32; 6],
    w2: f32,
}

impl Curvature {
    pub fn add(&mut self, w: f32, x: f32, y: f32, v: f32, variance: f32) {
        let (xx, xy, yy) = (x * x, x * y, y * y);
        let basis = [
            1.,
            x,
            y,
            xx,
            xy,
            yy,
            xx * x,
            xx * y,
            x * yy,
            yy * y,
            xx * xx,
            xx * xy,
            xx * yy,
            xy * yy,
            yy * yy,
        ];
        let nv = w * w * variance.max(0.);
        for (i, b) in basis.iter().enumerate() {
            self.normal[i] += w * b;
            self.noise[i] += nv * b;
        }
        for (r, b) in self.rhs.iter_mut().zip(basis.iter()) {
            *r += w * v * b;
        }
        self.w2 += w * w;
    }

    /// Called only after the existing plane's support/centroid guards pass.
    pub fn correction(&self) -> Option<(f32, f32)> {
        self.checked_correction().ok()
    }

    pub fn effective_samples(&self) -> f32 {
        self.normal[0] * self.normal[0] / self.w2.max(1e-30)
    }

    fn variance_of(&self, a: &[f64; 6], b: &[f64; 6]) -> f64 {
        (0..6).map(|i| (0..6)
            .map(|j| a[i] * b[j] * self.noise[PRODUCT[i][j]] as f64)
            .sum::<f64>()).sum()
    }

    /// Diagnostic only: variances and covariance before the amplification gate.
    /// May be available even when the effective-sample gate rejects the fit.
    pub fn variance_probe(&self) -> Option<[f64; 3]> {
        let quad = response(&self.normal, 6)?;
        let linear = response(&self.normal, 3)?;
        Some([self.variance_of(&linear, &linear), self.variance_of(&quad, &quad),
            self.variance_of(&linear, &quad)])
    }

    /// First failing guard; numerical operations and thresholds match correction.
    pub fn checked_correction(&self) -> Result<(f32, f32), &'static str> {
        self.checked_with_policy(false)
    }

    pub fn checked_with_policy(&self, bounded_blend: bool) -> Result<(f32, f32), &'static str> {
        if self.normal[0] * self.normal[0] < 12. * self.w2 {
            return Err("effective_samples");
        }
        let quad = response(&self.normal, 6).ok_or("quadratic_condition")?;
        let linear = response(&self.normal, 3).ok_or("linear_condition")?;
        let vp = self.variance_of(&linear, &linear);
        let vq = self.variance_of(&quad, &quad);
        if !vp.is_finite() || !vq.is_finite() || vp <= 0. || vq <= 0. {
            return Err("variance");
        }
        let cap = if vq > 4. * vp {
            if !bounded_blend { return Err("variance"); }
            variance_blend_cap(vp, vq, self.variance_of(&linear, &quad)).ok_or("variance")?
        } else { 1. };
        let diff = std::array::from_fn(|i| quad[i] - linear[i]);
        let vd = self.variance_of(&diff, &diff).max(0.);
        let delta = diff
            .iter()
            .zip(self.rhs)
            .map(|(r, v)| r * v as f64)
            .sum::<f64>();
        // Three-sigma positive-part shrinkage. Smoothly returns to the plane
        // where the measured curvature correction is indistinguishable from noise.
        let blend = (1. - 9. * vd / (delta * delta).max(1e-30)).clamp(0., 1.).min(cap);
        (delta.is_finite() && blend.is_finite()).then_some((delta as f32, blend as f32)).ok_or("nonfinite")
    }
}

/// Largest convex interpolation coefficient satisfying the existing 4x ceiling.
fn variance_blend_cap(vp: f64, vq: f64, covariance: f64) -> Option<f64> {
    let a = vq + vp - 2.*covariance;
    let b = 2.*(covariance-vp);
    if !covariance.is_finite() || a <= 0. { return None; }
    let cap = 6.*vp / ((b*b+12.*a*vp).sqrt()+b);
    (cap.is_finite() && cap > 0.).then_some(cap.min(1.))
}

/// Row zero of the weighted polynomial estimator. Normalize the normal matrix
/// before Cholesky so conditioning is independent of weight/coordinate units.
fn response(m: &[f32; 15], n: usize) -> Option<[f64; 6]> {
    let mut scale = [0.; 6];
    for i in 0..n {
        scale[i] = (m[PRODUCT[i][i]] as f64).sqrt();
        if !scale[i].is_finite() || scale[i] <= 1e-15 {
            return None;
        }
    }
    let mut l = [[0.; 6]; 6];
    for i in 0..n {
        for j in 0..=i {
            let v = m[PRODUCT[i][j]] as f64 / (scale[i] * scale[j])
                - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !v.is_finite() || v < 1e-3 {
                    return None;
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    let mut b = [0.; 6];
    b[0] = 1. / scale[0];
    for i in 0..n {
        b[i] = (b[i] - (0..i).map(|j| l[i][j] * b[j]).sum::<f64>()) / l[i][i];
    }
    for i in (0..n).rev() {
        b[i] = (b[i] - (i + 1..n).map(|j| l[j][i] * b[j]).sum::<f64>()) / l[i][i];
    }
    for i in 0..n {
        b[i] /= scale[i];
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_mixture_retains_variance_ceiling_including_shared_smaller_blends() {
        for ratio in [4.001_f64, 4.5, 10., 100.] {
            for correlation in [-0.9,0.,0.9] {
                let covariance=correlation*ratio.sqrt();
                let cap=variance_blend_cap(1.,ratio,covariance).unwrap();
                for fraction in [0.,0.1,0.5,1.] {
                    let t=cap*fraction;
                    let v=(1.-t).powi(2)+t*t*ratio+2.*t*(1.-t)*covariance;
                    assert!(v <= 4.+1e-12, "{ratio} {correlation} {t} {v}");
                }
            }
        }
    }

    #[test]
    fn curved_profile_recovers_centre_without_changing_a_flat_field() {
        for curvature in [0., -0.2] {
            let mut m = Curvature::default();
            let mut sum = 0.;
            let mut weight = 0.;
            for y in -5..=5 {
                for x in -5..=5 {
                    let (x, y) = (x as f32 / 5., y as f32 / 5.);
                    let w = (-(x * x + y * y) / 0.4).exp();
                    let v = 0.5 + curvature * (x * x + y * y);
                    m.add(w, x, y, v, 1e-7);
                    sum += w * v;
                    weight += w;
                }
            }
            let (delta, blend) = m.correction().unwrap();
            let result = sum / weight + blend*delta;
            assert!((result - 0.5).abs() < 1e-4, "{result}");
        }
    }

    #[test]
    fn sparse_or_collinear_samples_do_not_fit_curvature() {
        for count in [5, 30] {
            let mut m = Curvature::default();
            for i in 0..count {
                m.add(1., i as f32 / 30., 0., 0.5, 1e-5);
            }
            assert!(m.correction().is_none());
            assert_eq!(m.checked_correction().unwrap_err(),
                if count == 5 { "effective_samples" } else { "quadratic_condition" });
        }
    }
}
