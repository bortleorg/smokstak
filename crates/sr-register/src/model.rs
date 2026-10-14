//! Robust fitting of the global transform hierarchy.
//!
//! Model selection is not cosmetic. A homography, or even a full affine, can
//! absorb genuine local deformation into a global warp and thereby invent
//! detail that the sensor never measured. So we start at translation and only
//! accept a richer model when the residuals clearly demand it.

use serde::{Deserialize, Serialize};
use sr_core::geometry::{GlobalTransform, TransformModel};
use sr_core::math::{cholesky_solve, huber_weight, mad_sigma, median, tukey_weight};

/// One measured correspondence, in reference-frame coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Correspondence {
    /// Position in the reference frame.
    pub x: f32,
    pub y: f32,
    /// Where that position should move to, i.e. `(x + rx, y + ry)`.
    pub rx: f32,
    pub ry: f32,
    /// Prior confidence from the correlation peak.
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitResult {
    pub transform: GlobalTransform,
    pub model: TransformModel,
    /// Robust RMS residual, reference pixels.
    pub residual_rms: f32,
    /// Residual at the 50th, 90th and 99th percentile.
    pub residual_p50: f32,
    pub residual_p90: f32,
    pub residual_p99: f32,
    pub inliers: usize,
    pub total: usize,
}

impl FitResult {
    pub fn inlier_fraction(&self) -> f32 {
        self.inliers as f32 / self.total.max(1) as f32
    }
}

fn solve_weighted(
    corr: &[Correspondence],
    w: &[f32],
    model: TransformModel,
) -> Option<GlobalTransform> {
    match model {
        TransformModel::Translation => {
            let mut sw = 0.0f64;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            for (c, &wi) in corr.iter().zip(w) {
                let wi = wi as f64;
                sw += wi;
                sx += wi * c.rx as f64;
                sy += wi * c.ry as f64;
            }
            if sw < 1e-9 {
                return None;
            }
            Some(GlobalTransform::translation(
                (sx / sw) as f32,
                (sy / sw) as f32,
            ))
        }
        TransformModel::Euclidean | TransformModel::Similarity => {
            // x' = a*x - b*y + tx ; y' = b*x + a*y + ty
            let n = 4;
            let mut ata = vec![0.0f64; n * n];
            let mut atb = vec![0.0f64; n];
            for (c, &wi) in corr.iter().zip(w) {
                let wi = wi as f64;
                let (x, y) = (c.x as f64, c.y as f64);
                let (xp, yp) = ((c.x + c.rx) as f64, (c.y + c.ry) as f64);
                // Row for x': [x, -y, 1, 0]
                let r1 = [x, -y, 1.0, 0.0];
                // Row for y': [y,  x, 0, 1]
                let r2 = [y, x, 0.0, 1.0];
                for i in 0..n {
                    for j in 0..n {
                        ata[i * n + j] += wi * (r1[i] * r1[j] + r2[i] * r2[j]);
                    }
                    atb[i] += wi * (r1[i] * xp + r2[i] * yp);
                }
            }
            if !cholesky_solve(&mut ata, &mut atb, n) {
                return None;
            }
            let (mut a, mut b) = (atb[0] as f32, atb[1] as f32);
            if model == TransformModel::Euclidean {
                let m = (a * a + b * b).sqrt();
                if m < 1e-9 {
                    return None;
                }
                a /= m;
                b /= m;
            }
            Some(GlobalTransform {
                m: [a, -b, atb[2] as f32, b, a, atb[3] as f32],
            })
        }
        TransformModel::Affine => {
            // Two independent 3-parameter problems sharing one normal matrix.
            let n = 3;
            let mut ata = vec![0.0f64; n * n];
            let mut atbx = vec![0.0f64; n];
            let mut atby = vec![0.0f64; n];
            for (c, &wi) in corr.iter().zip(w) {
                let wi = wi as f64;
                let r = [c.x as f64, c.y as f64, 1.0];
                let xp = (c.x + c.rx) as f64;
                let yp = (c.y + c.ry) as f64;
                for i in 0..n {
                    for j in 0..n {
                        ata[i * n + j] += wi * r[i] * r[j];
                    }
                    atbx[i] += wi * r[i] * xp;
                    atby[i] += wi * r[i] * yp;
                }
            }
            let mut ata2 = ata.clone();
            if !cholesky_solve(&mut ata, &mut atbx, n) {
                return None;
            }
            if !cholesky_solve(&mut ata2, &mut atby, n) {
                return None;
            }
            Some(GlobalTransform {
                m: [
                    atbx[0] as f32,
                    atbx[1] as f32,
                    atbx[2] as f32,
                    atby[0] as f32,
                    atby[1] as f32,
                    atby[2] as f32,
                ],
            })
        }
    }
}

fn residuals(corr: &[Correspondence], t: &GlobalTransform) -> Vec<f32> {
    corr.iter()
        .map(|c| {
            let (px, py) = t.apply(c.x, c.y);
            let dx = px - (c.x + c.rx);
            let dy = py - (c.y + c.ry);
            (dx * dx + dy * dy).sqrt()
        })
        .collect()
}

/// Iteratively reweighted fit of one model.
///
/// Tukey rather than Huber for the final passes: a probe that latched onto a
/// moving object or a repeated texture should be removed from the fit, not
/// merely trusted less.
pub fn fit_model(
    corr: &[Correspondence],
    model: TransformModel,
    iters: usize,
) -> Option<FitResult> {
    if corr.len() < model.dof() {
        return None;
    }
    // Seed from the component-wise median displacement. Tukey is non-convex,
    // so an ordinary least-squares start is not safe: with a fifth of the
    // probes latched onto a moving object, the first fit is dragged far enough
    // that the reweighting never recovers. The median has a 50% breakdown
    // point and costs nothing here.
    let rxs: Vec<f32> = corr.iter().map(|c| c.rx).collect();
    let rys: Vec<f32> = corr.iter().map(|c| c.ry).collect();
    let (mx, my) = (median(&rxs), median(&rys));
    let seed_res: Vec<f32> = corr
        .iter()
        .map(|c| ((c.rx - mx).powi(2) + (c.ry - my).powi(2)).sqrt())
        .collect();
    let seed_sigma = mad_sigma(&seed_res).max(1e-3);

    let mut w: Vec<f32> = corr
        .iter()
        .zip(&seed_res)
        .map(|(c, &r)| c.weight.max(1e-6) * tukey_weight(r, 4.0 * seed_sigma))
        .collect();
    if w.iter().filter(|&&x| x > 1e-6).count() < model.dof() {
        w = corr.iter().map(|c| c.weight.max(1e-6)).collect();
    }
    let mut t = solve_weighted(corr, &w, model)?;

    // Huber first (convex, keeps every probe in play), then Tukey to cut the
    // survivors that are genuinely inconsistent.
    let huber_iters = iters / 2;
    for it in 0..iters {
        let res = residuals(corr, &t);
        let sigma = mad_sigma(&res).max(1e-3);
        for i in 0..corr.len() {
            let base = corr[i].weight.max(1e-6);
            w[i] = base
                * if it < huber_iters {
                    huber_weight(res[i], 2.0 * sigma)
                } else {
                    tukey_weight(res[i], 4.0 * sigma)
                };
        }
        if w.iter().filter(|&&x| x > 1e-6).count() < model.dof() {
            break;
        }
        match solve_weighted(corr, &w, model) {
            Some(nt) => t = nt,
            None => break,
        }
    }

    let res = residuals(corr, &t);
    let sigma = mad_sigma(&res).max(1e-4);
    let inlier_cut = 4.0 * sigma;
    let inlier_res: Vec<f32> = res.iter().copied().filter(|&r| r <= inlier_cut).collect();
    let inliers = inlier_res.len();
    let rms = if inliers > 0 {
        (inlier_res.iter().map(|r| (r * r) as f64).sum::<f64>() / inliers as f64).sqrt() as f32
    } else {
        f32::INFINITY
    };

    let mut sorted = res.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |p: f32| sorted[((sorted.len() - 1) as f32 * p) as usize];

    Some(FitResult {
        transform: t,
        model,
        residual_rms: rms,
        residual_p50: pick(0.50),
        residual_p90: pick(0.90),
        residual_p99: pick(0.99),
        inliers,
        total: corr.len(),
    })
}

/// Fit the transform hierarchy and return the simplest adequate model.
///
/// `tolerance` is the residual improvement (in reference pixels) a richer model
/// must deliver before it is preferred. Set it from the noise floor of the
/// correspondence estimates, not from wishful thinking: below that, extra
/// degrees of freedom are fitting noise.
pub fn fit_best(corr: &[Correspondence], tolerance: f32, iters: usize) -> Option<FitResult> {
    let ladder = [
        TransformModel::Translation,
        TransformModel::Euclidean,
        TransformModel::Similarity,
        TransformModel::Affine,
    ];
    let mut fits: Vec<FitResult> = Vec::new();
    for m in ladder {
        if let Some(f) = fit_model(corr, m, iters) {
            fits.push(f);
        }
    }
    if fits.is_empty() {
        return None;
    }
    let best_rms = fits
        .iter()
        .map(|f| f.residual_rms)
        .fold(f32::INFINITY, f32::min);
    // Walk from simplest to richest and stop at the first model that comes
    // within `tolerance` of the best achievable residual.
    for f in &fits {
        if f.residual_rms <= best_rms + tolerance {
            return Some(f.clone());
        }
    }
    fits.into_iter()
        .min_by(|a, b| a.residual_rms.partial_cmp(&b.residual_rms).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(t: &GlobalTransform, n: usize, outliers: usize) -> Vec<Correspondence> {
        let mut v = Vec::new();
        for i in 0..n {
            let x = ((i * 37) % 200) as f32 * 5.0;
            let y = ((i * 53) % 150) as f32 * 5.0;
            let (px, py) = t.apply(x, y);
            v.push(Correspondence {
                x,
                y,
                rx: px - x,
                ry: py - y,
                weight: 1.0,
            });
        }
        for i in 0..outliers {
            let x = ((i * 71) % 200) as f32 * 5.0;
            let y = ((i * 29) % 150) as f32 * 5.0;
            v.push(Correspondence {
                x,
                y,
                rx: 25.0,
                ry: -18.0,
                weight: 1.0,
            });
        }
        v
    }

    #[test]
    fn recovers_pure_translation() {
        let t = GlobalTransform::translation(3.25, -1.75);
        let f = fit_model(&synth(&t, 60, 0), TransformModel::Translation, 8).unwrap();
        assert!((f.transform.m[2] - 3.25).abs() < 1e-3);
        assert!((f.transform.m[5] + 1.75).abs() < 1e-3);
        assert!(f.residual_rms < 1e-3);
    }

    #[test]
    fn recovers_similarity() {
        let t = GlobalTransform::similarity(0.004, 1.0008, 2.0, -3.0);
        let f = fit_model(&synth(&t, 80, 0), TransformModel::Similarity, 8).unwrap();
        for i in 0..6 {
            assert!(
                (f.transform.m[i] - t.m[i]).abs() < 1e-3,
                "param {i}: {} vs {}",
                f.transform.m[i],
                t.m[i]
            );
        }
    }

    #[test]
    fn rejects_gross_outliers() {
        let t = GlobalTransform::similarity(0.003, 1.0, 1.5, 2.5);
        // 20% of probes latched onto something moving.
        let f = fit_model(&synth(&t, 80, 20), TransformModel::Similarity, 12).unwrap();
        assert!(
            (f.transform.m[2] - 1.5).abs() < 0.05,
            "tx {}",
            f.transform.m[2]
        );
        assert!(
            (f.transform.m[5] - 2.5).abs() < 0.05,
            "ty {}",
            f.transform.m[5]
        );
        assert!(f.inlier_fraction() > 0.7, "inliers {}", f.inlier_fraction());
    }

    #[test]
    fn prefers_translation_when_that_is_all_there_is() {
        let t = GlobalTransform::translation(2.0, -4.0);
        let f = fit_best(&synth(&t, 100, 0), 0.02, 10).unwrap();
        assert_eq!(f.model, TransformModel::Translation);
    }

    #[test]
    fn escalates_to_affine_when_the_data_demands_it() {
        // Shear cannot be represented by a similarity.
        let t = GlobalTransform {
            m: [1.0, 0.02, 3.0, -0.005, 1.001, -2.0],
        };
        let f = fit_best(&synth(&t, 120, 0), 0.02, 10).unwrap();
        assert_eq!(f.model, TransformModel::Affine);
        assert!(f.residual_rms < 1e-2);
    }
}
