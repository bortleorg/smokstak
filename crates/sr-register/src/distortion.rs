//! Smooth stellar distortion, accepted only on disjoint validation stars.
//!
//! A flipped optical field can leave curved residuals after affine alignment.
//! Fit twelve coefficients, not independent patches; use the existing forward
//! deformation mesh for deposition without resampling the input images.

use crate::refine::Grid;
use rayon::prelude::*;
use sr_core::geometry::{DeformationField, GlobalTransform};
use sr_core::math::{cholesky_solve, median};
use sr_core::star::Star;

type Pair = ([f64; 2], [f64; 2]);

/// Veto a correlation field when independently measured stars do not improve.
/// None means too few stars or insufficient spatial coverage to judge it.
pub fn validates_correlation(
    reference: &[Star],
    target: &[Star],
    global: &GlobalTransform,
    field: &DeformationField,
    w: usize,
    h: usize,
) -> Option<bool> {
    if reference.len() < 150 || target.len() < 150 {
        return None;
    }
    let grid = Grid::new(reference, 6.);
    let mapped: Vec<_> = target
        .iter()
        .map(|s| {
            let (x, y) = global.apply(s.x, s.y);
            Star { x, y, flux: s.flux }
        })
        .collect();
    let reverse = Grid::new(&mapped, 6.);
    let pairs: Vec<Pair> = mapped
        .iter()
        .filter_map(|s| {
            let r = grid.nearest(reference, s.x, s.y)?;
            reverse
                .nearest(&mapped, r.x, r.y)
                .filter(|back| std::ptr::eq(*back, s))?;
            Some(([s.x as f64, s.y as f64], [r.x as f64, r.y as f64]))
        })
        .collect();
    if pairs.len() < 150 || !covered(&pairs, w as f64, h as f64, 4) {
        return None;
    }
    let mut before = Vec::new();
    let mut after = Vec::new();
    for &(s, d) in &pairs {
        let u = field.sample(s[0] as f32, s[1] as f32);
        before.push((s[0] - d[0]).hypot(s[1] - d[1]) as f32);
        after.push((s[0] + u.0 as f64 - d[0]).hypot(s[1] + u.1 as f64 - d[1]) as f32);
    }
    let (b, a) = (median(&before), median(&after));
    let mut bs = before.clone();
    let mut az = after.clone();
    bs.sort_by(f32::total_cmp);
    az.sort_by(f32::total_cmp);
    if a > b * 0.8 || b - a < 0.05 || az[az.len() * 9 / 10] > bs[bs.len() * 9 / 10] * 0.8 {
        return Some(false);
    }
    for cy in 0..4 {
        for cx in 0..4 {
            let ids: Vec<_> = pairs
                .iter()
                .enumerate()
                .filter(|(_, (p, _))| {
                    p[0] >= (cx * w) as f64 / 4.
                        && p[0] < ((cx + 1) * w) as f64 / 4.
                        && p[1] >= (cy * h) as f64 / 4.
                        && p[1] < ((cy + 1) * h) as f64 / 4.
                })
                .map(|(i, _)| i)
                .collect();
            let b: Vec<_> = ids.iter().map(|&i| before[i]).collect();
            let a: Vec<_> = ids.iter().map(|&i| after[i]).collect();
            if median(&a) > median(&b) + 0.03 {
                return Some(false);
            }
        }
    }
    Some(true)
}

fn basis(p: [f64; 2], w: f64, h: f64) -> [f64; 6] {
    let (x, y) = (2. * p[0] / w - 1., 2. * p[1] / h - 1.);
    [1., x, y, x * x, x * y, y * y]
}

fn displacement(c: &[[f64; 6]; 2], b: &[f64; 6]) -> [f64; 2] {
    std::array::from_fn(|axis| c[axis].iter().zip(b).map(|(a, b)| a * b).sum())
}

fn fit(pairs: &[Pair], w: f64, h: f64, n: usize) -> Option<[[f64; 6]; 2]> {
    let mut keep = vec![true; pairs.len()];
    let mut coeff = [[0.; 6]; 2];
    for _ in 0..4 {
        if keep.iter().filter(|&&k| k).count() < 50 {
            return None;
        }
        let mut normal = vec![0.; n * n];
        let mut rhs = [vec![0.; n], vec![0.; n]];
        for (i, &(src, dst)) in pairs.iter().enumerate() {
            if !keep[i] {
                continue;
            }
            let b = basis(src, w, h);
            for j in 0..n {
                for k in 0..n {
                    normal[j * n + k] += b[j] * b[k];
                }
                for axis in 0..2 {
                    rhs[axis][j] += b[j] * (dst[axis] - src[axis]);
                }
            }
        }
        for axis in 0..2 {
            if !cholesky_solve(&mut normal.clone(), &mut rhs[axis], n) {
                return None;
            }
            coeff[axis][..n].copy_from_slice(&rhs[axis]);
        }
        let errors = errors(pairs, &coeff, w, h);
        let cut = (3. * median(&errors)).max(0.3);
        keep = errors.iter().map(|&e| e < cut).collect();
    }
    if coeff.iter().flatten().all(|v| v.is_finite()) {
        Some(coeff)
    } else {
        None
    }
}

fn errors(pairs: &[Pair], c: &[[f64; 6]; 2], w: f64, h: f64) -> Vec<f32> {
    pairs
        .iter()
        .map(|&(s, d)| {
            let u = displacement(c, &basis(s, w, h));
            ((s[0] + u[0] - d[0]).hypot(s[1] + u[1] - d[1])) as f32
        })
        .collect()
}

fn covered(pairs: &[Pair], w: f64, h: f64, minimum: usize) -> bool {
    let mut cells = [0; 16];
    for &(p, _) in pairs {
        if p[0] < 0. || p[1] < 0. || p[0] >= w || p[1] >= h {
            continue;
        }
        let x = (p[0] / w * 4.) as usize;
        let y = (p[1] / h * 4.) as usize;
        cells[y * 4 + x] += 1;
    }
    let valid = cells.iter().all(|&n| n >= minimum);
    if !valid {
        log::debug!("stellar distortion coverage requires {minimum}: {cells:?}");
    }
    valid
}

fn field(pairs: &[Pair], w: usize, h: usize) -> Option<DeformationField> {
    if pairs.len() < 150 || w == 0 || h == 0 {
        return None;
    }
    let (wf, hf) = (w as f64, h as f64);
    let (train, test): (Vec<_>, Vec<_>) = pairs.iter().enumerate().partition(|(i, _)| i % 3 != 0);
    let train: Vec<Pair> = train.into_iter().map(|(_, &p)| p).collect();
    let test: Vec<Pair> = test.into_iter().map(|(_, &p)| p).collect();
    if !covered(&train, wf, hf, 4) || !covered(&test, wf, hf, 2) {
        return None;
    }
    let affine = fit(&train, wf, hf, 3)?;
    let curved = fit(&train, wf, hf, 6)?;
    let before = errors(&test, &affine, wf, hf);
    let after = errors(&test, &curved, wf, hf);
    let b = median(&before);
    let a = median(&after);
    let mut sorted = after.clone();
    sorted.sort_by(f32::total_cmp);
    let mut original = before.clone();
    original.sort_by(f32::total_cmp);
    // Require a material absolute and relative gain, and good held-out tails.
    log::debug!(
        "stellar distortion validation: {} pairs, median {b:.3} -> {a:.3}, p90 {:.3}",
        pairs.len(),
        sorted[sorted.len() * 9 / 10]
    );
    // The detector uses small core apertures; their phase-dependent centroid
    // scatter is not optical distortion. Demand improvement in both the median
    // and tail, with an absolute ceiling, rather than fitting that scatter.
    if a > b * 0.8
        || b - a < 0.05
        || a > 0.75
        || sorted[sorted.len() * 9 / 10] > 1.25
        || sorted[sorted.len() * 9 / 10] > original[original.len() * 9 / 10] * 0.8
    {
        return None;
    }
    // Do not trade one corner against the rest of the frame.
    for cy in 0..4 {
        for cx in 0..4 {
            let ids: Vec<_> = test
                .iter()
                .enumerate()
                .filter(|(_, (p, _))| {
                    p[0] >= cx as f64 * wf / 4.
                        && p[0] < (cx + 1) as f64 * wf / 4.
                        && p[1] >= cy as f64 * hf / 4.
                        && p[1] < (cy + 1) as f64 * hf / 4.
                })
                .map(|(i, _)| i)
                .collect();
            let eb: Vec<_> = ids.iter().map(|&i| before[i]).collect();
            let ea: Vec<_> = ids.iter().map(|&i| after[i]).collect();
            if median(&ea) > median(&eb) + 0.03 {
                return None;
            }
        }
    }
    // Keep the validated fit. Refitting with the test stars would invalidate
    // the acceptance measurement for the coefficients actually deposited.
    let spacing = w.min(h) as f32 / 32.;
    let gw = (w as f32 / spacing).ceil() as usize + 1;
    let gh = (h as f32 / spacing).ceil() as usize + 1;
    let mut result = DeformationField::zeros((0., 0.), spacing, gw, gh);
    for y in 0..gh {
        for x in 0..gw {
            let u = displacement(
                &curved,
                &basis(
                    [x as f64 * spacing as f64, y as f64 * spacing as f64],
                    wf,
                    hf,
                ),
            );
            if u[0].hypot(u[1]) > 6. {
                return None;
            }
            result.u[y * gw + x] = [u[0] as f32, u[1] as f32];
            result.conf[y * gw + x] = 1.;
        }
    }
    // Bound the field derivative too: no folds or rapidly varying extrapolation.
    for y in 0..gh {
        for x in 0..gw {
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= gw || ny >= gh {
                    continue;
                }
                let a = result.node(x, y);
                let b = result.node(nx, ny);
                if (a[0] - b[0]).hypot(a[1] - b[1]) / spacing > 0.02 {
                    return None;
                }
            }
        }
    }
    log::debug!(
        "stellar distortion: {} train / {} held out, median {:.3} -> {:.3} sensor px",
        train.len(),
        test.len(),
        b,
        a
    );
    Some(result)
}

/// Optical distortion in reference sensor coordinates; automatic per frame.
/// Reciprocal matches prevent one reference star from supplying several pairs.
pub fn refine_stars(
    reference: usize,
    stars: &[Vec<Star>],
    transforms: &[GlobalTransform],
    w: usize,
    h: usize,
) -> Vec<Option<DeformationField>> {
    let ref_stars = &stars[reference];
    if ref_stars.len() < 150 {
        return vec![None; stars.len()];
    }
    let grid = Grid::new(ref_stars, 6.);
    (0..stars.len())
        .into_par_iter()
        .map(|i| {
            if i == reference || stars[i].len() < 150 {
                return None;
            }
            let mapped: Vec<_> = stars[i]
                .iter()
                .map(|s| {
                    let (x, y) = transforms[i].apply(s.x, s.y);
                    Star { x, y, flux: s.flux }
                })
                .collect();
            let reverse = Grid::new(&mapped, 6.);
            let mut pairs = Vec::new();
            for s in &mapped {
                if let Some(r) = grid.nearest(ref_stars, s.x, s.y)
                    && reverse
                        .nearest(&mapped, r.x, r.y)
                        .is_some_and(|back| std::ptr::eq(back, s))
                {
                    pairs.push(([s.x as f64, s.y as f64], [r.x as f64, r.y as f64]));
                }
            }
            log::debug!(
                "frame {i}: testing stellar distortion on {} reciprocal pairs",
                pairs.len()
            );
            let result = field(&pairs, w, h);
            if result.is_some() {
                log::info!(
                    "frame {i}: validated stellar distortion on {} reciprocal pairs",
                    pairs.len()
                );
            }
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(curved: bool) -> Vec<Pair> {
        (0..32 * 24)
            .map(|i| {
                let p = [20. + (i % 32) as f64 * 30., 20. + (i / 32) as f64 * 30.];
                let b = basis(p, 1000., 750.);
                let u = if curved {
                    [1.8 * b[3] - 0.6, 1.2 * b[4]]
                } else {
                    [0., 0.]
                };
                let noise = 0.025 * ((i * 73 % 101) as f64 / 50. - 1.);
                (p, [p[0] + u[0] + noise, p[1] + u[1] - noise])
            })
            .collect()
    }

    #[test]
    fn recovers_curved_field_on_unseen_positions_and_rejects_noise() {
        let f = field(&pairs(true), 1000, 750).unwrap();
        for y in [0., 133., 500., 750.] {
            for x in [0., 175., 550., 1000.] {
                let b = basis([x, y], 1000., 750.);
                let u = f.sample(x as f32, y as f32);
                assert!((u.0 as f64 - (1.8 * b[3] - 0.6)).abs() < 0.015);
                assert!((u.1 as f64 - 1.2 * b[4]).abs() < 0.015);
            }
        }
        assert!(field(&pairs(false), 1000, 750).is_none());
    }

    #[test]
    fn refuses_unsupported_edges_and_large_extrapolation() {
        let mut p = pairs(true);
        p.retain(|(s, _)| s[0] < 700.);
        assert!(field(&p, 1000, 750).is_none());
        let p: Vec<_> = pairs(true)
            .into_iter()
            .map(|(s, d)| (s, [s[0] + 10. * (d[0] - s[0]), s[1] + 10. * (d[1] - s[1])]))
            .collect();
        assert!(field(&p, 1000, 750).is_none());
    }

    #[test]
    fn held_out_stars_veto_a_training_only_curve() {
        let p: Vec<_> = pairs(true)
            .into_iter()
            .enumerate()
            .map(|(i, (s, d))| if i % 3 == 0 { (s, s) } else { (s, d) })
            .collect();
        assert!(field(&p, 1000, 750).is_none());
    }

    #[test]
    fn correlation_field_must_improve_independent_star_geometry() {
        let p = pairs(true);
        let reference: Vec<_> = p
            .iter()
            .map(|&(_, d)| Star {
                x: d[0] as f32,
                y: d[1] as f32,
                flux: 1.,
            })
            .collect();
        let target: Vec<_> = p
            .iter()
            .map(|&(s, _)| Star {
                x: s[0] as f32,
                y: s[1] as f32,
                flux: 1.,
            })
            .collect();
        let good = field(&p, 1000, 750).unwrap();
        assert_eq!(
            validates_correlation(
                &reference,
                &target,
                &GlobalTransform::IDENTITY,
                &good,
                1000,
                750
            ),
            Some(true)
        );
        let mut bad = good.clone();
        bad.u.fill([0.6, -0.4]);
        assert_eq!(
            validates_correlation(
                &reference,
                &target,
                &GlobalTransform::IDENTITY,
                &bad,
                1000,
                750
            ),
            Some(false)
        );
        assert_eq!(
            validates_correlation(
                &reference,
                &target[..100],
                &GlobalTransform::IDENTITY,
                &bad,
                1000,
                750
            ),
            None
        );
    }

    #[test]
    fn flipped_sensor_samples_use_the_reference_field_and_inverse() {
        use sr_core::geometry::WarpField;
        let global = GlobalTransform::similarity(std::f32::consts::PI, 1., 1000., 750.);
        let inverse = global.inverse().unwrap();
        let pairs = pairs(true);
        let reference: Vec<_> = pairs
            .iter()
            .map(|&(_, d)| Star {
                x: d[0] as f32,
                y: d[1] as f32,
                flux: 1.,
            })
            .collect();
        let target: Vec<_> = pairs
            .iter()
            .map(|&(s, _)| {
                let (x, y) = inverse.apply(s[0] as f32, s[1] as f32);
                Star { x, y, flux: 1. }
            })
            .collect();
        let fields = refine_stars(
            0,
            &[reference, target.clone()],
            &[GlobalTransform::IDENTITY, global],
            1000,
            750,
        );
        assert!(fields[0].is_none());
        let warp = WarpField {
            global,
            local: Some(fields[1].clone().unwrap()),
        };
        for (i, s) in target.iter().enumerate().step_by(17) {
            let (x, y) = warp.map(s.x, s.y);
            assert!((x as f64 - pairs[i].1[0]).hypot(y as f64 - pairs[i].1[1]) < 0.05);
            let (sx, sy) = warp.inverse_map(x, y).unwrap();
            assert!((sx - s.x).hypot(sy - s.y) < 0.025);
            let q = warp.rescale(0.5).map(s.x * 0.5, s.y * 0.5);
            assert!((q.0 * 2. - x).hypot(q.1 * 2. - y) < 0.001);
        }
    }
}
