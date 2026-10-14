//! Joint relative sky geometry, with shared sensor distortion per instrument.
//! Correspondences must be established independently; held-out pairs never enter
//! the fit. This solves internal alignment, not an absolute astrometric WCS.
use serde::{Deserialize, Serialize};
use sr_core::math::cholesky_solve;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleFrame {
    pub center: [f64; 2],
    pub normalization_scale: f64,
    pub instrument: usize,
    /// Sensor-normalized to anchor-normalized projective matrix.
    pub homography: [f64; 9],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct StarPair {
    pub a: usize,
    pub b: usize,
    pub source_a: [f64; 2],
    pub source_b: [f64; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleSolution {
    pub frames: Vec<BundleFrame>,
    pub distortion: Vec<[f64; 3]>,
    pub iterations: usize,
    pub converged: bool,
    pub initial_cost: f64,
    pub final_cost: f64,
}

/// Brown-Conrady correction in normalized sensor coordinates.
pub fn undistort(p: [f64; 2], k: [f64; 3]) -> [f64; 2] {
    let [x, y] = p;
    let r = x * x + y * y;
    [
        x + k[0] * x * r + 2. * k[1] * x * y + k[2] * (r + 2. * x * x),
        y + k[0] * y * r + k[1] * (r + 2. * y * y) + 2. * k[2] * x * y,
    ]
}

fn evaluate(
    frame: &BundleFrame,
    k: [f64; 3],
    p: [f64; 2],
    scale: f64,
) -> Option<([f64; 2], [[f64; 11]; 2])> {
    let [x, y] = [
        (p[0] - frame.center[0]) / frame.normalization_scale,
        (p[1] - frame.center[1]) / frame.normalization_scale,
    ];
    let r = x * x + y * y;
    let [u, v] = undistort([x, y], k);
    let h = frame.homography;
    let z = h[6] * u + h[7] * v + h[8];
    if !z.is_finite() || z <= 0.05 {
        return None;
    }
    let q = [
        (h[0] * u + h[1] * v + h[2]) / z,
        (h[3] * u + h[4] * v + h[5]) / z,
    ];
    if q.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut j = [[0.; 11]; 2];
    j[0][..8].copy_from_slice(&[
        u / z,
        v / z,
        1. / z,
        0.,
        0.,
        0.,
        -q[0] * u / z,
        -q[0] * v / z,
    ]);
    j[1][..8].copy_from_slice(&[
        0.,
        0.,
        0.,
        u / z,
        v / z,
        1. / z,
        -q[1] * u / z,
        -q[1] * v / z,
    ]);
    let b = [
        [x * r, y * r],
        [2. * x * y, r + 2. * y * y],
        [r + 2. * x * x, 2. * x * y],
    ];
    for axis in 0..2 {
        let dx = (h[axis * 3] - q[axis] * h[6]) / z;
        let dy = (h[axis * 3 + 1] - q[axis] * h[7]) / z;
        for t in 0..3 {
            j[axis][8 + t] = dx * b[t][0] + dy * b[t][1];
        }
        for v in &mut j[axis] {
            *v *= scale;
        }
    }
    let position = [q[0] * scale, q[1] * scale];
    if position
        .iter()
        .chain(j.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return None;
    }
    Some((position, j))
}

/// Error lengths on a disjoint set of correspondences, in anchor sensor pixels.
pub fn errors(solution: &BundleSolution, pairs: &[StarPair], scale: f64) -> Option<Vec<f64>> {
    pairs
        .iter()
        .map(|p| {
            let a = solution.frames.get(p.a)?;
            let b = solution.frames.get(p.b)?;
            let pa = evaluate(
                a,
                *solution.distortion.get(a.instrument)?,
                p.source_a,
                scale,
            )?
            .0;
            let pb = evaluate(
                b,
                *solution.distortion.get(b.instrument)?,
                p.source_b,
                scale,
            )?
            .0;
            Some((pa[0] - pb[0]).hypot(pa[1] - pb[1]))
        })
        .collect()
}

/// Robust damped least squares with an anchored projective gauge.
/// A result is a candidate even when converged: callers must check withheld
/// errors, graph connectivity, support, and every full-frame mapping for folds.
pub fn solve(
    frames: &[BundleFrame],
    pairs: &[StarPair],
    anchor: usize,
    instruments: usize,
    scale: f64,
    max_iterations: usize,
) -> Option<BundleSolution> {
    if instruments > frames.len() {
        return None;
    }
    solve_with_fixed_distortion(
        frames,
        pairs,
        anchor,
        instruments,
        scale,
        max_iterations,
        &vec![None; instruments],
    )
}

/// Solve while holding selected instruments' Brown coefficients exactly fixed.
/// `fixed` has one entry per instrument; `None` retains the ordinary free fit.
/// Fixed coefficients have no columns in the optimization, rather than a large
/// penalty. Fixing the anchor instrument to zero intentionally keeps its native
/// distorted raster as the relative output gauge; it does not assert physically
/// distortion-free optics. All ordinary held-out and footprint gates still apply.
pub fn solve_with_fixed_distortion(
    frames: &[BundleFrame],
    pairs: &[StarPair],
    anchor: usize,
    instruments: usize,
    scale: f64,
    max_iterations: usize,
    fixed: &[Option<[f64; 3]>],
) -> Option<BundleSolution> {
    if frames.len() < 2
        || anchor >= frames.len()
        || instruments == 0
        || instruments > frames.len()
        || fixed.len() != instruments
        || fixed.iter().flatten().flatten().any(|v| !v.is_finite())
        || !scale.is_finite()
        || scale <= 0.
        || max_iterations == 0
        || pairs.len() < 30
        || frames.iter().any(|f| {
            f.instrument >= instruments
                || !f.normalization_scale.is_finite()
                || f.normalization_scale <= 0.
                || f.center.iter().chain(&f.homography).any(|v| !v.is_finite())
                || f.homography[8].abs() < 1e-12
        })
        || pairs.iter().any(|p| {
            p.a >= frames.len()
                || p.b >= frames.len()
                || p.a == p.b
                || p.source_a.iter().chain(&p.source_b).any(|v| !v.is_finite())
        })
    {
        return None;
    }
    let mut seen = vec![false; frames.len()];
    seen[anchor] = true;
    for _ in 0..frames.len() {
        for p in pairs {
            if seen[p.a] || seen[p.b] {
                seen[p.a] = true;
                seen[p.b] = true;
            }
        }
    }
    if seen.iter().any(|v| !v) {
        return None;
    }
    let slots: Vec<_> = (0..frames.len())
        .map(|i| {
            if i == anchor {
                None
            } else {
                Some(8 * (i - usize::from(i > anchor)))
            }
        })
        .collect();
    let optical = 8 * (frames.len() - 1);
    let mut n = optical;
    let distortion_slots: Vec<_> = fixed
        .iter()
        .map(|constraint| {
            if constraint.is_some() {
                None
            } else {
                let slot = n;
                n += 3;
                Some(slot)
            }
        })
        .collect();
    if n > 1024 {
        return None;
    } // bounded dense normal matrix, not a survey-scale solver
    let mut state = BundleSolution {
        frames: frames.to_vec(),
        distortion: fixed.iter().map(|v| v.unwrap_or([0.; 3])).collect(),
        iterations: 0,
        converged: false,
        initial_cost: 0.,
        final_cost: 0.,
    };
    for f in &mut state.frames {
        let s = f.homography[8];
        for v in &mut f.homography {
            *v /= s;
        }
    }
    let robust_cost = |s: &BundleSolution| -> Option<f64> {
        let cost: f64 = errors(s, pairs, scale)?
            .iter()
            // Algebraically the same pseudo-Huber loss, without subtracting
            // nearly equal numbers near a perfect solution or squaring e.
            .map(|&e| e * (0.3 * (e / (e.hypot(0.3) + 0.3))))
            .sum();
        cost.is_finite().then_some(cost)
    };
    let mut cost = robust_cost(&state)?;
    state.initial_cost = cost;
    let mut damping = 1e-3;
    for iteration in 0..max_iterations {
        state.iterations = iteration + 1;
        let mut normal = vec![0.; n * n];
        let mut rhs = vec![0.; n];
        let mut max_residual = 0.0f64;
        for p in pairs {
            let a = &state.frames[p.a];
            let b = &state.frames[p.b];
            let (pa, ja) = evaluate(a, state.distortion[a.instrument], p.source_a, scale)?;
            let (pb, jb) = evaluate(b, state.distortion[b.instrument], p.source_b, scale)?;
            let residual = [pa[0] - pb[0], pa[1] - pb[1]];
            max_residual = max_residual.max(residual[0].hypot(residual[1]));
            let w =
                1. / (1. + (residual[0] * residual[0] + residual[1] * residual[1]) / 0.09).sqrt();
            for axis in 0..2 {
                let mut row: Vec<(usize, f64)> = Vec::with_capacity(22);
                for (index, f, j, sign) in [(p.a, a, &ja, 1.), (p.b, b, &jb, -1.)] {
                    if let Some(s) = slots[index] {
                        for (k, &derivative) in j[axis][..8].iter().enumerate() {
                            row.push((s + k, sign * derivative));
                        }
                    }
                    if let Some(s) = distortion_slots[f.instrument] {
                        for k in 0..3 {
                            let slot = s + k;
                            let value = sign * j[axis][8 + k];
                            if let Some(e) = row.iter_mut().find(|e| e.0 == slot) {
                                e.1 += value;
                            } else {
                                row.push((slot, value));
                            }
                        }
                    }
                }
                for &(i, di) in &row {
                    rhs[i] -= w * di * residual[axis];
                    for &(j, dj) in &row {
                        normal[i * n + j] += w * di * dj;
                    }
                }
            }
        }
        if normal.iter().chain(&rhs).any(|v| !v.is_finite()) {
            return None;
        }
        // An exact seed has no strictly improving LM step. Recognize numerical
        // zero before the strict decrease test; this is not a rank/support gate.
        if max_residual <= 1e-9 {
            state.converged = true;
            break;
        }
        let mut accepted = false;
        for _ in 0..12 {
            let mut matrix = normal.clone();
            let mut delta = rhs.clone();
            for i in 0..n {
                matrix[i * n + i] += damping * normal[i * n + i].max(1.);
            }
            if !cholesky_solve(&mut matrix, &mut delta, n) {
                damping *= 10.;
                continue;
            }
            let mut candidate = state.clone();
            for (i, f) in candidate.frames.iter_mut().enumerate() {
                if let Some(s) = slots[i] {
                    for k in 0..8 {
                        f.homography[k] += delta[s + k];
                    }
                }
            }
            for (i, k) in candidate.distortion.iter_mut().enumerate() {
                if let Some(s) = distortion_slots[i] {
                    for t in 0..3 {
                        k[t] += delta[s + t];
                    }
                }
            }
            let next = robust_cost(&candidate).unwrap_or(f64::INFINITY);
            if next < cost {
                let improvement = cost - next;
                state = candidate;
                cost = next;
                damping = (damping / 3.).max(1e-10);
                accepted = true;
                if improvement < 1e-8 * (1. + cost) {
                    state.converged = true;
                }
                break;
            }
            damping *= 10.;
        }
        if state.converged {
            break;
        }
        if !accepted {
            break;
        }
    }
    state.final_cost = cost;
    Some(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_native_gauge_recovers_other_instrument_on_withheld_points() {
        let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let mut frames: Vec<_> = (0..3)
            .map(|i| BundleFrame {
                center: [0., 0.],
                normalization_scale: 1000.,
                instrument: usize::from(i > 0),
                homography: identity,
            })
            .collect();
        let truth = [0.015, -0.001, 0.0005];
        let mut pairs = Vec::new();
        for i in 0..240 {
            let sensor = [(i % 20) as f64 * 0.08 - 0.8, (i / 20) as f64 * 0.1 - 0.6];
            let q = undistort(sensor, truth);
            for b in [1, 2] {
                let dx = b as f64 * 0.12;
                let dy = b as f64 * -0.07;
                pairs.push(StarPair {
                    a: 0,
                    b,
                    source_a: [(q[0] + dx) * 1000., (q[1] + dy) * 1000.],
                    source_b: [sensor[0] * 1000., sensor[1] * 1000.],
                });
                frames[b].homography[2] = dx + 0.002;
                frames[b].homography[5] = dy - 0.003;
            }
        }
        let train: Vec<_> = pairs
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 5 != 0)
            .map(|(_, p)| *p)
            .collect();
        let test: Vec<_> = pairs
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 5 == 0)
            .map(|(_, p)| *p)
            .collect();
        let fit =
            solve_with_fixed_distortion(&frames, &train, 0, 2, 1000., 150, &[Some([0.; 3]), None])
                .unwrap();
        assert!(fit.converged);
        assert_eq!(fit.frames[0].homography, identity);
        assert_eq!(fit.distortion[0], [0.; 3]);
        for (actual, expected) in fit.distortion[1].iter().zip(truth) {
            assert!((actual - expected).abs() < 1e-7);
        }
        assert!(
            errors(&fit, &test, 1000.)
                .unwrap()
                .iter()
                .all(|e| *e < 1e-5)
        );
        for fixed in [
            vec![],
            vec![None],
            vec![Some([f64::NAN, 0., 0.]), None],
            vec![None, Some([0., f64::INFINITY, 0.])],
        ] {
            assert!(
                solve_with_fixed_distortion(&frames, &train, 0, 2, 1000., 150, &fixed).is_none()
            );
        }
        let old = solve(&frames, &train, 0, 2, 1000., 150).unwrap();
        let free =
            solve_with_fixed_distortion(&frames, &train, 0, 2, 1000., 150, &[None, None]).unwrap();
        assert_eq!(old.distortion, free.distortion);
        assert_eq!(old.final_cost, free.final_cost);
        for (a, b) in old.frames.iter().zip(&free.frames) {
            assert_eq!(a.homography, b.homography);
        }
    }

    #[test]
    fn nonzero_fixed_coefficients_are_preserved_when_every_instrument_is_fixed() {
        let frame = BundleFrame {
            center: [0., 0.],
            normalization_scale: 1000.,
            instrument: 0,
            homography: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        };
        let pairs: Vec<_> = (0..100)
            .map(|i| {
                let p = [(i % 10) as f64 * 80., (i / 10) as f64 * 50.];
                StarPair {
                    a: 0,
                    b: 1,
                    source_a: p,
                    source_b: p,
                }
            })
            .collect();
        let k = [0.005, -0.001, 0.002];
        let fit = solve_with_fixed_distortion(
            &[frame.clone(), frame],
            &pairs,
            0,
            1,
            1000.,
            10,
            &[Some(k)],
        )
        .unwrap();
        assert!(fit.converged);
        assert_eq!(fit.distortion, vec![k]);
        assert_eq!(fit.final_cost, 0.);
    }

    #[test]
    fn exact_and_near_exact_seeds_converge_without_an_improving_step() {
        let frame = BundleFrame {
            center: [0., 0.],
            normalization_scale: 1000.,
            instrument: 0,
            homography: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        };
        let pairs: Vec<_> = (0..100)
            .map(|i| {
                let source = [(i % 10) as f64 * 70., (i / 10) as f64 * 60.];
                StarPair {
                    a: 0,
                    b: 1,
                    source_a: source,
                    source_b: source,
                }
            })
            .collect();
        for dx in [0., 1e-13] {
            let mut frames = vec![frame.clone(); 2];
            frames[1].homography[2] = dx;
            let fit = solve(&frames, &pairs, 0, 1, 1000., 30).unwrap();
            assert!(fit.converged);
            assert_eq!(fit.iterations, 1);
            assert_eq!(fit.frames[1].homography, frames[1].homography);
            assert_eq!(fit.initial_cost, fit.final_cost);
            if dx > 0. {
                assert!(
                    fit.final_cost > 0.,
                    "small residual costs must not cancel to zero"
                );
            }
        }
        let mut frames = vec![frame; 2];
        frames[0].center[0] = f64::NAN;
        assert!(solve(&frames, &pairs, 0, 1, 1000., 30).is_none());
        frames[0].center[0] = 0.;
        frames[0].normalization_scale = 1e-300;
        assert!(solve(&frames, &pairs, 0, 1, 1000., 30).is_none());
    }

    #[test]
    fn recovers_joint_translation_with_withheld_stars_and_refuses_disconnected_graph() {
        let frames = vec![
            BundleFrame {
                center: [0., 0.],
                normalization_scale: 1000.,
                instrument: 0,
                homography: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            },
            BundleFrame {
                center: [0., 0.],
                normalization_scale: 1000.,
                instrument: 0,
                homography: [1., 0., 0.09, 0., 1., -0.06, 0., 0., 1.],
            },
        ];
        let pairs: Vec<_> = (0..200)
            .map(|i| {
                let p = [(i % 20) as f64 * 50., (i / 20) as f64 * 80.];
                StarPair {
                    a: 0,
                    b: 1,
                    source_a: p,
                    source_b: [p[0] - 100., p[1] + 50.],
                }
            })
            .collect();
        let fit = solve(&frames, &pairs[..140], 0, 1, 1000., 150).unwrap();
        assert!(fit.final_cost < fit.initial_cost * 1e-6);
        assert!(
            errors(&fit, &pairs[140..], 1000.)
                .unwrap()
                .iter()
                .all(|e| *e < 0.01)
        );
        let mut disconnected = frames.clone();
        disconnected.push(frames[1].clone());
        assert!(solve(&disconnected, &pairs, 0, 1, 1000., 50).is_none());
    }
}
