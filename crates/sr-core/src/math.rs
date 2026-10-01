//! Small numerical helpers kept out of the hot data structures.
//!
//! Deliberately concrete: fixed-size solvers and simple robust estimators, so
//! inner loops stay inlinable and SIMD-friendly instead of dragging a general
//! linear-algebra abstraction through every pixel.

/// Solve a small dense symmetric-positive-definite system by Cholesky.
/// `a` is `n x n` row-major and is overwritten. Returns `false` if not SPD.
pub fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 1e-18 {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    // Forward substitution.
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    // Back substitution.
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    true
}

/// Eigen-decomposition of a symmetric 2x2 matrix `[[a, b], [b, c]]`.
/// Returns `(lambda1, lambda2, e1)` with `lambda1 >= lambda2` and `e1` the unit
/// eigenvector of `lambda1`.
#[inline]
pub fn eig_sym2(a: f32, b: f32, c: f32) -> (f32, f32, [f32; 2]) {
    let tr = a + c;
    let diff = a - c;
    let disc = (diff * diff + 4.0 * b * b).max(0.0).sqrt();
    let l1 = 0.5 * (tr + disc);
    let l2 = 0.5 * (tr - disc);
    // Eigenvector for l1; fall back to an axis when the matrix is isotropic.
    let (ex, ey) = if b.abs() > 1e-20 {
        (l1 - c, b)
    } else if a >= c {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let n = (ex * ex + ey * ey).sqrt();
    let e = if n > 1e-20 { [ex / n, ey / n] } else { [1.0, 0.0] };
    (l1, l2, e)
}

/// Median of a slice, by partial sort of a copy.
///
/// For an even-sized sample this is the mean of the two central values. Taking
/// the upper of the two instead biases the result upward, which matters
/// wherever a median is used as a smoother over a truncated neighbourhood.
pub fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return 0.0;
    }
    let n = s.len();
    let mid = n / 2;
    s.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let hi = s[mid];
    if n % 2 == 1 {
        return hi;
    }
    // The lower half is already partitioned below `mid`, so its maximum is the
    // other central value.
    let lo = s[..mid]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    0.5 * (lo + hi)
}

/// Median absolute deviation, scaled to be a consistent estimator of sigma for
/// Gaussian data.
pub fn mad_sigma(v: &[f32]) -> f32 {
    let m = median(v);
    let dev: Vec<f32> = v.iter().map(|x| (x - m).abs()).collect();
    1.4826 * median(&dev)
}

/// Huber weight for a residual normalised by scale.
#[inline]
pub fn huber_weight(r: f32, k: f32) -> f32 {
    let a = r.abs();
    if a <= k {
        1.0
    } else {
        k / a.max(1e-12)
    }
}

/// Tukey biweight, which fully rejects gross outliers rather than merely
/// down-weighting them.
#[inline]
pub fn tukey_weight(r: f32, k: f32) -> f32 {
    let a = (r / k).abs();
    if a >= 1.0 {
        0.0
    } else {
        let t = 1.0 - a * a;
        t * t
    }
}

/// Parabolic sub-pixel peak refinement from three samples around a maximum.
/// Returns the offset from the centre sample in `[-0.5, 0.5]`.
#[inline]
pub fn parabolic_peak(left: f32, centre: f32, right: f32) -> f32 {
    let denom = left - 2.0 * centre + right;
    if denom.abs() < 1e-20 {
        return 0.0;
    }
    let d = 0.5 * (left - right) / denom;
    d.clamp(-1.0, 1.0)
}

/// Mean and variance of a slice in one pass.
pub fn mean_var(v: &[f32]) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let n = v.len() as f64;
    let mut s = 0.0f64;
    let mut ss = 0.0f64;
    for &x in v {
        let x = x as f64;
        s += x;
        ss += x * x;
    }
    let m = s / n;
    ((m) as f32, ((ss / n) - m * m).max(0.0) as f32)
}

/// Peak signal-to-noise ratio against a reference, for data in `[0, 1]`.
pub fn psnr(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut se = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) as f64;
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse <= 1e-20 {
        return 99.0;
    }
    (10.0 * (1.0 / mse).log10()) as f32
}

/// Global SSIM with an 8x8 block approximation. Adequate for regression
/// tracking; not a substitute for the full windowed formulation.
pub fn ssim_blocks(a: &[f32], b: &[f32], width: usize, height: usize, block: usize) -> f32 {
    assert_eq!(a.len(), b.len());
    let c1 = 0.01f64 * 0.01;
    let c2 = 0.03f64 * 0.03;
    let mut total = 0.0f64;
    let mut n = 0usize;
    let mut y = 0;
    while y + block <= height {
        let mut x = 0;
        while x + block <= width {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            for dy in 0..block {
                for dx in 0..block {
                    let i = (y + dy) * width + (x + dx);
                    let va = a[i] as f64;
                    let vb = b[i] as f64;
                    sa += va;
                    sb += vb;
                    saa += va * va;
                    sbb += vb * vb;
                    sab += va * vb;
                }
            }
            let m = (block * block) as f64;
            let ma = sa / m;
            let mb = sb / m;
            let va = (saa / m - ma * ma).max(0.0);
            let vb = (sbb / m - mb * mb).max(0.0);
            let cov = sab / m - ma * mb;
            let s = ((2.0 * ma * mb + c1) * (2.0 * cov + c2))
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            total += s;
            n += 1;
            x += block;
        }
        y += block;
    }
    if n == 0 {
        return 0.0;
    }
    (total / n as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_known_system() {
        // [[4,1],[1,3]] x = [1,2]  ->  x = [1/11, 7/11]
        let mut a = vec![4.0, 1.0, 1.0, 3.0];
        let mut b = vec![1.0, 2.0];
        assert!(cholesky_solve(&mut a, &mut b, 2));
        assert!((b[0] - 1.0 / 11.0).abs() < 1e-12, "{:?}", b);
        assert!((b[1] - 7.0 / 11.0).abs() < 1e-12, "{:?}", b);
    }

    #[test]
    fn eig_sym2_orders_and_orients() {
        let (l1, l2, e) = eig_sym2(3.0, 0.0, 1.0);
        assert!((l1 - 3.0).abs() < 1e-6);
        assert!((l2 - 1.0).abs() < 1e-6);
        assert!((e[0].abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parabolic_peak_finds_offset() {
        // Parabola peaking at +0.25 from centre.
        let f = |x: f32| -(x - 0.25) * (x - 0.25);
        let d = parabolic_peak(f(-1.0), f(0.0), f(1.0));
        assert!((d - 0.25).abs() < 1e-4, "got {d}");
    }

    #[test]
    fn median_of_an_even_sample_is_the_midpoint() {
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-6);
        assert!((median(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-6);
        assert!((median(&[5.0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn mad_sigma_is_robust_to_outliers() {
        let mut v: Vec<f32> = (0..100).map(|i| (i % 2) as f32 * 0.0 + 1.0).collect();
        v[0] = 1000.0;
        assert!(mad_sigma(&v) < 1e-6);
    }
}
