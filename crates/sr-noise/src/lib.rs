//! Stage 3: heteroscedastic sensor noise estimation.
//!
//! `var(x) = alpha * x + beta`, estimated from the burst itself rather than
//! assumed. Two independent estimators are provided because they fail in
//! different ways:
//!
//! * [`estimate_spatial`] uses within-frame local statistics. It works on a
//!   single frame but is biased upward by texture.
//! * [`estimate_temporal`] uses frame-to-frame differences at matched sites.
//!   It is nearly texture-free but needs the burst to be roughly aligned.
//!
//! The temporal estimate is preferred when available; both are reported.
//!
//! [`defects`] handles the part of a sensor's error that is not random at all:
//! sites that misread the same way in every frame. Averaging cannot remove
//! those, and aligning before averaging turns them into streaks.

pub mod defects;
pub mod spatial;

use rayon::prelude::*;
use sr_core::cfa::CfaColor;
use sr_core::frame::{NoiseModel, NoiseSource, RawFrame};
use sr_core::math;

/// One (mean, variance) observation used for the noise-level regression.
#[derive(Clone, Copy, Debug)]
struct Sample {
    mean: f32,
    var: f32,
}

/// Mean and residual variance of a `block x block` patch after removing a
/// least-squares linear plane.
///
/// Without this, any illumination gradient inside the patch is measured as
/// noise. On a real telephoto scene that bias is easily larger than the sensor
/// noise itself, so detrending is not optional. The grid is regular and
/// centred, which makes the normal equations diagonal and the fit a few sums.
fn detrended_mean_var(vals: &[f32], block: usize) -> (f32, f32) {
    let n = vals.len();
    debug_assert_eq!(n, block * block);
    if n < 4 {
        return math::mean_var(vals);
    }
    let c = (block - 1) as f64 * 0.5;
    let (mut s, mut ss, mut su, mut sw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut suu = 0.0f64;
    for by in 0..block {
        let wv = by as f64 - c;
        for bx in 0..block {
            let u = bx as f64 - c;
            let v = vals[by * block + bx] as f64;
            s += v;
            ss += v * v;
            su += u * v;
            sw += wv * v;
        }
    }
    // sum(u^2) over the whole patch, identical for both axes by symmetry.
    for bx in 0..block {
        let u = bx as f64 - c;
        suu += u * u;
    }
    suu *= block as f64;

    let nf = n as f64;
    let c0 = s / nf;
    let c1 = if suu > 0.0 { su / suu } else { 0.0 };
    let c2 = if suu > 0.0 { sw / suu } else { 0.0 };
    let sse = (ss - nf * c0 * c0 - c1 * c1 * suu - c2 * c2 * suu).max(0.0);
    let dof = (nf - 3.0).max(1.0);
    (c0 as f32, (sse / dof) as f32)
}

/// Weighted least-squares fit of `var = alpha * mean + beta` over the lower
/// envelope of the observations.
///
/// The lower envelope matters: a block containing an edge has a variance far
/// above the noise floor, so a plain fit measures texture, not the sensor. We
/// bin by intensity, take a low percentile of the variance in each bin, and fit
/// to that.
fn fit_lower_envelope(samples: &[Sample], bins: usize, percentile: f32) -> Option<(f32, f32)> {
    if samples.len() < 32 {
        return None;
    }
    let (lo, hi) = samples
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), s| {
            (a.min(s.mean), b.max(s.mean))
        });
    // Negated deliberately: if either end is NaN there is no range to bucket
    // over, and `hi <= lo` would answer that question with `false`.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(hi > lo) {
        return None;
    }
    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); bins];
    for s in samples {
        let t = ((s.mean - lo) / (hi - lo) * (bins - 1) as f32).round() as usize;
        buckets[t.min(bins - 1)].push(s.var);
    }

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut ws: Vec<f64> = Vec::new();
    for (i, b) in buckets.iter_mut().enumerate() {
        if b.len() < 8 {
            continue;
        }
        b.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let k = ((b.len() - 1) as f32 * percentile).round() as usize;
        let centre = lo + (hi - lo) * (i as f32 / (bins - 1) as f32);
        xs.push(centre as f64);
        ys.push(b[k] as f64);
        ws.push(b.len() as f64);
    }
    if xs.len() < 3 {
        return None;
    }

    // Weighted linear regression, then one reweighting pass that suppresses
    // bins still sitting well above the fitted line.
    let mut alpha = 0.0f64;
    let mut beta = 0.0f64;
    let mut w = ws.clone();
    for _ in 0..3 {
        let sw: f64 = w.iter().sum();
        let sx: f64 = xs.iter().zip(&w).map(|(x, w)| x * w).sum();
        let sy: f64 = ys.iter().zip(&w).map(|(y, w)| y * w).sum();
        let sxx: f64 = xs.iter().zip(&w).map(|(x, w)| x * x * w).sum();
        let sxy: f64 = xs.iter().zip(ys.iter()).zip(&w).map(|((x, y), w)| x * y * w).sum();
        let den = sw * sxx - sx * sx;
        if den.abs() < 1e-18 {
            return None;
        }
        alpha = (sw * sxy - sx * sy) / den;
        beta = (sy - alpha * sx) / sw;
        for i in 0..xs.len() {
            let pred = alpha * xs[i] + beta;
            let r = ys[i] - pred;
            // Only positive residuals are suspicious: texture inflates variance.
            w[i] = ws[i] * if r > 0.0 { 1.0 / (1.0 + 4.0 * (r / pred.abs().max(1e-9))) } else { 1.0 };
        }
    }
    let alpha = alpha.max(0.0) as f32;
    let beta = beta.max(1e-10) as f32;
    if !alpha.is_finite() || !beta.is_finite() {
        return None;
    }
    Some((alpha, beta))
}

/// Single-frame estimate from local statistics of same-colour sites.
///
/// Blocks are read per CFA channel with a stride of 2, so mosaic structure is
/// never mistaken for signal variance.
pub fn estimate_spatial(frame: &RawFrame, block: usize) -> Option<NoiseModel> {
    let (w, h) = (frame.width, frame.height);
    let step = block * 2;
    let mut samples: Vec<Sample> = Vec::new();

    let mut y0 = 0;
    while y0 + step <= h {
        let mut x0 = 0;
        while x0 + step <= w {
            for cell in 0..4 {
                let ox = cell % 2;
                let oy = cell / 2;
                let mut vals: Vec<f32> = Vec::with_capacity(block * block);
                let mut ok = true;
                for by in 0..block {
                    for bx in 0..block {
                        let x = x0 + ox + 2 * bx;
                        let y = y0 + oy + 2 * by;
                        let i = y * w + x;
                        let v = frame.value(x, y);
                        if !frame.usable_value(i, v) {
                            ok = false;
                            break;
                        }
                        vals.push(v);
                    }
                    if !ok {
                        break;
                    }
                }
                if !ok || vals.len() < 16 {
                    continue;
                }
                let (m, v) = detrended_mean_var(&vals, block);
                if m > 0.0 && m < 0.95 {
                    samples.push(Sample { mean: m, var: v });
                }
            }
            x0 += step;
        }
        y0 += step;
    }

    // Detrending removes smooth gradients; a low percentile still guards
    // against blocks containing real high-frequency texture.
    fit_lower_envelope(&samples, 24, 0.15)
        .map(|(a, b)| NoiseModel::new(a, b, NoiseSource::Measured))
}

/// Burst estimate from temporal differences between two frames at the same
/// sensor sites.
///
/// If the burst has sub-pixel motion the difference also contains a signal
/// term, so blocks whose spatial gradient is high are discarded: what is left
/// is dominated by sensor noise.
pub fn estimate_temporal(frames: &[&RawFrame], block: usize, max_pairs: usize) -> Option<NoiseModel> {
    if frames.len() < 2 {
        return None;
    }
    let (w, h) = (frames[0].width, frames[0].height);
    let pairs: Vec<(usize, usize)> = (0..frames.len() - 1)
        .map(|i| (i, i + 1))
        .take(max_pairs)
        .collect();

    let step = block * 2;
    let samples: Vec<Sample> = pairs
        .par_iter()
        .flat_map(|&(ia, ib)| {
            let a = frames[ia];
            let b = frames[ib];
            let mut local: Vec<Sample> = Vec::new();
            let mut y0 = 0;
            while y0 + step <= h {
                let mut x0 = 0;
                while x0 + step <= w {
                    for cell in 0..4 {
                        let ox = cell % 2;
                        let oy = cell / 2;
                        let mut diffs: Vec<f32> = Vec::with_capacity(block * block);
                        let mut means: Vec<f32> = Vec::with_capacity(block * block);
                        let mut grad = 0.0f32;
                        let mut ok = true;
                        for by in 0..block {
                            for bx in 0..block {
                                let x = x0 + ox + 2 * bx;
                                let y = y0 + oy + 2 * by;
                                let i = y * w + x;
                                let va = a.value(x, y);
                                let vb = b.value(x, y);
                                if !a.usable_value(i, va) || !b.usable_value(i, vb) {
                                    ok = false;
                                    break;
                                }
                                diffs.push(va - vb);
                                means.push(0.5 * (va + vb));
                                if bx > 0 {
                                    grad += (va - a.value(x - 2, y)).abs();
                                }
                            }
                            if !ok {
                                break;
                            }
                        }
                        if !ok || diffs.len() < 16 {
                            continue;
                        }
                        let (m, _) = math::mean_var(&means);
                        // Reject textured blocks: with sub-pixel shift they
                        // would report scene detail as noise.
                        let g = grad / diffs.len() as f32;
                        if g > 0.01 || !(m > 0.0 && m < 0.95) {
                            continue;
                        }
                        // var(a - b) = 2 * var(noise) for independent frames.
                        let sigma = math::mad_sigma(&diffs);
                        local.push(Sample { mean: m, var: 0.5 * sigma * sigma });
                    }
                    x0 += step;
                }
                y0 += step;
            }
            local
        })
        .collect();

    // Temporal differences are already texture-suppressed, so a mid percentile
    // is appropriate here rather than a low one.
    fit_lower_envelope(&samples, 24, 0.5)
        .map(|(a, b)| NoiseModel::new(a, b, NoiseSource::Measured))
}

/// Estimate a model for the burst, preferring temporal evidence and falling
/// back through spatial statistics to the ISO-derived nominal model.
pub fn estimate_burst(frames: &[RawFrame]) -> (NoiseModel, String) {
    let refs: Vec<&RawFrame> = frames.iter().take(8).collect();
    estimate_burst_refs(&refs)
}

/// The same estimator over a selected population, without cloning full frames.
/// The caller supplies at least one frame, in the same order as reconstruction.
pub fn estimate_burst_refs(frames: &[&RawFrame]) -> (NoiseModel, String) {
    let selected = &frames[..frames.len().min(8)];
    if let Some(m) = estimate_temporal(selected, 8, 4) {
        if m.alpha.is_finite() && m.beta.is_finite() && m.alpha >= 0.0 {
            return (m, "temporal frame differences".to_string());
        }
    }
    if let Some(m) = estimate_spatial(frames[0], 8) {
        return (m, "single-frame local statistics".to_string());
    }
    (frames[0].noise, "ISO-derived nominal model".to_string())
}

/// Signal-to-noise ratio of a mid-grey patch, for human-readable reporting.
pub fn snr_at(model: &NoiseModel, level: f32) -> f32 {
    level / model.std_dev(level).max(1e-12)
}

/// Per-channel mean level of a frame; used to sanity check the noise fit.
pub fn channel_means(frame: &RawFrame) -> [f32; 3] {
    let mut sum = [0.0f64; 3];
    let mut cnt = [0u64; 3];
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = y * frame.width + x;
            let v = frame.value(x, y);
            if !frame.usable_value(i, v) {
                continue;
            }
            let c = frame.cfa.color_at(x, y).index();
            sum[c] += v as f64;
            cnt[c] += 1;
        }
    }
    [
        (sum[0] / cnt[0].max(1) as f64) as f32,
        (sum[1] / cnt[1].max(1) as f64) as f32,
        (sum[2] / cnt[2].max(1) as f64) as f32,
    ]
}

/// Convenience: the colour of a site, re-exported so callers of this crate do
/// not need `sr_core::cfa` directly for simple reporting.
pub fn color_name(c: CfaColor) -> &'static str {
    c.name()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::cfa::CfaPattern;
    use sr_core::samples::{DefectMask, SamplePlane};

    /// Deterministic normal deviates: splitmix64 plus Box-Muller, so the
    /// planted noise level is exact and the test measures the estimator, not
    /// the generator.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            // Open interval, so ln() below never sees zero.
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f32 {
            let u1 = self.uniform();
            let u2 = self.uniform();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        }
    }

    fn synth_frame(w: usize, h: usize, alpha: f32, beta: f32, seed: u64) -> RawFrame {
        let mut rng = Rng(seed);
        let mut data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                // Smooth ramp so the lower-envelope fit has a range of levels
                // to work with but almost no high-frequency texture.
                let level = 0.05 + 0.8 * (x as f32 / w as f32);
                let sigma = (alpha * level + beta).sqrt();
                data[y * w + x] = level + sigma * rng.normal();
            }
        }
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_normalised(w, h, data),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            noise: NoiseModel::nominal(100.0, 16000.0),
            metadata: Default::default(),
        }
    }

    #[test]
    fn spatial_estimate_recovers_planted_noise() {
        let alpha = 2.0e-5;
        let beta = 4.0e-6;
        let f = synth_frame(512, 512, alpha, beta, 12345);
        let m = estimate_spatial(&f, 8).expect("fit");
        // Order-of-magnitude agreement is the bar here: the estimator sees a
        // ramp, and a low percentile deliberately biases slightly low.
        let sigma_true = (alpha * 0.5 + beta).sqrt();
        let sigma_est = m.std_dev(0.5);
        let ratio = sigma_est / sigma_true;
        assert!(ratio > 0.6 && ratio < 1.6, "sigma ratio {ratio} (est {sigma_est}, true {sigma_true})");
    }

    #[test]
    fn temporal_estimate_recovers_planted_noise() {
        let alpha = 2.0e-5;
        let beta = 4.0e-6;
        let a = synth_frame(512, 512, alpha, beta, 1);
        let b = synth_frame(512, 512, alpha, beta, 999);
        let m = estimate_temporal(&[&a, &b], 8, 1).expect("fit");
        let sigma_true = (alpha * 0.5 + beta).sqrt();
        let ratio = m.std_dev(0.5) / sigma_true;
        assert!(ratio > 0.75 && ratio < 1.35, "sigma ratio {ratio}");
    }

    #[test]
    fn borrowed_population_uses_identical_noise_evidence_and_fallback() {
        let frames: Vec<_> = (0..10).map(|seed|
            synth_frame(256, 256, 2.0e-5, 4.0e-6, seed + 1)
        ).collect();
        let refs: Vec<_> = frames.iter().collect();
        let owned = estimate_burst(&frames);
        let borrowed = estimate_burst_refs(&refs);
        assert_eq!(owned, borrowed);
        assert_eq!(borrowed.1, "temporal frame differences");
        assert_eq!(borrowed, estimate_burst_refs(&refs[..8]));
        // An image too small for spatial/temporal evidence uses that selected
        // frame's nominal model, never an unrelated frame outside the slice.
        let tiny = vec![synth_frame(8, 8, 2.0e-5, 4.0e-6, 999)];
        let fallback = estimate_burst_refs(&[&tiny[0]]);
        assert_eq!(fallback, estimate_burst(&tiny));
        assert_eq!(fallback.0, tiny[0].noise);
        assert_eq!(fallback.1, "ISO-derived nominal model");
    }
}
