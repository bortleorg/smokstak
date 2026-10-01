//! Optional final restoration.
//!
//! Super-resolution and sharpening are different claims and are kept apart on
//! purpose. The default output has nothing applied. When restoration is asked
//! for it is conservative, edge-aware and reported, so a reader can always tell
//! how much of the apparent detail came from the merge and how much from a
//! filter — and the unrestored result remains available.

use sr_core::plane::Plane;

/// What a restoration pass did, for the manifest.
#[derive(Clone, Copy, Debug)]
pub struct RestorationReport {
    pub amount: f32,
    pub radius_passes: usize,
    /// Mean absolute change, in output units.
    pub mean_change: f32,
    /// Largest change applied to any pixel.
    pub max_change: f32,
}

/// Edge-aware unsharp mask.
///
/// Two properties keep this honest: the correction is suppressed where the
/// local contrast is comparable to the noise, so grain is not amplified; and it
/// is limited to the local range of the image, so it cannot manufacture
/// overshoot beyond what was already there.
pub fn mild_sharpen(
    rgb: &mut [Plane<f32>; 3],
    channels: usize,
    amount: f32,
    passes: usize,
    noise_sigma: f32,
) -> RestorationReport {
    let channels = channels.clamp(1, 3);
    let amount = amount.clamp(0.0, 2.0);
    let passes = passes.clamp(1, 6);
    let mut total_change = 0.0f64;
    let mut max_change = 0.0f32;
    let n = rgb[0].data.len();

    // Work from luminance so the correction cannot shift hue. A monochrome
    // reconstruction has one plane and two empty ones, and *is* its luminance.
    let mut luma = Plane::<f32>::new(rgb[0].width, rgb[0].height);
    for i in 0..n {
        luma.data[i] = if channels == 1 {
            rgb[0].data[i]
        } else {
            0.25 * rgb[0].data[i] + 0.5 * rgb[1].data[i] + 0.25 * rgb[2].data[i]
        };
    }
    let blurred = luma.blur_n(passes);

    let (w, h) = (luma.width, luma.height);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let detail = luma.data[i] - blurred.data[i];

            // Suppress where the detail is not clearly above the noise floor.
            let t = (detail.abs() / (2.0 * noise_sigma).max(1e-6)).min(1.0);
            let gate = t * t;

            // Never push a pixel outside the range of its own neighbourhood:
            // that is what produces haloes.
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let v = luma.at_clamped(x as i64 + dx, y as i64 + dy);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let target = (luma.data[i] + amount * gate * detail).clamp(lo, hi);
            let delta = target - luma.data[i];
            if delta != 0.0 {
                total_change += delta.abs() as f64;
                max_change = max_change.max(delta.abs());
                for p in rgb.iter_mut().take(channels) {
                    p.data[i] += delta;
                }
            }
        }
    }

    RestorationReport {
        amount,
        radius_passes: passes,
        mean_change: (total_change / n as f64) as f32,
        max_change,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A monochrome product carries one plane and two empty ones. Reading the
    /// missing two to build a luminance panicked on every `--postprocess` run
    /// of a narrowband burst.
    #[test]
    fn a_monochrome_product_is_sharpened_on_the_plane_it_has() {
        let full = step(32);
        let mut mono = [full[0].clone(), Plane::new(0, 0), Plane::new(0, 0)];
        let before = mono[0].clone();
        let r = mild_sharpen(&mut mono, 1, 0.6, 2, 0.002);
        assert!(r.max_change > 0.0, "nothing was sharpened");
        assert!(mono[1].data.is_empty(), "an absent channel was written to");
        let moved = (0..before.data.len())
            .filter(|&i| (mono[0].data[i] - before.data[i]).abs() > 1e-6)
            .count();
        assert!(moved > 0);
    }

    fn step(n: usize) -> [Plane<f32>; 3] {
        let mut p = Plane::<f32>::new(n, n);
        for y in 0..n {
            for x in 0..n {
                p.data[y * n + x] = if x < n / 2 { 0.3 } else { 0.7 };
            }
        }
        let p = p.blur_n(3);
        [p.clone(), p.clone(), p]
    }

    #[test]
    fn sharpening_increases_edge_gradient() {
        let before = step(64);
        let mut after = before.clone();
        mild_sharpen(&mut after, 3, 1.0, 3, 0.002);
        let grad = |img: &[Plane<f32>; 3]| {
            let p = &img[1];
            let mut m = 0.0f32;
            for x in 1..63 {
                m = m.max((p.data[32 * 64 + x + 1] - p.data[32 * 64 + x - 1]).abs());
            }
            m
        };
        assert!(grad(&after) > grad(&before), "edge did not sharpen");
    }

    #[test]
    fn sharpening_does_not_create_overshoot() {
        let before = step(64);
        let mut after = before.clone();
        mild_sharpen(&mut after, 3, 2.0, 3, 0.002);
        let (lo_b, hi_b) = before[1].min_max();
        let (lo_a, hi_a) = after[1].min_max();
        assert!(lo_a >= lo_b - 1e-5, "undershoot: {lo_a} < {lo_b}");
        assert!(hi_a <= hi_b + 1e-5, "overshoot: {hi_a} > {hi_b}");
    }

    #[test]
    fn noise_is_not_amplified() {
        let mut seed = 5u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let n = 64;
        let mut p = Plane::filled(n, n, 0.5);
        for v in p.data.iter_mut() {
            *v += 0.004 * rnd();
        }
        let before = [p.clone(), p.clone(), p];
        let mut after = before.clone();
        // Told the noise sigma, the gate should leave grain essentially alone.
        mild_sharpen(&mut after, 3, 1.5, 3, 0.004);
        let var = |img: &[Plane<f32>; 3]| {
            let (_, v) = sr_core::math::mean_var(&img[1].data);
            v
        };
        assert!(
            var(&after) < var(&before) * 1.3,
            "noise amplified: {} -> {}",
            var(&before),
            var(&after)
        );
    }

    #[test]
    fn zero_amount_is_a_no_op() {
        let before = step(32);
        let mut after = before.clone();
        let r = mild_sharpen(&mut after, 3, 0.0, 3, 0.002);
        assert_eq!(r.mean_change, 0.0);
        for c in 0..3 {
            for i in 0..before[c].data.len() {
                assert!((before[c].data[i] - after[c].data[i]).abs() < 1e-7);
            }
        }
    }
}
