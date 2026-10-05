//! Phase correlation on square patches.
//!
//! Phase correlation is the workhorse here rather than feature matching. For a
//! telephoto burst the inter-frame motion is small and the scene is often
//! repetitive (foliage, brickwork, test charts) — exactly the conditions where
//! descriptor matching becomes ambiguous and a whole-patch frequency method is
//! both cheaper and better conditioned.
//!
//! Sign convention, fixed by test: [`Correlator::shift`] returns `(dx, dy)`
//! such that a point at `p` in the **target** patch corresponds to `p + (dx,
//! dy)` in the **reference** patch.

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Result of correlating one patch pair.
#[derive(Clone, Copy, Debug)]
pub struct PatchShift {
    pub dx: f32,
    pub dy: f32,
    /// Correlation peak height, normalised to the surface energy.
    pub peak: f32,
    /// Peak height divided by the strongest competing peak elsewhere in the
    /// surface. Values near 1 mean the match is ambiguous.
    pub peak_ratio: f32,
}

impl PatchShift {
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// Reusable FFT plans plus scratch buffers for one patch size.
///
/// Construct one per thread: the plans are shared cheaply, the buffers are not.
pub struct Correlator {
    n: usize,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    a: Vec<Complex32>,
    b: Vec<Complex32>,
    scratch: Vec<Complex32>,
    surface: Vec<f32>,
    /// Whitened cross-power spectrum, kept for sub-pixel refinement.
    cps: Vec<Complex32>,
    /// `exp(i2*pi*m/n)` for integer `m`.
    wtab: Vec<Complex32>,
    /// `exp(i2*pi*f*offset/n)` for each signed frequency and fine-grid offset.
    etab: Vec<Complex32>,
    /// Fine-grid offsets used by the upsampled peak search.
    offsets: Vec<f32>,
    /// Scratch for the separable upsampled DFT.
    up_t: Vec<Complex32>,
}

impl Correlator {
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 8 && n.is_multiple_of(2),
            "patch size must be even and >= 8"
        );
        let mut planner = FftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let scratch_len = fwd
            .get_inplace_scratch_len()
            .max(inv.get_inplace_scratch_len())
            .max(1);
        // Separable Hann window: without it the patch edges act as a strong
        // synthetic feature and the correlation locks onto the frame, not the
        // scene.
        let mut window = vec![0.0f32; n * n];
        let w1: Vec<f32> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
            .collect();
        for y in 0..n {
            for x in 0..n {
                window[y * n + x] = w1[y] * w1[x];
            }
        }
        // Fine grid for the upsampled peak search: +/- 0.6 px at 0.02 px.
        // Fine enough that the residual interpolation error is far below the
        // sub-pixel accuracy the reconstruction actually needs.
        let step = 0.02f32;
        let span = 0.6f32;
        let k = (2.0 * span / step).round() as usize + 1;
        let offsets: Vec<f32> = (0..k).map(|i| -span + i as f32 * step).collect();

        let tau = std::f32::consts::TAU;
        let wtab: Vec<Complex32> = (0..n)
            .map(|m| Complex32::from_polar(1.0, tau * m as f32 / n as f32))
            .collect();
        let half = (n / 2) as i64;
        let mut etab = vec![Complex32::new(0.0, 0.0); n * k];
        for fi in 0..n {
            let f = if fi as i64 >= half {
                fi as i64 - n as i64
            } else {
                fi as i64
            };
            for (ui, &off) in offsets.iter().enumerate() {
                etab[fi * k + ui] = Complex32::from_polar(1.0, tau * f as f32 * off / n as f32);
            }
        }

        Self {
            n,
            fwd,
            inv,
            window,
            a: vec![Complex32::new(0.0, 0.0); n * n],
            b: vec![Complex32::new(0.0, 0.0); n * n],
            scratch: vec![Complex32::new(0.0, 0.0); scratch_len],
            surface: vec![0.0f32; n * n],
            cps: vec![Complex32::new(0.0, 0.0); n * n],
            wtab,
            etab,
            up_t: vec![Complex32::new(0.0, 0.0); n * k],
            offsets,
        }
    }

    /// Evaluate the correlation surface on a fine grid around the integer peak
    /// and return the sub-pixel offset of its maximum.
    ///
    /// Parabolic interpolation of a phase-correlation peak is biased toward
    /// zero by several percent of a pixel, and that bias would land directly in
    /// the registration. Evaluating the inverse transform where we actually
    /// want it removes the interpolation step. The evaluation is separable, so
    /// it costs `O(n^2 k + n k^2)` rather than `O(n^2 k^2)`.
    fn refine_peak(&mut self, ix: i64, iy: i64) -> (f32, f32) {
        let n = self.n;
        let k = self.offsets.len();
        let nn = n as i64;

        // T[fy][u] = sum_fx C[fy][fx] * exp(i2*pi*fx*(ix + off_u)/n)
        for fy in 0..n {
            let row = &self.cps[fy * n..(fy + 1) * n];
            let out = &mut self.up_t[fy * k..(fy + 1) * k];
            out.fill(Complex32::new(0.0, 0.0));
            for (fx, &c) in row.iter().enumerate() {
                if c.re == 0.0 && c.im == 0.0 {
                    continue;
                }
                let f = if fx as i64 >= nn / 2 {
                    fx as i64 - nn
                } else {
                    fx as i64
                };
                let phase = self.wtab[(f * ix).rem_euclid(nn) as usize];
                let base = c * phase;
                let e = &self.etab[fx * k..(fx + 1) * k];
                for ui in 0..k {
                    out[ui] += base * e[ui];
                }
            }
        }

        // S[v][u] = sum_fy T[fy][u] * exp(i2*pi*fy*(iy + off_v)/n)
        let mut best = f32::NEG_INFINITY;
        let mut bu = 0usize;
        let mut bv = 0usize;
        let mut acc = vec![Complex32::new(0.0, 0.0); k];
        for vi in 0..k {
            acc.iter_mut().for_each(|c| *c = Complex32::new(0.0, 0.0));
            for fy in 0..n {
                let f = if fy as i64 >= nn / 2 {
                    fy as i64 - nn
                } else {
                    fy as i64
                };
                let phase = self.wtab[(f * iy).rem_euclid(nn) as usize] * self.etab[fy * k + vi];
                let t = &self.up_t[fy * k..(fy + 1) * k];
                for ui in 0..k {
                    acc[ui] += t[ui] * phase;
                }
            }
            for (ui, a) in acc.iter().enumerate().take(k) {
                let v = a.re;
                if v > best {
                    best = v;
                    bu = ui;
                    bv = vi;
                }
            }
        }
        (self.offsets[bu], self.offsets[bv])
    }

    pub fn size(&self) -> usize {
        self.n
    }

    fn fft2(&mut self, which: bool) {
        let n = self.n;
        let buf = if which { &mut self.a } else { &mut self.b };
        // Rows.
        for y in 0..n {
            self.fwd
                .process_with_scratch(&mut buf[y * n..(y + 1) * n], &mut self.scratch);
        }
        // Columns, via transpose-free strided gather.
        let mut col = vec![Complex32::new(0.0, 0.0); n];
        for x in 0..n {
            for y in 0..n {
                col[y] = buf[y * n + x];
            }
            self.fwd.process_with_scratch(&mut col, &mut self.scratch);
            for y in 0..n {
                buf[y * n + x] = col[y];
            }
        }
    }

    fn ifft2_a(&mut self) {
        let n = self.n;
        for y in 0..n {
            self.inv
                .process_with_scratch(&mut self.a[y * n..(y + 1) * n], &mut self.scratch);
        }
        let mut col = vec![Complex32::new(0.0, 0.0); n];
        for x in 0..n {
            for (y, c) in col.iter_mut().enumerate() {
                *c = self.a[y * n + x];
            }
            self.inv.process_with_scratch(&mut col, &mut self.scratch);
            for (y, c) in col.iter().enumerate() {
                self.a[y * n + x] = *c;
            }
        }
    }

    /// Correlate two `n x n` patches given as row-major slices.
    ///
    /// Returns `None` when either patch is effectively featureless, since a
    /// shift estimated from noise is worse than no estimate at all.
    pub fn shift(&mut self, reference: &[f32], target: &[f32]) -> Option<PatchShift> {
        let n = self.n;
        debug_assert_eq!(reference.len(), n * n);
        debug_assert_eq!(target.len(), n * n);

        // Remove the DC term before windowing: a brightness difference between
        // frames otherwise dominates the spectrum.
        let mr: f32 = reference.iter().sum::<f32>() / (n * n) as f32;
        let mt: f32 = target.iter().sum::<f32>() / (n * n) as f32;
        let mut energy_r = 0.0f64;
        let mut energy_t = 0.0f64;
        for i in 0..n * n {
            let wr = (reference[i] - mr) * self.window[i];
            let wt = (target[i] - mt) * self.window[i];
            energy_r += (wr * wr) as f64;
            energy_t += (wt * wt) as f64;
            self.a[i] = Complex32::new(wr, 0.0);
            self.b[i] = Complex32::new(wt, 0.0);
        }
        let denom = (n * n) as f64;
        if (energy_r / denom).sqrt() < 1e-7 || (energy_t / denom).sqrt() < 1e-7 {
            return None;
        }

        self.fft2(true);
        self.fft2(false);

        // Cross-power spectrum, whitened but gated.
        //
        // Full whitening gives every frequency bin equal weight, which is what
        // makes phase correlation insensitive to brightness and contrast
        // differences. Taken literally it also promotes bins that contain
        // nothing but leakage and rounding noise to the same authority as bins
        // carrying real structure, which wrecks the estimate on narrowband
        // content. Two guards: drop bins whose cross-magnitude is negligible
        // against the strongest bin, and roll off the top of the band, which is
        // noise-dominated in any real burst.
        let mut max_mag = 0.0f32;
        for i in 0..n * n {
            let m = self.a[i].norm() * self.b[i].norm();
            if m > max_mag {
                max_mag = m;
            }
        }
        let gate = max_mag * 1e-4;
        let half = n as i64 / 2;
        let cutoff = 0.45f32 * n as f32;
        for y in 0..n {
            let fy = if y as i64 > half {
                y as f32 - n as f32
            } else {
                y as f32
            };
            for x in 0..n {
                let fx = if x as i64 > half {
                    x as f32 - n as f32
                } else {
                    x as f32
                };
                let i = y * n + x;
                let ra = self.a[i];
                let tb = self.b[i];
                if ra.norm() * tb.norm() <= gate {
                    self.a[i] = Complex32::new(0.0, 0.0);
                    continue;
                }
                let c = ra * tb.conj();
                let m = c.norm();
                if m <= 1e-20 {
                    self.a[i] = Complex32::new(0.0, 0.0);
                    continue;
                }
                let r = (fx * fx + fy * fy).sqrt() / cutoff;
                let lp = (-0.5 * r * r * 2.0).exp();
                self.a[i] = c * (lp / m);
            }
        }
        self.cps.copy_from_slice(&self.a);
        self.ifft2_a();

        let inv_scale = 1.0 / (n * n) as f32;
        for i in 0..n * n {
            self.surface[i] = self.a[i].re * inv_scale;
        }

        // Locate the peak.
        let mut best = f32::NEG_INFINITY;
        let mut bi = 0usize;
        for (i, &v) in self.surface.iter().enumerate() {
            if v > best {
                best = v;
                bi = i;
            }
        }
        let px = bi % n;
        let py = bi / n;

        // Second-strongest peak outside a small exclusion zone, as an
        // ambiguity measure.
        let mut second = 0.0f32;
        let excl = 3i64;
        for y in 0..n {
            for x in 0..n {
                let mut dx = x as i64 - px as i64;
                let mut dy = y as i64 - py as i64;
                if dx > n as i64 / 2 {
                    dx -= n as i64;
                }
                if dx < -(n as i64) / 2 {
                    dx += n as i64;
                }
                if dy > n as i64 / 2 {
                    dy -= n as i64;
                }
                if dy < -(n as i64) / 2 {
                    dy += n as i64;
                }
                if dx.abs() <= excl && dy.abs() <= excl {
                    continue;
                }
                let v = self.surface[y * n + x];
                if v > second {
                    second = v;
                }
            }
        }

        // Unwrap the peak position into a signed shift, then refine it.
        let mut ix = px as i64;
        let mut iy = py as i64;
        if ix > half {
            ix -= n as i64;
        }
        if iy > half {
            iy -= n as i64;
        }
        let (sx, sy) = self.refine_peak(ix, iy);

        // With `R * conj(T)` the surface peaks at `-s`, where `s` is how far
        // the target's content has moved relative to the reference. A target
        // point therefore maps onto the reference by adding the peak position
        // directly.
        let dx = ix as f32 + sx;
        let dy = iy as f32 + sy;

        if std::env::var_os("SR_DEBUG_CORR").is_some() {
            eprintln!(
                "peak idx=({px},{py}) ix={ix} iy={iy} sx={sx} sy={sy} best={best} second={second}"
            );
        }
        Some(PatchShift {
            dx,
            dy,
            peak: best,
            peak_ratio: if second > 1e-9 { best / second } else { 10.0 },
        })
    }
}

/// Correlators keyed by patch size.
///
/// Patch size has to adapt to the pyramid level: a 128 px patch on a 96 px
/// image is one heavily edge-clamped probe, which is not enough to fit
/// anything. Building an FFT plan per probe would be wasteful, so plans are
/// cached per size and reused.
#[derive(Default)]
pub struct CorrelatorCache {
    map: std::collections::HashMap<usize, Correlator>,
}

impl CorrelatorCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&mut self, n: usize) -> &mut Correlator {
        self.map.entry(n).or_insert_with(|| Correlator::new(n))
    }
}

/// Largest usable patch size for an image of these dimensions.
///
/// Returns `None` when the image is too small for any patch to carry enough
/// structure to correlate. Skipping a pyramid level is better than fitting a
/// transform to clamped border pixels.
pub fn patch_for(width: usize, height: usize, want: usize) -> Option<usize> {
    let limit = (width.min(height) / 2) & !1;
    if limit < 16 {
        return None;
    }
    Some((want.min(limit).max(16)) & !1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Broadband, band-limited test content: a sum of many sinusoids with
    /// pseudo-random directions, evaluated analytically so that a fractional
    /// shift is exact rather than interpolated.
    fn textured(n: usize, ox: f32, oy: f32) -> Vec<f32> {
        let mut v = vec![0.5f32; n * n];
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        // Frequencies up to ~0.35 cycles/px keep the content inside the
        // correlator's low-pass and away from Nyquist.
        let comps: Vec<(f32, f32, f32)> = (0..48)
            .map(|_| (next() * 2.0, next() * 2.0, next()))
            .collect();
        for y in 0..n {
            for x in 0..n {
                let fx = x as f32 - ox;
                let fy = y as f32 - oy;
                let mut acc = 0.0f32;
                for &(kx, ky, ph) in &comps {
                    acc += 0.03 * (kx * fx + ky * fy + ph * std::f32::consts::TAU).sin();
                }
                v[y * n + x] = 0.5 + acc;
            }
        }
        v
    }

    #[test]
    fn mono_proxy_tracks_sky_instead_of_stationary_sensor_spikes() {
        use sr_core::{
            cfa::CfaPattern,
            frame::{NoiseModel, RawFrame},
            samples::{DefectMask, SamplePlane},
        };
        let frame = |ox: f32, oy: f32| {
            let mut data = vec![0.01; 128 * 128];
            for (sx, sy) in [(28.0, 35.0), (77.0, 43.0), (51.0, 87.0), (98.0, 94.0)] {
                for y in 0..128 {
                    for x in 0..128 {
                        let r2 = (x as f32 - sx - ox).powi(2) + (y as f32 - sy - oy).powi(2);
                        data[y * 128 + x] += 0.08 * (-r2 / 4.5).exp();
                    }
                }
            }
            for i in 0..80 {
                let x = 4 + (i * 37) % 120;
                let y = 4 + (i * 53) % 120;
                data[y * 128 + x] = 0.8;
            }
            RawFrame {
                width: 128,
                height: 128,
                cfa: CfaPattern::MONO,
                samples: SamplePlane::from_normalised(128, 128, data),
                defects: DefectMask::none(128, 128),
                noise: NoiseModel::nominal(100.0, 65535.0),
                metadata: Default::default(),
            }
        };
        let (a, b) = (frame(0.0, 0.0), frame(6.0, -4.0));
        let mut c = Correlator::new(64);
        let old = c
            .shift(&a.guide_rgb().luma().data, &b.guide_rgb().luma().data)
            .unwrap();
        assert!(
            old.magnitude() < 0.2,
            "fixture must reproduce stationary-pattern lock: {old:?}"
        );
        let corrected = c
            .shift(&a.registration_luma().data, &b.registration_luma().data)
            .unwrap();
        assert!(
            (corrected.dx + 3.0).abs() < 0.2 && (corrected.dy - 2.0).abs() < 0.2,
            "sky shift was not recovered: {corrected:?}"
        );
    }

    #[test]
    fn recovers_integer_shift_with_documented_sign() {
        let n = 64;
        let reference = textured(n, 0.0, 0.0);
        // Target content is the reference displaced by +3 in x, +2 in y:
        // a feature at reference x maps to target x + 3.
        let target = textured(n, 3.0, 2.0);
        let mut c = Correlator::new(n);
        let s = c.shift(&reference, &target).expect("shift");
        // Target point p corresponds to reference point p + (dx, dy), so a
        // feature that moved to +3 needs -3 to get back.
        //
        // The tolerance is loose on purpose. The analysis window is fixed while
        // the content moves through it, so a large *uncorrected* shift biases
        // the peak by a fraction of a pixel. The pipeline never relies on this
        // regime: it re-extracts the target through the current estimate, and
        // what it needs to be accurate on is the small residual, covered by
        // `subpixel_accuracy_on_small_residuals`.
        assert_eq!(s.dx.round(), -3.0, "dx {}", s.dx);
        assert_eq!(s.dy.round(), -2.0, "dy {}", s.dy);
        assert!((s.dx + 3.0).abs() < 0.2, "dx {}", s.dx);
        assert!((s.dy + 2.0).abs() < 0.2, "dy {}", s.dy);
    }

    #[test]
    fn recovers_subpixel_shift() {
        let n = 64;
        let reference = textured(n, 0.0, 0.0);
        let target = textured(n, 0.37, -0.62);
        let mut c = Correlator::new(n);
        let s = c.shift(&reference, &target).expect("shift");
        assert!((s.dx + 0.37).abs() < 0.06, "dx {}", s.dx);
        assert!((s.dy - 0.62).abs() < 0.06, "dy {}", s.dy);
    }

    #[test]
    fn subpixel_accuracy_on_small_residuals() {
        // The operating regime of the registration loop: sub-pixel residuals
        // after a coarse estimate has been applied. This is what limits how
        // much real resolution the merge can recover, so the bar is tight.
        let n = 128;
        let mut c = Correlator::new(n);
        let reference = textured(n, 0.0, 0.0);
        let mut worst = 0.0f32;
        for i in -5..=5 {
            let t = i as f32 * 0.1;
            let target = textured(n, t, -t * 0.5);
            let s = c.shift(&reference, &target).expect("shift");
            worst = worst.max((s.dx + t).abs()).max((s.dy - t * 0.5).abs());
        }
        assert!(worst < 0.02, "worst sub-pixel error {worst}");
    }

    #[test]
    fn flat_patches_are_rejected() {
        let n = 32;
        let flat = vec![0.5f32; n * n];
        let mut c = Correlator::new(n);
        assert!(c.shift(&flat, &flat).is_none());
    }

    #[test]
    fn noise_only_patches_score_far_below_real_matches() {
        let n = 64;
        let mut seed = 99u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let a: Vec<f32> = (0..n * n).map(|_| 0.5 + 0.01 * rnd()).collect();
        let b: Vec<f32> = (0..n * n).map(|_| 0.5 + 0.01 * rnd()).collect();
        let mut c = Correlator::new(n);
        let noise = c.shift(&a, &b).expect("shift");

        let r = textured(n, 0.0, 0.0);
        let t = textured(n, 1.0, 0.0);
        let real = c.shift(&r, &t).expect("shift");

        assert!(
            real.peak_ratio > noise.peak_ratio * 2.0,
            "real {} vs noise {}",
            real.peak_ratio,
            noise.peak_ratio
        );
    }
}
