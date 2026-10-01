//! Bounded-memory statistics. No frame decoding or report presentation here.
use serde::{Deserialize, Serialize};
use sr_noise::spatial::{detrended_tile_sigma, lower_decile_noise};

pub const METHOD: &str = "Equal-weight nearest-detector samples on a common footprint. At even N, MAD sigma per patch of (sum odd - sum even)/N; median across patches. Spatial residual RMS is reported separately. Fits use even geometric checkpoints and a fourfold window.";
pub const LIMITATION: &str = "Difference noise cancels shared scene structure AND shared systematic errors. Seeing, registration and illumination differences can remain; robust MAD measures the distribution core, not total outlier energy. No split measurement at odd N. The ideal line assumes equal independent noise and is anchored at N=2; it is not a prediction for changing frame quality.";

pub const TILE: usize = 16;
pub const PIXELS: usize = TILE * TILE;
pub const PITCH: usize = 2;
pub const DEFAULT_SAMPLES: usize = 262_144;

/// Deterministic, data-independent stratification over the central 80%.
/// Whole tiles, without overlap. Selection never favours a low-noise first frame.
pub fn tiles(width: usize, height: usize, samples: usize) -> Vec<(usize, usize)> {
    let (mx, my) = (width / 10, height / 10);
    let (nx, ny) = (
        (width - 2 * mx) / (TILE * PITCH),
        (height - 2 * my) / (TILE * PITCH),
    );
    let available = nx * ny;
    let count = (samples / PIXELS).min(available);
    (0..count)
        .map(|i| {
            let cell = ((2 * i + 1) * available / (2 * count)).min(available - 1);
            (
                mx + (cell % nx) * TILE * PITCH,
                my + (cell / nx) * TILE * PITCH,
            )
        })
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fit {
    pub amplitude: f64,
    pub exponent: f64,
    /// Descriptive only: cumulative observations are correlated, not IID trials.
    pub r_squared: f64,
    pub points: usize,
    pub first_n: usize,
    pub last_n: usize,
}

/// OLS in log-log space on geometrically spaced checkpoints, N >= 4.
/// At least five points spanning a factor of two are required.
pub fn fit(points: &[(usize, f64)]) -> Option<Fit> {
    if points.len() < 5 {
        return None;
    }
    let first_n = points.first()?.0;
    let last_n = points.last()?.0;
    if first_n < 4 || last_n < 2 * first_n {
        return None;
    }
    let mut sums = [0.0; 5];
    for &(n, noise) in points {
        if noise <= 0.0 || !noise.is_finite() {
            return None;
        }
        let (x, y) = ((n as f64).ln(), noise.ln());
        for (s, v) in sums.iter_mut().zip([x, y, x * x, x * y, y * y]) {
            *s += v;
        }
    }
    let len = points.len() as f64;
    let [sx, sy, sxx, sxy, syy] = sums;
    let xx = sxx - sx * sx / len;
    let yy = syy - sy * sy / len;
    if xx <= 0.0 {
        return None;
    }
    let xy = sxy - sx * sy / len;
    let exponent = xy / xx;
    Some(Fit {
        amplitude: ((sy - exponent * sx) / len).exp(),
        exponent,
        r_squared: if yy > 1e-20 {
            (xy * xy / (xx * yy)).clamp(0.0, 1.0)
        } else {
            0.0
        },
        points: points.len(),
        first_n,
        last_n,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Depth {
    pub frame_count: usize,
    pub frame_index: usize,
    /// Sum of known positive exposure durations; unknown durations are counted.
    pub integration_seconds: f64,
    pub unknown_exposures: usize,
    pub noise: Option<f64>,
    pub pair_noise: Option<f64>,
    pub spatial_residual: f64,
    pub ideal_noise: Option<f64>,
    pub measured_to_ideal: Option<f64>,
    pub local_fit: Option<Fit>,
    pub fit_checkpoint: bool,
}

pub struct Accumulator {
    sums: Vec<f64>,
    differences: Vec<f64>,
    previous: Vec<f32>,
    valid: Vec<bool>,
    n: usize,
    exposure: f64,
    unknown: usize,
    first_noise: f64,
    checkpoints: Vec<(usize, f64)>,
    next_checkpoint: usize,
    latest: Option<(usize, f64)>,
}

impl Accumulator {
    pub fn new(valid: Vec<bool>) -> Self {
        Self {
            sums: vec![0.0; valid.len() * PIXELS],
            differences: vec![0.0; valid.len() * PIXELS],
            previous: vec![0.0; valid.len() * PIXELS],
            valid,
            n: 0,
            exposure: 0.0,
            unknown: 0,
            first_noise: 0.0,
            checkpoints: Vec::new(),
            next_checkpoint: 4,
            latest: None,
        }
    }

    pub fn push(&mut self, values: &[f32], exposure: Option<f32>, index: usize) -> Depth {
        assert_eq!(values.len(), self.sums.len());
        self.n += 1;
        match exposure.filter(|t| t.is_finite() && *t > 0.0) {
            Some(t) => self.exposure += t as f64,
            None => self.unknown += 1,
        }
        let mut sigmas = Vec::with_capacity(self.valid.len());
        let mut tile = [0.0f32; PIXELS];
        let mut difference = [0.0f32; PIXELS];
        let mut split_sigmas = Vec::with_capacity(self.valid.len());
        let mut pair_sigmas = Vec::with_capacity(self.valid.len());
        let mut pair = [0.0f32; PIXELS];
        for (j, &valid) in self.valid.iter().enumerate() {
            if !valid {
                continue;
            }
            for (k, v) in tile.iter_mut().enumerate() {
                let i = j * PIXELS + k;
                assert!(values[i].is_finite());
                pair[k] = (values[i] - self.previous[i]) / std::f32::consts::SQRT_2;
                self.previous[i] = values[i];
                self.sums[i] += values[i] as f64;
                self.differences[i] += if self.n % 2 == 1 {
                    values[i] as f64
                } else {
                    -(values[i] as f64)
                };
                difference[k] = (self.differences[i] / self.n as f64) as f32;
                *v = (self.sums[i] / self.n as f64) as f32;
            }
            sigmas.push(detrended_tile_sigma(&tile, TILE));
            if self.n % 2 == 0 {
                split_sigmas.push(sr_core::math::mad_sigma(&difference));
                pair_sigmas.push(sr_core::math::mad_sigma(&pair));
            }
        }
        let spatial_residual = lower_decile_noise(&mut sigmas) as f64;
        // Balanced halves give exactly Var(full mean) for independent samples,
        // even with unequal frame variances. Odd counts deliberately have no
        // measurement: rescaling unequal halves would bias that identity.
        let noise = (!split_sigmas.is_empty()).then(|| sr_core::math::median(&split_sigmas) as f64);
        if let Some(noise) = noise {
            self.latest = Some((self.n, noise));
            if self.n == 2 {
                self.first_noise = noise;
            }
            if self.n >= self.next_checkpoint {
                self.checkpoints.push((self.n, noise));
                self.next_checkpoint = ((self.n as f64 * 1.25).ceil() as usize).div_ceil(2) * 2;
            }
        }
        // Local fits use bounded checkpoint history plus the measured endpoint.
        // The report plots checkpoints, not a dense staircase of refits.
        let mut local: Vec<_> = self
            .checkpoints
            .iter()
            .rev()
            .take_while(|&&(n, _)| n * 4 >= self.n)
            .copied()
            .collect();
        local.reverse();
        if let Some(noise) = noise.filter(|_| self.n >= 4) {
            if local.last().is_none_or(|p| p.0 != self.n) {
                local.push((self.n, noise));
            }
        }
        let ideal_noise = noise.map(|_| self.first_noise * (2.0 / self.n as f64).sqrt());
        Depth {
            frame_count: self.n,
            frame_index: index,
            integration_seconds: self.exposure,
            unknown_exposures: self.unknown,
            noise,
            pair_noise: (!pair_sigmas.is_empty())
                .then(|| sr_core::math::median(&pair_sigmas) as f64),
            spatial_residual,
            ideal_noise,
            measured_to_ideal: noise
                .zip(ideal_noise)
                .filter(|&(_, ideal)| ideal > 0.0)
                .map(|(noise, ideal)| noise / ideal),
            local_fit: noise.and_then(|_| fit(&local)),
            fit_checkpoint: self.checkpoints.last().is_some_and(|p| p.0 == self.n),
        }
    }

    /// Sparse, matched scene patches for display and explicit region comparisons.
    /// Pixel previews are quantized only for display; statistics stay full precision.
    pub fn snapshot(&self) -> Snapshot {
        let mut patches = Vec::new();
        for (id, &valid) in self.valid.iter().enumerate() {
            if !valid {
                continue;
            }
            let range = id * PIXELS..(id + 1) * PIXELS;
            let mean: Vec<f32> = self.sums[range.clone()]
                .iter()
                .map(|v| (v / self.n as f64) as f32)
                .collect();
            let median = sr_core::math::median(&mean);
            let spread = sr_core::math::mad_sigma(&mean).max(1e-8);
            let low = median - 8.0 * spread;
            let high = median + 8.0 * spread;
            let pixels: String = mean
                .iter()
                .map(|&v| {
                    format!(
                        "{:04x}",
                        (((v - low) / (high - low)).clamp(0.0, 1.0) * 65535.0).round() as u16
                    )
                })
                .collect();
            let noise = (self.n % 2 == 0).then(|| {
                let difference: Vec<_> = self.differences[range]
                    .iter()
                    .map(|v| (v / self.n as f64) as f32)
                    .collect();
                sr_core::math::mad_sigma(&difference)
            });
            patches.push(Patch {
                id,
                median,
                noise,
                low,
                high,
                pixels,
            });
        }
        Snapshot {
            frame_count: self.n,
            integration_seconds: self.exposure,
            patches,
        }
    }

    pub fn global_fit(&self) -> Option<Fit> {
        let mut points = self.checkpoints.clone();
        if let Some(last) = self.latest.filter(|p| p.0 >= 4 && points.last() != Some(p)) {
            points.push(last);
        }
        fit(&points)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Patch {
    pub id: usize,
    pub median: f32,
    pub noise: Option<f32>,
    pub low: f32,
    pub high: f32,
    /// 256 unsigned 16-bit values as four hex characters each; display only.
    pub pixels: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub frame_count: usize,
    pub integration_seconds: f64,
    pub patches: Vec<Patch>,
}

pub fn snapshot_due(n: usize, total: usize) -> bool {
    n.is_power_of_two() || n == total || (total % 2 == 1 && n + 1 == total)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Projection {
    pub noise_reduction_fraction: f64,
    pub additional_frames: Option<usize>,
    pub additional_hours: Option<f64>,
}

pub fn projections(model: Option<&Fit>, n: usize, seconds: f64, unknown: usize) -> Vec<Projection> {
    [0.05f64, 0.10, 0.20]
        .into_iter()
        .map(|fraction| {
            let additional_frames = model
                .filter(|m| m.exponent < -0.05 && m.r_squared >= 0.8)
                .and_then(|m| {
                    let total = n as f64 * (1.0 - fraction).powf(1.0 / m.exponent);
                    (total.is_finite() && total < 2_000_000_000.0)
                        .then(|| (total.ceil() as usize).saturating_sub(n))
                });
            Projection {
                noise_reduction_fraction: fraction,
                additional_frames,
                additional_hours: additional_frames
                    .filter(|_| n > 0 && unknown == 0)
                    .map(|extra| extra as f64 * seconds / n as f64 / 3600.0),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gaussian(state: &mut u64) -> f32 {
        let mut uniform = || {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*state >> 32) as f64) + 0.5) / 4294967296.0
        };
        ((-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos()) as f32
    }
    #[test]
    fn ideal_and_fixed_floor_are_distinguishable() {
        let mut seed = 20260923;
        let floor: Vec<_> = (0..128 * PIXELS)
            .map(|_| gaussian(&mut seed) * 0.5)
            .collect();
        let mut ideal = Accumulator::new(vec![true; 128]);
        let mut correlated = Accumulator::new(vec![true; 128]);
        let mut last = None;
        for n in 0..128 {
            let v: Vec<_> = floor.iter().map(|_| gaussian(&mut seed)).collect();
            ideal.push(&v, Some(180.0), n);
            last = Some(correlated.push(
                &v.iter().zip(&floor).map(|(a, b)| a + b).collect::<Vec<_>>(),
                Some(180.0),
                n,
            ));
        }
        assert!((ideal.global_fit().unwrap().exponent + 0.5).abs() < 0.04);
        let floor_fit = correlated.global_fit().unwrap();
        assert!((floor_fit.exponent + 0.5).abs() < 0.04, "{floor_fit:?}");
        let last = last.unwrap();
        assert!(last.spatial_residual > 3.0 * last.noise.unwrap());
    }
    #[test]
    fn accumulator_exposure_and_mean() {
        let mut a = Accumulator::new(vec![true, false]);
        let x: Vec<_> = (0..2 * PIXELS).map(|i| (i % 7) as f32).collect();
        a.push(&x, Some(30.0), 0);
        let d = a.push(&x.iter().map(|v| -v).collect::<Vec<_>>(), None, 1);
        assert_eq!(d.spatial_residual, 0.0);
        assert!(d.noise.unwrap() > 0.0);
        assert_eq!(d.integration_seconds, 30.0);
        assert_eq!(d.unknown_exposures, 1);
    }

    #[test]
    fn snapshots_preserve_common_geometry_and_physical_display_scale() {
        let mut acc = Accumulator::new(vec![true, false, true]);
        let values: Vec<_> = (0..3 * PIXELS)
            .map(|i| 0.01 + (i % 17) as f32 * 0.0001)
            .collect();
        let first = acc.push(&values, Some(30.0), 0);
        assert!(first.pair_noise.is_none());
        let second = acc.push(&values, Some(30.0), 1);
        assert_eq!(second.pair_noise, Some(0.0));
        let snap = acc.snapshot();
        assert_eq!(snap.frame_count, 2);
        assert_eq!(snap.integration_seconds, 60.0);
        assert_eq!(
            snap.patches.iter().map(|p| p.id).collect::<Vec<_>>(),
            [0, 2]
        );
        for patch in snap.patches {
            assert_eq!(patch.noise, Some(0.0));
            assert_eq!(patch.pixels.len(), PIXELS * 4);
            assert_eq!(
                patch.median,
                sr_core::math::median(&values[patch.id * PIXELS..(patch.id + 1) * PIXELS])
            );
            for (k, encoded) in patch.pixels.as_bytes().chunks_exact(4).enumerate() {
                let q = u16::from_str_radix(std::str::from_utf8(encoded).unwrap(), 16).unwrap();
                let reconstructed = patch.low + q as f32 / 65535.0 * (patch.high - patch.low);
                let original = values[patch.id * PIXELS + k];
                assert!((reconstructed - original).abs() < (patch.high - patch.low) / 65535.0);
            }
        }
        assert_eq!(
            (1..=17)
                .filter(|&n| snapshot_due(n, 17))
                .collect::<Vec<_>>(),
            [1, 2, 4, 8, 16, 17]
        );
        assert_eq!(
            (1..=19)
                .filter(|&n| snapshot_due(n, 19))
                .collect::<Vec<_>>(),
            [1, 2, 4, 8, 16, 18, 19]
        );
    }
    #[test]
    fn structured_incremental_stack_agrees_with_known_noise() {
        // Full raster reference stack, with known scene and random residual.
        // Moving compact stars model fractional sampling / changing seeing;
        // sparse extreme hits exercise contamination without clipping the stack.
        for alternating_phase in [false, true] {
            let mut seed = 19384;
            let mut acc = Accumulator::new(vec![true; 64]);
            let mut clean_sum = vec![0.0f64; 64 * PIXELS];
            let mut last = None;
            for n in 0..128 {
                let values: Vec<_> = (0..clean_sum.len())
                    .map(|i| {
                        let noise = gaussian(&mut seed);
                        clean_sum[i] += noise as f64;
                        let x = (i % TILE) as f32;
                        let y = ((i / TILE) % TILE) as f32;
                        let shift = if alternating_phase {
                            if n % 2 == 0 {
                                0.3
                            } else {
                                -0.3
                            }
                        } else {
                            (n as f32 * 1.73).sin() * 0.3
                        };
                        let star =
                            30.0 * (-((x - 7.0 - shift).powi(2) + (y - 7.0).powi(2)) / 1.5).exp();
                        let hit = if i % PIXELS == (n * 73) % PIXELS && n % 16 == 0 {
                            1000.0
                        } else {
                            0.0
                        };
                        noise + 10.0 + (x * 0.3).sin() * 5.0 + star + hit + n as f32 * 0.1
                    })
                    .collect();
                let d = acc.push(&values, Some(30.0), n);
                if n % 2 == 0 {
                    assert!(d.noise.is_none());
                }
                last = Some(d);
            }
            let truth = (clean_sum.iter().map(|v| (v / 128.0).powi(2)).sum::<f64>()
                / clean_sum.len() as f64)
                .sqrt();
            let d = last.unwrap();
            if alternating_phase {
                // A PSF mismatch locked to half membership remains contamination.
                assert!(d.noise.unwrap() > 1.1 * truth);
            } else {
                assert!(
                    (d.noise.unwrap() / truth - 1.0).abs() < 0.18,
                    "split={:?}, truth={truth}",
                    d.noise
                );
                assert!(d.spatial_residual > 3.0 * truth);
                assert!((acc.global_fit().unwrap().exponent + 0.5).abs() < 0.08);
            }
        }
    }

    #[test]
    fn unequal_noise_and_bad_new_frames_are_not_forced_downward() {
        let mut seed = 82;
        let mut acc = Accumulator::new(vec![true; 128]);
        let mut variance_sum = 0.0f64;
        let mut previous = 0.0;
        for n in 1..=66 {
            let sigma = if n > 64 {
                30.0
            } else if n % 2 == 0 {
                3.0
            } else {
                1.0
            };
            variance_sum += sigma * sigma;
            let values: Vec<_> = (0..128 * PIXELS)
                .map(|_| gaussian(&mut seed) * sigma as f32)
                .collect();
            let d = acc.push(&values, None, n);
            if n == 64 {
                previous = d.noise.unwrap();
            }
            if n == 66 {
                let expected = variance_sum.sqrt() / n as f64;
                assert!((d.noise.unwrap() / expected - 1.0).abs() < 0.08);
                assert!(d.noise.unwrap() > 2.0 * previous);
            }
        }
    }
    #[test]
    fn fits_and_projection() {
        let p: Vec<_> = [4, 8, 16, 32, 64]
            .into_iter()
            .map(|n| (n, 3.0 / (n as f64).sqrt()))
            .collect();
        let f = fit(&p).unwrap();
        assert!((f.exponent + 0.5).abs() < 1e-10);
        assert!((f.amplitude - 3.0).abs() < 1e-10);
        assert!(fit(&p[..3]).is_none());
        let proj = projections(Some(&f), 100, 360000.0, 0);
        assert_eq!(proj[1].additional_frames, Some(24));
        assert_eq!(proj[1].additional_hours, Some(24.0));
        assert!(projections(Some(&f), 100, 0.0, 1)[0]
            .additional_hours
            .is_none());
    }
    #[test]
    fn stratification_is_unique_bounded_and_deterministic() {
        let p = tiles(9576, 6388, DEFAULT_SAMPLES);
        assert_eq!(p.len() * PIXELS, DEFAULT_SAMPLES);
        assert_eq!(p, tiles(9576, 6388, DEFAULT_SAMPLES));
        let unique: std::collections::BTreeSet<_> = p.iter().collect();
        assert_eq!(p.len(), unique.len());
        assert!(p.iter().all(|&(x, y)| x + TILE < 9576 && y + TILE < 6388));
        assert!(tiles(8, 8, DEFAULT_SAMPLES).is_empty());
    }
    /// Run explicitly in release mode; excludes file I/O and sample generation.
    #[test]
    #[ignore]
    fn benchmark_linear_accumulation() {
        let mut seed = 20260923;
        for samples in [65_536, 262_144, 524_288] {
            let values: Vec<_> = (0..samples).map(|_| gaussian(&mut seed)).collect();
            for n in [250, 500, 1000] {
                let mut acc = Accumulator::new(vec![true; samples / PIXELS]);
                let start = std::time::Instant::now();
                for i in 0..n {
                    std::hint::black_box(acc.push(&values, Some(180.0), i));
                }
                println!(
                    "samples={samples}, frames={n}, analysis_seconds={:.6}",
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
}
