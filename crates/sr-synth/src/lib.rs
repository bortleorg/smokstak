//! Stage 0: the synthetic validation harness.
//!
//! Ground truth first. Without a burst whose geometry, blur, noise and content
//! we chose ourselves, "it looks sharper" is the only available verdict, and
//! that verdict cannot distinguish reconstruction from sharpening.
//!
//! The generator runs the forward model the reconstruction inverts:
//!
//! ```text
//! latent HR scene -> geometric warp -> optical PSF -> CFA sampling -> noise
//! ```
//!
//! and hands back both the synthetic RAW frames and the band-limited truth the
//! reconstruction should be compared against.

pub mod metrics;
pub mod scene;

use rand::Rng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use sr_core::cfa::CfaPattern;
use sr_core::frame::{FrameMetadata, NoiseModel, NoiseSource, RawFrame};
use sr_core::geometry::{DeformationField, GlobalTransform, WarpField};
use sr_core::plane::Plane;
use sr_core::samples::{DefectMask, Levels, SamplePlane};

pub use scene::LatentScene;

#[derive(Clone, Debug)]
pub struct SynthConfig {
    /// Sensor frame edge in sensor pixels.
    pub sensor: usize,
    /// Latent pixels per sensor pixel. Also the scale the reconstruction is
    /// expected to recover.
    pub scale: f32,
    pub frames: usize,
    /// Standard deviation of inter-frame translation, in sensor pixels.
    pub shift_sigma: f32,
    /// Standard deviation of inter-frame rotation, in degrees.
    pub rotation_sigma: f32,
    /// Optical PSF sigma, in sensor pixels.
    pub psf_sigma: f32,
    /// Amplitude of a smooth per-frame local deformation, in sensor pixels.
    pub local_warp_amp: f32,
    pub noise: NoiseModel,
    /// Relative exposure jitter, e.g. 0.02 for +/-2%.
    pub exposure_jitter: f32,
    /// Additive background that ramps from zero on the first frame to this
    /// fraction of full scale on the last, as a night sky brightens.
    ///
    /// Distinct from `exposure_jitter` on purpose: one scales the scene, the
    /// other adds to it, and a correction that assumes the wrong one of the two
    /// mis-scales everything real in the frame.
    pub sky_drift: f32,
    pub hot_pixels: usize,
    /// Generate a sensor with no colour filter array: one channel, sampled at
    /// every site.
    pub mono: bool,
    /// Paint a moving object into a fraction of the frames.
    pub moving_object: bool,
    pub seed: u64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            sensor: 256,
            scale: 2.0,
            frames: 16,
            shift_sigma: 1.5,
            rotation_sigma: 0.0,
            psf_sigma: 0.6,
            local_warp_amp: 0.0,
            noise: NoiseModel::new(2.0e-5, 4.0e-6, NoiseSource::Manual),
            exposure_jitter: 0.0,
            sky_drift: 0.0,
            hot_pixels: 0,
            mono: false,
            moving_object: false,
            seed: 0xC0FFEE,
        }
    }
}

/// Dark-current offset planted at a hot site, in normalised units.
///
/// Exposed so that a harness can tell a site it should have found from one
/// buried in a highlight that was already at white, which is neither
/// detectable nor harmful.
pub const HOT_PIXEL_EXCESS: f32 = 0.35;

/// A generated burst plus the truth it was generated from.
pub struct SynthBurst {
    pub frames: Vec<RawFrame>,
    /// True sensor-coordinate warps, for scoring the registration.
    pub true_warps: Vec<WarpField>,
    /// Latent scene band-limited by the optical PSF and resampled onto the
    /// output grid: the fair target for the reconstruction.
    pub truth: [Plane<f32>; 3],
    /// Unblurred latent, for reference.
    pub latent: LatentScene,
    pub output_width: usize,
    pub output_height: usize,
    pub exposure_scale: Vec<f32>,
    /// Sensor indices of the planted hot sites, so a detector can be scored
    /// against them. The frames themselves do not carry this: a camera does
    /// not hand over a list of its bad pixels, and generating one would hide
    /// the problem a stacker has to solve.
    pub hot_pixels: Vec<usize>,
}

/// Gaussian blur of a plane by repeated binomial passes, with a fractional
/// final pass so that the effective sigma is continuous.
fn gaussian_blur(p: &Plane<f32>, sigma: f32) -> Plane<f32> {
    if sigma <= 1e-3 {
        return p.clone();
    }
    // Each `blur3` pass adds variance 0.5.
    let passes = (2.0 * sigma * sigma).round().max(1.0) as usize;
    p.blur_n(passes)
}

/// Sample a latent RGB channel at a sub-pixel position.
#[inline]
fn sample(p: &Plane<f32>, x: f32, y: f32) -> f32 {
    p.bilinear(
        x.clamp(0.0, (p.width - 1) as f32),
        y.clamp(0.0, (p.height - 1) as f32),
    )
}

/// Generate a synthetic burst.
pub fn generate(cfg: &SynthConfig) -> SynthBurst {
    let hr = (cfg.sensor as f32 * cfg.scale).round() as usize;
    let latent = scene::build(hr, cfg.scale, cfg.seed);

    // Optical band limit, applied in latent pixels.
    let psf_hr = cfg.psf_sigma * cfg.scale;
    // The sensor's own pixel aperture integrates over one sensor pixel; a
    // Gaussian of this sigma is the usual stand-in for that box.
    let aperture_hr = cfg.scale * 0.29;
    let total_sigma = (psf_hr * psf_hr + aperture_hr * aperture_hr).sqrt();
    let blurred: Vec<Plane<f32>> = latent
        .rgb
        .iter()
        .map(|p| gaussian_blur(p, total_sigma))
        .collect();

    // The truth is the band-limited scene on the output grid, which is the
    // latent grid: an honest reconstruction cannot beat this.
    let truth = [blurred[0].clone(), blurred[1].clone(), blurred[2].clone()];

    let mut rng = Pcg64Mcg::new(cfg.seed as u128 | 1);
    let mut transforms = Vec::with_capacity(cfg.frames);
    let mut exposure = Vec::with_capacity(cfg.frames);
    let mut locals: Vec<Option<DeformationField>> = Vec::with_capacity(cfg.frames);

    for i in 0..cfg.frames {
        // Frame 0 is the anchor, so the true warps are exactly identity for it
        // and the registration has an unambiguous target.
        let (dx, dy, rot) = if i == 0 {
            (0.0, 0.0, 0.0)
        } else {
            (
                rng.gen_range(-1.0..1.0) * cfg.shift_sigma,
                rng.gen_range(-1.0..1.0) * cfg.shift_sigma,
                rng.gen_range(-1.0..1.0) * cfg.rotation_sigma.to_radians(),
            )
        };
        // Rotation about the frame centre rather than the origin.
        let c = cfg.sensor as f32 * 0.5;
        let to_centre = GlobalTransform::translation(-c, -c);
        let back = GlobalTransform::translation(c + dx, c + dy);
        let rotate = GlobalTransform::similarity(rot, 1.0, 0.0, 0.0);
        transforms.push(back.compose(&rotate.compose(&to_centre)));

        exposure.push(if cfg.exposure_jitter > 0.0 && i > 0 {
            1.0 + rng.gen_range(-1.0..1.0) * cfg.exposure_jitter
        } else {
            1.0
        });

        if cfg.local_warp_amp > 0.0 && i > 0 {
            let spacing = 32.0f32;
            let gw = (cfg.sensor as f32 / spacing).ceil() as usize + 1;
            let gh = gw;
            let mut d = DeformationField::zeros((0.0, 0.0), spacing, gw, gh);
            // A smooth, frame-specific deformation, as atmospheric seeing
            // produces: low spatial frequency, not per-pixel jitter.
            let px: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let py: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            for gy in 0..gh {
                for gx in 0..gw {
                    let fx = gx as f32 * spacing / cfg.sensor as f32 * std::f32::consts::TAU;
                    let fy = gy as f32 * spacing / cfg.sensor as f32 * std::f32::consts::TAU;
                    d.u[gy * gw + gx] = [
                        cfg.local_warp_amp * (fy + px).sin(),
                        cfg.local_warp_amp * (fx + py).cos(),
                    ];
                    d.conf[gy * gw + gx] = 1.0;
                }
            }
            locals.push(Some(d));
        } else {
            locals.push(None);
        }
    }

    let true_warps: Vec<WarpField> = transforms
        .iter()
        .zip(locals.iter())
        .map(|(t, l)| WarpField {
            global: *t,
            local: l.clone(),
        })
        .collect();

    // Hot pixels are a property of the sensor, so they sit at the same sites in
    // every frame — which is exactly what makes them dangerous for a stacker
    // that aligns frames before rejecting outliers.
    let mut hot: Vec<usize> = Vec::with_capacity(cfg.hot_pixels);
    for _ in 0..cfg.hot_pixels {
        hot.push(rng.gen_range(0..cfg.sensor * cfg.sensor));
    }

    let cfa = if cfg.mono {
        CfaPattern::MONO
    } else {
        CfaPattern::RGGB
    };
    let frames: Vec<RawFrame> = (0..cfg.frames)
        .into_par_iter()
        .map(|i| {
            let mut frng = Pcg64Mcg::new((cfg.seed ^ ((i as u64 + 1) << 32)) as u128 | 1);
            let warp = &true_warps[i];
            let inv = warp.global.inverse().unwrap();
            let n = cfg.sensor;
            // Generated as 16-bit integers, which is what a sensor delivers and
            // what the pipeline stores. Quantisation is part of the forward
            // model, not an artefact of it.
            let mut raw = vec![0u16; n * n];
            let defects = DefectMask::none(n, n);
            let exp = exposure[i];
            let sky = if cfg.frames > 1 {
                cfg.sky_drift * i as f32 / (cfg.frames - 1) as f32
            } else {
                0.0
            };
            let white = 65535.0f32;

            // A moving object crossing the frame in later exposures.
            let obj = if cfg.moving_object && i >= cfg.frames / 2 {
                let t = (i - cfg.frames / 2) as f32 / (cfg.frames as f32 * 0.5).max(1.0);
                Some((
                    n as f32 * (0.15 + 0.6 * t),
                    n as f32 * 0.55,
                    n as f32 * 0.06,
                ))
            } else {
                None
            };

            for y in 0..n {
                for x in 0..n {
                    let idx = y * n + x;
                    let c = cfa.color_at(x, y).index();

                    // Sensor site -> reference sensor coords -> latent coords.
                    // The local field is defined in reference coordinates, so
                    // it is applied after the global part, exactly as the
                    // reconstruction assumes.
                    let (rx, ry) = {
                        let (gx, gy) = warp.global.apply(x as f32, y as f32);
                        match &warp.local {
                            Some(d) => {
                                let (ux, uy) = d.sample(gx, gy);
                                (gx + ux, gy + uy)
                            }
                            None => (gx, gy),
                        }
                    };
                    let hx = (rx + 0.5) * cfg.scale - 0.5;
                    let hy = (ry + 0.5) * cfg.scale - 0.5;

                    let mut v = sample(&blurred[c], hx, hy) * exp + sky;

                    if let Some((ox, oy, orad)) = obj {
                        let d = ((x as f32 - ox).powi(2) + (y as f32 - oy).powi(2)).sqrt();
                        if d < orad {
                            v = 0.85;
                        }
                    }

                    // Shot plus read noise, from the model the pipeline will
                    // later try to estimate from the data.
                    let sigma = cfg.noise.std_dev(v);
                    let u1: f64 = frng.gen_range(1e-9..1.0);
                    let u2: f64 = frng.gen_range(0.0..1.0);
                    let gauss =
                        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32;
                    v += sigma * gauss;

                    if hot.contains(&idx) {
                        // A dark-current offset, which is what a hot site
                        // actually is, added before the sensor's own clipping.
                        // Not pinned to full scale: a site sitting at white is
                        // discarded as unusable before any detector sees it,
                        // and planting one would test the saturation rule
                        // rather than the detector.
                        v += HOT_PIXEL_EXCESS;
                    }
                    // Quantise to the sensor's own levels. Saturation and black
                    // clipping need no flag: they are what the extreme codes
                    // mean.
                    let code = (v * white).round().clamp(0.0, white);
                    raw[idx] = code as u16;
                }
            }

            let _ = inv;
            RawFrame {
                width: n,
                height: n,
                samples: SamplePlane::from_u16(n, n, raw, Levels::new([0.0; 4], [white; 4])),
                cfa,
                defects,
                noise: cfg.noise,
                metadata: FrameMetadata {
                    file_name: format!("synthetic-{i:04}.raw"),
                    make: "smokstak".into(),
                    model: "synthetic".into(),
                    clean_model: "synthetic".into(),
                    iso: Some(100.0),
                    exposure_time: Some(0.01),
                    aperture: Some(5.6),
                    focal_length: Some(400.0),
                    wb_coeffs: [1.0, 1.0, 1.0],
                    xyz_to_cam: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    white_level: 16383.0,
                    crop: (0, 0, n, n),
                    full_width: n,
                    full_height: n,
                    ..Default::default()
                },
            }
        })
        .collect();

    SynthBurst {
        frames,
        true_warps,
        truth,
        latent,
        output_width: hr,
        output_height: hr,
        exposure_scale: exposure.iter().map(|e| 1.0 / e).collect(),
        hot_pixels: hot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_frames_have_the_requested_geometry() {
        let cfg = SynthConfig {
            sensor: 128,
            frames: 6,
            ..Default::default()
        };
        let b = generate(&cfg);
        assert_eq!(b.frames.len(), 6);
        assert_eq!(b.frames[0].width, 128);
        assert_eq!(b.output_width, 256);
        assert_eq!(b.truth[0].dims(), (256, 256));
    }

    #[test]
    fn frame_zero_is_the_anchor() {
        let b = generate(&SynthConfig {
            sensor: 64,
            frames: 4,
            ..Default::default()
        });
        let (x, y) = b.true_warps[0].map(10.0, 20.0);
        assert!((x - 10.0).abs() < 1e-6 && (y - 20.0).abs() < 1e-6);
    }

    #[test]
    fn frames_actually_differ_by_the_requested_shifts() {
        let cfg = SynthConfig {
            sensor: 64,
            frames: 8,
            shift_sigma: 2.0,
            ..Default::default()
        };
        let b = generate(&cfg);
        let shifts: Vec<f32> = b
            .true_warps
            .iter()
            .map(|w| {
                let (dx, dy) = w.global.shift();
                (dx * dx + dy * dy).sqrt()
            })
            .collect();
        assert!(
            shifts[1..].iter().any(|&s| s > 0.3),
            "no motion generated: {shifts:?}"
        );
        assert!(
            shifts.iter().all(|&s| s < 4.0),
            "motion exceeded the request: {shifts:?}"
        );
    }

    #[test]
    fn noise_is_present_and_of_the_requested_size() {
        let cfg = SynthConfig {
            sensor: 128,
            frames: 2,
            shift_sigma: 0.0,
            noise: NoiseModel::new(0.0, 1.0e-4, NoiseSource::Manual),
            ..Default::default()
        };
        let b = generate(&cfg);
        // Two frames at identical geometry differ only by noise.
        let n = b.frames[0].samples.len();
        let mut acc = 0.0f64;
        for i in 0..n {
            let d = (b.frames[0].value_at(i) - b.frames[1].value_at(i)) as f64;
            acc += d * d;
        }
        let sigma = (acc / n as f64 / 2.0).sqrt() as f32;
        let want = 1.0e-2;
        assert!(
            (sigma / want - 1.0).abs() < 0.15,
            "noise sigma {sigma} vs requested {want}"
        );
    }

    #[test]
    fn a_monochrome_sensor_samples_every_site() {
        let colour = generate(&SynthConfig {
            sensor: 64,
            frames: 2,
            ..Default::default()
        });
        let mono = generate(&SynthConfig {
            sensor: 64,
            frames: 2,
            mono: true,
            ..Default::default()
        });
        assert!(!colour.frames[0].is_mono());
        assert!(mono.frames[0].is_mono());
        assert_eq!(mono.frames[0].channels(), 1);
        // Every site carries the same latent channel, so the frame has no
        // mosaic structure: neighbouring values differ by scene and noise, not
        // by which filter they sat under.
        let f = &mono.frames[0];
        for y in 0..64 {
            for x in 0..64 {
                assert_eq!(f.channel_at(x, y), 0);
            }
        }
    }

    #[test]
    fn hot_pixels_read_high_at_the_same_sites_in_every_frame() {
        let cfg = SynthConfig {
            sensor: 64,
            frames: 3,
            hot_pixels: 10,
            ..Default::default()
        };
        let b = generate(&cfg);
        assert!(!b.hot_pixels.is_empty());
        // The frames must not advertise them: the point of the fixture is that
        // a detector has to find these the way it would on a real sensor.
        assert!(b.frames.iter().all(|f| f.defects.is_empty()));
        for &i in &b.hot_pixels {
            let (x, y) = (i % 64, i / 64);
            for f in &b.frames {
                let v = f.value(x, y);
                let n = 0.25
                    * (f.value(x.saturating_sub(2), y)
                        + f.value((x + 2).min(63), y)
                        + f.value(x, y.saturating_sub(2))
                        + f.value(x, (y + 2).min(63)));
                assert!(
                    v > n + 0.2,
                    "site {i} reads {v} against a neighbourhood of {n}"
                );
            }
        }
    }
}
