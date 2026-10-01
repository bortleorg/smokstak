//! End-to-end validation against synthetic ground truth.
//!
//! This is the harness the plan insists on building before optimising anything:
//! bursts whose geometry, blur, noise and content we chose, run through the
//! same registration and merge code as real data, and scored against the truth
//! they were generated from.
//!
//! It answers the questions that matter and that no real burst can answer:
//!
//! * does registration recover the shifts we planted, and to what accuracy?
//! * does the raw-domain merge beat a single frame upscaled — in MTF, not just
//!   in noise?
//! * does the raw-domain merge beat demosaic-then-average?
//! * does the structure-aware kernel beat isotropic drizzle?
//! * does a burst with no sub-pixel diversity correctly report that it cannot
//!   be super-resolved?

use std::path::Path;
use std::time::Instant;

use anyhow::Result;

use sr_core::config::{Backend, LocalWarpMode, ReconstructionConfig};
use sr_core::frame::{NoiseModel, NoiseSource, RawFrame};
use sr_core::geometry::WarpField;
use sr_core::plane::Plane;
use sr_reconstruct::kernel::KernelField;
use sr_reconstruct::robustness::RobustnessMaps;
use sr_synth::metrics::{
    Comparison, MtfResult, StarChroma, compare, compare_excluding, slanted_edge_mtf, star_chroma,
};
use sr_synth::{SynthBurst, SynthConfig};

/// Ratchets on the colour of a point source.
///
/// Not bars a correct reconstruction is known to meet -- there is no such
/// number for four synthetic stars, whose ring ratios move with the dither
/// pattern and the reference frame -- but the level the suite holds today, set
/// so that the defect that prompted them (cores of random colour, a rim that
/// drifted 0.67 on the clean scenario) fails and the current merge passes.
/// Their job is to notice a regression, and the two columns in the table are
/// what to read for the actual state.
///
/// Demosaic-then-average was the bar at first, and is not one: it blurs a
/// star to a fifth of its height, and a heavily blurred profile has smooth
/// rings and a low ratio scatter for that reason alone.
const STAR_CORE_RATCHET: f32 = 0.17;
const STAR_RIM_RATCHET: f32 = 0.40;

struct Scenario {
    name: &'static str,
    description: &'static str,
    config: SynthConfig,
    /// Whether this scenario is expected to support the full scale.
    expect_super_resolution: bool,
}

impl Scenario {
    /// How far ahead of a single interpolated frame the merge has to finish.
    ///
    /// On a mosaic the single frame is penalised twice — demosaic *and*
    /// interpolation — so a large margin is the honest bar. A monochrome sensor
    /// measures every site, so a single frame upsampled is already a fair
    /// image and the merge starts from much closer. Holding it to the mosaic
    /// margin would not be a stricter test, it would be a different one.
    fn psnr_margin_db(&self) -> f32 {
        if self.config.mono { 0.3 } else { 1.0 }
    }

    /// The same, for resolution: on a monochrome burst the merge has to resolve
    /// more than one frame does, rather than more by a set fraction.
    fn mtf_margin(&self) -> f32 {
        if self.config.mono { 1.0 } else { 1.05 }
    }
}

fn scenarios(frames: usize, sensor: usize) -> Vec<Scenario> {
    let base = SynthConfig {
        sensor,
        scale: 2.0,
        frames,
        shift_sigma: 1.5,
        psf_sigma: 0.55,
        noise: NoiseModel::new(2.0e-5, 4.0e-6, NoiseSource::Manual),
        seed: 0xA11CE,
        ..Default::default()
    };
    vec![
        Scenario {
            name: "clean",
            description: "sub-pixel shifts, low noise, static scene",
            config: base.clone(),
            expect_super_resolution: true,
        },
        Scenario {
            name: "noisy",
            description: "same geometry at roughly 5x the noise",
            config: SynthConfig {
                noise: NoiseModel::new(5.0e-4, 1.0e-4, NoiseSource::Manual),
                seed: 0xB0B,
                ..base.clone()
            },
            expect_super_resolution: true,
        },
        Scenario {
            name: "static",
            description: "no inter-frame motion at all",
            config: SynthConfig {
                shift_sigma: 0.0,
                seed: 0xC0C0,
                ..base.clone()
            },
            expect_super_resolution: false,
        },
        Scenario {
            name: "motion",
            description: "an object crossing the frame in half the exposures",
            config: SynthConfig {
                moving_object: true,
                seed: 0xD1CE,
                ..base.clone()
            },
            expect_super_resolution: true,
        },
        Scenario {
            name: "drifting",
            description: "illumination that scales and a background that rises across the burst",
            config: SynthConfig {
                exposure_jitter: 0.08,
                sky_drift: 0.06,
                seed: 0xF00D,
                ..base.clone()
            },
            expect_super_resolution: true,
        },
        Scenario {
            name: "hotpixels",
            description: "a sensor with bad sites that no dark frame declared",
            config: SynthConfig {
                hot_pixels: 150,
                seed: 0x5EED,
                ..base.clone()
            },
            expect_super_resolution: true,
        },
        Scenario {
            name: "mono",
            description: "a monochrome sensor at narrowband noise: every site measured, no mosaic",
            config: SynthConfig {
                mono: true,
                noise: NoiseModel::new(5.0e-4, 1.0e-4, NoiseSource::Manual),
                seed: 0x0FF1,
                ..base.clone()
            },
            expect_super_resolution: true,
        },
        Scenario {
            name: "turbulence",
            description: "smooth per-frame local deformation, as atmospheric seeing produces",
            config: SynthConfig {
                local_warp_amp: 1.2,
                rotation_sigma: 0.05,
                seed: 0xE1E1,
                ..base.clone()
            },
            expect_super_resolution: true,
        },
    ]
}

/// Registration accuracy against the planted geometry.
struct RegistrationScore {
    /// RMS error of the recovered global shift, in sensor pixels.
    global_rms: f32,
    worst: f32,
    /// RMS positional error over the whole frame, including rotation and any
    /// local field, sampled on a grid.
    field_rms: f32,
}

fn score_registration(
    synth: &SynthBurst,
    estimated: &[WarpField],
    reference: usize,
    sensor: usize,
) -> RegistrationScore {
    // The pipeline registers against its own chosen reference, while the truth
    // is expressed relative to frame 0. Compose out that difference before
    // comparing, or a correct registration would score as a constant error.
    let ref_true = &synth.true_warps[reference];
    let mut errs = Vec::new();
    let mut field = 0.0f64;
    let mut field_n = 0u64;

    for (i, est) in estimated.iter().enumerate() {
        // True mapping from frame i into the reference frame's coordinates.
        let true_map = |x: f32, y: f32| -> (f32, f32) {
            let (ax, ay) = synth.true_warps[i].map(x, y);
            // Undo the reference frame's own warp to land in reference coords.
            ref_true.inverse_map(ax, ay).unwrap_or((ax, ay))
        };

        let c = sensor as f32 * 0.5;
        let (tx, ty) = true_map(c, c);
        let (ex, ey) = est.map(c, c);
        let e = ((tx - ex).powi(2) + (ty - ey).powi(2)).sqrt();
        errs.push(e);

        let step = (sensor / 8).max(1);
        let mut y = step;
        while y + step < sensor {
            let mut x = step;
            while x + step < sensor {
                let (a, b) = true_map(x as f32, y as f32);
                let (p, q) = est.map(x as f32, y as f32);
                let d = ((a - p) as f64).powi(2) + ((b - q) as f64).powi(2);
                field += d;
                field_n += 1;
                x += step;
            }
            y += step;
        }
    }

    let rms = (errs.iter().map(|e| (e * e) as f64).sum::<f64>() / errs.len().max(1) as f64).sqrt();
    RegistrationScore {
        global_rms: rms as f32,
        worst: errs.iter().cloned().fold(0.0f32, f32::max),
        field_rms: (field / field_n.max(1) as f64).sqrt() as f32,
    }
}

/// Baseline A: the best single frame, demosaiced and interpolated up to the
/// output size. If multi-frame reconstruction cannot beat this, it is not
/// contributing anything.
fn single_frame_baseline(
    frame: &RawFrame,
    scale: f32,
    out_w: usize,
    out_h: usize,
) -> [Plane<f32>; 3] {
    let rgb = as_three(&sr_raw::demosaic_bilinear(frame), frame.channels());
    let mut out = [
        Plane::<f32>::new(out_w, out_h),
        Plane::<f32>::new(out_w, out_h),
        Plane::<f32>::new(out_w, out_h),
    ];
    for c in 0..3 {
        for y in 0..out_h {
            let sy = (y as f32 + 0.5) / scale - 0.5;
            for x in 0..out_w {
                let sx = (x as f32 + 0.5) / scale - 0.5;
                out[c].data[y * out_w + x] = rgb[c].bilinear(
                    sx.clamp(0.0, (frame.width - 1) as f32),
                    sy.clamp(0.0, (frame.height - 1) as f32),
                );
            }
        }
    }
    out
}

struct BackendScore {
    label: String,
    comparison: Comparison,
    /// Comparison restricted to the low-contrast texture panel.
    texture: Comparison,
    mtf: Option<MtfResult>,
    /// Colour at the planted stars: what the whole-frame numbers cannot see.
    ///
    /// A star's core and its rim are a few hundred pixels out of millions, so
    /// PSNR, SSIM and the chroma error are all indifferent to them, and a
    /// slanted-edge MTF never looks at a point source at all. They are also
    /// the first thing a person notices.
    chroma: Option<StarChroma>,
    elapsed_ms: u128,
}

/// Compare a single region against the truth.
///
/// A whole-frame score is dominated by whatever occupies the most area, and a
/// slanted-edge MTF only ever looks at a high-contrast step. Neither notices a
/// merge that is quietly smoothing away low-contrast fine texture, which is
/// most of what a real photograph is made of.
/// Expand a monochrome image to three identical channels.
///
/// A monochrome burst samples the latent scene's green channel at every site,
/// so its reconstruction and the truth it is scored against are both that one
/// channel. Every metric here works in three, and rewriting them all to carry a
/// channel count would be a great deal of code to say the same thing three
/// times.
fn as_three(img: &[Plane<f32>; 3], channels: usize) -> [Plane<f32>; 3] {
    if channels == 1 {
        [img[0].clone(), img[0].clone(), img[0].clone()]
    } else {
        img.clone()
    }
}

fn compare_region(
    recon: &[Plane<f32>; 3],
    truth: &[Plane<f32>; 3],
    rect: (usize, usize, usize, usize),
) -> Comparison {
    let (x, y, w, h) = rect;
    let w = w.min(recon[0].width.saturating_sub(x));
    let h = h.min(recon[0].height.saturating_sub(y));
    let crop = |p: &[Plane<f32>; 3]| {
        [
            p[0].crop(x, y, w, h),
            p[1].crop(x, y, w, h),
            p[2].crop(x, y, w, h),
        ]
    };
    compare(&crop(recon), &crop(truth), 2)
}

fn measure_mtf(img: &[Plane<f32>; 3], synth: &SynthBurst) -> Option<MtfResult> {
    let edge = *synth.latent.edges.first()?;
    let half_width = synth.output_width as f32 * 0.05;
    let half_length = synth.output_width as f32 * 0.07;
    slanted_edge_mtf(&img[1], edge, half_width, half_length)
}

fn run_scenario(
    sc: &Scenario,
    out_dir: &Path,
    keep: bool,
    photometric_match: bool,
    detect_defects: bool,
) -> Result<bool> {
    println!("\n=== {} — {} ===", sc.name, sc.description);
    let mut synth = sr_synth::generate(&sc.config);
    let sensor = sc.config.sensor;
    let scale = sc.config.scale;

    // Same entry points the real pipeline uses.
    let guide_luma: Vec<Plane<f32>> = synth.frames.iter().map(|f| f.guide_rgb().luma()).collect();
    let mut qualities: Vec<sr_core::frame::FrameQuality> = guide_luma
        .iter()
        .zip(&synth.frames)
        .map(|(l, f)| sr_quality::analyse(l, f.saturation_fraction()))
        .collect();
    sr_quality::normalise_against_median(&mut qualities);

    let cfg_base = ReconstructionConfig {
        scale,
        tile: 256,
        photometric_match,
        detect_defects,
        local_warp: if sc.config.local_warp_amp > 0.0 {
            LocalWarpMode::On
        } else {
            LocalWarpMode::Auto
        },
        ..Default::default()
    };

    let proxies: Vec<sr_register::pyramid::RegistrationImage> = guide_luma
        .iter()
        .map(|l| {
            sr_register::pyramid::RegistrationImage::build(l, cfg_base.registration.pyramid_levels)
        })
        .collect();
    let mut choice =
        sr_register::reference::select_reference(&proxies, &qualities, &cfg_base.registration);
    // The truth is the latent scene, which is frame 0's geometry: frame 0 is
    // the generator's anchor and carries the identity warp. Reconstructing onto
    // any other frame's grid is then scored against a scene that has been
    // deformed away from it, so the fidelity numbers measure the reference's
    // own deformation rather than the merge, and a correct reconstruction is
    // penalised for being faithful to the frame it was asked to match.
    //
    // Rigid scenarios do not have this problem -- the truth is the same scene
    // under a translation the comparison's crop absorbs -- so reference
    // selection is left to do its own work everywhere else, which is most of
    // the suite.
    if sc.config.local_warp_amp > 0.0 && choice.index != 0 {
        println!(
            "reference pinned to frame 0 (selection preferred {}): the truth is \
             that frame's geometry",
            choice.index
        );
        choice.index = 0;
    }
    let registrations =
        sr_register::global::register_burst(&proxies, choice.index, &cfg_base.registration);
    let mut warps: Vec<WarpField> = registrations
        .iter()
        .map(|r| WarpField::global_only(r.transform).rescale(2.0))
        .collect();
    // Score the global fit on its own first, so the local stage can be credited
    // with the improvement it actually delivers rather than an absolute number.
    let global_only_score = score_registration(&synth, &warps, choice.index, sensor);
    let local_applied = sr_warp::maybe_refine(
        &proxies,
        choice.index,
        &registrations,
        &cfg_base,
        &mut warps,
    );

    let reg_score = score_registration(&synth, &warps, choice.index, sensor);
    println!(
        "reference frame: {} of {}",
        choice.index,
        synth.frames.len()
    );
    println!(
        "registration: global RMS {:.3} px, worst {:.3} px, full-field RMS {:.3} px, local warp {}",
        reg_score.global_rms,
        reg_score.worst,
        reg_score.field_rms,
        if local_applied { "on" } else { "off" }
    );
    if local_applied {
        println!(
            "  full-field RMS before local refinement: {:.3} px",
            global_only_score.field_rms
        );
    }

    let coverage = sr_reconstruct::coverage::analyse_coverage(&synth.frames, &warps, scale, 8);
    println!(
        "sampling diversity: {:.2}x supported (requested {:.2}x)",
        coverage.recommended_scale, scale
    );

    let residual_sigma = {
        let rms: Vec<f32> = registrations
            .iter()
            .filter(|r| r.probes > 0)
            .map(|r| r.residual_rms)
            .collect();
        if rms.is_empty() {
            0.5
        } else {
            sr_core::math::median(&rms) * 2.0
        }
    };

    // Fixed-pattern defects, found the way the real pipeline finds them: from
    // the burst, with no help from the generator. The frames do not carry a
    // mask, so if the detector stops working the streaks land in the scores.
    // The same gate the pipeline applies, from the planted geometry rather
    // than the estimated one: with no motion a star is a fixed pattern and the
    // scan would delete the sky. The `static` scenario is here to prove that
    // the gate holds.
    let burst_motion = {
        let c = sensor as f32 * 0.5;
        let centres: Vec<(f32, f32)> = synth
            .true_warps
            .iter()
            .map(|w| w.global.apply(c, c))
            .collect();
        let xs: Vec<f32> = centres.iter().map(|p| p.0).collect();
        let ys: Vec<f32> = centres.iter().map(|p| p.1).collect();
        let (mx, my) = (sr_core::math::median(&xs), sr_core::math::median(&ys));
        let d: Vec<f32> = centres
            .iter()
            .map(|&(x, y)| ((x - mx).powi(2) + (y - my).powi(2)).sqrt())
            .collect();
        sr_core::math::median(&d)
    };
    let defects = if cfg_base.detect_defects && burst_motion >= sr_noise::defects::MIN_MOTION_PX {
        let (mask, report) = sr_noise::defects::find_fixed_pattern(
            &synth.frames.iter().collect::<Vec<_>>(),
            &sc.config.noise,
        );
        for f in synth.frames.iter_mut() {
            f.defects = mask.clone();
        }
        Some((mask, report))
    } else {
        None
    };

    // Measured rather than taken from the generator. The matcher is in the
    // path of every real run, so the harness has to exercise it: if it stopped
    // recovering the planted illumination change, the scores below are what
    // would say so.
    let photometry: Vec<sr_quality::photometry::PhotometricMatch> = if cfg_base.photometric_match {
        sr_quality::photometry::match_burst(
            &synth.frames,
            &warps,
            choice.index,
            &synth.exposure_scale,
        )
        .iter()
        .map(|p| p.map)
        .collect()
    } else {
        synth
            .exposure_scale
            .iter()
            .map(|&s| sr_quality::photometry::PhotometricMatch::from_exposure(s))
            .collect()
    };
    let robustness = sr_reconstruct::robustness::build_maps(
        &synth.frames,
        &warps,
        choice.index,
        &vec![true; synth.frames.len()],
        &photometry,
        &sc.config.noise,
        residual_sigma,
        &cfg_base.robustness,
    );
    let no_robustness = RobustnessMaps {
        width: sensor / 2,
        height: sensor / 2,
        maps: Vec::new(),
        rejected_fraction: vec![0.0; synth.frames.len()],
        consensus_luma: None,
    };
    let weights = vec![1.0f32; synth.frames.len()];

    let mut scores: Vec<BackendScore> = Vec::new();

    // Baseline A: one frame, interpolated.
    let single = single_frame_baseline(
        &synth.frames[choice.index],
        scale,
        synth.output_width,
        synth.output_height,
    );
    let border = (synth.output_width / 12).max(8);
    let panel = synth.latent.texture_panel;
    // A monochrome sensor measures the latent's green channel, so that is the
    // truth its reconstruction is scored against.
    let truth = if sc.config.mono {
        [
            synth.truth[1].clone(),
            synth.truth[1].clone(),
            synth.truth[1].clone(),
        ]
    } else {
        synth.truth.clone()
    };
    // Where the point sources landed, and how wide they are once the optics
    // and the pixel aperture have had their say. The latent grid is the output
    // grid, so these need no mapping.
    let star_at: Vec<(f32, f32)> = synth.latent.stars.iter().map(|s| (s.x, s.y)).collect();
    let star_psf = scale * 0.8;
    // A monochrome sensor has no channel ratios to measure, and as_three would
    // make every one of them exactly 1.
    let chroma_of = |img: &[Plane<f32>; 3]| -> Option<StarChroma> {
        if sc.config.mono {
            None
        } else {
            star_chroma(img, &star_at, star_psf)
        }
    };

    // A saturated core carries no information: the sensor recorded full scale
    // and nothing above it, so no reconstruction can put back what is missing
    // and scoring one against the latent scene there is scoring the
    // impossible. Left in, four stars whose true peaks are several times full
    // scale dominate a whole-frame PSNR. So the cores are excluded from the
    // fidelity metrics and judged by `star_chroma` instead, which asks the
    // question that can be answered: whatever value came out, is it one
    // colour, and does that colour hold with radius.
    let star_disks: Vec<(f32, f32, f32)> = synth
        .latent
        .stars
        .iter()
        .map(|s| (s.x, s.y, star_psf * 4.0))
        .collect();

    scores.push(BackendScore {
        label: "single frame + bilinear upscale".into(),
        comparison: compare_excluding(&single, &truth, border, &star_disks),
        texture: compare_region(&single, &truth, panel),
        mtf: measure_mtf(&single, &synth),
        chroma: chroma_of(&single),
        elapsed_ms: 0,
    });

    let backends = [
        (Backend::RgbMeanBaseline, "demosaic + align + average"),
        (Backend::CfaDrizzle, "CFA drizzle (isotropic)"),
        (Backend::HandheldBurstSr, "burst SR (structure-aware)"),
    ];
    for (backend, label) in backends {
        let cfg = ReconstructionConfig {
            backend,
            ..cfg_base.clone()
        };
        let kernel_guide = synth.frames[choice.index].structure_guide_rgb().luma();
        let ref_luma = &kernel_guide;
        let noise_sigma = sr_reconstruct::kernel::guide_noise_sigma(ref_luma)
            .max(0.05 * sc.config.noise.std_dev(ref_luma.mean().max(0.0)));
        let kernels = match backend {
            Backend::HandheldBurstSr => KernelField::for_sensor(
                ref_luma,
                noise_sigma,
                synth.frames.len(),
                synth.frames[choice.index].cfa,
                &cfg.kernel,
            ),
            _ => KernelField::isotropic(
                ref_luma.width,
                ref_luma.height,
                cfg.kernel.k_detail.max(0.25),
                cfg.kernel.radius,
                2,
            ),
        };
        let rob: &RobustnessMaps = if matches!(backend, Backend::RgbMeanBaseline) {
            &no_robustness
        } else {
            &robustness
        };
        let inputs = sr_reconstruct::MergeInputs {
            frames: &synth.frames,
            warps: &warps,
            reference: choice.index,
            photometry: &photometry,
            noise: sc.config.noise,
            robustness: rob,
            kernels: &kernels,
            frame_weight: &weights,
            lucky: None,
            // The synthetic generator images every channel through the same
            // geometry, so there is no aberration to correct here.
            chroma: sr_core::geometry::RadialChroma::identity(),
        };
        let t = Instant::now();
        let product = sr_reconstruct::reconstruct(&inputs, &cfg)?;
        let elapsed = t.elapsed();
        let scored = as_three(&product.rgb, product.channels);
        scores.push(BackendScore {
            label: label.into(),
            comparison: compare_excluding(&scored, &truth, border, &star_disks),
            texture: compare_region(&scored, &truth, panel),
            mtf: measure_mtf(&scored, &synth),
            chroma: chroma_of(&scored),
            elapsed_ms: elapsed.as_millis(),
        });

        if keep {
            std::fs::create_dir_all(out_dir)?;
            sr_output::write_product32f(
                &out_dir.join(format!("{}-{}.linear.tif", sc.name, backend.name())),
                &product.rgb,
                product.channels,
            )?;
            let p = out_dir.join(format!("{}-{}.tif", sc.name, backend.name()));
            sr_output::write_product16(&p, &sr_color::to_rendered(&product.rgb), product.channels)?;
        }
    }

    if keep {
        std::fs::create_dir_all(out_dir)?;
        // Keep native values for downstream accuracy audits. The display
        // TIFFs clip highlights and cannot establish linear colour accuracy.
        sr_output::write_rgb32f(
            &out_dir.join(format!("{}-truth.linear.tif", sc.name)),
            &truth,
        )?;
        sr_output::write_rgb32f(
            &out_dir.join(format!("{}-single-frame.linear.tif", sc.name)),
            &single,
        )?;
        sr_output::write_rgb16(
            &out_dir.join(format!("{}-truth.tif", sc.name)),
            &sr_color::to_rendered(&truth),
        )?;
        sr_output::write_rgb16(
            &out_dir.join(format!("{}-single-frame.tif", sc.name)),
            &sr_color::to_rendered(&single),
        )?;
    }

    println!(
        "{:<34} {:>9} {:>8} {:>10} {:>10} {:>9} {:>9} {:>9} {:>8} {:>6}",
        "method",
        "PSNR dB",
        "SSIM",
        "MTF50",
        "overshoot",
        "chroma",
        "texture",
        "star core",
        "star rim",
        "ms"
    );
    for s in &scores {
        println!(
            "{:<34} {:>9.2} {:>8.4} {:>10} {:>10} {:>9.5} {:>9.2} {:>9} {:>8} {:>6}",
            s.label,
            s.comparison.psnr_mean_db,
            s.comparison.ssim,
            s.mtf
                .as_ref()
                .map(|m| format!("{:.4}", m.mtf50))
                .unwrap_or_else(|| "-".into()),
            s.mtf
                .as_ref()
                .map(|m| format!("{:.3}", m.overshoot))
                .unwrap_or_else(|| "-".into()),
            s.comparison.chroma_error,
            s.texture.psnr_mean_db,
            s.chroma
                .map(|c| format!("{:.4}", c.core_scatter))
                .unwrap_or_else(|| "-".into()),
            s.chroma
                .map(|c| format!("{:.4}", c.halo_drift))
                .unwrap_or_else(|| "-".into()),
            s.elapsed_ms
        );
    }

    // Checks. These are the claims the program makes about itself.
    let mut ok = true;
    let single = &scores[0];
    let drizzle = &scores[2];
    let burst_sr = &scores[3];
    let rgb_mean = &scores[1];

    let mut check = |pass: bool, msg: String| {
        println!("  [{}] {}", if pass { "PASS" } else { "FAIL" }, msg);
        if !pass {
            ok = false;
        }
    };

    if sc.config.local_warp_amp > 0.0 {
        // No rigid transform can absorb a deformation that varies across the
        // frame, so the meaningful question is how much of it the local stage
        // removes.
        check(
            reg_score.field_rms < global_only_score.field_rms * 0.7,
            format!(
                "local refinement cuts registration error from {:.3} to {:.3} px",
                global_only_score.field_rms, reg_score.field_rms
            ),
        );
    } else {
        check(
            reg_score.global_rms < 0.15,
            format!(
                "registration recovers planted shifts to {:.3} px (target < 0.15)",
                reg_score.global_rms
            ),
        );
    }

    if sc.expect_super_resolution {
        check(
            coverage.recommended_scale > scale - 0.05,
            format!(
                "sampling diversity reported as {:.2}x for a burst that has it",
                coverage.recommended_scale
            ),
        );
        check(
            burst_sr.comparison.psnr_mean_db > single.comparison.psnr_mean_db + sc.psnr_margin_db(),
            format!(
                "burst SR beats a single interpolated frame by {:.2} dB",
                burst_sr.comparison.psnr_mean_db - single.comparison.psnr_mean_db
            ),
        );
        // The check the roof dataset taught us to make. A merge can win on
        // edges and on whole-frame PSNR while smoothing away exactly the
        // low-contrast fine texture that most of a photograph is made of.
        check(
            burst_sr.texture.psnr_mean_db > single.texture.psnr_mean_db,
            format!(
                "burst SR preserves low-contrast texture better than one frame \
                 ({:.2} vs {:.2} dB on the texture panel)",
                burst_sr.texture.psnr_mean_db, single.texture.psnr_mean_db
            ),
        );

        if let (Some(a), Some(b)) = (&burst_sr.mtf, &single.mtf) {
            check(
                a.mtf50 > b.mtf50 * sc.mtf_margin(),
                format!(
                    "burst SR resolves finer detail than one frame: MTF50 {:.4} vs {:.4} cycles/px",
                    a.mtf50, b.mtf50
                ),
            );
        }
        // Colour at a point source, which every other number here is blind to.
        // Ratchets, not bars; see the constants.
        if let Some(a) = burst_sr.chroma {
            let b = rgb_mean.chroma.map(|c| c.core_scatter).unwrap_or(f32::NAN);
            check(
                a.core_scatter <= STAR_CORE_RATCHET,
                format!(
                    "star cores are one colour: scatter {:.4} over {} stars (ratchet {:.2}; \
                     demosaic-then-average {:.4})",
                    a.core_scatter, a.stars, STAR_CORE_RATCHET, b
                ),
            );
            let b = rgb_mean.chroma.map(|c| c.halo_drift).unwrap_or(f32::NAN);
            check(
                a.halo_drift <= STAR_RIM_RATCHET,
                format!(
                    "stars have no coloured rim: colour drifts {:.4} from the inner ring \
                     outwards (ratchet {:.2}; demosaic-then-average {:.4})",
                    a.halo_drift, STAR_RIM_RATCHET, b
                ),
            );
        }
        check(
            drizzle.comparison.psnr_mean_db > rgb_mean.comparison.psnr_mean_db - 0.25,
            format!(
                "raw-domain merge is no worse than demosaic-then-average ({:.2} vs {:.2} dB)",
                drizzle.comparison.psnr_mean_db, rgb_mean.comparison.psnr_mean_db
            ),
        );
        // Colour is judged against the single interpolated frame, which is
        // what a user would otherwise have. Demosaic-then-average scores lower
        // on chroma only because bilinear demosaicing smooths chroma away
        // together with the detail it also loses, so it is not a standard worth
        // matching.
        check(
            burst_sr.comparison.chroma_error <= single.comparison.chroma_error,
            format!(
                "raw-domain merge has no more false colour than one interpolated frame \
                 ({:.5} vs {:.5}; demosaic-then-average {:.5})",
                burst_sr.comparison.chroma_error,
                single.comparison.chroma_error,
                rgb_mean.comparison.chroma_error
            ),
        );
    } else {
        check(
            coverage.recommended_scale < 1.3,
            format!(
                "a burst with no motion is honestly reported as unable to super-resolve \
                 (recommended {:.2}x)",
                coverage.recommended_scale
            ),
        );
    }

    if sc.name == "motion" {
        let mean_rejected = robustness.rejected_fraction.iter().sum::<f32>()
            / robustness.rejected_fraction.len() as f32;
        check(
            mean_rejected > 0.001,
            format!(
                "the moving object was detected and rejected ({:.3}% of samples)",
                mean_rejected * 100.0
            ),
        );
    }

    if sc.name == "turbulence" {
        check(
            local_applied,
            "local warp refinement engaged on a deformed burst".into(),
        );
    }

    if sc.name == "static" {
        check(
            defects.is_none(),
            format!(
                "defect detection declined on a burst that moved {burst_motion:.2} px, where \
                 a star cannot be told from a hot pixel"
            ),
        );
    }

    if sc.config.hot_pixels > 0 {
        let mut planted: Vec<usize> = synth.hot_pixels.clone();
        planted.sort_unstable();
        planted.dedup();
        // Sites near the sensor edge have no same-colour neighbours to be
        // compared against, so they are undetectable by construction and are
        // not counted against the detector.
        let n = sc.config.sensor;
        planted.retain(|&i| {
            let (x, y) = (i % n, i / n);
            x >= 4 && y >= 4 && x + 4 < n && y + 4 < n
        });
        // A site whose scene is already at white carries no excess to find and
        // is discarded as unusable by the merge anyway, so it is neither a
        // miss nor a defect anyone is harmed by.
        planted.retain(|&i| {
            let (x, y) = (i % n, i / n);
            let buried = synth.frames.iter().filter(|f| f.value(x, y) >= 1.0).count();
            buried * 2 <= synth.frames.len()
        });
        let (mask, report) = defects.as_ref().expect("the defect scan runs by default");
        let found = planted.iter().filter(|&&i| mask.get(i)).count();
        let spurious = report.total().saturating_sub(found);
        // Two thirds, not all of them. This scene is the hard case for the
        // detector by construction: a high-contrast chart under a dither of
        // about a pixel and a half, where scene structure at a site persists
        // from frame to frame nearly as well as a defect does. On a flat field
        // — which is what the astronomical bursts this was built for look
        // like — recall is essentially complete. The bar is here to catch the
        // detector breaking, not to be tightened by tuning against a chart.
        check(
            found * 3 >= planted.len() * 2,
            format!(
                "fixed-pattern defects found: {found} of {} planted sites ({:.0}%)",
                planted.len(),
                found as f32 / planted.len().max(1) as f32 * 100.0
            ),
        );
        check(
            spurious * 10 <= planted.len(),
            format!(
                "{spurious} sites were called defective that were not ({:.4}% of the sensor)",
                spurious as f32 / (n * n) as f32 * 100.0
            ),
        );
    }

    Ok(ok)
}

/// The burst size every margin in this file was chosen against. It is the CLI's
/// default too, and the two have to agree.
const DEFAULT_FRAMES: usize = 24;

pub fn run(
    out: &Path,
    frames: usize,
    size: usize,
    keep: bool,
    photometric_match: bool,
    detect_defects: bool,
) -> Result<()> {
    println!(
        "Synthetic validation: {frames} frames, {size} px sensor, 2x output.\n\
         Every scenario runs through the same registration and merge code as real bursts."
    );
    // The margins below are calibrated for the default burst. A short one is a
    // legal thing to ask for and a fast way to see the harness run, but two
    // things move: the geometry is redrawn, so which frame wins the reference
    // and how well it happens to be placed both change; and at 2x a mosaic's
    // red and blue sites deliver a quarter of a sample per output pixel per
    // frame, so under about sixteen frames those channels are marginally
    // determined and the merge's own score falls. Twelve frames scores 31.7 dB
    // on the clean scenario where twenty-four scores 36.0, against a
    // single-frame baseline that barely moves. That is the burst being short,
    // not the code being broken.
    if frames < DEFAULT_FRAMES {
        println!(
            "\nNote: {frames} frames is below the {DEFAULT_FRAMES} these margins are \
             calibrated for. A short burst undersamples red and blue at 2x and redraws the \
             geometry, so a failure here may be the burst rather than the code. Confirm with \
             the default before concluding anything."
        );
    }
    let mut all_ok = true;
    for sc in scenarios(frames, size) {
        match run_scenario(&sc, out, keep, photometric_match, detect_defects) {
            Ok(ok) => all_ok &= ok,
            Err(e) => {
                println!("  [FAIL] scenario {} errored: {e}", sc.name);
                all_ok = false;
            }
        }
    }
    println!(
        "\n{}",
        if all_ok {
            "All checks passed."
        } else {
            "Some checks FAILED."
        }
    );
    if keep {
        println!("Outputs kept in {}", out.display());
    }
    anyhow::ensure!(all_ok, "synthetic validation failed");
    Ok(())
}
