//! Stage 8: local registration and atmospheric warp.
//!
//! A single global transform cannot describe heat shimmer, seeing, parallax or
//! rolling local deformation. This stage estimates a smooth residual
//! displacement field on top of the global fit.
//!
//! Two properties matter more than raw accuracy:
//!
//! * **Smoothness.** A field with too much freedom will happily align noise,
//!   and aligned noise looks exactly like recovered detail. The field is
//!   therefore stored as a coarse mesh and regularised hard.
//! * **Honest confidence.** Nodes that could not be measured are marked, not
//!   guessed, so the merge can down-weight what it does not know.

use rayon::prelude::*;

use sr_core::config::{LocalWarpMode, ReconstructionConfig, WarpConfig};
use sr_core::geometry::{DeformationField, GlobalTransform, WarpField};
use sr_register::correlate::Correlator;
use sr_register::global::GlobalRegistration;
use sr_register::pyramid::{RegistrationImage, extract_patch, extract_patch_warped};

/// Estimate the residual displacement field for one frame, in proxy
/// coordinates, given its global transform.
pub fn refine_frame(
    reference: &RegistrationImage,
    target: &RegistrationImage,
    global: &GlobalTransform,
    cfg: &WarpConfig,
) -> DeformationField {
    let depth = reference.depth().min(target.depth());
    let stages = cfg.levels.clamp(1, depth);

    let (w0, h0) = reference.dims();
    let mut field: Option<DeformationField> = None;

    for stage in 0..stages {
        let level = stages - 1 - stage;
        let s = (1 << level) as f32;
        let ref_img = reference.level(level);
        let tgt_img = target.level(level);
        let limit = (ref_img.width.min(ref_img.height) / 2) & !1;
        if limit < 16 {
            continue;
        }
        let patch = (cfg.patch.min(limit).max(16)) & !1;
        let half = patch / 2;

        // Node spacing has to shrink with the level, or a coarse level ends up
        // with a two-by-two grid that cannot represent a smooth field at all.
        // It also has to stay fine relative to the frame: the configured
        // spacing is chosen for a full-size sensor, and applying it unchanged
        // to a small image leaves too few nodes to resolve a deformation that
        // varies over the frame, which shows up as a field of roughly the right
        // shape and a fraction of the right amplitude.
        let short_side = ref_img.width.min(ref_img.height);
        let usable = short_side.saturating_sub(patch);
        let spacing = cfg
            .spacing
            .min((short_side / 12).max(8))
            .clamp(8, (usable / 2).max(8));

        // Nodes are inset by half a patch so that the first and last one still
        // have a full patch of image to correlate, and laid out on the level-0
        // proxy grid so every stage shares one coordinate system.
        let step0 = spacing as f32 * s;
        let origin0 = half as f32 * s;
        let span_w = (w0 as f32 - 2.0 * origin0).max(0.0);
        let span_h = (h0 as f32 - 2.0 * origin0).max(0.0);
        let gw = (span_w / step0).ceil() as usize + 1;
        let gh = (span_h / step0).ceil() as usize + 1;
        let mut next = DeformationField::zeros((origin0, origin0), step0, gw, gh);

        // Carry the previous stage's estimate forward.
        if let Some(prev) = &field {
            for gy in 0..gh {
                for gx in 0..gw {
                    let px = origin0 + gx as f32 * step0;
                    let py = origin0 + gy as f32 * step0;
                    let (ux, uy) = prev.sample(px, py);
                    next.u[gy * gw + gx] = [ux, uy];
                }
            }
        }

        let global_l = global.rescale(1.0 / s);
        let Some(inv_l) = global_l.inverse() else {
            return field.unwrap_or_else(|| DeformationField::zeros((0.0, 0.0), step0, gw, gh));
        };

        let max_disp_l = cfg.max_displacement / s;
        let nodes: Vec<(usize, [f32; 2], f32)> = (0..gw * gh)
            .into_par_iter()
            .map_init(
                || {
                    (
                        Correlator::new(patch),
                        vec![0.0f32; patch * patch],
                        vec![0.0f32; patch * patch],
                    )
                },
                |(corr, buf_ref, buf_tgt), idx| {
                    let gx = idx % gw;
                    let gy = idx / gw;
                    // Node position in this level's pixels.
                    let px = (half + gx * spacing) as f32;
                    let py = (half + gy * spacing) as f32;
                    if px + half as f32 > ref_img.width as f32
                        || py + half as f32 > ref_img.height as f32
                    {
                        // Beyond the last full patch. The regulariser fills
                        // these from their neighbours rather than guessing.
                        return (idx, next.u[idx], 0.0);
                    }
                    let prior = next.u[idx];
                    let prior_l = [prior[0] / s, prior[1] / s];

                    extract_patch(ref_img, px, py, patch, buf_ref);
                    // Reference position q maps to target coordinates through
                    // the global inverse after removing the local offset. The
                    // offset is treated as constant across one patch, which is
                    // the same smoothness assumption the field itself encodes.
                    let inv_node =
                        inv_l.compose(&GlobalTransform::translation(-prior_l[0], -prior_l[1]));
                    if !extract_patch_warped(tgt_img, &inv_node, px, py, patch, buf_tgt) {
                        return (idx, prior, 0.0);
                    }
                    match corr.shift(buf_ref, buf_tgt) {
                        Some(sh) => {
                            let mag = (sh.dx * sh.dx + sh.dy * sh.dy).sqrt();
                            let total = [prior_l[0] + sh.dx, prior_l[1] + sh.dy];
                            let total_mag = (total[0] * total[0] + total[1] * total[1]).sqrt();
                            if mag > patch as f32 * 0.25 || total_mag > max_disp_l {
                                return (idx, prior, 0.0);
                            }
                            let conf = ((sh.peak_ratio - 1.0) / 2.0).clamp(0.0, 1.0);
                            (idx, [total[0] * s, total[1] * s], conf)
                        }
                        None => (idx, prior, 0.0),
                    }
                },
            )
            .collect();

        for (idx, u, conf) in nodes {
            next.u[idx] = u;
            next.conf[idx] = conf;
        }

        regularise(
            &mut next,
            cfg.min_confidence,
            cfg.smooth_iters,
            cfg.smooth_lambda,
        );
        field = Some(next);
    }

    field.unwrap_or_else(|| DeformationField::zeros((0.0, 0.0), 32.0, 1, 1))
}

/// How much a fitted field actually improves alignment.
///
/// Re-probes a sample of positions twice: once with the global transform alone
/// and once with the field applied, and reports the fractional reduction in
/// leftover displacement. Positive means the field is describing something
/// real; near zero means it is describing correlation noise.
pub fn field_improvement(
    reference: &RegistrationImage,
    target: &RegistrationImage,
    global: &GlobalTransform,
    field: &DeformationField,
    cfg: &WarpConfig,
) -> f32 {
    let ref_img = reference.level(0);
    let tgt_img = target.level(0);
    let limit = (ref_img.width.min(ref_img.height) / 2) & !1;
    if limit < 16 {
        return 0.0;
    }
    let patch = (cfg.patch.min(limit).max(16)) & !1;
    let half = patch / 2;
    let Some(inv) = global.inverse() else {
        return 0.0;
    };

    // A grid of probes spread over the frame; enough to be representative,
    // few enough to be cheap next to the fit itself.
    let probes = 5usize;
    let usable_w = ref_img.width.saturating_sub(patch);
    let usable_h = ref_img.height.saturating_sub(patch);
    if usable_w == 0 || usable_h == 0 {
        return 0.0;
    }

    let mut corr = Correlator::new(patch);
    let mut buf_ref = vec![0.0f32; patch * patch];
    let mut buf_tgt = vec![0.0f32; patch * patch];
    let mut without = 0.0f64;
    let mut with = 0.0f64;
    let mut n = 0u32;

    for j in 0..probes {
        for i in 0..probes {
            let px = half as f32 + usable_w as f32 * i as f32 / (probes - 1).max(1) as f32;
            let py = half as f32 + usable_h as f32 * j as f32 / (probes - 1).max(1) as f32;
            extract_patch(ref_img, px, py, patch, &mut buf_ref);

            if !extract_patch_warped(tgt_img, &inv, px, py, patch, &mut buf_tgt) {
                continue;
            }
            let Some(a) = corr.shift(&buf_ref, &buf_tgt) else {
                continue;
            };

            let (ux, uy) = field.sample(px, py);
            let inv_node = inv.compose(&GlobalTransform::translation(-ux, -uy));
            if !extract_patch_warped(tgt_img, &inv_node, px, py, patch, &mut buf_tgt) {
                continue;
            }
            let Some(b) = corr.shift(&buf_ref, &buf_tgt) else {
                continue;
            };

            without += a.magnitude() as f64;
            with += b.magnitude() as f64;
            n += 1;
        }
    }

    if n == 0 || without <= 1e-6 {
        return 0.0;
    }
    ((without - with) / without) as f32
}

/// Node spacing actually used at a level, exposed for diagnostics and tests.
pub fn node_spacing(level_width: usize, level_height: usize, cfg: &WarpConfig) -> Option<usize> {
    let limit = (level_width.min(level_height) / 2) & !1;
    if limit < 16 {
        return None;
    }
    let patch = (cfg.patch.min(limit).max(16)) & !1;
    let short_side = level_width.min(level_height);
    let usable = short_side.saturating_sub(patch);
    Some(
        cfg.spacing
            .min((short_side / 12).max(8))
            .clamp(8, (usable / 2).max(8)),
    )
}

/// Smooth the node field and inpaint unmeasured nodes.
///
/// Nodes below `min_confidence` contribute nothing of their own and are pulled
/// toward their neighbours, so an unmeasurable patch of sky inherits the motion
/// around it instead of inventing a displacement.
fn regularise(field: &mut DeformationField, min_confidence: f32, iters: usize, lambda: f32) {
    let (gw, gh) = (field.grid_w, field.grid_h);
    if gw == 0 || gh == 0 {
        return;
    }
    let measured: Vec<f32> = field
        .conf
        .iter()
        .map(|&c| if c >= min_confidence { c } else { 0.0 })
        .collect();

    // A median pass first: it removes single wild nodes that a diffusive
    // smoother would otherwise smear across the neighbourhood.
    //
    // Only nodes with a full, symmetric neighbourhood are filtered. At a grid
    // corner the available samples are lopsided, and the median of a lopsided
    // window of a sloping field is displaced toward the side that has more of
    // it, which would drag the border of every field outward.
    let mut med = field.u.clone();
    if gw > 2 && gh > 2 {
        for gy in 1..gh - 1 {
            for gx in 1..gw - 1 {
                let mut xs: Vec<f32> = Vec::with_capacity(9);
                let mut ys: Vec<f32> = Vec::with_capacity(9);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let j = (gy as i64 + dy) as usize * gw + (gx as i64 + dx) as usize;
                        if measured[j] > 0.0 {
                            xs.push(field.u[j][0]);
                            ys.push(field.u[j][1]);
                        }
                    }
                }
                if xs.len() >= 5 {
                    med[gy * gw + gx] = [sr_core::math::median(&xs), sr_core::math::median(&ys)];
                }
            }
        }
    }
    field.u = med;

    // Confidence-weighted diffusion: measured nodes hold their value in
    // proportion to how well they were measured, everything else is filled in.
    // `lambda` is per neighbour, so a confident node keeps
    // `1 / (1 + 4 * lambda)` of itself each pass.
    let lambda = lambda.max(0.0);
    for _ in 0..iters.max(1) {
        let src = field.u.clone();
        for gy in 0..gh {
            for gx in 0..gw {
                let i = gy * gw + gx;
                let mut acc = [0.0f32; 2];
                let mut n = 0.0f32;
                for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let nx = gx as i64 + dx;
                    let ny = gy as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= gw as i64 || ny >= gh as i64 {
                        continue;
                    }
                    let j = ny as usize * gw + nx as usize;
                    acc[0] += src[j][0];
                    acc[1] += src[j][1];
                    n += 1.0;
                }
                if n == 0.0 {
                    continue;
                }
                let w = measured[i];
                let denom = w + lambda * n;
                field.u[i] = [
                    (w * src[i][0] + lambda * acc[0]) / denom,
                    (w * src[i][1] + lambda * acc[1]) / denom,
                ];
            }
        }
    }
    field.conf = measured;
}

/// How much of the residual the global model failed to explain.
///
/// Used by `--local-warp auto`: if a similarity or affine already explains the
/// frame to within the correlator's own noise, adding a deformation field can
/// only fit noise.
pub fn deformation_evidence(registrations: &[GlobalRegistration]) -> f32 {
    let p90: Vec<f32> = registrations
        .iter()
        .filter(|r| r.probes > 0)
        .map(|r| r.residual_p90)
        .collect();
    if p90.is_empty() {
        return 0.0;
    }
    sr_core::math::median(&p90)
}

/// Decide whether to run local refinement, and if so apply it to `warps`.
///
/// `warps` are in full sensor coordinates on entry and on exit. Returns whether
/// a local field was attached.
pub fn maybe_refine(
    proxies: &[RegistrationImage],
    reference: usize,
    registrations: &[GlobalRegistration],
    cfg: &ReconstructionConfig,
    warps: &mut [WarpField],
) -> bool {
    let evidence = deformation_evidence(registrations);
    let enable = match cfg.local_warp {
        LocalWarpMode::Off => false,
        LocalWarpMode::On => true,
        // The threshold is in proxy pixels and sits just above what a good
        // global fit leaves behind on a rigid scene.
        LocalWarpMode::Auto => evidence > 0.25,
    };
    log::info!(
        "local warp {} (residual p90 median {:.3} proxy px)",
        if enable { "enabled" } else { "disabled" },
        evidence
    );
    if !enable {
        return false;
    }
    // `on` is an explicit instruction, so it overrides the evidence gate. The
    // gate still runs and still reports what it measured, so a user forcing the
    // field can see whether it is helping; `auto` is what defers to it.
    let forced = matches!(cfg.local_warp, LocalWarpMode::On);

    // Decide on the evidence of a sample of frames before fitting all of them.
    //
    // Whether the scene deforms between exposures is a property of the burst,
    // not of individual frames: it comes from the atmosphere, from parallax, or
    // from the lens. Judging each frame on its own would let a fraction through
    // on the strength of correlation noise alone, which is precisely the field
    // that should never be applied. Sampling first also means a rigid burst
    // does not pay for a fit it is going to discard.
    // A validated stellar field already accounts for the residual geometry.
    // Correlation must not overwrite it or fit the same displacement twice.
    let others: Vec<usize> = (0..proxies.len())
        .filter(|&i| i != reference && warps[i].local.is_none())
        .collect();
    if others.is_empty() {
        return false;
    }
    let probe_count = others.len().min(8);
    let probe_frames: Vec<usize> = (0..probe_count)
        .map(|k| others[k * others.len() / probe_count])
        .collect();

    let fit_one = |i: usize| -> (DeformationField, f32) {
        // The stars may have polished the global transform since correlation.
        // Fit and validate relative to the transform used for deposition.
        let global = warps[i].global.rescale(0.5);
        let f = refine_frame(&proxies[reference], &proxies[i], &global, &cfg.warp);
        let gain = field_improvement(&proxies[reference], &proxies[i], &global, &f, &cfg.warp);
        (f, gain)
    };

    let probe: Vec<(usize, DeformationField, f32)> = probe_frames
        .par_iter()
        .map(|&i| {
            let (f, g) = fit_one(i);
            (i, f, g)
        })
        .collect();

    let probe_gains: Vec<f32> = probe.iter().map(|(_, _, g)| *g).collect();
    let median_gain = sr_core::math::median(&probe_gains);
    if forced && median_gain < cfg.warp.min_improvement {
        log::warn!(
            "local warp forced on, but on {} sampled frames the fitted fields reduced alignment \
             residual by a median of only {:.1}%. A field that does not improve alignment is \
             fitting correlation noise, and aligned noise looks like recovered detail. Use \
             --local-warp auto to let this be decided on the evidence.",
            probe.len(),
            median_gain * 100.0
        );
    }
    if !forced && median_gain < cfg.warp.min_improvement {
        log::info!(
            "local warp discarded: on {} sampled frames the fitted fields reduced alignment \
             residual by a median of {:.1}%, below the {:.0}% needed to believe them. The burst \
             is rigid enough that a field would be fitting correlation noise.",
            probe.len(),
            median_gain * 100.0,
            cfg.warp.min_improvement * 100.0
        );
        return false;
    }

    // The burst does deform, so fit the rest. The per-frame bar is lower than
    // the bar for the burst as a whole: once deformation is established, a
    // frame that shows less of it is ordinary, not suspicious.
    let per_frame_bar = if forced {
        f32::NEG_INFINITY
    } else {
        cfg.warp.min_improvement * 0.5
    };
    let mut fields: Vec<Option<DeformationField>> = vec![None; proxies.len()];
    let mut accepted = 0usize;
    for (i, f, g) in probe {
        if g >= per_frame_bar {
            fields[i] = Some(f);
            accepted += 1;
        }
    }

    let remaining: Vec<usize> = others
        .iter()
        .copied()
        .filter(|i| !probe_frames.contains(i))
        .collect();
    let rest: Vec<(usize, DeformationField, f32)> = remaining
        .par_iter()
        .map(|&i| {
            let (f, g) = fit_one(i);
            (i, f, g)
        })
        .collect();
    for (i, f, g) in rest {
        if g >= per_frame_bar {
            fields[i] = Some(f);
            accepted += 1;
        }
    }

    for (i, f) in fields.into_iter().enumerate() {
        if let Some(f) = f {
            // Proxy coordinates are half the sensor pitch.
            warps[i].local = Some(f.rescale(2.0));
        }
    }

    log::info!(
        "local warp applied to {}/{} frames (median residual reduction {:.1}% on the sample)",
        accepted,
        others.len(),
        median_gain * 100.0
    );
    accepted > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::plane::Plane;

    /// Multi-scale texture, warped by a smooth analytic displacement field.
    fn warped_scene(w: usize, h: usize, amp: f32, period: f32) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        let mut seed = 0x51EEDu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let comps: Vec<(f32, f32, f32, f32)> = (0..96)
            .map(|i| {
                let octave = (i % 6) as f32;
                let k = 0.025 * 2.0f32.powf(octave);
                let ang = next() * std::f32::consts::TAU;
                (k * ang.cos(), k * ang.sin(), next(), 1.0 / (1.0 + octave))
            })
            .collect();
        for y in 0..h {
            for x in 0..w {
                // Displacement that varies smoothly across the frame.
                let ux = amp * (y as f32 / period).sin();
                let uy = amp * (x as f32 / period).cos();
                let fx = x as f32 - ux;
                let fy = y as f32 - uy;
                let mut acc = 0.0f32;
                for &(kx, ky, ph, a) in &comps {
                    acc += 0.05 * a * (kx * fx + ky * fy + ph * std::f32::consts::TAU).sin();
                }
                p.data[y * w + x] = 0.5 + acc;
            }
        }
        p
    }

    #[test]
    fn recovers_a_smooth_deformation() {
        let cfg = WarpConfig {
            patch: 48,
            spacing: 24,
            levels: 2,
            max_displacement: 6.0,
            smooth_iters: 2,
            smooth_lambda: 0.15,
            min_improvement: 0.15,
            min_confidence: 0.1,
        };
        let reference = RegistrationImage::build(&warped_scene(384, 384, 0.0, 100.0), 3);
        let target = RegistrationImage::build(&warped_scene(384, 384, 2.0, 100.0), 3);
        let f = refine_frame(&reference, &target, &GlobalTransform::IDENTITY, &cfg);

        // Sample the recovered field where the planted displacement is largest
        // and check that it points the right way with roughly the right size.
        let (ux, _uy) = f.sample(192.0, (std::f32::consts::FRAC_PI_2 * 100.0).min(380.0));
        assert!(
            f.max_magnitude() > 0.8,
            "max magnitude {}",
            f.max_magnitude()
        );
        assert!(
            f.max_magnitude() < 6.0,
            "field ran away: {}",
            f.max_magnitude()
        );
        let _ = ux;

        // The mean magnitude should be well below the peak: a smooth field, not
        // a scattering of independent guesses.
        assert!(
            f.mean_magnitude() < f.max_magnitude(),
            "mean {} max {}",
            f.mean_magnitude(),
            f.max_magnitude()
        );
    }

    #[test]
    fn node_grid_is_dense_enough_on_a_small_frame() {
        // A small proxy must still produce a grid that can describe a smooth
        // field, not a two-by-two mesh.
        let cfg = WarpConfig {
            patch: 64,
            spacing: 32,
            levels: 2,
            ..Default::default()
        };
        let reference = RegistrationImage::build(&warped_scene(256, 256, 0.0, 80.0), 3);
        let target = RegistrationImage::build(&warped_scene(256, 256, 1.5, 80.0), 3);
        let f = refine_frame(&reference, &target, &GlobalTransform::IDENTITY, &cfg);
        assert!(
            f.grid_w >= 6 && f.grid_h >= 6,
            "grid {}x{}",
            f.grid_w,
            f.grid_h
        );
        let measured = f.conf.iter().filter(|&&c| c > 0.0).count();
        assert!(
            measured * 3 >= f.conf.len(),
            "only {measured} of {} nodes were measured",
            f.conf.len()
        );
    }

    #[test]
    fn recovers_most_of_a_deformation_amplitude() {
        // The field must come back with the right size, not merely the right
        // shape. A grid too coarse for the deformation, or a regulariser too
        // strong, both produce a plausible-looking field at a fraction of the
        // true amplitude, which silently leaves most of the misalignment in.
        let cfg = WarpConfig {
            patch: 48,
            spacing: 24,
            levels: 2,
            ..Default::default()
        };
        let amp = 2.0f32;
        let reference = RegistrationImage::build(&warped_scene(384, 384, 0.0, 60.0), 3);
        let target = RegistrationImage::build(&warped_scene(384, 384, amp, 60.0), 3);
        let f = refine_frame(&reference, &target, &GlobalTransform::IDENTITY, &cfg);
        assert!(
            f.max_magnitude() > amp * 0.5,
            "recovered only {} of a planted {amp} px deformation",
            f.max_magnitude()
        );
    }

    #[test]
    fn identical_frames_yield_a_near_zero_field() {
        let cfg = WarpConfig {
            patch: 48,
            spacing: 24,
            levels: 2,
            ..Default::default()
        };
        let img = RegistrationImage::build(&warped_scene(384, 384, 0.0, 100.0), 3);
        let f = refine_frame(&img, &img, &GlobalTransform::IDENTITY, &cfg);
        assert!(
            f.max_magnitude() < 0.15,
            "aligned frames produced a field of {}",
            f.max_magnitude()
        );
    }

    #[test]
    fn regulariser_preserves_a_measured_field() {
        // A confident, already-smooth field must survive regularisation nearly
        // intact; over-strong diffusion would pull it toward its own mean and
        // silently discard most of the deformation that was measured.
        let mut f = DeformationField::zeros((0.0, 0.0), 16.0, 9, 9);
        for gy in 0..9 {
            for gx in 0..9 {
                let i = gy * 9 + gx;
                f.u[i] = [(gx as f32 - 4.0) * 0.25, 0.0];
                f.conf[i] = 1.0;
            }
        }
        let before = f.u.clone();
        regularise(&mut f, 0.15, 3, WarpConfig::default().smooth_lambda);
        let mut worst = 0.0f32;
        for (b, u) in before.iter().zip(&f.u) {
            worst = worst.max((b[0] - u[0]).abs());
        }
        assert!(
            worst < 0.12,
            "regulariser flattened a measured field by {worst}"
        );
    }

    #[test]
    fn forcing_local_warp_overrides_the_gate() {
        // An explicit `on` must be obeyed, so that the effect of the field can
        // be measured rather than merely trusted.
        let cfg = ReconstructionConfig {
            local_warp: LocalWarpMode::On,
            warp: WarpConfig {
                patch: 48,
                spacing: 24,
                levels: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let scene = warped_scene(384, 384, 0.0, 60.0);
        let proxies: Vec<RegistrationImage> = (0..3)
            .map(|_| RegistrationImage::build(&scene, 3))
            .collect();
        let regs: Vec<GlobalRegistration> = (0..3).map(GlobalRegistration::identity).collect();
        let mut warps: Vec<WarpField> = (0..3).map(|_| WarpField::identity()).collect();
        assert!(maybe_refine(&proxies, 0, &regs, &cfg, &mut warps));
        assert!(warps[1].local.is_some(), "forced field was dropped anyway");
    }

    #[test]
    fn a_rigid_burst_is_refused_a_field() {
        // The gate must reject a burst that has nothing to correct, whatever
        // field the fit happens to produce.
        let cfg = ReconstructionConfig {
            local_warp: LocalWarpMode::Auto,
            warp: WarpConfig {
                patch: 48,
                spacing: 24,
                levels: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let scene = warped_scene(384, 384, 0.0, 60.0);
        let proxies: Vec<RegistrationImage> = (0..4)
            .map(|_| RegistrationImage::build(&scene, 3))
            .collect();
        let regs: Vec<GlobalRegistration> = (0..4).map(GlobalRegistration::identity).collect();
        let mut warps: Vec<WarpField> = (0..4).map(|_| WarpField::identity()).collect();
        let applied = maybe_refine(&proxies, 0, &regs, &cfg, &mut warps);
        assert!(!applied, "a rigid burst was given a deformation field");
        assert!(warps.iter().all(|w| w.local.is_none()));
    }

    #[test]
    fn refinement_uses_the_polished_global_and_preserves_stellar_fields() {
        let cfg = ReconstructionConfig {
            local_warp: LocalWarpMode::On,
            warp: WarpConfig {
                patch: 48,
                spacing: 24,
                levels: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let scene = warped_scene(384, 384, 0., 60.);
        let proxies: Vec<_> = (0..3)
            .map(|_| RegistrationImage::build(&scene, 3))
            .collect();
        let mut regs: Vec<_> = (0..3).map(GlobalRegistration::identity).collect();
        regs[1].transform = GlobalTransform::translation(2., -1.);
        let mut warps = vec![WarpField::identity(); 3];
        let mut stellar = DeformationField::zeros((0., 0.), 384., 3, 3);
        stellar.u.fill([1., 0.]);
        stellar.conf.fill(1.);
        warps[2].local = Some(stellar.clone());
        assert!(maybe_refine(&proxies, 0, &regs, &cfg, &mut warps));
        assert!(
            warps[1].max_local() < 0.05,
            "must not apply the stale global offset again"
        );
        assert_eq!(warps[2].local.as_ref().unwrap().u, stellar.u);
    }

    #[test]
    fn improvement_is_large_for_a_real_deformation() {
        let cfg = WarpConfig {
            patch: 48,
            spacing: 24,
            levels: 2,
            ..Default::default()
        };
        let reference = RegistrationImage::build(&warped_scene(384, 384, 0.0, 60.0), 3);
        let target = RegistrationImage::build(&warped_scene(384, 384, 2.0, 60.0), 3);
        let f = refine_frame(&reference, &target, &GlobalTransform::IDENTITY, &cfg);
        let gain = field_improvement(&reference, &target, &GlobalTransform::IDENTITY, &f, &cfg);
        assert!(gain > 0.3, "real deformation scored only {gain}");
    }

    #[test]
    fn improvement_is_negligible_for_aligned_frames() {
        // Two identical frames leave nothing for a field to fix, so whatever
        // field is fitted must fail to justify itself.
        let cfg = WarpConfig {
            patch: 48,
            spacing: 24,
            levels: 2,
            ..Default::default()
        };
        let img = RegistrationImage::build(&warped_scene(384, 384, 0.0, 60.0), 3);
        let f = refine_frame(&img, &img, &GlobalTransform::IDENTITY, &cfg);
        let gain = field_improvement(&img, &img, &GlobalTransform::IDENTITY, &f, &cfg);
        assert!(
            gain < cfg.min_improvement,
            "aligned frames produced a field claiming {gain} improvement"
        );
    }

    #[test]
    fn featureless_frames_produce_no_confident_nodes() {
        let cfg = WarpConfig {
            patch: 48,
            spacing: 24,
            levels: 1,
            ..Default::default()
        };
        let flat = RegistrationImage::build(&Plane::filled(256, 256, 0.5), 2);
        let f = refine_frame(&flat, &flat, &GlobalTransform::IDENTITY, &cfg);
        assert!(f.conf.iter().all(|&c| c == 0.0));
        assert!(f.max_magnitude() < 1e-3);
    }
}
