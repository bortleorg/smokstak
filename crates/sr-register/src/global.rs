//! Stage 7: coarse-to-fine global registration.
//!
//! The loop is: probe the reference with a grid of patches, pull the matching
//! patches out of the target *through the current transform estimate*, measure
//! what is left over, fit an update, repeat. Working on residuals keeps the
//! correlator in the small-shift regime where it is most accurate, and it means
//! rotation and scale are handled by resampling rather than by a search.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use sr_core::config::RegistrationConfig;
use sr_core::geometry::{GlobalTransform, TransformModel};
use sr_core::math::{mad_sigma, median};

use crate::correlate::{CorrelatorCache, patch_for};
use crate::model::{Correspondence, fit_best, fit_model};
use crate::pyramid::{RegistrationImage, extract_patch, extract_patch_warped, probe_grid};

/// Registration of one frame against the reference, in proxy coordinates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalRegistration {
    pub frame: usize,
    pub transform: GlobalTransform,
    pub model: TransformModel,
    pub residual_rms: f32,
    pub residual_p50: f32,
    pub residual_p90: f32,
    pub residual_p99: f32,
    pub inliers: usize,
    /// Correspondences the fit was measured on -- probes that found anything
    /// at all, as opposed to probes attempted.
    pub measured: usize,
    pub probes: usize,
    pub rejected: usize,
    /// Fraction of the reference frame that the target actually covers.
    pub overlap: f32,
    /// Overall trust in this registration, `[0, 1]`.
    pub confidence: f32,
    /// How far the frame's centre moved, in proxy pixels.
    ///
    /// The centre rather than the transform's translation component, which is
    /// measured at the origin and so reports a rotation's lever arm as motion.
    pub centre_shift: (f32, f32),
    pub rotation_deg: f32,
    pub scale: f32,
}

impl GlobalRegistration {
    pub fn identity(frame: usize) -> Self {
        Self {
            frame,
            transform: GlobalTransform::IDENTITY,
            model: TransformModel::Translation,
            residual_rms: 0.0,
            residual_p50: 0.0,
            residual_p90: 0.0,
            residual_p99: 0.0,
            inliers: 0,
            measured: 0,
            probes: 0,
            rejected: 0,
            overlap: 1.0,
            confidence: 1.0,
            centre_shift: (0.0, 0.0),
            rotation_deg: 0.0,
            scale: 1.0,
        }
    }
}

/// Measure residual displacements between the reference and a target that has
/// been brought into approximate alignment by `t`.
// Eight arguments, and grouping them would mean inventing a struct that exists
// only to be unpacked here.
#[allow(clippy::too_many_arguments)]
fn probe_residuals(
    reference: &RegistrationImage,
    target: &RegistrationImage,
    level: usize,
    t: &GlobalTransform,
    max_across: usize,
    min_peak_ratio: f32,
    patch: usize,
    cache: &mut CorrelatorCache,
) -> (Vec<Correspondence>, usize) {
    let ref_img = reference.level(level);
    let tgt_img = target.level(level);
    let inv = match t.inverse() {
        Some(i) => i,
        None => return (Vec::new(), 0),
    };
    let probes = probe_grid(ref_img.width, ref_img.height, patch, max_across);

    let mut out = Vec::with_capacity(probes.len());
    let mut rejected = 0usize;
    let mut buf_ref = vec![0.0f32; patch * patch];
    let mut buf_tgt = vec![0.0f32; patch * patch];
    let corr = cache.get(patch);

    for &(cx, cy) in &probes {
        extract_patch(ref_img, cx, cy, patch, &mut buf_ref);
        if !extract_patch_warped(tgt_img, &inv, cx, cy, patch, &mut buf_tgt) {
            rejected += 1;
            continue;
        }
        match corr.shift(&buf_ref, &buf_tgt) {
            Some(s) if s.peak_ratio >= min_peak_ratio => {
                // Beyond about 40% of the patch the Hann-windowed overlap is
                // too small to trust: such a peak is a different match, not a
                // refinement. Coarse pyramid levels use small patches, so this
                // bound is what decides how much motion the search can follow.
                if s.magnitude() > patch as f32 * 0.4 {
                    rejected += 1;
                    continue;
                }
                out.push(Correspondence {
                    x: cx,
                    y: cy,
                    rx: s.dx,
                    ry: s.dy,
                    // Confidence grows with peak distinctiveness but saturates:
                    // one spectacular probe should not outvote twenty good ones.
                    weight: (s.peak_ratio - 1.0).clamp(0.0, 4.0),
                });
            }
            _ => rejected += 1,
        }
    }
    (out, rejected)
}

// Residual scale in level pixels at which a set of correspondences stops
// being evidence. A fit that agrees is two orders of magnitude tighter
// than one that does not -- tenths of a pixel against tens -- so anywhere
// in the gap does, and a pixel is the value with a meaning: the model
// explains where every probe landed to within one pixel of the level it
// was measured on.
const AGREES_WITHIN: f32 = 1.0;

/// An under-parameterised fine-level fit must not prevent the supported affine
/// model from being tried. Keep the same evidence threshold for that model.
fn supported_update(
    corrs: &[Correspondence],
    fit: crate::model::FitResult,
    finest: bool,
    agreed: bool,
    cfg: &RegistrationConfig,
) -> Option<crate::model::FitResult> {
    if fit.residual_p50 <= AGREES_WITHIN || !agreed {
        return Some(fit);
    }
    if finest && corrs.len() >= 8 {
        let richer = fit_best(corrs, cfg.model_selection_tolerance, cfg.irls_iters)?;
        if richer.residual_p50 <= AGREES_WITHIN {
            return Some(richer);
        }
    }
    None
}

/// Register one target frame against the reference.
pub fn register_pair(
    frame: usize,
    reference: &RegistrationImage,
    target: &RegistrationImage,
    cfg: &RegistrationConfig,
    cache: &mut CorrelatorCache,
) -> GlobalRegistration {
    register_pair_from(
        frame,
        reference,
        target,
        cfg,
        cache,
        GlobalTransform::IDENTITY,
    )
}

/// The same, started from a given estimate rather than from the identity.
///
/// Refinement can follow a burst that drifts and cannot follow one that flips:
/// a mount crossing the meridian turns the camera through 180 degrees, and no
/// amount of correlation from a standing start will find that. A seed from the
/// frame's plate solve puts the search in the right place; the pixels do the
/// rest, exactly as before.
pub fn register_pair_from(
    frame: usize,
    reference: &RegistrationImage,
    target: &RegistrationImage,
    cfg: &RegistrationConfig,
    cache: &mut CorrelatorCache,
    seed: GlobalTransform,
) -> GlobalRegistration {
    let depth = reference
        .depth()
        .min(target.depth())
        .min(cfg.pyramid_levels);

    // Transform is carried in level-0 proxy coordinates throughout; only the
    // translation changes meaning between levels.
    let mut t = seed;
    let mut last: Option<crate::model::FitResult> = None;
    let mut probes_total = 0usize;
    let mut rejected_total = 0usize;

    // Whether any level has agreed yet. Until one has, an update is better
    // than nothing: a frame left at identity is worse than one placed roughly.
    let mut agreed = false;

    for level in (0..depth).rev() {
        let s = (1 << level) as f32;
        let ref_level = reference.level(level);
        // The patch has to fit the level, not the other way round.
        let Some(patch) = patch_for(ref_level.width, ref_level.height, cfg.global_patch) else {
            continue;
        };
        // Express the current estimate in this level's coordinates.
        let mut t_l = t.rescale(1.0 / s);

        // Coarse levels only need to find the gross offset, so a translation is
        // both sufficient and far better conditioned. The richer models are
        // fitted where the residuals are meaningful.
        let iters_here = if level == 0 { 3 } else { 2 };
        for it in 0..iters_here {
            let (corrs, rejected) = probe_residuals(
                reference,
                target,
                level,
                &t_l,
                cfg.global_probes,
                cfg.min_peak_ratio,
                patch,
                cache,
            );
            probes_total = corrs.len() + rejected;
            rejected_total = rejected;
            if corrs.is_empty() {
                break;
            }
            // Only fit what the evidence supports. A single good probe still
            // determines a translation, and refusing to use it would leave the
            // frame at identity, which is far worse than an
            // under-parameterised fit.
            let fit = if corrs.len() < 8 {
                fit_model(&corrs, TransformModel::Translation, cfg.irls_iters)
            } else if level == 0 && it + 1 == iters_here {
                fit_best(&corrs, cfg.model_selection_tolerance, cfg.irls_iters)
            } else if level == 0 {
                fit_model(&corrs, TransformModel::Similarity, cfg.irls_iters)
            } else {
                fit_model(&corrs, TransformModel::Translation, cfg.irls_iters)
            };
            let Some(fit) = fit else { break };

            // Does this level's evidence agree with itself?
            //
            // On a wide-field star field the finest level does not. The guide
            // image is half sensor resolution, so a star is barely more than
            // one pixel across, and correlating two such patches returns
            // shifts spread uniformly across the search radius with no mode at
            // all -- median residual twenty-seven pixels where a level that
            // works gives a tenth of one. Composing that update is a random
            // walk away from an answer the coarse levels already found, and
            // taking its residual as the frame's quality metric is how a burst
            // that stacks to round stars came to report a sixty-pixel
            // registration error.
            //
            // Residual magnitude cannot be compared between levels -- a coarse
            // level is smoother and always looks more accurate -- so the test
            // is against a fixed scale in the level's own pixels, not against
            // the other levels.
            log::trace!(
                "frame {frame} level {level} it {it}: n {} p50 {:.2} rms {:.2} inliers {}",
                corrs.len(),
                fit.residual_p50,
                fit.residual_rms,
                fit.inliers
            );
            let Some(fit) = supported_update(&corrs, fit, level == 0, agreed, cfg) else {
                break;
            };
            agreed = agreed || fit.residual_p50 <= AGREES_WITHIN;

            // The fit is an update in reference coordinates; compose it on the
            // left of the current estimate.
            t_l = fit.transform.compose(&t_l);
            // Residuals are in this level's pixels. Scaled to level-0 ones so
            // that what is reported means the same thing whichever level
            // produced it.
            let mut scaled = fit.clone();
            scaled.residual_rms *= s;
            scaled.residual_p50 *= s;
            scaled.residual_p90 *= s;
            scaled.residual_p99 *= s;
            last = Some(scaled);
        }
        t = t_l.rescale(s);
    }

    let (w, h) = reference.dims();
    let overlap = overlap_fraction(&t, w, h);
    let (_fitted_kind, rms, p50, p90, p99, inliers, corrs_used) = match &last {
        Some(f) => (
            f.model,
            f.residual_rms,
            f.residual_p50,
            f.residual_p90,
            f.residual_p99,
            f.inliers,
            f.total,
        ),
        None => (
            TransformModel::Translation,
            f32::INFINITY,
            0.0,
            0.0,
            0.0,
            0,
            0,
        ),
    };

    // Provisional confidence, from what this pair alone can know. The residual
    // scale here is a fixed guess; `normalise_confidence` replaces it with one
    // measured from the burst as soon as the whole set has been registered.
    let inlier_frac = inliers as f32 / probes_total.max(1) as f32;
    let rt = 1.0 / (1.0 + (rms / 0.25).max(0.0));
    let confidence = (inlier_frac * rt * overlap).clamp(0.0, 1.0);

    GlobalRegistration {
        frame,
        transform: t,
        // What the accumulated transform does, not what its last refinement
        // step fitted. A tenth of a sensor pixel is the threshold, which is a
        // twentieth of a proxy pixel.
        model: t.effective_model(w as f32, h as f32, 0.05),
        residual_rms: if rms.is_finite() { rms } else { 9999.0 },
        residual_p50: p50,
        residual_p90: p90,
        residual_p99: p99,
        inliers,
        measured: corrs_used,
        probes: probes_total,
        rejected: rejected_total,
        overlap,
        confidence,
        centre_shift: t.centre_offset(w as f32, h as f32),
        rotation_deg: t.rotation().to_degrees(),
        scale: t.scale(),
    }
}

/// Fraction of the reference raster that the transformed target still covers.
fn overlap_fraction(t: &GlobalTransform, w: usize, h: usize) -> f32 {
    let Some(inv) = t.inverse() else { return 0.0 };
    let (wf, hf) = (w as f32, h as f32);
    let mut inside = 0;
    let mut total = 0;
    // Sample the reference raster coarsely; exact area is not needed.
    let step = 16;
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let (sx, sy) = inv.apply(x as f32, y as f32);
            if sx >= 0.0 && sy >= 0.0 && sx < wf && sy < hf {
                inside += 1;
            }
            total += 1;
            x += step;
        }
        y += step;
    }
    inside as f32 / total.max(1) as f32
}

/// How much of the measured evidence the transform explains.
///
/// The caller passes the smaller of two counts, because each saturates where
/// the other discriminates. `inliers` is counted against a scale estimated
/// from the residuals: on a clean scene that scale is small and the count is a
/// real quality signal, and on a star field half the correspondences are false
/// matches, the scale inflates, and it admits them all. `tight` is counted
/// against a fixed pixel: on a star field it separates the frames that placed
/// from the frames that did not, and on a clean scene every correspondence is
/// inside it and it says nothing. Neither alone is enough.
///
/// Divided by what was measured rather than by what was attempted -- a probe
/// that found nothing to correlate, which on a sparse star field is most of
/// them, is not a registration failure -- and held down when little was
/// measured, because three correspondences agreeing is not the statement that
/// two hundred agreeing is.
fn agreement(agreeing: usize, measured: usize) -> f32 {
    if measured == 0 {
        return 0.0;
    }
    let share = (agreeing as f32 / measured as f32).min(1.0);
    let evidence = (measured as f32 / MIN_CORRESPONDENCES as f32).min(1.0);
    share * evidence
}

/// Correspondences below which a fit is not yet evidence of anything.
const MIN_CORRESPONDENCES: usize = 20;

/// Re-express per-frame confidence relative to the rest of the burst.
///
/// What a caller needs from `confidence` is "should I trust this frame as much
/// as the others?", and that cannot be answered by comparing a residual against
/// a constant. The residual a good fit leaves behind is a property of the
/// scene: a tripod shot of a flat test chart settles around 0.1 proxy pixels,
/// while a hand-held burst of a textured roof at 400 mm settles around 0.6 —
/// with the same inlier fractions, the same overlap, and nothing wrong with it.
///
/// Judged against a fixed scale the second burst reads as 40% failures, and
/// those frames then get down-weighted in the merge for the crime of being
/// photographed outdoors. So the scale comes from the burst: a frame is
/// penalised only for the residual it carries *in excess* of its peers, in
/// units of their spread.
pub fn normalise_confidence(regs: &mut [GlobalRegistration]) {
    // The median correspondence residual, not the RMS. Both are per frame and
    // the median is the robust one: half a star field's correspondences are
    // false matches, which inflates an RMS and leaves a median alone.
    let res: Vec<f32> = regs
        .iter()
        .filter(|r| r.probes > 0 && r.residual_p50.is_finite())
        .map(|r| r.residual_p50)
        .collect();
    if res.len() < 4 {
        return;
    }
    let med = median(&res);
    // Floor the spread so an unusually consistent burst does not make ordinary
    // variation look like failure.
    let spread = mad_sigma(&res).max(0.15 * med).max(1e-4);

    for r in regs.iter_mut() {
        if r.probes == 0 {
            continue;
        }
        let excess = ((r.residual_p50 - med) / (3.0 * spread)).max(0.0);
        let residual_term = 1.0 / (1.0 + excess);
        // `inliers` over `probes` was two mistakes at once. A probe that
        // produced no correspondence -- featureless sky, and on a star field
        // that is most of them -- is not a registration failure, so dividing
        // by every probe attempted penalised every star field equally. And
        // `inliers` is counted against a scale estimated from the residuals,
        // which inflates when half of them are false matches and then admits
        // those too. What is wanted is the share of the evidence the transform
        // actually explains, so: correspondences within a pixel, over
        // correspondences measured, scaled down when there were too few of
        // them to mean anything.
        r.confidence =
            (agreement(r.inliers, r.measured) * residual_term * r.overlap).clamp(0.0, 1.0);
    }
}

/// Register a whole burst against one reference frame, in parallel.
pub fn register_burst(
    proxies: &[RegistrationImage],
    reference: usize,
    cfg: &RegistrationConfig,
) -> Vec<GlobalRegistration> {
    register_burst_seeded(proxies, reference, cfg, &vec![None; proxies.len()])
}

/// How far from the identity a seed has to be before it is worth trying.
///
/// Below this the correlator finds the frame on its own and a seed can only
/// add a failure mode. Above it — a different night's framing, or a meridian
/// flip — the correlator has nothing to lock onto and the seed is the whole
/// difference between using a frame and discarding it.
const SEED_WORTH_TRYING_PX: f32 = 20.0;

/// Register a burst, offered a seed for each frame.
///
/// A seed is a suggestion and never an answer. Where one is materially
/// different from the identity, the frame is registered twice — once from each
/// start — and the result that actually agrees with the pixels is kept. A seed
/// from a stale or wrong plate solve therefore costs time and changes nothing
/// else.
pub fn register_burst_seeded(
    proxies: &[RegistrationImage],
    reference: usize,
    cfg: &RegistrationConfig,
    seeds: &[Option<GlobalTransform>],
) -> Vec<GlobalRegistration> {
    let completed = std::sync::atomic::AtomicUsize::new(0);
    let started = std::time::Instant::now();
    let mut out: Vec<GlobalRegistration> = (0..proxies.len())
        .into_par_iter()
        .map_init(CorrelatorCache::new, |cache, i| {
            if i == reference {
                return GlobalRegistration::identity(i);
            }
            register_pair_seeded(
                i,
                &proxies[reference],
                &proxies[i],
                cfg,
                cache,
                seeds.get(i).copied().flatten(),
            )
        })
        .inspect(|_| {
            let count = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if proxies.len() >= 32
                && (count == 1 || count.is_multiple_of(25) || count == proxies.len())
            {
                log::info!(
                    "global registration: measured {count}/{} frames in {:.1?}",
                    proxies.len(),
                    started.elapsed()
                );
            }
        })
        .collect();
    out.sort_by_key(|r| r.frame);
    normalise_confidence(&mut out);
    out
}

/// Register a pair with the same seed-versus-identity decision as a burst.
/// Confidence is raw; callers normalise it over their completed filter group.
pub fn register_pair_seeded(
    i: usize,
    reference: &RegistrationImage,
    target: &RegistrationImage,
    cfg: &RegistrationConfig,
    cache: &mut CorrelatorCache,
    seed: Option<GlobalTransform>,
) -> GlobalRegistration {
    let (w, h) = reference.dims();
    let seed = seed.filter(|s| {
        let (dx, dy) = s.centre_offset(w as f32, h as f32);
        dx.hypot(dy) > SEED_WORTH_TRYING_PX || s.rotation().to_degrees().abs() > 0.5
    });
    let Some(seed) = seed else {
        return register_pair(i, reference, target, cfg, cache);
    };

    // Both starts, always. Accepting a seed that merely converged
    // would be a third of the registration time cheaper and is not
    // sound: a wrong seed can converge, and on a scene with any
    // symmetry it does — which is what `a_wrong_seed_is_discarded`
    // demonstrates. The frame's own pixels get to decide.
    let seeded = register_pair_from(i, reference, target, cfg, cache, seed);
    let plain = register_pair(i, reference, target, cfg, cache);
    if seeded.residual_p50 < plain.residual_p50 {
        seeded
    } else {
        plain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::plane::Plane;

    #[test]
    fn fine_affine_evidence_is_not_rejected_by_a_similarity_residual() {
        let cfg = RegistrationConfig::default();
        let corrs: Vec<_> = (0..100)
            .map(|i| {
                let x = (i % 10) as f32 * 100.;
                let y = (i / 10) as f32 * 100.;
                Correspondence {
                    x,
                    y,
                    rx: 0.02 * x,
                    ry: -0.02 * y,
                    weight: 1.,
                }
            })
            .collect();
        let similarity = fit_model(&corrs, TransformModel::Similarity, cfg.irls_iters).unwrap();
        assert!(similarity.residual_p50 > AGREES_WITHIN);
        let accepted = supported_update(&corrs, similarity.clone(), true, true, &cfg).unwrap();
        assert!(accepted.residual_p50 < 0.01);
        assert!(supported_update(&corrs, similarity, false, true, &cfg).is_none());
        let scattered: Vec<_> = corrs
            .iter()
            .enumerate()
            .map(|(i, c)| Correspondence {
                rx: ((i * 37 % 101) as f32 - 50.) * 0.6,
                ry: ((i * 61 % 97) as f32 - 48.) * 0.6,
                ..*c
            })
            .collect();
        let fit = fit_model(&scattered, TransformModel::Similarity, cfg.irls_iters).unwrap();
        assert!(supported_update(&scattered, fit, true, true, &cfg).is_none());
    }

    fn scene(w: usize, h: usize, ox: f32, oy: f32) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        let mut seed = 0xC0FFEEu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        // Log-spaced spatial frequencies from very coarse to near Nyquist.
        // A single-scale scene would vanish after two decimations and the
        // coarse pyramid levels would have nothing to lock onto - which is not
        // how real imagery behaves.
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
                let fx = x as f32 - ox;
                let fy = y as f32 - oy;
                let mut acc = 0.0f32;
                for &(kx, ky, ph, amp) in &comps {
                    acc += 0.05 * amp * (kx * fx + ky * fy + ph * std::f32::consts::TAU).sin();
                }
                p.data[y * w + x] = 0.5 + acc;
            }
        }
        p
    }

    #[test]
    fn recovers_a_known_translation_between_frames() {
        let cfg = RegistrationConfig {
            global_patch: 64,
            global_probes: 8,
            ..Default::default()
        };
        let a = RegistrationImage::build(&scene(512, 384, 0.0, 0.0), 3);
        // Target content displaced by (+2.4, -1.3).
        let b = RegistrationImage::build(&scene(512, 384, 2.4, -1.3), 3);
        let mut cache = CorrelatorCache::new();
        let r = register_pair(1, &a, &b, &cfg, &mut cache);
        // Target-to-reference transform must undo the content displacement.
        assert!(
            (r.centre_shift.0 + 2.4).abs() < 0.1,
            "dx {:?}",
            r.centre_shift
        );
        assert!(
            (r.centre_shift.1 - 1.3).abs() < 0.1,
            "dy {:?}",
            r.centre_shift
        );
        assert!(r.residual_rms < 0.1, "rms {}", r.residual_rms);
        assert!(r.confidence > 0.5, "confidence {}", r.confidence);
    }

    /// A field of stars: isolated Gaussian points on a flat background, which
    /// is what registering astronomical frames actually has to work with. It
    /// matters that this is not the sinusoid scene above — a sum of sinusoids
    /// is symmetric enough under a half turn that correlation finds the flip
    /// unaided, and a star field is not.
    fn star_field(w: usize, h: usize, ox: f32, oy: f32) -> Plane<f32> {
        let mut p = Plane::filled(w, h, 0.02);
        let mut seed = 0x5EEDu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / (1u32 << 31) as f32
        };
        let sigma = 1.6f32;
        for _ in 0..320 {
            let sx = next() * (w as f32 - 20.0) + 10.0 - ox;
            let sy = next() * (h as f32 - 20.0) + 10.0 - oy;
            let amp = 0.2 + 0.8 * next();
            let r = 5i64;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y) = (sx.round() as i64 + dx, sy.round() as i64 + dy);
                    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                        continue;
                    }
                    let (fx, fy) = (x as f32 - sx, y as f32 - sy);
                    let g = (-(fx * fx + fy * fy) / (2.0 * sigma * sigma)).exp();
                    p.data[y as usize * w + x as usize] += amp * g;
                }
            }
        }
        p
    }

    /// Rotate a plane by half a turn about its centre, which is what a mount
    /// crossing the meridian does to the camera.
    fn half_turn(p: &Plane<f32>) -> Plane<f32> {
        let mut out = Plane::new(p.width, p.height);
        for y in 0..p.height {
            for x in 0..p.width {
                out.data[y * p.width + x] =
                    p.data[(p.height - 1 - y) * p.width + (p.width - 1 - x)];
            }
        }
        out
    }

    /// The case plate-solve seeding exists for.
    ///
    /// Correlation refines and does not search, so from a standing start a
    /// flipped frame registers to nothing. Given the seed the solve provides,
    /// the same code finds it — and this is the whole difference between using
    /// a night's data and discarding it.
    #[test]
    fn a_flipped_frame_needs_a_seed_and_then_registers() {
        let cfg = RegistrationConfig::default();
        let a = RegistrationImage::build(&star_field(384, 384, 0.0, 0.0), 3);
        let b = RegistrationImage::build(&half_turn(&star_field(384, 384, 2.0, -1.0)), 3);

        let mut cache = CorrelatorCache::new();
        let cold = register_pair(1, &a, &b, &cfg, &mut cache);
        assert!(
            cold.residual_p50 > 2.0,
            "a half turn was found from the identity, so this test proves nothing: p50 {}",
            cold.residual_p50
        );

        // A half turn about the centre, in the coordinates registration uses.
        let (w, h) = (384.0f32, 384.0f32);
        let seed = GlobalTransform {
            m: [-1.0, 0.0, w - 1.0, 0.0, -1.0, h - 1.0],
        };
        let warm = register_pair_from(1, &a, &b, &cfg, &mut cache, seed);
        assert!(
            warm.residual_p50 < 1.0,
            "seeded registration still did not converge: p50 {}",
            warm.residual_p50
        );
        assert!(warm.residual_p50 < cold.residual_p50 * 0.2);
        // And the recovered transform really is the half turn.
        let rot = warm.transform.rotation().to_degrees().abs();
        assert!((rot - 180.0).abs() < 1.0, "recovered rotation {rot}");
    }

    /// A seed is offered, not obeyed. One that is wrong has to lose to the
    /// frame's own pixels, or a stale plate solve would quietly ruin a burst.
    #[test]
    fn a_wrong_seed_is_discarded() {
        let cfg = RegistrationConfig::default();
        let proxies = vec![
            RegistrationImage::build(&scene(384, 384, 0.0, 0.0), 3),
            RegistrationImage::build(&scene(384, 384, 2.4, -1.3), 3),
        ];
        let truth = register_burst(&proxies, 0, &cfg);

        // A seed claiming a half turn that did not happen.
        let liar = GlobalTransform {
            m: [-1.0, 0.0, 383.0, 0.0, -1.0, 383.0],
        };
        let seeded = register_burst_seeded(&proxies, 0, &cfg, &[None, Some(liar)]);

        let (tx, ty) = truth[1].transform.centre_offset(384.0, 384.0);
        let (sx, sy) = seeded[1].transform.centre_offset(384.0, 384.0);
        assert!(
            (tx - sx).abs() < 0.05 && (ty - sy).abs() < 0.05,
            "a wrong seed changed the answer: ({tx}, {ty}) became ({sx}, {sy})"
        );
    }

    #[test]
    fn identity_registration_is_exact() {
        let cfg = RegistrationConfig {
            global_patch: 64,
            global_probes: 6,
            ..Default::default()
        };
        let a = RegistrationImage::build(&scene(384, 384, 0.0, 0.0), 3);
        let mut cache = CorrelatorCache::new();
        let r = register_pair(1, &a, &a, &cfg, &mut cache);
        assert!(r.centre_shift.0.abs() < 0.02, "dx {:?}", r.centre_shift);
        assert!(r.centre_shift.1.abs() < 0.02, "dy {:?}", r.centre_shift);
    }

    #[test]
    fn confidence_is_relative_to_the_burst_not_an_absolute_residual() {
        // Two bursts, identical in every way that matters, differing only in
        // the residual scale their scene supports. Both should be trusted.
        let make = |rms: f32| -> Vec<GlobalRegistration> {
            (0..12)
                .map(|i| {
                    let mut r = GlobalRegistration::identity(i);
                    if i > 0 {
                        r.probes = 100;
                        r.inliers = 90;
                        // Ninety correspondences measured, all agreeing: the
                        // scene's residual scale is not what confidence is
                        // about.
                        r.measured = 95;
                        r.overlap = 0.98;
                        // A little spread, as any real burst has.
                        r.residual_p50 = rms * (1.0 + 0.05 * ((i % 5) as f32 - 2.0));
                    }
                    r
                })
                .collect()
        };

        let mut tight = make(0.10);
        let mut loose = make(0.60);
        normalise_confidence(&mut tight);
        normalise_confidence(&mut loose);

        let conf = |v: &[GlobalRegistration]| {
            let c: Vec<f32> = v
                .iter()
                .filter(|r| r.probes > 0)
                .map(|r| r.confidence)
                .collect();
            sr_core::math::median(&c)
        };
        let (a, b) = (conf(&tight), conf(&loose));
        assert!(a > 0.8, "tight burst confidence {a}");
        assert!(b > 0.8, "loose burst confidence {b}");
        assert!(
            (a - b).abs() < 0.05,
            "confidence depended on the scene's residual scale: {a} vs {b}"
        );
    }

    /// The failure this replaced.
    ///
    /// A star field answers roughly half its probes: the rest land on empty
    /// sky and correlate against nothing. Dividing agreement by every probe
    /// attempted therefore capped every star field's confidence near a half,
    /// and frame weight is confidence times sharpness -- so a 211-frame set of
    /// NGC 7023 merged with an effective 105 frames, throwing away half the
    /// data for the crime of being a picture of stars.
    #[test]
    fn empty_sky_does_not_count_against_a_frame() {
        let mut regs: Vec<GlobalRegistration> = (0..12)
            .map(|i| {
                let mut r = GlobalRegistration::identity(i);
                if i > 0 {
                    // Two hundred probes, ninety of which found anything at
                    // all, and every one of those agreeing with the fit.
                    r.probes = 200;
                    r.inliers = 90;
                    r.measured = 90;
                    r.overlap = 0.99;
                    r.residual_p50 = 0.20;
                }
                r
            })
            .collect();
        normalise_confidence(&mut regs);
        let c = regs[3].confidence;
        assert!(
            c > 0.9,
            "a frame whose every correspondence agreed scored {c}"
        );
    }

    /// And the other half of it: a frame whose correspondences were measured
    /// but do not agree is exactly what low confidence is for.
    #[test]
    fn correspondences_that_disagree_do_count_against_a_frame() {
        let mut regs: Vec<GlobalRegistration> = (0..12)
            .map(|i| {
                let mut r = GlobalRegistration::identity(i);
                if i > 0 {
                    r.probes = 200;
                    r.inliers = 90;
                    r.measured = 90;
                    r.inliers = if i == 5 { 4 } else { 88 };
                    r.overlap = 0.99;
                    r.residual_p50 = if i == 5 { 30.0 } else { 0.20 };
                }
                r
            })
            .collect();
        normalise_confidence(&mut regs);
        assert!(
            regs[5].confidence < 0.2,
            "the odd one out scored {}",
            regs[5].confidence
        );
        assert!(
            regs[4].confidence > 0.8,
            "an ordinary frame scored {}",
            regs[4].confidence
        );
    }

    #[test]
    fn confidence_still_singles_out_a_genuine_outlier() {
        let mut regs: Vec<GlobalRegistration> = (0..12)
            .map(|i| {
                let mut r = GlobalRegistration::identity(i);
                if i > 0 {
                    r.probes = 100;
                    r.inliers = 90;
                    r.measured = 95;
                    r.overlap = 0.98;
                    r.residual_p50 = 0.60 + 0.02 * ((i % 5) as f32 - 2.0);
                }
                r
            })
            .collect();
        // One frame that really did fail.
        regs[7].residual_p50 = 6.0;
        regs[7].inliers = 20;
        normalise_confidence(&mut regs);
        let typical = regs[3].confidence;
        assert!(
            regs[7].confidence < typical * 0.4,
            "outlier scored {} against a typical {typical}",
            regs[7].confidence
        );
    }

    #[test]
    fn reports_low_confidence_on_unrelated_content() {
        let cfg = RegistrationConfig {
            global_patch: 64,
            global_probes: 6,
            ..Default::default()
        };
        let a = RegistrationImage::build(&scene(384, 384, 0.0, 0.0), 3);
        let flat = RegistrationImage::build(&Plane::filled(384, 384, 0.5), 3);
        let mut cache = CorrelatorCache::new();
        let r = register_pair(1, &a, &flat, &cfg, &mut cache);
        assert!(r.confidence < 0.2, "confidence {}", r.confidence);
    }
}
