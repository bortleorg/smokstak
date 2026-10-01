//! Stage 6: reference-frame selection.
//!
//! Picking the single sharpest frame is a trap. The reference defines the
//! output grid, so a frame at the edge of the burst's motion envelope forces
//! every other frame to be extrapolated and costs coverage around the border.
//! We want a frame that is both good and *central*.

use serde::{Deserialize, Serialize};

use sr_core::config::RegistrationConfig;
use sr_core::frame::FrameQuality;
use sr_core::math::median;

use sr_core::plane::Plane;

use crate::correlate::CorrelatorCache;
use crate::global::register_pair;
use crate::pyramid::RegistrationImage;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceChoice {
    pub index: usize,
    pub reason: String,
    /// Estimated position of each frame in the burst's motion envelope, in
    /// proxy pixels relative to the provisional reference.
    pub positions: Vec<(f32, f32)>,
    /// 90th-percentile radius of the burst motion envelope, proxy pixels.
    /// A percentile rather than a maximum so one stray frame does not define it.
    pub envelope_radius: f32,
    pub scores: Vec<f32>,
}

/// How far each frame departs from what the burst agrees on.
///
/// The reference defines truth for everything downstream: the output grid, and
/// — through the robustness model — which samples count as scene and which
/// count as intrusion. Choose a frame with a bird, a car or a passer-by in it
/// and the logic inverts, because the intruder becomes the reference content
/// and every clean frame is rejected for disagreeing with it. Sharpness and
/// centrality say nothing about this.
///
/// So each frame is compared against the burst's own median image, coarsely
/// aligned. A frame carrying something transient deviates from that median far
/// more than its neighbours do, whatever its other merits.
///
/// Returns a robust per-frame deviation, in image units.
fn typicality_deviation(proxies: &[RegistrationImage], positions: &[(f32, f32)]) -> Vec<f32> {
    let n = proxies.len();
    if n < 3 {
        return vec![0.0; n];
    }
    // Coarsest level that still carries enough pixels to be meaningful. This
    // measurement is about large intrusions, not about detail.
    let depth = proxies.iter().map(|p| p.depth()).min().unwrap_or(1);
    let mut level = depth - 1;
    while level > 0
        && proxies[0]
            .level(level)
            .width
            .min(proxies[0].level(level).height)
            < 48
    {
        level -= 1;
    }
    let scale = (1 << level) as f32;
    let img0 = proxies[0].level(level);
    let (w, h) = (img0.width, img0.height);

    // Keep clear of the border, where a frame's own shift runs off the edge.
    let margin = 4usize;
    if w <= 2 * margin + 4 || h <= 2 * margin + 4 {
        return vec![0.0; n];
    }

    // Every frame resampled into reference coordinates at this level.
    let aligned: Vec<Plane<f32>> = proxies
        .iter()
        .zip(positions)
        .map(|(p, &(dx, dy))| {
            let src = p.level(level);
            let (sx, sy) = (dx / scale, dy / scale);
            let mut out = Plane::<f32>::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    // The surveyed shift maps target coordinates onto the
                    // reference, so reading the reference position back out of
                    // the target subtracts it.
                    out.data[y * w + x] = src.bilinear(x as f32 - sx, y as f32 - sy);
                }
            }
            out
        })
        .collect();

    // Per-pixel median across the burst: the scene as the burst agrees on it.
    let mut median_img = Plane::<f32>::new(w, h);
    let mut column = vec![0.0f32; n];
    for i in 0..w * h {
        for (k, a) in aligned.iter().enumerate() {
            column[k] = a.data[i];
        }
        median_img.data[i] = median(&column);
    }

    aligned
        .iter()
        .map(|a| {
            let mut dev: Vec<f32> = Vec::with_capacity((w - 2 * margin) * (h - 2 * margin));
            for y in margin..h - margin {
                for x in margin..w - margin {
                    let i = y * w + x;
                    dev.push((a.data[i] - median_img.data[i]).abs());
                }
            }
            // A high percentile rather than the median: an intrusion covers a
            // small part of the frame, and a median over the whole frame would
            // average it away to nothing.
            dev.sort_by(|p, q| p.partial_cmp(q).unwrap());
            dev[((dev.len() - 1) as f32 * 0.98) as usize]
        })
        .collect()
}

/// Choose a reference frame.
///
/// Runs a cheap coarse registration of every frame against frame 0 to learn the
/// burst's motion distribution, then scores candidates on quality, centrality
/// and how typical of the burst each frame is.
pub fn select_reference(
    proxies: &[RegistrationImage],
    qualities: &[FrameQuality],
    cfg: &RegistrationConfig,
) -> ReferenceChoice {
    assert_eq!(proxies.len(), qualities.len());
    let n = proxies.len();
    if n == 1 {
        return ReferenceChoice {
            index: 0,
            reason: "only one frame in the burst".to_string(),
            positions: vec![(0.0, 0.0)],
            envelope_radius: 0.0,
            scores: vec![1.0],
        };
    }

    // Coarse survey: translation only, few probes, shallow pyramid. This is
    // about the shape of the motion cloud, not final geometry.
    let survey = RegistrationConfig {
        pyramid_levels: cfg.pyramid_levels.min(3),
        global_patch: cfg.global_patch,
        global_probes: 6,
        ..*cfg
    };

    use rayon::prelude::*;
    let mut positions: Vec<(usize, (f32, f32), f32)> = (0..n)
        .into_par_iter()
        .map_init(CorrelatorCache::new, |cache, i| {
            if i == 0 {
                (0, (0.0, 0.0), 1.0)
            } else {
                let r = register_pair(i, &proxies[0], &proxies[i], &survey, cache);
                (i, r.centre_shift, r.confidence)
            }
        })
        .collect();
    positions.sort_by_key(|p| p.0);

    let pos: Vec<(f32, f32)> = positions.iter().map(|p| p.1).collect();
    let mut conf: Vec<f32> = positions.iter().map(|p| p.2).collect();
    // The survey anchor registers against itself, so it would otherwise score a
    // free 1.0 while every other frame carries a measured value. On a burst
    // where registration is hard that bias alone can elect frame 0, however far
    // from the centre of the motion it sits. Give it the burst's median instead.
    if conf.len() > 2 {
        conf[0] = median(&conf[1..]);
    }

    // Robust centre of the motion cloud.
    let xs: Vec<f32> = pos.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = pos.iter().map(|p| p.1).collect();
    let (cx, cy) = (median(&xs), median(&ys));
    let dists: Vec<f32> = pos
        .iter()
        .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
        .collect();
    let envelope = {
        let mut d = dists.clone();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d[((d.len() - 1) as f32 * 0.9) as usize]
    };
    let scale = median(&dists).max(0.25);

    let sharp: Vec<f32> = qualities.iter().map(|q| q.sharpness).collect();
    let med_sharp = median(&sharp).max(1e-20);

    let deviation = typicality_deviation(proxies, &pos);
    let med_dev = median(&deviation).max(1e-6);

    let scores: Vec<f32> = (0..n)
        .map(|i| {
            let q = qualities[i];
            let quality = (q.sharpness / med_sharp).min(3.0);
            let saturation = 1.0 - q.saturation_fraction.min(0.5) * 2.0;
            // Centrality falls off smoothly; a frame one median-radius from the
            // centre is worth about two thirds of a perfectly central one.
            let centrality = 1.0 / (1.0 + dists[i] / scale);
            // Frames the survey could not register are poor references
            // regardless of how sharp they look.
            let connectivity = conf[i].max(0.05);
            // A frame that disagrees with the burst is carrying something the
            // burst does not, and must not become the definition of the scene.
            let typicality = 1.0 / (1.0 + (deviation[i] / med_dev - 1.0).max(0.0));
            quality * saturation * centrality * connectivity * typicality
        })
        .collect();

    let index = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let sharpest = sharp
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let reason = if index == sharpest {
        format!(
            "sharpest frame and well centred (sharpness {:.2}x burst median, {:.2} px from motion centre)",
            sharp[index] / med_sharp,
            dists[index]
        )
    } else {
        format!(
            "best balance of sharpness and centrality (sharpness {:.2}x median vs {:.2}x for the sharpest frame, {:.2} px from motion centre vs {:.2} px)",
            sharp[index] / med_sharp,
            sharp[sharpest] / med_sharp,
            dists[index],
            dists[sharpest]
        )
    };

    let reason = if deviation[index] > med_dev * 1.5 {
        format!(
            "{reason}; note that this frame still deviates from the burst median more than most"
        )
    } else {
        reason
    };

    ReferenceChoice {
        index,
        reason,
        positions: pos,
        envelope_radius: envelope,
        scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(w: usize, h: usize, ox: f32, oy: f32, blur: usize) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        let mut seed = 0xBEEFu64;
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
        if blur > 0 { p.blur_n(blur) } else { p }
    }

    fn quality_of(p: &Plane<f32>) -> FrameQuality {
        let mut q = FrameQuality::default();
        // Tenengrad, inlined to avoid a dependency cycle with sr-quality.
        let (w, h) = (p.width, p.height);
        let mut acc = 0.0f64;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = y * w + x;
                let gx = p.data[i + 1] - p.data[i - 1];
                let gy = p.data[i + w] - p.data[i - w];
                acc += (gx * gx + gy * gy) as f64;
            }
        }
        q.sharpness = (acc / ((w - 2) * (h - 2)) as f64) as f32;
        q.registration_confidence = 1.0;
        q
    }

    #[test]
    fn prefers_a_central_frame_over_an_outlying_sharper_one() {
        let cfg = RegistrationConfig {
            global_patch: 64,
            global_probes: 6,
            ..Default::default()
        };
        // Frames 0..4 clustered near the origin, frame 5 far away but sharpest.
        let offsets = [
            (0.0f32, 0.0f32),
            (1.0, 0.5),
            (-0.8, 0.6),
            (0.5, -1.0),
            (-0.4, -0.7),
            (40.0, 30.0),
        ];
        let planes: Vec<Plane<f32>> = offsets
            .iter()
            .enumerate()
            .map(|(i, &(ox, oy))| scene(384, 384, ox, oy, if i == 5 { 0 } else { 1 }))
            .collect();
        let proxies: Vec<RegistrationImage> = planes
            .iter()
            .map(|p| RegistrationImage::build(p, 3))
            .collect();
        let quals: Vec<FrameQuality> = planes.iter().map(quality_of).collect();

        let choice = select_reference(&proxies, &quals, &cfg);
        assert_ne!(
            choice.index, 5,
            "picked the outlying frame: {}",
            choice.reason
        );
        // The survey must also have *located* the outlier: its content is
        // displaced by (+40, +30), so the transform that brings it back is the
        // negative of that.
        let (px, py) = choice.positions[5];
        assert!(
            (px + 40.0).abs() < 1.0 && (py + 30.0).abs() < 1.0,
            "outlier at ({px}, {py})"
        );
        // `envelope_radius` is a 90th percentile, so with six frames it
        // deliberately ignores the single stray one.
        assert!(
            choice.envelope_radius < 5.0,
            "envelope {}",
            choice.envelope_radius
        );
    }

    #[test]
    fn refuses_a_frame_containing_a_transient_object() {
        // Eight frames of a static scene; the last three have a bright blob
        // crossing them. Picking one of those as the reference would make the
        // blob the definition of the scene.
        let cfg = RegistrationConfig {
            global_patch: 64,
            global_probes: 6,
            ..Default::default()
        };
        let planes: Vec<Plane<f32>> = (0..8)
            .map(|i| {
                let mut p = scene(256, 256, 0.0, 0.0, 0);
                if i >= 5 {
                    let cx = 60 + (i - 5) * 40;
                    for y in 90..150 {
                        for x in cx..(cx + 60).min(256) {
                            p.data[y * 256 + x] = 0.95;
                        }
                    }
                }
                p
            })
            .collect();
        let proxies: Vec<RegistrationImage> = planes
            .iter()
            .map(|p| RegistrationImage::build(p, 3))
            .collect();
        // The intruding frames also look *sharper*, because a hard-edged blob
        // adds gradient energy. Sharpness alone would elect one of them.
        let quals: Vec<FrameQuality> = planes.iter().map(quality_of).collect();
        assert!(
            quals[7].sharpness > quals[0].sharpness,
            "test premise: the blob should raise measured sharpness"
        );

        let choice = select_reference(&proxies, &quals, &cfg);
        assert!(
            choice.index < 5,
            "picked frame {} which contains the transient object: {}",
            choice.index,
            choice.reason
        );
    }

    #[test]
    fn single_frame_burst_is_its_own_reference() {
        let cfg = RegistrationConfig::default();
        let p = scene(128, 128, 0.0, 0.0, 0);
        let proxies = vec![RegistrationImage::build(&p, 2)];
        let quals = vec![quality_of(&p)];
        let c = select_reference(&proxies, &quals, &cfg);
        assert_eq!(c.index, 0);
    }
}
