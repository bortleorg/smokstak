//! Stage 5: frame and region quality analysis.
//!
//! Quality is deliberately reported as a vector, and locally as a map. A single
//! global "sharpness" number is the wrong abstraction for a long-lens burst:
//! with atmospheric seeing, different parts of the frame are sharp in different
//! exposures, and the merge should be able to exploit that.
//!
//! All metrics run on the half-resolution guide, never on the mosaic, so that
//! CFA structure is not mistaken for detail.
//!
//! Two modules are exceptions to both statements. [`photometry`] measures how
//! bright a frame is relative to another rather than how sharp. [`stars`]
//! measures the size and shape of point sources, which cannot be done on a
//! guide that samples a star at one pixel. Both read the mosaic directly.

pub mod photometry;
pub mod stars;

use rayon::prelude::*;
use sr_core::frame::{FrameQuality, LocalQualityMap};
use sr_core::math;
use sr_core::plane::Plane;

/// Squared-gradient (Tenengrad) energy, the primary sharpness signal.
pub fn tenengrad(img: &Plane<f32>) -> f32 {
    let (w, h) = (img.width, img.height);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = img.data[i + 1] - img.data[i - 1];
            let gy = img.data[i + w] - img.data[i - w];
            acc += (gx * gx + gy * gy) as f64;
        }
    }
    (acc / ((w - 2) * (h - 2)) as f64) as f32
}

/// Multi-scale Laplacian response. Variance-of-Laplacian alone is easily fooled
/// by noise, so the response is accumulated over three scales and normalised by
/// the local signal level.
pub fn multiscale_laplacian(img: &Plane<f32>) -> f32 {
    let mut acc = 0.0f32;
    let mut cur = img.clone();
    let mut weight = 1.0f32;
    for level in 0..3 {
        if cur.width < 8 || cur.height < 8 {
            break;
        }
        let blurred = cur.blur3();
        let mut e = 0.0f64;
        for i in 0..cur.data.len() {
            let d = (cur.data[i] - blurred.data[i]) as f64;
            e += d * d;
        }
        acc += weight * (e / cur.data.len() as f64) as f32;
        weight *= 0.5;
        if level < 2 {
            cur = blurred.downsample2();
        }
    }
    acc
}

/// RMS local contrast: standard deviation of the high-pass residual.
pub fn local_contrast(img: &Plane<f32>) -> f32 {
    let blurred = img.blur_n(2);
    let mut e = 0.0f64;
    for i in 0..img.data.len() {
        let d = (img.data[i] - blurred.data[i]) as f64;
        e += d * d;
    }
    ((e / img.data.len().max(1) as f64).sqrt()) as f32
}

/// Directional gradient statistics.
///
/// Returns `(anisotropy, dominant_angle)`. Uniform defocus is isotropic; motion
/// blur suppresses gradients along one direction only, so a high anisotropy at
/// a frame level is evidence of camera movement during the exposure rather than
/// of a soft lens.
pub fn gradient_anisotropy(img: &Plane<f32>) -> (f32, f32) {
    let (w, h) = (img.width, img.height);
    if w < 3 || h < 3 {
        return (0.0, 0.0);
    }
    let (mut jxx, mut jxy, mut jyy) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = (img.data[i + 1] - img.data[i - 1]) as f64;
            let gy = (img.data[i + w] - img.data[i - w]) as f64;
            jxx += gx * gx;
            jxy += gx * gy;
            jyy += gy * gy;
        }
    }
    let n = ((w - 2) * (h - 2)) as f64;
    let (l1, l2, e) = math::eig_sym2((jxx / n) as f32, (jxy / n) as f32, (jyy / n) as f32);
    let sum = l1 + l2;
    let aniso = if sum > 1e-20 { (l1 - l2) / sum } else { 0.0 };
    (aniso, e[1].atan2(e[0]))
}

/// Blur radius proxy, in guide pixels.
///
/// Estimated from how much extra energy a known Gaussian blur removes: a
/// already-soft frame loses proportionally less than a sharp one. The value is
/// monotonic in true blur, which is all the ranking needs.
pub fn estimated_blur(img: &Plane<f32>) -> f32 {
    let e0 = tenengrad(img);
    if e0 <= 1e-20 {
        return 10.0;
    }
    let e1 = tenengrad(&img.blur_n(2));
    let ratio = (e1 / e0).clamp(1e-6, 0.999999);
    // For a Gaussian PSF, gradient energy falls off as 1/(sigma^2 + s^2); solve
    // for the sigma consistent with the observed drop after a known blur.
    let s2 = 1.0f32; // variance added by blur_n(2)
    (ratio * s2 / (1.0 - ratio)).max(0.0).sqrt()
}

/// Full quality vector for one guide image.
pub fn analyse(guide_luma: &Plane<f32>, saturation_fraction: f32) -> FrameQuality {
    let (aniso, _angle) = gradient_anisotropy(guide_luma);
    FrameQuality {
        sharpness: tenengrad(guide_luma),
        laplacian: multiscale_laplacian(guide_luma),
        contrast: local_contrast(guide_luma),
        saturation_fraction,
        estimated_blur: estimated_blur(guide_luma),
        blur_anisotropy: aniso,
        registration_confidence: 1.0,
        mean_level: guide_luma.mean(),
    }
}

/// Per-region sharpness, on a grid of `region x region` guide pixels.
///
/// This is the input to lucky-region selection: region A may be best served by
/// frames 2, 7 and 12 while region B wants 3, 4 and 15.
pub fn local_quality(guide_luma: &Plane<f32>, region: usize) -> LocalQualityMap {
    let (w, h) = (guide_luma.width, guide_luma.height);
    let gw = w.div_ceil(region);
    let gh = h.div_ceil(region);
    let mut sharpness = vec![0.0f32; gw * gh];

    sharpness
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let gx = idx % gw;
            let gy = idx / gw;
            let x0 = gx * region;
            let y0 = gy * region;
            let x1 = (x0 + region).min(w);
            let y1 = (y0 + region).min(h);
            if x1 <= x0 + 2 || y1 <= y0 + 2 {
                *out = 0.0;
                return;
            }
            let mut acc = 0.0f64;
            let mut n = 0u64;
            for y in (y0 + 1)..(y1 - 1) {
                for x in (x0 + 1)..(x1 - 1) {
                    let i = y * w + x;
                    let gx = guide_luma.data[i + 1] - guide_luma.data[i - 1];
                    let gy = guide_luma.data[i + w] - guide_luma.data[i - w];
                    acc += (gx * gx + gy * gy) as f64;
                    n += 1;
                }
            }
            *out = (acc / n.max(1) as f64) as f32;
        });

    LocalQualityMap { grid_w: gw, grid_h: gh, region, sharpness }
}

/// Rank frames per region, best first.
///
/// Returned as `grid_w * grid_h` lists of frame indices. Used by the lucky
/// backend; the caller decides how many to keep, and must weigh sampling
/// diversity as well as raw sharpness.
pub fn rank_frames_per_region(maps: &[LocalQualityMap]) -> Vec<Vec<u16>> {
    if maps.is_empty() {
        return Vec::new();
    }
    let gw = maps[0].grid_w;
    let gh = maps[0].grid_h;
    (0..gw * gh)
        .into_par_iter()
        .map(|cell| {
            let mut order: Vec<(u16, f32)> = maps
                .iter()
                .enumerate()
                .map(|(i, m)| (i as u16, *m.sharpness.get(cell).unwrap_or(&0.0)))
                .collect();
            order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            order.into_iter().map(|(i, _)| i).collect()
        })
        .collect()
}

/// Normalise a set of frame quality vectors against the burst median, so that
/// "sharpness 1.03" means "3% above this burst's typical frame".
/// Replace the gradient-derived sharpness with one measured on point sources,
/// where the burst has them.
///
/// Not an addition to the vector but a substitution, so that every consumer —
/// reference selection, `--select sharpest`, the quality table — gets the
/// better measurement without knowing it exists. `sharpness` keeps its meaning
/// throughout: larger is sharper, and one is the burst median.
///
/// The substitution needs most of the burst to have been measured. A metric
/// that is a half-flux diameter on some frames and a gradient energy on others
/// is not a ranking, it is two rankings interleaved, and the frames that
/// happened to be measured would sort against frames that happened not to be.
pub fn apply_star_sharpness(qualities: &mut [FrameQuality], stars: &[Option<stars::StarMetrics>]) -> bool {
    let measured: Vec<f32> = stars.iter().flatten().map(|s| s.hfd).filter(|h| *h > 0.0).collect();
    if measured.len() * 4 < qualities.len() * 3 || measured.len() < 3 {
        return false;
    }
    let med = math::median(&measured).max(1e-6);
    let worst = measured.iter().cloned().fold(0.0f32, f32::max);
    for (q, s) in qualities.iter_mut().zip(stars) {
        // Inverted, because a half-flux diameter is a width and sharpness is
        // not.
        //
        // A frame that could not be measured while its peers could is ranked
        // below all of them rather than at the median. Failing to measure is
        // not a neutral outcome here: the reasons a frame yields no usable
        // point sources are cloud, gross defocus and trailing so bad the
        // sources leave the measurement window, and every one of them is a
        // frame worth putting last. Giving it the median instead would hide
        // exactly the frame the operator wants dropped.
        q.sharpness = match s {
            Some(m) if m.hfd > 0.0 => med / m.hfd,
            _ => 0.9 * med / worst.max(1e-6),
        };
    }
    true
}

pub fn normalise_against_median(qualities: &mut [FrameQuality]) {
    let sharp: Vec<f32> = qualities.iter().map(|q| q.sharpness).collect();
    let med = math::median(&sharp).max(1e-20);
    for q in qualities.iter_mut() {
        q.sharpness /= med;
    }
    let lap: Vec<f32> = qualities.iter().map(|q| q.laplacian).collect();
    let medl = math::median(&lap).max(1e-20);
    for q in qualities.iter_mut() {
        q.laplacian /= medl;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: usize, h: usize, period: usize) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if ((x / period) + (y / period)).is_multiple_of(2) { 0.7 } else { 0.3 };
                p.data[y * w + x] = v;
            }
        }
        p
    }

    #[test]
    fn blurring_reduces_measured_sharpness() {
        let sharp = checkerboard(128, 128, 4);
        let soft = sharp.blur_n(3);
        assert!(tenengrad(&sharp) > tenengrad(&soft) * 2.0);
        assert!(estimated_blur(&sharp) < estimated_blur(&soft));
    }

    #[test]
    fn horizontal_streaks_read_as_anisotropic() {
        let mut p = Plane::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                // Vertical bars: gradient energy is entirely horizontal.
                p.data[y * 128 + x] = if (x / 4) % 2 == 0 { 0.8 } else { 0.2 };
            }
        }
        let (aniso, _) = gradient_anisotropy(&p);
        assert!(aniso > 0.9, "anisotropy {aniso}");

        let iso = checkerboard(128, 128, 4);
        let (aniso_iso, _) = gradient_anisotropy(&iso);
        assert!(aniso_iso < 0.2, "anisotropy {aniso_iso}");
    }

    #[test]
    fn local_quality_finds_the_detailed_half() {
        let mut p = Plane::filled(128, 64, 0.5);
        // Left half detailed, right half flat.
        for y in 0..64 {
            for x in 0..64 {
                p.data[y * 128 + x] = if (x / 2) % 2 == 0 { 0.8 } else { 0.2 };
            }
        }
        let m = local_quality(&p, 32);
        assert_eq!((m.grid_w, m.grid_h), (4, 2));
        assert!(m.sharpness[0] > m.sharpness[3] * 10.0);
    }

    #[test]
    fn per_region_ranking_prefers_the_sharper_frame() {
        let a = local_quality(&checkerboard(64, 64, 2), 32);
        let b = local_quality(&checkerboard(64, 64, 2).blur_n(4), 32);
        let ranks = rank_frames_per_region(&[b, a]);
        // Frame 1 (unblurred) should win every region.
        assert!(ranks.iter().all(|r| r[0] == 1), "{ranks:?}");
    }
}
