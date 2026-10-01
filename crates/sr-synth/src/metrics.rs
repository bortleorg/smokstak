//! Quantitative comparison against ground truth.
//!
//! PSNR alone is not enough to tell reconstruction from sharpening — an
//! over-sharpened image can score well and a genuinely resolved one can score
//! badly on a slight brightness offset. So this module also measures the
//! slanted-edge MTF, which is the measurement that actually distinguishes
//! "more detail" from "more contrast at the same detail".

use sr_core::math::{psnr, ssim_blocks};
use sr_core::plane::Plane;

/// Comparison of a reconstruction against a truth image.
#[derive(Clone, Debug)]
pub struct Comparison {
    pub psnr_db: [f32; 3],
    pub psnr_mean_db: f32,
    pub ssim: f32,
    /// Best-fit gain applied before comparison, and the residual after it.
    pub gain: f32,
    pub offset: f32,
    /// Mean absolute colour error between channels, a false-colour proxy.
    pub chroma_error: f32,
}

/// Least-squares gain and offset that best map `a` onto `b`.
///
/// Applied before scoring so that a global exposure difference — which no
/// reconstruction claim depends on — does not dominate the metric.
fn fit_gain_offset(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len() as f64;
    let (mut sa, mut sb, mut saa, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        sa += x;
        sb += y;
        saa += x * x;
        sab += x * y;
    }
    let den = n * saa - sa * sa;
    if den.abs() < 1e-12 {
        return (1.0, 0.0);
    }
    let gain = (n * sab - sa * sb) / den;
    let offset = (sb - gain * sa) / n;
    (gain as f32, offset as f32)
}

/// Compare a reconstruction with the truth, ignoring a `border` of pixels where
/// the merge has incomplete support.
pub fn compare(recon: &[Plane<f32>; 3], truth: &[Plane<f32>; 3], border: usize) -> Comparison {
    compare_excluding(recon, truth, border, &[])
}

/// The same, with disks of `(x, y, radius)` left out of the scoring entirely.
///
/// For places where the truth holds something the sensor could not record. A
/// saturated star core is the case this exists for: the exposure went to full
/// scale and stopped, so the latent peak above it was never measured and no
/// reconstruction can be held to it.
///
/// Left in, four such stars dominate a whole-frame PSNR. Substituting the truth
/// for them instead of excluding them is worse still, and was tried first: it
/// makes a set of perfectly agreeing high-value pixels, and those then dominate
/// the least-squares gain and offset fit, which flatters whichever image was
/// the more poorly scaled. Only genuine exclusion is neutral.
///
/// SSIM is the one exception. It is computed in blocks and a masked block has
/// no meaning, so there the excluded pixels are set equal in both images; a
/// block that matches perfectly contributes 1.0 to an average over thousands,
/// and the disks are well under a percent of the frame.
pub fn compare_excluding(
    recon: &[Plane<f32>; 3],
    truth: &[Plane<f32>; 3],
    border: usize,
    exclude: &[(f32, f32, f32)],
) -> Comparison {
    let (w, h) = (recon[0].width, recon[0].height);
    assert_eq!(truth[0].dims(), (w, h), "reconstruction and truth differ in size");
    let bw = w.saturating_sub(2 * border);
    let bh = h.saturating_sub(2 * border);
    assert!(bw > 8 && bh > 8, "border leaves nothing to compare");

    let crop = |p: &Plane<f32>| p.crop(border, border, bw, bh);

    // Which cropped pixels count. Empty exclusions leave every one of them in,
    // so the plain `compare` pays only one pass over a bool vector.
    let mut keep = vec![true; bw * bh];
    if !exclude.is_empty() {
        for j in 0..bh {
            for i in 0..bw {
                let (px, py) = ((i + border) as f32, (j + border) as f32);
                if exclude
                    .iter()
                    .any(|&(x, y, r)| (px - x).hypot(py - y) <= r)
                {
                    keep[j * bw + i] = false;
                }
            }
        }
    }
    let kept: Vec<usize> = keep.iter().enumerate().filter(|(_, &k)| k).map(|(i, _)| i).collect();
    assert!(kept.len() > 64, "the exclusions leave nothing to compare");

    let mut psnr_db = [0.0f32; 3];
    let mut gains = [1.0f32; 3];
    let mut offsets = [0.0f32; 3];
    let mut cropped_recon = Vec::with_capacity(3);
    let mut cropped_truth = Vec::with_capacity(3);

    for c in 0..3 {
        let r = crop(&recon[c]);
        let t = crop(&truth[c]);
        let rk: Vec<f32> = kept.iter().map(|&i| r.data[i]).collect();
        let tk: Vec<f32> = kept.iter().map(|&i| t.data[i]).collect();
        let (g, o) = fit_gain_offset(&rk, &tk);
        let adjusted: Vec<f32> = r.data.iter().map(|&v| v * g + o).collect();
        let ak: Vec<f32> = kept.iter().map(|&i| adjusted[i]).collect();
        psnr_db[c] = psnr(&ak, &tk);
        gains[c] = g;
        offsets[c] = o;
        cropped_recon.push(Plane::from_vec(bw, bh, adjusted));
        cropped_truth.push(t);
    }

    // See the note on the function: blocks cannot be masked, so the excluded
    // pixels are made to agree instead.
    if !exclude.is_empty() {
        for (r, t) in cropped_recon.iter_mut().zip(cropped_truth.iter()) {
            for (i, &k) in keep.iter().enumerate() {
                if !k {
                    r.data[i] = t.data[i];
                }
            }
        }
    }

    let ssim = (0..3)
        .map(|c| ssim_blocks(&cropped_recon[c].data, &cropped_truth[c].data, bw, bh, 8))
        .sum::<f32>()
        / 3.0;

    // Chroma error: how far the reconstruction's colour differences drift from
    // the truth's. Catches false colour that a per-channel PSNR can hide.
    let mut chroma = 0.0f64;
    for &i in &kept {
        let rg_r = cropped_recon[0].data[i] - cropped_recon[1].data[i];
        let rg_t = cropped_truth[0].data[i] - cropped_truth[1].data[i];
        let bg_r = cropped_recon[2].data[i] - cropped_recon[1].data[i];
        let bg_t = cropped_truth[2].data[i] - cropped_truth[1].data[i];
        chroma += ((rg_r - rg_t).abs() + (bg_r - bg_t).abs()) as f64;
    }

    Comparison {
        psnr_db,
        psnr_mean_db: (psnr_db[0] + psnr_db[1] + psnr_db[2]) / 3.0,
        ssim,
        gain: (gains[0] + gains[1] + gains[2]) / 3.0,
        offset: (offsets[0] + offsets[1] + offsets[2]) / 3.0,
        chroma_error: (chroma / kept.len() as f64 / 2.0) as f32,
    }
}

/// Result of a slanted-edge measurement.
#[derive(Clone, Debug)]
pub struct MtfResult {
    /// Frequency at which contrast falls to 50%, in cycles per output pixel.
    pub mtf50: f32,
    /// Frequency at 10% contrast.
    pub mtf10: f32,
    /// Peak overshoot near the edge, relative to the step height. Large values
    /// mean sharpening haloes rather than resolution.
    pub overshoot: f32,
    /// Number of pixels that contributed.
    pub samples: usize,
}

/// Measure the MTF from a slanted edge.
///
/// The edge is specified in the same coordinates as the image. Pixels in the
/// region are projected onto the edge normal to build a super-sampled edge
/// spread function; its derivative is the line spread function, and the
/// magnitude of that function's transform is the MTF.
pub fn slanted_edge_mtf(
    img: &Plane<f32>,
    edge: (f32, f32, f32, f32),
    half_width: f32,
    half_length: f32,
) -> Option<MtfResult> {
    let (x0, y0, x1, y1) = edge;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-6 {
        return None;
    }
    let (tx, ty) = (dx / len, dy / len);
    let (nx, ny) = (ty, -tx);
    let (cx, cy) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));

    // Bin at a quarter pixel: the edge slant supplies the sub-pixel phases.
    let bin = 0.25f32;
    let nbins = ((2.0 * half_width) / bin).round() as usize;
    if nbins < 16 {
        return None;
    }
    let mut sum = vec![0.0f64; nbins];
    let mut cnt = vec![0.0f64; nbins];
    let mut samples = 0usize;

    let x_lo = (cx - half_length - half_width).floor().max(0.0) as usize;
    let x_hi = ((cx + half_length + half_width).ceil() as usize).min(img.width);
    let y_lo = (cy - half_length - half_width).floor().max(0.0) as usize;
    let y_hi = ((cy + half_length + half_width).ceil() as usize).min(img.height);

    for y in y_lo..y_hi {
        for x in x_lo..x_hi {
            let px = x as f32 - cx;
            let py = y as f32 - cy;
            let along = px * tx + py * ty;
            let across = px * nx + py * ny;
            if along.abs() > half_length || across.abs() >= half_width {
                continue;
            }
            let b = ((across + half_width) / bin) as usize;
            if b >= nbins {
                continue;
            }
            sum[b] += img.data[y * img.width + x] as f64;
            cnt[b] += 1.0;
            samples += 1;
        }
    }

    // Fill any empty bins by interpolation rather than leaving a spike that the
    // derivative would turn into a false high-frequency component.
    let mut esf = vec![0.0f32; nbins];
    let mut last = None;
    for i in 0..nbins {
        if cnt[i] > 0.0 {
            esf[i] = (sum[i] / cnt[i]) as f32;
            last = Some(i);
        }
    }
    last?;
    for i in 0..nbins {
        if cnt[i] == 0.0 {
            let prev = (0..i).rev().find(|&j| cnt[j] > 0.0);
            let next = (i + 1..nbins).find(|&j| cnt[j] > 0.0);
            esf[i] = match (prev, next) {
                (Some(p), Some(n)) => {
                    let t = (i - p) as f32 / (n - p) as f32;
                    esf[p] + (esf[n] - esf[p]) * t
                }
                (Some(p), None) => esf[p],
                (None, Some(n)) => esf[n],
                (None, None) => return None,
            };
        }
    }

    // Step height, from the plateaus at each end.
    let edge_n = (nbins / 8).max(2);
    let lo: f32 = esf[..edge_n].iter().sum::<f32>() / edge_n as f32;
    let hi: f32 = esf[nbins - edge_n..].iter().sum::<f32>() / edge_n as f32;
    let step = (hi - lo).abs();
    if step < 1e-4 {
        return None;
    }

    // Overshoot: how far past the plateaus the profile swings.
    let (mut min_v, mut max_v) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &esf {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let overshoot = ((max_v - hi.max(lo)).max(0.0) + (lo.min(hi) - min_v).max(0.0)) / step;

    // Line spread function.
    let mut lsf = vec![0.0f32; nbins - 1];
    for i in 0..nbins - 1 {
        lsf[i] = esf[i + 1] - esf[i];
    }
    // Hann window to suppress the truncation of the tails.
    let m = lsf.len();
    for (i, v) in lsf.iter_mut().enumerate() {
        let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / (m - 1) as f32).cos();
        *v *= w;
    }

    // Discrete transform magnitude. `bin` is in output pixels, so frequency
    // comes out in cycles per output pixel directly.
    let dc: f32 = lsf.iter().sum::<f32>().abs();
    if dc < 1e-9 {
        return None;
    }
    let freq_step = 1.0 / (bin * m as f32);
    let nyquist = 0.5 / bin;
    let steps = (nyquist / freq_step) as usize;

    let mut mtf50 = None;
    let mut mtf10 = None;
    let mut prev = (0.0f32, 1.0f32);
    for k in 1..=steps {
        let f = k as f32 * freq_step;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, &v) in lsf.iter().enumerate() {
            let ph = -std::f32::consts::TAU * f * (i as f32 * bin);
            re += v * ph.cos();
            im += v * ph.sin();
        }
        let mag = (re * re + im * im).sqrt() / dc;
        let cur = (f, mag);
        if mtf50.is_none() && mag < 0.5 && prev.1 >= 0.5 {
            let t = (prev.1 - 0.5) / (prev.1 - mag).max(1e-9);
            mtf50 = Some(prev.0 + (f - prev.0) * t);
        }
        if mtf10.is_none() && mag < 0.1 && prev.1 >= 0.1 {
            let t = (prev.1 - 0.1) / (prev.1 - mag).max(1e-9);
            mtf10 = Some(prev.0 + (f - prev.0) * t);
        }
        prev = cur;
        if mtf50.is_some() && mtf10.is_some() {
            break;
        }
    }

    Some(MtfResult {
        mtf50: mtf50.unwrap_or(nyquist),
        mtf10: mtf10.unwrap_or(nyquist),
        overshoot,
        samples,
    })
}

/// A straight edge found inside a region.
#[derive(Clone, Copy, Debug)]
pub struct FoundEdge {
    pub line: (f32, f32, f32, f32),
    /// Edge orientation in degrees from vertical, in `[-90, 90]`.
    pub angle_deg: f32,
    /// Contrast across the edge, in image units.
    pub contrast: f32,
    /// How straight and single the edge is: the structure tensor's anisotropy,
    /// in `[0, 1]`. Low values mean the box holds texture, not one edge.
    pub straightness: f32,
}

/// Locate the dominant straight edge inside a box.
///
/// A slanted-edge measurement needs the edge's true position and angle, and
/// getting either slightly wrong biases the result. This finds both from the
/// data: the gradient-weighted centroid gives the position, and the structure
/// tensor of the same region gives the orientation.
///
/// Returns `None` when the region does not contain a single dominant straight
/// edge, so that a meaningless MTF is never reported as a real one.
pub fn fit_edge_in_box(
    img: &Plane<f32>,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Option<FoundEdge> {
    let x1 = (x0 + w).min(img.width);
    let y1 = (y0 + h).min(img.height);
    if x1 < x0 + 8 || y1 < y0 + 8 {
        return None;
    }

    let (mut jxx, mut jxy, mut jyy) = (0.0f64, 0.0f64, 0.0f64);
    let (mut cx, mut cy, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;

    for y in (y0 + 1)..(y1 - 1) {
        for x in (x0 + 1)..(x1 - 1) {
            let i = y * img.width + x;
            let gx = 0.5 * (img.data[i + 1] - img.data[i - 1]);
            let gy = 0.5 * (img.data[i + img.width] - img.data[i - img.width]);
            let m = (gx * gx + gy * gy) as f64;
            jxx += (gx * gx) as f64;
            jxy += (gx * gy) as f64;
            jyy += (gy * gy) as f64;
            cx += x as f64 * m;
            cy += y as f64 * m;
            wsum += m;
            lo = lo.min(img.data[i]);
            hi = hi.max(img.data[i]);
        }
    }
    if wsum <= 1e-12 {
        return None;
    }
    let (cx, cy) = ((cx / wsum) as f32, (cy / wsum) as f32);

    let (l1, l2, e1) = crate::metrics::eig2(jxx as f32, jxy as f32, jyy as f32);
    let sum = l1 + l2;
    let straightness = if sum > 1e-20 { ((l1 - l2) / sum).clamp(0.0, 1.0) } else { 0.0 };
    // The edge runs perpendicular to the dominant gradient.
    let (ex, ey) = (-e1[1], e1[0]);

    let half = (x1 - x0).min(y1 - y0) as f32 * 0.35;
    let line = (cx - ex * half, cy - ey * half, cx + ex * half, cy + ey * half);
    let angle_deg = ey.atan2(ex).to_degrees();
    // Fold into [-90, 90]: an edge and its reverse are the same edge.
    let angle_deg = if angle_deg > 90.0 {
        angle_deg - 180.0
    } else if angle_deg < -90.0 {
        angle_deg + 180.0
    } else {
        angle_deg
    };

    Some(FoundEdge { line, angle_deg, contrast: hi - lo, straightness })
}

/// Symmetric 2x2 eigen-decomposition, re-exported locally so this crate does
/// not depend on the core solver purely for one call.
fn eig2(a: f32, b: f32, c: f32) -> (f32, f32, [f32; 2]) {
    sr_core::math::eig_sym2(a, b, c)
}

/// Noise level in the flattest tiles of an image.
///
/// On a resolution chart the flattest tiles are the uniform patches, which is
/// exactly where noise should be judged. Each tile is detrended before its
/// residual is measured so that shading is not counted as grain.
pub use sr_noise::spatial::flat_field_noise;

/// Colour fringing at luminance edges.
///
/// A neutral edge should stay neutral. Lateral chromatic aberration displaces
/// the red and blue images relative to green, so at a high-contrast edge the
/// channels cross over at slightly different places and a coloured rim appears.
///
/// The difficulty is separating that rim from ordinary colour. A red brick
/// against a blue sky has an enormous chroma difference exactly where the
/// luminance edge is, and no fringing at all. What distinguishes them is *how*
/// the chroma relates to the luminance:
///
/// * a genuine colour boundary has chroma proportional to luminance — both step
///   at the same place, so `c = alpha * l + beta` fits it;
/// * a fringe from a channel displaced by `d` has `c ≈ d * dl/dn`, a spike at
///   the edge that returns to baseline on both sides, which no multiple of `l`
///   can reproduce.
///
/// So chroma is locally regressed onto luminance and only the residual is
/// counted. The regression is done with box statistics rather than per-pixel
/// fits, which makes it a handful of blurs.
pub fn edge_chroma_fringing(rgb: &[Plane<f32>; 3]) -> f32 {
    let (w, h) = rgb[0].dims();
    if w < 16 || h < 16 {
        return 0.0;
    }
    let n = w * h;
    let mut luma = Plane::<f32>::new(w, h);
    let mut chroma = [Plane::<f32>::new(w, h), Plane::<f32>::new(w, h)];
    for i in 0..n {
        let (r, g, b) = (rgb[0].data[i], rgb[1].data[i], rgb[2].data[i]);
        luma.data[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        chroma[0].data[i] = r - g;
        chroma[1].data[i] = b - g;
    }

    // Local first and second moments, for a least-squares fit of chroma to
    // luminance over each neighbourhood.
    const PASSES: usize = 4;
    let ml = luma.blur_n(PASSES);
    let mut ll = Plane::<f32>::new(w, h);
    for i in 0..n {
        ll.data[i] = luma.data[i] * luma.data[i];
    }
    let mll = ll.blur_n(PASSES);

    let mut grad: Vec<(f32, usize)> = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = luma.data[i + 1] - luma.data[i - 1];
            let gy = luma.data[i + w] - luma.data[i - w];
            grad.push(((gx * gx + gy * gy).sqrt(), i));
        }
    }
    if grad.is_empty() {
        return 0.0;
    }
    grad.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = (grad.len() / 10).max(1);

    let mut acc = 0.0f64;
    for ch in chroma.iter().take(2) {
        let mc = ch.blur_n(PASSES);
        let mut lc = Plane::<f32>::new(w, h);
        for i in 0..n {
            lc.data[i] = luma.data[i] * ch.data[i];
        }
        let mlc = lc.blur_n(PASSES);

        for &(_, i) in &grad[..take] {
            let var_l = (mll.data[i] - ml.data[i] * ml.data[i]).max(0.0);
            let cov = mlc.data[i] - ml.data[i] * mc.data[i];
            // With no local luminance variation there is no colour boundary to
            // explain away, and the chroma deviation stands as it is.
            let alpha = if var_l > 1e-8 { cov / var_l } else { 0.0 };
            let predicted = mc.data[i] + alpha * (luma.data[i] - ml.data[i]);
            let resid = (ch.data[i] - predicted) as f64;
            acc += resid * resid;
        }
    }
    ((acc / (2.0 * take as f64)).sqrt()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slanted step edge, optionally blurred.
    fn edge_image(n: usize, slant: f32, blur_passes: usize) -> (Plane<f32>, (f32, f32, f32, f32)) {
        let mut p = Plane::<f32>::new(n, n);
        let c = n as f32 * 0.5;
        for y in 0..n {
            for x in 0..n {
                // Edge tilted by `slant` pixels per row.
                let boundary = c + (y as f32 - c) * slant;
                p.data[y * n + x] = if (x as f32) > boundary { 0.8 } else { 0.2 };
            }
        }
        let p = if blur_passes > 0 { p.blur_n(blur_passes) } else { p };
        // Edge line, from top to bottom of the region.
        let half = n as f32 * 0.35;
        let e = (c - slant * half, c - half, c + slant * half, c + half);
        (p, e)
    }

    #[test]
    fn identical_images_score_perfectly() {
        let (p, _) = edge_image(64, 0.1, 1);
        let img = [p.clone(), p.clone(), p];
        let c = compare(&img, &img, 4);
        assert!(c.psnr_mean_db > 90.0, "psnr {}", c.psnr_mean_db);
        assert!(c.ssim > 0.999, "ssim {}", c.ssim);
        assert!((c.gain - 1.0).abs() < 1e-4);
    }

    #[test]
    fn blur_lowers_both_psnr_and_ssim() {
        let (sharp, _) = edge_image(64, 0.1, 0);
        let soft = sharp.blur_n(4);
        let a = [sharp.clone(), sharp.clone(), sharp.clone()];
        let b = [soft.clone(), soft.clone(), soft];
        let c = compare(&b, &a, 6);
        assert!(c.psnr_mean_db < 40.0, "psnr {}", c.psnr_mean_db);
        assert!(c.ssim < 0.99, "ssim {}", c.ssim);
    }

    #[test]
    fn a_pure_exposure_difference_is_discounted() {
        let (p, _) = edge_image(64, 0.1, 1);
        let scaled = Plane::from_vec(
            p.width,
            p.height,
            p.data.iter().map(|v| v * 1.3 + 0.02).collect(),
        );
        let a = [p.clone(), p.clone(), p];
        let b = [scaled.clone(), scaled.clone(), scaled];
        let c = compare(&b, &a, 4);
        assert!(c.psnr_mean_db > 60.0, "gain fitting failed: psnr {}", c.psnr_mean_db);
        assert!((c.gain - 1.0 / 1.3).abs() < 0.02, "gain {}", c.gain);
    }

    #[test]
    fn mtf_falls_when_the_edge_is_blurred() {
        let (sharp, e) = edge_image(128, 0.12, 0);
        let soft = sharp.blur_n(6);
        let a = slanted_edge_mtf(&sharp, e, 12.0, 40.0).expect("sharp mtf");
        let b = slanted_edge_mtf(&soft, e, 12.0, 40.0).expect("soft mtf");
        assert!(
            a.mtf50 > b.mtf50 * 1.5,
            "blur did not reduce MTF50: sharp {} soft {}",
            a.mtf50,
            b.mtf50
        );
        assert!(a.samples > 100);
    }

    #[test]
    fn mtf_detects_sharpening_overshoot() {
        let (sharp, e) = edge_image(128, 0.12, 2);
        // Unsharp mask: raises apparent contrast and produces haloes.
        let blurred = sharp.blur_n(3);
        let sharpened = Plane::from_vec(
            sharp.width,
            sharp.height,
            (0..sharp.data.len())
                .map(|i| sharp.data[i] + 1.2 * (sharp.data[i] - blurred.data[i]))
                .collect(),
        );
        let plain = slanted_edge_mtf(&sharp, e, 12.0, 40.0).expect("plain");
        let over = slanted_edge_mtf(&sharpened, e, 12.0, 40.0).expect("sharpened");
        assert!(
            over.overshoot > plain.overshoot + 0.02,
            "overshoot not detected: plain {} sharpened {}",
            plain.overshoot,
            over.overshoot
        );
    }

    #[test]
    fn edge_finder_recovers_position_and_angle() {
        let (img, _) = edge_image(128, 0.15, 1);
        let found = fit_edge_in_box(&img, 24, 24, 80, 80).expect("edge");
        // slant 0.15 px per row is atan(0.15) from vertical.
        let want = 0.15f32.atan().to_degrees();
        let got = 90.0 - found.angle_deg.abs();
        assert!((got - want).abs() < 2.0, "angle {got} vs {want}");
        assert!(found.straightness > 0.8, "straightness {}", found.straightness);
        assert!(found.contrast > 0.4, "contrast {}", found.contrast);
    }

    #[test]
    fn edge_finder_rejects_texture() {
        let mut p = Plane::<f32>::new(96, 96);
        let mut seed = 3u64;
        for v in p.data.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (seed >> 33) as f32 / (1u32 << 31) as f32;
        }
        let found = fit_edge_in_box(&p, 8, 8, 80, 80).expect("returns something");
        assert!(found.straightness < 0.3, "texture read as an edge: {}", found.straightness);
    }

    #[test]
    fn flat_field_noise_recovers_a_planted_sigma() {
        let mut p = Plane::filled(128, 128, 0.5);
        let mut seed = 11u64;
        let sigma = 0.01f32;
        for v in p.data.iter_mut() {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            let u1 = (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z2 = seed;
            z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z2 = (z2 ^ (z2 >> 27)).wrapping_mul(0x94D049BB133111EB);
            let u2 = (((z2 ^ (z2 >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            *v += sigma * g as f32;
        }
        let est = flat_field_noise(&p, 32);
        assert!((est / sigma - 1.0).abs() < 0.25, "estimated {est} for a planted {sigma}");
    }

    #[test]
    fn fringing_metric_responds_to_channel_misregistration() {
        // A neutral slanted edge. Displacing red against green is exactly what
        // lateral aberration does, and must raise the measure; a genuine colour
        // difference across the edge must not.
        let n = 128;
        let (mono, _) = edge_image(n, 0.12, 2);
        let clean = [mono.clone(), mono.clone(), mono.clone()];
        let base = edge_chroma_fringing(&clean);

        // Red shifted by half a pixel.
        let mut shifted = Plane::<f32>::new(n, n);
        for y in 0..n {
            for x in 0..n {
                shifted.data[y * n + x] = mono.bilinear(x as f32 + 0.5, y as f32);
            }
        }
        let fringed = [shifted, mono.clone(), mono.clone()];
        let with_ca = edge_chroma_fringing(&fringed);
        assert!(
            with_ca > base * 3.0,
            "misregistration not detected: {with_ca} vs baseline {base}"
        );

        // A strongly coloured but correctly registered edge stays clean.
        let scaled = Plane::from_vec(n, n, mono.data.iter().map(|v| v * 0.4).collect());
        let coloured = [mono.clone(), mono.clone(), scaled];
        let col = edge_chroma_fringing(&coloured);
        assert!(
            col < with_ca,
            "a colour difference was mistaken for fringing: {col} vs {with_ca}"
        );
    }

    #[test]
    fn flat_regions_have_no_measurable_edge() {
        let flat = Plane::filled(64, 64, 0.5);
        assert!(slanted_edge_mtf(&flat, (32.0, 8.0, 34.0, 56.0), 10.0, 20.0).is_none());
    }
}

/// Colour measured around point sources, ring by ring.
///
/// Two things go wrong at a star that no whole-frame colour statistic sees,
/// because both live in a few hundred pixels of a twenty-six-million-pixel
/// frame and both are confined to where the profile is steep.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StarChroma {
    /// Pixel-to-pixel scatter of the channel ratios inside the core, as a
    /// fraction of their own level.
    ///
    /// A star's core is one colour. If the three channels are each
    /// reconstructed from a different and small set of surviving samples --
    /// which is what happens when the saturated ones are discarded and nothing
    /// notices that almost all of them were -- the core comes out as random
    /// colour, and this is the number that says so.
    pub core_scatter: f32,
    /// The largest fractional drift of a channel ratio between the inner ring
    /// and an outer one.
    ///
    /// Measured from the inner ring rather than from the core so that a core
    /// ruined by saturation cannot leak into it: the two numbers are meant to
    /// name two different faults.
    ///
    /// A colour that changes with radius is a coloured halo. On a mosaic it is
    /// the expected failure: green sits on a lattice of half the pitch of red
    /// and blue, so on a steep profile green reconstructs sharper and the other
    /// two reconstruct wider, and the star acquires a rim.
    pub halo_drift: f32,
    /// How many stars both numbers are medians over.
    pub stars: usize,
}

/// Measure [`StarChroma`] around each of `stars`, at a profile width of `psf`
/// output pixels.
///
/// Returns `None` when no star could be measured, rather than a zero that would
/// read as a pass.
pub fn star_chroma(rgb: &[Plane<f32>; 3], stars: &[(f32, f32)], psf: f32) -> Option<StarChroma> {
    let psf = psf.max(0.5);
    let sky_in = psf * 5.0;
    let sky_out = psf * 7.0;
    let reach = sky_out.ceil() as i64;
    let (w, h) = (rgb[0].width as i64, rgb[0].height as i64);
    let rings = [(1.0f32, 2.0f32), (2.0, 3.0), (3.0, 4.0)];

    let mut scatters = Vec::new();
    let mut drifts = Vec::new();
    for &(sx, sy) in stars {
        if sx < reach as f32 || sy < reach as f32
            || sx >= (w - reach) as f32
            || sy >= (h - reach) as f32
        {
            continue;
        }
        // Everything below is on sky-subtracted values, and the sky comes from
        // a ring far enough out that the star does not reach it.
        let mut sky_samples: [Vec<f32>; 3] = Default::default();
        let mut core: [Vec<f32>; 3] = Default::default();
        let mut ring_sum = [[0.0f64; 3]; 3];
        let mut ring_n = [0u32; 3];
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let (px, py) = ((sx.round() as i64 + dx), (sy.round() as i64 + dy));
                let r = (px as f32 - sx).hypot(py as f32 - sy) / psf;
                let i = py as usize * rgb[0].width + px as usize;
                if r * psf >= sky_in && r * psf <= sky_out {
                    for c in 0..3 {
                        sky_samples[c].push(rgb[c].data[i]);
                    }
                } else if r <= 1.0 {
                    for c in 0..3 {
                        core[c].push(rgb[c].data[i]);
                    }
                } else {
                    for (k, &(lo, hi)) in rings.iter().enumerate() {
                        if r >= lo && r < hi {
                            for c in 0..3 {
                                ring_sum[k][c] += rgb[c].data[i] as f64;
                            }
                            ring_n[k] += 1;
                        }
                    }
                }
            }
        }
        if core[1].len() < 4 || sky_samples[1].len() < 8 {
            continue;
        }
        let sky: Vec<f32> = (0..3).map(|c| median_of(&mut sky_samples[c])).collect();
        // The sky's own scatter, so that a ring can be asked whether it holds
        // signal or only noise. A ratio of two noisy numbers near zero is
        // anything at all, and taking the worst of several such ratios makes a
        // faint star's rim look like a fault while a heavily blurred image,
        // whose rings are smooth, looks like a virtue. Both are wrong.
        let sigma_sky = {
            let mut dev: Vec<f32> = sky_samples[1].iter().map(|v| (v - sky[1]).abs()).collect();
            median_of(&mut dev) * 1.4826
        };
        let core_g = {
            let mut v: Vec<f32> = core[1].iter().map(|g| g - sky[1]).collect();
            median_of(&mut v)
        };
        if core_g < CORE_MIN_SNR * sigma_sky {
            continue;
        }

        // The core, pixel by pixel: how much does its colour vary across it?
        //
        // A pixel where some channels are at the ceiling and others are not
        // is left out. That is the edge of a saturated plateau, and the three
        // channels reach the ceiling at three different radii because the star
        // is not grey, so the ratio there changes for a reason that has
        // nothing to do with the reconstruction.
        let straddles = |i: usize| -> bool {
            let at = [
                core[0][i] >= RING_CEILING,
                core[1][i] >= RING_CEILING,
                core[2][i] >= RING_CEILING,
            ];
            at.iter().any(|&a| a) && !at.iter().all(|&a| a)
        };
        let mut worst = 0.0f32;
        for c in [0usize, 2] {
            let q: Vec<f32> = core[c]
                .iter()
                .zip(core[1].iter())
                .enumerate()
                .filter_map(|(i, (&a, &g))| {
                    let g = g - sky[1];
                    if g > 0.0 && !straddles(i) {
                        Some((a - sky[c]) / g)
                    } else {
                        None
                    }
                })
                .collect();
            if q.len() < 4 {
                continue;
            }
            let m = median_of(&mut q.clone());
            if m.abs() < 1e-6 {
                continue;
            }
            let mut dev: Vec<f32> = q.iter().map(|v| (v - m).abs()).collect();
            worst = worst.max(median_of(&mut dev) * 1.4826 / m.abs());
        }
        scatters.push(worst);

        // The rings: does that colour change as the profile falls away?
        let ratio = |k: usize, c: usize| -> Option<f32> {
            if ring_n[k] == 0 {
                return None;
            }
            let n = ring_n[k] as f64;
            let g = ring_sum[k][1] / n - sky[1] as f64;
            let a = ring_sum[k][c] / n - sky[c] as f64;
            if g > 0.0 && a > 0.0 {
                Some((a / g) as f32)
            } else {
                None
            }
        };
        // A ring counts only where every channel is both well above the noise
        // and below the ceiling: a ring inside a saturated plateau has no colour
        // to speak of, and one that is mostly noise has no colour worth
        // believing. The noise of a ring mean is the sky's scatter over the
        // root of its pixel count.
        let usable = |k: usize| -> bool {
            if ring_n[k] == 0 {
                return false;
            }
            let n = ring_n[k] as f64;
            let noise = sigma_sky as f64 / n.sqrt();
            (0..3).all(|c| {
                let m = ring_sum[k][c] / n;
                m < RING_CEILING as f64 && (m - sky[c] as f64) > RING_MIN_SNR as f64 * noise
            })
        };
        let inner = match (0..rings.len()).find(|&k| usable(k)) {
            Some(k) => k,
            None => continue,
        };
        let mut drift = 0.0f32;
        let mut judged = false;
        for c in [0usize, 2] {
            if let Some(base) = ratio(inner, c) {
                for k in inner + 1..rings.len() {
                    if !usable(k) {
                        continue;
                    }
                    if let Some(outer) = ratio(k, c) {
                        drift = drift.max((outer / base - 1.0).abs());
                        judged = true;
                    }
                }
            }
        }
        if judged {
            drifts.push(drift);
        }
    }

    if scatters.is_empty() || drifts.is_empty() {
        return None;
    }
    Some(StarChroma {
        core_scatter: median_of(&mut scatters.clone()),
        halo_drift: median_of(&mut drifts),
        stars: scatters.len(),
    })
}

/// A star's core must stand this many sky deviations above the sky to be
/// measured at all.
const CORE_MIN_SNR: f32 = 20.0;
/// A ring's sky-subtracted mean must stand this many of its own deviations
/// above zero, in every channel, before its colour is believed.
const RING_MIN_SNR: f32 = 10.0;
/// A ring whose mean reaches this, in any channel, is inside the saturated
/// plateau and has no colour.
const RING_CEILING: f32 = 0.98;

fn median_of(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

#[cfg(test)]
mod star_chroma_tests {
    use super::*;
    /// A star of the given width on a flat sky, with each channel a fixed
    /// multiple of green -- the thing a reconstruction is supposed to produce.
    fn one_star(psf: f32, colour: [f32; 3]) -> ([Plane<f32>; 3], (f32, f32)) {
        let n = 64usize;
        let (cx, cy) = (31.5f32, 31.5f32);
        let mut p = [
            Plane::filled(n, n, 0.05f32),
            Plane::filled(n, n, 0.05f32),
            Plane::filled(n, n, 0.05f32),
        ];
        for y in 0..n {
            for x in 0..n {
                let r = (x as f32 - cx).hypot(y as f32 - cy);
                let a = (-(r * r) / (2.0 * psf * psf)).exp();
                for c in 0..3 {
                    p[c].data[y * n + x] += a * colour[c];
                }
            }
        }
        (p, (cx, cy))
    }

    #[test]
    fn a_clean_star_scores_near_zero_on_both_counts() {
        let (p, at) = one_star(2.0, [1.0, 0.9, 0.8]);
        let m = star_chroma(&p, &[at], 2.0).expect("a star was placed");
        assert_eq!(m.stars, 1);
        assert!(m.core_scatter < 0.02, "core scatter {} on a clean star", m.core_scatter);
        assert!(m.halo_drift < 0.02, "halo drift {} on a clean star", m.halo_drift);
    }

    #[test]
    fn a_core_of_random_colour_is_caught_and_the_halo_is_not_blamed() {
        let (mut p, at) = one_star(2.0, [1.0, 0.9, 0.8]);
        // What discarding the saturated samples and reconstructing from the
        // survivors does: each channel independently wrong, inside the core
        // only.
        let mut seed = 12345u64;
        for y in 28..36 {
            for x in 28..36 {
                if (x as f32 - at.0).hypot(y as f32 - at.1) > 2.0 {
                    continue;
                }
                for c in [0usize, 2] {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u = ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5;
                    p[c].data[y * 64 + x] *= 1.0 + u;
                }
            }
        }
        let m = star_chroma(&p, &[at], 2.0).expect("a star was placed");
        assert!(m.core_scatter > 0.10, "speckle went unnoticed: {}", m.core_scatter);
        assert!(m.halo_drift < 0.05, "the halo was blamed for a core fault: {}", m.halo_drift);
    }

    #[test]
    fn a_coloured_halo_is_caught_and_the_core_is_not_blamed() {
        // Green reconstructed from a finer lattice: sharper than the other two,
        // so their ratio to it rises with radius and the star gets a rim. The
        // core keeps one colour throughout.
        let n = 64usize;
        let (cx, cy) = (31.5f32, 31.5f32);
        let mut p = [
            Plane::filled(n, n, 0.05f32),
            Plane::filled(n, n, 0.05f32),
            Plane::filled(n, n, 0.05f32),
        ];
        let widths = [2.4f32, 2.0, 2.4];
        let colour = [1.0f32, 0.9, 0.8];
        for y in 0..n {
            for x in 0..n {
                let r = (x as f32 - cx).hypot(y as f32 - cy);
                for c in 0..3 {
                    let w = widths[c];
                    // Same total flux, spread over a different width.
                    let a = (2.0 / (w * w)) * (-(r * r) / (2.0 * w * w)).exp();
                    p[c].data[y * n + x] += a * colour[c];
                }
            }
        }
        let m = star_chroma(&p, &[(cx, cy)], 2.0).expect("a star was placed");
        assert!(m.halo_drift > 0.10, "the rim went unnoticed: {}", m.halo_drift);
        assert!(m.core_scatter < 0.05, "the core was blamed for a rim: {}", m.core_scatter);
    }

    #[test]
    fn a_star_too_close_to_the_edge_is_declined_rather_than_guessed() {
        let (p, _) = one_star(2.0, [1.0, 0.9, 0.8]);
        assert!(star_chroma(&p, &[(3.0, 3.0)], 2.0).is_none());
    }
}
