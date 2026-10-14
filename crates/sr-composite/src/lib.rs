//! Combining separately-stacked channels into one image.
//!
//! A narrowband set is not one burst. Each filter is stacked on its own —
//! merging them would be merging different measurements of the sky — and what
//! comes out is several single-channel masters that have to be brought together
//! afterwards. This is that step, and it is three separate problems:
//!
//! 1. **They do not share a grid.** Each stack chose its own reference frame
//!    from within its own filter, so the masters differ by whatever dither sat
//!    between the sequences, plus any rotation over the hours between them.
//! 2. **They do not share a scale.** Different filters pass different amounts
//!    of light through different bandwidths onto a sensor with a different
//!    quantum efficiency at each wavelength. Combined as they are, one channel
//!    dominates and the result is monochrome with a tint.
//! 3. **Which channel goes where is a choice**, not a fact. Narrowband has no
//!    true colour: sulphur, hydrogen and oxygen are all deep red, teal and
//!    blue-green at wavelengths no palette respects. The mapping is an
//!    aesthetic convention and belongs to the operator.
//!
//! ## On resampling twice
//!
//! Aligning finished masters means resampling an image that was already
//! resampled by the merge. That is a real cost and it is paid here rather than
//! avoided, because avoiding it means reconstructing every filter onto one grid
//! in a single run — which requires holding every frame of every filter at once
//! and registering across filters, where the robustness model and the
//! photometric match would both have to be told to stay inside their own group.
//! Worth doing one day; not the same size of job.
//!
//! The cost is kept small by interpolating with Catmull-Rom rather than
//! bilinear, and by the fact that the offsets between filters in a single
//! session are a few pixels, so most of the image moves by less than one.

use serde::{Deserialize, Serialize};
use sr_core::math;
use sr_core::plane::Plane;

/// One stacked channel, named by its filter.
#[derive(Clone, Debug)]
pub struct Channel {
    pub name: String,
    pub image: Plane<f32>,
}

/// How channels are brought onto a common scale before being combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Fit {
    /// Leave the channels as they are.
    ///
    /// Right when the channels are already comparable — a real RGB set through
    /// one telescope — and wrong for narrowband, where the filters pass
    /// different amounts of light and the brightest simply wins.
    None,
    /// Match the background level only, leaving the scale alone.
    ///
    /// Keeps the relative intensity of the channels, which is the physical
    /// measurement, while removing the difference in sky background that would
    /// otherwise tint the whole frame.
    Offset,
    /// Match background and scale: `a * v + b`, fitted robustly.
    ///
    /// The usual choice for a narrowband palette, and the one that discards
    /// most information: after it, the channels are equal by construction and
    /// what survives is where they differ *spatially*, which is exactly what a
    /// palette is meant to show. It is a presentation decision, not a
    /// measurement, and it is why `Offset` exists beside it.
    Linear,
}

/// What matching one channel to the reference did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelFit {
    pub name: String,
    pub gain: f32,
    pub offset: f32,
    /// Blocks that survived the robust weighting, of those compared.
    pub inliers: usize,
    pub blocks: usize,
}

/// What aligning one channel to the reference did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelAlignment {
    pub name: String,
    pub shift: (f32, f32),
    pub rotation_deg: f32,
    /// Median residual of the correspondences, in output pixels.
    pub residual: f32,
    pub confidence: f32,
}

/// Blocks across the image used for the photometric fit.
const GRID: usize = 24;

/// Median of each block of an image, with `None` where a block is unusable.
fn block_medians(p: &Plane<f32>) -> Vec<Option<f32>> {
    let mut out = Vec::with_capacity(GRID * GRID);
    let mut buf: Vec<f32> = Vec::new();
    for by in 0..GRID {
        let y0 = by * p.height / GRID;
        let y1 = ((by + 1) * p.height / GRID).max(y0 + 1);
        for bx in 0..GRID {
            let x0 = bx * p.width / GRID;
            let x1 = ((bx + 1) * p.width / GRID).max(x0 + 1);
            buf.clear();
            let sx = ((x1 - x0) / 40).max(1);
            let sy = ((y1 - y0) / 40).max(1);
            let mut y = y0;
            while y < y1 {
                let mut x = x0;
                while x < x1 {
                    let v = p.data[y * p.width + x];
                    if v.is_finite() {
                        buf.push(v);
                    }
                    x += sx;
                }
                y += sy;
            }
            out.push(if buf.len() >= 16 {
                Some(math::median(&buf))
            } else {
                None
            });
        }
    }
    out
}

/// Robust line through matched block medians.
///
/// The reweighting matters more here than anywhere else this pattern is used.
/// Two narrowband channels agree about the *sky* and disagree about the
/// *object* — that is the entire reason for imaging both — so the nebula is a
/// large, one-sided departure from the relationship being fitted. An unweighted
/// fit would bend itself around the subject and then subtract it.
fn fit_line(pairs: &[(f32, f32)], mode: Fit) -> (f32, f32, usize) {
    if pairs.len() < 8 || mode == Fit::None {
        return (1.0, 0.0, 0);
    }
    let mut w = vec![1.0f32; pairs.len()];
    let (mut gain, mut offset) = (1.0f32, 0.0f32);
    for pass in 0..3 {
        if mode == Fit::Offset {
            // Gain fixed at one: only the pedestal moves, so the channels keep
            // their relative brightness.
            let d: Vec<f32> = pairs
                .iter()
                .zip(&w)
                .filter(|&(_, &wi)| wi > 0.0)
                .map(|(p, _)| p.1 - p.0)
                .collect();
            if d.is_empty() {
                break;
            }
            gain = 1.0;
            offset = math::median(&d);
        } else {
            let (mut sw, mut sx, mut sy, mut sxx, mut sxy) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (i, &(x, y)) in pairs.iter().enumerate() {
                let (wi, x, y) = (w[i] as f64, x as f64, y as f64);
                sw += wi;
                sx += wi * x;
                sy += wi * y;
                sxx += wi * x * x;
                sxy += wi * x * y;
            }
            let denom = sw * sxx - sx * sx;
            if sw <= 0.0 || denom.abs() < 1e-18 {
                break;
            }
            gain = ((sw * sxy - sx * sy) / denom) as f32;
            offset = ((sy - gain as f64 * sx) / sw) as f32;
        }
        if pass == 2 {
            break;
        }
        let resid: Vec<f32> = pairs
            .iter()
            .map(|&(x, y)| y - (gain * x + offset))
            .collect();
        let sigma = math::mad_sigma(&resid).max(1e-9);
        for (i, r) in resid.iter().enumerate() {
            w[i] = math::tukey_weight(*r, 3.0 * sigma);
        }
    }
    let inliers = w.iter().filter(|&&wi| wi > 0.0).count();
    (gain, offset, inliers)
}

/// Bring every channel onto the reference channel's photometric scale.
pub fn fit_channels(channels: &mut [Channel], reference: usize, mode: Fit) -> Vec<ChannelFit> {
    if channels.is_empty() {
        return Vec::new();
    }
    let reference = reference.min(channels.len() - 1);
    let target = block_medians(&channels[reference].image);
    let mut out = Vec::with_capacity(channels.len());
    #[allow(clippy::needless_range_loop)]
    for i in 0..channels.len() {
        if i == reference || mode == Fit::None {
            out.push(ChannelFit {
                name: channels[i].name.clone(),
                gain: 1.0,
                offset: 0.0,
                inliers: 0,
                blocks: 0,
            });
            continue;
        }
        let src = block_medians(&channels[i].image);
        let pairs: Vec<(f32, f32)> = src
            .iter()
            .zip(&target)
            .filter_map(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => Some((*a, *b)),
                _ => None,
            })
            .collect();
        let (gain, offset, inliers) = fit_line(&pairs, mode);
        // A gain far from one is a fit that failed rather than a filter that
        // differs: even the widest disparity between narrowband filters is well
        // inside this, and beyond it the pairing has gone wrong.
        let (gain, offset) =
            if gain.is_finite() && offset.is_finite() && (0.02..50.0).contains(&gain) {
                (gain, offset)
            } else {
                (1.0, 0.0)
            };
        for v in channels[i].image.data.iter_mut() {
            *v = gain * *v + offset;
        }
        out.push(ChannelFit {
            name: channels[i].name.clone(),
            gain,
            offset,
            inliers,
            blocks: pairs.len(),
        });
    }
    out
}

/// Register every channel onto the reference channel and resample it there.
///
/// Uses the same machinery as frame registration, run on the masters
/// themselves: they are single-channel images of the same star field, which is
/// what that code is for. Level zero of the pyramid is the master's own
/// resolution, so the transform comes back in output pixels with no rescaling.
pub fn align_channels(
    channels: &mut [Channel],
    reference: usize,
    cfg: &sr_core::config::RegistrationConfig,
) -> Vec<ChannelAlignment> {
    use sr_register::global::register_pair;
    use sr_register::pyramid::RegistrationImage;

    if channels.len() < 2 {
        return Vec::new();
    }
    let reference = reference.min(channels.len() - 1);
    let (w, h) = (
        channels[reference].image.width,
        channels[reference].image.height,
    );
    let ref_pyr = RegistrationImage::build(&channels[reference].image, cfg.pyramid_levels);
    let mut cache = sr_register::correlate::CorrelatorCache::new();
    let mut out = Vec::with_capacity(channels.len());

    #[allow(clippy::needless_range_loop)]
    for i in 0..channels.len() {
        if i == reference {
            out.push(ChannelAlignment {
                name: channels[i].name.clone(),
                shift: (0.0, 0.0),
                rotation_deg: 0.0,
                residual: 0.0,
                confidence: 1.0,
            });
            continue;
        }
        if channels[i].image.width != w || channels[i].image.height != h {
            // Different sizes mean different regions of interest or different
            // scales, and guessing which is not this function's job.
            out.push(ChannelAlignment {
                name: channels[i].name.clone(),
                shift: (f32::NAN, f32::NAN),
                rotation_deg: 0.0,
                residual: f32::INFINITY,
                confidence: 0.0,
            });
            continue;
        }
        let pyr = RegistrationImage::build(&channels[i].image, cfg.pyramid_levels);
        let r = register_pair(i, &ref_pyr, &pyr, cfg, &mut cache);

        // Correlation refines and does not search, and separately stacked
        // filters need searching: each master sits on the grid of whatever
        // frame its own stack chose as reference, and those were taken at
        // whatever rotator angle the night had. On the two-season Iris set the
        // filters' references stand 8 to 13 degrees apart, which at nine
        // thousand pixels is a thousand pixels of displacement at the corner
        // -- far outside anything a correlator can follow.
        //
        // The masters are star fields, so when correlation comes back unsure
        // the pattern of their stars can place them at any angle. Same terms
        // as everywhere else: it is tried, and kept only if it explains the
        // pixels better than what correlation found.
        // Judged on the residual rather than on confidence: confidence is
        // relative to a burst and there is no burst here, only two images.
        // A channel that placed lands within a pixel or so; one that did not
        // lands wherever the correlator's noise put it.
        let placed = r.residual_p50 < 2.0 && r.confidence >= 0.1;
        let (t, residual, rotation, confidence) = if placed {
            (r.transform, r.residual_p50, r.rotation_deg, r.confidence)
        } else {
            match stars_place(&channels[reference].image, &channels[i].image) {
                Some(m) => {
                    log::info!(
                        "{}: correlation was unsure, so its stars placed it -- {} pairs, \
                         turned {:.2} deg, residual {:.2} px",
                        channels[i].name,
                        m.pairs,
                        m.rotation_deg,
                        m.residual
                    );
                    (m.transform, m.residual, m.rotation_deg, 1.0)
                }
                None => (r.transform, r.residual_p50, r.rotation_deg, r.confidence),
            }
        };
        // The transform carries this channel onto the reference, so resampling
        // reads through its inverse.
        if let Some(inv) = t.inverse() {
            let src = channels[i].image.clone();
            let (sw, sh) = (src.width as f32 - 1.0, src.height as f32 - 1.0);
            for y in 0..h {
                for x in 0..w {
                    let (sx, sy) = inv.apply(x as f32, y as f32);
                    // Outside the channel is not the same as dark. Sampling
                    // clamps at the edge, so a rotated channel would otherwise
                    // smear its border pixels across everything the rotation
                    // uncovered -- long coloured streaks along two sides of
                    // the frame, which is what a twelve-degree turn produced
                    // the first time this ran.
                    channels[i].image.data[y * w + x] =
                        if sx < 0.0 || sy < 0.0 || sx > sw || sy > sh {
                            NO_DATA
                        } else {
                            src.catmull_rom(sx, sy)
                        };
                }
            }
        }
        out.push(ChannelAlignment {
            name: channels[i].name.clone(),
            shift: t.centre_offset(w as f32, h as f32),
            rotation_deg: rotation,
            residual,
            confidence,
        });
    }
    out
}

/// Assemble three named channels into RGB.
///
/// `map` gives, for red, green and blue, the index of the channel that goes
/// there. One channel may appear more than once, which is what a two-filter
/// palette does.
pub fn combine(channels: &[Channel], map: [usize; 3]) -> [Plane<f32>; 3] {
    let (w, h) = (channels[map[0]].image.width, channels[map[0]].image.height);
    let mut out = [Plane::new(w, h), Plane::new(w, h), Plane::new(w, h)];
    for c in 0..3 {
        out[c] = channels[map[c].min(channels.len() - 1)].image.clone();
    }
    out
}

/// Replace the luminance of an RGB image with a separately measured one.
///
/// The reason LRGB exists: brightness needs signal-to-noise and colour does
/// not. The eye reads detail almost entirely from luminance, so an unfiltered
/// channel — which collects several times the photons of any colour filter —
/// carries the structure, and thin colour data tints it. That is why a set
/// like this is 98 hours of L against 18 of R, G and B together.
///
/// The colour is preserved by scaling all three channels by the same factor,
/// the ratio of the new luminance to the old, rather than by writing the new
/// luminance into a computed channel. Scaling keeps the *ratios* between the
/// primaries, which is what hue is; anything else shifts colour wherever the
/// luminance disagrees with the RGB brightness, which is everywhere, since
/// disagreeing is what it is for.
///
/// `strength` blends between the RGB's own luminance at 0 and the measured one
/// at 1, because a colour set with real signal in it need not be overruled
/// entirely.
pub fn apply_luminance(rgb: &mut [Plane<f32>; 3], luminance: &Plane<f32>, strength: f32) {
    let strength = strength.clamp(0.0, 1.0);
    let n = rgb[0].data.len().min(luminance.data.len());
    for i in 0..n {
        // Where the luminance does not reach -- the corners a rotation
        // uncovered -- the colour keeps its own brightness. The picture
        // degrades to RGB at its edges instead of going black there.
        if luminance.data[i] <= NO_DATA {
            continue;
        }
        // Rec. 709, the weighting the eye actually applies.
        let own = 0.2126 * rgb[0].data[i] + 0.7152 * rgb[1].data[i] + 0.0722 * rgb[2].data[i];
        let want = own + strength * (luminance.data[i] - own);
        // Where the colour data has nothing, there is no hue to preserve and
        // the ratio is meaningless; the luminance stands on its own.
        let k = if own > 1e-5 { want / own } else { 0.0 };
        let k = if k.is_finite() {
            k.clamp(0.0, 64.0)
        } else {
            0.0
        };
        if own > 1e-5 {
            for c in rgb.iter_mut() {
                c.data[i] *= k;
            }
        } else {
            for c in rgb.iter_mut() {
                c.data[i] = want;
            }
        }
    }
}

/// What a channel holds where it has nothing to say: outside its own frame
/// after being turned onto another's grid.
///
/// Zero rather than a sentinel, because everything downstream already treats a
/// channel with no signal as having no signal, and a NaN would spread.
pub const NO_DATA: f32 = 0.0;

/// Place one finished image on another by the pattern of their stars.
///
/// The detector in `sr-quality` works on a mosaic, where it has to know which
/// sites carry which colour. A stacked master is a plain image and needs none
/// of that: threshold above the background, keep local maxima that are not
/// crowded, and take the centroid of each.
fn stars_place(
    reference: &Plane<f32>,
    target: &Plane<f32>,
) -> Option<sr_register::asterism::Match> {
    let a = find_stars(reference, 200);
    let b = find_stars(target, 200);
    sr_register::asterism::match_stars(&a, &b)
}

fn find_stars(p: &Plane<f32>, limit: usize) -> Vec<sr_core::star::Star> {
    // Background and spread from a strided sample: a median over the whole of a
    // sixty-megapixel plane is not worth the sort.
    let mut sample: Vec<f32> = p.data.iter().copied().step_by(97).collect();
    if sample.len() < 64 {
        return Vec::new();
    }
    sample.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let median = sample[sample.len() / 2];
    let mut dev: Vec<f32> = sample.iter().map(|v| (v - median).abs()).collect();
    dev.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let sigma = 1.4826 * dev[dev.len() / 2];
    // Twelve sigma above the background, or a fraction of the way to the
    // brightest thing in the frame, whichever is larger. The second is what
    // catches an image with no noise to speak of, where a robust sigma is zero
    // and every threshold built on it collapses onto the background.
    let bright = sample[sample.len() * 999 / 1000];
    let threshold = median + (12.0 * sigma).max(0.05 * (bright - median).max(0.0));
    // Negated deliberately: a NaN threshold has to take this branch, and
    // `threshold <= median` would answer that question with `false`.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(threshold > median) {
        return Vec::new();
    }

    // One candidate per cell, so the list covers the frame rather than its
    // brightest corner, and the same stars come back from both images.
    const CELL: usize = 64;
    let (gw, gh) = (p.width.div_ceil(CELL), p.height.div_ceil(CELL));
    let mut best: Vec<Option<(f32, usize, usize)>> = vec![None; gw * gh];
    let r = 4usize;
    for y in r..p.height.saturating_sub(r) {
        for x in r..p.width.saturating_sub(r) {
            let v = p.data[y * p.width + x];
            if v < threshold {
                continue;
            }
            let cell = (y / CELL) * gw + x / CELL;
            if best[cell].map(|b| b.0 >= v).unwrap_or(false) {
                continue;
            }
            // A local maximum, or it is the shoulder of one already counted.
            let peak =
                (y - 1..=y + 1).all(|j| (x - 1..=x + 1).all(|i| p.data[j * p.width + i] <= v));
            if peak {
                best[cell] = Some((v, x, y));
            }
        }
    }

    let mut out: Vec<sr_core::star::Star> = best
        .iter()
        .flatten()
        .filter_map(|&(_, x, y)| {
            let (mut cx, mut cy, mut cw) = (0.0f32, 0.0, 0.0);
            for dy in -(r as i64)..=r as i64 {
                for dx in -(r as i64)..=r as i64 {
                    let w = p.data[(y as i64 + dy) as usize * p.width + (x as i64 + dx) as usize]
                        - median;
                    if w > 0.0 {
                        cx += w * dx as f32;
                        cy += w * dy as f32;
                        cw += w;
                    }
                }
            }
            (cw > 0.0).then(|| sr_core::star::Star {
                x: x as f32 + cx / cw,
                y: y as f32 + cy / cw,
                flux: cw,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(limit);
    out
}

/// A named arrangement of filters into colour.
///
/// None of these is a colour anything really is. Sulphur, hydrogen and oxygen
/// emit at 672, 656 and 501 nm — two deep reds and a blue-green — so a faithful
/// rendering would be red, red and teal, and would show almost nothing. Every
/// palette below is a convention for making structure visible, and choosing one
/// is the operator's decision.
pub fn palette(name: &str) -> Option<[&'static str; 3]> {
    match name.to_ascii_lowercase().as_str() {
        // The Hubble palette.
        "sho" | "hubble" => Some(["S", "H", "O"]),
        "hso" => Some(["H", "S", "O"]),
        // Two filters, oxygen doing duty for both green and blue.
        "hoo" => Some(["H", "O", "O"]),
        "ohh" => Some(["O", "H", "H"]),
        "rgb" | "natural" => Some(["R", "G", "B"]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two masters of the same field at very different angles, which is what
    /// separately stacked filters are: each sits on the grid of whatever frame
    /// its own stack chose, and those come from different nights at different
    /// rotator angles. Correlation cannot follow that; the stars can.
    #[test]
    fn channels_are_aligned_even_when_one_is_turned_right_round() {
        let n = 512;
        let field = |dx: f32, dy: f32, deg: f32| -> Plane<f32> {
            let mut p = Plane::filled(n, n, 0.05);
            let mut seed = 0xA57E_u64;
            let mut next = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 33) as f32 / (1u32 << 31) as f32
            };
            let th = deg.to_radians();
            let (s, c) = (th.sin(), th.cos());
            let mid = n as f32 * 0.5;
            for _ in 0..140 {
                let (ux, uy) = (next() * 0.8 + 0.1, next() * 0.8 + 0.1);
                let amp = 0.3 + 0.7 * next();
                // Sky position, then this image's own framing.
                let (px, py) = (ux * n as f32 - mid, uy * n as f32 - mid);
                let (rx, ry) = (c * px - s * py + mid + dx, s * px + c * py + mid + dy);
                for j in -4i64..=4 {
                    for i in -4i64..=4 {
                        let (x, y) = (rx.round() as i64 + i, ry.round() as i64 + j);
                        if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
                            continue;
                        }
                        let (fx, fy) = (x as f32 - rx, y as f32 - ry);
                        let g = (-(fx * fx + fy * fy) / 3.0).exp();
                        p.data[y as usize * n + x as usize] += amp * g;
                    }
                }
            }
            p
        };

        let mut channels = vec![
            Channel {
                name: "L".into(),
                image: field(0.0, 0.0, 0.0),
            },
            Channel {
                name: "R".into(),
                image: field(6.0, -4.0, 12.0),
            },
        ];
        let cfg = sr_core::config::RegistrationConfig::default();
        let a = align_channels(&mut channels, 0, &cfg);
        assert!(
            (a[1].rotation_deg.abs() - 12.0).abs() < 1.0,
            "recovered {:.2} deg, wanted 12",
            a[1].rotation_deg
        );
        assert!(a[1].residual < 2.0, "residual {}", a[1].residual);
    }

    #[test]
    fn luminance_replaces_brightness_and_keeps_hue() {
        let n = 64;
        let mut rgb = [Plane::new(n, n), Plane::new(n, n), Plane::new(n, n)];
        // A flat magenta: twice as much red as green, blue between them.
        for i in 0..n * n {
            rgb[0].data[i] = 0.40;
            rgb[1].data[i] = 0.20;
            rgb[2].data[i] = 0.30;
        }
        let mut lum = Plane::new(n, n);
        for (i, v) in lum.data.iter_mut().enumerate() {
            // Structure the colour channels do not have.
            *v = 0.10 + 0.40 * ((i % n) as f32 / n as f32);
        }
        let before = [rgb[0].data[10], rgb[1].data[10], rgb[2].data[10]];
        apply_luminance(&mut rgb, &lum, 1.0);

        for i in [5usize, 500, 2000] {
            let out = [rgb[0].data[i], rgb[1].data[i], rgb[2].data[i]];
            let y = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
            assert!(
                (y - lum.data[i]).abs() < 1e-4,
                "luminance {y} wanted {}",
                lum.data[i]
            );
            // Hue is the ratio between primaries, and it must not have moved.
            assert!(
                (out[0] / out[1] - before[0] / before[1]).abs() < 1e-3,
                "red/green went from {} to {}",
                before[0] / before[1],
                out[0] / out[1]
            );
            assert!((out[2] / out[1] - before[2] / before[1]).abs() < 1e-3);
        }
    }

    #[test]
    fn strength_zero_changes_nothing() {
        let n = 16;
        let mut rgb = [Plane::new(n, n), Plane::new(n, n), Plane::new(n, n)];
        for i in 0..n * n {
            rgb[0].data[i] = 0.3 + 0.001 * i as f32;
            rgb[1].data[i] = 0.2;
            rgb[2].data[i] = 0.1;
        }
        let copy = rgb.clone();
        let lum = Plane::filled(n, n, 0.9);
        apply_luminance(&mut rgb, &lum, 0.0);
        for c in 0..3 {
            for i in 0..n * n {
                assert!((rgb[c].data[i] - copy[c].data[i]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn colourless_pixels_take_the_luminance_rather_than_a_ratio() {
        // Dividing by a black colour pixel is how an LRGB composite acquires
        // its coloured speckle. There is no hue to preserve there.
        let n = 8;
        let mut rgb = [Plane::new(n, n), Plane::new(n, n), Plane::new(n, n)];
        let lum = Plane::filled(n, n, 0.5);
        apply_luminance(&mut rgb, &lum, 1.0);
        for c in rgb.iter() {
            for v in &c.data {
                assert!((v - 0.5).abs() < 1e-6, "got {v}");
            }
        }
    }

    fn scene(w: usize, h: usize, gain: f32, offset: f32, dx: f32, dy: f32) -> Plane<f32> {
        let mut p = Plane::<f32>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 - dx, y as f32 - dy);
                // A background plus stars plus one broad object, so a fit has
                // both sky and subject to disagree over.
                let mut v = 0.10 + 0.00004 * fx + 0.00002 * fy;
                for k in 0..40 {
                    let sx = 12.0 + ((k * 37) % (w - 24)) as f32;
                    let sy = 12.0 + ((k * 53) % (h - 24)) as f32;
                    let d2 = (fx - sx).powi(2) + (fy - sy).powi(2);
                    v += 0.5 * (-d2 / 4.0).exp();
                }
                if fx > w as f32 * 0.3 && fx < w as f32 * 0.6 && fy > 20.0 && fy < h as f32 * 0.5 {
                    v += 0.12;
                }
                p.data[y * w + x] = gain * v + offset;
            }
        }
        p
    }

    fn channels(shift: (f32, f32), gain: f32, offset: f32) -> Vec<Channel> {
        vec![
            Channel {
                name: "H".into(),
                image: scene(192, 160, 1.0, 0.0, 0.0, 0.0),
            },
            Channel {
                name: "O".into(),
                image: scene(192, 160, gain, offset, shift.0, shift.1),
            },
        ]
    }

    #[test]
    fn a_linear_fit_recovers_a_planted_gain_and_offset() {
        let mut c = channels((0.0, 0.0), 0.4, 0.03);
        let f = fit_channels(&mut c, 0, Fit::Linear);
        // Carrying O onto H undoes the planted 0.4 and 0.03.
        assert!((f[1].gain - 2.5).abs() < 0.1, "gain {}", f[1].gain);
        assert!((f[1].offset + 0.075).abs() < 0.02, "offset {}", f[1].offset);
        // And afterwards the two agree on the background.
        let bg = |p: &Plane<f32>| math::median(&p.data[..2000]);
        assert!(
            (bg(&c[0].image) - bg(&c[1].image)).abs() < 0.01,
            "backgrounds still differ: {} vs {}",
            bg(&c[0].image),
            bg(&c[1].image)
        );
    }

    #[test]
    fn an_offset_fit_keeps_the_relative_brightness() {
        // The distinction between the two modes. A channel that is genuinely
        // half as bright must stay half as bright under Offset, and must not
        // under Linear.
        let mut c = channels((0.0, 0.0), 0.5, 0.02);
        let before = c[1].image.data.iter().cloned().fold(0.0f32, f32::max)
            - math::median(&c[1].image.data[..2000]);
        let f = fit_channels(&mut c, 0, Fit::Offset);
        assert_eq!(f[1].gain, 1.0);
        let after = c[1].image.data.iter().cloned().fold(0.0f32, f32::max)
            - math::median(&c[1].image.data[..2000]);
        assert!(
            (after - before).abs() < 0.02 * before.abs().max(1e-6),
            "an offset fit changed the amplitude from {before} to {after}"
        );
    }

    #[test]
    fn alignment_recovers_a_planted_shift() {
        let cfg = sr_core::config::RegistrationConfig {
            global_patch: 64,
            global_probes: 6,
            ..Default::default()
        };
        let mut c = channels((3.0, -2.0), 1.0, 0.0);
        let a = align_channels(&mut c, 0, &cfg);
        // The second image's content sits at +3, -2, so carrying it onto the
        // first is a shift of -3, +2.
        assert!((a[1].shift.0 + 3.0).abs() < 0.3, "dx {:?}", a[1].shift);
        assert!((a[1].shift.1 - 2.0).abs() < 0.3, "dy {:?}", a[1].shift);

        // Resampled, the two should now agree pixel for pixel away from the
        // border the shift pulled in.
        let (w, h) = (c[0].image.width, c[0].image.height);
        let mut worst = 0.0f32;
        for y in 12..h - 12 {
            for x in 12..w - 12 {
                worst = worst.max((c[0].image.data[y * w + x] - c[1].image.data[y * w + x]).abs());
            }
        }
        assert!(worst < 0.06, "aligned channels still differ by {worst}");
    }

    #[test]
    fn combining_maps_channels_where_the_palette_says() {
        let c = channels((0.0, 0.0), 1.0, 0.0);
        let rgb = combine(&c, [1, 0, 1]);
        assert_eq!(rgb[0].data[500], c[1].image.data[500]);
        assert_eq!(rgb[1].data[500], c[0].image.data[500]);
        assert_eq!(rgb[2].data[500], c[1].image.data[500]);
    }

    #[test]
    fn palettes_are_named_the_way_the_field_names_them() {
        assert_eq!(palette("sho"), Some(["S", "H", "O"]));
        assert_eq!(palette("HUBBLE"), Some(["S", "H", "O"]));
        assert_eq!(palette("hoo"), Some(["H", "O", "O"]));
        assert_eq!(palette("nonsense"), None);
    }

    #[test]
    fn no_fit_leaves_the_channels_alone() {
        let mut c = channels((0.0, 0.0), 0.4, 0.03);
        let before = c[1].image.data.clone();
        let f = fit_channels(&mut c, 0, Fit::None);
        assert_eq!(f[1].gain, 1.0);
        assert_eq!(c[1].image.data, before);
    }
}
