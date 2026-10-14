//! Display stretch for images whose content occupies a sliver of the range.
//!
//! A deep-sky result is not dark. It is *flat*: on the reconstructions this was
//! written against the median sits near mid-grey and the robust spread is under
//! a percent, so the whole image is one shade with a few white dots on it. A
//! daytime frame from the same pipeline has fifty times the spread and needs
//! nothing done to it.
//!
//! So this is not a brightness correction and not a tone curve. It is the
//! standard astronomical screen transfer: clip the shadows a few deviations
//! below the background, then apply a midtone transfer function chosen so the
//! background lands at a fixed target. Two parameters, both derived from the
//! image's own statistics, and no fitting.
//!
//! ## It is a preview, not a result
//!
//! The stretch is not invertible in any useful sense and destroys the linear
//! relationship the rest of the pipeline works hard to preserve. It is applied
//! to a copy, written to a separate file, and never to the output.
//!
//! ## The gate
//!
//! Applied to an image that did not need it, a background-targeting transfer
//! makes things worse: a well-exposed daytime frame has a median well above the
//! target, so the transfer *darkens* it. Rather than pick a threshold to
//! separate the cases, the stretch is computed and then measured — if it does
//! not widen the robust spread, it is not applied. An image that did not need
//! stretching says so by not being improved.

use sr_core::math;
use sr_core::plane::Plane;

/// Where the background is put, as a fraction of full scale.
///
/// A quarter is the figure astronomical software has settled on: dark enough to
/// read as sky, bright enough that what sits just above it is visible.
const TARGET_BACKGROUND: f32 = 0.25;

/// Where the highlight anchor is put, as a fraction of full scale.
///
/// The anchor is the `HIGHLIGHT_QUANTILE` of the channel, which on a star
/// field is faint stars and the brighter nebulosity. Well below white, so that
/// the bright stars above it have somewhere to go, and well above the
/// background, so that what sits between the two is spread out.
const TARGET_HIGHLIGHT: f32 = 0.65;
const HIGHLIGHT_QUANTILE: f64 = 0.995;

/// Shadow clipping, in robust deviations below the background.
///
/// Discards the bottom tail of the noise so that the stretch does not spend
/// most of its range on values that carry nothing.
const SHADOW_CLIP: f32 = 2.8;

/// The stretch must widen the robust spread by at least this much to be worth
/// applying.
const MIN_GAIN: f32 = 1.5;

/// A midtone transfer function and the shadow clipping that precedes it.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Stretch {
    /// Values at or below this become zero.
    pub shadows: f32,
    /// Midtone balance. Below 0.5 brightens; 0.5 is the identity.
    pub midtone: f32,
}

impl Stretch {
    /// The stretch anchored on the picture's content: the sky `median` goes to
    /// the target background and the `high` quantile of the signal goes to
    /// the highlight target, both parameters solved from those two facts.
    ///
    /// The preview's rule -- shadows a few noise deviations below the sky --
    /// fixes the display size of the *noise*, which means it amplifies every
    /// real variation in inverse proportion to how clean the image is. A stack
    /// two and a half times less noisy than another gets its faint structure
    /// blown up two and a half times harder by that rule, and on a deep stack
    /// the whole frame goes to the vignetting and the sky gradient. Anchoring
    /// on the signal instead gives the same picture for the same sky, however
    /// clean the stack.
    ///
    /// The transfer is monotone in the shadows for a fixed midtone rule, so
    /// the shadows are found by bisection and the midtone follows from them.
    pub fn for_anchors(median: f32, high: f32) -> Stretch {
        let high = high.max(median + 1e-4).min(0.999);
        let at = |shadows: f32| -> (Stretch, f32) {
            let x0 = ((median - shadows) / (1.0 - shadows).max(1e-6)).clamp(1e-6, 1.0 - 1e-6);
            let s = Stretch {
                shadows,
                midtone: midtone_for(x0, TARGET_BACKGROUND),
            };
            (s, s.apply(high))
        };
        let (mut lo, mut hi) = (0.0f32, (median * 0.999).max(0.0));
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            let (_, got) = at(mid);
            // Shadows nearer the sky mean a steeper transfer and a higher
            // landing for the highlight anchor.
            if got > TARGET_HIGHLIGHT {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        at(0.5 * (lo + hi)).0
    }

    /// The stretch that puts a background of `median` with robust spread `mad`
    /// at the target level, shadows clipped below it.
    pub fn for_background(median: f32, mad: f32) -> Stretch {
        let shadows = (median - SHADOW_CLIP * mad).clamp(0.0, 0.999);
        let x0 = ((median - shadows) / (1.0 - shadows).max(1e-6)).clamp(1e-6, 1.0 - 1e-6);
        Stretch {
            shadows,
            midtone: midtone_for(x0, TARGET_BACKGROUND),
        }
    }

    #[inline]
    pub fn apply(&self, v: f32) -> f32 {
        let x = ((v - self.shadows) / (1.0 - self.shadows).max(1e-6)).clamp(0.0, 1.0);
        midtone_transfer(x, self.midtone)
    }

    /// How much this brightens the background, for reporting.
    pub fn describe(&self, median: f32) -> String {
        format!(
            "shadows clipped at {:.4}, midtone {:.4}; background {:.3} becomes {:.3}",
            self.shadows,
            self.midtone,
            median,
            self.apply(median)
        )
    }
}

/// The midtone transfer function used throughout astronomical imaging.
///
/// `m` is where the input midpoint ends up. It is continuous, monotonic on
/// `[0, 1]`, and fixes both endpoints, which is what makes it safe to apply to
/// an image without clipping anything that was not already at an extreme.
#[inline]
pub fn midtone_transfer(x: f32, m: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if m <= 0.0 {
        return 1.0;
    }
    if m >= 1.0 {
        return 0.0;
    }
    let d = (2.0 * m - 1.0) * x - m;
    if d.abs() < 1e-12 {
        return x;
    }
    (((m - 1.0) * x) / d).clamp(0.0, 1.0)
}

/// The midtone that sends `x` to `target`.
///
/// Inverting the transfer for `m` rather than searching for it: the function is
/// a Mobius transformation in `m` as well as in `x`, so this is algebra and not
/// an optimisation.
fn midtone_for(x: f32, target: f32) -> f32 {
    let denom = 2.0 * target * x - target - x;
    if denom.abs() < 1e-9 {
        return 0.5;
    }
    (x * (target - 1.0) / denom).clamp(1e-4, 1.0 - 1e-4)
}

/// Robust centre and spread of the channels that carry data.
/// Background level and robust spread, as the stretch measures them.
///
/// Public so that a caller can say what a stretch did to the background rather
/// than only what its coefficients were.
pub fn background(rgb: &[Plane<f32>], channels: usize) -> Option<(f32, f32)> {
    statistics(rgb, channels)
}

/// The sky median and the highlight quantile of one plane, for the anchored
/// stretch. `None` when there is too little to say.
fn anchors(p: &Plane<f32>) -> Option<(f32, f32)> {
    if p.data.is_empty() {
        return None;
    }
    let step = (p.data.len() / 400_000).max(1);
    let mut v: Vec<f32> = p
        .data
        .iter()
        .step_by(step)
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if v.len() < 64 {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = v[v.len() / 2];
    let i = ((v.len() as f64) * HIGHLIGHT_QUANTILE).floor() as usize;
    let high = v[i.min(v.len() - 1)];
    if high <= median {
        return None;
    }
    Some((median, high))
}

fn statistics(rgb: &[Plane<f32>], channels: usize) -> Option<(f32, f32)> {
    // A stride: a median over a hundred thousand samples is exact enough to
    // place a background, and a full sort of a hundred million is not free.
    let mut v: Vec<f32> = Vec::new();
    for p in rgb.iter().take(channels) {
        if p.data.is_empty() {
            continue;
        }
        let step = (p.data.len() / 120_000).max(1);
        v.extend(p.data.iter().step_by(step).filter(|x| x.is_finite()));
    }
    if v.len() < 64 {
        return None;
    }
    let median = math::median(&v);
    Some((median, math::mad_sigma(&v)))
}

/// Choose a stretch for an image, or decline.
///
/// Linked across the channels: one transfer for all of them, derived from their
/// combined statistics. Stretching each channel to its own background would
/// neutralise the colour, which is a decision about the image rather than about
/// how to look at it, and not one a preview should make silently.
///
/// `None` when the image does not need it — see the module note on the gate.
pub fn choose(rgb: &[Plane<f32>], channels: usize) -> Option<Stretch> {
    let (median, mad) = statistics(rgb, channels)?;
    if !(mad.is_finite() && mad > 0.0) {
        return None;
    }
    let s = Stretch::for_background(median, mad);

    // Would it help? The spread after the stretch against the spread before,
    // both measured the same way. An image that was already legible comes out
    // at or below one and is left alone.
    let after_lo = s.apply(median - mad);
    let after_hi = s.apply(median + mad);
    let gain = (after_hi - after_lo) / (2.0 * mad).max(1e-9);
    if gain < MIN_GAIN {
        return None;
    }
    Some(s)
}

/// Apply a stretch to a copy of the image.
///
/// The result is display-referred and must not be passed through the sRGB
/// transfer curve afterwards: the midtone transfer *is* the display transform,
/// and encoding it again washes it out.
pub fn apply(rgb: &[Plane<f32>], channels: usize, s: &Stretch) -> Vec<Plane<f32>> {
    rgb.iter()
        .take(channels)
        .map(|p| {
            let mut out = p.clone();
            for v in out.data.iter_mut() {
                *v = s.apply(*v);
            }
            out
        })
        .collect()
}

/// How a linear product was turned into the picture in the 16-bit file.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Rendering {
    /// Each channel divided by its own ceiling and put through its own
    /// transfer, so that the sky comes out neutral at the target level and a
    /// saturated star comes out white.
    Stretched { per_channel: Vec<Stretch> },
    /// The image was already legible and got only the sRGB transfer curve.
    Encoded,
}

/// Render a linear product for display.
///
/// `white` is, per channel, the value in `linear` at which the sensor
/// saturated -- the exposure gain times that channel's white-balance
/// multiplier. Dividing by it first is what keeps a clipped star white: after
/// white balance a saturated site reads higher in red and blue than in green,
/// which is the magenta core every raw converter has to deal with, and the
/// cure everywhere is to clip all three at the same place.
///
/// The transfer is then chosen per channel, which puts each channel's sky at
/// the same level and so neutralises the background. That is a decision about
/// the picture and not merely about how to look at it, and the preview
/// deliberately does not make it; the 16-bit file does, because it is the
/// picture. Anyone who wants the sky's own colour has the linear file, where
/// nothing has been decided at all.
///
/// The gate is the same one the preview uses, judged on the channels together:
/// an image that is already legible -- a daytime frame -- is not stretched, and
/// gets the sRGB curve as before.
pub fn render(
    linear: &[Plane<f32>],
    channels: usize,
    white: &[f32; 3],
) -> (Vec<Plane<f32>>, Rendering) {
    let scaled: Vec<Plane<f32>> = linear
        .iter()
        .take(channels)
        .enumerate()
        .map(|(c, p)| {
            let w = if white[c].is_finite() && white[c] > 1e-6 {
                white[c]
            } else {
                1.0
            };
            let mut q = p.clone();
            for v in q.data.iter_mut() {
                *v = (*v / w).clamp(0.0, 1.0);
            }
            q
        })
        .collect();
    // Gated on one channel, not on the channels pooled: a colour camera's sky
    // sits at three different levels, and pooled statistics read that spread
    // as a wide, legible image and decline to stretch exactly the pictures
    // that need it most. Green for a mosaic, the only plane for a monochrome.
    if choose(
        std::slice::from_ref(&scaled[channels.min(scaled.len()) / 2]),
        1,
    )
    .is_none()
    {
        let out = scaled
            .iter()
            .map(|p| {
                let mut q = p.clone();
                for v in q.data.iter_mut() {
                    *v = crate::srgb_encode(*v);
                }
                q
            })
            .collect();
        return (out, Rendering::Encoded);
    }
    // Colour is smoothed a little before the transfer, luminance not at all.
    // The three channels are reconstructed from three separate sets of
    // samples, and where the profile is steep -- every star -- each carries
    // its own sample noise, so a star wears a fringe of colour speckle that a
    // resampled-and-averaged stack does not. Measured around stars, the
    // scatter of the channel ratios was three times a conventional stack's; a
    // Gaussian of 0.6 px on the colour differences alone brings it down and moves
    // the luminance width by nothing, because it does not touch it. The eye
    // resolves colour at a fraction of the resolution it resolves brightness,
    // and every viewer and every processing tool does this somewhere.
    let scaled = if channels == 3 {
        smooth_chroma(&scaled, CHROMA_SMOOTH_SIGMA)
    } else {
        scaled
    };
    let mut per_channel = Vec::with_capacity(channels);
    let mut out = Vec::with_capacity(channels);
    for p in &scaled {
        let s = match anchors(p) {
            Some((median, high)) => Stretch::for_anchors(median, high),
            // A channel with no spread at all has nothing to place; the
            // identity leaves it as it is rather than inventing a level.
            None => Stretch {
                shadows: 0.0,
                midtone: 0.5,
            },
        };
        let mut q = p.clone();
        for v in q.data.iter_mut() {
            *v = s.apply(*v);
        }
        per_channel.push(s);
        out.push(q);
    }
    (out, Rendering::Stretched { per_channel })
}

/// Standard deviation of the Gaussian applied to the colour differences of
/// the rendered picture, in output pixels.
const CHROMA_SMOOTH_SIGMA: f32 = 0.6;

/// Luminance kept, colour differences blurred by a small Gaussian.
fn smooth_chroma(rgb: &[Plane<f32>], sigma: f32) -> Vec<Plane<f32>> {
    let (w, h) = (rgb[0].width, rgb[0].height);
    let n = w * h;
    let luma: Vec<f32> = (0..n)
        .map(|i| 0.2126 * rgb[0].data[i] + 0.7152 * rgb[1].data[i] + 0.0722 * rgb[2].data[i])
        .collect();
    // Five taps are enough for a sigma under one pixel.
    let taps: Vec<f32> = (-2..=2)
        .map(|k| (-(k * k) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f32 = taps.iter().sum();
    let taps: Vec<f32> = taps.iter().map(|t| t / norm).collect();
    rgb.iter()
        .map(|p| {
            let mut chroma: Vec<f32> = (0..n).map(|i| p.data[i] - luma[i]).collect();
            let mut tmp = vec![0.0f32; n];
            // Horizontal pass, edges clamped.
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0f32;
                    for (k, t) in taps.iter().enumerate() {
                        let xx = (x as i64 + k as i64 - 2).clamp(0, w as i64 - 1) as usize;
                        acc += t * chroma[y * w + xx];
                    }
                    tmp[y * w + x] = acc;
                }
            }
            // Vertical pass.
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0f32;
                    for (k, t) in taps.iter().enumerate() {
                        let yy = (y as i64 + k as i64 - 2).clamp(0, h as i64 - 1) as usize;
                        acc += t * tmp[yy * w + x];
                    }
                    chroma[y * w + x] = acc;
                }
            }
            let mut out = p.clone();
            for i in 0..n {
                out.data[i] = (luma[i] + chroma[i]).clamp(0.0, 1.0);
            }
            out
        })
        .collect()
}

/// The white point of each channel as the data holds it: the level of the
/// brightest hundred-thousandth of the pixels.
///
/// On a star field that is the saturated plateau of the bright stars, and it is
/// read from the data rather than derived from the exposure gain and the white
/// balance because the plateau does not land where that arithmetic says it
/// should -- the photometric match moves each frame's ceiling a little, and the
/// merge averages them -- and a saturated star rendered at 0.94 of white has a
/// tint. On a frame with nothing saturated this clips the very brightest
/// pixels, which is what every screen transfer does and is the price of a
/// white that is white.
pub fn ceiling_from_data(planes: &[Plane<f32>], channels: usize) -> [f32; 3] {
    let mut out = [1.0f32; 3];
    for (c, p) in planes.iter().take(channels.min(3)).enumerate() {
        if p.data.is_empty() {
            continue;
        }
        let step = (p.data.len() / 2_000_000).max(1);
        let mut v: Vec<f32> = p
            .data
            .iter()
            .step_by(step)
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if v.len() < 1000 {
            v = p.data.iter().copied().filter(|x| x.is_finite()).collect();
        }
        if v.is_empty() {
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let i = ((v.len() as f64) * (1.0 - CEILING_FRACTION)).floor() as usize;
        let w = v[i.min(v.len() - 1)];
        if w.is_finite() && w > 1e-6 {
            out[c] = w;
        }
    }
    if channels == 1 {
        out = [out[0]; 3];
    }
    out
}

/// The share of a channel's pixels taken to be at or above its white point.
const CEILING_FRACTION: f64 = 1e-5;

/// Stretch if it helps, and say whether it did.
pub fn auto(rgb: &[Plane<f32>], channels: usize) -> (Vec<Plane<f32>>, Option<Stretch>) {
    match choose(rgb, channels) {
        Some(s) => (apply(rgb, channels, &s), Some(s)),
        None => (rgb.iter().take(channels).cloned().collect(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An image with a background at `level` and a robust spread of `spread`,
    /// plus a handful of bright points.
    fn image(level: f32, spread: f32, n: usize) -> Vec<Plane<f32>> {
        let mut p = Plane::<f32>::new(n, n);
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for (i, v) in p.data.iter_mut().enumerate() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 40) & 0xffff) as f32 / 65535.0 - 0.5;
            *v = (level + u * spread * 3.4).clamp(0.0, 1.0);
            if i % 977 == 0 {
                *v = 0.95;
            }
        }
        vec![p.clone(), p.clone(), p]
    }

    #[test]
    fn the_midtone_transfer_fixes_its_endpoints_and_its_midpoint() {
        for m in [0.1f32, 0.25, 0.5, 0.75, 0.9] {
            assert_eq!(midtone_transfer(0.0, m), 0.0);
            assert_eq!(midtone_transfer(1.0, m), 1.0);
        }
        // The identity, which is what a midtone of a half means.
        for x in [0.1f32, 0.3, 0.5, 0.9] {
            assert!((midtone_transfer(x, 0.5) - x).abs() < 1e-6, "x {x}");
        }
    }

    #[test]
    fn the_midtone_transfer_is_monotonic() {
        let m = 0.2;
        let mut last = -1.0;
        for i in 0..=100 {
            let v = midtone_transfer(i as f32 / 100.0, m);
            assert!(v >= last - 1e-6, "not monotonic at {i}: {v} after {last}");
            last = v;
        }
    }

    #[test]
    fn the_midtone_is_solved_rather_than_searched_for() {
        for x in [0.05f32, 0.2, 0.5, 0.8] {
            for target in [0.1f32, 0.25, 0.5] {
                let m = midtone_for(x, target);
                let got = midtone_transfer(x, m);
                assert!(
                    (got - target).abs() < 1e-4,
                    "x {x} target {target}: m {m} gave {got}"
                );
            }
        }
    }

    #[test]
    fn a_flat_astronomical_image_is_stretched_to_the_target() {
        // The case this exists for: median near mid-grey, spread under a
        // percent, which reads on screen as one flat shade.
        let img = image(0.42, 0.0067, 256);
        let s = choose(&img, 3).expect("a flat image needs a stretch");
        let out = apply(&img, 3, &s);
        let (median, mad) = statistics(&out, 3).unwrap();
        assert!(
            (median - TARGET_BACKGROUND).abs() < 0.02,
            "background landed at {median}, wanted {TARGET_BACKGROUND}"
        );
        // And the point of it: the spread is now visible rather than a percent.
        assert!(mad > 0.02, "spread after stretching is still only {mad}");
    }

    #[test]
    fn an_image_that_does_not_need_it_is_left_alone() {
        // A daytime frame: median high, spread half the range. A transfer that
        // targets a background of a quarter would darken it, which is worse
        // than doing nothing, so nothing is what happens.
        let img = image(0.75, 0.35, 256);
        assert!(
            choose(&img, 3).is_none(),
            "stretched an image that was already legible"
        );
        let (out, applied) = auto(&img, 3);
        assert!(applied.is_none());
        assert_eq!(out[0].data, img[0].data);
    }

    #[test]
    fn the_stretch_is_shared_across_channels() {
        // Per-channel stretching would neutralise the background and change the
        // colour of the image, which is a decision and not a way of looking.
        let mut img = image(0.42, 0.0067, 192);
        for v in img[0].data.iter_mut() {
            *v *= 0.7;
        }
        let s = choose(&img, 3).unwrap();
        let out = apply(&img, 3, &s);
        // Red stays darker than green, in the same proportion the transfer
        // implies rather than being equalised away.
        let r = math::median(&out[0].data);
        let g = math::median(&out[1].data);
        assert!(r < g * 0.9, "channels were equalised: {r} against {g}");
    }

    #[test]
    fn one_channel_is_enough() {
        let img = image(0.42, 0.0067, 128);
        let mono = vec![img[0].clone(), Plane::new(0, 0), Plane::new(0, 0)];
        let (out, applied) = auto(&mono, 1);
        assert_eq!(out.len(), 1);
        assert!(applied.is_some());
        assert!((math::median(&out[0].data) - TARGET_BACKGROUND).abs() < 0.02);
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;

    /// A flat sky per channel with a saturated star in the middle, as the
    /// linear product holds it after white balance and exposure gain: the
    /// star's channels sit at different ceilings.
    fn product(sky: [f32; 3], white: [f32; 3]) -> Vec<Plane<f32>> {
        let n = 96usize;
        let mut planes = Vec::new();
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for c in 0..3 {
            let mut p = Plane::<f32>::new(n, n);
            for (i, v) in p.data.iter_mut().enumerate() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 40) & 0xffff) as f32 / 65535.0 - 0.5;
                *v = sky[c] * (1.0 + 0.01 * u);
                let (x, y) = ((i % n) as i32 - 48, (i / n) as i32 - 48);
                if x * x + y * y <= 9 {
                    *v = white[c];
                }
            }
            planes.push(p);
        }
        planes
    }

    #[test]
    fn a_saturated_star_comes_out_white_in_every_channel() {
        // Red and blue ceilings above green's, as white balance leaves them.
        let white = [2.1f32, 1.0, 1.6];
        let planes = product([0.22, 0.39, 0.27], white);
        let found = ceiling_from_data(&planes, 3);
        for c in 0..3 {
            assert!(
                (found[c] - white[c]).abs() < 1e-4,
                "ceiling {c} read as {}",
                found[c]
            );
        }
        let (out, how) = render(&planes, 3, &found);
        assert!(matches!(how, Rendering::Stretched { .. }));
        let n = 96;
        for (c, plane) in out.iter().enumerate() {
            let v = plane.data[48 * n + 48];
            assert!(v > 0.999, "channel {c} of a saturated star rendered at {v}");
        }
    }

    #[test]
    fn the_sky_lands_at_the_same_level_in_every_channel() {
        // A green-dominated sky, as a colour camera records it. Rendered per
        // channel, all three backgrounds end at the target.
        let white = [1.0f32, 1.0, 1.0];
        let planes = product([0.22, 0.39, 0.27], white);
        let (out, _) = render(&planes, 3, &white);
        for (c, plane) in out.iter().enumerate() {
            let m = math::median(&plane.data);
            assert!(
                (m - TARGET_BACKGROUND).abs() < 0.03,
                "channel {c} sky rendered at {m}, wanted {TARGET_BACKGROUND}"
            );
        }
    }

    #[test]
    fn smoothing_the_colour_leaves_the_luminance_alone() {
        // A sharp, coloured star on a grey sky: after the chroma smoothing the
        // luminance at every pixel is what it was, and only the colour spread.
        let n = 32usize;
        let mut planes: Vec<Plane<f32>> = (0..3).map(|_| Plane::filled(n, n, 0.2f32)).collect();
        for (c, gain) in [1.0f32, 0.6, 0.3].iter().enumerate() {
            planes[c].data[16 * n + 16] = 0.2 + 0.5 * gain;
        }
        let out = smooth_chroma(&planes, 0.6);
        for i in 0..n * n {
            let before = 0.2126 * planes[0].data[i]
                + 0.7152 * planes[1].data[i]
                + 0.0722 * planes[2].data[i];
            let after = 0.2126 * out[0].data[i] + 0.7152 * out[1].data[i] + 0.0722 * out[2].data[i];
            assert!(
                (before - after).abs() < 1e-5,
                "luminance moved at {i}: {before} -> {after}"
            );
        }
        // And the colour did spread: the neighbour now carries some of it.
        let centre_before = planes[0].data[16 * n + 16] - planes[2].data[16 * n + 16];
        let centre_after = out[0].data[16 * n + 16] - out[2].data[16 * n + 16];
        let next_after = out[0].data[16 * n + 17] - out[2].data[16 * n + 17];
        assert!(centre_after < centre_before && next_after > 0.0);
    }

    #[test]
    fn a_legible_image_is_only_encoded() {
        // Wide spread, mid-grey: a daytime frame. The gate declines and the
        // sRGB curve is all that happens.
        let n = 64usize;
        let mut p = Plane::<f32>::new(n, n);
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = 0.05 + 0.9 * ((i % n) as f32 / n as f32);
        }
        let planes = vec![p.clone(), p.clone(), p];
        let (out, how) = render(&planes, 3, &[1.0; 3]);
        assert_eq!(how, Rendering::Encoded);
        let mid = out[1].data[32 * n + 32];
        let expect = crate::srgb_encode(planes[1].data[32 * n + 32]);
        assert!((mid - expect).abs() < 1e-5, "got {mid}, wanted {expect}");
    }

    #[test]
    fn a_monochrome_product_renders_its_one_plane() {
        let white = [1.5f32, 1.5, 1.5];
        let planes = product([0.3, 0.3, 0.3], white);
        let (out, how) = render(&planes[..1], 1, &white);
        assert_eq!(out.len(), 1);
        assert!(matches!(how, Rendering::Stretched { per_channel } if per_channel.len() == 1));
        assert!(out[0].data[48 * 96 + 48] > 0.999);
    }
}
