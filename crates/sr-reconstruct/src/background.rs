//! Background modelling: vignetting and sky gradient, fitted from the image.
//!
//! Two things make a deep-sky background uneven, and they are not the same kind
//! of thing:
//!
//! * **Vignetting** is the optics passing less light towards the edge of the
//!   field. It scales whatever arrives, so it is *multiplicative*, roughly
//!   symmetric about the optical axis, and identical in every frame from that
//!   telescope.
//! * **A sky gradient** is light pollution or moonlight added to the frame. It
//!   *adds*, it is smooth and roughly planar across a few degrees of sky, and it
//!   changes through a session as the target moves.
//!
//! Correcting them the same way is wrong in a way that is easy to miss. Dividing
//! out an additive gradient scales bright things more than faint ones; the
//! background comes out flat, and every real brightness in the frame has been
//! altered by a different factor. So the model separates them, and the
//! correction divides by one and subtracts the other.
//!
//! ## What this cannot do
//!
//! It is not a substitute for a flat frame, and the difference is not one of
//! degree. Dust motes and per-pixel sensitivity are small and sharp-edged: a
//! model smooth enough to be safe cannot see them, and a model sharp enough to
//! see them removes nebulosity along with them. Only a real flat fixes those.
//!
//! There is also a limit no amount of modelling escapes. Nothing in one image
//! distinguishes "this corner is dim because of the optics" from "this corner is
//! dim because the object is not there". The model is deliberately restricted to
//! five coefficients per channel — a constant, two radial terms and a plane —
//! because that is roughly the most that can be fitted without competing with
//! the subject. On the burst this was written against, the residual left after
//! fitting it is *visibly the nebula*, which is the sign that the limit has been
//! reached rather than that a richer model is needed.
//!
//! Off by default for that reason.

use serde::{Deserialize, Serialize};
use sr_core::math;
use sr_core::plane::Plane;

/// Blocks across the image the model is fitted to. Coarse on purpose: a block
/// median is robust to stars, and the model has five degrees of freedom, so
/// hundreds of blocks is already a heavily over-determined fit.
const GRID_X: usize = 32;
const GRID_Y: usize = 24;

/// A per-channel background model in normalised output coordinates.
///
/// `x` and `y` run from -1 to +1 across the image, and `r` is the distance from
/// the centre divided by the distance to the corner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BackgroundModel {
    /// Relative vignetting per channel as `1 + k2 r^2 + k4 r^4`, normalised so
    /// that the centre is exactly 1.
    pub vignette: [[f32; 2]; 3],
    /// Additive sky gradient per channel as `gx * x + gy * y`, in normalised
    /// signal units. Its mean over the image is zero by construction, so
    /// subtracting it moves no overall level.
    pub gradient: [[f32; 2]; 3],
    /// Background level per channel at the centre, for reporting.
    pub level: [f32; 3],
    /// How many of the three entries above mean anything. A monochrome
    /// reconstruction fills one; the rest are untouched and must not be read.
    pub channels: usize,
}

impl BackgroundModel {
    /// The channel a single number about this model should come from.
    ///
    /// Green for a mosaic, because that is where the luminance and the noise
    /// both live; the only channel there is for a monochrome sensor.
    #[inline]
    fn primary(&self) -> usize {
        if self.channels <= 1 { 0 } else { 1 }
    }

    /// Multiplicative falloff at normalised radius `r`.
    #[inline]
    pub fn vignette_at(&self, channel: usize, r: f32) -> f32 {
        let k = self.vignette[channel.min(2)];
        let r2 = r * r;
        1.0 + k[0] * r2 + k[1] * r2 * r2
    }

    /// Additive gradient at normalised position.
    #[inline]
    pub fn gradient_at(&self, channel: usize, x: f32, y: f32) -> f32 {
        let g = self.gradient[channel.min(2)];
        g[0] * x + g[1] * y
    }

    /// Vignetting at the corner, as a percentage change from the centre.
    pub fn corner_falloff_percent(&self) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for (c, o) in out.iter_mut().enumerate().take(self.channels.max(1)) {
            *o = 100.0 * (self.vignette_at(c, 1.0) - 1.0);
        }
        out
    }

    /// Peak-to-peak gradient across the image, as a fraction of the centre
    /// level, and the direction it runs in.
    pub fn gradient_percent_and_angle(&self) -> (f32, f32) {
        let g = self.gradient[self.primary()];
        let level = self.level[self.primary()].max(1e-9);
        let span = 2.0 * (g[0] * g[0] + g[1] * g[1]).sqrt();
        (100.0 * span / level, g[1].atan2(g[0]).to_degrees())
    }

    pub fn describe(&self) -> String {
        let v = self.corner_falloff_percent();
        let (gp, ga) = self.gradient_percent_and_angle();
        let vignette = if self.channels == 1 {
            format!("{:+.2}% at the corner", v[0])
        } else {
            format!(
                "{:+.2}% red, {:+.2}% green, {:+.2}% blue at the corner",
                v[0], v[1], v[2]
            )
        };
        format!(
            "Background model: vignetting {vignette}; sky gradient {gp:.2}% of the background \
             level across the frame, rising towards {ga:.0} deg"
        )
    }
}

/// Block medians of one channel, with the block centres in normalised
/// coordinates.
fn blocks(p: &Plane<f32>) -> Vec<(f32, f32, f32)> {
    let mut out = Vec::with_capacity(GRID_X * GRID_Y);
    let mut buf: Vec<f32> = Vec::new();
    for by in 0..GRID_Y {
        let y0 = by * p.height / GRID_Y;
        let y1 = ((by + 1) * p.height / GRID_Y).max(y0 + 1);
        for bx in 0..GRID_X {
            let x0 = bx * p.width / GRID_X;
            let x1 = ((bx + 1) * p.width / GRID_X).max(x0 + 1);
            buf.clear();
            // Strided: a few thousand samples fix a median far below the noise.
            let sx = (((x1 - x0) / 48).max(1)).max(1);
            let sy = (((y1 - y0) / 48).max(1)).max(1);
            let mut y = y0;
            while y < y1 {
                let mut x = x0;
                while x < x1 {
                    buf.push(p.data[y * p.width + x]);
                    x += sx;
                }
                y += sy;
            }
            if buf.len() < 16 {
                continue;
            }
            let cx = (x0 + x1) as f32 * 0.5 / p.width as f32 * 2.0 - 1.0;
            let cy = (y0 + y1) as f32 * 0.5 / p.height as f32 * 2.0 - 1.0;
            out.push((cx, cy, math::median(&buf)));
        }
    }
    out
}

/// Least-squares fit of `[1, r^2, r^4, x, y]`, with two Tukey reweightings.
///
/// The reweighting is what keeps the subject out of the model. A nebula covering
/// a third of the frame is a large, one-sided departure from a smooth
/// background, which is exactly what a redescending weight discards — and
/// exactly what an unweighted fit would bend itself around.
fn fit_channel(samples: &[(f32, f32, f32)]) -> Option<[f64; 5]> {
    if samples.len() < 20 {
        return None;
    }
    let basis = |x: f32, y: f32| -> [f64; 5] {
        let (xf, yf) = (x as f64, y as f64);
        // Radius normalised so that a corner is 1.
        let r2 = (xf * xf + yf * yf) / 2.0;
        [1.0, r2, r2 * r2, xf, yf]
    };

    let mut w = vec![1.0f64; samples.len()];
    let mut best: Option<[f64; 5]> = None;
    for pass in 0..3 {
        let mut ata = [0.0f64; 25];
        let mut atb = [0.0f64; 5];
        for (s, &(x, y, v)) in samples.iter().enumerate() {
            let b = basis(x, y);
            let wi = w[s];
            for i in 0..5 {
                atb[i] += wi * b[i] * v as f64;
                for j in 0..5 {
                    ata[i * 5 + j] += wi * b[i] * b[j];
                }
            }
        }
        let mut a = ata;
        let mut c = atb;
        if !math::cholesky_solve(&mut a, &mut c, 5) {
            return best;
        }
        best = Some(c);
        if pass == 2 {
            break;
        }
        let resid: Vec<f32> = samples
            .iter()
            .map(|&(x, y, v)| {
                let b = basis(x, y);
                let m: f64 = (0..5).map(|i| b[i] * c[i]).sum();
                v - m as f32
            })
            .collect();
        let sigma = math::mad_sigma(&resid).max(1e-9);
        for (s, r) in resid.iter().enumerate() {
            w[s] = math::tukey_weight(*r, 3.0 * sigma) as f64;
        }
    }
    best
}

/// Fit a background model to a reconstructed image.
pub fn fit(rgb: &[Plane<f32>; 3], channels: usize) -> Option<BackgroundModel> {
    let channels = channels.clamp(1, 3);
    let mut m = BackgroundModel {
        channels,
        ..Default::default()
    };
    for (c, plane) in rgb.iter().enumerate().take(channels) {
        let s = blocks(plane);
        let coeff = fit_channel(&s)?;
        let level = coeff[0] as f32;
        if !(level.is_finite() && level.abs() > 1e-9) {
            return None;
        }
        // The radial terms are a fraction of the centre level, which is what
        // makes them a multiplicative falloff rather than an amount of light.
        m.vignette[c] = [(coeff[1] / coeff[0]) as f32, (coeff[2] / coeff[0]) as f32];

        // The gradient is fitted through the vignetting, because that is how it
        // reaches the sensor: light pollution comes in through the same optics.
        // So the fitted plane is the sky gradient already attenuated, and the
        // correction subtracts *after* dividing the falloff out. Rescaling by
        // the mean falloff undoes that, and the mean has a closed form over a
        // uniform grid on [-1, 1]: mean(r^2) is 1/3 and mean(r^4) is 7/45.
        //
        // Left out, this is an 8% error in the gradient for a 20% vignette,
        // which is small and entirely avoidable.
        let mean_v = 1.0 + m.vignette[c][0] / 3.0 + m.vignette[c][1] * 7.0 / 45.0;
        let inv = 1.0 / mean_v.clamp(0.2, 5.0);
        m.gradient[c] = [coeff[3] as f32 * inv, coeff[4] as f32 * inv];
        m.level[c] = level;
        if !m.vignette[c]
            .iter()
            .chain(m.gradient[c].iter())
            .all(|v| v.is_finite())
        {
            return None;
        }
    }
    Some(m)
}

/// How far a model departs from doing nothing, as a fraction of the background.
///
/// Used to decline a correction that would not change anything, so that a run on
/// an already-flat image does not claim to have flattened it.
pub fn magnitude(m: &BackgroundModel) -> f32 {
    let v = m.corner_falloff_percent()[m.primary()].abs();
    let (g, _) = m.gradient_percent_and_angle();
    (v + g) / 100.0
}

/// Divide out the vignetting and subtract the gradient.
///
/// The overall level is preserved: only the *variation* of the gradient is
/// removed, because pushing the background to zero would put half of it below
/// the black point, where the rest of the pipeline treats a sample as unusable.
pub fn apply(rgb: &mut [Plane<f32>; 3], m: &BackgroundModel) {
    for (c, p) in rgb.iter_mut().enumerate().take(m.channels.max(1)) {
        let (w, h) = (p.width, p.height);
        for y in 0..h {
            let ny = (y as f32 + 0.5) / h as f32 * 2.0 - 1.0;
            for x in 0..w {
                let nx = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
                let r = ((nx * nx + ny * ny) * 0.5).sqrt();
                let v = m.vignette_at(c, r).max(0.2);
                let i = y * w + x;
                p.data[i] = p.data[i] / v - m.gradient_at(c, nx, ny);
            }
        }
    }
}

/// Render the model itself, for looking at.
///
/// Returned *relative to the centre of each channel*, so a value of 1.0 is "no
/// correction here" and 0.97 is "three percent was taken off". Absolute units
/// would be useless as a picture: the three channels sit at different
/// background levels, those differences are ten times larger than the shape,
/// and normalising such an image for display shows the colour of the sky and
/// nothing else.
///
/// Worth writing out, because the failure mode of any background model is that
/// it started fitting the subject. That is obvious in a picture of the model
/// and invisible in its coefficients.
pub fn render(m: &BackgroundModel, width: usize, height: usize) -> [Plane<f32>; 3] {
    let sized = |c: usize| {
        if c < m.channels.max(1) {
            Plane::<f32>::new(width, height)
        } else {
            Plane::<f32>::new(0, 0)
        }
    };
    let mut out = [sized(0), sized(1), sized(2)];
    for (c, p) in out.iter_mut().enumerate().take(m.channels.max(1)) {
        let inv_level = 1.0 / m.level[c].abs().max(1e-9);
        for y in 0..height {
            let ny = (y as f32 + 0.5) / height as f32 * 2.0 - 1.0;
            for x in 0..width {
                let nx = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
                let r = ((nx * nx + ny * ny) * 0.5).sqrt();
                p.data[y * width + x] = m.vignette_at(c, r) + m.gradient_at(c, nx, ny) * inv_level;
            }
        }
    }
    out
}

/// Stretch a rendered model into something viewable.
///
/// The whole range of the field is a few percent, so it is scaled to fill the
/// output. The scale is shared across the three channels, so their differences
/// stay legible rather than each being normalised into agreement.
pub fn render_for_display(m: &BackgroundModel, width: usize, height: usize) -> [Plane<f32>; 3] {
    let mut f = render(m, width, height);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for p in f.iter().take(m.channels.max(1)) {
        for &v in &p.data {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let span = (hi - lo).max(1e-6);
    for p in f.iter_mut().take(m.channels.max(1)) {
        for v in p.data.iter_mut() {
            *v = (*v - lo) / span;
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planted(
        w: usize,
        h: usize,
        level: f32,
        k2: f32,
        gx: f32,
        gy: f32,
        object: bool,
    ) -> [Plane<f32>; 3] {
        let mut out = [Plane::new(w, h), Plane::new(w, h), Plane::new(w, h)];
        for (c, p) in out.iter_mut().enumerate() {
            for y in 0..h {
                let ny = (y as f32 + 0.5) / h as f32 * 2.0 - 1.0;
                for x in 0..w {
                    let nx = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
                    let r2 = (nx * nx + ny * ny) * 0.5;
                    // Everything the telescope sees is vignetted, light
                    // pollution included: it arrives through the same optics.
                    // Planting the gradient outside the falloff instead would
                    // make the two corrections fight each other and would be a
                    // fixture that does not describe a telescope.
                    let mut v = level + gx * nx + gy * ny;
                    // A bright object over a third of the frame, as a nebula is.
                    if object && nx > -0.2 && nx < 0.5 && ny > -0.3 && ny < 0.2 {
                        v += 0.4 * level * (1.0 + c as f32 * 0.2);
                    }
                    p.data[y * w + x] = v * (1.0 + k2 * r2);
                }
            }
        }
        out
    }

    #[test]
    fn recovers_a_planted_vignette_and_gradient() {
        let img = planted(320, 240, 0.08, -0.20, 0.004, -0.010, false);
        let m = fit(&img, 3).expect("a smooth background is fittable");
        assert!(
            (m.corner_falloff_percent()[1] + 20.0).abs() < 1.0,
            "corner falloff {:?}",
            m.corner_falloff_percent()
        );
        assert!(
            (m.gradient[1][0] - 0.004).abs() < 5e-4,
            "gx {:?}",
            m.gradient[1]
        );
        assert!(
            (m.gradient[1][1] + 0.010).abs() < 5e-4,
            "gy {:?}",
            m.gradient[1]
        );
        assert!((m.level[1] - 0.08).abs() < 0.002, "level {:?}", m.level);
    }

    #[test]
    fn applying_it_flattens_the_background() {
        let img = planted(320, 240, 0.08, -0.20, 0.004, -0.010, false);
        let m = fit(&img, 3).unwrap();
        let mut corrected = img.clone();
        apply(&mut corrected, &m);
        let before = &img[1].data;
        let after = &corrected[1].data;
        let spread = |v: &[f32]| {
            let (lo, hi) = v
                .iter()
                .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
            hi - lo
        };
        assert!(
            spread(after) < 0.03 * spread(before),
            "spread {:.5} then {:.5}",
            spread(before),
            spread(after)
        );
        // The level is kept: a correction that pushed the background to zero
        // would put half of it below the black point.
        let mean_before: f32 = before.iter().sum::<f32>() / before.len() as f32;
        let mean_after: f32 = after.iter().sum::<f32>() / after.len() as f32;
        assert!((mean_after - mean_before).abs() < 0.1 * mean_before);
    }

    #[test]
    fn a_bright_object_does_not_drag_the_model() {
        // The test the whole robust fit exists for. The same background twice,
        // once with a nebula over a third of the frame: the model must come out
        // the same, or it is subtracting the subject.
        let plain = fit(&planted(320, 240, 0.08, -0.20, 0.004, -0.010, false), 3).unwrap();
        let with = fit(&planted(320, 240, 0.08, -0.20, 0.004, -0.010, true), 3).unwrap();
        let dv = (plain.corner_falloff_percent()[1] - with.corner_falloff_percent()[1]).abs();
        assert!(dv < 3.0, "vignetting moved by {dv:.2} percentage points");
        for k in 0..2 {
            assert!(
                (plain.gradient[1][k] - with.gradient[1][k]).abs() < 2e-3,
                "gradient {k} moved from {:?} to {:?}",
                plain.gradient[1],
                with.gradient[1]
            );
        }
    }

    #[test]
    fn an_additive_gradient_is_not_divided_out() {
        // The distinction the module exists to make. A gradient that is purely
        // additive must be subtracted, so two objects of equal brightness in
        // different parts of the frame stay equal afterwards. Dividing instead
        // would scale them differently.
        let level = 0.08f32;
        let mut img = planted(320, 240, level, 0.0, 0.0, -0.02, false);
        // Two identical sources, one in the bright half and one in the dark.
        for (x, y) in [(80usize, 40usize), (240, 200)] {
            for p in img.iter_mut() {
                for dy in 0..6 {
                    for dx in 0..6 {
                        p.data[(y + dy) * 320 + x + dx] += 0.05;
                    }
                }
            }
        }
        let m = fit(&img, 3).unwrap();
        let mut corrected = img.clone();
        apply(&mut corrected, &m);
        let bg = |p: &Plane<f32>, x: usize, y: usize| p.data[(y + 10) * 320 + x];
        let a = corrected[1].data[42 * 320 + 82] - bg(&corrected[1], 82, 42);
        let b = corrected[1].data[202 * 320 + 242] - bg(&corrected[1], 242, 202);
        assert!(
            (a - b).abs() < 0.05 * a.abs().max(b.abs()),
            "equal sources came out at {a:.5} and {b:.5}"
        );
    }

    #[test]
    fn a_monochrome_image_is_fitted_on_the_channel_it_has() {
        // The two absent channels are zero-sized planes, and reading them is
        // what this used to do: --flatten-background panicked on every
        // monochrome reconstruction, after the result had been written.
        let full = planted(320, 240, 0.08, -0.20, 0.004, -0.010, false);
        let mono = [full[0].clone(), Plane::new(0, 0), Plane::new(0, 0)];
        let m = fit(&mono, 1).expect("one channel is enough to fit a background");
        assert_eq!(m.channels, 1);
        assert!((m.corner_falloff_percent()[0] + 20.0).abs() < 1.0);
        // The single number reported comes from the channel that exists.
        assert!(magnitude(&m) > 0.05, "magnitude {}", magnitude(&m));
        assert!(!m.describe().contains("green"), "{}", m.describe());

        let mut corrected = mono.clone();
        apply(&mut corrected, &m);
        assert!(
            corrected[1].data.is_empty(),
            "an absent channel was written to"
        );
        let spread = |v: &[f32]| {
            let (lo, hi) = v
                .iter()
                .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
            hi - lo
        };
        assert!(spread(&corrected[0].data) < 0.05 * spread(&mono[0].data));

        let field = render_for_display(&m, 64, 48);
        assert_eq!(field[0].data.len(), 64 * 48);
        assert!(field[1].data.is_empty());
    }

    #[test]
    fn a_flat_image_produces_a_model_worth_declining() {
        let img = planted(320, 240, 0.08, 0.0, 0.0, 0.0, false);
        let m = fit(&img, 3).unwrap();
        assert!(
            magnitude(&m) < 0.005,
            "magnitude {} on a flat image",
            magnitude(&m)
        );
    }
}
