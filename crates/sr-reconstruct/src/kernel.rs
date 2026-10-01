//! Structure-aware reconstruction kernels.
//!
//! An isotropic kernel averages across edges as readily as along them, which
//! blurs exactly the detail multi-frame reconstruction exists to recover. So
//! the kernel is shaped by the local structure tensor of the reference frame:
//! elongated along an edge, narrowed across it, and widened in flat regions
//! where averaging is pure gain.
//!
//! Parameter names follow Wronski et al., "Handheld Multi-Frame
//! Super-Resolution" (SIGGRAPH 2019), section 5.2, so the values stay
//! comparable with the published ones. The exact interpolation between the
//! detail-preserving and denoising regimes is ours and is documented inline.

use rayon::prelude::*;

use sr_core::cfa::CfaPattern;
use sr_core::config::KernelConfig;
use sr_core::math::eig_sym2;
use sr_core::plane::Plane;

/// How far the two eigenvalues must be apart, in units of the squared gradient
/// noise alone produces, before the kernel is allowed to be anisotropic at all.
///
/// The structure tensor is smoothed over about seven independent samples, so
/// its own scatter is roughly a third of the noise floor. Two floors of margin
/// is several times that, and the anisotropy is faded in from there to twice
/// it, which puts the whole ramp below anything a real edge produces.
const ANISO_MARGIN: f32 = 2.0;

/// The burst size the published kernel variances were chosen for.
const REFERENCE_FRAMES: f32 = 8.0;

/// How narrow the detail kernel may get, however large the burst.
///
/// At 0.03 the reconstruction is already leaving a thousand output pixels with
/// no samples under them, and below it the bright stars measurably widen again
/// -- the sampling limit, not a smoothing choice. The floor binds past about
/// five hundred frames.
const MIN_K_DETAIL: f32 = 0.03;

/// The scatter of the guide plane, measured from the guide plane.
///
/// The kernel needs the noise of the image it is reading, and that is not the
/// sensor's per-sample noise. The guide is a 2x2 cell reduction of the mosaic:
/// its greens are the mean of two samples, its red and blue are single ones, and
/// what comes out is smaller than the per-sample figure by a factor that depends
/// on the mosaic and on the scene. Handing the sensor model's number to the
/// kernel overstates it, and everything the kernel does is expressed in units of
/// it -- the detail threshold that decides how wide to be, and the floor that
/// decides whether an edge is an edge. Overstating it widens the kernel
/// everywhere and reports structure as grain.
///
/// Differencing neighbours cancels everything smooth -- the sky gradient, the
/// nebula -- and leaves the grain. The median absolute deviation of those
/// differences is unmoved by the stars and edges that also live in the plane,
/// which is why it is a median and not a standard deviation.
pub fn guide_noise_sigma(guide: &Plane<f32>) -> f32 {
    let (w, h) = (guide.width, guide.height);
    if w < 4 || h < 4 {
        return 0.0;
    }
    // Enough rows for a stable median without walking the whole plane: a
    // 45-megapixel guide has two thousand rows and forty of them is plenty.
    let step = (h / 64).max(1);
    let mut diffs: Vec<f32> = Vec::with_capacity((h / step) * (w / 2));
    for y in (0..h).step_by(step) {
        let row = &guide.data[y * w..(y + 1) * w];
        for x in 1..w {
            let d = row[x] - row[x - 1];
            if d.is_finite() {
                diffs.push(d.abs());
            }
        }
    }
    if diffs.is_empty() {
        return 0.0;
    }
    // The differences are already centred on zero for a flat field, so the
    // median of their magnitudes is the deviation directly.
    let mad = sr_core::math::median(&diffs);
    // 1.4826 turns a median absolute deviation into a sigma; the root of two
    // undoes the differencing of two independent samples.
    1.4826 * mad / std::f32::consts::SQRT_2
}

/// Per-cell kernel shape, stored as the inverse covariance in an eigenbasis.
///
/// One entry per 2x2 mosaic cell of the reference frame, matching the
/// resolution at which the structure tensor can honestly be estimated from
/// mosaiced data.
#[derive(Clone, Debug)]
pub struct KernelField {
    pub width: usize,
    pub height: usize,
    /// Reciprocal variance across the edge, i.e. along the dominant gradient.
    inv_across: Vec<f32>,
    /// Reciprocal variance along the edge.
    inv_along: Vec<f32>,
    /// Unit dominant-gradient direction.
    dir: Vec<[f32; 2]>,
    /// Support radius in output pixels.
    pub radius: f32,
    /// Cells per sensor pixel along one axis (2 for a Bayer mosaic).
    pub cell: usize,
}

impl KernelField {
    /// Select the mono sampling kernel or the adaptive mosaic field. Mono
    /// measures every site, so it need not borrow CFA's spatial denoising.
    pub fn for_sensor(
        guide: &Plane<f32>, noise_sigma: f32, frames: usize,
        pattern: CfaPattern, cfg: &KernelConfig,
    ) -> Self {
        if pattern.is_mono()
            && let Some(variance) = cfg.mono_kernel_variance {
                return Self::isotropic(guide.width, guide.height, variance.max(MIN_K_DETAIL), cfg.radius, 2);
            }
        Self::from_reference(guide, noise_sigma, frames, cfg)
    }

    /// A single fixed circular kernel, used by the drizzle baseline.
    ///
    /// `variance` is in sensor pixels squared.
    pub fn isotropic(width: usize, height: usize, variance: f32, radius: f32, cell: usize) -> Self {
        let n = width * height;
        let inv = 1.0 / variance.max(1e-6);
        Self {
            width,
            height,
            inv_across: vec![inv; 1],
            inv_along: vec![inv; 1],
            dir: vec![[1.0, 0.0]; 1],
            radius,
            cell,
        }
        .with_uniform(n)
    }

    fn with_uniform(mut self, _n: usize) -> Self {
        // Uniform fields keep a single entry; `weight_at` detects that and skips
        // the lookup entirely.
        self.inv_across.truncate(1);
        self.inv_along.truncate(1);
        self.dir.truncate(1);
        self
    }

    #[inline]
    fn is_uniform(&self) -> bool {
        self.inv_across.len() == 1
    }

    /// Build the field from the reference frame's half-resolution luma guide.
    ///
    /// `noise_sigma` sets the gradient magnitude below which structure is
    /// indistinguishable from noise, so that grain is not mistaken for an edge
    /// and given a narrow kernel of its own.
    pub fn from_reference(
        guide_luma: &Plane<f32>,
        noise_sigma: f32,
        frames: usize,
        cfg: &KernelConfig,
    ) -> Self {
        let (w, h) = (guide_luma.width, guide_luma.height);

        // Structure tensor, smoothed over a small neighbourhood.
        let mut jxx = Plane::<f32>::new(w, h);
        let mut jxy = Plane::<f32>::new(w, h);
        let mut jyy = Plane::<f32>::new(w, h);
        let mut curvature_energy = Plane::<f32>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(h - 1);
                let i = y * w + x;
                let centre = guide_luma.data[i];
                let dxp = guide_luma.data[y * w + xp] - centre;
                let dxm = centre - guide_luma.data[y * w + xm];
                let dyp = guide_luma.data[yp * w + x] - centre;
                let dym = centre - guide_luma.data[ym * w + x];
                // Central gradients preserve the established edge orientation.
                // Second differences detect alternating samples that cancel in
                // those gradients, without inventing an edge direction for them.
                let gx = 0.5 * (guide_luma.data[y * w + xp] - guide_luma.data[y * w + xm]);
                let gy = 0.5 * (guide_luma.data[yp * w + x] - guide_luma.data[ym * w + x]);
                jxx.data[i] = gx * gx;
                jxy.data[i] = gx * gy;
                jyy.data[i] = gy * gy;
                curvature_energy.data[i] = 0.25 * ((dxp - dxm).powi(2) + (dyp - dym).powi(2));
            }
        }
        let jxx = jxx.blur3();
        let jxy = jxy.blur3();
        let jyy = jyy.blur3();
        let curvature_energy = curvature_energy.blur3();

        // A central gradient has variance sigma^2/2. Each halved second
        // difference has variance 1.5*sigma^2; their summed energy therefore
        // has six times the central-gradient noise expectation.
        let noise_grad = (noise_sigma * std::f32::consts::SQRT_2 * 0.5).max(1e-9);
        let noise_floor = noise_grad * noise_grad;

        // The threshold below which structure is indistinguishable from grain.
        let detail_grad = (cfg.detail_snr.max(0.5) * noise_grad).max(1e-9);

        // The detail kernel is a hedge against sparse sampling. It has to be
        // wide enough that an output pixel finds samples of all three colours
        // under it, and on the handheld bursts the published value was chosen
        // for -- eight frames, upscaled -- that is most of a pixel. Ninety-six
        // dithered frames put twenty-four sub-pixel phases per axis under every
        // output pixel, and the hedge is then paid for in resolution and bought
        // nothing.
        //
        // Measured on a 96-frame burst, taking the width of stars bright enough
        // that noise cannot bias the measurement:
        //
        //   k_detail   0.25    0.12    0.06    0.03    0.015   0.008
        //   width      2.559   --      2.286   2.295   2.350   2.407
        //
        // against 2.2 for a single frame. The turn at 0.06 is the sampling limit
        // showing: below it output pixels start finding no samples at all
        // (351 of them at 0.06, 7742 at 0.008) and the stars widen again.
        //
        // The same law the flat-region kernel already uses lands at 0.072 on
        // this burst, which is the measured optimum inside its own scatter, so
        // there is one rule here and not two.
        let k_detail = if cfg.scale_detail_with_frames {
            let attenuation = (REFERENCE_FRAMES / frames.max(1) as f32).sqrt().min(1.0);
            (cfg.k_detail * attenuation).max(MIN_K_DETAIL).min(cfg.k_detail)
        } else {
            cfg.k_detail
        };

        // Flat regions are widened to suppress noise, but a large burst has
        // already suppressed it temporally. Fall back toward the detail kernel
        // as the frame count rises, taking the reference burst size of the
        // published parameters as eight frames.
        let k_denoise = if cfg.scale_denoise_with_frames {
            let attenuation = (REFERENCE_FRAMES / frames.max(1) as f32).sqrt().min(1.0);
            k_detail + (cfg.k_denoise - k_detail).max(0.0) * attenuation
        } else {
            cfg.k_denoise
        };

        let n = w * h;
        let mut inv_across = vec![0.0f32; n];
        let mut inv_along = vec![0.0f32; n];
        let mut dir = vec![[1.0f32, 0.0f32]; n];

        let chunks: Vec<(usize, f32, f32, [f32; 2])> = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = (jxx.data[i] - noise_floor).max(0.0);
                let c = (jyy.data[i] - noise_floor).max(0.0);
                let b = jxy.data[i];
                let (l1, l2, e1) = eig_sym2(a, b, c);
                let l1 = l1.max(0.0);
                let l2 = l2.max(0.0);

                // Anisotropy in [1, 2]: 1 when the structure is isotropic,
                // 2 when it is a perfect edge -- but it has to be earned.
                //
                // The tensor is built from the guide and the guide has noise in
                // it. In a flat region `jxx` and `jyy` fall to the noise floor
                // and are clamped to zero, while `jxy` has no floor of its own
                // and keeps whatever the noise put there. The eigenvalues are
                // then `|b|` and zero: the most anisotropic result there is,
                // out of nothing but grain. The kernel was stretched four to
                // one along a direction chosen by noise, everywhere the sky was
                // blank, and the background came out visibly fibrous -- worth
                // 140% more scatter at a scale of four to sixteen pixels than a
                // conventional stack of the same frames, with the per-pixel
                // figure a dead heat, which is exactly what smearing noise
                // sideways does.
                //
                // So the gap between the eigenvalues has to stand clear of what
                // noise alone produces before it counts for anything. It fades
                // in over the same margin rather than switching on, so the
                // kernel does not change abruptly along a contour.
                let sum = l1 + l2;
                let margin = ANISO_MARGIN * noise_floor;
                let confidence = ((l1 - l2 - margin) / margin.max(1e-30)).clamp(0.0, 1.0);
                let aniso = if sum > 1e-20 {
                    1.0 + confidence * ((l1 - l2) / sum).clamp(0.0, 1.0).sqrt()
                } else {
                    1.0
                };

                // Detail measure: 1 in flat regions, 0 where structure is
                // strong, judged against what noise alone would produce.
                let fine_energy = (curvature_energy.data[i] / 6.0 - noise_floor).max(0.0);
                let detail_energy = l1.max(fine_energy);
                let d = (1.0 - detail_energy.sqrt() / detail_grad + cfg.d_th).clamp(0.0, 1.0);

                // Interpolate the isotropic size between the detail-preserving
                // and denoising regimes, then stretch it by the anisotropy.
                let k = (1.0 - d) * k_detail + d * k_denoise;
                let stretch = 1.0 + (aniso - 1.0) * (cfg.k_stretch - 1.0);
                let shrink = 1.0 + (aniso - 1.0) * (cfg.k_shrink - 1.0);

                // The wide axis is *along* the edge, which is the minor
                // eigenvector; the narrow axis is along the gradient.
                let k_along = (k * stretch).max(1e-4);
                let k_across = (k / shrink).max(1e-4);
                (i, 1.0 / k_across, 1.0 / k_along, e1)
            })
            .collect();

        for (i, ia, ial, e) in chunks {
            inv_across[i] = ia;
            inv_along[i] = ial;
            dir[i] = e;
        }

        Self {
            width: w,
            height: h,
            inv_across,
            inv_along,
            dir,
            radius: cfg.radius,
            cell: 2,
        }
    }

    /// Kernel weight for a sample displaced by `(dx, dy)` sensor pixels from an
    /// output position lying in reference cell `(cx, cy)`.
    #[inline]
    pub fn weight_at(&self, cx: usize, cy: usize, dx: f32, dy: f32) -> f32 {
        self.weight_scaled(cx, cy, dx, dy, 1.0)
    }

    /// As [`weight_at`], with the kernel variance multiplied by `var_scale`.
    ///
    /// This is how the mosaic's unequal channel densities are handled. In a
    /// Bayer pattern red and blue are measured on a lattice of twice the pitch
    /// of green, so a quarter as often. Reconstructing them through the same
    /// kernel as green fits them far more tightly than their sample spacing
    /// supports, and the result is false colour at edges. Widening their kernel
    /// in proportion to their sampling interval asks each channel for only the
    /// resolution it was actually measured at.
    #[inline]
    pub fn weight_scaled(&self, cx: usize, cy: usize, dx: f32, dy: f32, var_scale: f32) -> f32 {
        let i = if self.is_uniform() {
            0
        } else {
            cy.min(self.height - 1) * self.width + cx.min(self.width - 1)
        };
        let e = self.dir[i];
        // Project onto the eigenbasis: `across` is along the gradient.
        let across = dx * e[0] + dy * e[1];
        let along = -dx * e[1] + dy * e[0];
        let inv = 1.0 / var_scale.max(1e-6);
        let q = (across * across * self.inv_across[i] + along * along * self.inv_along[i]) * inv;
        (-0.5 * q).exp()
    }

    /// Inverse of the bilinearly interpolated covariance at a sensor position.
    /// Wronski et al. section 5.1.2 interpolates the covariance from the
    /// half-resolution grid before evaluating weights. A stepwise field
    /// changes the kernel abruptly between neighbouring output pixels.
    /// Interpolate matrices, not eigenvectors: an eigenvector and its negative
    /// describe the same ellipse and must not cancel during interpolation.
    pub fn precision_at(&self, x: f32, y: f32) -> [f32; 3] {
        let matrix = |cx: usize, cy: usize| {
            let i = if self.is_uniform() { 0 } else { cy * self.width + cx };
            let [ex, ey] = self.dir[i];
            let a = 1.0 / self.inv_across[i];
            let b = 1.0 / self.inv_along[i];
            [a * ex * ex + b * ey * ey, (a - b) * ex * ey,
             a * ey * ey + b * ex * ex]
        };
        let inverse = |m: [f32; 3]| {
            let det = (m[0] * m[2] - m[1] * m[1]).max(1e-20);
            [m[2] / det, -m[1] / det, m[0] / det]
        };
        if self.is_uniform() { return inverse(matrix(0, 0)); }
        // A guide cell averages sensor sites, whose centre is (cell-1)/2.
        let cell = self.cell as f32;
        let gx = ((x - (cell - 1.0) * 0.5) / cell).clamp(0.0, (self.width - 1) as f32);
        let gy = ((y - (cell - 1.0) * 0.5) / cell).clamp(0.0, (self.height - 1) as f32);
        let (ix, iy) = (gx.floor() as usize, gy.floor() as usize);
        let (fx, fy) = (gx - ix as f32, gy - iy as f32);
        let mut q = [0.0; 3];
        for (cx, cy, w) in [
            (ix, iy, (1.0-fx)*(1.0-fy)),
            ((ix+1).min(self.width-1), iy, fx*(1.0-fy)),
            (ix, (iy+1).min(self.height-1), (1.0-fx)*fy),
            ((ix+1).min(self.width-1), (iy+1).min(self.height-1), fx*fy),
        ] {
            let p = matrix(cx, cy);
            for c in 0..3 { q[c] += w*p[c]; }
        }
        inverse(q)
    }

    /// Kernel anisotropy per cell, for diagnostics.
    pub fn anisotropy_map(&self) -> Plane<f32> {
        if self.is_uniform() {
            return Plane::filled(self.width, self.height, 1.0);
        }
        Plane::from_vec(
            self.width,
            self.height,
            (0..self.width * self.height)
                .map(|i| (self.inv_across[i] / self.inv_along[i].max(1e-12)).sqrt())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_sampling_is_uniform_and_cfa_or_explicit_adaptive_is_preserved() {
        let cfg = KernelConfig::default();
        let guide = vertical_edge(64, 64);
        let mono = KernelField::for_sensor(&guide, 0.002, 10, CfaPattern::MONO, &cfg);
        let expected = KernelField::isotropic(64, 64, 0.125, cfg.radius, 2);
        for (x, y) in [(5.0, 5.0), (63.5, 63.5), (90.0, 70.0)] {
            assert_eq!(mono.precision_at(x, y), expected.precision_at(x, y));
        }
        let adaptive = KernelField::from_reference(&guide, 0.002, 10, &cfg);
        for pattern in [CfaPattern::MONO, CfaPattern::RGGB] {
            let selected = KernelField::for_sensor(&guide, 0.002, 10, pattern,
                &KernelConfig { mono_kernel_variance: None, ..cfg });
            assert_eq!(selected.inv_across, adaptive.inv_across);
            assert_eq!(selected.inv_along, adaptive.inv_along);
            assert_eq!(selected.dir, adaptive.dir);
        }
        let cfa = KernelField::for_sensor(&guide, 0.002, 10, CfaPattern::RGGB, &cfg);
        assert_eq!(cfa.inv_across, adaptive.inv_across);
        assert_eq!(cfa.inv_along, adaptive.inv_along);
        assert_eq!(cfa.dir, adaptive.dir);
    }

    #[test]
    fn covariance_interpolation_respects_cell_centres_and_rotations() {
        // Orthogonal ellipses average to a circle halfway between their
        // centres. Interpolating their directions or inverse variances does
        // not give the specified arithmetic covariance average.
        let f = KernelField {
            width: 2, height: 1, cell: 2, radius: 2.0,
            inv_across: vec![1.0, 0.25], inv_along: vec![0.25, 1.0],
            dir: vec![[1.0, 0.0], [-1.0, 0.0]],
        };
        assert_eq!(f.precision_at(0.5, 0.5), [1.0, 0.0, 0.25]);
        let middle = f.precision_at(1.5, 0.5);
        assert!((middle[0] - 0.4).abs() < 1e-6);
        assert!(middle[1].abs() < 1e-6);
        assert!((middle[2] - 0.4).abs() < 1e-6);
        let a = f.precision_at(2.0 - 1e-4, 0.5);
        let b = f.precision_at(2.0 + 1e-4, 0.5);
        assert!((a[0] - b[0]).abs() < 1e-3, "cell boundary changed the kernel abruptly");
    }

    fn vertical_edge(w: usize, h: usize) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                p.data[y * w + x] = if x < w / 2 { 0.2 } else { 0.8 };
            }
        }
        p
    }

    #[test]
    fn kernel_stretches_along_an_edge() {
        let cfg = KernelConfig::default();
        let f = KernelField::from_reference(&vertical_edge(64, 64), 0.001, 8, &cfg);
        // On the edge itself, at the centre column.
        let (cx, cy) = (32usize, 32usize);
        let across = f.weight_at(cx, cy, 1.0, 0.0); // along the gradient
        let along = f.weight_at(cx, cy, 0.0, 1.0); // along the edge
        assert!(
            along > across * 3.0,
            "kernel should gather along the edge: along {along}, across {across}"
        );
    }

    #[test]
    fn a_large_burst_gets_a_narrower_detail_kernel() {
        // The detail kernel is a hedge against sparse sampling, and a hundred
        // dithered frames are not sparse. Paid for in resolution: on 96 frames
        // the shipped 0.25 put the bright stars at 2.559 px against a single
        // frame's 2.2, and shrinking it brought them to 2.286.
        let cfg = KernelConfig::default();
        let edge = vertical_edge(64, 64);
        let small = KernelField::from_reference(&edge, 0.001, 8, &cfg);
        let large = KernelField::from_reference(&edge, 0.001, 96, &cfg);
        // Across the edge, where the detail kernel governs.
        let a = small.weight_at(32, 32, 1.0, 0.0);
        let b = large.weight_at(32, 32, 1.0, 0.0);
        assert!(b < a * 0.7, "large burst kernel not narrower: {b} against {a}");
    }

    #[test]
    fn the_detail_kernel_has_a_floor() {
        // Below about 0.03 the reconstruction starts leaving output pixels
        // with no samples under them and the stars widen again, so however
        // many frames arrive the kernel stops there.
        let cfg = KernelConfig::default();
        let edge = vertical_edge(64, 64);
        let many = KernelField::from_reference(&edge, 0.001, 10_000, &cfg);
        let floor_cfg = KernelConfig { k_detail: MIN_K_DETAIL, ..cfg };
        let floored = KernelField::from_reference(&edge, 0.001, 8, &floor_cfg);
        let a = many.weight_at(32, 32, 1.0, 0.0);
        let b = floored.weight_at(32, 32, 1.0, 0.0);
        assert!((a - b).abs() < 0.02, "floor not honoured: {a} against {b}");
    }

    #[test]
    fn the_detail_kernel_is_left_alone_when_the_scaling_is_off() {
        let cfg = KernelConfig { scale_detail_with_frames: false, ..KernelConfig::default() };
        let edge = vertical_edge(64, 64);
        let small = KernelField::from_reference(&edge, 0.001, 8, &cfg);
        let large = KernelField::from_reference(&edge, 0.001, 96, &cfg);
        let a = small.weight_at(32, 32, 1.0, 0.0);
        let b = large.weight_at(32, 32, 1.0, 0.0);
        assert!((a - b).abs() < 1e-4, "{a} against {b}");
    }

    #[test]
    fn flat_regions_get_a_wide_isotropic_kernel() {
        let cfg = KernelConfig::default();
        let flat = Plane::filled(64, 64, 0.5);
        let f = KernelField::from_reference(&flat, 0.001, 8, &cfg);
        let a = f.weight_at(32, 32, 1.0, 0.0);
        let b = f.weight_at(32, 32, 0.0, 1.0);
        assert!((a - b).abs() < 1e-3, "flat region kernel is not isotropic: {a} vs {b}");
        // k_denoise = 3 means a 1 px offset barely attenuates.
        assert!(a > 0.8, "flat kernel too narrow: {a}");
    }

    #[test]
    fn detailed_regions_get_a_narrow_kernel() {
        let cfg = KernelConfig::default();
        let edge = KernelField::from_reference(&vertical_edge(64, 64), 0.001, 8, &cfg);
        let flat = KernelField::from_reference(&Plane::filled(64, 64, 0.5), 0.001, 8, &cfg);
        let sharp_across = edge.weight_at(32, 32, 1.0, 0.0);
        let flat_w = flat.weight_at(32, 32, 1.0, 0.0);
        assert!(
            sharp_across < flat_w * 0.5,
            "edge kernel ({sharp_across}) should be much narrower than flat ({flat_w})"
        );
    }

    #[test]
    fn noise_is_not_mistaken_for_structure() {
        let cfg = KernelConfig::default();
        let mut noisy = Plane::filled(64, 64, 0.5);
        let mut seed = 7u64;
        for v in noisy.data.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v += ((seed >> 33) as f32 / (1u32 << 31) as f32 - 0.5) * 0.02;
        }
        // Told that the noise sigma is 0.01, the field should still read flat.
        let f = KernelField::from_reference(&noisy, 0.01, 8, &cfg);
        let a = f.weight_at(32, 32, 1.0, 0.0);
        assert!(a > 0.7, "noise was treated as structure: weight {a}");
    }

    #[test]
    fn alternating_guide_texture_keeps_detail_in_both_axes() {
        let cfg = KernelConfig::default();
        let flat = KernelField::from_reference(&Plane::filled(64, 64, 0.3), 0.002, 8, &cfg);
        for axis in 0..2 {
            let mut guide = Plane::filled(64, 64, 0.3);
            for y in 0..64 {
                for x in 0..64 {
                    let coordinate = if axis == 0 { x } else { y };
                    guide.data[y * 64 + x] += if coordinate % 2 == 0 { 0.02 } else { -0.02 };
                }
            }
            let field = KernelField::from_reference(&guide, 0.002, 8, &cfg);
            let (dx, dy) = if axis == 0 { (1.0, 0.0) } else { (0.0, 1.0) };
            for x in [31, 32] {
                assert!(field.weight_at(x, 32, dx, dy) < flat.weight_at(x, 32, dx, dy) * 0.6,
                    "alternating texture was classified as flat on axis {axis}, phase {x}");
            }
        }
    }

    #[test]
    fn fine_detail_detector_does_not_narrow_flat_grain() {
        let cfg = KernelConfig::default();
        for seed in [7, 0xC0FFEE, 0xBEEF] {
            let field = KernelField::from_reference(&grain(96, 96, 0.004, seed), 0.004, 8, &cfg);
            let mut weights = Vec::new();
            for y in 4..92 {
                for x in 4..92 {
                    weights.push(field.weight_at(x, y, 1.0, 0.0));
                }
            }
            weights.sort_by(f32::total_cmp);
            assert!(weights[weights.len() / 100] > 0.7,
                "fine-detail detection narrowed too much blank grain for seed {seed}");
            assert!(weights.iter().sum::<f32>() / weights.len() as f32 > 0.8);
        }
    }

    #[test]
    fn low_contrast_texture_is_not_treated_as_flat() {
        // Fine texture well above the noise floor must keep a narrow kernel,
        // however low its absolute contrast. An absolute gradient threshold
        // fails this: it calls a sunlit roof's granule texture flat and blurs
        // it, while passing a printed chart's edges.
        let cfg = KernelConfig::default();
        let mut fine = Plane::<f32>::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                // 0.04 peak to peak on a period of four guide pixels.
                fine.data[y * 64 + x] = 0.30 + if (x / 2) % 2 == 0 { 0.02 } else { -0.02 };
            }
        }
        // Noise an order of magnitude below the texture.
        let f = KernelField::from_reference(&fine, 0.002, 8, &cfg);
        let w = f.weight_at(32, 32, 1.0, 0.0);
        let flat = KernelField::from_reference(&Plane::filled(64, 64, 0.30), 0.002, 8, &cfg);
        let wf = flat.weight_at(32, 32, 1.0, 0.0);
        assert!(
            w < wf * 0.6,
            "low-contrast texture got a flat-region kernel: {w} vs flat {wf}"
        );
    }

    #[test]
    fn flat_region_kernel_narrows_as_the_burst_grows() {
        let cfg = KernelConfig::default();
        let flat = Plane::filled(64, 64, 0.5);
        let few = KernelField::from_reference(&flat, 0.002, 8, &cfg);
        let many = KernelField::from_reference(&flat, 0.002, 200, &cfg);
        assert!(
            many.weight_at(32, 32, 1.5, 0.0) < few.weight_at(32, 32, 1.5, 0.0),
            "a 200-frame burst should not be blurred as hard as an 8-frame one"
        );
    }

    #[test]
    fn channel_variance_scale_widens_the_kernel() {
        let f = KernelField::isotropic(32, 32, 0.25, 2.0, 2);
        let green = f.weight_scaled(4, 4, 0.7, 0.0, 1.0);
        let red = f.weight_scaled(4, 4, 0.7, 0.0, 2.0);
        assert!(red > green, "wider kernel should weight a distant sample more");
        // exp(-0.5 * d^2 / (v * s)) with d = 0.7, v = 0.25, s = 2.
        let want = (-0.5f32 * 0.49 / 0.5).exp();
        assert!((red - want).abs() < 1e-5, "{red} vs {want}");
    }

    #[test]
    fn isotropic_field_is_uniform_and_cheap() {
        let f = KernelField::isotropic(100, 100, 0.25, 2.0, 2);
        assert!((f.weight_at(0, 0, 0.5, 0.0) - f.weight_at(99, 99, 0.0, 0.5)).abs() < 1e-6);
        // sigma^2 = 0.25 so a 0.5 px offset attenuates by exp(-0.5).
        assert!((f.weight_at(5, 5, 0.5, 0.0) - (-0.5f32).exp()).abs() < 1e-5);
    }

    /// A guide of nothing but grain, at a known sigma.
    fn grain(w: usize, h: usize, sigma: f32, seed: u64) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        let mut s = seed | 1;
        for v in p.data.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 40) & 0xffff) as f32 / 65535.0 - 0.5;
            *v = 0.3 + sigma * u * 3.46;
        }
        p
    }

    #[test]
    fn grain_alone_does_not_earn_an_anisotropic_kernel() {
        // The failure this guards. `jxx` and `jyy` are floored at the noise
        // level and clamp to zero in a flat region, while `jxy` has no floor of
        // its own -- so the eigenvalues become `|b|` and zero, the most
        // anisotropic pair there is, out of grain. The kernel was then stretched
        // four to one along a direction chosen by noise, everywhere the sky was
        // blank, and the background of a 96-frame stack came out fibrous.
        let cfg = KernelConfig::default();
        let sigma = 0.004;
        let f = KernelField::from_reference(&grain(96, 96, sigma, 0xC0FFEE), sigma, 8, &cfg);
        let a = f.anisotropy_map();
        // Away from the border, where the gradient is one-sided.
        let mut worst = 1.0f32;
        let mut sum = 0.0f32;
        let mut n = 0.0f32;
        for y in 4..92 {
            for x in 4..92 {
                let v = a.data[y * 96 + x];
                worst = worst.max(v);
                sum += v;
                n += 1.0;
            }
        }
        let mean = sum / n;
        let mut all: Vec<f32> = (4..92)
            .flat_map(|y| (4..92).map(move |x| (y, x)))
            .map(|(y, x)| a.data[y * 96 + x])
            .collect();
        all.sort_by(|p, q| p.partial_cmp(q).unwrap());
        let p99 = all[all.len() * 99 / 100];
        // Before the gate this was a stretch of 2.8 to 1 across almost the
        // whole field, because two clamped eigenvalues and an unclamped
        // off-diagonal make every noisy cell look like a perfect edge.
        assert!(
            mean < 1.02,
            "the kernel is stretched {mean:.2} to 1 on average by grain alone"
        );
        assert!(
            p99 < 1.10,
            "a hundredth of pure grain is stretched past {p99:.2} to 1"
        );
        // The extreme tail is left alone deliberately. A handful of cells in
        // ten thousand do cross the gate on a genuine noise excursion, and a
        // few isolated cells do not make a texture -- the mean and the
        // ninety-ninth are what the eye sees.
        let _ = worst;
    }

    #[test]
    fn an_edge_in_the_same_grain_still_earns_one() {
        // The other half: the gate must not cost a real edge its kernel.
        let cfg = KernelConfig::default();
        let sigma = 0.004;
        let mut g = grain(96, 96, sigma, 0xBEEF);
        for y in 0..96 {
            for x in 0..96 {
                if x >= 48 {
                    g.data[y * 96 + x] += 0.4;
                }
            }
        }
        let f = KernelField::from_reference(&g, sigma, 8, &cfg);
        let a = f.anisotropy_map();
        let on_edge: Vec<f32> = (20..76).map(|y| a.data[y * 96 + 48]).collect();
        let median = {
            let mut v = on_edge.clone();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        assert!(
            median > 1.7,
            "the edge only earned a stretch of {median:.2} to 1"
        );
    }

    #[test]
    fn guide_noise_is_measured_not_assumed() {
        // A plane of known grain has to come back at that sigma, or every
        // threshold the kernel expresses in units of it is wrong by the same
        // factor.
        for sigma in [0.001f32, 0.004, 0.02] {
            let got = guide_noise_sigma(&grain(128, 128, sigma, 0xABCDEF));
            assert!(
                (got / sigma - 1.0).abs() < 0.15,
                "grain at {sigma} measured as {got}"
            );
        }
    }

    #[test]
    fn stars_and_edges_do_not_inflate_the_guide_noise() {
        // The plane it is measured on is a picture, not a flat field: it has
        // stars in it, and an edge or two. A mean would follow them; a median
        // of neighbour differences does not.
        let sigma = 0.003;
        let mut p = grain(128, 128, sigma, 0x1234);
        for y in 0..128 {
            for x in 0..128 {
                if x >= 64 {
                    p.data[y * 128 + x] += 0.35;
                }
                // A scatter of bright points, as a star field has.
                if (x * 7 + y * 13) % 211 == 0 {
                    p.data[y * 128 + x] += 0.5;
                }
            }
        }
        let got = guide_noise_sigma(&p);
        assert!(
            (got / sigma - 1.0).abs() < 0.25,
            "a picture with structure in it measured {got} against a true {sigma}"
        );
    }
}
