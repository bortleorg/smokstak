//! Stages 11A/11B: the merge.
//!
//! Every backend shares one engine: original sensor samples are pushed forward
//! through their own frame's warp onto the output grid and accumulated with a
//! kernel. No frame is ever resampled into an intermediate raster, and for the
//! CFA backends no frame is demosaiced at all.
//!
//! The engine is tiled over the *output*. For each tile and each frame the
//! source rectangle is found by inverting the frame's transform, so the working
//! set stays in cache and peak memory is set by the tile size rather than by
//! the frame count.
//!
//! Backends differ only in three things, which is deliberate â€” it makes the
//! comparison between them a controlled experiment rather than a comparison of
//! two different programs:
//!
//! | backend | samples | kernel | robustness |
//! |---|---|---|---|
//! | `RgbMeanBaseline` | bilinear-demosaiced RGB | isotropic | off |
//! | `CfaDrizzle` | raw mosaic | isotropic | on |
//! | `HandheldBurstSr` | raw mosaic | structure-aware anisotropic | on |

use rayon::prelude::*;

use sr_core::cfa::CfaPattern;
use sr_core::config::{Backend, ReconstructionConfig};
use sr_core::frame::{NoiseModel, RawFrame};
use sr_core::geometry::{RadialChroma, WarpField};
use sr_core::plane::Plane;
use sr_core::product::{ProductStats, ReconstructionProduct};
use sr_core::{Result, SrError};
use sr_quality::photometry::PhotometricMatch;

use crate::kernel::KernelField;
use crate::curvature::Curvature;
use crate::lucky::LuckySelection;
use crate::robustness::RobustnessMaps;

/// Output pixels per cell of the effective-frame-count diagnostic.
pub const EFFECTIVE_FRAMES_CELL: usize = 8;

/// Upper bound on the chroma kernel widening, used to size tile halos and
/// source rectangles so they never clip a legitimate contribution.
const MAX_CHANNEL_SCALE: f32 = 4.0;

/// Everything the merge needs, gathered by the caller.
pub struct MergeInputs<'a> {
    pub frames: &'a [RawFrame],
    /// Sensor-coordinate warps, one per frame.
    pub warps: &'a [WarpField],
    pub reference: usize,
    /// Per-frame, per-channel map onto the reference's photometric scale.
    /// Per-frame, per-channel map onto the reference's photometric scale.
    pub photometry: &'a [PhotometricMatch],
    pub noise: NoiseModel,
    pub robustness: &'a RobustnessMaps,
    pub kernels: &'a KernelField,
    /// Per-frame quality weight, already normalised to about 1.
    pub frame_weight: &'a [f32],
    pub lucky: Option<&'a LuckySelection>,
    /// Per-channel lateral chromatic aberration correction, in each frame's own
    /// sensor coordinates. Identity when there is nothing to correct.
    ///
    /// Applied to a sample's position before the frame's own warp, because the
    /// aberration is a property of the lens and is therefore fixed relative to
    /// the sensor, not to the scene.
    pub chroma: RadialChroma,
}

struct Geometry {
    scale: f32,
    /// Output-grid coordinate of local pixel (0, 0).
    origin: (f32, f32),
    radius: f32,
    /// Kernel variance multiplier applied to red and blue.
    chroma_variance: f32,
    /// Largest displacement the chromatic correction applies, in sensor pixels.
    chroma_pad: f32,
}

/// One independently reconstructed mono tile. Unsupported pixels are NaN;
/// weights and contribution counts are the original deposit diagnostics.
pub struct MonoTile {
    pub values: Vec<f32>,
    pub weight: Vec<f32>,
    pub count: Vec<f32>,
    pub stats: ProductStats,
}

/// Reconstruct native mono samples on an explicit output grid, without a full
/// output allocation or hole filling. `origin` uses the existing ROI boundary
/// convention: (0, 0) reproduces an uncropped reconstruction, and (x, y)
/// reproduces a sensor ROI starting there. `tile` is in output-grid pixels.
/// Sky estimates must be computed once per frame, not separately per tile.
/// Unlike the legacy burst entry point, each frame's own noise model determines
/// its inverse-variance weight and profile-fit uncertainty.
/// This first entry point excludes lucky selection, RGB and chromatic correction.
pub fn reconstruct_mono_tile(
    inputs: &MergeInputs,
    cfg: &ReconstructionConfig,
    origin: (f32, f32),
    tile: sr_core::geometry::Rect,
    frame_sky: &[[f32; 3]],
) -> Result<MonoTile> {
    reconstruct_mono_tile_feathered(inputs, cfg, origin, tile, frame_sky, 0.0)
}

/// Opt-in native detector edge taper for overlapping mono frames. A cubic
/// smoothstep scales statistical contribution weights within `feather_fraction`
/// of the shorter full detector dimension. This changes noise/exposure weighting;
/// it does not calibrate backgrounds or gradients. Cropped windows use their
/// full-detector dimensions and offsets, so window edges never become seams.
/// Zero preserves the ordinary mono tile path exactly.
pub fn reconstruct_mono_tile_feathered(
    inputs: &MergeInputs,
    cfg: &ReconstructionConfig,
    origin: (f32, f32),
    tile: sr_core::geometry::Rect,
    frame_sky: &[[f32; 3]],
    feather_fraction: f32,
) -> Result<MonoTile> {
    let invalid = |message: &str| SrError::Reconstruction(message.into());
    if !feather_fraction.is_finite() || !(0.0..=0.25).contains(&feather_fraction) {
        return Err(invalid("mono tile feather fraction must be finite in 0..=0.25"));
    }
    let n = inputs.frames.len();
    if n == 0 || inputs.reference >= n || inputs.warps.len() != n
        || inputs.photometry.len() != n || inputs.frame_weight.len() != n
        || frame_sky.len() != n
    {
        return Err(invalid("mono tile requires a valid reference and matching per-frame arrays"));
    }
    // Coordinates beyond this lose individual pixel centres in the f32 warp
    // engine. Reject them rather than silently collapse adjacent output sites.
    const COORD_LIMIT: usize = 1 << 23;
    if !(1.0..=8.0).contains(&cfg.scale)
        || !origin.0.is_finite() || !origin.1.is_finite()
        || (origin.0 * cfg.scale).abs() > COORD_LIMIT as f32
        || (origin.1 * cfg.scale).abs() > COORD_LIMIT as f32
        || !(1..=4096).contains(&tile.width) || !(1..=4096).contains(&tile.height)
        || tile.x.checked_add(tile.width).is_none_or(|v| v > COORD_LIMIT)
        || tile.y.checked_add(tile.height).is_none_or(|v| v > COORD_LIMIT)
        || !cfg.kernel.radius.is_finite() || !(0.0..=4096.0).contains(&cfg.kernel.radius)
    {
        return Err(invalid("invalid mono tile scale, origin, dimensions or kernel radius"));
    }
    if cfg.backend == Backend::RgbMeanBaseline || inputs.lucky.is_some()
        || !inputs.chroma.is_identity() || cfg.roi.is_some()
    {
        return Err(invalid("mono tile requires a raw backend, explicit origin, no ROI, lucky selection or chromatic correction"));
    }
    if !inputs.noise.alpha.is_finite() || !inputs.noise.beta.is_finite()
        || inputs.noise.alpha < 0.0 || inputs.noise.beta < 0.0
        || !inputs.noise.variance(0.18).is_finite()
        || inputs.noise.variance(0.18) <= 0.0
        || inputs.kernels.width == 0 || inputs.kernels.height == 0 || inputs.kernels.cell == 0
        || inputs.frame_weight.iter().any(|v| !v.is_finite() || *v < 0.0)
        || frame_sky.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(invalid("invalid mono tile noise, kernel, frame weights or sky"));
    }
    for (frame, sky) in inputs.frames.iter().zip(frame_sky) {
        if !frame.is_mono() || frame.width == 0 || frame.height == 0
            || frame.width > COORD_LIMIT || frame.height > COORD_LIMIT
            || frame.width.checked_mul(frame.height) != Some(frame.samples.len())
            || (frame.samples.width, frame.samples.height) != (frame.width, frame.height)
            || (frame.defects.width, frame.defects.height) != (frame.width, frame.height)
        {
            return Err(invalid("mono tile requires well-formed monochrome frame planes"));
        }
        if feather_fraction > 0.0 {
            for (full, offset, size) in [
                (frame.metadata.full_width, frame.metadata.crop.0, frame.width),
                (frame.metadata.full_height, frame.metadata.crop.1, frame.height),
            ] {
                if full > 0 && offset.checked_add(size).is_none_or(|end| end > full) {
                    return Err(invalid("mono feather crop lies outside the full detector"));
                }
            }
        }
        let noise = frame.noise;
        if !noise.alpha.is_finite() || !noise.beta.is_finite()
            || noise.alpha < 0.0 || noise.beta < 0.0
            || noise.alpha * 0.18 + noise.beta <= 0.0
            || !noise.variance(0.18).is_finite()
            || !noise.variance(sky[0]).is_finite()
        {
            return Err(invalid("mono tile requires finite nonnegative per-frame noise with positive variance"));
        }
    }
    if !inputs.robustness.maps.is_empty() && (inputs.robustness.maps.len() != n
        || inputs.robustness.maps.iter().any(|m| !m.data.is_empty()
            && (m.width == 0 || m.height == 0 || m.width.checked_mul(m.height) != Some(m.data.len()))))
    {
        return Err(invalid("invalid mono tile robustness maps"));
    }
    for photo in inputs.photometry {
        if photo.gain.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || photo.offset.iter().any(|v| !v.is_finite())
            || photo.field.iter().flatten().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid mono tile photometric map"));
        }
    }
    for warp in inputs.warps {
        if warp.global.m.iter().any(|v| !v.is_finite()) || warp.global.inverse().is_none() {
            return Err(invalid("invalid mono tile global warp"));
        }
        if let Some(local) = &warp.local
            && (local.grid_w == 0 || local.grid_h == 0
                || local.grid_w.checked_mul(local.grid_h) != Some(local.u.len())
                || local.conf.len() != local.u.len() || !local.spacing.is_finite() || local.spacing <= 0.0
                || !local.origin.0.is_finite() || !local.origin.1.is_finite()
                || local.u.iter().flatten().any(|v| !v.is_finite())
                || local.conf.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)))
            {
                return Err(invalid("invalid mono tile local warp"));
            }
    }
    let geom = Geometry { scale: cfg.scale,
        origin: (origin.0 * cfg.scale, origin.1 * cfg.scale),
        radius: cfg.kernel.radius.max(0.5), chroma_variance: 1.0, chroma_pad: 0.0 };
    let reference=&inputs.frames[inputs.reference];
    let pad=geom.radius*cfg.scale+2.;
    let lo=geom.to_ref(tile.x as f32-pad,tile.y as f32-pad);
    let hi=geom.to_ref((tile.x+tile.width) as f32+pad,(tile.y+tile.height) as f32+pad);
    let u=[2.*lo.0/reference.width as f32-1.,2.*hi.0/reference.width as f32-1.];
    let v=[2.*lo.1/reference.height as f32-1.,2.*hi.1/reference.height as f32-1.];
    for ((photo,frame),sky) in inputs.photometry.iter().zip(inputs.frames).zip(frame_sky) {
        if photo.log_gain.is_some() {
            let (minimum,maximum)=photo.gain_bounds(0,u,v).ok_or_else(||invalid("invalid spatial gain across mono tile and halo"))?;
            for raw in [0.,1.,sky[0]] {for gain in [minimum,maximum] {
                let variance=frame.noise.variance(raw)*gain*gain;
                if !variance.is_finite() || variance<=0. {return Err(invalid("spatial gain overflows or collapses mono tile noise"));}
            }}
        }
    }
    // Guard the inverse corners before the existing source-bounds integer casts.
    for warp in inputs.warps {
        let inv = warp.global.inverse().unwrap();
        for x in [tile.x as f32 - geom.radius, (tile.x + tile.width) as f32 + geom.radius] {
            for y in [tile.y as f32 - geom.radius, (tile.y + tile.height) as f32 + geom.radius] {
                let (rx, ry) = geom.to_ref(x, y);
                let (sx, sy) = inv.apply(rx, ry);
                if !sx.is_finite() || !sy.is_finite()
                    || sx.abs() + warp.max_local() > COORD_LIMIT as f32
                    || sy.abs() + warp.max_local() > COORD_LIMIT as f32
                {
                    return Err(invalid("mono tile inverse footprint exceeds supported coordinates"));
                }
            }
        }
    }
    let curved = cfg.kernel.fit_curvature.unwrap_or(cfg.backend == Backend::HandheldBurstSr);
    let mut accum = Tile::new(tile.x, tile.y, tile.width, tile.height, 1,
        cfg.kernel.debias_deposit, curved);
    for y in 0..tile.height { for x in 0..tile.width {
        let (rx, ry) = geom.to_ref((tile.x + x) as f32, (tile.y + y) as f32);
        let q = inputs.kernels.precision_at(rx, ry);
        if q.iter().any(|v| !v.is_finite()) || q[0] <= 0.0 || q[2] <= 0.0
            || q[0] * q[2] - q[1] * q[1] <= 0.0
        {
            return Err(invalid("mono tile requires a finite positive definite kernel"));
        }
        accum.precision[y * tile.width + x] = q;
    }}
    let mut frame_den = vec![0.0; accum.block_w * accum.block_h];
    for (i, &sky) in frame_sky.iter().enumerate() {
        accumulate_frame(&mut accum, &geom, inputs, i, cfg.backend,
            inputs.noise.variance(0.18), inputs.frames[i].noise, sky,
            Some((inputs.frames[inputs.reference].width as f32, inputs.frames[inputs.reference].height as f32)),
            feather_fraction,
            &mut frame_den);
    }
    let mut values = vec![f32::NAN; tile.width * tile.height];
    for (p, value) in values.iter_mut().enumerate() {
        if !accum.den[0][p].is_finite() || !accum.num[0][p].is_finite() {
            return Err(invalid("non-finite mono tile accumulation"));
        }
        if accum.den[0][p] <= 0.0 {
            accum.stats.unsupported_pixels += 1;
            continue;
        }
        *value = accum.num[0][p] / accum.den[0][p];
        if accum.mom[0].is_empty() { continue; }
        let m = &accum.mom[0][p];
        if m.clipped {
            if cfg.kernel.fit_clipped == Some(true)
                && let Some(v) = bounded_clipped_plane(std::slice::from_ref(m)) { *value = v[0]; }
        } else if let Some(v) = plane_fit(m) {
            *value = v;
            if !accum.curvature[0].is_empty()
                && let Some((delta, blend)) = accum.curvature[0][p].correction() {
                    let proposed = v + blend * delta;
                    if proposed.is_finite() && proposed >= 0.0 && blend > 0.0 { *value = proposed; }
                }
        }
    }
    if values.iter().zip(&accum.den[0]).any(|(v, &w)| w > 0.0 && !v.is_finite()) {
        return Err(invalid("non-finite supported mono tile estimate"));
    }
    let effective: Vec<_> = accum.block_den.iter().zip(&accum.block_den2)
        .filter_map(|(&d, &d2)| (d2 > 0.0).then_some(d * d / d2)).collect();
    if !effective.is_empty() {
        accum.stats.min_effective_frames = effective.iter().copied().fold(f32::INFINITY, f32::min);
        accum.stats.mean_effective_frames = effective.iter().sum::<f32>() / effective.len() as f32;
    }
    let [weight, _, _] = accum.den;
    let [count, _, _] = accum.cnt;
    Ok(MonoTile { values, weight, count, stats: accum.stats })
}

impl Geometry {
    /// Reference sensor coordinate -> local output coordinate.
    #[inline]
    fn to_out(&self, rx: f32, ry: f32) -> (f32, f32) {
        (
            (rx + 0.5) * self.scale - 0.5 - self.origin.0,
            (ry + 0.5) * self.scale - 0.5 - self.origin.1,
        )
    }

    /// Local output coordinate -> reference sensor coordinate.
    #[inline]
    fn to_ref(&self, ox: f32, oy: f32) -> (f32, f32) {
        (
            (ox + self.origin.0 + 0.5) / self.scale - 0.5,
            (oy + self.origin.1 + 0.5) / self.scale - 0.5,
        )
    }
}

/// Accumulators for one output tile.
struct Tile {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    num: [Vec<f32>; 3],
    den: [Vec<f32>; 3],
    cnt: [Vec<f32>; 3],
    /// Moments of the samples under each pixel, for `plane_fit`. Empty when
    /// the plane fit is off, in which case the deposit is a plain mean.
    mom: [Vec<Moments>; 3],
    curvature: [Vec<Curvature>; 3],
    precision: Vec<[f32; 3]>,
    rejected: Vec<f32>,
    /// Per-frame weight totals, on the decimated diagnostic grid.
    block_den: Vec<f32>,
    block_den2: Vec<f32>,
    block_w: usize,
    block_h: usize,
    stats: ProductStats,
}

impl Tile {
    /// `channels` is one for a monochrome sensor and three for a mosaic. The
    /// unused planes are allocated empty rather than full of zeros: at a 2x
    /// reconstruction of a 26 MP sensor the two spare channels would be two and
    /// a half gigabytes of nothing.
    fn new(x0: usize, y0: usize, w: usize, h: usize, channels: usize, fit: bool, curved: bool) -> Self {
        let n = w * h;
        let sized = |c: usize| vec![0.0; if c < channels { n } else { 0 }];
        let moments =
            |c: usize| vec![Moments::default(); if fit && c < channels { n } else { 0 }];
        let bw = w.div_ceil(EFFECTIVE_FRAMES_CELL);
        let bh = h.div_ceil(EFFECTIVE_FRAMES_CELL);
        Self {
            x0,
            y0,
            w,
            h,
            num: [sized(0), sized(1), sized(2)],
            den: [sized(0), sized(1), sized(2)],
            cnt: [sized(0), sized(1), sized(2)],
            mom: [moments(0), moments(1), moments(2)],
            curvature: std::array::from_fn(|c| vec![Curvature::default(); if curved && fit && c < channels { n } else { 0 }]),
            precision: vec![[0.0; 3]; n],
            rejected: vec![0.0; n],
            block_den: vec![0.0; bw * bh],
            block_den2: vec![0.0; bw * bh],
            block_w: bw,
            block_h: bh,
            stats: ProductStats::default(),
        }
    }


}

/// How far a pixel's samples may sit off its centre before the plane fit is
/// refused, in output pixels.
///
/// This is the guard that matters, and the first version of it did not work. It
/// compared the distance the fit moved the pixel against the local gradient
/// times the sample spread -- a bound that grows in step with the thing it is
/// bounding, so it never fired. Where the samples were sparse the fit
/// extrapolated, a handful of pixels came out astronomically large, and the
/// exposure normalisation then crushed the whole image to nothing: at
/// `k_detail` 0.03 the sky came out at 2.4e-9 of its proper level.
///
/// So the bound is absolute. It was a quarter of a pixel, on the argument that
/// beyond that the plane is being asked about a place it has no samples -- and
/// that refused the fit exactly where it was needed. The offset is not random
/// per pixel: every pixel of one channel sees the same dither phases, so a
/// whole channel sits a channel-specific fraction of a pixel off centre, and
/// for red and blue on a lattice of twice green's pitch that fraction is
/// typically a quarter. The guard fell back to the mean, which carries the
/// bias in full, and red and blue came out shifted against each other. Within
/// the kernel's footprint a plane is a fair model, and the footprint is a few
/// pixels; the degenerate fits that the old bound was really protecting
/// against are the condition number's job, below.
const MAX_CENTROID_OFFSET: f32 = 0.6;

/// Smallest usable determinant of the sample scatter, relative to its own
/// scale, before the fit is called degenerate and the mean is kept.
const MIN_FIT_CONDITION: f32 = 0.05;

/// The value at a pixel's centre, rather than at wherever its samples averaged.
///
/// The merge sets an output pixel to the weighted mean of the samples under it.
/// That is the right estimator only if those samples sit symmetrically about
/// the pixel's centre, and they do not: every pixel gets whatever the dither
/// happened to leave there, with its own centroid a little off centre. For a
/// locally linear signal the mean is
///
/// ```text
///   mean = v(pixel) + g . (centroid of the samples, relative to the pixel)
/// ```
///
/// so it is the value not at the pixel but a little way towards where its
/// samples happened to lie. The offset differs for every pixel, so the error is
/// not a blur that could be characterised and undone -- it varies pixel to
/// pixel like noise, and being proportional to the local gradient it appears
/// around stars and not on blank sky.
///
/// Fitting a plane instead of taking a mean removes it. The fit needs the first
/// and second moments of the sample positions and the first moments of the
/// values, all of which the deposit can accumulate as it goes, so the estimate
/// at a pixel uses only that pixel's own samples. That matters for more than
/// tidiness: a correction that read neighbouring output pixels would give a
/// different answer at the edge of a region than in the middle of a frame, and
/// reconstructing a crop would stop agreeing with reconstructing the whole.
///
/// This matters most where samples are sparse, which is exactly where the
/// kernel wants to be narrow. On a 96-frame burst, pushing `k_detail` below
/// 0.06 made the bright stars wider rather than sharper; that is this bias
/// growing as the samples thin out.
#[inline]
fn plane_fit(m: &Moments) -> Option<f32> {
    checked_plane_fit(m).ok()
}

fn checked_plane_fit(m: &Moments) -> std::result::Result<f32, &'static str> {
    let (s0, sx, sy) = (m.w, m.wx, m.wy);
    if s0 <= 0.0 || m.clipped {
        return Err("empty_or_clipped");
    }
    // Central second moments: the scatter of the samples about their own
    // centroid, which is what says whether a plane is determined at all.
    let (cx, cy) = (sx / s0, sy / s0);
    // How far from the pixel the fit is being asked to reach.
    if cx.hypot(cy) > MAX_CENTROID_OFFSET {
        return Err("centroid");
    }
    let mxx = m.wxx / s0 - cx * cx;
    let mxy = m.wxy / s0 - cx * cy;
    let myy = m.wyy / s0 - cy * cy;
    let det = mxx * myy - mxy * mxy;
    let scale = (mxx + myy) * 0.5;
    if scale <= 1e-12 || det <= MIN_FIT_CONDITION * scale * scale {
        return Err("condition");
    }
    // Covariance of value with position, about the centroid.
    let mean = m.wv / s0;
    let cvx = m.wvx / s0 - cx * mean;
    let cvy = m.wvy / s0 - cy * mean;
    // Gradient from the normal equations, then the plane's value at the pixel
    // centre, which is the centroid displaced by -(cx, cy).
    let gx = (myy * cvx - mxy * cvy) / det;
    let gy = (mxx * cvy - mxy * cvx) / det;
    let moved = -(gx * cx + gy * cy);
    if !moved.is_finite() {
        return Err("nonfinite");
    }
    Ok(mean + moved)
}

/// One pixel's accumulated moments, in one channel.
#[derive(Clone, Copy, Default)]
struct Moments {
    /// Whether any sample under this pixel was at the sensor's ceiling.
    ///
    /// A plane is a model of a locally linear signal, and a clipped profile is
    /// not one: at the edge of a saturated plateau the samples inside read
    /// full scale and the samples outside read the flank, the fitted gradient
    /// is enormous, and the plane read at the pixel centre overshoots the
    /// ceiling by a random amount in each channel. That was the speckle inside
    /// bright star cores, after the ceiling deposit had given the mean the one
    /// fallback. Any centroid correction there must use the separate bounded,
    /// shared-channel fit; the ordinary unconstrained plane is not used.
    clipped: bool,
    w: f32,
    wx: f32,
    wy: f32,
    wxx: f32,
    wxy: f32,
    wyy: f32,
    wv: f32,
    wvx: f32,
    wvy: f32,
    vmin: f32,
    vmax: f32,
}

impl Moments {
    #[inline]
    fn add(&mut self, w: f32, dx: f32, dy: f32, v: f32) {
        if self.w == 0. {
            self.vmin = v;
            self.vmax = v;
        } else {
            self.vmin = self.vmin.min(v);
            self.vmax = self.vmax.max(v);
        }
        self.w += w;
        self.wx += w * dx;
        self.wy += w * dy;
        self.wxx += w * dx * dx;
        self.wxy += w * dx * dy;
        self.wyy += w * dy * dy;
        self.wv += w * v;
        self.wvx += w * v * dx;
        self.wvy += w * v * dy;
    }
}

/// Correct the sampling centroid only if all channels constrain a plane. One
/// shared blend keeps every channel within its measured sample range; no
/// channel independently switches to an extrapolated clipped-core estimate.
fn bounded_clipped_plane(moments: &[Moments]) -> Option<[f32; 3]> {
    let mut means = [0.; 3];
    let mut moved = [0.; 3];
    let mut blend = 1_f32;
    for (c, m) in moments.iter().enumerate() {
        let value = plane_fit(&Moments { clipped: false, ..*m })?;
        if !value.is_finite() { return None; }
        let mean = m.wv / m.w;
        means[c] = mean;
        moved[c] = value - mean;
        if moved[c] > 0. {
            blend = blend.min((m.vmax - mean) / moved[c]);
        } else if moved[c] < 0. {
            blend = blend.min((m.vmin - mean) / moved[c]);
        }
    }
    let blend = blend.clamp(0., 1.);
    Some(std::array::from_fn(|c| means[c] + blend * moved[c]))
}

fn write_clipped_trace(
    writer: &mut dyn std::io::Write, x:usize, y:usize, moments:&[Moments], enabled:bool, applied:[f32;3],
) -> std::io::Result<()> {
    let fits:Vec<_>=moments.iter().map(|m|checked_plane_fit(&Moments {clipped:false,..*m})).collect();
    let limits:Vec<_>=moments.iter().zip(&fits).map(|(m,fit)| {
        let Ok(value)=fit else {return f32::NAN;};
        let mean=m.wv/m.w;let delta=value-mean;
        if delta>0. {((m.vmax-mean)/delta).clamp(0.,1.)}
        else if delta<0. {((m.vmin-mean)/delta).clamp(0.,1.)} else {1.}
    }).collect();
    let shared=if fits.iter().all(|f|matches!(f,Ok(v) if v.is_finite())) {
        limits.iter().copied().fold(1f32,f32::min)
    } else {0.};
    for (c,m) in moments.iter().enumerate() {
        let reason=fits[c].as_ref().map_or_else(|e|*e,|v|if v.is_finite() {"pass"} else {"nonfinite"});
        let plane=fits[c].as_ref().copied().unwrap_or(f32::NAN);
        writeln!(writer,"{x},{y},{c},{},{reason},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            m.clipped,m.w,m.wx,m.wy,m.wxx,m.wxy,m.wyy,m.wv,m.wvx,m.wvy,m.vmin,m.vmax,m.wv/m.w,plane,limits[c],shared,enabled,applied[c])?;
    }
    Ok(())
}

/// Bilinear demosaic of one rectangle of a mosaic, used only by the RGB
/// baseline. Kept local to this crate: the reconstruction path never calls it.
fn demosaic_region(
    frame: &RawFrame,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> [Vec<f32>; 3] {
    let mut out = [vec![0.0f32; w * h], vec![0.0f32; w * h], vec![0.0f32; w * h]];
    for j in 0..h {
        for i in 0..w {
            let x = x0 + i;
            let y = y0 + j;
            let mut acc = [0.0f32; 3];
            let mut wsum = [0.0f32; 3];
            // A 5x5 neighbourhood always contains every colour of a Bayer
            // mosaic, whatever the phase.
            for dy in -2i64..=2 {
                for dx in -2i64..=2 {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= frame.width as i64 || ny >= frame.height as i64 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    let idx = ny * frame.width + nx;
                    let v = frame.value(nx, ny);
                    if !frame.usable_value(idx, v) {
                        continue;
                    }
                    let c = frame.channel_at(nx, ny);
                    let d2 = (dx * dx + dy * dy) as f32;
                    let wt = 1.0 / (1.0 + d2);
                    acc[c] += v * wt;
                    wsum[c] += wt;
                }
            }
            // A monochrome frame fills all three planes with its one value, so
            // that the RGB baseline backend and the preview stay grey rather
            // than becoming a red image with two empty channels.
            for (c, o) in out.iter_mut().enumerate() {
                let src = if frame.is_mono() { 0 } else { c };
                o[j * w + i] = if wsum[src] > 0.0 { acc[src] / wsum[src] } else { 0.0 };
            }
        }
    }
    out
}

/// Source rectangle in one frame that can influence a tile.
fn source_bounds(
    geom: &Geometry,
    tile: &Tile,
    warp: &WarpField,
    frame_w: usize,
    frame_h: usize,
) -> Option<(usize, usize, usize, usize)> {
    // Reference-coordinate rectangle covered by the tile, widened by the kernel
    // support so that samples just outside still contribute.
    let r = geom.radius * geom.chroma_variance.sqrt();
    let (rx0, ry0) = geom.to_ref(tile.x0 as f32 - r, tile.y0 as f32 - r);
    let (rx1, ry1) = geom.to_ref(
        (tile.x0 + tile.w) as f32 + r,
        (tile.y0 + tile.h) as f32 + r,
    );

    let inv = warp.global.inverse()?;
    // The local field displaces by at most this much, and the inverse is taken
    // through the global part only, so pad by it. The chromatic correction
    // moves samples too, and by up to a couple of pixels at the frame corner.
    // Local displacement is measured in reference pixels. After inverse
    // scaling, especially across instruments, it can span more source pixels.
    // Retain the old padding as a lower bound to preserve previous coverage.
    let local = warp.max_local();
    let pad_x = local * (inv.m[0].abs() + inv.m[1].abs()).max(1.0) + geom.chroma_pad + 1.0;
    let pad_y = local * (inv.m[3].abs() + inv.m[4].abs()).max(1.0) + geom.chroma_pad + 1.0;

    let corners = [(rx0, ry0), (rx1, ry0), (rx0, ry1), (rx1, ry1)];
    let mut lo = (f32::INFINITY, f32::INFINITY);
    let mut hi = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &(x, y) in &corners {
        let (sx, sy) = inv.apply(x, y);
        lo.0 = lo.0.min(sx);
        lo.1 = lo.1.min(sy);
        hi.0 = hi.0.max(sx);
        hi.1 = hi.1.max(sy);
    }

    let x0 = (lo.0 - pad_x).floor().max(0.0) as usize;
    let y0 = (lo.1 - pad_y).floor().max(0.0) as usize;
    let x1 = ((hi.0 + pad_x).ceil().max(0.0) as usize).saturating_add(1).min(frame_w);
    let y1 = ((hi.1 + pad_y).ceil().max(0.0) as usize).saturating_add(1).min(frame_h);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0, y0, x1 - x0, y1 - y0))
}

/// A frame's sky level in each channel. Sample complete Bayer cells so a
/// stride cannot silently omit a color. Mono retains its scalar sampling.
/// Raw per-channel background levels used for reconstruction's noise weights.
/// Also exposed for diagnostic probes so they use the same estimator.
pub fn sky_of(frame: &RawFrame) -> [f32; 3] {
    let n = frame.width * frame.height;
    let mut per: [Vec<f32>; 3] = Default::default();
    let mut sample = |x: usize, y: usize| {
        let i = y * frame.width + x;
        let v = frame.samples.value_in_cell(i, (y & 1) * 2 + (x & 1));
        if frame.usable_value(i, v) {
            let c = if frame.is_mono() { 0 } else { frame.cfa.color_at(x, y).index() };
            per[c].push(v);
        }
    };
    if frame.is_mono() {
        let stride = (n / 240_000).max(1);
        for i in (0..n).step_by(stride) { sample(i % frame.width, i / frame.width); }
    } else {
        let cw = frame.width.div_ceil(2);
        let cells = cw * frame.height.div_ceil(2);
        let stride = (cells / 60_000).max(1);
        for i in (0..cells).step_by(stride) {
            let (x, y) = (2 * (i % cw), 2 * (i / cw));
            for dy in 0..2 { for dx in 0..2 {
                if x+dx < frame.width && y+dy < frame.height { sample(x+dx, y+dy); }
            }}
        }
    }
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        if per[c].is_empty() {
            continue;
        }
        per[c].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out[c] = per[c][per[c].len() / 2];
    }
    if frame.is_mono() {
        out = [out[0]; 3];
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn accumulate_frame(
    tile: &mut Tile,
    geom: &Geometry,
    inputs: &MergeInputs,
    frame_index: usize,
    backend: Backend,
    weight_norm: f32,
    noise_model: NoiseModel,
    sky: [f32; 3],
    photo_extent: Option<(f32, f32)>,
    feather_fraction: f32,
    frame_den: &mut [f32],
) {
    let frame = &inputs.frames[frame_index];
    let warp = &inputs.warps[frame_index];
    let Some((sx0, sy0, sw, sh)) =
        source_bounds(geom, tile, warp, frame.width, frame.height)
    else {
        return;
    };

    let channels = frame.channels();
    let rgb_source = matches!(backend, Backend::RgbMeanBaseline);
    let demosaiced = if rgb_source {
        Some(demosaic_region(frame, sx0, sy0, sw, sh))
    } else {
        None
    };

    let photo = inputs.photometry[frame_index];
    let frame_w = inputs.frame_weight[frame_index];
    let (full_w, crop_x) = if frame.metadata.full_width > 0 {
        (frame.metadata.full_width, frame.metadata.crop.0)
    } else { (frame.width, 0) };
    let (full_h, crop_y) = if frame.metadata.full_height > 0 {
        (frame.metadata.full_height, frame.metadata.crop.1)
    } else { (frame.height, 0) };
    let feather_span = feather_fraction as f64 * full_w.min(full_h) as f64;
    // A frame with no weight contributes nothing, and walking its source
    // rectangle to discover that a few million times is the difference between
    // reconstructing one filter of a set and reconstructing all of them.
    // Negated deliberately: a weight that is NaN has to take this branch too,
    // and `frame_w <= 0.0` would let it through into the accumulators.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(frame_w > 0.0) {
        return;
    }
    let use_robustness = !matches!(backend, Backend::RgbMeanBaseline);
    let cfa: CfaPattern = frame.cfa;
    let mono = frame.is_mono();
    let radius = geom.radius;
    // Red and blue are sampled a quarter as densely as green, so their kernels
    // are widened to compensate. A monochrome sensor samples everything at full
    // density and needs none of that.
    let channel_variance = if frame.is_mono() {
        [1.0, 1.0, 1.0]
    } else {
        [geom.chroma_variance, 1.0, geom.chroma_variance]
    };

    frame_den.iter_mut().for_each(|v| *v = 0.0);


    // Photometric fields and obstruction masks are fitted to block medians
    // on the reference grid. Read them on that same grid, including across a
    // meridian flip. A source-sensor lookup applies them to the opposite sky.
    let (fw, fh) = photo_extent.unwrap_or((frame.width.max(1) as f32, frame.height.max(1) as f32));
    for j in 0..sh {
        let y = sy0 + j;
        for i in 0..sw {
            let x = sx0 + i;
            let (photo_x, photo_y) = warp.map(x as f32, y as f32);
            let pu = 2.0 * photo_x / fw - 1.0;
            let pv = 2.0 * photo_y / fh - 1.0;
            let sidx = y * frame.width + x;
            tile.stats.total_samples += 1;
            // Something stood between this part of the sensor and the sky.
            // Not a measurement of the scene; see `PhotometricMatch::blocked`.
            if photo.blocked_at(pu, pv) {
                tile.stats.masked_samples += 1;
                continue;
            }
            let raw = frame.samples.value_in_cell(sidx, (y & 1) * 2 + (x & 1));
            // A saturated sample is kept, at the ceiling. It is a measurement:
            // the value here is at least full scale. Discarding it -- which is
            // what `usable_value` does, and rightly, for every stage that fits
            // something to the data -- leaves the pixel to be reconstructed
            // from the few samples that fell below white, a different biased
            // few in each channel, and the core of every bright star comes out
            // as random colour. Depositing it at the ceiling gives the core the
            // one value it can honestly have, in all three channels alike.
            let saturated = raw >= 1.0 && !frame.defects.get(sidx);
            if !saturated && !frame.usable_value(sidx, raw) {
                tile.stats.masked_samples += 1;
                continue;
            }
            if saturated {
                tile.stats.saturated_samples += 1;
            }
            let raw = raw.min(1.0);

            // The sample's channel decides both its kernel width and its
            // chromatic correction, so it is resolved before anything else.
            let (channel, var_scale) = match &demosaiced {
                Some(_) => (usize::MAX, 1.0f32),
                None => {
                    let c = if mono { 0 } else { cfa.color_at(x, y).index() };
                    (c, channel_variance[c])
                }
            };

            // Undo the lens's per-channel magnification in the frame's own
            // sensor coordinates, then map the frame onto the reference.
            let (cx_s, cy_s) = if channel == usize::MAX {
                (x as f32, y as f32)
            } else {
                inputs.chroma.apply(channel, x as f32, y as f32)
            };
            let (rx, ry) = warp.map(cx_s, cy_s);
            let (ox, oy) = geom.to_out(rx, ry);

            // Reject early if the sample cannot reach this tile.
            let lx = ox - tile.x0 as f32;
            let ly = oy - tile.y0 as f32;
            let reach = radius * geom.chroma_variance.sqrt();
            if lx < -reach
                || ly < -reach
                || lx > tile.w as f32 + reach
                || ly > tile.h as f32 + reach
            {
                continue;
            }

            // Weights that do not depend on the output pixel.
            //
            // The noise weight is taken from the frame's sky level in this
            // channel, not from the sample's own value. Inverse-variance
            // weighting is optimal when every sample under a pixel measures the
            // same thing; a weight that follows the sample's value does not do
            // that, it prefers whichever samples happened to read lower. On a
            // gradient that pulls the estimate towards the dim side -- a star's
            // flank is weighted more than its peak, a sky that is brighter in
            // some frames than others is weighted towards the frames where it
            // was darkest -- and on one test burst, where a third of the frames
            // carried cloud, the stack's sky took on a shape none of the frames
            // had. Per frame, the weight still says what it should: a frame
            // shot through haze or a brighter sky counts for less, everywhere
            // in it alike.
            //
            // The photometric map scales the value, so it scales the noise
            // with it; the pedestal does not move the variance at all.
            let gain = photo.gain_at(if channel == usize::MAX {1} else {channel},pu,pv);
            let level = if channel == usize::MAX { sky[1] } else { sky[channel] };
            let variance = noise_model.variance(level) * gain * gain;
            let w_noise = weight_norm / variance.max(1e-12);

            let w_rob = if use_robustness {
                inputs.robustness.at(frame_index, rx, ry)
            } else {
                1.0
            };
            let w_local = warp.local_confidence_at_ref(rx, ry).max(0.05);
            let w_lucky = match inputs.lucky {
                Some(l) => l.weight(frame_index, rx, ry),
                None => 1.0,
            };
            let base = w_noise * w_rob * w_local * w_lucky * frame_w;
            if base <= 1e-9 {
                if w_rob <= 1e-6 || w_lucky <= 1e-6 {
                    tile.stats.rejected_samples += 1;
                    let px = (lx.round() as i64).clamp(0, tile.w as i64 - 1) as usize;
                    let py = (ly.round() as i64).clamp(0, tile.h as i64 - 1) as usize;
                    tile.rejected[py * tile.w + px] += 1.0;
                }
                continue;
            }


            // Native pixel centres keep first/last detector rows positive.
            // Taper after the existing rejection threshold so tiny legitimate
            // feather weights cannot introduce unsupported edge pixels.
            let base = if feather_span > 0.0 {
                let fx = (x + crop_x) as f64 + 0.5;
                let fy = (y + crop_y) as f64 + 0.5;
                let distance = fx.min(full_w as f64 - fx).min(fy.min(full_h as f64 - fy));
                let t = (distance / feather_span).clamp(0.0, 1.0);
                base * (t * t * (3.0 - 2.0 * t)) as f32
            } else { base };

            let radius_c = radius * var_scale.sqrt();

            let x_lo = ((lx - radius_c).ceil() as i64).max(0) as usize;
            let x_hi = ((lx + radius_c).floor() as i64).min(tile.w as i64 - 1);
            let y_lo = ((ly - radius_c).ceil() as i64).max(0) as usize;
            let y_hi = ((ly + radius_c).floor() as i64).min(tile.h as i64 - 1);
            if x_hi < x_lo as i64 || y_hi < y_lo as i64 {
                tile.stats.out_of_bounds_samples += 1;
                continue;
            }

            let inv_scale = 1.0 / geom.scale;
            let mut deposited = false;

            for py in y_lo..=(y_hi as usize) {
                let dy_out = py as f32 - ly;
                let dy = dy_out * inv_scale;
                for px in x_lo..=(x_hi as usize) {
                    let dx_out = px as f32 - lx;
                    let dx = dx_out * inv_scale;
                    // Every contribution uses the same continuous output kernel.
                    let p = py * tile.w + px;
                    let [qxx, qxy, qyy] = tile.precision[p];
                    let q = (qxx * dx * dx + 2.0 * qxy * dx * dy + qyy * dy * dy)
                        / var_scale.max(1e-6);
                    let k = (-0.5 * q).exp();
                    if k <= 1e-4 {
                        continue;
                    }
                    let w = base * k;
                    let bp = (py / EFFECTIVE_FRAMES_CELL) * tile.block_w
                        + (px / EFFECTIVE_FRAMES_CELL);

                    match &demosaiced {
                        Some(rgbs) => {
                            for (c, plane) in rgbs.iter().enumerate().take(channels) {
                                let v = photo.apply_at(c, plane[j * sw + i], pu, pv);
                                tile.num[c][p] += w * v;
                                tile.den[c][p] += w;
                                tile.cnt[c][p] += 1.0;
                                if !tile.mom[c].is_empty() {
                                    tile.mom[c][p].add(w, dx_out, dy_out, v);
                                    if saturated {
                                        tile.mom[c][p].clipped = true;
                                    }
                                }
                            }
                            frame_den[bp] += 3.0 * w;
                        }
                        None => {
                            let v = photo.apply_at(channel, raw, pu, pv);
                            tile.num[channel][p] += w * v;
                            tile.den[channel][p] += w;
                            tile.cnt[channel][p] += 1.0;
                            if !tile.mom[channel].is_empty() {
                                tile.mom[channel][p].add(w, dx_out, dy_out, v);
                                if !tile.curvature[channel].is_empty() {
                                    let noise = noise_model.variance(raw)*gain*gain;
                                    tile.curvature[channel][p].add(w, dx_out, dy_out, v, noise);
                                }
                                // Mono has dense support in one channel: a saturated
                                // site's weak (<1% of its peak) kernel tail must not
                                // disable the surrounding wing's profile fit. Keep
                                // the stricter shared-channel guard for sparse CFA.
                                if saturated && (!mono || k > 0.01) {
                                    tile.mom[channel][p].clipped = true;
                                }
                            }
                            frame_den[bp] += w;
                        }
                    }
                    deposited = true;
                }
            }
            if deposited {
                tile.stats.accumulated_samples += 1;
            }
        }
    }


    // Fold this frame's totals into the effective-frame-count accumulators.
    for (b, &d) in frame_den.iter().enumerate() {
        if d > 0.0 {
            tile.block_den[b] += d;
            tile.block_den2[b] += d * d;
        }
    }
}

/// Fill output pixels with no support, by growing outward from supported
/// neighbours of the same channel.
///
/// Filled pixels are counted and reported: an output that needed a lot of this
/// is an output whose scale was too ambitious for the burst.
fn fill_holes(rgb: &mut [Plane<f32>; 3], den: &[Plane<f32>; 3], channels: usize) -> usize {
    let (w, h) = (rgb[0].width, rgb[0].height);
    let mut filled = 0usize;
    for c in 0..channels {
        let missing: Vec<usize> = (0..w * h).filter(|&i| den[c].data[i] <= 0.0).collect();
        if missing.is_empty() {
            continue;
        }
        filled += missing.len();
        for &i in &missing {
            let (x, y) = (i % w, i / w);
            let mut acc = 0.0f32;
            let mut n = 0.0f32;
            'radius: for r in 1..6i64 {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx.abs() != r && dy.abs() != r {
                            continue;
                        }
                        let nx = x as i64 + dx;
                        let ny = y as i64 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if den[c].data[j] > 0.0 {
                            acc += rgb[c].data[j];
                            n += 1.0;
                        }
                    }
                }
                if n > 0.0 {
                    break 'radius;
                }
            }
            if n > 0.0 {
                rgb[c].data[i] = acc / n;
            }
        }
    }
    filled
}

/// Run the merge.
pub fn reconstruct(
    inputs: &MergeInputs,
    cfg: &ReconstructionConfig,
) -> Result<ReconstructionProduct> {
    reconstruct_impl(inputs, cfg, None, false, None, None,None)
}

/// Explicit, potentially large per-channel fit trace for controlled diagnostics.
/// Does not alter fit thresholds, sample accumulation, or reconstruction output.
pub fn reconstruct_with_fit_trace(
    inputs: &MergeInputs,
    cfg: &ReconstructionConfig,
    trace: &mut dyn std::io::Write,
) -> Result<ReconstructionProduct> {
    reconstruct_impl(inputs, cfg, Some(trace), false, None, None,None)
}

/// Read-only clipped-fit diagnostic. Rows contain the actual accumulated moments
/// and applied output, only where a contributing sample clipped. Coordinates
/// are ROI-local. Does not alter the ordinary estimator or the existing fit CSV.
pub fn reconstruct_with_clipped_trace(
    inputs: &MergeInputs, cfg: &ReconstructionConfig, trace: &mut dyn std::io::Write,
) -> Result<ReconstructionProduct> {
    reconstruct_impl(inputs,cfg,None,false,None,None,Some(trace))
}

/// Experimental variance-bounded curvature blend; never selected by normal reconstruction.
pub fn reconstruct_variance_blend_experiment(
    inputs: &MergeInputs, cfg: &ReconstructionConfig, trace: &mut dyn std::io::Write,
) -> Result<ReconstructionProduct> {
    reconstruct_impl(inputs, cfg, Some(trace), true, None, None,None)
}

/// Experimental multi-pass output. Statistics remain per source pass: effective
/// frame counts/rejection maps cannot honestly be combined by RGB selection.
pub struct SupportSelectionExperiment {
    pub rgb: [Plane<f32>; 3],
    pub weight: [Plane<f32>; 3],
    pub count: [Plane<f32>; 3],
    /// 0/1/2 is the chosen kernel; fallback retains kernel zero.
    /// The clipped-preserving experiment additionally uses 3 for production.
    pub selection: Plane<u8>,
    pub fallback_pixels: usize,
    pub source_stats: Vec<ProductStats>,
}

/// Reference implementation of the frozen offline rule. Three sequential
/// merges, no trace CSV parsing. Caller supplies circular kernels in increasing
/// support order. This is a validation path, not an application default.
pub fn reconstruct_support_selection_experiment(
    inputs: &MergeInputs, cfg: &ReconstructionConfig, kernels: [&KernelField; 3],
) -> Result<SupportSelectionExperiment> {
    support_selection_impl(inputs,cfg,kernels,None)
}

fn support_selection_impl(
    inputs: &MergeInputs, cfg: &ReconstructionConfig, kernels: [&KernelField; 3],
    mut clipped_union: Option<&mut Vec<bool>>,
) -> Result<SupportSelectionExperiment> {
    if inputs.frames.is_empty() || inputs.reference >= inputs.frames.len()
        || inputs.frames[inputs.reference].channels() != 3
        || cfg.backend != Backend::HandheldBurstSr
        || cfg.kernel.fit_curvature != Some(true) || !cfg.kernel.debias_deposit {
        return Err(SrError::Reconstruction("support selection requires explicit CFA curvature and plane fitting".into()));
    }
    let mut result: Option<SupportSelectionExperiment> = None;
    let mut chosen: Vec<bool> = Vec::new();
    for (index, kernel) in kernels.into_iter().enumerate() {
        let pass = MergeInputs { frames:inputs.frames,warps:inputs.warps,reference:inputs.reference,
            photometry:inputs.photometry,noise:inputs.noise,robustness:inputs.robustness,
            kernels:kernel,frame_weight:inputs.frame_weight,lucky:inputs.lucky,chroma:inputs.chroma };
        let mut eligible = Vec::new();
        let output = reconstruct_impl(&pass,cfg,None,true,Some(&mut eligible),clipped_union.as_deref_mut(),None)?;
        if let Some(ref mut selected) = result {
            for p in 0..chosen.len() {
                if !chosen[p] && eligible[p] {
                    for c in 0..3 {
                        selected.rgb[c].data[p]=output.rgb[c].data[p];
                        selected.weight[c].data[p]=output.weight[c].data[p];
                        selected.count[c].data[p]=output.count[c].data[p];
                    }
                    selected.selection.data[p]=index as u8;
                    chosen[p]=true;
                }
            }
            selected.source_stats.push(output.stats);
        } else {
            chosen=eligible;
            result=Some(SupportSelectionExperiment {rgb:output.rgb,weight:output.weight,count:output.count,
                selection:Plane::filled(output.width,output.height,0),fallback_pixels:0,source_stats:vec![output.stats]});
        }
    }
    let mut result=result.expect("three kernels supplied");
    result.fallback_pixels=chosen.iter().filter(|&&v|!v).count();
    Ok(result)
}

/// Four-pass reference experiment. Preserves production RGB/weights/counts
/// wherever any production or candidate pass encounters a clipped footprint.
/// No threshold, dilation, blending or per-colour selection is introduced.
pub struct ClippedSupportSelectionExperiment {
    pub output: SupportSelectionExperiment,
    pub preserved_pixels: usize,
    pub production_stats: ProductStats,
}

pub fn reconstruct_clipped_support_selection_experiment(
    inputs: &MergeInputs, production_cfg: &ReconstructionConfig, kernels: [&KernelField; 3],
) -> Result<ClippedSupportSelectionExperiment> {
    let mut candidate_cfg=production_cfg.clone();
    candidate_cfg.kernel.fit_curvature=Some(true);
    let mut clipped_union=Vec::new();
    let mut output=support_selection_impl(inputs,&candidate_cfg,kernels,Some(&mut clipped_union))?;
    let production=reconstruct_impl(inputs,production_cfg,None,false,None,Some(&mut clipped_union),None)?;
    let mut preserved_pixels=0;
    for (p,&clipped) in clipped_union.iter().enumerate() {
        if clipped {
            preserved_pixels+=1;
            for c in 0..3 {
                output.rgb[c].data[p]=production.rgb[c].data[p];
                output.weight[c].data[p]=production.weight[c].data[p];
                output.count[c].data[p]=production.count[c].data[p];
            }
            output.selection.data[p]=3;
        }
    }
    // fallback_pixels still describes candidate eligibility before preservation.
    Ok(ClippedSupportSelectionExperiment {output,preserved_pixels,production_stats:production.stats})
}

/// Experimental RGB output. The blend fraction is not statistical confidence.
pub struct TaperedSupportSelectionExperiment {
    pub rgb: [Plane<f32>; 3],
    /// Shared RGB coefficient: zero retains production, one retains candidate.
    pub candidate_fraction: Plane<f32>,
    /// Original candidate choice, including pixels later blended with production.
    pub candidate_selection: Plane<u8>,
    /// Three unblended candidate passes in increasing support order.
    pub candidate_stats: Vec<ProductStats>,
    pub production_stats: ProductStats,
}

/// Frozen native-pixel taper. Full frame prevents truncation of the clipping halo.
pub fn reconstruct_tapered_support_selection_experiment(
    inputs: &MergeInputs, cfg: &ReconstructionConfig, kernels: [&KernelField; 3],
) -> Result<TaperedSupportSelectionExperiment> {
    if cfg.scale != 1. || cfg.roi.is_some() {
        return Err(SrError::Reconstruction("tapered experiment requires native scale and full frame".into()));
    }
    let mut candidate_cfg=cfg.clone();
    candidate_cfg.kernel.fit_curvature=Some(true);
    let mut mask=Vec::new();
    let candidate=support_selection_impl(inputs,&candidate_cfg,kernels,Some(&mut mask))?;
    let production=reconstruct_impl(inputs,cfg,None,false,None,Some(&mut mask),None)?;
    let (rgb,candidate_fraction)=taper_clipped_rgb(candidate.rgb,&production.rgb,&mask);
    Ok(TaperedSupportSelectionExperiment {rgb,candidate_fraction,
        candidate_selection:candidate.selection,candidate_stats:candidate.source_stats,
        production_stats:production.stats})
}

fn taper_clipped_rgb(mut candidate: [Plane<f32>;3], production: &[Plane<f32>;3], mask: &[bool])
    -> ([Plane<f32>;3],Plane<f32>) {
    let (w,h)=(candidate[0].width,candidate[0].height);
    // Distances >= 3 retain candidate exactly, including the empty-mask case.
    let mut distance2=vec![9u8;w*h];
    for (p,&clipped) in mask.iter().enumerate() {
        if !clipped {continue;}
        let (x,y)=((p%w) as isize,(p/w) as isize);
        for dy in -2isize..=2 {for dx in -2isize..=2 {
            let (nx,ny)=(x+dx,y+dy);
            if nx>=0 && ny>=0 && nx<w as isize && ny<h as isize {
                let q=ny as usize*w+nx as usize;
                distance2[q]=distance2[q].min((dx*dx+dy*dy) as u8);
            }
        }}
    }
    let mut fraction=Plane::filled(w,h,1.);
    for (p,&d2) in distance2.iter().enumerate() {
        if d2==9 {continue;}
        // Match the frozen offline float64 calculation before storing float32.
        let t=((f64::from(d2).sqrt()-1.)/2.).clamp(0.,1.);
        let alpha=t*t*(3.-2.*t);
        fraction.data[p]=alpha as f32;
        for c in 0..3 {
            candidate[c].data[p]=if alpha==0. {production[c].data[p]} else {
                let base=f64::from(production[c].data[p]);
                (base+alpha*(f64::from(candidate[c].data[p])-base)) as f32
            };
        }
    }
    (candidate,fraction)
}

/// Memory-bounded native-scale validation path. Source stats are ordered
/// stripe-major, three passes per stripe. Hole filling across stripes is not
/// validated: fail explicitly if any pass has unsupported pixels.
pub fn reconstruct_support_selection_striped_experiment(
    inputs: &MergeInputs, cfg: &ReconstructionConfig, kernels: [&KernelField; 3], stripe_height: usize,
) -> Result<SupportSelectionExperiment> {
    if cfg.scale != 1. || cfg.roi.is_some() || stripe_height == 0 {
        return Err(SrError::Reconstruction("striped experiment requires native scale, full frame and positive stripe height".into()));
    }
    let reference=inputs.frames.get(inputs.reference).ok_or_else(||SrError::Reconstruction("invalid reference".into()))?;
    let (w,h)=(reference.width,reference.height);
    let mut result=SupportSelectionExperiment {
        rgb:std::array::from_fn(|_|Plane::new(w,h)),weight:std::array::from_fn(|_|Plane::new(w,h)),
        count:std::array::from_fn(|_|Plane::new(w,h)),selection:Plane::new(w,h),fallback_pixels:0,source_stats:Vec::new(),
    };
    for y in (0..h).step_by(stripe_height) {
        let height=stripe_height.min(h-y);
        let mut local=cfg.clone();
        local.roi=Some((0,y,w,height));
        let stripe=reconstruct_support_selection_experiment(inputs,&local,kernels)?;
        if stripe.source_stats.iter().any(|s|s.unsupported_pixels>0) {
            return Err(SrError::Reconstruction("striped support selection has unsupported pixels; use the full-frame reference for this case".into()));
        }
        let range=y*w..(y+height)*w;
        for c in 0..3 {
            result.rgb[c].data[range.clone()].copy_from_slice(&stripe.rgb[c].data);
            result.weight[c].data[range.clone()].copy_from_slice(&stripe.weight[c].data);
            result.count[c].data[range.clone()].copy_from_slice(&stripe.count[c].data);
        }
        result.selection.data[range].copy_from_slice(&stripe.selection.data);
        result.fallback_pixels+=stripe.fallback_pixels;
        result.source_stats.extend(stripe.source_stats);
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_impl(
    inputs: &MergeInputs,
    cfg: &ReconstructionConfig,
    mut trace: Option<&mut dyn std::io::Write>,
    bounded_blend: bool,
    mut eligibility: Option<&mut Vec<bool>>,
    mut clipped_union: Option<&mut Vec<bool>>,
    mut clipped_trace: Option<&mut dyn std::io::Write>,
) -> Result<ReconstructionProduct> {
    if let Some(writer) = trace.as_mut() {
        writeln!(writer,"x,y,channel,plane,curvature,effective_samples,delta,channel_blend,shared_blend,valid,changed,plane_variance,quadratic_variance,plane_quad_covariance")?;
    }
    if let Some(writer)=clipped_trace.as_mut() {
        writeln!(writer,"x,y,channel,channel_clipped,reason,w,wx,wy,wxx,wxy,wyy,wv,wvx,wvy,vmin,vmax,mean,plane,local_limit,shared_limit,fit_enabled,applied")?;
    }
    let n = inputs.frames.len();
    if n == 0 {
        return Err(SrError::Reconstruction("empty burst".into()));
    }
    if inputs.photometry.iter().any(|p|p.log_gain.is_some()) {
        return Err(SrError::Reconstruction("spatial gain is supported only by the experimental mono tile path".into()));
    }
    if inputs.warps.len() != n
        || inputs.photometry.len() != n
        || inputs.frame_weight.len() != n
    {
        return Err(SrError::Reconstruction(
            "per-frame input arrays disagree on the frame count".into(),
        ));
    }
    if !(cfg.scale >= 1.0 && cfg.scale <= 8.0) {
        return Err(SrError::Reconstruction(format!(
            "scale {} is outside the supported range 1.0..8.0",
            cfg.scale
        )));
    }

    let reference = &inputs.frames[inputs.reference];
    let roi = cfg
        .roi
        .unwrap_or((0, 0, reference.width, reference.height));
    if roi.0 + roi.2 > reference.width || roi.1 + roi.3 > reference.height {
        return Err(SrError::Reconstruction(format!(
            "roi {:?} does not fit inside the {}x{} reference frame",
            roi, reference.width, reference.height
        )));
    }

    let scale = cfg.scale;
    let out_w = ((roi.2 as f32) * scale).round() as usize;
    let out_h = ((roi.3 as f32) * scale).round() as usize;
    let geom = Geometry {
        scale,
        origin: (roi.0 as f32 * scale, roi.1 as f32 * scale),
        radius: cfg.kernel.radius.max(0.5),
        chroma_variance: cfg.kernel.chroma_variance.clamp(1.0, MAX_CHANNEL_SCALE),
        chroma_pad: {
            // How far the correction can move a sample, at the far corner.
            inputs
                .chroma
                .max_displacement(reference.width as f32, reference.height as f32)
        },
    };

    // Normalise inverse-variance weights so the accumulators sit near unity and
    // f32 keeps its precision across a hundred frames.
    let weight_norm = inputs.noise.variance(0.18);

    // Each frame's sky level per channel, which is what its samples' noise
    // weight is taken from. See `accumulate_frame`.
    let frame_sky: Vec<[f32; 3]> = inputs.frames.par_iter().map(sky_of).collect();

    // One channel for a monochrome sensor, three for a mosaic. The spare
    // planes are allocated at zero size rather than full of nothing.
    let channels = reference.channels();
    let fit_curvature = cfg.kernel.fit_curvature
        .unwrap_or(channels == 1 && cfg.backend == Backend::HandheldBurstSr);
    let fit_clipped = cfg.kernel.fit_clipped
        .unwrap_or(channels == 3 && cfg.backend == Backend::HandheldBurstSr);
    let sized_w = |c: usize| if c < channels { out_w } else { 0 };
    let sized_h = |c: usize| if c < channels { out_h } else { 0 };

    if let Some(ref mut eligible) = eligibility { eligible.resize(out_w*out_h,false); }
    // Repeated passes OR into this mask; a new caller starts with an empty Vec.
    if let Some(ref mut clipped) = clipped_union { clipped.resize(out_w*out_h,false); }
    let tile_size = cfg.tile.clamp(64, 4096);
    let mut tiles: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut ty = 0;
    while ty < out_h {
        let th = tile_size.min(out_h - ty);
        let mut tx = 0;
        while tx < out_w {
            let tw = tile_size.min(out_w - tx);
            tiles.push((tx, ty, tw, th));
            tx += tw;
        }
        ty += th;
    }

    let mut rgb = [
        Plane::<f32>::new(out_w, out_h),
        Plane::<f32>::new(sized_w(1), sized_h(1)),
        Plane::<f32>::new(sized_w(2), sized_h(2)),
    ];
    let mut weight = [
        Plane::<f32>::new(out_w, out_h),
        Plane::<f32>::new(sized_w(1), sized_h(1)),
        Plane::<f32>::new(sized_w(2), sized_h(2)),
    ];
    let mut count = [
        Plane::<f32>::new(out_w, out_h),
        Plane::<f32>::new(sized_w(1), sized_h(1)),
        Plane::<f32>::new(sized_w(2), sized_h(2)),
    ];
    let mut fitted = 0u64;
    let mut curved = 0u64;
    let mut rejected = Plane::<f32>::new(out_w, out_h);
    let eff_w = out_w.div_ceil(EFFECTIVE_FRAMES_CELL);
    let eff_h = out_h.div_ceil(EFFECTIVE_FRAMES_CELL);
    let mut effective = Plane::<f32>::new(eff_w, eff_h);
    let mut stats = ProductStats::default();

    // Tiles run in parallel; results are blitted in bounded batches so peak
    // memory stays a small multiple of one tile rather than of the output.
    let batch = (rayon::current_num_threads() * 2).max(4);
    for chunk in tiles.chunks(batch) {
        let done: Vec<Tile> = chunk
            .par_iter()
            .map(|&(tx, ty, tw, th)| {
                let mut tile = Tile::new(tx, ty, tw, th, channels, cfg.kernel.debias_deposit, fit_curvature);
                for y in 0..th {
                    for x in 0..tw {
                        let (rx, ry) = geom.to_ref((tx + x) as f32, (ty + y) as f32);
                        tile.precision[y * tw + x] = inputs.kernels.precision_at(rx, ry);
                    }
                }
                let mut frame_den = vec![0.0f32; tile.block_w * tile.block_h];
                for (i, &sky) in frame_sky.iter().enumerate().take(n) {
                    accumulate_frame(
                        &mut tile,
                        &geom,
                        inputs,
                        i,
                        cfg.backend,
                        weight_norm,
                        inputs.noise,
                        sky,
                        None,
                        0.0,
                        &mut frame_den,
                    );
                }
                tile
            })
            .collect();

        for tile in done {
            for y in 0..tile.h {
                for x in 0..tile.w {
                    let src = y * tile.w + x;
                    let dst = (tile.y0 + y) * out_w + (tile.x0 + x);
                    let clipped = tile.mom.iter().any(|m| !m.is_empty() && m[src].clipped);
                    if let Some(ref mut mask) = clipped_union { mask[dst] |= clipped; }
                    let mut values = [0.; 3];
                    let mut corrections = [None; 3];
                    for c in 0..channels {
                        let d = tile.den[c][src];
                        weight[c].data[dst] = d;
                        count[c].data[dst] = tile.cnt[c][src];
                        values[c] = if d > 0. { tile.num[c][src] / d } else { 0. };
                        // A clipped footprint keeps the same mean estimator in every colour.
                        if !tile.mom[c].is_empty() && d > 0. && !clipped
                            && let Some(v) = plane_fit(&tile.mom[c][src]) {
                                fitted += 1;
                                values[c] = v;
                                if !tile.curvature[c].is_empty() {
                                    corrections[c] = if bounded_blend {
                                        tile.curvature[c][src].checked_with_policy(true).ok()
                                    } else { tile.curvature[c][src].correction() };
                                }
                            }
                    }
                    if clipped && fit_clipped {
                        let moments: [Moments; 3] = std::array::from_fn(|c|
                            if c < channels { tile.mom[c][src] } else { Moments::default() });
                        if let Some(v) = bounded_clipped_plane(&moments[..channels]) {
                            values = v;
                        }
                    }
                    if clipped
                        && let Some(writer)=clipped_trace.as_mut() {
                            let moments:Vec<_>=(0..channels).map(|c|tile.mom[c][src]).collect();
                            write_clipped_trace(*writer,tile.x0+x,tile.y0+y,&moments,fit_clipped,values)?;
                        }
                    // All colours need adequate support. A shared blend prevents
                    // fitting curvature in green while leaving sparse R/B on a plane.
                    // The corrections themselves remain independent channel profiles.
                    let blend = corrections[..channels].iter()
                        .map(|c| c.map_or(0., |(_, b)| b)).fold(1_f32, f32::min);
                    let proposed: [f32; 3] = std::array::from_fn(|c|
                        values[c] + blend*corrections[c].map_or(0., |(d, _)| d));
                    let valid = proposed[..channels].iter().all(|v| v.is_finite() && *v >= 0.);
                    if let Some(ref mut eligible) = eligibility {
                        eligible[dst]=valid && corrections[..channels].iter().all(Option::is_some)
                            && (0..channels).all(|c| tile.curvature[c][src].effective_samples()>=12.);
                    }
                    for c in 0..channels {
                        let v = if valid && blend > 0. { proposed[c] } else { values[c] };
                        if v != values[c] { curved += 1; }
                        rgb[c].data[dst] = v;
                        if let Some(writer) = trace.as_mut() {
                            let plane = if clipped { "clipped_footprint" }
                                else if tile.mom[c].is_empty() { "disabled" }
                                else { checked_plane_fit(&tile.mom[c][src]).map_or_else(|e| e, |_| "pass") };
                            let (reason, neff, delta, local_blend) = if tile.curvature[c].is_empty() {
                                ("disabled", 0., 0., 0.)
                            } else {
                                let m = &tile.curvature[c][src];
                                if plane != "pass" { ("plane_blocked", m.effective_samples(), 0., 0.) }
                                else { match m.checked_with_policy(bounded_blend) {
                                    Ok((d,b)) => (if b > 0. {"pass"} else {"shrinkage_zero"}, m.effective_samples(), d, b),
                                    Err(reason) => (reason, m.effective_samples(), 0., 0.),
                                }}
                            };
                            let vp = if tile.curvature[c].is_empty() { None }
                                else { tile.curvature[c][src].variance_probe() }.unwrap_or([f64::NAN;3]);
                            writeln!(writer,"{},{},{},{},{},{},{},{},{},{},{},{},{},{}",tile.x0+x,tile.y0+y,c,
                                plane,reason,neff,delta,local_blend,blend,valid,v != values[c],vp[0],vp[1],vp[2])?;
                        }
                    }
                }
            }
            for y in 0..tile.h {
                for x in 0..tile.w {
                    rejected.data[(tile.y0 + y) * out_w + (tile.x0 + x)] =
                        tile.rejected[y * tile.w + x];
                }
            }
            for by in 0..tile.block_h {
                for bx in 0..tile.block_w {
                    let b = by * tile.block_w + bx;
                    let d = tile.block_den[b];
                    let d2 = tile.block_den2[b];
                    let e = if d2 > 0.0 { d * d / d2 } else { 0.0 };
                    let gx = (tile.x0 / EFFECTIVE_FRAMES_CELL) + bx;
                    let gy = (tile.y0 / EFFECTIVE_FRAMES_CELL) + by;
                    if gx < eff_w && gy < eff_h {
                        effective.data[gy * eff_w + gx] = e;
                    }
                }
            }
            stats.total_samples += tile.stats.total_samples;
            stats.accumulated_samples += tile.stats.accumulated_samples;
            stats.rejected_samples += tile.stats.rejected_samples;
            stats.masked_samples += tile.stats.masked_samples;
            stats.saturated_samples += tile.stats.saturated_samples;
            stats.out_of_bounds_samples += tile.stats.out_of_bounds_samples;
        }
    }

    if cfg.kernel.debias_deposit {
        let total: u64 = (0..channels).map(|c| rgb[c].data.len() as u64).sum();
        log::info!(
            "deposit placed at the pixel centre by a plane fit on {:.1}% of pixels; \
             the rest kept the weighted mean",
            100.0 * fitted as f32 / total.max(1) as f32
        );
    }

    let holes_filled = fill_holes(&mut rgb, &weight, channels);
    if fit_curvature {
        log::info!("guarded curvature changed {curved} channel-pixels from their plane estimates");
    }
    stats.unsupported_pixels = holes_filled as u64;
    stats.min_effective_frames = effective.data.iter().copied().fold(f32::INFINITY, f32::min);
    if !stats.min_effective_frames.is_finite() {
        stats.min_effective_frames = 0.0;
    }
    stats.mean_effective_frames = effective.mean();

    Ok(ReconstructionProduct {
        channels,
        width: out_w,
        height: out_h,
        rgb,
        weight,
        count,
        effective_frames: effective,
        effective_frames_cell: EFFECTIVE_FRAMES_CELL,
        rejected,
        holes_filled,
        stats,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn clipped_trace_identifies_shared_limiter_and_failed_channel() {
        let mut moments=[Moments {w:1.,wx:0.5,wxx:1.,wyy:1.,wv:1.,wvx:1.25,
            vmin:0.,vmax:2.,..Default::default()};3];
        moments[1].clipped=true;moments[1].vmin=0.875;
        let applied=bounded_clipped_plane(&moments).unwrap();
        assert_eq!(applied,[0.875;3]);
        let mut trace=Vec::new();
        write_clipped_trace(&mut trace,2,4,&moments,true,applied).unwrap();
        for (c,row) in String::from_utf8(trace).unwrap().lines().enumerate() {
            let fields:Vec<_>=row.split(',').collect();
            assert_eq!(fields.len(),22);
            assert_eq!(fields[4],"pass");
            assert_eq!(fields[18].parse::<f32>().unwrap(),if c==1 {0.25} else {1.});
            assert_eq!(fields[19],"0.25");assert_eq!(fields[21],"0.875");
        }
        moments[0].wx=2.;
        assert!(bounded_clipped_plane(&moments).is_none());
        let mut trace=Vec::new();
        write_clipped_trace(&mut trace,2,4,&moments,true,[1.;3]).unwrap();
        for (c,row) in String::from_utf8(trace).unwrap().lines().enumerate() {
            let fields:Vec<_>=row.split(',').collect();
            assert_eq!(fields[4],if c==0 {"centroid"} else {"pass"});
            assert_eq!(fields[19],"0");
        }
    }
    #[test]
    fn clipped_taper_matches_global_distance_oracle() {
        let (w,h)=(13,11);
        for points in [vec![],vec![0],vec![w*h-1],vec![3*w+4,7*w+8],(0..w*h).collect()] {
            let mut mask=vec![false;w*h];
            for &p in &points {mask[p]=true;}
            let candidate: [Plane<f32>;3]=std::array::from_fn(|c| {
                let mut plane=Plane::new(w,h);
                for (p,v) in plane.data.iter_mut().enumerate() {*v=(p+1) as f32*0.003+c as f32*0.11;}
                plane
            });
            let production=std::array::from_fn(|c|Plane::filled(w,h,0.07+c as f32*0.03));
            let (actual,fraction)=taper_clipped_rgb(candidate.clone(),&production,&mask);
            for p in 0..w*h {
                let d=points.iter().map(|&q| {
                    let dx=(p%w) as f64-(q%w) as f64;
                    let dy=(p/w) as f64-(q/w) as f64;
                    (dx*dx+dy*dy).sqrt()
                }).fold(f64::INFINITY,f64::min);
                let t=((d-1.)/2.).clamp(0.,1.);
                let alpha=t*t*(3.-2.*t);
                assert_eq!(fraction.data[p],alpha as f32);
                for c in 0..3 {
                    let base=f64::from(production[c].data[p]);
                    let expected=if alpha==0. {production[c].data[p]} else if alpha==1. {candidate[c].data[p]} else {
                        (base+alpha*(f64::from(candidate[c].data[p])-base)) as f32
                    };
                    assert_eq!(actual[c].data[p].to_bits(),expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn tapered_selection_is_tile_invariant_and_rejects_cropped_halos() {
        let mut f=fixture(128,96,&[(0.,0.),(0.5,0.),(0.,0.5),(0.5,0.5)]);
        for frame in &mut f.frames {
            let mut data=vec![0.1;128*96];
            data[63*128+63]=1.;
            frame.samples=SamplePlane::from_normalised(128,96,data);
        }
        let kernels=[&f.kernels;3];
        let mut cfg=ReconstructionConfig {scale:1.,tile:64,..Default::default()};
        let a=reconstruct_tapered_support_selection_experiment(&inputs(&f),&cfg,kernels).unwrap();
        assert!(a.candidate_fraction.data.contains(&0.));
        assert!(a.candidate_fraction.data.iter().any(|&v|v>0. && v<1.));
        cfg.tile=128;
        let b=reconstruct_tapered_support_selection_experiment(&inputs(&f),&cfg,kernels).unwrap();
        for c in 0..3 {assert_eq!(a.rgb[c].data,b.rgb[c].data);}
        assert_eq!(a.candidate_fraction,b.candidate_fraction);
        assert_eq!(a.candidate_selection,b.candidate_selection);
        cfg.roi=Some((64,64,32,32));
        assert!(reconstruct_tapered_support_selection_experiment(&inputs(&f),&cfg,kernels).is_err());
        cfg.roi=None;cfg.scale=2.;
        assert!(reconstruct_tapered_support_selection_experiment(&inputs(&f),&cfg,kernels).is_err());
    }
    use super::*;
    use sr_core::cfa::CfaPattern;
    use sr_core::config::{KernelConfig, RobustnessConfig};
    use sr_core::frame::NoiseSource;
    use sr_core::geometry::GlobalTransform;
    use sr_core::samples::{DefectMask, SamplePlane};

    fn noise() -> NoiseModel {
        NoiseModel::new(1e-6, 1e-8, NoiseSource::Measured)
    }

    #[test]
    fn mono_feather_preserves_shared_scene_gain_and_tile_boundaries() {
        use sr_core::geometry::Rect;
        for constant in [true, false] {
            let shifts = [(0.,0.),(12.,0.)];
            let mut f = fixture_with(32,24,&shifts,true);
            for (i, (frame, &(dx,dy))) in f.frames.iter_mut().zip(&shifts).enumerate() {
                *frame = mosaic_of(32,24,dx,dy,|x,y| {
                    let scene = if constant {0.4} else {0.2+0.3*x/64.+0.2*y/64.};
                    scene / if i == 0 {1.} else {2.}
                });
                frame.cfa = CfaPattern::MONO;
            }
            f.photometry[1].gain = [2.;3];
            let sky:Vec<_> = f.frames.iter().map(sky_of).collect();
            let cfg = ReconstructionConfig {scale:1.,..Default::default()};
            let rect = Rect::new(0,0,44,24);
            let plain = reconstruct_mono_tile(&inputs(&f),&cfg,(0.,0.),rect,&sky).unwrap();
            let zero = reconstruct_mono_tile_feathered(&inputs(&f),&cfg,(0.,0.),rect,&sky,0.).unwrap();
            assert_eq!(plain.values,zero.values);
            assert_eq!(plain.weight,zero.weight);
            assert_eq!(plain.count,zero.count);
            let full = reconstruct_mono_tile_feathered(&inputs(&f),&cfg,(0.,0.),rect,&sky,0.25).unwrap();
            assert_eq!(full.count,plain.count,"feathering must not invent or discard native samples");
            assert!(full.weight.iter().zip(&plain.weight).any(|(a,b)|a<b));
            assert!(full.weight.iter().all(|v|*v>0.),"true detector edge pixels remain supported");
            for y in 0..24 {for x in 0..44 {
                if constant || ((4..40).contains(&x) && (4..20).contains(&y)) {
                    let expected = if constant {0.4} else {0.2+0.3*x as f32/64.+0.2*y as f32/64.};
                    assert!((full.values[y*44+x]-expected).abs()<1e-5,"gain/scene bias at {x},{y}");
                }
            }}
            for y in (0..24).step_by(7) {for x in (0..44).step_by(7) {
                let part=Rect::new(x,y,7.min(44-x),7.min(24-y));
                let out=reconstruct_mono_tile_feathered(&inputs(&f),&cfg,(0.,0.),part,&sky,0.25).unwrap();
                for dy in 0..part.height {for dx in 0..part.width {
                    let a=dy*part.width+dx;let b=(y+dy)*44+x+dx;
                    assert_eq!(out.values[a],full.values[b]);
                    assert_eq!(out.weight[a],full.weight[b]);
                }}
            }}
        }
    }

    #[test]
    fn mono_feather_cropped_windows_use_true_detector_edges() {
        use sr_core::geometry::Rect;
        let full=fixture_with(96,64,&[(0.,0.)],true);
        let sky:Vec<_>=full.frames.iter().map(sky_of).collect();
        let cfg=ReconstructionConfig {scale:1.,..Default::default()};
        for rect in [Rect::new(0,0,16,16),Rect::new(14,20,16,16),Rect::new(80,48,16,16)] {
            let expected=reconstruct_mono_tile_feathered(&inputs(&full),&cfg,(0.,0.),rect,&sky,0.25).unwrap();
            let cx=rect.x.saturating_sub(4);let cy=rect.y.saturating_sub(4);
            let w=(rect.x+rect.width+4).min(96)-cx;
            let h=(rect.y+rect.height+4).min(64)-cy;
            let mut cropped=fixture_with(w,h,&[(0.,0.)],true);
            let samples=&full.frames[0].samples;
            let data=(cy..cy+h).flat_map(|y| (cx..cx+w).map(move |x|samples.value(x,y))).collect();
            cropped.frames[0].samples=SamplePlane::from_normalised(w,h,data);
            cropped.frames[0].metadata.full_width=96;
            cropped.frames[0].metadata.full_height=64;
            cropped.frames[0].metadata.crop=(cx,cy,w,h);
            cropped.warps[0]=WarpField::global_only(GlobalTransform::translation(cx as f32,cy as f32));
            cropped.kernels=full.kernels.clone();
            let actual=reconstruct_mono_tile_feathered(&inputs(&cropped),&cfg,(0.,0.),rect,&sky,0.25).unwrap();
            assert_eq!(actual.values,expected.values);
            assert_eq!(actual.weight,expected.weight);
            assert_eq!(actual.count,expected.count);
        }
        for invalid in [-0.1,0.251,f32::NAN,f32::INFINITY] {
            assert!(reconstruct_mono_tile_feathered(&inputs(&full),&cfg,(0.,0.),Rect::new(0,0,8,8),&sky,invalid).is_err());
        }
        let mut invalid=fixture_with(16,16,&[(0.,0.)],true);
        invalid.frames[0].metadata.full_width=20;
        invalid.frames[0].metadata.crop.0=usize::MAX;
        assert!(reconstruct_mono_tile_feathered(&inputs(&invalid),&cfg,(0.,0.),Rect::new(0,0,8,8),&sky,0.1).is_err());
    }

    #[test]
    fn mono_tile_uses_per_frame_sky_noise_without_changing_legacy_weights() {
        use sr_core::geometry::Rect;
        let mut f = fixture_with(16, 16, &[(0., 0.), (0., 0.)], true);
        for (frame, value) in f.frames.iter_mut().zip([0.2, 0.8]) {
            frame.samples = SamplePlane::from_normalised(16, 16, vec![value; 256]);
        }
        let mut cfg = ReconstructionConfig { scale: 1., ..Default::default() };
        cfg.kernel.debias_deposit = false;
        let legacy = reconstruct(&inputs(&f), &cfg).unwrap();
        // Sky is fixed independently of these constant sample values. Nonzero
        // alpha proves the deposit does not accidentally weight each sample by
        // its own intensity instead of the caller's fixed per-frame sky.
        let sky = [[0.1; 3]; 2];
        for alpha in [0., 1e-5] {
            f.frames[0].noise = NoiseModel::new(alpha, 1e-6, NoiseSource::Measured);
            f.frames[1].noise = NoiseModel::new(alpha, 4e-6, NoiseSource::Measured);
            let variances = f.frames.iter().map(|fr| fr.noise.variance(0.1)).collect::<Vec<_>>();
            let expected = (0.2 / variances[0] + 0.8 / variances[1])
                / (1. / variances[0] + 1. / variances[1]);
            let out = reconstruct_mono_tile(&inputs(&f), &cfg, (0., 0.), Rect::new(4,4,8,8), &sky).unwrap();
            for value in out.values { assert!((value - expected).abs() < 2e-6, "{value} vs {expected}"); }
            let unchanged = reconstruct(&inputs(&f), &cfg).unwrap();
            assert_eq!(legacy.rgb[0].data, unchanged.rgb[0].data);
            assert_eq!(legacy.weight[0].data, unchanged.weight[0].data);
        }
        for bad in [NoiseModel::new(-1., 1e-6, NoiseSource::Measured),
            NoiseModel::new(0., 0., NoiseSource::Measured),
            NoiseModel::new(0., f32::NAN, NoiseSource::Measured)] {
            f.frames[0].noise = bad;
            assert!(reconstruct_mono_tile(&inputs(&f), &cfg, (0.,0.), Rect::new(4,4,8,8), &sky).is_err());
        }
    }

    #[test]
    fn mono_tile_assembly_preserves_production_estimators_and_diagnostics() {
        use sr_core::geometry::Rect;
        let shifts: Vec<_> = (0..16).map(|i| ((i % 4) as f32 / 4., (i / 4) as f32 / 4.)).collect();
        let mut f = fixture_with(32, 32, &shifts, true);
        for (frame, &(dx, dy)) in f.frames.iter_mut().zip(&shifts) {
            *frame = mosaic_of(32, 32, dx, dy, |x, y|
                0.1 + 1.4 * (-((x-16.).powi(2)+(y-16.).powi(2))/8.).exp());
            frame.cfa = CfaPattern::MONO;
        }
        let sky: Vec<_> = f.frames.iter().map(sky_of).collect();
        for scale in [1., 2.] { for fit in [false, true] { for clipped in [false, true] {
            let mut cfg = ReconstructionConfig { scale, ..Default::default() };
            cfg.kernel.debias_deposit = fit;
            cfg.kernel.fit_clipped = Some(clipped);
            let product = reconstruct(&inputs(&f), &cfg).unwrap();
            let w = product.rgb[0].width;
            for step in [7, 19] {
                for y in (0..w).step_by(step) { for x in (0..w).step_by(step) {
                    let rect = Rect::new(x, y, step.min(w-x), step.min(w-y));
                    let out = reconstruct_mono_tile(&inputs(&f), &cfg, (0., 0.), rect, &sky).unwrap();
                    for dy in 0..rect.height { for dx in 0..rect.width {
                        let a = dy * rect.width + dx;
                        let b = (y + dy) * w + x + dx;
                        assert_eq!(out.weight[a], product.weight[0].data[b]);
                        assert_eq!(out.count[a], product.count[0].data[b]);
                        if out.weight[a] > 0. {
                            assert_eq!(out.values[a], product.rgb[0].data[b], "scale {scale}, fit {fit}, clipped {clipped}, {x},{y}");
                        } else { assert!(out.values[a].is_nan()); }
                    }}
                }}
            }
        }}}
    }

    #[test]
    fn mono_tile_explicit_origin_supports_negative_and_outside_reference_grids() {
        use sr_core::geometry::Rect;
        let f = fixture_with(32, 32, &[(0., 0.), (0.5, 0.5)], true);
        let sky: Vec<_> = f.frames.iter().map(sky_of).collect();
        let cfg = ReconstructionConfig { scale: 2., ..Default::default() };
        let normal = reconstruct_mono_tile(&inputs(&f), &cfg, (0.,0.), Rect::new(0,0,32,32), &sky).unwrap();
        let shifted = reconstruct_mono_tile(&inputs(&f), &cfg, (-8.,-8.), Rect::new(16,16,32,32), &sky).unwrap();
        assert_eq!(normal.values, shifted.values);
        assert_eq!(normal.weight, shifted.weight);
        let empty = reconstruct_mono_tile(&inputs(&f), &cfg, (-100.,-100.), Rect::new(0,0,7,9), &sky).unwrap();
        assert!(empty.values.iter().all(|v| v.is_nan()));
        assert!(empty.weight.iter().all(|v| *v == 0.));
        assert!(empty.count.iter().all(|v| *v == 0.));
        assert_eq!(empty.stats.unsupported_pixels, 63);
        let cropped = reconstruct_mono_tile(&inputs(&f), &cfg, (4.,5.), Rect::new(0,0,16,14), &sky).unwrap();
        let roi_cfg = ReconstructionConfig { roi: Some((4,5,8,7)), ..cfg };
        let product = reconstruct(&inputs(&f), &roi_cfg).unwrap();
        assert_eq!(cropped.values.as_slice(), &product.rgb[0].data[..]);
        assert_eq!(cropped.weight.as_slice(), &product.weight[0].data[..]);
    }

    #[test]
    fn mono_tile_downscale_local_warp_preserves_contributions_at_tile_edges() {
        use sr_core::geometry::{DeformationField, Rect};
        let mut f = fixture_with(96, 96, &[(0., 0.)], true);
        f.warps[0].global = GlobalTransform::similarity(0., 0.25, 0., 0.);
        let mut local = DeformationField::zeros((0.,0.), 32., 2, 2);
        local.u.fill([6., -4.]);
        local.conf.fill(1.);
        f.warps[0].local = Some(local);
        let sky: Vec<_> = f.frames.iter().map(sky_of).collect();
        let cfg = ReconstructionConfig { scale: 1., ..Default::default() };
        let full = reconstruct_mono_tile(&inputs(&f), &cfg, (0.,0.), Rect::new(0,0,32,32), &sky).unwrap();
        for y in (0..32).step_by(4) { for x in (0..32).step_by(4) {
            let out = reconstruct_mono_tile(&inputs(&f), &cfg, (0.,0.), Rect::new(x,y,4,4), &sky).unwrap();
            for dy in 0..4 { for dx in 0..4 {
                let a = dy * 4 + dx;
                let b = (y+dy)*32+x+dx;
                assert_eq!(out.count[a], full.count[b], "omitted native contribution at {},{}", x+dx,y+dy);
                assert_eq!(out.weight[a], full.weight[b]);
                assert_eq!(out.values[a].to_bits(), full.values[b].to_bits());
            }}
        }}
    }

    #[test]
    fn mono_tile_rejects_invalid_inputs_before_accumulation() {
        use sr_core::geometry::Rect;
        let f = fixture_with(16,16,&[(0.,0.)],true);
        let sky: Vec<_> = f.frames.iter().map(sky_of).collect();
        let cfg = ReconstructionConfig { scale: 1., ..Default::default() };
        let rect = Rect::new(0,0,8,8);
        let mut input = inputs(&f);
        input.reference = 1;
        assert!(reconstruct_mono_tile(&input,&cfg,(0.,0.),rect,&sky).is_err());
        input.reference = 0;
        for bad in [Rect::new(0,0,0,1), Rect::new(usize::MAX,0,8,8), Rect::new(0,0,4097,1)] {
            assert!(reconstruct_mono_tile(&input,&cfg,(0.,0.),bad,&sky).is_err());
        }
        assert!(reconstruct_mono_tile(&input,&cfg,(f32::NAN,0.),rect,&sky).is_err());
        assert!(reconstruct_mono_tile(&input,&cfg,(0.,0.),rect,&[]).is_err());
        assert!(reconstruct_mono_tile(&input,&cfg,(0.,0.),rect,&[[f32::NAN;3]]).is_err());
        for scale in [0., 9., f32::NAN] {
            let bad = ReconstructionConfig {scale,..cfg.clone()};
            assert!(reconstruct_mono_tile(&input,&bad,(0.,0.),rect,&sky).is_err());
        }
        let bad = ReconstructionConfig {backend:Backend::RgbMeanBaseline,..cfg.clone()};
        assert!(reconstruct_mono_tile(&input,&bad,(0.,0.),rect,&sky).is_err());
        input.frame_weight = &[];
        assert!(reconstruct_mono_tile(&input,&cfg,(0.,0.),rect,&sky).is_err());
        let mut bad = fixture_with(16,16,&[(0.,0.)],true);
        bad.frames[0].width = 15;
        assert!(reconstruct_mono_tile(&inputs(&bad),&cfg,(0.,0.),rect,&sky).is_err());
        let color = fixture(16,16,&[(0.,0.)]);
        assert!(reconstruct_mono_tile(&inputs(&color),&cfg,(0.,0.),rect,&sky).is_err());
    }

    /// Mosaic a continuous scene function at sensor sites, with a sub-pixel
    /// offset, so that a burst genuinely samples different phases.
    fn mosaic_of<F: Fn(f32, f32) -> f32>(
        w: usize,
        h: usize,
        dx: f32,
        dy: f32,
        f: F,
    ) -> RawFrame {
        let mut data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = f(x as f32 + dx, y as f32 + dy);
            }
        }
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_normalised(w, h, data),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            noise: noise(),
            metadata: Default::default(),
        }
    }

    struct Fixture {
        frames: Vec<RawFrame>,
        warps: Vec<WarpField>,
        photometry: Vec<PhotometricMatch>,
        fweight: Vec<f32>,
        robust: RobustnessMaps,
        kernels: KernelField,
    }

    /// A burst of a grey ramp, shifted over the four half-pixel phases.
    fn fixture(w: usize, h: usize, shifts: &[(f32, f32)]) -> Fixture {
        fixture_with(w, h, shifts, false)
    }

    fn fixture_with(w: usize, h: usize, shifts: &[(f32, f32)], mono: bool) -> Fixture {
        let scene = |x: f32, y: f32| 0.2 + 0.3 * (x / 64.0) + 0.2 * (y / 64.0);
        let frames: Vec<RawFrame> = shifts
            .iter()
            .map(|&(dx, dy)| {
                let mut f = mosaic_of(w, h, dx, dy, scene);
                if mono {
                    // The same scene, sampled at every site instead of through
                    // a mosaic. The generator already writes one value per
                    // site, so relabelling the pattern is the whole change.
                    f.cfa = CfaPattern::MONO;
                }
                f
            })
            .collect();
        // Frame content sampled at x + dx corresponds to reference x + dx, so
        // the map back onto the reference subtracts the shift.
        let warps: Vec<WarpField> = shifts
            .iter()
            .map(|&(dx, dy)| WarpField::global_only(GlobalTransform::translation(dx, dy)))
            .collect();
        let n = shifts.len();
        let robust = RobustnessMaps {
            width: w / 2,
            height: h / 2,
            maps: Vec::new(),
            rejected_fraction: vec![0.0; n],
            consensus_luma: None,
        };
        let kernels = KernelField::isotropic(w / 2, h / 2, 0.25, 2.0, 2);
        Fixture {
            frames,
            warps,
            photometry: vec![PhotometricMatch::IDENTITY; n],
            fweight: vec![1.0; n],
            robust,
            kernels,
        }
    }

    fn inputs<'a>(f: &'a Fixture) -> MergeInputs<'a> {
        MergeInputs {
            frames: &f.frames,
            warps: &f.warps,
            reference: 0,
            photometry: &f.photometry,
            noise: noise(),
            robustness: &f.robust,
            kernels: &f.kernels,
            frame_weight: &f.fweight,
            lucky: None,
            chroma: RadialChroma::identity(),
        }
    }

    #[test]
    fn clipped_selection_preserves_trace_union_and_candidate_elsewhere() {
        let shifts:Vec<_>=(0..64).map(|i|((i%8) as f32/4.-0.875,(i/8) as f32/4.-0.875)).collect();
        for all_colours in [false,true] {
            let mut f=fixture(32,32,&shifts);
            for (frame,&(dx,dy)) in f.frames.iter_mut().zip(&shifts) {
                let data=(0..32*32).map(|p| {
                    let (x,y)=((p%32) as f32+dx,(p/32) as f32+dy);
                    let c=frame.cfa.color_at(p%32,p/32).index();
                    let amplitude=if all_colours || c==0 {3.} else {0.5};
                    (0.03+amplitude*(-((x-16.2).powi(2)+(y-15.7).powi(2))/3.).exp()
                        +0.15*(-((x-22.1).powi(2)+(y-16.4).powi(2))/2.).exp()).min(1.)
                }).collect();
                frame.samples=SamplePlane::from_normalised(32,32,data);
            }
            f.kernels=KernelField::isotropic(16,16,if all_colours {0.06} else {0.8},2.,2);
            let kernels=[0.1,0.25,0.375].map(|v|KernelField::isotropic(16,16,v,2.,2));
            for fit_clipped in [false,true] {
                let mut cfg=ReconstructionConfig {scale:1.,..Default::default()};
                cfg.kernel.fit_clipped=Some(fit_clipped);
                let protected=reconstruct_clipped_support_selection_experiment(&inputs(&f),&cfg,[&kernels[0],&kernels[1],&kernels[2]]).unwrap();
                let production=reconstruct(&inputs(&f),&cfg).unwrap();
                let mut clipped_trace=Vec::new();
                let traced=reconstruct_with_clipped_trace(&inputs(&f),&cfg,&mut clipped_trace).unwrap();
                for c in 0..3 {
                    assert_eq!(production.rgb[c].data,traced.rgb[c].data);
                    assert_eq!(production.weight[c].data,traced.weight[c].data);
                    assert_eq!(production.count[c].data,traced.count[c].data);
                }
                let text=String::from_utf8(clipped_trace).unwrap();
                assert!(text.lines().count()>30);
                for row in text.lines().skip(1) {
                    let fields:Vec<_>=row.split(',').collect();
                    let x=fields[0].parse::<usize>().unwrap();let y=fields[1].parse::<usize>().unwrap();
                    let c=fields[2].parse::<usize>().unwrap();
                    assert_eq!(fields[21].parse::<f32>().unwrap(),production.rgb[c].data[y*32+x]);
                }
                let mut curved=cfg.clone();curved.kernel.fit_curvature=Some(true);
                let candidate=reconstruct_support_selection_experiment(&inputs(&f),&curved,[&kernels[0],&kernels[1],&kernels[2]]).unwrap();
                let mut union=vec![false;32*32];
                for kernel in [&f.kernels,&kernels[0],&kernels[1],&kernels[2]] {
                    let mut pass=inputs(&f);pass.kernels=kernel;
                    let mut trace=Vec::new();
                    reconstruct_with_fit_trace(&pass,&cfg,&mut trace).unwrap();
                    for row in String::from_utf8(trace).unwrap().lines().skip(1) {
                        let fields:Vec<_>=row.split(',').collect();
                        if fields[3]=="clipped_footprint" {
                            union[fields[1].parse::<usize>().unwrap()*32+fields[0].parse::<usize>().unwrap()]=true;
                        }
                    }
                }
                assert!(protected.preserved_pixels>10 && protected.preserved_pixels<32*32);
                assert_eq!(protected.preserved_pixels,union.iter().filter(|&&v|v).count());
                for (p,&preserve) in union.iter().enumerate() {
                    assert_eq!(protected.output.selection.data[p],if preserve {3} else {candidate.selection.data[p]});
                    for c in 0..3 {
                        let (rgb,weight,count)=if preserve {(&production.rgb,&production.weight,&production.count)} else {(&candidate.rgb,&candidate.weight,&candidate.count)};
                        assert_eq!(protected.output.rgb[c].data[p],rgb[c].data[p]);
                        assert_eq!(protected.output.weight[c].data[p],weight[c].data[p]);
                        assert_eq!(protected.output.count[c].data[p],count[c].data[p]);
                    }
                }
            }
        }
    }

    #[test]
    fn support_selection_preserves_clipped_rgb_footprints_for_both_clipping_policies() {
        let shifts:Vec<_>=(0..64).map(|i|((i%8) as f32/4.-0.875,(i/8) as f32/4.-0.875)).collect();
        for all_colours in [false,true] {
            let mut f=fixture(32,32,&shifts);
            for (frame,&(dx,dy)) in f.frames.iter_mut().zip(&shifts) {
                let mut data=Vec::with_capacity(32*32);
                for y in 0..32 { for x in 0..32 {
                    let c=frame.cfa.color_at(x,y).index();
                    let amplitude=if all_colours || c==0 {3.0} else {0.5};
                    let radius=(x as f32+dx-16.2).powi(2)+(y as f32+dy-15.7).powi(2);
                    data.push((0.03+amplitude*(-radius/3.).exp()).min(1.));
                }}
                frame.samples=SamplePlane::from_normalised(32,32,data);
            }
            let kernels=[0.1,0.25,0.375].map(|v|KernelField::isotropic(16,16,v,2.,2));
            f.kernels=kernels[0].clone();
            for clipped_fit in [false,true] {
                let mut cfg=ReconstructionConfig {scale:1.,..Default::default()};
                cfg.kernel.fit_curvature=Some(true);cfg.kernel.fit_clipped=Some(clipped_fit);
                let mut trace=Vec::new();
                let narrow=reconstruct_variance_blend_experiment(&inputs(&f),&cfg,&mut trace).unwrap();
                let selected=reconstruct_support_selection_experiment(&inputs(&f),&cfg,[&kernels[0],&kernels[1],&kernels[2]]).unwrap();
                let mut clipped_pixels=0;
                for row in String::from_utf8(trace).unwrap().lines().skip(1) {
                    let fields:Vec<_>=row.split(',').collect();
                    if fields[2]!="0" || fields[3]!="clipped_footprint" {continue;}
                    let p=fields[1].parse::<usize>().unwrap()*32+fields[0].parse::<usize>().unwrap();
                    clipped_pixels+=1;
                    assert_eq!(selected.selection.data[p],0);
                    for c in 0..3 {
                        assert_eq!(selected.rgb[c].data[p],narrow.rgb[c].data[p]);
                        assert_eq!(selected.weight[c].data[p],narrow.weight[c].data[p]);
                    }
                }
                assert!(clipped_pixels>10,"fixture must exercise a clipped footprint");
                assert!(narrow.stats.saturated_samples>0);
            }
        }
    }

    #[test]
    fn support_selection_is_tile_invariant_and_preserves_identical_kernel_control() {
        let f=fixture(128,96,&[(0.,0.),(0.5,0.),(0.,0.5),(0.5,0.5)]);
        let mut cfg=ReconstructionConfig {scale:1.,tile:64,..Default::default()};
        cfg.kernel.fit_curvature=Some(true);
        let kernels=[&f.kernels,&f.kernels,&f.kernels];
        let a=reconstruct_support_selection_experiment(&inputs(&f),&cfg,kernels).unwrap();
        let protected_a=reconstruct_clipped_support_selection_experiment(&inputs(&f),&cfg,kernels).unwrap();
        cfg.tile=128;
        let b=reconstruct_support_selection_experiment(&inputs(&f),&cfg,kernels).unwrap();
        let protected_b=reconstruct_clipped_support_selection_experiment(&inputs(&f),&cfg,kernels).unwrap();
        let striped=reconstruct_support_selection_striped_experiment(&inputs(&f),&cfg,kernels,17).unwrap();
        let mut sink=std::io::sink();
        let first=reconstruct_variance_blend_experiment(&inputs(&f),&cfg,&mut sink).unwrap();
        for c in 0..3 {
            assert_eq!(a.rgb[c].data,b.rgb[c].data);
            assert_eq!(a.weight[c].data,b.weight[c].data);
            assert_eq!(a.rgb[c].data,first.rgb[c].data);
            assert_eq!(a.rgb[c].data,striped.rgb[c].data);
            assert_eq!(a.weight[c].data,striped.weight[c].data);
            assert_eq!(protected_a.output.rgb[c].data,protected_b.output.rgb[c].data);
            assert_eq!(protected_a.output.weight[c].data,protected_b.output.weight[c].data);
            assert_eq!(protected_a.output.count[c].data,protected_b.output.count[c].data);
        }
        assert_eq!(a.selection,b.selection);
        assert!(a.selection.data.iter().all(|&v|v==0));
        assert_eq!(a.fallback_pixels,b.fallback_pixels);
        assert_eq!(a.selection,striped.selection);
        assert_eq!(a.fallback_pixels,striped.fallback_pixels);
        assert_eq!(a.source_stats.len(),3);
        assert_eq!(protected_a.output.selection,protected_b.output.selection);
        assert_eq!(protected_a.preserved_pixels,protected_b.preserved_pixels);
    }

    #[test]
    fn striped_selection_rejects_unsupported_pixels_instead_of_seaming_hole_fill() {
        let f=fixture(32,32,&[(0.,0.)]);
        let narrow=KernelField::isotropic(16,16,0.01,0.5,2);
        let mut cfg=ReconstructionConfig {scale:1.,..Default::default()};
        cfg.kernel.fit_curvature=Some(true);cfg.kernel.radius=0.5;
        let result=reconstruct_support_selection_striped_experiment(&inputs(&f),&cfg,[&narrow;3],8);
        assert!(matches!(result,Err(SrError::Reconstruction(ref message)) if message.contains("unsupported pixels")));
    }

    #[test]
    fn fit_trace_preserves_reconstruction_and_reports_each_channel() {
        let f = fixture(32, 32, &[(0.,0.),(0.5,0.),(0.,0.5),(0.5,0.5)]);
        let mut cfg = ReconstructionConfig { scale: 1., ..Default::default() };
        cfg.kernel.fit_curvature = Some(true);
        let baseline = reconstruct(&inputs(&f), &cfg).unwrap();
        let mut trace = Vec::new();
        let traced = reconstruct_with_fit_trace(&inputs(&f), &cfg, &mut trace).unwrap();
        for c in 0..3 {
            assert_eq!(baseline.rgb[c].data, traced.rgb[c].data);
            assert_eq!(baseline.weight[c].data, traced.weight[c].data);
        }
        let text = String::from_utf8(trace).unwrap();
        assert_eq!(text.lines().count(), 1 + 32*32*3);
        assert!(text.lines().skip(1).any(|row| row.split(',').nth(4) == Some("effective_samples")));
        assert!(text.lines().skip(1).all(|row| row.split(',').count() == 14));
    }

    #[test]
    fn photometry_and_obstructions_follow_the_sky_across_a_meridian_flip() {
        let mut f = fixture_with(64, 64, &[(0.0, 0.0), (0.0, 0.0)], true);
        f.frames[0] = mosaic_of(64, 64, 0.0, 0.0, |_, _| 0.3);
        f.frames[1] = mosaic_of(64, 64, 0.0, 0.0, |x, _| {
            let rx = 63.0 - x;
            0.3 + 0.05 * (2.0 * rx / 64.0 - 1.0)
                + if rx < 12.0 { 0.4 } else { 0.0 }
        });
        for frame in &mut f.frames { frame.cfa = CfaPattern::MONO; }
        f.warps[1] = WarpField::global_only(GlobalTransform {
            m: [-1.0, 0.0, 63.0, 0.0, -1.0, 63.0],
        });
        let map = &mut f.photometry[1];
        let n = map.field[0].len();
        for row in &mut map.field[0] {
            for (x, value) in row.iter_mut().enumerate() {
                *value = -0.05 * (2.0 * x as f32 / (n - 1) as f32 - 1.0);
            }
        }
        for row in &mut map.blocked {
            for value in row.iter_mut().take(n / 4) { *value = true; }
        }
        let cfg = ReconstructionConfig {
            scale: 1.0,
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let p = reconstruct(&inputs(&f), &cfg).unwrap();
        for y in 8..56 {
            for x in 8..56 {
                assert!((p.rgb[0].data[y * 64 + x] - 0.3).abs() < 1e-4,
                    "reference-grid correction misplaced at {x},{y}: {}", p.rgb[0].data[y * 64 + x]);
            }
        }
    }

    #[test]
    fn a_clipped_channel_switches_all_colours_to_the_same_estimator() {
        let mut f = fixture(32, 32, &[(0.0, 0.0), (0.25, 0.25)]);
        for frame in &mut f.frames {
            let mut data: Vec<f32> = (0..32 * 32).map(|i| frame.value(i % 32, i / 32)).collect();
            data[16 * 32 + 16] = 1.0;
            frame.samples = SamplePlane::from_normalised(32, 32, data);
        }
        let cfg = ReconstructionConfig {
            scale: 2.0,
            kernel: KernelConfig { fit_clipped: Some(false), ..Default::default() },
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let mut mean_cfg = cfg.clone();
        mean_cfg.kernel.debias_deposit = false;
        let fitted = reconstruct(&inputs(&f), &cfg).unwrap();
        let mean = reconstruct(&inputs(&f), &mean_cfg).unwrap();
        let mut curved_cfg = cfg.clone();
        curved_cfg.kernel.fit_curvature = Some(true);
        let curved = reconstruct(&inputs(&f), &curved_cfg).unwrap();
        let p = 32 * 64 + 32;
        for c in 0..3 {
            assert!((fitted.rgb[c].data[p] - mean.rgb[c].data[p]).abs() < 1e-6,
                "channel {c} used a different estimator beside a clipped red sample");
            assert_eq!(curved.rgb[c].data[p], mean.rgb[c].data[p],
                "curvature changed a clipped footprint in channel {c}");
        }
    }

    #[test]
    fn curvature_needs_support_in_every_colour() {
        let shifts: Vec<_> = (0..64).map(|i| ((i % 8) as f32/4.-0.875, (i / 8) as f32/4.-0.875)).collect();
        let mut f = fixture(32, 32, &shifts);
        for (frame, &(dx, dy)) in f.frames.iter_mut().zip(&shifts) {
            *frame = mosaic_of(32, 32, dx, dy, |x, y|
                0.1 + 0.6*(-((x-16.).powi(2)+(y-16.).powi(2))/8.).exp());
        }
        let base = ReconstructionConfig {
            scale: 1., robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let mut cfg = base.clone();
        cfg.kernel.fit_curvature = Some(true);
        let plain = reconstruct(&inputs(&f), &base).unwrap();
        let curved = reconstruct(&inputs(&f), &cfg).unwrap();
        let p = 16*32+16;
        assert!((plain.rgb[1].data[p]-curved.rgb[1].data[p]).abs() > 0.001,
            "control must actually apply curvature");
        // Leave blue supported by only one exposure. R/G still constrain a
        // quadratic, but changing only those channels would change the PSF by colour.
        for frame in f.frames.iter_mut().skip(1) {
            for y in (1..32).step_by(2) { for x in (1..32).step_by(2) {
                frame.defects.set(y*32+x);
            }}
        }
        let plain = reconstruct(&inputs(&f), &base).unwrap();
        let curved = reconstruct(&inputs(&f), &cfg).unwrap();
        for c in 0..3 { assert_eq!(plain.rgb[c].data, curved.rgb[c].data); }
    }

    #[test]
    fn sky_noise_sampling_covers_every_bayer_phase() {
        let expected = [0.03, 0.07, 0.12];
        for pattern in [CfaPattern::RGGB, CfaPattern::BGGR, CfaPattern::GRBG, CfaPattern::GBRG] {
            for (w, h) in [(768, 768), (767, 769)] {
                let mut frame = mosaic_of(w, h, 0., 0., |_, _| 0.1);
                frame.cfa = pattern;
                let data = (0..w*h).map(|i| expected[pattern.color_at(i%w, i/w).index()]).collect();
                frame.samples = SamplePlane::from_normalised(w, h, data);
                let measured = sky_of(&frame);
                for c in 0..3 {
                    assert!((measured[c]-expected[c]).abs() < 1e-6,
                        "sky sampling missed channel {c} for {pattern:?} at {w}x{h}: {measured:?}");
                }
            }
        }
    }

    #[test]
    fn bounded_clipped_planes_reduce_phase_colour_without_extrapolation() {
        let mut moments = [Moments::default(); 3];
        for (c, offset) in [-0.2, 0., 0.2].into_iter().enumerate() {
            for iy in -2..=2 { for ix in -2..=2 {
                let x = ix as f32 * 0.3 + offset;
                let v = (0.85 + 0.6*x).min(1.);
                moments[c].add(1., x, iy as f32*0.3, v);
                moments[c].clipped |= v >= 1.;
            }}
        }
        let means = moments.map(|m| m.wv/m.w);
        let corrected = bounded_clipped_plane(&moments).unwrap();
        let spread = |v: [f32; 3]| v.into_iter().fold(f32::NEG_INFINITY, f32::max)
            - v.into_iter().fold(f32::INFINITY, f32::min);
        assert!(spread(corrected) < spread(means)*0.25);
        for c in 0..3 {
            assert!(corrected[c] >= moments[c].vmin && corrected[c] <= moments[c].vmax);
        }
        // A missing channel must leave every channel on the common fallback.
        moments[2] = Moments::default();
        assert!(bounded_clipped_plane(&moments).is_none());
    }

    #[test]
    fn bounded_clipped_fit_is_automatic_for_cfa_only() {
        // Deliberately unequal subpixel support. A balanced fine phase grid
        // makes the mean already centred and cannot prove the switch matters.
        let shifts: Vec<_> = (0..16).map(|i|
            ((i % 4) as f32/4., (i / 4) as f32/4.)).collect();
        let mut f = fixture(32, 32, &shifts);
        for (frame, &(dx, dy)) in f.frames.iter_mut().zip(&shifts) {
            *frame = mosaic_of(32, 32, dx, dy, |x, y|
                0.1 + 1.5*(-((x-16.).powi(2)+(y-16.).powi(2))/2.).exp());
        }
        let auto = ReconstructionConfig { scale: 1.,
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default() };
        let mut on = auto.clone(); on.kernel.fit_clipped = Some(true);
        let mut off = auto.clone(); off.kernel.fit_clipped = Some(false);
        let a = reconstruct(&inputs(&f), &auto).unwrap();
        let b = reconstruct(&inputs(&f), &on).unwrap();
        let c = reconstruct(&inputs(&f), &off).unwrap();
        for channel in 0..3 { assert_eq!(a.rgb[channel].data, b.rgb[channel].data); }
        assert!(a.rgb[0].data.iter().zip(&c.rgb[0].data).any(|(x,y)| (x-y).abs() > 1e-4));
        for frame in &mut f.frames { frame.cfa = CfaPattern::MONO; }
        let a = reconstruct(&inputs(&f), &auto).unwrap();
        let b = reconstruct(&inputs(&f), &off).unwrap();
        assert_eq!(a.rgb[0].data, b.rgb[0].data);
    }

    #[test]
    fn bounded_clipped_planes_share_the_most_restrictive_limit() {
        let mut moments = [Moments::default(); 3];
        for (c, offset) in [0.2, 0.1, 0.].into_iter().enumerate() {
            for iy in -2..=2 { for ix in -1..=1 {
                let x = ix as f32*0.1 + offset;
                moments[c].add(1., x, iy as f32*0.1, 0.5+x);
            }}
            moments[c].clipped = true;
        }
        let corrected = bounded_clipped_plane(&moments).unwrap();
        // The red footprint needs extrapolation to reach x=0. Its bound limits
        // the blend to one half in all colours, including the green channel.
        assert!((corrected[0]-0.6).abs() < 1e-6);
        assert!((corrected[1]-0.55).abs() < 1e-6);
        assert!((corrected[2]-0.5).abs() < 1e-6);
    }

    #[test]
    fn curvature_is_automatic_only_for_monochrome_bursts() {
        let shifts: Vec<_> = (0..16).map(|i| ((i%4) as f32/4., (i/4) as f32/4.)).collect();
        let mut f = fixture(32, 32, &shifts);
        for (frame, &(dx, dy)) in f.frames.iter_mut().zip(&shifts) {
            *frame = mosaic_of(32, 32, dx, dy, |x, y|
                0.1+0.6*(-((x-16.).powi(2)+(y-16.).powi(2))/8.).exp());
        }
        let auto = ReconstructionConfig {
            scale: 1., robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let mut off = auto.clone(); off.kernel.fit_curvature = Some(false);
        let a = reconstruct(&inputs(&f), &auto).unwrap();
        let b = reconstruct(&inputs(&f), &off).unwrap();
        for c in 0..3 { assert_eq!(a.rgb[c].data, b.rgb[c].data); }
        for frame in &mut f.frames { frame.cfa = CfaPattern::MONO; }
        let mut on = auto.clone(); on.kernel.fit_curvature = Some(true);
        let a = reconstruct(&inputs(&f), &auto).unwrap();
        let b = reconstruct(&inputs(&f), &on).unwrap();
        let c = reconstruct(&inputs(&f), &off).unwrap();
        assert_eq!(a.rgb[0].data, b.rgb[0].data);
        assert!((a.rgb[0].data[16*32+16]-c.rgb[0].data[16*32+16]).abs() > 0.001);
    }

    #[test]
    fn output_kernel_gives_equal_weight_to_opposite_samples() {
        // A spatially changing guide must not make the estimate prefer one
        // side of a pixel. Compare equal impulses one pixel either side of
        // the same output location, keeping the guide fixed in both runs.
        let mut f = fixture_with(32, 32, &[(0.0, 0.0)], true);
        let mut guide = Plane::<f32>::new(16, 16);
        for y in 0..16 {
            for x in 8..16 {
                guide.data[y * 16 + x] = 1.0;
            }
        }
        f.kernels = KernelField::from_reference(&guide, 0.001, 1, &KernelConfig::default());
        let cfg = ReconstructionConfig {
            scale: 1.0,
            kernel: KernelConfig { debias_deposit: false, ..Default::default() },
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let mut values = Vec::new();
        // At x=12 the support crosses the change in kernel shape.
        for impulse_x in [11, 13] {
            let mut frame = mosaic_of(32, 32, 0.0, 0.0, |x, y| {
                if x == impulse_x as f32 && y == 16.0 { 0.6 } else { 0.2 }
            });
            frame.cfa = CfaPattern::MONO;
            f.frames[0] = frame;
            values.push(reconstruct(&inputs(&f), &cfg).unwrap().rgb[0].data[16 * 32 + 12]);
        }
        assert!((values[0] - values[1]).abs() < 1e-6,
            "opposite samples acquired different kernel weights: {values:?}");
    }

    #[test]
    fn known_sensor_colour_shift_recovers_blue_signal_across_a_flip() {
        // Isolate the blue reconstruction with exact geometry. Adjusting the
        // global translation below is equivalent, for blue only, to moving its
        // detector coordinates before the warp. Other channels are not scored.
        // No registration, inferred chromatic model, or robust guide is involved.
        let shifts: Vec<_> = (0..32).map(|i| {
            (((i * 13 % 31) as f32 / 31.0) * 2.0,
             ((i * 19 % 29) as f32 / 29.0) * 2.0)
        }).collect();
        let cfg = ReconstructionConfig {
            scale: 1.0,
            backend: Backend::CfaDrizzle,
            tile: 64,
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        // Numerical pixel integration; a sensor-fixed tail changes direction
        // on the sky after the flip but retains the analytic flux centroid.
        let profile = |x: f32, y: f32, sign: f32, tail: f32| {
            let mut sum = 0.0;
            for iy in 0..8 {
                for ix in 0..8 {
                    let dx = x + (ix as f32 + 0.5) / 8.0 - 0.5 - 31.27;
                    let dy = y + (iy as f32 + 0.5) / 8.0 - 0.5 - 30.63;
                    let g = |cy: f32| (-0.5 * (dx*dx/2.25 + (dy-cy).powi(2)/2.7225)).exp();
                    sum += (1.0-tail)*g(-sign*tail*2.0) + tail*g(sign*(1.0-tail)*2.0);
                }
            }
            0.03 + 0.35*sum/64.0
        };
        for (tail, injected) in [(0.0, 0.20), (0.2, 0.20), (0.0, 0.0), (0.2, 0.0)] {
            let mut f = fixture(64, 64, &shifts);
            for (i, &(dx, dy)) in shifts.iter().enumerate() {
                let sign = if i < 16 { 1.0 } else { -1.0 };
                let origin = if sign > 0.0 { 0.0 } else { 63.0 };
                f.warps[i] = WarpField::global_only(GlobalTransform {
                    m: [sign, 0.0, origin+dx, 0.0, sign, origin+dy],
                });
                // Apparent blue image centroid is displaced -injected sensor Y.
                f.frames[i] = mosaic_of(64, 64, 0.0, 0.0, |x, y| {
                    profile(sign*x+origin+dx, sign*(y+injected)+origin+dy, sign, tail)
                });
            }
            let uncorrected = reconstruct(&inputs(&f), &cfg).unwrap();
            for (i, warp) in f.warps.iter_mut().enumerate() {
                warp.global.m[5] += if i < 16 { 0.20 } else { -0.20 };
            }
            let corrected = reconstruct(&inputs(&f), &cfg).unwrap();
            let mut mse = [0.0f64; 2];
            for y in 23..40 {
                for x in 23..40 {
                    let truth = (profile(x as f32,y as f32,1.0,tail)
                        + profile(x as f32,y as f32,-1.0,tail))*0.5;
                    for (j, image) in [&uncorrected, &corrected].iter().enumerate() {
                        mse[j] += (image.rgb[2].data[y*64+x]-truth).powi(2) as f64 / 289.0;
                    }
                }
            }
            eprintln!("blue shift tail={tail}, injected={injected}: truth MSE {mse:?}");
            if injected > 0.0 {
                assert!(mse[1] < mse[0], "known correction degraded blue truth: tail={tail}, {mse:?}");
            } else {
                assert!(mse[1] > mse[0], "unnecessary correction improved truth: tail={tail}, {mse:?}");
            }
        }
    }

    #[test]
    fn a_monochrome_burst_reconstructs_one_channel() {
        let f = fixture_with(64, 64, &[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)], true);
        let cfg = ReconstructionConfig {
            scale: 2.0,
            backend: Backend::CfaDrizzle,
            tile: 64,
            kernel: KernelConfig { radius: 2.0, ..Default::default() },
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let p = reconstruct(&inputs(&f), &cfg).unwrap();
        assert_eq!(p.channels, 1);
        assert_eq!((p.width, p.height), (128, 128));

        // The two channels that do not exist cost nothing. A monochrome
        // reconstruction of a 26 MP sensor at 2x would otherwise carry two and
        // a half gigabytes of zeros.
        assert!(p.rgb[1].data.is_empty(), "an absent channel was allocated");
        assert!(p.rgb[2].data.is_empty(), "an absent channel was allocated");
        assert!(p.weight[1].data.is_empty());

        let scene = |x: f32, y: f32| 0.2 + 0.3 * (x / 64.0) + 0.2 * (y / 64.0);
        let mut worst = 0.0f32;
        for oy in 16..112 {
            for ox in 16..112 {
                assert!(
                    p.weight[0].data[oy * 128 + ox] > 0.0,
                    "interior pixel ({ox}, {oy}) had no support"
                );
                let rx = (ox as f32 + 0.5) / 2.0 - 0.5;
                let ry = (oy as f32 + 0.5) / 2.0 - 0.5;
                worst = worst.max((p.rgb[0].data[oy * 128 + ox] - scene(rx, ry)).abs());
            }
        }
        // Tighter than the mosaiced case, and it should be: every site
        // contributes to the one channel rather than one site in four.
        assert!(worst < 0.005, "ramp reconstruction error {worst}");
    }


    #[test]
    fn reconstructs_a_ramp_at_2x_without_holes() {
        let f = fixture(64, 64, &[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]);
        let cfg = ReconstructionConfig {
            scale: 2.0,
            backend: Backend::CfaDrizzle,
            tile: 64,
            kernel: KernelConfig { radius: 2.0, ..Default::default() },
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let p = reconstruct(&inputs(&f), &cfg).unwrap();
        assert_eq!((p.width, p.height), (128, 128));

        // Unsupported pixels are legitimate at the very border, where the
        // outermost output samples sit beyond the last sensor site of every
        // frame. They must not appear anywhere else.
        assert!(
            p.holes_filled * 100 < p.width * p.height * 3,
            "{} unsupported pixels is more than a border's worth",
            p.holes_filled
        );
        for c in 0..3 {
            for y in 4..124 {
                for x in 4..124 {
                    assert!(
                        p.weight[c].data[y * 128 + x] > 0.0,
                        "interior pixel ({x}, {y}) channel {c} had no support"
                    );
                }
            }
        }

        // Every channel should reproduce the underlying ramp. Check well inside
        // the border, where the kernel footprint is complete.
        let scene = |x: f32, y: f32| 0.2 + 0.3 * (x / 64.0) + 0.2 * (y / 64.0);
        let mut worst = 0.0f32;
        for oy in 16..112 {
            for ox in 16..112 {
                let rx = (ox as f32 + 0.5) / 2.0 - 0.5;
                let ry = (oy as f32 + 0.5) / 2.0 - 0.5;
                let want = scene(rx, ry);
                for c in 0..3 {
                    worst = worst.max((p.rgb[c].data[oy * 128 + ox] - want).abs());
                }
            }
        }
        assert!(worst < 0.01, "ramp reconstruction error {worst}");
    }

    #[test]
    fn averaging_reduces_noise_by_root_n() {
        // A flat field plus independent noise per frame: the merge should
        // deliver close to the sqrt(N) improvement.
        let n = 16usize;
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            let u = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
            (u as f32 - 0.5) * 3.4641016 // uniform with unit variance
        };
        let sigma = 0.02f32;
        let shifts: Vec<(f32, f32)> = (0..n)
            .map(|i| ((i % 4) as f32 * 0.25, (i / 4) as f32 * 0.25))
            .collect();
        let mut f = fixture(64, 64, &shifts);
        for fr in f.frames.iter_mut() {
            let (w, h) = (fr.width, fr.height);
            let vals: Vec<f32> = (0..w * h).map(|_| 0.4 + sigma * rnd()).collect();
            fr.samples = SamplePlane::from_normalised(w, h, vals);
        }
        let cfg = ReconstructionConfig {
            scale: 2.0,
            backend: Backend::CfaDrizzle,
            tile: 64,
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let p = reconstruct(&inputs(&f), &cfg).unwrap();

        let mut acc = 0.0f64;
        let mut cnt = 0u64;
        for oy in 20..108 {
            for ox in 20..108 {
                let d = (p.rgb[1].data[oy * 128 + ox] - 0.4) as f64;
                acc += d * d;
                cnt += 1;
            }
        }
        let out_sigma = (acc / cnt as f64).sqrt() as f32;
        assert!(
            out_sigma < sigma / 3.0,
            "merged sigma {out_sigma} vs input {sigma}: expected a large reduction"
        );
    }

    #[test]
    fn effective_frame_count_tracks_the_burst_size() {
        let shifts: Vec<(f32, f32)> = (0..8)
            .map(|i| ((i % 4) as f32 * 0.25, (i / 4) as f32 * 0.25))
            .collect();
        let f = fixture(64, 64, &shifts);
        let cfg = ReconstructionConfig {
            scale: 2.0,
            backend: Backend::CfaDrizzle,
            tile: 64,
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let p = reconstruct(&inputs(&f), &cfg).unwrap();
        // Away from the border every block should see all eight frames.
        let e = &p.effective_frames;
        let mid = e.data[(e.height / 2) * e.width + e.width / 2];
        assert!(mid > 7.0 && mid <= 8.01, "effective frames {mid}");
    }

    /// Samples on a known linear ramp, placed where the caller says.
    fn moments_on_ramp(at: &[(f32, f32)], gx: f32, gy: f32, at_centre: f32) -> Moments {
        let mut m = Moments::default();
        for &(dx, dy) in at {
            // `dx` is pixel-minus-sample, as the deposit accumulates it, so the
            // sample sits at -(dx, dy) from the pixel centre.
            let v = at_centre + gx * -dx + gy * -dy;
            m.add(1.0, dx, dy, v);
        }
        m
    }

    #[test]
    fn a_pixel_with_a_clipped_sample_keeps_the_mean() {
        // The same well-determined scatter as below, but one of its samples was
        // at the ceiling. A plane through a clipped profile is not a model of
        // anything, and the fit must decline rather than extrapolate past the
        // sensor.
        let at = [(0.3f32, 0.1f32), (0.1, -0.15), (0.25, 0.2), (0.15, 0.0)];
        let mut m = moments_on_ramp(&at, 2.0, -1.0, 0.75);
        assert!(plane_fit(&m).is_some());
        m.clipped = true;
        assert!(plane_fit(&m).is_none(), "fitted a plane through a saturated sample");
    }

    #[test]
    fn a_plane_fit_recovers_the_value_the_mean_misses() {
        // Four samples, all off to one side, on a ramp. Their mean is the value
        // where they sit, not the value at the pixel; a plane through them
        // reads the pixel correctly.
        let at = [(0.3f32, 0.1f32), (0.1, -0.15), (0.25, 0.2), (0.15, 0.0)];
        let m = moments_on_ramp(&at, 2.0, -1.0, 0.75);
        let mean = m.wv / m.w;
        let fitted = plane_fit(&m).expect("a plane is determined by four scattered samples");
        assert!(
            (mean - 0.75).abs() > 0.3,
            "the mean was supposed to be wrong here, and was {mean}"
        );
        assert!((fitted - 0.75).abs() < 1e-4, "plane fit gave {fitted}, wanted 0.75");
    }

    #[test]
    fn samples_far_off_centre_keep_the_mean() {
        // The guard that the first version got wrong. Samples all half a pixel
        // to one side determine a plane perfectly well, and evaluating it back
        // at the pixel is extrapolation into a place nothing was measured. The
        // first guard scaled with the gradient, so it never fired, and a
        // handful of runaway pixels took the exposure normalisation with them.
        let at = [(0.9f32, 0.1f32), (0.8, -0.2), (0.95, 0.3), (0.85, 0.0)];
        let m = moments_on_ramp(&at, 2.0, -1.0, 0.75);
        assert!(plane_fit(&m).is_none(), "extrapolated from samples 0.9 px away");
    }

    #[test]
    fn samples_in_a_line_keep_the_mean() {
        // A plane is not determined by collinear samples. Fitting one anyway
        // extrapolates along the direction nothing constrains.
        let at = [(0.2f32, 0.2f32), (0.4, 0.4), (-0.3, -0.3), (0.1, 0.1)];
        let m = moments_on_ramp(&at, 1.0, 1.0, 0.5);
        assert!(plane_fit(&m).is_none(), "fitted a plane to a line");
    }

    #[test]
    fn one_sample_keeps_the_mean() {
        let m = moments_on_ramp(&[(0.3, -0.2)], 1.0, 1.0, 0.5);
        assert!(plane_fit(&m).is_none());
    }

    #[test]
    fn symmetric_samples_leave_the_mean_alone() {
        // The case the mean is already right for: nothing should move.
        let at = [(0.3f32, 0.0f32), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3)];
        let m = moments_on_ramp(&at, 3.0, -2.0, 0.6);
        let mean = m.wv / m.w;
        let fitted = plane_fit(&m).expect("a plane is determined");
        assert!((mean - 0.6).abs() < 1e-5, "mean {mean}");
        assert!((fitted - mean).abs() < 1e-5, "fit moved a symmetric pixel to {fitted}");
    }

    #[test]
    fn roi_reconstructs_the_same_pixels_as_the_full_frame() {
        let f = fixture(64, 64, &[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]);
        let base = ReconstructionConfig {
            scale: 2.0,
            backend: Backend::CfaDrizzle,
            tile: 32,
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let full = reconstruct(&inputs(&f), &base).unwrap();
        let cropped = reconstruct(
            &inputs(&f),
            &ReconstructionConfig { roi: Some((16, 16, 16, 16)), ..base.clone() },
        )
        .unwrap();
        assert_eq!((cropped.width, cropped.height), (32, 32));
        let mut worst = 0.0f32;
        for y in 0..32 {
            for x in 0..32 {
                let a = cropped.rgb[1].data[y * 32 + x];
                let b = full.rgb[1].data[(32 + y) * 128 + (32 + x)];
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-5, "roi disagreed with the full frame by {worst}");
    }

    #[test]
    fn tile_size_does_not_change_the_result() {
        let f = fixture(64, 64, &[(0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (0.25, 0.25)]);
        let base = ReconstructionConfig {
            scale: 2.0,
            backend: Backend::CfaDrizzle,
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let a = reconstruct(&inputs(&f), &ReconstructionConfig { tile: 64, ..base.clone() }).unwrap();
        let b = reconstruct(&inputs(&f), &ReconstructionConfig { tile: 96, ..base.clone() }).unwrap();
        let mut worst = 0.0f32;
        for c in 0..3 {
            for i in 0..a.rgb[c].data.len() {
                worst = worst.max((a.rgb[c].data[i] - b.rgb[c].data[i]).abs());
            }
        }
        assert!(worst < 1e-5, "tiling changed the result by {worst}");
    }

    #[test]
    fn chromatic_correction_moves_only_its_own_channel() {
        // A deliberate 1% red magnification, corrected during the merge. Green
        // must be untouched, and red must land where green already is.
        let f = fixture(64, 64, &[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]);
        let cfg = ReconstructionConfig {
            scale: 2.0,
            backend: Backend::CfaDrizzle,
            tile: 64,
            robustness: RobustnessConfig { enabled: false, ..Default::default() },
            ..Default::default()
        };
        let plain = reconstruct(&inputs(&f), &cfg).unwrap();

        let mut with_ca = inputs(&f);
        with_ca.chroma = RadialChroma {
            centre: (32.0, 32.0),
            norm: 45.0,
            coeff: [[0.01, 0.0], [0.0, 0.0], [0.0, 0.0]],
        };
        let corrected = reconstruct(&with_ca, &cfg).unwrap();

        // Green is identical.
        let mut green_diff = 0.0f32;
        for i in 0..plain.rgb[1].data.len() {
            green_diff = green_diff.max((plain.rgb[1].data[i] - corrected.rgb[1].data[i]).abs());
        }
        assert!(green_diff < 1e-6, "green moved: {green_diff}");

        // Red changed, and only away from the centre of the correction.
        let mut red_diff = 0.0f32;
        for i in 0..plain.rgb[0].data.len() {
            red_diff = red_diff.max((plain.rgb[0].data[i] - corrected.rgb[0].data[i]).abs());
        }
        assert!(red_diff > 1e-4, "red did not move: {red_diff}");
    }

    #[test]
    fn rejects_an_impossible_scale() {
        let f = fixture(32, 32, &[(0.0, 0.0)]);
        let cfg = ReconstructionConfig { scale: 20.0, ..Default::default() };
        assert!(reconstruct(&inputs(&f), &cfg).is_err());
    }

    #[test]
    fn rejects_an_out_of_range_roi() {
        let f = fixture(32, 32, &[(0.0, 0.0)]);
        let cfg = ReconstructionConfig { roi: Some((16, 16, 32, 32)), ..Default::default() };
        assert!(reconstruct(&inputs(&f), &cfg).is_err());
    }
}
