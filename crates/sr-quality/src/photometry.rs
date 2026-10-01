//! Photometric matching between frames of a burst.
//!
//! Registration answers where a frame's samples go. This answers what they are
//! worth once they get there, which is a separate question and is not settled
//! by the exposure metadata: a burst can be shot at one shutter speed and one
//! gain from first frame to last and still change brightness, because the
//! *scene* changed. Cloud crosses the sun; the sky brightens as the moon rises
//! or the target sinks into worse air.
//!
//! Left uncorrected this is not a cosmetic problem. Every stage that compares a
//! frame against the reference — motion rejection above all — reads a shifted
//! level as scene change, and reads it *everywhere*, so an entire frame is
//! distrusted for having been taken later in the night. On the 36-frame
//! IC 1340 burst the sky rises 14% from first frame to last, which is nearly
//! three times the tolerance the robustness model allows, and the merge
//! discarded a third of the samples it was given.
//!
//! The correction is one affine map per colour channel, `gain * v + offset`,
//! estimated from the pixels rather than from the header. Affine because both
//! mechanisms occur: haze and airmass scale the scattered light already there,
//! while moonlight and light pollution add to it, and a burst usually has some
//! of each.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sr_core::frame::RawFrame;
use sr_core::geometry::WarpField;
use sr_core::star::Star;

/// A per-channel affine map carrying one frame onto another's photometric
/// scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhotometricMatch {
    pub gain: [f32; 3],
    /// Experimental mono-mosaic log gain in the same normalized reference
    /// coordinates as `apply_at`: [constant, u, v, u², uv, v²]. None preserves
    /// the ordinary scalar path exactly. This is not an additive sky model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_gain: Option<[[f32; 6]; 3]>,
    pub offset: [f32; 3],
    /// Additive offset across the frame, per channel, on a `FIELD` x `FIELD`
    /// grid read bilinearly in normalised reference coordinates where the frame spans
    /// -1 to 1 on each axis. Zero-mean: whatever constant part it had is in
    /// `offset`.
    ///
    /// Transparency multiplies and is the same everywhere in a frame; sky
    /// brightness adds and is not. Moonlight, a town on the horizon and dawn
    /// all put a gradient across the frame that moves between exposures, and a
    /// single offset per frame can only match their average -- which leaves
    /// the difference to be rejected as if it were an outlier, or averaged in
    /// as if it were signal. So the gain stays global and the offset is
    /// allowed to vary.
    ///
    /// This was a plane, and a plane was not enough. Moonlight falls off from
    /// where the moon is; over a wide field that is curved, and a plane takes
    /// the tilt out and leaves the curvature in. On a 96-frame set where 89
    /// frames needed a gradient, the merge was rejecting 21% of all samples,
    /// and what survived carried the leftover curvature as background
    /// structure.
    ///
    /// What makes a grid safe here, where fitting a background model to a
    /// single frame would not be, is that this is fitted to the *difference*
    /// between two frames of the same sky. Real structure is in both and
    /// cancels; only what changed between the exposures survives to be fitted.
    #[serde(default = "no_field")]
    pub field: [[[f32; FIELD]; FIELD]; 3],
    /// Cells of the frame where something stood between the sensor and the
    /// sky. Samples there are not measurements of the scene and are given to
    /// nobody: the merge skips them and the consensus does not count them.
    ///
    /// Found against the burst: a block where this frame, on the reference's
    /// scale, is a quarter darker than the burst typically is there. Not
    /// against the frame's own sky, because a dark nebula is dark in every
    /// frame and a tree is dark in this one. The last six frames of one
    /// test burst were shot into trees at dawn, thirty to eighty percent dark over
    /// half the frame, and the additive field, asked to explain the residual,
    /// explained the trees as a gradient and lifted them back into the stack.
    /// Obstructed blocks take no part in the gain, the pedestal or the field,
    /// and are carried here so that nothing downstream uses them either.
    #[serde(default = "no_blocks")]
    pub blocked: [[bool; FIELD]; FIELD],
}

/// No cell blocked, for the serde default and for maps that have none.
fn no_blocks() -> [[bool; FIELD]; FIELD] {
    [[false; FIELD]; FIELD]
}

/// Fraction of the expected sky in the source frame (scaled by its gain, but
/// excluding the additive matching pedestal) that must be missing to mark an
/// obstruction. An arbitrary reference sky offset must not hide a shadow.
///
/// The steepest sky gradient in one 96-frame test burst spans seventeen percent of a
/// frame, which puts its dark side eight or nine below the frame's own level;
/// the trees in its last frames took thirty to eighty at their thickest and
/// twelve to thirty where the branches thinned, and a quarter left the thin
/// part in. Fifteen takes it and leaves a gradient of thirty percent across
/// the frame alone, which is more than moonlight makes.
const OBSTRUCTION_FRACTION: f32 = 0.15;

/// And by this many times the burst's own scatter at that block, so that a
/// block the dither carries across an edge of the subject is not read as
/// obstructed for taking a different median in every frame.
const OBSTRUCTION_SIGMAS: f32 = 5.0;

/// Grid resolution of the additive field, per axis.
///
/// Eight is a compromise found by measurement rather than argument. Two would
/// be the bilinear surface the plane already was. Much finer and each cell
/// starts to cover less sky than the structures we must not absorb, and the
/// guard has to do all the work; at eight, one cell of a full-frame sensor is
/// still several hundred pixels across.
pub const FIELD: usize = 8;

/// A field of zeros, for the serde default and for maps that have none.
fn no_field() -> [[[f32; FIELD]; FIELD]; 3] {
    [[[0.0; FIELD]; FIELD]; 3]
}

impl Default for PhotometricMatch {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl PhotometricMatch {
    pub const IDENTITY: PhotometricMatch = PhotometricMatch {
        gain: [1.0; 3],
        log_gain: None,
        offset: [0.0; 3],
        field: [[[0.0; FIELD]; FIELD]; 3],
        blocked: [[false; FIELD]; FIELD],
    };

    /// The map a plain exposure ratio implies: a gain and no pedestal.
    pub fn from_exposure(scale: f32) -> Self {
        PhotometricMatch {
            gain: [scale; 3],
            log_gain: None,
            offset: [0.0; 3],
            field: [[[0.0; FIELD]; FIELD]; 3],
            blocked: [[false; FIELD]; FIELD],
        }
    }

    /// Whether the frame is obstructed at a place, in the same normalised
    /// coordinates as `apply_at`. Nearest cell: an obstruction has an edge and
    /// reading it bilinearly would invent a half-obstructed fringe.
    #[inline]
    pub fn blocked_at(&self, u: f32, w: f32) -> bool {
        let last = (FIELD - 1) as f32;
        let gx = ((u + 1.0) * 0.5 * last).round().clamp(0.0, last) as usize;
        let gy = ((w + 1.0) * 0.5 * last).round().clamp(0.0, last) as usize;
        self.blocked[gy][gx]
    }

    /// The share of the frame's cells that are obstructed.
    pub fn blocked_fraction(&self) -> f32 {
        let n = self.blocked.iter().flatten().filter(|&&b| b).count();
        n as f32 / (FIELD * FIELD) as f32
    }

    #[inline]
    pub fn apply(&self, channel: usize, v: f32) -> f32 {
        self.apply_at(channel, v, 0.0, 0.0)
    }

    /// The map at a place in the frame, in normalised coordinates spanning
    /// -1 to 1. `apply` is this at the centre, which is where a frame with a
    /// zero-mean field is closest to having none.
    #[inline]
    pub fn apply_at(&self, channel: usize, v: f32, u: f32, w: f32) -> f32 {
        let c = channel.min(2);
        self.gain_at(c, u, w) * v + self.offset[c] + self.field_at(c, u, w)
    }

    /// Local multiplicative correction, evaluated analytically without clipping.
    #[inline]
    pub fn gain_at(&self, channel: usize, u: f32, v: f32) -> f32 {
        let c=channel.min(2);
        let Some(field)=&self.log_gain else { return self.gain[c]; };
        let [constant,x,y,xx,xy,yy]=field[c].map(f64::from);
        let (u,v)=(f64::from(u),f64::from(v));
        (f64::from(self.gain[c]) * (constant+x*u+y*v+xx*u*u+xy*u*v+yy*v*v).exp()) as f32
    }

    /// Conservative positive finite gain range on a rectangle. Interval bounds
    /// also cover interior quadratic extrema; corner samples alone do not.
    pub fn gain_bounds(&self, channel:usize, u:[f32;2], v:[f32;2]) -> Option<(f32,f32)> {
        let c=channel.min(2);let gain=self.gain[c];
        if !gain.is_finite() || gain<=0. || u.iter().chain(&v).any(|x|!x.is_finite())
            || u[0]>u[1] || v[0]>v[1] { return None; }
        let Some(field)=&self.log_gain else {return Some((gain,gain));};
        let coefficients=field[c].map(f64::from);
        if coefficients.iter().any(|x|!x.is_finite()) {return None;}
        let u=u.map(f64::from);let v=v.map(f64::from);
        let square=|a:[f64;2]| [if a[0]<=0. && a[1]>=0. {0.} else {a[0].powi(2).min(a[1].powi(2))},a[0].powi(2).max(a[1].powi(2))];
        let products=[u[0]*v[0],u[0]*v[1],u[1]*v[0],u[1]*v[1]];
        let cross=[products.into_iter().fold(f64::INFINITY,f64::min),products.into_iter().fold(f64::NEG_INFINITY,f64::max)];
        let mut range=[0.;2];
        for (coefficient,basis) in coefficients.into_iter().zip([[1.,1.],u,v,square(u),cross,square(v)]) {
            let a=coefficient*basis[0];let b=coefficient*basis[1];range[0]+=a.min(b);range[1]+=a.max(b);
        }
        let bounds=range.map(|x|(f64::from(gain)*x.exp()) as f32);
        (bounds.iter().all(|g|g.is_finite() && *g>0.)).then_some((bounds[0],bounds[1]))
    }

    /// The additive field at a place, read bilinearly.
    ///
    /// Bilinear and not nearest: a field read in steps would put its own grid
    /// into the background, which is the thing it exists to remove.
    #[inline]
    pub fn field_at(&self, channel: usize, u: f32, w: f32) -> f32 {
        let f = &self.field[channel.min(2)];
        let last = (FIELD - 1) as f32;
        let gx = ((u + 1.0) * 0.5 * last).clamp(0.0, last);
        let gy = ((w + 1.0) * 0.5 * last).clamp(0.0, last);
        let (x0, y0) = (gx as usize, gy as usize);
        let (x1, y1) = ((x0 + 1).min(FIELD - 1), (y0 + 1).min(FIELD - 1));
        let (tx, ty) = (gx - x0 as f32, gy - y0 as f32);
        let top = f[y0][x0] + (f[y0][x1] - f[y0][x0]) * tx;
        let bottom = f[y1][x0] + (f[y1][x1] - f[y1][x0]) * tx;
        top + (bottom - top) * ty
    }

    /// Whether this map varies across the frame at all.
    #[inline]
    pub fn varies_across_frame(&self) -> bool {
        self.field.iter().flatten().flatten().any(|v| *v != 0.0)
            || self.log_gain.is_some_and(|f|f.iter().any(|c|c[1..].iter().any(|v|*v!=0.)))
    }

    /// Largest correction the field applies anywhere in the frame, for
    /// reporting: the span from its lowest cell to its highest.
    pub fn field_amplitude(&self, channel: usize) -> f32 {
        let f = &self.field[channel.min(2)];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for v in f.iter().flatten() {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        if lo.is_finite() && hi.is_finite() { hi - lo } else { 0.0 }
    }

    #[inline]
    pub fn gain_of(&self, channel: usize) -> f32 {
        self.gain[channel.min(2)]
    }

    /// Undo this map: the frame that `self` carries onto a scale, expressed the
    /// other way round. `None` when a gain has collapsed.
    pub fn inverse(&self) -> Option<PhotometricMatch> {
        // Dividing the additive field by an exponential is outside this model.
        if self.log_gain.is_some() { return None; }
        let mut out = PhotometricMatch::IDENTITY;
        out.blocked = self.blocked;
        for c in 0..3 {
            if self.gain[c].abs() < 1e-6 {
                return None;
            }
            out.gain[c] = 1.0 / self.gain[c];
            out.offset[c] = -self.offset[c] / self.gain[c];
            for y in 0..FIELD {
                for x in 0..FIELD {
                    out.field[c][y][x] = -self.field[c][y][x] / self.gain[c];
                }
            }
        }
        Some(out)
    }

    /// `self` applied after `other`. Legacy scalar-gain maps only: composition
    /// of spatial gains with additive fields is not closed in this model.
    /// Panics when either map has an experimental log-gain field.
    pub fn compose(&self, other: &PhotometricMatch) -> PhotometricMatch {
        assert!(self.log_gain.is_none() && other.log_gain.is_none(),
            "PhotometricMatch::compose does not support spatial log gain");
        let mut out = PhotometricMatch::IDENTITY;
        for (row, (a, b)) in out.blocked.iter_mut().zip(self.blocked.iter().zip(other.blocked.iter())) {
            for (cell, (x, y)) in row.iter_mut().zip(a.iter().zip(b.iter())) {
                *cell = *x || *y;
            }
        }
        for c in 0..3 {
            out.gain[c] = self.gain[c] * other.gain[c];
            out.offset[c] = self.gain[c] * other.offset[c] + self.offset[c];
            // Both fields act on the same normalised coordinates, and the outer
            // map scales whatever the inner one produced.
            for y in 0..FIELD {
                for x in 0..FIELD {
                    out.field[c][y][x] =
                        self.gain[c] * other.field[c][y][x] + self.field[c][y][x];
                }
            }
        }
        out
    }

    /// How far this map moves a mid-range value, as a fraction of that value.
    /// A single number for reporting; zero means the frame needed nothing.
    pub fn relative_change(&self, level: f32) -> f32 {
        (0..3)
            .map(|c| ((self.apply(c, level) - level) / level.max(1e-6)).abs())
            .fold(0.0f32, f32::max)
    }
}

/// Target size of one measurement block, in sensor pixels.
///
/// The fit compares block medians rather than whole-frame quantiles, and the
/// reason is that a block median has a *place*. Two frames of the same scene
/// agree block for block, so the regression is over matched pairs and its slope
/// is the illumination change. Sorted whole-frame quantiles have no such
/// pairing: on a scene with little tonal range their spread is mostly noise,
/// the two frames' noise is independent, and the slope through them is
/// meaningless — while looking perfectly well-conditioned.
const BLOCK: usize = 128;
const MIN_BLOCKS: usize = 8;
const MAX_BLOCKS: usize = 32;

/// Conservative limits for gains inferred from sky blocks. Matched stellar
/// fluxes can support larger changes (different exposures or electronic gains)
/// when independent stars agree; see `supported_stellar_gain`.
const GAIN_LIMITS: (f32, f32) = (0.5, 2.0);

/// How much better than the uncertainty of a block median the spread of block
/// medians has to be before a gain is identifiable from them.
const SPREAD_MARGIN: f32 = 20.0;

/// How fine the fitted field may be, in cells per axis.
///
/// The stored grid is `FIELD` across, but a field fitted that finely does not
/// only follow the sky: it follows whatever the block medians did, and what
/// they did includes their own scatter. Each frame then carries a little
/// invented structure at the cell scale, and ninety-six frames of it average
/// into a lumpy sky -- on a 96-frame burst the mid-scale structure fell by a tenth with the
/// field removed altogether, and rejection with it, because frames were being
/// made to disagree with a burst carrying everyone else's invented lumps.
///
/// Four cells is still a curved surface -- the thing a plane could not do, and
/// the reason the plane was replaced -- while being too coarse to
/// reproduce anything at the scale the eye reads as blotches.
const FIELD_MAX_CELLS: usize = 4;

/// How far across the frame the field must reach, in units of the uncertainty
/// of one block median, before it is believed to be a gradient rather than a
/// surface through the scatter of the blocks.
const FIELD_MIN_AMPLITUDE: f32 = 12.0;

/// Blocks needed before a field across the frame is fitted at all.
const FIELD_MIN_BLOCKS: usize = 24;

/// Blocks a field cell needs before its own median is believed. Cells below
/// this are filled from their neighbours instead.
const FIELD_MIN_PER_CELL: usize = 4;

/// The fraction of the residual spread a field must leave behind. Above this
/// it has explained too little to be believed, and is discarded rather than
/// imposed.
const FIELD_MUST_EXPLAIN: f32 = 0.85;

// There is deliberately no smoothing pass. The grid is already no finer than
// the blocks support, which is the smoothing -- and blurring it on top of that
// flattens the fit against its own clamped edges: on a quadratic glow one pass
// took 112 codes off the outermost cell and left the frame edge further out
// than a plain plane would have.

/// Per-channel median of each block of the frame, in raster order.
///
/// `None` for a block whose samples are mostly unusable, so that a saturated
/// or vignetted corner does not enter the fit as a dark reading.
struct BlockMedians {
    grid_w: usize,
    grid_h: usize,
    /// Per channel, one entry per block.
    values: [Vec<Option<f32>>; 3],
    /// Typical number of samples behind one block median, per channel.
    samples_per_block: [usize; 3],
    /// Partly on sensor; unsuitable for full-block photometry, but still
    /// needs a common-footprint obstruction check before its pixels merge.
    partial: Vec<bool>,
}

/// Measure a frame's blocks on a grid defined in *reference* coordinates.
///
/// `offset` gives, per block, where that block's scene content sits in this
/// frame's own sensor coordinates. Without it the pairing is by sensor
/// position, which silently assumes the burst barely moved: on a burst that
/// steps by most of a block width, block `k` of one frame and block `k` of
/// another look at different parts of the world, and the regression through
/// them measures the scene rather than the light.
///
/// The offset is rounded to a whole mosaic cell so that a block always holds
/// the same colours in the same proportion. Rotation within a block is left
/// uncorrected: over a hundred-odd pixels it moves the content by a fraction
/// of a block, which a median absorbs.
fn block_medians(frame: &RawFrame, offset: &[(i32, i32)]) -> BlockMedians {
    let grid_w = (frame.width / BLOCK).clamp(MIN_BLOCKS, MAX_BLOCKS);
    let grid_h = (frame.height / BLOCK).clamp(MIN_BLOCKS, MAX_BLOCKS);
    let n = grid_w * grid_h;
    let mut values: [Vec<Option<f32>>; 3] =
        [vec![None; n], vec![None; n], vec![None; n]];
    let mut counts = [0usize; 3];
    let mut partial = vec![false; n];

    let mut buf: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for by in 0..grid_h {
        for bx in 0..grid_w {
            let idx0 = by * grid_w + bx;
            let (ox, oy) = offset.get(idx0).copied().unwrap_or((0, 0));
            let rx0 = (bx * frame.width / grid_w) as i32 + ox;
            let rx1 = ((bx + 1) * frame.width / grid_w) as i32 + ox;
            let ry0 = (by * frame.height / grid_h) as i32 + oy;
            let ry1 = ((by + 1) * frame.height / grid_h) as i32 + oy;
            // A clipped block and its full reference block cover different
            // scene footprints. Their median difference is not a change in
            // illumination, so neither partial nor absent blocks enter the fit.
            if rx0 < 0 || ry0 < 0 || rx1 > frame.width as i32 || ry1 > frame.height as i32 {
                partial[idx0] = rx1 > 0 && ry1 > 0
                    && rx0 < frame.width as i32 && ry0 < frame.height as i32;
                continue;
            }
            let x0 = rx0.max(0) as usize;
            let x1 = (rx1.min(frame.width as i32)).max(0) as usize;
            let y0 = ry0.max(0) as usize;
            let y1 = (ry1.min(frame.height as i32)).max(0) as usize;
            if x1 <= x0 + 4 || y1 <= y0 + 4 {
                continue;
            }
            for b in buf.iter_mut() {
                b.clear();
            }
            // Stride in whole mosaic cells so that a stride cannot favour one
            // colour, and so that the cost of this does not grow with sensor
            // size beyond a few thousand samples per block.
            let cells_x = (x1 - x0) / 2;
            let cells_y = (y1 - y0) / 2;
            let stride = (((cells_x * cells_y) as f32 / 2048.0).sqrt().max(1.0)) as usize;
            let mut cy = y0 & !1;
            while cy + 1 < y1 {
                let mut cx = x0 & !1;
                while cx + 1 < x1 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (x, y) = (cx + dx, cy + dy);
                            let i = y * frame.width + x;
                            let v = frame.value(x, y);
                            if frame.usable_value(i, v) {
                                buf[frame.channel_at(x, y)].push(v);
                            }
                        }
                    }
                    cx += 2 * stride;
                }
                cy += 2 * stride;
            }
            let idx = by * grid_w + bx;
            for c in 0..3 {
                if buf[c].len() >= 32 {
                    values[c][idx] = Some(sr_core::math::median(&buf[c]));
                    counts[c] = counts[c].max(buf[c].len());
                }
            }
        }
    }
    BlockMedians { grid_w, grid_h, values, samples_per_block: counts, partial }
}

/// Read all colour phases near the same sky position, without interpolating
/// different CFA colours. The robust comparisons below tolerate the small
/// position differences within a mosaic cell; stars do not define a sky block.
fn sky_sample(frame: &RawFrame, warp: Option<&WarpField>, rx: f32, ry: f32) -> [Option<f32>; 3] {
    if frame.width < 2 || frame.height < 2 { return [None; 3]; }
    let Some((sx, sy)) = warp.map_or(Some((rx, ry)), |w| w.inverse_map(rx, ry)) else {
        return [None; 3];
    };
    if !sx.is_finite() || !sy.is_finite() || sx < 0.0 || sy < 0.0
        || sx >= (frame.width - 1) as f32 || sy >= (frame.height - 1) as f32 {
        return [None; 3];
    }
    let (x, y) = (sx as usize & !1, sy as usize & !1);
    let mut sums = [0.0; 3];
    let mut counts = [0; 3];
    for dy in 0..2 {
        for dx in 0..2 {
            let (x, y) = (x + dx, y + dy);
            let v = frame.value(x, y);
            if frame.usable_value(y * frame.width + x, v) {
                let c = frame.channel_at(x, y);
                sums[c] += v;
                counts[c] += 1;
            }
        }
    }
    std::array::from_fn(|c| (counts[c] > 0).then(|| sums[c] / counts[c] as f32))
}

/// Full-block matching deliberately skips partial footprints. Detect shadows
/// there separately by comparing *the same reference positions* across peers,
/// rather than comparing a clipped median with a different, full sky region.
/// These samples never enter the gain or sky-field fit.
fn clipped_obstructions(
    frames: &[&RawFrame],
    warps: &[WarpField],
    blocks: &[BlockMedians],
    maps: &[FramePhotometry],
    measured: &[bool],
    exclude: &[Vec<bool>],
) -> Vec<(usize, usize)> {
    const SIDE: usize = 12;
    const MIN_POINTS: usize = 32;
    let (width, height) = (frames[0].width, frames[0].height);
    let grid_w = (width / BLOCK).clamp(MIN_BLOCKS, MAX_BLOCKS);
    let grid_h = (height / BLOCK).clamp(MIN_BLOCKS, MAX_BLOCKS);
    let candidates: Vec<_> = (0..grid_w * grid_h).filter(|&b| {
        measured.iter().enumerate().any(|(i, &m)| m && blocks[i].partial[b])
    }).collect();
    candidates.into_par_iter().flat_map_iter(|b| {
        let (bx, by) = (b % grid_w, b / grid_w);
        let mut samples = vec![vec![[None; 3]; SIDE * SIDE]; frames.len()];
        for py in 0..SIDE {
            for px in 0..SIDE {
                let rx = (bx as f32 + (px as f32 + 0.5) / SIDE as f32) * width as f32 / grid_w as f32;
                let ry = (by as f32 + (py as f32 + 0.5) / SIDE as f32) * height as f32 / grid_h as f32;
                for (i, &m) in measured.iter().enumerate() {
                    if m {
                        samples[i][py * SIDE + px] = sky_sample(frames[i], warps.get(i), rx, ry);
                    }
                }
            }
        }
        let mut found = Vec::new();
        for (i, &m) in measured.iter().enumerate() {
            if !m || !blocks[i].partial[b] { continue; }
            for c in 0..frames[i].channels() {
                let n = samples[i].iter().filter(|s| s[c].is_some()).count();
                if n < MIN_POINTS { continue; }
                let mut peer_deficits = Vec::new();
                let mut peer_levels = Vec::new();
                let mut differences = Vec::new();
                let mut levels = Vec::new();
                let mut support = n;
                for (j, &peer_measured) in measured.iter().enumerate() {
                    if j == i || !peer_measured || exclude[j][b] { continue; }
                    differences.clear();
                    levels.clear();
                    for (own, peer) in samples[i].iter().zip(&samples[j]) {
                        if let (Some(a), Some(v)) = (own[c], peer[c]) {
                            let own = maps[i].map.gain[c] * a + maps[i].map.offset[c];
                            let peer = maps[j].map.gain[c] * v + maps[j].map.offset[c];
                            differences.push(peer - own);
                            levels.push(peer);
                        }
                    }
                    // Each comparison covers nearly the whole candidate's
                    // footprint, with identical sky positions in both frames.
                    if differences.len() < MIN_POINTS || differences.len() * 5 < n * 4 { continue; }
                    support = support.min(differences.len());
                    peer_deficits.push(sr_core::math::median(&differences));
                    peer_levels.push(sr_core::math::median(&levels));
                }
                if peer_deficits.len() < 3 { continue; }
                let deficit = sr_core::math::median(&peer_deficits);
                let expected = (sr_core::math::median(&peer_levels) - maps[i].map.offset[c]).max(0.0);
                // The decision concerns a regional median. Compare its spread
                // across frames, not single-pixel shot noise. Do not divide this
                // scatter by sqrt(samples): coherent sky differences must remain
                // a veto even when a block has many samples.
                let scatter = sr_core::math::mad_sigma(&peer_deficits);
                let gain = maps[i].map.gain[c];
                let sigma = gain * frames[i].noise.std_dev(expected / gain) / (support as f32).sqrt();
                if deficit > OBSTRUCTION_FRACTION * expected
                    && deficit > OBSTRUCTION_SIGMAS * sigma.max(1e-9)
                    && deficit > OBSTRUCTION_SIGMAS * scatter {
                    found.push((i, b));
                    break;
                }
            }
        }
        found
    }).collect()
}

/// Least-squares line through matched pairs, with two Tukey reweightings.
///
/// The reweighting is what keeps a moving object, a satellite trail or a
/// passing headlight from tilting the illumination estimate: those affect a few
/// blocks out of hundreds and are exactly what a redescending weight discards.
fn fit_line(pairs: &[(f32, f32)]) -> Option<(f32, f32)> {
    if pairs.len() < 8 {
        return None;
    }
    let mut w = vec![1.0f32; pairs.len()];
    let mut best = None;
    for pass in 0..3 {
        let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (i, &(x, y)) in pairs.iter().enumerate() {
            let wi = w[i] as f64;
            let (x, y) = (x as f64, y as f64);
            sw += wi;
            sx += wi * x;
            sy += wi * y;
            sxx += wi * x * x;
            sxy += wi * x * y;
        }
        let denom = sw * sxx - sx * sx;
        if sw <= 0.0 || denom.abs() < 1e-18 {
            return best;
        }
        let gain = (sw * sxy - sx * sy) / denom;
        let offset = (sy - gain * sx) / sw;
        best = Some((gain as f32, offset as f32));
        if pass == 2 {
            break;
        }
        let resid: Vec<f32> = pairs
            .iter()
            .map(|&(x, y)| y - (gain as f32 * x + offset as f32))
            .collect();
        let sigma = sr_core::math::mad_sigma(&resid).max(1e-9);
        let cut = 4.0 * sigma;
        for (i, r) in resid.iter().enumerate() {
            let u = r / cut;
            w[i] = if u.abs() < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
        }
    }
    best
}

/// Why a frame ended up with the map it did, for the run manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhotometrySource {
    /// Gain and pedestal both fitted against the anchor.
    Measured,
    /// Pedestal only. Either the scene had no structure to fit a gain to, or
    /// the gain that came out was not one an illumination change explains —
    /// which on a night sky is what a frame under cloud looks like. Matching
    /// the level is still right and still worth doing; the disagreement that
    /// remains is local, and local disagreement is what motion rejection is
    /// for.
    Level,
    /// Not enough usable blocks to compare at all; the exposure metadata was
    /// used.
    Exposure,
    /// Matching is switched off.
    Disabled,
}

impl PhotometrySource {
    pub fn name(self) -> &'static str {
        match self {
            PhotometrySource::Measured => "measured",
            PhotometrySource::Level => "level-only",
            PhotometrySource::Exposure => "exposure",
            PhotometrySource::Disabled => "disabled",
        }
    }
}

/// One frame's photometric relationship to the anchor.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FramePhotometry {
    pub map: PhotometricMatch,
    pub source: PhotometrySource,
}

/// Where each block's scene content sits in one frame's own sensor
/// coordinates, relative to where it sits in the reference.
///
/// Rounded to a whole mosaic cell so the block keeps its colour composition.
fn block_offsets(
    warp: &WarpField,
    width: usize,
    height: usize,
    grid_w: usize,
    grid_h: usize,
) -> Vec<(i32, i32)> {
    let mut out = Vec::with_capacity(grid_w * grid_h);
    for by in 0..grid_h {
        for bx in 0..grid_w {
            let rx = (bx as f32 + 0.5) * width as f32 / grid_w as f32;
            let ry = (by as f32 + 0.5) * height as f32 / grid_h as f32;
            let (dx, dy) = match warp.inverse_map(rx, ry) {
                Some((sx, sy)) => (sx - rx, sy - ry),
                None => (0.0, 0.0),
            };
            out.push((
                ((dx * 0.5).round() * 2.0) as i32,
                ((dy * 0.5).round() * 2.0) as i32,
            ));
        }
    }
    out
}

/// Match every frame of a burst onto the reference.
///
/// `warps` carries each frame onto the reference, and is what lets a block of
/// one frame be compared with the same *scene* in another rather than the same
/// sensor position. Running after registration is therefore not an ordering
/// convenience, it is a requirement: paired by sensor position alone, a burst
/// that panned by most of a block width has its scene change measured as a
/// change of light.
///
/// `exposure_scale` is the fallback, and is never applied on top of a measured
/// fit: a real exposure difference is already visible in the pixels, so a fit
/// that saw it has accounted for it and multiplying again would correct twice.
pub fn match_burst(
    frames: &[RawFrame],
    warps: &[WarpField],
    reference: usize,
    exposure_scale: &[f32],
) -> Vec<FramePhotometry> {
    match_subset(frames, warps, reference, exposure_scale, &[])
}

/// Match a subset of a burst, leaving the rest untouched.
///
/// `active` empty means every frame. Otherwise only the selected frames are
/// measured, which is what reconstructing several filters onto one grid needs:
/// a fit between an H-alpha frame and an OIII one is a fit between two
/// different pictures of the sky, and both the number and the frame it was
/// measured on would be meaningless.
pub fn match_subset(
    frames: &[RawFrame],
    warps: &[WarpField],
    reference: usize,
    exposure_scale: &[f32],
    active: &[bool],
) -> Vec<FramePhotometry> {
    match_with_stars(frames, warps, reference, exposure_scale, active, &[], true)
}

/// The same, with each frame's point sources to take the gain from.
///
/// Transparency multiplies what the sky *and* the stars deliver; sky
/// brightness only adds. So a gain is a statement about the stars, and a
/// pedestal is a statement about the sky, and the two are separable only if
/// the gain is measured on something that has no sky in it.
///
/// Fitted to block medians alone they are not. A block median is the sky, and
/// a line through the blocks of two frames whose skies differ additively --
/// twilight, moonrise, a town below one horizon -- reads part of that
/// difference as slope, because the blocks carry vignetting and the subject
/// and so are not all at one level. On one test burst the frames shot at dusk
/// came out with gains of 0.56 against a reference taken in full dark, which
/// is not a transparency any sky has: every star in those frames was then
/// merged at half its flux, and the parts of the frame where those frames
/// happened to overlap were left with a different sky from the parts where
/// they did not.
///
/// A star's flux is already sky-subtracted. The ratio of the same star's flux
/// in two frames is the transparency between them and nothing else.
pub fn match_with_stars(
    frames: &[RawFrame],
    warps: &[WarpField],
    reference: usize,
    exposure_scale: &[f32],
    active: &[bool],
    stars: &[Vec<Star>],
    sky_field: bool,
) -> Vec<FramePhotometry> {
    match_with_star_refs(
        &frames.iter().collect::<Vec<_>>(),
        warps,
        reference,
        exposure_scale,
        active,
        stars,
        sky_field,
    )
}

/// Borrowed-frame counterpart for streaming callers. Uses exactly the same
/// photometric estimator without copying the reference's detector samples.
pub fn match_with_star_refs(
    frames: &[&RawFrame],
    warps: &[WarpField],
    reference: usize,
    exposure_scale: &[f32],
    active: &[bool],
    stars: &[Vec<Star>],
    sky_field: bool,
) -> Vec<FramePhotometry> {
    let live = |i: usize| active.is_empty() || active.get(i).copied().unwrap_or(false);
    if frames.is_empty() {
        return Vec::new();
    }
    let reference = reference.min(frames.len() - 1);
    let (w, h) = (frames[0].width, frames[0].height);
    let grid_w = (w / BLOCK).clamp(MIN_BLOCKS, MAX_BLOCKS);
    let grid_h = (h / BLOCK).clamp(MIN_BLOCKS, MAX_BLOCKS);
    let none = vec![(0i32, 0i32); grid_w * grid_h];

    let blocks: Vec<BlockMedians> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            if !live(i) && i != reference {
                return BlockMedians {
                    grid_w: 0,
                    grid_h: 0,
                    values: [Vec::new(), Vec::new(), Vec::new()],
                    samples_per_block: [0; 3],
                    partial: Vec::new(),
                };
            }
            let offsets = match warps.get(i) {
                Some(warp) => block_offsets(warp, w, h, grid_w, grid_h),
                None => none.clone(),
            };
            block_medians(f, &offsets)
        })
        .collect();
    let target = &blocks[reference];

    // Transparency from the stars, where there are stars to take it from.
    let star_gain = star_gains(warps, reference, stars, frames.len());

    // One frame's match, with some of its blocks left out of every fit.
    let fit = |i: usize, exclude: &[bool]| -> FramePhotometry {
        {
            if i == reference {
                return FramePhotometry {
                    map: PhotometricMatch::IDENTITY,
                    source: PhotometrySource::Measured,
                };
            }
            if !live(i) {
                return FramePhotometry {
                    map: PhotometricMatch::IDENTITY,
                    source: PhotometrySource::Disabled,
                };
            }
            let fallback = FramePhotometry {
                map: PhotometricMatch::from_exposure(
                    exposure_scale.get(i).copied().unwrap_or(1.0),
                ),
                source: PhotometrySource::Exposure,
            };
            let src = &blocks[i];
            if src.grid_w != target.grid_w || src.grid_h != target.grid_h {
                return fallback;
            }
            let mut map = PhotometricMatch::IDENTITY;
            let mut source = PhotometrySource::Measured;
            // A monochrome frame has one channel; the other two hold nothing
            // and fitting them would fail every frame into the metadata
            // fallback, which is how this was found.
            for c in 0..frames[i].channels() {
                // Position travels with each pair so that the residual left
                // by the global match can be fitted as a slope across the
                // frame. Normalised to -1..1 so the coefficients mean the same
                // thing whatever the grid.
                let placed: Vec<(f32, f32, f32, f32)> = src.values[c]
                    .iter()
                    .zip(target.values[c].iter())
                    .enumerate()
                    .filter_map(|(b, (a, t))| match (a, t) {
                        (Some(a), Some(t)) if !exclude.get(b).copied().unwrap_or(false) => {
                            let (bx, by) = (b % src.grid_w, b / src.grid_w);
                            let u = 2.0 * (bx as f32 + 0.5) / src.grid_w as f32 - 1.0;
                            let w = 2.0 * (by as f32 + 0.5) / src.grid_h as f32 - 1.0;
                            Some((*a, *t, u, w))
                        }
                        _ => None,
                    })
                    .collect();
                let pairs: Vec<(f32, f32)> =
                    placed.iter().map(|&(a, t, _, _)| (a, t)).collect();
                if pairs.len() < 8 {
                    return fallback;
                }

                // A gain is only identifiable if the blocks differ from each
                // other by more than the uncertainty of a block median. On a
                // scene with no large-scale structure — fog, blank sky, a
                // wall — they do not, and a slope fitted through them is
                // fitted to whatever noise survived the median. Match the
                // level instead and leave the gain at one, which is the honest
                // answer rather than a confident wrong one.
                let level = median_of(&pairs, |p| p.0);
                let median_sigma = frames[i].noise.std_dev(level)
                    / (src.samples_per_block[c].max(1) as f32).sqrt();
                let identifiable =
                    block_spread(&pairs) >= SPREAD_MARGIN * median_sigma.max(1e-9);
                // The stars settle the gain where they can; the blocks are
                // then asked only for the pedestal, which is what they can
                // actually see.
                let stellar = star_gain.get(i).copied().flatten();
                let fitted = match stellar {
                    Some(g) => {
                        let g = g[c.min(2)];
                        Some((g, median_of(&pairs, |p| p.1 - g * p.0)))
                    }
                    None if identifiable => fit_line(&pairs),
                    None => None,
                };
                match fitted {
                    Some((gain, offset))
                        if gain.is_finite()
                            && offset.is_finite()
                            && (stellar.is_some() || (GAIN_LIMITS.0..=GAIN_LIMITS.1).contains(&gain)) =>
                    {
                        map.gain[c] = gain;
                        map.offset[c] = offset;
                    }
                    _ => {
                        map.gain[c] = 1.0;
                        map.offset[c] = median_of(&pairs, |p| p.1 - p.0);
                        source = PhotometrySource::Level;
                    }
                }

                // What the global match could not explain, as a function of
                // where in the frame it was left -- and, first, which of it is
                // not sky at all. A block still a tenth of the sky darker than
                // the reference after the gain and the pedestal is obstructed,
                // and a field fitted through it would explain the obstruction
                // as a gradient and lift it back into the stack.
                let residual: Vec<(f32, f32, f32)> = placed
                    .iter()
                    .map(|&(a, t, u, w)| (u, w, t - (map.gain[c] * a + map.offset[c])))
                    .collect();
                let fitted_field =
                    if sky_field { fit_field(&residual, median_sigma) } else { None };
                if let Some((field, constant)) = fitted_field {
                    map.field[c] = field;
                    map.offset[c] += constant;
                }
            }
            FramePhotometry { map, source }
        }
    };
    let none: Vec<bool> = Vec::new();
    let mut out: Vec<FramePhotometry> = (0..frames.len()).map(|i| fit(i, &none)).collect();

    // Obstructions: blocks where this frame, brought onto the reference's
    // scale, is far darker than the burst typically is at that block. Judged
    // against the burst and not against the frame's own sky, because a dark
    // nebula is dark in every frame and a tree is dark in this one; and not
    // against the reference alone, because the reference has its own night.
    let measured: Vec<bool> = (0..frames.len())
        .map(|i| {
            i == reference
                || (live(i)
                    && matches!(
                        out[i].source,
                        PhotometrySource::Measured | PhotometrySource::Level
                    )
                    && blocks[i].grid_w == target.grid_w
                    && blocks[i].grid_h == target.grid_h)
        })
        .collect();
    let nblocks = target.grid_w * target.grid_h;
    let corrected = |i: usize, c: usize, b: usize| -> Option<f32> {
        let a = blocks[i].values[c].get(b).copied().flatten()?;
        Some(out[i].map.gain[c] * a + out[i].map.offset[c])
    };
    let channels = frames[reference].channels();
    // The typical value at each block and how much the frames scatter about
    // it. A block on a strong edge of the subject -- a nebula's rim, a printed
    // step on the synthetic chart -- takes a different median in every frame
    // as the dither moves the edge across it, and that scatter is not
    // obstruction; a block of sky agrees across the burst to a fraction of a
    // percent, and a tree is a departure from that by many times its scatter.
    let mut typical = vec![[f32::NAN; 3]; nblocks];
    let mut scatter = vec![[f32::NAN; 3]; nblocks];
    let mut cell = Vec::with_capacity(frames.len());
    for (b, (typ, sct)) in typical.iter_mut().zip(scatter.iter_mut()).enumerate() {
        for c in 0..channels {
            cell.clear();
            cell.extend((0..frames.len()).filter(|&i| measured[i]).filter_map(|i| corrected(i, c, b)));
            if cell.len() >= 3 {
                typ[c] = sr_core::math::median(&cell);
                sct[c] = sr_core::math::mad_sigma(&cell);
            }
        }
    }
    let mut excluded = vec![vec![false; nblocks]; frames.len()];
    for (i, &is_measured) in measured.iter().enumerate() {
        if !is_measured {
            continue;
        }
        let exclude = &mut excluded[i];
        for (b, ex) in exclude.iter_mut().enumerate() {
            for (c, &t) in typical[b].iter().enumerate().take(channels) {
                if !t.is_finite() {
                    continue;
                }
                let sct = scatter[b][c];
                if let Some(v) = corrected(i, c, b) {
                    let deficit = t - v;
                    let map = &out[i].map;
                    // t and v share the matching offset. It cancels from the
                    // deficit, and must also be removed from its fractional
                    // scale. Otherwise a large positive offset (e.g. mixed
                    // camera gains) makes even a deep foreground shadow look
                    // like a small fraction of the reference sky.
                    let expected = (t - map.offset[c]).max(0.0);
                    let gain = map.gain[c];
                    let median_sigma = gain * frames[i].noise.std_dev(expected / gain)
                        / (blocks[i].samples_per_block[c].max(1) as f32).sqrt();
                    if deficit > OBSTRUCTION_FRACTION * expected
                        && deficit > OBSTRUCTION_SIGMAS * median_sigma.max(1e-9)
                        && (!sct.is_finite() || deficit > OBSTRUCTION_SIGMAS * sct)
                    {
                        *ex = true;
                    }
                }
            }
        }
    }
    for (i, b) in clipped_obstructions(frames, warps, &blocks, &out, &measured, &excluded) {
        excluded[i][b] = true;
    }
    for (i, exclude) in excluded.into_iter().enumerate().filter(|(_, e)| e.iter().any(|&v| v)) {
        let mut blocked = no_blocks();
        for (b, &ex) in exclude.iter().enumerate() {
            if ex {
                let (bx, by) = (b % target.grid_w, b / target.grid_w);
                let u = 2.0 * (bx as f32 + 0.5) / target.grid_w as f32 - 1.0;
                let w = 2.0 * (by as f32 + 0.5) / target.grid_h as f32 - 1.0;
                mark_blocked(&mut blocked, u, w);
            }
        }
        // A block half covered by foliage has the sky's median and passes the
        // threshold, so an obstruction always has a fringe of blocks that are
        // partly in it. One cell of dilation takes the fringe with it; on the
        // six obstructed frames of a ninety-six-frame burst that costs nothing
        // anyone would miss.
        dilate_blocked(&mut blocked);
        extend_blocked_to_the_edge(&mut blocked);
        if i != reference {
            out[i] = fit(i, &exclude);
        }
        out[i].map.blocked = blocked;
    }

    let participates: Vec<bool> = (0..frames.len())
        .map(|i| {
            i == reference
                || (live(i)
                    && matches!(
                        out[i].source,
                        PhotometrySource::Measured | PhotometrySource::Level
                    ))
        })
        .collect();
    take_the_burst_shape(&mut out, &participates);
    out
}

/// Transparency of each frame against the reference, from the flux of the
/// stars they share.
///
/// A star is matched to the reference's nearest star after the frame's warp is
/// applied, within `STAR_MATCH_RADIUS`, and only where that match is
/// unambiguous -- the next-nearest reference star must be further off than
/// `STAR_MATCH_MARGIN` times the distance -- so that a crowded field does not
/// pair a faint star with a bright neighbour and call the ratio transparency.
/// The gain is the median of the flux ratios, which needs no fit and no
/// weighting: every star is one estimate of the same number.
/// Catalogues must carry detector-unit aperture flux, for example from
/// `stars::positions_for_photometry`, never registration's noise-normalized rank.
pub fn star_gains(
    warps: &[WarpField],
    reference: usize,
    stars: &[Vec<Star>],
    frames: usize,
) -> Vec<Option<[f32; 3]>> {
    let mut out = vec![None; frames];
    if stars.len() != frames || reference >= frames {
        return out;
    }
    let mut paired: Vec<usize> = Vec::new();
    let anchor = &stars[reference];
    if anchor.len() < STAR_GAIN_MIN {
        return out;
    }
    let grid = StarGrid::new(anchor);
    let ref_warp = &warps[reference];
    for (i, mine) in stars.iter().enumerate() {
        if i == reference {
            out[i] = Some([1.0; 3]);
            continue;
        }
        if mine.len() < STAR_GAIN_MIN {
            continue;
        }
        let warp = &warps[i];
        let mut ratios: [Vec<f32>; 3] = Default::default();
        for st in mine.iter() {
            if !st.flux.is_finite() || st.flux <= 0.0 {
                continue;
            }
            // Into the reference frame's own sensor coordinates.
            let (rx, ry) = warp.map(st.x, st.y);
            let Some((ax, ay)) = ref_warp.inverse_map(rx, ry) else {
                continue;
            };
            let Some((j, second)) = grid.nearest_two(ax, ay, STAR_MATCH_RADIUS) else {
                continue;
            };
            if let Some(d2) = second {
                if d2 < STAR_MATCH_MARGIN * STAR_MATCH_MARGIN * grid.last_distance2() {
                    continue;
                }
            }
            // One gain for the three channels, from the detector's own
            // achromatic flux.
            //
            // Transparency is wavelength dependent and the stars can see it:
            // measured channel by channel, the red-to-green ratio moves by a
            // fifth across this burst. Taken per channel it measured no better
            // than one gain -- red and blue are sampled at a quarter of
            // green's density, so three separately measured gains are three
            // separately noisy ones -- and the extra noise would go into the
            // colour of every frame before it is merged. What differential
            // extinction is left is smooth in position and time, and the
            // per-channel pedestal and field carry it.
            let anchor_flux = anchor[j].flux;
            if anchor_flux.is_finite() && anchor_flux > 0.0 {
                ratios[0].push(anchor_flux / st.flux);
            }
        }
        if ratios[0].len() < STAR_GAIN_MIN {
            continue;
        }
        if let Some(g) = supported_stellar_gain(&ratios[0]) {
            out[i] = Some([g; 3]);
            paired.push(ratios[0].len());
        }
    }
    if !paired.is_empty() {
        let mut got: Vec<f32> = out.iter().flatten().map(|g| g[1]).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        log::info!(
            "photometry: transparency from the stars on {} of {frames} frames, {} stars \
             paired at the median; gain {:.3} to {:.3}",
            paired.len(),
            sr_core::math::median(&paired.iter().map(|&n| n as f32).collect::<Vec<_>>()),
            got.first().copied().unwrap_or(1.0),
            got.last().copied().unwrap_or(1.0)
        );
    }
    out
}

/// How near a warped star must land to a reference star to be the same star,
/// in sensor pixels. Registration is good to a fraction of a pixel by this
/// stage; two is slack for the frames it was worst on.
const STAR_MATCH_RADIUS: f32 = 2.0;

/// How much further the second-nearest reference star must be before the
/// nearest is believed to be the match.
const STAR_MATCH_MARGIN: f32 = 2.5;

/// Matched stars needed before their median ratio is preferred to the blocks.
const STAR_GAIN_MIN: usize = 30;

/// Preserve ordinary-range matching, but require strong agreement before
/// extending beyond the historical 0.5–2 range. These are conservative screening
/// limits, not formal confidence intervals. No camera gain setting is assumed
/// proportional to detector response.
fn supported_stellar_gain(ratios: &[f32]) -> Option<f32> {
    let valid: Vec<f32> = ratios.iter().copied().filter(|r| r.is_finite() && *r > 0.0).collect();
    if valid.len() < STAR_GAIN_MIN { return None; }
    let gain = sr_core::math::median(&valid);
    let squared = gain * gain;
    if !squared.is_finite() || squared < f32::MIN_POSITIVE { return None; }
    if (GAIN_LIMITS.0..=GAIN_LIMITS.1).contains(&gain) { return Some(gain); }
    let deviations: Vec<f32> = valid.iter().map(|r| (r / gain - 1.0).abs()).collect();
    let mad = sr_core::math::median(&deviations);
    let agreement = deviations.iter().filter(|&&d| d <= 0.2).count();
    let supported = mad <= 0.1 && agreement * 5 >= valid.len() * 4;
    log::info!("photometry: stellar gain {gain:.4} outside 0.5–2, {}/{} pairs agree within 20%, relative median deviation {:.1}%: {}",
        agreement, valid.len(), mad * 100.0, if supported { "supported" } else { "unsupported" });
    supported.then_some(gain)
}

/// A bucketed index over the reference stars, for nearest-neighbour lookup.
struct StarGrid<'a> {
    stars: &'a [Star],
    cell: f32,
    origin: (f32, f32),
    cols: usize,
    rows: usize,
    buckets: Vec<Vec<u32>>,
    last: std::cell::Cell<f32>,
}

impl<'a> StarGrid<'a> {
    fn new(stars: &'a [Star]) -> Self {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for s in stars {
            x0 = x0.min(s.x);
            y0 = y0.min(s.y);
            x1 = x1.max(s.x);
            y1 = y1.max(s.y);
        }
        let cell = 32.0f32;
        let cols = (((x1 - x0) / cell).ceil() as usize + 1).max(1);
        let rows = (((y1 - y0) / cell).ceil() as usize + 1).max(1);
        let mut buckets = vec![Vec::new(); cols * rows];
        for (i, s) in stars.iter().enumerate() {
            let cx = ((s.x - x0) / cell) as usize;
            let cy = ((s.y - y0) / cell) as usize;
            buckets[cy.min(rows - 1) * cols + cx.min(cols - 1)].push(i as u32);
        }
        Self { stars, cell, origin: (x0, y0), cols, rows, buckets, last: std::cell::Cell::new(0.0) }
    }

    /// Index of the nearest star within `radius`, and the squared distance to
    /// the second nearest if there is one.
    fn nearest_two(&self, x: f32, y: f32, radius: f32) -> Option<(usize, Option<f32>)> {
        let cx = ((x - self.origin.0) / self.cell).floor();
        let cy = ((y - self.origin.1) / self.cell).floor();
        if !cx.is_finite() || !cy.is_finite() {
            return None;
        }
        let (cx, cy) = (cx as i64, cy as i64);
        let r2 = radius * radius;
        let (mut best, mut best_d2) = (None, f32::MAX);
        let mut second_d2 = f32::MAX;
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (bx, by) = (cx + dx, cy + dy);
                if bx < 0 || by < 0 || bx >= self.cols as i64 || by >= self.rows as i64 {
                    continue;
                }
                for &j in &self.buckets[by as usize * self.cols + bx as usize] {
                    let s = &self.stars[j as usize];
                    let _ = s.flux;
                    let d2 = (s.x - x) * (s.x - x) + (s.y - y) * (s.y - y);
                    if d2 < best_d2 {
                        second_d2 = best_d2;
                        best_d2 = d2;
                        best = Some(j as usize);
                    } else if d2 < second_d2 {
                        second_d2 = d2;
                    }
                }
            }
        }
        if best_d2 > r2 {
            return None;
        }
        self.last.set(best_d2);
        Some((best?, if second_d2.is_finite() { Some(second_d2) } else { None }))
    }

    fn last_distance2(&self) -> f32 {
        self.last.get().max(1e-6)
    }
}

/// Mark the field cell a block centre falls in as obstructed.
fn mark_blocked(blocked: &mut [[bool; FIELD]; FIELD], u: f32, w: f32) {
    let last = (FIELD - 1) as f32;
    let gx = ((u + 1.0) * 0.5 * last).round().clamp(0.0, last) as usize;
    let gy = ((w + 1.0) * 0.5 * last).round().clamp(0.0, last) as usize;
    blocked[gy][gx] = true;
}

/// Every cell touching a blocked cell is blocked too; see the call site.
fn dilate_blocked(blocked: &mut [[bool; FIELD]; FIELD]) {
    let before = *blocked;
    for gy in 0..FIELD {
        for gx in 0..FIELD {
            if before[gy][gx] {
                continue;
            }
            let near = (gy.saturating_sub(1)..=(gy + 1).min(FIELD - 1))
                .any(|y| (gx.saturating_sub(1)..=(gx + 1).min(FIELD - 1)).any(|x| before[y][x]));
            if near {
                blocked[gy][gx] = true;
            }
        }
    }
}

/// A border cell next to a blocked cell is blocked too; see the call site.
fn extend_blocked_to_the_edge(blocked: &mut [[bool; FIELD]; FIELD]) {
    let last = FIELD - 1;
    let before = *blocked;
    for gy in 0..FIELD {
        for gx in 0..FIELD {
            if before[gy][gx] {
                continue;
            }
            let on_edge = gx == 0 || gy == 0 || gx == last || gy == last;
            if !on_edge {
                continue;
            }
            // The neighbour one step towards the interior on each edge axis,
            // and the diagonal for a corner.
            let ix = if gx == 0 { 1 } else if gx == last { last - 1 } else { gx };
            let iy = if gy == 0 { 1 } else if gy == last { last - 1 } else { gy };
            if before[iy][ix] || before[gy][ix] || before[iy][gx] {
                blocked[gy][gx] = true;
            }
        }
    }
}

/// Move the additive fields so that the stack's sky takes the shape of the
/// typical frame rather than the reference's.
///
/// Each frame's field is fitted against the reference, so applying them as
/// fitted carries every frame onto the reference's own sky: the reference is
/// then not just the geometric anchor but the photometric truth about what
/// shape the sky had, and its particular tilt -- the moon on its side of the
/// frame at that hour, a town below that horizon -- is imprinted on the whole
/// burst at full strength. On one ninety-six frame burst the reference was
/// tilted nine percent left-to-right against the median frame, and the stack
/// came out with that tilt on top of the sky's own.
///
/// The typical frame is a spatial median of whole fields, the reference
/// counted at zero. Subtracting it from every field, the reference's included,
/// leaves the frames agreeing with each other exactly as before -- the merge
/// does not care what the common shape is, only that they share one -- and
/// makes that common shape the burst's rather than one exposure's. Frames
/// with no measured field take part at zero; a frame whose photometry came
/// from the metadata does not, since nothing was measured about it.
fn take_the_burst_shape(maps: &mut [FramePhotometry], participates: &[bool]) {
    // Keep the contributors fixed across the field. Switching exposures at
    // an obstruction boundary can introduce a discontinuity into every map.
    let clear: Vec<bool> = maps.iter().zip(participates)
        .map(|(m, &p)| p && m.map.blocked_fraction() == 0.0)
        .collect();
    let n = clear.iter().filter(|&&p| p).count();
    if n < 2 {
        return;
    }
    let mut common = no_field();
    for (c, plane) in common.iter_mut().enumerate() {
        let fields: Vec<_> = maps
            .iter()
            .zip(&clear)
            .filter(|(_, &p)| p)
            .map(|(m, _)| {
                m.map.field[c]
                    .iter()
                    .flatten()
                    .map(|v| *v as f64)
                    .collect::<Vec<_>>()
            })
            .collect();
        // A separate median at each node can switch exposures at a crossing
        // and turn perfectly planar skies into a ridge. Weiszfeld reweighting
        // gives each whole field one weight, preserving their spatial basis.
        // A tiny distance floor handles coincident fields; f64 keeps the
        // iteration stable when many exposures have an identical zero field.
        let size = FIELD * FIELD;
        let mut centre = vec![0.0; size];
        for field in &fields {
            for (v, f) in centre.iter_mut().zip(field) { *v += f / n as f64; }
        }
        for _ in 0..200 {
            let mut next = vec![0.0; size];
            let mut total = 0.0;
            for field in &fields {
                let distance = (field.iter().zip(&centre).map(|(f, v)| (f-v).powi(2)).sum::<f64>()
                    / size as f64).sqrt();
                let weight = 1.0 / distance.max(1e-12);
                total += weight;
                for (v, f) in next.iter_mut().zip(field) { *v += weight * f; }
            }
            for v in &mut next { *v /= total; }
            let change = next.iter().zip(&centre).map(|(a, b)| (a-b).abs()).fold(0.0f64, f64::max);
            centre = next;
            if change < 1e-10 { break; }
        }
        for (v, estimate) in plane.iter_mut().flatten().zip(centre) {
            *v = estimate as f32;
        }
    }
    for (c, plane) in common.iter().enumerate() {
        let (lo, hi) = plane.iter().flatten().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
        log::info!(
            "photometry: the typical frame's sky differs from the reference's by {:.3}% of full \
             scale across the field in channel {c} ({n} frames measured)",
            (hi - lo) * 100.0
        );
    }
    for (m, &p) in maps.iter_mut().zip(participates) {
        if !p {
            continue;
        }
        for (plane, shared) in m.map.field.iter_mut().zip(common.iter()) {
            for (row, shared_row) in plane.iter_mut().zip(shared.iter()) {
                for (value, shared_value) in row.iter_mut().zip(shared_row.iter()) {
                    *value -= shared_value;
                }
            }
        }
    }
}

/// Spread of the source block medians, as a robust interdecile range.
/// A smooth additive field through the residuals the global match left behind.
///
/// Returns the zero-mean field and the constant taken out of it, or `None` when
/// the field does not explain enough of the residual to be worth believing.
/// That guard matters as much as the fit: a field fitted to noise is not
/// neutral, it is structure invented and then imposed on every pixel.
///
/// Three things keep it from following anything real.
///
/// * It is fitted to the *difference* between two frames of the same sky, so
///   the nebula, the stars and the dust are in both and cancel before this sees
///   them. What is left is what changed between the exposures. This is what
///   makes a grid safe here where fitting a background model to a single frame
///   would not be.
/// * It is never finer than the evidence supports. The fit happens on a grid
///   sized from the number of blocks available and is then resampled onto the
///   stored one, so a small frame gets something close to a plane and a full
///   sensor gets the full grid — and neither is asked to invent detail.
/// * Each cell is a median of block medians, and each block median is itself a
///   median of thousands of pixels. A satellite crossing one block does not
///   move the cell.
fn fit_field(points: &[(f32, f32, f32)], noise: f32) -> Option<([[f32; FIELD]; FIELD], f32)> {
    if points.len() < FIELD_MIN_BLOCKS {
        return None;
    }

    // The resolution this many blocks can support, at `FIELD_MIN_PER_CELL`
    // blocks per cell. Two is a bilinear surface, which is the plane this
    // replaced; `FIELD` is as fine as the stored grid goes.
    let k = ((points.len() / FIELD_MIN_PER_CELL) as f32).sqrt() as usize;
    let k = k.clamp(2, FIELD_MAX_CELLS);

    // Cells, not nodes: each cell owns an equal share of the frame, so the
    // blocks land in it evenly. Binning onto nodes instead gives the corner
    // cells a quarter of the blocks of an interior one, and on a small grid
    // that leaves the corners empty.
    let mut bins: Vec<Vec<f32>> = vec![Vec::new(); k * k];
    for &(u, v, r) in points {
        if !r.is_finite() {
            continue;
        }
        let gx = ((u + 1.0) * 0.5 * k as f32) as usize;
        let gy = ((v + 1.0) * 0.5 * k as f32) as usize;
        bins[gy.min(k - 1) * k + gx.min(k - 1)].push(r);
    }

    let mut coarse = vec![0.0f32; k * k];
    let mut known = vec![false; k * k];
    let mut filled = 0usize;
    for i in 0..k * k {
        if bins[i].len() >= FIELD_MIN_PER_CELL {
            coarse[i] = sr_core::math::median(&bins[i]);
            known[i] = true;
            filled += 1;
        }
    }
    // Too little of the frame measured and the rest would be invented rather
    // than carried, so nothing is claimed at all.
    if filled * 3 < k * k {
        return None;
    }

    // Cells with nothing of their own take the mean of whatever neighbours have
    // been settled, spreading outwards. A frame that overlaps the reference
    // only partly still gets a field over the part it covers, and the rest is
    // carried out from the edge rather than extrapolated.
    while filled < k * k {
        let snapshot = known.clone();
        let mut progressed = false;
        for y in 0..k {
            for x in 0..k {
                if snapshot[y * k + x] {
                    continue;
                }
                let mut acc = 0.0f32;
                let mut n = 0.0f32;
                for (dy, dx) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (ny, nx) = (y as i32 + dy, x as i32 + dx);
                    if ny < 0 || nx < 0 || ny >= k as i32 || nx >= k as i32 {
                        continue;
                    }
                    if snapshot[ny as usize * k + nx as usize] {
                        acc += coarse[ny as usize * k + nx as usize];
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    coarse[y * k + x] = acc / n;
                    known[y * k + x] = true;
                    filled += 1;
                    progressed = true;
                }
            }
        }
        if !progressed {
            return None;
        }
    }

    // Onto the stored grid, which is a fixed size so that the map stays a plain
    // value that can be copied and serialised.
    let mut grid = [[0.0f32; FIELD]; FIELD];
    for (y, row) in grid.iter_mut().enumerate() {
        let v = 2.0 * y as f32 / (FIELD - 1) as f32 - 1.0;
        for (x, cell) in row.iter_mut().enumerate() {
            let u = 2.0 * x as f32 / (FIELD - 1) as f32 - 1.0;
            *cell = sample_coarse(&coarse, k, u, v);
        }
    }

    // The constant part belongs in `offset`, so that a map with no gradient
    // still reads as one and `apply` at the centre keeps its meaning.
    let mean: f32 = grid.iter().flatten().sum::<f32>() / (FIELD * FIELD) as f32;
    for v in grid.iter_mut().flatten() {
        *v -= mean;
    }

    // How much of the residual the field accounted for, as robust spreads so
    // that a handful of bad blocks cannot make a useless field look good.
    // Measured through the field as it will actually be stored and read.
    let before = spread_of(points.iter().map(|p| p.2));
    let probe = PhotometricMatch { gain: [1.0; 3], offset: [0.0; 3], field: [grid; 3], ..PhotometricMatch::IDENTITY };
    let after = spread_of(
        points.iter().map(|&(u, v, r)| r - (mean + probe.field_at(0, u, v))),
    );

    // Explaining *something* is not evidence. Fitting `k*k` free values to `n`
    // points removes about `k*k/n` of the variance whatever the points are, so
    // the bar has to be that much lower than one before it means anything --
    // otherwise a fine grid on a flat pair of frames passes for having found a
    // gradient, and then imposes it. Without this a 4x4 field on two frames of
    // nothing but sky invented 54 codes of gradient across the frame.
    let chance =
        (1.0 - (k * k) as f32 / points.len() as f32).max(0.0).sqrt();
    if before <= 1e-9 || after > before * chance * FIELD_MUST_EXPLAIN {
        return None;
    }

    // And it has to be worth removing.
    //
    // The test above asks whether the field explains the residual, not whether
    // there was a residual worth explaining. A frame differing from the
    // reference by a hair still has a shape to its difference, and a field
    // fitted to it is mostly the block medians' own scatter -- carried into
    // every frame, and not averaging out across the burst, because every frame
    // is fitted against the same reference.
    //
    // On a 96-frame burst that was worth a seventh of the sky's mid-scale
    // structure and a third of its broad colour, and it cost rejection too:
    // frames were made to disagree with a burst carrying everyone else's
    // invented lumps, 8.4% of samples against 5.0% with no field at all. The
    // field was added when the photometric gain was being read off the sky,
    // and the residuals it was mopping up were that error; with
    // the gain taken from the stars, most frames no longer have one.
    //
    // So the span the field would remove has to stand clear of what a block
    // median is uncertain by. A real moving gradient -- a rising moon, a town
    // below one horizon -- is orders of magnitude above this; the fields this
    // turns away are the ones fitted to nothing.
    let (lo, hi) = grid
        .iter()
        .flatten()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if hi - lo < FIELD_MIN_AMPLITUDE * noise.max(1e-9) {
        return None;
    }
    Some((grid, mean))
}

/// Read a `k` by `k` grid of cell values, in the normalised coordinates the map
/// uses. Cell `i` stands at the middle of its share of the frame, hence the
/// half.
///
/// Beyond the outermost cell centres the value is carried on by the slope of
/// the last pair rather than held flat. Half a cell of the frame lies outside
/// them on each side, and holding the edge value there leaves that strip
/// uncorrected -- which on a plain gradient is a third of the gradient still
/// present at the two edges, measured.
fn sample_coarse(g: &[f32], k: usize, u: f32, v: f32) -> f32 {
    if k == 1 {
        return g[0];
    }
    let place = |t: f32| -> (usize, f32) {
        let p = (t + 1.0) * 0.5 * k as f32 - 0.5;
        let i = (p.floor()).clamp(0.0, (k - 2) as f32) as usize;
        (i, p - i as f32)
    };
    let (x0, tx) = place(u);
    let (y0, ty) = place(v);
    let (x1, y1) = (x0 + 1, y0 + 1);
    let top = g[y0 * k + x0] + (g[y0 * k + x1] - g[y0 * k + x0]) * tx;
    let bottom = g[y1 * k + x0] + (g[y1 * k + x1] - g[y1 * k + x0]) * tx;
    top + (bottom - top) * ty
}

/// Robust spread of a sequence, as a median absolute deviation.
fn spread_of<I: Iterator<Item = f32>>(v: I) -> f32 {
    let mut all: Vec<f32> = v.collect();
    if all.is_empty() {
        return 0.0;
    }
    let m = sr_core::math::median(&all);
    for x in all.iter_mut() {
        *x = (*x - m).abs();
    }
    1.4826 * sr_core::math::median(&all)
}

fn block_spread(pairs: &[(f32, f32)]) -> f32 {
    let mut v: Vec<f32> = pairs.iter().map(|p| p.0).collect();
    if v.len() < 4 {
        return 0.0;
    }
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = v[(v.len() as f32 * 0.1) as usize];
    let hi = v[((v.len() - 1) as f32 * 0.9) as usize];
    hi - lo
}

fn median_of<F: Fn(&(f32, f32)) -> f32>(pairs: &[(f32, f32)], f: F) -> f32 {
    let v: Vec<f32> = pairs.iter().map(f).collect();
    sr_core::math::median(&v)
}

/// Everything matching disabled: each frame keeps whatever the exposure
/// metadata says and nothing else.
pub fn from_exposure_only(exposure_scale: &[f32]) -> Vec<FramePhotometry> {
    exposure_scale
        .iter()
        .map(|&s| FramePhotometry {
            map: PhotometricMatch::from_exposure(s),
            source: PhotometrySource::Disabled,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::cfa::CfaPattern;
    use sr_core::frame::{FrameMetadata, NoiseModel};
    use sr_core::geometry::GlobalTransform;
    use sr_core::samples::{DefectMask, Levels, SamplePlane};

    /// A frame whose sky sits at `sky` sensor codes, brightening across the
    /// field as light pollution does, with a scattering of point sources and a
    /// little noise. Mosaiced RGGB.
    ///
    /// The gradient is the part that matters: a gain is only identifiable from
    /// a scene whose blocks differ from one another, and a frame of uniform sky
    /// plus noise deliberately does not offer one.
    /// The same frame with a gradient laid across it, as a rising moon or a
    /// town on one horizon puts there. `shape` is given the position across
    /// the frame in 0..1 and returns the extra sky there, in sensor codes.
    fn frame_with_glow(
        sky: [u16; 3],
        gain: f32,
        seed: u64,
        shape: impl Fn(f32, f32) -> f32,
    ) -> RawFrame {
        let mut f = frame(sky, gain, seed);
        let (w, h) = (f.width, f.height);
        let mut data = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as f32 / (w - 1) as f32, y as f32 / (h - 1) as f32);
                let s = f.samples.value_in_cell(y * w + x, (y & 1) * 2 + (x & 1)) * 65535.0;
                data[y * w + x] = (s + shape(u, v)).clamp(0.0, 65535.0) as u16;
            }
        }
        f.samples = SamplePlane::from_u16(w, h, data, Levels::new([0.0; 4], [65535.0; 4]));
        f
    }

    /// How far apart two frames' skies still are at a place in the frame after
    /// each has had its own map applied, in sensor codes. This is the quantity
    /// the merge sees: it does not care what the common shape is, only that
    /// every frame has been brought to it.
    fn disagreement(
        a: &PhotometricMatch,
        sky_a: f32,
        b: &PhotometricMatch,
        sky_b: f32,
        u: f32,
        v: f32,
    ) -> f32 {
        let got_a = a.apply_at(1, sky_a / 65535.0, u, v);
        let got_b = b.apply_at(1, sky_b / 65535.0, u, v);
        (got_b - got_a) * 65535.0
    }

    #[test]
    fn translated_scene_blocks_do_not_invent_edge_photometry() {
        // The illumination is identical: only the sensor footprint moves.
        // Every retained paired median must therefore measure the same scene,
        // including near an edge where half a measurement block is off sensor.
        for (dx, dy) in [(-16, 0), (16, 0), (0, -16), (0, 16)] {
            let scene = |shift_x: i32, shift_y: i32| {
                let mut f = flat(1000, 1);
                let data = (0..f.height)
                    .flat_map(|y| {
                        (0..f.width).map(move |x| {
                            (10000 + 20 * (x as i32 - shift_x) + 12 * (y as i32 - shift_y)) as u16
                        })
                    })
                    .collect();
                f.samples = SamplePlane::from_u16(
                    f.width,
                    f.height,
                    data,
                    Levels::new([0.0; 4], [65535.0; 4]),
                );
                f
            };
            let reference = block_medians(&scene(0, 0), &[]);
            let offsets = vec![(dx, dy); reference.grid_w * reference.grid_h];
            let shifted = block_medians(&scene(dx, dy), &offsets);
            for c in 0..3 {
                let differences: Vec<f32> = reference.values[c].iter()
                    .zip(&shifted.values[c])
                    .filter_map(|(a, b)| Some((a.as_ref()? - b.as_ref()?).abs() * 65535.0))
                    .collect();
                assert!(differences.len() >= reference.values[c].len() / 2);
                let worst = differences.into_iter().fold(0.0f32, f32::max);
                assert!(worst < 1.0,
                    "same scene acquired a {worst:.1}-code median difference at shift ({dx},{dy}), channel {c}");
            }
        }
    }

    #[test]
    fn a_gradient_across_the_frame_is_matched_not_just_its_average() {
        // One frame has a gradient the other does not. A single offset can
        // only match the average of it, leaving half the frame too bright and
        // half too dark -- which the merge would then either reject as motion
        // or average in as signal.
        let reference = frame([1200, 1400, 1300], 1.0, 0xA11CE);
        let glow = |u: f32, _v: f32| 900.0 * u;
        let tilted = frame_with_glow([1200, 1400, 1300], 1.0, 0xA11CE, glow);
        let frames = vec![reference, tilted];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let out = match_burst(&frames, &warps, 0, &[1.0, 1.0]);

        let m = out[1].map;
        assert!(m.varies_across_frame(), "no gradient was found at all");
        // Corrected, the two frames should agree with each other at both
        // edges, whatever common shape they were brought to.
        let left = disagreement(&out[0].map, 1400.0, &m, 1400.0 + glow(0.0, 0.5), -1.0, 0.0);
        let right = disagreement(&out[0].map, 1400.0, &m, 1400.0 + glow(1.0, 0.5), 1.0, 0.0);
        assert!(
            left.abs() < 0.25 * 900.0 && right.abs() < 0.25 * 900.0,
            "the frames still differ by {left:.0} and {right:.0} codes of the 900 planted"
        );
    }

    #[test]
    fn large_stellar_gains_require_agreement_and_enough_finite_matches() {
        assert!(supported_stellar_gain(&[18.0; 29]).is_none());
        let mut coherent = vec![18.0; 34];
        coherent.extend([1.0, 2.0, 100.0, 200.0, 300.0, 400.0]);
        assert_eq!(supported_stellar_gain(&coherent), Some(18.0));
        let mut incoherent = vec![18.0; 20];
        incoherent.extend([36.0; 20]);
        assert!(supported_stellar_gain(&incoherent).is_none());
        let mut invalid = vec![18.0; 29];
        invalid.extend([f32::NAN, f32::INFINITY, 0.0, -1.0]);
        assert!(supported_stellar_gain(&invalid).is_none());
    }

    #[test]
    fn coherent_stars_match_large_gain_changes_in_both_directions() {
        let reference = frame([1200, 1400, 1300], 1.0, 0xA11CE);
        let mut faint = reference.clone();
        faint.samples = SamplePlane::from_normalised(reference.width, reference.height,
            (0..reference.width * reference.height)
                .map(|i| reference.value(i % reference.width, i / reference.width) / 18.0 + 0.03)
                .collect());
        let field: Vec<Star> = (0..40).map(|k| Star {
            x: 22.0 * (1 + k % 8) as f32 + 0.5,
            y: 22.0 * (1 + k / 8) as f32 + 0.5, flux: 1.0,
        }).collect();
        let faint_stars: Vec<_> = field.iter().map(|s| Star { flux: s.flux / 18.0, ..*s }).collect();
        let frames = vec![reference, faint];
        let stars = vec![field, faint_stars];
        let warps = vec![WarpField::identity(); 2];
        for reference in [0, 1] {
            let target = 1 - reference;
            let expected = if reference == 0 { 18.0 } else { 1.0 / 18.0 };
            let fits = match_with_stars(&frames, &warps, reference, &[1.0; 2], &[true; 2], &stars, false);
            assert!((fits[target].map.gain[0] / expected - 1.0).abs() < 1e-5,
                "valid gain became a level-only fallback: {}", fits[target].map.gain[0]);
            for (x, y) in [(5, 5), (90, 90), (210, 170)] {
                let mapped = fits[target].map.apply_at(0, frames[target].value(x, y), 0.0, 0.0);
                assert!((mapped - frames[reference].value(x, y)).abs() < 2e-5);
            }
        }
    }

    #[test]
    fn twilight_is_a_pedestal_and_the_stars_say_so() {
        // Dusk frames: the same stars through the same air, on a
        // sky that is much brighter. Block medians alone read that as a gain
        // of about a half; the stars say the transparency did not change.
        // Twilight is seen through the same vignetting as everything else, so
        // a brighter sky is brighter *proportionally*: its blocks spread wider
        // in exactly the ratio of the sky levels. A line through them then has
        // that ratio for its slope, and reports it as transparency.
        let reference = frame([1200, 1400, 1300], 1.0, 0xA11CE);
        let dusk = frame([2100, 2450, 2275], 1.0, 0xA11CE);
        let frames = vec![reference, dusk];
        let warps = vec![WarpField::identity(), WarpField::identity()];

        let blocks_only = match_burst(&frames, &warps, 0, &[1.0, 1.0]);
        // The same forty stars in both frames, at the same flux: the air did
        // not change, only the sky behind them. Built here rather than
        // detected, because the detector is a separate thing with its own
        // tests and this one is about what the ratios say.
        // On the frame helper's own stars, which sit in 2x2 blocks every
        // eleven mosaic cells: the gain is measured from the frames at these
        // positions, so they have to be where the light is.
        let field: Vec<Star> = (0..40)
            .map(|k| Star {
                x: 22.0 * (1 + k % 8) as f32 + 0.5,
                y: 22.0 * (1 + k / 8) as f32 + 0.5,
                flux: 1.0,
            })
            .collect();
        let stars = vec![field.clone(), field];
        // The detector is not run here, so the gain comes from the frames
        // themselves at these positions -- which is what the real path does.
        let with_stars =
            match_with_stars(&frames, &warps, 0, &[1.0, 1.0], &[true; 2], &stars, true);

        let blind = blocks_only[1].map.gain[1];
        let seen = with_stars[1].map.gain[1];
        assert!(
            blind < 0.7,
            "the block fit was supposed to read the sky ratio, and read {blind:.3}"
        );
        assert!(
            (seen - 1.0).abs() < 0.06,
            "the stars read the twilight as a gain of {seen:.3}"
        );
        // And the sky is still matched: the pedestal carries the whole of it.
        let left = with_stars[1].map.apply_at(1, 2450.0 / 65535.0, 0.0, 0.0)
            - 1400.0 / 65535.0;
        assert!(
            left.abs() * 65535.0 < 400.0,
            "the sky is still {:.0} codes out after the match",
            left.abs() * 65535.0
        );
    }

    #[test]
    fn clipped_edges_compare_the_same_sky_before_masking_obstructions() {
        let size = 512;
        for cfa in [CfaPattern::MONO, CfaPattern::RGGB] {
            for edge in 0..4 {
                for obstructed in [false, true] {
                    for noise in [0.0_f32, 0.012] {
                        let (dx, dy) = match edge {
                            0 => (-12.0, 0.0), 1 => (12.0, 0.0),
                            2 => (0.0, -12.0), _ => (0.0, 12.0),
                        };
                        let mut frames = Vec::new();
                        let mut warps = vec![WarpField::identity(); 4];
                        warps[3].global = GlobalTransform::translation(dx, dy);
                        let mut stars = Vec::new();
                        for (i, warp) in warps.iter().enumerate() {
                            let gain = if i == 3 { 0.06 } else { 1.0 };
                            let mut f = flat(1000, i as u64 + 1);
                            f.width = size; f.height = size; f.cfa = cfa;
                            f.defects = DefectMask::none(size, size);
                            let mut values = Vec::new();
                            let mut rng = (i as u64 + 1) * 0x1234567;
                            for y in 0..size {
                                for x in 0..size {
                                    let (rx, ry) = warp.map(x as f32, y as f32);
                                    let (distance, along) = match edge {
                                        0 => (size as f32 - 1.0 - rx, ry),
                                        1 => (rx, ry),
                                        2 => (size as f32 - 1.0 - ry, rx),
                                        _ => (ry, rx),
                                    };
                                    // The clipped/full medians differ even in the
                                    // clean case: a bright real feature straddles
                                    // the missing footprint. Compare common sky.
                                    let scene = 0.02 + 0.002 * along / size as f32
                                        + if distance < 32.0 { 0.01 } else { 0.0 };
                                    let shadow = if i == 3 && obstructed && distance < 64.0 && along < 64.0 {
                                        0.012
                                    } else { 0.0 };
                                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                                    let random = (((rng >> 40) & 0xffff) as f32 / 32767.5 - 1.0) * noise;
                                    values.push((scene - shadow + random) / gain + if i == 0 { 0.0078 } else { 0.0 });
                                }
                            }
                            f.noise = NoiseModel::new(0.0, (noise / gain).powi(2) / 3.0, sr_core::frame::NoiseSource::Manual);
                            f.samples = SamplePlane::from_normalised(size, size, values);
                            frames.push(f);
                            stars.push((0..36).map(|s| {
                                let (x, y) = warp.inverse_map(60.0 + (s % 6) as f32 * 60.0,
                                    60.0 + (s / 6) as f32 * 60.0).unwrap();
                                Star { x, y, flux: 100.0 / gain }
                            }).collect());
                        }
                        let out = match_with_stars(&frames, &warps, 0, &[1.0; 4], &[], &stars, true);
                        let (u, v) = match edge { 0 => (0.95, -0.95), 1 => (-0.95, -0.95),
                            2 => (-0.95, 0.95), _ => (-0.95, -0.95) };
                        assert_eq!(out[3].map.blocked_at(u, v), obstructed,
                            "edge {edge}, CFA {cfa:?}, obstructed {obstructed}, noise {noise}");
                        for m in &out[..3] { assert_eq!(m.map.blocked_fraction(), 0.0, "clean peer masked"); }
                        if !obstructed { assert_eq!(out[3].map.blocked_fraction(), 0.0, "partial real feature masked"); }
                    }
                }
            }
        }
    }

    #[test]
    fn clipped_edge_peer_scatter_is_not_averaged_away() {
        // Many pixels cannot turn disagreement between whole peer regions
        // into convincing obstruction evidence. Only agreeing peers can.
        for spread in [0.0, 0.005] {
            let frames: Vec<_> = [0.02 - spread, 0.02, 0.02 + spread, 0.012]
                .into_iter().enumerate().map(|(i, level)| {
                    let mut f = flat(1000, i as u64 + 1);
                    f.cfa = CfaPattern::MONO;
                    f.samples = SamplePlane::from_normalised(256, 256, vec![level; 256 * 256]);
                    f
                }).collect();
            let mut blocks: Vec<_> = frames.iter().map(|f| block_medians(f, &[])).collect();
            blocks[3].partial[7] = true;
            let maps = vec![FramePhotometry { map: PhotometricMatch::IDENTITY, source: PhotometrySource::Measured }; 4];
            let mut warps = vec![WarpField::identity(); 4];
            warps[3].global = GlobalTransform::translation(-12.0, 0.0);
            let found = clipped_obstructions(&frames.iter().collect::<Vec<_>>(), &warps,
                &blocks, &maps, &[true; 4], &vec![vec![false; 64]; 4]);
            assert_eq!(found.contains(&(3, 7)), spread == 0.0);
        }
    }

    #[test]
    fn mixed_gain_obstructions_do_not_depend_on_reference_sky_pedestal() {
        // SII regression: a gain-300 frame mapped by ~0.06 onto a gain-50
        // reference receives a large additive sky offset. That offset must
        // not make the very same foreground shadow disappear from the mask.
        let size = 512;
        let mut previous = None;
        for pedestal in [0.0, 0.0078, 0.04] {
            let mut frames = Vec::new();
            let mut stars = Vec::new();
            for i in 0..5 {
                let gain: f32 = if i == 0 { 1.0 } else { 0.06 };
                let mut frame = flat(1000, i as u64 + 1);
                frame.width = size;
                frame.height = size;
                frame.cfa = CfaPattern::MONO;
                frame.defects = DefectMask::none(size, size);
                frame.noise = NoiseModel::new(0.0, (0.000001 / gain).powi(2),
                    sr_core::frame::NoiseSource::Manual);
                let mut values = Vec::with_capacity(size * size);
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        // Shared diffuse emission and a dark cloud are real
                        // sky structure, and must remain unmasked.
                        let cloud = if u < 0.25 && v < 0.25 { -0.0003 } else { 0.0 };
                        let scene = 0.0015 + cloud
                            + 0.00015 * (-((u - 0.4).powi(2) + (v - 0.5).powi(2)) / 0.07).exp();
                        let gradient = if i == 3 { 0.00015 * (u - 0.5) } else { 0.0 };
                        let shadow = if i == 4 && u > 0.65 && v > 0.65 { -0.0008 } else { 0.0 };
                        let noise = ((x * 17 + y * 31 + i * 13) % 11) as f32 * 0.0000002;
                        values.push((scene + gradient + shadow + noise) / gain
                            + if i == 0 { pedestal } else { 0.0 });
                    }
                }
                frame.samples = SamplePlane::from_normalised(size, size, values);
                frames.push(frame);
                stars.push((0..36).map(|s| Star {
                    x: 40.0 + (s % 6) as f32 * 65.0,
                    y: 40.0 + (s / 6) as f32 * 65.0,
                    flux: 100.0 / gain,
                }).collect());
            }
            let maps = match_with_stars(&frames, &vec![WarpField::identity(); 5],
                0, &[1.0; 5], &[], &stars, true);
            let mask = maps[4].map.blocked;
            assert!(maps[4].map.blocked_at(0.9, 0.9),
                "shadow missed at reference pedestal {pedestal}");
            assert!(!maps[4].map.blocked_at(-0.8, -0.8), "shared dark cloud was masked");
            for (i, m) in maps[..4].iter().enumerate() {
                assert_eq!(m.map.blocked_fraction(), 0.0,
                    "clean frame/gradient {i} masked at pedestal {pedestal}");
            }
            if let Some(before) = previous {
                assert_eq!(mask, before, "reference pedestal changed the obstruction mask");
            }
            previous = Some(mask);
            if pedestal == 0.0 {
                // Even unanimous clean peers cannot make an uncertain source
                // block precise. The per-frame median-noise floor must veto
                // a deficit that is below that source's own uncertainty.
                frames[4].noise = NoiseModel::new(0.0, (0.05_f32 / 0.06).powi(2),
                    sr_core::frame::NoiseSource::Manual);
                let uncertain = match_with_stars(&frames, &vec![WarpField::identity(); 5],
                    0, &[1.0; 5], &[], &stars, true);
                assert_eq!(uncertain[4].map.blocked_fraction(), 0.0,
                    "a sub-noise source-block deficit became an obstruction veto");
            }
        }
    }

    #[test]
    fn a_dark_corner_is_an_obstruction_not_a_gradient() {
        // A tree in the corner of one frame: half the sky gone over a sixth of
        // the frame. The field must not follow it, and the cells must be
        // marked so that the merge gives those samples to nobody.
        let reference = frame([1200, 1400, 1300], 1.0, 0xA11CE);
        let clean = frame([1200, 1400, 1300], 1.0, 0xC1EA);
        let tree = |u: f32, v: f32| if u > 0.6 && v > 0.6 { -700.0 } else { 0.0 };
        let dark = frame_with_glow([1200, 1400, 1300], 1.0, 0xBEEF, tree);
        let frames = vec![reference, clean, dark];
        let warps = vec![WarpField::identity(); 3];
        let out = match_burst(&frames, &warps, 0, &[1.0; 3]);
        let m = &out[2].map;
        assert!(m.blocked_at(0.9, 0.9), "the corner under the tree was not marked");
        assert!(!m.blocked_at(-0.9, -0.9), "the far corner was marked");
        assert!(!m.blocked_at(-0.3, -0.3), "the clear side of the frame was marked");
        let frac = m.blocked_fraction();
        assert!(
            (0.04..0.40).contains(&frac),
            "{:.0}% of the frame was marked for a tree over a sixth of it",
            frac * 100.0
        );
        // The field did not chase the tree: whatever it fitted is small.
        for c in 0..3 {
            let amp = m.field_amplitude(c) * 65535.0;
            assert!(amp < 120.0, "channel {c} fitted {amp:.0} codes of field to a tree");
        }
        // And the clean frames are clean.
        assert_eq!(out[0].map.blocked_fraction(), 0.0);
        assert_eq!(out[1].map.blocked_fraction(), 0.0);
    }

    #[test]
    fn obstructed_cells_cannot_define_the_common_sky_shape() {
        // Bad exposures can be a majority in a corner even when the clear
        // exposures still determine the sky there. Extrapolated fits through
        // an obstruction must not be transferred into every clean frame.
        let mut fits = Vec::new();
        for (i, amplitude) in [0.0, 0.002, 0.2, 0.3, 0.4].into_iter().enumerate() {
            let mut map = PhotometricMatch::IDENTITY;
            for c in 0..3 {
                for y in 0..FIELD {
                    for x in 0..FIELD {
                        map.field[c][y][x] = amplitude * (2.0 * x as f32 / (FIELD - 1) as f32 - 1.0);
                    }
                }
            }
            if i >= 2 { map.blocked[FIELD - 1][FIELD - 1] = true; }
            fits.push(FramePhotometry { map, source: PhotometrySource::Measured });
        }
        take_the_burst_shape(&mut fits, &[true; 5]);
        let corner = fits[0].map.field[1][FIELD - 1][FIELD - 1];
        assert!(corner.abs() <= 0.0021,
            "obstructed fits imposed {corner} on the clean reference sky");
        // The input fields are planes; excluding a contributor at just one
        // node would create a kink at the obstruction boundary.
        for row in &fits[0].map.field[1] {
            for triple in row.windows(3) {
                assert!((triple[0] - 2.0 * triple[1] + triple[2]).abs() < 1e-6,
                    "common sky acquired a kink at an obstruction boundary");
            }
        }
    }

    #[test]
    fn crossing_sky_planes_cannot_create_a_ridge_in_the_common_shape() {
        let mut fits = Vec::new();
        for (a, b) in [(0.002, 0.0), (0.0, 0.002), (-0.002, -0.002)] {
            let mut map = PhotometricMatch::IDENTITY;
            for plane in &mut map.field {
                for (y, row) in plane.iter_mut().enumerate() {
                    for (x, value) in row.iter_mut().enumerate() {
                        *value = a * (2.0 * x as f32 / (FIELD - 1) as f32 - 1.0)
                            + b * (2.0 * y as f32 / (FIELD - 1) as f32 - 1.0);
                    }
                }
            }
            fits.push(FramePhotometry { map, source: PhotometrySource::Measured });
        }
        let before = fits.clone();
        take_the_burst_shape(&mut fits, &[true; 3]);
        for plane in &fits[0].map.field {
            for y in 1..FIELD-1 {
                for x in 1..FIELD-1 {
                    assert!((plane[y][x-1] - 2.0*plane[y][x] + plane[y][x+1]).abs() < 1e-8,
                        "independent node medians introduced a horizontal ridge");
                    assert!((plane[y-1][x] - 2.0*plane[y][x] + plane[y+1][x]).abs() < 1e-8,
                        "independent node medians introduced a vertical ridge");
                    let original_difference = before[0].map.field[1][y][x] - before[1].map.field[1][y][x];
                    let final_difference = fits[0].map.field[1][y][x] - fits[1].map.field[1][y][x];
                    assert!((original_difference-final_difference).abs() < 1e-8);
                }
            }
        }
    }

    #[test]
    fn the_stack_takes_the_shape_of_the_typical_frame_not_the_reference() {
        // Two frames carry the same tilt and the reference does not. Matched
        // onto the reference, the stack would be flat and the reference's
        // accident of a sky would be everyone's. Matched onto the typical
        // frame, it is the reference that is brought to the others.
        let reference = frame([1200, 1400, 1300], 1.0, 0xA11CE);
        let glow = |u: f32, _v: f32| 900.0 * u;
        let t1 = frame_with_glow([1200, 1400, 1300], 1.0, 0xBEEF, glow);
        let t2 = frame_with_glow([1200, 1400, 1300], 1.0, 0xCAFE, glow);
        let frames = vec![reference, t1, t2];
        let warps = vec![WarpField::identity(); 3];
        let out = match_burst(&frames, &warps, 0, &[1.0; 3]);

        // The reference's own map now tilts its flat sky up towards where the
        // others are brighter.
        let r = &out[0].map;
        let rise = (r.apply_at(1, 1400.0 / 65535.0, 1.0, 0.0)
            - r.apply_at(1, 1400.0 / 65535.0, -1.0, 0.0))
            * 65535.0;
        assert!(
            rise > 0.5 * 900.0,
            "the reference was tilted by only {rise:.0} codes towards the typical frame"
        );
        // And the tilted frames are left nearly alone.
        let t = &out[1].map;
        let kept = (t.apply_at(1, (1400.0 + 900.0) / 65535.0, 1.0, 0.0)
            - t.apply_at(1, 1400.0 / 65535.0, -1.0, 0.0))
            * 65535.0;
        assert!(
            (kept - 900.0).abs() < 0.35 * 900.0,
            "a typical frame had its tilt changed to {kept:.0} codes"
        );
        // Everyone still agrees.
        for u in [-1.0f32, 0.0, 1.0] {
            let d = disagreement(r, 1400.0, t, 1400.0 + glow(0.5 * (u + 1.0), 0.5), u, 0.0);
            assert!(d.abs() < 0.25 * 900.0, "frames disagree by {d:.0} codes at u = {u}");
        }
    }

    #[test]
    fn a_glow_that_is_not_a_plane_is_still_taken_out() {
        // The reason this is a field and not a plane. Moonlight falls off from
        // where the moon is, so its gradient is curved; a plane takes the tilt
        // out and leaves the curvature, and the curvature is what the merge
        // then rejects as disagreement.
        //
        // The glow here is quadratic across the frame. The best straight line
        // through a parabola leaves a sixth of its amplitude behind -- 200 of
        // these 1200 codes, at both edges and in the middle -- so anything
        // comfortably under that is doing something a plane could not.
        let reference = frame([1200, 1400, 1300], 1.0, 0x5EED);
        let glow = |u: f32, _v: f32| 1200.0 * u * u;
        let lit = frame_with_glow([1200, 1400, 1300], 1.0, 0x5EED, glow);
        let frames = vec![reference, lit];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let out = match_burst(&frames, &warps, 0, &[1.0, 1.0]);
        let m = out[1].map;
        assert!(m.varies_across_frame(), "no glow was found at all");

        // Sampled over the whole frame, not just the edges: a line can be made
        // to agree at two points and be wrong everywhere between them.
        let mut worst = 0.0f32;
        for i in 0..9 {
            for j in 0..5 {
                let (u, v) = (i as f32 / 8.0, j as f32 / 4.0);
                // Against the other frame's corrected sky, not against the
                // flat one: the two are brought to a common shape, and that
                // shape is not the reference's.
                let e = disagreement(
                    &out[0].map,
                    1400.0,
                    &m,
                    1400.0 + glow(u, v),
                    2.0 * u - 1.0,
                    2.0 * v - 1.0,
                );
                worst = worst.max(e.abs());
            }
        }
        assert!(
            worst < 120.0,
            "the worst place in the frame is still {worst:.0} codes of the 1200 \
             planted, and a plane would already have got that to 200"
        );
    }

    #[test]
    fn no_gradient_is_invented_where_there_is_none() {
        // The guard that matters. A field fitted to noise is not neutral: it
        // is structure invented and then imposed on every pixel.
        let a = frame([1200, 1400, 1300], 1.0, 0xBEEF);
        let b = frame([1200, 1400, 1300], 1.0, 0xC0FFEE);
        let frames = vec![a, b];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let out = match_burst(&frames, &warps, 0, &[1.0, 1.0]);
        for c in 0..3 {
            let amplitude = out[1].map.field_amplitude(c) * 65535.0;
            assert!(
                amplitude < 20.0,
                "channel {c} invented a gradient of {amplitude:.0} codes across the frame"
            );
        }
    }

    #[test]
    fn the_field_does_not_swallow_what_is_in_both_frames() {
        // A field fitted to one frame's background would absorb real structure.
        // This one is fitted to the difference between two frames of the same
        // sky, so anything present in both cancels before the fit sees it --
        // and that is what makes a grid safe here. Planting the same broad
        // structure in both frames must leave the field flat.
        let shape = |u: f32, v: f32| 2500.0 * ((u - 0.5).powi(2) + (v - 0.5).powi(2));
        let a = frame_with_glow([1200, 1400, 1300], 1.0, 0xF00D, shape);
        let b = frame_with_glow([1200, 1400, 1300], 1.0, 0xD00D, shape);
        let frames = vec![a, b];
        let warps = vec![WarpField::identity(), WarpField::identity()];
        let out = match_burst(&frames, &warps, 0, &[1.0, 1.0]);
        for c in 0..3 {
            let amplitude = out[1].map.field_amplitude(c) * 65535.0;
            assert!(
                amplitude < 60.0,
                "channel {c} took {amplitude:.0} codes of a structure that was in both frames"
            );
        }
    }

    #[test]
    fn a_field_survives_being_composed_and_undone() {
        let mut m = PhotometricMatch {
            gain: [1.1, 0.9, 1.3],
            log_gain: None,
            offset: [0.01, -0.02, 0.005],
            field: [[[0.0; FIELD]; FIELD]; 3],
            blocked: [[false; FIELD]; FIELD],
        };
        for c in 0..3 {
            for y in 0..FIELD {
                for x in 0..FIELD {
                    m.field[c][y][x] =
                        0.004 * ((c + 1) as f32) * ((x as f32) - 3.5) / 3.5
                            - 0.002 * ((y as f32) - 3.5) / 3.5;
                }
            }
        }
        let back = m.inverse().expect("gains are healthy");
        let round = m.compose(&back);
        for c in 0..3 {
            assert!((round.gain[c] - 1.0).abs() < 1e-5, "gain {c}");
            assert!(round.offset[c].abs() < 1e-5, "offset {c}");
            assert!(
                round.field_amplitude(c) < 1e-5,
                "field {c} left {}",
                round.field_amplitude(c)
            );
        }
        // And a value away from the centre comes back to itself.
        for &(u, v) in &[(0.0, 0.0), (1.0, 1.0), (-1.0, 0.6), (0.3, -0.8)] {
            let there = m.apply_at(1, 0.3, u, v);
            let home = back.apply_at(1, there, u, v);
            assert!((home - 0.3).abs() < 1e-5, "at {u},{v} came back {home}");
        }
    }

    #[test]
    fn the_field_is_read_as_a_surface_not_as_cells() {
        // Read by nearest cell, the correction would arrive in steps and put
        // its own grid into the background it exists to smooth.
        let mut m = PhotometricMatch::IDENTITY;
        for c in 0..3 {
            for y in 0..FIELD {
                for x in 0..FIELD {
                    m.field[c][y][x] = x as f32;
                }
            }
        }
        let row: Vec<f32> = (0..29).map(|i| m.field_at(0, -1.0 + i as f32 / 14.0, 0.0)).collect();
        let steps: Vec<f32> = row.windows(2).map(|w| w[1] - w[0]).collect();
        let smallest = steps.iter().cloned().fold(f32::INFINITY, f32::min);
        let largest = steps.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(smallest > 0.0, "the field is flat somewhere along a ramp: {row:?}");
        assert!(
            largest - smallest < 0.02 * largest,
            "the field arrives in steps of {smallest} to {largest}, which is a staircase"
        );
        // And a cell centre still reads its own value.
        for x in 0..FIELD {
            let u = 2.0 * x as f32 / (FIELD - 1) as f32 - 1.0;
            assert!((m.field_at(0, u, -1.0) - x as f32).abs() < 1e-5, "cell {x}");
        }
    }

    fn frame(sky: [u16; 3], gain: f32, seed: u64) -> RawFrame {
        let (w, h) = (256usize, 256usize);
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) & 0x3fff) as i32) - 8192
        };
        let mut data = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                let c = CfaPattern::RGGB.color_at(x, y).index();
                let ramp = 1.0 + 0.6 * (x + y) as f32 / (w + h) as f32;
                let star = if (x / 2) % 11 == 0 && (y / 2) % 11 == 0 { 6000.0 } else { 0.0 };
                let v = (sky[c] as f32 * ramp + star) * gain + rnd() as f32 * 0.02;
                data[y * w + x] = v.clamp(0.0, 65535.0) as u16;
            }
        }
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_u16(w, h, data, Levels::new([0.0; 4], [65535.0; 4])),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            // Roughly the shot noise of a 14-bit sensor, so the identifiability
            // guard is exercised against a realistic figure.
            noise: NoiseModel::new(6e-5, 1e-9, sr_core::frame::NoiseSource::Manual),
            metadata: FrameMetadata::default(),
        }
    }

    /// A frame of featureless sky: no gradient, no sources, nothing a gain
    /// could be fitted to.
    fn flat(level: u16, seed: u64) -> RawFrame {
        let (w, h) = (256usize, 256usize);
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) & 0x3fff) as i32) - 8192
        };
        let data: Vec<u16> = (0..w * h)
            .map(|_| (level as f32 + rnd() as f32 * 0.02).clamp(0.0, 65535.0) as u16)
            .collect();
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_u16(w, h, data, Levels::new([0.0; 4], [65535.0; 4])),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            noise: NoiseModel::new(6e-5, 1e-9, sr_core::frame::NoiseSource::Manual),
            metadata: FrameMetadata::default(),
        }
    }

    /// Sky with structure that varies block to block, so that moving the frame
    /// by most of a block genuinely changes what each block is looking at. The
    /// smooth ramp of [`frame`] does not: shifting it leaves the block medians
    /// almost where they were, which would let the pairing test pass without
    /// testing anything.
    ///
    /// `pan` slides the scene: `patchy(s, k, 24)` holds at `x` what
    /// `patchy(s, k, 0)` holds at `x + 24`. Generated rather than resampled, so
    /// there is no clamped edge band to confuse the comparison.
    fn patchy(sky: [u16; 3], seed: u64, pan: i32, gain: f32) -> RawFrame {
        let (w, h) = (256usize, 256usize);
        let mut data = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                let c = CfaPattern::RGGB.color_at(x, y).index();
                // A hash per 32-pixel cell: uncorrelated between neighbours,
                // identical between frames of the same scene.
                let sx = (x as i32 + pan).rem_euclid(4096) as u64;
                let cell = (sx / 32) * 977 + (y as u64 / 32) * 5171 + seed;
                // A proper finalizer, not one multiply. Multiplying a small
                // counter by a constant leaves the high bits nearly linear in
                // it, which makes a "random" field a smooth ramp — and a ramp
                // is exactly the scene a mis-paired regression can still fit,
                // so the test would pass while proving nothing.
                let mut z = cell.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let f = (z & 0x3ff) as f32 / 1023.0;
                let v = sky[c] as f32 * (0.6 + 0.8 * f) * gain;
                data[y * w + x] = v.clamp(0.0, 65535.0) as u16;
            }
        }
        RawFrame {
            width: w,
            height: h,
            samples: SamplePlane::from_u16(w, h, data, Levels::new([0.0; 4], [65535.0; 4])),
            cfa: CfaPattern::RGGB,
            defects: DefectMask::none(w, h),
            noise: NoiseModel::new(6e-5, 1e-9, sr_core::frame::NoiseSource::Manual),
            metadata: FrameMetadata::default(),
        }
    }

    /// Identity warps, for the tests where the burst did not move.
    fn identity(n: usize) -> Vec<WarpField> {
        (0..n).map(|_| WarpField::identity()).collect()
    }

    /// The same frame with a constant added to every site: a sky that
    /// brightened without the scene changing.
    fn with_pedestal(f: &RawFrame, add: u16) -> RawFrame {
        let mut out = f.clone();
        let data: Vec<u16> = (0..f.width * f.height)
            .map(|i| match &f.samples.data {
                sr_core::samples::SampleData::U16(v) => v[i].saturating_add(add),
                _ => 0,
            })
            .collect();
        out.samples =
            SamplePlane::from_u16(f.width, f.height, data, Levels::new([0.0; 4], [65535.0; 4]));
        out
    }

    #[test]
    fn identical_frames_need_no_correction() {
        let frames = vec![frame([3700, 5300, 4900], 1.0, 1), frame([3700, 5300, 4900], 1.0, 2)];
        let m = match_burst(&frames, &identity(2), 0, &[1.0, 1.0]);
        assert_eq!(m[1].source, PhotometrySource::Measured);
        for c in 0..3 {
            assert!((m[1].map.gain[c] - 1.0).abs() < 0.01, "gain {}", m[1].map.gain[c]);
            assert!(m[1].map.offset[c].abs() < 0.002, "offset {}", m[1].map.offset[c]);
        }
    }

    #[test]
    fn a_brighter_sky_is_measured_as_a_pedestal() {
        // This is the deep-sky failure this module exists for: the sky rises by
        // 800 codes over the session and nothing else changes.
        let a = frame([3700, 5300, 4900], 1.0, 3);
        let b = with_pedestal(&a, 800);
        let m = match_burst(&[a, b], &identity(2), 0, &[1.0, 1.0]);
        let map = m[1].map;
        assert_eq!(m[1].source, PhotometrySource::Measured);
        for c in 0..3 {
            assert!((map.gain[c] - 1.0).abs() < 0.01, "gain {}", map.gain[c]);
            let expected = -800.0 / 65535.0;
            assert!(
                (map.offset[c] - expected).abs() < 0.0005,
                "offset {} wanted {expected}",
                map.offset[c]
            );
        }
        // Applying it must bring the frame back onto the anchor's scale.
        let sky_b = (5300.0 + 800.0) / 65535.0;
        assert!((map.apply(1, sky_b) - 5300.0 / 65535.0).abs() < 0.001);
    }

    #[test]
    fn a_dimmer_frame_is_measured_as_a_gain() {
        let a = frame([3700, 5300, 4900], 1.0, 5);
        let b = frame([3700, 5300, 4900], 0.8, 6);
        let m = match_burst(&[a, b], &identity(2), 0, &[1.0, 1.0]);
        for c in 0..3 {
            assert!(
                (m[1].map.gain[c] - 1.25).abs() < 0.02,
                "channel {c} gain {}",
                m[1].map.gain[c]
            );
        }
    }

    #[test]
    fn the_reference_is_whichever_frame_was_named() {
        let frames = vec![
            frame([3700, 5300, 4900], 1.0, 7),
            frame([3700, 5300, 4900], 0.9, 8),
            frame([3700, 5300, 4900], 0.8, 9),
        ];
        let m = match_burst(&frames, &identity(3), 1, &[1.0; 3]);
        for c in 0..3 {
            assert!((m[1].map.gain[c] - 1.0).abs() < 1e-4);
            assert!(m[1].map.offset[c].abs() < 1e-5);
        }
        // Frame 2 is 0.8 of frame 0 and frame 1 is 0.9, so onto frame 1 it
        // needs 0.9 / 0.8.
        assert!(
            (m[2].map.gain[1] - 0.9 / 0.8).abs() < 0.03,
            "gain {}",
            m[2].map.gain[1]
        );
    }

    #[test]
    fn blocks_follow_the_scene_rather_than_the_sensor() {
        // A burst that panned by most of a block width. Paired by sensor
        // position the two frames' blocks look at different scene and the fit
        // measures the pan; paired through the warp they agree.
        let shift = 24.0f32;
        let a = patchy([3700, 5300, 4900], 31, 0, 1.0);
        // Panned *and* 20% brighter, so there is a right answer for the fit to
        // get: carrying b onto a needs a gain of 1 / 1.2.
        let b = patchy([3700, 5300, 4900], 31, shift as i32, 1.2);
        // `b` holds at x what `a` holds at x + 24, so carrying b onto a is a
        // translation of +24: the convention is frame to reference.
        let warps = vec![
            WarpField::identity(),
            WarpField { global: GlobalTransform::translation(shift, 0.0), local: None },
        ];
        let aligned = match_burst(&[a.clone(), b.clone()], &warps, 0, &[1.0, 1.0]);
        let naive = match_burst(&[a, b], &identity(2), 0, &[1.0, 1.0]);
        let want = 1.0 / 1.2;
        for c in 0..3 {
            assert!(
                (aligned[1].map.gain[c] - want).abs() < 0.02,
                "channel {c} gain {} with the warp, wanted {want}",
                aligned[1].map.gain[c]
            );
        }
        // Paired by sensor position instead, the blocks are looking at
        // different scene and the gain is not recovered.
        let best = (0..3)
            .map(|c| (naive[1].map.gain[c] - want).abs())
            .fold(f32::MAX, f32::min);
        assert!(best > 0.05, "sensor-paired fit was off by only {best}, so this proves nothing");
    }

    #[test]
    fn an_implausible_gain_degrades_to_matching_the_level() {
        // A frame ten times brighter than the anchor is beyond anything a
        // change of illumination explains within one burst, so the gain is
        // refused. Matching the level is still right, and still leaves the
        // frame in a state motion rejection can reason about locally.
        let frames = vec![frame([3700, 5300, 4900], 1.0, 11), frame([3700, 5300, 4900], 10.0, 12)];
        let m = match_burst(&frames, &identity(2), 0, &[1.0, 0.25]);
        assert_eq!(m[1].source, PhotometrySource::Level);
        assert_eq!(m[1].map.gain, [1.0; 3]);
        assert!(m[1].map.offset[1] < -0.4, "offset {}", m[1].map.offset[1]);
    }

    #[test]
    fn a_featureless_frame_gets_a_level_match_and_no_gain() {
        // The case that makes whole-frame quantile matching unusable: two
        // frames of blank sky differing only by a pedestal. Their spread is
        // noise, the noise is independent, and any gain fitted to it is
        // fiction. The level still has to be matched.
        let a = flat(20000, 21);
        let b = flat(21000, 22);
        let m = match_burst(&[a, b], &identity(2), 0, &[1.0, 1.0]);
        assert_eq!(m[1].source, PhotometrySource::Level);
        for c in 0..3 {
            assert_eq!(m[1].map.gain[c], 1.0, "channel {c} invented a gain");
            let expected = -1000.0 / 65535.0;
            assert!(
                (m[1].map.offset[c] - expected).abs() < 0.0008,
                "channel {c} offset {} wanted {expected}",
                m[1].map.offset[c]
            );
        }
    }

    #[test]
    fn composition_and_inversion_agree() {
        let m = PhotometricMatch {
            gain: [1.1, 0.9, 1.3],
            log_gain: None,
            offset: [0.01, -0.02, 0.005],
            field: [[[0.0; FIELD]; FIELD]; 3],
            blocked: [[false; FIELD]; FIELD],
        };
        let round = m.inverse().unwrap().compose(&m);
        for c in 0..3 {
            assert!((round.gain[c] - 1.0).abs() < 1e-5);
            assert!(round.offset[c].abs() < 1e-6);
        }
    }
}
