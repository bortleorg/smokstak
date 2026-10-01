//! Refining a fitted transform against the stars themselves.
//!
//! Correlation places a frame by matching patches of it against the reference,
//! which is the right tool when the scene is texture and there is nothing else
//! to go on. On a star field there is something else: hundreds of points whose
//! centres can be measured to a fraction of a pixel, individually, without
//! reference to any patch.
//!
//! The difference is not small. On a 96-frame burst, the stacked point spread
//! function was 3.4 pixels across where a single frame's is 2.2: the stack was
//! not preserving the frames' own sharpness. Taking the kernel's contribution out in
//! quadrature leaves about a pixel of scatter in where we were putting each
//! frame, which is what a correlation residual of 0.4 *proxy* pixels means at
//! sensor scale.
//!
//! So the correlation result is kept as the starting point -- it is what finds
//! the frame at all, through a meridian flip or a hundred pixels of drift --
//! and the stars are used to polish it.

use rayon::prelude::*;

use sr_core::geometry::GlobalTransform;
use sr_core::star::Star;

/// How near a mapped star must land to a reference star to be called the same
/// star, in sensor pixels.
///
/// Generous, because the transform being refined is the one that was wrong: at
/// a pixel of error the true match is a pixel away, and a tolerance tighter
/// than the error finds nothing. Wrong pairings at this radius are rare on a
/// field whose stars are further apart than this, and the fit is robust to the
/// ones that happen.
const MATCH_RADIUS: f32 = 4.0;

/// Pairs needed before a correction is fitted at all. Six is the minimum for an
/// affine; this is well above it, so the fit is over-determined enough that no
/// single pairing decides it.
const MIN_PAIRS: usize = 25;

/// The correction must bring the pairs at least this much closer together, or
/// it is discarded. A refinement that does not refine is a refinement fitted to
/// mismatches.
const MUST_IMPROVE: f32 = 0.95;

/// How far a correction may move a star before it is disbelieved, in sensor
/// pixels. A polish is a fraction of a pixel; anything larger means the pairs
/// are not the stars we think they are.
const MAX_CORRECTION: f32 = 6.0;

/// What one frame's refinement did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Refinement {
    pub pairs: usize,
    /// Median pair separation before and after, in sensor pixels.
    pub before: f32,
    pub after: f32,
    /// The furthest any star was moved by the fitted correction.
    pub worst: f32,
    pub applied: bool,
}

/// Polish each frame's transform against the reference's stars.
///
/// `stars[i]` are frame `i`'s star positions in its own sensor coordinates, and
/// `transforms[i]` maps those into the reference's. Returns the corrections to
/// compose onto each, and a report per frame.
pub fn refine_against_stars(
    reference: usize,
    stars: &[Vec<Star>],
    transforms: &[GlobalTransform],
) -> Vec<(GlobalTransform, Refinement)> {
    let ref_stars = &stars[reference];
    if ref_stars.len() < MIN_PAIRS {
        return vec![(GlobalTransform::IDENTITY, Refinement::default()); stars.len()];
    }
    // A grid over the reference's stars, so the nearest one is found without
    // walking all of them for every star of every frame.
    let grid = Grid::new(ref_stars, MATCH_RADIUS);

    (0..stars.len())
        .into_par_iter()
        .map(|i| {
            if i == reference || stars[i].len() < MIN_PAIRS {
                return (GlobalTransform::IDENTITY, Refinement::default());
            }
            let t = transforms[i];
            let mut src = Vec::with_capacity(stars[i].len());
            let mut dst = Vec::with_capacity(stars[i].len());
            let mut before = Vec::with_capacity(stars[i].len());
            for s in &stars[i] {
                let (mx, my) = t.apply(s.x, s.y);
                if let Some(r) = grid.nearest(ref_stars, mx, my) {
                    src.push((mx, my));
                    dst.push((r.x, r.y));
                    before.push((r.x - mx).hypot(r.y - my));
                }
            }
            if src.len() < MIN_PAIRS {
                return (GlobalTransform::IDENTITY, Refinement::default());
            }
            let Some(correction) = fit_affine(&src, &dst) else {
                return (GlobalTransform::IDENTITY, Refinement::default());
            };

            let mut after = Vec::with_capacity(src.len());
            let mut worst = 0.0f32;
            for (k, &(x, y)) in src.iter().enumerate() {
                let (cx, cy) = correction.apply(x, y);
                after.push((dst[k].0 - cx).hypot(dst[k].1 - cy));
                worst = worst.max((cx - x).hypot(cy - y));
            }
            let b = median(&mut before.clone());
            let a = median(&mut after);
            if log::log_enabled!(log::Level::Trace) {
                let (worst_cell, n) = cell_structure(&src, &dst, &correction);
                log::trace!(
                    "  frame {i}: residual left in the worst eighth-of-a-frame \
                     cell {worst_cell:.3} px over {n} stars"
                );
            }
            let report =
                Refinement { pairs: src.len(), before: b, after: a, worst, applied: false };
            if worst > MAX_CORRECTION || a > b * MUST_IMPROVE {
                return (GlobalTransform::IDENTITY, report);
            }
            (correction, Refinement { applied: true, ..report })
        })
        .collect()
}

/// The largest mean residual in any cell of an 8x8 grid over the frame, and how
/// many pairs backed it.
///
/// A least-squares affine leaves residuals whose *mean over many stars* is zero
/// wherever the affine is the right model. Where it is not -- field distortion,
/// differential refraction, anything that bends across the frame -- the
/// residuals in one part of the frame all point the same way, and the mean over
/// a cell stops falling as the square root of its count. That is the number
/// that says whether six parameters are enough.
fn cell_structure(
    src: &[(f32, f32)],
    dst: &[(f32, f32)],
    t: &GlobalTransform,
) -> (f32, usize) {
    const CELLS: usize = 8;
    let (mut x0, mut y0) = (f32::MAX, f32::MAX);
    let (mut x1, mut y1) = (f32::MIN, f32::MIN);
    for &(x, y) in src {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let (w, h) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
    let mut acc = vec![(0.0f64, 0.0f64, 0usize); CELLS * CELLS];
    for (k, &(x, y)) in src.iter().enumerate() {
        let (cx, cy) = t.apply(x, y);
        let gx = (((x - x0) / w * CELLS as f32) as usize).min(CELLS - 1);
        let gy = (((y - y0) / h * CELLS as f32) as usize).min(CELLS - 1);
        let e = &mut acc[gy * CELLS + gx];
        e.0 += (dst[k].0 - cx) as f64;
        e.1 += (dst[k].1 - cy) as f64;
        e.2 += 1;
    }
    let mut worst = 0.0f32;
    let mut count = 0usize;
    for &(sx, sy, n) in &acc {
        if n < 8 {
            continue;
        }
        let m = ((sx / n as f64).hypot(sy / n as f64)) as f32;
        if m > worst {
            worst = m;
            count = n;
        }
    }
    (worst, count)
}

/// Least squares affine through matched pairs, with one robust pass.
///
/// The first fit uses every pair, including whatever wrong pairings the match
/// radius admitted; the second drops the pairs the first fit left furthest out
/// and refits. Two passes, because the input is mostly right and a third
/// changes nothing measurable.
fn fit_affine(src: &[(f32, f32)], dst: &[(f32, f32)]) -> Option<GlobalTransform> {
    let first = solve_affine(src, dst, None)?;
    let mut residual: Vec<f32> = src
        .iter()
        .enumerate()
        .map(|(k, &(x, y))| {
            let (cx, cy) = first.apply(x, y);
            (dst[k].0 - cx).hypot(dst[k].1 - cy)
        })
        .collect();
    let cut = {
        let mut r = residual.clone();
        let m = median(&mut r);
        (3.0 * m).max(0.5)
    };
    let keep: Vec<bool> = residual.drain(..).map(|r| r < cut).collect();
    if keep.iter().filter(|k| **k).count() < MIN_PAIRS {
        return Some(first);
    }
    solve_affine(src, dst, Some(&keep)).or(Some(first))
}

fn solve_affine(
    src: &[(f32, f32)],
    dst: &[(f32, f32)],
    keep: Option<&[bool]>,
) -> Option<GlobalTransform> {
    // Normal equations for [x y 1] * p = target, once for each output axis.
    let mut ata = [[0.0f64; 3]; 3];
    let mut atx = [0.0f64; 3];
    let mut aty = [0.0f64; 3];
    let mut n = 0usize;
    for (k, &(x, y)) in src.iter().enumerate() {
        if let Some(keep) = keep
            && !keep[k] {
                continue;
            }
        n += 1;
        let row = [x as f64, y as f64, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
            atx[i] += row[i] * dst[k].0 as f64;
            aty[i] += row[i] * dst[k].1 as f64;
        }
    }
    if n < MIN_PAIRS {
        return None;
    }
    let px = solve3(ata, atx)?;
    let py = solve3(ata, aty)?;
    let m = [
        px[0] as f32,
        px[1] as f32,
        px[2] as f32,
        py[0] as f32,
        py[1] as f32,
        py[2] as f32,
    ];
    if m.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(GlobalTransform { m })
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in col + 1..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..3 {
            let f = a[r][col] / a[col][col];
            let pivot = a[col];
            for (c, v) in a[r].iter_mut().enumerate().skip(col) {
                *v -= f * pivot[c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for j in i + 1..3 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// A coarse bucket grid over the reference stars, for nearest-neighbour lookup.
pub(crate) struct Grid {
    cell: f32,
    w: usize,
    h: usize,
    x0: f32,
    y0: f32,
    buckets: Vec<Vec<u32>>,
}

impl Grid {
    pub(crate) fn new(stars: &[Star], radius: f32) -> Grid {
        let cell = radius.max(1.0);
        let (mut x0, mut y0) = (f32::MAX, f32::MAX);
        let (mut x1, mut y1) = (f32::MIN, f32::MIN);
        for s in stars {
            x0 = x0.min(s.x);
            y0 = y0.min(s.y);
            x1 = x1.max(s.x);
            y1 = y1.max(s.y);
        }
        let w = (((x1 - x0) / cell).ceil() as usize + 2).max(1);
        let h = (((y1 - y0) / cell).ceil() as usize + 2).max(1);
        let mut buckets = vec![Vec::new(); w * h];
        for (i, s) in stars.iter().enumerate() {
            let gx = (((s.x - x0) / cell) as usize).min(w - 1);
            let gy = (((s.y - y0) / cell) as usize).min(h - 1);
            buckets[gy * w + gx].push(i as u32);
        }
        Grid { cell, w, h, x0, y0, buckets }
    }

    pub(crate) fn nearest<'a>(&self, stars: &'a [Star], x: f32, y: f32) -> Option<&'a Star> {
        let gx = ((x - self.x0) / self.cell).floor();
        let gy = ((y - self.y0) / self.cell).floor();
        if !gx.is_finite() || !gy.is_finite() {
            return None;
        }
        let mut best: Option<(f32, usize)> = None;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let cx = gx as i32 + dx;
                let cy = gy as i32 + dy;
                if cx < 0 || cy < 0 || cx >= self.w as i32 || cy >= self.h as i32 {
                    continue;
                }
                for &k in &self.buckets[cy as usize * self.w + cx as usize] {
                    let s = &stars[k as usize];
                    let d = (s.x - x).hypot(s.y - y);
                    if d <= self.cell && best.map(|(bd, _)| d < bd).unwrap_or(true) {
                        best = Some((d, k as usize));
                    }
                }
            }
        }
        best.map(|(_, k)| &stars[k])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field of stars, scattered but never two within the match radius of
    /// each other, so a pairing is unambiguous.
    fn field(n: usize, seed: u64) -> Vec<Star> {
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 33) as f32) / (u32::MAX as f32 / 2.0)
        };
        let mut out: Vec<Star> = Vec::new();
        while out.len() < n {
            let x = rnd() * 2000.0;
            let y = rnd() * 1400.0;
            if out.iter().any(|o: &Star| (o.x - x).hypot(o.y - y) < 3.0 * MATCH_RADIUS) {
                continue;
            }
            out.push(Star { x, y, flux: 1.0 });
        }
        out
    }

    /// The same field seen through a transform, as a frame sees it.
    fn seen_through(stars: &[Star], t: &GlobalTransform) -> Vec<Star> {
        let inv = t.inverse().expect("invertible");
        stars
            .iter()
            .map(|s| {
                let (x, y) = inv.apply(s.x, s.y);
                Star { x, y, flux: s.flux }
            })
            .collect()
    }

    #[test]
    fn a_transform_left_a_pixel_out_is_brought_back() {
        // What this exists for. Correlation places a frame to within about a
        // pixel; the stars know better, and a stack is only as sharp as the
        // worst of the two.
        let reference = field(300, 0xC0FFEE);
        let truth = GlobalTransform::similarity(0.0008, 1.0002, 1.1, -0.7);
        let frame = seen_through(&reference, &truth);
        // The registration got close but not exact.
        let guessed = GlobalTransform::similarity(0.0, 1.0, 0.0, 0.0);

        let stars = vec![reference.clone(), frame];
        let out = refine_against_stars(0, &stars, &[GlobalTransform::IDENTITY, guessed]);
        let (correction, report) = out[1];
        assert!(report.applied, "no correction was made at all");
        assert!(report.pairs > 200, "only {} pairs", report.pairs);
        assert!(
            report.after < 0.05,
            "left {:.3} px on the table, from {:.3}",
            report.after,
            report.before
        );

        // And the corrected transform agrees with the truth across the frame.
        let refined = correction.compose(&guessed);
        for &(x, y) in &[(0.0f32, 0.0f32), (2000.0, 0.0), (0.0, 1400.0), (2000.0, 1400.0)] {
            let (ax, ay) = refined.apply(x, y);
            let (bx, by) = truth.apply(x, y);
            assert!(
                (ax - bx).hypot(ay - by) < 0.05,
                "corner ({x},{y}) is {:.3} px out",
                (ax - bx).hypot(ay - by)
            );
        }
    }

    #[test]
    fn a_frame_that_is_already_right_is_left_alone() {
        // The guard that matters. A correction fitted to nothing is not
        // neutral: it is a shift invented and then applied to every sample.
        let reference = field(300, 0xBEEF);
        let stars = vec![reference.clone(), reference.clone()];
        let out =
            refine_against_stars(0, &stars, &[GlobalTransform::IDENTITY; 2]);
        let (correction, report) = out[1];
        // Either it declined, or what it did is far below a tenth of a pixel.
        let mut worst = 0.0f32;
        for &(x, y) in &[(0.0f32, 0.0f32), (2000.0, 0.0), (0.0, 1400.0), (2000.0, 1400.0)] {
            let (ax, ay) = correction.apply(x, y);
            worst = worst.max((ax - x).hypot(ay - y));
        }
        assert!(worst < 0.02, "moved a corner {worst:.3} px for nothing");
        assert!(report.pairs >= 300 || !report.applied);
    }

    #[test]
    fn two_fields_that_are_not_the_same_sky_are_declined() {
        // Different stars, so any pairing inside the match radius is chance.
        // Fitting those would move the frame somewhere arbitrary.
        let reference = field(300, 0x1234);
        let other = field(300, 0x9999);
        let stars = vec![reference, other];
        let out = refine_against_stars(0, &stars, &[GlobalTransform::IDENTITY; 2]);
        assert!(!out[1].1.applied, "fitted a correction to unrelated fields");
    }

    #[test]
    fn the_reference_is_never_moved() {
        let reference = field(200, 0x5EED);
        let stars = vec![reference.clone(), reference];
        let out = refine_against_stars(0, &stars, &[GlobalTransform::IDENTITY; 2]);
        assert_eq!(out[0].0.m, GlobalTransform::IDENTITY.m);
        assert!(!out[0].1.applied);
    }

    #[test]
    fn a_correction_larger_than_a_polish_is_refused() {
        // If the pairs say the frame is six pixels out, the pairs are wrong:
        // at that distance the match radius has been pairing each star with a
        // different one. Refuse rather than apply it.
        let reference = field(300, 0xABCD);
        let truth = GlobalTransform::translation(30.0, 0.0);
        let frame = seen_through(&reference, &truth);
        let stars = vec![reference, frame];
        let out = refine_against_stars(0, &stars, &[GlobalTransform::IDENTITY; 2]);
        assert!(!out[1].1.applied, "applied a 30 px correction as if it were a polish");
    }
}
