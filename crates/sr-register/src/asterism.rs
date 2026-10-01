//! Placing a frame by the pattern of its stars.
//!
//! Correlation refines and does not search. It follows a burst that drifts by
//! tens of pixels and fails outright on one that has been rotated, which is
//! routine in astronomical data: a mount crossing the meridian turns the camera
//! through half a turn, and a target revisited a season later is framed at
//! whatever angle the rotator happened to be at. Plate-solve seeding covers the
//! case where the capture program wrote a solve into the header. This solves
//! the case where it did not.
//!
//! The dataset that asked for it: 2654 frames of NGC 7023 over two seasons, of
//! which 657 carry no solve at all and sit 96 degrees rotated from the rest,
//! with a 1% difference in scale. No amount of refinement reaches that, and no
//! fixed set of trial rotations is a general answer.
//!
//! ## How
//!
//! A triangle of stars has a shape that does not depend on where the frame was
//! pointed, which way up it was, or what the plate scale was: sort its sides
//! and take the two ratios against the longest. Two frames of the same sky
//! therefore contain many triangles with the same shape, and a shape is a
//! two-number key that can be hashed.
//!
//! Matching those keys gives a pile of candidate triangle pairs, most of them
//! coincidences. Each pair implies a rotation and a scale, and the true pairs
//! agree on both while the coincidences scatter, so the answer is the mode of
//! the votes. From the frames that voted with the mode, a similarity is fitted,
//! stars are paired under it, and the fit is repeated with a shrinking
//! tolerance.
//!
//! What comes back is a *seed*, on the same terms as a plate solve: the
//! registration that uses it re-runs from the identity as well and keeps
//! whichever result the pixels agree with.

use std::collections::HashMap;

use sr_core::geometry::GlobalTransform;
use sr_core::star::Star;

/// Brightest stars used from each frame.
///
/// The bright end is the part two frames of the same field reliably share; the
/// faint end differs with transparency and exposure and mostly adds triangles
/// that match nothing.
const STARS: usize = 200;

/// Neighbours each star forms triangles with.
const NEIGHBOURS: usize = 12;

/// Quantisation of the shape key. Two triangles land in the same bucket when
/// their side ratios agree to this, which has to be loose enough to absorb
/// centroid noise and tight enough that coincidences stay rare.
const SHAPE_TOLERANCE: f32 = 0.008;

/// Rotation histogram resolution, in degrees.
const ROTATION_BIN: f32 = 1.0;

/// Scale changes beyond this are not the same instrument on the same target.
const SCALE_LIMIT: f32 = 1.10;

/// A fit is believed when at least this many stars pair under it.
const MIN_PAIRS: usize = 12;

/// How close a pair has to land, in pixels, at the tightest refinement.
const PAIR_TOLERANCE: f32 = 2.0;

/// What a match found, for reporting.
#[derive(Clone, Copy, Debug)]
pub struct Match {
    pub transform: GlobalTransform,
    /// Stars that paired under the final fit.
    pub pairs: usize,
    /// Median distance between paired stars, in pixels.
    pub residual: f32,
    pub rotation_deg: f32,
    pub scale: f32,
}

/// One triangle pair's opinion about how the two frames are related.
#[derive(Clone, Copy)]
struct Vote {
    scale: f32,
    deg: f32,
    reference: usize,
    target: usize,
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// A triangle, as a shape key and the three stars that made it.
struct Triangle {
    key: (f32, f32),
    /// Vertices ordered by the length of the side opposite them, so that two
    /// triangles of the same shape have their vertices in the same order.
    v: [usize; 3],
    /// Longest side, from `v[0]` to `v[1]`, for recovering rotation and scale.
    long: f32,
}

fn triangles(stars: &[Star]) -> Vec<Triangle> {
    let n = stars.len().min(STARS);
    let mut out = Vec::with_capacity(n * NEIGHBOURS * NEIGHBOURS / 2);
    // Neighbours by brute force: sixty stars is nothing, and a spatial index
    // would be more code than the loop it replaces.
    for i in 0..n {
        let mut near: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let d = (stars[i].x - stars[j].x).hypot(stars[i].y - stars[j].y);
                (d, j)
            })
            .collect();
        near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        near.truncate(NEIGHBOURS);
        for a in 0..near.len() {
            for b in a + 1..near.len() {
                if let Some(t) = shape(stars, [i, near[a].1, near[b].1]) {
                    out.push(t);
                }
            }
        }
    }
    out
}

fn shape(stars: &[Star], v: [usize; 3]) -> Option<Triangle> {
    let p = |k: usize| (stars[v[k]].x, stars[v[k]].y);
    let d = |a: usize, b: usize| {
        let (ax, ay) = p(a);
        let (bx, by) = p(b);
        (ax - bx).hypot(ay - by)
    };
    // Side k is the one opposite vertex k.
    let sides = [d(1, 2), d(0, 2), d(0, 1)];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| sides[a].partial_cmp(&sides[b]).unwrap_or(std::cmp::Ordering::Equal));
    let (s0, s1, s2) = (sides[order[0]], sides[order[1]], sides[order[2]]);
    if s2 < 8.0 || s0 < 1.0 {
        // Degenerate, or so small that centroid noise dominates the shape.
        return None;
    }
    let key = (s0 / s2, s1 / s2);
    // The two vertices at the ends of the longest side, in a fixed order: the
    // longest side is opposite `order[2]`, so it joins the other two.
    let far = order[2];
    let ends = [(far + 1) % 3, (far + 2) % 3];
    // Break the remaining symmetry with the shorter of the two other sides.
    let (first, second) = if sides[ends[0]] <= sides[ends[1]] {
        (ends[0], ends[1])
    } else {
        (ends[1], ends[0])
    };
    Some(Triangle { key, v: [v[first], v[second], v[far]], long: s2 })
}

fn bucket(key: (f32, f32)) -> (i32, i32) {
    (
        (key.0 / SHAPE_TOLERANCE) as i32,
        (key.1 / SHAPE_TOLERANCE) as i32,
    )
}

/// Find the similarity that maps `target` onto `reference`.
///
/// Both are lists of star positions in their own frame's pixels, brightest
/// first. `None` when the two do not share enough of a pattern to be sure,
/// which is the answer for an unrelated field, a cloud, or a frame with too few
/// stars to say anything.
pub fn match_stars(reference: &[Star], target: &[Star]) -> Option<Match> {
    if reference.len() < MIN_PAIRS || target.len() < MIN_PAIRS {
        return None;
    }
    let rt = triangles(reference);
    let tt = triangles(target);
    if rt.is_empty() || tt.is_empty() {
        return None;
    }

    let mut index: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, t) in rt.iter().enumerate() {
        index.entry(bucket(t.key)).or_default().push(i);
    }

    // Each candidate pair votes for a rotation and a scale. True pairs agree;
    // coincidences do not.
    let mut votes: HashMap<i32, Vec<Vote>> = HashMap::new();
    for (ti, t) in tt.iter().enumerate() {
        let b = bucket(t.key);
        for dx in -1..=1 {
            for dy in -1..=1 {
                let Some(cands) = index.get(&(b.0 + dx, b.1 + dy)) else {
                    continue;
                };
                for &ri in cands {
                    let r = &rt[ri];
                    if (r.key.0 - t.key.0).abs() > SHAPE_TOLERANCE
                        || (r.key.1 - t.key.1).abs() > SHAPE_TOLERANCE
                    {
                        continue;
                    }
                    let scale = r.long / t.long.max(1e-6);
                    if !(1.0 / SCALE_LIMIT..=SCALE_LIMIT).contains(&scale) {
                        continue;
                    }
                    let rv = (
                        reference[r.v[1]].x - reference[r.v[0]].x,
                        reference[r.v[1]].y - reference[r.v[0]].y,
                    );
                    let tv = (
                        target[t.v[1]].x - target[t.v[0]].x,
                        target[t.v[1]].y - target[t.v[0]].y,
                    );
                    let rot = rv.1.atan2(rv.0) - tv.1.atan2(tv.0);
                    let deg = rot.to_degrees().rem_euclid(360.0);
                    votes
                        .entry((deg / ROTATION_BIN) as i32)
                        .or_default()
                        .push(Vote { scale, deg, reference: ri, target: ti });
                }
            }
        }
    }

    // The mode, taking the neighbouring bins with it so that a peak straddling
    // a boundary is not split in two.
    let bins: Vec<i32> = votes.keys().copied().collect();
    let best = bins.iter().copied().max_by_key(|b| {
        (-1..=1).map(|d| votes.get(&(b + d)).map_or(0, |v| v.len())).sum::<usize>()
    })?;
    let mut mode: Vec<Vote> = Vec::new();
    for d in -1..=1 {
        if let Some(v) = votes.get(&(best + d)) {
            mode.extend(v.iter().copied());
        }
    }
    log::debug!(
        "asterism: {} x {} triangles, {} vote bins, mode {} at {} deg",
        rt.len(), tt.len(), votes.len(), mode.len(), best as f32 * ROTATION_BIN
    );
    if mode.len() < 4 {
        return None;
    }

    // The first estimate comes from the mode itself, not from a fit over the
    // pairs that voted for it. Those pairs still contain the coincidences that
    // happened to land in the same bin, and least squares has no defence
    // against them: fitting them directly puts the estimate far enough out
    // that nothing pairs afterwards, which is precisely how this failed the
    // first time it was written. Medians have a defence.
    let mut rot: Vec<f32> = mode.iter().map(|v| v.deg).collect();
    let mut sc: Vec<f32> = mode.iter().map(|v| v.scale).collect();
    let rot = median(&mut rot).to_radians();
    let sc = median(&mut sc);
    let (sin, cos) = (rot.sin() * sc, rot.cos() * sc);
    let mut dx: Vec<f32> = Vec::with_capacity(mode.len() * 3);
    let mut dy: Vec<f32> = Vec::with_capacity(mode.len() * 3);
    for v in &mode {
        for k in 0..3 {
            let (r, g) = (&reference[rt[v.reference].v[k]], &target[tt[v.target].v[k]]);
            dx.push(r.x - (cos * g.x - sin * g.y));
            dy.push(r.y - (sin * g.x + cos * g.y));
        }
    }
    let t = GlobalTransform {
        m: [cos, -sin, median(&mut dx), sin, cos, median(&mut dy)],
    };

    if let Some(m) = refine_match(reference, target, t) {
        return Some(m);
    }

    // A rotation mode can still contain more accidental triangles than real
    // ones. Its median angle then misses the sky by many pixels on a large
    // sensor. Keep each triangle's angle, scale and translation together and
    // ask independent stars which candidate actually places the field.
    // Limit voting to the same bright catalogue used to form the triangles;
    // the full supplied catalogue is used for the final refinement below.
    let bright_ref = &reference[..reference.len().min(STARS)];
    let bright_target = &target[..target.len().min(STARS)];
    let mut best: Option<(usize, f32, GlobalTransform)> = None;
    for v in &mode {
        let angle = v.deg.to_radians();
        let (sin, cos) = (angle.sin() * v.scale, angle.cos() * v.scale);
        let r = reference[rt[v.reference].v[0]];
        let g = target[tt[v.target].v[0]];
        let candidate = GlobalTransform { m: [cos, -sin,
            r.x - (cos*g.x - sin*g.y), sin, cos,
            r.y - (sin*g.x + cos*g.y)] };
        let pairs = pair_up(bright_ref, bright_target, &candidate, 2.0 * PAIR_TOLERANCE);
        if pairs.len() < MIN_PAIRS { continue; }
        let residual = median_residual(bright_ref, bright_target, &candidate, &pairs);
        if best.as_ref().is_none_or(|&(count, error, _)|
            pairs.len() > count || (pairs.len() == count && residual < error)) {
            best = Some((pairs.len(), residual, candidate));
        }
    }
    let (pairs, _, candidate) = best?;
    log::debug!("asterism: recovered candidate supported by {pairs} distinct bright stars");
    refine_match(reference, target, candidate)
}

/// Seed registration between instruments with a known approximate pixel-scale
/// ratio (target arcsec/pixel divided by reference arcsec/pixel).
/// The existing narrow search and evidence requirements operate in reference
/// pixels after this coordinate conversion. This remains a seed, not validation.
/// No resampling or change to the default same-instrument matcher is involved.
pub fn match_stars_at_scale(
    reference: &[Star],
    target: &[Star],
    target_to_reference_scale: f32,
) -> Option<Match> {
    let scale = target_to_reference_scale;
    if !scale.is_finite() || scale <= 0.0
        || reference.iter().any(|s| !s.x.is_finite() || !s.y.is_finite())
    {
        return None;
    }
    let scaled: Vec<_> = target.iter().map(|s| Star {
        x: s.x * scale, y: s.y * scale, flux: s.flux,
    }).collect();
    if scaled.iter().any(|s| !s.x.is_finite() || !s.y.is_finite()) {
        return None;
    }
    let mut result = match_stars(reference, &scaled)?;
    result.transform = result.transform.compose(&GlobalTransform::similarity(0.0, scale, 0.0, 0.0));
    result.scale = result.transform.scale();
    Some(result)
}

fn refine_match(reference: &[Star], target: &[Star], mut t: GlobalTransform) -> Option<Match> {
    // Then pair every star under it and refit, tightening as it converges.
    let mut result = None;
    for tol in [16.0f32, 8.0, 4.0, PAIR_TOLERANCE] {
        let paired = pair_up(reference, target, &t, tol);
        if paired.len() < MIN_PAIRS {
            break;
        }
        let Some(next) = fit_similarity(reference, target, &paired) else {
            break;
        };
        t = next;
        let residual = median_residual(reference, target, &t, &paired);
        result = Some(Match {
            transform: t,
            pairs: paired.len(),
            residual,
            rotation_deg: t.rotation().to_degrees(),
            scale: t.scale(),
        });
    }
    result.filter(|m| m.pairs >= MIN_PAIRS && m.residual <= PAIR_TOLERANCE)
}

/// Nearest reference star to each transformed target star, within `tol`.
fn pair_up(
    reference: &[Star],
    target: &[Star],
    t: &GlobalTransform,
    tol: f32,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (ti, s) in target.iter().enumerate() {
        let (x, y) = t.apply(s.x, s.y);
        let mut best = (tol, usize::MAX);
        for (ri, r) in reference.iter().enumerate() {
            let d = (r.x - x).hypot(r.y - y);
            if d < best.0 {
                best = (d, ri);
            }
        }
        if best.1 != usize::MAX {
            out.push((best.1, ti));
        }
    }
    // One reference star cannot answer for two target stars.
    out.sort_unstable();
    out.dedup_by_key(|p| p.0);
    out
}

/// Least-squares similarity mapping target positions onto reference ones.
fn fit_similarity(
    reference: &[Star],
    target: &[Star],
    pairs: &[(usize, usize)],
) -> Option<GlobalTransform> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let (mut rx, mut ry, mut tx, mut ty) = (0.0f64, 0.0, 0.0, 0.0);
    for &(ri, ti) in pairs {
        rx += reference[ri].x as f64;
        ry += reference[ri].y as f64;
        tx += target[ti].x as f64;
        ty += target[ti].y as f64;
    }
    let (rx, ry, tx, ty) = (rx / n, ry / n, tx / n, ty / n);
    let (mut sxx, mut sxy, mut stt) = (0.0f64, 0.0, 0.0);
    for &(ri, ti) in pairs {
        let (ax, ay) = ((target[ti].x as f64 - tx), (target[ti].y as f64 - ty));
        let (bx, by) = ((reference[ri].x as f64 - rx), (reference[ri].y as f64 - ry));
        sxx += ax * bx + ay * by;
        sxy += ax * by - ay * bx;
        stt += ax * ax + ay * ay;
    }
    if stt < 1e-9 {
        return None;
    }
    let (a, b) = (sxx / stt, sxy / stt);
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    // [a -b; b a] about the target centroid, landing on the reference centroid.
    Some(GlobalTransform {
        m: [
            a as f32,
            -b as f32,
            (rx - a * tx + b * ty) as f32,
            b as f32,
            a as f32,
            (ry - b * tx - a * ty) as f32,
        ],
    })
}

fn median_residual(
    reference: &[Star],
    target: &[Star],
    t: &GlobalTransform,
    pairs: &[(usize, usize)],
) -> f32 {
    let mut d: Vec<f32> = pairs
        .iter()
        .map(|&(ri, ti)| {
            let (x, y) = t.apply(target[ti].x, target[ti].y);
            (reference[ri].x - x).hypot(reference[ri].y - y)
        })
        .collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d[d.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_instrument_scale_recovers_rotation_and_preserves_default_limit() {
        let a = field(100, 0x71285);
        let b = transform(&a, 57.0, 458.0 / 336.0, 1200.0, -350.0);
        assert!(match_stars(&a, &b).is_none());
        let m = match_stars_at_scale(&a, &b, 336.0 / 458.0).unwrap();
        assert!(m.pairs >= 90);
        assert!(m.residual < 0.01);
        for (r, t) in a.iter().zip(&b) {
            let p = m.transform.apply(t.x, t.y);
            assert!((p.0-r.x).hypot(p.1-r.y) < 0.01);
        }
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(match_stars_at_scale(&a, &b, invalid).is_none());
        }
    }

    #[test]
    fn explicit_scale_does_not_accept_unrelated_fields() {
        assert!(match_stars_at_scale(&field(100, 42), &field(100, 123), 0.73).is_none());
    }

    #[test]
    fn recovers_iris_frames_when_accidental_triangles_bias_the_rotation_mode() {
        // Brightest 200 detections from the original reference and three
        // rejected 9576x6388 Iris exposures. Each target was independently
        // registered through a neighbouring exposure (0.28-0.36 sensor-pixel
        // residual). These are positions only, rounded to 0.001 sensor pixel.
        let stars: Vec<Star> = include_str!("../tests/data/iris-bright-stars.txt")
            .lines().filter(|line| !line.is_empty()).map(|line| {
                let mut values = line.split_whitespace().map(|v| v.parse::<f32>().unwrap());
                Star { x: values.next().unwrap(), y: values.next().unwrap(), flux: 1. }
            }).collect();
        assert_eq!(stars.len(), 800);
        let reference = &stars[..200];
        for (target, rotation) in stars[200..].chunks_exact(200).zip([-179.64f32, -8.82, -9.02]) {
            let m = match_stars(reference, target).expect("valid rotated exposure must be placed");
            assert!(m.pairs >= MIN_PAIRS);
            assert!(m.residual < PAIR_TOLERANCE);
            assert!((m.rotation_deg - rotation).abs() < 0.1, "{m:?}");
        }
    }

    #[test]
    fn candidate_fallback_does_not_invent_matches_in_unrelated_fields() {
        let reference = field(200, 0x123456);
        for seed in 0..12 {
            assert!(match_stars(&reference, &field(200, 0xABCD + seed*17)).is_none());
        }
    }

    fn field(n: usize, seed: u64) -> Vec<Star> {
        let mut s = seed | 1;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u32 << 31) as f32
        };
        (0..n)
            .map(|_| Star {
                x: next() * 3000.0,
                y: next() * 2000.0,
                flux: 1.0 + next(),
            })
            .collect()
    }

    fn transform(stars: &[Star], deg: f32, scale: f32, dx: f32, dy: f32) -> Vec<Star> {
        // The inverse of what the matcher should recover.
        let th = deg.to_radians();
        let (s, c) = (th.sin() * scale, th.cos() * scale);
        stars
            .iter()
            .map(|p| Star {
                x: c * p.x - s * p.y + dx,
                y: s * p.x + c * p.y + dy,
                flux: p.flux,
            })
            .collect()
    }

    #[test]
    fn recovers_a_quarter_turn_with_a_scale_change() {
        // The case from the two-season NGC 7023 set: about 96 degrees apart
        // with a 1% difference in plate scale, and no header to say so.
        let a = field(90, 0xA11CE);
        let b = transform(&a, 96.0, 1.01, -400.0, 250.0);
        let m = match_stars(&a, &b).expect("a rotated copy of a star field must be found");
        assert!(m.pairs >= 40, "only {} pairs", m.pairs);
        assert!(m.residual < 0.5, "residual {}", m.residual);
        // Recovering the map back: rotation -96, scale 1/1.01.
        let rot = m.rotation_deg;
        assert!((rot + 96.0).abs() < 0.5 || (rot - 264.0).abs() < 0.5, "rotation {rot}");
        assert!((m.scale - 1.0 / 1.01).abs() < 0.01, "scale {}", m.scale);
    }

    #[test]
    fn recovers_a_half_turn() {
        let a = field(80, 0xBEEF);
        let b = transform(&a, 180.0, 1.0, 4000.0, 3000.0);
        let m = match_stars(&a, &b).expect("a meridian flip must be found");
        assert!((m.rotation_deg.abs() - 180.0).abs() < 0.5, "rotation {}", m.rotation_deg);
        assert!(m.residual < 0.5);
    }

    #[test]
    fn survives_stars_the_other_frame_does_not_have() {
        // Transparency differs between nights, so the two lists overlap rather
        // than agree. Half of each is unique here.
        let shared = field(50, 0xC0FFEE);
        let mut a = shared.clone();
        a.extend(field(50, 0x1111));
        let mut b = transform(&shared, 30.0, 1.0, 120.0, -80.0);
        b.extend(field(50, 0x2222));
        let m = match_stars(&a, &b).expect("half a field in common is plenty");
        assert!((m.rotation_deg + 30.0).abs() < 0.5, "rotation {}", m.rotation_deg);
    }

    #[test]
    fn two_unrelated_fields_are_refused() {
        // The property that makes this safe to use as a seed: no answer is
        // better than a confident wrong one.
        let a = field(90, 0x5EED);
        let b = field(90, 0x9999);
        assert!(match_stars(&a, &b).is_none());
    }

    #[test]
    fn too_few_stars_is_no_answer() {
        let a = field(6, 0x1234);
        let b = transform(&a, 45.0, 1.0, 0.0, 0.0);
        assert!(match_stars(&a, &b).is_none());
    }

    #[test]
    fn centroid_noise_does_not_break_it() {
        let a = field(90, 0xD00D);
        let mut b = transform(&a, 120.0, 1.0, -900.0, 400.0);
        let mut s = 7u64;
        for p in b.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = ((s >> 33) as f32 / (1u32 << 31) as f32 - 0.5) * 0.6;
            p.x += j;
            p.y -= j;
        }
        let m = match_stars(&a, &b).expect("a third of a pixel of jitter is nothing");
        assert!((m.rotation_deg + 120.0).abs() < 0.5, "rotation {}", m.rotation_deg);
        assert!(m.residual < 1.0, "residual {}", m.residual);
    }
}
