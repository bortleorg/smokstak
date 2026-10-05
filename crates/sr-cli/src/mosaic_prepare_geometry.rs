//! Automatic relative geometry for mono mosaics. Training selects the graph;
//! frozen, disjoint faint detections assess the final solution.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sr_core::{GlobalTransform, projection::FrameProjection, star::Star};
use sr_register::{
    asterism,
    bundle::{self, BundleFrame, StarPair},
    projective::{self, ProjectiveTransform},
};
use std::{collections::HashMap, path::PathBuf};

const TRAIN: usize = 1000;
const MIN_PAIRS: usize = 30;
const EDGE_PAIRS: usize = 384;
const P50: f64 = 0.75;
const P90: f64 = 1.5;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Catalog {
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    #[serde(with = "star_list")]
    pub stars: Vec<Star>,
    /// Optional angular pixel size, in arcseconds per pixel.
    pub pixel_scale: Option<f64>,
    /// Optical metadata fingerprint, including telescope/focal length, not just camera.
    pub optical_id: String,
}

mod star_list {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        stars: &[Star],
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        stars
            .iter()
            .map(|s| [s.x, s.y, s.flux])
            .collect::<Vec<_>>()
            .serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Vec<Star>, D::Error> {
        Ok(Vec::<[f32; 3]>::deserialize(deserializer)?
            .into_iter()
            .map(|s| Star {
                x: s[0],
                y: s[1],
                flux: s[2],
            })
            .collect())
    }
}

pub struct FrameGeometry {
    pub projection: FrameProjection,
    pub group: usize,
    pub registration_p50: f64,
    pub registration_p90: f64,
    pub validation_stars: usize,
}

pub struct Geometry {
    pub frames: Vec<FrameGeometry>,
    pub notes: Vec<String>,
    /// Global catalogue indices, including within-panel exposures.
    pub pairs: Vec<(usize, usize)>,
    pub anchor: usize,
}

#[derive(Clone)]
struct PairFit {
    h: ProjectiveTransform,
    train: Vec<(usize, usize)>,
    test: Vec<(usize, usize)>,
    scale: f64,
}

fn xy(s: &Star) -> [f64; 2] {
    [s.x as f64, s.y as f64]
}
fn training_count(c: &Catalog) -> usize {
    TRAIN.min(c.stars.len() * 2 / 3)
}

fn bounded_pairs(pairs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if pairs.len() <= EDGE_PAIRS {
        return pairs;
    }
    // Sample the entire ordered list, not just its bright end. Selection is
    // independent of measured residuals and preserves broad catalogue coverage.
    (0..EDGE_PAIRS)
        .map(|i| pairs[i * (pairs.len() - 1) / (EDGE_PAIRS - 1)])
        .collect()
}
fn center(c: &Catalog) -> [f64; 2] {
    [c.width as f64 / 2., c.height as f64 / 2.]
}
fn norm(c: &Catalog) -> f64 {
    c.width.max(c.height) as f64 / 2.
}
fn identity() -> ProjectiveTransform {
    ProjectiveTransform {
        m: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
    }
}
fn normalizer(c: &Catalog) -> ProjectiveTransform {
    let s = norm(c);
    let [x, y] = center(c);
    ProjectiveTransform {
        m: [1. / s, 0., -x / s, 0., 1. / s, -y / s, 0., 0., 1.],
    }
}
fn inverse(h: ProjectiveTransform) -> Option<ProjectiveTransform> {
    let m = h.m;
    let a = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ];
    let d = m[0] * a[0] + m[1] * a[3] + m[2] * a[6];
    (d.is_finite() && d.abs() > 1e-12).then(|| ProjectiveTransform {
        m: a.map(|v| v / d),
    })
}

fn quantiles(mut e: Vec<f64>) -> Option<(f64, f64)> {
    if e.is_empty() || e.iter().any(|v| !v.is_finite()) {
        return None;
    }
    e.sort_by(f64::total_cmp);
    Some((
        e[(e.len() - 1) / 2],
        e[((e.len() - 1) as f64 * 0.9).ceil() as usize],
    ))
}

/// Spatial hash pairing is linear in detections on ordinary star fields. Each
/// split is associated independently, so held-out detections cannot displace a
/// training match. Return (reference index, target index).
fn reciprocal(
    reference: &[Star],
    target: &[Star],
    transform: GlobalTransform,
    offsets: (usize, usize),
) -> Vec<(usize, usize)> {
    let refs: Vec<_> = reference.iter().map(|s| (s.x as f64, s.y as f64)).collect();
    let mapped: Vec<_> = target
        .iter()
        .map(|s| {
            let q = transform.apply(s.x, s.y);
            (q.0 as f64, q.1 as f64)
        })
        .collect();
    fn nearest(points: &[(f64, f64)], queries: &[(f64, f64)]) -> Vec<Option<usize>> {
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            grid.entry(((p.0 / 4.).floor() as i64, (p.1 / 4.).floor() as i64))
                .or_default()
                .push(i);
        }
        queries
            .iter()
            .map(|q| {
                let key = ((q.0 / 4.).floor() as i64, (q.1 / 4.).floor() as i64);
                let mut best = None;
                let mut distance = 16.;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(indices) = grid.get(&(key.0 + dx, key.1 + dy)) {
                            for &i in indices {
                                let p = points[i];
                                let d = (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2);
                                if d < distance {
                                    best = Some(i);
                                    distance = d;
                                }
                            }
                        }
                    }
                }
                best
            })
            .collect()
    }
    let a = nearest(&refs, &mapped);
    let b = nearest(&mapped, &refs);
    a.into_iter()
        .enumerate()
        .filter_map(|(j, i)| {
            let i = i?;
            (b[i] == Some(j)).then_some((i + offsets.0, j + offsets.1))
        })
        .collect()
}

fn seed_scales(a: &Catalog, b: &Catalog) -> Vec<f64> {
    let mut scales = vec![1.];
    if let (Some(x), Some(y)) = (a.pixel_scale, b.pixel_scale)
        && x.is_finite()
        && y.is_finite()
        && x > 0.
        && y > 0.
    {
        return if (y / x - 1.).abs() < 1e-6 {
            vec![1.]
        } else {
            vec![y / x, 1.]
        };
    }
    // Overlapping +/-10% windows cover 1:4 through 4:1 when headers are absent
    // missing. This search changes no ordinary burst matcher defaults.
    for i in 1..=9 {
        scales.push(1.18f64.powi(i));
        scales.push(1.18f64.powi(-i));
    }
    scales.dedup_by(|x, y| (*x - *y).abs() < 1e-6);
    scales
}

fn pair(a: &Catalog, b: &Catalog) -> Option<PairFit> {
    let (na, nb) = (training_count(a), training_count(b));
    let (ra, tb) = (&a.stars[..na], &b.stars[..nb]);
    for scale in seed_scales(a, b) {
        let Some(seed) = asterism::match_stars_at_scale(ra, tb, scale as f32) else {
            continue;
        };
        let refine = sr_register::refine::refine_against_stars(
            0,
            &[ra.to_vec(), tb.to_vec()],
            &[GlobalTransform::IDENTITY, seed.transform],
        );
        let affine = refine[1].0.compose(&seed.transform);
        let train = reciprocal(ra, tb, affine, (0, 0));
        if train.len() < MIN_PAIRS {
            continue;
        }
        let coordinates: Vec<_> = train
            .iter()
            .map(|&(i, j)| (xy(&b.stars[j]), xy(&a.stars[i])))
            .collect();
        let Some(h) = projective::fit(&coordinates) else {
            continue;
        };
        if h.bounds(b.width, b.height).is_none() {
            continue;
        }
        let errors: Option<Vec<_>> = coordinates
            .iter()
            .map(|&(q, r)| h.apply(q[0], q[1]).map(|p| (p.0 - r[0]).hypot(p.1 - r[1])))
            .collect();
        let Some((p50, p90)) = errors.and_then(quantiles) else {
            continue;
        };
        if p50 > 1.5 || p90 > 3. {
            continue;
        }
        // Fixed under training affine, never reassociated after fitting optics.
        let test = reciprocal(&a.stars[na..], &b.stars[nb..], affine, (na, nb));
        if test.len() < MIN_PAIRS {
            continue;
        }
        return Some(PairFit {
            h,
            train: bounded_pairs(train),
            test: bounded_pairs(test),
            scale: seed.scale as f64,
        });
    }
    None
}

fn same_optics(a: &Catalog, b: &Catalog, scale: f64) -> bool {
    a.width == b.width
        && a.height == b.height
        && (scale - 1.).abs() < 0.015
        && (a.optical_id.is_empty() || b.optical_id.is_empty() || a.optical_id == b.optical_id)
        && match (a.pixel_scale, b.pixel_scale) {
            (Some(x), Some(y)) => (x / y - 1.).abs() < 0.015,
            _ => true,
        }
}

fn same_panel(a: &Catalog, b: &Catalog, p: &PairFit) -> bool {
    if !same_optics(a, b, p.scale) {
        return false;
    }
    let Some([x0, y0, x1, y1]) = p.h.bounds(b.width, b.height) else {
        return false;
    };
    // A bounding rectangle alone overestimates overlap for rotated fields;
    // count a regular source grid actually landing inside the reference.
    if x1 < x0 || y1 < y0 {
        return false;
    }
    let mut inside = 0;
    for y in 0..9 {
        for x in 0..9 {
            if let Some((u, v)) = p.h.apply(
                x as f64 * (b.width - 1) as f64 / 8.,
                y as f64 * (b.height - 1) as f64 / 8.,
            ) {
                inside +=
                    usize::from(u >= 0. && v >= 0. && u < a.width as f64 && v < a.height as f64);
            }
        }
    }
    inside >= 61
}

fn projection(frame: &BundleFrame, distortion: [f64; 3], anchor: &Catalog) -> FrameProjection {
    FrameProjection {
        center: frame.center,
        normalization_scale: frame.normalization_scale,
        distortion,
        homography: frame.homography,
        output_center: center(anchor),
        output_scale: norm(anchor),
    }
}

fn normalized(c: &Catalog, s: &Star, k: [f64; 3]) -> [f64; 2] {
    let p = xy(s);
    let center = center(c);
    let scale = norm(c);
    bundle::undistort([(p[0] - center[0]) / scale, (p[1] - center[1]) / scale], k)
}

fn pose(
    a: &Catalog,
    b: &Catalog,
    anchor: &FrameProjection,
    pair: &PairFit,
) -> Result<FrameProjection> {
    let points: Option<Vec<_>> = pair
        .train
        .iter()
        .map(|&(i, j)| {
            let p = xy(&a.stars[i]);
            let q = anchor.map(p[0], p[1])?;
            Some((
                normalized(b, &b.stars[j], anchor.distortion),
                [
                    (q.0 - anchor.output_center[0]) / anchor.output_scale,
                    (q.1 - anchor.output_center[1]) / anchor.output_scale,
                ],
            ))
        })
        .collect();
    let h = points
        .and_then(|p| projective::fit(&p))
        .context("exposure pose fit failed")?;
    Ok(FrameProjection {
        center: center(b),
        normalization_scale: norm(b),
        homography: h.m,
        ..anchor.clone()
    })
}

fn validation(
    a: &Catalog,
    b: &Catalog,
    pa: &FrameProjection,
    pb: &FrameProjection,
    pairs: &[(usize, usize)],
) -> Result<(f64, f64, usize)> {
    ensure!(
        pairs.len() >= MIN_PAIRS,
        "{} and {} have only {} independent validation stars; need at least {MIN_PAIRS}",
        a.path.display(),
        b.path.display(),
        pairs.len()
    );
    let errors: Option<Vec<_>> = pairs
        .iter()
        .map(|&(i, j)| {
            let r = xy(&a.stars[i]);
            let t = xy(&b.stars[j]);
            let r = pa.map(r[0], r[1])?;
            let t = pb.map(t[0], t[1])?;
            Some((r.0 - t.0).hypot(r.1 - t.1))
        })
        .collect();
    let (p50, p90) = errors
        .and_then(quantiles)
        .context("nonfinite held-out projection")?;
    ensure!(
        p50 <= P50 && p90 <= P90,
        "registration quality failed for {} against {}: held-out p50 {p50:.3}, p90 {p90:.3} pixels (limits {P50}/{P90}); no frames were silently discarded",
        b.path.display(),
        a.path.display()
    );
    Ok((p50, p90, pairs.len()))
}

fn footprint(c: &Catalog, p: &FrameProjection) -> Result<()> {
    p.validate()?;
    // Same checked full-frame adapter used by reconstruction: probes horizons,
    // folds and inverse closure and rejects mesh/bounds outside its budget.
    p.to_warp_with_node_budget(c.width, c.height, 0.05, 1_000_000)
        .with_context(|| format!("invalid complete footprint for {}", c.path.display()))?;
    Ok(())
}

/// Discover repeated pointings, solve their connected overlap graph, then place
/// every selected exposure under inherited optics. Failure never drops a frame.
pub fn prepare(catalogs: &[Catalog], mut progress: impl FnMut(&str)) -> Result<Geometry> {
    ensure!(
        catalogs.len() >= 2,
        "select at least two mono exposures for a mosaic"
    );
    for c in catalogs {
        ensure!(
            c.width > 0 && c.height > 0 && c.stars.len() >= 3 * MIN_PAIRS,
            "{} needs at least {} star detections for independent registration checks",
            c.path.display(),
            3 * MIN_PAIRS
        );
        ensure!(
            c.stars.iter().all(|s| s.x.is_finite()
                && s.y.is_finite()
                && s.flux.is_finite()
                && s.x >= 0.
                && s.y >= 0.
                && s.x < c.width as f32
                && s.y < c.height as f32),
            "invalid star coordinates in {}",
            c.path.display()
        );
        ensure!(
            c.pixel_scale.is_none_or(|s| s.is_finite() && s > 0.),
            "invalid angular pixel scale in {}",
            c.path.display()
        );
    }
    let mut representatives: Vec<usize> = Vec::new();
    let mut groups = vec![0; catalogs.len()];
    let mut member_pairs: Vec<Option<PairFit>> = vec![None; catalogs.len()];
    let mut cache: HashMap<(usize, usize), Option<PairFit>> = HashMap::new();
    for i in 0..catalogs.len() {
        progress(&format!(
            "Identifying panel {} of {}",
            i + 1,
            catalogs.len()
        ));
        let mut assigned = false;
        for (g, &r) in representatives.iter().enumerate().rev() {
            let a = &catalogs[r];
            let b = &catalogs[i];
            let prior = match (a.pixel_scale, b.pixel_scale) {
                (Some(x), Some(y)) => y / x,
                _ => 1.,
            };
            if !same_optics(a, b, prior) {
                continue;
            }
            let p = cache
                .entry((r, i))
                .or_insert_with(|| pair(&catalogs[r], &catalogs[i]));
            if let Some(p) = p
                && same_panel(&catalogs[r], &catalogs[i], p)
            {
                groups[i] = g;
                member_pairs[i] = Some(p.clone());
                assigned = true;
                break;
            }
        }
        if !assigned {
            groups[i] = representatives.len();
            representatives.push(i);
            let mut positive: Vec<_> = cache
                .iter()
                .filter(|(_, v)| v.is_some())
                .map(|(&k, _)| k)
                .collect();
            positive.sort_unstable();
            let excess = positive.len().saturating_sub(256);
            for key in positive.into_iter().take(excess) {
                cache.remove(&key);
            }
        } else {
            // Once this is an exposure of an existing pointing, its attempted
            // links will never be used by the representative graph.
            cache.retain(|&(_, target), _| target != i);
        }
        ensure!(
            representatives.len() <= 120,
            "more than 120 distinct pointings require a larger graph solver; no selected frames were dropped"
        );
    }
    progress(&format!(
        "Connecting {} detected panels",
        representatives.len()
    ));
    let mut edges = Vec::new();
    let mut component: Vec<_> = (0..representatives.len()).collect();
    let mut degree_count = vec![0usize; representatives.len()];
    fn root(component: &[usize], mut i: usize) -> usize {
        while component[i] != i {
            i = component[i];
        }
        i
    }
    for a in 0..representatives.len() {
        for b in a + 1..representatives.len() {
            let (ra, rb) = (representatives[a], representatives[b]);
            let candidate = cache
                .remove(&(ra, rb))
                .unwrap_or_else(|| pair(&catalogs[ra], &catalogs[rb]));
            if let Some(p) = candidate {
                let (ar, br) = (root(&component, a), root(&component, b));
                // Retain a spanning forest regardless of degree, plus a
                // bounded set of loop closures. Every accepted connection can
                // join components; a dense field cannot create an unbounded
                // dense normal-equation input or catalogue-pair cache.
                if ar != br || (degree_count[a] < 8 && degree_count[b] < 8) {
                    component[br] = ar;
                    degree_count[a] += 1;
                    degree_count[b] += 1;
                    edges.push((a, b, p));
                }
            }
        }
    }
    drop(cache);
    // Choose the best-sampled instrument as output gauge; graph degree breaks
    // ties using training evidence alone.
    let degree = |i: usize| edges.iter().filter(|(a, b, _)| *a == i || *b == i).count();
    let mut relative = vec![None; representatives.len()];
    relative[0] = Some(1.);
    for _ in 0..representatives.len() {
        for (a, b, p) in &edges {
            match (relative[*a], relative[*b]) {
                (Some(s), None) => relative[*b] = Some(s * p.scale),
                (None, Some(s)) => relative[*a] = Some(s / p.scale),
                _ => {}
            }
        }
    }
    let metadata_scale = representatives
        .iter()
        .enumerate()
        .find_map(|(i, &r)| Some(catalogs[r].pixel_scale? / relative[i]?))
        .unwrap_or(1.);
    let sampling: Vec<_> = representatives
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            catalogs[r]
                .pixel_scale
                .unwrap_or_else(|| relative[i].unwrap_or(f64::INFINITY) * metadata_scale)
        })
        .collect();
    let finest = sampling.iter().copied().fold(f64::INFINITY, f64::min);
    let anchor = (0..representatives.len())
        .filter(|&i| sampling[i] <= finest * 1.015)
        .min_by(|&a, &b| degree(b).cmp(&degree(a)).then(a.cmp(&b)))
        .unwrap();
    let mut poses = vec![None; representatives.len()];
    poses[anchor] = Some(identity());
    for _ in 0..representatives.len() {
        for (a, b, p) in &edges {
            match (poses[*a], poses[*b]) {
                (Some(h), None) => poses[*b] = Some(h.compose(&p.h)),
                (None, Some(h)) => poses[*a] = inverse(p.h).map(|i| h.compose(&i)),
                _ => {}
            }
        }
    }
    let missing: Vec<_> = poses
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_none())
        .map(|(i, _)| catalogs[representatives[i]].path.display().to_string())
        .collect();
    ensure!(
        missing.is_empty(),
        "mosaic overlap graph is disconnected; could not place {}. Add overlapping exposures; no frames were dropped",
        missing.join(", ")
    );
    let reference = &catalogs[representatives[anchor]];
    let mut instruments: Vec<usize> = Vec::new();
    let mut instrument_for = vec![0; representatives.len()];
    for (i, &r) in representatives.iter().enumerate() {
        let mut found = None;
        for (k, &previous) in instruments.iter().enumerate() {
            let prev = representatives[previous];
            let ratio = match (catalogs[prev].pixel_scale, catalogs[r].pixel_scale) {
                (Some(a), Some(b)) => b / a,
                _ => {
                    let h = poses[i].unwrap().m;
                    let h0 = poses[previous].unwrap().m;
                    (h[0] * h[4] - h[1] * h[3]).abs().sqrt()
                        / (h0[0] * h0[4] - h0[1] * h0[3]).abs().sqrt()
                }
            };
            if same_optics(&catalogs[prev], &catalogs[r], ratio) {
                found = Some(k);
                break;
            }
        }
        instrument_for[i] = found.unwrap_or_else(|| {
            instruments.push(i);
            instruments.len() - 1
        });
    }
    let frames: Vec<_> = representatives
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let h = normalizer(reference)
                .compose(&poses[i].unwrap())
                .compose(&inverse(normalizer(&catalogs[r])).unwrap());
            BundleFrame {
                center: center(&catalogs[r]),
                normalization_scale: norm(&catalogs[r]),
                instrument: instrument_for[i],
                homography: h.m,
            }
        })
        .collect();
    let mut projections = if representatives.len() == 1 {
        vec![projection(&frames[0], [0.; 3], reference)]
    } else {
        progress("Refining shared optics and panel geometry");
        let mut training = Vec::new();
        for (a, b, p) in &edges {
            for &(i, j) in &p.train {
                training.push(StarPair {
                    a: *a,
                    b: *b,
                    source_a: xy(&catalogs[representatives[*a]].stars[i]),
                    source_b: xy(&catalogs[representatives[*b]].stars[j]),
                });
            }
        }
        // Two scopes observing one field have a relative optical gauge. Hold
        // the output instrument native instead of claiming absolute optics.
        let mut fixed = vec![None; instruments.len()];
        if instruments.len() > 1 {
            fixed[instrument_for[anchor]] = Some([0.; 3]);
        }
        let solved = bundle::solve_with_fixed_distortion(
            &frames,
            &training,
            anchor,
            instruments.len(),
            norm(reference),
            300,
            &fixed,
        )
        .context("joint panel/optical solve failed")?;
        solved
            .frames
            .iter()
            .map(|f| projection(f, solved.distortion[f.instrument], reference))
            .collect::<Vec<_>>()
    };
    let mut evidence = vec![(0f64, 0f64, 0usize); representatives.len()];
    let mut pairs = Vec::new();
    for (a, b, p) in &edges {
        let (ra, rb) = (representatives[*a], representatives[*b]);
        let e = validation(
            &catalogs[ra],
            &catalogs[rb],
            &projections[*a],
            &projections[*b],
            &p.test,
        )?;
        for i in [*a, *b] {
            evidence[i].0 = evidence[i].0.max(e.0);
            evidence[i].1 = evidence[i].1.max(e.1);
            evidence[i].2 += e.2;
        }
        pairs.push((ra, rb));
    }
    let mut out = Vec::with_capacity(catalogs.len());
    for (i, c) in catalogs.iter().enumerate() {
        progress(&format!(
            "Validating exposure {} of {}",
            i + 1,
            catalogs.len()
        ));
        let g = groups[i];
        let r = representatives[g];
        let (p, e) = if let Some(pair) = &member_pairs[i] {
            let p = pose(&catalogs[r], c, &projections[g], pair)?;
            let e = validation(&catalogs[r], c, &projections[g], &p, &pair.test)?;
            evidence[g].0 = evidence[g].0.max(e.0);
            evidence[g].1 = evidence[g].1.max(e.1);
            evidence[g].2 += e.2;
            pairs.push((r, i));
            (p, e)
        } else {
            (projections[g].clone(), evidence[g])
        };
        footprint(c, &p)?;
        out.push(FrameGeometry {
            projection: p,
            group: g,
            registration_p50: e.0,
            registration_p90: e.1,
            validation_stars: e.2,
        });
    }
    // A one-panel anchor gets its evidence from the independently placed
    // exposures, not a fabricated identity/self-match measurement.
    for (g, &r) in representatives.iter().enumerate() {
        ensure!(
            evidence[g].2 >= MIN_PAIRS,
            "panel {} lacks independent validation evidence",
            g + 1
        );
        out[r].registration_p50 = evidence[g].0;
        out[r].registration_p90 = evidence[g].1;
        out[r].validation_stars = evidence[g].2;
    }
    projections.clear();
    Ok(Geometry {frames:out,pairs,anchor:representatives[anchor],notes:vec![format!("Automatically placed {} exposures in {} pointing groups with {} optical groups; every selected exposure passed independent p50 <= {P50} / p90 <= {P90} pixel checks.",catalogs.len(),representatives.len(),instruments.len()),"Relative sky geometry; no absolute astrometric solution. Optical grouping is inferred from metadata and measured scale.".into()]})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn catalog(name: &str) -> Catalog {
        let mut seed = 734234u64;
        let mut stars = Vec::new();
        for i in 0..1400 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = 50. + (seed >> 32) as f32 / u32::MAX as f32 * 1900.;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = 50. + (seed >> 32) as f32 / u32::MAX as f32 * 1400.;
            stars.push(Star {
                x,
                y,
                flux: (1400 - i) as f32,
            });
        }
        Catalog {
            path: name.into(),
            width: 2000,
            height: 1500,
            stars,
            pixel_scale: Some(2.),
            optical_id: "scope-a".into(),
        }
    }
    #[test]
    fn shifted_exposures_are_grouped_and_all_validated() {
        let a = catalog("a");
        let mut b = a.clone();
        b.path = "b".into();
        for s in &mut b.stars {
            s.x += 3.;
            s.y -= 2.;
        }
        let result = prepare(&[a, b], |_| {}).unwrap();
        assert_eq!(result.frames.len(), 2);
        assert_eq!(result.frames[0].group, result.frames[1].group);
        assert_eq!(result.pairs.len(), 1);
        assert!(
            result
                .frames
                .iter()
                .all(|f| f.validation_stars >= 300 && f.registration_p90 < 0.01)
        );
    }
    #[test]
    fn frozen_validation_rejects_moved_faint_stars() {
        let a = catalog("a");
        let mut b = a.clone();
        b.path = "b".into();
        let p = pair(&a, &b).unwrap();
        let split = training_count(&b);
        for s in &mut b.stars[split..] {
            s.x += 2.;
        }
        let base = FrameProjection {
            center: center(&a),
            normalization_scale: norm(&a),
            distortion: [0.; 3],
            homography: identity().m,
            output_center: center(&a),
            output_scale: norm(&a),
        };
        let candidate = pose(&a, &b, &base, &p).unwrap();
        assert!(validation(&a, &b, &base, &candidate, &p.test).is_err());
    }
    #[test]
    fn metadata_distinguishes_optics_and_scale_search_covers_missing_headers() {
        let a = catalog("a");
        let mut b = a.clone();
        b.optical_id = "scope-b".into();
        assert!(!same_optics(&a, &b, 1.));
        b.pixel_scale = None;
        let scales = seed_scales(&a, &b);
        for ratio in [0.25, 0.5, 0.73, 1., 1.37, 2., 4.] {
            assert!(scales.iter().any(|s| (ratio / s - 1.).abs() < 0.09));
        }
    }
    #[test]
    fn serialized_catalog_preserves_star_order() {
        let c = catalog("roundtrip");
        let s = serde_json::to_vec(&c).unwrap();
        let back: Catalog = serde_json::from_slice(&s).unwrap();
        assert_eq!(c.stars, back.stars);
    }

    #[test]
    fn unknown_mixed_scale_is_found_without_header_or_manual_ratio() {
        let mut a = catalog("wide");
        a.pixel_scale = None;
        let mut b = a.clone();
        b.path = "narrow".into();
        b.width = 1400;
        b.height = 1050;
        b.optical_id = "scope-b".into();
        for s in &mut b.stars {
            s.x *= 0.7;
            s.y *= 0.7;
        }
        let result = prepare(&[a, b], |_| {}).unwrap();
        assert_ne!(result.frames[0].group, result.frames[1].group);
        assert!(
            result
                .frames
                .iter()
                .all(|f| f.registration_p90 < 0.01 && f.validation_stars >= MIN_PAIRS)
        );
    }

    #[test]
    fn disconnected_fields_fail_instead_of_dropping_an_input() {
        let a = catalog("field-a");
        let mut b = catalog("unrelated-field-b");
        for (i, s) in b.stars.iter_mut().enumerate() {
            s.x = (s.x + i as f32 * 37.19).rem_euclid(1950.) + 10.;
            s.y = (s.y + i as f32 * 13.43).rem_euclid(1450.) + 10.;
        }
        let error = prepare(&[a, b], |_| {}).err().unwrap().to_string();
        assert!(error.contains("disconnected"), "{error}");
        assert!(error.contains("unrelated-field-b"), "{error}");
    }

    #[test]
    fn sparse_catalogues_use_disjoint_adaptive_splits() {
        let mut a = catalog("sparse");
        a.stars.truncate(150);
        let mut b = a.clone();
        b.path = "sparse-second".into();
        for s in &mut b.stars {
            s.x += 2.;
        }
        let result = prepare(&[a, b], |_| {}).unwrap();
        assert_eq!(result.frames.len(), 2);
        assert!(result.frames.iter().all(|f| f.validation_stars >= 45));
    }

    #[test]
    #[ignore = "requires an explicitly supplied local catalogue fixture"]
    fn local_catalogue_acceptance() {
        let path = std::env::var("SMOKSTAK_GEOMETRY_FIXTURE")
            .expect("set SMOKSTAK_GEOMETRY_FIXTURE to a JSON array of Catalog records");
        let catalogs: Vec<Catalog> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let result = prepare(&catalogs, |s| eprintln!("{s}")).unwrap();
        eprintln!(
            "Accepted {} frames, {} edges, anchor {}",
            result.frames.len(),
            result.pairs.len(),
            result.anchor
        );
        if let Ok(path) = std::env::var("SMOKSTAK_GEOMETRY_RESULT") {
            let frames:Vec<_>=result.frames.iter().map(|f|serde_json::json!({"projection":f.projection,"group":f.group,"p50":f.registration_p50,"p90":f.registration_p90,"validation_stars":f.validation_stars})).collect();
            std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"frames":frames,"pairs":result.pairs,"anchor":result.anchor,"notes":result.notes})).unwrap()).unwrap();
        }
    }
}
