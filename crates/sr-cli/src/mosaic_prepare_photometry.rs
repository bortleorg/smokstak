//! Bounded native stellar photometry and independently tested same-sky corrections.
//! Does not flatten individual images: the anchor retains its complete scene.
use crate::mosaic::FrameSpec;
use anyhow::{ensure, Context, Result};
use sr_core::projection::FrameProjection;
use sr_core::{NoiseModel, NoiseSource};
use sr_raw::window::{FitsWindowReader, SampleUnits};
use std::collections::{HashMap, VecDeque};

/// Rows per cached band. Bands span the full image width, so each file row is
/// read in one contiguous request at most once while its band stays cached;
/// square tiles re-read the same file rows once per tile column.
const CACHE_BAND_ROWS: usize = 64;
const CACHE_BANDS: usize = 8;
struct CachedBand {
    y: usize,
    height: usize,
    pixels: Vec<f32>,
}
/// Exact normalized samples, cached only to amortize small native-window I/O.
/// Eviction happens before loading a new band; each source retains at most
/// CACHE_BANDS full-width bands (12.8 MiB for a 6248-pixel-wide sensor).
struct WindowCache {
    source: FitsWindowReader,
    tiles: VecDeque<CachedBand>,
    #[cfg(test)]
    direct: bool,
}
impl WindowCache {
    fn new(source: FitsWindowReader) -> Self {
        Self {
            source,
            tiles: VecDeque::new(),
            #[cfg(test)]
            direct: false,
        }
    }
    fn open(path: &std::path::Path) -> Result<Self> {
        Ok(Self::new(FitsWindowReader::open(
            path,
            &sr_raw::ReadOptions::default(),
        )?))
    }
    fn dimensions(&self) -> (usize, usize) {
        self.source.dimensions()
    }
    fn saturation_threshold(&self) -> Option<f32> {
        self.source.saturation_threshold()
    }
    fn band(&mut self, y: usize) -> Result<&CachedBand> {
        if let Some(index) = self.tiles.iter().position(|t| t.y == y) {
            let band = self.tiles.remove(index).expect("known cached band");
            self.tiles.push_back(band);
        } else {
            if self.tiles.len() == CACHE_BANDS {
                self.tiles.pop_front();
            }
            let (w, h) = self.dimensions();
            let height = CACHE_BAND_ROWS.min(h - y);
            let pixels = self.source.read_rect(0, y, w, height)?;
            self.tiles.push_back(CachedBand { y, height, pixels });
        }
        Ok(self.tiles.back().expect("just loaded cached band"))
    }
    fn read_rect(&mut self, x: usize, y: usize, w: usize, h: usize) -> Result<Vec<f32>> {
        #[cfg(test)]
        if self.direct {
            return Ok(self.source.read_rect(x, y, w, h)?);
        }
        let (width, height) = self.dimensions();
        ensure!(
            w > 0
                && h > 0
                && x.checked_add(w).is_some_and(|v| v <= width)
                && y.checked_add(h).is_some_and(|v| v <= height),
            "invalid cached source window"
        );
        let mut result = vec![0.; w.checked_mul(h).context("cached window size overflow")?];
        for by in (y / CACHE_BAND_ROWS..=(y + h - 1) / CACHE_BAND_ROWS).map(|b| b * CACHE_BAND_ROWS) {
            let band = self.band(by)?;
            let y0 = y.max(by);
            let y1 = (y + h).min(by + band.height);
            for yy in y0..y1 {
                let src = (yy - by) * width + x;
                let dst = (yy - y) * w;
                result[dst..dst + w].copy_from_slice(&band.pixels[src..src + w]);
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Measurement {
    x: f64,
    y: f64,
    flux: f64,
    hfd: f64,
    peak: f64,
}

fn aperture_catalog(
    reader: &mut WindowCache,
    catalog: &[[f64; 3]],
    radius: f64,
    projection: &FrameProjection,
) -> Result<Vec<Option<Measurement>>> {
    let mut order: Vec<_> = (0..catalog.len())
        .filter(|&j| catalog[j].iter().all(|v| v.is_finite()))
        .collect();
    order.sort_by_key(|&j| ((catalog[j][1] / CACHE_BAND_ROWS as f64).floor() as i64, j));
    let mut measurements = vec![None; catalog.len()];
    for j in order {
        measurements[j] = aperture(reader, catalog[j], radius, projection)?;
    }
    Ok(measurements)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    Some(if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    } else {
        values[n / 2]
    })
}

fn aperture(
    reader: &mut WindowCache,
    star: [f64; 3],
    radius: f64,
    projection: &FrameProjection,
) -> Result<Option<Measurement>> {
    let [x, y, _] = star;
    let scale = pixel_area(projection, x, y)?.sqrt();
    let radius = radius / scale;
    ensure!(
        (3.0..=40.0).contains(&radius),
        "common angular aperture exceeds supported native sampling range"
    );
    let outer = radius * (19.0 / 12.0);
    // Enclose modest affine anisotropy, then reject any aperture whose actual
    // projected rim touches this bounded window rather than truncating it.
    let extent = outer * 2.0;
    let center = projection.map(x, y).context("star outside projection")?;
    let (width, height) = reader.dimensions();
    if !x.is_finite()
        || !y.is_finite()
        || x - extent < 0.0
        || y - extent < 0.0
        || x + extent > (width - 1) as f64
        || y + extent > (height - 1) as f64
    {
        return Ok(None);
    }
    let x0 = (x - extent).floor() as usize;
    let y0 = (y - extent).floor() as usize;
    let w = (x + extent).ceil() as usize - x0 + 1;
    let h = (y + extent).ceil() as usize - y0 + 1;
    let values = reader.read_rect(x0, y0, w, h)?;
    let saturation = reader.saturation_threshold().map(f64::from);
    let mut sky = Vec::new();
    let mut signal = Vec::new();
    for iy in 0..h {
        for ix in 0..w {
            let q = projection
                .map((x0 + ix) as f64, (y0 + iy) as f64)
                .context("invalid aperture projection")?;
            let distance = (q.0 - center.0).hypot(q.1 - center.1) / scale;
            ensure!(
                distance > outer || (ix > 0 && iy > 0 && ix + 1 < w && iy + 1 < h),
                "projected stellar aperture exceeds bounded native window"
            );
            if distance > outer {
                continue;
            }
            let value = values[iy * w + ix] as f64;
            if !value.is_finite() || saturation.is_some_and(|s| value >= s) {
                return Ok(None);
            }
            if distance <= radius {
                signal.push((distance, value));
            } else if distance >= radius * (15.0 / 12.0) {
                sky.push(value);
            }
        }
    }
    let Some(background) = median(&mut sky) else {
        return Ok(None);
    };
    let flux = signal.iter().map(|s| s.1 - background).sum::<f64>();
    let peak = signal.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
    let positive = signal
        .iter()
        .map(|s| (s.1 - background).max(0.0))
        .sum::<f64>();
    if !flux.is_finite() || flux <= 0.0 || positive <= 0.0 {
        return Ok(None);
    }
    signal.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut cumulative = 0.0;
    let mut hfd = 0.0;
    for (distance, value) in signal {
        cumulative += (value - background).max(0.0);
        if cumulative >= 0.5 * positive {
            hfd = 2.0 * distance * scale;
            break;
        }
    }
    Ok(Some(Measurement {
        x,
        y,
        flux,
        hfd,
        peak,
    }))
}

/// Fixed 8x8 layout; one 32x32 (or smaller native image) patch at a time.
/// Difference MAD suppresses isolated stars without fitting a sky surface.
fn sky_noise(reader: &mut FitsWindowReader) -> Result<(f64, f64, usize)> {
    let (width, height) = reader.dimensions();
    let (w, h) = (width.min(32), height.min(32));
    let mut skies = Vec::new();
    let mut sigmas = Vec::new();
    let saturation = reader.saturation_threshold();
    let usable = |v: f32| v.is_finite() && saturation.is_none_or(|s| v < s);
    for gy in 0..8 {
        for gx in 0..8 {
            let x = (width - w) * gx / 7;
            let y = (height - h) * gy / 7;
            let pixels = reader.read_rect(x, y, w, h)?;
            let mut sky: Vec<f64> = pixels
                .iter()
                .copied()
                .filter(|v| usable(*v))
                .map(f64::from)
                .collect();
            let mut differences = Vec::new();
            for iy in 0..h {
                for ix in 0..w {
                    let v = pixels[iy * w + ix];
                    if !usable(v) {
                        continue;
                    }
                    if ix + 1 < w && usable(pixels[iy * w + ix + 1]) {
                        differences.push(pixels[iy * w + ix + 1] as f64 - v as f64);
                    }
                    if iy + 1 < h && usable(pixels[(iy + 1) * w + ix]) {
                        differences.push(pixels[(iy + 1) * w + ix] as f64 - v as f64);
                    }
                }
            }
            if let (Some(sky), Some(center)) = (median(&mut sky), median(&mut differences)) {
                for d in &mut differences {
                    *d = (*d - center).abs();
                }
                if let Some(mad) = median(&mut differences) {
                    skies.push(sky);
                    sigmas.push(1.482602218505602 * mad / std::f64::consts::SQRT_2);
                }
            }
        }
    }
    let count = skies.len();
    Ok((
        median(&mut skies).context("no usable sky patches")?,
        median(&mut sigmas).context("no usable noise patches")?,
        count,
    ))
}

// Interval arithmetic encloses each straight source-edge segment, including
// extrema between sampled endpoints. Dependencies may widen, never shrink it.
#[derive(Clone, Copy)]
struct Interval(f64, f64);
impl Interval {
    fn add(self, b: Self) -> Self {
        Self(self.0 + b.0, self.1 + b.1)
    }
    fn mul(self, b: Self) -> Self {
        let q = [self.0 * b.0, self.0 * b.1, self.1 * b.0, self.1 * b.1];
        Self(
            q.iter().copied().fold(f64::INFINITY, f64::min),
            q.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }
    fn scale(self, s: f64) -> Self {
        self.mul(Self(s, s))
    }
    fn square(self) -> Self {
        Self(
            if self.0 <= 0.0 && self.1 >= 0.0 {
                0.0
            } else {
                self.0.powi(2).min(self.1.powi(2))
            },
            self.0.powi(2).max(self.1.powi(2)),
        )
    }
}
fn edge_interval(p: &FrameProjection, a: (f64, f64), b: (f64, f64)) -> Result<[f64; 4]> {
    let x = Interval(
        (a.0.min(b.0) - p.center[0]) / p.normalization_scale,
        (a.0.max(b.0) - p.center[0]) / p.normalization_scale,
    );
    let y = Interval(
        (a.1.min(b.1) - p.center[1]) / p.normalization_scale,
        (a.1.max(b.1) - p.center[1]) / p.normalization_scale,
    );
    let r = x.square().add(y.square());
    let [k, t1, t2] = p.distortion;
    let u = x
        .add(x.mul(r).scale(k))
        .add(x.mul(y).scale(2.0 * t1))
        .add(r.add(x.square().scale(2.0)).scale(t2));
    let v = y
        .add(y.mul(r).scale(k))
        .add(r.add(y.square().scale(2.0)).scale(t1))
        .add(x.mul(y).scale(2.0 * t2));
    let m = p.homography;
    let row = |i: usize| {
        u.scale(m[i])
            .add(v.scale(m[i + 1]))
            .add(Interval(m[i + 2], m[i + 2]))
    };
    let z = row(6);
    ensure!(
        z.0 > 1e-12 || z.1 < -1e-12,
        "projection perimeter approaches/crosses a horizon"
    );
    let inv = Interval(1.0 / z.1, 1.0 / z.0);
    let qx = row(0)
        .mul(inv)
        .scale(p.output_scale)
        .add(Interval(p.output_center[0], p.output_center[0]));
    let qy = row(3)
        .mul(inv)
        .scale(p.output_scale)
        .add(Interval(p.output_center[1], p.output_center[1]));
    let bounds = [qx.0, qy.0, qx.1, qy.1];
    ensure!(
        bounds.iter().all(|v| v.is_finite()),
        "projection perimeter overflow"
    );
    Ok(bounds)
}
pub(crate) fn bounds(p: &FrameProjection, w: usize, h: usize) -> Result<[f64; 4]> {
    p.validate()?;
    ensure!(
        w > 0 && h > 0 && w <= 1_000_000 && h <= 1_000_000,
        "invalid frame dimensions"
    );
    let mut result = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let corners = [
        (-0.5, -0.5),
        (w as f64 - 0.5, -0.5),
        (w as f64 - 0.5, h as f64 - 0.5),
        (-0.5, h as f64 - 0.5),
    ];
    for i in 0..4 {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        let n = ((a.0 - b.0).abs().max((a.1 - b.1).abs()) / 32.0)
            .ceil()
            .max(1.0) as usize;
        for j in 0..n {
            let at = |t: f64| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            let q = edge_interval(p, at(j as f64 / n as f64), at((j + 1) as f64 / n as f64))?;
            result[0] = result[0].min(q[0]);
            result[1] = result[1].min(q[1]);
            result[2] = result[2].max(q[2]);
            result[3] = result[3].max(q[3]);
        }
    }
    Ok(result)
}

fn quantile(v: &[f64], q: f64) -> f64 {
    v[((v.len() - 1) as f64 * q).round() as usize]
}
fn patch(
    reader: &mut WindowCache,
    p: &FrameProjection,
    x: f64,
    y: f64,
    radius: f64,
) -> Result<Option<[f64; 5]>> {
    let Some(c) = p.inverse_map(x, y) else {
        return Ok(None);
    };
    let (Some(xp), Some(xm), Some(yp), Some(ym)) = (
        p.inverse_map(x + 1.0, y),
        p.inverse_map(x - 1.0, y),
        p.inverse_map(x, y + 1.0),
        p.inverse_map(x, y - 1.0),
    ) else {
        return Ok(None);
    };
    let rx = (radius * ((xp.0 - xm.0) * 0.5).hypot((yp.0 - ym.0) * 0.5)).ceil() + 2.0;
    let ry = (radius * ((xp.1 - xm.1) * 0.5).hypot((yp.1 - ym.1) * 0.5)).ceil() + 2.0;
    if !rx.is_finite() || !ry.is_finite() || rx > 32.0 || ry > 32.0 {
        return Ok(None);
    }
    let (w, h) = reader.dimensions();
    if c.0 - rx < 0.0 || c.1 - ry < 0.0 || c.0 + rx > (w - 1) as f64 || c.1 + ry > (h - 1) as f64 {
        return Ok(None);
    }
    let x0 = (c.0 - rx).floor() as usize;
    let y0 = (c.1 - ry).floor() as usize;
    let width = (c.0 + rx).ceil() as usize - x0 + 1;
    let height = (c.1 + ry).ceil() as usize - y0 + 1;
    let data = reader.read_rect(x0, y0, width, height)?;
    if data.iter().any(|v| !v.is_finite() || *v >= 1.0) {
        return Ok(None);
    }
    let mut values = Vec::new();
    for iy in 0..height {
        for ix in 0..width {
            let Some(q) = p.map((x0 + ix) as f64, (y0 + iy) as f64) else {
                return Ok(None);
            };
            if (q.0 - x).hypot(q.1 - y) <= radius {
                values.push(data[iy * width + ix] as f64);
            }
        }
    }
    if values.len() < 5 {
        return Ok(None);
    }
    values.sort_by(f64::total_cmp);
    let median = quantile(&values, 0.5);
    let p10 = quantile(&values, 0.1);
    let p90 = quantile(&values, 0.9);
    let spread = p90 - p10;
    Ok(Some([median, values.len() as f64, spread, p10, p90]))
}

fn pixel_area(p: &FrameProjection, x: f64, y: f64) -> Result<f64> {
    let a = p.map(x + 0.5, y).context("invalid projection derivative")?;
    let b = p.map(x - 0.5, y).context("invalid projection derivative")?;
    let c = p.map(x, y + 0.5).context("invalid projection derivative")?;
    let d = p.map(x, y - 0.5).context("invalid projection derivative")?;
    let area = ((a.0 - b.0) * (c.1 - d.1) - (a.1 - b.1) * (c.0 - d.0)).abs();
    ensure!(
        area.is_finite() && area > 1e-6 && area < 1e6,
        "invalid projected pixel area"
    );
    Ok(area)
}

fn shot_coefficient(white_level: f32, electrons_per_native_adu: f64) -> f64 {
    // Same left-shift correction as the full FITS decoder: an N-bit converter
    // stored in 16 bits has white=65536-step, while EGAIN describes native ADU.
    (65536.0 - white_level as f64) / (electrons_per_native_adu * white_level as f64)
}

#[derive(Clone, Copy)]
struct Star {
    x: f64,
    y: f64,
    flux: f64,
    /// Aperture flux over sky noise in the aperture; shot noise is ignored.
    snr: f64,
}

fn nearest(a: &[Star], b: &[Star]) -> Vec<Option<usize>> {
    let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, s) in b.iter().enumerate() {
        cells
            .entry(((s.x / 2.).floor() as i64, (s.y / 2.).floor() as i64))
            .or_default()
            .push(i);
    }
    a.iter()
        .map(|s| {
            let (x, y) = ((s.x / 2.).floor() as i64, (s.y / 2.).floor() as i64);
            let mut best = (2.25, None);
            for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    if let Some(ids) = cells.get(&(xx, yy)) {
                        for &i in ids {
                            let dist = (s.x - b[i].x).powi(2) + (s.y - b[i].y).powi(2);
                            if dist < best.0 {
                                best = (dist, Some(i));
                            }
                        }
                    }
                }
            }
            best.1
        })
        .collect()
}

/// Sparse graph row: value equals basis dot (correction[b]-correction[a]).
#[derive(Clone)]
struct Row {
    a: usize,
    b: usize,
    basis: [f64; 6],
    value: f64,
    test: bool,
    precision_weight: f64,
}
fn prediction(row: &Row, x: &[f64], anchor: usize, dim: usize) -> f64 {
    let mut v = 0.;
    for k in 0..dim {
        if row.b != anchor {
            v += row.basis[k] * x[row.b * dim + k];
        }
        if row.a != anchor {
            v -= row.basis[k] * x[row.a * dim + k];
        }
    }
    v
}
fn accumulate(row: &Row, value: f64, out: &mut [f64], anchor: usize, dim: usize) {
    for k in 0..dim {
        if row.b != anchor {
            out[row.b * dim + k] += row.basis[k] * value;
        }
        if row.a != anchor {
            out[row.a * dim + k] -= row.basis[k] * value;
        }
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Jacobi-preconditioned conjugate gradients keeps memory linear in evidence.
fn solve(rows: &[Row], n: usize, anchor: usize, dim: usize) -> Result<Vec<f64>> {
    let mut connected = vec![false; n];
    connected[anchor] = true;
    for _ in 0..n {
        for r in rows.iter().filter(|r| !r.test) {
            if connected[r.a] || connected[r.b] {
                connected[r.a] = true;
                connected[r.b] = true;
            }
        }
    }
    ensure!(
        connected.iter().all(|v| *v),
        "photometric overlap graph disconnected; {} of {n} frames have support",
        connected.iter().filter(|v| **v).count()
    );
    let mut weights: Vec<_> = rows.iter().map(|r| r.precision_weight).collect();
    let mut x = vec![0.; n * dim];
    for _ in 0..8 {
        let mut rhs = vec![0.; n * dim];
        let mut diagonal = vec![0.; n * dim];
        let mut blocks:HashMap<(usize,usize),[[f64;6];6]>=HashMap::new();
        for (r, &w) in rows.iter().zip(&weights).filter(|(r, _)| !r.test) {
            accumulate(r, w * r.value, &mut rhs, anchor, dim);
            if dim==6 {
                let block=blocks.entry((r.a,r.b)).or_insert([[0.;6];6]);
                for (i,line) in block.iter_mut().enumerate(){for (j,v) in line.iter_mut().enumerate(){*v+=w*r.basis[i]*r.basis[j];}}
            }
            for k in 0..dim {
                for f in [r.a, r.b] {
                    if f != anchor {
                        diagonal[f * dim + k] += w * r.basis[k].powi(2);
                    }
                }
            }
        }
        for k in 0..dim {
            diagonal[anchor * dim + k] = 1.;
        }
        ensure!(
            diagonal.iter().all(|v| v.is_finite() && *v > 1e-15),
            "overlap planes lack spatial support"
        );
        // A frame's six response terms are strongly correlated with one another
        // (a canvas-wide basis over a small footprint), so the six-term fit is
        // preconditioned by each frame's own 6x6 block rather than its diagonal.
        let mut frame_blocks = vec![[[0.; 6]; 6]; if dim == 6 { n } else { 0 }];
        for (&(a, b), block) in blocks.iter().filter(|_| dim == 6) {
            for f in [a, b] {
                for (target, source) in frame_blocks[f].iter_mut().zip(block) {
                    for (t, s) in target.iter_mut().zip(source) {
                        *t += s;
                    }
                }
            }
        }
        let identity: [[f64; 6]; 6] = std::array::from_fn(|i| std::array::from_fn(|j| f64::from(u8::from(i == j))));
        let inverse_blocks = frame_blocks
            .into_iter()
            .enumerate()
            .map(|(f, block)| {
                if f == anchor {
                    Some(identity)
                } else {
                    solve6(block, identity)
                }
            })
            .collect::<Option<Vec<_>>>()
            .context("a frame's stellar response is not constrained")?;
        let precondition = |r: &[f64]| -> Vec<f64> {
            if dim == 6 {
                (0..n * 6)
                    .map(|i| dot(&inverse_blocks[i / 6][i % 6], &r[i / 6 * 6..i / 6 * 6 + 6]))
                    .collect()
            } else {
                r.iter().zip(&diagonal).map(|(r, d)| r / d).collect()
            }
        };
        let apply = |v: &[f64]| {
            let mut out = vec![0.; n * dim];
            if dim==6{
                for (&(a,b),block) in &blocks {
                    let difference:Vec<_>=(0..6).map(|k|if b==anchor{0.}else{v[b*6+k]}-if a==anchor{0.}else{v[a*6+k]}).collect();
                    for (i,line) in block.iter().enumerate(){
                        let value=dot(line,&difference);
                        if a!=anchor{out[a*6+i]-=value;}
                        if b!=anchor{out[b*6+i]+=value;}
                    }
                }
            }else{
                for (r, &w) in rows.iter().zip(&weights).filter(|(r, _)| !r.test) {
                    accumulate(r, w * prediction(r, v, anchor, dim), &mut out, anchor, dim);
                }
            }
            out
        };
        let ax = apply(&x);
        let mut residual: Vec<_> = rhs.iter().zip(ax).map(|(b, a)| b - a).collect();
        let mut z = precondition(&residual);
        let mut direction = z.clone();
        let mut rz = dot(&residual, &z);
        let tolerance = dot(&rhs, &rhs).max(1e-24) * 1e-18;
        for _ in 0..(n * dim * 8).max(100) {
            if dot(&residual, &residual) <= tolerance {
                break;
            }
            let ad = apply(&direction);
            let denominator = dot(&direction, &ad);
            ensure!(
                denominator > 0. && denominator.is_finite(),
                "singular overlap constraint graph"
            );
            let step = rz / denominator;
            for i in 0..x.len() {
                x[i] += step * direction[i];
                residual[i] -= step * ad[i];
            }
            z = precondition(&residual);
            let next = dot(&residual, &z);
            let beta = next / rz;
            for i in 0..x.len() {
                direction[i] = z[i] + beta * direction[i];
            }
            rz = next;
        }
        ensure!(
            dot(&residual, &residual) <= tolerance * 100.,
            "overlap fit did not converge; geometry does not constrain a stable plane"
        );
        let mut residuals: Vec<_> = rows
            .iter()
            .filter(|r| !r.test)
            .map(|r| r.value - prediction(r, &x, anchor, dim))
            .collect();
        let center = median(&mut residuals).context("no photometric constraints")?;
        for v in &mut residuals {
            *v = (*v - center).abs();
        }
        // Gain residuals are standardized by independently measured edge
        // uncertainty, matching the original soft-L1 graph with f_scale=2.
        let sigma = if dim == 1 || dim == 6 {
            1.0
        } else {
            (1.4826 * median(&mut residuals).unwrap()).max(1e-7)
        };
        for (r, w) in rows.iter().zip(&mut weights) {
            *w = r.precision_weight
                / (1.
                    + ((r.value - prediction(r, &x, anchor, dim)) * r.precision_weight.sqrt()
                        / (2. * sigma))
                        .powi(2))
                .sqrt();
        }
    }
    ensure!(x.iter().all(|v| v.is_finite()), "nonfinite photometric fit");
    Ok(x)
}

struct GainEvidence {
    matches: usize,
    withheld: usize,
    scatter: f64,
    standard_error: f64,
}

/// Withheld stars must be this far above the sky noise in both frames. A
/// fainter star's aperture disagreement is dominated by its own measurement
/// (nebulosity in the annulus, neighbours) and tests that, not the gain model.
/// Fainter withheld stars only top an overlap up to MIN_WITHHELD; the rest are
/// dropped, never moved into training.
const VALIDATION_SNR: f64 = 50.;
/// Floor on a matched star's log-flux uncertainty in the spatial fit, so the
/// brightest stars cannot claim more precision than aperture photometry has.
const STAR_LOG_FLUX_FLOOR: f64 = 0.01;
/// Fewest withheld stars an overlap may contribute.
const MIN_WITHHELD: usize = 5;
/// Fewest withheld stars for an overlap to be validated on its own; sparser
/// overlaps are validated together.
const MIN_OVERLAP_VALIDATION: usize = 10;
/// Prior standard deviation of each footprint-local log-response shape
/// coefficient (the footprint spans -1..1): about ten percent across a frame,
/// the size of the differences measured between uncalibrated instruments.
/// Stars decide wherever overlaps measure the shape; the prior decides only
/// the part no overlap can see, such as a shape shared by every frame of one
/// instrument outside the other instrument's field.
const RESPONSE_PRIOR: f64 = 0.1;
/// The spatial response must cut the pooled withheld-star median by this
/// factor to replace the simpler per-frame gain.
const SPATIAL_IMPROVEMENT: f64 = 0.95;

/// Common-coordinate normalization shared by every frame's spatial response.
#[derive(Clone, Copy)]
struct Normalization {
    center: [f64; 2],
    scale: f64,
}
impl Normalization {
    fn basis(&self, x: f64, y: f64) -> [f64; 6] {
        let u = (x - self.center[0]) / self.scale;
        let v = (y - self.center[1]) / self.scale;
        [1., u, v, u * u, u * v, v * v]
    }
    /// Centred on one footprint, spanning -1..1 along its longer side.
    fn of_footprint(bounds: [f64; 4]) -> Self {
        Self {
            center: [(bounds[0] + bounds[2]) / 2., (bounds[1] + bounds[3]) / 2.],
            scale: ((bounds[2] - bounds[0]).max(bounds[3] - bounds[1]) / 2.).max(f64::MIN_POSITIVE),
        }
    }
}

/// The matrix taking a quadratic's coefficients in `global` to the same
/// quadratic's coefficients in `local`. Exact: both bases span the same
/// polynomials, and a 3x3 lattice over the footprint determines a quadratic.
/// Global coefficients of an off-centre frame are large and cancel, which
/// makes interval bounds useless; local ones state the shape over the frame.
fn reparameterization(global: Normalization, local: Normalization, bounds: [f64; 4]) -> Result<[[f64; 6]; 6]> {
    let mut normal = [[0.; 6]; 6];
    let mut cross = [[0.; 6]; 6];
    for iy in 0..3 {
        for ix in 0..3 {
            let x = bounds[0] + (bounds[2] - bounds[0]) * ix as f64 / 2.;
            let y = bounds[1] + (bounds[3] - bounds[1]) * iy as f64 / 2.;
            let (l, g) = (local.basis(x, y), global.basis(x, y));
            for i in 0..6 {
                for j in 0..6 {
                    normal[i][j] += l[i] * l[j];
                    cross[i][j] += l[i] * g[j];
                }
            }
        }
    }
    solve6(normal, cross).context("degenerate stellar-response footprint")
}

/// Solves `a * X = b` for 6x6 matrices by Gauss-Jordan with partial pivoting.
fn solve6(mut a: [[f64; 6]; 6], mut b: [[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
    let size = a.iter().flatten().fold(0_f64, |m, v| m.max(v.abs()));
    for column in 0..6 {
        let pivot = (column..6).max_by(|&p, &q| a[p][column].abs().total_cmp(&a[q][column].abs()))?;
        let magnitude = a[pivot][column].abs();
        if magnitude.is_nan() || magnitude <= 1e-12 * size.max(f64::MIN_POSITIVE) {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in 0..6 {
            if row != column {
                let factor = a[row][column] / a[column][column];
                for k in 0..6 {
                    a[row][k] -= factor * a[column][k];
                    b[row][k] -= factor * b[column][k];
                }
            }
        }
    }
    for (row, line) in b.iter_mut().enumerate() {
        let d = a[row][row];
        for v in line.iter_mut() {
            *v /= d;
        }
    }
    b.iter().flatten().all(|v| v.is_finite()).then_some(b)
}

fn gain_rows(
    a: usize,
    b: usize,
    aa: &[Star],
    bb: &[Star],
    normalization: Normalization,
) -> Result<(Vec<Row>, GainEvidence)> {
    let ab = nearest(aa, bb);
    let ba = nearest(bb, aa);
    let matched: Vec<_> = ab
        .iter()
        .enumerate()
        .filter_map(|(i, j)| j.filter(|&j| ba[j] == Some(i)).map(|j| (i, j)))
        .collect();
    let mut training: Vec<_> = matched
        .iter()
        .filter(|(_, j)| j % 5 != 0)
        .map(|&(i, j)| (aa[i].flux / bb[j].flux).ln())
        .collect();
    // Withheld stars, most precise first: every one at VALIDATION_SNR, topped
    // up with the next brightest only where an overlap has fewer than the
    // minimum, so a sparse overlap is still tested rather than dropped.
    let snr = |&(i, j): &(usize, usize)| aa[i].snr.min(bb[j].snr);
    let mut candidates: Vec<_> = matched.iter().filter(|m| m.1 % 5 == 0).collect();
    candidates.sort_by(|p, q| snr(q).total_cmp(&snr(p)).then(p.1.cmp(&q.1)));
    let precise_count = candidates.iter().filter(|m| snr(m) >= VALIDATION_SNR).count();
    let validating: std::collections::HashSet<usize> = candidates
        .iter()
        .take(precise_count.max(MIN_WITHHELD))
        .map(|m| m.1)
        .collect();
    let held = validating.len();
    ensure!(
        training.len() >= 20 && held >= MIN_WITHHELD,
        "too few independent stellar flux matches ({}/{held})",
        training.len()
    );
    let center = median(&mut training).unwrap();
    let mut deviations: Vec<_> = training.iter().map(|v| (v - center).abs()).collect();
    let scatter = 1.4826 * median(&mut deviations).unwrap();
    ensure!(
        scatter <= 0.2 && 1.2533 * scatter / (training.len() as f64).sqrt() <= 0.02,
        "stellar photometry insufficiently precise"
    );
    let rows: Vec<_> = matched
        .iter()
        .filter_map(|&(i, j)| {
            let value = (aa[i].flux / bb[j].flux).ln();
            let test = j % 5 == 0;
            if test && !validating.contains(&j) {
                return None;
            }
            let shot = (aa[i].snr.powi(-2) + bb[j].snr.powi(-2)).sqrt();
            (test || (value - center).abs() <= 0.03_f64.max(3. * scatter)).then_some(Row {
                a,
                b,
                basis: normalization.basis(aa[i].x, aa[i].y),
                value,
                test,
                precision_weight: 1. / (shot.powi(2) + STAR_LOG_FLUX_FLOOR.powi(2)),
            })
        })
        .collect();
    let matches = rows.iter().filter(|r| !r.test).count();
    ensure!(matches >= 20, "too few robust stellar flux matches");
    let standard_error = scatter / (matches as f64).sqrt();
    ensure!(
        1.2533 * standard_error <= 0.02,
        "retained stellar photometry insufficiently precise"
    );
    Ok((
        rows,
        GainEvidence {
            matches,
            withheld: held,
            scatter,
            standard_error,
        },
    ))
}

/// `noise_floor` applies to background planes only: the p90 that withheld
/// patch differences would show from sampling noise alone. Below it, before and
/// after cannot be told apart, so a correction is not rejected for getting there.
fn validate(rows: &[Row], x: &[f64], anchor: usize, dim: usize, noise_floor: f64) -> Result<(f64, f64)> {
    let mut before: Vec<_> = rows
        .iter()
        .filter(|r| r.test)
        .map(|r| r.value.abs())
        .collect();
    let mut after: Vec<_> = rows
        .iter()
        .filter(|r| r.test)
        .map(|r| (r.value - prediction(r, x, anchor, dim)).abs())
        .collect();
    ensure!(
        after.len() >= 5,
        "insufficient withheld photometry evidence"
    );
    before.sort_by(f64::total_cmp);
    after.sort_by(f64::total_cmp);
    let (b, a) = (quantile(&before, 0.9), quantile(&after, 0.9));
    if dim == 1 || dim == 6 {
        let median_absolute=median(&mut after).unwrap();
        let mut signed:Vec<_>=rows.iter().filter(|r|r.test).map(|r|r.value-prediction(r,x,anchor,dim)).collect();
        let bias=median(&mut signed).unwrap();
        ensure!(
            median_absolute <= 0.05 && a <= 0.2,
            "{} withheld stellar fluxes reject gain: median absolute disagreement {median_absolute:.5} (limit 0.05000), p90 {a:.5} (limit 0.20000), signed median bias {bias:.5}",after.len()
        );
    } else {
        ensure!(
            a <= (1.15 * b).max(noise_floor).max(3e-5),
            "withheld overlap patches reject background plane ({b:.6} before, {a:.6} after, sampling-noise p90 {noise_floor:.6})"
        );
    }
    Ok((b, a))
}

/// Native aperture measurements are retained only in small per-frame catalogs;
/// at most two FITS handles and bounded patches are alive during overlap reads.
pub(crate) fn prepare(
    frames: &mut [FrameSpec],
    pairs: &[(usize, usize)],
    anchor: usize,
    memory_mb: usize,
    catalogs: &[Vec<[f64; 3]>],
) -> Result<Vec<String>> {
    let n = frames.len();
    ensure!(
        (2..=512).contains(&n) && anchor < n && catalogs.len() == n,
        "invalid photometry frame/catalog count"
    );
    ensure!(
        pairs.len() <= 4096 && pairs.iter().all(|&(a, b)| a < n && b < n && a != b),
        "invalid photometry pair budget"
    );
    ensure!(
        memory_mb >= 64 && catalogs.iter().all(|c| c.len() <= 6000),
        "photometry requires 64 MiB and at most 6000 detections/frame"
    );
    let budget = memory_mb
        .checked_mul(1024 * 1024)
        .context("photometry memory budget overflow")?;
    let catalog_bytes = catalogs
        .iter()
        .map(|c| c.capacity() * std::mem::size_of::<[f64; 3]>())
        .sum::<usize>();
    let star_bytes = catalogs
        .iter()
        .map(|c| c.len() * std::mem::size_of::<Star>())
        .sum::<usize>();
    ensure!(
        catalog_bytes + star_bytes + 24 * 1024 * 1024 <= budget,
        "stellar catalogs exceed preparation memory budget"
    );
    let row_limit = ((budget - catalog_bytes - 24 * 1024 * 1024)
        / (std::mem::size_of::<Row>() + 32))
        .min(1_048_576);
    let maximum_scale = frames
        .iter()
        .map(|f| {
            pixel_area(&f.projection, f.width as f64 / 2., f.height as f64 / 2.).map(f64::sqrt)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .fold(0., f64::max);
    let radius = 12. * maximum_scale;
    let frame_bounds = frames
        .iter()
        .map(|f| bounds(&f.projection, f.width, f.height))
        .collect::<Result<Vec<_>>>()?;
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for b in &frame_bounds {
        for k in 0..2 {
            lo[k] = lo[k].min(b[k]);
            hi[k] = hi[k].max(b[k + 2]);
        }
    }
    let center = [(lo[0] + hi[0]) / 2., (lo[1] + hi[1]) / 2.];
    let scale = ((hi[0] - lo[0]).max(hi[1] - lo[1])) / 2.;
    ensure!(scale.is_finite() && scale > 0., "invalid photometry extent");
    let normalization = Normalization { center, scale };
    let mut stars = Vec::new();
    let mut notes = Vec::new();
    for (i, f) in frames.iter_mut().enumerate() {
        eprintln!(
            "Measuring native stellar flux and noise: frame {}/{n}",
            i + 1
        );
        let mut reader = FitsWindowReader::open(&f.path, &sr_raw::ReadOptions::default())?;
        ensure!(
            reader.dimensions() == (f.width, f.height),
            "source dimensions changed"
        );
        ensure!(
            matches!(reader.sample_units(), SampleUnits::NormalizedInteger { .. }),
            "automatic photometry currently requires normalized integer FITS"
        );
        let (sky, sigma, patches) = sky_noise(&mut reader)?;
        ensure!(
            patches >= 32 && sigma.is_finite() && sigma > 0.,
            "frame {} has insufficient measured noise",
            i + 1
        );
        let (header, _) = sr_raw::fits::read_header(&f.path)?;
        let alpha = match (
            reader.sample_units(),
            header.number("EGAIN").filter(|g| g.is_finite() && *g > 0.),
        ) {
            (SampleUnits::NormalizedInteger { white_level }, Some(g)) => {
                shot_coefficient(white_level, g)
            }
            _ => 0.,
        };
        f.sky = sky as f32;
        f.noise = NoiseModel::new(
            alpha as f32,
            (sigma * sigma - alpha * sky).max(sigma * sigma * 0.05) as f32,
            NoiseSource::Measured,
        );
        let mut measured = Vec::with_capacity(catalogs[i].len());
        let mut hfds = Vec::new();
        let mut reader = WindowCache::new(reader);
        let measurements = aperture_catalog(&mut reader, &catalogs[i], radius, &f.projection)?;
        // Restore the original brightness/catalog order before any scientific
        // selection: withheld-star identity must not depend on I/O locality.
        for m in measurements.into_iter().flatten() {
            let area = pixel_area(&f.projection, m.x, m.y)?;
            let snr = m.flux / (sigma * (std::f64::consts::PI * radius * radius / area).sqrt());
            if m.peak < 0.8 && m.hfd >= 1.1 * area.sqrt() && snr >= 20. {
                let q = f
                    .projection
                    .map(m.x, m.y)
                    .context("invalid star projection")?;
                measured.push(Star {
                    x: q.0,
                    y: q.1,
                    flux: m.flux * area,
                    snr,
                });
                if snr >= 50. {
                    hfds.push(m.hfd);
                }
            }
        }
        ensure!(
            hfds.len() >= 10,
            "frame {} has fewer than ten high-SNR resolved stars for PSF measurement",
            i + 1
        );
        f.psf_hfd = median(&mut hfds).unwrap() as f32;
        notes.push(format!(
            "Frame {}: {} measured stellar fluxes, {} PSF stars, noise sigma {:.7}",
            i + 1,
            measured.len(),
            hfds.len(),
            sigma
        ));
        stars.push(measured);
    }
    let mut flux_rows = Vec::new();
    let mut star_rows = Vec::new();
    let mut evidence_by_pair = Vec::new();
    for &(a, b) in pairs {
        match gain_rows(a, b, &stars[a], &stars[b], normalization) {
            Ok((rows, evidence)) => {
                let mut train: Vec<_> = rows.iter().filter(|r| !r.test).map(|r| r.value).collect();
                flux_rows.push(Row {
                    a,
                    b,
                    basis: [1., 0., 0., 0., 0., 0.],
                    value: median(&mut train).unwrap(),
                    test: false,
                    precision_weight: 1. / evidence.standard_error.max(0.003).powi(2),
                });
                ensure!(
                    star_rows.len() + rows.len() <= row_limit,
                    "stellar evidence exceeds preparation memory budget"
                );
                let start = star_rows.len();
                star_rows.extend(rows);
                evidence_by_pair.push((a, b, start..star_rows.len(), evidence));
            }
            Err(e) => {
                if notes.len() < 700 {
                    notes.push(format!(
                        "Pair {}–{} lacks reliable gain evidence: {e}",
                        a + 1,
                        b + 1
                    ));
                }
            }
        }
    }
    drop(stars);
    let usable_pairs: Vec<_> = evidence_by_pair.iter().map(|e| (e.0, e.1)).collect();
    eprintln!(
        "Solving relative stellar fluxes across {} supported overlaps",
        usable_pairs.len()
    );
    // The scalar solve also proves the overlap graph connected; the spatial
    // solve below cannot, because its prior rows tie every frame to the anchor.
    let scalar = solve(&flux_rows, n, anchor, 1)?;
    drop(flux_rows);

    // Candidate two: one log-quadratic relative response per frame, fitted to
    // individual matched stars. Rows share one canvas-wide basis; each frame's
    // result is restated over its own footprint. The weak prior acts on those
    // footprint-local shape terms and only keeps what no overlap constrains
    // from wandering; measured stars outweigh it wherever they exist.
    let local: Vec<_> = frame_bounds.iter().map(|&b| Normalization::of_footprint(b)).collect();
    let to_local = frame_bounds
        .iter()
        .zip(&local)
        .map(|(&b, &l)| reparameterization(normalization, l, b))
        .collect::<Result<Vec<_>>>()?;
    let training_rows = star_rows.len();
    for f in (0..n).filter(|&f| f != anchor) {
        for &basis in &to_local[f][1..] {
            star_rows.push(Row {
                a: anchor,
                b: f,
                basis,
                value: 0.,
                test: false,
                precision_weight: 1. / RESPONSE_PRIOR.powi(2),
            });
        }
    }
    eprintln!("Solving relative stellar response across the field");
    let spatial = solve(&star_rows, n, anchor, 6);
    star_rows.truncate(training_rows);

    // Footprint-local coefficients of frame f: constant first, then shape.
    let local_coefficients = |x: &[f64], f: usize| -> [f64; 6] {
        std::array::from_fn(|k| dot(&to_local[f][k], &x[6 * f..6 * f + 6]))
    };
    let response = |x: &[f64], f: usize| {
        let c = local_coefficients(x, f);
        crate::mosaic::RelativeLogGain {
            center: local[f].center,
            normalization_scale: local[f].scale,
            coefficients: [c[1], c[2], c[3], c[4], c[5]],
        }
    };
    // Every applied overlap must pass its own withheld stars; the pooled
    // withheld median then compares candidates on identical evidence.
    // An overlap with too few withheld stars for its own median and p90 to mean
    // anything is not skipped: its stars join one pooled set that must pass
    // the same limits.
    let evaluate = |x: &[f64], dim: usize| -> Result<(f64, Vec<Option<f64>>)> {
        let mut p90s = Vec::new();
        let mut sparse = Vec::new();
        for (a, b, range, _) in &evidence_by_pair {
            let rows = &star_rows[range.clone()];
            if rows.iter().filter(|r| r.test).count() < MIN_OVERLAP_VALIDATION {
                sparse.extend(rows.iter().filter(|r| r.test).cloned());
                p90s.push(None);
                continue;
            }
            let (_, p90) = validate(rows, x, anchor, dim, 0.).with_context(|| {
                format!(
                    "stellar gain validation for frame {} ({}) and frame {} ({})",
                    a + 1,
                    frames[*a].path.display(),
                    b + 1,
                    frames[*b].path.display()
                )
            })?;
            p90s.push(Some(p90));
        }
        if !sparse.is_empty() {
            validate(&sparse, x, anchor, dim, 0.).with_context(|| {
                format!(
                    "stellar gain validation pooled over {} sparse overlaps",
                    p90s.iter().filter(|p| p.is_none()).count()
                )
            })?;
        }
        let mut pooled: Vec<_> = star_rows
            .iter()
            .filter(|r| r.test)
            .map(|r| (r.value - prediction(r, x, anchor, dim)).abs())
            .collect();
        Ok((median(&mut pooled).context("no withheld stars")?, p90s))
    };
    let scalar_result = evaluate(&scalar, 1);
    let spatial_result = spatial.and_then(|x| {
        for f in (0..n).filter(|&f| f != anchor) {
            let [lo, hi] = response(&x, f).log_bounds(frame_bounds[f])?;
            ensure!(
                lo >= -crate::mosaic::MAX_RELATIVE_RESPONSE.ln() && hi <= crate::mosaic::MAX_RELATIVE_RESPONSE.ln(),
                "relative stellar response for frame {} spans {:.3}–{:.3} over its footprint, outside {}",
                f + 1,
                lo.exp(),
                hi.exp(),
                crate::mosaic::RESPONSE_RANGE_TEXT
            );
        }
        evaluate(&x, 6).map(|r| (x, r))
    });
    let use_spatial = match (&scalar_result, &spatial_result) {
        (Ok((s, _)), Ok((_, (p, _)))) => *p <= SPATIAL_IMPROVEMENT * s,
        (Err(_), Ok(_)) => true,
        (Ok(_), Err(_)) => false,
        (Err(s), Err(p)) => anyhow::bail!(
            "no relative flux model passes withheld stars.\nPer-frame gain: {s:#}\nSpatial response: {p:#}"
        ),
    };
    match (&scalar_result, &spatial_result) {
        (Ok((s, _)), Ok((_, (p, _)))) => notes.push(format!(
            "Withheld-star median log-flux disagreement: per-frame gain {s:.5}, spatial response {p:.5}; {} applied.",
            if use_spatial { "spatial response" } else { "per-frame gain" }
        )),
        (Ok((s, _)), Err(e)) => notes.push(format!(
            "Per-frame gain applied (withheld median {s:.5}); spatial response rejected: {e:#}"
        )),
        (Err(e), Ok((_, (p, _)))) => notes.push(format!(
            "Spatial response applied (withheld median {p:.5}); per-frame gain rejected: {e:#}"
        )),
        (Err(_), Err(_)) => unreachable!(),
    }
    let p90s = if use_spatial {
        let (x, (_, p90s)) = spatial_result?;
        for (f, frame) in frames.iter_mut().enumerate() {
            frame.gain = local_coefficients(&x, f)[0].exp() as f32;
            frame.relative_log_gain = (f != anchor).then(|| response(&x, f));
        }
        p90s
    } else {
        let (_, p90s) = scalar_result?;
        for (frame, g) in frames.iter_mut().zip(&scalar) {
            frame.gain = g.exp() as f32;
            frame.relative_log_gain = None;
        }
        p90s
    };
    for frame in frames.iter() {
        ensure!(
            frame.gain.is_finite() && frame.gain > 0.,
            "invalid relative flux gain"
        );
    }
    for ((a, b, _, evidence), p90) in evidence_by_pair.iter().zip(p90s) {
        if notes.len() < 850 {
            let p90 = p90.map_or("pooled with sparse overlaps".into(), |p| format!("{p:.5}"));
            notes.push(format!("Stellar overlap {}–{}: {} training / {} withheld stars (SNR ≥ {VALIDATION_SNR} first); log-flux scatter {:.5}, measured standard error {:.5} (fit floor 0.003), withheld p90 {p90}",a+1,b+1,evidence.matches,evidence.withheld,evidence.scatter,evidence.standard_error));
        }
    }
    drop(star_rows);
    let mut rows = Vec::with_capacity((usable_pairs.len() * 256).min(row_limit));
    let mut noise_floors = HashMap::new();
    for (pair_index, &(a, b)) in usable_pairs.iter().enumerate() {
        eprintln!(
            "Measuring same-sky overlap: {}/{}",
            pair_index + 1,
            usable_pairs.len()
        );
        let ab = frame_bounds[a];
        let bb = frame_bounds[b];
        let region = [
            ab[0].max(bb[0]),
            ab[1].max(bb[1]),
            ab[2].min(bb[2]),
            ab[3].min(bb[3]),
        ];
        if region[2] <= region[0] || region[3] <= region[1] {
            continue;
        }
        let mut ar = WindowCache::open(&frames[a].path)?;
        let mut br = WindowCache::open(&frames[b].path)?;
        // A 16x16 normalized overlap lattice bounds I/O and metadata per pair.
        // Separate checkerboard support is frozen before the training-only cut.
        let mut samples = Vec::new();
        let mut spreads = Vec::new();
        let separation = 16. * maximum_scale + 2.;
        let nx = (((region[2] - region[0]) / separation).floor() as usize).min(16);
        let ny = (((region[3] - region[1]) / separation).floor() as usize).min(16);
        if nx < 3 || ny < 3 {
            continue;
        }
        for iy in 0..ny {
            for ix in 0..nx {
                let x = region[0] + (ix as f64 + 0.5) * (region[2] - region[0]) / nx as f64;
                let y = region[1] + (iy as f64 + 0.5) * (region[3] - region[1]) / ny as f64;
                if let (Some(av), Some(bv)) = (
                    patch(&mut ar, &frames[a].projection, x, y, 8. * maximum_scale)?,
                    patch(&mut br, &frames[b].projection, x, y, 8. * maximum_scale)?,
                ) {
                    let (ga, gb) = (frames[a].matched_gain(x, y), frames[b].matched_gain(x, y));
                    let spread = (av[2] * ga).max(bv[2] * gb);
                    // Sampling noise of the two patch medians, from each frame's
                    // measured sky noise rather than the patch spread, which
                    // also contains real structure.
                    let median_noise = |f: &FrameSpec, gain: f64, count: f64| {
                        1.2533 * gain * f64::from(f.noise.variance(f.sky)).sqrt() / count.sqrt()
                    };
                    let noise = median_noise(&frames[a], ga, av[1]).hypot(median_noise(&frames[b], gb, bv[1]));
                    let test = (ix + iy) % 2 != 0;
                    if !test {
                        spreads.push(spread);
                    }
                    samples.push((
                        Row {
                            a,
                            b,
                            basis: [1., (x - center[0]) / scale, (y - center[1]) / scale, 0., 0., 0.],
                            value: ga * av[0] - gb * bv[0],
                            test,
                            precision_weight: 1.,
                        },
                        spread,
                        noise,
                    ));
                }
            }
        }
        if spreads.len() < 12 {
            continue;
        }
        spreads.sort_by(f64::total_cmp);
        let cutoff = quantile(&spreads, 0.75);
        let samples: Vec<_> = samples.into_iter().filter(|(_, s, _)| *s <= cutoff).collect();
        let mut test_noise: Vec<_> = samples.iter().filter(|(r, _, _)| r.test).map(|(_, _, n)| *n).collect();
        let selected: Vec<_> = samples.into_iter().map(|(r, _, _)| r).collect();
        if selected.iter().filter(|r| r.test).count() < 8
            || selected.iter().filter(|r| !r.test).count() < 12
        {
            continue;
        }
        spatial_support(&selected)?;
        ensure!(
            rows.len() + selected.len() <= row_limit,
            "overlap evidence exceeds preparation memory budget"
        );
        rows.extend(selected);
        // p90 of |N(0, sigma)| at the overlap's typical withheld-patch noise.
        noise_floors.insert((a, b), 1.645 * median(&mut test_noise).unwrap_or(0.));
    }
    eprintln!("Solving overlap-only additive planes with independent validation");
    let planes = solve(&rows, n, anchor, 3)?;
    for &(a, b) in &usable_pairs {
        let pair: Vec<_> = rows
            .iter()
            .filter(|r| r.a == a && r.b == b)
            .cloned()
            .collect();
        if pair.is_empty() {
            continue;
        }
        let (before, after) = validate(&pair, &planes, anchor, 3, noise_floors[&(a, b)])
            .with_context(|| format!("background validation for frames {}–{}", a + 1, b + 1))?;
        if notes.len() < 1000 {
            notes.push(format!(
                "Overlap {}–{}: withheld background p90 {:.7} → {:.7}",
                a + 1,
                b + 1,
                before,
                after
            ));
        }
    }
    for (i, f) in frames.iter_mut().enumerate() {
        let dx = planes[3 * i + 1] / scale;
        let dy = planes[3 * i + 2] / scale;
        f.offset = (planes[3 * i] - dx * center[0] - dy * center[1]) as f32;
        f.background_plane = [dx as f32, dy as f32];
        f.background_quadratic = [0.; 3];
    }
    notes.push("Automatic native common-angular-aperture stellar gains (optionally with a relative log-quadratic response) and overlap-only additive planes; held-out stars (SNR ≥ 50) and checkerboard patches validated every applied overlap. Anchor retains its full diffuse scene. No individual-image flattening or flat-field recovery.".into());
    notes.push("Additive planes outside the measured overlaps are extrapolation, not validated calibration. Measured adjacent-difference noise is an empirical approximation; source calibration and optical flat-field response remain unknown.".into());
    Ok(notes)
}

fn spatial_support(rows: &[Row]) -> Result<()> {
    let train: Vec<_> = rows.iter().filter(|r| !r.test).collect();
    let n = train.len() as f64;
    let x = train.iter().map(|r| r.basis[1]).sum::<f64>() / n;
    let y = train.iter().map(|r| r.basis[2]).sum::<f64>() / n;
    let xx = train.iter().map(|r| (r.basis[1] - x).powi(2)).sum::<f64>();
    let yy = train.iter().map(|r| (r.basis[2] - y).powi(2)).sum::<f64>();
    let xy = train
        .iter()
        .map(|r| (r.basis[1] - x) * (r.basis[2] - y))
        .sum::<f64>();
    ensure!(
        xx > 1e-10 && yy > 1e-10 && xx * yy - xy * xy > 1e-4 * xx * yy,
        "overlap patches do not constrain a two-dimensional background plane"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn projection(scale: f64) -> FrameProjection {
        FrameProjection {
            center: [0.; 2],
            normalization_scale: 1.,
            distortion: [0.; 3],
            homography: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            output_center: [0.; 2],
            output_scale: scale,
        }
    }
    fn fixture(scale: f64, gain: f64, extra: bool) -> (Fixture, FrameSpec, Vec<[f64; 3]>) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "smokstak-auto-photo-{}-{}.fits",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let width = (512. / scale).round() as usize;
        let mut header = String::new();
        for (key, value) in [
            ("SIMPLE", "T".to_string()),
            ("BITPIX", "16".to_string()),
            ("NAXIS", "2".to_string()),
            ("NAXIS1", width.to_string()),
            ("NAXIS2", width.to_string()),
            ("BZERO", "32768".to_string()),
        ] {
            header.push_str(&format!("{:<80}", format!("{key:<8}= {value}")));
        }
        header.push_str(&format!("{:<80}", "END"));
        let mut bytes = header.into_bytes();
        bytes.resize(2880, b' ');
        let mut catalog = Vec::new();
        for gy in 0..9 {
            for gx in 0..9 {
                catalog.push([
                    (64. + gx as f64 * 48.) / scale,
                    (64. + gy as f64 * 48.) / scale,
                    1.,
                ]);
            }
        }
        for iy in 0..width {
            for ix in 0..width {
                let x = ix as f64 * scale;
                let y = iy as f64 * scale;
                let mut scene = 0.1 + 0.008 * (x / 180.).sin() * (y / 130.).cos() + 0.00002 * x;
                for s in &catalog {
                    let r2 = (x - s[0] * scale).powi(2) + (y - s[1] * scale).powi(2);
                    if r2 < 250. {
                        scene += 0.18 * (-r2 / 8.).exp();
                    }
                }
                let extra = if extra {
                    0.006 + 0.000002 * x - 0.000003 * y
                } else {
                    0.
                };
                let hash = (ix as u64 * 73856093) ^ (iy as u64 * 19349663);
                let noise = (hash % 17) as f64 - 8.;
                let value = ((scene / gain + extra) * 65535. + noise).round() as i32;
                bytes.extend_from_slice(&((value - 32768) as i16).to_be_bytes());
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let f = FrameSpec {
            path: path.clone(),
            label: String::new(),
            group: String::new(),
            width,
            height: width,
            bytes: bytes.len() as u64,
            sha256: String::new(),
            projection: projection(scale),
            noise: NoiseModel::new(0., 1e-6, NoiseSource::Measured),
            sky: 0.,
            gain: 1.,
            relative_log_gain: None,
            offset: 0.,
            background_plane: [0.; 2],
            background_quadratic: [0.; 3],
            weight: 1.,
            psf_hfd: 1.,
            registration_p50: 0.,
            registration_p90: 0.,
            validation_stars: 100,
        };
        (Fixture(path), f, catalog)
    }
    #[test]
    fn native_mixed_scale_preserves_stellar_flux_and_diffuse_scene() {
        let (_fa, a, ca) = fixture(1., 1., false);
        let (_fb, b, cb) = fixture(2. / 3., 1.4, true);
        let mut frames = vec![a, b];
        let notes = prepare(&mut frames, &[(0, 1)], 0, 64, &[ca, cb]).unwrap();
        assert!(
            (frames[1].gain - 1.4).abs() < 0.01,
            "gain {}",
            frames[1].gain
        );
        assert_eq!(frames[0].offset, 0.);
        assert_eq!(frames[0].background_plane, [0.; 2]);
        for (x, y) in [(80., 80.), (256., 300.), (440., 430.)] {
            let scene =
                0.1 + 0.008_f64 * (x / 180.0_f64).sin() * (y / 130.0_f64).cos() + 0.00002 * x;
            let measured = scene / 1.4 + 0.006 + 0.000002 * x - 0.000003 * y;
            let recovered = measured * frames[1].gain as f64
                + frames[1].offset as f64
                + frames[1].background_plane[0] as f64 * x
                + frames[1].background_plane[1] as f64 * y;
            assert!(
                (recovered - scene).abs() < 0.0002,
                "diffuse signal {scene} -> {recovered}"
            );
        }
        assert!((frames[0].psf_hfd - frames[1].psf_hfd).abs() < 0.8);
        assert!(frames.iter().all(|f| f.noise.beta > 0.));
        assert!(notes
            .iter()
            .any(|note| note.contains("No individual-image flattening")));
    }
    #[test]
    fn withheld_background_disagreement_is_not_clipped() {
        let mut rows = Vec::new();
        for iy in 0..8 {
            for ix in 0..8 {
                let test = (ix + iy) % 2 != 0;
                let basis = [1., ix as f64 / 8., iy as f64 / 8., 0., 0., 0.];
                rows.push(Row {
                    a: 0,
                    b: 1,
                    basis,
                    value: if test { 0. } else { 0.01 + basis[1] * 0.02 },
                    test,
                    precision_weight: 1.,
                });
            }
        }
        spatial_support(&rows).unwrap();
        let result = solve(&rows, 2, 0, 3).unwrap();
        assert!(validate(&rows, &result, 0, 3, 0.).is_err());
    }
    #[test]
    fn gain_test_stars_cannot_be_clipped_to_force_acceptance() {
        let aa: Vec<_> = (0..100)
            .map(|i| Star {
                x: (i % 10) as f64 * 5.,
                y: (i / 10) as f64 * 5.,
                flux: 1.,
                snr: 100.,
            })
            .collect();
        let bb: Vec<_> = aa
            .iter()
            .enumerate()
            .map(|(i, s)| Star {
                x: s.x,
                y: s.y,
                flux: if i % 5 == 0 { 0.5 } else { 1. },
                snr: 100.,
            })
            .collect();
        let (rows, _) = gain_rows(0, 1, &aa, &bb, UNIT).unwrap();
        let result = solve(&rows, 2, 0, 1).unwrap();
        assert!(validate(&rows, &result, 0, 1, 0.).is_err());
    }

    const UNIT: Normalization = Normalization { center: [25., 25.], scale: 25. };

    /// A star field seen by two frames whose relative response differs by
    /// `log_ratio(u, v)`, with every star at the given SNR.
    fn field(log_ratio: impl Fn(f64, f64) -> f64, snr: impl Fn(usize) -> f64) -> (Vec<Star>, Vec<Star>) {
        let aa: Vec<_> = (0..400)
            .map(|i| Star { x: (i % 20) as f64 * 2.5, y: (i / 20) as f64 * 2.5, flux: 1., snr: snr(i) })
            .collect();
        let bb = aa
            .iter()
            .map(|s| {
                let [_, u, v, ..] = UNIT.basis(s.x, s.y);
                Star { flux: s.flux * (-log_ratio(u, v)).exp(), ..*s }
            })
            .collect();
        (aa, bb)
    }

    #[test]
    fn a_frame_response_restated_over_its_footprint_is_the_same_surface() {
        // An off-centre frame on a wide canvas: large global terms that cancel
        // become small local ones, and the surface itself is unchanged.
        let global = Normalization { center: [0., 0.], scale: 10_000. };
        let bounds = [6_000., -2_500., 9_000., -500.];
        let local = Normalization::of_footprint(bounds);
        let map = reparameterization(global, local, bounds).unwrap();
        let x = [0.4, -3.1, 1.7, 2.2, -0.9, 1.3];
        let y: [f64; 6] = std::array::from_fn(|k| dot(&map[k], &x));
        for (px, py) in [(6_000., -2_500.), (7_123., -1_111.), (9_000., -500.), (8_500., -2_400.)] {
            let a = dot(&global.basis(px, py), &x);
            let b = dot(&local.basis(px, py), &y);
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn background_disagreement_within_sampling_noise_is_not_a_rejection() {
        // Withheld patches already agree to within their own noise; a plane
        // pulled slightly by the rest of the graph leaves them just as noisy.
        let rows: Vec<_> = (0..20)
            .map(|i| Row {
                a: 0,
                b: 1,
                basis: [1., 0., 0., 0., 0., 0.],
                value: if i % 2 == 0 { 1e-4 } else { -1e-4 },
                test: true,
                precision_weight: 1.,
            })
            .collect();
        let plane = [0., 0., 0., 4e-5, 0., 0.];
        assert!(validate(&rows, &plane, 0, 3, 0.).is_err());
        assert!(validate(&rows, &plane, 0, 3, 2e-4).is_ok());
        assert!(validate(&rows, &plane, 0, 3, 1.2e-4).is_err(), "the floor does not excuse a worse-than-noise fit");
    }

    #[test]
    fn faint_withheld_stars_neither_validate_nor_train() {
        // Every fifth star of b is withheld; half of those are faint, and the
        // faint ones disagree wildly. They must be dropped, not tested and
        // not moved into training, while the bright withheld stars decide.
        let (aa, mut bb) = field(|_, _| 0.1, |i| if i % 10 == 0 { 25. } else { 100. });
        for (i, s) in bb.iter_mut().enumerate() {
            if i % 10 == 0 {
                s.flux *= 2.;
            }
        }
        let (rows, evidence) = gain_rows(0, 1, &aa, &bb, UNIT).unwrap();
        assert_eq!(evidence.withheld, 40);
        assert_eq!(rows.iter().filter(|r| r.test).count(), 40);
        assert_eq!(rows.iter().filter(|r| !r.test).count(), 320);
        let fit = solve(&rows, 2, 0, 1).unwrap();
        assert!((fit[1] - 0.1).abs() < 1e-6, "{}", fit[1]);
        validate(&rows, &fit, 0, 1, 0.).unwrap();
    }

    #[test]
    fn spatial_response_is_recovered_where_a_scalar_gain_fails() {
        // Vignetting that differs between two optical trains: a radial bowl
        // of 0.3 in log flux at the corners of the shared field.
        let truth = [0.05, 0.02, -0.03, 0.15, 0.01, 0.15];
        let (aa, bb) = field(
            |u, v| truth[0] + truth[1] * u + truth[2] * v + truth[3] * u * u + truth[4] * u * v + truth[5] * v * v,
            |_| 200.,
        );
        let (mut rows, _) = gain_rows(0, 1, &aa, &bb, UNIT).unwrap();
        let mut medians: Vec<_> = rows.iter().filter(|r| !r.test).map(|r| r.value).collect();
        let scalar_row = Row { a: 0, b: 1, basis: [1., 0., 0., 0., 0., 0.], value: median(&mut medians).unwrap(), test: false, precision_weight: 1. };
        let scalar = solve(&[scalar_row], 2, 0, 1).unwrap();
        assert!(validate(&rows, &scalar, 0, 1, 0.).is_err(), "a scalar cannot absorb a 0.3 bowl");
        let training = rows.len();
        for k in 1..6 {
            let mut basis = [0.; 6];
            basis[k] = 1.;
            rows.push(Row { a: 0, b: 1, basis, value: 0., test: false, precision_weight: 1. / RESPONSE_PRIOR.powi(2) });
        }
        let fit = solve(&rows, 2, 0, 6).unwrap();
        rows.truncate(training);
        for k in 0..6 {
            assert!((fit[6 + k] - truth[k]).abs() < 2e-3, "term {k}: {} vs {}", fit[6 + k], truth[k]);
        }
        validate(&rows, &fit, 0, 6, 0.).unwrap();
    }
    #[test]
    fn disconnected_or_collinear_evidence_is_rejected() {
        let rows: Vec<_> = (0..20)
            .map(|i| Row {
                a: 0,
                b: 1,
                basis: [1., i as f64 / 20., i as f64 / 20., 0., 0., 0.],
                value: 0.,
                test: false,
                precision_weight: 1.,
            })
            .collect();
        assert!(spatial_support(&rows).is_err());
        assert!(solve(&rows, 3, 0, 1).is_err());
    }
    #[test]
    fn precise_gain_edges_dominate_noisy_measurements() {
        let rows = vec![
            Row {
                a: 0,
                b: 1,
                basis: [1., 0., 0., 0., 0., 0.],
                value: 0.1,
                test: false,
                precision_weight: 1. / 0.003_f64.powi(2),
            },
            Row {
                a: 0,
                b: 1,
                basis: [1., 0., 0., 0., 0., 0.],
                value: 0.2,
                test: false,
                precision_weight: 1. / 0.05_f64.powi(2),
            },
        ];
        let fit = solve(&rows, 2, 0, 1).unwrap();
        assert!(
            (fit[1] - 0.1).abs() < 0.001,
            "precise edge was overwhelmed: {}",
            fit[1]
        );
    }
    #[test]
    fn shot_noise_matches_left_shifted_fits_units() {
        assert!((shot_coefficient(65532., 0.25) - 16. / 65532.).abs() < 1e-15);
        assert!((shot_coefficient(65535., 0.25) - 4. / 65535.).abs() < 1e-15);
    }
    #[test]
    fn cached_rectangles_preserve_exact_samples_across_tiles_edges_and_eviction() {
        let path =
            std::env::temp_dir().join(format!("smokstak-window-cache-{}.fits", std::process::id()));
        let _fixture = Fixture(path.clone());
        let mut header = String::new();
        for (key, value) in [
            ("SIMPLE", "T"),
            ("BITPIX", "16"),
            ("NAXIS", "2"),
            ("NAXIS1", "1031"),
            ("NAXIS2", "1553"),
            ("BZERO", "32768"),
        ] {
            header.push_str(&format!("{:<80}", format!("{key:<8}= {value}")));
        }
        header.push_str(&format!("{:<80}", "END"));
        let mut bytes = header.into_bytes();
        bytes.resize(2880, b' ');
        for y in 0..1553 {
            for x in 0..1031 {
                let value = ((x * 127 + y * 31) % 65536) - 32768;
                bytes.extend_from_slice(&(value as i16).to_be_bytes());
            }
        }
        std::fs::write(&path, bytes).unwrap();
        let mut direct = FitsWindowReader::open(&path, &sr_raw::ReadOptions::default()).unwrap();
        let mut cached = WindowCache::open(&path).unwrap();
        // Visit distinct bands, forcing eviction, before revisiting windows
        // that straddle band edges, the partial last band, and a window taller
        // than the whole cache.
        let mut windows = Vec::new();
        for y in [0, 512, 1024, 1536] {
            for x in [0, 512, 1024] {
                windows.push((x, y, 3, 3));
            }
        }
        windows.extend([
            (0, 0, 1, 1),
            (503, 508, 35, 29),
            (1020, 1530, 11, 23),
            (500, 1000, 531, 553),
        ]);
        for (x, y, w, h) in windows {
            let expected = direct.read_rect(x, y, w, h).unwrap();
            let actual = cached.read_rect(x, y, w, h).unwrap();
            assert_eq!(actual, expected, "window {x},{y} {w}x{h}");
            assert!(cached.tiles.len() <= CACHE_BANDS);
            assert!(
                cached
                    .tiles
                    .iter()
                    .map(|t| t.pixels.capacity() * 4)
                    .sum::<usize>()
                    <= CACHE_BANDS * CACHE_BAND_ROWS * 1031 * 4
            );
        }
        assert!(cached.read_rect(1030, 0, 2, 1).is_err());
        assert!(cached.read_rect(usize::MAX, 0, 2, 1).is_err());
    }
    #[test]
    fn cached_apertures_restore_exact_original_catalog_order() {
        let (_file, frame, mut catalog) = fixture(2. / 3., 1.4, true);
        catalog.sort_by_key(|s| ((s[0] as usize * 73) ^ (s[1] as usize * 19)) % 997);
        let mut direct = WindowCache::open(&frame.path).unwrap();
        direct.direct = true;
        let start = std::time::Instant::now();
        let expected: Vec<_> = catalog
            .iter()
            .map(|&s| aperture(&mut direct, s, 12., &frame.projection).unwrap())
            .collect();
        let original_time = start.elapsed();
        let mut cached = WindowCache::open(&frame.path).unwrap();
        let start = std::time::Instant::now();
        let actual = aperture_catalog(&mut cached, &catalog, 12., &frame.projection).unwrap();
        eprintln!(
            "81 native apertures: direct {original_time:?}, cached {:?}",
            start.elapsed()
        );
        assert_eq!(actual, expected);
    }
    /// Explicit developer diagnostic: bounded catalogs and original native
    /// aperture evidence are saved for investigating a rejected real overlap.
    #[test]
    #[ignore]
    fn real_pair_diagnostic() {
        #[derive(serde::Deserialize)]
        struct Request {
            frames: Vec<FrameSpec>,
            catalogs: Vec<Vec<[f64; 3]>>,
        }
        let path = PathBuf::from(
            std::env::var("SMOKSTAK_PHOTOMETRY_DIAGNOSTIC").expect("set diagnostic request path"),
        );
        let request: Request = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(request.frames.len() <= 4 && request.frames.len() == request.catalogs.len());
        assert!(request.catalogs.iter().all(|c| c.len() <= 6000));
        let radius = 12.
            * request
                .frames
                .iter()
                .map(|f| {
                    pixel_area(&f.projection, f.width as f64 / 2., f.height as f64 / 2.)
                        .unwrap()
                        .sqrt()
                })
                .fold(0., f64::max);
        let mut result = Vec::new();
        for (f, catalog) in request.frames.iter().zip(&request.catalogs) {
            let mut source =
                FitsWindowReader::open(&f.path, &sr_raw::ReadOptions::default()).unwrap();
            let (sky, sigma, _) = sky_noise(&mut source).unwrap();
            let mut cached = WindowCache::new(source);
            let measured = aperture_catalog(&mut cached, catalog, radius, &f.projection).unwrap();
            let stars:Vec<_>=measured.iter().enumerate().filter_map(|(index,m)|m.as_ref().map(|m|{
                let area=pixel_area(&f.projection,m.x,m.y).unwrap();let q=f.projection.map(m.x,m.y).unwrap();
                serde_json::json!({"index":index,"x":q.0,"y":q.1,"source_x":m.x,"source_y":m.y,"flux":m.flux*area,"hfd":m.hfd,"peak":m.peak,"snr":m.flux/(sigma*(std::f64::consts::PI*radius*radius/area).sqrt()),"resolved":m.hfd>=1.1*area.sqrt()})
            })).collect();
            result.push(serde_json::json!({"path":f.path,"sky":sky,"sigma":sigma,"stars":stars}));
        }
        let output = path.with_extension("measurements.json");
        std::fs::write(
            &output,
            serde_json::to_vec_pretty(&serde_json::json!({"radius":radius,"frames":result}))
                .unwrap(),
        )
        .unwrap();
        eprintln!("{}", output.display());
    }
}
