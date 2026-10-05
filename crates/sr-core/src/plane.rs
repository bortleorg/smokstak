//! Dense 2D buffers.
//!
//! Every raster in the pipeline is a `Plane<T>`: row-major, tightly packed, no
//! stride padding. Sensor samples, weights, masks and diagnostics all use it so
//! that geometry code can be written once.

use crate::buffer::{Buffer, Primitive};
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, PartialEq)]
pub struct Plane<T> {
    pub width: usize,
    pub height: usize,
    pub data: Buffer<T>,
}

impl<T: Clone + Default> Plane<T> {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![T::default(); width * height].into(),
        }
    }
}

impl<T: Clone> Plane<T> {
    pub fn filled(width: usize, height: usize, value: T) -> Self {
        Self {
            width,
            height,
            data: vec![value; width * height].into(),
        }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), width * height, "plane data length mismatch");
        Self {
            width,
            height,
            data: data.into(),
        }
    }
}

impl<T> Plane<T> {
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.data[y * self.width + x]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T
    where
        T: Clone,
    {
        &mut self.data[y * self.width + x]
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T]
    where
        T: Clone,
    {
        let w = self.width;
        &mut self.data[y * w..(y + 1) * w]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

impl<T> Index<(usize, usize)> for Plane<T> {
    type Output = T;
    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &T {
        &self.data[y * self.width + x]
    }
}

impl<T: Clone> IndexMut<(usize, usize)> for Plane<T> {
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut T {
        let w = self.width;
        &mut self.data[y * w + x]
    }
}

impl<T: Primitive> Plane<T> {
    pub fn spill(&mut self, dir: &std::path::Path) -> std::io::Result<()> {
        self.data.spill(dir)
    }
}

impl Plane<f32> {
    /// Nearest-neighbour fetch with edge clamping. Used where an out-of-bounds
    /// read must not fail (kernel footprints straddling the border).
    #[inline]
    pub fn at_clamped(&self, x: i64, y: i64) -> f32 {
        let x = x.clamp(0, self.width as i64 - 1) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        self.data[y * self.width + x]
    }

    /// Bilinear sample in pixel-centre coordinates: integer `(x, y)` addresses
    /// the centre of pixel `(x, y)`.
    #[inline]
    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let x0 = x0 as i64;
        let y0 = y0 as i64;
        let p00 = self.at_clamped(x0, y0);
        let p10 = self.at_clamped(x0 + 1, y0);
        let p01 = self.at_clamped(x0, y0 + 1);
        let p11 = self.at_clamped(x0 + 1, y0 + 1);
        let a = p00 + (p10 - p00) * fx;
        let b = p01 + (p11 - p01) * fx;
        a + (b - a) * fy
    }

    /// Bilinear sample that reports whether the footprint was fully inside.
    #[inline]
    pub fn bilinear_checked(&self, x: f32, y: f32) -> Option<f32> {
        if x < 0.0 || y < 0.0 || x > (self.width - 1) as f32 || y > (self.height - 1) as f32 {
            return None;
        }
        Some(self.bilinear(x, y))
    }

    pub fn min_max(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in &self.data {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if lo > hi { (0.0, 1.0) } else { (lo, hi) }
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let s: f64 = self.data.iter().map(|&v| v as f64).sum();
        (s / self.data.len() as f64) as f32
    }

    /// Percentile over finite values. `p` in `[0, 1]`.
    pub fn percentile(&self, p: f32) -> f32 {
        let mut v: Vec<f32> = self
            .data
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i = ((v.len() - 1) as f32 * p.clamp(0.0, 1.0)).round() as usize;
        v[i]
    }

    /// 2x decimation by 2x2 box average. Cheap pyramid level for registration.
    pub fn downsample2(&self) -> Plane<f32> {
        let w = self.width / 2;
        let h = self.height / 2;
        let mut out = Plane::new(w, h);
        for y in 0..h {
            let r0 = self.row(2 * y);
            let r1 = self.row(2 * y + 1);
            let o = out.row_mut(y);
            for x in 0..w {
                o[x] = 0.25 * (r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1]);
            }
        }
        out
    }

    /// Separable 3-tap binomial blur (1 2 1)/4, edge-clamped.
    /// Catmull-Rom bicubic sample, edge-clamped.
    ///
    /// Bilinear is what the registration uses, because there it is sampling a
    /// proxy to measure a displacement and the loss does not reach the output.
    /// A composite resamples a finished image, so the loss *is* the output, and
    /// bilinear costs real contrast at the scale where the detail of a 2x
    /// reconstruction lives. Catmull-Rom is interpolating — it passes through
    /// the samples — and costs four taps per axis instead of two.
    pub fn catmull_rom(&self, x: f32, y: f32) -> f32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let w = |t: f32| -> [f32; 4] {
            let t2 = t * t;
            let t3 = t2 * t;
            [
                0.5 * (-t3 + 2.0 * t2 - t),
                0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
                0.5 * (-3.0 * t3 + 4.0 * t2 + t),
                0.5 * (t3 - t2),
            ]
        };
        let (wx, wy) = (w(fx), w(fy));
        let mut acc = 0.0f32;
        for (j, wyj) in wy.iter().enumerate() {
            let sy = y0 as i64 + j as i64 - 1;
            let mut row = 0.0f32;
            for (i, wxi) in wx.iter().enumerate() {
                row += wxi * self.at_clamped(x0 as i64 + i as i64 - 1, sy);
            }
            acc += wyj * row;
        }
        acc
    }

    pub fn blur3(&self) -> Plane<f32> {
        let (w, h) = (self.width, self.height);
        let mut tmp = Plane::new(w, h);
        for y in 0..h {
            let src = self.row(y);
            let dst = tmp.row_mut(y);
            for x in 0..w {
                let a = src[x.saturating_sub(1)];
                let b = src[x];
                let c = src[(x + 1).min(w - 1)];
                dst[x] = 0.25 * (a + 2.0 * b + c);
            }
        }
        let mut out = Plane::new(w, h);
        for y in 0..h {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            for x in 0..w {
                out.data[y * w + x] = 0.25
                    * (tmp.data[ym * w + x] + 2.0 * tmp.data[y * w + x] + tmp.data[yp * w + x]);
            }
        }
        out
    }

    /// Repeated `blur3` approximates a Gaussian; `n` passes give sigma ~= sqrt(n/2).
    pub fn blur_n(&self, n: usize) -> Plane<f32> {
        let mut cur = self.clone();
        for _ in 0..n {
            cur = cur.blur3();
        }
        cur
    }

    pub fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> Plane<f32> {
        let mut out = Plane::new(w, h);
        for y in 0..h {
            let src = &self.data[(y0 + y) * self.width + x0..(y0 + y) * self.width + x0 + w];
            out.row_mut(y).copy_from_slice(src);
        }
        out
    }
}
