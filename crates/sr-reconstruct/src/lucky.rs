//! Stage 12: lucky-region frame selection.
//!
//! Over a long atmospheric path, no single frame is sharp everywhere and no
//! frame is soft everywhere. Ranking frames per region and merging only the
//! good ones is the core idea of planetary lucky imaging.
//!
//! The trap, for a *super-resolution* stacker, is that the sharpest ten frames
//! in a region may all sit at nearly the same sub-pixel phase. Dropping the
//! eleventh because it is marginally softer can throw away the only sample at a
//! phase nothing else covers, and cost more resolution than the sharpness gain
//! buys. So selection here is quality-ranked but diversity-constrained: each
//! phase bin gets a quota before raw ranking is allowed to fill the rest.

use sr_core::frame::LocalQualityMap;
use sr_core::geometry::WarpField;
use sr_core::math::median;
use sr_core::plane::Plane;

/// Per-frame, per-region contribution weights.
#[derive(Clone, Debug)]
pub struct LuckySelection {
    /// Region edge in reference *sensor* pixels.
    pub region: f32,
    pub grid_w: usize,
    pub grid_h: usize,
    pub frames: usize,
    /// Row-major `[frame][cell]`, values in `[0, 1]`.
    weights: Vec<f32>,
    /// Fraction of frames kept, for reporting.
    pub fraction: f32,
}

impl LuckySelection {
    /// Weight for a sample landing at reference sensor coordinate `(rx, ry)`.
    ///
    /// Bilinear across regions: a hard per-region mask would print the region
    /// grid into the output wherever the selected set changes.
    #[inline]
    pub fn weight(&self, frame: usize, rx: f32, ry: f32) -> f32 {
        if self.grid_w == 0 || self.grid_h == 0 {
            return 1.0;
        }
        let gx = (rx / self.region - 0.5).clamp(0.0, (self.grid_w - 1) as f32);
        let gy = (ry / self.region - 0.5).clamp(0.0, (self.grid_h - 1) as f32);
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(self.grid_w - 1);
        let y1 = (y0 + 1).min(self.grid_h - 1);
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let base = frame * self.grid_w * self.grid_h;
        let a = self.weights[base + y0 * self.grid_w + x0] * (1.0 - fx)
            + self.weights[base + y0 * self.grid_w + x1] * fx;
        let b = self.weights[base + y1 * self.grid_w + x0] * (1.0 - fx)
            + self.weights[base + y1 * self.grid_w + x1] * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Number of frames kept per region, as a diagnostic plane.
    pub fn contributor_map(&self) -> Plane<f32> {
        let n = self.grid_w * self.grid_h;
        let mut out = Plane::<f32>::new(self.grid_w, self.grid_h);
        for f in 0..self.frames {
            for i in 0..n {
                out.data[i] += self.weights[f * n + i];
            }
        }
        out
    }
}

/// How much the burst's local sharpness actually varies between frames.
///
/// Returns the median across regions of `p90 / p50` of the per-frame sharpness.
/// A rigid tripod burst of a still scene sits near 1.0; a burst through heat
/// shimmer is visibly above it.
pub fn seeing_variability(maps: &[LocalQualityMap]) -> f32 {
    if maps.len() < 4 {
        return 1.0;
    }
    let cells = maps[0].sharpness.len();
    let mut ratios = Vec::with_capacity(cells);
    for c in 0..cells {
        let mut v: Vec<f32> = maps.iter().map(|m| m.sharpness[c]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = v[v.len() / 2];
        let p90 = v[((v.len() - 1) as f32 * 0.9) as usize];
        if p50 > 1e-12 {
            ratios.push(p90 / p50);
        }
    }
    if ratios.is_empty() {
        1.0
    } else {
        median(&ratios)
    }
}

/// Build a selection.
///
/// `maps` are per-frame local quality maps computed on each frame's *own*
/// guide, so each region centre is mapped back through that frame's warp before
/// its quality is read.
pub fn build(
    maps: &[LocalQualityMap],
    warps: &[WarpField],
    sensor_w: usize,
    sensor_h: usize,
    scale: f32,
    fraction: f32,
) -> LuckySelection {
    let n = maps.len();
    assert_eq!(n, warps.len());
    // Quality regions are measured in guide pixels; the reference grid is in
    // sensor pixels, which are half the pitch.
    let region_sensor = maps[0].region as f32 * 2.0;
    let grid_w = ((sensor_w as f32 / region_sensor).ceil() as usize).max(1);
    let grid_h = ((sensor_h as f32 / region_sensor).ceil() as usize).max(1);
    let cells = grid_w * grid_h;

    let keep = ((n as f32 * fraction).round() as usize).clamp(1, n);
    // Phase bins per axis: exactly the number of distinct positions the output
    // grid can distinguish.
    let bins = (scale.round() as usize).clamp(1, 4);
    let bin_quota = ((keep as f32 / (bins * bins) as f32).ceil() as usize).max(1);

    let mut weights = vec![0.0f32; n * cells];

    for cell in 0..cells {
        let gx = cell % grid_w;
        let gy = cell / grid_w;
        let cx = (gx as f32 + 0.5) * region_sensor;
        let cy = (gy as f32 + 0.5) * region_sensor;

        // Quality and phase of every frame at this region.
        let mut cand: Vec<(usize, f32, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            let Some((sx, sy)) = warps[i].inverse_map(cx, cy) else {
                continue;
            };
            let m = &maps[i];
            let qx = (sx * 0.5 / m.region as f32) as i64;
            let qy = (sy * 0.5 / m.region as f32) as i64;
            if qx < 0 || qy < 0 || qx >= m.grid_w as i64 || qy >= m.grid_h as i64 {
                continue;
            }
            let q = m.sharpness[qy as usize * m.grid_w + qx as usize];

            // Phase must come from where this frame's *sensor sites* land, not
            // from the region centre: mapping the centre back and forward again
            // returns the centre, which would give every frame the same phase.
            let (rx, ry) = warps[i].map(sx.round(), sy.round());
            let ox = (rx + 0.5) * scale - 0.5;
            let oy = (ry + 0.5) * scale - 0.5;
            let bx = (((ox - ox.floor()) * bins as f32) as usize).min(bins - 1);
            let by = (((oy - oy.floor()) * bins as f32) as usize).min(bins - 1);
            cand.push((i, q, by * bins + bx));
        }
        if cand.is_empty() {
            for i in 0..n {
                weights[i * cells + cell] = 1.0;
            }
            continue;
        }
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // First pass: fill each phase bin's quota in quality order, so no bin is
        // starved by a run of sharp frames that happen to share a phase.
        let mut used = vec![0usize; bins * bins];
        let mut accepted = 0usize;
        let mut chosen = vec![false; n];
        for &(i, _, b) in &cand {
            if accepted >= keep {
                break;
            }
            if used[b] < bin_quota {
                used[b] += 1;
                chosen[i] = true;
                accepted += 1;
            }
        }
        // Second pass: any remaining slots go to the best frames left.
        for &(i, _, _) in &cand {
            if accepted >= keep {
                break;
            }
            if !chosen[i] {
                chosen[i] = true;
                accepted += 1;
            }
        }
        for i in 0..n {
            weights[i * cells + cell] = if chosen[i] { 1.0 } else { 0.0 };
        }
    }

    LuckySelection {
        region: region_sensor,
        grid_w,
        grid_h,
        frames: n,
        weights,
        fraction: keep as f32 / n as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::geometry::GlobalTransform;

    fn map_with(grid: usize, region: usize, values: &[f32]) -> LocalQualityMap {
        LocalQualityMap {
            grid_w: grid,
            grid_h: grid,
            region,
            sharpness: values.to_vec(),
        }
    }

    /// Four frames, uniform quality per frame, phases tiling the 2x grid.
    fn uniform_burst(quals: &[f32]) -> (Vec<LocalQualityMap>, Vec<WarpField>) {
        let grid = 4usize;
        let maps: Vec<LocalQualityMap> = quals
            .iter()
            .map(|&q| map_with(grid, 32, &vec![q; grid * grid]))
            .collect();
        let warps: Vec<WarpField> = (0..quals.len())
            .map(|i| {
                WarpField::global_only(GlobalTransform::translation(
                    (i % 2) as f32 * 0.25,
                    (i / 2) as f32 * 0.25,
                ))
            })
            .collect();
        (maps, warps)
    }

    #[test]
    fn keeps_the_sharpest_frames() {
        let (maps, warps) = uniform_burst(&[0.1, 0.9, 0.2, 0.8]);
        let sel = build(&maps, &warps, 256, 256, 2.0, 0.5);
        // Two of four kept.
        assert!((sel.fraction - 0.5).abs() < 1e-6);
        let total: f32 = (0..4).map(|f| sel.weight(f, 128.0, 128.0)).sum();
        assert!((total - 2.0).abs() < 1e-4, "kept {total} frames");
        assert!(sel.weight(1, 128.0, 128.0) > 0.9, "sharpest frame was dropped");
    }

    #[test]
    fn diversity_quota_keeps_a_softer_frame_that_covers_a_missing_phase() {
        // Frames 0 and 1 are sharpest but share phase (0, 0); frame 2 is softer
        // and is the only one at a different phase.
        let grid = 2usize;
        let maps: Vec<LocalQualityMap> = [0.90f32, 0.89, 0.40]
            .iter()
            .map(|&q| map_with(grid, 32, &vec![q; grid * grid]))
            .collect();
        let warps: Vec<WarpField> = vec![
            WarpField::global_only(GlobalTransform::translation(0.0, 0.0)),
            WarpField::global_only(GlobalTransform::translation(0.0, 0.0)),
            WarpField::global_only(GlobalTransform::translation(0.25, 0.25)),
        ];
        let sel = build(&maps, &warps, 128, 128, 2.0, 2.0 / 3.0);
        let kept: Vec<usize> = (0..3)
            .filter(|&f| sel.weight(f, 64.0, 64.0) > 0.5)
            .collect();
        assert!(
            kept.contains(&2),
            "the only frame at its phase was dropped: kept {kept:?}"
        );
    }

    #[test]
    fn a_full_fraction_keeps_everything() {
        let (maps, warps) = uniform_burst(&[0.1, 0.9, 0.2, 0.8]);
        let sel = build(&maps, &warps, 256, 256, 2.0, 1.0);
        for f in 0..4 {
            assert!(sel.weight(f, 128.0, 128.0) > 0.99, "frame {f} was dropped");
        }
    }

    #[test]
    fn selection_varies_by_region() {
        // Frame 0 is sharp on the left, frame 1 sharp on the right.
        let grid = 2usize;
        let left = map_with(grid, 32, &[0.9, 0.1, 0.9, 0.1]);
        let right = map_with(grid, 32, &[0.1, 0.9, 0.1, 0.9]);
        let warps = vec![
            WarpField::global_only(GlobalTransform::IDENTITY),
            WarpField::global_only(GlobalTransform::IDENTITY),
        ];
        let sel = build(&[left, right], &warps, 128, 128, 2.0, 0.5);
        assert!(sel.weight(0, 20.0, 64.0) > sel.weight(1, 20.0, 64.0), "left region");
        assert!(sel.weight(1, 110.0, 64.0) > sel.weight(0, 110.0, 64.0), "right region");
    }

    #[test]
    fn seeing_variability_separates_steady_from_shimmering_bursts() {
        let steady: Vec<LocalQualityMap> =
            (0..8).map(|_| map_with(4, 32, &[1.0; 16])).collect();
        assert!(seeing_variability(&steady) < 1.05);

        let shimmer: Vec<LocalQualityMap> = (0..8)
            .map(|i| {
                let v: Vec<f32> = (0..16)
                    .map(|c| if (c + i) % 3 == 0 { 2.0 } else { 0.5 })
                    .collect();
                map_with(4, 32, &v)
            })
            .collect();
        assert!(seeing_variability(&shimmer) > 1.5, "{}", seeing_variability(&shimmer));
    }
}
