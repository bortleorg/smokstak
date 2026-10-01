//! Registration proxies and patch extraction.
//!
//! The registration proxy is *not* reconstruction data. It exists only to
//! estimate geometry cheaply and is allowed to be blurred, decimated and
//! normalised in ways that would be unacceptable for the merge.

use sr_core::geometry::GlobalTransform;
use sr_core::plane::Plane;

/// A Gaussian-ish pyramid over one frame's registration proxy.
#[derive(Clone, Debug)]
pub struct RegistrationImage {
    /// Level 0 is the full proxy resolution; each level halves it.
    pub levels: Vec<Plane<f32>>,
}

impl RegistrationImage {
    pub fn build(base: &Plane<f32>, levels: usize) -> Self {
        let mut out = Vec::with_capacity(levels);
        out.push(base.clone());
        for _ in 1..levels {
            let prev = out.last().unwrap();
            if prev.width < 32 || prev.height < 32 {
                break;
            }
            // Blur before decimating, or aliasing shows up as a false shift.
            out.push(prev.blur3().downsample2());
        }
        Self { levels: out }
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    pub fn level(&self, l: usize) -> &Plane<f32> {
        &self.levels[l.min(self.levels.len() - 1)]
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.levels[0].width, self.levels[0].height)
    }
}

/// Copy an axis-aligned `n x n` patch centred on `(cx, cy)`, edge-clamped.
pub fn extract_patch(img: &Plane<f32>, cx: f32, cy: f32, n: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), n * n);
    let half = n as f32 * 0.5;
    let x0 = (cx - half).round() as i64;
    let y0 = (cy - half).round() as i64;
    for j in 0..n {
        for i in 0..n {
            out[j * n + i] = img.at_clamped(x0 + i as i64, y0 + j as i64);
        }
    }
}

/// Copy an `n x n` patch from `img` at the positions that a transform maps the
/// reference window onto.
///
/// `inv` maps reference coordinates to target coordinates, i.e. it is the
/// inverse of the target-to-reference transform. Returns `false` if too much of
/// the patch falls outside the target frame to be trusted.
pub fn extract_patch_warped(
    img: &Plane<f32>,
    inv: &GlobalTransform,
    cx: f32,
    cy: f32,
    n: usize,
    out: &mut [f32],
) -> bool {
    debug_assert_eq!(out.len(), n * n);
    let half = n as f32 * 0.5;
    let x0 = (cx - half).round();
    let y0 = (cy - half).round();
    let mut outside = 0usize;
    let wf = (img.width - 1) as f32;
    let hf = (img.height - 1) as f32;
    for j in 0..n {
        for i in 0..n {
            let (sx, sy) = inv.apply(x0 + i as f32, y0 + j as f32);
            if sx < 0.0 || sy < 0.0 || sx > wf || sy > hf {
                outside += 1;
            }
            out[j * n + i] = img.bilinear(sx.clamp(0.0, wf), sy.clamp(0.0, hf));
        }
    }
    // Clamped border pixels are constant, which the correlator would read as a
    // strong edge feature at a fixed position.
    outside * 20 < n * n
}

/// Probe centres for a level, laid out on a regular grid inset by half a patch.
pub fn probe_grid(width: usize, height: usize, patch: usize, max_across: usize) -> Vec<(f32, f32)> {
    let half = patch / 2;
    if width < patch + 2 || height < patch + 2 {
        return vec![(width as f32 * 0.5, height as f32 * 0.5)];
    }
    let usable_w = width - patch;
    let usable_h = height - patch;
    // Aim for roughly `max_across` probes along the longer axis, but never
    // space them more than one patch apart or the fit loses spatial support.
    let long = usable_w.max(usable_h) as f32;
    let step = (long / max_across.max(1) as f32).max(patch as f32 * 0.5);
    let nx = ((usable_w as f32 / step).floor() as usize).max(1);
    let ny = ((usable_h as f32 / step).floor() as usize).max(1);
    let mut out = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            let x = half as f32 + usable_w as f32 * (i as f32 / nx as f32);
            let y = half as f32 + usable_h as f32 * (j as f32 / ny as f32);
            out.push((x, y));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_halves_each_level() {
        let p = Plane::<f32>::new(512, 256);
        let py = RegistrationImage::build(&p, 4);
        assert_eq!(py.level(0).dims(), (512, 256));
        assert_eq!(py.level(1).dims(), (256, 128));
        assert_eq!(py.level(2).dims(), (128, 64));
    }

    #[test]
    fn pyramid_stops_before_it_gets_useless() {
        let p = Plane::<f32>::new(64, 64);
        let py = RegistrationImage::build(&p, 8);
        assert!(py.depth() <= 3, "depth {}", py.depth());
    }

    #[test]
    fn warped_extraction_follows_the_transform() {
        let mut img = Plane::<f32>::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.data[y * 64 + x] = x as f32;
            }
        }
        // Reference-to-target map that shifts by +4 in x.
        let inv = GlobalTransform::translation(4.0, 0.0);
        let mut buf = vec![0.0f32; 16 * 16];
        assert!(extract_patch_warped(&img, &inv, 32.0, 32.0, 16, &mut buf));
        // Patch origin is 32 - 8 = 24; sampled at 28.
        assert!((buf[0] - 28.0).abs() < 1e-4, "{}", buf[0]);
    }

    #[test]
    fn mostly_out_of_frame_patches_are_refused() {
        let img = Plane::<f32>::new(64, 64);
        let inv = GlobalTransform::translation(500.0, 0.0);
        let mut buf = vec![0.0f32; 16 * 16];
        assert!(!extract_patch_warped(&img, &inv, 32.0, 32.0, 16, &mut buf));
    }

    #[test]
    fn probe_grid_covers_the_frame() {
        let g = probe_grid(1024, 512, 128, 8);
        assert!(g.len() >= 8);
        assert!(g.iter().all(|&(x, y)| x >= 64.0 && y >= 64.0));
        assert!(g.iter().all(|&(x, y)| x <= 1024.0 - 64.0 && y <= 512.0 - 64.0));
    }
}
