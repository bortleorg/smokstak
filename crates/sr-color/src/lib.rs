//! Stage 13: colour reconstruction.
//!
//! The merge produces linear camera-RGB on the high-resolution grid. This crate
//! turns that into a colorimetric result, and nothing else: no tone curve, no
//! saturation boost, no local contrast. A `linear` output is the reference
//! product; a `rendered` output adds only the sRGB transfer function so the
//! file is viewable.
//!
//! [`stretch`] is the exception, and is quarantined in its own module for it:
//! a deep-sky result is legible only after a transfer that destroys the linear
//! relationship deliberately. It is applied to a copy, written to a preview
//! file, and never to the output.

pub mod stretch;

use serde::{Deserialize, Serialize};
use sr_core::frame::FrameMetadata;
use sr_core::plane::Plane;

/// sRGB primaries expressed in XYZ (D65), rows = X, Y, Z.
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.119192, 0.9503041],
];

fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut o = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

fn mat3_inverse(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ColorTransform {
    /// White-balance multipliers applied to camera RGB, green normalised to 1.
    pub wb: [f32; 3],
    /// Camera RGB (white balanced) to linear sRGB.
    pub cam_to_srgb: [[f32; 3]; 3],
    /// True when the camera matrix was unusable and a pass-through was
    /// substituted; the colour of such an output is not colorimetric.
    pub fallback: bool,
}

impl ColorTransform {
    /// Derive the transform from a frame's metadata.
    ///
    /// The camera matrix is normalised so that a neutral camera signal maps to
    /// a neutral sRGB signal — the standard dcraw construction. Without it the
    /// matrix alone would introduce a colour cast.
    pub fn from_metadata(meta: &FrameMetadata) -> ColorTransform {
        let wb = {
            let g = if meta.wb_coeffs[1].abs() > 1e-6 { meta.wb_coeffs[1] } else { 1.0 };
            [meta.wb_coeffs[0] / g, 1.0, meta.wb_coeffs[2] / g]
        };
        let wb = [
            if wb[0].is_finite() && wb[0] > 0.0 { wb[0] } else { 1.0 },
            1.0,
            if wb[2].is_finite() && wb[2] > 0.0 { wb[2] } else { 1.0 },
        ];

        let xyz_to_cam = meta.xyz_to_cam;
        let usable = xyz_to_cam
            .iter()
            .flatten()
            .any(|v| v.is_finite() && v.abs() > 1e-6);

        if !usable {
            return ColorTransform {
                wb,
                cam_to_srgb: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                fallback: true,
            };
        }

        // sRGB -> camera, then row-normalise so that (1, 1, 1) stays neutral.
        let mut rgb_to_cam = mat3_mul(&xyz_to_cam, &SRGB_TO_XYZ);
        for row in rgb_to_cam.iter_mut() {
            let s = row[0] + row[1] + row[2];
            if s.abs() > 1e-9 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        }

        match mat3_inverse(&rgb_to_cam) {
            Some(m) => ColorTransform { wb, cam_to_srgb: m, fallback: false },
            None => ColorTransform {
                wb,
                cam_to_srgb: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                fallback: true,
            },
        }
    }

    /// White balance and matrix, in place. Values stay linear and unclamped so
    /// that a float output keeps its highlights.
    pub fn apply(&self, rgb: &mut [Plane<f32>; 3]) {
        // A monochrome sensor has no colour to convert. There is no meaningful
        // white balance for one channel and no matrix that maps it anywhere,
        // so the samples are left as measured.
        if rgb[1].data.is_empty() {
            return;
        }
        let n = rgb[0].data.len();
        let m = self.cam_to_srgb;
        let wb = self.wb;
        for i in 0..n {
            let r = rgb[0].data[i] * wb[0];
            let g = rgb[1].data[i] * wb[1];
            let b = rgb[2].data[i] * wb[2];
            rgb[0].data[i] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
            rgb[1].data[i] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
            rgb[2].data[i] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
        }
    }
}

/// The sRGB transfer function.
#[inline]
pub fn srgb_encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Undo [`srgb_encode`].
///
/// Needed because this program writes 16-bit results through the transfer
/// curve, and anything that reads one back to measure or combine it has to work
/// in linear light. Fitting one narrowband channel to another through a gamma
/// would fit the curve as much as the sky.
pub fn srgb_decode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// What writing an image as a display-referred integer file will do to it.
///
/// The linear product and the 16-bit one are two different claims. The linear
/// file is a measurement and is judged by every metric this program has. The
/// 16-bit file is a picture, and until this existed nothing looked at it at
/// all -- so it was possible to ship one whose sky sat at two thirds of the
/// range with its star cores clipped, and for every check to pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Encoding {
    /// Where the background lands after encoding, as a fraction of full scale.
    pub sky: f32,
    /// What is left above the sky for everything the picture contains.
    pub headroom: f32,
    /// Pixels per channel at or beyond full scale in the linear image, so lost
    /// to the integer file.
    pub clipped: [u64; 3],
    /// How much the background level varies across the frame, as a fraction
    /// of its own level.
    ///
    /// A gradient is not a defect -- vignetting and a real sky both make one,
    /// and the program deliberately leaves it alone because it cannot tell
    /// them apart. But it interacts with `sky`: with the background at two
    /// thirds of full scale, a few percent of variation in the data becomes a
    /// large and very visible swing in the encoded picture, so the two numbers
    /// have to be read together.
    pub sky_span: f32,
    /// Pixels where some channels clip and others do not.
    ///
    /// These are the expensive ones. A highlight that clips in all three stays
    /// white; one that clips in green alone comes out magenta, and a star core
    /// rendered magenta is the first thing anyone notices.
    pub uneven: u64,
    pub pixels: u64,
}

impl Encoding {
    /// The share of the frame that clips unevenly, which is the share that
    /// changes colour.
    pub fn uneven_fraction(&self) -> f32 {
        if self.pixels == 0 {
            0.0
        } else {
            self.uneven as f32 / self.pixels as f32
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "sky at {:.3} of full scale, {:.3} of range above it, varying {:.1}% \
             across the frame; clipped R/G/B {}/{}/{} pixels, {} of them in some \
             channels and not others",
            self.sky,
            self.headroom,
            self.sky_span * 100.0,
            self.clipped[0],
            self.clipped[1],
            self.clipped[2],
            self.uneven
        )
    }
}

/// Measure what [`to_rendered`] and a 16-bit write will make of this image.
pub fn encoding_of(rgb: &[Plane<f32>; 3]) -> Encoding {
    encoding_with(rgb, true)
}

/// The same for an image that is already display-referred, so the sky is read
/// as it stands rather than through the transfer curve a second time.
pub fn encoding_of_rendered(rgb: &[Plane<f32>]) -> Encoding {
    if rgb.len() < 3 {
        return Encoding::default();
    }
    encoding_with(&[rgb[0].clone(), rgb[1].clone(), rgb[2].clone()], false)
}

fn encoding_with(rgb: &[Plane<f32>; 3], encode: bool) -> Encoding {
    let n = rgb[0].data.len();
    if n == 0 || rgb[1].data.len() != n || rgb[2].data.len() != n {
        return Encoding::default();
    }
    let mut clipped = [0u64; 3];
    let mut uneven = 0u64;
    for i in 0..n {
        let mut hit = 0;
        for c in 0..3 {
            if rgb[c].data[i] >= 1.0 {
                clipped[c] += 1;
                hit += 1;
            }
        }
        if hit > 0 && hit < 3 {
            uneven += 1;
        }
    }
    // The sky as the encoded file will hold it, which is the number that says
    // whether the picture has anywhere to go.
    let stride = (n / 200_000).max(1);
    let mut lum: Vec<f32> = (0..n)
        .step_by(stride)
        .map(|i| {
            let l = 0.2126 * rgb[0].data[i] + 0.7152 * rgb[1].data[i] + 0.0722 * rgb[2].data[i];
            if encode { srgb_encode(l) } else { l.clamp(0.0, 1.0) }
        })
        .collect();
    lum.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sky = if lum.is_empty() { 0.0 } else { lum[lum.len() / 2] };
    Encoding {
        sky,
        headroom: (1.0 - sky).max(0.0),
        sky_span: sky_span(&rgb[1]),
        clipped,
        uneven,
        pixels: n as u64,
    }
}

/// Spread of the background between tiles, as a fraction of its own level.
///
/// Taken on a three by three so that a smooth gradient shows and pixel noise
/// does not, and from medians so that stars do not.
fn sky_span(g: &Plane<f32>) -> f32 {
    let (w, h) = (g.width, g.height);
    if w < 12 || h < 12 {
        return 0.0;
    }
    let (tw, th) = (w / 3, h / 3);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for ty in 0..3 {
        for tx in 0..3 {
            let mut v: Vec<f32> = Vec::new();
            let mut y = ty * th;
            while y < (ty + 1) * th {
                let mut x = tx * tw;
                while x < (tx + 1) * tw {
                    v.push(g.data[y * w + x]);
                    x += 3;
                }
                y += 3;
            }
            if v.is_empty() {
                continue;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = v[v.len() / 2];
            lo = lo.min(m);
            hi = hi.max(m);
        }
    }
    if !lo.is_finite() || lo <= 0.0 {
        0.0
    } else {
        hi / lo - 1.0
    }
}

/// Apply the sRGB transfer function to a copy of the image.
pub fn to_rendered(rgb: &[Plane<f32>; 3]) -> [Plane<f32>; 3] {
    let mut out = rgb.clone();
    for p in out.iter_mut() {
        for v in p.data.iter_mut() {
            *v = srgb_encode(*v);
        }
    }
    out
}

/// Scale so that a chosen highlight percentile lands at 1.0.
///
/// Reconstruction output is scene-linear and typically occupies only the lower
/// part of the range; without this a correct result looks like an underexposed
/// one. Returns the multiplier applied, so it can be recorded in the manifest
/// and the transformation reversed.
pub fn normalise_exposure(rgb: &mut [Plane<f32>; 3], percentile: f32, headroom: f32) -> f32 {
    let mut samples: Vec<f32> = Vec::new();
    let stride = (rgb[0].data.len() / 200_000).max(1);
    let mono = rgb[1].data.is_empty();
    let mut i = 0;
    while i < rgb[0].data.len() {
        let v = if mono {
            rgb[0].data[i]
        } else {
            rgb[0].data[i].max(rgb[1].data[i]).max(rgb[2].data[i])
        };
        if v.is_finite() {
            samples.push(v);
        }
        i += stride;
    }
    if samples.is_empty() {
        return 1.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f32 * percentile.clamp(0.0, 1.0)) as usize;
    let hi = samples[idx];
    if hi <= 1e-6 {
        return 1.0;
    }
    let gain = headroom / hi;
    for p in rgb.iter_mut() {
        for v in p.data.iter_mut() {
            *v *= gain;
        }
    }
    gain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_with(xyz_to_cam: [[f32; 3]; 3], wb: [f32; 3]) -> FrameMetadata {
        FrameMetadata { xyz_to_cam, wb_coeffs: wb, ..Default::default() }
    }

    #[test]
    fn neutral_camera_signal_stays_neutral() {
        // A plausible camera matrix.
        let m = [
            [0.7866, -0.1414, -0.0813],
            [-0.5867, 1.3573, 0.2373],
            [-0.0682, 0.1263, 0.7284],
        ];
        let t = ColorTransform::from_metadata(&meta_with(m, [2.0, 1.0, 1.5]));
        assert!(!t.fallback);
        // Feed a signal that is neutral *after* white balance.
        let mut rgb = [
            Plane::filled(4, 4, 0.5 / t.wb[0]),
            Plane::filled(4, 4, 0.5),
            Plane::filled(4, 4, 0.5 / t.wb[2]),
        ];
        t.apply(&mut rgb);
        let (r, g, b) = (rgb[0].data[0], rgb[1].data[0], rgb[2].data[0]);
        assert!((r - g).abs() < 1e-3 && (b - g).abs() < 1e-3, "not neutral: {r} {g} {b}");
        assert!((g - 0.5).abs() < 1e-3, "brightness changed: {g}");
    }

    #[test]
    fn unusable_matrix_falls_back_and_says_so() {
        let t = ColorTransform::from_metadata(&meta_with([[0.0; 3]; 3], [1.0, 1.0, 1.0]));
        assert!(t.fallback);
        let mut rgb = [
            Plane::filled(2, 2, 0.3),
            Plane::filled(2, 2, 0.4),
            Plane::filled(2, 2, 0.5),
        ];
        t.apply(&mut rgb);
        assert!((rgb[1].data[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn white_balance_is_green_normalised() {
        let t = ColorTransform::from_metadata(&meta_with([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [2.0, 1.0, 1.5]));
        assert!((t.wb[1] - 1.0).abs() < 1e-9);
        assert!((t.wb[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn srgb_transfer_hits_the_known_anchors() {
        assert!((srgb_encode(0.0)).abs() < 1e-9);
        assert!((srgb_encode(1.0) - 1.0).abs() < 1e-6);
        // Mid-grey: linear 0.2140 encodes to about 0.5.
        assert!((srgb_encode(0.2140) - 0.5).abs() < 2e-3);
    }

    #[test]
    fn exposure_normalisation_puts_the_percentile_at_the_target() {
        let mut rgb = [
            Plane::filled(64, 64, 0.05),
            Plane::filled(64, 64, 0.05),
            Plane::filled(64, 64, 0.05),
        ];
        let gain = normalise_exposure(&mut rgb, 0.999, 1.0);
        assert!((gain - 20.0).abs() < 0.5, "gain {gain}");
        assert!((rgb[1].data[0] - 1.0).abs() < 0.05);
    }
}

#[cfg(test)]
mod encoding_tests {
    use super::*;

    fn image(w: usize, h: usize, v: [f32; 3]) -> [Plane<f32>; 3] {
        [
            Plane::filled(w, h, v[0]),
            Plane::filled(w, h, v[1]),
            Plane::filled(w, h, v[2]),
        ]
    }

    #[test]
    fn a_dark_sky_leaves_the_range_above_it() {
        let e = encoding_of(&image(64, 64, [0.10, 0.17, 0.12]));
        assert!(e.sky < 0.5, "sky at {} of full scale", e.sky);
        assert!(e.headroom > 0.5, "only {} of range above the sky", e.headroom);
        assert_eq!(e.clipped, [0, 0, 0]);
        assert_eq!(e.uneven, 0);
    }

    #[test]
    fn a_sky_that_has_eaten_the_range_says_so() {
        // What an exposure normalisation with too much gain produces: the
        // background alone occupies most of what the file can hold.
        let e = encoding_of(&image(64, 64, [0.22, 0.39, 0.27]));
        assert!(e.sky > 0.55, "sky at only {} of full scale", e.sky);
        assert!(e.headroom < 0.45, "{} of range left above the sky", e.headroom);
    }

    #[test]
    fn a_highlight_that_clips_in_one_channel_only_is_counted_separately() {
        let mut p = image(32, 32, [0.2, 0.35, 0.25]);
        // Green over full scale, red and blue below it: the magenta core.
        for i in 0..10 {
            p[1].data[i] = 1.4;
        }
        // And a highlight that clips in all three, which stays white.
        for plane in p.iter_mut() {
            for i in 100..104 {
                plane.data[i] = 2.0;
            }
        }
        let e = encoding_of(&p);
        assert_eq!(e.clipped, [4, 14, 4]);
        assert_eq!(e.uneven, 10, "the one-channel clips were not separated out");
        assert!(e.uneven_fraction() > 0.0);
    }

    #[test]
    fn an_empty_image_reports_nothing_rather_than_a_pass() {
        let e = encoding_of(&[
            Plane::filled(0, 0, 0.0f32),
            Plane::filled(0, 0, 0.0f32),
            Plane::filled(0, 0, 0.0f32),
        ]);
        assert_eq!(e, Encoding::default());
        assert_eq!(e.uneven_fraction(), 0.0);
    }

}

#[cfg(test)]
mod sky_span_tests {
    use super::*;

    #[test]
    fn a_flat_sky_spans_nothing() {
        let e = encoding_of(&[
            Plane::filled(90, 90, 0.10f32),
            Plane::filled(90, 90, 0.17f32),
            Plane::filled(90, 90, 0.12f32),
        ]);
        assert!(e.sky_span < 0.01, "a flat sky reported a span of {}", e.sky_span);
    }

    #[test]
    fn a_gradient_across_the_frame_is_reported() {
        let mut p = [
            Plane::filled(90, 90, 0.10f32),
            Plane::filled(90, 90, 0.17f32),
            Plane::filled(90, 90, 0.12f32),
        ];
        for y in 0..90 {
            for x in 0..90 {
                // Ten percent from one corner to the other.
                let t = (x + y) as f32 / 178.0;
                p[1].data[y * 90 + x] = 0.17 * (1.0 + 0.10 * t);
            }
        }
        let e = encoding_of(&p);
        assert!(
            e.sky_span > 0.05 && e.sky_span < 0.12,
            "a ten percent gradient came back as {}",
            e.sky_span
        );
    }

    #[test]
    fn stars_do_not_count_as_a_gradient() {
        let mut p = [
            Plane::filled(90, 90, 0.10f32),
            Plane::filled(90, 90, 0.17f32),
            Plane::filled(90, 90, 0.12f32),
        ];
        // A bright clump in one tile. A mean would move; a median does not.
        for y in 6..14 {
            for x in 6..14 {
                p[1].data[y * 90 + x] = 0.9;
            }
        }
        let e = encoding_of(&p);
        assert!(e.sky_span < 0.01, "stars were read as a gradient: {}", e.sky_span);
    }
}
