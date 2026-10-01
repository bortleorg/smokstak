//! The central data structure: one decoded RAW frame, still mosaiced.
//!
//! A `RawFrame` holds linear sensor measurements, not an image. Registration
//! results live beside it as metadata, so no stage of the pipeline has to
//! produce a resampled RGB intermediate.

use serde::{Deserialize, Serialize};

use crate::cfa::{CfaColor, CfaPattern};
use crate::plane::Plane;
use crate::samples::{self, DefectMask, SamplePlane};

/// Heteroscedastic sensor noise: `var(x) = alpha * x + beta`, with `x` the
/// black-subtracted, white-normalised signal in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseModel {
    /// Signal-dependent (shot) term.
    pub alpha: f32,
    /// Signal-independent (read) term.
    pub beta: f32,
    /// How the model was obtained, for the run manifest.
    pub source: NoiseSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseSource {
    /// Estimated from this burst's own statistics.
    Measured,
    /// Fallback derived from ISO and bit depth.
    Nominal,
    /// Supplied on the command line.
    Manual,
}

impl NoiseModel {
    pub fn new(alpha: f32, beta: f32, source: NoiseSource) -> Self {
        Self { alpha, beta, source }
    }

    /// Conservative default when nothing better is known.
    pub fn nominal(iso: f32, white_minus_black: f32) -> Self {
        // Full-well electrons scale roughly with sensor level range; treat the
        // signal as Poisson in raw levels and convert to normalised units.
        let gain = (iso / 100.0).max(1.0);
        let electrons_per_level = 1.0 / gain;
        let alpha = 1.0 / (white_minus_black.max(1.0) * electrons_per_level.max(1e-3));
        let read_levels = 2.0 * gain.sqrt();
        let beta = (read_levels / white_minus_black.max(1.0)).powi(2);
        Self { alpha, beta, source: NoiseSource::Nominal }
    }

    #[inline]
    pub fn variance(&self, x: f32) -> f32 {
        (self.alpha * x.max(0.0) + self.beta).max(1e-12)
    }

    #[inline]
    pub fn std_dev(&self, x: f32) -> f32 {
        self.variance(x).sqrt()
    }

    /// Inverse-variance weight for a measurement of value `x`.
    #[inline]
    pub fn weight(&self, x: f32) -> f32 {
        1.0 / self.variance(x)
    }
}

/// Everything we learned about a frame from its file, before touching pixels.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrameMetadata {
    pub path: String,
    pub file_name: String,
    pub make: String,
    pub model: String,
    pub clean_model: String,
    pub iso: Option<f32>,
    pub exposure_time: Option<f32>,
    /// Capture start as recorded by the astronomical header (normally ISO 8601).
    #[serde(default)]
    pub capture_time: Option<String>,
    pub aperture: Option<f32>,
    pub focal_length: Option<f32>,
    /// Sensor pitch in micrometres, where the file records it. With the focal
    /// length this gives the plate scale, which is what turns a blur measured
    /// in pixels into one an operator can compare against the seeing.
    pub pixel_pitch_um: Option<f32>,
    /// Filter in the light path, where the file names one. Frames taken through
    /// different filters are different measurements of the sky and cannot be
    /// merged, however well they register.
    pub filter: Option<String>,
    pub orientation: u16,
    /// Camera white-balance multipliers, RGB order.
    pub wb_coeffs: [f32; 3],
    /// XYZ -> camera RGB matrix, 3x3 row-major.
    pub xyz_to_cam: [[f32; 3]; 3],
    /// Per-CFA-cell black level, raster order within the 2x2 cell.
    pub black_levels: [f32; 4],
    pub white_level: f32,
    /// Active area used, relative to the full decoded array.
    pub crop: (usize, usize, usize, usize),
    pub full_width: usize,
    pub full_height: usize,
    pub sha256_prefix: String,
    /// Where the capture program's plate solve says this frame was pointed.
    ///
    /// Registration seeds itself from this when both frames have one, which is
    /// what lets a burst survive a meridian flip. It is never the answer: see
    /// [`crate::wcs`].
    #[serde(default)]
    pub wcs: Option<crate::wcs::Wcs>,
}

/// Frame-level quality descriptors. Deliberately a vector, not a single score:
/// collapsing early hides why a frame was ranked where it was.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct FrameQuality {
    /// Tenengrad gradient energy, normalised against the burst median.
    pub sharpness: f32,
    /// Multi-scale Laplacian response.
    pub laplacian: f32,
    /// RMS local contrast.
    pub contrast: f32,
    pub saturation_fraction: f32,
    /// Blur radius proxy in proxy pixels; smaller is sharper.
    pub estimated_blur: f32,
    /// Anisotropy of the gradient distribution; high values suggest directional
    /// motion blur rather than uniform softness.
    pub blur_anisotropy: f32,
    pub registration_confidence: f32,
    pub mean_level: f32,
}

impl FrameQuality {
    /// Single scalar used only for ordering candidate references and for
    /// display; the individual terms remain available.
    pub fn composite(&self) -> f32 {
        let sat_penalty = 1.0 - self.saturation_fraction.min(1.0);
        self.sharpness * sat_penalty * self.registration_confidence.max(0.05)
    }
}

/// Local (per-region) quality, used for lucky-region selection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalQualityMap {
    pub grid_w: usize,
    pub grid_h: usize,
    /// Region size in proxy pixels.
    pub region: usize,
    pub sharpness: Vec<f32>,
}

impl LocalQualityMap {
    pub fn plane(&self) -> Plane<f32> {
        Plane::from_vec(self.grid_w, self.grid_h, self.sharpness.clone())
    }
}

/// One decoded mosaiced frame.
///
/// Samples are held as decoded and normalised on read; see [`SamplePlane`].
/// Whether a sample is usable is likewise derived rather than stored, because
/// for real sensor data it is entirely a statement about the value: at or above
/// white is saturated, at or below black is clipped. Only defects that the
/// value cannot reveal — a dead site, a decode failure — need recording, and
/// those are rare enough to keep in a bitset that is usually absent.
#[derive(Clone, Debug)]
pub struct RawFrame {
    pub width: usize,
    pub height: usize,
    /// One measurement per sensor site.
    pub samples: SamplePlane,
    pub cfa: CfaPattern,
    /// Sites known bad for reasons the value does not show.
    pub defects: DefectMask,
    pub noise: NoiseModel,
    pub metadata: FrameMetadata,
}

impl RawFrame {
    #[inline]
    pub fn color_at(&self, x: usize, y: usize) -> CfaColor {
        self.cfa.color_at(x, y)
    }

    /// Whether this frame came from a sensor with no colour filter array.
    #[inline]
    pub fn is_mono(&self) -> bool {
        self.cfa.is_mono()
    }

    /// How many channels the reconstruction of this frame has: one or three.
    #[inline]
    pub fn channels(&self) -> usize {
        if self.is_mono() {
            1
        } else {
            3
        }
    }

    /// Which output channel a site contributes to.
    ///
    /// Not the same as [`color_at`](Self::color_at): a monochrome frame's sites
    /// are all labelled green so that the mosaic geometry still works, but they
    /// all go to channel zero, because there is only one.
    #[inline]
    pub fn channel_at(&self, x: usize, y: usize) -> usize {
        if self.is_mono() {
            0
        } else {
            self.cfa.color_at(x, y).index()
        }
    }

    /// Normalised linear value at a sensor site. Values outside `[0, 1]` are
    /// preserved rather than clamped, so that "dark" and "clipped" stay
    /// distinguishable.
    #[inline]
    pub fn value(&self, x: usize, y: usize) -> f32 {
        self.samples.value(x, y)
    }

    #[inline]
    pub fn value_at(&self, i: usize) -> f32 {
        self.samples.value_at(i)
    }

    /// Whether a measurement can be used, given its value.
    ///
    /// Takes the value the caller has already read, so that a hot loop is not
    /// forced to fetch it twice.
    #[inline]
    pub fn usable_value(&self, i: usize, value: f32) -> bool {
        value > 0.0 && value < 1.0 && !self.defects.get(i)
    }

    #[inline]
    pub fn usable(&self, x: usize, y: usize) -> bool {
        let i = y * self.width + x;
        self.usable_value(i, self.samples.value_in_cell(i, samples::Levels::cell(x, y)))
    }

    /// Bytes this frame occupies.
    pub fn bytes(&self) -> usize {
        self.samples.bytes() + self.defects.bytes()
    }

    pub fn saturation_fraction(&self) -> f32 {
        let mut n = 0usize;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.samples.value(x, y) >= 1.0 {
                    n += 1;
                }
            }
        }
        n as f32 / (self.width * self.height).max(1) as f32
    }

    /// Half-resolution per-cell RGB "guide" image: one RGB triple per 2x2 CFA
    /// cell, greens averaged. Cheap, alias-free enough for registration,
    /// robustness testing and quality metrics — and it is never used as
    /// reconstruction data.
    pub fn guide_rgb(&self) -> GuideImage {
        self.make_guide(false)
    }

    /// Geometry-only mono proxy. Isolated sensor spikes must not supply a
    /// stationary pattern to phase correlation before the temporal defect
    /// mask can be estimated. Samples and reconstruction guides are unchanged.
    pub fn registration_luma(&self) -> Plane<f32> {
        if !self.is_mono() {
            return self.guide_rgb().luma();
        }
        let mut out = Plane::new(self.width / 2, self.height / 2);
        for cy in 0..out.height {
            for cx in 0..out.width {
                let mut sum = 0.0;
                let mut count = 0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (x, y) = (2 * cx + dx, 2 * cy + dy);
                        let mut v = self.value(x, y);
                        if !self.usable_value(y * self.width + x, v) {
                            continue;
                        }
                        if x > 0 && y > 0 && x + 1 < self.width && y + 1 < self.height {
                            let mut neighbors = [0.0f32; 8];
                            let mut n = 0;
                            for yy in y - 1..=y + 1 {
                                for xx in x - 1..=x + 1 {
                                    if xx != x || yy != y {
                                        neighbors[n] = self.value(xx, yy);
                                        n += 1;
                                    }
                                }
                            }
                            // Preserve extended peaks and plateaus. Only a
                            // peak over 32 times its surrounding contrast
                            // is replaced, in this disposable proxy alone.
                            if neighbors.iter().all(|q| q.is_finite() && *q < v) {
                                neighbors.sort_unstable_by(f32::total_cmp);
                                let median = (neighbors[3] + neighbors[4]) * 0.5;
                                if v - median > 32.0 * (neighbors[7] - median) {
                                    v = median;
                                }
                            }
                        }
                        sum += v;
                        count += 1;
                    }
                }
                let mean = if count > 0 { sum / count as f32 } else { 0.0 };
                // Match GuideImage::luma's arithmetic when no spike changes
                // the cell, so ordinary proxies remain bitwise identical.
                out[(cx, cy)] = 0.25 * mean + 0.5 * mean + 0.25 * mean;
            }
        }
        out
    }

    /// Structural guide after defect masking. Clipped plateaus retain their
    /// endpoint values rather than turning into black holes. Registration
    /// keeps the conservative guide above, since defects are not known yet.
    pub fn structure_guide_rgb(&self) -> GuideImage {
        self.make_guide(true)
    }

    fn make_guide(&self, keep_clipped: bool) -> GuideImage {
        let gw = self.width / 2;
        let gh = self.height / 2;
        let mut r = Plane::new(gw, gh);
        let mut g = Plane::new(gw, gh);
        let mut b = Plane::new(gw, gh);
        if self.is_mono() {
            // The same value in all three planes rather than one plane and two
            // of zeros. Every consumer of a guide — the luma it is registered
            // on, the quality metrics, the robustness comparison — then behaves
            // identically without knowing this frame has one channel, and the
            // luma comes out as the cell mean rather than half of it.
            for cy in 0..gh {
                for cx in 0..gw {
                    let mut acc = 0.0f32;
                    let mut cnt = 0.0f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (x, y) = (2 * cx + dx, 2 * cy + dy);
                            let i = y * self.width + x;
                            let v = self.samples.value_in_cell(i, samples::Levels::cell(x, y));
                            if self.usable_value(i, v)
                                || (keep_clipped && v.is_finite() && !self.defects.get(i)) {
                                acc += v.clamp(0.0, 1.0);
                                cnt += 1.0;
                            }
                        }
                    }
                    let m = if cnt > 0.0 { acc / cnt } else { 0.0 };
                    let i = cy * gw + cx;
                    r.data[i] = m;
                    g.data[i] = m;
                    b.data[i] = m;
                }
            }
            return GuideImage { width: gw, height: gh, r, g, b };
        }
        for cy in 0..gh {
            for cx in 0..gw {
                let mut acc = [0.0f32; 3];
                let mut cnt = [0.0f32; 3];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let x = 2 * cx + dx;
                        let y = 2 * cy + dy;
                        let i = y * self.width + x;
                        let c = self.cfa.color_at(x, y).index();
                        let v = self.samples.value_in_cell(i, samples::Levels::cell(x, y));
                        if self.usable_value(i, v)
                            || (keep_clipped && v.is_finite() && !self.defects.get(i)) {
                            acc[c] += v.clamp(0.0, 1.0);
                            cnt[c] += 1.0;
                        }
                    }
                }
                let i = cy * gw + cx;
                r.data[i] = if cnt[0] > 0.0 { acc[0] / cnt[0] } else { 0.0 };
                g.data[i] = if cnt[1] > 0.0 { acc[1] / cnt[1] } else { 0.0 };
                b.data[i] = if cnt[2] > 0.0 { acc[2] / cnt[2] } else { 0.0 };
            }
        }
        GuideImage { width: gw, height: gh, r, g, b }
    }
}

/// Half-resolution RGB proxy derived from a mosaiced frame.
#[derive(Clone, Debug)]
pub struct GuideImage {
    pub width: usize,
    pub height: usize,
    pub r: Plane<f32>,
    pub g: Plane<f32>,
    pub b: Plane<f32>,
}

impl GuideImage {
    /// Luminance-ish plane for registration. Green dominates because Bayer
    /// sensors sample it twice as densely and terrestrial detail lives there.
    pub fn luma(&self) -> Plane<f32> {
        let mut out = Plane::new(self.width, self.height);
        for i in 0..out.data.len() {
            out.data[i] = 0.25 * self.r.data[i] + 0.5 * self.g.data[i] + 0.25 * self.b.data[i];
        }
        out
    }

    #[inline]
    pub fn channel(&self, c: usize) -> &Plane<f32> {
        match c {
            0 => &self.r,
            1 => &self.g,
            _ => &self.b,
        }
    }
}

/// A validated set of frames plus the shared decisions made about them.
#[derive(Debug)]
pub struct Burst {
    pub frames: Vec<RawFrame>,
    pub reference: usize,
    pub reference_reason: String,
}

impl Burst {
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
    pub fn reference_frame(&self) -> &RawFrame {
        &self.frames[self.reference]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_ignores_isolated_mono_spikes_without_changing_samples() {
        let mut values = vec![0.01; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                let r2 = (x as f32 - 32.0).powi(2) + (y as f32 - 32.0).powi(2);
                values[y * 64 + x] += 0.3 * (-r2 / 4.5).exp();
            }
        }
        let mut frame = RawFrame {
            width: 64, height: 64, cfa: CfaPattern::MONO,
            samples: SamplePlane::from_normalised(64, 64, values.clone()),
            defects: DefectMask::none(64, 64),
            noise: NoiseModel::nominal(100.0, 65535.0),
            metadata: Default::default(),
        };
        let clean = frame.registration_luma();
        assert_eq!(clean.data, frame.guide_rgb().luma().data,
                   "a resolved Gaussian star should retain its geometry");
        values[10 * 64 + 10] = 0.9;
        frame.samples = SamplePlane::from_normalised(64, 64, values);
        assert_eq!(frame.registration_luma().data, clean.data);
        assert_eq!(frame.value(10, 10), 0.9, "science samples must remain original");
        assert!(frame.structure_guide_rgb().luma()[(5, 5)] > 0.2);
        frame.cfa = CfaPattern::RGGB;
        assert_eq!(frame.registration_luma().data, frame.guide_rgb().luma().data,
                   "CFA registration must preserve its existing proxy");
    }

    #[test]
    fn registration_preserves_narrow_stars_at_different_pixel_phases() {
        for sigma in [0.55f32, 0.75, 1.0] {
            for phase in [0.0f32, 0.25, 0.5] {
                let mut values = vec![0.01; 32 * 32];
                for y in 0..32 {
                    for x in 0..32 {
                        let r2 = (x as f32-16.0-phase).powi(2)+(y as f32-16.0-phase).powi(2);
                        values[y*32+x] += 0.3*(-r2/(2.0*sigma*sigma)).exp();
                    }
                }
                let frame = RawFrame { width:32, height:32, cfa:CfaPattern::MONO,
                    samples:SamplePlane::from_normalised(32,32,values),
                    defects:DefectMask::none(32,32), noise:NoiseModel::nominal(100.0,65535.0),
                    metadata:Default::default() };
                assert_eq!(frame.registration_luma().data, frame.guide_rgb().luma().data,
                           "sigma {sigma}, pixel phase {phase}");
            }
        }
    }

    #[test]
    fn a_saturated_guide_stays_bright_for_bayer_and_mono() {
        for cfa in [CfaPattern::RGGB, CfaPattern::MONO] {
            let frame = RawFrame {
                width: 4, height: 4, cfa,
                samples: SamplePlane::from_normalised(4, 4, vec![1.0; 16]),
                defects: DefectMask::none(4, 4),
                noise: NoiseModel::nominal(100.0, 65535.0),
                metadata: Default::default(),
            };
            assert!(!frame.usable(0, 0), "clipped data must still be excluded from fits");
            let guide = frame.structure_guide_rgb();
            for c in 0..3 {
                assert!(guide.channel(c).data.iter().all(|v| *v == 1.0),
                    "saturated channel {c} became dark in the structural guide");
            }
        }
    }
}
