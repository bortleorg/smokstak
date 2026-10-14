//! Read-only evidence from a completed production build. Pixels are display
//! evidence, never another quality estimator or a source of rejection decisions.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sr_core::{frame::RawFrame, geometry::WarpField};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewFrame {
    pub path: PathBuf,
    pub filter: String,
    pub status: String,
    pub reason: String,
    #[serde(default)]
    pub capture_time: Option<String>,
    #[serde(default)]
    pub exposure_seconds: Option<f32>,
    /// Header ISO or camera gain setting, not electrons per ADU.
    #[serde(default)]
    pub iso_or_gain: Option<f32>,
    #[serde(default)]
    pub photometric_gain: Option<Vec<f32>>,
    #[serde(default)]
    pub photometry_source: Option<String>,
    /// Production mask in full-reference coordinates; absent in older audits
    /// and on frames excluded before photometry. Not a raw-thumbnail mask.
    #[serde(default)]
    pub obstruction_mask: Option<Vec<Vec<bool>>>,
    pub weight: Option<f32>,
    pub hfd: Option<f32>,
    pub eccentricity: Option<f32>,
    pub residual: Option<f32>,
    pub suppressed: Option<f32>,
    pub preview_asset: Option<String>,
    pub preview_note: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Patch {
    width: usize,
    height: usize,
    center: [f32; 2],
    pixels: Vec<Option<f32>>,
}

fn value(frame: &RawFrame, x: i64, y: i64) -> Option<f32> {
    if x < 0 || y < 0 || x >= frame.width as i64 || y >= frame.height as i64 {
        return None;
    }
    let value = frame.value(x as usize, y as usize);
    value.is_finite().then_some(value)
}

fn bilinear(frame: &RawFrame, x: f32, y: f32) -> Option<f32> {
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    // Preserve exact pixel values at integer centres, including the last row.
    if fx == 0.0 && fy == 0.0 {
        return value(frame, ix, iy);
    }
    let a = value(frame, ix, iy)?;
    let b = value(frame, ix + 1, iy)?;
    let c = value(frame, ix, iy + 1)?;
    let d = value(frame, ix + 1, iy + 1)?;
    Some((a + (b - a) * fx) * (1.0 - fy) + (c + (d - c) * fx) * fy)
}

fn green_phase(frame: &RawFrame) -> (i64, i64) {
    let i = frame
        .cfa
        .codes
        .iter()
        .position(|&c| c == sr_core::cfa::CfaColor::G)
        .unwrap_or(0);
    ((i % 2) as i64, (i / 2) as i64)
}

/// Resample a single green CFA phase. Never interpolate across Bayer colours.
fn inspection_sample(frame: &RawFrame, x: f32, y: f32) -> Option<f32> {
    if frame.is_mono() {
        return bilinear(frame, x, y);
    }
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    let (px, py) = green_phase(frame);
    let (gx, gy) = ((x - px as f32) / 2.0, (y - py as f32) / 2.0);
    let (ix, iy) = (gx.floor() as i64 * 2 + px, gy.floor() as i64 * 2 + py);
    let (fx, fy) = (gx - gx.floor(), gy - gy.floor());
    let a = value(frame, ix, iy)?;
    if fx == 0.0 && fy == 0.0 {
        return Some(a);
    }
    let b = value(frame, ix + 2, iy)?;
    let c = value(frame, ix, iy + 2)?;
    let d = value(frame, ix + 2, iy + 2)?;
    Some((a + (b - a) * fx) * (1.0 - fy) + (c + (d - c) * fx) * fy)
}

fn native(frame: &RawFrame, center: [f32; 2]) -> Patch {
    let size = 96;
    let origin = [center[0].round() as i64 - 48, center[1].round() as i64 - 48];
    Patch {
        width: size,
        height: size,
        center,
        pixels: (0..size * size)
            .map(|i| {
                value(
                    frame,
                    origin[0] + (i % size) as i64,
                    origin[1] + (i / size) as i64,
                )
            })
            .collect(),
    }
}

fn aligned(frame: &RawFrame, warp: &WarpField, center: [f32; 2]) -> Patch {
    let size = 96;
    Patch {
        width: size,
        height: size,
        center,
        pixels: (0..size * size)
            .map(|i| {
                let rx = center[0].round() - 48.0 + (i % size) as f32;
                let ry = center[1].round() - 48.0 + (i / size) as f32;
                warp.inverse_map(rx, ry)
                    .and_then(|(x, y)| inspection_sample(frame, x, y))
            })
            .collect(),
    }
}

fn overview(frame: &RawFrame) -> Patch {
    let width = 64;
    let height =
        ((frame.height as f64 / frame.width as f64 * width as f64).round() as usize).clamp(1, 128);
    Patch {
        width,
        height,
        center: [frame.width as f32 / 2.0, frame.height as f32 / 2.0],
        pixels: (0..width * height)
            .map(|i| {
                let mut x = (((i % width) as f64 + 0.5) * frame.width as f64 / width as f64) as i64;
                let mut y =
                    (((i / width) as f64 + 0.5) * frame.height as f64 / height as f64) as i64;
                if !frame.is_mono() {
                    let (px, py) = green_phase(frame);
                    x = x / 2 * 2 + px;
                    y = y / 2 * 2 + py;
                }
                value(frame, x, y)
            })
            .collect(),
    }
}

fn sky_overview(frame: &RawFrame, warp: &WarpField) -> Patch {
    let width = 128;
    let height =
        ((frame.height as f64 / frame.width as f64 * width as f64).round() as usize).clamp(1, 256);
    Patch {
        width,
        height,
        center: [frame.width as f32 / 2.0, frame.height as f32 / 2.0],
        pixels: (0..width * height)
            .map(|i| {
                let rx = (i % width) as f32 / (width - 1) as f32 * (frame.width - 1) as f32;
                let ry =
                    (i / width) as f32 / (height.max(2) - 1) as f32 * (frame.height - 1) as f32;
                warp.inverse_map(rx, ry)
                    .and_then(|(x, y)| inspection_sample(frame, x, y))
            })
            .collect(),
    }
}

/// Prefer a detected star near the field/ROI centre, so shape can be inspected.
/// If no suitable star is present, retain the centre and report its coordinates.
pub fn center(frame: &RawFrame, roi: Option<(usize, usize, usize, usize)>) -> [f32; 2] {
    let (x, y, w, h) = roi.unwrap_or((0, 0, frame.width, frame.height));
    let middle = [x as f32 + w as f32 / 2.0, y as f32 + h as f32 / 2.0];
    sr_quality::stars::positions(frame, 512)
        .into_iter()
        .filter(|s| {
            s.x >= x as f32 + 48.0
                && s.y >= y as f32 + 48.0
                && s.x + 48.0 < (x + w) as f32
                && s.y + 48.0 < (y + h) as f32
        })
        .min_by(|a, b| {
            ((a.x - middle[0]).powi(2) + (a.y - middle[1]).powi(2))
                .total_cmp(&((b.x - middle[0]).powi(2) + (b.y - middle[1]).powi(2)))
        })
        .map_or(middle, |s| [s.x, s.y])
}

/// Read one already-decoded frame, write small lazy assets, retain no full raster.
pub fn capture(
    dir: &Path,
    row: &mut ReviewFrame,
    frame: &RawFrame,
    warp: Option<&WarpField>,
    center: [f32; 2],
) -> Result<()> {
    row.capture_time = frame.metadata.capture_time.clone();
    row.exposure_seconds = frame
        .metadata
        .exposure_time
        .filter(|v| v.is_finite() && *v > 0.0);
    row.iso_or_gain = frame.metadata.iso.filter(|v| v.is_finite());
    let id = format!(
        "{:x}",
        Sha256::digest(row.path.to_string_lossy().as_bytes())
    );
    let assets = dir.join("review-assets");
    std::fs::create_dir_all(&assets)?;
    let sensor_center = warp
        .and_then(|w| w.inverse_map(center[0], center[1]))
        .map(|(x, y)| [x, y])
        .unwrap_or([frame.width as f32 / 2.0, frame.height as f32 / 2.0]);
    let images = serde_json::json!({"thumb":overview(frame), "native":native(frame,sensor_center),
        "aligned":warp.map(|w| aligned(frame,w,center)),
        "sky":warp.map(|w| sky_overview(frame,w))});
    let script = format!(
        "globalThis.smokstakReviewPatch(\"{id}\",{});",
        serde_json::to_string(&images)?
    );
    std::fs::write(assets.join(format!("{id}.js")), script)?;
    row.preview_asset = Some(format!("review-assets/{id}.js"));
    row.preview_note = if warp.is_some() {
        format!(
            "Common reference centre ({:.1}, {:.1}) sensor px. Registered crop: bilinear resampling at 1× sensor pitch. Native crop: original sensor pixels near that location, original orientation. Raw normalized detector levels; no photometric correction or defect masking. Overview is sparsely sampled.",
            center[0], center[1]
        )
    } else {
        "No reliable alignment used: native crop is the sensor centre, not a verified common sky area. Raw normalized detector levels; overview is sparsely sampled.".into()
    };
    if !frame.is_mono() {
        row.preview_note.push_str(" OSC: native crop shows the original Bayer mosaic in grayscale, not demosaiced RGB. Registered crop and overview use one green CFA phase only; green samples are two sensor pixels apart, resampled to sensor pitch in the registered view. Use the RGB master or original astro software for colour assessment.");
    }
    Ok(())
}

pub fn write(dir: &Path, frames: &[ReviewFrame]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let data = serde_json::json!({"schema":1,"frames":frames,"notes":[
        "Read-only evidence from this build. Changing a displayed selection does not change any stack.",
        "Used means eligible with a positive frame factor; coverage and pixel rejection may still prevent contribution at particular locations. A guide suppression fraction is not a percentage of exposures discarded.",
        "Frame factor is registration confidence × bounded sharpness, before noise, local, pixel and kernel weights. It is not effective integration time.",
        "Native crops preserve detector samples. Registered crops are resampled for location comparison and should not be used to measure noise. Previews do not establish that excluding a frame improved the master."
    ]});
    let json = serde_json::to_string(&data)?;
    std::fs::write(dir.join("frame-review.json"), &json)?;
    // Frame paths/reasons are untrusted text, never executable markup.
    let safe = json
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    let html = include_str!("frame_review.html")
        .replace("__REVIEW_JS__", include_str!("frame_review.js"))
        .replace("__REVIEW_JSON__", &safe);
    std::fs::write(dir.join("frame-review.html"), html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> RawFrame {
        RawFrame {
            width: 128,
            height: 128,
            samples: sr_core::samples::SamplePlane::from_normalised(
                128,
                128,
                (0..128 * 128).map(|i| i as f32 / 20000.0 - 0.1).collect(),
            ),
            cfa: sr_core::cfa::CfaPattern::MONO,
            defects: sr_core::samples::DefectMask::none(128, 128),
            noise: sr_core::frame::NoiseModel::nominal(100.0, 65535.0),
            metadata: Default::default(),
        }
    }
    #[test]
    fn sky_overview_keeps_the_reference_orientation_after_a_meridian_flip() {
        let f = frame();
        let mut warp = WarpField::identity();
        warp.global = sr_core::geometry::GlobalTransform {
            m: [-1.0, 0.0, 127.0, 0.0, -1.0, 127.0],
        };
        let sky = sky_overview(&f, &warp);
        assert_eq!(sky.pixels[0], Some(f.value(127, 127)));
        assert_eq!(*sky.pixels.last().unwrap(), Some(f.value(0, 0)));
        warp.global = sr_core::geometry::GlobalTransform::translation(3.0, 0.0);
        assert_eq!(sky_overview(&f, &warp).pixels[0], None);
    }

    #[test]
    fn native_and_identity_crop_keep_exact_detector_values_and_missing_edges() {
        let f = frame();
        let crop = native(&f, [64.0, 64.0]);
        assert_eq!(
            crop.pixels,
            aligned(&f, &WarpField::identity(), [64.0, 64.0]).pixels
        );
        assert_eq!(crop.pixels[0], Some(f.value(16, 16)));
        assert!(native(&f, [0.0, 0.0]).pixels[0].is_none());
        assert_eq!(bilinear(&f, 0.0, 0.0), Some(-0.1));
        assert!(bilinear(&f, -1.0, 5.0).is_none());
    }
    #[test]
    fn registered_crop_uses_inverse_warp_and_marks_missing_coverage() {
        let f = frame();
        let mut warp = WarpField::identity();
        warp.global = sr_core::geometry::GlobalTransform::translation(3.0, -2.0);
        let crop = aligned(&f, &warp, [64.0, 64.0]);
        assert_eq!(crop.pixels[0], Some(f.value(13, 18)));
    }
    #[test]
    fn osc_previews_never_interpolate_different_cfa_colours() {
        for cfa in [
            sr_core::cfa::CfaPattern::RGGB,
            sr_core::cfa::CfaPattern::BGGR,
            sr_core::cfa::CfaPattern::GRBG,
            sr_core::cfa::CfaPattern::GBRG,
        ] {
            let mut f = frame();
            f.cfa = cfa;
            f.samples = sr_core::samples::SamplePlane::from_normalised(
                128,
                128,
                (0..128 * 128)
                    .map(|i| match cfa.color_at(i % 128, i / 128) {
                        sr_core::cfa::CfaColor::G => 0.25,
                        _ => 100.0,
                    })
                    .collect(),
            );
            assert_eq!(inspection_sample(&f, 63.3, 65.7), Some(0.25));
            assert!(overview(&f).pixels.into_iter().flatten().all(|v| v == 0.25));
            assert!(
                aligned(&f, &WarpField::identity(), [64.0, 64.0])
                    .pixels
                    .into_iter()
                    .flatten()
                    .all(|v| v == 0.25)
            );
            assert!(native(&f, [64.0, 64.0]).pixels.contains(&Some(100.0)));
        }
    }
    #[test]
    fn report_keeps_reasons_as_data_not_html() {
        let dir = std::env::temp_dir().join(format!(
            "smokstak-review-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let row = ReviewFrame {
            path: PathBuf::from("original.fits"),
            filter: "H".into(),
            status: "excluded".into(),
            capture_time: None,
            exposure_seconds: None,
            iso_or_gain: None,
            photometric_gain: None,
            photometry_source: None,
            reason: "</script><img src=x onerror=alert(1)>".into(),
            weight: None,
            hfd: None,
            eccentricity: None,
            residual: None,
            suppressed: None,
            obstruction_mask: None,
            preview_asset: None,
            preview_note: "Source unavailable".into(),
        };
        write(&dir, &[row]).unwrap();
        let html = std::fs::read_to_string(dir.join("frame-review.html")).unwrap();
        assert!(!html.contains("</script><img"));
        assert!(html.contains("\\u003c/script\\u003e"));
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("frame-review.json")).unwrap()).unwrap();
        assert_eq!(
            json["frames"][0]["reason"],
            "</script><img src=x onerror=alert(1)>"
        );
        std::fs::remove_file(dir.join("frame-review.html")).unwrap();
        std::fs::remove_file(dir.join("frame-review.json")).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
