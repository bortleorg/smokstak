//! Image output.
//!
//! 16-bit integer TIFF for the deliverable, 32-bit float TIFF for anything that
//! will be measured. Diagnostic rasters get their own helper that records the
//! value range it normalised by, because a diagnostic you cannot put a number
//! on is decoration.

pub mod scientific;
pub use scientific::write_scientific_copies;

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use tiff::encoder::{colortype, TiffEncoder};

use sr_core::plane::Plane;

/// Write linear or encoded RGB as a 16-bit TIFF. Values are clamped to `[0, 1]`.
/// Write a preview as an 8-bit PNG, box-averaged down to fit `max_edge`.
///
/// PNG rather than TIFF because the point of a preview is that it opens
/// anywhere without thinking about it, and eight bits because it is for looking
/// at rather than measuring. Downsampled because a 2x reconstruction of a 26 MP
/// sensor is a hundred megapixels, and a preview that needs its own viewer is
/// not a preview.
///
/// The planes are taken as display-referred: whatever transfer they need has
/// already been applied. Adding one here would wash out a stretched image,
/// which is the main thing this is used for.
pub fn write_preview_png(path: &Path, planes: &[Plane<f32>], max_edge: usize) -> Result<()> {
    anyhow::ensure!(!planes.is_empty(), "a preview needs at least one channel");
    let channels = if planes.len() >= 3 { 3 } else { 1 };
    let (w, h) = (planes[0].width, planes[0].height);
    anyhow::ensure!(w > 0 && h > 0, "cannot preview an empty image");

    // Box average rather than point sampling: a preview of a star field made by
    // dropping pixels loses most of the stars, which is precisely the content.
    let factor = ((w.max(h) + max_edge - 1) / max_edge.max(1)).max(1);
    let (ow, oh) = ((w / factor).max(1), (h / factor).max(1));
    let mut buf = vec![0u8; ow * oh * channels];
    for oy in 0..oh {
        for ox in 0..ow {
            for c in 0..channels {
                let p = &planes[c.min(planes.len() - 1)];
                let mut acc = 0.0f32;
                let mut n = 0.0f32;
                for sy in oy * factor..((oy + 1) * factor).min(h) {
                    for sx in ox * factor..((ox + 1) * factor).min(w) {
                        acc += p.data[sy * w + sx];
                        n += 1.0;
                    }
                }
                let v = if n > 0.0 { acc / n } else { 0.0 };
                buf[(oy * ow + ox) * channels + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }

    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), ow as u32, oh as u32);
    enc.set_color(if channels == 3 { png::ColorType::Rgb } else { png::ColorType::Grayscale });
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc
        .write_header()
        .with_context(|| format!("writing {}", path.display()))?;
    writer
        .write_image_data(&buf)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Write a single-channel 16-bit image.
pub fn write_gray16(path: &Path, plane: &Plane<f32>) -> Result<()> {
    let mut buf = vec![0u16; plane.width * plane.height];
    for (o, &v) in buf.iter_mut().zip(&plane.data) {
        *o = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
    }
    let file = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut enc = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    enc.write_image::<tiff::encoder::colortype::Gray16>(
        plane.width as u32,
        plane.height as u32,
        &buf,
    )?;
    Ok(())
}

pub fn write_rgb16(path: &Path, rgb: &[Plane<f32>; 3]) -> Result<()> {
    let (w, h) = (rgb[0].width, rgb[0].height);
    anyhow::ensure!(w > 0 && h > 0, "refusing to write an empty image");
    let mut interleaved = vec![0u16; w * h * 3];
    for i in 0..w * h {
        for c in 0..3 {
            let v = rgb[c].data[i];
            let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
            interleaved[i * 3 + c] = (v * 65535.0).round() as u16;
        }
    }
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = TiffEncoder::new(BufWriter::new(file))?;
    enc.write_image::<colortype::RGB16>(w as u32, h as u32, &interleaved)?;
    Ok(())
}

/// Write RGB as a 32-bit float TIFF, unclamped.
///
/// This is the product to measure against: no clipping, no quantisation, so a
/// highlight that the merge reconstructed above 1.0 survives.
pub fn write_rgb32f(path: &Path, rgb: &[Plane<f32>; 3]) -> Result<()> {
    let (w, h) = (rgb[0].width, rgb[0].height);
    anyhow::ensure!(w > 0 && h > 0, "refusing to write an empty image");
    let mut interleaved = vec![0f32; w * h * 3];
    for i in 0..w * h {
        for c in 0..3 {
            let v = rgb[c].data[i];
            interleaved[i * 3 + c] = if v.is_finite() { v } else { 0.0 };
        }
    }
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = TiffEncoder::new(BufWriter::new(file))?;
    enc.write_image::<colortype::RGB32Float>(w as u32, h as u32, &interleaved)?;
    Ok(())
}

/// Write a single-channel 32-bit float TIFF.
pub fn write_gray32f(path: &Path, plane: &Plane<f32>) -> Result<()> {
    anyhow::ensure!(
        plane.width > 0 && plane.height > 0,
        "refusing to write an empty image"
    );
    let data: Vec<f32> = plane
        .data
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect();
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = TiffEncoder::new(BufWriter::new(file))?;
    enc.write_image::<colortype::Gray32Float>(plane.width as u32, plane.height as u32, &data)?;
    Ok(())
}

/// Write a reconstruction, in however many channels it has.
///
/// A monochrome sensor produces one plane and two empty ones, and writing
/// three channels from that reads past the end of an empty vector. Every
/// caller that has a product in hand should use this rather than choosing the
/// writer itself, because that choice has been got wrong twice.
pub fn write_product16(path: &Path, rgb: &[Plane<f32>; 3], channels: usize) -> Result<()> {
    if channels == 1 {
        write_gray16(path, &rgb[0])
    } else {
        write_rgb16(path, rgb)
    }
}

/// The same, unclamped and in 32-bit float.
pub fn write_product32f(path: &Path, rgb: &[Plane<f32>; 3], channels: usize) -> Result<()> {
    if channels == 1 {
        write_gray32f(path, &rgb[0])
    } else {
        write_rgb32f(path, rgb)
    }
}

/// How a diagnostic raster was scaled into `[0, 1]`.
#[derive(Clone, Copy, Debug)]
pub struct DiagnosticScale {
    pub lo: f32,
    pub hi: f32,
}

/// Write a diagnostic raster as 16-bit grey, normalised, returning the range
/// used so the caller can record it.
///
/// The float original is written alongside when `also_float` is set, because a
/// normalised preview is for looking at and the float is for measuring.
pub fn write_diagnostic(
    path: &Path,
    plane: &Plane<f32>,
    range: Option<(f32, f32)>,
    also_float: bool,
) -> Result<DiagnosticScale> {
    let (lo, hi) = match range {
        Some(r) => r,
        None => {
            // Percentile bounds, so one runaway pixel does not flatten the map.
            let lo = plane.percentile(0.001);
            let hi = plane.percentile(0.999);
            if hi > lo {
                (lo, hi)
            } else {
                plane.min_max()
            }
        }
    };
    let span = if (hi - lo).abs() > 1e-20 { hi - lo } else { 1.0 };
    let data: Vec<u16> = plane
        .data
        .iter()
        .map(|&v| {
            let t = if v.is_finite() { (v - lo) / span } else { 0.0 };
            (t.clamp(0.0, 1.0) * 65535.0).round() as u16
        })
        .collect();
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = TiffEncoder::new(BufWriter::new(file))?;
    enc.write_image::<colortype::Gray16>(plane.width as u32, plane.height as u32, &data)?;

    if also_float {
        let fp = path.with_extension("f32.tif");
        write_gray32f(&fp, plane)?;
    }
    Ok(DiagnosticScale { lo, hi })
}

/// Read a single-channel image, and report whether the file stored floating
/// point.
///
/// The distinction matters to a caller that has to decide whether the samples
/// are linear. This program writes integers through the sRGB transfer curve and
/// floats linear, so the storage type is the best available evidence about
/// which, and the caller is left to decide rather than being guessed at.
pub fn read_plane(path: &Path) -> Result<(Plane<f32>, bool)> {
    use tiff::decoder::DecodingResult;

    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading {}", path.display()))?
        .with_limits(tiff::decoder::Limits::unlimited());
    let (w, h) = dec.dimensions()?;
    let (w, h) = (w as usize, h as usize);
    let channels = match dec.colortype()? {
        tiff::ColorType::RGB(_) => 3usize,
        tiff::ColorType::RGBA(_) => 4,
        tiff::ColorType::Gray(_) => 1,
        other => anyhow::bail!("unsupported colour type {other:?} in {}", path.display()),
    };
    let (samples, floating): (Vec<f32>, bool) = match dec.read_image()? {
        DecodingResult::U8(v) => (v.iter().map(|&x| x as f32 / 255.0).collect(), false),
        DecodingResult::U16(v) => (v.iter().map(|&x| x as f32 / 65535.0).collect(), false),
        DecodingResult::F32(v) => (v, true),
        DecodingResult::F64(v) => (v.iter().map(|&x| x as f32).collect(), true),
        other => anyhow::bail!("unsupported sample format {other:?} in {}", path.display()),
    };
    anyhow::ensure!(
        samples.len() >= w * h * channels,
        "{}: got {} samples, expected {}",
        path.display(),
        samples.len(),
        w * h * channels
    );
    let mut out = Plane::<f32>::new(w, h);
    for i in 0..w * h {
        // A colour file collapses to its green channel, which is where the
        // luminance of anything this program writes lives.
        out.data[i] = samples[if channels == 1 { i } else { i * channels + 1.min(channels - 1) }];
    }
    Ok((out, floating))
}

/// Read an image back as linear planes.
///
/// Accepts the depths this crate writes, and normalises integer samples to
/// `[0, 1]` so a measurement does not depend on how the file was stored.
/// Measurement tools need this: a 16-bit result read as 8-bit silently loses
/// the precision the whole pipeline exists to preserve.
pub fn read_rgb(path: &Path) -> Result<[Plane<f32>; 3]> {
    use tiff::decoder::DecodingResult;

    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // The decoder's default buffer limit is sized for ordinary photographs and
    // refuses a full-resolution super-resolved frame, which is precisely the
    // output this crate exists to write.
    let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading {}", path.display()))?
        .with_limits(tiff::decoder::Limits::unlimited());
    let (w, h) = dec.dimensions()?;
    let (w, h) = (w as usize, h as usize);
    let colour = dec.colortype()?;
    let channels = match colour {
        tiff::ColorType::RGB(_) => 3usize,
        tiff::ColorType::RGBA(_) => 4,
        tiff::ColorType::Gray(_) => 1,
        other => anyhow::bail!("unsupported colour type {other:?} in {}", path.display()),
    };

    let samples: Vec<f32> = match dec.read_image()? {
        DecodingResult::U8(v) => v.iter().map(|&x| x as f32 / 255.0).collect(),
        DecodingResult::U16(v) => v.iter().map(|&x| x as f32 / 65535.0).collect(),
        DecodingResult::U32(v) => v.iter().map(|&x| x as f32 / u32::MAX as f32).collect(),
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.iter().map(|&x| x as f32).collect(),
        other => anyhow::bail!("unsupported sample format {other:?} in {}", path.display()),
    };
    anyhow::ensure!(
        samples.len() >= w * h * channels,
        "{}: got {} samples, expected {}",
        path.display(),
        samples.len(),
        w * h * channels
    );

    let mut out = [
        Plane::<f32>::new(w, h),
        Plane::<f32>::new(w, h),
        Plane::<f32>::new(w, h),
    ];
    for i in 0..w * h {
        for (c, p) in out.iter_mut().enumerate() {
            let src = if channels == 1 { i } else { i * channels + c.min(channels - 1) };
            p.data[i] = samples[src];
        }
    }
    Ok(out)
}

/// Luminance of an RGB image, for measurements that are about detail rather
/// than colour.
pub fn luma(rgb: &[Plane<f32>; 3]) -> Plane<f32> {
    let mut out = Plane::<f32>::new(rgb[0].width, rgb[0].height);
    for i in 0..out.data.len() {
        out.data[i] =
            0.2126 * rgb[0].data[i] + 0.7152 * rgb[1].data[i] + 0.0722 * rgb[2].data[i];
    }
    out
}

#[cfg(test)]
mod tests {
    fn read_png(path: &std::path::Path) -> ((usize, usize), Vec<u8>) {
        let dec = png::Decoder::new(std::fs::File::open(path).expect("open preview"));
        let mut reader = dec.read_info().expect("png header");
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("png data");
        buf.truncate(info.buffer_size());
        ((info.width as usize, info.height as usize), buf)
    }

    #[test]
    fn a_preview_is_box_averaged_down_to_fit() {
        // Point sampling a star field down by eight would drop most of the
        // stars, which are the content. A box average keeps them as dimmer
        // pixels, which is what a preview should show.
        let (w, h) = (800usize, 600usize);
        let mut p = Plane::<f32>::new(w, h);
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = if i % 97 == 0 { 1.0 } else { 0.0 };
        }
        let f = scratch_dir().join("preview.png");
        write_preview_png(&f, std::slice::from_ref(&p), 100).unwrap();

        let (dims, pixels) = read_png(&f);
        assert_eq!(dims, (100, 75));
        // The bright fraction survives the downsample rather than being
        // sampled away: one in 97 pixels lit, averaged, is a mean near 0.0103.
        let mean = pixels.iter().map(|&v| v as f32).sum::<f32>() / pixels.len() as f32 / 255.0;
        assert!((mean - 1.0 / 97.0).abs() < 0.006, "mean after downsampling {mean}");
        std::fs::remove_file(&f).ok();
    }

    #[test]
    fn a_small_image_is_previewed_at_its_own_size() {
        let p = Plane::<f32>::new(40, 30);
        let f = scratch_dir().join("preview-small.png");
        write_preview_png(&f, std::slice::from_ref(&p), 2000).unwrap();
        assert_eq!(read_png(&f).0, (40, 30));
        std::fs::remove_file(&f).ok();
    }

    use super::*;

    /// Scratch directory for test artefacts.
    ///
    /// Derived from the test binary's own location rather than the system
    /// temp directory, so the suite writes to the same volume as the build
    /// output. A full system drive should not be able to fail these tests.
    fn scratch_dir() -> std::path::PathBuf {
        let mut p = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(std::env::temp_dir);
        p.push("sr-test-scratch");
        std::fs::create_dir_all(&p).expect("create scratch dir");
        p
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = scratch_dir();
        p.push(format!("smokstak-test-{}-{}", std::process::id(), name));
        p
    }

    fn ramp(w: usize, h: usize) -> [Plane<f32>; 3] {
        let mut r = Plane::<f32>::new(w, h);
        let mut g = Plane::<f32>::new(w, h);
        let mut b = Plane::<f32>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                r.data[i] = x as f32 / w as f32;
                g.data[i] = y as f32 / h as f32;
                b.data[i] = 0.5;
            }
        }
        [r, g, b]
    }

    #[test]
    fn writes_a_readable_16_bit_rgb_tiff() {
        let p = tmp("rgb16.tif");
        write_rgb16(&p, &ramp(64, 32)).unwrap();
        let f = File::open(&p).unwrap();
        let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(f)).unwrap();
        assert_eq!(dec.dimensions().unwrap(), (64, 32));
        match dec.read_image().unwrap() {
            tiff::decoder::DecodingResult::U16(v) => {
                assert_eq!(v.len(), 64 * 32 * 3);
                // Blue is a constant 0.5 everywhere.
                assert!((v[2] as i32 - 32768).abs() < 2, "blue {}", v[2]);
            }
            other => panic!("unexpected sample format: {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn float_output_keeps_values_above_one() {
        let p = tmp("rgb32f.tif");
        let mut img = ramp(16, 16);
        img[0].data[0] = 3.5;
        write_rgb32f(&p, &img).unwrap();
        let f = File::open(&p).unwrap();
        let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(f)).unwrap();
        match dec.read_image().unwrap() {
            tiff::decoder::DecodingResult::F32(v) => {
                assert!((v[0] - 3.5).abs() < 1e-6, "highlight was clipped: {}", v[0]);
            }
            other => panic!("unexpected sample format: {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn sixteen_bit_output_clamps_and_reports_nothing_odd() {
        let p = tmp("clamp.tif");
        let mut img = ramp(8, 8);
        img[0].data[0] = 5.0;
        img[1].data[1] = -2.0;
        img[2].data[2] = f32::NAN;
        write_rgb16(&p, &img).unwrap();
        let f = File::open(&p).unwrap();
        let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(f)).unwrap();
        match dec.read_image().unwrap() {
            tiff::decoder::DecodingResult::U16(v) => {
                // Interleaved, so pixel `i` channel `c` is at `i * 3 + c`.
                let at = |i: usize, c: usize| v[i * 3 + c];
                assert_eq!(at(0, 0), 65535);
                assert_eq!(at(1, 1), 0);
                assert_eq!(at(2, 2), 0);
            }
            other => panic!("unexpected sample format: {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn sixteen_bit_output_round_trips_at_full_precision() {
        // The whole point of a 16-bit deliverable is that a measurement made on
        // it is not limited to 8-bit steps.
        let p = tmp("roundtrip.tif");
        let mut img = ramp(64, 64);
        img[1].data[0] = 0.30001;
        img[1].data[1] = 0.30004;
        write_rgb16(&p, &img).unwrap();
        let back = read_rgb(&p).unwrap();
        assert_eq!(back[0].dims(), (64, 64));
        assert!(
            (back[1].data[0] - 0.30001).abs() < 1e-4,
            "value {} lost precision",
            back[1].data[0]
        );
        assert!(
            back[1].data[0] != back[1].data[1],
            "two values a 16-bit step apart came back identical"
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn float_output_round_trips() {
        let p = tmp("roundtripf.tif");
        let mut img = ramp(32, 32);
        img[2].data[5] = 2.75;
        write_rgb32f(&p, &img).unwrap();
        let back = read_rgb(&p).unwrap();
        assert!((back[2].data[5] - 2.75).abs() < 1e-6);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn diagnostic_normalisation_reports_its_range() {
        let p = tmp("diag.tif");
        let mut plane = Plane::<f32>::new(32, 32);
        for (i, v) in plane.data.iter_mut().enumerate() {
            *v = 10.0 + i as f32 * 0.1;
        }
        let s = write_diagnostic(&p, &plane, None, false).unwrap();
        assert!(s.lo >= 10.0 && s.hi <= 10.0 + 1024.0 * 0.1);
        assert!(s.hi > s.lo);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn constant_diagnostic_does_not_divide_by_zero() {
        let p = tmp("const.tif");
        let plane = Plane::filled(8, 8, 4.0);
        let s = write_diagnostic(&p, &plane, None, false).unwrap();
        assert!(s.lo.is_finite() && s.hi.is_finite());
        std::fs::remove_file(&p).ok();
    }
}
