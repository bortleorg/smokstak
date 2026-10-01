//! Stage 1: frame ingestion.
//!
//! We do not write a Nikon decoder. We do not run a camera ISP either: no auto
//! brightness, no tone curve, no gamma, no demosaic. What comes out of here is
//! the linear mosaic, its per-site validity mask, and the metadata needed to
//! reason about it.
//!
//! Three sources feed that: camera raw by way of `rawler`, FITS, which has no
//! decoder to defer to and is handled here in [`fits`], and XISF in [`xisf`]. All three produce the same [`RawFrame`], so nothing downstream
//! knows which it is looking at.

pub mod fits;
mod validate;
pub mod window;
pub mod xisf;

pub use fits::RowOrder;
pub use validate::{BurstValidation, FieldReport, Severity, exposure_level, validate_burst};

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use rawler::RawImageData;
use rawler::rawimage::RawPhotometricInterpretation;
use rawler::rawsource::RawSource;
use sha2::{Digest, Sha256};

use sr_core::cfa::{CfaColor, CfaPattern};
use sr_core::frame::{FrameMetadata, NoiseModel, RawFrame};
use sr_core::plane::Plane;
use sr_core::samples::{DefectMask, Levels, SamplePlane};
use sr_core::{Result, SrError};

/// Extensions the FITS reader claims. `fts` is the old three-letter form and
/// `fit` is what Windows-era capture programs still write.
const FITS_EXTENSIONS: &[&str] = &["fits", "fit", "fts"];

/// Extensions the XISF reader claims. There is only the one, and the format
/// carries its own signature besides.
const XISF_EXTENSIONS: &[&str] = &["xisf"];

/// Camera raw extensions `rawler` decodes. Not exhaustive of what it supports,
/// but enough that a directory of frames from any of them is recognised without
/// the operator naming the format.
const RAW_EXTENSIONS: &[&str] = &[
    "nef", "nrw", "cr2", "cr3", "crw", "arw", "srf", "sr2", "dng", "raf", "orf", "rw2", "pef",
    "srw", "iiq", "3fr", "mos", "kdc", "dcr", "erf", "mrw", "x3f",
];

/// Which reader handles a path, decided by its extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Camera raw, decoded by `rawler`.
    CameraRaw,
    Fits,
    /// XISF, which is how a set that has been through another calibration
    /// pipeline often arrives.
    Xisf,
}

fn extension_of(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

/// The reader for a path, or `None` if the extension is not one we claim.
pub fn format_of(path: &Path) -> Option<Format> {
    let ext = extension_of(path);
    if FITS_EXTENSIONS.contains(&ext.as_str()) {
        Some(Format::Fits)
    } else if XISF_EXTENSIONS.contains(&ext.as_str()) {
        Some(Format::Xisf)
    } else if RAW_EXTENSIONS.contains(&ext.as_str()) {
        Some(Format::CameraRaw)
    } else {
        None
    }
}

/// How a frame should be read, for the decisions a file cannot settle alone.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReadOptions {
    /// Whether a FITS array's first row is the bottom or the top of the image.
    pub fits_row_order: fits::RowOrder,
}

/// Read only the header of a frame, for the questions that can be answered
/// without decoding it.
///
/// Used to sort a directory by filter before committing to reading gigabytes of
/// the wrong ones.
pub fn peek_filter(path: &Path) -> Option<String> {
    match format_of(path) {
        Some(Format::Fits) => {
            let (hdr, _) = fits::read_header(path).ok()?;
            hdr.any_text(&["FILTER", "FILTER1", "FILTNAME"])
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        }
        Some(Format::Xisf) => xisf::peek_filter(path),
        _ => None,
    }
}

/// Collect burst members from a directory or an explicit file list, sorted by
/// name so a rerun sees the same ordering.
///
/// With no `pattern`, every file whose extension names a format we read is
/// taken. A directory holding two such formats is an error rather than a
/// guess — a burst cannot span them, and silently picking one would hide the
/// other from a run that looked like it had succeeded.
pub fn collect_files(input: &Path, pattern: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if input.is_file() {
        // A file we can read is a burst of one. A file we cannot is taken to be
        // a list of the frames to use, which is the only way to reconstruct a
        // selection that does not correspond to a directory — the output of a
        // catalogue query, say, or frames that survived a review.
        if format_of(input).is_some() {
            out.push(input.to_path_buf());
            return Ok(out);
        }
        return read_file_list(input, pattern);
    }
    if !input.is_dir() {
        return Err(SrError::Input(format!(
            "{} is neither a file nor a directory",
            input.display()
        )));
    }

    let want = pattern.map(|p| {
        p.trim_start_matches('*')
            .trim_start_matches('.')
            .to_ascii_lowercase()
    });
    let mut kinds_found: std::collections::BTreeSet<String> = Default::default();
    for entry in std::fs::read_dir(input)? {
        let p = entry?.path();
        if !p.is_file() {
            continue;
        }
        let ext = extension_of(&p);
        match &want {
            Some(w) if !w.is_empty() => {
                if &ext == w {
                    out.push(p);
                }
            }
            _ => {
                if format_of(&p).is_some() {
                    kinds_found.insert(ext);
                    out.push(p);
                }
            }
        }
    }
    if kinds_found.len() > 1 {
        let kinds: Vec<&str> = kinds_found.iter().map(String::as_str).collect();
        return Err(SrError::Input(format!(
            "{} holds more than one format ({}); a burst cannot span them, so name the one \
             you want with --pattern",
            input.display(),
            kinds.join(", ")
        )));
    }
    out.sort();
    if out.is_empty() {
        return Err(match &want {
            Some(w) if !w.is_empty() => {
                SrError::Input(format!("no files matching *.{w} under {}", input.display()))
            }
            _ => SrError::Input(format!(
                "no camera raw, FITS or XISF files under {}; recognised extensions are {}, {}                  and {}",
                input.display(),
                RAW_EXTENSIONS.join(", "),
                FITS_EXTENSIONS.join(", "),
                XISF_EXTENSIONS.join(", ")
            )),
        });
    }
    Ok(out)
}

/// Frames named one per line, rather than found in a directory.
///
/// Blank lines and `#` comments are skipped, surrounding quotes and whitespace
/// are trimmed, and a relative path is resolved against the list's own
/// directory so a list can travel with the frames it names.
///
/// Every named file has to exist and be a format we read: a list is an explicit
/// statement about which frames to use, so silently dropping one would produce
/// a reconstruction of something other than what was asked for. The list itself
/// is only ever read.
fn read_file_list(list: &Path, pattern: Option<&str>) -> Result<Vec<PathBuf>> {
    let text = std::fs::read_to_string(list).map_err(|e| {
        SrError::Input(format!(
            "{} is not a format we read, and could not be read as a list of frames either: {e}",
            list.display()
        ))
    })?;
    let base = list.parent().unwrap_or(Path::new("."));
    let want = pattern.map(|p| {
        p.trim_start_matches('*')
            .trim_start_matches('.')
            .to_ascii_lowercase()
    });

    let mut out = Vec::new();
    let mut kinds_found: std::collections::BTreeSet<String> = Default::default();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim().trim_matches('"');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let raw = Path::new(line);
        // Against the list's own directory first, so a list can travel with
        // the frames it names; against the working directory second, because
        // `ls > frames.txt` from somewhere else is how most lists get made.
        let p = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            let beside = base.join(raw);
            if beside.is_file() {
                beside
            } else {
                raw.to_path_buf()
            }
        };
        let where_ = format!("{}, line {}", list.display(), n + 1);
        if !p.is_file() {
            return Err(SrError::Input(format!(
                "{}: {} is not a file",
                where_,
                p.display()
            )));
        }
        let ext = extension_of(&p);
        if let Some(w) = &want
            && !w.is_empty()
            && &ext != w
        {
            continue;
        }
        if format_of(&p).is_none() {
            return Err(SrError::Input(format!(
                "{}: {} is not a format we read; recognised extensions are {}, {} and {}",
                where_,
                p.display(),
                RAW_EXTENSIONS.join(", "),
                FITS_EXTENSIONS.join(", "),
                XISF_EXTENSIONS.join(", ")
            )));
        }
        kinds_found.insert(ext);
        out.push(p);
    }
    if kinds_found.len() > 1 {
        let kinds: Vec<&str> = kinds_found.iter().map(String::as_str).collect();
        return Err(SrError::Input(format!(
            "{} names more than one format ({}); a burst cannot span them, so name the one \
             you want with --pattern",
            list.display(),
            kinds.join(", ")
        )));
    }
    if out.is_empty() {
        return Err(SrError::Input(format!(
            "{} names no frames",
            list.display()
        )));
    }
    // Sorted like a directory listing, so the same set of frames produces the
    // same reference and the same result however the list was ordered.
    out.sort();
    out.dedup();
    log::info!("{}: {} frames listed", list.display(), out.len());
    Ok(out)
}

/// One file that is byte-for-byte another file already in the list.
pub struct Duplicate {
    pub dropped: PathBuf,
    pub same_as: PathBuf,
}

/// Remove files whose contents another file in the list already supplies.
///
/// A frame that appears twice is merged twice: it gets double weight, it
/// agrees perfectly with its own copy so outlier rejection sees a consensus
/// where there is one measurement, and the effective frame count -- the number
/// an operator reads as "how deep is this" -- counts it as two exposures when
/// only one was taken.
///
/// This is not hypothetical tidiness. A list of 41 frames of NGC 7000 handed
/// to this program held the same 20 exposures under two directories; nothing
/// noticed, and it changed whether the local warp was applied.
///
/// Identity is the same bounded head-and-tail digest the cache uses, which
/// includes the file's length. Two different exposures agreeing on all of that
/// is not a thing that happens; two copies of one file agreeing is certain.
/// The first occurrence in the list order is the one kept.
pub fn deduplicate(paths: Vec<PathBuf>) -> (Vec<PathBuf>, Vec<Duplicate>) {
    use rayon::prelude::*;
    use std::collections::HashMap;
    let digests: Vec<Option<String>> = paths.par_iter().map(|p| sha256_prefix(p).ok()).collect();

    let mut seen: HashMap<String, PathBuf> = HashMap::new();
    let mut kept = Vec::with_capacity(paths.len());
    let mut dropped = Vec::new();
    for (p, d) in paths.into_iter().zip(digests) {
        match d {
            // A file we could not read is left in the list: decoding will
            // report it far better than this can.
            None => kept.push(p),
            Some(d) => match seen.get(&d) {
                Some(first) => dropped.push(Duplicate {
                    dropped: p,
                    same_as: first.clone(),
                }),
                None => {
                    seen.insert(d, p.clone());
                    kept.push(p);
                }
            },
        }
    }
    (kept, dropped)
}

pub(crate) fn sha256_prefix(path: &Path) -> std::io::Result<String> {
    // Hash a bounded head+tail rather than 24 MB per file: enough to detect a
    // changed input for cache invalidation, cheap enough to run on 100 frames.
    let mut f = File::open(path)?;
    let len = f.metadata()?.len();
    let mut hasher = Sha256::new();
    hasher.update(len.to_le_bytes());
    let mut head = vec![0u8; (1 << 16).min(len as usize)];
    f.read_exact(&mut head)?;
    hasher.update(&head);
    if len > (1 << 17) {
        use std::io::Seek;
        f.seek(std::io::SeekFrom::End(-(1 << 16)))?;
        let mut tail = vec![0u8; 1 << 16];
        f.read_exact(&mut tail)?;
        hasher.update(&tail);
    }
    let digest = hasher.finalize();
    Ok(digest[..8].iter().map(|b| format!("{b:02x}")).collect())
}

/// XYZ-to-camera matrix for a frame, as 3x3 row-major.
///
/// `rawler` exposes this two ways. `color_matrix` is the current one, keyed by
/// illuminant, and is where modern camera entries actually carry their data;
/// `xyz_to_cam` is the older flat field and is empty for cameras whose entries
/// were written after it was deprecated — the Nikon Z 7II among them. Reading
/// only the old field yields a zero matrix, which silently costs colorimetric
/// output, so both are consulted.
///
/// Daylight illuminants are preferred because that is what the sRGB working
/// space assumes; any populated matrix beats none.
fn colour_matrix(img: &rawler::RawImage) -> [[f32; 3]; 3] {
    use rawler::imgop::xyz::Illuminant;

    let preference = [
        Illuminant::D65,
        Illuminant::Daylight,
        Illuminant::FineWeather,
        Illuminant::D50,
        Illuminant::D55,
        Illuminant::D75,
        Illuminant::CloudyWeather,
        Illuminant::Shade,
        Illuminant::Flash,
        Illuminant::A,
        Illuminant::Tungsten,
        Illuminant::Unknown,
    ];

    let flat = img
        .color_matrix_find_first(preference)
        .map(|(_, m)| m)
        .filter(|m| m.len() >= 9)
        .or_else(|| {
            // Any populated entry, whatever its illuminant.
            img.color_matrix
                .values()
                .find(|m| m.len() >= 9 && m.iter().any(|v| v.is_finite() && v.abs() > 1e-6))
                .cloned()
        });

    if let Some(m) = flat {
        let mut out = [[0.0f32; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = m[r * 3 + c];
            }
        }
        if out
            .iter()
            .flatten()
            .any(|v| v.is_finite() && v.abs() > 1e-6)
        {
            return out;
        }
    }

    // Fall back to the deprecated field, which arrives as 4 rows (RGBE).
    let mut out = [[0.0f32; 3]; 3];
    out.copy_from_slice(&img.xyz_to_cam[..3]);
    out
}

fn cfa_from_name(name: &str) -> Result<CfaPattern> {
    CfaPattern::from_name(name).ok_or_else(|| {
        SrError::Input(format!(
            "unsupported CFA pattern {name:?}; only 2x2 RGB Bayer mosaics are handled in v0"
        ))
    })
}

/// Decode one frame, choosing the reader from the file extension.
pub fn decode(path: &Path) -> Result<RawFrame> {
    decode_with(path, &ReadOptions::default())
}

/// Decode one frame with explicit reader options.
pub fn decode_with(path: &Path, opts: &ReadOptions) -> Result<RawFrame> {
    match format_of(path) {
        Some(Format::Fits) => fits::decode(path, opts.fits_row_order),
        Some(Format::Xisf) => xisf::decode(path),
        // An unrecognised extension is handed to `rawler` rather than refused:
        // it knows more formats than the list above names, and its own error is
        // more use than ours would be.
        _ => decode_camera_raw(path),
    }
}

/// Decode one camera RAW file into a normalised mosaic.
///
/// Normalisation is `(raw - black_c) / (white_c - black_c)` with the black
/// level taken per CFA cell position. Values outside `[0, 1]` are preserved
/// rather than clamped, and flagged in the mask, so downstream stages can tell
/// "dark" from "clipped".
fn decode_camera_raw(path: &Path) -> Result<RawFrame> {
    let src =
        RawSource::new(path).map_err(|e| SrError::Input(format!("{}: {e}", path.display())))?;
    let decoder = rawler::get_decoder(&src)
        .map_err(|e| SrError::Input(format!("{}: {e}", path.display())))?;
    let params = rawler::decoders::RawDecodeParams::default();
    let md = decoder
        .raw_metadata(&src, &params)
        .map_err(|e| SrError::Input(format!("{}: metadata: {e}", path.display())))?;
    let img = decoder
        .raw_image(&src, &params, false)
        .map_err(|e| SrError::Input(format!("{}: raw image: {e}", path.display())))?;

    if img.cpp != 1 {
        return Err(SrError::Input(format!(
            "{}: expected a mosaiced single-component image, got cpp={}",
            path.display(),
            img.cpp
        )));
    }
    if !matches!(img.photometric, RawPhotometricInterpretation::Cfa(_)) {
        return Err(SrError::Input(format!(
            "{}: file is not CFA mosaiced; linear-RGB RAW is out of scope for v0",
            path.display()
        )));
    }

    let full_w = img.width;
    let full_h = img.height;

    // Prefer the camera's recommended crop, fall back to the active area.
    let rect = img.crop_area.or(img.active_area);
    let (cx, cy, cw, ch) = match rect {
        Some(r) => (r.p.x, r.p.y, r.d.w, r.d.h),
        None => (0, 0, full_w, full_h),
    };
    // Keep the crop origin even so the 2x2 mosaic phase of the cropped array is
    // a clean shift of the sensor's, and so half-resolution guides tile exactly.
    let (cx, cy) = (cx & !1, cy & !1);
    let cw = cw & !1;
    let ch = ch & !1;

    let base_cfa = cfa_from_name(&img.camera.cfa.name)?;
    let cfa = base_cfa.shifted(cx, cy);

    let black = img.blacklevel.shift(cx, cy).as_bayer_array();
    let white = img.whitelevel.as_bayer_array();

    let data = match &img.data {
        RawImageData::Integer(d) => d,
        RawImageData::Float(_) => {
            return Err(SrError::Input(format!(
                "{}: floating point RAW is not handled in v0",
                path.display()
            )));
        }
    };
    if data.len() < full_w * full_h {
        return Err(SrError::Input(format!(
            "{}: decoded {} samples, expected {}",
            path.display(),
            data.len(),
            full_w * full_h
        )));
    }

    // Samples are kept as decoded and normalised on read: the values are 14-bit
    // integers and storing them as `f32` would double the burst's footprint to
    // hold no extra information. Saturation and black clipping are not recorded
    // either, because they are exactly "the normalised value is at or beyond the
    // ends of the range" and can be recomputed for free.
    let mut raw = vec![0u16; cw * ch];
    for y in 0..ch {
        let src_row = &data[(cy + y) * full_w + cx..(cy + y) * full_w + cx + cw];
        raw[y * cw..(y + 1) * cw].copy_from_slice(src_row);
    }
    let levels = Levels::new(black, white);
    let samples = SamplePlane::from_u16(cw, ch, raw, levels);
    let defects = DefectMask::none(cw, ch);

    let iso = md.exif.iso_speed_ratings.map(|v| v as f32);
    let black_mean = 0.25 * (black[0] + black[1] + black[2] + black[3]);
    let noise = NoiseModel::nominal(iso.unwrap_or(100.0), white[0] - black_mean);

    let xyz_to_cam = colour_matrix(&img);

    let wb = img.wb_coeffs;
    let wb_rgb = [
        if wb[0].is_finite() && wb[0] > 0.0 {
            wb[0]
        } else {
            1.0
        },
        if wb[1].is_finite() && wb[1] > 0.0 {
            wb[1]
        } else {
            1.0
        },
        if wb[2].is_finite() && wb[2] > 0.0 {
            wb[2]
        } else {
            1.0
        },
    ];

    let metadata = FrameMetadata {
        // Camera raw files carry no plate solve.
        wcs: None,
        path: path.display().to_string(),
        file_name: path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        make: img.make.trim().to_string(),
        model: img.model.trim().to_string(),
        clean_model: img.clean_model.trim().to_string(),
        iso,
        exposure_time: md
            .exif
            .exposure_time
            .map(|r| r.n as f32 / r.d.max(1) as f32),
        capture_time: None,
        aperture: md.exif.fnumber.map(|r| r.n as f32 / r.d.max(1) as f32),
        focal_length: md.exif.focal_length.map(|r| r.n as f32 / r.d.max(1) as f32),
        // Camera RAW does not carry the sensor pitch in any portable field, and
        // guessing it from a body name is not something this needs to do.
        pixel_pitch_um: None,
        filter: None,
        orientation: 0,
        wb_coeffs: wb_rgb,
        xyz_to_cam,
        black_levels: black,
        white_level: white[0],
        crop: (cx, cy, cw, ch),
        full_width: full_w,
        full_height: full_h,
        sha256_prefix: sha256_prefix(path).unwrap_or_default(),
    };

    Ok(RawFrame {
        width: cw,
        height: ch,
        samples,
        cfa,
        defects,
        noise,
        metadata,
    })
}

/// Decode a whole burst in parallel.
pub fn decode_all(paths: &[PathBuf], opts: &ReadOptions) -> Result<Vec<RawFrame>> {
    use rayon::prelude::*;
    let mut results: Vec<(usize, Result<RawFrame>)> = paths
        .par_iter()
        .enumerate()
        .map(|(i, p)| (i, decode_with(p, opts)))
        .collect();
    results.sort_by_key(|(i, _)| *i);
    let mut frames = Vec::with_capacity(results.len());
    for (_, r) in results {
        frames.push(r?);
    }
    Ok(frames)
}

/// Cheap bilinear demosaic. Used only for the RGB baseline backend and for
/// preview output — never as reconstruction input.
pub fn demosaic_bilinear(frame: &RawFrame) -> [Plane<f32>; 3] {
    let (w, h) = (frame.width, frame.height);
    let mut sum = [
        Plane::<f32>::new(w, h),
        Plane::<f32>::new(w, h),
        Plane::<f32>::new(w, h),
    ];
    let mut cnt = [
        Plane::<f32>::new(w, h),
        Plane::<f32>::new(w, h),
        Plane::<f32>::new(w, h),
    ];

    // Scatter each measured site into its own channel, then fill the gaps from
    // the neighbourhood of the same channel.
    for y in 0..h {
        for x in 0..w {
            let c = frame.cfa.color_at(x, y).index();
            let i = y * w + x;
            let v = frame.value(x, y);
            if frame.usable_value(i, v) {
                sum[c].data[i] = v;
                cnt[c].data[i] = 1.0;
            }
        }
    }
    let mut out = [
        Plane::<f32>::new(w, h),
        Plane::<f32>::new(w, h),
        Plane::<f32>::new(w, h),
    ];
    // A monochrome sensor's sites are all recorded as green, so only the middle
    // plane is filled. Everything downstream expects a monochrome image in the
    // shape a monochrome reconstruction has — channel 0, and nothing else — so
    // it is returned that way rather than as a green rectangle.
    let channels = if frame.is_mono() { 1 } else { 3 };
    for c in 0..3 {
        if channels == 1 && c != 1 {
            continue;
        }
        let radius: i64 = if c == 1 { 1 } else { 2 };
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if cnt[c].data[i] > 0.0 {
                    out[c].data[i] = sum[c].data[i];
                    continue;
                }
                let mut acc = 0.0f32;
                let mut wsum = 0.0f32;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x as i64 + dx;
                        let ny = y as i64 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if cnt[c].data[j] <= 0.0 {
                            continue;
                        }
                        let d2 = (dx * dx + dy * dy) as f32;
                        let wt = 1.0 / (1.0 + d2);
                        acc += sum[c].data[j] * wt;
                        wsum += wt;
                    }
                }
                out[c].data[i] = if wsum > 0.0 { acc / wsum } else { 0.0 };
            }
        }
    }
    if channels == 1 {
        out.swap(0, 1);
        out[1] = Plane::new(0, 0);
        out[2] = Plane::new(0, 0);
    }
    out
}

/// Fraction of sites carrying each CFA colour; a sanity check on pattern
/// detection.
pub fn cfa_histogram(frame: &RawFrame) -> [f32; 3] {
    let mut n = [0u64; 3];
    for y in 0..frame.height.min(64) {
        for x in 0..frame.width.min(64) {
            n[frame.cfa.color_at(x, y).index()] += 1;
        }
    }
    let total = (n[0] + n[1] + n[2]).max(1) as f32;
    [
        n[0] as f32 / total,
        n[1] as f32 / total,
        n[2] as f32 / total,
    ]
}

/// Colour of the CFA site, exposed for callers that only hold a pattern.
pub fn color_at(cfa: &CfaPattern, x: usize, y: usize) -> CfaColor {
    cfa.color_at(x, y)
}

#[cfg(test)]
mod dedup_tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push("sr-raw-dedup-tests");
        d.push(name);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn the_same_file_under_two_names_is_one_exposure() {
        // What a list built from two directories of the same session looks
        // like. Merging it twice would give one exposure the weight of two.
        let dir = scratch("copies");
        let a = write(&dir, "a.fit", b"frame one contents");
        let b = write(&dir, "b.fit", b"frame two contents");
        let copy = write(&dir, "a-copy.fit", b"frame one contents");

        let (kept, dropped) = deduplicate(vec![a.clone(), b.clone(), copy.clone()]);
        assert_eq!(kept, vec![a.clone(), b]);
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].dropped, copy);
        // And it says which frame the dropped one duplicates, so the report
        // can name both.
        assert_eq!(dropped[0].same_as, a);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frames_that_merely_resemble_each_other_are_both_kept() {
        // Two exposures of the same field differ in their pixels however
        // similar the header. Only identity counts.
        let dir = scratch("similar");
        let a = write(&dir, "a.fit", b"header........ pixels AAAA");
        let b = write(&dir, "b.fit", b"header........ pixels AAAB");
        let (kept, dropped) = deduplicate(vec![a, b]);
        assert_eq!(kept.len(), 2);
        assert!(dropped.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_first_listed_copy_is_the_one_kept() {
        let dir = scratch("order");
        let first = write(&dir, "1.fit", b"same");
        let second = write(&dir, "2.fit", b"same");
        let third = write(&dir, "3.fit", b"same");
        let (kept, dropped) = deduplicate(vec![first.clone(), second, third]);
        assert_eq!(kept, vec![first]);
        assert_eq!(dropped.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_that_cannot_be_read_is_left_for_the_decoder_to_report() {
        let dir = scratch("unreadable");
        let a = write(&dir, "a.fit", b"real");
        let missing = dir.join("gone.fit");
        let (kept, dropped) = deduplicate(vec![a, missing.clone()]);
        assert!(
            kept.contains(&missing),
            "an unreadable file was silently dropped"
        );
        assert!(dropped.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod list_tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push("sr-raw-list-tests");
        d.push(name);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Not a real FITS, but `collect_files` only ever looks at the extension.
    fn touch(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, b"not really a frame").unwrap();
        p
    }

    #[test]
    fn a_list_names_the_frames_to_use() {
        let dir = scratch("basic");
        let a = touch(&dir, "a.fit");
        let b = touch(&dir, "b.fit");
        let list = dir.join("frames.txt");
        std::fs::write(
            &list,
            format!("# a comment\n\n{}\n\"{}\"\n", a.display(), b.display()),
        )
        .unwrap();

        let got = collect_files(&list, None).unwrap();
        assert_eq!(got, vec![a.clone(), b.clone()]);

        // Relative paths are resolved against the list, so it can travel with
        // the frames it names.
        std::fs::write(&list, "b.fit\na.fit\n").unwrap();
        let got = collect_files(&list, None).unwrap();
        assert_eq!(got, vec![a, b], "a list is sorted like a directory listing");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_named_frame_that_is_missing_is_an_error() {
        // The whole point of a list is that it is explicit. Dropping a line
        // would reconstruct something other than what was asked for.
        let dir = scratch("missing");
        touch(&dir, "a.fit");
        let list = dir.join("frames.txt");
        std::fs::write(&list, "a.fit\nnot-here.fit\n").unwrap();
        let e = collect_files(&list, None).unwrap_err().to_string();
        assert!(e.contains("line 2"), "{e}");
        assert!(e.contains("not-here.fit"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_list_cannot_mix_formats_any_more_than_a_directory_can() {
        let dir = scratch("mixed");
        touch(&dir, "a.fit");
        touch(&dir, "b.nef");
        let list = dir.join("frames.txt");
        std::fs::write(&list, "a.fit\nb.nef\n").unwrap();
        let e = collect_files(&list, None).unwrap_err().to_string();
        assert!(e.contains("more than one format"), "{e}");

        // ...and --pattern picks one, as it does for a directory.
        assert_eq!(collect_files(&list, Some("fit")).unwrap().len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_frame_we_can_read_is_still_a_burst_of_one() {
        let dir = scratch("single");
        let a = touch(&dir, "a.fit");
        assert_eq!(collect_files(&a, None).unwrap(), vec![a]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_list_naming_nothing_says_so() {
        let dir = scratch("empty");
        let list = dir.join("frames.txt");
        std::fs::write(&list, "# every line a comment\n\n").unwrap();
        let e = collect_files(&list, None).unwrap_err().to_string();
        assert!(e.contains("names no frames"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
