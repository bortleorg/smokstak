//! FITS ingestion.
//!
//! FITS is not a camera raw format. There is no decoder to defer to and no
//! maker note to consult: the file is an ASCII header followed by a big-endian
//! array, and everything the pipeline needs has to be recovered from keywords
//! that capture programs write inconsistently or not at all.
//!
//! Three of those recoveries are load-bearing, and all three are done from the
//! pixels rather than from the header, because the header is often wrong:
//!
//! * **Which sites are green.** `BAYERPAT` describes the image as the capture
//!   program displays it, and whether that matches the array as stored depends
//!   on `ROWORDER`, which is frequently absent. Getting it wrong swaps red for
//!   blue. The two green sites of a mosaic cell see almost the same light, so
//!   the green diagonal is plain in the cell means of any real frame, and that
//!   is what decides it here.
//! * **The white level.** Astronomy cameras with 12- or 14-bit converters write
//!   16-bit files by shifting left, so the largest attainable value is not
//!   65535 and a star that saturated the converter never compares equal to it.
//!   The shift shows up as low bits that no sample in the frame ever sets.
//! * **The shot-noise coefficient**, when `EGAIN` is present: a gain in
//!   electrons per ADU plus the quantisation step is exactly what the
//!   heteroscedastic noise model wants.
//!
//! The black level is deliberately taken as zero. A deep-sky frame sits a few
//! percent above the camera's pedestal, and the pedestal is not recorded in any
//! portable way; subtracting an estimate of it would push sky background to or
//! below zero, where [`RawFrame::usable_value`] discards it. Leaving it in
//! costs nothing: the noise model's constant term absorbs it, and every other
//! stage works in differences.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use sr_core::cfa::{CfaColor, CfaPattern};
use sr_core::frame::{FrameMetadata, NoiseModel, NoiseSource, RawFrame};
use sr_core::samples::{DefectMask, Levels, SamplePlane};
use sr_core::wcs::Wcs;
use sr_core::{Result, SrError};

/// FITS blocks are fixed at 2880 bytes, which is 36 cards of 80 characters.
const BLOCK: usize = 2880;
const CARD: usize = 80;
/// A header longer than this is a malformed file, not a rich one.
const MAX_HEADER_BLOCKS: usize = 256;

/// Whether the first row of the stored array is the bottom or the top of the
/// image.
///
/// The FITS standard puts the origin at the bottom left, but capture programs
/// overwhelmingly write sensor readout order and only sometimes say so with
/// `ROWORDER`. This decides one thing only — whether the result comes out
/// mirrored top to bottom — because the mosaic is settled against the pixels
/// afterwards either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RowOrder {
    /// Believe `ROWORDER`, and without it take the array as it is stored.
    ///
    /// Not the standard's bottom-up, deliberately. Assuming a flip that was not
    /// there reinterprets the file silently; declining one that was there
    /// leaves a mirrored image the operator can see and correct. It also keeps
    /// FITS from being the one format here that does not hand back storage
    /// order.
    #[default]
    Auto,
    BottomUp,
    TopDown,
}

impl RowOrder {
    pub fn parse(s: &str) -> Option<RowOrder> {
        match s.trim().to_ascii_uppercase().replace('_', "-").as_str() {
            "AUTO" => Some(RowOrder::Auto),
            "BOTTOM-UP" | "BOTTOMUP" => Some(RowOrder::BottomUp),
            "TOP-DOWN" | "TOPDOWN" => Some(RowOrder::TopDown),
            _ => None,
        }
    }
}

/// Header keywords in file order, with the value text left unparsed.
#[derive(Clone, Debug, Default)]
pub struct Header {
    pub cards: Vec<(String, String)>,
}

impl Header {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.cards
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    /// The first of several keywords that different programs use for one thing.
    pub fn any(&self, keys: &[&str]) -> Option<&str> {
        keys.iter().find_map(|k| self.get(k))
    }

    pub fn number(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.trim().parse::<f64>().ok())
    }

    pub fn any_number(&self, keys: &[&str]) -> Option<f64> {
        keys.iter().find_map(|k| self.number(k))
    }

    pub fn int(&self, key: &str) -> Option<i64> {
        self.number(key).map(|v| v as i64)
    }

    /// A string value with its FITS quoting removed.
    pub fn text(&self, key: &str) -> Option<String> {
        self.get(key).map(unquote)
    }

    pub fn any_text(&self, keys: &[&str]) -> Option<String> {
        keys.iter().find_map(|k| self.text(k))
    }
}

fn unquote(v: &str) -> String {
    let t = v.trim();
    let Some(inner) = t.strip_prefix('\'') else {
        return t.to_string();
    };
    let inner = match inner.rfind('\'') {
        Some(i) => &inner[..i],
        None => inner,
    };
    // Doubled quotes are the FITS escape for a literal quote.
    inner.replace("''", "'").trim().to_string()
}

/// The plate solve, if the header carries a usable one.
///
/// Only the gnomonic projection is read, and only its linear part: `CTYPE`
/// has to say TAN (with or without SIP, whose coefficients are a sub-pixel
/// correction this does not need), and the matrix has to be there in one of
/// its two spellings. `CDELT` with `CROTA` is the older form and is converted;
/// anything else is treated as a frame with no solution, which is not an error
/// — most of the world's raw files have no solve at all.
pub(crate) fn plate_solve(hdr: &Header) -> Option<Wcs> {
    let ctype1 = hdr.text("CTYPE1").unwrap_or_default().to_ascii_uppercase();
    let ctype2 = hdr.text("CTYPE2").unwrap_or_default().to_ascii_uppercase();
    if !ctype1.contains("TAN") || !ctype2.contains("TAN") {
        return None;
    }
    // FITS counts pixels from one, and from the centre of the first pixel.
    let crpix = (hdr.number("CRPIX1")? - 1.0, hdr.number("CRPIX2")? - 1.0);
    let crval = (hdr.number("CRVAL1")?, hdr.number("CRVAL2")?);

    let cd = match (
        hdr.number("CD1_1"),
        hdr.number("CD1_2"),
        hdr.number("CD2_1"),
        hdr.number("CD2_2"),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => [[a, b], [c, d]],
        _ => {
            // The PC + CDELT spelling, and the older CDELT + CROTA2 one.
            let (d1, d2) = (hdr.number("CDELT1")?, hdr.number("CDELT2")?);
            match (
                hdr.number("PC1_1"),
                hdr.number("PC1_2"),
                hdr.number("PC2_1"),
                hdr.number("PC2_2"),
            ) {
                (Some(a), Some(b), Some(c), Some(d)) => [[a * d1, b * d1], [c * d2, d * d2]],
                _ => {
                    let rot = hdr.number("CROTA2").unwrap_or(0.0).to_radians();
                    let (s, c) = (rot.sin(), rot.cos());
                    [[d1 * c, -d2 * s], [d1 * s, d2 * c]]
                }
            }
        }
    };
    let w = Wcs { crpix, crval, cd };
    w.is_plausible().then_some(w)
}

/// Split one card into keyword and value text, discarding the comment.
///
/// Comment stripping has to respect quoting: a slash inside a quoted value is
/// part of the value, not the start of a comment.
fn parse_card(card: &str) -> Option<(String, String)> {
    let key = card.get(..8)?.trim();
    if key.is_empty() || key == "COMMENT" || key == "HISTORY" || key == "END" {
        return None;
    }
    if card.get(8..10)? != "= " {
        return None;
    }
    let rest = card.get(10..)?;
    let mut in_quotes = false;
    let mut end = rest.len();
    for (i, c) in rest.char_indices() {
        match c {
            '\'' => in_quotes = !in_quotes,
            '/' if !in_quotes => {
                end = i;
                break;
            }
            _ => {}
        }
    }
    Some((key.to_string(), rest[..end].trim().to_string()))
}

/// Read the primary header, returning it and the byte offset of the data unit.
pub fn read_header(path: &Path) -> Result<(Header, u64)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut header = Header::default();
    let mut buf = [0u8; BLOCK];
    for block in 0..MAX_HEADER_BLOCKS {
        r.read_exact(&mut buf).map_err(|e| {
            SrError::Input(format!("{}: truncated FITS header: {e}", path.display()))
        })?;
        for c in 0..BLOCK / CARD {
            let card = String::from_utf8_lossy(&buf[c * CARD..(c + 1) * CARD]);
            if card.starts_with("END") && card[3..].trim().is_empty() {
                return Ok((header, ((block + 1) * BLOCK) as u64));
            }
            if let Some(kv) = parse_card(&card) {
                header.cards.push(kv);
            }
        }
    }
    Err(SrError::Input(format!(
        "{}: no END card in the first {MAX_HEADER_BLOCKS} header blocks",
        path.display()
    )))
}

/// How far apart the attainable sample values are.
///
/// A camera whose converter is narrower than the file's 16 bits writes every
/// value shifted left, so the low bits are zero in every sample of the frame.
/// The step is then the factor between a file value and a converter code, and
/// it is found by asking which low bits no sample ever sets.
///
/// Deliberately not a greatest common divisor. A gcd of a synthetic ramp picks
/// up whatever the values happen to share — a frame of multiples of 28 is not a
/// camera that steps in 28s — whereas a left shift is by construction a power
/// of two, and asking only about low bits cannot answer anything else.
fn quantisation_step(data: &[u16]) -> u32 {
    let mut bits = 0u32;
    for &v in data {
        bits |= v as u32;
        // An odd sample settles it, and for an unshifted file that is the first
        // one or two. This is why the whole frame can be scanned.
        if bits & 1 != 0 {
            return 1;
        }
    }
    if bits == 0 {
        return 1;
    }
    // Lowest set bit. Capped because a 16-bit file shifted by more than four
    // places would be a 12-bit converter, and beyond that the frame is more
    // likely to be empty than the camera exotic.
    (bits & bits.wrapping_neg()).min(16)
}

/// Mean of each of the four positions in the 2x2 mosaic cell.
fn cell_means(data: &[u16], w: usize, h: usize) -> [f64; 4] {
    let mut sum = [0f64; 4];
    let mut n = [0u64; 4];
    // Include every mosaic cell. Sparse periodic sampling aliases sharp stars
    // into unequal green means and can silently change a correct Bayer label.
    for y in (0..h.saturating_sub(1)).step_by(2) {
        for x in (0..w.saturating_sub(1)).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    sum[dy * 2 + dx] += data[(y + dy) * w + x + dx] as f64;
                    n[dy * 2 + dx] += 1;
                }
            }
        }
    }
    let mut out = [0f64; 4];
    for i in 0..4 {
        out[i] = sum[i] / n[i].max(1) as f64;
    }
    out
}

/// Which diagonal of the mosaic cell holds the two green sites, judged from the
/// pixels.
///
/// `Some(true)` means the anti-diagonal, sites `(1,0)` and `(0,1)`, as in RGGB
/// and BGGR. `None` means the frame does not say clearly enough to overrule the
/// header.
fn green_is_antidiagonal(means: [f64; 4]) -> Option<bool> {
    let scale = (means.iter().sum::<f64>() / 4.0).max(1.0);
    let main = (means[0] - means[3]).abs() / scale;
    let anti = (means[1] - means[2]).abs() / scale;
    // Two greens of one cell differ by a fraction of a percent of the frame
    // mean; two different filters differ by far more. Requiring a factor of
    // four makes a genuinely grey scene abstain instead of guessing.
    if anti * 4.0 < main {
        Some(true)
    } else if main * 4.0 < anti {
        Some(false)
    } else {
        None
    }
}

/// Bring the header's pattern into agreement with the pixels.
///
/// Only a one-row shift is ever applied. A disagreement about which diagonal is
/// green means the array's row parity is not what the header assumed, which is
/// exactly what a bottom-up array of even height does to a pattern quoted for
/// the displayed image. A genuine red/blue swap would not move the greens and
/// is not detectable this way.
fn reconcile_pattern(cfa: CfaPattern, means: [f64; 4], file: &str) -> CfaPattern {
    let Some(observed) = green_is_antidiagonal(means) else {
        return cfa;
    };
    let stated = cfa.color_at(1, 0) == CfaColor::G && cfa.color_at(0, 1) == CfaColor::G;
    if observed == stated {
        return cfa;
    }
    let fixed = cfa.shifted(0, 1);
    // Once per run. Every frame of a burst comes from one camera and one
    // capture program, so the second report onwards says nothing new and would
    // bury the rest of the log under one line per frame.
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if !REPORTED.swap(true, Ordering::Relaxed) {
        log::warn!(
            "{file}: header calls the mosaic {} but the green sites are on the other diagonal, \
             so it is being read as {}. That is what a row order the header did not describe \
             looks like. Reported once per run.",
            cfa.name(),
            fixed.name()
        );
    }
    fixed
}

/// Decode one FITS file into a mosaiced frame.
pub fn decode(path: &Path, row_order: RowOrder) -> Result<RawFrame> {
    let file = path.display().to_string();
    let (hdr, data_offset) = read_header(path)?;

    if hdr.get("SIMPLE").map(|v| v.trim() != "T").unwrap_or(true) {
        return Err(SrError::Input(format!(
            "{file}: not a simple FITS file (no SIMPLE = T)"
        )));
    }
    let bitpix = hdr
        .int("BITPIX")
        .ok_or_else(|| SrError::Input(format!("{file}: no BITPIX")))?;
    let naxis = hdr
        .int("NAXIS")
        .ok_or_else(|| SrError::Input(format!("{file}: no NAXIS")))?;
    if !(2..=3).contains(&naxis) {
        return Err(SrError::Input(format!(
            "{file}: NAXIS = {naxis}; only 2-D images are handled"
        )));
    }
    if naxis == 3 {
        let planes = hdr.int("NAXIS3").unwrap_or(1);
        if planes != 1 {
            return Err(SrError::Input(format!(
                "{file}: NAXIS3 = {planes}; this is an already-separated colour image, and the \
                 reconstruction needs the undemosaiced mosaic"
            )));
        }
    }
    let w = hdr.int("NAXIS1").unwrap_or(0);
    let h = hdr.int("NAXIS2").unwrap_or(0);
    if w < 4 || h < 4 {
        return Err(SrError::Input(format!(
            "{file}: implausible image size {w}x{h}"
        )));
    }
    let (w, h) = (w as usize, h as usize);

    let bzero = hdr.number("BZERO").unwrap_or(0.0);
    let bscale = hdr.number("BSCALE").unwrap_or(1.0);
    let raw = read_samples(path, data_offset, w, h, bitpix, bzero, bscale)?;

    // Crop to even dimensions so the 2x2 mosaic tiles exactly, as elsewhere.
    let (cw, ch) = (w & !1, h & !1);
    let flip = match row_order {
        RowOrder::TopDown => false,
        RowOrder::BottomUp => true,
        RowOrder::Auto => hdr
            .text("ROWORDER")
            .map(|s| s.to_ascii_uppercase().starts_with("BOTTOM"))
            .unwrap_or(false),
    };
    let mut data = vec![0u16; cw * ch];
    for y in 0..ch {
        let src = if flip { h - 1 - y } else { y };
        data[y * cw..(y + 1) * cw].copy_from_slice(&raw[src * w..src * w + cw]);
    }
    drop(raw);

    let wcs = plate_solve(&hdr);
    build_frame(
        path,
        &hdr,
        Mosaic {
            data,
            width: cw,
            height: ch,
            full_width: w,
            full_height: h,
            wcs,
        },
    )
}

/// A decoded mosaic and the little the container knows that the header does
/// not.
///
/// The two containers this crate reads differ in how the array is stored and
/// nowhere else: once the samples are in display order and scaled to 16-bit
/// levels, everything left to decide — the white level, which sites are green,
/// the noise model, the metadata — is the same work on the same keywords. So
/// it is done once, [`build_frame`], and both readers hand it one of these.
pub(crate) struct Mosaic {
    /// Samples in stored order, cropped to even dimensions.
    pub data: Vec<u16>,
    pub width: usize,
    pub height: usize,
    /// Dimensions before the crop, which is what the frame records as the
    /// sensor area it came from.
    pub full_width: usize,
    pub full_height: usize,
    /// The plate solve, from wherever the container keeps one. FITS has it in
    /// the header; XISF may have it there or in its own properties, so it is
    /// settled by the reader rather than here.
    pub wcs: Option<Wcs>,
}

/// Everything that follows having the pixels: white level, mosaic, noise and
/// metadata.
pub(crate) fn build_frame(path: &Path, hdr: &Header, m: Mosaic) -> Result<RawFrame> {
    let file = path.display().to_string();
    let Mosaic {
        data,
        width: cw,
        height: ch,
        full_width: w,
        full_height: h,
        wcs,
    } = m;

    let step = quantisation_step(&data);
    // A converter narrower than the file leaves the top codes unreachable, so
    // the largest attainable value is the largest multiple of the step.
    let white = ((65535 / step) * step) as f32;

    let means = cell_means(&data, cw, ch);
    // No mosaic named in the header means a monochrome sensor. There is no way
    // to tell that from the pixels — a mono frame and a badly-labelled colour
    // one look alike — so the header is believed here, and it is the one place
    // in this reader that it is.
    let cfa = match mosaic_pattern(hdr, &file)? {
        Some(p) => reconcile_pattern(p, means, &file),
        None => CfaPattern::MONO,
    };

    let levels = Levels::new([0.0; 4], [white; 4]);
    let samples = SamplePlane::from_u16(cw, ch, data, levels);

    // Shot noise in normalised units. A gain in electrons per ADU refers to the
    // converter's own codes, so the file's quantisation step converts it:
    // var(ADU) = step * ADU / egain, and dividing through by the white level
    // twice puts it in the [0, 1] units the model works in.
    let noise = match hdr.number("EGAIN").filter(|g| g.is_finite() && *g > 1e-3) {
        Some(g) => {
            let alpha = (step as f64 / (g * white as f64)) as f32;
            // Read noise has no portable keyword. Seed it at a few converter
            // codes; the burst estimator replaces this before it is used.
            let read = 3.0 * step as f32 / white;
            NoiseModel::new(alpha, read * read, NoiseSource::Nominal)
        }
        None => NoiseModel::nominal(hdr.number("GAIN").unwrap_or(100.0) as f32, white),
    };

    let instrument = hdr
        .any_text(&["INSTRUME", "CAMERA", "DETNAM"])
        .unwrap_or_default();
    let (make, model) = match instrument.split_once(' ') {
        Some((a, b)) => (a.trim().to_string(), b.trim().to_string()),
        None => (String::new(), instrument.clone()),
    };

    let metadata = FrameMetadata {
        path: file.clone(),
        file_name: path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        make,
        model,
        clean_model: instrument,
        // Not an ISO speed. `GAIN` is the camera's gain setting, and what this
        // field is used for is noticing that it changed mid-burst.
        iso: hdr.any_number(&["GAIN", "ISOSPEED"]).map(|v| v as f32),
        exposure_time: hdr.any_number(&["EXPTIME", "EXPOSURE"]).map(|v| v as f32),
        capture_time: hdr.any_text(&["DATE-OBS"]),
        aperture: None,
        focal_length: hdr.number("FOCALLEN").map(|v| v as f32),
        filter: hdr
            .any_text(&["FILTER", "FILTER1", "FILTNAME"])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        pixel_pitch_um: hdr
            .any_number(&["XPIXSZ", "PIXSIZE1", "PIXSIZE"])
            .map(|v| v as f32)
            .filter(|v| *v > 0.0),
        orientation: 0,
        wb_coeffs: [1.0, 1.0, 1.0],
        // No colorimetric characterisation exists for an astronomy camera with
        // an arbitrary filter in front of it. Left at zero so that the colour
        // stage says so rather than inventing one.
        xyz_to_cam: [[0.0; 3]; 3],
        black_levels: [0.0; 4],
        white_level: white,
        crop: (0, 0, cw, ch),
        full_width: w,
        full_height: h,
        sha256_prefix: super::sha256_prefix(path).unwrap_or_default(),
        wcs,
    };

    Ok(RawFrame {
        width: cw,
        height: ch,
        samples,
        cfa,
        defects: DefectMask::none(cw, ch),
        noise,
        metadata,
    })
}

/// The mosaic named by the header, with any Bayer origin offset applied.
fn mosaic_pattern(hdr: &Header, file: &str) -> Result<Option<CfaPattern>> {
    let name = match hdr.any_text(&["BAYERPAT", "BAYPAT", "COLORTYP"]) {
        Some(n) if !n.trim().is_empty() => n.trim().to_ascii_uppercase(),
        _ => return Ok(None),
    };
    if name == "MONO" || name == "NONE" {
        return Ok(None);
    }
    let base = CfaPattern::from_name(&name).ok_or_else(|| {
        SrError::Input(format!(
            "{file}: BAYERPAT = {name:?}; only 2x2 RGB Bayer mosaics are handled"
        ))
    })?;
    // Some programs record a sub-frame's offset into the sensor's mosaic
    // separately from the pattern name.
    let dx = hdr.any_number(&["XBAYROFF", "BAYOFFX"]).unwrap_or(0.0) as usize;
    let dy = hdr.any_number(&["YBAYROFF", "BAYOFFY"]).unwrap_or(0.0) as usize;
    Ok(Some(base.shifted(dx & 1, dy & 1)))
}

/// Read the data unit and convert it to 16-bit levels.
///
/// Everything is scaled into `[0, 65535]` rather than kept in its own units, so
/// that a frame costs two bytes per site whatever the file's type and so that
/// the rest of the pipeline sees one representation. Floating-point files are
/// the only lossy case, and they are the ones that have already been through
/// somebody else's calibration.
fn read_samples(
    path: &Path,
    offset: u64,
    w: usize,
    h: usize,
    bitpix: i64,
    bzero: f64,
    bscale: f64,
) -> Result<Vec<u16>> {
    let n = w * h;
    let width = match bitpix {
        8 => 1usize,
        16 => 2,
        32 | -32 => 4,
        64 | -64 => 8,
        other => {
            return Err(SrError::Input(format!(
                "{}: BITPIX = {other} is not a FITS data type",
                path.display()
            )));
        }
    };
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0u8; n * width];
    f.read_exact(&mut bytes).map_err(|e| {
        SrError::Input(format!(
            "{}: data unit is short of the {w}x{h} the header declares: {e}",
            path.display()
        ))
    })?;

    let mut out = vec![0u16; n];
    // The common case by a wide margin: 16-bit signed shifted into unsigned by
    // BZERO, which is exactly a 16-bit unsigned image and needs no arithmetic.
    if bitpix == 16 && bscale == 1.0 && bzero == 32768.0 {
        for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *o = (i16::from_be_bytes(*c) as i32 + 32768) as u16;
        }
        return Ok(out);
    }

    let physical: fn(&[u8]) -> f64 = match bitpix {
        8 => |c| c[0] as f64,
        16 => |c| i16::from_be_bytes([c[0], c[1]]) as f64,
        32 => |c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64,
        -32 => |c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64,
        64 => |c| i64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64,
        _ => |c| f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
    };

    // Floating-point and wide-integer files carry no declared full scale, so
    // one has to be inferred. The choices are the two conventions in the wild:
    // already normalised to [0, 1], or in 16-bit units.
    let mut peak = 0.0f64;
    for c in bytes.chunks_exact(width).step_by(11) {
        let v = bzero + bscale * physical(c);
        if v.is_finite() && v > peak {
            peak = v;
        }
    }
    let full_scale = if bitpix < 0 || bitpix == 32 || bitpix == 64 {
        if peak <= 1.5 {
            1.0
        } else if peak <= 65535.0 {
            65535.0
        } else {
            peak
        }
    } else {
        65535.0
    };
    let gain = 65535.0 / full_scale.max(1e-9);
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(width)) {
        let v = (bzero + bscale * physical(c)) * gain;
        *o = if v.is_finite() {
            v.clamp(0.0, 65535.0) as u16
        } else {
            0
        };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(key: &str, value: &str) -> String {
        let s = format!("{key:<8}= {value:<70}");
        s[..CARD].to_string()
    }

    /// Build a minimal 16-bit unsigned FITS file in memory.
    fn synth(w: usize, h: usize, extra: &[(&str, &str)], pixels: &[u16]) -> Vec<u8> {
        let mut cards = vec![
            card("SIMPLE", "                   T"),
            card("BITPIX", "                  16"),
            card("NAXIS", "                   2"),
            card("NAXIS1", &format!("{w:20}")),
            card("NAXIS2", &format!("{h:20}")),
            card("BZERO", "               32768"),
            card("BSCALE", "                   1"),
        ];
        for (k, v) in extra {
            cards.push(card(k, v));
        }
        cards.push(format!("{:<80}", "END"));
        let mut text = cards.concat();
        while text.len() % BLOCK != 0 {
            text.push(' ');
        }
        let mut bytes = text.into_bytes();
        for &p in pixels {
            bytes.extend_from_slice(&((p as i32 - 32768) as i16).to_be_bytes());
        }
        while bytes.len() % BLOCK != 0 {
            bytes.push(0);
        }
        bytes
    }

    /// Deliberately not the system temp directory: on the machine this was
    /// developed on that is a different volume from the workspace, and it has
    /// been full.
    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut d = std::env::current_exe().unwrap();
        d.pop();
        d.push("fits-tests");
        std::fs::create_dir_all(&d).unwrap();
        d.push(name);
        std::fs::write(&d, bytes).unwrap();
        d
    }

    /// The keywords an ASIAIR writes, values taken from a real frame of the
    /// NGC 6871 burst.
    fn asiair_solve() -> Vec<(&'static str, &'static str)> {
        vec![
            ("CTYPE1", "'RA---TAN-SIP'"),
            ("CTYPE2", "'DEC--TAN-SIP'"),
            ("CRPIX1", "        3943.41202799"),
            ("CRPIX2", "        1069.41572062"),
            ("CRVAL1", "        300.082313872"),
            ("CRVAL2", "        34.7827621294"),
            ("CD1_1", "     -2.03473314368E-05"),
            ("CD1_2", "      0.00124206328656"),
            ("CD2_1", "     -0.00123976252717"),
            ("CD2_2", "     -1.79194235287E-05"),
        ]
    }

    #[test]
    fn a_plate_solve_is_read_when_the_header_carries_one() {
        let bytes = synth(8, 8, &asiair_solve(), &[1000u16; 64]);
        let p = write("solved.fit", &bytes);
        let f = decode(&p, RowOrder::Auto).unwrap();
        let w = f.metadata.wcs.expect("the solve should have been read");

        // FITS counts from one; everything downstream counts from zero.
        assert!((w.crpix.0 - 3942.41202799).abs() < 1e-6);
        assert!((w.crpix.1 - 1068.41572062).abs() < 1e-6);
        // 3.76 um at 173 mm.
        assert!(
            (w.scale_arcsec() - 4.47).abs() < 0.05,
            "scale {}",
            w.scale_arcsec()
        );
    }

    #[test]
    fn a_header_with_no_solve_simply_has_none() {
        let bytes = synth(8, 8, &[("BAYERPAT", "'RGGB    '")], &[1000u16; 64]);
        let p = write("unsolved.fit", &bytes);
        assert!(decode(&p, RowOrder::Auto).unwrap().metadata.wcs.is_none());
    }

    /// A projection we do not implement is not a solve we can use. Reading the
    /// linear terms anyway would put frames in the wrong place with no
    /// indication that anything had gone wrong.
    #[test]
    fn a_projection_we_do_not_know_is_declined() {
        let mut cards = asiair_solve();
        cards[0] = ("CTYPE1", "'RA---SIN'");
        cards[1] = ("CTYPE2", "'DEC--SIN'");
        let bytes = synth(8, 8, &cards, &[1000u16; 64]);
        let p = write("sin.fit", &bytes);
        assert!(decode(&p, RowOrder::Auto).unwrap().metadata.wcs.is_none());
    }

    /// The older spelling, which plenty of software still writes.
    #[test]
    fn cdelt_and_crota_are_understood_as_well_as_a_cd_matrix() {
        let cards = vec![
            ("CTYPE1", "'RA---TAN'"),
            ("CTYPE2", "'DEC--TAN'"),
            ("CRPIX1", "                3124"),
            ("CRPIX2", "                2088"),
            ("CRVAL1", "        300.082313872"),
            ("CRVAL2", "        34.7827621294"),
            ("CDELT1", "     -0.00124388888889"),
            ("CDELT2", "      0.00124388888889"),
            ("CROTA2", "                  90"),
        ];
        let bytes = synth(8, 8, &cards, &[1000u16; 64]);
        let p = write("crota.fit", &bytes);
        let w = decode(&p, RowOrder::Auto).unwrap().metadata.wcs.unwrap();
        assert!(
            (w.scale_arcsec() - 4.478).abs() < 0.05,
            "scale {}",
            w.scale_arcsec()
        );
    }

    #[test]
    fn cards_split_on_the_value_indicator() {
        let c = format!("{:<80}", "BAYERPAT= 'RGGB    '           / Bayer pattern");
        assert_eq!(
            parse_card(&c),
            Some(("BAYERPAT".into(), "'RGGB    '".into()))
        );
        assert_eq!(unquote("'RGGB    '"), "RGGB");
    }

    #[test]
    fn a_slash_inside_a_string_is_not_a_comment() {
        let c = format!(
            "{:<80}",
            "FILE    = 'a/b.fit'            / where it came from"
        );
        let (_, v) = parse_card(&c).unwrap();
        assert_eq!(unquote(&v), "a/b.fit");
    }

    #[test]
    fn comment_cards_carry_no_value() {
        let c = format!("{:<80}", "COMMENT   FITS is defined in A&A 376, 359");
        assert!(parse_card(&c).is_none());
    }

    #[test]
    fn the_quantisation_step_finds_a_shifted_converter() {
        // 14 bits written into 16: every value is a multiple of four.
        let shifted: Vec<u16> = (0..4000u16).map(|v| v * 4).collect();
        assert_eq!(quantisation_step(&shifted), 4);
        let full: Vec<u16> = (0..4000u16).map(|v| v * 3 + 1).collect();
        assert_eq!(quantisation_step(&full), 1);
        // A common factor that is not a power of two is not a converter shift.
        let by_28: Vec<u16> = (1..2000u16).map(|v| v * 28).collect();
        assert_eq!(quantisation_step(&by_28), 4);
        assert_eq!(quantisation_step(&[0, 0, 0]), 1);
    }

    #[test]
    fn the_green_diagonal_is_read_from_the_pixels() {
        // R low, G equal, B middling: an ordinary light-polluted sky.
        assert_eq!(
            green_is_antidiagonal([3790.0, 5361.0, 5362.0, 4950.0]),
            Some(true)
        );
        // The same frame stored bottom-up: the greens move to the main diagonal.
        assert_eq!(
            green_is_antidiagonal([5361.0, 4950.0, 3790.0, 5362.0]),
            Some(false)
        );
        // A grey frame says nothing and must abstain rather than guess.
        assert_eq!(
            green_is_antidiagonal([1000.0, 1001.0, 1002.0, 1003.0]),
            None
        );
    }

    #[test]
    fn sparse_star_samples_cannot_overrule_the_mosaic_pattern() {
        let mut pixels = vec![0u16; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                pixels[y * 32 + x] = [2000, 6000, 6000, 4000][(y % 2) * 2 + x % 2];
            }
        }
        // A narrow feature can put the two green sites at very different
        // levels in a sampled cell. Looking only at every fourth cell aliases
        // this structure into the channel means and declares the R/B sites G.
        for y in (0..32).step_by(8) {
            for x in (0..32).step_by(8) {
                for (dy, dx, value) in [(0, 0, 5000), (0, 1, 1000), (1, 0, 8000), (1, 1, 5000)] {
                    pixels[(y + dy) * 32 + x + dx] = value;
                }
            }
        }
        let pattern = reconcile_pattern(CfaPattern::RGGB, cell_means(&pixels, 32, 32), "stars");
        assert_eq!(pattern, CfaPattern::RGGB);
    }

    #[test]
    fn a_row_flipped_mosaic_is_corrected() {
        let fixed = reconcile_pattern(CfaPattern::RGGB, [5361.0, 4950.0, 3790.0, 5362.0], "test");
        assert_eq!(fixed, CfaPattern::GBRG);
        // Agreement leaves the header's pattern alone.
        let kept = reconcile_pattern(CfaPattern::RGGB, [3790.0, 5361.0, 5362.0, 4950.0], "test");
        assert_eq!(kept, CfaPattern::RGGB);
    }

    #[test]
    fn a_minimal_bayer_file_decodes() {
        // 4x4 RGGB, every value a multiple of four but not of eight, so the
        // white level must come out at the largest such multiple below 65536.
        let mut px = vec![0u16; 16];
        for y in 0..4 {
            for x in 0..4 {
                px[y * 4 + x] = match (y % 2, x % 2) {
                    (0, 0) => 1004,
                    (1, 1) => 2004,
                    _ => 4004,
                };
            }
        }
        let bytes = synth(
            4,
            4,
            &[("BAYERPAT", "'RGGB    '"), ("ROWORDER", "'TOP-DOWN'")],
            &px,
        );
        let p = write("minimal.fit", &bytes);
        let f = decode(&p, RowOrder::Auto).unwrap();
        assert_eq!((f.width, f.height), (4, 4));
        assert_eq!(f.cfa, CfaPattern::RGGB);
        assert_eq!(f.metadata.white_level, 65532.0);
        assert!(
            (f.value(0, 0) - 1004.0 / 65532.0).abs() < 1e-6,
            "{}",
            f.value(0, 0)
        );
    }

    #[test]
    fn odd_dimensions_are_cropped_to_keep_the_mosaic_whole() {
        let px = vec![4000u16; 5 * 5];
        let bytes = synth(5, 5, &[("BAYERPAT", "'RGGB    '")], &px);
        let p = write("odd.fit", &bytes);
        let f = decode(&p, RowOrder::TopDown).unwrap();
        assert_eq!((f.width, f.height), (4, 4));
    }

    #[test]
    fn bottom_up_storage_is_flipped_to_screen_order() {
        // A gradient down the image: reading it bottom-up must invert it.
        let mut px = vec![0u16; 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                px[y * 4 + x] = 1000 + 1000 * y as u16;
            }
        }
        let bytes = synth(4, 4, &[("BAYERPAT", "'RGGB    '")], &px);
        let p = write("bottomup.fit", &bytes);
        let up = decode(&p, RowOrder::BottomUp).unwrap();
        let down = decode(&p, RowOrder::TopDown).unwrap();
        assert!(up.value(0, 0) > up.value(0, 3));
        assert!(down.value(0, 0) < down.value(0, 3));
    }

    #[test]
    fn a_file_without_a_mosaic_is_read_as_monochrome() {
        let px = vec![4000u16; 16];
        let bytes = synth(4, 4, &[("FILTER", "'Ha      '")], &px);
        let p = write("mono.fit", &bytes);
        let f = decode(&p, RowOrder::TopDown).unwrap();
        assert!(f.is_mono());
        assert_eq!(f.channels(), 1);
        assert_eq!(f.cfa.name(), "Mono");
        assert_eq!(f.metadata.filter.as_deref(), Some("Ha"));
        // Every site goes to the one channel there is.
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(f.channel_at(x, y), 0);
            }
        }
    }

    #[test]
    fn a_monochrome_guide_carries_the_cell_mean_in_every_plane() {
        let px: Vec<u16> = (0..16).map(|i| 1000 + 100 * i as u16).collect();
        let bytes = synth(4, 4, &[], &px);
        let p = write("monoguide.fit", &bytes);
        let f = decode(&p, RowOrder::TopDown).unwrap();
        let g = f.guide_rgb();
        // Top-left cell holds 1000, 1100, 1400, 1500; the mean is 1250.
        let want = 1250.0 / f.metadata.white_level;
        for c in 0..3 {
            assert!(
                (g.channel(c).data[0] - want).abs() < 1e-6,
                "channel {c} gave {}",
                g.channel(c).data[0]
            );
        }
        assert!((g.luma().data[0] - want).abs() < 1e-6);
    }

    #[test]
    fn shot_noise_comes_from_the_gain_when_the_header_gives_one() {
        let px: Vec<u16> = (0..16).map(|i| 1000 + 4 * i as u16).collect();
        let bytes = synth(4, 4, &[("BAYERPAT", "'RGGB    '"), ("EGAIN", "1.0")], &px);
        let p = write("egain.fit", &bytes);
        let f = decode(&p, RowOrder::TopDown).unwrap();
        // alpha = step / (egain * white) = 4 / 65532.
        assert!(
            (f.noise.alpha - 4.0 / 65532.0).abs() < 1e-9,
            "{}",
            f.noise.alpha
        );
    }
}
