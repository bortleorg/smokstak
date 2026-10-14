//! Bounded, read-only access to uncompressed mono FITS images.
//!
//! Integer16 values follow the existing decoder's normalization, including its
//! u16 conversion and inferred white level. Opening scans the legacy even crop
//! with fixed scratch space to infer that level. Unlike the Bayer-oriented full
//! decoder, mono windows retain odd trailing rows and columns. Floating-point
//! values retain their physical `BZERO + BSCALE * sample` units, signed values,
//! and nonfinite values: the legacy decoder's lossy u16 conversion is deliberately
//! not applied. Callers must not assume float files are in normalized units.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::{ReadOptions, RowOrder, fits};
use sr_core::{Result, SrError};

/// Largest single read `read_rect` issues. Rows of a window are strided by the
/// full image width, so a window covering at least half of each row is read as
/// runs of whole file rows: one request per run instead of one per row. On a
/// network share each request is a round trip, and for a wide window per-row
/// reads were four to five times slower. A narrow window keeps per-row reads,
/// where whole rows would multiply the bytes transferred instead (measured
/// twice as slow for a mosaic build reading ~5% of each row).
pub const MAX_READ_SPAN_BYTES: usize = 8 << 20;

/// Units of samples returned by a window reader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleUnits {
    /// Legacy integer normalization; 1.0 is the inferred saturation level.
    NormalizedInteger { white_level: f32 },
    /// FITS physical values. No saturation threshold can be inferred safely.
    PhysicalFloat,
}

/// An open primary mono FITS image. No full-frame allocation or mutable file access.
#[derive(Debug)]
pub struct FitsWindowReader {
    file: File,
    width: usize,
    height: usize,
    offset: u64,
    bitpix: i64,
    bytes_per_sample: usize,
    bzero: f64,
    bscale: f64,
    flip: bool,
    inverse_white: f32,
}

fn invalid(message: impl Into<String>) -> SrError {
    SrError::Input(format!("FITS window: {}", message.into()))
}

impl FitsWindowReader {
    /// Open a 2-D primary image (a singleton third axis is also accepted).
    /// Supports BITPIX 16 and -32; unsupported storage is an explicit error.
    pub fn open(path: &Path, options: &ReadOptions) -> Result<Self> {
        let (hdr, offset) = fits::read_header(path)?;
        if hdr.get("SIMPLE").map(str::trim) != Some("T") {
            return Err(invalid("requires SIMPLE = T"));
        }
        if hdr.get("ZIMAGE").map(str::trim) == Some("T")
            || hdr.get("GROUPS").map(str::trim) == Some("T")
        {
            return Err(invalid(
                "compressed images and random groups are unsupported",
            ));
        }
        let integer = |key: &str| -> Result<i64> {
            hdr.get(key)
                .and_then(|s| s.trim().parse::<i64>().ok())
                .ok_or_else(|| invalid(format!("missing or invalid {key}")))
        };
        match integer("NAXIS")? {
            2 => {}
            3 if integer("NAXIS3")? == 1 => {}
            _ => return Err(invalid("requires a mono 2-D image")),
        }
        if let Some(pattern) = hdr.any_text(&["BAYERPAT", "BAYPAT", "COLORTYP"])
            && !matches!(
                pattern.trim().to_ascii_uppercase().as_str(),
                "" | "MONO" | "NONE"
            )
        {
            return Err(invalid("Bayer and color images are unsupported"));
        }
        let width = usize::try_from(integer("NAXIS1")?).map_err(|_| invalid("invalid NAXIS1"))?;
        let height = usize::try_from(integer("NAXIS2")?).map_err(|_| invalid("invalid NAXIS2"))?;
        if width == 0 || height == 0 {
            return Err(invalid("empty image"));
        }
        let bitpix = integer("BITPIX")?;
        let bytes_per_sample = match bitpix {
            16 => 2,
            -32 => 4,
            _ => {
                return Err(invalid(format!(
                    "unsupported BITPIX {bitpix}; supported: 16, -32"
                )));
            }
        };
        let scaling = |key: &str, default: f64| -> Result<f64> {
            match hdr.get(key) {
                None => Ok(default),
                Some(_) => hdr
                    .number(key)
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| invalid(format!("invalid {key}"))),
            }
        };
        let file = File::open(path)?;
        let length = (width as u64)
            .checked_mul(height as u64)
            .and_then(|n| n.checked_mul(bytes_per_sample as u64))
            .and_then(|n| n.checked_add(offset))
            .ok_or_else(|| invalid("data size overflow"))?;
        if file.metadata()?.len() < length {
            return Err(invalid("truncated image data"));
        }
        let flip = match options.fits_row_order {
            RowOrder::TopDown => false,
            RowOrder::BottomUp => true,
            RowOrder::Auto => hdr
                .text("ROWORDER")
                .is_some_and(|s| s.to_ascii_uppercase().starts_with("BOTTOM")),
        };
        let mut reader = Self {
            file,
            width,
            height,
            offset,
            bitpix,
            bytes_per_sample,
            bzero: scaling("BZERO", 0.0)?,
            bscale: scaling("BSCALE", 1.0)?,
            flip,
            inverse_white: 1.0,
        };
        if bitpix == 16 {
            // Match quantisation_step over exactly the full decoder's even crop,
            // including its flipped-row selection. Stop once an odd code settles it.
            let mut bits = 0u16;
            let mut scratch = [0u8; 8192];
            'rows: for y in 0..(height & !1) {
                for x in (0..(width & !1)).step_by(scratch.len() / 2) {
                    let count = ((width & !1) - x).min(scratch.len() / 2);
                    reader.seek_pixel(x, y)?;
                    reader.file.read_exact(&mut scratch[..count * 2])?;
                    for c in scratch[..count * 2].as_chunks::<2>().0 {
                        bits |= reader.integer_sample(c);
                    }
                    if bits & 1 != 0 {
                        break 'rows;
                    }
                }
            }
            let step = if bits == 0 {
                1
            } else {
                1u32 << bits.trailing_zeros()
            };
            reader.inverse_white = 1.0 / ((65535 / step) * step) as f32;
        }
        Ok(reader)
    }

    /// Full mono dimensions, without the legacy decoder's even-dimension crop.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn sample_units(&self) -> SampleUnits {
        if self.bitpix == 16 {
            SampleUnits::NormalizedInteger {
                white_level: (1.0 / self.inverse_white).round(),
            }
        } else {
            SampleUnits::PhysicalFloat
        }
    }

    /// Threshold in returned units, only available for integer detector codes.
    pub fn saturation_threshold(&self) -> Option<f32> {
        (self.bitpix == 16).then_some(1.0)
    }

    fn integer_sample(&self, bytes: &[u8]) -> u16 {
        let value = self.bzero + self.bscale * i16::from_be_bytes([bytes[0], bytes[1]]) as f64;
        // Required for compatibility with fits::read_samples, not a float policy.
        if value.is_finite() {
            value.clamp(0.0, 65535.0) as u16
        } else {
            0
        }
    }

    fn seek_pixel(&mut self, x: usize, y: usize) -> Result<()> {
        let row = if self.flip { self.height - 1 - y } else { y };
        let position = (row as u64)
            .checked_mul(self.width as u64)
            .and_then(|n| n.checked_add(x as u64))
            .and_then(|n| n.checked_mul(self.bytes_per_sample as u64))
            .and_then(|n| n.checked_add(self.offset))
            .ok_or_else(|| invalid("pixel offset overflow"))?;
        self.file.seek(SeekFrom::Start(position))?;
        Ok(())
    }

    /// Read a nonempty rectangle in display row order into a row-major vector.
    /// Memory is the output plus at most `MAX_READ_SPAN_BYTES` of encoded
    /// scratch (or one row, if a single row is larger).
    /// Coordinates, products, allocation failures and truncated reads are errors.
    pub fn read_rect(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        self.read_rect_spanning(x, y, width, height, MAX_READ_SPAN_BYTES)
    }

    fn read_rect_spanning(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        max_span: usize,
    ) -> Result<Vec<f32>> {
        if width == 0
            || height == 0
            || x.checked_add(width).is_none_or(|end| end > self.width)
            || y.checked_add(height).is_none_or(|end| end > self.height)
        {
            return Err(invalid("rectangle is empty or outside the image"));
        }
        let count = width
            .checked_mul(height)
            .ok_or_else(|| invalid("rectangle size overflow"))?;
        let row_bytes = width
            .checked_mul(self.bytes_per_sample)
            .ok_or_else(|| invalid("row size overflow"))?;
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| invalid("rectangle allocation failed"))?;
        let file_row_bytes = self
            .width
            .checked_mul(self.bytes_per_sample)
            .ok_or_else(|| invalid("row size overflow"))?;
        // Whole file rows per request, bounded; the last row of a run needs only
        // the requested columns, so a run of n rows spans (n-1) full rows + one.
        let rows_per_run = if 2 * row_bytes >= file_row_bytes {
            (max_span.saturating_sub(row_bytes) / file_row_bytes + 1).clamp(1, height)
        } else {
            1
        };
        let mut scratch = Vec::new();
        let scratch_bytes = (rows_per_run - 1) * file_row_bytes + row_bytes;
        scratch
            .try_reserve_exact(scratch_bytes)
            .map_err(|_| invalid("read buffer allocation failed"))?;
        scratch.resize(scratch_bytes, 0u8);
        let mut first = 0;
        while first < height {
            let rows = rows_per_run.min(height - first);
            // File rows of this run are contiguous, ascending or (flipped)
            // descending in display order; read from the lowest one.
            let file_row = |dy: usize| {
                if self.flip {
                    self.height - 1 - (y + dy)
                } else {
                    y + dy
                }
            };
            let low = file_row(first).min(file_row(first + rows - 1));
            let span = (rows - 1) * file_row_bytes + row_bytes;
            let position = (low as u64)
                .checked_mul(file_row_bytes as u64)
                .and_then(|n| n.checked_add((x * self.bytes_per_sample) as u64))
                .and_then(|n| n.checked_add(self.offset))
                .ok_or_else(|| invalid("pixel offset overflow"))?;
            self.file.seek(SeekFrom::Start(position))?;
            self.file.read_exact(&mut scratch[..span])?;
            for dy in first..first + rows {
                let start = (file_row(dy) - low) * file_row_bytes;
                for c in scratch[start..start + row_bytes].chunks_exact(self.bytes_per_sample) {
                    out.push(if self.bitpix == 16 {
                        self.integer_sample(c) as f32 * self.inverse_white
                    } else {
                        (self.bzero
                            + self.bscale * f32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64)
                            as f32
                    });
                }
            }
            first += rows;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn fixture(
        bitpix: i64,
        width: usize,
        height: usize,
        extra: &[(&str, &str)],
        data: &[u8],
    ) -> Fixture {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "sr-window-{}-{}.fits",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut header = String::new();
        for (key, value) in [
            ("SIMPLE", "T".to_owned()),
            ("BITPIX", bitpix.to_string()),
            ("NAXIS", "2".to_owned()),
            ("NAXIS1", width.to_string()),
            ("NAXIS2", height.to_string()),
        ] {
            header.push_str(&format!("{:<80}", format!("{key:<8}= {value}")));
        }
        for (key, value) in extra {
            header.push_str(&format!("{:<80}", format!("{key:<8}= {value}")));
        }
        header.push_str(&format!("{:<80}", "END"));
        let mut bytes = header.into_bytes();
        bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
        Fixture(path)
    }

    #[test]
    fn integer_crops_match_decoder_in_both_orientations_with_odd_edges() {
        // Multiples of four require the same inferred 65532 white level.
        let values: Vec<u16> = (0..35).map(|i| (1000 + i * 4) as u16).collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| ((v as i32 - 32768) as i16).to_be_bytes())
            .collect();
        let source = fixture(
            16,
            7,
            5,
            &[("BZERO", "32768"), ("ROWORDER", "'BOTTOM-UP'")],
            &bytes,
        );
        for order in [RowOrder::Auto, RowOrder::BottomUp, RowOrder::TopDown] {
            let options = ReadOptions {
                fits_row_order: order,
            };
            let full = crate::decode_with(&source.0, &options).unwrap();
            let mut window = FitsWindowReader::open(&source.0, &options).unwrap();
            assert_eq!(window.dimensions(), (7, 5));
            assert_eq!(
                window.sample_units(),
                SampleUnits::NormalizedInteger {
                    white_level: 65532.0
                }
            );
            assert_eq!(window.saturation_threshold(), Some(1.0));
            for (x, y, w, h) in [(0, 0, 6, 4), (1, 1, 3, 3), (5, 3, 1, 1)] {
                let got = window.read_rect(x, y, w, h).unwrap();
                for dy in 0..h {
                    for dx in 0..w {
                        assert_eq!(got[dy * w + dx], full.value(x + dx, y + dy));
                    }
                }
            }
            let stored_y = if order == RowOrder::TopDown { 4 } else { 0 };
            assert_eq!(
                window.read_rect(6, 4, 1, 1).unwrap(),
                vec![values[stored_y * 7 + 6] as f32 * (1.0 / 65532.0)]
            );
        }
    }

    #[test]
    fn runs_of_any_length_read_the_same_window_in_both_orientations() {
        let (width, height) = (9, 11);
        let bytes: Vec<u8> = (0..width * height)
            .flat_map(|i| ((i as i32 * 37 % 4000 * 4 - 32768) as i16).to_be_bytes())
            .collect();
        for order in ["'BOTTOM-UP'", "'TOP-DOWN'"] {
            let source = fixture(
                16,
                width,
                height,
                &[("BZERO", "32768"), ("ROWORDER", order)],
                &bytes,
            );
            let mut window = FitsWindowReader::open(&source.0, &ReadOptions::default()).unwrap();
            for (x, y, w, h) in [(0, 0, 9, 11), (2, 3, 4, 7), (8, 10, 1, 1), (1, 0, 7, 11)] {
                let row_at_a_time = window.read_rect_spanning(x, y, w, h, 0).unwrap();
                for span in [2 * width * 2, 3 * width * 2 + 1, MAX_READ_SPAN_BYTES] {
                    assert_eq!(
                        window.read_rect_spanning(x, y, w, h, span).unwrap(),
                        row_at_a_time,
                        "{order} rect {x},{y} {w}x{h} span {span}"
                    );
                }
            }
        }
    }

    #[test]
    fn scaled_signed_integer_matches_legacy_conversion() {
        let bytes: Vec<u8> = (-12i16..12).flat_map(i16::to_be_bytes).collect();
        let source = fixture(16, 6, 4, &[("BZERO", "5.5"), ("BSCALE", "2.5")], &bytes);
        let full = crate::decode(&source.0).unwrap();
        let mut window = FitsWindowReader::open(&source.0, &ReadOptions::default()).unwrap();
        let got = window.read_rect(0, 0, 6, 4).unwrap();
        for y in 0..4 {
            for x in 0..6 {
                assert_eq!(got[y * 6 + x], full.value(x, y));
            }
        }
    }

    #[test]
    fn floats_preserve_signed_nonfinite_and_physical_units() {
        let values = [
            -2.0f32,
            0.25,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            70000.0,
        ];
        let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_be_bytes).collect();
        let source = fixture(-32, 3, 2, &[("BZERO", "0.5"), ("BSCALE", "2")], &bytes);
        let mut window = FitsWindowReader::open(&source.0, &ReadOptions::default()).unwrap();
        assert_eq!(window.sample_units(), SampleUnits::PhysicalFloat);
        assert_eq!(window.saturation_threshold(), None);
        let got = window.read_rect(0, 0, 3, 2).unwrap();
        assert_eq!(got[0], -3.5);
        assert_eq!(got[1], 1.0);
        assert!(got[2].is_nan());
        assert_eq!(got[3], f32::INFINITY);
        assert_eq!(got[4], f32::NEG_INFINITY);
        assert_eq!(got[5], 140000.5);
    }

    #[test]
    fn rejects_truncation_invalid_rectangles_and_unsupported_storage() {
        let source = fixture(16, 5, 5, &[], &[0; 50]);
        let mut window = FitsWindowReader::open(&source.0, &ReadOptions::default()).unwrap();
        for rect in [
            (0, 0, 0, 1),
            (4, 4, 2, 1),
            (0, 5, 1, 1),
            (usize::MAX, 0, 2, 1),
        ] {
            assert!(window.read_rect(rect.0, rect.1, rect.2, rect.3).is_err());
        }
        let short = fixture(16, 5, 5, &[], &[0; 49]);
        assert!(FitsWindowReader::open(&short.0, &ReadOptions::default()).is_err());
        let unsupported = fixture(32, 5, 5, &[], &[0; 100]);
        assert!(FitsWindowReader::open(&unsupported.0, &ReadOptions::default()).is_err());
        let color = fixture(16, 5, 5, &[("BAYERPAT", "'RGGB'")], &[0; 50]);
        assert!(FitsWindowReader::open(&color.0, &ReadOptions::default()).is_err());
    }
}
