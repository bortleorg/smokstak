//! Uncompressed floating-point astronomy masters. Samples are planar and unclamped.
use anyhow::{ensure, Context, Result};
use sr_core::plane::Plane;
use std::{
    fs::File,
    io::{BufWriter, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

fn dimensions(rgb: &[Plane<f32>; 3], channels: usize) -> Result<(usize, usize)> {
    ensure!(
        channels == 1 || channels == 3,
        "expected mono or RGB output"
    );
    let (w, h) = rgb[0].dims();
    ensure!(w > 0 && h > 0, "cannot write an empty master");
    let n = w.checked_mul(h).context("image dimensions overflow")?;
    ensure!(
        rgb[..channels]
            .iter()
            .all(|p| p.dims() == (w, h) && p.data.len() == n),
        "inconsistent image planes"
    );
    Ok((w, h))
}

/// IEEE Float32 FITS, bottom-up rows, planar RGB, big-endian samples.
pub fn write_fits(path: &Path, rgb: &[Plane<f32>; 3], channels: usize) -> Result<()> {
    let (w, h) = dimensions(rgb, channels)?;
    let header = fits_header(w, h, channels);
    let mut f =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    f.write_all(&header)?;
    for p in &rgb[..channels] {
        for row in p.data.chunks_exact(w).rev() {
            for &v in row {
                f.write_all(&v.to_be_bytes())?;
            }
        }
    }
    let bytes = w * h * channels * 4;
    f.write_all(&vec![0; (2880 - bytes % 2880) % 2880])?;
    f.flush()?;
    Ok(())
}

fn fits_header(w: usize, h: usize, channels: usize) -> Vec<u8> {
    let mut cards = vec![
        format!("{:<8}= {:>20}", "SIMPLE", "T"),
        format!("{:<8}= {:>20}", "BITPIX", -32),
        format!("{:<8}= {:>20}", "NAXIS", if channels == 1 { 2 } else { 3 }),
        format!("{:<8}= {:>20}", "NAXIS1", w),
        format!("{:<8}= {:>20}", "NAXIS2", h),
    ];
    if channels == 3 {
        cards.push(format!("{:<8}= {:>20}", "NAXIS3", 3));
    }
    cards.extend([
        "ROWORDER= 'BOTTOM-UP'".into(),
        format!("COLORSPC= '{}'", if channels == 1 { "Gray" } else { "RGB" }),
        "IMAGETYP= 'MASTER'".into(),
        "HISTORY Written by smokstak; floating-point samples without display transfer".into(),
        "END".into(),
    ]);
    let mut header = Vec::new();
    for card in cards {
        header.extend_from_slice(format!("{card:<80}").as_bytes());
    }
    header.resize(header.len().div_ceil(2880) * 2880, b' ');
    header
}

/// Stateful mono FITS output accepting each tile of a fixed grid exactly once.
/// Tiles may arrive in any order. Publish the staging stream only after `finish`
/// succeeds; an I/O failure leaves an incomplete artifact. Storage is one byte per
/// tile (at most 16 MiB) and one serialized tile row (at most 16 KiB).
pub struct MonoFitsTileWriter<W: Write + Seek> {
    writer: W,
    width: usize,
    height: usize,
    tile: usize,
    columns: usize,
    written: Vec<u8>,
    remaining: usize,
    row: Vec<u8>,
    payload_end: u64,
    padding: usize,
    header_len: u64,
}

impl<W: Write + Seek> MonoFitsTileWriter<W> {
    pub fn new(mut writer: W, width: usize, height: usize, tile: usize) -> Result<Self> {
        ensure!(width > 0 && height > 0, "cannot write an empty master");
        ensure!((1..=4096).contains(&tile), "tile must be in 1..=4096");
        let payload = u64::try_from(width)?
            .checked_mul(u64::try_from(height)?)
            .and_then(|n| n.checked_mul(4))
            .context("FITS payload size overflow")?;
        let header = fits_header(width, height, 1);
        let header_len = header.len() as u64;
        let payload_end = header_len
            .checked_add(payload)
            .context("FITS file size overflow")?;
        let padding = ((2880 - payload % 2880) % 2880) as usize;
        payload_end
            .checked_add(padding as u64)
            .filter(|&n| n <= i64::MAX as u64)
            .context("FITS file size overflow")?;
        let columns = width.div_ceil(tile);
        let count = columns
            .checked_mul(height.div_ceil(tile))
            .filter(|&n| n <= 16 * 1024 * 1024)
            .context("FITS tile bookkeeping exceeds 16 MiB; increase tile size")?;
        let mut written = Vec::new();
        written
            .try_reserve_exact(count)
            .context("allocating FITS tile bookkeeping")?;
        written.resize(count, 0);
        let mut row = Vec::new();
        row.try_reserve_exact(tile.min(width) * 4)
            .context("allocating FITS row")?;
        ensure!(
            writer.seek(SeekFrom::End(0))? == 0,
            "FITS staging stream must be empty"
        );
        writer.write_all(&header)?;
        Ok(Self {
            writer,
            width,
            height,
            tile,
            columns,
            written,
            remaining: count,
            row,
            payload_end,
            padding,
            header_len,
        })
    }

    pub fn write_tile(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        data: &[f32],
    ) -> Result<()> {
        ensure!(
            x < self.width && y < self.height,
            "FITS tile origin out of bounds"
        );
        ensure!(
            x % self.tile == 0 && y % self.tile == 0,
            "FITS tile origin is not on the tile grid"
        );
        ensure!(
            width == self.tile.min(self.width - x) && height == self.tile.min(self.height - y),
            "FITS tile dimensions do not match the tile grid"
        );
        ensure!(
            width.checked_mul(height) == Some(data.len()),
            "FITS tile sample count mismatch"
        );
        let index = (y / self.tile) * self.columns + x / self.tile;
        ensure!(self.written[index] == 0, "FITS tile already written");
        for dy in 0..height {
            self.row.clear();
            for &value in &data[dy * width..(dy + 1) * width] {
                self.row.extend_from_slice(&value.to_be_bytes());
            }
            let offset = ((self.height - 1 - y - dy) as u64)
                .checked_mul(self.width as u64)
                .and_then(|n| n.checked_add(x as u64))
                .and_then(|n| n.checked_mul(4))
                .and_then(|n| n.checked_add(self.header_len))
                .context("FITS tile offset overflow")?;
            self.writer.seek(SeekFrom::Start(offset))?;
            self.writer.write_all(&self.row)?;
        }
        self.written[index] = 1;
        self.remaining -= 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<W> {
        ensure!(
            self.remaining == 0,
            "FITS output is missing {} tiles",
            self.remaining
        );
        self.writer.seek(SeekFrom::Start(self.payload_end))?;
        self.writer.write_all(&[0; 2880][..self.padding])?;
        self.writer.flush()?;
        Ok(self.writer)
    }
}

/// Write a mono Float32 FITS from bounded tiles in top-down sensor coordinates.
/// The callback fills exactly `width * height` samples, preserving negative,
/// over-range and missing (NaN) values. Memory is one tile plus one row and the
/// writer's bounded tile coverage bookkeeping.
/// Seek offsets are checked u64 values; this has no classic-TIFF 4 GiB limit.
/// The caller owns publication: use a new staging stream and publish only after
/// success. On any callback or I/O error the stream is an incomplete artifact.
pub fn write_mono_fits_tiles<W, F>(
    writer: &mut W,
    width: usize,
    height: usize,
    tile: usize,
    mut fill: F,
) -> Result<()>
where
    W: Write + Seek,
    F: FnMut(usize, usize, usize, usize, &mut [f32]) -> Result<()>,
{
    let mut output = MonoFitsTileWriter::new(writer, width, height, tile)?;
    let mut samples = vec![f32::NAN; tile.min(width) * tile.min(height)];
    for y in (0..height).step_by(tile) {
        let th = tile.min(height - y);
        for x in (0..width).step_by(tile) {
            let tw = tile.min(width - x);
            let data = &mut samples[..tw * th];
            data.fill(f32::NAN);
            fill(x, y, tw, th, data)?;
            output.write_tile(x, y, tw, th, data)?;
        }
    }
    output.finish()?;
    Ok(())
}

/// XISF 1.0 with one uncompressed Float32 image and an aligned data attachment.
pub fn write_xisf(path: &Path, rgb: &[Plane<f32>; 3], channels: usize) -> Result<()> {
    let (w, h) = dimensions(rgb, channels)?;
    let bytes = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(channels * 4))
        .context("image size overflow")?;
    let xml=format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?><xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\"><Image geometry=\"{w}:{h}:{channels}\" sampleFormat=\"Float32\" colorSpace=\"{}\" pixelStorage=\"Planar\" byteOrder=\"little\" bounds=\"0:1\" location=\"attachment:4096:{bytes}\"/></xisf>",if channels==1 {"Gray"} else {"RGB"});
    ensure!(
        xml.len() + 16 <= 4096,
        "XISF header exceeds attachment offset"
    );
    let mut f =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    f.write_all(b"XISF0100")?;
    f.write_all(&(xml.len() as u32).to_le_bytes())?;
    f.write_all(&[0; 4])?;
    f.write_all(xml.as_bytes())?;
    f.write_all(&vec![0; 4096 - 16 - xml.len()])?;
    for p in &rgb[..channels] {
        for &v in &p.data {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

/// Additional scientific masters alongside the rendered TIFF.
pub fn write_scientific_copies(
    output: &Path,
    rgb: &[Plane<f32>; 3],
    channels: usize,
    fits: bool,
    xisf: bool,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if fits {
        let p = output.with_extension("fits");
        write_fits(&p, rgb, channels)?;
        paths.push(p);
    }
    if xisf {
        let p = output.with_extension("xisf");
        write_xisf(&p, rgb, channels)?;
        paths.push(p);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stateful_tiles_support_out_of_order_and_reject_invalid_submissions() {
        let mut output =
            MonoFitsTileWriter::new(std::io::Cursor::new(Vec::new()), 3, 3, 2).unwrap();
        for (x, y, w, h, samples) in [
            (1, 0, 2, 2, vec![0.; 4]),
            (0, 0, 1, 2, vec![0.; 2]),
            (0, 0, 2, 2, vec![0.; 3]),
            (3, 0, 1, 2, vec![0.; 2]),
            (0, usize::MAX, 2, 2, vec![0.; 4]),
            (0, 0, usize::MAX, usize::MAX, vec![]),
        ] {
            assert!(output.write_tile(x, y, w, h, &samples).is_err());
        }
        output.write_tile(2, 2, 1, 1, &[8.]).unwrap();
        assert!(output
            .write_tile(2, 2, 1, 1, &[80.])
            .unwrap_err()
            .to_string()
            .contains("already written"));
        output.write_tile(0, 2, 2, 1, &[6., 7.]).unwrap();
        output.write_tile(2, 0, 1, 2, &[2., 5.]).unwrap();
        output.write_tile(0, 0, 2, 2, &[0., 1., 3., 4.]).unwrap();
        let bytes = output.finish().unwrap().into_inner();
        let mut expected = std::io::Cursor::new(Vec::new());
        write_mono_fits_tiles(&mut expected, 3, 3, 2, |x, y, w, h, data| {
            for dy in 0..h {
                for dx in 0..w {
                    data[dy * w + dx] = ((y + dy) * 3 + x + dx) as f32;
                }
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(bytes, expected.into_inner());
    }

    #[test]
    fn stateful_tiles_require_complete_coverage_and_bounded_bookkeeping() {
        let mut stream = std::io::Cursor::new(Vec::new());
        let mut output = MonoFitsTileWriter::new(&mut stream, 3, 3, 2).unwrap();
        output.write_tile(0, 0, 2, 2, &[0.; 4]).unwrap();
        assert!(output
            .finish()
            .unwrap_err()
            .to_string()
            .contains("missing 3 tiles"));
        let mut stream = std::io::Cursor::new(Vec::new());
        assert!(MonoFitsTileWriter::new(&mut stream, 4097, 4096, 1).is_err());
        assert!(stream.into_inner().is_empty());
        let mut stream = std::io::Cursor::new(vec![42]);
        assert!(MonoFitsTileWriter::new(&mut stream, 3, 3, 2).is_err());
        assert_eq!(stream.into_inner(), vec![42]);
    }

    #[test]
    fn tiled_mono_fits_is_byte_identical_at_every_tile_boundary() {
        let mut pixels: Vec<f32> = (0..19 * 13).map(|i| i as f32 / 20. - 2.).collect();
        pixels[0] = f32::from_bits(0x7fc01234);
        pixels[18] = -0.0;
        pixels[19] = f32::INFINITY;
        let planes = [
            Plane::from_vec(19, 13, pixels.clone()),
            Plane::new(0, 0),
            Plane::new(0, 0),
        ];
        let path =
            std::env::temp_dir().join(format!("smokstak-tiled-fits-{}.fits", std::process::id()));
        write_fits(&path, &planes, 1).unwrap();
        let expected = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        for tile in [1, 3, 7, 19, 64] {
            let mut out = std::io::Cursor::new(Vec::new());
            write_mono_fits_tiles(&mut out, 19, 13, tile, |x, y, w, h, data| {
                for dy in 0..h {
                    data[dy * w..(dy + 1) * w]
                        .copy_from_slice(&pixels[(y + dy) * 19 + x..(y + dy) * 19 + x + w]);
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(out.into_inner(), expected, "tile {tile}");
        }
    }

    #[test]
    fn tiled_fits_refuses_bad_sizes_existing_streams_and_propagates_failures() {
        for (w, h, tile) in [
            (0, 10, 2),
            (10, 0, 2),
            (10, 10, 0),
            (10, 10, 4097),
            (usize::MAX, usize::MAX, 64),
        ] {
            let mut out = std::io::Cursor::new(Vec::new());
            assert!(
                write_mono_fits_tiles(&mut out, w, h, tile, |_, _, _, _, _| panic!(
                    "invalid geometry must not read tiles"
                ))
                .is_err()
            );
            assert!(out.into_inner().is_empty());
        }
        let mut out = std::io::Cursor::new(vec![42]);
        assert!(write_mono_fits_tiles(&mut out, 10, 10, 4, |_, _, _, _, _| Ok(())).is_err());
        assert_eq!(out.into_inner(), vec![42]);
        let mut out = std::io::Cursor::new(Vec::new());
        assert!(
            write_mono_fits_tiles(&mut out, 10, 10, 4, |_, _, _, _, _| anyhow::bail!(
                "source unavailable"
            ))
            .unwrap_err()
            .to_string()
            .contains("source unavailable")
        );
    }

    #[test]
    fn tiled_fits_seeks_beyond_four_gib_without_allocating_the_image() {
        struct Probe {
            position: u64,
            large_write: Option<u64>,
        }
        impl Write for Probe {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.position > u32::MAX as u64 {
                    self.large_write = Some(self.position);
                    return Err(std::io::Error::other("stop after observing large offset"));
                }
                self.position += bytes.len() as u64;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl Seek for Probe {
            fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
                self.position = match p {
                    SeekFrom::Start(n) => n,
                    SeekFrom::End(0) => 0,
                    _ => panic!("unexpected seek"),
                };
                Ok(self.position)
            }
        }
        let mut probe = Probe {
            position: 0,
            large_write: None,
        };
        let err = write_mono_fits_tiles(&mut probe, 65536, 17000, 16, |_, _, _, _, data| {
            data.fill(0.5);
            Ok(())
        })
        .unwrap_err();
        assert!(err.to_string().contains("large offset"));
        assert_eq!(probe.large_write, Some(2880 + 16999u64 * 65536 * 4));
    }

    #[test]
    fn float_masters_preserve_channels_orientation_and_out_of_range_values() {
        for channels in [1, 3] {
            let rgb = std::array::from_fn(|c| Plane {
                width: 16,
                height: 18,
                data: (0..288)
                    .map(|i| i as f32 / 100.0 - 0.25 + c as f32)
                    .collect(),
            });
            let base = std::env::temp_dir()
                .join(format!("smokstak-export-{}-{channels}", std::process::id()));
            let fit = base.with_extension("fits");
            write_fits(&fit, &rgb, channels).unwrap();
            let bytes = std::fs::read(&fit).unwrap();
            assert_eq!(bytes.len() % 2880, 0);
            assert!(String::from_utf8_lossy(&bytes[..2880]).contains("BOTTOM-UP"));
            let expected: Vec<_> = rgb[..channels]
                .iter()
                .flat_map(|p| p.data.iter().copied())
                .collect();
            let decoded: Vec<_> = bytes[2880..2880 + expected.len() * 4]
                .chunks_exact(4)
                .map(|b| f32::from_be_bytes(b.try_into().unwrap()))
                .collect();
            let fits_expected: Vec<_> = rgb[..channels]
                .iter()
                .flat_map(|p| p.data.chunks_exact(16).rev().flatten().copied())
                .collect();
            assert_eq!(decoded, fits_expected);
            let (header, offset) = sr_raw::fits::read_header(&fit).unwrap();
            assert_eq!(offset, 2880);
            assert_eq!(header.int("BITPIX"), Some(-32));
            assert_eq!(header.int("NAXIS"), Some(if channels == 1 { 2 } else { 3 }));
            let xisf = base.with_extension("xisf");
            write_xisf(&xisf, &rgb, channels).unwrap();
            let payload = std::fs::read(&xisf).unwrap();
            let decoded: Vec<_> = payload[4096..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(decoded, expected, "export must not clip scientific values");
            // The raw ingestion reader normalizes/clamps to nominal bounds;
            // validate its geometry, while the payload above checks exact samples.
            let (header, planes) = sr_raw::xisf::read_planes(&xisf).unwrap();
            assert_eq!(header.channels, channels);
            for c in 0..channels {
                assert_eq!(
                    planes[c].data,
                    rgb[c]
                        .data
                        .iter()
                        .map(|v| v.clamp(0.0, 1.0))
                        .collect::<Vec<_>>()
                );
                assert_eq!(planes[c].dims(), (16, 18));
            }
            std::fs::remove_file(fit).unwrap();
            std::fs::remove_file(xisf).unwrap();
        }
    }
}
