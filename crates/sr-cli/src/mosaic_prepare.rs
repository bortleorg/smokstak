//! Automatic, bounded mono mosaic preparation. Input pixels are never modified.
use crate::mosaic::{FrameSpec, Grid, Plan};
use crate::mosaic_prepare_geometry::Catalog;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sr_core::{
    CfaPattern, DefectMask, FrameMetadata, NoiseModel, NoiseSource, RawFrame, SamplePlane,
    star::Star,
};
use sr_raw::window::{FitsWindowReader, SampleUnits};
use std::{
    collections::HashSet,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

const MAX_FRAMES: usize = 512;
const MAX_STARS: usize = 6000;
const CATALOG_VERSION: u32 = 2;

#[derive(Serialize, Deserialize)]
struct CachedCatalog {
    version: u32,
    sha256: String,
    width: usize,
    height: usize,
    stars: Vec<[f32; 3]>,
}

/// Copy `path` into `stage` as `<sha256>.fits` while fingerprinting it, so a
/// network source is read once, sequentially, and everything after works on
/// local disk. An existing staged copy is verified, not trusted by name.
fn stage_copy(path: &Path, stage: &Path) -> Result<(String, PathBuf)> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let partial = stage.join(format!(
        ".{}-{}.partial",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let copied = (|| -> Result<String> {
        let mut source = fs::File::open(path)?;
        let mut target = std::io::BufWriter::new(fs::File::create_new(&partial)?);
        let mut buffer = vec![0u8; 4 * 1024 * 1024];
        let mut hash = Sha256::new();
        loop {
            let count = source.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
            std::io::Write::write_all(&mut target, &buffer[..count])?;
        }
        target
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        Ok(format!("{:x}", hash.finalize()))
    })();
    let hash = match copied {
        Ok(hash) => hash,
        Err(e) => {
            let _ = fs::remove_file(&partial);
            return Err(e.context(format!(
                "Staging {} into {}",
                path.display(),
                stage.display()
            )));
        }
    };
    let staged = stage.join(format!("{hash}.fits"));
    if staged.exists() && fingerprint(&staged).ok().as_deref() == Some(hash.as_str()) {
        fs::remove_file(&partial)?;
    } else {
        fs::rename(&partial, &staged)?;
    }
    Ok((hash, staged))
}

fn fingerprint(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hash = Sha256::new();
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Folders include panel subfolders. Symbolic directory links are not followed.
/// All supplied paths must resolve; no unreadable exposure is silently dropped.
pub(crate) fn collect(input: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if input.is_dir() {
        let mut pending = vec![input.to_path_buf()];
        let mut directories = 0;
        while let Some(directory) = pending.pop() {
            directories += 1;
            ensure!(
                directories <= 10000,
                "Too many input subfolders; choose the exposure folder more specifically"
            );
            for entry in fs::read_dir(&directory)? {
                let entry = entry?;
                let kind = entry.file_type()?;
                if kind.is_dir() {
                    ensure!(
                        directories + pending.len() < 10000,
                        "Too many input subfolders; choose a more specific folder"
                    );
                    pending.push(entry.path());
                } else if kind.is_file() && sr_raw::format_of(&entry.path()).is_some() {
                    paths.push(entry.path());
                    ensure!(
                        paths.len() <= MAX_FRAMES,
                        "Automatic mosaics currently support at most {MAX_FRAMES} exposures"
                    );
                }
            }
        }
    } else {
        ensure!(
            fs::metadata(input)?.len() <= 4 * 1024 * 1024 || sr_raw::format_of(input).is_some(),
            "Input list exceeds 4 MiB"
        );
        if sr_raw::format_of(input).is_some() {
            paths.push(input.to_path_buf());
        } else {
            let base = input.parent().unwrap_or(Path::new("."));
            for (line_number, line) in fs::read_to_string(input)?.lines().enumerate() {
                let line = line
                    .trim()
                    .trim_start_matches('\u{feff}')
                    .trim()
                    .trim_matches('"');
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let raw = Path::new(line);
                let path = if raw.is_absolute() {
                    raw.to_owned()
                } else if base.join(raw).is_file() {
                    base.join(raw)
                } else {
                    raw.to_owned()
                };
                ensure!(
                    path.is_file() && sr_raw::format_of(&path).is_some(),
                    "{}, line {}: not a readable exposure: {}",
                    input.display(),
                    line_number + 1,
                    path.display()
                );
                paths.push(path);
                ensure!(
                    paths.len() <= MAX_FRAMES,
                    "Automatic mosaics support at most {MAX_FRAMES} exposures"
                );
            }
        }
    }
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for path in paths {
        let path =
            fs::canonicalize(&path).with_context(|| format!("Cannot read {}", path.display()))?;
        if seen.insert(path.clone()) {
            result.push(path);
        }
    }
    result.sort();
    ensure!(
        (2..=MAX_FRAMES).contains(&result.len()),
        "Choose 2–{MAX_FRAMES} mono Hα exposures (found {})",
        result.len()
    );
    Ok(result)
}

/// Native-resolution detection in fixed windows. Each core owns its detections;
/// an even 32-pixel halo preserves the detector's 2x2 lattice at window edges.
fn detect(reader: &mut FitsWindowReader, noise: NoiseModel) -> Result<Vec<Star>> {
    let (width, height) = reader.dimensions();
    let mut stars = Vec::new();
    for y in (0..height).step_by(1024) {
        for x in (0..width).step_by(1024) {
            let left = x.saturating_sub(32);
            let top = y.saturating_sub(32);
            let right = (x + 1056).min(width);
            let bottom = (y + 1056).min(height);
            let (w, h) = (right - left, bottom - top);
            let pixels = reader.read_rect(left, top, w, h)?;
            let frame = RawFrame {
                width: w,
                height: h,
                samples: SamplePlane::from_normalised(w, h, pixels),
                cfa: CfaPattern::MONO,
                defects: DefectMask::none(w, h),
                noise,
                metadata: FrameMetadata::default(),
            };
            for mut star in sr_quality::stars::positions(&frame, MAX_STARS) {
                star.x += left as f32;
                star.y += top as f32;
                if star.x >= x as f32
                    && star.x < (x + 1024).min(width) as f32
                    && star.y >= y as f32
                    && star.y < (y + 1024).min(height) as f32
                {
                    stars.push(star);
                }
            }
            stars.sort_unstable_by(|a, b| {
                b.flux
                    .total_cmp(&a.flux)
                    .then(a.y.total_cmp(&b.y))
                    .then(a.x.total_cmp(&b.x))
            });
            stars.truncate(MAX_STARS);
        }
    }
    ensure!(
        stars.len() >= 100,
        "Only {} usable stars detected; at least 100 are needed for independent alignment checks",
        stars.len()
    );
    Ok(stars)
}

fn load_catalog(
    original: &Path,
    cache: &Path,
    stage: Option<&Path>,
) -> Result<(Catalog, String, u64, String)> {
    let before = fs::metadata(original)?;
    let (hash, staged) = match stage {
        Some(stage) => {
            let (hash, staged) = stage_copy(original, stage)?;
            (hash, Some(staged))
        }
        None => (fingerprint(original)?, None),
    };
    let after = fs::metadata(original)?;
    ensure!(
        before.len() == after.len() && before.modified()? == after.modified()?,
        "Source changed during preparation: {}",
        original.display()
    );
    let path = staged.as_deref().unwrap_or(original);
    let (header, _) = sr_raw::fits::read_header(path)?;
    ensure!(
        header.int("BITPIX") == Some(16),
        "{}: first-pass automatic mosaics require uncompressed 16-bit mono FITS",
        path.display()
    );
    let filter = sr_raw::peek_filter(path).context("Every exposure must identify its Hα filter")?;
    let normalized = filter
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-', '_'], "");
    ensure!(
        ["h", "ha", "halpha", "hα"].contains(&normalized.as_str()),
        "{}: expected Hα; found filter {filter}",
        path.display()
    );
    let mut reader = FitsWindowReader::open(path, &sr_raw::ReadOptions::default())?;
    ensure!(
        matches!(reader.sample_units(), SampleUnits::NormalizedInteger { .. }),
        "Unsupported source units"
    );
    let (width, height) = reader.dimensions();
    let cached_path = cache.join(format!("catalog-v{CATALOG_VERSION}-{hash}.json"));
    let cached = fs::File::open(&cached_path)
        .ok()
        .and_then(|f| {
            let mut bytes = Vec::new();
            f.take(2 * 1024 * 1024 + 1).read_to_end(&mut bytes).ok()?;
            (bytes.len() <= 2 * 1024 * 1024).then_some(bytes)
        })
        .and_then(|bytes| serde_json::from_slice::<CachedCatalog>(&bytes).ok())
        .filter(|c| {
            c.version == CATALOG_VERSION
                && c.sha256 == hash
                && c.width == width
                && c.height == height
                && (100..=MAX_STARS).contains(&c.stars.len())
                && c.stars.iter().all(|s| {
                    s.iter().all(|v| v.is_finite())
                        && s[0] >= 0.0
                        && s[1] >= 0.0
                        && s[0] < width as f32
                        && s[1] < height as f32
                        && s[2] > 0.0
                })
        });
    let stars = if let Some(c) = cached {
        c.stars
            .into_iter()
            .map(|s| Star {
                x: s[0],
                y: s[1],
                flux: s[2],
            })
            .collect()
    } else {
        let SampleUnits::NormalizedInteger { white_level } = reader.sample_units() else {
            unreachable!()
        };
        let step = 65536.0 - white_level;
        let noise = match header
            .number("EGAIN")
            .filter(|v| v.is_finite() && *v > 1e-3)
        {
            Some(gain) => NoiseModel::new(
                step / (gain as f32 * white_level),
                (3.0 * step / white_level).powi(2),
                NoiseSource::Nominal,
            ),
            None => NoiseModel::nominal(header.number("GAIN").unwrap_or(100.0) as f32, white_level),
        };
        let stars = detect(&mut reader, noise)
            .with_context(|| format!("Detecting stars in {}", path.display()))?;
        let value = CachedCatalog {
            version: CATALOG_VERSION,
            sha256: hash.clone(),
            width,
            height,
            stars: stars.iter().map(|s| [s.x, s.y, s.flux]).collect(),
        };
        // Cache is an optimization; interrupted/invalid entries are recomputed.
        let _ = fs::write(&cached_path, serde_json::to_vec(&value)?);
        stars
    };
    let after = fs::metadata(path)?;
    ensure!(
        staged.is_some()
            || (before.len() == after.len() && before.modified()? == after.modified()?),
        "Source changed during preparation: {}",
        path.display()
    );
    let focal = header
        .number("FOCALLEN")
        .filter(|v| v.is_finite() && *v > 0.0);
    let pitch = header
        .any_number(&["XPIXSZ", "PIXSIZE1", "PIXSIZE"])
        .filter(|v| v.is_finite() && *v > 0.0);
    let pixel_scale = focal.zip(pitch).map(|(f, p)| 206.265 * p / f);
    let optical_id = format!(
        "{}|{}|{:?}|{:?}|{width}x{height}",
        header
            .any_text(&["TELESCOP", "TELESCOPE"])
            .unwrap_or_default(),
        header.any_text(&["INSTRUME", "CAMERA"]).unwrap_or_default(),
        focal,
        pitch
    );
    Ok((
        Catalog {
            path: path.to_owned(),
            width,
            height,
            stars,
            pixel_scale,
            optical_id,
        },
        hash,
        before.len(),
        filter,
    ))
}

fn canvas(frames: &[FrameSpec]) -> Result<Grid> {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for frame in frames {
        frame
            .projection
            .to_warp_with_node_budget(frame.width, frame.height, 0.025, 1_000_000)?;
        let bounds =
            crate::mosaic_prepare_photometry::bounds(&frame.projection, frame.width, frame.height)?;
        for axis in 0..2 {
            lo[axis] = lo[axis].min(bounds[axis]);
            hi[axis] = hi[axis].max(bounds[axis + 2]);
        }
    }
    let origin = [(lo[0] - 2.0).floor(), (lo[1] - 2.0).floor()];
    let size = [
        (hi[0] + 2.0).ceil() - origin[0] + 1.0,
        (hi[1] + 2.0).ceil() - origin[1] + 1.0,
    ];
    ensure!(
        origin
            .iter()
            .chain(size.iter())
            .all(|v| v.is_finite() && v.abs() < 8_000_000.0),
        "Mosaic footprint is implausibly large; registration needs review"
    );
    Ok(Grid {
        origin: origin.map(|v| v as f32),
        width: size[0] as usize,
        height: size[1] as usize,
        scale: 1.0,
    })
}

pub(crate) fn run(
    input: &Path,
    output: &Path,
    memory_mb: usize,
    cache_dir: Option<&Path>,
    stage_dir: Option<&Path>,
) -> Result<()> {
    ensure!(
        (128..=65536).contains(&memory_mb),
        "Preparation memory budget must be 128–65536 MiB"
    );
    ensure!(
        !output.exists(),
        "Preparation output already exists; choose a new folder"
    );
    let paths = collect(input)?;
    // Catalogs, pair correspondences and a representative bundle are bounded;
    // raster windows never scale with the full image or output canvas.
    ensure!(
        memory_mb * 1024 * 1024 >= 64 * 1024 * 1024 + paths.len() * MAX_STARS * 64,
        "{} exposures need a preparation budget of at least {} MiB",
        paths.len(),
        64 + (paths.len() * MAX_STARS * 64).div_ceil(1024 * 1024)
    );
    fs::create_dir_all(output.parent().unwrap_or(Path::new(".")))?;
    fs::create_dir(output)?;
    let cache = cache_dir.map(Path::to_owned).unwrap_or_else(|| {
        output
            .parent()
            .unwrap_or(Path::new("."))
            .join(".smokstak-mosaic-cache")
    });
    fs::create_dir_all(&cache)?;
    if let Some(stage) = stage_dir {
        fs::create_dir_all(stage)?;
    }
    let result = (|| -> Result<()> {
        let mut catalogs = Vec::new();
        let mut sources = Vec::new();
        let mut filter = None;
        let mut hashes = HashSet::new();
        for (i, path) in paths.iter().enumerate() {
            eprintln!(
                "mosaic prepare catalog {}/{}: {}",
                i + 1,
                paths.len(),
                path.display()
            );
            let (catalog, hash, bytes, name) = load_catalog(path, &cache, stage_dir)?;
            eprintln!(
                "mosaic prepare catalog: {} usable stars",
                catalog.stars.len()
            );
            ensure!(
                hashes.insert(hash.clone()),
                "Identical exposure selected more than once: {}; remove the duplicate copy",
                path.display()
            );
            if let Some(ref filter) = filter {
                ensure!(
                    *filter == name,
                    "Filter labels differ ({filter} / {name}); select one consistent Hα set"
                );
            } else {
                filter = Some(name);
            }
            sources.push((hash, bytes));
            catalogs.push(catalog);
        }
        eprintln!("mosaic prepare geometry: finding panels and registering overlaps");
        let geometry = crate::mosaic_prepare_geometry::prepare(&catalogs, |s| {
            eprintln!("mosaic prepare geometry: {s}")
        })?;
        let mut frames: Vec<_> = catalogs
            .iter()
            .zip(&geometry.frames)
            .zip(&sources)
            .zip(&paths)
            .map(|(((c, g), (hash, bytes)), original)| FrameSpec {
                path: c.path.clone(),
                label: original
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                group: format!("Panel / field {}", g.group + 1),
                width: c.width,
                height: c.height,
                bytes: *bytes,
                sha256: hash.clone(),
                projection: g.projection.clone(),
                noise: NoiseModel::new(0.0, 0.0, NoiseSource::Measured),
                sky: 0.0,
                gain: 1.0,
                relative_log_gain: None,
                offset: 0.0,
                background_plane: [0.0; 2],
                background_quadratic: [0.0; 3],
                weight: 1.0,
                psf_hfd: 0.0,
                registration_p50: g.registration_p50 as f32,
                registration_p90: g.registration_p90 as f32,
                validation_stars: g.validation_stars,
            })
            .collect();
        let grid = canvas(&frames)?;
        // Reproducible evidence for a failed photometry run, deliberately not a
        // buildable Plan: only plan.json publishes a completed preparation.
        let mut diagnostic = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("registration.json"))?;
        serde_json::to_writer(
            &mut diagnostic,
            &serde_json::json!({
                "frames": &frames, "pairs": &geometry.pairs, "anchor": geometry.anchor,
            }),
        )?;
        diagnostic.sync_all()?;
        drop(diagnostic);
        let stars = catalogs
            .iter()
            .map(|c| {
                c.stars
                    .iter()
                    .map(|s| [s.x as f64, s.y as f64, s.flux as f64])
                    .collect()
            })
            .collect::<Vec<_>>();
        drop(catalogs);
        let mut notes = geometry.notes;
        if let Some(stage) = stage_dir {
            notes.push(format!(
                "Sources were copied into {} before preparation, verified by SHA-256; this plan reads the copies. Labels keep the original file names.",
                stage.display()
            ));
        }
        eprintln!("mosaic prepare photometry: measuring stars and shared sky");
        notes.extend(crate::mosaic_prepare_photometry::prepare(
            &mut frames,
            &geometry.pairs,
            geometry.anchor,
            memory_mb,
            &stars,
        )?);
        notes.push("Automatically prepared from every selected exposure. Relative registration only; no absolute astrometric WCS. Detector calibration is unknown and no dark/flat correction is applied.".into());
        let plan = Plan {
            version: 1,
            filter: filter.context("No filter")?,
            calibration: "uncalibrated or unknown; no calibration applied".into(),
            grid,
            frames,
            notes,
        };
        let pending = output.join("plan.pending.json");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)?;
        file.write_all(&serde_json::to_vec_pretty(&plan)?)?;
        file.sync_all()?;
        drop(file);
        crate::mosaic::read_review_plan_fingerprinted(&pending, 256, memory_mb)?;
        fs::rename(&pending, output.join("plan.json"))?;
        println!(
            "Prepared mosaic plan: {}",
            output.join("plan.json").display()
        );
        Ok(())
    })();
    if let Err(ref error) = result {
        let _ = fs::write(
            output.join("FAILED.txt"),
            format!(
                "{error:#}\nNo completed plan was published. Input exposures were not changed.\n"
            ),
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staged_copies_are_exact_named_by_content_and_reused() {
        let root = std::env::temp_dir().join(format!("mosaic-stage-test-{}", std::process::id()));
        let stage = root.join("stage");
        fs::create_dir_all(&stage).unwrap();
        let source = root.join("frame.fits");
        let bytes: Vec<u8> = (0..5_000_000u32).map(|i| (i * 7 % 251) as u8).collect();
        fs::write(&source, &bytes).unwrap();
        let (hash, staged) = stage_copy(&source, &stage).unwrap();
        assert_eq!(hash, fingerprint(&source).unwrap());
        assert_eq!(staged, stage.join(format!("{hash}.fits")));
        assert!(fs::read(&staged).unwrap() == bytes);
        // A damaged copy under the right name is replaced, not trusted.
        fs::write(&staged, b"damaged").unwrap();
        assert_eq!(
            stage_copy(&source, &stage).unwrap(),
            (hash.clone(), staged.clone())
        );
        assert!(fs::read(&staged).unwrap() == bytes);
        let (again, _) = stage_copy(&source, &stage).unwrap();
        assert_eq!(again, hash);
        let names: Vec<_> = fs::read_dir(&stage)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 1, "no partial copies left behind: {names:?}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn preparation_rejects_existing_output_without_writes() {
        let root = std::env::temp_dir();
        assert!(
            run(Path::new("missing"), &root, 2048, None, None)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
    }
    #[test]
    fn invalid_budget_fails_before_input_access() {
        assert!(
            run(
                Path::new("missing"),
                Path::new("missing-output"),
                0,
                None,
                None
            )
            .unwrap_err()
            .to_string()
            .contains("budget")
        );
    }
    #[test]
    fn folder_recursion_and_explicit_bom_list_have_distinct_membership() {
        let root = std::env::temp_dir().join(format!(
            "mosaic-input-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("panel")).unwrap();
        for name in ["a.fits", "b.fits", "panel/c.fit"] {
            fs::write(root.join(name), []).unwrap();
        }
        let list = root.join("selected.txt");
        fs::write(&list, "\u{feff}\"a.fits\"\n\"b.fits\"\n").unwrap();
        assert_eq!(collect(&root).unwrap().len(), 3);
        let selected = collect(&list).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|p| p.file_name().unwrap() != "c.fit"));
        for name in ["a.fits", "b.fits", "panel/c.fit", "selected.txt"] {
            fs::remove_file(root.join(name)).unwrap();
        }
        fs::remove_dir(root.join("panel")).unwrap();
        fs::remove_dir(root).unwrap();
    }
}
