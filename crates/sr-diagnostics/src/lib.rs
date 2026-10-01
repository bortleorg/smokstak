//! Run manifests, per-frame tables and diagnostic rasters.
//!
//! The point of this crate is that a user can answer, after the fact: which
//! frames were used, where was the output rejected, was 2x actually supported,
//! and would this run reproduce? A reconstruction that cannot be interrogated
//! is not a measurement, it is a picture.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use sr_core::config::ReconstructionConfig;
use sr_core::frame::{FrameQuality, RawFrame};
use sr_core::plane::Plane;
use sr_core::product::{ProductStats, ReconstructionProduct, SamplingCoverage};

/// Everything needed to reproduce a run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunManifest {
    pub program: String,
    pub version: String,
    pub build_profile: String,
    /// Source revision if the build recorded one.
    pub revision: String,
    pub started_utc: String,
    pub host: HostInfo,
    pub dependencies: BTreeMap<String, String>,

    pub input: String,
    /// Files in the exact order they were processed, with a content hash.
    pub sources: Vec<SourceFile>,
    pub reference_index: usize,
    pub reference_file: String,
    pub reference_reason: String,
    /// Brightness reference in `sources`; may differ from the output-grid
    /// reference in split-filter runs. None when matching is disabled or an
    /// older manifest did not record it.
    #[serde(default)]
    pub photometric_reference_index: Option<usize>,

    pub config: ReconstructionConfig,
    pub seed: u64,
    pub noise: NoiseRecord,
    pub color: ColorRecord,

    pub output_width: usize,
    pub output_height: usize,
    pub output_files: Vec<String>,

    pub coverage: Option<SamplingCoverage>,
    pub stats: ProductStats,
    pub timings_ms: BTreeMap<String, u128>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostInfo {
    pub os: String,
    pub arch: String,
    pub threads: usize,
    pub compute: String,
}

impl Default for HostInfo {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            threads: rayon::current_num_threads(),
            compute: "cpu".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceFile {
    pub index: usize,
    pub path: String,
    pub hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoiseRecord {
    pub alpha: f32,
    pub beta: f32,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColorRecord {
    pub wb: [f32; 3],
    pub cam_to_srgb: [[f32; 3]; 3],
    pub matrix_fallback: bool,
    pub exposure_gain: f32,
}

/// Collects timings without threading a stopwatch through every call site.
#[derive(Clone, Default)]
pub struct Timings {
    entries: BTreeMap<String, u128>,
}

impl Timings {
    pub fn record(&mut self, name: &str, d: Duration) {
        *self.entries.entry(name.to_string()).or_insert(0) += d.as_millis();
    }
    pub fn into_map(self) -> BTreeMap<String, u128> {
        self.entries
    }
}

pub fn dependency_versions() -> BTreeMap<String, String> {
    // Recorded from the lockfile at build time would be better; these are the
    // majors this workspace is written against.
    let mut m = BTreeMap::new();
    m.insert("rawler".into(), "0.8".into());
    m.insert("rustfft".into(), "6".into());
    m.insert("tiff".into(), "0.11".into());
    m.insert("rayon".into(), "1".into());
    m.insert("rustc".into(), option_env!("SRSTACK_RUSTC").unwrap_or("unknown").into());
    m
}

pub fn write_manifest(path: &Path, manifest: &RunManifest) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Per-frame table: everything that was decided about each frame.
#[derive(Clone, Debug, Serialize)]
pub struct FrameRow {
    pub index: usize,
    pub file: String,
    pub used: bool,
    pub sharpness: f32,
    pub contrast: f32,
    pub estimated_blur: f32,
    pub blur_anisotropy: f32,
    pub saturation_fraction: f32,
    /// Point-source shape, where the frame had point sources: half-flux
    /// diameter in sensor pixels, eccentricity, and how many were measured.
    /// A count of zero means the frame carried none and the columns say
    /// nothing.
    pub star_hfd: f32,
    pub star_eccentricity: f32,
    pub star_count: usize,
    pub exposure_scale: f32,
    /// Green-channel gain and pedestal of the photometric map onto the
    /// reference, and how they were arrived at. Green because it is the
    /// channel every sensor samples twice and the one a reader compares frames
    /// by; all three channels are in the run manifest.
    pub photometric_gain: f32,
    pub photometric_offset: f32,
    pub photometric_source: String,
    pub transform_model: String,
    pub shift_x: f32,
    pub shift_y: f32,
    pub rotation_deg: f32,
    pub scale: f32,
    pub residual_rms: f32,
    pub residual_p90: f32,
    pub inliers: usize,
    pub probes: usize,
    pub overlap: f32,
    pub confidence: f32,
    pub local_warp_mean: f32,
    pub local_warp_max: f32,
    pub robustness_rejected: f32,
}

pub fn write_frames_csv(path: &Path, rows: &[FrameRow]) -> Result<()> {
    let mut s = String::from(
        "index,file,used,sharpness,contrast,estimated_blur,blur_anisotropy,saturation_fraction,\
         star_hfd,star_eccentricity,star_count,\
         exposure_scale,photometric_gain,photometric_offset,photometric_source,\
         transform_model,shift_x,shift_y,rotation_deg,scale,residual_rms,\
         residual_p90,inliers,probes,overlap,confidence,local_warp_mean,local_warp_max,\
         robustness_rejected\n",
    );
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{:.5},{:.6},{:.4},{:.4},{:.6},{:.4},{:.4},{},{:.5},{:.5},{:.6},{},{},{:.5},{:.5},{:.5},{:.7},{:.5},{:.5},{},{},{:.4},{:.4},{:.4},{:.4},{:.5}\n",
            r.index,
            r.file,
            r.used,
            r.sharpness,
            r.contrast,
            r.estimated_blur,
            r.blur_anisotropy,
            r.saturation_fraction,
            r.star_hfd,
            r.star_eccentricity,
            r.star_count,
            r.exposure_scale,
            r.photometric_gain,
            r.photometric_offset,
            r.photometric_source,
            r.transform_model,
            r.shift_x,
            r.shift_y,
            r.rotation_deg,
            r.scale,
            r.residual_rms,
            r.residual_p90,
            r.inliers,
            r.probes,
            r.overlap,
            r.confidence,
            r.local_warp_mean,
            r.local_warp_max,
            r.robustness_rejected,
        ));
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Frame quality table, kept separate so it can be produced by `inspect` too.
pub fn write_quality_csv(path: &Path, frames: &[RawFrame], q: &[FrameQuality]) -> Result<()> {
    let mut s = String::from(
        "index,file,sharpness,laplacian,contrast,estimated_blur,blur_anisotropy,\
         saturation_fraction,mean_level\n",
    );
    for (i, (f, q)) in frames.iter().zip(q).enumerate() {
        s.push_str(&format!(
            "{},{},{:.5},{:.5},{:.6},{:.4},{:.4},{:.6},{:.6}\n",
            i,
            f.metadata.file_name,
            q.sharpness,
            q.laplacian,
            q.contrast,
            q.estimated_blur,
            q.blur_anisotropy,
            q.saturation_fraction,
            q.mean_level
        ));
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Sub-pixel phase histogram as CSV, so it can be plotted without a TIFF
/// viewer.
pub fn write_phase_histogram_csv(path: &Path, cov: &SamplingCoverage) -> Result<()> {
    let mut s = String::from("phase_x_bin,phase_y_bin,samples\n");
    for by in 0..cov.bins {
        for bx in 0..cov.bins {
            s.push_str(&format!("{},{},{}\n", bx, by, cov.phase_histogram[by * cov.bins + bx]));
        }
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Write the standard diagnostic raster bundle for a reconstruction.
pub fn write_product_diagnostics(dir: &Path, product: &ReconstructionProduct) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let mut written = Vec::new();

    let mut put = |name: &str, plane: &Plane<f32>, range: Option<(f32, f32)>| -> Result<()> {
        let p = dir.join(name);
        sr_output::write_diagnostic(&p, plane, range, false)?;
        written.push(p);
        Ok(())
    };

    // A monochrome reconstruction has one channel, and the other two planes are
    // allocated at zero size. Writing them produced a file with no dimensions
    // and failed the run *after* the result had been written, which is the
    // worst place to fail: the work was done and the command still reported an
    // error.
    if product.channels == 1 {
        put("coverage.tif", &product.count[0], None)?;
    } else {
        for (c, name) in ["r-coverage.tif", "g-coverage.tif", "b-coverage.tif"].iter().enumerate() {
            put(name, &product.count[c], None)?;
        }
    }
    put("weight-map.tif", &product.weight[product.channels.min(2) - 1], None)?;
    put("effective-frame-count.tif", &product.effective_frames, None)?;
    put("rejection-count.tif", &product.rejected, None)?;
    Ok(written)
}

/// Human-readable summary of a finished run.
pub fn summarise(product: &ReconstructionProduct, coverage: Option<&SamplingCoverage>) -> String {
    let s = &product.stats;
    let mut out = String::new();
    out.push_str(&format!(
        "Output:            {} x {} px\n",
        product.width, product.height
    ));
    out.push_str(&format!(
        "Samples:           {} examined, {} merged, {} masked, {} rejected by robustness\n",
        s.total_samples, s.accumulated_samples, s.masked_samples, s.rejected_samples
    ));
    let pct = if s.total_samples > 0 {
        100.0 * s.rejected_samples as f64 / s.total_samples as f64
    } else {
        0.0
    };
    out.push_str(&format!("Rejected:          {pct:.3}% of examined samples\n"));
    out.push_str(&format!(
        "Effective frames:  mean {:.1}, minimum {:.1} (per {} px cell)\n",
        s.mean_effective_frames, s.min_effective_frames, product.effective_frames_cell
    ));
    if product.holes_filled > 0 {
        let total = product.width * product.height * product.channels;
        out.push_str(&format!(
            "Unsupported:       {} of {} channel-pixels had no samples and were interpolated ({:.4}%)\n",
            product.holes_filled,
            total,
            100.0 * product.holes_filled as f64 / total as f64
        ));
    } else {
        out.push_str("Unsupported:       none; every output pixel is backed by real samples\n");
    }
    if let Some(c) = coverage {
        out.push('\n');
        out.push_str(&c.describe());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
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

    fn tmpdir(name: &str) -> PathBuf {
        let mut p = scratch_dir();
        p.push(format!("smokstak-diag-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn dummy_product(w: usize, h: usize) -> ReconstructionProduct {
        ReconstructionProduct {
            channels: 3,
            width: w,
            height: h,
            rgb: [Plane::filled(w, h, 0.5), Plane::filled(w, h, 0.5), Plane::filled(w, h, 0.5)],
            weight: [Plane::filled(w, h, 1.0), Plane::filled(w, h, 1.0), Plane::filled(w, h, 1.0)],
            count: [Plane::filled(w, h, 4.0), Plane::filled(w, h, 8.0), Plane::filled(w, h, 4.0)],
            effective_frames: Plane::filled(w / 8, h / 8, 12.0),
            effective_frames_cell: 8,
            rejected: Plane::<f32>::new(w, h),
            holes_filled: 0,
            stats: ProductStats {
                total_samples: 1000,
                accumulated_samples: 950,
                rejected_samples: 20,
                masked_samples: 30,
                mean_effective_frames: 12.0,
                min_effective_frames: 11.0,
                ..Default::default()
            },
        }
    }

    #[test]
    fn writes_the_diagnostic_bundle() {
        let d = tmpdir("bundle");
        let files = write_product_diagnostics(&d, &dummy_product(64, 64)).unwrap();
        assert_eq!(files.len(), 6);
        for f in &files {
            assert!(f.exists(), "{} was not written", f.display());
            assert!(std::fs::metadata(f).unwrap().len() > 0);
        }
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn frames_csv_has_one_row_per_frame_plus_a_header() {
        let d = tmpdir("csv");
        let rows: Vec<FrameRow> = (0..3)
            .map(|i| FrameRow {
                index: i,
                file: format!("DSC_{i}.NEF"),
                used: true,
                sharpness: 1.0,
                contrast: 0.01,
                estimated_blur: 0.8,
                blur_anisotropy: 0.1,
                saturation_fraction: 0.0,
                star_hfd: 0.0,
                star_eccentricity: 0.0,
                star_count: 0,
                exposure_scale: 1.0,
                photometric_gain: 1.0,
                photometric_offset: 0.0,
                photometric_source: "measured".into(),
                transform_model: "translation".into(),
                shift_x: 0.1,
                shift_y: -0.2,
                rotation_deg: 0.0,
                scale: 1.0,
                residual_rms: 0.05,
                residual_p90: 0.09,
                inliers: 100,
                probes: 110,
                overlap: 1.0,
                confidence: 0.9,
                local_warp_mean: 0.0,
                local_warp_max: 0.0,
                robustness_rejected: 0.0,
            })
            .collect();
        let p = d.join("frames.csv");
        write_frames_csv(&p, &rows).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert_eq!(text.lines().count(), 4);
        assert!(text.lines().next().unwrap().starts_with("index,file,used"));
        assert!(text.contains("DSC_1.NEF"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn summary_reports_unsupported_pixels_when_there_are_some() {
        let mut p = dummy_product(64, 64);
        p.holes_filled = 12;
        let s = summarise(&p, None);
        assert!(s.contains("12 of 12288") && s.contains("0.0977%"), "{s}");
        p.channels = 1;
        let mono = summarise(&p, None);
        assert!(mono.contains("12 of 4096") && mono.contains("0.2930%"), "{mono}");
        let clean = dummy_product(64, 64);
        assert!(summarise(&clean, None).contains("none"));
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let m = RunManifest {
            program: "smokstak".into(),
            version: "0.1.0".into(),
            build_profile: "release".into(),
            revision: "unknown".into(),
            started_utc: "2026-01-01T00:00:00Z".into(),
            host: HostInfo::default(),
            dependencies: dependency_versions(),
            input: "burst".into(),
            sources: vec![SourceFile { index: 0, path: "a.NEF".into(), hash: "abc".into() }],
            reference_index: 0,
            reference_file: "a.NEF".into(),
            reference_reason: "only frame".into(),
            photometric_reference_index: Some(0),
            config: ReconstructionConfig::default(),
            seed: 7,
            noise: NoiseRecord { alpha: 1.0, beta: 2.0, source: "measured".into() },
            color: ColorRecord {
                wb: [2.0, 1.0, 1.5],
                cam_to_srgb: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                matrix_fallback: false,
                exposure_gain: 1.0,
            },
            output_width: 128,
            output_height: 128,
            output_files: vec!["result.tif".into()],
            coverage: None,
            stats: ProductStats::default(),
            timings_ms: BTreeMap::new(),
            warnings: vec![],
        };
        let d = tmpdir("manifest");
        let p = d.join("run.json");
        write_manifest(&p, &m).unwrap();
        let back: RunManifest =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(back.reference_file, "a.NEF");
        assert_eq!(back.photometric_reference_index, Some(0));
        let mut legacy = serde_json::to_value(&m).unwrap();
        legacy.as_object_mut().unwrap().remove("photometric_reference_index");
        let legacy: RunManifest = serde_json::from_value(legacy).unwrap();
        assert_eq!(legacy.photometric_reference_index, None);
        assert_eq!(back.config.scale, m.config.scale);
        assert_eq!(back.seed, 7);
        std::fs::remove_dir_all(&d).ok();
    }
}
