//! Recompute statistics from a saved report and verified compact samples.
//! Does not decode source images or bypass cache validation in normal analyze.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Write,
    path::PathBuf,
};
#[allow(dead_code)]
#[path = "../src/analyze/stats.rs"]
mod stats;

fn retain_subset(report: &mut Value, list: &std::path::Path) -> Result<()> {
    let text = fs::read_to_string(list)?;
    let requested: HashSet<String> = text
        .lines()
        .map(|line| line.trim().trim_matches('"'))
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect();
    let original = report["frames"].as_array().context("frames missing")?;
    ensure!(!requested.is_empty(), "retained list is empty");
    let known: HashSet<_> = original.iter().filter_map(|f| f["path"].as_str()).collect();
    ensure!(requested.iter().all(|p| known.contains(p.as_str())), "retained list contains paths absent from the saved report; exact original paths are required");
    let mut frames: Vec<_> = original
        .iter()
        .filter(|f| requested.contains(f["path"].as_str().unwrap_or_default()))
        .cloned()
        .collect();
    let (mut seconds, mut accepted_seconds) = (0.0, 0.0);
    for (index, frame) in frames.iter_mut().enumerate() {
        frame["source_input_index"] = frame["index"].clone();
        frame["index"] = json!(index);
        let exposure = frame["exposure_seconds"].as_f64().unwrap_or(0.0);
        seconds += exposure;
        if frame["accepted"] == true {
            accepted_seconds += exposure;
        }
        frame["known_integration_seconds"] = json!(seconds);
        frame["accepted_integration_seconds"] = json!(accepted_seconds);
        frame["cache_hit"] = json!(frame["accepted"] == true);
    }
    let accepted = frames.iter().filter(|f| f["accepted"] == true).count();
    report["summary"] = json!({"input_frames":frames.len(), "accepted_frames":accepted,
        "rejected_frames":frames.len()-accepted, "known_integration_seconds":seconds,
        "accepted_integration_seconds":accepted_seconds, "duplicate_files_dropped":0,
        "unknown_exposures":frames.iter().filter(|f| f["exposure_seconds"].is_null()).count(), "cache_hits":accepted});
    let groups = report["filters"]
        .as_array_mut()
        .context("filters missing")?;
    groups.retain(|g| frames.iter().any(|f| f["filter"] == g["filter"]));
    for group in groups {
        let rows: Vec<_> = frames
            .iter()
            .filter(|f| f["filter"] == group["filter"])
            .collect();
        let accepted: Vec<_> = rows.iter().filter(|f| f["accepted"] == true).collect();
        group["accepted_frames"] = json!(accepted.len());
        group["rejected_frames"] = json!(rows.len() - accepted.len());
        group["known_integration_seconds"] = json!(accepted
            .iter()
            .filter_map(|f| f["exposure_seconds"].as_f64())
            .sum::<f64>());
        group["unknown_exposures"] = json!(accepted
            .iter()
            .filter(|f| f["exposure_seconds"].is_null())
            .count());
        for (output, input) in [
            ("median_hfd_sensor_px", "hfd"),
            ("median_eccentricity", "eccentricity"),
            (
                "median_registration_residual_sensor_px",
                "registration_residual_sensor_px",
            ),
        ] {
            let mut values: Vec<_> = rows.iter().filter_map(|f| f[input].as_f64()).collect();
            values.sort_by(f64::total_cmp);
            group[output] = if values.is_empty() {
                Value::Null
            } else {
                json!((values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0)
            };
        }
        for field in ["integration_depth", "snapshots", "projections"] {
            group[field] = json!([]);
        }
        for field in ["global_fit", "recent_fit"] {
            group[field] = Value::Null;
        }
        let mut warnings = vec![json!("Subset replay: original registration, photometry, common footprint and per-frame advisory flags retained; no source-image remeasurement.")];
        let fallback = accepted
            .iter()
            .filter(|f| f["stellar_gain"] == false)
            .count();
        if fallback > 0 {
            warnings.push(json!(format!("{fallback} accepted frames lack a stellar gain; interpret noise scaling cautiously.")));
        }
        if group["unknown_exposures"].as_u64().unwrap_or(0) > 0 {
            warnings.push(json!(
                "Integration totals exclude unknown exposure durations."
            ));
        }
        if group["filter"] == "" {
            warnings.push(json!("Filter is missing: verify one shared passband."));
        }
        group["warnings"] = json!(warnings);
    }
    report["frames"] = json!(frames);
    report["project"] = json!(list);
    Ok(())
}

#[cfg(test)]
mod subset_tests {
    use super::*;
    #[test]
    fn subset_reindexes_and_recomputes_accounting_without_mutating_source() {
        let source = json!({"frames":[
            {"index":0,"path":"/a.fit","filter":"H","accepted":true,"exposure_seconds":100,"hfd":2},
            {"index":1,"path":"/b.fit","filter":"O","accepted":true,"exposure_seconds":200,"hfd":4},
            {"index":2,"path":"/c.fit","filter":"H","accepted":true,"exposure_seconds":300,"hfd":6},
            {"index":3,"path":"/d.fit","filter":"H","accepted":false,"exposure_seconds":null}
        ],"filters":[{"filter":"H","sample_count":123},{"filter":"O","sample_count":456}]});
        let path = std::env::temp_dir().join(format!("smokstak-subset-{}.txt", std::process::id()));
        fs::write(&path, "# selected\n/c.fit\n/d.fit\n").unwrap();
        let mut report = source.clone();
        retain_subset(&mut report, &path).unwrap();
        assert_eq!(source["frames"].as_array().unwrap().len(), 4);
        assert_eq!(report["summary"]["input_frames"], 2);
        assert_eq!(report["summary"]["accepted_frames"], 1);
        assert_eq!(report["summary"]["accepted_integration_seconds"], 300.0);
        assert_eq!(report["summary"]["unknown_exposures"], 1);
        assert_eq!(report["frames"][0]["index"], 0);
        assert_eq!(report["frames"][0]["source_input_index"], 2);
        assert_eq!(report["frames"][1]["known_integration_seconds"], 300.0);
        assert_eq!(report["filters"].as_array().unwrap().len(), 1);
        assert_eq!(report["filters"][0]["median_hfd_sensor_px"], 6.0);
        assert_eq!(report["filters"][0]["sample_count"], 123);
        assert_eq!(report["filters"][0]["integration_depth"], json!([]));
        fs::write(&path, "/unknown.fit\n").unwrap();
        assert!(retain_subset(&mut source.clone(), &path).is_err());
        fs::remove_file(path).unwrap();
    }
}

fn main() -> Result<()> {
    let started = std::time::Instant::now();
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3 || args.len() == 4,
        "usage: replay_analysis REPORT.json CACHE_DIRECTORY NEW_OUTPUT_DIRECTORY [RETAINED_FRAMES.txt]"
    );
    let source = PathBuf::from(&args[0]);
    let cache = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    ensure!(!output.exists(), "output directory must not already exist");
    let mut report: Value = serde_json::from_slice(&fs::read(&source)?)?;
    ensure!(
        report["schema_version"] == 1
            || report["schema_version"] == 2
            || report["schema_version"] == 3,
        "unsupported report schema"
    );
    let mut records = HashMap::new();
    for entry in fs::read_dir(cache)? {
        let path = entry?.path();
        if path.extension().is_none_or(|s| s != "json") {
            continue;
        }
        let record: Value = serde_json::from_slice(&fs::read(&path)?)?;
        let Some(input) = record["metadata"]["path"].as_str().map(str::to_owned) else {
            continue;
        };
        ensure!(records.insert(input.to_owned(), (path, record)).is_none(), "multiple cache entries for {input}; use a cache directory containing one measurement set");
    }
    let original_frames = report["frames"]
        .as_array()
        .context("frames missing")?
        .clone();
    if let Some(list) = args.get(3) {
        retain_subset(&mut report, &PathBuf::from(list))?;
    }
    let frames = report["frames"]
        .as_array()
        .context("frames missing")?
        .clone();
    for group in report["filters"]
        .as_array_mut()
        .context("filters missing")?
    {
        let rows: Vec<_> = frames
            .iter()
            .filter(|f| f["filter"] == group["filter"] && f["accepted"] == true)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let count = group["candidate_tiles"]
            .as_u64()
            .context("candidate_tiles missing")? as usize;
        let mut valid = vec![true; count];
        // Keep the original common footprint, even when removing frames would
        // make additional tiles available. This isolates the selection comparison.
        for frame in original_frames
            .iter()
            .filter(|f| f["filter"] == group["filter"] && f["accepted"] == true)
        {
            let (_, record) = records
                .get(frame["path"].as_str().context("path missing")?)
                .context("cache entry missing")?;
            ensure!(
                record["photometry"] == frame["photometry"],
                "photometry differs from report"
            );
            let flags = record["valid_tiles"]
                .as_array()
                .context("valid tiles missing")?;
            ensure!(flags.len() == count, "sample layout differs");
            for (v, flag) in valid.iter_mut().zip(flags) {
                *v &= flag.as_bool().context("invalid tile flag")?;
            }
        }
        ensure!(
            valid.iter().filter(|&&v| v).count() as u64
                == group["common_tiles"]
                    .as_u64()
                    .context("common tiles missing")?,
            "common footprint differs"
        );
        if valid.iter().filter(|&&v| v).count() < 32 {
            continue;
        }
        let mut acc = stats::Accumulator::new(valid);
        let mut depth = Vec::new();
        let mut snapshots = Vec::new();
        let total = rows.len();
        let first_record = &records[rows[0]["path"].as_str().unwrap()].1;
        group["patch_positions"] = serde_json::to_value(stats::tiles(
            first_record["width"].as_u64().context("width missing")? as usize,
            first_record["height"].as_u64().context("height missing")? as usize,
            count * stats::PIXELS,
        ))?;
        for frame in rows {
            let (path, record) = &records[frame["path"].as_str().unwrap()];
            let bytes = fs::read(path.with_extension("samples"))?;
            ensure!(
                bytes.len() == count * stats::PIXELS * 4,
                "sample length differs"
            );
            ensure!(
                format!("{:x}", Sha256::digest(&bytes))
                    == record["samples_digest"]
                        .as_str()
                        .context("digest missing")?,
                "sample checksum mismatch"
            );
            let values: Vec<f32> = bytes
                .as_chunks::<4>().0.iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            depth.push(acc.push(
                &values,
                frame["exposure_seconds"].as_f64().map(|t| t as f32),
                frame["index"].as_u64().context("index missing")? as usize,
            ));
            if stats::snapshot_due(depth.len(), total) {
                snapshots.push(acc.snapshot());
            }
        }
        group["snapshots"] = serde_json::to_value(snapshots)?;
        let recent = depth
            .iter()
            .rev()
            .find(|d| d.noise.is_some())
            .and_then(|d| d.local_fit.as_ref());
        group["global_fit"] = serde_json::to_value(acc.global_fit())?;
        group["recent_fit"] = serde_json::to_value(recent)?;
        group["projections"] = serde_json::to_value(stats::projections(
            recent,
            depth.len(),
            group["known_integration_seconds"].as_f64().unwrap_or(0.0),
            group["unknown_exposures"].as_u64().unwrap_or(0) as usize,
        ))?;
        if let Some(last) = depth.iter().rev().find(|d| d.noise.is_some()) {
            println!(
                "{}: N={}, difference={:.8}, spatial={:.8}, exponent={}",
                group["filter"],
                last.frame_count,
                last.noise.unwrap(),
                last.spatial_residual,
                group["global_fit"]["exponent"]
            );
        }
        group["integration_depth"] = serde_json::to_value(depth)?;
    }
    report["schema_version"] = json!(3);
    report["method"] = json!(stats::METHOD);
    report["limitations"] = json!([stats::LIMITATION, "Spatial residuals include astronomical structure and are not a random-noise measurement.", "Equal weighting and nearest-detector sampling differ from a resampled, weighted, rejection-based production stack.", "Fits are descriptive; projections assume the recent trend continues and are not forecasts.", "Recomputed selected saved samples using original registration, photometry and common footprint; original advisory flags, measurement timings and build identity are retained."]);
    report["statistics_replay"] = json!({"source_report":source, "retain_list":args.get(3).map(PathBuf::from), "original_frame_count":original_frames.len(), "retained_frame_count":frames.len(), "fixed_original_footprint":true, "elapsed_seconds":started.elapsed().as_secs_f64(), "statistics_executable_sha256":format!("{:x}", Sha256::digest(fs::read(std::env::current_exe()?)?)), "verified_sample_checksums":true});
    let embedded = serde_json::to_string(&report)?
        .replace('<', "\\u003c")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    let html = include_str!("../src/analyze/report.html")
        .replace("__REPORT_JSON__", &embedded)
        .replace(
            "__INSIGHTS_JS__",
            include_str!("../src/analyze/insights.js"),
        )
        .replace("__REVIEW_JS__", include_str!("../src/analyze/review.js"));
    fs::create_dir_all(&output)?;
    for (name, bytes) in [
        ("project-analysis.json", serde_json::to_vec_pretty(&report)?),
        ("project-analysis.html", html.into_bytes()),
    ] {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join(name))?
            .write_all(&bytes)?;
    }
    Ok(())
}
