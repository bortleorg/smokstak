//! End-to-end CLI contract using real FITS decode, registration and photometry.
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "smokstak-analyze-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fits(path: &Path, index: usize, filter: &str, exposure: Option<u32>) {
    let mut header = String::new();
    for (k, v) in [
        ("SIMPLE", "T".into()),
        ("BITPIX", "-32".into()),
        ("NAXIS", "2".into()),
        ("NAXIS1", "512".into()),
        ("NAXIS2", "512".into()),
        ("ROWORDER", "'TOP-DOWN'".into()),
        ("FILTER", format!("'{filter}'")),
        ("DATE-OBS", format!("'2026-09-{:02}T01:02:03'", index + 1)),
    ] {
        header.push_str(&format!("{k:<8}= {v:<70}"));
    }
    if let Some(e) = exposure {
        header.push_str(&format!("{:<8}= {:<70}", "EXPTIME", e));
    }
    header.push_str(&format!("{:<80}", "END"));
    while !header.len().is_multiple_of(2880) {
        header.push(' ');
    }
    let mut bytes = header.into_bytes();
    let mut seed = index as u64 + 17;
    let shift = (index % 3) as f32;
    for y in 0..512 {
        for x in 0..512 {
            let (sx, sy) = (x as f32 - shift, y as f32);
            // An irregular lattice with varied stellar flux, rather than identical
            // periodic peaks that give the registration matcher several answers.
            let mut value = 0.08 + index as f32 * 0.005;
            let (gx, gy) = ((sx / 37.0).floor() as i32, (sy / 41.0).floor() as i32);
            for cy in gy - 1..=gy + 1 {
                for cx in gx - 1..=gx + 1 {
                    let star = (cx * 53 + cy * 97).rem_euclid(23);
                    let (px, py) = (
                        cx as f32 * 37.0 + 10.0 + star as f32 * 0.35,
                        cy as f32 * 41.0 + 13.0 + star as f32 * 0.21,
                    );
                    let r2 = (sx - px).powi(2) + (sy - py).powi(2);
                    value += (0.15 + star as f32 * 0.006) * (-r2 / 6.0).exp();
                }
            }
            let mut noise = 0.0;
            for _ in 0..8 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                noise += (seed >> 32) as f32 / u32::MAX as f32 - 0.5;
            }
            bytes.extend((value + noise * 0.003).to_be_bytes());
        }
    }
    while !bytes.len().is_multiple_of(2880) {
        bytes.push(0);
    }
    fs::write(path, bytes).unwrap();
}

fn run(dir: &Path, input: &Path, extra: &[&str]) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_smokstak"))
        .current_dir(dir)
        .args(["--threads", "2", "--log", "warn", "analyze"])
        .arg(input)
        .args(["--samples", "8192"])
        .args(extra)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&fs::read(dir.join("project-analysis.json")).unwrap()).unwrap()
}

#[test]
fn filters_order_exposure_cache_and_offline_report() {
    let scratch = Scratch::new();
    let dir = &scratch.0;
    let lights = dir.join("lights");
    fs::create_dir(&lights).unwrap();
    for i in 0..6 {
        fits(
            &lights.join(format!("{i:02}.fit")),
            i,
            if i % 2 == 0 { "H" } else { "O" },
            (i != 5).then_some(120),
        );
    }
    fs::write(lights.join("bad.fit"), "not fits").unwrap();
    let cold = run(dir, Path::new("lights"), &[]);
    assert!(
        cold["frames"]
            .as_array()
            .unwrap()
            .iter()
            .all(|frame| { Path::new(frame["path"].as_str().unwrap()).is_absolute() })
    );
    assert_eq!(cold["schema_version"], 3);
    assert_eq!(cold["summary"]["input_frames"], 7);
    assert_eq!(cold["summary"]["cache_hits"], 0);
    assert_eq!(cold["summary"]["accepted_frames"], 6, "{cold:#}");
    assert_eq!(cold["summary"]["known_integration_seconds"], 600.0);
    assert_eq!(cold["summary"]["unknown_exposures"], 2);
    assert_eq!(cold["frames"][0]["capture_time"], "2026-09-01T01:02:03");
    assert!(cold["frames"][6]["rejection"].is_string());
    for g in cold["filters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|g| g["filter"] != "")
    {
        assert_eq!(g["integration_depth"].as_array().unwrap().len(), 3);
        let indices: Vec<_> = g["integration_depth"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["frame_index"].as_u64().unwrap())
            .collect();
        assert!(indices.windows(2).all(|p| p[0] < p[1]));
    }
    let warm = run(dir, &lights, &[]);
    assert_eq!(warm["summary"]["cache_hits"], 6);
    assert_eq!(warm["filters"], cold["filters"]);
    let html = fs::read_to_string(dir.join("project-analysis.html")).unwrap();
    assert!(html.contains("<canvas"));
    assert!(!html.contains("__REPORT_JSON__"));
    assert!(!html.contains("__REVIEW_JS__"));
    assert!(html.contains("retained-frames.txt"));
    assert!(!html.contains("https://"));
    fits(&lights.join("06.fit"), 6, "H", Some(240));
    let incremental = run(dir, &lights, &[]);
    assert_eq!(incremental["summary"]["cache_hits"], 6);
    assert_eq!(incremental["summary"]["known_integration_seconds"], 840.0);
    let cache = dir.join("cache/analysis");
    let entry = fs::read_dir(&cache)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|s| s == "samples"))
        .unwrap();
    fs::write(entry, b"broken").unwrap();
    let repaired = run(dir, &lights, &[]);
    assert_eq!(repaired["summary"]["cache_hits"], 6);
    assert_eq!(incremental["filters"], repaired["filters"]);
    let list = dir.join("frames.txt");
    fs::write(
        &list,
        "# relative list, sorted by discovery\nlights/04.fit\nlights/00.fit\nlights/02.fit\n",
    )
    .unwrap();
    let listed = run(dir, &list, &["--filter", "h"]);
    assert_eq!(listed["summary"]["input_frames"], 3);
    assert_eq!(listed["frames"][0]["file"], "00.fit");
    let no_cache = run(dir, &list, &["--no-cache"]);
    assert_eq!(no_cache["filters"], listed["filters"]);
    assert_eq!(no_cache["summary"]["cache_hits"], 0);
}
