//! End-to-end CLI contract using real FITS decode, registration and photometry.
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "smokstak-project-e2e-{}-{nonce}-{}",
            std::process::id(), SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}

fn command(dir: &Path, args: &[&str], success: bool) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_smokstak"))
        .current_dir(dir).args(["--threads", "2", "--log", "warn", "project"])
        .args(args).output().unwrap();
    let text = format!("{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.success(), success, "{args:?}\n{text}");
    text
}

fn json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn gui_runs_analysis_stack_and_versioned_updates() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::{Duration, Instant};
    struct Server(std::process::Child);
    impl Drop for Server {
        fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); }
    }
    fn call(address: &str, route: &str, body: Option<Value>) -> Value {
        let mut stream = TcpStream::connect(address).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let method = if body.is_some() { "POST" } else { "GET" };
        let body = body.map(|b|b.to_string()).unwrap_or_default();
        write!(stream,"{method} {route} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
        let mut response=String::new();stream.read_to_string(&mut response).unwrap();
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap()
    }
    fn finish(address: &str) -> Value {
        let start=Instant::now();
        loop {
            let state=call(address,"/status",None);
            if state["finished"]==true {
                assert_eq!(state["failed"],false,"{state}");
                return state;
            }
            assert!(start.elapsed()<Duration::from_secs(240),"GUI run timed out: {state}");
            std::thread::sleep(Duration::from_millis(150));
        }
    }
    fn workflow(address: &str, body: Value) -> Value {
        let response=call(address,"/workflow",Some(body));
        assert_eq!(response["ok"],true,"{response}");
        finish(address)
    }
    let scratch=Scratch::new();let dir=&scratch.0;
    for i in 0..4 { fits(&dir.join(format!("frame {i}.fit")),i,"H",Some(120)); }
    fs::write(dir.join("first.txt"),"frame 0.fit\nframe 1.fit\nframe 2.fit\n").unwrap();
    let stdout=dir.join("gui.log");
    let child=Command::new(env!("CARGO_BIN_EXE_smokstak")).current_dir(dir)
        .args(["gui","--no-open","--port","0"]).env("SMOKSTAK_HISTORY",dir.join("history.json"))
        .stdout(fs::File::create(&stdout).unwrap()).stderr(fs::File::create(dir.join("gui-errors.log")).unwrap())
        .spawn().unwrap();
    let _server=Server(child);
    let started=Instant::now();
    let address=loop {
        let log=fs::read_to_string(&stdout).unwrap();
        if let Some((_,tail))=log.split_once("http://") { break tail.lines().next().unwrap().trim().to_string(); }
        assert!(started.elapsed()<Duration::from_secs(10));std::thread::sleep(Duration::from_millis(50));
    };
    let analysis=workflow(&address,serde_json::json!({"action":"analyze","input":"first.txt","directory":"reports","workers":2}));
    let report=analysis["outputs"].as_array().unwrap().iter().find_map(|v|v.as_str().filter(|s|s.ends_with("project-analysis.html"))).unwrap();
    assert!(dir.join(report).is_file());
    assert!(fs::read_to_string(dir.join(report)).unwrap().contains("Noise"));
    // Requesting an unrelated HTML file must not invoke an OS opener.
    let denied=call(&address,"/open-report",Some(serde_json::json!({"path":"first.txt"})));
    assert_ne!(denied["ok"],true);
    let response=call(&address,"/start",Some(serde_json::json!({"input":"first.txt","output":"ordinary.tif",
        "scale":1,"threads":2,"diagnostics":true,"fits":true,"preview":true})));
    assert_eq!(response["ok"],true,"{response}");
    let stacked=finish(&address);
    let audit=stacked["outputs"].as_array().unwrap().iter().find_map(|v|v.as_str().filter(|s|s.ends_with("frame-review.html"))).unwrap();
    assert!(dir.join(audit).is_file());
    assert!(fs::read_to_string(dir.join(audit)).unwrap().contains("Sky + mask"));
    let project=workflow(&address,serde_json::json!({"action":"create","input":"first.txt","directory":"project","reference":"frame 0.fit"}));
    assert!(project["outputs"].as_array().unwrap().iter().all(|v| !v.as_str().unwrap().contains(".pending")));
    let original=fs::read(dir.join("project/runs/00000001/master_H.fits")).unwrap();
    workflow(&address,serde_json::json!({"action":"add","input":"frame 3.fit","directory":"project"}));
    assert_eq!(json(dir.join("project/runs/00000002/frame-review.json"))["frames"].as_array().unwrap().len(),4);
    assert_eq!(original,fs::read(dir.join("project/runs/00000001/master_H.fits")).unwrap());
    fs::write(dir.join("review.txt"),"frame 3.fit\n").unwrap();
    workflow(&address,serde_json::json!({"action":"exclude","directory":"project","list":"review.txt","reason":"Inspected cloud"}));
    let audit=json(dir.join("project/runs/00000003/frame-review.json"));
    assert_eq!(audit["frames"].as_array().unwrap().iter().filter(|f| f["status"]=="excluded").count(),1);
    let status=workflow(&address,serde_json::json!({"action":"status","directory":"project"}));
    assert!(status["outputs"].as_array().unwrap().iter().any(|v|v.as_str().unwrap().contains("00000003")));
    assert!(status["log"].as_array().unwrap().iter().any(|v|v.as_str().unwrap().contains("3 selected; 1 excluded")));
}

fn compare_runs(a: &Path, b: &Path) {
    let left = json(a.join("project-build.json"));
    let right = json(b.join("project-build.json"));
    assert_eq!(left["input_ids"], right["input_ids"]);
    assert_eq!(left["recipe"], right["recipe"]);
    assert_eq!(left["reference"], right["reference"]);
    let artifacts = left["artifact_sha256"].as_object().unwrap();
    let mut images = 0;
    for name in artifacts.keys() {
        if name.ends_with(".tif") || name.ends_with(".fits") {
            assert_eq!(fs::read(a.join(name)).unwrap(), fs::read(b.join(name)).unwrap(), "different pixels: {name}");
            images += 1;
        }
        if name.ends_with("run.json") {
            let l = json(a.join(name));
            let r = json(b.join(name));
            for key in ["sources", "reference_index", "reference_file", "photometric_reference_index", "config", "noise", "color", "coverage", "stats"] {
                assert_eq!(l[key], r[key], "different decision {name}: {key}");
            }
            for output in l["output_files"].as_array().unwrap() {
                assert!(Path::new(output.as_str().unwrap()).is_file(), "published manifest contains stale output path");
            }
        }
    }
    assert!(images >= 4, "must compare scientific images and diagnostics");
    for run in [a, b] {
        assert!(!run.join("spool").exists() || fs::read_dir(run.join("spool")).unwrap().next().is_none(), "scratch maps leaked");
        let audit = json(run.join("frame-review.json"));
        let snapshot = json(run.join("project-revision.json"));
        let frames = audit["frames"].as_array().unwrap();
        assert_eq!(frames.len(), snapshot["frames"].as_array().unwrap().len());
        for row in frames {
            assert!(row["capture_time"].as_str().unwrap().starts_with("2026-09-"));
            assert_eq!(row["exposure_seconds"].as_f64(), Some(120.0));
            assert_eq!(row["iso_or_gain"].as_f64(), Some(50.0));
            let source = snapshot["frames"].as_array().unwrap().iter().find(|s|s["path"]==row["path"]).unwrap();
            if !source["exclusion_reason"].is_null() {
                assert_eq!(row["status"], "excluded");
                assert!(row["weight"].is_null());
                assert!(row["reason"].as_str().unwrap().contains(source["exclusion_reason"].as_str().unwrap()));
            } else if row["status"] == "used" {
                assert!(row["weight"].as_f64().unwrap()>0.0);
                let gains = row["photometric_gain"].as_array().unwrap();
                assert_eq!(gains.len(), 1, "mono audit must record the applied gain");
                assert!(gains[0].as_f64().is_some_and(|g| g.is_finite() && g > 0.0));
                assert!(!row["photometry_source"].as_str().unwrap().is_empty());
            }
            let asset = fs::read_to_string(run.join(row["preview_asset"].as_str().unwrap())).unwrap();
            let pixels: Value = serde_json::from_str(asset.split_once(',').unwrap().1.strip_suffix(");").unwrap()).unwrap();
            assert_eq!(pixels["native"]["pixels"].as_array().unwrap().len(),96*96);
            if row["status"] == "excluded" { assert!(pixels["aligned"].is_null()); }
            else if row["status"] == "used" { assert!(pixels["aligned"].is_object()); }
        }
        let html=fs::read_to_string(run.join("frame-review.html")).unwrap();
        assert!(!html.contains("__REVIEW_JSON__"));
        assert!(!html.contains("__REVIEW_JS__"));
    }
}

#[test]
fn update_matches_clean_build_and_resident_and_cached_paths() {
    let scratch = Scratch::new();
    let dir = &scratch.0;
    fs::create_dir(dir.join("lights")).unwrap();
    for i in 0..8 {
        fits(&dir.join(format!("lights/{i:02}.fit")), i, if i < 6 { "H" } else { "O" }, Some(120));
    }
    fs::write(dir.join("a.txt"), "lights/00.fit\nlights/01.fit\nlights/02.fit\nlights/03.fit\n").unwrap();
    fs::write(dir.join("b.txt"), "lights/04.fit\nlights/05.fit\nlights/06.fit\nlights/07.fit\n").unwrap();
    let recipe = sr_core::config::ReconstructionConfig {
        scale: 1.0, roi: Some((128,128,128,128)), ..Default::default()
    };
    fs::write(dir.join("recipe.json"), serde_json::to_vec(&recipe).unwrap()).unwrap();
    command(dir, &["init", "incremental", "a.txt", "--reference-file", "lights/00.fit", "--recipe", "recipe.json"], true);
    command(dir, &["build", "incremental"], true);
    let first = fs::read(dir.join("incremental/runs/00000001/project-build.json")).unwrap();
    command(dir, &["add", "incremental", "b.txt"], true);
    command(dir, &["build", "incremental"], true);
    assert_eq!(fs::read(dir.join("incremental/runs/00000001/project-build.json")).unwrap(), first);
    command(dir, &["init", "clean", "lights", "--reference-file", "lights/00.fit", "--recipe", "recipe.json"], true);
    command(dir, &["build", "clean", "--in-memory", "--no-cache"], true);
    compare_runs(&dir.join("incremental/runs/00000002"), &dir.join("clean/runs/00000001"));
    let warm = command(dir, &["build", "incremental"], true);
    assert!(warm.contains("Registration: reused from cache"), "{warm}");
    compare_runs(&dir.join("incremental/runs/00000002"), &dir.join("incremental/runs/00000003"));

    // Content identity survives a copied/renamed file and preserves decisions.
    fs::copy(dir.join("lights/03.fit"), dir.join("renamed.fit")).unwrap();
    command(dir, &["add", "incremental", "renamed.fit"], true);
    assert!(!dir.join("incremental/revisions/00000003.json").exists());
    fs::write(dir.join("review.txt"), "lights/00.fit\n").unwrap();
    command(dir, &["exclude", "incremental", "review.txt", "--reason", "reference deliberately excluded"], true);
    command(dir, &["exclude", "clean", "review.txt", "--reason", "reference deliberately excluded"], true);
    command(dir, &["build", "incremental"], true);
    command(dir, &["build", "clean", "--in-memory", "--no-cache"], true);
    compare_runs(&dir.join("incremental/runs/00000004"), &dir.join("clean/runs/00000002"));
    let revision = json(dir.join("incremental/revisions/00000003.json"));
    assert_eq!(revision["frames"].as_array().unwrap().iter().filter(|f| !f["exclusion_reason"].is_null()).count(), 1);
    command(dir, &["include", "incremental", "review.txt"], true);
    assert!(json(dir.join("incremental/revisions/00000004.json"))["frames"].as_array().unwrap().iter().all(|f| f["exclusion_reason"].is_null()));

    // Detect source edits, preserve every published master, and refuse a live lock.
    let previous = fs::read(dir.join("incremental/runs/00000004/project-build.json")).unwrap();
    fs::write(dir.join("incremental/project.lock"), "test lock").unwrap();
    assert!(command(dir, &["build", "incremental"], false).contains("cannot lock"));
    fs::remove_file(dir.join("incremental/project.lock")).unwrap();
    fits(&dir.join("lights/01.fit"), 19, "H", Some(120));
    assert!(command(dir, &["build", "incremental"], false).contains("source content changed"));
    assert_eq!(fs::read(dir.join("incremental/runs/00000004/project-build.json")).unwrap(), previous);
    assert!(!dir.join("incremental/runs/00000005").exists());
}

#[test]
fn moved_sources_keep_identity_reviews_and_grid_and_reject_output_collisions() {
    let scratch = Scratch::new();
    let dir = &scratch.0;
    fs::create_dir(dir.join("lights")).unwrap();
    fits(&dir.join("lights/anchor.fit"), 0, "H", Some(120));
    fits(&dir.join("lights/second.fit"), 1, "H", Some(120));
    command(dir, &["init", "project", "lights", "--reference-file", "lights/anchor.fit"], true);
    fs::write(dir.join("review.txt"), "lights/anchor.fit\n").unwrap();
    command(dir, &["exclude", "project", "review.txt", "--reason", "cloud"], true);
    let before = json(dir.join("project/revisions/00000002.json"));
    fs::rename(dir.join("lights"), dir.join("moved")).unwrap();
    command(dir, &["relink", "project", "moved"], true);
    let after = json(dir.join("project/revisions/00000003.json"));
    assert_eq!(before["reference"]["id"], after["reference"]["id"]);
    assert!(Path::new(after["reference"]["path"].as_str().unwrap()).is_file());
    for (a,b) in before["frames"].as_array().unwrap().iter().zip(after["frames"].as_array().unwrap()) {
        assert_eq!(a["id"], b["id"]);
        assert_eq!(a["exclusion_reason"], b["exclusion_reason"]);
        assert!(Path::new(b["path"].as_str().unwrap()).is_file());
    }
    fits(&dir.join("unknown.fit"), 2, "H", Some(120));
    command(dir, &["relink", "project", "unknown.fit"], false);
    fits(&dir.join("collision.fit"), 3, "h", Some(120));
    command(dir, &["add", "project", "collision.fit"], false);
    fits(&dir.join("reserved.fit"), 4, "CON", Some(120));
    command(dir, &["add", "project", "reserved.fit"], false);
    assert!(!dir.join("project/revisions/00000004.json").exists());
    let ids = after["frames"].as_array().unwrap().iter().map(|f| f["id"].as_str().unwrap()).collect::<Vec<_>>().join("\n");
    fs::write(dir.join("all.txt"), ids).unwrap();
    command(dir, &["exclude", "project", "all.txt", "--reason", "review pending"], true);
    assert!(command(dir, &["build", "project"], false).contains("all project frames are excluded"));
    assert!(!dir.join("project/runs").exists());
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
        ("GAIN", "50".into()),
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
