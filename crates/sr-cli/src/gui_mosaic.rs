//! Prepared mono mosaic plans, using the GUI's existing child-process lifecycle.
use super::*;
use anyhow::{ensure, Context};

const ASSET_LIMIT: u64 = 16 * 1024 * 1024;
const OUTPUTS: [&str; 6] = ["image.fits", "weight.fits", "samples.fits", "preview.png", "plan.json", "result.json"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewRequest { plan: String, tile: usize, memory_mb: usize }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StartRequest { plan: String, plan_sha256: String, output: String, tile: usize, memory_mb: usize }

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrepareRequest {
    #[serde(default)] path: String,
    #[serde(default)] text: String,
    output: String,
    memory_mb: usize,
    #[serde(default)] exclude: Vec<String>,
    /// Copy sources beside the result first; see `mosaic prepare --stage-dir`.
    #[serde(default)] stage: bool,
}

pub(super) fn prepare(body: &str, exe: &Path, job: &Arc<Mutex<Job>>) -> Result<String> {
    let req: PrepareRequest = serde_json::from_str(body)?;
    ensure!(!req.output.trim().is_empty(), "Choose a new result folder first; preparation is saved beside it");
    ensure!((128..=65536).contains(&req.memory_mb), "Preparation needs a memory budget of at least 128 MiB");
    let output = std::path::absolute(req.output.trim())?;
    ensure!(!output.try_exists()?, "Result folder already exists; choose a new folder name");
    let parent = output.parent().context("Result folder needs a parent directory")?;
    let stem = output.file_name().context("Choose a result folder name")?.to_string_lossy();
    let preparation = workflows::unique_child(parent, &format!("{stem}-preparation"));
    let mut input = resolve_input(&InspectRequest { path:req.path, text:req.text,
        name:preparation.file_name().unwrap_or_default().to_string_lossy().into_owned() })?;
    if !req.exclude.is_empty() {
        let excluded = req.exclude.iter().map(std::fs::canonicalize).collect::<std::io::Result<std::collections::HashSet<_>>>()?;
        let selected = crate::mosaic_prepare::collect(&input)?;
        let text = selected.iter().filter(|p| !excluded.contains(*p))
            .map(|p| format!("{}\n", p.display())).collect::<String>();
        input = list_from_text(&text, "mosaic-selected.txt")?;
    }
    let mut args = vec!["mosaic".into(),"prepare".into(),input.to_string_lossy().into_owned(),
        "--output".into(),preparation.to_string_lossy().into_owned(),"--memory-mb".into(),req.memory_mb.to_string()];
    if req.stage {
        args.extend(["--stage-dir".into(), parent.join(".smokstak-mosaic-sources").to_string_lossy().into_owned()]);
    }
    let planned = Planned {
        step:"Find panels, align stars and match shared sky".into(), program:exe.to_owned(),
        args,
        kind:"mosaic_prepare", last_stage:2,
    };
    start_plan(vec![planned], job, format!("Prepare mosaic · {stem}"), None)
}

pub(super) fn preparation_completed(job: &mut Job, args: &[String]) -> Result<()> {
    let directory = args.windows(2).find(|a|a[0]=="--output").context("Missing preparation directory")?;
    let path = Path::new(&directory[1]).join("plan.json");
    let memory = args.windows(2).find(|a|a[0]=="--memory-mb").context("Missing memory budget")?[1].parse()?;
    review(&path.to_string_lossy(),256,memory)?;
    job.progress.preparation_detail = Some("Layout ready".into());
    job.progress.prepared_plan = Some(path.to_string_lossy().into_owned());
    job.announce(path.to_string_lossy().into_owned());
    Ok(())
}

pub(super) fn preparation_progress(p: &mut Progress, line: &str) {
    fn counts(text: &str, separator: &str) -> Option<(usize,usize)> {
        let (done,total)=text.trim().split_once(separator)?;
        let (done,total)=(done.parse::<usize>().ok()?,total.parse::<usize>().ok()?);
        (done>0 && done<=total && total<=1_000_000).then_some((done,total))
    }
    fn count(text: &str) -> Option<usize> {
        let n=text.parse::<usize>().ok()?;
        (n>0 && n<=1_000_000).then_some(n)
    }
    let line=line.trim();
    if p.stage==0 {
        if let Some(rest)=line.strip_prefix("mosaic prepare catalog ") {
            if let Some((done,total))=rest.split_once(':').and_then(|(numbers,_)|counts(numbers,"/")) {
                p.preparation_detail=Some(format!("Detecting stars · frame {done} of {total}"));
            }
        }
    }
    if let Some(detail)=line.strip_prefix("mosaic prepare geometry:") {
        if p.stage>1 { return; }
        p.stage=1;
        let detail=detail.trim();
        let message=if let Some((done,total))=detail.strip_prefix("Identifying panel ").and_then(|s|counts(s," of ")) {
            Some(format!("Finding panels · frame {done} of {total}"))
        } else if let Some(n)=detail.strip_prefix("Connecting ").and_then(|s|s.strip_suffix(" detected panels")).and_then(count) {
            Some(format!("Connecting {n} panels"))
        } else if detail=="Refining shared optics and panel geometry" {
            Some("Refining alignment and optical differences".into())
        } else if let Some((done,total))=detail.strip_prefix("Validating exposure ").and_then(|s|counts(s," of ")) {
            Some(format!("Checking alignment · frame {done} of {total}"))
        } else if detail=="finding panels and registering overlaps" {
            Some("Finding panels and aligning stars".into())
        } else { None };
        p.preparation_detail=Some(message.unwrap_or_else(||"Finding panels and aligning stars".into()));
        return;
    }
    if line.starts_with("mosaic prepare photometry:") {
        p.stage=2;
        p.preparation_detail=Some("Measuring brightness and shared sky".into());
        return;
    }
    let detail=if let Some((done,total))=line.strip_prefix("Measuring native stellar flux and noise: frame ").and_then(|s|counts(s,"/")) {
        Some(format!("Measuring brightness and noise · frame {done} of {total}"))
    } else if let Some(n)=line.strip_prefix("Solving relative stellar fluxes across ").and_then(|s|s.strip_suffix(" supported overlaps")).and_then(count) {
        Some(format!("Matching brightness across {n} overlaps"))
    } else if let Some((done,total))=line.strip_prefix("Measuring same-sky overlap: ").and_then(|s|counts(s,"/")) {
        Some(format!("Measuring shared sky · overlap {done} of {total}"))
    } else if line=="Solving overlap-only additive planes with independent validation" {
        Some("Matching sky gradients and validating overlaps".into())
    } else { None };
    if let Some(detail)=detail { p.stage=2; p.preparation_detail=Some(detail); }
}

pub(super) fn pick_plan() -> Result<String> {
    ensure!(!PICKING.swap(true, Ordering::SeqCst), "A file or folder chooser is already open");
    let result = plan_chooser().and_then(|mut c| Ok(c.output()?));
    PICKING.store(false, Ordering::SeqCst);
    let output = result?;
    ensure!(output.status.success() || output.status.code() == Some(1), "Cannot open file chooser; paste the plan path instead");
    Ok(serde_json::json!({"ok":true,"path":folder_from(&output.stdout)}).to_string())
}

fn plan_chooser() -> Result<Command> {
    #[cfg(windows)] {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("powershell");
        c.creation_flags(0x08000000);
        c.args(["-NoProfile","-STA","-Command",
            "Add-Type -AssemblyName System.Windows.Forms; [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; $d = New-Object System.Windows.Forms.OpenFileDialog; $d.Title = 'Choose a prepared mosaic plan'; $d.Filter = 'JSON plans (*.json)|*.json'; $d.CheckFileExists = $true; if ($d.ShowDialog() -eq 'OK') { [Console]::Out.Write($d.FileName) }"]);
        Ok(c)
    }
    #[cfg(target_os = "macos")] {
        let mut c = Command::new("osascript");
        c.args(["-e","try\nPOSIX path of (choose file with prompt \"Choose a prepared mosaic plan\" of type {\"public.json\"})\nend try"]); Ok(c)
    }
    #[cfg(all(unix, not(target_os = "macos")))] {
        for (exe,args) in [("zenity",vec!["--file-selection","--title=Choose a prepared mosaic plan","--file-filter=JSON plans | *.json"]),
            ("kdialog",vec!["--getopenfilename",".","*.json|JSON plans"])] {
            if which(exe) { let mut c=Command::new(exe); c.args(args); return Ok(c); }
        }
        anyhow::bail!("No file chooser available; paste the plan path instead")
    }
}

fn review(path: &str, tile: usize, memory_mb: usize) -> Result<(PathBuf, crate::mosaic::Plan, String)> {
    ensure!(!path.trim().is_empty(), "Choose a prepared mosaic plan");
    let path = std::fs::canonicalize(path).context("Cannot open mosaic plan")?;
    let (plan, fingerprint) = crate::mosaic::read_review_plan_fingerprinted(&path, tile, memory_mb)?;
    for frame in &plan.frames {
        let meta = std::fs::metadata(&frame.path).with_context(|| format!("Source unavailable: {}", frame.path.display()))?;
        ensure!(meta.is_file() && meta.len() == frame.bytes, "Source size changed: {}", frame.path.display());
    }
    Ok((path, plan, fingerprint))
}

pub(super) fn inspect(body: &str) -> Result<String> {
    let req: ReviewRequest = serde_json::from_str(body)?;
    let (path, plan, fingerprint) = review(&req.plan, req.tile, req.memory_mb)?;
    let cohorts = crate::mosaic::global_psf_cohort_representatives(&plan.frames.iter().map(|f| f.psf_hfd).collect::<Vec<_>>());
    let frames = plan.frames.iter().enumerate().map(|(index, f)| -> Result<_> {
        let mut footprint = Vec::with_capacity(32);
        for edge in 0..4 { for i in 0..8 {
            let t = i as f64 / 8.0;
            let w = f.width as f64; let h = f.height as f64;
            let (x,y) = match edge { 0 => (t*w,0.0), 1 => (w,t*h), 2 => ((1.0-t)*w,h), _ => (0.0,(1.0-t)*h) };
            let (x,y) = f.projection.map(x-0.5,y-0.5).context("Invalid projected footprint")?;
            footprint.push([(x-f64::from(plan.grid.origin[0])+0.5)*f64::from(plan.grid.scale)-0.5,
                (y-f64::from(plan.grid.origin[1])+0.5)*f64::from(plan.grid.scale)-0.5]);
        }}
        Ok(serde_json::json!({"index":index,"path":f.path,"label":f.label,"group":f.group,"psf_cohort":cohorts[index],"psf_hfd":f.psf_hfd,"registration_p50":f.registration_p50,
            "registration_p90":f.registration_p90,"validation_stars":f.validation_stars,"footprint":footprint}))
    }).collect::<Result<Vec<_>>>()?;
    Ok(serde_json::json!({"ok":true,"plan":path,"plan_sha256":fingerprint,"filter":plan.filter,"calibration":plan.calibration,
        "width":plan.grid.width,"height":plan.grid.height,"scale":plan.grid.scale,"frames":frames,"notes":plan.notes}).to_string())
}

fn planned(req: StartRequest, exe: &Path) -> Result<Planned> {
    ensure!(!req.output.trim().is_empty(), "Choose a new output directory");
    let output = std::path::absolute(&req.output)?;
    ensure!(!output.try_exists()?, "Output already exists; choose a new directory");
    ensure!(output.file_name().is_some(), "Output needs a directory name");
    let (path, _, fingerprint) = review(&req.plan, req.tile, req.memory_mb)?;
    ensure!(req.plan_sha256 == fingerprint, "Mosaic plan changed since inspection; review it again before starting");
    Ok(Planned { step: "Experimental mono mosaic".into(), program: exe.to_path_buf(),
        args: vec!["mosaic".into(),"build".into(),path.to_string_lossy().into_owned(),"--output".into(),
            output.to_string_lossy().into_owned(),"--tile".into(),req.tile.to_string(),"--memory-mb".into(),req.memory_mb.to_string(),
            "--plan-sha256".into(),fingerprint,"--experimental".into()],
        kind:"mosaic",last_stage:2 })
}

pub(super) fn start(body: &str, exe: &Path, job: &Arc<Mutex<Job>>) -> Result<String> {
    let req: StartRequest = serde_json::from_str(body)?;
    let title = format!("Mosaic · {}", Path::new(&req.output).file_name().unwrap_or_default().to_string_lossy());
    start_plan(vec![planned(req, exe)?], job, title, None)
}

pub(super) fn progress(p: &mut Progress, line: &str) {
    if let Some(rest) = line.split("mosaic tile ").nth(1) {
        if let Some((done,total)) = rest.split(',').next().unwrap_or("").split_once('/') {
            if let (Ok(done),Ok(total)) = (done.parse::<usize>(),total.parse::<usize>()) {
                if total > 0 && done <= total && done >= p.tiles_done.unwrap_or(0) {
                    p.stage = p.stage.max(1); p.tiles_done = Some(done); p.tiles_total = Some(total);
                }
            }
        }
    }
    // Publication is confirmed by successful process completion, never a log line.
}

fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    ensure!(file.metadata()?.len() <= limit, "Mosaic asset exceeds serving limit");
    let mut bytes = Vec::new(); file.take(limit+1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "Mosaic asset exceeds serving limit");
    Ok(bytes)
}

pub(super) fn completed(job: &mut Job, args: &[String]) -> Result<()> {
    let output = args.windows(2).find(|a| a[0] == "--output").map(|a| Path::new(&a[1]))
        .context("Mosaic completion has no output directory")?;
    // A failed/killed child never calls this; partial staging paths are never recorded.
    ensure!(OUTPUTS.iter().all(|name| output.join(name).is_file()), "Mosaic exited successfully but its published output set is incomplete");
    let value: serde_json::Value = serde_json::from_slice(&bounded_read(&output.join("result.json"), 2*1024*1024)?)?;
    let coverage = &value["coverage"];
    let supported = coverage["supported_pixels"].as_u64().context("Mosaic result has no supported-pixel count")?;
    let gaps = coverage["unsupported_pixels"].as_u64().context("Mosaic result has no gap-pixel count")?;
    let mut coverage_text = format!("{supported} supported pixels; {gaps} gap pixels.");
    if let Some(minimum) = coverage["minimum_effective_frames"].as_f64() {
        coverage_text.push_str(&format!(" Minimum effective frames: {minimum:.2}."));
    }
    let limits = value["limitations"].as_array().context("Mosaic result has no limitations")?
        .iter().map(|v| v.as_str().context("Invalid mosaic limitation")).collect::<Result<Vec<_>>>()?
        .into_iter().map(|s| format!("{}.",s.trim().trim_end_matches('.'))).collect::<Vec<_>>().join(" ");
    for name in OUTPUTS {
        let path = output.join(name);
        job.mosaic_outputs.insert(path.clone());
        job.mosaic_outputs.insert(std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone()));
        job.announce(path.to_string_lossy().into_owned());
    }
    for (label,text) in [("Coverage",coverage_text),("Experimental limits",limits)] {
        job.progress.findings.push(Finding { group:String::new(),label:label.into(),text });
    }
    Ok(())
}

pub(super) fn is_output(job: &Job, path: &Path) -> bool {
    job.mosaic_outputs.contains(path)
}

pub(super) fn serve(path: &Path) -> Result<(String,Vec<u8>)> {
    let kind = match path.file_name().and_then(|n| n.to_str()) {
        Some("preview.png") => "image/png",
        Some("plan.json" | "result.json") => "application/json",
        _ => anyhow::bail!("Scientific mosaic outputs must be opened in the file manager"),
    };
    Ok((kind.into(),bounded_read(path,ASSET_LIMIT)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preparation_details_show_current_phase_and_counts_without_paths() {
        let mut p=Progress::default();
        preparation_progress(&mut p,"mosaic prepare catalog 7/134: C:/private/<script>.fits");
        assert_eq!(p.stage,0);
        assert_eq!(p.preparation_detail.as_deref(),Some("Detecting stars · frame 7 of 134"));
        preparation_progress(&mut p,"mosaic prepare geometry: finding panels and registering overlaps");
        assert_eq!(p.stage,1);
        assert_eq!(p.preparation_detail.as_deref(),Some("Finding panels and aligning stars"));
        for (line,want) in [
            ("Identifying panel 32 of 134","Finding panels · frame 32 of 134"),
            ("Connecting 16 detected panels","Connecting 16 panels"),
            ("Refining shared optics and panel geometry","Refining alignment and optical differences"),
            ("Validating exposure 41 of 134","Checking alignment · frame 41 of 134"),
        ] {
            preparation_progress(&mut p,&format!("mosaic prepare geometry: {line}"));
            assert_eq!(p.preparation_detail.as_deref(),Some(want));
        }
        preparation_progress(&mut p,"mosaic prepare photometry: measuring stars and shared sky");
        assert_eq!(p.stage,2);
        assert_eq!(p.preparation_detail.as_deref(),Some("Measuring brightness and shared sky"));
        for (line,want) in [
            ("Measuring native stellar flux and noise: frame 19/134","Measuring brightness and noise · frame 19 of 134"),
            ("Solving relative stellar fluxes across 142 supported overlaps","Matching brightness across 142 overlaps"),
            ("Measuring same-sky overlap: 28/142","Measuring shared sky · overlap 28 of 142"),
            ("Solving overlap-only additive planes with independent validation","Matching sky gradients and validating overlaps"),
        ] {
            preparation_progress(&mut p,line);
            assert_eq!(p.preparation_detail.as_deref(),Some(want));
        }
        preparation_progress(&mut p,"mosaic prepare catalog 134/134: old.fits");
        preparation_progress(&mut p,"mosaic prepare geometry: Validating exposure 134 of 134");
        assert_eq!(p.stage,2);
        assert_eq!(p.preparation_detail.as_deref(),Some("Matching sky gradients and validating overlaps"));
        assert!(p.prepared_plan.is_none());
        assert!(p.outputs.is_empty());
    }
    #[test]
    fn preparation_count_parser_ignores_invalid_counts_and_unrelated_lines() {
        let mut p=Progress::default();
        preparation_progress(&mut p,"mosaic prepare catalog 2/10: frame.fits");
        for line in ["mosaic prepare catalog 11/10: bad.fits", "mosaic prepare catalog 0/10: bad.fits",
            "mosaic prepare catalog 1/0: bad.fits", "mosaic prepare catalog 1/1000001: bad.fits",
            "mosaic prepare catalog NaN/10: bad.fits", "mosaic prepare catalog: 6000 usable stars",
            "Measuring native stellar flux and noise: frame 7/3", "Measuring same-sky overlap: 0/8",
            "Solving relative stellar fluxes across 0 supported overlaps", "<b>random message</b>"] {
            preparation_progress(&mut p,line);
        }
        assert_eq!(p.stage,0);
        assert_eq!(p.preparation_detail.as_deref(),Some("Detecting stars · frame 2 of 10"));
    }
    #[cfg(windows)]
    #[test]
    fn plan_chooser_uses_only_fixed_script_arguments() {
        let c = plan_chooser().unwrap();
        let args: Vec<_> = c.get_args().map(|a| a.to_string_lossy()).collect();
        assert_eq!(&args[..3], &["-NoProfile","-STA","-Command"]);
        assert!(args[3].contains("OpenFileDialog"));
        assert!(args[3].contains("*.json"));
    }
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = workflows::unique_child(&std::env::temp_dir(), "mosaic gui spaces");
            std::fs::create_dir(&path).unwrap(); Self(path)
        }
        fn plan(&self) -> PathBuf {
            let frames = (0..2).map(|i| {
                let path = self.0.join(format!("source {i}.fits"));
                std::fs::write(&path, "not decoded during review").unwrap();
                serde_json::json!({"path":path,"width":100,"height":80,"bytes":25,"sha256":"a".repeat(64),
                    "projection":{"center":[0.,0.],"normalization_scale":1.,"distortion":[0.,0.,0.],
                    "homography":[1.,0.,0.,0.,1.,0.,0.,0.,1.],"output_center":[0.,0.],"output_scale":1.},
                    "noise":sr_core::NoiseModel::new(0.,0.001,sr_core::NoiseSource::Manual),"sky":0.1,"gain":1.,"offset":0.,
                    "weight":1.,"psf_hfd":2.,"registration_p50":0.1,"registration_p90":0.2,"validation_stars":40})
            }).collect::<Vec<_>>();
            let path = self.0.join("prepared plan.json");
            std::fs::write(&path,serde_json::json!({"version":1,"filter":"Ha","calibration":"uncalibrated",
                "grid":{"origin":[0.,0.],"width":100,"height":80,"scale":1.},"frames":frames,"notes":[]}).to_string()).unwrap(); path
        }
    }
    impl Drop for Fixture { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
    #[test]
    fn review_and_command_preserve_paths_and_footprints() {
        let fixture = Fixture::new(); let path = fixture.plan();
        let body = serde_json::json!({"plan":path,"tile":64,"memory_mb":64}).to_string();
        let value: serde_json::Value = serde_json::from_str(&inspect(&body).unwrap()).unwrap();
        assert_eq!(value["frames"][0]["footprint"][0],serde_json::json!([-0.5,-0.5]));
        assert_eq!(value["frames"][0]["footprint"].as_array().unwrap().len(),32);
        let req = StartRequest { plan:path.to_string_lossy().into_owned(),plan_sha256:value["plan_sha256"].as_str().unwrap().into(),output:fixture.0.join("new output").to_string_lossy().into_owned(),tile:64,memory_mb:64 };
        let p = planned(req,Path::new("program with spaces.exe")).unwrap();
        assert_eq!(p.args[2], std::fs::canonicalize(path).unwrap().to_string_lossy());
        assert!(p.args[4].ends_with("new output"));
        assert_eq!(p.command().get_args().count(),12);
        assert_eq!(p.args[9],"--plan-sha256");
        assert_eq!(p.args[10],value["plan_sha256"].as_str().unwrap());
        assert_eq!(p.args.last().unwrap(),"--experimental");
    }
    #[test]
    fn rejects_existing_output_invalid_requests_and_source_size() {
        let fixture = Fixture::new(); let path = fixture.plan();
        assert!(planned(StartRequest {plan:path.to_string_lossy().into_owned(),plan_sha256:String::new(),output:fixture.0.to_string_lossy().into_owned(),tile:64,memory_mb:64},Path::new("exe")).err().unwrap().to_string().contains("already exists"));
        for body in ["{}",r#"{"plan":"","tile":64,"memory_mb":64}"#] { assert!(inspect(body).is_err()); }
        std::fs::write(fixture.0.join("source 0.fits"),"changed").unwrap();
        assert!(review(path.to_str().unwrap(),64,64).err().unwrap().to_string().contains("size changed"));
    }
    #[test]
    fn progress_is_monotonic_and_publication_is_not_a_log_event() {
        let mut p = Progress::default();
        progress(&mut p,"log: mosaic tile 7/20, 1.2s");
        progress(&mut p,"mosaic tile 3/20, 1.3s");
        progress(&mut p,"mosaic tile 99/20, 1.4s");
        progress(&mut p,"Mosaic evaluation output: fake");
        assert_eq!((p.stage,p.tiles_done,p.tiles_total),(1,Some(7),Some(20)));
        assert!(p.outputs.is_empty());
    }
    #[test]
    fn preparation_only_publishes_a_completed_valid_plan() {
        let fixture=Fixture::new(); let path=fixture.plan();
        let args=vec!["--output".into(),fixture.0.to_string_lossy().into_owned(),"--memory-mb".into(),"2048".into()];
        let mut job=Job::default();
        preparation_progress(&mut job.progress,"Prepared mosaic plan: arbitrary.json");
        assert!(job.progress.prepared_plan.is_none());
        assert!(preparation_completed(&mut job,&args).is_err());
        assert!(job.progress.outputs.is_empty());
        std::fs::rename(path,fixture.0.join("plan.json")).unwrap();
        preparation_completed(&mut job,&args).unwrap();
        assert_eq!(job.progress.outputs.len(),1);
        assert_eq!(job.progress.prepared_plan.as_ref(),job.progress.outputs.first());
    }
    #[test]
    fn changed_review_plan_is_refused_before_start() {
        let fixture = Fixture::new(); let path = fixture.plan();
        let (_,_,fingerprint) = review(path.to_str().unwrap(),64,64).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["notes"] = serde_json::json!(["changed after review"]);
        std::fs::write(&path,value.to_string()).unwrap();
        let error = planned(StartRequest {plan:path.to_string_lossy().into_owned(),plan_sha256:fingerprint,
            output:fixture.0.join("new output").to_string_lossy().into_owned(),tile:64,memory_mb:64},Path::new("exe")).err().unwrap();
        assert!(error.to_string().contains("changed since inspection"));
        assert!(!fixture.0.join("new output").exists());
    }
    #[test]
    fn only_complete_fixed_outputs_are_published_and_assets_are_bounded() {
        let fixture = Fixture::new(); let args = vec!["--output".into(),fixture.0.to_string_lossy().into_owned()];
        let mut job = Job::default(); job.progress.kind = "mosaic".into();
        for name in &OUTPUTS[..5] { std::fs::write(fixture.0.join(name),"data").unwrap(); }
        assert!(completed(&mut job,&args).is_err()); assert!(job.progress.outputs.is_empty());
        std::fs::write(fixture.0.join("result.json"),"{}").unwrap();
        assert!(completed(&mut job,&args).is_err()); assert!(job.progress.outputs.is_empty());
        std::fs::write(fixture.0.join("result.json"),r#"{"coverage":{"supported_pixels":4,"unsupported_pixels":2,"minimum_effective_frames":1.5},"limitations":["Experimental","Uncalibrated."]}"#).unwrap();
        std::fs::write(fixture.0.join("unexpected.png"),"data").unwrap();
        completed(&mut job,&args).unwrap(); assert_eq!(job.progress.outputs.len(),6); assert_eq!(job.progress.findings.len(),2);
        assert_eq!(job.progress.findings[0].text,"4 supported pixels; 2 gap pixels. Minimum effective frames: 1.50.");
        assert_eq!(job.progress.findings[1].text,"Experimental. Uncalibrated.");
        assert!(serve(&fixture.0.join("image.fits")).is_err());
        assert!(serve(&fixture.0.join("preview.png")).is_ok());
        let arc = Arc::new(Mutex::new(job));
        assert!(serve_file(&format!("path={}",fixture.0.join("image.fits").display()),&arc).is_err());
        assert!(serve_file(&format!("path={}",fixture.0.join("unexpected.png").display()),&arc).is_err());
        std::fs::File::create(fixture.0.join("preview.png")).unwrap().set_len(ASSET_LIMIT+1).unwrap();
        assert!(serve(&fixture.0.join("preview.png")).is_err());
        let mut restored = Job::default(); restore(&mut restored,vec![arc.lock().unwrap().progress.clone()]);
        assert!(is_output(&restored,&fixture.0.join("image.fits")));
        let restored = Arc::new(Mutex::new(restored));
        assert!(serve_file(&format!("path={}",fixture.0.join("result.json").display()),&restored).is_ok());
        assert!(serve_file(&format!("path={}",fixture.0.join("image.fits").display()),&restored).is_err());
    }
}
