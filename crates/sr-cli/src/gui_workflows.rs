//! Thin GUI adapters for the existing analysis and versioned-project commands.
use super::*;

#[derive(Deserialize)]
struct WorkflowRequest {
    action: String,
    #[serde(default)]
    input: String,
    directory: String,
    #[serde(default)]
    reference: String,
    #[serde(default)]
    list: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    workers: usize,
    #[serde(default)]
    exclude: Vec<String>,
}

pub(super) fn unique_child(parent: &Path, label: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_nanos();
    parent.join(format!("{label}-{}-{stamp}", std::process::id()))
}

fn plan(req: &WorkflowRequest, exe: &Path) -> Result<Vec<Planned>> {
    anyhow::ensure!(!req.directory.trim().is_empty(), "Choose an output or project folder");
    let directory = req.directory.trim().to_string();
    let workers = if req.workers == 0 { 2 } else { req.workers };
    anyhow::ensure!(workers <= 256, "Use between 1 and 256 workers");
    let item = |args, step: &str, kind| Planned {
        step: step.into(), program: exe.to_path_buf(), args, last_stage: STAGES.len()-1, kind,
    };
    let require = |value: &str, name: &str| -> Result<String> {
        anyhow::ensure!(!value.trim().is_empty(), "{name} is required");
        Ok(value.trim().into())
    };
    if req.action == "analyze" {
        let input = require(&req.input, "Input frames")?;
        let output = unique_child(Path::new(&directory), "analysis");
        return Ok(vec![item(vec!["analyze".into(), input,
            "--html".into(), output.join("project-analysis.html").to_string_lossy().into_owned(),
            "--json".into(), output.join("project-analysis.json").to_string_lossy().into_owned(),
            "--cache-dir".into(), Path::new(&directory).join("analysis-cache").to_string_lossy().into_owned(),
            "--threads".into(), workers.to_string()], "Measure noise and integration", "analysis")]);
    }
    let mut result = Vec::new();
    let mut args = vec!["project".into()];
    match req.action.as_str() {
        "create" => args.extend(["init".into(), directory.clone(), require(&req.input, "Input frames")?,
            "--reference-file".into(), require(&req.reference, "Reference frame")?]),
        "add" | "relink" => args.extend([req.action.clone(), directory.clone(), require(&req.input, "Input frames")?]),
        "exclude" | "include" => {
            args.extend([req.action.clone(), directory.clone(), require(&req.list, "Review list")?]);
            if req.action == "exclude" { args.extend(["--reason".into(), require(&req.reason, "Exclusion reason")?]); }
        }
        "status" => args.extend(["status".into(), directory.clone()]),
        "build" => {},
        _ => anyhow::bail!("Unknown workflow action"),
    }
    if req.action != "build" { result.push(item(args, &format!("Project: {}", req.action), "project")); }
    if req.action != "status" {
        result.push(item(vec!["project".into(), "build".into(), directory,
            "--workers".into(), workers.to_string()], "Rebuild all selected exposures", "project"));
    }
    Ok(result)
}

pub(super) fn start(body: &str, exe: &Path, job: &Arc<Mutex<Job>>) -> Result<String> {
    let mut req: WorkflowRequest = serde_json::from_str(body)?;
    {
        let j = job.lock().unwrap();
        anyhow::ensure!(!j.progress.running && !j.survey.running, "Stop or finish the current run first");
    }
    if req.action == "analyze" && !req.exclude.is_empty() {
        req.input = without(&req.input, &req.exclude)?.0.to_string_lossy().into_owned();
    }
    let commands = plan(&req, exe)?;
    let title = format!("{} · {}", if req.action == "analyze" { "Noise analysis" } else { "Mono project" }, req.directory);
    start_plan(commands, job, title, None)
}

pub(super) fn is_report(path: &Path) -> bool {
    path.is_file() && matches!(path.file_name().and_then(|v| v.to_str()),
        Some("frame-review.html" | "project-analysis.html"))
}

fn add_audit(j: &mut Job, directory: &Path) {
    let page = directory.join("frame-review.html");
    if is_report(&page) { j.announce(page.to_string_lossy().into_owned()); }
}

fn published(j: &mut Job, directory: &Path) {
    // Pending output paths are intentionally not published by the GUI. Only
    // the CLI's successful, atomically published run can supply these masters.
    if !directory.join("project-build.json").is_file() { return; }
    add_audit(j, directory);
    if let Ok(entries) = std::fs::read_dir(directory) {
        let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| {
            p.is_file() && p.file_name().unwrap_or_default().to_string_lossy().starts_with("master")
                && matches!(p.extension().and_then(|v|v.to_str()), Some("fits"|"tif"|"png"))
        }).collect();
        paths.sort();
        for path in paths { j.announce(path.to_string_lossy().into_owned()); }
    }
}

pub(super) fn record_outputs(j: &mut Job, line: &str, kind: &str) {
    if kind == "project" {
        if let Some(path) = line.strip_prefix("Frame review: ") {
            if let Some(directory) = Path::new(path.trim()).parent() { published(j, directory); }
        }
        return;
    }
    if let Some(path) = announced_path(line) { j.announce(path.into()); }
    if let Some(path) = line.strip_prefix("Report: ").or_else(||line.strip_prefix("Data: ")) {
        j.announce(path.trim().into());
    }
    if let Some(directory) = line.strip_prefix("Diagnostics written to ") {
        let directory = Path::new(directory.trim());
        add_audit(j, directory);
        if let Ok(entries) = std::fs::read_dir(directory) {
            for entry in entries.flatten().filter(|e| e.path().is_dir()) { add_audit(j, &entry.path()); }
        }
    }
}

pub(super) fn completed(j: &mut Job, kind: &str, args: &[String]) {
    if kind == "project" && args.get(1).map(String::as_str) == Some("status") {
        if let Some(directory) = args.get(2) {
            if let Ok(entries) = std::fs::read_dir(Path::new(directory).join("runs")) {
                let mut runs: Vec<_> = entries.flatten().map(|e|e.path())
                    .filter(|p| p.file_name().unwrap_or_default().to_string_lossy().chars().all(|c|c.is_ascii_digit())
                        && p.join("project-build.json").is_file()).collect();
                runs.sort();
                if let Some(run) = runs.last() { published(j, run); }
            }
        }
    }
}

fn report_to_open(body: &str, job: &Arc<Mutex<Job>>) -> Result<PathBuf> {
    let req: RevealRequest = serde_json::from_str(body)?;
    let path = std::fs::canonicalize(req.path)?;
    anyhow::ensure!(is_report(&path) && job.lock().unwrap().served.contains(&path),
        "Only a report from a recorded run can be opened");
    Ok(path)
}

pub(super) fn open_report(body: &str, job: &Arc<Mutex<Job>>) -> Result<String> {
    let path = report_to_open(body, job)?;
    // Open locally so the report's relative lazy assets remain usable. Do not
    // expose a directory tree through HTTP or assemble a shell command.
    #[cfg(windows)]
    Command::new("explorer.exe").arg(&path).spawn()?;
    #[cfg(target_os = "macos")]
    Command::new("open").arg(&path).spawn()?;
    #[cfg(all(unix, not(target_os = "macos")))]
    Command::new("xdg-open").arg(&path).spawn()?;
    Ok(serde_json::json!({"ok":true}).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(action: &str) -> WorkflowRequest {
        serde_json::from_value(serde_json::json!({"action":action,"input":"data with spaces.txt",
            "directory":"my project","reference":"reference.fit","list":"review.txt","reason":"Cloud"})).unwrap()
    }
    #[test]
    fn updates_and_review_rebuild_the_combined_population() {
        for action in ["create","add","exclude","include","relink","build"] {
            let commands=plan(&request(action),Path::new("smokstak")).unwrap();
            assert_eq!(&commands.last().unwrap().args, &["project","build","my project","--workers","2"]);
            assert!(!commands.iter().any(|c| c.args.contains(&"--accumulate".into())));
        }
        let mut bad=request("exclude");bad.reason.clear();assert!(plan(&bad,Path::new("exe")).is_err());
        assert!(plan(&request("arbitrary-command"),Path::new("exe")).is_err());
    }
    #[test]
    fn analysis_preserves_existing_reports_and_argument_boundaries() {
        let first=plan(&request("analyze"),Path::new("exe")).unwrap();
        let second=plan(&request("analyze"),Path::new("exe")).unwrap();
        assert_eq!(first[0].args[1],"data with spaces.txt");
        assert_ne!(first[0].args[3],second[0].args[3]);
        assert!(first[0].args[3].ends_with("project-analysis.html"));
    }
    #[test]
    fn report_access_and_publication_are_explicit() {
        let dir=unique_child(&std::env::temp_dir(),"gui-reports");std::fs::create_dir_all(&dir).unwrap();
        let page=dir.join("frame-review.html");std::fs::write(&page,"report").unwrap();
        let job=Arc::new(Mutex::new(Job::default()));
        let body=serde_json::json!({"path":page}).to_string();
        assert!(report_to_open(&body,&job).is_err());
        record_outputs(&mut job.lock().unwrap(),&format!("Wrote {}",page.display()),"project");
        assert!(job.lock().unwrap().progress.outputs.is_empty());
        record_outputs(&mut job.lock().unwrap(),&format!("Diagnostics written to {}",dir.display()),"stack");
        assert!(report_to_open(&body,&job).is_ok());
        let history=job.lock().unwrap().progress.clone();
        let mut restored=Job::default();restore(&mut restored,vec![history]);
        assert!(restored.served.contains(&std::fs::canonicalize(&page).unwrap()));
        std::fs::remove_file(&page).unwrap();std::fs::remove_dir(&dir).unwrap();
    }
}
