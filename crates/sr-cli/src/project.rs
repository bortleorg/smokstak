//! Versioned projects use the ordinary production stacker over the complete
//! selected population. Reuse never freezes decisions from an earlier batch.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use clap::{Args as ClapArgs, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sr_core::config::ReconstructionConfig;

use crate::pipeline::{self, FrameSelect, InputSpec};

#[derive(ClapArgs, Debug)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Create a project from compatible, calibrated monochrome exposures.
    Init {
        directory: PathBuf,
        input: PathBuf,
        /// Pin the geometry to this original, even if it is later excluded.
        #[arg(long)]
        reference_file: PathBuf,
        /// Full ReconstructionConfig JSON; defaults to the production pipeline at 1x.
        #[arg(long)]
        recipe: Option<PathBuf>,
        #[arg(long, default_value = "auto")]
        fits_row_order: String,
    },
    /// Add new content; renamed duplicate exposures are not counted again.
    Add { directory: PathBuf, input: PathBuf },
    /// Relocate known originals by full content identity, preserving review decisions.
    Relink { directory: PathBuf, input: PathBuf },
    /// Preserve a reason for excluding listed paths or full SHA256 identities.
    Exclude {
        directory: PathBuf,
        list: PathBuf,
        #[arg(long)]
        reason: String,
    },
    /// Restore listed paths or full SHA256 identities to the selected population.
    Include { directory: PathBuf, list: PathBuf },
    /// Rebuild all selected contributions with global production decisions.
    Build {
        directory: PathBuf,
        #[arg(long, default_value_t = 2)]
        workers: usize,
        /// Use resident decoded samples, for comparison against the disk-backed path.
        #[arg(long)]
        in_memory: bool,
        #[arg(long)]
        no_cache: bool,
    },
    /// Show the latest revision and preserved review decisions.
    Status { directory: PathBuf },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Frame {
    id: String,
    path: PathBuf,
    filter: String,
    exclusion_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Reference {
    id: String,
    path: PathBuf,
    width: usize,
    height: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Snapshot {
    schema: u32,
    revision: u64,
    created_unix_seconds: u64,
    operation: String,
    reference: Reference,
    fits_row_order: String,
    recipe: ReconstructionConfig,
    /// Sorted by full content identity, independent of ingest or filename order.
    frames: Vec<Frame>,
}

/// A crashed process deliberately leaves the lock for an operator to inspect.
/// Never guess that a long-running stack has died and remove its lock.
struct ProjectLock {
    file: Option<File>,
    path: PathBuf,
}

impl ProjectLock {
    fn acquire(directory: &Path) -> Result<Self> {
        let path = directory.join("project.lock");
        let file = OpenOptions::new().write(true).create_new(true).open(&path)
            .with_context(|| format!("cannot lock {}; another command may be running. Inspect an abandoned lock before removing it", path.display()))?;
        let mut lock = Self {
            file: Some(file),
            path,
        };
        let file = lock.file.as_mut().expect("new lock has a file");
        writeln!(file, "pid={} started={}", std::process::id(), now())?;
        file.sync_all()?;
        Ok(lock)
    }
}

impl Drop for ProjectLock {
    fn drop(&mut self) {
        drop(self.file.take());
        if let Err(error) = fs::remove_file(&self.path) {
            log::warn!(
                "could not release project lock {}: {error}",
                self.path.display()
            );
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_options(row: &str) -> Result<sr_raw::ReadOptions> {
    Ok(sr_raw::ReadOptions {
        fits_row_order: sr_raw::RowOrder::parse(row)
            .with_context(|| format!("invalid FITS row order: {row}"))?,
    })
}

fn validate_recipe(recipe: &ReconstructionConfig) -> Result<()> {
    ensure!(
        recipe.scale.is_finite() && recipe.scale > 0.0,
        "recipe scale must be positive and finite"
    );
    ensure!(recipe.tile > 0, "recipe tile must be nonzero");
    ensure!(
        recipe.reference.is_none(),
        "project recipes use --reference-file, not an index into a changing population"
    );
    Ok(())
}

fn canonical(path: &Path) -> Result<PathBuf> {
    fs::canonicalize(path).with_context(|| format!("resolving {}", path.display()))
}

fn validate_source(
    path: &Path,
    read: &sr_raw::ReadOptions,
    reference: Option<&Reference>,
) -> Result<(usize, usize)> {
    ensure!(
        matches!(
            sr_raw::format_of(path),
            Some(sr_raw::Format::Fits | sr_raw::Format::Xisf)
        ),
        "project inputs must be calibrated monochrome FITS or XISF: {}",
        path.display()
    );
    let frame = sr_raw::decode_with(path, read)
        .with_context(|| format!("validating {}", path.display()))?;
    ensure!(
        frame.is_mono(),
        "project currently supports monochrome inputs only: {}",
        path.display()
    );
    if let Some(reference) = reference {
        ensure!(
            (frame.width, frame.height) == (reference.width, reference.height),
            "incompatible dimensions for {}: {}x{}, reference {}x{}",
            path.display(),
            frame.width,
            frame.height,
            reference.width,
            reference.height
        );
    }
    Ok((frame.width, frame.height))
}

fn validate_filters<'a>(filters: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut spellings = BTreeMap::new();
    for filter in filters {
        ensure!(
            !filter.is_empty()
                && filter
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "_-+".contains(c)),
            "filter name must contain only letters, digits, underscore, plus or hyphen: {filter:?}"
        );
        let folded = filter.to_ascii_uppercase();
        let reserved = matches!(folded.as_str(), "CON" | "NUL" | "PRN" | "AUX")
            || ["COM", "LPT"].iter().any(|prefix| {
                folded
                    .strip_prefix(prefix)
                    .is_some_and(|n| n.len() == 1 && matches!(n.as_bytes()[0], b'1'..=b'9'))
            });
        ensure!(
            !reserved,
            "filter name {filter:?} is a reserved Windows directory name"
        );
        if let Some(previous) = spellings.insert(folded, filter) {
            ensure!(previous == filter,
                "filter names {previous:?} and {filter:?} collide on case-insensitive filesystems; use one consistent filter name");
        }
    }
    Ok(())
}

fn ingest(input: &Path, snapshot: &mut Snapshot) -> Result<(usize, usize)> {
    validate_filters(snapshot.frames.iter().map(|f| f.filter.as_str()))?;
    let paths = sr_raw::collect_files(input, None)?;
    ensure!(!paths.is_empty(), "no input frames");
    let read = read_options(&snapshot.fits_row_order)?;
    let mut known: BTreeMap<String, usize> = snapshot
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| (f.id.clone(), i))
        .collect();
    let mut added = 0;
    let mut duplicate = 0;
    let total = paths.len();
    for (index, path) in paths.into_iter().enumerate() {
        if index % 10 == 0 || index + 1 == total {
            log::info!("validating project input {}/{total}: {}", index + 1, path.display());
        }
        let path = canonical(&path)?;
        let id = hash_file(&path)?;
        if known.contains_key(&id) {
            duplicate += 1;
            continue;
        }
        ensure!(!snapshot.frames.iter().any(|f| f.path == path),
            "source changed at {}; retain the original and ingest revised calibration under a new path", path.display());
        let filter = sr_raw::peek_filter(&path)
            .filter(|f| !f.trim().is_empty())
            .with_context(|| format!("missing filter metadata: {}", path.display()))?;
        // Filters become output directory names in the production pipeline.
        validate_filters(
            snapshot
                .frames
                .iter()
                .map(|f| f.filter.as_str())
                .chain(std::iter::once(filter.as_str())),
        )?;
        validate_source(&path, &read, Some(&snapshot.reference))?;
        ensure!(
            hash_file(&path)? == id,
            "source changed during ingest: {}",
            path.display()
        );
        known.insert(id.clone(), snapshot.frames.len());
        snapshot.frames.push(Frame {
            id,
            path,
            filter,
            exclusion_reason: None,
        });
        added += 1;
    }
    snapshot.frames.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((added, duplicate))
}

fn apply_relinks(
    snapshot: &mut Snapshot,
    replacements: &BTreeMap<String, PathBuf>,
) -> Result<usize> {
    ensure!(
        replacements
            .keys()
            .all(|id| id == &snapshot.reference.id || snapshot.frames.iter().any(|f| &f.id == id)),
        "relink input contains content that is not a member or reference of this project"
    );
    let mut revised = snapshot.clone();
    let mut changed = 0;
    for (id, path) in replacements {
        let mut moved = false;
        if let Some(frame) = revised.frames.iter_mut().find(|f| &f.id == id) {
            if frame.path != *path {
                frame.path = path.clone();
                moved = true;
            }
        }
        if revised.reference.id == *id && revised.reference.path != *path {
            revised.reference.path = path.clone();
            moved = true;
        }
        changed += usize::from(moved);
    }
    let unique: BTreeSet<_> = revised.frames.iter().map(|f| &f.path).collect();
    ensure!(
        unique.len() == revised.frames.len(),
        "relink would assign different exposure identities to the same source path"
    );
    ensure!(
        !revised
            .frames
            .iter()
            .any(|f| f.path == revised.reference.path && f.id != revised.reference.id),
        "relink would assign another exposure to the pinned reference path"
    );
    *snapshot = revised;
    Ok(changed)
}

fn relink(input: &Path, snapshot: &mut Snapshot) -> Result<usize> {
    let paths = sr_raw::collect_files(input, None)?;
    ensure!(!paths.is_empty(), "no relink inputs");
    let read = read_options(&snapshot.fits_row_order)?;
    let mut replacements = BTreeMap::new();
    for path in paths {
        let path = canonical(&path)?;
        let id = hash_file(&path)?;
        let member = snapshot.frames.iter().find(|f| f.id == id);
        ensure!(
            member.is_some() || snapshot.reference.id == id,
            "relink input contains unknown content: {}; use project add for new exposures",
            path.display()
        );
        validate_source(&path, &read, Some(&snapshot.reference))?;
        if let Some(frame) = member {
            ensure!(
                sr_raw::peek_filter(&path).as_deref() == Some(frame.filter.as_str()),
                "filter metadata disagrees with project for {}",
                path.display()
            );
        }
        ensure!(
            hash_file(&path)? == id,
            "source changed during relink: {}",
            path.display()
        );
        if let Some(previous) = replacements.insert(id, path.clone()) {
            ensure!(previous == path, "relink input names multiple copies of one exposure; supply one location per content identity");
        }
    }
    // Old sources need not remain online. Validate supplied replacements here;
    // the next build still checks every selected original and the reference.
    for (id, path) in &replacements {
        ensure!(
            hash_file(path)? == *id,
            "source changed before relink publication: {}",
            path.display()
        );
    }
    apply_relinks(snapshot, &replacements)
}

fn revisions(directory: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut paths = Vec::new();
    for entry in
        fs::read_dir(directory.join("revisions")).context("project has no revisions directory")?
    {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            if let Some(n) = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.parse::<u64>().ok())
            {
                paths.push((n, path));
            }
        }
    }
    paths.sort_by_key(|p| p.0);
    Ok(paths)
}

fn load(directory: &Path) -> Result<Snapshot> {
    let entries = revisions(directory)?;
    let (revision, path) = entries
        .last()
        .context("project has no committed revision")?;
    let snapshot: Snapshot = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        snapshot.schema == 1 && snapshot.revision == *revision,
        "unsupported or inconsistent project revision"
    );
    ensure!(
        !snapshot.frames.is_empty(),
        "project revision has no frames"
    );
    ensure!(
        snapshot.frames.windows(2).all(|p| p[0].id < p[1].id),
        "project membership is not uniquely ordered by content identity"
    );
    validate_recipe(&snapshot.recipe)?;
    validate_filters(snapshot.frames.iter().map(|f| f.filter.as_str()))?;
    read_options(&snapshot.fits_row_order)?;
    Ok(snapshot)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {} without overwriting", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn save(directory: &Path, snapshot: &Snapshot) -> Result<()> {
    let revisions = directory.join("revisions");
    fs::create_dir_all(&revisions)?;
    let target = revisions.join(format!("{:08}.json", snapshot.revision));
    ensure!(
        !target.exists(),
        "revision {} already exists",
        snapshot.revision
    );
    let temporary = revisions.join(format!(
        ".{:08}-{}.tmp",
        snapshot.revision,
        std::process::id()
    ));
    write_new(&temporary, &serde_json::to_vec_pretty(snapshot)?)?;
    // The project lock protects this publication, and revisions are never replaced.
    fs::rename(&temporary, &target)?;
    Ok(())
}

fn resolve_review(list: &Path, snapshot: &Snapshot) -> Result<BTreeSet<String>> {
    let text = fs::read_to_string(list)?;
    let mut ids = BTreeSet::new();
    for (number, line) in text.lines().enumerate() {
        let value = line.trim().trim_start_matches('\u{feff}').trim_matches('"');
        if value.is_empty() || value.starts_with('#') {
            continue;
        }
        if let Some(frame) = snapshot.frames.iter().find(|f| f.id == value) {
            ids.insert(frame.id.clone());
            continue;
        }
        let path = Path::new(value);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            list.parent().unwrap_or(Path::new(".")).join(path)
        };
        let path = canonical(&path)?;
        let frame = snapshot
            .frames
            .iter()
            .find(|f| f.path == path)
            .with_context(|| {
                format!(
                    "{}, line {}: path is not a member of this project",
                    list.display(),
                    number + 1
                )
            })?;
        ids.insert(frame.id.clone());
    }
    ensure!(!ids.is_empty(), "review list is empty");
    Ok(ids)
}

fn set_review(
    snapshot: &mut Snapshot,
    ids: &BTreeSet<String>,
    reason: Option<&str>,
) -> Result<usize> {
    if let Some(reason) = reason {
        ensure!(
            !reason.trim().is_empty(),
            "exclusion reason must not be empty"
        );
    }
    ensure!(
        ids.iter()
            .all(|id| snapshot.frames.iter().any(|f| &f.id == id)),
        "review contains unknown identities"
    );
    let mut changed = 0;
    for frame in &mut snapshot.frames {
        if ids.contains(&frame.id) && frame.exclusion_reason.as_deref() != reason {
            frame.exclusion_reason = reason.map(str::to_owned);
            changed += 1;
        }
    }
    Ok(changed)
}

fn verify_sources(snapshot: &Snapshot) -> Result<()> {
    log::info!("verifying pinned reference identity");
    ensure!(
        hash_file(&snapshot.reference.path)? == snapshot.reference.id,
        "pinned reference has changed"
    );
    let selected: Vec<_> = snapshot
        .frames
        .iter()
        .filter(|f| f.exclusion_reason.is_none())
        .collect();
    for (index, frame) in selected.iter().enumerate() {
        if index % 25 == 0 || index + 1 == selected.len() {
            log::info!("verifying project source {}/{}", index + 1, selected.len());
        }
        ensure!(
            hash_file(&frame.path)? == frame.id,
            "source content changed: {}",
            frame.path.display()
        );
    }
    Ok(())
}

/// Pipeline manifests record output paths. Publication moves the entire run;
/// relocate only path strings under this staging directory before committing it.
fn relocate_json(value: &mut serde_json::Value, stage: &str, published: &str) {
    match value {
        serde_json::Value::String(s) => {
            if s == stage
                || s.strip_prefix(stage)
                    .is_some_and(|tail| tail.starts_with(['/', '\\']))
            {
                *s = format!("{published}{}", &s[stage.len()..]);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                relocate_json(value, stage, published);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values_mut() {
                relocate_json(value, stage, published);
            }
        }
        _ => {}
    }
}

fn run_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        ensure!(!kind.is_symlink(), "unexpected symlink in build outputs");
        if kind.is_dir() {
            files.extend(run_files(&entry.path())?);
        } else if kind.is_file() {
            files.push(entry.path());
        }
    }
    files.sort();
    Ok(files)
}

/// Join production decisions with explicit project exclusions. An excluded
/// original is decoded only for visual inspection, never supplied to the stack.
fn build_review(stage: &Path, snapshot: &Snapshot) -> Result<()> {
    use crate::frame_review::{self, ReviewFrame};
    let assets = stage.join("review-assets");
    fs::create_dir_all(&assets)?;
    let mut rows = BTreeMap::new();
    for report in run_files(&stage.join("diagnostics"))?.into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "frame-review.json")) {
        let value: serde_json::Value = serde_json::from_slice(&fs::read(&report)?)?;
        let group: Vec<ReviewFrame> = serde_json::from_value(value["frames"].clone())?;
        for row in group {
            if let Some(asset) = &row.preview_asset {
                let name = Path::new(asset).file_name().context("missing review asset name")?;
                fs::copy(report.parent().unwrap().join(asset), assets.join(name))?;
            }
            ensure!(rows.insert(row.path.clone(),row).is_none(), "duplicate production review record");
        }
    }
    let read = read_options(&snapshot.fits_row_order)?;
    let mut frames = Vec::new();
    for source in &snapshot.frames {
        if let Some(reason) = &source.exclusion_reason {
            let mut row = ReviewFrame { path:source.path.clone(), filter:source.filter.clone(),
                status:"excluded".into(),reason:format!("Project review exclusion: {reason}"),
                capture_time:None,exposure_seconds:None,iso_or_gain:None,photometric_gain:None,photometry_source:None,
                weight:None,hfd:None,eccentricity:None,residual:None,suppressed:None,obstruction_mask:None,
                preview_asset:None,preview_note:String::new() };
            let preview = (|| -> Result<()> {
                ensure!(hash_file(&source.path)? == source.id,"excluded source content has changed");
                let frame = sr_raw::decode_with(&source.path,&read)?;
                ensure!(hash_file(&source.path)? == source.id,"excluded source changed during preview");
                frame_review::capture(stage,&mut row,&frame,None,[frame.width as f32/2.0,frame.height as f32/2.0])
            })();
            if let Err(error) = preview {
                row.preview_asset=None;
                row.preview_note=format!("Original preview unavailable: {error:#}. Exclusion remains recorded; no replacement pixels are shown.");
            }
            frames.push(row);
        } else {
            frames.push(rows.remove(&source.path).context("selected input is missing production review evidence")?);
        }
    }
    ensure!(rows.is_empty(), "production review contains inputs outside project membership");
    frame_review::write(stage,&frames)
}

fn build(
    directory: &Path,
    snapshot: &Snapshot,
    workers: usize,
    in_memory: bool,
    no_cache: bool,
) -> Result<()> {
    ensure!(workers > 0, "workers must be positive");
    let paths: Vec<_> = snapshot
        .frames
        .iter()
        .filter(|f| f.exclusion_reason.is_none())
        .map(|f| f.path.clone())
        .collect();
    ensure!(!paths.is_empty(), "all project frames are excluded");
    verify_sources(snapshot).context("checking source identities before build")?;
    let executable = hash_file(&std::env::current_exe()?)?;
    let snapshot_bytes = serde_json::to_vec(snapshot)?;
    let revision_hash = hash_bytes(&snapshot_bytes);
    let runs = directory.join("runs");
    fs::create_dir_all(&runs)?;
    let mut sequence = 1u64;
    while runs.join(format!("{sequence:08}")).exists()
        || runs.join(format!(".{sequence:08}.pending")).exists()
    {
        sequence += 1;
    }
    let stage = runs.join(format!(".{sequence:08}.pending"));
    let published = runs.join(format!("{sequence:08}"));
    fs::create_dir(&stage)?;
    let input = stage.join("selected-frames.txt");
    let input_text = paths
        .iter()
        .map(|p| p.to_string_lossy())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    write_new(&input, input_text.as_bytes())?;
    write_new(&stage.join("project-revision.json"), &snapshot_bytes)?;
    let cache = directory
        .join("cache")
        .join(&executable)
        .join(&revision_hash);
    let spool = stage.join("spool");
    let spec = InputSpec {
        path: &input,
        pattern: None,
        max_frames: None,
        select: FrameSelect::First,
        read: read_options(&snapshot.fits_row_order)?,
        star_metrics: true,
        filter: None,
        cache_dir: (!no_cache).then_some(cache.as_path()),
        reference_file: Some(&snapshot.reference.path),
        spill_dir: (!in_memory).then_some(spool.as_path()),
        ordered_paths: Some(&paths),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let result = pool.install(|| {
        pipeline::stack(
            &spec,
            true,
            &snapshot.recipe,
            &stage.join("master.tif"),
            Some(&stage.join("diagnostics")),
            Some(&stage.join("master.preview.png")),
            1600,
            true,
            true,
            false,
            None,
            false,
            false,
        )
    });
    if let Err(error) = result {
        bail!("build failed; previous masters unchanged; partial diagnostics preserved at {}: {error:#}", stage.display());
    }
    verify_sources(snapshot).context("sources changed during build; result remains unpublished")?;
    build_review(&stage, snapshot)?;
    let stage_text = stage.to_string_lossy();
    let published_text = published.to_string_lossy();
    let mut artifacts = BTreeMap::new();
    for path in run_files(&stage)? {
        // Disk-backed decoded samples are disposable execution state, not masters.
        if path.starts_with(&spool) {
            continue;
        }
        if path.file_name().is_some_and(|f| f == "run.json") {
            let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
            relocate_json(&mut value, &stage_text, &published_text);
            let mut file = OpenOptions::new().write(true).truncate(true).open(&path)?;
            file.write_all(&serde_json::to_vec_pretty(&value)?)?;
            file.sync_all()?;
        }
        artifacts.insert(
            path.strip_prefix(&stage)?.to_string_lossy().into_owned(),
            hash_file(&path)?,
        );
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?
            .sync_all()?;
    }
    let manifest = serde_json::json!({
        "schema": 1, "revision": snapshot.revision, "revision_sha256": revision_hash,
        "executable_sha256": executable, "created_unix_seconds": now(),
        "recipe": snapshot.recipe, "reference": snapshot.reference,
        "fits_row_order": snapshot.fits_row_order, "workers": workers,
        "sample_storage": if in_memory { "resident" } else { "disk-backed" },
        "cache_enabled": !no_cache, "input_ids": snapshot.frames.iter().filter(|f| f.exclusion_reason.is_none()).map(|f| &f.id).collect::<Vec<_>>(),
        "artifact_sha256": artifacts,
        "update_strategy": "recompute global production decisions over all selected exposures"
    });
    write_new(
        &stage.join("project-build.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    ensure!(!published.exists(), "build output already exists");
    fs::rename(&stage, &published)?;
    println!(
        "Completed project revision {}: {}",
        snapshot.revision,
        published.display()
    );
    println!("Frame review: {}", published.join("frame-review.html").display());
    for path in artifacts
        .keys()
        .filter(|p| p.ends_with(".fits") || p.ends_with(".tif"))
    {
        if !path.starts_with("diagnostics") {
            println!("Master: {}", published.join(path).display());
        }
    }
    Ok(())
}

pub fn run(args: &Args) -> Result<()> {
    match &args.command {
        Command::Init {
            directory,
            input,
            reference_file,
            recipe,
            fits_row_order,
        } => {
            fs::create_dir_all(directory)?;
            let directory = canonical(directory)?;
            let _lock = ProjectLock::acquire(&directory)?;
            ensure!(
                fs::read_dir(&directory)?.all(|e| e.is_ok_and(|e| e.file_name() == "project.lock")),
                "project initialization requires an empty directory"
            );
            let read = read_options(fits_row_order)?;
            let path = canonical(reference_file)?;
            let id = hash_file(&path)?;
            let (width, height) = validate_source(&path, &read, None)?;
            ensure!(
                hash_file(&path)? == id,
                "reference changed during initialization"
            );
            let recipe = if let Some(path) = recipe {
                serde_json::from_slice(&fs::read(path)?).context("reading project recipe")?
            } else {
                ReconstructionConfig {
                    scale: 1.0,
                    ..ReconstructionConfig::default()
                }
            };
            validate_recipe(&recipe)?;
            let mut snapshot = Snapshot {
                schema: 1,
                revision: 1,
                created_unix_seconds: now(),
                operation: "init".into(),
                reference: Reference {
                    id,
                    path,
                    width,
                    height,
                },
                fits_row_order: fits_row_order.clone(),
                recipe,
                frames: Vec::new(),
            };
            let (added, duplicates) = ingest(input, &mut snapshot)?;
            save(&directory, &snapshot)?;
            println!(
                "Initialized {}: {added} frames, {duplicates} duplicate contents ignored",
                directory.display()
            );
        }
        Command::Add { directory, input } => {
            let directory = canonical(directory)?;
            let _lock = ProjectLock::acquire(&directory)?;
            let mut snapshot = load(&directory)?;
            let (added, duplicates) = ingest(input, &mut snapshot)?;
            if added > 0 {
                snapshot.revision += 1;
                snapshot.created_unix_seconds = now();
                snapshot.operation = "add".into();
                save(&directory, &snapshot)?;
            }
            println!(
                "Revision {}: added {added}; {duplicates} duplicate contents ignored",
                snapshot.revision
            );
        }
        Command::Relink { directory, input } => {
            let directory = canonical(directory)?;
            let _lock = ProjectLock::acquire(&directory)?;
            let mut snapshot = load(&directory)?;
            let changed = relink(input, &mut snapshot)?;
            if changed > 0 {
                snapshot.revision += 1;
                snapshot.created_unix_seconds = now();
                snapshot.operation = "relink".into();
                save(&directory, &snapshot)?;
            }
            println!(
                "Revision {}: {changed} content identities relocated; review decisions preserved",
                snapshot.revision
            );
        }
        Command::Exclude {
            directory,
            list,
            reason,
        } => review(directory, list, Some(reason))?,
        Command::Include { directory, list } => review(directory, list, None)?,
        Command::Build {
            directory,
            workers,
            in_memory,
            no_cache,
        } => {
            let directory = canonical(directory)?;
            let _lock = ProjectLock::acquire(&directory)?;
            build(
                &directory,
                &load(&directory)?,
                *workers,
                *in_memory,
                *no_cache,
            )?;
        }
        Command::Status { directory } => {
            let snapshot = load(directory)?;
            let selected = snapshot
                .frames
                .iter()
                .filter(|f| f.exclusion_reason.is_none())
                .count();
            println!(
                "Revision {}: {} frames; {selected} selected; {} excluded",
                snapshot.revision,
                snapshot.frames.len(),
                snapshot.frames.len() - selected
            );
            println!("Pinned reference: {}", snapshot.reference.path.display());
            for frame in snapshot
                .frames
                .iter()
                .filter(|f| f.exclusion_reason.is_some())
            {
                println!(
                    "Excluded {}: {} ({})",
                    frame.id,
                    frame.path.display(),
                    frame.exclusion_reason.as_deref().unwrap_or_default()
                );
            }
        }
    }
    Ok(())
}

fn review(directory: &Path, list: &Path, reason: Option<&str>) -> Result<()> {
    let directory = canonical(directory)?;
    let _lock = ProjectLock::acquire(&directory)?;
    let mut snapshot = load(&directory)?;
    let ids = resolve_review(list, &snapshot)?;
    let changed = set_review(&mut snapshot, &ids, reason)?;
    if changed > 0 {
        snapshot.revision += 1;
        snapshot.created_unix_seconds = now();
        snapshot.operation = if reason.is_some() {
            "exclude"
        } else {
            "include"
        }
        .into();
        save(&directory, &snapshot)?;
    }
    println!(
        "Revision {}: {changed} review decisions changed",
        snapshot.revision
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> Snapshot {
        Snapshot {
            schema: 1,
            revision: 1,
            created_unix_seconds: 0,
            operation: "init".into(),
            reference: Reference {
                id: "ref".into(),
                path: "reference.fits".into(),
                width: 16,
                height: 16,
            },
            fits_row_order: "auto".into(),
            recipe: ReconstructionConfig::default(),
            frames: vec![
                Frame {
                    id: "a".into(),
                    path: "a.fits".into(),
                    filter: "H".into(),
                    exclusion_reason: None,
                },
                Frame {
                    id: "b".into(),
                    path: "b.fits".into(),
                    filter: "H".into(),
                    exclusion_reason: None,
                },
            ],
        }
    }

    #[test]
    fn reviews_are_explicit_reversible_and_idempotent() {
        let mut s = snapshot();
        let ids = BTreeSet::from(["a".to_owned()]);
        assert!(set_review(&mut s, &ids, Some(" ")).is_err());
        assert_eq!(set_review(&mut s, &ids, Some("cloud")).unwrap(), 1);
        assert_eq!(set_review(&mut s, &ids, Some("cloud")).unwrap(), 0);
        assert_eq!(s.frames[1].exclusion_reason, None);
        assert_eq!(set_review(&mut s, &ids, None).unwrap(), 1);
        assert_eq!(s.frames[0].exclusion_reason, None);
        assert!(set_review(&mut s, &BTreeSet::from(["unknown".into()]), Some("cloud")).is_err());
    }

    #[test]
    fn manifest_relocation_respects_path_boundaries() {
        let mut value = serde_json::json!({"output": "C:/run/pending/master.tif", "other": "C:/run/pending-original.fits"});
        relocate_json(&mut value, "C:/run/pending", "C:/run/complete");
        assert_eq!(value["output"], "C:/run/complete/master.tif");
        assert_eq!(value["other"], "C:/run/pending-original.fits");
    }

    #[test]
    fn lock_and_publication_never_overwrite() {
        let directory = std::env::temp_dir().join(format!(
            "smokstak-project-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let lock = ProjectLock::acquire(&directory).unwrap();
        assert!(ProjectLock::acquire(&directory).is_err());
        let s = snapshot();
        save(&directory, &s).unwrap();
        let first = fs::read(directory.join("revisions/00000001.json")).unwrap();
        assert!(save(&directory, &s).is_err());
        assert_eq!(
            fs::read(directory.join("revisions/00000001.json")).unwrap(),
            first
        );
        assert_eq!(load(&directory).unwrap().frames.len(), 2);
        // Loading also checks the invariant: a hand-edited or older revision
        // must not bypass the same checks that protect ingestion.
        let mut invalid = s.clone();
        invalid.revision = 2;
        invalid.frames[1].filter = "h".into();
        save(&directory, &invalid).unwrap();
        assert!(load(&directory).is_err());
        drop(lock);
        assert!(!directory.join("project.lock").exists());
        fs::remove_file(directory.join("revisions/00000001.json")).unwrap();
        fs::remove_file(directory.join("revisions/00000002.json")).unwrap();
        fs::remove_dir(directory.join("revisions")).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn filter_output_names_are_portable_and_collision_free() {
        assert!(validate_filters(["H", "H", "OIII", "SII", "Ha-7nm", "H+O"]).is_ok());
        assert!(validate_filters(["H", "h"]).is_err());
        assert!(validate_filters(["OIII", "oiii"]).is_err());
        for name in [
            "CON", "con", "NUL", "PRN", "aux", "COM1", "com9", "LPT1", "lpt9", "", "../H",
            "H/other",
        ] {
            assert!(validate_filters([name]).is_err(), "accepted {name:?}");
        }
        assert!(validate_filters(["COM10", "LPT0", "CON_H"]).is_ok());
    }

    #[test]
    fn relink_preserves_identity_reviews_and_reference() {
        let mut s = snapshot();
        s.reference.id = "a".into();
        s.reference.path = "a.fits".into();
        s.frames[0].exclusion_reason = Some("cloud".into());
        let replacements = BTreeMap::from([("a".into(), PathBuf::from("moved/a.fits"))]);
        assert_eq!(apply_relinks(&mut s, &replacements).unwrap(), 1);
        assert_eq!(s.frames[0].id, "a");
        assert_eq!(s.frames[0].exclusion_reason.as_deref(), Some("cloud"));
        assert_eq!(s.frames[0].filter, "H");
        assert_eq!(s.reference.path, PathBuf::from("moved/a.fits"));
        assert_eq!(s.reference.id, "a");
        assert_eq!(apply_relinks(&mut s, &replacements).unwrap(), 0);
        let before = serde_json::to_vec(&s).unwrap();
        assert!(apply_relinks(
            &mut s,
            &BTreeMap::from([("unknown".into(), PathBuf::from("x.fits"))])
        )
        .is_err());
        assert!(apply_relinks(
            &mut s,
            &BTreeMap::from([("b".into(), PathBuf::from("moved/a.fits"))])
        )
        .is_err());
        assert_eq!(serde_json::to_vec(&s).unwrap(), before);
        s.reference.id = "external".into();
        assert_eq!(
            apply_relinks(
                &mut s,
                &BTreeMap::from([("external".into(), PathBuf::from("moved/ref.fits"))])
            )
            .unwrap(),
            1
        );
        assert_eq!(s.reference.path, PathBuf::from("moved/ref.fits"));
        assert_eq!(s.frames.len(), 2);
    }

    #[test]
    fn content_identity_hashes_every_byte() {
        assert_eq!(
            hash_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let a = vec![0; 1024 * 1024];
        let mut b = a.clone();
        b[512 * 1024] = 1;
        assert_ne!(hash_bytes(&a), hash_bytes(&b));
    }
}
