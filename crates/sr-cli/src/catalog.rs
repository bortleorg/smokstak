//! Frames chosen by asking the catalog, rather than by pointing at a folder.
//!
//! Every frame the observatory takes is indexed by a catalog server that knows
//! its target, its filter, the night it was taken, how sharp its stars were and
//! whether anybody rejected it. A query against that is a better way of saying
//! "every good Ha frame of the Rosette" than a folder is: one target's frames
//! are spread across a folder per night, and after the next clear night there
//! are more of them. So the page asks the catalog, finds each frame it names on
//! this computer, and writes a list — which is what the stacker already reads,
//! so nothing past this module knows there was a catalog at all.
//!
//! A query can be bookmarked. What is kept is the question and not the answer,
//! so running a bookmark again picks up every frame indexed since.
//!
//! ## Why curl
//!
//! The catalog is served over HTTPS, and speaking TLS from Rust means a TLS
//! stack and a set of certificate roots: a dozen crates, one of them a C and
//! assembly build, for a program that otherwise builds with cargo alone. `curl`
//! ships with Windows 10 and later, with macOS and with every Linux desktop,
//! and it trusts the certificates the platform trusts. The page already asks
//! the desktop for its folder chooser the same way.
//!
//! The credentials reach curl on standard input, as a config file, and never
//! as arguments, where they would sit in the process list for as long as the
//! request takes. It is never told to follow a redirect, so a login page sent
//! back in place of an answer is reported as a refusal rather than followed
//! with the password attached.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The catalog this was written against. A default only; the page can point
/// somewhere else.
const DEFAULT_BASE: &str = "https://ap.smerty.com/api/v1";

/// Frames asked for at once, which is the most the server will hand out.
const PER_PAGE: usize = 500;

/// A query matching more than this is not one target, it is the catalog, and
/// fetching eighty thousand paths to find that out is the wrong way round.
const MOST: usize = 20_000;

/// What finding a frame and describing a set need, and nothing else. Left to
/// itself the server sends every group it has, a few kilobytes a frame.
const FIELDS: &str = "file.path,file.host_path,capture.object,capture.filter,\
                      capture.date_obs,capture.exposure_s,capture.camera";

/// What the catalog's screening said about a frame, asked for whenever the
/// server screens at all. How the frame compares with the rest of its session
/// is a group of its own, added with `expand=sequence`.
const SCREEN_FIELDS: &str = "quality.trailed,quality.occluded,quality.satellite,status.grade";

/// The screens a query can leave frames out by, in the order a frame is
/// blamed on them.
const SCREENS: [&str; 4] = ["trailed", "occluded", "anomaly", "satellite"];

/// Parameters this module sets itself, and a query may not.
const RESERVED: [&str; 5] = ["page", "per_page", "fields", "expand", "api_key"];

/// How to reach the catalog, and how its paths become this computer's.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    /// The API root, up to and including the version: `https://…/api/v1`.
    base: String,
    user: String,
    password: String,
    /// Sent as `X-API-Key`, for a server that asks for one.
    api_key: String,
    /// Tried in order, the first that matches wins.
    rewrites: Vec<Rewrite>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            base: DEFAULT_BASE.into(),
            user: String::new(),
            password: String::new(),
            api_key: String::new(),
            rewrites: Vec::new(),
        }
    }
}

/// One folder as the catalog names it, and the same folder as this computer
/// names it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct Rewrite {
    from: String,
    to: String,
}

/// A question to ask the catalog.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct Query {
    /// Parameters of `GET /frames`, by name, as the server documents them.
    params: BTreeMap<String, String>,
    /// A query saved in the catalog itself, asked through
    /// `/queries/{id}/frames` instead. Zero for none.
    saved: u64,
    /// What that query is called there, which is also what the list is named.
    saved_name: String,
    /// Which of the catalog's screens leave frames out: `anomaly`, `trailed`,
    /// `occluded`, `satellite`. A frame the catalog has not judged is kept.
    screen: BTreeSet<String>,
}

/// A query kept to be asked again.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct Bookmark {
    name: String,
    query: Query,
    /// What it found the last time it was asked, and when, in seconds since
    /// the epoch: "412 frames on 9 Sep" says whether there is anything new.
    last_frames: usize,
    last_run: f64,
}

/// Everything kept between sessions, in one file.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Store {
    settings: Settings,
    bookmarks: Vec<Bookmark>,
}

/// Held while the file is read, changed and written back, so that two
/// requests at once cannot each write over what the other kept.
static STORE: Mutex<()> = Mutex::new(());

/// Where the catalog's settings and bookmarks are kept.
///
/// Beside the list of earlier runs. `SMOKSTAK_CATALOG` names another file, and
/// set to nothing keeps nothing.
fn store_file() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SMOKSTAK_CATALOG") {
        return (!p.is_empty()).then(|| PathBuf::from(p));
    }
    crate::gui::data_dir().map(|d| d.join("catalog.json"))
}

/// The kept settings, for reading. A file that cannot be read reads as the
/// defaults, which is enough to show the page; changing anything refuses
/// instead, so that the bookmarks in it are not written over.
fn load() -> Store {
    let Some(path) = store_file() else { return Store::default() };
    load_from(&path).unwrap_or_else(|e| {
        log::warn!("{e}");
        Store::default()
    })
}

fn load_from(path: &Path) -> Result<Store> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Store::default()),
        Err(e) => return Err(anyhow!("{} could not be read: {e}", path.display())),
    };
    serde_json::from_str(&text)
        .map_err(|e| anyhow!("{} could not be read ({e}); mend it or delete it", path.display()))
}

/// Written beside itself and renamed into place, so a server closed part way
/// leaves the last file and not half of this one. It holds a password, so on
/// a system with file modes only its owner may read it.
fn save_to(path: &Path, store: &Store) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(store)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read the kept file, change it, and write it back, as one step.
fn update<T>(change: impl FnOnce(&mut Store) -> Result<T>) -> Result<T> {
    let path = store_file().ok_or_else(|| {
        anyhow!("there is nowhere to keep the catalog's settings; set SMOKSTAK_CATALOG to a file")
    })?;
    let _held = STORE.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = load_from(&path)?;
    let out = change(&mut store)?;
    save_to(&path, &store)?;
    Ok(out)
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---- talking to the server ------------------------------------------------

/// What came back: the status, and the body as it was sent.
struct Reply {
    status: u16,
    body: Vec<u8>,
}

/// A value for curl's config file, quoted. A line break would end the line
/// and begin another option, so a setting holding one is refused.
fn quoted(s: &str) -> Result<String> {
    ensure!(!s.contains(['\n', '\r', '\0']), "a catalog setting holds a line break");
    Ok(format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
}

/// Everything curl is told about one request, credentials included.
fn curl_config(s: &Settings, url: &str) -> Result<String> {
    let mut c = format!("url = {}\n", quoted(url)?);
    if !s.user.is_empty() || !s.password.is_empty() {
        // Always with the colon: without one curl asks for the password on
        // the terminal, which here is the config it is reading.
        c.push_str(&format!("user = {}\n", quoted(&format!("{}:{}", s.user, s.password))?));
    }
    if !s.api_key.is_empty() {
        c.push_str(&format!("header = {}\n", quoted(&format!("X-API-Key: {}", s.api_key))?));
    }
    Ok(c)
}

fn get(s: &Settings, url: &str) -> Result<Reply> {
    let config = curl_config(s, url)?;
    let mut child = Command::new("curl")
        .args([
            "--silent",
            "--show-error",
            "--globoff",
            "--connect-timeout",
            "15",
            "--max-time",
            "180",
            // After the body, which is how the status is told from it without
            // a second stream: the last three bytes.
            "--write-out",
            "%{http_code}",
            "--config",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                anyhow!(
                    "the catalog is reached through curl, which is not installed here \
                     (it comes with Windows 10 and later, macOS and most Linux)"
                )
            } else {
                anyhow!("could not start curl: {e}")
            }
        })?;
    child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("curl has no input"))?
        .write_all(config.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        let said = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(anyhow!(
            "could not reach the catalog: {}",
            if said.is_empty() { format!("curl ended with {}", out.status) } else { said }
        ));
    }
    let mut body = out.stdout;
    ensure!(body.len() >= 3, "the catalog sent nothing back");
    let code = body.split_off(body.len() - 3);
    let status = std::str::from_utf8(&code)
        .ok()
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| anyhow!("curl did not say how the request went"))?;
    Ok(Reply { status, body })
}

/// The JSON a request was answered with, or what went wrong said plainly.
fn answer(r: Reply) -> Result<Value> {
    let said = || -> Option<String> {
        let v: Value = serde_json::from_slice(&r.body).ok()?;
        v["error"]["message"].as_str().map(str::to_string)
    };
    match r.status {
        200 => serde_json::from_slice(&r.body).context("the catalog answered with something other than JSON"),
        // The server sits behind a login, and a request it does not accept is
        // sent to the login page rather than refused.
        300..=399 => Err(anyhow!(
            "the catalog sent a login page instead of an answer, so the user name and \
             password were not accepted; they are under Connection and paths"
        )),
        401 | 403 => Err(anyhow!(
            "the catalog refused the request{}; the credentials are under Connection and paths",
            said().map(|m| format!(" ({m})")).unwrap_or_default()
        )),
        code => Err(anyhow!("the catalog said: {}", said().unwrap_or_else(|| format!("HTTP {code}")))),
    }
}

/// The API root, checked.
fn base_of(s: &Settings) -> Result<String> {
    let base = s.base.trim().trim_end_matches('/');
    ensure!(!base.is_empty(), "no catalog is set up; its address goes under Connection and paths");
    ensure!(
        base.starts_with("https://") || base.starts_with("http://"),
        "the catalog's address should start with https://"
    );
    Ok(base.to_string())
}

/// What a catalog says it can do, read from its own specification.
#[derive(Clone, Debug, Default)]
struct Capabilities {
    base: String,
    version: String,
    /// The parameters `GET /frames` takes, less the ones set here, each with
    /// what the server says it does.
    frame_params: Vec<(String, String)>,
    /// Whether frames carry the catalog's screening: trailed, occluded,
    /// satellite, and how each compares with its session.
    screening: bool,
    /// Whether a frame can be fetched larger than its thumbnail.
    preview: bool,
}

fn capabilities_of(spec: &Value, base: &str) -> Capabilities {
    let params: Vec<(String, String)> = spec["paths"]["/frames"]["get"]["parameters"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|p| {
            let name = p["name"].as_str()?;
            let said = p["description"].as_str().or_else(|| p["schema"]["description"].as_str());
            Some((name.to_string(), said.unwrap_or_default().to_string()))
        })
        .filter(|(name, _)| !RESERVED.contains(&name.as_str()))
        .collect();
    let schemas = &spec["components"]["schemas"];
    Capabilities {
        base: base.to_string(),
        version: spec["info"]["version"].as_str().unwrap_or_default().to_string(),
        frame_params: params,
        screening: schemas["Frame"]["properties"]["sequence"].is_object()
            && ["trailed", "occluded", "satellite"]
                .iter()
                .all(|k| schemas["QualityInfo"]["properties"][*k].is_object()),
        preview: spec["paths"]["/frames/{frame_id}/preview"].is_object(),
    }
}

/// What the catalog last said it can do, so that every query is not preceded
/// by a fetch of the whole specification.
static CAPABILITIES: Mutex<Option<Capabilities>> = Mutex::new(None);

/// The catalog's capabilities, from the last reading unless `fresh`. `None`
/// when the specification cannot be read, and queries then go unchecked
/// rather than refused.
fn capabilities(s: &Settings, fresh: bool) -> Option<Capabilities> {
    let base = base_of(s).ok()?;
    let mut held = CAPABILITIES.lock().unwrap_or_else(|e| e.into_inner());
    if !fresh {
        if let Some(c) = held.as_ref().filter(|c| c.base == base) {
            return Some(c.clone());
        }
    }
    match get(s, &format!("{base}/openapi.json")).and_then(answer) {
        Ok(spec) => {
            let c = capabilities_of(&spec, &base);
            *held = Some(c.clone());
            Some(c)
        }
        Err(e) => {
            log::warn!("the catalog's specification could not be read, so queries go unchecked: {e}");
            None
        }
    }
}

/// A query this catalog cannot answer as asked, refused before asking.
///
/// The server ignores a parameter it does not know, so a mistyped filter, or
/// one from a newer catalog, would otherwise quietly match every frame it was
/// written to leave out.
fn check_supported(q: &Query, caps: &Capabilities) -> Result<()> {
    ensure!(
        q.screen.is_empty() || caps.screening,
        "this catalog (API {}) does not screen frames, so its screening cannot leave any out",
        caps.version
    );
    if q.saved > 0 || caps.frame_params.is_empty() {
        return Ok(());
    }
    for k in q.params.keys() {
        ensure!(
            caps.frame_params.iter().any(|(name, _)| name == k),
            "the catalog (API {}) takes no parameter called {k}; the ones it takes are listed \
             under Anything else",
            caps.version
        );
    }
    Ok(())
}

/// Everything but the unreserved characters, `%`-encoded. A `where`
/// expression is full of characters that mean something in a query string.
fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// A query's own parameters, checked, with the empty ones dropped.
fn params_of(q: &Query) -> Result<Vec<(String, String)>> {
    for s in &q.screen {
        ensure!(SCREENS.contains(&s.as_str()), "{s:?} is not one of the catalog's screens");
    }
    let mut out = Vec::new();
    for (k, v) in &q.params {
        ensure!(
            !k.is_empty() && k.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
            "{k:?} is not the name of a catalog parameter"
        );
        ensure!(!RESERVED.contains(&k.as_str()), "{k} is set by the page itself, not by a query");
        if q.saved > 0 {
            // The saved query makes its own selection, and the server takes
            // nothing else beside it.
            ensure!(
                matches!(k.as_str(), "include_rejected" | "sort"),
                "a query saved in the catalog takes only include_rejected and sort, not {k}"
            );
        }
        if !v.trim().is_empty() {
            out.push((k.clone(), v.trim().to_string()));
        }
    }
    Ok(out)
}

/// One page of a query's frames.
fn frames_url(base: &str, q: &Query, page: usize, screening: bool) -> Result<String> {
    let mut pairs = params_of(q)?;
    // Oldest first, so a list reads in the order the night went, and one
    // order every time, so a page boundary cannot move under the paging.
    if !q.params.contains_key("sort") {
        pairs.push(("sort".into(), "capture.date_obs".into()));
    }
    if screening {
        pairs.push(("fields".into(), format!("{FIELDS},{SCREEN_FIELDS}")));
        pairs.push(("expand".into(), "sequence".into()));
    } else {
        pairs.push(("fields".into(), FIELDS.into()));
    }
    pairs.push(("page".into(), page.to_string()));
    pairs.push(("per_page".into(), PER_PAGE.to_string()));
    let path = if q.saved > 0 {
        format!("{base}/queries/{}/frames", q.saved)
    } else {
        format!("{base}/frames")
    };
    let query: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}={}", encode(v))).collect();
    Ok(format!("{path}?{}", query.join("&")))
}

/// One frame, as much of it as this needs.
#[derive(Clone, Debug, Default)]
struct Listed {
    id: u64,
    /// The path as the catalog's own server sees it: `/mnt/astro/…`.
    path: String,
    /// The same, rewritten by the server for the machines that stack: `Z:\…`.
    host_path: String,
    object: String,
    filter: String,
    date: String,
    exposure: f64,
    camera: String,
    /// What the catalog's screening said, where it has said anything: `None`
    /// is a frame not judged, which is not the same as a frame judged good.
    trailed: Option<bool>,
    occluded: Option<bool>,
    satellite: Option<bool>,
    /// `accepted`, `rejected`, or empty when nobody graded it.
    grade: String,
    /// Whether the frame stands out from the rest of its session.
    anomaly: Option<bool>,
    /// How far it departs from its session, by each measure the catalog took.
    deviations: BTreeMap<&'static str, f64>,
}

impl Listed {
    fn from_json(v: &Value) -> Self {
        let text = |group: &str, leaf: &str| v[group][leaf].as_str().unwrap_or_default().to_string();
        Self {
            id: v["id"].as_u64().unwrap_or(0),
            path: text("file", "path"),
            host_path: text("file", "host_path"),
            object: text("capture", "object"),
            filter: text("capture", "filter"),
            date: text("capture", "date_obs"),
            exposure: v["capture"]["exposure_s"].as_f64().unwrap_or(0.0),
            camera: text("capture", "camera"),
            trailed: v["quality"]["trailed"].as_bool(),
            occluded: v["quality"]["occluded"].as_bool(),
            satellite: v["quality"]["satellite"].as_bool(),
            grade: text("status", "grade"),
            anomaly: v["sequence"]["anomaly"].as_bool(),
            deviations: ["star_drop", "hfr_rise", "empty_rise", "spread_rise", "flux_drop"]
                .into_iter()
                .filter_map(|k| Some((k, v["sequence"][k].as_f64()?)))
                .collect(),
        }
    }
}

/// Every frame a query matches, a page at a time, and how many the catalog
/// said there were.
fn fetch_all(s: &Settings, caps: Option<&Capabilities>, q: &Query) -> Result<(Vec<Listed>, usize)> {
    let base = base_of(s)?;
    if let Some(c) = caps {
        check_supported(q, c)?;
    }
    // Whenever the server screens, so that what it would leave out can be
    // said even when the query leaves nothing out.
    let screening = !q.screen.is_empty() || caps.is_some_and(|c| c.screening);
    let mut out: Vec<Listed> = Vec::new();
    let mut seen = HashSet::new();
    let mut page = 1;
    loop {
        let doc = answer(get(s, &frames_url(&base, q, page, screening)?)?)?;
        let total = doc["total"].as_u64().unwrap_or(0) as usize;
        ensure!(
            total <= MOST,
            "that matches {total} frames, which is most of the catalog rather than a target; \
             narrow it with a target, a filter or some dates"
        );
        let items = doc["items"].as_array().map(Vec::as_slice).unwrap_or_default();
        // A frame indexed while this pages can push one already fetched onto
        // the next page; it is the same frame and is listed once.
        out.extend(items.iter().map(Listed::from_json).filter(|f| seen.insert(f.id)));
        let pages = doc["pages"].as_u64().unwrap_or(1) as usize;
        if items.is_empty() || page >= pages {
            return Ok((out, total));
        }
        page += 1;
    }
}

/// What one screen said about a frame: `None` when it has not judged it.
fn verdict(f: &Listed, screen: &str) -> Option<bool> {
    match screen {
        "trailed" => f.trailed,
        "occluded" => f.occluded,
        "anomaly" => f.anomaly,
        "satellite" => f.satellite,
        _ => None,
    }
}

/// Why the catalog's screening leaves a frame out, if it does.
///
/// Only a frame the catalog has judged is left out. While it is still
/// analyzing — after an upgrade, or for a night just indexed — most of these
/// answers are null, and a frame nobody has looked at is not a bad frame.
fn screened_by(f: &Listed, q: &Query) -> Option<&'static str> {
    // Graded rejected, under the same choice as marked rejected.
    let rejecting = if q.saved > 0 {
        q.params.get("include_rejected").map(String::as_str) != Some("true")
    } else {
        q.params.get("rejected").map(String::as_str) == Some("false")
    };
    if rejecting && f.grade == "rejected" {
        return Some("rejected");
    }
    SCREENS
        .into_iter()
        .filter(|s| q.screen.contains(*s))
        .find(|s| verdict(f, s) == Some(true))
}

/// For each screen asked for, how many of the frames kept it has not judged.
fn unjudged(frames: &[Listed], q: &Query) -> BTreeMap<&'static str, usize> {
    SCREENS
        .into_iter()
        .filter(|s| q.screen.contains(*s))
        .map(|s| (s, frames.iter().filter(|f| verdict(f, s).is_none()).count()))
        .collect()
}

fn file_name(f: &Listed) -> &str {
    let p = if f.host_path.is_empty() { &f.path } else { &f.host_path };
    p.rsplit(['/', '\\']).next().unwrap_or(p)
}

// ---- finding the frames here -----------------------------------------------

/// A catalog path under the first rule whose folder it is in, or `None`.
///
/// Matched without regard to case or to which way the separators lean, since
/// the catalog and Windows agree on neither, and the rest of the path is given
/// the separators of the folder it is moved into.
fn rewrite(path: &str, rules: &[Rewrite]) -> Option<String> {
    // Both changes keep every byte where it was, so a length measured on the
    // normalised text is a length in the original.
    let norm = |s: &str| s.replace('\\', "/").to_ascii_lowercase();
    let p = norm(path);
    rules.iter().find_map(|r| {
        let from = norm(r.from.trim());
        if from.is_empty() || !p.starts_with(&from) {
            return None;
        }
        let to = r.to.trim();
        let windows = to.contains('\\') || to.as_bytes().get(1) == Some(&b':');
        let (sep, other) = if windows { ('\\', '/') } else { ('/', '\\') };
        let rest = path[from.len()..].replace(other, &sep.to_string());
        let to_ends = to.ends_with(['/', '\\']);
        let rest_starts = rest.starts_with(sep);
        Some(match (to_ends, rest_starts) {
            (true, true) => format!("{to}{}", &rest[1..]),
            (false, false) if !to.is_empty() && !rest.is_empty() => format!("{to}{sep}{rest}"),
            _ => format!("{to}{rest}"),
        })
    })
}

/// Where a frame is on this computer, if it is.
#[derive(Clone, Debug, PartialEq)]
struct Located {
    /// The path used, or when none was there, the one looked at first.
    path: String,
    found: bool,
}

/// The first of the frame's names that is a file here.
///
/// A rule the reader wrote comes first, because it was written for a reason;
/// then the path the server rewrote for Windows machines, then its own. So a
/// catalog that already names `Z:\` needs no rules on a machine where that
/// drive is mounted, and one that does not can be given them.
fn locate(f: &Listed, rules: &[Rewrite], exists: impl Fn(&str) -> bool) -> Located {
    let mut tried: Vec<String> = Vec::new();
    let names = [&f.host_path, &f.path];
    let rewritten = names.iter().filter_map(|n| rewrite(n, rules));
    for name in rewritten.chain(names.iter().map(|n| n.to_string())) {
        if !name.is_empty() && !tried.contains(&name) {
            tried.push(name);
        }
    }
    match tried.iter().find(|t| exists(t.as_str())) {
        Some(t) => Located { path: t.clone(), found: true },
        None => Located { path: tried.first().cloned().unwrap_or_default(), found: false },
    }
}

/// `k` indices spread across `n`, first and last included.
fn spread(n: usize, k: usize) -> Vec<usize> {
    if n <= k {
        return (0..n).collect();
    }
    if k < 2 {
        return (0..k).collect();
    }
    (0..k).map(|i| i * (n - 1) / (k - 1)).collect()
}

/// What the page is told about the frames a query found.
fn summarise(listed: &[Listed], located: &[Located], total: usize) -> Value {
    let found = located.iter().filter(|l| l.found).count();
    // Described as what would be stacked, whenever any of it is here.
    let set: Vec<&Listed> = listed
        .iter()
        .zip(located)
        .filter(|(_, l)| l.found || found == 0)
        .map(|(f, _)| f)
        .collect();

    let mut filters: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    let mut objects: BTreeMap<&str, usize> = BTreeMap::new();
    let mut cameras: BTreeSet<&str> = BTreeSet::new();
    let mut dates: BTreeSet<&str> = BTreeSet::new();
    let mut taken: BTreeSet<&str> = BTreeSet::new();
    let mut seconds = 0.0;
    for f in &set {
        let e = filters.entry(f.filter.as_str()).or_default();
        e.0 += 1;
        e.1 += f.exposure;
        seconds += f.exposure;
        if !f.object.is_empty() {
            *objects.entry(&f.object).or_default() += 1;
        }
        if !f.camera.is_empty() {
            cameras.insert(&f.camera);
        }
        // `N/A` is what some capture software writes for a date it lacks.
        if let Some(day) = f.date.get(..10).filter(|d| d.as_bytes()[0].is_ascii_digit()) {
            dates.insert(day);
            taken.insert(&f.date);
        }
    }
    let mut filters: Vec<_> = filters.into_iter().collect();
    filters.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    let mut objects: Vec<_> = objects.into_iter().collect();
    objects.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    objects.truncate(5);

    let missing: Vec<Value> = listed
        .iter()
        .zip(located)
        .filter(|(_, l)| !l.found)
        .take(3)
        .map(|(f, l)| {
            let named = if f.host_path.is_empty() { &f.path } else { &f.host_path };
            json!({ "catalog": named, "tried": l.path })
        })
        .collect();

    json!({
        "total": total,
        "listed": listed.len(),
        "found": found,
        "missing": listed.len() - found,
        "missing_examples": missing,
        "filters": filters.iter().map(|(name, (n, s))| json!({
            "name": name, "frames": n, "hours": s / 3600.0,
        })).collect::<Vec<_>>(),
        "objects": objects.iter().map(|(name, n)| json!({ "name": name, "frames": n })).collect::<Vec<_>>(),
        "cameras": cameras,
        "dates": dates.len(),
        "first": taken.first(),
        "last": taken.last(),
        "hours": seconds / 3600.0,
        // A few frames from across the set, to see that it is the right sky.
        "thumbs": spread(set.len(), 8).into_iter().map(|i| set[i].id).collect::<Vec<_>>(),
    })
}

/// A name for a query's list when the page gives none.
fn default_name(q: &Query) -> String {
    if q.saved > 0 {
        return if q.saved_name.is_empty() { format!("catalog query {}", q.saved) } else { q.saved_name.clone() };
    }
    let parts: Vec<&str> = ["object", "filter"]
        .iter()
        .filter_map(|k| q.params.get(*k))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() { "catalog".into() } else { parts.join(" ") }
}

/// The query in a line, for the top of the list it made.
fn describe(q: &Query) -> String {
    let params: Vec<String> = q.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
    let params = params.join(", ");
    if q.saved == 0 {
        return params;
    }
    let also = if params.is_empty() { String::new() } else { format!(", {params}") };
    format!("saved query {} ({}){also}", q.saved, q.saved_name)
}

/// A name that can be a file name anywhere. The list's name becomes the
/// result's, so it is kept readable rather than made unique; the folder it is
/// written in is what is unique.
fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '_') {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let out: String = out.trim_matches(['-', '.']).chars().take(60).collect();
    let out = out.trim_end_matches(['-', '.']);
    if out.is_empty() { "catalog".into() } else { out.to_string() }
}

/// Write the frames found as a list the stacker reads.
///
/// A new folder each time. The same bookmark asked again next week names more
/// frames, and a list rewritten in place would change under a stack still
/// reading it, and under a measurement of every frame made of the old one.
fn write_list(name: &str, what: &str, paths: &[&str]) -> Result<PathBuf> {
    // The clock alone is not enough: two asked in the same millisecond would
    // share a folder.
    static WRITTEN: AtomicU64 = AtomicU64::new(0);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join("smokstak-catalog").join(format!(
        "{stamp}-{}-{}",
        std::process::id(),
        WRITTEN.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir)?;
    let list = dir.join(format!("{}.txt", slug(name)));
    let mut text = format!(
        "# {} frames from the catalog: {}\n",
        paths.len(),
        what.replace(['\r', '\n'], " ")
    );
    for p in paths {
        text.push_str(p);
        text.push('\n');
    }
    std::fs::write(&list, text)?;
    Ok(list)
}

/// Keep a bookmark, or change the query an existing one asks. What it found
/// last time belongs to the old question and goes with it.
fn keep_bookmark(marks: &mut Vec<Bookmark>, name: &str, query: Query) {
    match marks.iter_mut().find(|b| b.name == name) {
        Some(b) if b.query == query => {}
        Some(b) => *b = Bookmark { name: name.into(), query, ..Default::default() },
        None => marks.push(Bookmark { name: name.into(), query, ..Default::default() }),
    }
}

/// Record what a bookmark found, if the question asked was still its question.
fn note_run(name: &str, query: &Query, frames: usize) -> Result<Option<Vec<Bookmark>>> {
    update(|st| {
        let Some(b) = st.bookmarks.iter_mut().find(|b| b.name == name && b.query == *query) else {
            return Ok(None);
        };
        b.last_frames = frames;
        b.last_run = now();
        Ok(Some(st.bookmarks.clone()))
    })
}

// ---- what the page calls ---------------------------------------------------

/// The settings, less the secrets: the page is told whether there is a
/// password, never what it is.
pub fn settings_json() -> Result<String> {
    let st = load();
    Ok(json!({
        "ok": true,
        "base": st.settings.base,
        "user": st.settings.user,
        "has_password": !st.settings.password.is_empty(),
        "has_key": !st.settings.api_key.is_empty(),
        "rewrites": st.settings.rewrites,
        "bookmarks": st.bookmarks,
        "file": store_file(),
    })
    .to_string())
}

#[derive(Deserialize)]
struct SettingsRequest {
    #[serde(default)]
    base: String,
    #[serde(default)]
    user: String,
    /// Absent keeps the one already kept, which the page never had.
    password: Option<String>,
    api_key: Option<String>,
    #[serde(default)]
    rewrites: Vec<Rewrite>,
}

pub fn save_settings(body: &str) -> Result<String> {
    let req: SettingsRequest = serde_json::from_str(body)?;
    let base = req.base.trim().trim_end_matches('/').to_string();
    if !base.is_empty() {
        base_of(&Settings { base: base.clone(), ..Default::default() })?;
    }
    let rewrites: Vec<Rewrite> = req
        .rewrites
        .into_iter()
        .map(|r| Rewrite { from: r.from.trim().into(), to: r.to.trim().into() })
        .collect();
    ensure!(rewrites.iter().all(|r| !r.from.is_empty()), "a path rewrite needs a folder to replace");
    update(|st| {
        st.settings.base = base;
        st.settings.user = req.user.trim().to_string();
        if let Some(p) = req.password {
            st.settings.password = p;
        }
        if let Some(k) = req.api_key {
            st.settings.api_key = k.trim().to_string();
        }
        st.settings.rewrites = rewrites;
        Ok(())
    })?;
    settings_json()
}

/// What the catalog holds, for the page's suggestions and to show that the
/// connection works.
pub fn summary_json() -> Result<String> {
    let s = load().settings;
    let doc = answer(get(&s, &format!("{}/catalog", base_of(&s)?))?)?;
    // Read afresh whenever the page connects, which is when a catalog that has
    // just been upgraded ought to be noticed.
    let caps = capabilities(&s, true).unwrap_or_default();
    let keys = |v: &Value| v.as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
    Ok(json!({
        "ok": true,
        "total": doc["total"],
        "filters": doc["distinct"]["filters"],
        "imagetypes": doc["distinct"]["imagetypes"],
        "objects": keys(&doc["by_object"]),
        "version": caps.version,
        "screening": caps.screening,
        "preview": caps.preview,
        "params": caps.frame_params.iter()
            .map(|(name, said)| json!({ "name": name, "description": said }))
            .collect::<Vec<_>>(),
    })
    .to_string())
}

/// The queries saved in the catalog itself.
pub fn saved_queries_json() -> Result<String> {
    let s = load().settings;
    let doc = answer(get(&s, &format!("{}/queries", base_of(&s)?))?)?;
    let queries: Vec<Value> = doc
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .map(|q| json!({
            "id": q["id"], "name": q["name"], "folder": q["folder"],
            "expression": q["query"]["expression"],
        }))
        .collect();
    Ok(json!({ "ok": true, "queries": queries }).to_string())
}

#[derive(Deserialize)]
struct FramesRequest {
    query: Query,
    /// What to call the list.
    #[serde(default)]
    name: String,
    /// Write the list, for stacking, rather than only describe what was found.
    #[serde(default)]
    write: bool,
    /// The bookmark this was asked from, whose record of what it found is
    /// brought up to date.
    #[serde(default)]
    bookmark: String,
}

/// Ask a query, find its frames here, and describe them — and when the page is
/// about to stack them, write them down.
pub fn frames(body: &str) -> Result<String> {
    let req: FramesRequest = serde_json::from_str(body)?;
    let settings = load().settings;
    let caps = capabilities(&settings, false);
    let (fetched, total) = fetch_all(&settings, caps.as_ref(), &req.query)?;
    let matched = fetched.len();

    // The catalog's screening, applied here rather than asked of the server,
    // so that what it left out can be counted and a frame not yet judged kept.
    let mut screened: BTreeMap<&str, usize> = BTreeMap::new();
    let mut example: Option<Value> = None;
    let listed: Vec<Listed> = fetched
        .into_iter()
        .filter(|f| {
            let Some(why) = screened_by(f, &req.query) else { return true };
            *screened.entry(why).or_default() += 1;
            if why == "anomaly" && example.is_none() {
                example = Some(json!({ "file": file_name(f), "deviations": f.deviations }));
            }
            false
        })
        .collect();
    // A few hundred look-ups on a network drive, one at a time, is a wait.
    let located: Vec<Located> = listed
        .par_iter()
        .map(|f| locate(f, &settings.rewrites, |p| Path::new(p).is_file()))
        .collect();
    let found: Vec<&str> = located.iter().filter(|l| l.found).map(|l| l.path.as_str()).collect();

    let mut reply = summarise(&listed, &located, total);
    reply["matched"] = json!(matched);
    reply["screened"] = json!(screened);
    reply["unjudged"] = json!(unjudged(&listed, &req.query));
    reply["anomaly_example"] = json!(example);
    if !req.bookmark.is_empty() {
        match note_run(&req.bookmark, &req.query, found.len()) {
            Ok(Some(marks)) => reply["bookmarks"] = json!(marks),
            Ok(None) => {}
            Err(e) => log::warn!("what the bookmark found could not be kept: {e}"),
        }
    }
    if req.write {
        ensure!(matched > 0, "the catalog has no frames matching that");
        ensure!(
            !listed.is_empty(),
            "the catalog matched {matched} frames and its screening left out every one of them"
        );
        ensure!(
            !found.is_empty(),
            "none of the {} frames the catalog named is on this computer (the first was looked \
             for at {}); a path rewrite under Connection and paths maps its folders onto this \
             machine's",
            listed.len(),
            located[0].path
        );
        let name = if req.name.trim().is_empty() { default_name(&req.query) } else { req.name.trim().to_string() };
        let list = write_list(&name, &describe(&req.query), &found)?;
        reply["list"] = json!(list.to_string_lossy());
    }
    reply["ok"] = json!(true);
    Ok(reply.to_string())
}

#[derive(Deserialize)]
struct BookmarkRequest {
    name: String,
    #[serde(default)]
    query: Query,
}

pub fn save_bookmark(body: &str) -> Result<String> {
    let req: BookmarkRequest = serde_json::from_str(body)?;
    let name = req.name.trim().to_string();
    ensure!(!name.is_empty(), "a bookmark needs a name");
    // Refused now, rather than every time it is run.
    params_of(&req.query)?;
    let marks = update(|st| {
        keep_bookmark(&mut st.bookmarks, &name, req.query);
        Ok(st.bookmarks.clone())
    })?;
    Ok(json!({ "ok": true, "bookmarks": marks }).to_string())
}

#[derive(Deserialize)]
struct ForgetRequest {
    name: String,
}

pub fn forget_bookmark(body: &str) -> Result<String> {
    let req: ForgetRequest = serde_json::from_str(body)?;
    let marks = update(|st| {
        st.bookmarks.retain(|b| b.name != req.name);
        Ok(st.bookmarks.clone())
    })?;
    Ok(json!({ "ok": true, "bookmarks": marks }).to_string())
}

/// One frame's thumbnail, fetched with the credentials the page does not have.
///
/// Only a frame id is taken from the page, never an address, so this cannot be
/// used to make requests anywhere but the catalog.
pub fn thumbnail(query: &str) -> Result<(String, Vec<u8>)> {
    let id: u64 = query
        .split('&')
        .find_map(|kv| kv.strip_prefix("id="))
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| anyhow!("no frame asked for"))?;
    // With a size, the larger preview the catalog renders from the file itself.
    let size = query
        .split('&')
        .find_map(|kv| kv.strip_prefix("size="))
        .and_then(|v| v.parse::<u32>().ok());
    let s = load().settings;
    let base = base_of(&s)?;
    let url = match size {
        Some(px) => format!("{base}/frames/{id}/preview?size={}", px.clamp(256, 8192)),
        None => format!("{base}/frames/{id}/thumbnail"),
    };
    let r = get(&s, &url)?;
    ensure!(r.status == 200, "the catalog has no picture of frame {id} (HTTP {})", r.status);
    Ok(("image/jpeg".into(), r.body))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(from: &str, to: &str) -> Rewrite {
        Rewrite { from: from.into(), to: to.into() }
    }

    #[test]
    fn a_catalog_folder_becomes_this_computers_folder() {
        let unix_to_drive = [rule("/mnt/astro/", r"Z:\")];
        assert_eq!(
            rewrite("/mnt/astro/site1/2026-09-09/LIGHT/WR 134/f_0588.fits", &unix_to_drive).as_deref(),
            Some(r"Z:\site1\2026-09-09\LIGHT\WR 134\f_0588.fits")
        );
        // With and without the trailing separators, the join has exactly one.
        assert_eq!(rewrite("/mnt/astro/a.fits", &[rule("/mnt/astro", "Z:")]).as_deref(), Some(r"Z:\a.fits"));
        assert_eq!(rewrite("/mnt/astro/a.fits", &[rule("/mnt/astro", r"Z:\")]).as_deref(), Some(r"Z:\a.fits"));
        assert_eq!(rewrite("/mnt/astro/a.fits", &[rule("/mnt/astro/", "Z:")]).as_deref(), Some(r"Z:\a.fits"));
        // A drive to a share, whichever way the rule's separators lean and in
        // whatever case the catalog wrote the drive.
        assert_eq!(
            rewrite(r"z:\site1\a.fits", &[rule("Z:/", r"\\nas\astro\")]).as_deref(),
            Some(r"\\nas\astro\site1\a.fits")
        );
        // And onto a system with forward slashes.
        assert_eq!(
            rewrite(r"Z:\site1\a.fits", &[rule(r"Z:\", "/Volumes/astro")]).as_deref(),
            Some("/Volumes/astro/site1/a.fits")
        );
        // The first rule that matches, and none at all when none does.
        let two = [rule("/mnt/astro/site1/", "Y:"), rule("/mnt/astro/", "Z:")];
        assert_eq!(rewrite("/mnt/astro/site1/a.fits", &two).as_deref(), Some(r"Y:\a.fits"));
        assert_eq!(rewrite("/mnt/other/a.fits", &two), None);
        assert_eq!(rewrite("/mnt/astro/a.fits", &[rule("  ", "Z:")]), None, "an empty rule matches nothing");
    }

    fn frame(path: &str, host: &str) -> Listed {
        Listed { id: 1, path: path.into(), host_path: host.into(), ..Default::default() }
    }

    #[test]
    fn a_frame_is_found_under_whichever_of_its_names_is_here() {
        let f = frame("/mnt/astro/a.fits", r"Z:\a.fits");
        // The drive the server already rewrote for is mounted: no rules needed.
        let here = |p: &str| p == r"Z:\a.fits";
        assert_eq!(locate(&f, &[], here), Located { path: r"Z:\a.fits".into(), found: true });

        // It is not; a rule the reader wrote is looked at first.
        let share = |p: &str| p == r"\\nas\astro\a.fits";
        let rules = [rule(r"Z:\", r"\\nas\astro\")];
        assert_eq!(locate(&f, &rules, share), Located { path: r"\\nas\astro\a.fits".into(), found: true });

        // Nowhere: the name reported is the one looked at first.
        let nowhere = |_: &str| false;
        assert_eq!(locate(&f, &rules, nowhere), Located { path: r"\\nas\astro\a.fits".into(), found: false });
        assert_eq!(locate(&f, &[], nowhere), Located { path: r"Z:\a.fits".into(), found: false });
        // A catalog with no map configured names only its own path.
        assert_eq!(locate(&frame("/mnt/astro/a.fits", ""), &[], nowhere).path, "/mnt/astro/a.fits");
    }

    #[test]
    fn a_query_becomes_the_request_the_server_documents() {
        let mut q = Query::default();
        q.params.insert("object".into(), "WR 134".into());
        q.params.insert("where".into(), "hfr < median(hfr) + 2*mad(hfr) && path not like '%reject%'".into());
        q.params.insert("date_to".into(), "  ".into());
        let url = frames_url("https://cat.example/api/v1", &q, 2, false).unwrap();
        assert!(url.starts_with("https://cat.example/api/v1/frames?"), "{url}");
        assert!(url.contains("object=WR%20134"), "{url}");
        assert!(
            url.contains("where=hfr%20%3C%20median%28hfr%29%20%2B%202%2Amad%28hfr%29%20%26%26%20path%20not%20like%20%27%25reject%25%27"),
            "{url}"
        );
        assert!(!url.contains("date_to"), "an empty field is not a filter: {url}");
        assert!(url.contains("sort=capture.date_obs"), "{url}");
        assert!(url.contains("page=2") && url.contains("per_page=500"), "{url}");

        // Parameters the page sets are not the query's to set.
        let mut bad = Query::default();
        bad.params.insert("per_page".into(), "5".into());
        assert!(frames_url("https://c", &bad, 1, false).is_err());
        let mut odd = Query::default();
        odd.params.insert("x&y".into(), "1".into());
        assert!(frames_url("https://c", &odd, 1, false).is_err(), "a name cannot smuggle in another parameter");

        // A query saved in the catalog is asked by id, and takes nothing else.
        let saved = Query { saved: 3, saved_name: "Rosette Ha".into(), ..Default::default() };
        let url = frames_url("https://c/api/v1", &saved, 1, false).unwrap();
        assert!(url.starts_with("https://c/api/v1/queries/3/frames?"), "{url}");
        let mut extra = saved.clone();
        extra.params.insert("filter".into(), "Ha".into());
        assert!(frames_url("https://c", &extra, 1, false).is_err());
        extra.params.clear();
        extra.params.insert("include_rejected".into(), "true".into());
        assert!(frames_url("https://c", &extra, 1, false).unwrap().contains("include_rejected=true"));
    }

    #[test]
    fn credentials_go_to_curl_as_config_and_cannot_break_out_of_it() {
        let s = Settings { user: "astro".into(), password: "da\"ta\\".into(), ..Default::default() };
        let c = curl_config(&s, "https://c/api/v1/catalog").unwrap();
        assert_eq!(c, "url = \"https://c/api/v1/catalog\"\nuser = \"astro:da\\\"ta\\\\\"\n");
        // A line break would begin a second option of the password's choosing.
        let s = Settings { password: "x\nurl = \"https://elsewhere\"".into(), ..Default::default() };
        assert!(curl_config(&s, "https://c").is_err());
        // No credentials, no user line.
        assert!(!curl_config(&Settings::default(), "https://c").unwrap().contains("user"));
    }

    #[test]
    fn a_login_page_is_a_refusal_and_not_an_answer() {
        let e = answer(Reply { status: 302, body: b"<html>".to_vec() }).unwrap_err();
        assert!(e.to_string().contains("not accepted"), "{e}");
        let e = answer(Reply {
            status: 400,
            body: br#"{"error":{"code":"invalid_expression","message":"unknown variable hfrr"}}"#.to_vec(),
        })
        .unwrap_err();
        assert!(e.to_string().contains("unknown variable hfrr"), "the server's own words: {e}");
        assert_eq!(answer(Reply { status: 200, body: b"{\"total\":3}".to_vec() }).unwrap()["total"], 3);
    }

    #[test]
    fn what_was_found_is_described_as_what_would_be_stacked() {
        let page: Value = serde_json::from_str(
            r#"{"items":[
              {"id":10,"file":{"path":"/m/a.fits","host_path":"Z:\\a.fits"},
               "capture":{"object":"WR 134","filter":"Ha","date_obs":"2026-09-09T04:00:45","exposure_s":900.0,"camera":"ASI6200MM"}},
              {"id":11,"file":{"path":"/m/b.fits","host_path":"Z:\\b.fits"},
               "capture":{"object":"WR 134","filter":"OIII","date_obs":"2026-09-10T03:30:14","exposure_s":900.0,"camera":"ASI6200MM"}},
              {"id":12,"file":{"path":"/m/c.fits","host_path":"Z:\\c.fits"},
               "capture":{"object":"WR 134","filter":"Ha","date_obs":"N/A","exposure_s":900.0,"camera":null}}
            ]}"#,
        )
        .unwrap();
        let listed: Vec<Listed> = page["items"].as_array().unwrap().iter().map(Listed::from_json).collect();
        assert_eq!(listed[0].host_path, r"Z:\a.fits");
        assert_eq!(listed[2].camera, "", "null is unknown, not a camera called null");

        // Only the drive is here, and one frame is missing from it.
        let located: Vec<Located> = listed
            .iter()
            .map(|f| locate(f, &[], |p| p.starts_with("Z:") && p != r"Z:\b.fits"))
            .collect();
        let s = summarise(&listed, &located, 3);
        assert_eq!(s["found"], 2);
        assert_eq!(s["missing"], 1);
        assert_eq!(s["missing_examples"][0]["catalog"], r"Z:\b.fits");
        // The OIII frame is not here, so it is not in what is described.
        assert_eq!(s["filters"].as_array().unwrap().len(), 1);
        assert_eq!(s["filters"][0]["name"], "Ha");
        assert_eq!(s["filters"][0]["frames"], 2);
        assert_eq!(s["hours"], 0.5);
        assert_eq!(s["dates"], 1, "a date that is not one is not counted");
        assert_eq!(s["thumbs"], json!([10, 12]));
    }

    #[test]
    fn thumbnails_are_spread_across_the_set() {
        assert_eq!(spread(3, 8), [0, 1, 2]);
        assert_eq!(spread(0, 8), Vec::<usize>::new());
        let s = spread(100, 8);
        assert_eq!(s.len(), 8);
        assert_eq!((s[0], s[7]), (0, 99), "the first and the last");
    }

    #[test]
    fn a_list_is_named_for_its_query_in_a_name_any_system_accepts() {
        assert_eq!(slug("Iris / Iris Lum - Bad"), "Iris-Iris-Lum-Bad");
        assert_eq!(slug("WR 134 Ha"), "WR-134-Ha");
        assert_eq!(slug(r#"..\..\etc: "x""#), "etc-x");
        assert_eq!(slug("  ///  "), "catalog");

        let mut q = Query::default();
        assert_eq!(default_name(&q), "catalog");
        q.params.insert("object".into(), "WR 134".into());
        q.params.insert("filter".into(), "Ha".into());
        assert_eq!(default_name(&q), "WR 134 Ha");
        let saved = Query { saved: 3, saved_name: "Rosette / Rosette Ha".into(), ..Default::default() };
        assert_eq!(default_name(&saved), "Rosette / Rosette Ha");
    }

    #[test]
    fn a_written_list_is_one_the_stacker_reads() {
        let dir = std::env::temp_dir().join("smokstak-catalog-list-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let frames: Vec<String> = (0..3)
            .map(|i| {
                let p = dir.join(format!("WR 134_{i}.fits"));
                std::fs::write(&p, b"SIMPLE").unwrap();
                p.to_string_lossy().into_owned()
            })
            .collect();
        let refs: Vec<&str> = frames.iter().map(String::as_str).collect();
        let list = write_list("WR 134 Ha", "object=WR 134,\nfilter=Ha", &refs).unwrap();
        assert_eq!(list.file_name().unwrap(), "WR-134-Ha.txt");
        assert_eq!(sr_raw::collect_files(&list, None).unwrap().len(), 3);
        // Asked again, the list is a new file and the first is left alone.
        let again = write_list("WR 134 Ha", "", &refs[..1]).unwrap();
        assert_ne!(again, list);
        assert_eq!(sr_raw::collect_files(&list, None).unwrap().len(), 3);

        let _ = std::fs::remove_dir_all(list.parent().unwrap());
        let _ = std::fs::remove_dir_all(again.parent().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bookmark_keeps_its_question_and_forgets_the_old_answer_when_it_changes() {
        let mut q = Query::default();
        q.params.insert("filter".into(), "Ha".into());
        let mut marks = Vec::new();
        keep_bookmark(&mut marks, "Rosette Ha", q.clone());
        marks[0].last_frames = 412;
        keep_bookmark(&mut marks, "Rosette Ha", q.clone());
        assert_eq!(marks.len(), 1);
        assert_eq!(marks[0].last_frames, 412, "the same question, the same record");
        q.params.insert("rejected".into(), "false".into());
        keep_bookmark(&mut marks, "Rosette Ha", q.clone());
        assert_eq!(marks.len(), 1);
        assert_eq!(marks[0].last_frames, 0, "a different question has not been asked yet");
        assert_eq!(marks[0].query, q);
    }

    #[test]
    fn settings_and_bookmarks_outlive_the_server() {
        let dir = std::env::temp_dir().join("smokstak-catalog-store-test");
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("catalog.json");
        assert_eq!(load_from(&file).unwrap().settings.base, DEFAULT_BASE, "nothing kept yet");

        let mut st = Store::default();
        st.settings.user = "astro".into();
        st.settings.password = "data".into();
        st.settings.rewrites.push(rule("/mnt/astro/", r"Z:\"));
        keep_bookmark(&mut st.bookmarks, "WR 134", Query::default());
        save_to(&file, &st).unwrap();

        let back = load_from(&file).unwrap();
        assert_eq!(back.settings.password, "data");
        assert_eq!(back.settings.rewrites, st.settings.rewrites);
        assert_eq!(back.bookmarks, st.bookmarks);

        // A file that cannot be read is not quietly replaced by an empty one.
        std::fs::write(&file, b"{ not json").unwrap();
        assert!(load_from(&file).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_catalog_says_what_it_takes_and_a_query_it_would_ignore_is_refused() {
        let spec = json!({
            "info": { "version": "1.1.0" },
            "paths": {
                "/frames": { "get": { "parameters": [
                    { "name": "object", "description": "OBJECT contains this text." },
                    { "name": "page" },
                    { "name": "occluded", "schema": { "description": "true = part of the field blocked." } }
                ] } },
                "/frames/{frame_id}/preview": { "get": {} }
            },
            "components": { "schemas": {
                "Frame": { "properties": { "sequence": {} } },
                "QualityInfo": { "properties": { "trailed": {}, "occluded": {}, "satellite": {} } }
            } }
        });
        let caps = capabilities_of(&spec, "https://c/api/v1");
        assert_eq!(caps.version, "1.1.0");
        assert!(caps.screening && caps.preview);
        assert_eq!(
            caps.frame_params,
            [
                ("object".to_string(), "OBJECT contains this text.".to_string()),
                ("occluded".to_string(), "true = part of the field blocked.".to_string()),
            ],
            "what the page sets itself is not offered"
        );

        let mut q = Query::default();
        q.params.insert("object".into(), "WR 134".into());
        assert!(check_supported(&q, &caps).is_ok());
        // Mistyped: the server would ignore it and match every frame.
        q.params.insert("hfr_mx".into(), "2".into());
        let e = check_supported(&q, &caps).unwrap_err();
        assert!(e.to_string().contains("hfr_mx"), "{e}");

        // A catalog from before screening.
        let old = capabilities_of(
            &json!({ "info": { "version": "1.0.0" },
                     "paths": { "/frames": { "get": { "parameters": [{ "name": "object" }] } } } }),
            "https://c",
        );
        assert!(!old.screening && !old.preview);
        let screened = Query { screen: ["anomaly".to_string()].into(), ..Default::default() };
        assert!(check_supported(&screened, &old).is_err());
        assert!(check_supported(&screened, &caps).is_ok());

        // Screening asks for the verdicts and the session comparison.
        let url = frames_url("https://c/api/v1", &screened, 1, true).unwrap();
        assert!(url.contains("expand=sequence") && url.contains("quality.occluded"), "{url}");
        let unknown = Query { screen: ["cloudy".to_string()].into(), ..Default::default() };
        assert!(frames_url("https://c", &unknown, 1, false).is_err());
    }

    #[test]
    fn only_a_frame_the_catalog_has_judged_is_screened_out() {
        let page: Value = serde_json::from_str(
            r#"[
              {"id":1,"quality":{"trailed":false,"occluded":false,"satellite":true},"status":{"grade":null},
               "sequence":{"judged":true,"anomaly":true,"star_drop":0.25,"hfr_rise":0.27,"flux_drop":null}},
              {"id":2,"quality":{"trailed":null,"occluded":null,"satellite":null},"status":{"grade":null},
               "sequence":{"judged":false,"anomaly":null}},
              {"id":3,"quality":{"trailed":true,"occluded":true},"status":{"grade":"rejected"},
               "sequence":{"anomaly":true}},
              {"id":4}
            ]"#,
        )
        .unwrap();
        let frames: Vec<Listed> = page.as_array().unwrap().iter().map(Listed::from_json).collect();
        assert_eq!(frames[0].deviations.get("star_drop"), Some(&0.25));
        assert!(!frames[0].deviations.contains_key("flux_drop"), "null is not a measurement");

        let mut q = Query {
            screen: ["anomaly", "trailed", "occluded"].map(String::from).into(),
            ..Default::default()
        };
        let why: Vec<Option<&str>> = frames.iter().map(|f| screened_by(f, &q)).collect();
        // A satellite trail was not asked about; a frame not judged, and one
        // from a catalog that does not screen, are kept.
        assert_eq!(why, [Some("anomaly"), None, Some("trailed"), None]);
        assert_eq!(unjudged(&frames, &q).get("anomaly"), Some(&2));

        // Graded rejected counts when rejected frames are being left out.
        q.params.insert("rejected".into(), "false".into());
        assert_eq!(screened_by(&frames[2], &q), Some("rejected"));
        q.screen.clear();
        q.params.clear();
        assert!(frames.iter().all(|f| screened_by(f, &q).is_none()), "nothing asked, nothing left out");

        // A bookmark kept before screening existed asks for none.
        let old: Bookmark =
            serde_json::from_str(r#"{"name":"Rosette Ha","query":{"params":{"filter":"Ha"}}}"#).unwrap();
        assert!(old.query.screen.is_empty());
    }
}
