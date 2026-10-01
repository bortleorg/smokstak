//! A local page for people who would rather not type flags.
//!
//! The stacker is a command-line program and stays one: this serves a page on
//! the loopback interface and drives `smokstak` itself, one child process per
//! run, reading its output to say where the run has got to. Nothing here
//! reimplements any of the pipeline, so the two cannot disagree about what a
//! reconstruction is.
//!
//! ## Why a page and not a window
//!
//! The three things asked of the front end — take a file dropped on it, take a
//! list pasted into it, and open a file chooser — a browser already does, and
//! does properly on every platform. A native toolkit would bring a hundred
//! crates to a program that has thirty, for a window that would still have to
//! spawn the same child process. So the server is written here against
//! `std::net`, in about as much code as the argument parser, and the page is a
//! single file.
//!
//! ## What it will not do
//!
//! It binds to `127.0.0.1` and serves only its own page and its own endpoints.
//! It never builds a shell command: every argument is pushed onto a
//! `Command` as its own string, so a path with a space or a quote in it is a
//! path with a space or a quote in it and cannot become anything else.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::catalog;

#[path = "gui_workflows.rs"]
mod workflows;
#[path = "gui_mosaic.rs"]
mod mosaic;

/// The stages a run passes through, in the order the page draws them.
///
/// Derived from what the pipeline already prints rather than from a progress
/// callback threaded through every crate: the lines below are the ones it logs
/// at each transition, and a run that changes its mind about its own stages
/// will say so in the log before it says so here.
const STAGES: [(&str, &str); 8] = [
    ("read", "Reading the list"),
    ("decode", "Decoding frames"),
    ("register", "Registering"),
    ("photometry", "Matching brightness"),
    ("reject", "Rejecting outliers"),
    ("merge", "Merging"),
    ("finish", "Colour and background"),
    ("write", "Writing"),
];

/// Which stage a line of the child's output means we have reached.
///
/// First match wins and the stage only ever moves forward, so a late mention
/// of an earlier stage — the summary block at the end names most of them —
/// cannot walk the progress backwards.
fn stage_of(line: &str) -> Option<usize> {
    const MARKS: [(&str, usize); 14] = [
        ("frames listed", 0),
        ("decoding ", 1),
        ("decoded ", 2),
        ("registration proxies", 2),
        ("pyramids", 2),
        ("global registration", 2),
        ("star refinement", 2),
        ("photometry:", 3),
        ("Photometric match", 3),
        ("robustness", 4),
        ("merging ", 5),
        ("merge finished", 6),
        ("Rendered:", 6),
        ("Wrote ", 7),
    ];
    MARKS.iter().find(|(m, _)| line.contains(m)).map(|(_, s)| *s)
}

/// One palette, as a picture rather than as a name.
#[derive(Clone, Serialize, Deserialize)]
struct PaletteShot {
    id: String,
    label: String,
    preview: String,
}

/// A colour image combined from a run's masters.
///
/// Kept beside the run rather than in place of it. Choosing a palette is done
/// by making one, looking, and making another, and each of those used to
/// replace the stack on the page — its masters, what it said about itself, and
/// the chance to combine them again.
#[derive(Clone, Serialize, Deserialize)]
struct ColourImage {
    title: String,
    files: Vec<String>,
    /// Which filter went to which primary, as the compositor put it.
    mapping: String,
}

/// A decision the run made and reported, worth reading without reading the log.
///
/// A run says a great deal in passing — how much of the burst it threw away,
/// whether the frames were dithered enough for the scale asked of them, how it
/// stretched the result. Every word of it went into a four-hundred-line tail
/// that nobody reads, and these are the lines that answer "was that a good
/// stack?".
#[derive(Clone, Serialize, Deserialize)]
struct Finding {
    /// The filter this belongs to, empty for a run that had only one.
    group: String,
    label: String,
    text: String,
}

/// Lines worth lifting out of the log: what marks one, what to call it, and a
/// word the line must also carry.
///
/// The marker is looked for anywhere in the line rather than at its start,
/// because half of these arrive through the logger with a timestamp in front
/// and half are printed plainly. The third column is there because `kernel: `
/// prefixes two different lines and only one of them is the frame count.
const FINDINGS: [(&str, &str, &str); 8] = [
    ("Sampling headroom: ", "Sampling", ""),
    ("Lucky-region selection: ", "Lucky regions", ""),
    ("robustness: ", "Outliers", "suppressed on average"),
    // The other `robustness:` line, and a different fact: not how much was
    // thrown away but whether the frame everything is compared against was
    // itself the odd one out. Sharing a heading, only one of the two survived.
    ("robustness: ", "Reference", "disagreed with the reference"),
    ("kernel: ", "Frames used", "contributing frames"),
    ("Samples:", "Samples", "merged"),
    ("Encoding:", "Encoding", ""),
    ("Rendered:", "Rendered", ""),
];

fn finding_of(line: &str) -> Option<(&'static str, String)> {
    FINDINGS.iter().find_map(|(needle, label, must)| {
        if !must.is_empty() && !line.contains(must) {
            return None;
        }
        let at = line.find(needle)?;
        let text = line[at + needle.len()..].trim();
        (!text.is_empty()).then(|| (*label, text.to_string()))
    })
}

/// The file a line of the run's output says it wrote, if it says so.
///
/// The run names each file and then, mostly, says something about it in
/// parentheses. The path ends where that last group begins, and not at the
/// first parenthesis in the line: a list saved from a browser is
/// `approved_lights (10).txt`, the result is named after it, and cutting there
/// lost every file the run wrote.
fn announced_path(line: &str) -> Option<&str> {
    let rest = ["Wrote ", "Preview written to ", "Background model written to "]
        .iter()
        .find_map(|prefix| line.strip_prefix(prefix))?
        .trim();
    if rest.ends_with(')') {
        let mut depth = 0usize;
        for (i, c) in rest.char_indices().rev() {
            match c {
                ')' => depth += 1,
                '(' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(path) = rest[..i].strip_suffix(' ') {
                            return Some(path.trim_end());
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    Some(rest)
}

/// What the page is told about a run in progress.
///
/// Also what is kept of a finished run between sessions, which is why every
/// field has a default: a list written by an older build still reads.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Progress {
    prepared_plan: Option<String>,
    preparation_detail: Option<String>,
    tiles_done: Option<usize>,
    tiles_total: Option<usize>,
    running: bool,
    finished: bool,
    failed: bool,
    stage: usize,
    /// Seconds since the run began.
    elapsed: f64,
    /// The tail of the child's output, newest last.
    log: Vec<String>,
    /// Files the run said it wrote.
    outputs: Vec<String>,
    message: String,
    /// `stack` or `composite`. They are different lengths and pass through
    /// different stages, and the page draws them differently.
    kind: String,
    /// The handful of lines that say how the run went.
    findings: Vec<Finding>,
    /// Which filter the run is on, for a run doing one filter at a time.
    group: String,
    /// What this run was, in the few words that tell it from the last one.
    title: String,
    /// Which run of the work this is, and how many there are. A night stacked
    /// one filter at a time is several runs and one piece of work.
    step: usize,
    steps: usize,
    /// What the run in front of you is doing: "filter O, batch 2 of 2".
    step_name: String,
    /// The palettes tried on, for choosing between by eye.
    palettes: Vec<PaletteShot>,
    /// The colour images made from this run's masters, oldest first.
    colour: Vec<ColourImage>,
    /// When it ended, in seconds since the epoch, for the page to render in
    /// whatever the reader's clock says.
    finished_at: f64,
    /// Which piece of work this is. Palettes and colour images made afterwards
    /// carry the id of the stack they were made from, and are added to it.
    id: u64,
    /// How long the stack took. `elapsed` belongs to whatever ran last, and
    /// a combine that took four seconds is not how long the stack took.
    took: f64,
    /// Whether this is palettes or a colour image made from a finished stack,
    /// rather than a stack. Not read from `kind`: a night stacked in batches
    /// ends with a run that adds them, which is also a `composite`.
    extends: bool,
}

#[derive(Default)]
struct Job {
    mosaic_outputs: std::collections::HashSet<PathBuf>,
    progress: Progress,
    child: Option<Child>,
    cancel: Option<Arc<AtomicBool>>,
    /// When the whole piece of work began, so that several runs report one
    /// elapsed time rather than restarting the clock at each.
    began: Option<Instant>,
    /// Where palettes being tried on are written, if that is what is running.
    palette_dir: Option<PathBuf>,
    /// Stop was pressed. Kept because a plan is built before its first child
    /// exists — reading 634 headers takes a moment — and Stop in that moment
    /// used to be answered with "Stopped." and then a run starting anyway.
    stop: bool,
    /// Runs still to come, for work that is several runs long.
    ///
    /// Project mutations followed by a rebuild are one piece of work, so
    /// they share progress, a log and published outputs.
    queue: std::collections::VecDeque<Planned>,
    /// What this session's finished runs produced, newest last.
    ///
    /// A stack is slow enough that the way to use this page is to try
    /// something, look, and try again — and until now each look destroyed the
    /// one before it. Kept in the server rather than the page so that a reload
    /// does not lose them, and only for as long as the server runs: this is
    /// the record of an afternoon's work, not an archive.
    past: Vec<Progress>,
    /// Files this session's runs have said they wrote, and the only files the
    /// page is allowed to fetch back.
    ///
    /// The page shows the preview of a finished image rather than only naming
    /// it, which means the server has to hand a file back over HTTP. It hands
    /// back these and nothing else: not a path the page asks for, not a path
    /// under some root, only a path a run of this program announced writing.
    /// A browser is a hostile place to open a door in, and this one opens onto
    /// eight files.
    served: std::collections::HashSet<PathBuf>,
    /// The id the last piece of work was given.
    next_id: u64,
    /// Where `past` is kept between sessions, if anywhere.
    history: Option<PathBuf>,
    /// Every frame of a set being measured, or measured.
    survey: SurveyState,
    survey_child: Option<Child>,
    /// Which measurement is the current one. A reader who moves on to another
    /// set starts another, and what the first one's threads have left to say
    /// belongs to nobody.
    survey_gen: u64,
}

/// What the page is told about measuring every frame of a set.
#[derive(Clone, Default, Serialize)]
struct SurveyState {
    running: bool,
    finished: bool,
    stopped: bool,
    /// The set measured, so that one set's measurements are never shown as
    /// another's.
    input: String,
    done: usize,
    total: usize,
    error: String,
    /// One row per frame, as `smokstak survey` wrote them.
    frames: serde_json::Value,
}

/// One child process still to be run, and what to call it while it runs.
///
/// Held as its parts rather than as a `Command` because a queue has to outlive
/// the request that built it.
struct Planned {
    step: String,
    program: PathBuf,
    args: Vec<String>,
    /// The last stage this run can reach, for the progress list.
    last_stage: usize,
    kind: &'static str,
}

impl Planned {
    fn command(&self) -> Command {
        let mut c = Command::new(&self.program);
        c.args(&self.args);
        c
    }
}

impl Job {
    /// Record a file a run wrote, and allow the page to fetch it.
    fn announce(&mut self, path: String) {
        let p = PathBuf::from(&path);

        // A palette being tried on for size is not a result. Its preview goes
        // to the chooser and its full-size file is a means to that preview,
        // which nobody asked for and which is deleted once it has been made.
        if let Some(dir) = self.palette_dir.clone()
            && p.parent() == Some(dir.as_path()) {
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
                let id = stem.trim_end_matches(".preview").to_string();
                if p.extension().and_then(|e| e.to_str()) == Some("png") {
                    let label = CANDIDATES
                        .iter()
                        .find(|c| c.id == id)
                        .map(|c| c.label)
                        .unwrap_or("a palette");
                    self.progress.palettes.push(PaletteShot {
                        id,
                        label: label.to_string(),
                        preview: path.clone(),
                    });
                    self.served.insert(std::fs::canonicalize(&p).unwrap_or(p));
                } else {
                    let _ = std::fs::remove_file(&p);
                }
                return;
            }

        // A colour image is not a master. Put among them, it would be offered
        // as a channel of the next colour image.
        if self.progress.extends && self.progress.kind == "composite"
            && let Some(c) = self.progress.colour.last_mut() {
                if !c.files.contains(&path) {
                    c.files.push(path.clone());
                }
                self.served.insert(std::fs::canonicalize(&p).unwrap_or(p));
                return;
            }

        if !self.progress.outputs.contains(&path) {
            self.progress.outputs.push(path.clone());
        }
        self.served.insert(std::fs::canonicalize(&p).unwrap_or(p));
    }
}

/// Serve the page until the process is killed.
pub fn run(port: u16, open_browser: bool) -> Result<()> {
    let exe = std::env::current_exe()?;
    let listener = bind(port)?;
    let url = format!("http://{}", listener.local_addr()?);
    println!("smokstak is listening on {url}");
    println!("Leave this window open while you use it; close it to stop.");
    if open_browser {
        let _ = open_in_browser(&url);
    }

    let mut state = Job { history: history_file(), ..Default::default() };
    if let Some(file) = state.history.clone() {
        restore(&mut state, load_history(&file));
        if !state.past.is_empty() {
            println!("{} earlier runs, kept in {}", state.past.len(), file.display());
        }
    }
    let job: Arc<Mutex<Job>> = Arc::new(Mutex::new(state));
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let job = Arc::clone(&job);
        let exe = exe.clone();
        std::thread::spawn(move || {
            if let Err(e) = serve(stream, &job, &exe) {
                log::debug!("gui: {e}");
            }
        });
    }
    Ok(())
}

/// The requested port, or the next few if something already holds it.
fn bind(port: u16) -> Result<TcpListener> {
    for p in port..port.saturating_add(20) {
        if let Ok(l) = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, p)) {
            return Ok(l);
        }
    }
    Err(anyhow!("no free port between {port} and {}", port + 20))
}

fn open_in_browser(url: &str) -> Result<()> {
    #[cfg(windows)]
    let mut c = {
        let mut c = Command::new("cmd");
        c.args(["/C", "start", "", url]);
        c
    };
    #[cfg(target_os = "macos")]
    let mut c = {
        let mut c = Command::new("open");
        c.arg(url);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut c = {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };
    c.spawn()?;
    Ok(())
}

/// One request, one response, connection closed. There is one client.
fn serve(mut stream: TcpStream, job: &Arc<Mutex<Job>>, exe: &Path) -> Result<()> {
    let req = read_request(&mut stream)?;

    // A loopback server is reachable by any page the browser happens to be
    // showing: a site can point a name it controls at 127.0.0.1 and then make
    // requests here that the browser considers same-origin. What it cannot do
    // is send a Host of `localhost` or `127.0.0.1`, because that is not the
    // name it was loaded from. So that is what is required.
    if !host_is_loopback(&req.host) {
        return respond(&mut stream, 403, "text/plain", b"not for other sites".to_vec());
    }
    // What a site can do is post a form here: the browser sends it to the
    // address it was aimed at, so the Host is ours, and with the Origin of the
    // page that sent it, which is not. That mattered little while a post could
    // only start a stack. It matters now that one can change where the
    // catalog's password is sent.
    if req.method == "POST" && !req.origin.is_empty() && !origin_is_loopback(&req.origin) {
        return respond(&mut stream, 403, "text/plain", b"not for other sites".to_vec());
    }

    let (route, query) = match req.path.split_once('?') {
        Some((r, q)) => (r, q),
        None => (req.path.as_str(), ""),
    };
    let (status, kind, body) = match (req.method.as_str(), route) {
        ("GET", "/") => (
            200,
            "text/html; charset=utf-8".to_string(),
            // The stage list is defined once, in Rust, and pasted into the page
            // so the two cannot drift apart.
            include_str!("gui.html")
                .replace("__STAGES__", &stages_json())
                .into_bytes(),
        ),
        ("POST", "/inspect") => json_reply(inspect(&req.body, exe)),
        ("POST", "/start") => json_reply(start(&req.body, exe, job)),
        ("POST", "/mosaic/inspect") => json_reply(mosaic::inspect(&req.body)),
        ("POST", "/mosaic/pick-plan") => json_reply(mosaic::pick_plan()),
        ("POST", "/mosaic/start") => json_reply(mosaic::start(&req.body, exe, job)),
        ("POST", "/mosaic/prepare") => json_reply(mosaic::prepare(&req.body, exe, job)),
        ("POST", "/workflow") => json_reply(workflows::start(&req.body, exe, job)),
        ("POST", "/open-report") => json_reply(workflows::open_report(&req.body, job)),
        ("POST", "/composite") => json_reply(composite(&req.body, exe, job)),
        ("POST", "/palettes") => json_reply(palettes(&req.body, exe, job)),
        ("POST", "/survey") => json_reply(survey(&req.body, exe, job)),
        ("GET", "/survey") => {
            let s = job.lock().unwrap().survey.clone();
            (200, "application/json".to_string(), serde_json::to_string(&s)?.into_bytes())
        }
        ("POST", "/survey-stop") => {
            let mut j = job.lock().unwrap();
            if j.survey.running {
                j.survey.stopped = true;
                if let Some(c) = j.survey_child.as_mut() {
                    let _ = c.kill();
                }
            }
            (200, "application/json".to_string(), b"{\"ok\":true}".to_vec())
        }
        ("POST", "/forget") => {
            let mut j = job.lock().unwrap();
            j.past.clear();
            save_history(&j);
            (200, "application/json".to_string(), b"{\"ok\":true}".to_vec())
        }
        ("GET", "/status") => {
            let p = job.lock().unwrap().progress.clone();
            (200, "application/json".to_string(), serde_json::to_string(&p)?.into_bytes())
        }
        ("GET", "/history") => {
            // Newest first, which is the order they are wanted in.
            let mut past = job.lock().unwrap().past.clone();
            past.reverse();
            (200, "application/json".to_string(), serde_json::to_string(&past)?.into_bytes())
        }
        ("GET", "/file") => match serve_file(query, job) {
            Ok((kind, bytes)) => (200, kind, bytes),
            Err(e) => (404, "text/plain".to_string(), e.to_string().into_bytes()),
        },
        ("POST", "/cancel") => {
            let mut j = job.lock().unwrap();
            if let Some(c) = &j.cancel {
                c.store(true, Ordering::SeqCst);
            }
            // Whatever is running, and whatever was going to run after it: a
            // night stacked a filter at a time is several runs, and stopping
            // the one in front of you is not stopping the work.
            j.queue.clear();
            j.stop = true;
            if let Some(child) = j.child.as_mut() {
                let _ = child.kill();
            }
            if j.progress.running {
                j.progress.message = "Stopped.".into();
            }
            (200, "application/json".to_string(), b"{\"ok\":true}".to_vec())
        }
        ("POST", "/reveal") => json_reply(reveal(&req.body)),
        ("POST", "/pick") => json_reply(pick_folder()),
        ("GET", "/catalog/settings") => json_reply(catalog::settings_json()),
        ("POST", "/catalog/settings") => json_reply(catalog::save_settings(&req.body)),
        ("GET", "/catalog/summary") => json_reply(catalog::summary_json()),
        ("GET", "/catalog/queries") => json_reply(catalog::saved_queries_json()),
        ("POST", "/catalog/frames") => json_reply(catalog::frames(&req.body)),
        ("POST", "/catalog/bookmark") => json_reply(catalog::save_bookmark(&req.body)),
        ("POST", "/catalog/forget") => json_reply(catalog::forget_bookmark(&req.body)),
        ("GET", "/catalog/thumb") => match catalog::thumbnail(query) {
            Ok((kind, bytes)) => (200, kind, bytes),
            Err(e) => (404, "text/plain".to_string(), e.to_string().into_bytes()),
        },
        _ => (404, "text/plain".to_string(), b"not found".to_vec()),
    };
    respond(&mut stream, status, &kind, body)
}

fn respond(stream: &mut TcpStream, status: u16, kind: &str, body: Vec<u8>) -> Result<()> {
    let reason = match status {
        200 => "OK",
        403 => "Forbidden",
        404 => "Not Found",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

/// Whether the `Host` of a request is one this server can legitimately be
/// reached by. An absent header is allowed: that is a client speaking HTTP/1.0
/// or a hand-written request, not a browser, and every browser sends one.
fn host_is_loopback(host: &str) -> bool {
    if host.is_empty() {
        return true;
    }
    let name = host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host);
    let name = name.trim_matches(['[', ']']);
    name.eq_ignore_ascii_case("localhost") || name == "127.0.0.1" || name == "::1"
}

/// Whether the page a request came from is one served here. `null`, which is
/// what a sandboxed frame or a local file sends, is not.
fn origin_is_loopback(origin: &str) -> bool {
    origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
        .is_some_and(|rest| !rest.is_empty() && host_is_loopback(rest))
}

fn json_reply(r: Result<String>) -> (u16, String, Vec<u8>) {
    match r {
        Ok(s) => (200, "application/json".to_string(), s.into_bytes()),
        Err(e) => (
            200,
            "application/json".to_string(),
            serde_json::json!({ "ok": false, "error": e.to_string() })
                .to_string()
                .into_bytes(),
        ),
    }
}

/// Hand back one file a run wrote, so the page can show the picture rather
/// than only its path. Anything not on that list is a 404, including a file
/// that plainly exists.
fn serve_file(query: &str, job: &Arc<Mutex<Job>>) -> Result<(String, Vec<u8>)> {
    let want = query
        .split('&')
        .find_map(|kv| kv.strip_prefix("path="))
        .map(percent_decode)
        .ok_or_else(|| anyhow!("no file asked for"))?;
    let p = PathBuf::from(&want);
    let canonical = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
    {
        let j = job.lock().unwrap();
        if !j.served.contains(&canonical) && !j.served.contains(&p) {
            return Err(anyhow!("not a file this run wrote"));
        }
        if mosaic::is_output(&j, &canonical) || mosaic::is_output(&j, &p) {
            return mosaic::serve(&canonical);
        }
    }
    let kind = match p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("tif" | "tiff") => "image/tiff",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    };
    Ok((kind.to_string(), std::fs::read(&p)?))
}

/// `%20` and friends. The page encodes the paths it sends; Windows paths are
/// full of characters that mean something else in a query string.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(b[i]);
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// One HTTP request, as much of it as this server reads.
struct Request {
    method: String,
    path: String,
    host: String,
    origin: String,
    body: String,
}

/// Method, path, host and body of one HTTP request.
///
/// Enough of the protocol for a page talking to itself: a request line, headers
/// until a blank line, and exactly `Content-Length` bytes of body.
fn read_request(stream: &mut TcpStream) -> Result<Request> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let mut length = 0usize;
    let mut host = String::new();
    let mut origin = String::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            length = v.trim().parse().unwrap_or(0);
        }
        if let Some(v) = lower.strip_prefix("host:") {
            host = v.trim().to_string();
        }
        if let Some(v) = lower.strip_prefix("origin:") {
            origin = v.trim().to_string();
        }
    }
    let mut body = vec![0u8; length.min(64 * 1024 * 1024)];
    if !body.is_empty() {
        reader.read_exact(&mut body)?;
    }
    Ok(Request {
        method,
        path,
        host,
        origin,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

#[derive(Deserialize)]
struct InspectRequest {
    /// A path to a list, a directory or a single frame. Empty when the page is
    /// sending the text of a list instead.
    #[serde(default)]
    path: String,
    /// The contents of a list the user dropped or pasted.
    #[serde(default)]
    text: String,
    /// What the dropped file was called, so the working copy can keep its name.
    #[serde(default)]
    name: String,
}

/// Where a list pasted into the page is put so that the stacker can read it.
///
/// A dropped file arrives as text: the browser gives a page its contents and
/// its name and never its path, which is the right decision on the browser's
/// part and an inconvenience here. The copy is written to a unique temporary
/// file when paths are absolute, and refused when they are not, because a
/// relative path in a list means "relative to the list" and the original list's
/// directory is exactly what was not sent.
fn list_from_text(text: &str, name: &str) -> Result<PathBuf> {
    let entries: Vec<&str> = text
        .lines()
        .map(|l| l.trim().trim_start_matches('\u{feff}').trim().trim_matches('"'))
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if entries.is_empty() {
        return Err(anyhow!("that list has no frames in it"));
    }
    if let Some(rel) = entries.iter().find(|e| Path::new(e).is_relative()) {
        return Err(anyhow!(
            "the list holds relative paths, starting with {rel}. A dropped file \
             arrives without its folder, so relative paths cannot be resolved: \
             type the path to the list instead, or use a list of absolute paths."
        ));
    }
    let stem = Path::new(name).file_stem().and_then(|s| s.to_str()).unwrap_or("dropped");
    let stem: String = stem.chars().filter(|c|c.is_ascii_alphanumeric() || *c=='-').take(40).collect();
    static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..64 {
        let unique=NEXT_LIST.fetch_add(1,Ordering::Relaxed);
        let stamp=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let target=std::env::temp_dir().join(format!("smokstak-{stem}-{}-{stamp}-{unique}.txt",std::process::id()));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&target) {
            Ok(mut file)=> { file.write_all(format!("{}\n",entries.join("\n")).as_bytes())?; return Ok(target); }
            Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists => continue,
            Err(e)=>return Err(e.into()),
        }
    }
    Err(anyhow!("Cannot reserve a unique frame list"))
}

fn resolve_input(req: &InspectRequest) -> Result<PathBuf> {
    if !req.path.trim().is_empty() {
        let p = PathBuf::from(req.path.trim().trim_matches('"'));
        if !p.exists() {
            return Err(anyhow!("{} does not exist", p.display()));
        }
        return Ok(p);
    }
    if !req.text.trim().is_empty() {
        return list_from_text(&req.text, &req.name);
    }
    Err(anyhow!("nothing to read: drop a list, paste one, or type a path"))
}

/// The frames of a burst, gathered by the filter they were taken through.
///
/// Read from the headers rather than decoded: a filter name is a few bytes at
/// the front of the file, and 634 of them take about a second.
///
/// This is what decides whether a night can be stacked at all. Frames through
/// different filters cannot be merged, so they are stacked apart whatever
/// happens — and once that is true the memory a run needs is the memory of the
/// largest filter, not of the whole set. For a set of 634 exposures in three
/// filters that is the difference between 157 GiB, which no machine here has,
/// and 80, which this one does.
fn frames_by_filter(paths: &[PathBuf]) -> Vec<(String, Vec<PathBuf>)> {
    let mut groups: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for p in paths {
        let name = sr_raw::peek_filter(p).unwrap_or_default();
        match groups.iter_mut().find(|(n, _)| *n == name) {
            Some((_, v)) => v.push(p.clone()),
            None => groups.push((name, vec![p.clone()])),
        }
    }
    // Largest first: it is the one that decides what will fit.
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
    groups
}

/// What a child said, turned into something worth reading.
///
/// Running out of memory does not come back as an error. The allocator aborts
/// the process, and what reaches the page is `memory allocation of 61171488
/// bytes failed` followed by an invitation to set RUST_BACKTRACE — which names
/// the last straw rather than the load, and suggests debugging a program the
/// reader did not write. The run says what it needs before it starts, so that
/// is what to say back.
fn readable_failure(text: &str) -> String {
    let text = text.trim();
    let out_of_memory = text.contains("memory allocation of")
        || text.contains("Cannot allocate memory")
        || text.contains("not enough memory");
    if !out_of_memory {
        return text.to_string();
    }
    let needed = text
        .split_once("needs roughly ")
        .and_then(|(_, r)| r.split_once(" of memory"))
        .map(|(n, _)| n.to_string());
    match needed {
        Some(n) => format!(
            "This set does not fit in memory: it needs about {n} for the frames alone. \
             Stack fewer of them at a time — the frame count below sets how many — or \
             reconstruct part of the sky."
        ),
        None => "This set does not fit in memory. Stack fewer frames at a time — the frame \
                 count below sets how many."
            .to_string(),
    }
}

/// How much of a burst the page is willing to decode to describe it.
///
/// Reading the frames used to mean reading all of them, and a burst is held in
/// full: 634 exposures of a 61-megapixel sensor is 156 GiB, so the first thing
/// the page did with a real night's data was abort on an allocation failure,
/// before anything had been chosen. Describing a set does not need all of it —
/// a sample spread across the burst says what camera, what filters, what
/// exposure, and how sharp — so the page reads as much as fits in this and
/// says how much that was.
const INSPECT_BUDGET: u64 = 3 * 1024 * 1024 * 1024;

/// And no more frames than this even when they would fit, because what the
/// report says about a set — the camera, the filters, the exposure, how sharp
/// the night was — stops improving long before the hundredth frame.
const MAX_INSPECT_FRAMES: usize = 24;

/// Where the pipeline itself starts warning about the memory a burst needs.
///
/// Taken from the run rather than guessed at again here, so the page and the
/// program agree about what counts as a lot. Used only when the machine will
/// not say how much memory it has.
const HEAVY: u64 = 24 * 1024 * 1024 * 1024;

/// How much of the machine's memory a burst may be allowed to occupy.
///
/// The frames are held for the whole run and the reconstruction needs room
/// beside them, so this is well short of all of it.
const OF_MEMORY: f64 = 0.55;

/// What this machine has, in bytes.
///
/// Worth asking rather than assuming: the answer decides how much of a night
/// can be stacked in one go, and "24 GiB, because that is where the warning
/// starts" tells someone with 128 GB to use a sixth of their data.
fn total_memory() -> Option<u64> {
    #[cfg(windows)]
    {
        let out = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(target_os = "macos")]
    {
        let out = Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = text
            .lines()
            .find_map(|l| l.strip_prefix("MemTotal:"))?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        Some(kb * 1024)
    }
}

/// A number the pipeline wrote with `human_bytes`, back as bytes.
///
/// Read from the line the decode already logs rather than measured again here:
/// the projection that matters is the one the run itself will make.
fn parse_bytes(s: &str) -> Option<u64> {
    let (n, unit) = s.trim().split_once(' ')?;
    let n: f64 = n.parse().ok()?;
    let scale = match unit.trim() {
        "B" => 1.0,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        "TiB" => 1024.0f64.powi(4),
        _ => return None,
    };
    Some((n * scale) as u64)
}

/// What one frame of this burst costs, from the line the decode logs.
fn per_frame_bytes(log: &str) -> Option<u64> {
    log.lines().find_map(|l| {
        let (_, rest) = l.split_once(" per frame")?;
        let _ = rest;
        let before = l.split(" per frame").next()?;
        let start = before.rfind(", ")? + 2;
        parse_bytes(&before[start..])
    })
}

/// The frame a failed report names, if it names one of the set's that is not
/// already left out.
///
/// Found by looking for each frame's path in what was said rather than by
/// parsing the message, which is written for a person and free to change. The
/// longest match wins, so `f1.fit` is never blamed for what `f10.fit` did.
fn unreadable_in(text: &str, all: &[PathBuf], already: &[PathBuf]) -> Option<PathBuf> {
    all.iter()
        .filter(|p| !already.contains(p))
        .filter(|p| text.contains(p.to_string_lossy().as_ref()))
        .max_by_key(|p| p.as_os_str().len())
        .cloned()
}

/// Run `smokstak inspect` and hand the page what it said.
///
/// The report is parsed only far enough to fill in the form — the frame count,
/// the filters, a default output name — and is otherwise passed through whole,
/// because it is written for a person to read and the page has nothing to add.
fn inspect(body: &str, exe: &Path) -> Result<String> {
    let req: InspectRequest = serde_json::from_str(body)?;
    let input = resolve_input(&req)?;

    // How many frames there are, from the directory rather than from decoding
    // them. This is the number the memory projection multiplies, and asking for
    // it costs a listing.
    let total = sr_raw::collect_files(&input, None).map(|v| v.len()).unwrap_or(0);

    // What one of them costs. One frame is decoded to find out, which is what
    // the run itself does before committing to the rest.
    let probe = Command::new(exe)
        .args(["inspect"])
        .arg(&input)
        .args(["--max-frames", "1", "--no-star-metrics", "--log", "info"])
        .output()?;
    let probe_log = format!(
        "{}{}",
        String::from_utf8_lossy(&probe.stdout),
        String::from_utf8_lossy(&probe.stderr)
    );
    let per_frame = per_frame_bytes(&probe_log);
    let needed = per_frame.map(|b| b * total.max(1) as u64);
    let memory = total_memory();
    let affordable = memory.map(|m| (m as f64 * OF_MEMORY) as u64).unwrap_or(HEAVY);

    // The filters, and how much the largest of them would need. A set is
    // stacked one filter at a time whether or not it fits, so this — and not
    // the size of the whole set — is what says whether it can be done.
    let all_paths = sr_raw::collect_files(&input, None).unwrap_or_default();
    let groups = frames_by_filter(&all_paths);
    let group_json: Vec<serde_json::Value> = groups
        .iter()
        .map(|(name, files)| {
            serde_json::json!({
                "name": name,
                "frames": files.len(),
                "needs": per_frame.map(|b| b * files.len() as u64),
            })
        })
        .collect();
    let largest = groups.first().map(|(_, f)| f.len()).unwrap_or(0);
    let largest_needs = per_frame.map(|b| b * largest as u64);

    // A set that fits is read whole, as it always was. One that does not is
    // read in part, spread across the burst so the sample is of the whole night
    // and not of its first half hour.
    let sampled = match (per_frame, needed) {
        (Some(each), Some(need)) if each > 0 && need > INSPECT_BUDGET && total > 0 => {
            let fits = (INSPECT_BUDGET / each).max(1) as usize;
            fits.min(MAX_INSPECT_FRAMES).min(total)
        }
        _ => 0,
    };

    // A frame that cannot be decoded stops the report, and it used to stop the
    // page with it — at the one step before the choice of which frames to leave
    // out, which is the choice that frame needs. So it is left out of the
    // report instead, named, and left out of the stack unless put back.
    let mut unreadable: Vec<PathBuf> = Vec::new();
    let text = loop {
        let target = if unreadable.is_empty() {
            input.clone()
        } else {
            let names: Vec<String> =
                unreadable.iter().map(|p| p.to_string_lossy().into_owned()).collect();
            without(&input.to_string_lossy(), &names)?.0
        };
        let mut cmd = Command::new(exe);
        cmd.arg("inspect").arg(&target).args(["--log", "warn"]);
        if sampled > 0 {
            cmd.args(["--max-frames", &sampled.to_string(), "--select", "spread"]);
        }
        let out = cmd.output()?;
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if out.status.success() {
            break text;
        }
        match unreadable_in(&text, &all_paths, &unreadable) {
            // A handful is a few bad files. More is not a few bad files.
            Some(p) if unreadable.len() < 8 => unreadable.push(p),
            _ => return Err(anyhow!(readable_failure(&text))),
        }
    };

    let mut fields: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':')
            && !k.starts_with(' ') && !k.contains('[') {
                fields.insert(k.trim().to_string(), v.trim().to_string());
            }
    }
    let read_count = text
        .lines()
        .find_map(|l| l.strip_suffix(" files loaded"))
        .and_then(|n| n.trim().parse::<usize>().ok())
        .unwrap_or(0);
    // The set's size, not the sample's. A page that says "24 frames" about a
    // night of 634 is worse than one that says nothing.
    let frames = if total > 0 { total } else { read_count };

    // Whether the set holds more than one filter, which decides whether the
    // page offers a run per filter. The report writes one filter as
    // `H (30/30)` and several as `H (10), O (10), S (10)`, so the comma is the
    // tell — the page cannot work this out from the count, which is per filter
    // and not out of the total.
    let filter = fields.get("Filter").cloned().unwrap_or_default();
    let multi_filter = filter.contains(", ");

    // What the report already says about this set, lifted out of the report so
    // the page can put it where it will be read. A burst that will refuse to
    // stack says so on the line the report gives it, and that line was inside a
    // collapsed block: the page showed a tidy summary of a set it was about to
    // fail on.
    let mut notes: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|l| {
            let (level, text) = if let Some(r) = l.strip_prefix("fatal: ") {
                ("fatal", r)
            } else {
                ("warning", l.strip_prefix("warning: ")?)
            };
            Some(serde_json::json!({ "level": level, "text": text.trim() }))
        })
        .collect();
    for p in &unreadable {
        let name = p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        notes.push(serde_json::json!({
            "level": "warning",
            "text": format!("{name} could not be read, so it is left out of this report and of the stack"),
        }));
    }

    // Where a result is suggested, which is never among the frames.
    //
    // The frames are somebody's data. A stacker has no business writing its
    // output into an acquisition folder, and a default that does is worse than
    // an inconvenient one: it is the kind of thing noticed after it has already
    // happened. So a directory of frames gets a result beside it and not inside
    // it, and a list gets one beside the list.
    //
    // This is only a suggestion; the box says where it will go and the page
    // remembers wherever it was last pointed instead.
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("stack");
    let dir = input
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(std::env::temp_dir);
    let output = dir.join(format!("{stem}-stacked.tif"));

    Ok(serde_json::json!({
        "ok": true,
        "input": input.to_string_lossy(),
        // Freeze the actual nonrecursive stack selection for mosaic handoff.
        "mosaic_selection": if all_paths.len() <= 512 { Some(all_paths.iter().map(|p|
            (p.to_string_lossy().into_owned(),std::path::absolute(p).unwrap_or_else(|_|p.clone()).to_string_lossy().into_owned())
        ).collect::<Vec<_>>()) } else { None },
        "frames": frames,
        "report": text,
        "camera": fields.get("Camera").cloned().unwrap_or_default(),
        "dimensions": fields.get("Dimensions").cloned().unwrap_or_default(),
        "exposure": fields.get("Exposure").cloned().unwrap_or_default(),
        "filter": filter,
        "multi_filter": multi_filter,
        "notes": notes,
        "unreadable": unreadable.iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>(),
        // What the set costs, so the page can say what will fit before an hour
        // is spent finding out that nothing does.
        "read_count": read_count,
        "sampled": sampled,
        "per_frame": per_frame,
        "needed": needed,
        // What this burst may occupy here, and what the machine has, so the
        // page can say why the number it chose is the number it chose.
        "affordable": affordable,
        "memory": memory,
        "groups": group_json,
        "largest_needs": largest_needs,
        "cfa": fields.get("CFA").cloned().unwrap_or_default(),
        "output": output.to_string_lossy(),
        // So the page can put this set's name in a folder of the reader's
        // choosing rather than the one guessed at here.
        "name": format!("{stem}-stacked.tif"),
    })
    .to_string())
}

#[derive(Deserialize)]
struct StartRequest {
    input: String,
    output: String,
    #[serde(default = "one")]
    scale: f32,
    #[serde(default)]
    threads: u32,
    /// `deep` keeps the default flat-region kernel; `crisp` narrows it, which
    /// trades a little depth for grain the eye reads as detail.
    #[serde(default)]
    texture: String,
    #[serde(default)]
    flatten_background: bool,
    #[serde(default = "sky_field_default")]
    sky_field: bool,
    #[serde(default)]
    split_by_filter: bool,
    #[serde(default)]
    float_tiff: bool,
    #[serde(default)]
    fits: bool,
    #[serde(default)]
    xisf: bool,
    #[serde(default)]
    preview: bool,
    /// Stack only some of the frames. A full run is measured in hours and the
    /// settings that matter can be judged on a tenth of it, so the page offers
    /// a trial before the real thing.
    #[serde(default)]
    max_frames: u32,
    /// Which frames a trial keeps: `first`, `sharpest` or `spread`.
    #[serde(default)]
    select: String,
    /// Set once the page has asked about writing over an existing result.
    #[serde(default)]
    overwrite: bool,
    /// Blank uses private disk-backed buffers beside the output.
    #[serde(default)]
    scratch_dir: String,
    #[serde(default = "sky_field_default")]
    diagnostics: bool,
    /// Frames the reader left out, as the survey named them.
    #[serde(default)]
    exclude: Vec<String>,
}

/// Results already sitting where this run would put its own.
///
/// A stack takes hours and writes without asking, so a second run with the
/// same name quietly replaces the first — and the settings box remembers what
/// it was last given, which makes a name that has already been used the likely
/// case rather than the unlikely one. A per-filter run does not write the name
/// it was given at all; it writes that name with a filter on the end, so that
/// is what has to be looked for.
fn results_in_the_way(output: &Path, split_by_filter: bool) -> Vec<PathBuf> {
    let mut found = Vec::new();
    if !split_by_filter {
        if output.exists() {
            found.push(output.to_path_buf());
        }
        return found;
    }
    let (Some(dir), Some(stem), Some(ext)) = (
        output.parent(),
        output.file_stem().and_then(|s| s.to_str()),
        output.extension().and_then(|s| s.to_str()),
    ) else {
        return found;
    };
    let Ok(entries) = std::fs::read_dir(if dir.as_os_str().is_empty() { Path::new(".") } else { dir })
    else {
        return found;
    };
    for e in entries.flatten() {
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        // `m45.tif` becomes `m45_H.tif`; the linear copy and the preview are
        // written from the same stem and would be overwritten too, but naming
        // the masters is enough to say what is about to happen. What is left
        // between the stem and the extension has to be a filter and nothing
        // else — `m45_H.linear.tif` leaves `H.linear`, which is the linear copy
        // of one, not a second master.
        if let Some(rest) = name.strip_prefix(&format!("{stem}_"))
            && rest
                .strip_suffix(&format!(".{ext}"))
                .is_some_and(|f| !f.is_empty() && !f.contains('.'))
            {
                found.push(e.path());
            }
    }
    found.sort();
    found
}

fn results_with_exports(output: &Path, split: bool, fits: bool, xisf: bool) -> Vec<PathBuf> {
    let mut found = results_in_the_way(output, split);
    for (enabled, extension) in [(fits, "fits"), (xisf, "xisf")] {
        if enabled { found.extend(results_in_the_way(&output.with_extension(extension), split)); }
    }
    found.sort();
    found.dedup();
    found
}

fn one() -> f32 {
    1.0
}

fn sky_field_default() -> bool {
    sr_core::config::ReconstructionConfig::default().sky_field
}

/// Use the ordinary stacker with the complete selected population and a
/// shared grid. Disk-backed buffers preserve global photometry/rejection
/// decisions; independently processed batch accumulators do not.
fn plan_runs(req: &StartRequest, exe: &Path, how: &[String]) -> Result<Vec<Planned>> {
    let output = PathBuf::from(req.output.trim());
    let last = STAGES.len() - 1;
    let one = |args: Vec<String>, step: String| Planned {
        step,
        program: exe.to_path_buf(),
        args,
        last_stage: last,
        kind: "stack",
    };

    // What the reader asked for, unchanged, whenever it can be done in one go.
    let mut simple = vec!["stack".into(), req.input.trim().to_string()];
    simple.push("--output".into());
    simple.push(req.output.trim().to_string());
    simple.extend(how.iter().cloned());
    if req.max_frames > 0 {
        simple.push("--max-frames".into());
        simple.push(req.max_frames.to_string());
        // Which frames a trial takes changes what it tells you. `sharpest`
        // flatters the settings, `first` is whatever the night began with;
        // `spread` keeps the dither, which is what makes the extra resolution
        // possible and so is the only honest answer at 2x.
        simple.push("--select".into());
        simple.push(match req.select.as_str() {
            s @ ("sharpest" | "first" | "spread") => s.to_string(),
            _ => "spread".into(),
        });
    }
    if req.split_by_filter {
        simple.push("--split-by-filter".into());
    }

    // All selected frames share production decisions. Private disk-backed
    // buffers replace the former independently rejected batch accumulators.
    let scratch = if req.scratch_dir.trim().is_empty() {
        output.parent().unwrap_or(Path::new(".")).join(".smokstak-scratch")
    } else { PathBuf::from(req.scratch_dir.trim()) };
    simple.extend(["--scratch-dir".into(), scratch.to_string_lossy().into_owned()]);
    if req.diagnostics {
        let stem = output.file_stem().unwrap_or_default().to_string_lossy();
        let dir = workflows::unique_child(output.parent().unwrap_or(Path::new(".")), &format!("{stem}-review"));
        simple.extend(["--diagnostics".into(), dir.to_string_lossy().into_owned()]);
    }
    Ok(vec![one(simple, "All selected frames; disk-backed storage".into())])
}

fn start(body: &str, exe: &Path, job: &Arc<Mutex<Job>>) -> Result<String> {
    let mut req: StartRequest = serde_json::from_str(body)?;
    {
        let j = job.lock().unwrap();
        if j.progress.running {
            return Err(anyhow!("a run is already going"));
        }
        // Both decode every frame on every core, and the stack would be
        // slowed to a crawl by a measurement nobody is waiting for any more.
        if j.survey.running {
            return Err(anyhow!(
                "the frames are still being measured; wait for that, or stop it, before stacking"
            ));
        }
    }
    if req.input.trim().is_empty() || req.output.trim().is_empty() {
        return Err(anyhow!("both an input and an output are needed"));
    }
    if !req.overwrite {
        let in_the_way = results_with_exports(Path::new(req.output.trim()), req.split_by_filter, req.fits, req.xisf);
        if !in_the_way.is_empty() {
            // Answered by the page rather than refused outright: writing over
            // last night's stack is a perfectly ordinary thing to want, and
            // doing it without being asked is the part that is not.
            return Ok(serde_json::json!({
                "ok": false,
                "overwrite": in_the_way.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                "error": format!(
                    "{} already there",
                    if in_the_way.len() == 1 { "a result is".into() }
                    else { format!("{} results are", in_the_way.len()) }
                ),
            })
            .to_string());
        }
    }

    // Everything that describes the reconstruction rather than which frames go
    // into it, so that every run of a plan is the same reconstruction.
    let mut how: Vec<String> = vec![
        "--scale".into(),
        format!("{}", req.scale),
        "--log".into(),
        "info".into(),
    ];
    if req.threads > 0 {
        how.push("--threads".into());
        how.push(req.threads.to_string());
    }
    if req.texture == "crisp" {
        how.push("--k-denoise".into());
        how.push("1.0".into());
    }
    if req.flatten_background {
        how.push("--flatten-background".into());
    }
    how.push(if req.sky_field { "--sky-field".into() } else { "--no-sky-field".into() });
    if req.fits { how.push("--fits".into()); }
    if req.xisf { how.push("--xisf".into()); }
    if req.float_tiff {
        how.push("--float-tiff".into());
    }
    if req.preview {
        how.push("--preview".into());
    }

    // What tells this run from the last one, for the list of past runs. The
    // input's name, then only what was not left at its default: a list of runs
    // that all say "1x, deep" distinguishes nothing.
    let mut what = vec![Path::new(req.input.trim())
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| req.input.trim().to_string())];

    // Frames left out become a list of the rest, which every plan already
    // knows how to read — including one stacked a filter or a batch at a time.
    if !req.exclude.is_empty() {
        let (list, dropped) = without(&req.input, &req.exclude)?;
        req.input = list.to_string_lossy().into_owned();
        what.push(format!("{dropped} left out"));
    }

    let plan = plan_runs(&req, exe, &how)?;

    if req.max_frames > 0 {
        what.push(format!("{} frames", req.max_frames));
    }
    if (req.scale - 1.0).abs() > 1e-6 {
        what.push(format!("{}x", req.scale));
    }
    if req.texture == "crisp" {
        what.push("crisp".into());
    }
    if req.split_by_filter {
        what.push("per filter".into());
    }
    if req.flatten_background {
        what.push("flattened".into());
    }
    if !req.sky_field {
        what.push("no sky matching".into());
    }
    if plan.len() > 1 {
        what.push(format!("{} runs", plan.len()));
    }
    let title = what.join(" · ");

    start_plan(plan, job, title, None)
}

/// Begin a piece of work, which may be several runs long.
///
/// `of` names the finished stack that palettes or a colour image are being made
/// from; see [`begin`].
fn start_plan(
    plan: Vec<Planned>,
    job: &Arc<Mutex<Job>>,
    title: String,
    of: Option<u64>,
) -> Result<String> {
    let mut queue: std::collections::VecDeque<Planned> = plan.into();
    let first = queue.pop_front().ok_or_else(|| anyhow!("nothing to run"))?;
    {
        let mut j = job.lock().unwrap();
        anyhow::ensure!(!j.progress.running && !j.survey.running, "another run or survey is still going");
        j.progress = begin(&mut j, &first, queue.len() + 1, title, of)?;
        j.queue = queue;
        j.began = Some(Instant::now());
        j.stop = false;
    }
    spawn_next(first, job)
}

/// Whether a run is made from a finished stack's masters rather than being one.
fn extends_a_stack(kind: &str) -> bool {
    matches!(kind, "palette" | "composite")
}

/// What the page is told as a piece of work begins.
///
/// A stack is new work. Trying palettes on, or combining the masters into a
/// colour image, is part of looking at the stack that made them — whichever one
/// the page is showing, which is not always the one that ran last. Its masters,
/// its findings and the colour images already made from it all stay, and what
/// the new run makes is added to them.
fn begin(
    j: &mut Job,
    first: &Planned,
    steps: usize,
    title: String,
    of: Option<u64>,
) -> Result<Progress> {
    let fresh = Progress {
        running: true,
        message: "Starting".into(),
        kind: first.kind.to_string(),
        steps,
        step: 1,
        step_name: first.step.clone(),
        ..Default::default()
    };
    let Some(id) = of.filter(|_| extends_a_stack(first.kind)) else {
        j.palette_dir = None;
        j.next_id += 1;
        return Ok(Progress { id: j.next_id, title, ..fresh });
    };
    // The kept copy first: the progress in front of us may be a combine of the
    // same stack that failed part way. An id of 0 is a page that did not say.
    let base = j
        .past
        .iter()
        .rev()
        .find(|p| id != 0 && p.id == id)
        .cloned()
        .or_else(|| {
            (j.progress.finished && (id == 0 || j.progress.id == id)).then(|| j.progress.clone())
        })
        .ok_or_else(|| anyhow!("that run is no longer held; stack it again to combine it"))?;
    let mut colour = base.colour;
    let mut palettes = base.palettes;
    if first.kind == "composite" {
        colour.push(ColourImage { title, files: Vec::new(), mapping: String::new() });
    } else {
        palettes.clear();
    }
    Ok(Progress {
        extends: true,
        id: base.id,
        title: base.title,
        outputs: base.outputs,
        findings: base.findings,
        took: base.took,
        finished_at: base.finished_at,
        colour,
        palettes,
        ..fresh
    })
}

/// The end of a piece of work: what the page is told, and what is kept of it.
fn settle(j: &mut Job, ok: bool, stopped: bool, elapsed: f64, last_stage: usize) {
    j.queue.clear();
    j.progress.running = false;
    j.progress.finished = true;
    j.progress.elapsed = elapsed;
    if ok {
        j.progress.stage = last_stage;
        j.progress.message = "Done".into();
    } else if stopped {
        j.progress.failed = true;
        j.progress.message = "Stopped.".into();
    } else {
        j.progress.failed = true;
        j.progress.message =
            "The run did not finish. The log below says what it was doing.".into();
    }

    if j.progress.extends {
        // A colour image that was never written is not one to show.
        if !ok && j.progress.colour.last().is_some_and(|c| c.files.is_empty()) {
            j.progress.colour.pop();
        }
        // A palette or a colour image that failed failed on its own; the stack
        // it was made from is kept as it was.
        if !ok {
            return;
        }
        let mut done = j.progress.clone();
        done.log.clear();
        // In place of the stack it was made from, rather than beside it: one
        // stack combined three ways is one piece of work, not four.
        match j.past.iter_mut().find(|p| p.id == done.id) {
            Some(kept) => *kept = done,
            None => j.past.push(done),
        }
        save_history(j);
        return;
    }

    j.progress.took = elapsed;
    j.progress.finished_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    // Kept so that trying something, looking, and trying again does not
    // destroy what it is being compared against. A run that wrote nothing is
    // not worth going back to.
    if !j.progress.outputs.is_empty() {
        let mut done = j.progress.clone();
        done.log.clear(); // the tail is for the run in front of you
        j.past.push(done);
        let n = j.past.len();
        if n > 20 {
            j.past.drain(..n - 20);
        }
        save_history(j);
    }
}

/// Where the list of earlier runs is kept between sessions.
///
/// The per-user data folder the platform names, because the list is a record
/// of one person's work and not of any one project. `SMOKSTAK_HISTORY` names a
/// different file, and set to nothing keeps no list at all.
fn history_file() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SMOKSTAK_HISTORY") {
        return (!p.is_empty()).then(|| PathBuf::from(p));
    }
    data_dir().map(|d| d.join("history.json"))
}

/// The per-user folder the platform names for a program's own data, where
/// what the page keeps between sessions lives.
pub(crate) fn data_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"));
    #[cfg(all(unix, not(target_os = "macos")))]
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
    base.map(|b| b.join("smokstak"))
}

/// The runs an earlier session kept.
///
/// A list that cannot be read is started again rather than refused: it is a
/// convenience, and a page that will not open because of one is not.
fn load_history(path: &Path) -> Vec<Progress> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    serde_json::from_str(&text).unwrap_or_else(|e| {
        log::warn!("{} could not be read ({e}); starting a new list", path.display());
        Vec::new()
    })
}

/// Keep the list of earlier runs, if this session keeps one.
///
/// Written beside itself and renamed into place, so that a server closed part
/// way through writing leaves the last list and not half of this one.
fn save_history(j: &Job) {
    let Some(path) = &j.history else { return };
    let write = || -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string(&j.past)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    };
    if let Err(e) = write() {
        log::warn!("the list of earlier runs could not be kept in {}: {e}", path.display());
    }
}

/// Put back what an earlier session kept.
///
/// The page may fetch only what a run of this program wrote. These runs wrote
/// these files, so they are served again — only images and named Smokstak
/// reports, and only those still there: the list is a file anybody with the user's account can edit,
/// and an entry in it naming some other file does not make that file the
/// page's business.
fn restore(j: &mut Job, past: Vec<Progress>) {
    let is_image = |p: &Path| {
        p.is_file()
            && matches!(
                p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
                Some("png" | "tif" | "tiff" | "fits" | "xisf")
            )
    };
    j.next_id = j.next_id.max(past.iter().map(|p| p.id).max().unwrap_or(0));
    for mut p in past {
        p.running = false;
        p.log.clear();
        if p.id == 0 {
            j.next_id += 1;
            p.id = j.next_id;
        }
        // Palette thumbnails live in a temporary folder that may be gone.
        p.palettes.retain(|s| is_image(Path::new(&s.preview)));
        let files = p
            .outputs
            .iter()
            .chain(p.colour.iter().flat_map(|c| c.files.iter()))
            .chain(p.palettes.iter().map(|s| &s.preview));
        for f in files {
            let path = PathBuf::from(f);
            if p.kind == "mosaic" {
                j.mosaic_outputs.insert(path.clone());
                j.mosaic_outputs.insert(std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone()));
            }
            if is_image(&path) || workflows::is_report(&path)
                || (p.kind == "mosaic" && path.is_file()
                    && matches!(path.file_name().and_then(|n| n.to_str()), Some("plan.json" | "result.json"))) {
                j.served.insert(std::fs::canonicalize(&path).unwrap_or(path));
            }
        }
        j.past.push(p);
    }
    let n = j.past.len();
    if n > 20 {
        j.past.drain(..n - 20);
    }
}

/// The frames of a set less the ones the reader left out, written as a list
/// the stacker can read, and how many were left out.
fn without(input: &str, exclude: &[String]) -> Result<(PathBuf, usize)> {
    let all = sr_raw::collect_files(Path::new(input.trim()), None)?;
    let out: std::collections::HashSet<&str> = exclude.iter().map(|s| s.as_str()).collect();
    let kept: Vec<&PathBuf> =
        all.iter().filter(|p| !out.contains(p.to_string_lossy().as_ref())).collect();
    anyhow::ensure!(!kept.is_empty(), "every frame was left out");
    // The list lives in a temporary folder, and a relative path in a list is
    // read against the list's own folder.
    let here = std::env::current_dir()?;
    let text: String = kept
        .iter()
        .map(|p| format!("{}\n", if p.is_absolute() { p.to_path_buf() } else { here.join(p) }.display()))
        .collect();
    let list = list_from_text(&text, "kept.txt")?;
    Ok((list, all.len() - kept.len()))
}

#[derive(Deserialize)]
struct SurveyRequest {
    input: String,
}

/// Measure every frame of a set, beside the page rather than in front of it.
///
/// Minutes for a big night: every frame is decoded once, a core's worth at a
/// time, none of them held. The page asks how far it has got, and can go on
/// choosing settings meanwhile.
fn survey(body: &str, exe: &Path, job: &Arc<Mutex<Job>>) -> Result<String> {
    let req: SurveyRequest = serde_json::from_str(body)?;
    let input = req.input.trim().to_string();
    anyhow::ensure!(!input.is_empty(), "nothing to measure");
    let generation = {
        let mut j = job.lock().unwrap();
        if j.progress.running {
            return Err(anyhow!("a run is going; measure the frames once it has finished"));
        }
        if j.survey.running && j.survey.input == input {
            return Ok(serde_json::json!({ "ok": true }).to_string());
        }
        // Another set's measurement, which the reader has moved on from.
        if let Some(mut c) = j.survey_child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        j.survey_gen += 1;
        j.survey = SurveyState { running: true, input: input.clone(), ..Default::default() };
        j.survey_gen
    };
    let out = std::env::temp_dir()
        .join(format!("smokstak-survey-{}-{generation}.json", std::process::id()));
    let spawned = Command::new(exe)
        .arg("survey")
        .arg(&input)
        .arg("--json")
        .arg(&out)
        .args(["--log", "warn"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => {
            let mut j = job.lock().unwrap();
            j.survey.running = false;
            j.survey.finished = true;
            j.survey.error = format!("could not start measuring: {e}");
            return Err(anyhow!("could not start measuring: {e}"));
        }
    };
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;
    job.lock().unwrap().survey_child = Some(child);

    let counter = {
        let job = Arc::clone(job);
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let mut j = job.lock().unwrap();
                if j.survey_gen != generation {
                    return;
                }
                if let Some(n) = line.strip_prefix("surveying ").and_then(|r| r.strip_suffix(" frames")) {
                    j.survey.total = n.trim().parse().unwrap_or(0);
                }
                // Written from every worker at once, so not always in order.
                if let Some((n, m)) = line.strip_prefix("surveyed ").and_then(|r| r.split_once(" of ")) {
                    j.survey.done = j.survey.done.max(n.trim().parse().unwrap_or(0));
                    j.survey.total = m.trim().parse().unwrap_or(j.survey.total);
                }
            }
        })
    };
    let complaints = std::thread::spawn(move || {
        BufReader::new(stderr).lines().map_while(Result::ok).collect::<Vec<_>>()
    });

    let job = Arc::clone(job);
    std::thread::spawn(move || {
        // Watched rather than waited on, so that Stop can still reach it.
        let status = loop {
            let done = {
                let mut j = job.lock().unwrap();
                if j.survey_gen != generation {
                    return;
                }
                match j.survey_child.as_mut() {
                    Some(c) => c.try_wait(),
                    None => break None,
                }
            };
            match done {
                Ok(Some(s)) => break Some(s),
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(250)),
                Err(_) => break None,
            }
        };
        let _ = counter.join();
        let said = complaints.join().unwrap_or_default();
        let mut j = job.lock().unwrap();
        if j.survey_gen != generation {
            return;
        }
        j.survey_child = None;
        j.survey.running = false;
        j.survey.finished = true;
        let read_back = || -> Result<serde_json::Value> {
            let mut doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&out)?)?;
            Ok(doc["frames"].take())
        };
        match status {
            Some(s) if s.success() => match read_back() {
                Ok(frames) => {
                    j.survey.done = j.survey.total;
                    j.survey.frames = frames;
                }
                Err(e) => j.survey.error = format!("the measurements could not be read back: {e}"),
            },
            _ if j.survey.stopped => j.survey.error = "Stopped.".into(),
            _ => {
                let tail = said[said.len().saturating_sub(12)..].join("\n");
                j.survey.error = if tail.trim().is_empty() {
                    "measuring the frames did not finish".into()
                } else {
                    readable_failure(&tail)
                };
            }
        }
        let _ = std::fs::remove_file(&out);
    });
    Ok(serde_json::json!({ "ok": true }).to_string())
}

/// A way of turning the filters on hand into one colour picture.
struct Candidate {
    id: &'static str,
    label: &'static str,
    palette: &'static str,
    /// Which channel carries the detail, for an LRGB set.
    luminance: &'static str,
    needs: &'static [&'static str],
}

/// Every palette the channels on hand can actually make.
///
/// None of these is a colour anything really is — sulphur and hydrogen emit
/// 16 nm apart in the deep red, so a faithful rendering of a narrowband set
/// would be two reds and a teal and would show almost nothing. They are
/// conventions for making structure visible, which is exactly why the choice
/// wants to be made by looking rather than by reading a list of names.
const CANDIDATES: [Candidate; 6] = [
    Candidate { id: "sho", label: "SHO — the Hubble palette", palette: "sho", luminance: "", needs: &["S", "H", "O"] },
    Candidate { id: "hso", label: "HSO — hydrogen red, sulphur green", palette: "hso", luminance: "", needs: &["S", "H", "O"] },
    Candidate { id: "hoo", label: "HOO — hydrogen red, oxygen green and blue", palette: "hoo", luminance: "", needs: &["H", "O"] },
    Candidate { id: "ohh", label: "OHH — oxygen red, hydrogen green and blue", palette: "ohh", luminance: "", needs: &["H", "O"] },
    Candidate { id: "rgb", label: "RGB — the colours as measured", palette: "rgb", luminance: "", needs: &["R", "G", "B"] },
    Candidate { id: "lrgb", label: "LRGB — colour from RGB, detail from L", palette: "rgb", luminance: "L", needs: &["R", "G", "B", "L"] },
];

fn candidates_for(channels: &[String]) -> Vec<&'static Candidate> {
    let have: Vec<String> = channels.iter().map(|c| c.to_ascii_uppercase()).collect();
    CANDIDATES
        .iter()
        .filter(|c| c.needs.iter().all(|n| have.iter().any(|h| h == n)))
        .collect()
}

/// What the page sends to ask for a look at every palette its channels allow.
#[derive(Deserialize)]
struct PalettesRequest {
    /// `NAME=path` for each channel, as the composite takes them.
    channels: Vec<String>,
    #[serde(default)]
    stretch_channels: bool,
    /// The id of the stack whose masters these are.
    #[serde(default)]
    of: u64,
}

/// Make a small picture of each palette the channels allow, to choose by.
///
/// Choosing a palette from a list of names means running one, looking, running
/// another, and holding the first in your head while you do. They are quick
/// next to the stack that produced the masters — seconds against hours — so
/// the page makes all of them and shows them side by side.
fn palettes(body: &str, exe: &Path, job: &Arc<Mutex<Job>>) -> Result<String> {
    let req: PalettesRequest = serde_json::from_str(body)?;
    {
        let j = job.lock().unwrap();
        if j.progress.running {
            return Err(anyhow!("a run is already going"));
        }
    }
    // The same rule as everywhere else: only files this session stacked.
    {
        let j = job.lock().unwrap();
        for c in &req.channels {
            let path = c.split_once('=').map(|(_, p)| p).unwrap_or(c);
            let p = PathBuf::from(path);
            let canonical = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
            if !j.served.contains(&canonical) && !j.served.contains(&p) {
                return Err(anyhow!("{path} is not a file this session stacked"));
            }
        }
    }
    let names: Vec<String> = req
        .channels
        .iter()
        .filter_map(|c| c.split('=').next().map(|s| s.to_string()))
        .collect();
    let wanted = candidates_for(&names);
    anyhow::ensure!(!wanted.is_empty(), "these channels do not make a palette");

    // One folder per stack, so that trying palettes on one does not delete the
    // pictures another is still showing in the list of earlier runs.
    let id = if req.of == 0 { job.lock().unwrap().progress.id } else { req.of };
    let dir = std::env::temp_dir()
        .join(format!("smokstak-palettes-{}-{id}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let plan: Vec<Planned> = wanted
        .iter()
        .map(|c| {
            let mut args = vec!["composite".to_string()];
            args.extend(req.channels.iter().cloned());
            args.push("--output".into());
            args.push(dir.join(format!("{}.tif", c.id)).to_string_lossy().into_owned());
            args.push("--palette".into());
            args.push(c.palette.to_string());
            args.push("--preview".into());
            // Small: this is a thumbnail to choose by, not the picture.
            args.push("--preview-size".into());
            args.push("1000".into());
            args.push("--no-align".into());
            args.push("--log".into());
            args.push("warn".into());
            if req.stretch_channels {
                args.push("--stretch-channels".into());
            }
            if !c.luminance.is_empty() {
                args.push("--luminance".into());
                args.push(c.luminance.to_string());
            }
            Planned {
                step: c.label.to_string(),
                program: exe.to_path_buf(),
                args,
                last_stage: 0,
                kind: "palette",
            }
        })
        .collect();

    {
        let mut j = job.lock().unwrap();
        j.palette_dir = Some(dir);
    }
    start_plan(plan, job, format!("{} palettes", wanted.len()), Some(req.of))
}

/// What the page sends to combine separately stacked filters into one image.
#[derive(Deserialize)]
struct CompositeRequest {
    /// `NAME=path` for each channel, the names being the filters.
    channels: Vec<String>,
    output: String,
    #[serde(default)]
    fits: bool,
    #[serde(default)]
    xisf: bool,
    #[serde(default)]
    palette: String,
    /// Stretch each channel on its own histogram before combining. On for a
    /// narrowband palette, which is otherwise green: see the flag's own
    /// documentation.
    #[serde(default)]
    stretch_channels: bool,
    /// Channels stacked in one per-filter run already share a grid, so the
    /// alignment step is skipped by default and offered for the case where the
    /// masters came from separate runs.
    #[serde(default)]
    align: bool,
    #[serde(default)]
    luminance: String,
    #[serde(default)]
    overwrite: bool,
    /// The id of the stack whose masters these are.
    #[serde(default)]
    of: u64,
}

/// Combine the masters a per-filter run produced into one colour image.
///
/// Without this the page's answer to a narrowband set is three grey files and
/// an instruction to go and use the command line, which is the one thing the
/// page exists not to require.
fn composite(body: &str, exe: &Path, job: &Arc<Mutex<Job>>) -> Result<String> {
    let req: CompositeRequest = serde_json::from_str(body)?;
    {
        let j = job.lock().unwrap();
        if j.progress.running {
            return Err(anyhow!("a run is already going"));
        }
    }
    if req.channels.len() < 2 {
        return Err(anyhow!("a composite needs at least two channels"));
    }
    if req.output.trim().is_empty() {
        return Err(anyhow!("the combined image needs a name"));
    }
    if !req.overwrite {
        let in_the_way = results_with_exports(Path::new(req.output.trim()), false, req.fits, req.xisf);
        if !in_the_way.is_empty() {
            return Ok(serde_json::json!({
                "ok": false,
                "overwrite": in_the_way.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                "error": "a result is already there",
            })
            .to_string());
        }
    }
    // Only the files this session's runs wrote, for the same reason `/file`
    // serves only those: the page names them, and the page is a browser.
    {
        let j = job.lock().unwrap();
        for c in &req.channels {
            let path = c.split_once('=').map(|(_, p)| p).unwrap_or(c);
            let p = PathBuf::from(path);
            let canonical = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
            if !j.served.contains(&canonical) && !j.served.contains(&p) {
                return Err(anyhow!("{path} is not a file this session stacked"));
            }
        }
    }

    let mut cmd = Command::new(exe);
    cmd.arg("composite");
    for c in &req.channels {
        cmd.arg(c);
    }
    // The page sends the name of a choice, which may not be the name of a
    // palette: LRGB is the RGB palette with the detail taken from L. The
    // mapping lives with the choices rather than in the page.
    let chosen = CANDIDATES.iter().find(|c| c.id == req.palette);
    let palette = chosen.map(|c| c.palette).unwrap_or(if req.palette.is_empty() {
        "sho"
    } else {
        &req.palette
    });
    let luminance = match chosen {
        Some(c) if !c.luminance.is_empty() => c.luminance,
        _ => req.luminance.trim(),
    };
    cmd.arg("--output")
        .arg(&req.output)
        .arg("--palette")
        .arg(palette)
        .arg("--preview")
        .arg("--log")
        .arg("info");
    if req.fits { cmd.arg("--fits"); }
    if req.xisf { cmd.arg("--xisf"); }
    if req.stretch_channels {
        cmd.arg("--stretch-channels");
    }
    if !req.align {
        cmd.arg("--no-align");
    }
    if !luminance.is_empty() {
        cmd.arg("--luminance").arg(luminance);
    }

    let channels: Vec<&str> =
        req.channels.iter().filter_map(|c| c.split('=').next()).collect();
    let title = format!(
        "{} · {}{}",
        channels.join(""),
        chosen.map(|c| c.id).unwrap_or(palette),
        if req.stretch_channels { " · stretched" } else { "" }
    );
    let args: Vec<String> = cmd
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    start_plan(
        vec![Planned {
            step: String::new(),
            program: exe.to_path_buf(),
            args,
            last_stage: 0,
            kind: "composite",
        }],
        job,
        title,
        Some(req.of),
    )
}

/// Run one planned child, and follow it: its output becomes the page's log, the
/// stage it has reached, and the list of files it wrote. When it ends well and
/// more are queued behind it, the next one starts.
fn spawn_next(planned: Planned, job: &Arc<Mutex<Job>>) -> Result<String> {
    // Stop may have been pressed while this was being worked out.
    if job.lock().unwrap().stop {
        let mut j = job.lock().unwrap();
        j.progress.running = false;
        j.progress.finished = true;
        j.progress.failed = true;
        j.progress.message = "Stopped.".into();
        return Ok(serde_json::json!({ "ok": true }).to_string());
    }
    let last_stage = planned.last_stage;
    let mut child = planned
        .command()
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("could not start the stacker: {e}"))?;
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;

    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut j = job.lock().unwrap();
        // The stage list belongs to the run in front of you, and each run in a
        // plan walks it again; everything else — the log, the files, the
        // findings — belongs to the work as a whole and carries over.
        j.progress.stage = 0;
        j.progress.kind = planned.kind.to_string();
        j.progress.step_name = planned.step.clone();
        j.progress.running = true;
        j.progress.finished = false;
        j.child = Some(child);
        j.cancel = Some(Arc::clone(&cancel));
        if j.began.is_none() {
            j.began = Some(Instant::now());
        }
    }

    // The clock belongs to the work, not to this run of it.
    let began = job.lock().unwrap().began.unwrap_or_else(Instant::now);
    let kind = planned.kind;
    let completed_args = planned.args.clone();
    let mut readers = Vec::new();
    for source in [Box::new(stdout) as Box<dyn Read + Send>, Box::new(stderr)] {
        let job = Arc::clone(job);
        readers.push(std::thread::spawn(move || {
            for line in BufReader::new(source).lines().map_while(Result::ok) {
                let mut j = job.lock().unwrap();
                if kind == "mosaic" {
                    mosaic::progress(&mut j.progress, &line);
                } else if kind == "mosaic_prepare" {
                    mosaic::preparation_progress(&mut j.progress, &line);
                } else if let Some(s) = stage_of(&line) {
                    j.progress.stage = j.progress.stage.max(s);
                }
                // The run names each file it wrote as it writes it. The
                // preview announces itself differently from the rest, which is
                // how it came to be missing from the finished list.
                if kind != "mosaic" && kind != "mosaic_prepare" { workflows::record_outputs(&mut j, &line, kind); }
                // A per-filter run says which filter it has reached, and
                // everything after that line belongs to that filter.
                if let Some(rest) = line.strip_prefix("=== filter ") {
                    j.progress.group = rest.trim_end_matches(" ===").trim().to_string();
                }
                // The findings are what the stack said about itself. A combine
                // made from it afterwards has its own "Wrote" and would replace
                // them.
                if j.progress.extends {
                    if let Some(rest) = line.strip_prefix("Palette: ")
                        && let Some(c) = j.progress.colour.last_mut() {
                            c.mapping = rest.trim().to_string();
                        }
                } else if let Some((label, text)) = finding_of(&line) {
                    let group = j.progress.group.clone();
                    j.progress.findings.push(Finding { group, label: label.into(), text });
                }
                j.progress.elapsed = began.elapsed().as_secs_f64();
                j.progress.log.push(line);
                // The page shows a tail; keeping the whole log of a long run in
                // memory to throw most of it away on every poll is waste.
                let n = j.progress.log.len();
                if n > 400 {
                    j.progress.log.drain(..n - 400);
                }
            }
        }));
    }

    // A waiter, so that the page learns the run ended without polling the OS.
    {
        let job = Arc::clone(job);
        std::thread::spawn(move || {
            // Watched rather than waited on, so that the child stays in the job
            // where Stop can reach it. Waiting took it out of there first, which
            // left Stop with nothing to kill: it set the message and the flag,
            // said "Stopped.", and the run carried on to the end.
            let status = loop {
                let done = {
                    let mut j = job.lock().unwrap();
                    match j.child.as_mut() {
                        Some(c) => c.try_wait(),
                        None => break None,
                    }
                };
                match done {
                    Ok(Some(s)) => break Some(Ok(s)),
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(150)),
                    Err(e) => break Some(Err(e)),
                }
            };
            for reader in readers { let _ = reader.join(); }
            job.lock().unwrap().child = None;
            let mut j = job.lock().unwrap();
            let mut ok = matches!(&status, Some(Ok(s)) if s.success());
            let stopped = cancel.load(Ordering::SeqCst) || j.stop;
            if ok && !stopped {
                if kind == "mosaic" {
                    if let Err(e) = mosaic::completed(&mut j, &completed_args) {
                        j.progress.log.push(format!("Mosaic publication failed: {e:#}"));
                        ok = false;
                    }
                }
                else if kind == "mosaic_prepare" {
                    if let Err(e) = mosaic::preparation_completed(&mut j, &completed_args) {
                        j.progress.log.push(format!("Mosaic preparation publication failed: {e:#}"));
                        ok = false;
                    }
                }
                else { workflows::completed(&mut j, kind, &completed_args); }
            }

            // More to do, and reason to do it: the next run of the plan.
            if ok && !stopped && !j.queue.is_empty() {
                let next = j.queue.pop_front().expect("just checked");
                j.progress.step += 1;
                j.progress.elapsed = began.elapsed().as_secs_f64();
                drop(j);
                if let Err(e) = spawn_next(next, &job) {
                    let mut j = job.lock().unwrap();
                    j.progress.running = false;
                    j.progress.finished = true;
                    j.progress.failed = true;
                    j.progress.message = format!("The next run could not be started: {e}");
                }
                return;
            }

            settle(&mut j, ok, stopped, began.elapsed().as_secs_f64(), last_stage);
        });
    }

    Ok(serde_json::json!({ "ok": true }).to_string())
}

#[derive(Deserialize)]
struct RevealRequest {
    path: String,
}

/// Show a finished file in the platform's file manager.
fn reveal(body: &str) -> Result<String> {
    let req: RevealRequest = serde_json::from_str(body)?;
    let p = PathBuf::from(&req.path);
    if !p.exists() {
        return Err(anyhow!("{} is not there any more", p.display()));
    }
    #[cfg(windows)]
    Command::new("explorer").arg("/select,").arg(&p).spawn()?;
    #[cfg(target_os = "macos")]
    Command::new("open").arg("-R").arg(&p).spawn()?;
    #[cfg(all(unix, not(target_os = "macos")))]
    Command::new("xdg-open")
        .arg(p.parent().unwrap_or(Path::new(".")))
        .spawn()?;
    Ok(serde_json::json!({ "ok": true }).to_string())
}

/// Whether a folder dialog is already open, so a second click cannot put a
/// second one behind the first.
static PICKING: AtomicBool = AtomicBool::new(false);

/// Ask the desktop for a folder, and hand back the path it gives.
///
/// The one thing the page genuinely cannot do. A browser's file chooser exists
/// to give a page a file's *contents* without telling it where the file lives,
/// which is the right decision and leaves no way to say "that folder there" —
/// so a folder of frames had to be typed out by hand, in full, with the right
/// separators. The server is on the same machine as the person using the page,
/// so it asks the desktop instead.
fn pick_folder() -> Result<String> {
    if PICKING.swap(true, Ordering::SeqCst) {
        return Err(anyhow!("a folder chooser is already open"));
    }
    let out = folder_chooser().and_then(|mut c| Ok(c.output()?));
    PICKING.store(false, Ordering::SeqCst);
    let path = folder_from(&out?.stdout);
    Ok(serde_json::json!({ "ok": true, "path": path }).to_string())
}

/// What a chooser wrote on being closed.
///
/// Every one of them says nothing at all when the dialog is cancelled, and the
/// page reads an empty path as "nothing was chosen" rather than as an error —
/// cancelling is not a failure. A path is taken as it is apart from the
/// newline: folder names have spaces at the ends about as often as they have
/// spaces in the middle, which is to say it happens.
fn folder_from(stdout: &[u8]) -> String {
    String::from_utf8_lossy(stdout).trim_matches(['\r', '\n']).to_string()
}

/// The prompt every chooser puts above its tree.
const CHOOSE: &str = "Choose the folder holding the frames";

/// The platform's folder chooser, ready to run.
///
/// Built rather than run so that the choice of command and the reading of its
/// answer can be tested apart from the dialog itself, which by its nature waits
/// for a person.
fn folder_chooser() -> Result<Command> {
    #[cfg(windows)]
    {
        let mut c = Command::new("powershell");
        c.args([
            "-NoProfile",
            "-STA",
            "-Command",
            &format!(
                "Add-Type -AssemblyName System.Windows.Forms; \
                 $d = New-Object System.Windows.Forms.FolderBrowserDialog; \
                 $d.Description = '{CHOOSE}'; \
                 if ($d.ShowDialog() -eq 'OK') {{ [Console]::Out.Write($d.SelectedPath) }}"
            ),
        ]);
        Ok(c)
    }
    #[cfg(target_os = "macos")]
    {
        let mut c = Command::new("osascript");
        // `choose folder` raises on cancel rather than returning nothing, and
        // an unhandled raise is a non-zero exit and a message on stderr. The
        // `try` makes a cancel say nothing, like the others.
        c.args([
            "-e",
            &format!("try\nPOSIX path of (choose folder with prompt \"{CHOOSE}\")\nend try"),
        ]);
        Ok(c)
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // Whichever of the two desktop helpers is installed. `which` decides,
        // rather than running one to see whether it exists: running zenity to
        // find out puts a dialog on the screen before we know we want it.
        for (exe, args) in [
            ("zenity", vec!["--file-selection".into(), "--directory".into(), format!("--title={CHOOSE}")]),
            ("kdialog", vec!["--getexistingdirectory".into(), ".".to_string()]),
        ] {
            if which(exe) {
                let mut c = Command::new(exe);
                c.args(args);
                return Ok(c);
            }
        }
        Err(anyhow!(
            "no folder chooser on this desktop (zenity or kdialog); type the path instead"
        ))
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn which(exe: &str) -> bool {
    std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).any(|d| d.join(exe).is_file()))
        .unwrap_or(false)
}

/// The stage list, for the page to draw. Kept here so there is one definition.
fn stages_json() -> String {
    serde_json::to_string(
        &STAGES.iter().map(|(_, label)| *label).collect::<Vec<_>>(),
    )
    .unwrap_or_else(|_| "[]".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scientific_exports_are_included_in_overwrite_checks() {
        let dir=std::env::temp_dir().join(format!("smokstak-export-check-{}",std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("master.fits"),b"existing").unwrap();
        std::fs::write(dir.join("master_H.xisf"),b"existing").unwrap();
        assert_eq!(results_with_exports(&dir.join("master.tif"),false,true,true),vec![dir.join("master.fits")]);
        assert_eq!(results_with_exports(&dir.join("master.tif"),true,true,true),vec![dir.join("master_H.xisf")]);
        std::fs::remove_file(dir.join("master.fits")).unwrap();
        std::fs::remove_file(dir.join("master_H.xisf")).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn sky_matching_defaults_on_and_preserves_an_explicit_opt_out() {
        let default: StartRequest = serde_json::from_str(r#"{"input":"lights","output":"stack.tif"}"#).unwrap();
        assert!(default.sky_field);
        let disabled: StartRequest = serde_json::from_str(r#"{"input":"lights","output":"stack.tif","sky_field":false}"#).unwrap();
        assert!(!disabled.sky_field);
    }

    #[test]
    fn a_line_names_the_stage_it_belongs_to() {
        assert_eq!(stage_of("m45.txt: 96 frames listed"), Some(0));
        assert_eq!(stage_of("decoding 96 frames from m45.txt"), Some(1));
        assert_eq!(stage_of("global registration in 31.17s"), Some(2));
        assert_eq!(stage_of("robustness: 4.34% of the burst suppressed"), Some(4));
        assert_eq!(stage_of("merging 96 frames with the burst-sr backend"), Some(5));
        assert_eq!(stage_of("Wrote m45.tif (6248 x 4176)"), Some(7));
        assert_eq!(stage_of("something else entirely"), None);
    }

    #[test]
    fn every_stage_a_line_can_name_exists() {
        for line in [
            "frames listed",
            "decoding ",
            "decoded ",
            "photometry:",
            "robustness",
            "merging ",
            "merge finished",
            "Wrote ",
        ] {
            let s = stage_of(line).expect("a mark with no stage");
            assert!(s < STAGES.len(), "{line} names stage {s}, past the end");
        }
    }

    fn output_of(line: &str) -> Option<String> {
        announced_path(line).map(str::to_string)
    }

    #[test]
    fn every_file_the_run_announces_is_collected() {
        // A name with parentheses of its own, as a browser saves a second
        // download of the same list. Cut at the first one, the run had
        // written nothing the page could show.
        assert_eq!(
            output_of(r"Wrote C:\out\smokstak-approved_lights (10)-stacked_H.tif (6248 x 4176)")
                .as_deref(),
            Some(r"C:\out\smokstak-approved_lights (10)-stacked_H.tif")
        );
        assert_eq!(
            output_of(concat!(
                r"Preview written to C:\out\approved_lights (10)-stacked.preview_O.png ",
                "(shadows clipped at 0.0121, midtone 0.0019; background 0.010 becomes 0.250)"
            ))
            .as_deref(),
            Some(r"C:\out\approved_lights (10)-stacked.preview_O.png")
        );
        // Nothing said after it, so the parentheses are the name's.
        assert_eq!(output_of(r"Wrote C:\out\m45 (2).tif").as_deref(), Some(r"C:\out\m45 (2).tif"));
        assert_eq!(
            output_of("Wrote C:/out/m45 (2)/batch_1.acc (weights (and values))").as_deref(),
            Some("C:/out/m45 (2)/batch_1.acc")
        );
        assert_eq!(output_of("Wrote m45.tif (6248 x 4176)").as_deref(), Some("m45.tif"));
        assert_eq!(
            output_of("Wrote m45.linear.tif (32-bit float, linear, unrestored)").as_deref(),
            Some("m45.linear.tif")
        );
        // The one that was missing: it does not begin with "Wrote".
        assert_eq!(
            output_of("Preview written to m45.preview.png (shadows clipped at 0.2)").as_deref(),
            Some("m45.preview.png")
        );
        assert_eq!(output_of("merge finished in 21s"), None);
    }

    #[test]
    fn the_lines_worth_reading_are_the_ones_lifted_out() {
        assert_eq!(
            finding_of("[..INFO  smokstak::pipeline] kernel: 4 contributing frames out of 12 loaded"),
            Some(("Frames used", "4 contributing frames out of 12 loaded".to_string()))
        );
        // The same prefix, a different line, and not the frame count. Without
        // the guard this one arrived under the heading "Frames used".
        assert_eq!(
            finding_of("[..] kernel: guide noise 1.336e-3, against 2.949e-3 from the sensor model"),
            None
        );
        assert_eq!(
            finding_of("[..] robustness: 0.04% of the burst suppressed on average"),
            Some(("Outliers", "0.04% of the burst suppressed on average".to_string()))
        );
        // The same prefix again, and a different fact. Under one heading the
        // later of the two quietly replaced the earlier.
        assert_eq!(
            finding_of(
                "[..] robustness: 19561 sites where most frames disagreed with the reference \
                 rather than with each other"
            ),
            Some((
                "Reference",
                "19561 sites where most frames disagreed with the reference rather than with \
                 each other"
                    .to_string()
            ))
        );
        assert_eq!(
            finding_of("Lucky-region selection: off"),
            Some(("Lucky regions", "off".to_string()))
        );
        assert_eq!(
            finding_of("Samples:           107498809 examined, 105559565 merged, 1982 masked"),
            Some(("Samples", "107498809 examined, 105559565 merged, 1982 masked".to_string()))
        );
        assert_eq!(finding_of("merging 12 frames with the burst-sr backend"), None);
    }

    /// The rule `inspect` suggests an output with, held on its own so that it
    /// cannot drift back to writing among somebody's frames.
    fn suggested_output(input: &Path) -> PathBuf {
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("stack");
        let dir = input
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(std::env::temp_dir);
        dir.join(format!("{stem}-stacked.tif"))
    }

    #[test]
    fn the_palettes_offered_are_the_ones_the_channels_can_make() {
        let ids = |c: &[&str]| -> Vec<&'static str> {
            candidates_for(&c.iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .iter()
                .map(|c| c.id)
                .collect()
        };
        // A narrowband set: the two Hubble orderings, and the two-filter
        // versions that ignore sulphur.
        assert_eq!(ids(&["S", "H", "O"]), ["sho", "hso", "hoo", "ohh"]);
        // Two filters cannot make a three-filter palette.
        assert_eq!(ids(&["H", "O"]), ["hoo", "ohh"]);
        assert_eq!(ids(&["R", "G", "B"]), ["rgb"]);
        // The luminance one appears only when there is an L to take it from.
        assert_eq!(ids(&["R", "G", "B", "L"]), ["rgb", "lrgb"]);
        // Filters are named in whatever case the header used.
        assert_eq!(ids(&["r", "g", "b"]), ["rgb"]);
        // One filter is not a colour picture.
        assert!(ids(&["H"]).is_empty());
    }

    #[test]
    fn every_palette_offered_is_one_the_compositor_knows() {
        for c in &CANDIDATES {
            assert!(
                sr_composite::palette(c.palette).is_some(),
                "{} offers palette {:?}, which does not exist",
                c.id,
                c.palette
            );
        }
    }

    /// The plan for a request, without touching a disk: the shape of the
    /// arguments is the thing under test.
    fn plan_of(req: &StartRequest) -> Vec<Vec<String>> {
        let how = vec!["--log".to_string(), "info".to_string()];
        plan_runs(req, Path::new("smokstak"), &how)
            .expect("a plan")
            .into_iter()
            .map(|p| p.args)
            .collect()
    }

    fn request(json: &str) -> StartRequest {
        serde_json::from_str(json).expect("a request")
    }

    #[test]
    fn a_set_that_fits_is_still_one_run() {
        let plan = plan_of(&request(
            r#"{"input":"G:/data/m45","output":"G:/out/m45.tif","split_by_filter":true}"#,
        ));
        assert_eq!(plan.len(), 1, "nothing to split into: {plan:?}");
        assert!(plan[0].contains(&"--split-by-filter".to_string()), "{plan:?}");
        // And no reference is imposed: one run picks its own, as it always did.
        assert!(!plan[0].contains(&"--reference-file".to_string()), "{plan:?}");
    }

    #[test]
    fn asking_for_a_frame_count_is_answered_with_one_run() {
        // The reader has already said how much of the set to use, so using all
        // of it across several runs is not what was asked for.
        let plan = plan_of(&request(
            r#"{"input":"G:/data/m45","output":"G:/out/m45.tif",
                "whole_set":true,"max_frames":100}"#,
        ));
        assert_eq!(plan.len(), 1, "{plan:?}");
        assert!(plan[0].contains(&"--max-frames".to_string()), "{plan:?}");
    }

    #[test]
    fn large_sets_keep_global_decisions_and_report_the_masks() {
        let req = request(r#"{"input":"frames.txt","output":"out/master.tif",
            "whole_set":true,"per_frame":100,"affordable":200,"scratch_dir":"fast disk/scratch"}"#);
        let plan = plan_of(&req);
        assert_eq!(plan.len(), 1);
        assert!(!plan[0].contains(&"--accumulate".into()));
        assert!(plan[0].windows(2).any(|a| a == ["--scratch-dir", "fast disk/scratch"]));
        assert!(plan[0].contains(&"--diagnostics".into()));
        assert!(!plan_of(&request(r#"{"input":"a","output":"b.tif","diagnostics":false}"#))[0]
            .contains(&"--diagnostics".into()));
    }

    #[test]
    fn a_result_is_never_suggested_among_the_frames() {
        // A directory of frames gets a result beside it, not inside it.
        let frames = Path::new("Z:/site1/2026-07-23/LIGHT/WR 134");
        let out = suggested_output(frames);
        assert_eq!(out.parent().unwrap(), Path::new("Z:/site1/2026-07-23/LIGHT"));
        assert_ne!(out.parent().unwrap(), frames, "not among the frames");

        // A list gets one beside the list, and never beside what it names:
        // the frames are data, and this program does not write there.
        let list = Path::new("C:/Users/me/Downloads/approved_lights(13).txt");
        let out = suggested_output(list);
        assert_eq!(out.parent().unwrap(), Path::new("C:/Users/me/Downloads"));
        assert_eq!(
            out.file_name().unwrap().to_string_lossy(),
            "approved_lights(13)-stacked.tif"
        );
    }

    #[test]
    fn what_one_frame_costs_is_read_from_the_line_the_run_logs() {
        let line = "[2026-09-08T06:48:57.168Z INFO  smokstak::pipeline] decoding 1 frames from \
                    G:/data (about 107.8 MiB resident, 107.8 MiB per frame: 49.8 MiB of samples \
                    and 58.1 MiB of guide and pyramid)";
        assert_eq!(per_frame_bytes(line), Some((107.8 * 1024.0 * 1024.0) as u64));
        // 634 of those is the set that could not be read at all.
        let each = per_frame_bytes(line).unwrap();
        assert!(each * 634 > 60 * 1024 * 1024 * 1024, "a night of them is tens of gigabytes");
        assert_eq!(per_frame_bytes("nothing about memory here"), None);
    }

    #[test]
    fn running_out_of_memory_says_what_to_do_about_it() {
        let abort = "this burst needs roughly 156.5 GiB of memory for the frames, their guides \
                     and their pyramids, before any reconstruction buffers.\n\
                     memory allocation of 61171488 bytes failed\n\
                     note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace";
        let said = readable_failure(abort);
        assert!(said.contains("156.5 GiB"), "the load, not the last straw: {said}");
        assert!(said.contains("fewer"), "{said}");
        assert!(!said.contains("RUST_BACKTRACE"), "not the reader's program to debug: {said}");
        assert!(!said.contains("61171488"), "{said}");
        // Anything else is passed through as it was written.
        assert_eq!(readable_failure("no frames in G:\\data were readable"),
                   "no frames in G:\\data were readable");
    }

    #[test]
    fn a_run_notices_the_result_it_would_write_over() {
        let dir = std::env::temp_dir().join("smokstak-overwrite-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("m45.tif");

        assert!(results_in_the_way(&out, false).is_empty(), "nothing written yet");
        std::fs::write(&out, b"x").unwrap();
        assert_eq!(results_in_the_way(&out, false), vec![out.clone()]);

        // A per-filter run does not write the name it was given, so that name
        // being free says nothing about the masters that would be replaced.
        std::fs::remove_file(&out).unwrap();
        for f in ["m45_H.tif", "m45_O.tif", "m45_S.tif"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        // Written from the same stem, and not a master.
        std::fs::write(dir.join("m45_H.linear.tif"), b"x").unwrap();
        std::fs::write(dir.join("m45.preview_H.png"), b"x").unwrap();
        // Somebody else's stack of the same object.
        std::fs::write(dir.join("m45-old.tif"), b"x").unwrap();

        assert!(results_in_the_way(&out, false).is_empty(), "the given name is still free");
        let found = results_in_the_way(&out, true);
        let names: Vec<String> =
            found.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["m45_H.tif", "m45_O.tif", "m45_S.tif"], "{names:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cancelled_chooser_is_a_choice_and_not_a_failure() {
        // Every chooser says nothing when it is closed without a choice, and
        // the page reads that as "carry on", not as an error.
        assert_eq!(folder_from(b""), "");
        assert_eq!(folder_from(b"\n"), "");
        assert_eq!(folder_from(b"\r\n"), "");
    }

    #[test]
    fn a_chosen_folder_survives_being_read_back() {
        assert_eq!(folder_from(b"G:\\data\\M45\\lights"), r"G:\data\M45\lights");
        // zenity and osascript both end the line; the path does not.
        assert_eq!(folder_from(b"/home/a/M45 lights\n"), "/home/a/M45 lights");
        // A trailing space is part of the name, however unwise the name is.
        assert_eq!(folder_from(b"/home/a/lights \n"), "/home/a/lights ");
    }

    #[test]
    fn the_chooser_is_the_one_this_platform_has() {
        let c = folder_chooser().expect("this platform has a chooser, or says so");
        let exe = c.get_program().to_string_lossy().into_owned();
        let args: Vec<String> =
            c.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        if cfg!(windows) {
            assert_eq!(exe, "powershell");
            // -STA is not decoration: the shell dialog cannot be shown from a
            // multi-threaded apartment, which is what powershell uses by default.
            assert!(args.iter().any(|a| a == "-STA"), "{args:?}");
            assert!(args.iter().any(|a| a.contains("FolderBrowserDialog")), "{args:?}");
        } else if cfg!(target_os = "macos") {
            assert_eq!(exe, "osascript");
        } else {
            assert!(exe == "zenity" || exe == "kdialog", "{exe}");
        }
        assert!(
            args.iter().any(|a| a.contains(CHOOSE)),
            "the dialog should say what it is asking for: {args:?}"
        );
    }

    #[test]
    fn only_one_chooser_opens_at_a_time() {
        // Held, as a click that beat the first dialog onto the screen would.
        assert!(!PICKING.swap(true, Ordering::SeqCst));
        let e = pick_folder().expect_err("a second chooser must not open behind the first");
        assert!(e.to_string().contains("already open"), "{e}");
        // And the guard the refusal found is the one it leaves behind: a
        // refused second click must not release the first dialog's hold.
        assert!(PICKING.swap(false, Ordering::SeqCst));
    }

    #[test]
    fn only_loopback_names_are_served() {
        // A browser sends the name the page was loaded from, so this is what
        // separates our own page from a site that has pointed a name it owns
        // at 127.0.0.1 to reach this server through the browser.
        assert!(host_is_loopback("127.0.0.1:7878"));
        assert!(host_is_loopback("localhost:7878"));
        assert!(host_is_loopback("LocalHost"));
        assert!(host_is_loopback("[::1]:7878"));
        assert!(host_is_loopback(""), "a client that sends no Host is not a browser");
        assert!(!host_is_loopback("stack.example.com"));
        assert!(!host_is_loopback("stack.example.com:7878"));
        assert!(!host_is_loopback("localhost.example.com"));
    }

    #[test]
    fn only_our_own_page_may_post() {
        assert!(origin_is_loopback("http://127.0.0.1:7878"));
        assert!(origin_is_loopback("http://localhost:7879"));
        assert!(origin_is_loopback("http://[::1]:7878"));
        // A form on some other site, aimed at this server.
        assert!(!origin_is_loopback("https://stack.example.com"));
        assert!(!origin_is_loopback("http://localhost.example.com:7878"));
        // What a sandboxed frame or a file on disk sends.
        assert!(!origin_is_loopback("null"));
        assert!(!origin_is_loopback("http://"));
    }

    #[test]
    fn a_path_survives_the_query_string_it_travelled_in() {
        assert_eq!(
            percent_decode("C%3A%5CUsers%5CA%20B%5Cm45.preview.png"),
            r"C:\Users\A B\m45.preview.png"
        );
        assert_eq!(percent_decode("plain.png"), "plain.png");
        // A stray percent is a percent, not a reason to lose the rest.
        assert_eq!(percent_decode("100%"), "100%");
    }

    #[test]
    fn the_page_may_fetch_only_what_a_run_wrote() {
        let job: Arc<Mutex<Job>> = Arc::new(Mutex::new(Job::default()));
        let dir = std::env::temp_dir();
        let mine = dir.join("smokstak-served-test.png");
        let theirs = dir.join("smokstak-unserved-test.png");
        std::fs::write(&mine, b"\x89PNG\r\n\x1a\n").unwrap();
        std::fs::write(&theirs, b"\x89PNG\r\n\x1a\n").unwrap();

        job.lock().unwrap().announce(mine.to_string_lossy().into_owned());

        let q = |p: &Path| format!("path={}", p.to_string_lossy().replace('\\', "%5C"));
        let (kind, bytes) = serve_file(&q(&mine), &job).expect("a file this run wrote");
        assert_eq!(kind, "image/png");
        assert_eq!(bytes.len(), 8);

        // It exists, it is readable, it sits in the same directory, and it is
        // still none of the page's business.
        let e = serve_file(&q(&theirs), &job).expect_err("a file no run wrote");
        assert!(e.to_string().contains("not a file this run wrote"), "{e}");

        let _ = std::fs::remove_file(mine);
        let _ = std::fs::remove_file(theirs);
    }

    #[test]
    fn a_trial_takes_frames_spread_across_the_burst_unless_told_otherwise() {
        let r: StartRequest =
            serde_json::from_str(r#"{"input":"lights","output":"o.tif","max_frames":10}"#).unwrap();
        assert_eq!(r.max_frames, 10);
        assert_eq!(r.select, "", "the page may leave the choice to the server");
    }

    #[test]
    fn a_pasted_list_of_relative_paths_is_refused_rather_than_guessed() {
        let e = list_from_text("frame_0001.fit\nframe_0002.fit\n", "lights.txt")
            .expect_err("relative paths cannot be resolved without their folder");
        assert!(e.to_string().contains("relative"), "{e}");
    }

    #[test]
    fn an_empty_list_says_so() {
        let e = list_from_text("# only a comment\n\n", "lights.txt").expect_err("no frames");
        assert!(e.to_string().contains("no frames"), "{e}");
    }

    #[test]
    fn a_pasted_list_of_absolute_paths_is_written_somewhere_the_stacker_can_read() {
        let body = if cfg!(windows) {
            "C:\\data\\a.fit\nC:\\data\\b.fit\n"
        } else {
            "/data/a.fit\n/data/b.fit\n"
        };
        let p = list_from_text(body, "lights.txt").expect("absolute paths are fine");
        assert!(p.exists());
        assert_eq!(std::fs::read_to_string(&p).unwrap(), body);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn pasted_lists_are_isolated_and_accept_windows_copy_as_path() {
        let a=std::env::temp_dir().join("frame-a.fit");
        let b=std::env::temp_dir().join("frame-b.fit");
        let first=list_from_text(&format!("\u{feff}\"{}\"\n",a.display()),"same-list.txt").unwrap();
        let second=list_from_text(&format!("\"{}\"\n",b.display()),"same-list.txt").unwrap();
        assert_ne!(first,second);
        assert_eq!(std::fs::read_to_string(&first).unwrap(),format!("{}\n",a.display()));
        assert_eq!(std::fs::read_to_string(&second).unwrap(),format!("{}\n",b.display()));
        std::fs::remove_file(first).unwrap(); std::fs::remove_file(second).unwrap();
    }

    fn planned(kind: &'static str) -> Planned {
        Planned {
            step: String::new(),
            program: PathBuf::from("smokstak"),
            args: Vec::new(),
            last_stage: 0,
            kind,
        }
    }

    /// A stack that finished with two masters and something to say about itself.
    fn stacked(j: &mut Job, name: &str) -> u64 {
        j.progress = begin(j, &planned("stack"), 1, name.into(), None).unwrap();
        j.announce(format!("G:/out/{name}_H.tif"));
        j.announce(format!("G:/out/{name}_O.tif"));
        j.progress.findings.push(Finding {
            group: "H".into(),
            label: "Outliers".into(),
            text: "0.04% of the burst suppressed on average".into(),
        });
        settle(j, true, false, 3600.0, 7);
        j.progress.id
    }

    #[test]
    fn combining_the_masters_adds_to_the_stack_rather_than_replacing_it() {
        let mut j = Job::default();
        let id = stacked(&mut j, "m42");

        j.progress = begin(&mut j, &planned("composite"), 1, "HO · hoo".into(), Some(id)).unwrap();
        j.announce("G:/out/m42-hoo.tif".into());
        j.announce("G:/out/m42-hoo.preview.png".into());
        settle(&mut j, true, false, 4.0, 0);

        assert_eq!(j.progress.outputs, ["G:/out/m42_H.tif", "G:/out/m42_O.tif"],
                   "a colour image is not a master to combine next time");
        assert_eq!(j.progress.findings.len(), 1, "what the stack said about itself is still there");
        assert_eq!(j.progress.took, 3600.0, "the stack took an hour, not the seconds combining did");
        assert_eq!(j.progress.colour.len(), 1);
        assert_eq!(j.progress.colour[0].files, ["G:/out/m42-hoo.tif", "G:/out/m42-hoo.preview.png"]);
        assert_eq!(j.past.len(), 1, "one stack combined is one piece of work");

        // A second palette, and then one that fails.
        j.progress = begin(&mut j, &planned("composite"), 1, "HO · ohh".into(), Some(id)).unwrap();
        j.announce("G:/out/m42-ohh.tif".into());
        settle(&mut j, true, false, 4.0, 0);
        j.progress = begin(&mut j, &planned("composite"), 1, "HO · sho".into(), Some(id)).unwrap();
        settle(&mut j, false, false, 1.0, 0);

        assert!(j.progress.failed);
        assert_eq!(j.progress.colour.len(), 2, "a colour image never written is not shown");
        assert_eq!(j.past.len(), 1);
        assert_eq!(j.past[0].colour.len(), 2, "both made, both kept");
        assert!(!j.past[0].failed, "a combine that failed did not fail the stack");

        // And the next one starts from the kept stack, not from the failure.
        j.progress = begin(&mut j, &planned("composite"), 1, "HO · hoo".into(), Some(id)).unwrap();
        assert!(!j.progress.failed);
        assert_eq!(j.progress.colour.len(), 3);
    }

    #[test]
    fn an_earlier_stack_is_combined_from_its_own_masters() {
        let mut j = Job::default();
        let first = stacked(&mut j, "m42");
        let second = stacked(&mut j, "m45");
        assert_ne!(first, second);

        j.progress = begin(&mut j, &planned("composite"), 1, "HO · hoo".into(), Some(first)).unwrap();
        assert_eq!(j.progress.outputs, ["G:/out/m42_H.tif", "G:/out/m42_O.tif"],
                   "the stack the page was showing, not the one that ran last");
        j.announce("G:/out/m42-hoo.tif".into());
        settle(&mut j, true, false, 4.0, 0);
        assert_eq!(j.past.len(), 2);
        assert_eq!(j.past[0].colour.len(), 1);
        assert!(j.past[1].colour.is_empty());

        assert!(begin(&mut j, &planned("composite"), 1, String::new(), Some(99)).is_err());
    }

    #[test]
    fn a_night_stacked_in_batches_ends_as_a_stack_though_its_last_run_adds_them() {
        let mut j = Job::default();
        j.progress = begin(&mut j, &planned("stack"), 4, "m42 · 4 runs".into(), None).unwrap();
        // What `spawn_next` does as the run that adds the batches starts.
        j.progress.kind = "composite".into();
        j.announce("G:/out/m42.tif".into());
        settle(&mut j, true, false, 7200.0, 0);

        assert_eq!(j.progress.outputs, ["G:/out/m42.tif"], "the sum is the master, not a colour image");
        assert_eq!(j.progress.took, 7200.0);
        assert_eq!(j.past.len(), 1);
    }

    #[test]
    fn earlier_runs_outlive_the_server_but_only_their_pictures_are_served() {
        let dir = std::env::temp_dir().join("smokstak-history-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let master = dir.join("m42_H.tif");
        let note = dir.join("m42.txt");
        let gone = dir.join("m42_O.tif");
        std::fs::write(&master, b"x").unwrap();
        std::fs::write(&note, b"x").unwrap();
        let file = dir.join("history.json");

        let mut before = Job { history: Some(file.clone()), ..Default::default() };
        let id = stacked(&mut before, "m42");
        before.past[0].outputs = [&master, &note, &gone]
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        before.past[0].palettes.push(PaletteShot {
            id: "sho".into(),
            label: "SHO".into(),
            preview: dir.join("gone.preview.png").to_string_lossy().into_owned(),
        });
        save_history(&before);

        let mut after = Job::default();
        restore(&mut after, load_history(&file));
        assert_eq!(after.past.len(), 1);
        assert_eq!(after.past[0].id, id);
        assert_eq!(after.past[0].findings.len(), 1, "what the stack said about itself came back");
        assert_eq!(after.past[0].outputs.len(), 3, "the record is kept as it was");
        assert!(after.past[0].palettes.is_empty(), "a thumbnail that is gone is not offered");
        let served = |p: &Path| after.served.contains(&std::fs::canonicalize(p).unwrap_or(p.to_path_buf()));
        assert!(served(&master));
        assert!(!served(&note), "not an image, so not the page's business");
        assert!(!served(&gone));

        // And new work does not take the id of work already kept.
        after.progress = begin(&mut after, &planned("stack"), 1, "m45".into(), None).unwrap();
        assert!(after.progress.id > id);

        // A list that cannot be read is a new list, not a server that will not start.
        std::fs::write(&file, b"{ not json").unwrap();
        assert!(load_history(&file).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_frame_that_stops_the_report_is_named_so_it_can_be_left_out() {
        let all = vec![
            PathBuf::from(r"G:\data\f1.fit"),
            PathBuf::from(r"G:\data\f10.fit"),
        ];
        let said = r"Error: input error: G:\data\f10.fit: data unit is short of the 512x512 the header declares";
        assert_eq!(unreadable_in(said, &all, &[]), Some(all[1].clone()));
        assert_eq!(unreadable_in(said, &all, &[all[1].clone()]), None, "named once, not again");
        assert_eq!(unreadable_in("memory allocation of 61171488 bytes failed", &all, &[]), None);
    }

    #[test]
    fn frames_left_out_are_left_out_of_the_list_the_stacker_reads() {
        let dir = std::env::temp_dir().join("smokstak-without-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..4 {
            std::fs::write(dir.join(format!("f{i}.fits")), b"SIMPLE").unwrap();
        }
        let all = sr_raw::collect_files(&dir, None).unwrap();
        let trailed = all[2].to_string_lossy().into_owned();

        let (list, dropped) =
            without(&dir.to_string_lossy(), std::slice::from_ref(&trailed)).unwrap();
        assert_eq!(dropped, 1);
        let text = std::fs::read_to_string(&list).unwrap();
        assert_eq!(text.lines().count(), 3);
        assert!(!text.contains("f2.fits"), "{text}");
        assert_eq!(sr_raw::collect_files(&list, None).unwrap().len(), 3, "and the stacker reads it");

        let every: Vec<String> = all.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        assert!(without(&dir.to_string_lossy(), &every).is_err(), "nothing left to stack");

        let _ = std::fs::remove_file(list);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
