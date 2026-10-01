//! Reusing what a rerun would otherwise recompute.
//!
//! Tuning a reconstruction means changing the scale, the kernel or the region
//! and looking at the result. None of those change how the frames are aligned,
//! which sites are defective, or how bright each frame is — and on a small
//! region those stages are almost the entire run. Reconstructing a 400-pixel
//! crop of the 100-frame chart burst takes 68.7 seconds, of which the merge,
//! the only part that depends on what is being tuned, is **1.5**.
//!
//! ```text
//!   register             21.9 s   31.9%
//!   defects              15.6 s   22.6%
//!   robustness           15.0 s   21.8%
//!   decode                9.2 s   13.5%
//!   diagnostics           4.8 s    7.0%
//!   merge                 1.5 s    2.1%
//! ```
//!
//! ## What is cached, and what is not
//!
//! Registration and the defect scan: 37.5 of those 68.7 seconds. Both are pure
//! functions of the frames and a few configuration fields, and neither depends
//! on anything an operator adjusts between runs.
//!
//! Decode is not cached. It would need the decoded frames on disk — nine
//! gigabytes for that burst — and the frames are needed in memory anyway, so it
//! buys a read instead of a decode rather than nothing at all. Robustness is
//! not cached yet: it is the largest entry by far, a map per frame, and the
//! most entangled with things that do change.
//!
//! ## Why it is off by default
//!
//! A cache that returns a stale answer is worse than no cache, because the
//! answer is wrong and looks right. The fingerprint covers the input files by
//! content, and the configuration fields the stages actually read — but it
//! cannot cover a change to the *code*, which during development is exactly
//! what changes. It carries the program version and revision, which helps for a
//! release and not at all for a working tree.
//!
//! So it is opt-in, and the run says when it used an entry. Clear the directory
//! after changing registration or defect detection.

use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use sr_core::frame::RawFrame;
use sr_core::samples::DefectMask;

/// Bumped when the meaning of a stored entry changes, which invalidates every
/// entry written by an older build.
const FORMAT: u32 = 1;

/// A cache directory, or nothing.
pub struct Cache {
    dir: Option<PathBuf>,
}

impl Cache {
    pub fn new(dir: Option<&Path>) -> Self {
        Self { dir: dir.map(|d| d.to_path_buf()) }
    }

    fn path(&self, kind: &str, fingerprint: &str) -> Option<PathBuf> {
        Some(self.dir.as_ref()?.join(format!("{kind}-{fingerprint}.json")))
    }

    /// Read an entry, or `None` for any reason at all.
    ///
    /// A damaged or unreadable entry is a miss rather than an error: the whole
    /// point is that the cache is an optimisation, and the run has to be able
    /// to proceed without it.
    pub fn load<T: DeserializeOwned>(&self, kind: &str, fingerprint: &str) -> Option<T> {
        let p = self.path(kind, fingerprint)?;
        let text = std::fs::read_to_string(&p).ok()?;
        match serde_json::from_str(&text) {
            Ok(v) => {
                log::info!("cache hit: {}", p.display());
                Some(v)
            }
            Err(e) => {
                log::warn!("ignoring unreadable cache entry {}: {e}", p.display());
                None
            }
        }
    }

    pub fn store<T: Serialize>(&self, kind: &str, fingerprint: &str, value: &T) {
        let Some(p) = self.path(kind, fingerprint) else {
            return;
        };
        let write = || -> anyhow::Result<()> {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&p, serde_json::to_string(value)?)?;
            Ok(())
        };
        match write() {
            Ok(()) => log::info!("cached {}", p.display()),
            // Not being able to write a cache entry is not a reason to fail a
            // reconstruction that has already succeeded.
            Err(e) => log::warn!("could not write {}: {e}", p.display()),
        }
    }
}

/// Build a fingerprint from the things an entry depends on.
///
/// Every part is written into one hash, so a caller cannot accidentally produce
/// the same fingerprint from different inputs by reordering them.
pub struct Fingerprint {
    hasher: Sha256,
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self::new()
    }
}

impl Fingerprint {
    pub fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(FORMAT.to_le_bytes());
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
        hasher.update(option_env!("SRSTACK_REVISION").unwrap_or("unrecorded").as_bytes());
        Self { hasher }
    }

    pub fn text(mut self, s: &str) -> Self {
        self.hasher.update((s.len() as u64).to_le_bytes());
        self.hasher.update(s.as_bytes());
        self
    }

    /// Anything with a serialised form, which is how configuration structs get
    /// in: a field added to one of them changes the fingerprint without anyone
    /// having to remember to list it here.
    pub fn config<T: Serialize>(self, value: &T) -> Self {
        let s = serde_json::to_string(value).unwrap_or_default();
        self.text(&s)
    }

    /// The frames, by content rather than by name.
    ///
    /// `sha256_prefix` is a bounded head-and-tail digest, which is enough to
    /// notice that a file changed and cheap enough to have been computed during
    /// decode already.
    pub fn frames(mut self, frames: &[RawFrame]) -> Self {
        self.hasher.update((frames.len() as u64).to_le_bytes());
        for f in frames {
            self = self.text(&f.metadata.file_name).text(&f.metadata.sha256_prefix);
        }
        self
    }

    pub fn indices(mut self, idx: &[usize]) -> Self {
        self.hasher.update((idx.len() as u64).to_le_bytes());
        for &i in idx {
            self.hasher.update((i as u64).to_le_bytes());
        }
        self
    }

    pub fn finish(self) -> String {
        let d = self.hasher.finalize();
        d[..12].iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// A defect mask, stored as the sites that are set.
///
/// The mask is a bitset over every sensor site — 183 thousand words for a
/// 45 MP sensor, nearly all of them zero. What it actually holds is a thousand
/// or so indices, and storing those is a hundredth of the size and readable
/// besides.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedDefects {
    pub width: usize,
    pub height: usize,
    pub sites: Vec<u32>,
    pub hot: usize,
    pub cold: usize,
}

impl CachedDefects {
    pub fn from_mask(mask: &DefectMask, hot: usize, cold: usize) -> Self {
        let mut sites = Vec::new();
        for i in 0..mask.width * mask.height {
            if mask.get(i) {
                sites.push(i as u32);
            }
        }
        Self { width: mask.width, height: mask.height, sites, hot, cold }
    }

    pub fn to_mask(&self) -> DefectMask {
        let mut m = DefectMask::none(self.width, self.height);
        for &i in &self.sites {
            m.set(i as usize);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let mut d = std::env::current_exe().unwrap();
        d.pop();
        d.push("cache-tests");
        d.push(name);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_disabled_cache_stores_nothing_and_finds_nothing() {
        let c = Cache::new(None);
        c.store("thing", "abc", &42u32);
        assert_eq!(c.load::<u32>("thing", "abc"), None);
    }

    #[test]
    fn what_is_stored_comes_back() {
        let dir = scratch("roundtrip");
        let c = Cache::new(Some(&dir));
        c.store("thing", "abc", &vec![1u32, 2, 3]);
        assert_eq!(c.load::<Vec<u32>>("thing", "abc"), Some(vec![1, 2, 3]));
        // A different fingerprint is a different entry, which is the whole
        // mechanism: nothing is invalidated, it is simply not found.
        assert_eq!(c.load::<Vec<u32>>("thing", "def"), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_damaged_entry_is_a_miss_rather_than_a_failure() {
        let dir = scratch("damaged");
        let c = Cache::new(Some(&dir));
        c.store("thing", "abc", &vec![1u32, 2, 3]);
        std::fs::write(dir.join("thing-abc.json"), "{not json").unwrap();
        assert_eq!(c.load::<Vec<u32>>("thing", "abc"), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fingerprints_separate_what_should_be_separate() {
        let a = Fingerprint::new().text("one").text("two").finish();
        let b = Fingerprint::new().text("one").text("two").finish();
        assert_eq!(a, b, "the same inputs must give the same fingerprint");

        // Reordering, or moving a boundary, has to change it. Hashing the
        // concatenation without lengths would make "one","two" and "onet","wo"
        // the same entry.
        let c = Fingerprint::new().text("two").text("one").finish();
        let d = Fingerprint::new().text("onet").text("wo").finish();
        assert_ne!(a, c);
        assert_ne!(a, d);

        let e = Fingerprint::new().indices(&[1, 2]).finish();
        let f = Fingerprint::new().indices(&[2, 1]).finish();
        assert_ne!(e, f);
    }

    #[test]
    fn a_defect_mask_survives_the_round_trip() {
        let mut m = DefectMask::none(640, 480);
        for &i in &[0usize, 63, 64, 4095, 640 * 480 - 1] {
            m.set(i);
        }
        let stored = CachedDefects::from_mask(&m, 4, 1);
        assert_eq!(stored.sites.len(), 5);
        let back = stored.to_mask();
        for i in 0..640 * 480 {
            assert_eq!(m.get(i), back.get(i), "site {i}");
        }
        assert_eq!(back.count(), 5);
    }
}
