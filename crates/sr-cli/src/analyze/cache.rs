//! Content-addressed scalar records plus compact, checked sample blocks.
use super::{stats::PIXELS, Record};
use crate::cache::{Cache, Fingerprint};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn digest_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("reading {}", path.display()))?;
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

pub fn key(build: &str, input: &str, reference: &str, options: &str) -> String {
    Fingerprint::new()
        .text("project-analysis-v1")
        .text(build)
        .text(input)
        .text(reference)
        .text(options)
        .finish()
}

pub struct Store {
    dir: PathBuf,
    records: Cache,
    temporary: bool,
}

impl Store {
    pub fn new(dir: &Path, temporary: bool) -> Result<Self> {
        let dir = if temporary {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let p = std::env::temp_dir()
                .join(format!("smokstak-analysis-{}-{nonce}", std::process::id()));
            fs::create_dir(&p)?;
            p
        } else {
            fs::create_dir_all(dir)?;
            dir.to_path_buf()
        };
        Ok(Self {
            records: Cache::new(Some(&dir)),
            dir,
            temporary,
        })
    }
    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("analysis-{key}.samples"))
    }

    pub fn load(&self, key: &str) -> Option<Record> {
        let r: Record = self.records.load("analysis", key)?;
        let expected = r.valid_tiles.len().checked_mul(PIXELS)?.checked_mul(4)?;
        let p = self.path(key);
        if fs::metadata(&p).ok()?.len() != expected as u64 {
            return None;
        }
        if digest_file(&p).ok()? != r.samples_digest {
            return None;
        }
        Some(r)
    }

    pub fn save(&self, key: &str, record: &mut Record, values: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        record.samples_digest = format!("{:x}", Sha256::digest(&bytes));
        let path = self.path(key);
        // An interrupted write cannot become a hit: length and digest are
        // checked before any entry is reused. JSON is written after samples.
        File::create(path)?.write_all(&bytes)?;
        self.records.store("analysis", key, record);
        Ok(())
    }

    pub fn samples(&self, key: &str, record: &Record) -> Result<Vec<f32>> {
        let bytes = fs::read(self.path(key))?;
        anyhow::ensure!(
            bytes.len() == record.valid_tiles.len() * PIXELS * 4
                && format!("{:x}", Sha256::digest(&bytes)) == record.samples_digest,
            "sample cache changed during analysis; rerun the command"
        );
        Ok(bytes
            .as_chunks::<4>().0.iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        if self.temporary {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keys_cover_content_reference_config_and_build() {
        let original = key("build", "input", "reference", "options");
        assert_eq!(original, key("build", "input", "reference", "options"));
        for changed in [
            key("new", "input", "reference", "options"),
            key("build", "new", "reference", "options"),
            key("build", "input", "new", "options"),
            key("build", "input", "reference", "new"),
        ] {
            assert_ne!(original, changed);
        }
    }
    #[test]
    fn middle_edit_changes_digest_even_with_equal_length() {
        let store = Store::new(Path::new("unused"), true).unwrap();
        let p = store.dir.join("input");
        let mut bytes = vec![0; 300_000];
        fs::write(&p, &bytes).unwrap();
        let before = digest_file(&p).unwrap();
        bytes[150_000] = 1;
        fs::write(&p, &bytes).unwrap();
        assert_ne!(before, digest_file(&p).unwrap());
    }
}
