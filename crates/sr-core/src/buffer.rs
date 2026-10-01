//! Exact, optional spill storage for immutable raster buffers.
//!
//! Spill files are private scratch, not a persistent cache. Mapping removes
//! frame-count-sized heap allocations; the operating system controls residency.
//! Mutating a mapped buffer creates an independent owned copy.

use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    marker::PhantomData,
    mem::{align_of, size_of},
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u16 {}
    impl Sealed for u8 {}
}

/// Types with no padding and for which every bit pattern is valid.
pub trait Primitive: sealed::Sealed + Copy + Send + Sync + 'static {}
impl Primitive for f32 {}
impl Primitive for u16 {}
impl Primitive for u8 {}

struct Backing {
    map: Option<memmap2::Mmap>,
    file: Option<File>,
    path: PathBuf,
    delete_on_drop: bool,
}

impl Drop for Backing {
    fn drop(&mut self) {
        // Windows does not permit deleting a file while its mapping is open.
        drop(self.map.take());
        drop(self.file.take());
        if self.delete_on_drop {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Public only because it is the payload of `Buffer::Mapped`; construction is
/// private so all mapped typed slices have passed primitive/alignment checks.
pub struct MappedBuffer<T> {
    backing: Arc<Backing>,
    len: usize,
    marker: PhantomData<T>,
}

pub enum Buffer<T> {
    Owned(Vec<T>),
    Mapped(MappedBuffer<T>),
}

impl<T> Buffer<T> {
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Owned(values) => values,
            Self::Mapped(values) => {
                let map = values.backing.map.as_ref().expect("live mapped buffer");
                // SAFETY: only spill constructs this variant, only for sealed
                // primitives. It verifies byte length/alignment. Backing keeps
                // the immutable mapping alive for this entire borrow.
                unsafe { std::slice::from_raw_parts(map.as_ptr().cast::<T>(), values.len) }
            }
        }
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped(_))
    }
}

impl<T: Clone> Buffer<T> {
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if matches!(self, Self::Mapped(_)) {
            *self = Self::Owned(self.as_slice().to_vec());
        }
        match self {
            Self::Owned(values) => values,
            Self::Mapped(_) => unreachable!(),
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Owned(values) => values,
            Self::Mapped(_) => self.as_slice().to_vec(),
        }
    }
}

impl<T: Primitive> Buffer<T> {
    /// Move an owned buffer into immutable process-private scratch storage.
    /// Failure leaves the original buffer unchanged. Empty buffers stay owned.
    pub fn spill(&mut self, dir: &Path) -> io::Result<()> {
        if self.is_mapped() || self.is_empty() {
            return Ok(());
        }
        fs::create_dir_all(dir)?;
        static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
        let (path, file) = loop {
            let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!(
                "smokstak-spill-{}-{sequence}.bin",
                std::process::id()
            ));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            // Do not allow another Windows handle to modify/truncate this
            // backing file while typed slices refer to its mapping.
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                options.share_mode(1); // FILE_SHARE_READ
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => break (path, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        };
        let mut backing = Backing {
            map: None,
            file: Some(file),
            path,
            delete_on_drop: true,
        };
        let len = self.len();
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "spill size overflow"))?;
        // SAFETY: sealed primitives have no padding, so every source byte is
        // initialized and can be copied verbatim without conversion.
        let raw = unsafe { std::slice::from_raw_parts(self.as_ptr().cast::<u8>(), bytes) };
        let file = backing.file.as_mut().expect("new spill file");
        file.write_all(raw)?;
        file.flush()?;
        // SAFETY: this is a newly created, exclusively owned scratch file.
        // Its handle stays alive and no writes follow mapping. On Windows its
        // sharing mode prevents third-party writers; on Unix unlink below
        // makes it inaccessible by name for the mapping's lifetime.
        let map = unsafe { memmap2::MmapOptions::new().len(bytes).map(&*file)? };
        if map.len() != bytes || !(map.as_ptr() as usize).is_multiple_of(align_of::<T>()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid spill mapping layout",
            ));
        }
        backing.map = Some(map);
        #[cfg(unix)]
        {
            fs::remove_file(&backing.path)?;
            backing.delete_on_drop = false;
        }
        *self = Self::Mapped(MappedBuffer {
            backing: Arc::new(backing),
            len,
            marker: PhantomData,
        });
        Ok(())
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    fn from(values: Vec<T>) -> Self {
        Self::Owned(values)
    }
}
impl<T> FromIterator<T> for Buffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::Owned(iter.into_iter().collect())
    }
}
impl<T> AsRef<[T]> for Buffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: PartialEq> PartialEq<Vec<T>> for Buffer<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}
impl<T> Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: Clone> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
impl<T: Clone> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Owned(values) => Self::Owned(values.clone()),
            Self::Mapped(values) => Self::Mapped(MappedBuffer {
                backing: Arc::clone(&values.backing),
                len: values.len,
                marker: PhantomData,
            }),
        }
    }
}
impl<T: fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}
impl<T: PartialEq> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T: Eq> Eq for Buffer<T> {}
impl<T: Clone> IntoIterator for Buffer<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}
impl<'a, T> IntoIterator for &'a Buffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a, T: Clone> IntoIterator for &'a mut Buffer<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scratch() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "smokstak-buffer-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }
    #[test]
    fn spill_preserves_float_bits_and_copy_on_write_isolation() {
        let dir = scratch();
        let values = vec![
            -0.02f32,
            -0.0,
            0.75,
            f32::INFINITY,
            f32::from_bits(0x7fc0_0042),
        ];
        let bits: Vec<_> = values.iter().map(|v| v.to_bits()).collect();
        let mut a = Buffer::from(values);
        a.spill(&dir).unwrap();
        assert!(a.is_mapped());
        assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
        let mut b = a.clone();
        assert!(b.is_mapped());
        b[0] = 1.0;
        assert!(!b.is_mapped());
        assert_eq!(a[0].to_bits(), bits[0]);
        drop(a);
        assert_eq!(b[0], 1.0);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn spill_roundtrips_integers_and_last_clone_removes_file() {
        let dir = scratch();
        let mut values = Buffer::from(vec![0u16, 1, 40000, 40001, u16::MAX]);
        values.spill(&dir).unwrap();
        let survivor = values.clone();
        drop(values);
        #[cfg(windows)]
        {
            let files: Vec<_> = fs::read_dir(&dir).unwrap().collect();
            assert_eq!(files.len(), 1);
            let path = files[0].as_ref().unwrap().path();
            assert!(OpenOptions::new().write(true).open(path).is_err());
        }
        assert_eq!(survivor.as_slice(), &[0, 1, 40000, 40001, u16::MAX]);
        assert_eq!(
            survivor.into_iter().collect::<Vec<_>>(),
            vec![0, 1, 40000, 40001, u16::MAX]
        );
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn spill_failure_preserves_original_and_empty_creates_no_file() {
        let dir = scratch();
        let blocker = dir.join("file");
        fs::write(&blocker, b"occupied").unwrap();
        let mut values = Buffer::from(vec![5u8, 255]);
        assert!(values.spill(&blocker).is_err());
        assert!(!values.is_mapped());
        assert_eq!(values.as_slice(), &[5, 255]);
        let mut empty = Buffer::<u8>::default();
        empty.spill(&dir).unwrap();
        assert!(!empty.is_mapped());
        fs::remove_file(blocker).unwrap();
        fs::remove_dir(dir).unwrap();
    }
}
