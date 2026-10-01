//! Native mono windows adapted to the existing forward-deposit merger.
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use sr_core::projection::FrameProjection;
use sr_core::{
    CfaPattern, DefectMask, FrameMetadata, NoiseModel, RawFrame, Rect, SamplePlane, WarpField,
};
use sr_raw::window::{FitsWindowReader, SampleUnits};

pub struct PreparedSource {
    reader: FitsWindowReader,
    pub warp: WarpField,
    pub width: usize,
    pub height: usize,
    pub path: PathBuf,
    pub noise: NoiseModel,
    pub projection: FrameProjection,
}

pub struct SourceTile {
    pub frame: RawFrame,
}

/// Rows per cached band of a source. Bands span the full source width.
pub const BAND_ROWS: usize = 64;

/// Full-width row bands of every source, least recently used first out, within
/// a byte capacity the caller sets per tile. Output tiles advance along a row,
/// and each source window of one tile overlaps the next one's, so a band read
/// once (in one contiguous request) serves the whole tile row. Direct window
/// reads instead issue one request per source row per tile: a network round
/// trip each.
#[derive(Default)]
pub struct BandCache {
    capacity: usize,
    bytes: usize,
    bands: HashMap<(usize, usize), Vec<f32>>,
    order: VecDeque<(usize, usize)>,
    pub hits: u64,
    pub misses: u64,
}

impl BandCache {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Set the capacity and evict down to it.
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        while self.bytes > self.capacity {
            self.evict();
        }
    }

    fn evict(&mut self) {
        if let Some(key) = self.order.pop_front()
            && let Some(band) = self.bands.remove(&key) {
                self.bytes -= band.capacity() * std::mem::size_of::<f32>();
            }
    }
}

impl PreparedSource {
    #[cfg(test)]
    pub fn open(path: &Path, projection: &FrameProjection, noise: NoiseModel) -> Result<Self> {
        Self::open_with_node_budget(path, projection, noise, 1_000_000)
    }

    pub fn open_with_node_budget(
        path: &Path,
        projection: &FrameProjection,
        noise: NoiseModel,
        max_nodes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=1_000_000).contains(&max_nodes),
            "mesh node budget must be in 1..=1000000"
        );
        ensure!(
            noise.alpha.is_finite()
                && noise.alpha >= 0.0
                && noise.beta.is_finite()
                && noise.beta >= 0.0,
            "invalid source noise model"
        );
        let reader = FitsWindowReader::open(path, &sr_raw::ReadOptions::default())
            .with_context(|| format!("opening mosaic source {}", path.display()))?;
        if reader.sample_units() == SampleUnits::PhysicalFloat {
            bail!("{}: calibrated floating-point sources are not supported by this mosaic merge yet; the merger excludes values outside (0, 1)", path.display());
        }
        let (width, height) = reader.dimensions();
        let warp = projection
            .to_warp_with_node_budget(width, height, 0.025, max_nodes)
            .with_context(|| format!("preparing projection for {}", path.display()))?;
        Ok(Self {
            reader,
            warp,
            width,
            height,
            path: path.to_owned(),
            noise,
            projection: projection.clone(),
        })
    }

    /// Actual allocated mesh vector capacity, excluding inline struct metadata.
    pub fn mesh_bytes(&self) -> usize {
        self.warp.local.as_ref().map_or(0, |local| {
            local.u.capacity() * std::mem::size_of::<[f32; 2]>()
                + local.conf.capacity() * std::mem::size_of::<f32>()
        })
    }

    /// Conservatively bound native samples that can deposit in this output tile.
    /// `origin` is in reference pixels; tile coordinates and radius are output
    /// pixels. Uses the merger's pixel-center convention at arbitrary scale.
    /// Plans geometry only, allowing callers to budget allocations before reads.
    pub fn source_rect(
        &self,
        origin: (f32, f32),
        tile: Rect,
        scale: f32,
        radius: f32,
    ) -> Result<Option<Rect>> {
        const COORD_LIMIT: usize = 1 << 24;
        let x1 = tile.x.checked_add(tile.width).context("tile x overflow")?;
        let y1 = tile.y.checked_add(tile.height).context("tile y overflow")?;
        ensure!(
            tile.width > 0 && tile.height > 0 && x1 <= COORD_LIMIT && y1 <= COORD_LIMIT,
            "invalid output tile dimensions"
        );
        ensure!(
            scale.is_finite()
                && scale > 0.0
                && scale <= 64.0
                && radius.is_finite()
                && (0.0..=4096.0).contains(&radius)
                && origin.0.is_finite()
                && origin.1.is_finite()
                && (origin.0 as f64 * scale as f64).abs() <= COORD_LIMIT as f64
                && (origin.1 as f64 * scale as f64).abs() <= COORD_LIMIT as f64,
            "invalid output scale, origin or radius"
        );
        let inv = self
            .warp
            .global
            .inverse()
            .context("singular source affine")?;
        ensure!(
            inv.m.iter().all(|v| v.is_finite()),
            "nonfinite source affine inverse"
        );
        let local = self.warp.max_local() as f64;
        ensure!(local.is_finite(), "nonfinite local displacement");
        let pad_x = local * (inv.m[0] as f64).hypot(inv.m[1] as f64) + 1.0;
        let pad_y = local * (inv.m[3] as f64).hypot(inv.m[4] as f64) + 1.0;
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        let radius = radius.max(0.5) as f64;
        for ox in [tile.x as f64 - radius, x1 as f64 + radius] {
            for oy in [tile.y as f64 - radius, y1 as f64 + radius] {
                let rx = (ox + 0.5) / scale as f64 - 0.5 + origin.0 as f64;
                let ry = (oy + 0.5) / scale as f64 - 0.5 + origin.1 as f64;
                let q = [
                    inv.m[0] as f64 * rx + inv.m[1] as f64 * ry + inv.m[2] as f64,
                    inv.m[3] as f64 * rx + inv.m[4] as f64 * ry + inv.m[5] as f64,
                ];
                ensure!(q.iter().all(|v| v.is_finite()), "source bounds overflow");
                for axis in 0..2 {
                    lo[axis] = lo[axis].min(q[axis]);
                    hi[axis] = hi[axis].max(q[axis]);
                }
            }
        }
        lo[0] -= pad_x;
        lo[1] -= pad_y;
        hi[0] += pad_x;
        hi[1] += pad_y;
        if hi[0] < 0.0
            || hi[1] < 0.0
            || lo[0] >= (self.width as f64)
            || lo[1] >= (self.height as f64)
        {
            return Ok(None);
        }
        // Keep the native 2x2 guide lattice fixed across neighboring crops.
        // Enlarging bounds preserves every sample already selected above.
        let x = (lo[0].floor().max(0.0).min(self.width as f64) as usize) & !1;
        let y = (lo[1].floor().max(0.0).min(self.height as f64) as usize) & !1;
        let end_x = ((hi[0].ceil() + 1.0).max(0.0).min(self.width as f64) as usize)
            .div_ceil(2)
            .saturating_mul(2)
            .min(self.width);
        let end_y = ((hi[1].ceil() + 1.0).max(0.0).min(self.height as f64) as usize)
            .div_ceil(2)
            .saturating_mul(2)
            .min(self.height);
        if end_x <= x || end_y <= y {
            return Ok(None);
        }
        Ok(Some(Rect::new(x, y, end_x - x, end_y - y)))
    }

    /// Bytes of cached bands a window of this source needs.
    pub fn band_bytes(&self, rect: Rect) -> usize {
        let bands = (rect.y + rect.height - 1) / BAND_ROWS - rect.y / BAND_ROWS + 1;
        bands * BAND_ROWS * self.width * std::mem::size_of::<f32>()
    }

    /// The same samples as `read_rect`, assembled from cached full-width bands.
    /// `index` identifies this source in the shared cache.
    pub fn read_rect_cached(&mut self, index: usize, rect: Rect, cache: &mut BandCache) -> Result<SourceTile> {
        ensure!(rect.width > 0 && rect.height > 0
            && rect.x + rect.width <= self.width && rect.y + rect.height <= self.height,
            "cached source window outside {}", self.path.display());
        let mut values = vec![0.; rect.width * rect.height];
        for band_y in (rect.y / BAND_ROWS..=(rect.y + rect.height - 1) / BAND_ROWS).map(|b| b * BAND_ROWS) {
            let key = (index, band_y);
            if cache.bands.contains_key(&key) {
                cache.hits += 1;
                let position = cache.order.iter().position(|k| *k == key).expect("cached band is ordered");
                cache.order.remove(position);
            } else {
                cache.misses += 1;
                let rows = BAND_ROWS.min(self.height - band_y);
                let band = self.reader.read_rect(0, band_y, self.width, rows)?;
                let size = band.capacity() * std::mem::size_of::<f32>();
                while cache.bytes + size > cache.capacity && !cache.order.is_empty() {
                    cache.evict();
                }
                cache.bytes += size;
                cache.bands.insert(key, band);
            }
            cache.order.push_back(key);
            let band = &cache.bands[&key];
            let first = rect.y.max(band_y);
            let last = (rect.y + rect.height).min(band_y + BAND_ROWS);
            for y in first..last {
                let source = (y - band_y) * self.width + rect.x;
                let target = (y - rect.y) * rect.width;
                values[target..target + rect.width].copy_from_slice(&band[source..source + rect.width]);
            }
        }
        // A band larger than the whole capacity is used once and not retained.
        while cache.bytes > cache.capacity && !cache.order.is_empty() {
            cache.evict();
        }
        self.tile(rect, values)
    }

    /// Keep native detector values without cloning the full reference mesh.
    /// Callers rebase `projection` to the crop and their local output origin.
    pub fn read_rect(&mut self, rect: Rect) -> Result<SourceTile> {
        let values = self
            .reader
            .read_rect(rect.x, rect.y, rect.width, rect.height)?;
        self.tile(rect, values)
    }

    fn tile(&self, rect: Rect, values: Vec<f32>) -> Result<SourceTile> {
        let white_level = match self.reader.sample_units() {
            SampleUnits::NormalizedInteger { white_level } => white_level,
            SampleUnits::PhysicalFloat => {
                bail!("floating-point source cannot enter normalized merger")
            }
        };
        let frame = RawFrame {
            width: rect.width,
            height: rect.height,
            samples: SamplePlane::from_normalised(rect.width, rect.height, values),
            cfa: CfaPattern::MONO,
            defects: DefectMask::none(rect.width, rect.height),
            noise: self.noise,
            metadata: FrameMetadata {
                path: self.path.display().to_string(),
                file_name: self
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                full_width: self.width,
                full_height: self.height,
                crop: (rect.x, rect.y, rect.width, rect.height),
                white_level,
                wb_coeffs: [1.0; 3],
                ..FrameMetadata::default()
            },
        };
        Ok(SourceTile { frame })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::DeformationField;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn fixture(float: bool) -> Fixture {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mosaic-source-{}-{}.fits",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut text = String::new();
        for (key, value) in [
            ("SIMPLE", "T"),
            ("BITPIX", if float { "-32" } else { "16" }),
            ("NAXIS", "2"),
            ("NAXIS1", "32"),
            ("NAXIS2", "24"),
            ("BZERO", if float { "0" } else { "32768" }),
        ] {
            text.push_str(&format!("{:<80}", format!("{key:<8}= {value}")));
        }
        text.push_str(&format!("{:<80}", "END"));
        let mut bytes = text.into_bytes();
        bytes.resize(2880, b' ');
        for i in 0..32 * 24 {
            if float {
                bytes.extend_from_slice(&(-0.1f32).to_be_bytes());
            } else {
                bytes.extend_from_slice(&((1001 + i - 32768) as i16).to_be_bytes());
            }
        }
        std::fs::write(&path, bytes).unwrap();
        Fixture(path)
    }
    fn projection() -> FrameProjection {
        FrameProjection {
            center: [0.0; 2],
            normalization_scale: 1.0,
            distortion: [0.0; 3],
            homography: [0.7, -0.1, 12.0, 0.1, 0.7, -8.0, 0.0, 0.0, 1.0],
            output_center: [0.0; 2],
            output_scale: 1.0,
        }
    }
    #[test]
    fn cached_windows_match_direct_reads_under_any_capacity() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mosaic-bands-{}-{}.fits", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let (width, height) = (37, 3 * BAND_ROWS + 11);
        let mut bytes = String::new();
        for (key, value) in [("SIMPLE", "T".to_string()), ("BITPIX", "16".into()), ("NAXIS", "2".into()),
            ("NAXIS1", width.to_string()), ("NAXIS2", height.to_string()), ("BZERO", "32768".into())] {
            bytes.push_str(&format!("{:<80}", format!("{key:<8}= {value}")));
        }
        bytes.push_str(&format!("{:<80}", "END"));
        let mut bytes = bytes.into_bytes();
        bytes.resize(2880, b' ');
        for i in 0..width * height {
            bytes.extend_from_slice(&(((i * 53 % 30000) * 2) as i32 - 32768).to_be_bytes()[2..]);
        }
        std::fs::write(&path, bytes).unwrap();
        let _cleanup = Fixture(path.clone());
        let noise = NoiseModel::nominal(100.0, 65535.0);
        let mut source = PreparedSource::open(&path, &projection(), noise).unwrap();
        let band = BAND_ROWS * width * 4;
        let windows = [Rect::new(0, 0, 37, 5), Rect::new(3, 60, 20, 80), Rect::new(10, 150, 27, 53),
            Rect::new(0, 0, 37, height), Rect::new(36, 200, 1, 3), Rect::new(3, 60, 20, 80)];
        for capacity in [0, band / 2, band, 2 * band, 100 * band] {
            let mut cache = BandCache::default();
            cache.set_capacity(capacity);
            for rect in windows {
                let direct = source.read_rect(rect).unwrap().frame;
                let cached = source.read_rect_cached(0, rect, &mut cache).unwrap().frame;
                for y in 0..rect.height {
                    for x in 0..rect.width {
                        assert_eq!(cached.value(x, y), direct.value(x, y), "capacity {capacity} {rect:?}");
                    }
                }
                assert!(cache.bytes() <= capacity, "capacity {capacity}: {} retained", cache.bytes());
            }
            if capacity >= 4 * band {
                assert!(cache.hits > 0);
            }
        }
    }

    #[test]
    fn native_crop_preserves_samples_metadata_and_reference_local_field() {
        let f = fixture(false);
        let mut source =
            PreparedSource::open(&f.0, &projection(), NoiseModel::nominal(100.0, 65535.0)).unwrap();
        let mut local = DeformationField::zeros((-20.0, -20.0), 10.0, 10, 10);
        for y in 0..10 {
            for x in 0..10 {
                local.u[y * 10 + x] = [x as f32 * 0.02, y as f32 * -0.03];
            }
        }
        source.warp.local = Some(local);
        let rect = Rect::new(5, 7, 13, 9);
        let mesh_bytes = source.mesh_bytes();
        let mesh_pointer = source.warp.local.as_ref().unwrap().u.as_ptr();
        let tile = source.read_rect(rect).unwrap();
        assert_eq!(source.mesh_bytes(), mesh_bytes);
        assert_eq!(mesh_bytes, 10 * 10 * 12);
        assert_eq!(source.warp.local.as_ref().unwrap().u.as_ptr(), mesh_pointer);
        assert_eq!(tile.frame.metadata.crop, (5, 7, 13, 9));
        assert_eq!(
            (
                tile.frame.metadata.full_width,
                tile.frame.metadata.full_height
            ),
            (32, 24)
        );
        for y in 0..9 {
            for x in 0..13 {
                let cropped = source.warp.map((x + rect.x) as f32, (y + rect.y) as f32);
                assert!(cropped.0.is_finite() && cropped.1.is_finite());
                assert_eq!(
                    tile.frame.value(x, y),
                    (1001 + (y + 7) * 32 + x + 5) as f32 * (1.0 / 65535.0)
                );
            }
        }
        for scale in [0.5, 1.0, 2.0] {
            let origin = (10.0, -10.0);
            let output = Rect::new(3, 4, 7, 6);
            let bounds = source
                .source_rect(origin, output, scale, 2.0)
                .unwrap()
                .unwrap();
            for y in 0..24 {
                for x in 0..32 {
                    let q = source.warp.map(x as f32, y as f32);
                    let out = (
                        (q.0 + 0.5) * scale - 0.5 - origin.0 * scale,
                        (q.1 + 0.5) * scale - 0.5 - origin.1 * scale,
                    );
                    if out.0 >= output.x as f32 - 2.0
                        && out.0 <= output.x1() as f32 + 2.0
                        && out.1 >= output.y as f32 - 2.0
                        && out.1 <= output.y1() as f32 + 2.0
                    {
                        assert!(
                            x >= bounds.x && x < bounds.x1() && y >= bounds.y && y < bounds.y1()
                        );
                    }
                }
            }
        }
        assert!(source
            .source_rect((0.0, 0.0), Rect::new(10000, 10000, 10, 10), 1.0, 2.0)
            .unwrap()
            .is_none());
        assert!(source
            .source_rect((0.0, 0.0), Rect::new(0, 0, 10, 10), 0.0, 2.0)
            .is_err());
        assert!(source
            .source_rect((0.0, 0.0), Rect::new(usize::MAX, 0, 10, 10), 1.0, 2.0)
            .is_err());
    }
    #[test]
    fn calibrated_float_sources_are_explicitly_refused() {
        let f = fixture(true);
        let result = PreparedSource::open(&f.0, &projection(), NoiseModel::nominal(100.0, 65535.0));
        assert!(result.err().unwrap().to_string().contains("floating-point"));
    }
}
