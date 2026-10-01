//! Sensor sample storage.
//!
//! A burst is held in full for the whole run, so how one sample is stored is
//! multiplied by fifty million sites and by however many frames the operator
//! pointed at the program. Storing the normalised value as `f32` costs four
//! bytes to hold a number that arrived as a 14- or 16-bit integer and has not
//! gained any information since.
//!
//! So the decoded integer is kept as it came off the sensor and normalised on
//! read. This is exact: a 16-bit integer divided by a constant is representable
//! in `f32` without loss, so nothing is given up for the halving. What it does
//! give up is the ability to hold values that are not integers, which is why
//! [`SampleData`] keeps a floating-point variant — calibrated astronomical
//! frames are floating point and routinely negative after dark subtraction, and
//! foreclosing them to save memory on Bayer files would be a poor trade.
//!
//! Large runs may spill either representation to exact immutable scratch
//! storage. The mapping changes ownership and residency, never sample values
//! or normalisation; the operating system manages its resident pages.

use serde::{Deserialize, Serialize};
use crate::buffer::Buffer;
use std::{io, path::Path, sync::Arc};

/// How the samples of one frame are held.
#[derive(Clone, Debug, PartialEq)]
pub enum SampleData {
    /// Raw integer sensor levels, as decoded. The common case.
    U16(Vec<u16>),
    /// Values that are not integers: calibrated or synthetic frames.
    F32(Vec<f32>),
    /// Exact integer samples held in immutable scratch storage.
    MappedU16(Buffer<u16>),
    /// Exact floating-point samples held in immutable scratch storage.
    MappedF32(Buffer<f32>),
}

impl SampleData {
    pub fn len(&self) -> usize {
        match self {
            SampleData::U16(v) => v.len(),
            SampleData::F32(v) => v.len(),
            SampleData::MappedU16(v) => v.len(),
            SampleData::MappedF32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes of storage.
    pub fn bytes(&self) -> usize {
        match self {
            SampleData::U16(v) => v.len() * 2,
            SampleData::F32(v) => v.len() * 4,
            SampleData::MappedU16(v) => v.len() * 2,
            SampleData::MappedF32(v) => v.len() * 4,
        }
    }
}

/// Per-CFA-cell black and white levels, and the normalisation they imply.
///
/// Normalisation is `(raw - black) / (white - black)` with both taken for the
/// sample's own position in the 2x2 mosaic cell, because Nikon and others do
/// record per-channel black levels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Levels {
    pub black: [f32; 4],
    pub white: [f32; 4],
    /// Reciprocal of `white - black`, so that normalisation is a subtract and a
    /// multiply rather than a divide.
    inv_range: [f32; 4],
}

impl Levels {
    pub fn new(black: [f32; 4], white: [f32; 4]) -> Self {
        let mut inv_range = [1.0f32; 4];
        for i in 0..4 {
            inv_range[i] = 1.0 / (white[i] - black[i]).max(1.0);
        }
        Self { black, white, inv_range }
    }

    /// Identity, for sources that are already normalised.
    pub fn unit() -> Self {
        Self::new([0.0; 4], [1.0; 4])
    }

    #[inline]
    pub fn cell(x: usize, y: usize) -> usize {
        (y & 1) * 2 + (x & 1)
    }

    /// Normalise a raw level for a given cell.
    ///
    /// Written as the definition says — subtract black, then scale — rather
    /// than folded into a single multiply-add. Both cost the same, and doing
    /// the subtraction first is exact for integer input, so the result matches
    /// what a direct `(raw - black) / range` would have produced.
    #[inline]
    pub fn normalise(&self, raw: f32, cell: usize) -> f32 {
        (raw - self.black[cell]) * self.inv_range[cell]
    }
}

/// One frame's samples, plus what is needed to interpret them.
#[derive(Clone, Debug)]
pub struct SamplePlane {
    pub width: usize,
    pub height: usize,
    pub data: SampleData,
    pub levels: Levels,
}

impl SamplePlane {
    pub fn from_u16(width: usize, height: usize, data: Vec<u16>, levels: Levels) -> Self {
        assert_eq!(data.len(), width * height, "sample count mismatch");
        Self { width, height, data: SampleData::U16(data), levels }
    }

    /// Build from already-normalised floating-point values.
    pub fn from_normalised(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height, "sample count mismatch");
        Self { width, height, data: SampleData::F32(data), levels: Levels::unit() }
    }

    pub fn len(&self) -> usize {
        self.width * self.height
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        self.data.bytes()
    }

    /// Spill exact detector values to process-private scratch. Normalisation
    /// levels and dimensions are unchanged; failure preserves the samples.
    pub fn spill(&mut self, dir: &Path) -> io::Result<()> {
        // Move the Vec instead of cloning a full sensor image. Restore it on
        // failure so an unavailable scratch volume never destroys input data.
        let original = std::mem::replace(&mut self.data, SampleData::U16(Vec::new()));
        let (replacement, result) = match original {
            SampleData::U16(values) => {
                let mut buffer = Buffer::from(values);
                let result = buffer.spill(dir);
                let data = if result.is_ok() {
                    SampleData::MappedU16(buffer)
                } else {
                    SampleData::U16(buffer.into_vec())
                };
                (data, result)
            }
            SampleData::F32(values) => {
                let mut buffer = Buffer::from(values);
                let result = buffer.spill(dir);
                let data = if result.is_ok() {
                    SampleData::MappedF32(buffer)
                } else {
                    SampleData::F32(buffer.into_vec())
                };
                (data, result)
            }
            mapped => (mapped, Ok(())),
        };
        self.data = replacement;
        result
    }

    /// Normalised value at a linear index whose cell parity is already known.
    #[inline]
    pub fn value_in_cell(&self, i: usize, cell: usize) -> f32 {
        match &self.data {
            SampleData::U16(v) => self.levels.normalise(v[i] as f32, cell),
            SampleData::F32(v) => self.levels.normalise(v[i], cell),
            SampleData::MappedU16(v) => self.levels.normalise(v[i] as f32, cell),
            SampleData::MappedF32(v) => self.levels.normalise(v[i], cell),
        }
    }

    /// Normalised value at `(x, y)`.
    #[inline]
    pub fn value(&self, x: usize, y: usize) -> f32 {
        self.value_in_cell(y * self.width + x, Levels::cell(x, y))
    }

    /// Normalised value at a linear index.
    #[inline]
    pub fn value_at(&self, i: usize) -> f32 {
        let x = i % self.width;
        let y = i / self.width;
        self.value_in_cell(i, Levels::cell(x, y))
    }
}

/// Sites that carry no usable measurement for a reason the value alone cannot
/// reveal: a known-dead sensor site, or a decode failure.
///
/// Saturation and black clipping are deliberately *not* stored. They are exactly
/// "the normalised value is at or beyond the ends of the range", so keeping a
/// byte per pixel to record them would be storing a comparison. For an ordinary
/// RAW file this mask is therefore empty, and empty costs nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DefectMask {
    pub width: usize,
    pub height: usize,
    bits: Option<Arc<Vec<u64>>>,
}

impl DefectMask {
    pub fn none(width: usize, height: usize) -> Self {
        Self { width, height, bits: None }
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_none()
    }

    pub fn bytes(&self) -> usize {
        self.bits.as_ref().map(|b| b.len() * 8).unwrap_or(0)
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        match &self.bits {
            None => false,
            Some(b) => (b[i >> 6] >> (i & 63)) & 1 != 0,
        }
    }

    pub fn set(&mut self, i: usize) {
        let n = self.width * self.height;
        let bits = self
            .bits
            .get_or_insert_with(|| Arc::new(vec![0u64; n.div_ceil(64)]));
        let bits = Arc::make_mut(bits);
        bits[i >> 6] |= 1u64 << (i & 63);
    }

    pub fn count(&self) -> usize {
        match &self.bits {
            None => 0,
            Some(b) => b.iter().map(|w| w.count_ones() as usize).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalisation_matches_the_definition() {
        let levels = Levels::new([1008.0; 4], [15520.0; 4]);
        let p = SamplePlane::from_u16(2, 2, vec![1008, 15520, 8264, 933], levels);
        assert!((p.value(0, 0) - 0.0).abs() < 1e-6, "black maps to zero");
        assert!((p.value(1, 0) - 1.0).abs() < 1e-6, "white maps to one");
        // Midway between black and white.
        assert!((p.value(0, 1) - 0.5).abs() < 1e-4, "{}", p.value(0, 1));
        // Below black stays below zero rather than being clamped away.
        assert!(p.value(1, 1) < 0.0, "{}", p.value(1, 1));
    }

    #[test]
    fn per_cell_black_levels_are_respected() {
        let levels = Levels::new([100.0, 200.0, 300.0, 400.0], [1100.0, 1200.0, 1300.0, 1400.0]);
        let p = SamplePlane::from_u16(2, 2, vec![600, 700, 800, 900], levels);
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert!(
                (p.value(x, y) - 0.5).abs() < 1e-5,
                "cell ({x},{y}) gave {}",
                p.value(x, y)
            );
        }
    }

    #[test]
    fn sixteen_bit_values_survive_the_round_trip_exactly() {
        // The whole point of storing integers: normalising must not lose a
        // level. Two adjacent 16-bit codes must stay distinguishable.
        let levels = Levels::new([0.0; 4], [65535.0; 4]);
        let p = SamplePlane::from_u16(2, 1, vec![40000, 40001], levels);
        let (a, b) = (p.value(0, 0), p.value(1, 0));
        assert!(a != b, "adjacent codes collapsed to {a}");
        assert!((a * 65535.0 - 40000.0).abs() < 0.01, "{a}");
        assert!((b * 65535.0 - 40001.0).abs() < 0.01, "{b}");
    }

    #[test]
    fn float_samples_are_kept_as_they_are() {
        // Calibrated frames go negative after dark subtraction, and that has to
        // survive.
        let p = SamplePlane::from_normalised(2, 1, vec![-0.02, 0.75]);
        assert!((p.value(0, 0) + 0.02).abs() < 1e-6);
        assert!((p.value(1, 0) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn spilling_samples_preserves_normalisation_and_negative_values() {
        let dir = std::env::temp_dir().join(format!("smokstak-samples-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let levels = Levels::new([100.0, 200.0, 300.0, 400.0], [1100.0, 1200.0, 1300.0, 1400.0]);
        let mut integers = SamplePlane::from_u16(2, 2, vec![0, 65535, 301, 499], levels);
        let expected: Vec<_> = (0..4).map(|i| integers.value_at(i).to_bits()).collect();
        integers.spill(&dir).unwrap();
        assert!(matches!(integers.data, SampleData::MappedU16(_)));
        assert_eq!(integers.levels, levels);
        assert_eq!((0..4).map(|i| integers.value_at(i).to_bits()).collect::<Vec<_>>(), expected);
        assert_eq!(integers.bytes(), 8);
        let mut floats = SamplePlane::from_normalised(2, 1, vec![-0.02, 0.75]);
        let before: Vec<_> = (0..2).map(|i| floats.value_at(i).to_bits()).collect();
        floats.spill(&dir).unwrap();
        floats.spill(&dir).unwrap();
        assert_eq!((0..2).map(|i| floats.value_at(i).to_bits()).collect::<Vec<_>>(), before);
        drop(integers);
        drop(floats);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn integer_storage_is_half_the_size() {
        let n = 1000;
        let ints = SamplePlane::from_u16(n, 1, vec![0; n], Levels::unit());
        let floats = SamplePlane::from_normalised(n, 1, vec![0.0; n]);
        assert_eq!(ints.bytes() * 2, floats.bytes());
    }

    #[test]
    fn an_empty_defect_mask_costs_nothing() {
        let m = DefectMask::none(4096, 4096);
        assert_eq!(m.bytes(), 0);
        assert!(!m.get(12345));
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn defects_are_recorded_and_read_back() {
        let mut m = DefectMask::none(64, 64);
        for &i in &[0usize, 63, 64, 4095] {
            m.set(i);
        }
        assert_eq!(m.count(), 4);
        for &i in &[0usize, 63, 64, 4095] {
            assert!(m.get(i), "bit {i}");
        }
        assert!(!m.get(1));
        assert!(!m.get(65));
        // One bit per pixel, not one byte.
        assert!(m.bytes() <= 64 * 64 / 8 + 8);
    }

    #[test]
    fn cloned_defect_masks_share_storage_until_mutated() {
        let mut original = DefectMask::none(64, 64);
        original.set(63);
        let mut other = original.clone();
        assert!(Arc::ptr_eq(original.bits.as_ref().unwrap(), other.bits.as_ref().unwrap()));
        other.set(64);
        assert!(!Arc::ptr_eq(original.bits.as_ref().unwrap(), other.bits.as_ref().unwrap()));
        assert!(original.get(63));
        assert!(!original.get(64));
        assert!(other.get(63));
        assert!(other.get(64));
        assert_eq!(original.count(), 1);
        assert_eq!(other.count(), 2);
        original.set(65);
        assert!(!other.get(65));
    }
}
