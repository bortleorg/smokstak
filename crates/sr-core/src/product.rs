//! Reconstruction outputs and the evidence that comes with them.

use serde::{Deserialize, Serialize};

use crate::plane::Plane;

/// The reconstructed image plus every map a user needs to judge whether the
/// reconstruction is actually supported by measurements.
#[derive(Clone, Debug)]
pub struct ReconstructionProduct {
    pub width: usize,
    pub height: usize,
    /// Linear camera-RGB, normalised, before colour conversion.
    ///
    /// A monochrome reconstruction fills only `rgb[0]`; the other two are
    /// allocated at zero size. Read [`channels`](Self::channels) before
    /// indexing.
    pub rgb: [Plane<f32>; 3],
    /// One for a monochrome sensor, three for a mosaic.
    pub channels: usize,
    /// Accumulated kernel weight per channel.
    pub weight: [Plane<f32>; 3],
    /// Number of contributing samples per channel.
    pub count: [Plane<f32>; 3],
    /// Effective frame count: `(sum w)^2 / sum w^2` over per-frame weight
    /// totals, i.e. how many frames the merge really behaved like.
    ///
    /// Computed on a decimated grid (see `effective_frames_cell`): it needs a
    /// per-frame accumulator, and at full output resolution that would cost as
    /// much as the merge itself for a map nobody reads at pixel level.
    pub effective_frames: Plane<f32>,
    /// Output pixels per cell of `effective_frames`.
    pub effective_frames_cell: usize,
    /// Samples suppressed by the robustness model.
    pub rejected: Plane<f32>,
    /// Channel-pixels without support before interpolation, summed over channels.
    pub holes_filled: usize,
    pub stats: ProductStats,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProductStats {
    pub total_samples: u64,
    pub accumulated_samples: u64,
    pub rejected_samples: u64,
    pub masked_samples: u64,
    /// Samples at or above full scale, deposited at the ceiling rather than
    /// discarded. Counted separately because they are the one kind of
    /// measurement whose value is a bound and not a reading.
    pub saturated_samples: u64,
    pub out_of_bounds_samples: u64,
    pub min_effective_frames: f32,
    pub mean_effective_frames: f32,
    pub unsupported_pixels: u64,
}

/// Measured sub-pixel sampling diversity of the burst. This is what tells the
/// user whether 2x is real or wishful.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingCoverage {
    pub scale: f32,
    /// Histogram of sub-pixel phase over the output grid, `bins x bins`.
    pub bins: usize,
    pub phase_histogram: Vec<f64>,
    /// Per-channel occupancy of the phase histogram, `[0, 1]`.
    pub channel_occupancy: [f32; 3],
    /// How many of `channel_occupancy` mean anything: one for a monochrome
    /// sensor, three for a mosaic.
    pub channels: usize,
    /// Uniformity of the phase histogram, `[0, 1]`; 1.0 is ideal diversity.
    pub uniformity: f32,
    pub horizontal_diversity: f32,
    pub vertical_diversity: f32,
    /// Scale actually supported by the measured diversity.
    pub recommended_scale: f32,
    pub verdict: String,
}

impl SamplingCoverage {
    pub fn describe(&self) -> String {
        let grade = |v: f32| {
            if v > 0.8 {
                "excellent"
            } else if v > 0.6 {
                "good"
            } else if v > 0.4 {
                "moderate"
            } else if v > 0.2 {
                "poor"
            } else {
                "negligible"
            }
        };
        // A monochrome sensor has one channel, and reporting the two it does
        // not have as "negligible" reads as a fault rather than as a sensor.
        let per_channel = if self.channels == 1 {
            format!("  coverage: {}\n", grade(self.channel_occupancy[0]))
        } else {
            format!(
                "  R coverage: {}\n  G coverage: {}\n  B coverage: {}\n",
                grade(self.channel_occupancy[0]),
                grade(self.channel_occupancy[1]),
                grade(self.channel_occupancy[2]),
            )
        };
        format!(
            "Requested scale: {:.2}x\n\
             Estimated sampling support:\n  \
             horizontal: {}\n  vertical: {}\n\
             {}\
             Recommended scale: {:.2}x\n{}",
            self.scale,
            grade(self.horizontal_diversity),
            grade(self.vertical_diversity),
            per_channel,
            self.recommended_scale,
            self.verdict
        )
    }
}
