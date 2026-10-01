//! A detected point source.
//!
//! Lives here rather than beside the detector because two crates need it and
//! neither should depend on the other: `sr-quality` finds stars and measures
//! their shape, `sr-register` uses their positions to place a frame that
//! correlation cannot.

/// One star, at sub-pixel position, with the flux it was measured by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Star {
    /// Sensor coordinates, sub-pixel.
    pub x: f32,
    pub y: f32,
    /// Ranking brightness; the registration detector uses noise-normalized
    /// units. Photometric callers must explicitly remeasure detector flux.
    pub flux: f32,
}
