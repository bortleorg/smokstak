//! Per-sample validity flags.
//!
//! Bad measurements are flagged, never silently replaced: the reconstruction
//! needs to know which samples are real, and diagnostics need to show it.

use crate::plane::Plane;

pub type MaskPlane = Plane<u8>;

pub mod flags {
    /// Sample at or above the saturation level.
    pub const SATURATED: u8 = 1 << 0;
    /// Sample clipped at or below black.
    pub const BLACK_CLIPPED: u8 = 1 << 1;
    /// Known-defective sensor site.
    pub const HOT_OR_DEAD: u8 = 1 << 2;
    /// Outside the usable/active area.
    pub const OUT_OF_AREA: u8 = 1 << 3;
    /// Decoder reported a problem for this site.
    pub const DECODE_ERROR: u8 = 1 << 4;

    /// Any flag that makes a sample unusable for reconstruction.
    pub const UNUSABLE: u8 = SATURATED | BLACK_CLIPPED | HOT_OR_DEAD | OUT_OF_AREA | DECODE_ERROR;
}

#[inline]
pub fn is_usable(m: u8) -> bool {
    m & flags::UNUSABLE == 0
}
