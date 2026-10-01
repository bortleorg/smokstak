//! Colour-filter-array geometry.
//!
//! The pipeline never demosaics early, so every sensor sample must be able to
//! answer "which colour am I?" cheaply. That is all this module provides.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CfaColor {
    R = 0,
    G = 1,
    B = 2,
}

impl CfaColor {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            CfaColor::R => "R",
            CfaColor::G => "G",
            CfaColor::B => "B",
        }
    }
}

/// A 2x2 Bayer mosaic. `codes` is in raster order: (0,0), (1,0), (0,1), (1,1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CfaPattern {
    pub codes: [CfaColor; 4],
}

impl CfaPattern {
    pub const RGGB: CfaPattern = CfaPattern {
        codes: [CfaColor::R, CfaColor::G, CfaColor::G, CfaColor::B],
    };
    pub const BGGR: CfaPattern = CfaPattern {
        codes: [CfaColor::B, CfaColor::G, CfaColor::G, CfaColor::R],
    };
    pub const GRBG: CfaPattern = CfaPattern {
        codes: [CfaColor::G, CfaColor::R, CfaColor::B, CfaColor::G],
    };
    pub const GBRG: CfaPattern = CfaPattern {
        codes: [CfaColor::G, CfaColor::B, CfaColor::R, CfaColor::G],
    };

    /// A sensor with no colour filter array at all.
    ///
    /// Represented as a pattern whose four positions carry the same code rather
    /// than as a separate type, so that every piece of geometry that reasons
    /// about 2x2 cells — the crop parity, the guide image, the sampling phase —
    /// keeps working unchanged. What *does* change is the number of channels,
    /// and that is asked for explicitly through [`is_mono`](Self::is_mono).
    ///
    /// Green, because a monochrome sensor's response most resembles it and
    /// because it is the channel every other stage treats as luminance.
    pub const MONO: CfaPattern = CfaPattern {
        codes: [CfaColor::G, CfaColor::G, CfaColor::G, CfaColor::G],
    };

    /// Whether every position carries the same colour, which no real Bayer
    /// mosaic does.
    #[inline]
    pub fn is_mono(&self) -> bool {
        self.codes.iter().all(|&c| c == self.codes[0])
    }

    pub fn from_name(s: &str) -> Option<CfaPattern> {
        match s.to_ascii_uppercase().as_str() {
            "RGGB" => Some(Self::RGGB),
            "BGGR" => Some(Self::BGGR),
            "GRBG" => Some(Self::GRBG),
            "GBRG" => Some(Self::GBRG),
            "MONO" | "NONE" | "" => Some(Self::MONO),
            _ => None,
        }
    }

    pub fn name(&self) -> String {
        if self.is_mono() {
            return "Mono".to_string();
        }
        self.codes.iter().map(|c| c.name()).collect()
    }

    #[inline]
    pub fn color_at(&self, x: usize, y: usize) -> CfaColor {
        self.codes[(y & 1) * 2 + (x & 1)]
    }

    /// Offset of the first pixel of `color` within the 2x2 cell, if unique-ish.
    /// For green the *first* of the two greens is returned.
    pub fn offset_of(&self, color: CfaColor) -> Option<(usize, usize)> {
        self.codes
            .iter()
            .position(|&c| c == color)
            .map(|i| (i % 2, i / 2))
    }

    /// The two green offsets within the 2x2 cell.
    pub fn green_offsets(&self) -> [(usize, usize); 2] {
        let mut out = [(0usize, 0usize); 2];
        let mut n = 0;
        for (i, &c) in self.codes.iter().enumerate() {
            if c == CfaColor::G && n < 2 {
                out[n] = (i % 2, i / 2);
                n += 1;
            }
        }
        out
    }

    /// Shift the pattern origin by a crop offset, so that cropping the sensor
    /// array does not silently relabel colours.
    pub fn shifted(&self, dx: usize, dy: usize) -> CfaPattern {
        let mut codes = [CfaColor::G; 4];
        for y in 0..2 {
            for x in 0..2 {
                codes[y * 2 + x] = self.color_at(x + dx, y + dy);
            }
        }
        CfaPattern { codes }
    }
}
