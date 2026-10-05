//! Shared data model for the smokstak multi-frame super-resolution engine.
//!
//! The one architectural rule this crate exists to enforce: detector samples
//! stay detector samples. Geometry, quality, noise and confidence are carried
//! *alongside* the mosaic rather than baked into resampled RGB intermediates.

pub mod buffer;
pub mod cfa;
pub mod config;
pub mod frame;
pub mod geometry;
pub mod mask;
pub mod math;
pub mod plane;
pub mod product;
pub mod projection;
pub mod samples;
pub mod star;
pub mod wcs;

pub use cfa::{CfaColor, CfaPattern};
pub use config::{
    Backend, KernelConfig, LocalWarpMode, LuckyMode, PostProcess, ReconstructionConfig,
    RegistrationConfig, RobustnessConfig, WarpConfig,
};
pub use frame::{
    Burst, FrameMetadata, FrameQuality, GuideImage, LocalQualityMap, NoiseModel, NoiseSource,
    RawFrame,
};
pub use geometry::{
    DeformationField, GlobalTransform, RadialChroma, Rect, TransformModel, WarpField,
};
pub use mask::{MaskPlane, flags, is_usable};
pub use plane::Plane;
pub use product::{ProductStats, ReconstructionProduct, SamplingCoverage};
pub use samples::{DefectMask, Levels, SampleData, SamplePlane};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SrError {
    #[error("input error: {0}")]
    Input(String),
    #[error("burst is inconsistent: {0}")]
    Inconsistent(String),
    #[error("registration failed: {0}")]
    Registration(String),
    #[error("reconstruction failed: {0}")]
    Reconstruction(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, SrError>;
