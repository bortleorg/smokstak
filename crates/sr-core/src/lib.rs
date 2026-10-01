//! Shared data model for the smokstak multi-frame super-resolution engine.
//!
//! The one architectural rule this crate exists to enforce: detector samples
//! stay detector samples. Geometry, quality, noise and confidence are carried
//! *alongside* the mosaic rather than baked into resampled RGB intermediates.

pub mod cfa;
pub mod buffer;
pub mod config;
pub mod frame;
pub mod geometry;
pub mod projection;
pub mod mask;
pub mod math;
pub mod plane;
pub mod samples;
pub mod star;
pub mod product;
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
pub use mask::{flags, is_usable, MaskPlane};
pub use plane::Plane;
pub use samples::{DefectMask, Levels, SampleData, SamplePlane};
pub use product::{ProductStats, ReconstructionProduct, SamplingCoverage};

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
