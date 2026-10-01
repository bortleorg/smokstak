//! Run configuration. Everything here lands verbatim in `run.json`.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    /// Baseline B: aligned robust mean of naively demosaiced frames.
    RgbMeanBaseline,
    /// Baseline C: CFA-aware isotropic drizzle onto the high-resolution grid.
    CfaDrizzle,
    /// Wronski-style structure-aware anisotropic CFA merge.
    HandheldBurstSr,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::RgbMeanBaseline => "rgb-mean-baseline",
            Backend::CfaDrizzle => "cfa-drizzle",
            Backend::HandheldBurstSr => "burst-sr",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PostProcess {
    /// No subjective rendering at all. The scientific default.
    None,
    /// Conservative edge-aware sharpening.
    Mild,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LocalWarpMode {
    Off,
    On,
    /// Enable only when the burst shows evidence of non-global deformation.
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LuckyMode {
    Off,
    /// Keep the best `f` fraction of frames per region.
    Fraction(f32),
    Auto,
}

/// Tunables for the structure-aware kernel, following the naming in
/// Wronski et al. 2019 so the values stay comparable with the literature.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Fixed mono sampling variance in sensor pixels squared. None retains
    /// the adaptive CFA-style field. Uniform sampling preserves faint texture
    /// without giving each reference-noise fluctuation a different smoothing.
    #[serde(default = "default_mono_kernel_variance")]
    pub mono_kernel_variance: Option<f32>,
    /// Kernel variance in flat regions (denoising-dominant).
    pub k_denoise: f32,
    /// Kernel variance in detailed regions (detail-preserving).
    pub k_detail: f32,
    /// Maximum elongation along an edge.
    pub k_stretch: f32,
    /// Maximum narrowing across an edge.
    pub k_shrink: f32,
    /// Structure-to-noise ratio at which a region counts as fully detailed.
    ///
    /// Wronski et al. use an absolute gradient threshold here. An absolute
    /// threshold does not survive contact with a second scene: a printed test
    /// chart has edges of 0.6 contrast and a sunlit roof has granule texture of
    /// 0.05, and calling the second one "flat" blurs away exactly the detail
    /// the merge exists to recover. What separates detail from noise is how far
    /// the local gradient stands above the gradient noise alone would produce,
    /// and the noise model already tells us that.
    pub detail_snr: f32,
    /// Offset of the detail/denoise transition.
    pub d_th: f32,
    /// Put each output pixel's estimate at its own centre.
    ///
    /// A weighted mean of scattered samples is the value where those samples
    /// averaged to, not the value at the pixel, and the difference is the local
    /// gradient times however far off centre the dither happened to leave them.
    /// It varies pixel to pixel, so it behaves like noise rather than like a
    /// blur, and it grows as the samples thin out.
    pub debias_deposit: bool,
    /// Noise-guarded raw quadratic fit. None selects it for monochrome burst
    /// reconstruction; Some(false) explicitly disables it. CFA remains opt-in.
    #[serde(default)]
    pub fit_curvature: Option<bool>,
    /// Bounded shared centroid correction at clipped footprints. None selects
    /// it for CFA burst reconstruction; explicit false keeps the common mean.
    #[serde(default)]
    pub fit_clipped: Option<bool>,
    /// Shrink the detail kernel as the burst grows.
    ///
    /// The detail kernel is a hedge against sparse sampling: it has to be wide
    /// enough that an output pixel finds samples of all three colours under it.
    /// The published value was chosen for a handheld burst of about eight
    /// frames, where that is most of a pixel. A hundred dithered frames put
    /// twenty-odd sub-pixel phases per axis under every output pixel, and the
    /// hedge is then paid for in resolution and buys nothing.
    pub scale_detail_with_frames: bool,
    /// Shrink the flat-region kernel as the burst grows.
    ///
    /// Widening the kernel in flat regions buys noise reduction at the cost of
    /// resolution. That trade is worth making for the ten-frame handheld burst
    /// the published parameters were chosen for, and not for two hundred
    /// frames, where temporal averaging has already done the job an order of
    /// magnitude better.
    pub scale_denoise_with_frames: bool,
    /// Kernel support radius in output pixels.
    pub radius: f32,
    /// Kernel variance multiplier for red and blue.
    ///
    /// A Bayer mosaic samples red and blue on a lattice of twice green's pitch,
    /// so in principle their kernels should be wider by the square of that
    /// ratio. In practice the best value is a measurement, not a derivation:
    /// see the `chroma` row of `smokstak selftest`. Set to 1.0 to treat all
    /// three channels identically.
    pub chroma_variance: f32,
}

fn default_mono_kernel_variance() -> Option<f32> {
    Some(0.125)
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            mono_kernel_variance: default_mono_kernel_variance(),
            k_denoise: 3.0,
            k_detail: 0.25,
            k_stretch: 4.0,
            k_shrink: 2.0,
            detail_snr: 4.0,
            d_th: 0.121,
            debias_deposit: true,
            fit_curvature: None,
            fit_clipped: None,
            scale_detail_with_frames: true,
            scale_denoise_with_frames: true,
            radius: 2.0,
            chroma_variance: 1.0,
        }
    }
}

/// Robustness model parameters (motion / occlusion rejection).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RobustnessConfig {
    pub enabled: bool,
    /// Scale of the robustness falloff.
    pub s1: f32,
    /// Scale used where local structure is strong.
    pub s2: f32,
    /// Threshold subtracted before clamping to `[0, 1]`.
    pub t: f32,
    /// Floor on the local noise estimate, in normalised units.
    pub sigma_floor: f32,
    /// Below this robustness a sample is counted as rejected in diagnostics.
    pub reject_below: f32,
}

impl Default for RobustnessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            s1: 2.0,
            s2: 12.0,
            t: 0.12,
            sigma_floor: 0.002,
            reject_below: 0.25,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RegistrationConfig {
    /// Pyramid levels used for the coarse-to-fine search.
    pub pyramid_levels: usize,
    /// Patch size, in proxy pixels, for global displacement probes.
    pub global_patch: usize,
    /// Number of probes across the wider image dimension.
    pub global_probes: usize,
    /// Reject probes whose correlation peak is weaker than this.
    pub min_peak_ratio: f32,
    /// Robust fitting iterations.
    pub irls_iters: usize,
    /// Residual (proxy px) below which a simpler model is preferred.
    pub model_selection_tolerance: f32,
}

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            pyramid_levels: 4,
            global_patch: 128,
            global_probes: 24,
            min_peak_ratio: 1.15,
            irls_iters: 12,
            model_selection_tolerance: 0.02,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WarpConfig {
    /// Patch size in proxy pixels at the finest level.
    pub patch: usize,
    /// Node spacing in proxy pixels at the finest level.
    pub spacing: usize,
    /// Coarse-to-fine levels for the local search.
    pub levels: usize,
    /// Maximum accepted local displacement, proxy pixels.
    pub max_displacement: f32,
    /// Smoothing passes applied to the node field.
    pub smooth_iters: usize,
    /// Per-neighbour pull in the field regulariser.
    ///
    /// A measured node keeps `conf / (conf + 4 * lambda)` of its own value, so
    /// this trades smoothness against fidelity directly. Values near 0.6 leave
    /// a confident node holding under a third of what was measured, which
    /// flattens a real deformation field into its own average.
    pub smooth_lambda: f32,
    /// Reject nodes below this correlation confidence.
    pub min_confidence: f32,
    /// Fractional reduction in alignment residual a fitted field must deliver
    /// before it is accepted.
    ///
    /// Estimating a deformation field always produces one. The question is
    /// whether it describes the scene or the correlator's own noise, and a
    /// field fitted to noise is worse than none: it aligns grain between
    /// frames, which reads as recovered detail. So the field is measured
    /// against the alignment it was supposed to improve, and discarded if it
    /// does not.
    pub min_improvement: f32,
}

impl Default for WarpConfig {
    fn default() -> Self {
        Self {
            patch: 64,
            spacing: 32,
            levels: 3,
            max_displacement: 6.0,
            smooth_iters: 3,
            smooth_lambda: 0.15,
            min_confidence: 0.15,
            min_improvement: 0.15,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReconstructionConfig {
    pub scale: f32,
    pub backend: Backend,
    pub kernel: KernelConfig,
    /// Normalise each frame's brightness onto the reference before comparing
    /// and merging. See `sr_quality::photometry`.
    pub photometric_match: bool,
    /// Fit each frame an additive field across the frame, on top of the gain
    /// and the pedestal.
    ///
    /// On by default. This fits differences between aligned exposures,
    /// separately from `flatten_background`, which models the final image.
    pub sky_field: bool,
    /// Find the sensor's hot and dead sites from the burst and mask them.
    /// Needs a burst that moved. See `sr_noise::defects`.
    pub detect_defects: bool,
    /// Fit vignetting and a sky gradient to the result and remove them.
    /// Off by default: it is a correction that can compete with the subject.
    /// See `sr_reconstruct::background`.
    pub flatten_background: bool,
    pub robustness: RobustnessConfig,
    pub registration: RegistrationConfig,
    pub warp: WarpConfig,
    pub local_warp: LocalWarpMode,
    pub lucky: LuckyMode,
    pub postprocess: PostProcess,
    /// Output tile edge in output pixels. Bounds peak memory.
    pub tile: usize,
    /// Optional region of interest in reference sensor coordinates
    /// `(x, y, w, h)`, for fast iteration on a crop.
    pub roi: Option<(usize, usize, usize, usize)>,
    /// Explicit reference frame index, otherwise chosen automatically.
    pub reference: Option<usize>,
    /// Deterministic seed for any randomised step.
    pub seed: u64,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        Self {
            scale: 2.0,
            backend: Backend::HandheldBurstSr,
            kernel: KernelConfig::default(),
            photometric_match: true,
            sky_field: true,
            detect_defects: true,
            flatten_background: false,
            robustness: RobustnessConfig::default(),
            registration: RegistrationConfig::default(),
            warp: WarpConfig::default(),
            local_warp: LocalWarpMode::Auto,
            lucky: LuckyMode::Off,
            postprocess: PostProcess::None,
            tile: 512,
            roi: None,
            reference: None,
            seed: 0x5EED_5EED,
        }
    }
}
