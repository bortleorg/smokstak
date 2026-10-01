//! `smokstak` — command line front end.

mod analyze;
mod cache;
mod catalog;
mod frame_review;
mod gui;
mod mosaic;
mod mosaic_prepare;
mod mosaic_prepare_geometry;
mod mosaic_prepare_photometry;
mod mosaic_source;
mod pipeline;
mod project;
mod selftest;

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

use sr_core::config::{Backend, LocalWarpMode, LuckyMode, PostProcess, ReconstructionConfig};

#[derive(Parser, Debug)]
#[command(
    name = "smokstak",
    version,
    about = "Multi-frame RAW super-resolution for camera bursts",
    long_about = "Reconstructs one high-resolution image from a burst of RAW frames by \
                  merging the original CFA sensor samples on a common high-resolution grid. \
                  No per-frame demosaic, no upscaling filter."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Log level: error, warn, info, debug, trace.
    #[arg(long, global = true, default_value = "info")]
    log: String,

    /// Limit the worker thread count. Default uses every core.
    #[arg(long, global = true)]
    threads: Option<usize>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Decode a burst and report what it contains, without reconstructing.
    Inspect(InspectArgs),
    /// Measure every frame of a burst and name the ones worth leaving out.
    Survey(SurveyArgs),
    /// Analyze a mono project and write an offline interactive noise/health report.
    Analyze(analyze::Args),
    /// Maintain a reproducible mono project and publish rebuilt masters.
    Project(project::Args),
    /// Build an experimental mono mosaic from a validated geometry plan.
    Mosaic(mosaic::Args),
    /// Register a burst and write registration diagnostics.
    Register(RegisterArgs),
    /// Full reconstruction.
    Stack(Box<StackArgs>),
    /// Run the synthetic ground-truth validation suite.
    Selftest(SelftestArgs),
    /// Open a local page for stacking a burst without typing flags.
    Gui(GuiArgs),
    /// Measure resolution and noise on finished images.
    Measure(MeasureArgs),
    /// Combine separately stacked channels into one colour image.
    Composite(CompositeArgs),
    /// Add the accumulators of several batches into one image.
    Combine(CombineArgs),
}

#[derive(Parser, Debug)]
struct CombineArgs {
    /// Accumulator directories, as written by `stack --accumulate`.
    ///
    /// Every batch has to have been stacked against the same
    /// `--reference-file`, or its pixels do not correspond to the others' and
    /// this refuses rather than adding them.
    dirs: Vec<PathBuf>,

    /// Output TIFF path.
    #[arg(short, long, default_value = "combined.tif")]
    output: PathBuf,

    /// Also write a viewable PNG, as `<output>.preview.png`.
    #[arg(long)]
    preview: bool,

    /// Longest edge of the preview, in pixels.
    #[arg(long, default_value = "2000")]
    preview_size: usize,

    /// Also write the linear result as 32-bit float.
    #[arg(long)]
    float_tiff: bool,

    /// Also write a 32-bit floating-point FITS master.
    #[arg(long)]
    fits: bool,

    /// Also write a 32-bit floating-point XISF master.
    #[arg(long)]
    xisf: bool,

    /// Limit the worker thread count. Default uses every core.
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Parser, Debug)]
struct InputArgs {
    /// Directory of RAW frames, a single file, or a text file listing frames.
    ///
    /// A list is one path per line, `#` comments and blanks skipped, relative
    /// paths taken against the list's own directory. Use one when the frames
    /// you want are not a directory — the result of a catalogue query, or the
    /// frames that survived a review. The frames are only ever read.
    input: PathBuf,

    /// File extension to match inside a directory.
    ///
    /// Omit it and every camera RAW or FITS file in the directory is taken; a
    /// directory holding two formats is an error rather than a guess.
    #[arg(long)]
    pattern: Option<String>,

    /// Keep only frames taken through this filter.
    ///
    /// A narrowband set holds several in one directory. They register against
    /// each other perfectly well, which is exactly why merging them has to be
    /// refused rather than left to look plausible: they are separate
    /// measurements of the sky, not repeats of one.
    #[arg(long)]
    filter: Option<String>,

    /// Frame that fixes the output grid, whether or not it is one of the inputs.
    ///
    /// Every run otherwise picks its own reference, so two runs over different
    /// halves of a set produce images that cannot be added together. Naming
    /// one makes the grid a property of the project rather than of the batch,
    /// which is what lets a set too large for memory be reconstructed in
    /// pieces — see `--accumulate` — and what lets two people's data land on
    /// the same pixels.
    #[arg(long)]
    reference_file: Option<PathBuf>,

    /// Private disk-backed storage for decoded frames, guides and rejection maps.
    #[arg(long)]
    scratch_dir: Option<PathBuf>,

    /// Reuse alignment and defect scans between runs, in `./cache`.
    ///
    /// Tuning the scale, the kernel or the region does not change how the
    /// frames line up or which sensor sites are bad, and on a small region
    /// those stages are most of the run — 37 of the 69 seconds a 400-pixel crop
    /// of the 100-frame chart burst takes.
    ///
    /// Off by default because a stale entry is worse than no cache: the
    /// fingerprint covers the input files and the configuration those stages
    /// read, but it cannot cover a change to the code. Clear the directory
    /// after changing registration or defect detection.
    #[arg(long)]
    cache: bool,

    /// Where cached results go, when `--cache` is given.
    #[arg(long, default_value = "./cache")]
    cache_dir: PathBuf,

    /// Do not measure the shape of point sources.
    ///
    /// Frame sharpness is measured on stars where a burst has them, which on a
    /// star field is a far better ranking than gradient energy and is also what
    /// reports trailing. Turn it off to rank by gradient energy instead, or to
    /// see what the measurement is worth.
    #[arg(long)]
    no_star_metrics: bool,

    /// Whether a FITS array starts at the bottom or the top of the image.
    ///
    /// `auto` believes the file's ROWORDER keyword and, when it is absent,
    /// takes the array in the order it is stored, which is what capture
    /// programs overwhelmingly write. Masters are written in that same order
    /// and say so with ROWORDER = 'TOP-DOWN'. Override this only if the output
    /// comes out mirrored top to bottom; the mosaic pattern is checked against
    /// the pixels either way, so getting it wrong does not affect colour.
    #[arg(long, default_value = "auto")]
    fits_row_order: String,

    /// Use only N frames of the burst. See --select for which N.
    #[arg(long)]
    max_frames: Option<usize>,

    /// Which frames --max-frames keeps.
    ///
    /// `first` takes them in filename order, which is fastest and is what you
    /// want while iterating. `sharpest` surveys the whole burst first and keeps
    /// the sharpest N, at the cost of decoding everything twice. `spread` takes
    /// N evenly spaced across the burst, which preserves whatever camera motion
    /// the burst has — and that motion is what makes super-resolution possible.
    #[arg(long, value_enum, default_value = "first")]
    select: SelectArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SelectArg {
    First,
    Sharpest,
    Spread,
}

impl InputArgs {
    fn spec(&self) -> Result<pipeline::InputSpec<'_>> {
        let row = sr_raw::RowOrder::parse(&self.fits_row_order).ok_or_else(|| {
            anyhow::anyhow!(
                "--fits-row-order {:?}: expected auto, bottom-up or top-down",
                self.fits_row_order
            )
        })?;
        Ok(pipeline::InputSpec {
            ordered_paths: None,
            spill_dir: self.scratch_dir.as_deref(),
            path: &self.input,
            pattern: self.pattern.as_deref(),
            max_frames: self.max_frames,
            select: self.select.into(),
            read: sr_raw::ReadOptions {
                fits_row_order: row,
            },
            star_metrics: !self.no_star_metrics,
            filter: self.filter.as_deref(),
            cache_dir: self.cache.then_some(self.cache_dir.as_path()),
            reference_file: self.reference_file.as_deref(),
        })
    }
}

impl From<SelectArg> for pipeline::FrameSelect {
    fn from(a: SelectArg) -> pipeline::FrameSelect {
        match a {
            SelectArg::First => pipeline::FrameSelect::First,
            SelectArg::Sharpest => pipeline::FrameSelect::Sharpest,
            SelectArg::Spread => pipeline::FrameSelect::Spread,
        }
    }
}

#[derive(Parser, Debug)]
struct CompositeArgs {
    /// Channels, as `NAME=path`. Names are yours; the palettes below use
    /// S, H, O for narrowband and R, G, B for broadband.
    ///
    /// Example: `H=h.tif O=o.tif S=s.tif`
    #[arg(required = true)]
    channels: Vec<String>,

    /// Which filter goes to which primary.
    ///
    /// `sho` is the Hubble palette (red from sulphur, green from hydrogen,
    /// blue from oxygen), `hso` swaps the first two, `hoo` is the two-filter
    /// version with oxygen in both green and blue, `rgb` is a broadband set.
    ///
    /// None of these is a colour anything really is: sulphur and hydrogen emit
    /// 16 nm apart in the deep red, so a faithful rendering would be two reds
    /// and a teal and would show almost nothing. The mapping is a convention
    /// for making structure visible and the choice is yours.
    #[arg(long, default_value = "sho")]
    palette: String,

    /// Override the palette with an explicit mapping, as `R=H,G=O,B=O`.
    #[arg(long)]
    map: Option<String>,

    /// How channels are brought onto a common scale before combining.
    ///
    /// `linear` fits gain and offset and is the usual choice for a narrowband
    /// palette: afterwards the channels are equal by construction and what
    /// survives is where they differ spatially, which is what a palette is for.
    /// `offset` matches only the background, keeping the relative brightness
    /// the filters actually measured. `none` leaves them alone, which for
    /// narrowband means the brightest filter wins.
    #[arg(long, value_enum, default_value = "linear")]
    fit: FitArg,

    /// Channel the fit and the alignment are measured against. Defaults to
    /// whichever one the palette sends to green.
    #[arg(long)]
    reference: Option<String>,

    /// Do not register the channels against each other.
    ///
    /// Alignment is on because separately stacked filters do not share a grid:
    /// each stack chose its own reference frame from inside its own filter.
    #[arg(long)]
    no_align: bool,

    /// Treat 16-bit inputs as linear rather than sRGB-encoded.
    ///
    /// This program writes integers through the transfer curve and floats
    /// linear, so that is what is assumed. Pass this for 16-bit linear masters
    /// from other software.
    #[arg(long)]
    assume_linear: bool,

    /// Stretch each channel on its own histogram before combining them.
    ///
    /// In linear light a narrowband palette shows which filter is brightest,
    /// and for SHO that is hydrogen by a wide margin — the result is green
    /// with the other two buried in it. Stretching each channel separately is
    /// what makes a palette show *where* the filters differ instead. It is a
    /// display transform applied per channel, not a measurement, and the
    /// coefficients are reported.
    #[arg(long)]
    stretch_channels: bool,

    /// Channel to take the luminance from, as in `--luminance L`.
    ///
    /// Brightness needs signal-to-noise and colour does not, which is why an
    /// LRGB set is mostly L: an unfiltered channel collects several times the
    /// photons of any colour filter, so it carries the detail while thin
    /// colour data tints it. The colour channels are scaled by the ratio of
    /// the new luminance to their own, which keeps hue exactly.
    #[arg(long)]
    luminance: Option<String>,

    /// How much of the luminance to impose, from 0 (none) to 1 (all of it).
    #[arg(long, default_value_t = 1.0)]
    luminance_strength: f32,

    /// Output TIFF path.
    #[arg(long, short, default_value = "composite.tif")]
    output: PathBuf,

    /// Also write a viewable PNG, as `<output>.preview.png`.
    ///
    /// Deep-sky output is not dark but *flat*: the median sits near mid-grey
    /// with a robust spread under a percent, so it reads on screen as one shade
    /// with a few white dots. The preview applies the standard astronomical
    /// screen transfer — and only when it would actually widen the spread, so a
    /// daytime result comes out unchanged rather than darkened.
    ///
    /// Never applied to the result itself.
    #[arg(long)]
    preview: bool,

    /// Longest edge of the preview, in pixels.
    #[arg(long, default_value = "2000")]
    preview_size: usize,

    /// Write this batch's contribution to a directory, for `smokstak combine`.
    ///
    /// A set too large for memory, or spread across several people's disks, is
    /// stacked in batches that all name the same `--reference-file` and each
    /// write an accumulator here. `smokstak combine` adds them. Summing weights
    /// and weighted values is what the merge does within a batch, so ten
    /// batches recombined are the same arithmetic as one run over everything —
    /// not an average of averages, which would weight a thin batch like a
    /// thick one.
    #[arg(long)]
    accumulate: Option<PathBuf>,

    /// Also write the linear result as 32-bit float.
    #[arg(long)]
    float_tiff: bool,

    /// Also write a 32-bit floating-point FITS master.
    #[arg(long)]
    fits: bool,

    /// Also write a 32-bit floating-point XISF master.
    #[arg(long)]
    xisf: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FitArg {
    None,
    Offset,
    Linear,
}

impl From<FitArg> for sr_composite::Fit {
    fn from(a: FitArg) -> sr_composite::Fit {
        match a {
            FitArg::None => sr_composite::Fit::None,
            FitArg::Offset => sr_composite::Fit::Offset,
            FitArg::Linear => sr_composite::Fit::Linear,
        }
    }
}

#[derive(Parser, Debug)]
struct InspectArgs {
    #[command(flatten)]
    input: InputArgs,

    /// Skip the registration survey and sampling-diversity estimate.
    #[arg(long)]
    quick: bool,

    /// Write the report as JSON to this path in addition to stdout.
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct SurveyArgs {
    #[command(flatten)]
    input: InputArgs,

    /// Where to write every frame's measurements, and why any of them look
    /// worse than the rest of their filter, as JSON.
    #[arg(long)]
    json: PathBuf,
}

#[derive(Parser, Debug)]
struct RegisterArgs {
    #[command(flatten)]
    input: InputArgs,

    /// Directory for diagnostic output.
    #[arg(long, default_value = "./diagnostics")]
    diagnostics: PathBuf,

    /// Reference frame index; chosen automatically when omitted.
    #[arg(long)]
    reference: Option<usize>,

    /// Local warp refinement.
    #[arg(long, value_enum, default_value = "auto")]
    local_warp: LocalWarpArg,
}

#[derive(Parser, Debug)]
struct StackArgs {
    #[command(flatten)]
    input: InputArgs,

    /// Output linear resolution multiplier.
    #[arg(long, default_value = "2.0")]
    scale: f32,

    /// Reconstruction backend.
    #[arg(long, value_enum, default_value = "burst-sr")]
    backend: BackendArg,

    /// Output TIFF path.
    #[arg(long, short, default_value = "result.tif")]
    output: PathBuf,

    /// Reconstruct every filter in the directory, onto one shared grid.
    ///
    /// Each filter is stacked separately — merging them would merge different
    /// measurements of the sky — but all of them are registered to one
    /// reference frame, so the masters come out on the same pixel grid and
    /// `composite --no-align` is exact. Without this each filter picks its own
    /// reference and the masters land a dozen pixels apart, which the composite
    /// then has to resample away.
    ///
    /// The output path gains the filter name: `m.tif` becomes `m_H.tif`.
    #[arg(long, conflicts_with = "filter")]
    split_by_filter: bool,

    /// Fit vignetting and a sky gradient to the result and remove them.
    ///
    /// Off by default, and deliberately. Nothing in one image distinguishes a
    /// corner that is dim because of the optics from one that is dim because
    /// the object is not there, so the model is kept to five coefficients per
    /// channel and can still compete with a large subject. It is also not a
    /// substitute for a flat frame: dust motes and per-pixel sensitivity are
    /// too small for any model this smooth to see.
    ///
    /// With --diagnostics the fitted model is written out as an image, which is
    /// the way to tell whether it started fitting your nebula.
    #[arg(long)]
    flatten_background: bool,

    /// Do not mask the sensor's hot and dead sites.
    ///
    /// They are found from the burst: a site that reads the same way in every
    /// frame while the scene moves underneath it. Left in, each becomes a short
    /// streak in the output, because the merge aligns the scene and the defect
    /// does not follow.
    #[arg(long)]
    no_defect_map: bool,

    /// Merge every sample, rejecting none.
    ///
    /// A diagnostic, not a mode to finish in: anything that moved through the
    /// field — a satellite, an aircraft, a cosmic ray — is averaged in at full
    /// weight. What it answers is whether rejection is helping this burst or
    /// hurting it, which on a set whose sky changed a great deal is not
    /// obvious. Compare the noise and the star SNR of a run with and without.
    #[arg(long)]
    no_reject: bool,

    /// Do not normalise frame brightness onto the reference before merging.
    ///
    /// Matching is on by default and measured from the pixels. Turn it off to
    /// see what a burst looks like without it, or if the scene genuinely does
    /// change brightness and you want that preserved rather than levelled.
    #[arg(long)]
    no_photometric_match: bool,

    /// Fit each frame an additive sky gradient (enabled by default).
    ///
    /// Fits spatial differences between aligned exposures after matching
    /// stellar flux. This is separate from fitting a background to the final
    /// image: it addresses illumination that changed during the burst.
    #[arg(long, conflicts_with = "no_sky_field")]
    sky_field: bool,

    /// Disable spatial sky matching; retain global brightness matching.
    #[arg(long)]
    no_sky_field: bool,

    /// Directory for diagnostics; omit to skip them.
    #[arg(long)]
    diagnostics: Option<PathBuf>,

    /// Reference frame index; chosen automatically when omitted.
    #[arg(long)]
    reference: Option<usize>,

    /// Local warp refinement.
    #[arg(long, value_enum, default_value = "auto")]
    local_warp: LocalWarpArg,

    /// Lucky-region frame selection: `off`, `auto`, or a keep fraction like 0.5.
    #[arg(long, default_value = "off")]
    lucky: String,

    /// Final restoration.
    #[arg(long, value_enum, default_value = "none")]
    postprocess: PostProcessArg,

    /// Output tile size in output pixels; bounds peak memory.
    #[arg(long, default_value = "512")]
    tile: usize,

    /// Restrict reconstruction to a region of the reference sensor array:
    /// `X,Y,W,H` in sensor pixels.
    #[arg(long)]
    roi: Option<String>,

    /// Also write a viewable PNG, as `<output>.preview.png`.
    ///
    /// Deep-sky output is not dark but *flat*: the median sits near mid-grey
    /// with a robust spread under a percent, so it reads on screen as one shade
    /// with a few white dots. The preview applies the standard astronomical
    /// screen transfer — and only when it would actually widen the spread, so a
    /// daytime result comes out unchanged rather than darkened.
    ///
    /// Never applied to the result itself.
    #[arg(long)]
    preview: bool,

    /// Longest edge of the preview, in pixels.
    #[arg(long, default_value = "2000")]
    preview_size: usize,

    /// Write this batch's contribution to a directory, for `smokstak combine`.
    ///
    /// A set too large for memory, or spread across several people's disks, is
    /// stacked in batches that all name the same `--reference-file` and each
    /// write an accumulator here. `smokstak combine` adds them. Summing weights
    /// and weighted values is what the merge does within a batch, so ten
    /// batches recombined are the same arithmetic as one run over everything —
    /// not an average of averages, which would weight a thin batch like a
    /// thick one.
    #[arg(long)]
    accumulate: Option<PathBuf>,

    /// Also write a 32-bit float linear TIFF next to the 16-bit output.
    #[arg(long)]
    float_tiff: bool,

    /// Also write a 32-bit floating-point FITS master.
    #[arg(long)]
    fits: bool,

    /// Also write a 32-bit floating-point XISF master.
    #[arg(long)]
    xisf: bool,

    /// Proceed even when burst validation reports a fatal inconsistency.
    #[arg(long)]
    force: bool,

    /// Adaptive kernel variance in detailed regions, in sensor pixels squared.
    /// Setting an adaptive kernel parameter also selects adaptive mono kernels.
    #[arg(long)]
    k_detail: Option<f32>,

    /// Fixed monochrome sampling variance in sensor pixels squared (default 0.125).
    #[arg(long, conflicts_with = "adaptive_mono_kernel")]
    mono_kernel_variance: Option<f32>,

    /// Use spatially adaptive smoothing for mono, as used for CFA data.
    #[arg(long)]
    adaptive_mono_kernel: bool,

    /// Kernel variance in flat regions, in sensor pixels squared.
    #[arg(long)]
    k_denoise: Option<f32>,

    /// Maximum kernel elongation along an edge (1 disables elongation).
    #[arg(long)]
    k_stretch: Option<f32>,

    /// Maximum kernel narrowing across an edge (1 disables narrowing).
    #[arg(long)]
    k_shrink: Option<f32>,

    /// Structure-to-noise ratio at which a region counts as fully detailed.
    #[arg(long)]
    detail_snr: Option<f32>,

    /// Kernel support radius in output pixels.
    #[arg(long)]
    kernel_radius: Option<f32>,

    /// Kernel variance multiplier for red and blue.
    ///
    /// A Bayer mosaic samples red and blue on a lattice of twice green's
    /// pitch, so the same kernel rests on a quarter as many of their samples.
    /// Widening theirs trades colour resolution, which the eye has little of,
    /// for colour noise, which it sees at once.
    #[arg(long)]
    chroma_variance: Option<f32>,

    /// Do not shrink the flat-region kernel as the frame count rises.
    #[arg(long)]
    no_denoise_scaling: bool,

    /// Do not shrink the detail kernel as the frame count rises.
    #[arg(long)]
    no_detail_scaling: bool,

    /// Set each output pixel to the weighted mean of its samples rather than
    /// to a plane fitted through them.
    ///
    /// The mean is the value where the samples averaged to, which is not the
    /// pixel's centre unless the dither happened to leave them symmetric
    /// about it.
    #[arg(long)]
    no_plane_fit: bool,
    /// Fit raw profile curvature; automatic for mono, experimental for CFA.
    #[arg(long, conflicts_with_all = ["no_plane_fit", "no_curvature"])]
    fit_curvature: bool,
    /// Disable profile curvature fitting, retaining the local plane estimator.
    #[arg(long)]
    no_curvature: bool,

    /// Bounded centroid correction around clipped samples (automatic for CFA).
    #[arg(long, conflicts_with_all = ["no_plane_fit", "no_clipped_fit"])]
    fit_clipped: bool,

    /// Keep the common mean around clipped samples in every channel.
    #[arg(long)]
    no_clipped_fit: bool,

    /// Do not measure or correct lateral chromatic aberration.
    #[arg(long)]
    no_ca: bool,
}

impl StackArgs {
    fn selected_clipped_fit(&self) -> Option<bool> {
        if self.no_clipped_fit {
            Some(false)
        } else if self.fit_clipped {
            Some(true)
        } else {
            None
        }
    }

    fn selected_mono_variance(&self) -> Option<f32> {
        if let Some(v) = self.mono_kernel_variance {
            return Some(v);
        }
        if self.adaptive_mono_kernel
            || self.k_detail.is_some()
            || self.k_denoise.is_some()
            || self.k_stretch.is_some()
            || self.k_shrink.is_some()
            || self.detail_snr.is_some()
            || self.no_detail_scaling
            || self.no_denoise_scaling
        {
            return None;
        }
        sr_core::config::KernelConfig::default().mono_kernel_variance
    }
}

#[derive(Parser, Debug)]
struct GuiArgs {
    /// Port to listen on. The next few are tried if it is taken.
    #[arg(long, default_value_t = 7878)]
    port: u16,

    /// Print the address instead of opening a browser at it.
    #[arg(long)]
    no_open: bool,
}

#[derive(Parser, Debug)]
struct SelftestArgs {
    /// Run without photometric matching, to show what it is worth.
    #[arg(long)]
    no_photometric_match: bool,

    /// Run without masking fixed-pattern defects, to show what it is worth.
    #[arg(long)]
    no_defect_map: bool,

    /// Directory for the generated frames and reports.
    #[arg(long, default_value = "./selftest")]
    out: PathBuf,

    /// Number of synthetic frames.
    #[arg(long, default_value = "24")]
    frames: usize,

    /// Latent scene edge length in high-resolution pixels.
    #[arg(long, default_value = "512")]
    size: usize,

    /// Keep the generated inputs and reconstructions on disk.
    #[arg(long)]
    keep: bool,
}

#[derive(Parser, Debug)]
struct MeasureArgs {
    /// Images to measure. Append `@SCALE` to say what magnification each was
    /// produced at, e.g. `result.tif@2 single.tif@1`, so that results from
    /// different output sizes are directly comparable.
    inputs: Vec<String>,

    /// Box to find the slanted edge in, as `X,Y,W,H` in *sensor* pixels
    /// relative to the reconstructed region. Scaled by each file's own
    /// magnification. Without it the resolution columns read "no edge" and
    /// only noise, fringing and residual CA are reported.
    #[arg(long)]
    edge_box: Option<String>,

    /// Explicit edge as `X0,Y0,X1,Y1` in sensor pixels, instead of fitting one.
    #[arg(long)]
    edge: Option<String>,

    /// Box of uniform tone for the noise measurement, `X,Y,W,H` in sensor
    /// pixels. Defaults to the whole image, whose flattest tiles are used.
    #[arg(long)]
    flat_box: Option<String>,

    /// Optical centre, as `X,Y` in each image's own pixels, for the residual
    /// chromatic-aberration column. Defaults to the image centre, which is only
    /// correct for a full frame; on a crop, give the full frame's centre
    /// expressed in the crop's coordinates.
    #[arg(long)]
    ca_centre: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    RgbMean,
    CfaDrizzle,
    BurstSr,
}

impl From<BackendArg> for Backend {
    fn from(a: BackendArg) -> Backend {
        match a {
            BackendArg::RgbMean => Backend::RgbMeanBaseline,
            BackendArg::CfaDrizzle => Backend::CfaDrizzle,
            BackendArg::BurstSr => Backend::HandheldBurstSr,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LocalWarpArg {
    Off,
    On,
    Auto,
}

impl From<LocalWarpArg> for LocalWarpMode {
    fn from(a: LocalWarpArg) -> LocalWarpMode {
        match a {
            LocalWarpArg::Off => LocalWarpMode::Off,
            LocalWarpArg::On => LocalWarpMode::On,
            LocalWarpArg::Auto => LocalWarpMode::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PostProcessArg {
    None,
    Mild,
}

impl From<PostProcessArg> for PostProcess {
    fn from(a: PostProcessArg) -> PostProcess {
        match a {
            PostProcessArg::None => PostProcess::None,
            PostProcessArg::Mild => PostProcess::Mild,
        }
    }
}

fn parse_lucky(s: &str) -> Result<LuckyMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "off" => Ok(LuckyMode::Off),
        "auto" => Ok(LuckyMode::Auto),
        other => {
            let f: f32 = other.parse().map_err(|_| {
                anyhow::anyhow!("--lucky expects off, auto or a fraction; got {s:?}")
            })?;
            if !(0.05..=1.0).contains(&f) {
                anyhow::bail!("--lucky fraction must be between 0.05 and 1.0, got {f}");
            }
            Ok(LuckyMode::Fraction(f))
        }
    }
}

fn parse_roi(s: &str) -> Result<(usize, usize, usize, usize)> {
    let parts: Vec<&str> = s.split(&[',', 'x', ':'][..]).map(|p| p.trim()).collect();
    if parts.len() != 4 {
        anyhow::bail!("--roi expects X,Y,W,H; got {s:?}");
    }
    let v: Vec<usize> = parts
        .iter()
        .map(|p| p.parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| anyhow::anyhow!("--roi values must be non-negative integers; got {s:?}"))?;
    if v[2] == 0 || v[3] == 0 {
        anyhow::bail!("--roi width and height must be non-zero");
    }
    Ok((v[0], v[1], v[2], v[3]))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::new()
        .parse_filters(&cli.log)
        .format_timestamp_millis()
        .init();

    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .ok();
    }

    match cli.command {
        Command::Inspect(a) => pipeline::inspect(&a.input.spec()?, a.quick, a.json.as_deref()),
        Command::Survey(a) => pipeline::survey(&a.input.spec()?, &a.json),
        Command::Analyze(a) => analyze::run(&a),
        Command::Project(a) => project::run(&a),
        Command::Mosaic(a) => mosaic::run(&a),
        Command::Register(a) => {
            let cfg = ReconstructionConfig {
                reference: a.reference,
                local_warp: a.local_warp.into(),
                ..Default::default()
            };
            pipeline::register(&a.input.spec()?, &cfg, &a.diagnostics)
        }
        Command::Stack(a) => {
            // Preserve explicit adaptive tuning requests. A mono-specific
            // variance overrides them when supplied deliberately.
            let mut kernel = sr_core::config::KernelConfig {
                mono_kernel_variance: a.selected_mono_variance(),
                ..Default::default()
            };
            if let Some(v) = a.mono_kernel_variance {
                anyhow::ensure!(
                    v.is_finite() && v >= 0.03,
                    "mono kernel variance must be finite and at least 0.03"
                );
                kernel.mono_kernel_variance = Some(v);
            }
            if let Some(v) = a.k_detail {
                kernel.k_detail = v;
            }
            if let Some(v) = a.k_denoise {
                kernel.k_denoise = v;
            }
            if let Some(v) = a.k_stretch {
                kernel.k_stretch = v;
            }
            if let Some(v) = a.k_shrink {
                kernel.k_shrink = v;
            }
            if let Some(v) = a.detail_snr {
                kernel.detail_snr = v;
            }
            if let Some(v) = a.chroma_variance {
                kernel.chroma_variance = v;
            }
            if let Some(v) = a.kernel_radius {
                kernel.radius = v;
            }
            kernel.scale_denoise_with_frames = !a.no_denoise_scaling;
            kernel.scale_detail_with_frames = !a.no_detail_scaling;
            kernel.debias_deposit = !a.no_plane_fit;
            kernel.fit_clipped = a.selected_clipped_fit();
            kernel.fit_curvature = if a.no_curvature {
                Some(false)
            } else if a.fit_curvature {
                Some(true)
            } else {
                None
            };
            let cfg = ReconstructionConfig {
                scale: a.scale,
                photometric_match: !a.no_photometric_match,
                sky_field: a.sky_field || !a.no_sky_field,
                detect_defects: !a.no_defect_map,
                flatten_background: a.flatten_background,
                robustness: sr_core::config::RobustnessConfig {
                    enabled: !a.no_reject,
                    ..Default::default()
                },
                kernel,
                backend: a.backend.into(),
                local_warp: a.local_warp.into(),
                lucky: parse_lucky(&a.lucky)?,
                postprocess: a.postprocess.into(),
                tile: a.tile.max(64),
                roi: a.roi.as_deref().map(parse_roi).transpose()?,
                reference: a.reference,
                ..Default::default()
            };
            pipeline::stack(
                &a.input.spec()?,
                a.split_by_filter,
                &cfg,
                &a.output,
                a.diagnostics.as_deref(),
                preview_path(a.preview, &a.output).as_deref(),
                a.preview_size,
                a.float_tiff,
                a.fits,
                a.xisf,
                a.accumulate.as_deref(),
                a.force,
                !a.no_ca,
            )
        }
        Command::Combine(a) => pipeline::combine(
            &a.dirs,
            &a.output,
            preview_path(a.preview, &a.output).as_deref(),
            a.preview_size,
            a.float_tiff,
            a.fits,
            a.xisf,
        ),
        Command::Composite(a) => pipeline::composite(
            &a.channels,
            &a.palette,
            a.map.as_deref(),
            a.fit.into(),
            a.reference.as_deref(),
            !a.no_align,
            a.assume_linear,
            a.stretch_channels,
            a.luminance.as_deref(),
            a.luminance_strength,
            &a.output,
            preview_path(a.preview, &a.output).as_deref(),
            a.preview_size,
            a.float_tiff,
            a.fits,
            a.xisf,
        ),
        Command::Gui(a) => gui::run(a.port, !a.no_open),
        Command::Selftest(a) => pipeline::selftest(
            &a.out,
            a.frames,
            a.size,
            a.keep,
            !a.no_photometric_match,
            !a.no_defect_map,
        ),
        Command::Measure(a) => {
            anyhow::ensure!(!a.inputs.is_empty(), "measure needs at least one image");
            pipeline::measure(
                &a.inputs,
                a.edge_box.as_deref().map(parse_roi).transpose()?,
                a.edge.as_deref().map(parse_edge).transpose()?,
                a.flat_box.as_deref().map(parse_roi).transpose()?,
                a.ca_centre.as_deref().map(parse_point).transpose()?,
            )
        }
    }
}

/// Where a preview goes: beside the output, with its own suffix.
///
/// Not a path the caller supplies. An optional-valued flag cannot be told from
/// the positional input directory that follows it, and a preview is a
/// throwaway view of one specific result — tying it to that result's name is
/// what someone reading a directory of them would want anyway.
fn preview_path(wanted: bool, output: &Path) -> Option<PathBuf> {
    wanted.then(|| output.with_extension("preview.png"))
}

fn parse_point(s: &str) -> Result<(f32, f32)> {
    let v: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| anyhow::anyhow!("expected X,Y; got {s:?}"))?;
    anyhow::ensure!(v.len() == 2, "expected X,Y; got {s:?}");
    Ok((v[0], v[1]))
}

fn parse_edge(s: &str) -> Result<(f32, f32, f32, f32)> {
    let v: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| anyhow::anyhow!("--edge expects X0,Y0,X1,Y1; got {s:?}"))?;
    anyhow::ensure!(v.len() == 4, "--edge expects X0,Y0,X1,Y1; got {s:?}");
    Ok((v[0], v[1], v[2], v[3]))
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn clipped_fit_is_automatic_with_explicit_overrides() {
        for (flags, expected) in [
            (vec![], None),
            (vec!["--fit-clipped"], Some(true)),
            (vec!["--no-clipped-fit"], Some(false)),
        ] {
            let mut args = vec!["smokstak", "stack", "lights", "--output", "stack.tif"];
            args.extend(flags);
            let Command::Stack(parsed) = Cli::try_parse_from(args).unwrap().command else {
                panic!("expected stack command");
            };
            assert_eq!(parsed.selected_clipped_fit(), expected);
        }
        assert!(
            Cli::try_parse_from([
                "smokstak",
                "stack",
                "lights",
                "--output",
                "stack.tif",
                "--fit-clipped",
                "--no-clipped-fit"
            ])
            .is_err()
        );
        let mut saved = serde_json::to_value(sr_core::config::KernelConfig::default()).unwrap();
        saved.as_object_mut().unwrap().remove("fit_clipped");
        let loaded: sr_core::config::KernelConfig = serde_json::from_value(saved.clone()).unwrap();
        assert_eq!(loaded.fit_clipped, None);
        for value in [true, false] {
            saved["fit_clipped"] = value.into();
            let loaded: sr_core::config::KernelConfig =
                serde_json::from_value(saved.clone()).unwrap();
            assert_eq!(loaded.fit_clipped, Some(value));
        }
    }

    #[test]
    fn mono_kernel_defaults_and_explicit_tuning_are_preserved() {
        for (flags, expected) in [
            (vec![], Some(0.125)),
            (vec!["--adaptive-mono-kernel"], None),
            (vec!["--k-detail", "0.2"], None),
            (vec!["--no-denoise-scaling"], None),
            (vec!["--mono-kernel-variance", "0.2"], Some(0.2)),
            (
                vec!["--k-detail", "0.3", "--mono-kernel-variance", "0.2"],
                Some(0.2),
            ),
        ] {
            let mut args = vec!["smokstak", "stack", "lights", "--output", "stack.tif"];
            args.extend(flags);
            let Command::Stack(parsed) = Cli::try_parse_from(args).unwrap().command else {
                panic!("expected stack command");
            };
            assert_eq!(parsed.selected_mono_variance(), expected);
        }
        let mut saved = serde_json::to_value(sr_core::config::KernelConfig::default()).unwrap();
        saved
            .as_object_mut()
            .unwrap()
            .remove("mono_kernel_variance");
        let loaded: sr_core::config::KernelConfig = serde_json::from_value(saved.clone()).unwrap();
        assert_eq!(loaded.mono_kernel_variance, Some(0.125));
        saved["mono_kernel_variance"] = serde_json::Value::Null;
        let loaded: sr_core::config::KernelConfig = serde_json::from_value(saved).unwrap();
        assert_eq!(loaded.mono_kernel_variance, None);
    }

    #[test]
    fn sky_matching_can_be_disabled_without_breaking_the_existing_flag() {
        for (flags, expected) in [
            (vec![], true),
            (vec!["--sky-field"], true),
            (vec!["--no-sky-field"], false),
        ] {
            let mut args = vec!["smokstak", "stack", "lights", "--output", "stack.tif"];
            args.extend(flags);
            let Command::Stack(parsed) = Cli::try_parse_from(args).unwrap().command else {
                panic!("expected stack command");
            };
            assert_eq!(parsed.sky_field || !parsed.no_sky_field, expected);
        }
        assert!(
            Cli::try_parse_from([
                "smokstak",
                "stack",
                "lights",
                "--output",
                "stack.tif",
                "--sky-field",
                "--no-sky-field",
            ])
            .is_err()
        );
    }
}
