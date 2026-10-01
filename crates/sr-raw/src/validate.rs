//! Stage 2: burst consistency checking.
//!
//! Mild differences are reported, not rejected. The point is that the operator
//! finds out about the two frames shot at 1/400 s before spending an hour of
//! compute, and that the reconstruction can normalise for them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sr_core::frame::{FrameMetadata, RawFrame};

/// Exposure/gain level used by the existing metadata normalization fallback.
/// This is not an integration duration; missing headers use the legacy defaults.
pub fn exposure_level(metadata: &FrameMetadata) -> f32 {
    let t = metadata.exposure_time.unwrap_or(1.0).max(1e-9);
    let iso = metadata.iso.unwrap_or(100.0).max(1.0);
    t * iso
}

/// How one metadata field varied across the burst.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldReport {
    pub field: String,
    /// Distinct values and how many frames carried each, most common first.
    pub values: Vec<(String, usize)>,
    pub consistent: bool,
    /// True when a difference matters enough to change the reconstruction.
    pub severity: Severity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ok,
    Note,
    Warning,
    Fatal,
}

impl FieldReport {
    pub fn summary(&self) -> String {
        if self.values.len() == 1 {
            format!("{} ({}/{})", self.values[0].0, self.values[0].1, self.total())
        } else {
            self.values
                .iter()
                .map(|(v, n)| format!("{v} ({n})"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }

    pub fn total(&self) -> usize {
        self.values.iter().map(|(_, n)| n).sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurstValidation {
    pub frame_count: usize,
    pub fields: Vec<FieldReport>,
    pub warnings: Vec<String>,
    pub fatal: Vec<String>,
    /// The burst holds more than one filter. Fatal for a single
    /// reconstruction, and exactly what a per-filter run expects, so the
    /// decision is left to the caller rather than made here.
    pub filter_conflict: bool,
    /// Per-frame exposure normalisation factor relative to the burst median.
    pub exposure_scale: Vec<f32>,
}

impl BurstValidation {
    pub fn is_usable(&self) -> bool {
        self.fatal.is_empty()
    }

    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("{} files loaded\n\n", self.frame_count));
        for f in &self.fields {
            let marker = match f.severity {
                Severity::Ok => "",
                Severity::Note => "  <-- note",
                Severity::Warning => "  <-- warning",
                Severity::Fatal => "  <-- FATAL",
            };
            s.push_str(&format!("{:<16} {}{}\n", format!("{}:", f.field), f.summary(), marker));
        }
        if !self.warnings.is_empty() {
            s.push('\n');
            for w in &self.warnings {
                s.push_str(&format!("warning: {w}\n"));
            }
        }
        for w in &self.fatal {
            s.push_str(&format!("fatal: {w}\n"));
        }
        s
    }
}

fn tally<F>(frames: &[RawFrame], field: &str, sev_if_varying: Severity, f: F) -> FieldReport
where
    F: Fn(&RawFrame) -> String,
{
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for fr in frames {
        *counts.entry(f(fr)).or_insert(0) += 1;
    }
    let mut values: Vec<(String, usize)> = counts.into_iter().collect();
    values.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let consistent = values.len() <= 1;
    FieldReport {
        field: field.to_string(),
        values,
        consistent,
        severity: if consistent { Severity::Ok } else { sev_if_varying },
    }
}

fn fmt_opt(v: Option<f32>, unit: &str, decimals: usize) -> String {
    match v {
        Some(x) => format!("{x:.*}{unit}", decimals),
        None => "unknown".to_string(),
    }
}

fn fmt_shutter(v: Option<f32>) -> String {
    match v {
        Some(t) if t > 0.0 && t < 1.0 => format!("1/{:.0} s", 1.0 / t),
        Some(t) => format!("{t:.2} s"),
        None => "unknown".to_string(),
    }
}

pub fn validate_burst(frames: &[RawFrame]) -> BurstValidation {
    let mut fields = Vec::new();
    let mut warnings = Vec::new();
    let mut fatal = Vec::new();

    fields.push(tally(frames, "Camera", Severity::Fatal, |f| {
        format!("{} {}", f.metadata.make, f.metadata.model)
    }));
    fields.push(tally(frames, "Dimensions", Severity::Fatal, |f| {
        format!("{}x{}", f.width, f.height)
    }));
    fields.push(tally(frames, "CFA", Severity::Fatal, |f| f.cfa.name()));
    // Frames through different filters are different measurements of the sky.
    // They register perfectly well against each other, which is exactly why
    // this has to be refused rather than left to look right.
    fields.push(tally(frames, "Filter", Severity::Fatal, |f| {
        f.metadata.filter.clone().unwrap_or_else(|| "none".into())
    }));
    fields.push(tally(frames, "Crop", Severity::Fatal, |f| {
        let c = f.metadata.crop;
        format!("{}+{}", c.0, c.1)
    }));
    fields.push(tally(frames, "ISO", Severity::Warning, |f| {
        fmt_opt(f.metadata.iso, "", 0)
    }));
    fields.push(tally(frames, "Exposure", Severity::Warning, |f| {
        fmt_shutter(f.metadata.exposure_time)
    }));
    fields.push(tally(frames, "Aperture", Severity::Warning, |f| {
        f.metadata.aperture.map(|a| format!("f/{a:.1}")).unwrap_or_else(|| "unknown".into())
    }));
    fields.push(tally(frames, "Focal length", Severity::Warning, |f| {
        fmt_opt(f.metadata.focal_length, " mm", 0)
    }));
    fields.push(tally(frames, "Black level", Severity::Note, |f| {
        format!("{:.0}", f.metadata.black_levels[0])
    }));
    fields.push(tally(frames, "White level", Severity::Note, |f| {
        format!("{:.0}", f.metadata.white_level)
    }));

    for f in &fields {
        if !f.consistent {
            match f.severity {
                Severity::Fatal if f.field == "Filter" => fatal.push(format!(
                    "the burst holds more than one filter ({}); frames taken through \
                     different filters measure different light and cannot be merged. \
                     Stack each apart onto one shared grid with --split-by-filter, \
                     or take a single one with --filter",
                    f.summary()
                )),
                Severity::Fatal => fatal.push(format!(
                    "{} varies across the burst ({}); frames cannot be merged",
                    f.field,
                    f.summary()
                )),
                Severity::Warning => warnings.push(format!(
                    "{} varies across the burst ({})",
                    f.field,
                    f.summary()
                )),
                _ => {}
            }
        }
    }

    // Exposure normalisation: scale every frame to the burst's median exposure
    // so brightness differences do not masquerade as scene change.
    let mut exposure_scale = vec![1.0f32; frames.len()];
    let exposures: Vec<f32> = frames.iter().map(|f| exposure_level(&f.metadata)).collect();
    let med = sr_core::math::median(&exposures);
    if med > 0.0 {
        for (i, e) in exposures.iter().enumerate() {
            exposure_scale[i] = med / e.max(1e-9);
        }
        let spread = exposure_scale
            .iter()
            .fold(0.0f32, |acc, &s| acc.max((s - 1.0).abs()));
        if spread > 0.02 {
            warnings.push(format!(
                "exposure/ISO differs by up to {:.1}% across the burst; frames will be normalised",
                spread * 100.0
            ));
        }
    }

    let mean_sat: f32 =
        frames.iter().map(|f| f.saturation_fraction()).sum::<f32>() / frames.len().max(1) as f32;
    if mean_sat > 0.02 {
        warnings.push(format!(
            "{:.1}% of sensor sites are saturated on average; highlights will be poorly constrained",
            mean_sat * 100.0
        ));
    }

    if frames.len() < 8 {
        warnings.push(format!(
            "only {} frames: sub-pixel diversity is unlikely to support meaningful super-resolution",
            frames.len()
        ));
    }

    let filter_conflict = fields
        .iter()
        .any(|f| f.field == "Filter" && !f.consistent);
    BurstValidation {
        frame_count: frames.len(),
        fields,
        warnings,
        fatal,
        filter_conflict,
        exposure_scale,
    }
}
