//! Descriptive evidence for the applied warp, with no new fit or weight change.
//!
//! Residuals are conditional on reciprocal matches within four sensor pixels.
//! Always read them with the match counts and spatial coverage. Catalogues may
//! contain blends or defects, and may overlap the stars used to fit the warp:
//! these measurements are not independent validation or a calibrated probability.

use serde::Serialize;
use sr_core::{geometry::WarpField, star::Star};

const GRID: usize = 4;
const RADIUS: f32 = 4.0;

#[derive(Clone, Debug, Default, Serialize)]
pub struct CellEvidence {
    pub mapped_sources: usize,
    pub matches: usize,
    pub residual_p50: Option<f32>,
    pub median_dx: Option<f32>,
    pub median_dy: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AlignmentEvidence {
    pub source_detections: usize,
    pub reference_in_bounds: usize,
    pub mapped_in_bounds: usize,
    pub matches: usize,
    pub matched_fraction: Option<f32>,
    pub residual_p50: Option<f32>,
    pub residual_p90: Option<f32>,
    /// Row-major 4x4 cells covering the full reference sensor, not the catalogue extent.
    pub cells: Vec<CellEvidence>,
}

fn quantile(values: &[f32], fraction: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let t = (sorted.len() - 1) as f32 * fraction;
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * (t - lo as f32))
}

/// Measure the actual global-plus-local mapping in sensor pixels. Invalid and
/// off-sensor coordinates cannot provide evidence. Empty matches stay unknown.
pub fn applied_alignment(
    reference: &[Star],
    source: &[Star],
    warp: &WarpField,
    width: usize,
    height: usize,
) -> AlignmentEvidence {
    let inside = |&(x, y): &(f32, f32)| {
        x.is_finite()
            && y.is_finite()
            && x >= 0.0
            && y >= 0.0
            && x < width as f32
            && y < height as f32
    };
    let refs: Vec<_> = reference
        .iter()
        .map(|s| (s.x, s.y))
        .filter(inside)
        .collect();
    let mapped: Vec<_> = source
        .iter()
        .map(|s| (s.x, s.y))
        .filter(inside)
        .map(|(x, y)| warp.map(x, y))
        .filter(inside)
        .collect();
    let mut nearest_source = vec![None; refs.len()];
    let mut nearest_ref = vec![None; mapped.len()];
    let mut source_distance = vec![f32::INFINITY; refs.len()];
    let mut ref_distance = vec![f32::INFINITY; mapped.len()];
    // Catalogues are bounded by the caller. One pass finds both directions;
    // memory is linear even for a very large, sparsely populated sensor.
    for (i, &(x, y)) in mapped.iter().enumerate() {
        for (j, &(rx, ry)) in refs.iter().enumerate() {
            let d = (rx - x).powi(2) + (ry - y).powi(2);
            if d > RADIUS * RADIUS {
                continue;
            }
            if d < ref_distance[i] {
                ref_distance[i] = d;
                nearest_ref[i] = Some(j);
            }
            if d < source_distance[j] {
                source_distance[j] = d;
                nearest_source[j] = Some(i);
            }
        }
    }
    let mut cells = vec![CellEvidence::default(); GRID * GRID];
    let mut vectors = vec![Vec::new(); GRID * GRID];
    let mut residuals = Vec::new();
    for (i, &(x, y)) in mapped.iter().enumerate() {
        let col = (x / width as f32 * GRID as f32) as usize;
        let row = (y / height as f32 * GRID as f32) as usize;
        let cell = row.min(GRID - 1) * GRID + col.min(GRID - 1);
        cells[cell].mapped_sources += 1;
        if let Some(j) = nearest_ref[i] {
            if nearest_source[j] != Some(i) {
                continue;
            }
            let (dx, dy) = (refs[j].0 - x, refs[j].1 - y);
            vectors[cell].push((dx, dy));
            residuals.push(dx.hypot(dy));
        }
    }
    for (cell, pairs) in cells.iter_mut().zip(vectors) {
        cell.matches = pairs.len();
        cell.residual_p50 = quantile(
            &pairs.iter().map(|(x, y)| x.hypot(*y)).collect::<Vec<_>>(),
            0.5,
        );
        cell.median_dx = quantile(&pairs.iter().map(|p| p.0).collect::<Vec<_>>(), 0.5);
        cell.median_dy = quantile(&pairs.iter().map(|p| p.1).collect::<Vec<_>>(), 0.5);
    }
    AlignmentEvidence {
        source_detections: source.len(),
        reference_in_bounds: refs.len(),
        mapped_in_bounds: mapped.len(),
        matches: residuals.len(),
        matched_fraction: (!mapped.is_empty())
            .then(|| residuals.len() as f32 / mapped.len() as f32),
        residual_p50: quantile(&residuals, 0.5),
        residual_p90: quantile(&residuals, 0.9),
        cells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sr_core::geometry::GlobalTransform;
    fn star(x: f32, y: f32) -> Star {
        Star { x, y, flux: 1.0 }
    }

    #[test]
    fn reports_applied_error_without_fitting_it_away() {
        let refs = vec![star(10.0, 10.0), star(80.0, 80.0)];
        let warp = WarpField::global_only(GlobalTransform::translation(2.0, 0.0));
        let report = applied_alignment(&refs, &refs, &warp, 100, 100);
        assert_eq!(report.matches, 2);
        assert_eq!(report.residual_p50, Some(2.0));
        assert_eq!(report.cells[0].median_dx, Some(-2.0));
        assert_eq!(report.cells[15].matches, 1);
        assert_eq!(report.cells[5].residual_p50, None);
    }

    #[test]
    fn wrong_warp_has_no_false_zero_error_and_duplicates_are_not_extra_evidence() {
        let refs = vec![star(10.0, 10.0), star(80.0, 80.0)];
        let source = vec![refs[0], refs[0], refs[1]];
        let report = applied_alignment(&refs, &source, &WarpField::identity(), 100, 100);
        assert_eq!(report.matches, 2);
        assert_eq!(report.mapped_in_bounds, 3);
        let warp = WarpField::global_only(GlobalTransform::translation(8.0, 0.0));
        let report = applied_alignment(&refs, &source, &warp, 100, 100);
        assert_eq!(report.matches, 0);
        assert_eq!(report.matched_fraction, Some(0.0));
        assert_eq!(report.residual_p50, None);
    }

    #[test]
    fn nonfinite_off_sensor_and_empty_catalogues_are_not_alignment_evidence() {
        let bad = vec![star(f32::NAN, 0.0), star(-1.0, 0.0), star(100.0, 50.0)];
        let report = applied_alignment(&bad, &bad, &WarpField::identity(), 100, 100);
        assert_eq!(report.reference_in_bounds, 0);
        assert_eq!(report.mapped_in_bounds, 0);
        assert_eq!(report.matched_fraction, None);
        assert_eq!(report.residual_p90, None);
        let empty = applied_alignment(&[], &[], &WarpField::identity(), 0, 0);
        assert_eq!(empty.matches, 0);
    }

    #[test]
    fn measures_local_correction_after_global_placement() {
        let refs = vec![star(10.0, 10.0), star(80.0, 80.0)];
        let mut local = sr_core::geometry::DeformationField::zeros((0.0, 0.0), 100.0, 2, 2);
        local.u.fill([-2.0, 0.0]);
        let warp = WarpField {
            global: GlobalTransform::translation(2.0, 0.0),
            local: Some(local),
        };
        let report = applied_alignment(&refs, &refs, &warp, 100, 100);
        assert_eq!(report.matches, 2);
        assert_eq!(report.residual_p90, Some(0.0));
    }
}
