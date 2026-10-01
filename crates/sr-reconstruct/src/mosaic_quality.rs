//! Conservative mono mosaic PSF selection before consensus rejection.
//!
//! A measured global HFD is only a proxy: it does not capture spatially varying
//! aberrations, ellipticity, undersampling, or inaccurate measurements. Selecting
//! the finest available cohort preserves resolution at the cost of discarding
//! broader exposures' signal-to-noise benefit. It does not blur or deconvolve the
//! output. Selection lives on the existing guide lattice, so interpolated map
//! weights blend across approximately two reference pixels at footprint edges.

use sr_core::{NoiseModel, Plane, RawFrame, Result, SrError, WarpField, config::RobustnessConfig};
use sr_quality::photometry::PhotometricMatch;

use crate::robustness::{RobustnessMaps, build_maps_with_frame_noise};

/// Build independent rejection consensuses for compatible-PSF cohorts, then
/// retain the finest cohort with native geometric guide coverage at each
/// reference site, independently of rejection. Broader cohorts contribute only
/// outside finer footprints. Rejected fine samples remain rejected: this may
/// leave genuine holes instead of silently substituting a broader PSF.
/// `psf_hfd` must use common reference-pixel units; inactive dummy values are
/// ignored. Cohorts are sorted by HFD and bounded relative to their minimum,
/// preventing chained pairwise tolerances from admitting arbitrarily broad PSFs.
///
/// Rejection must be enabled to supply the per-frame admission maps.
/// Existing burst APIs and defaults are unchanged. The single-cohort case is
/// exactly `build_maps_with_frame_noise`. Multiple cohorts retain final maps,
/// one selected luma, one covered byte per guide pixel, and at most one cohort's
/// working buffers/luma; no collection of full cohort consensuses is retained.
#[allow(clippy::too_many_arguments)]
pub fn build_maps_with_psf_cohorts(
    frames: &[RawFrame],
    warps: &[WarpField],
    reference: usize,
    active: &[bool],
    photometry: &[PhotometricMatch],
    noise: &NoiseModel,
    registration_sigma: f32,
    cfg: &RobustnessConfig,
    typical: &[[f32; 3]],
    psf_hfd: &[f32],
    relative_tolerance: f32,
) -> Result<RobustnessMaps> {
    if active.len() != frames.len()
        || psf_hfd.len() != frames.len()
        || !(0.0..=0.25).contains(&relative_tolerance)
        || !cfg.enabled
        || !active.iter().any(|v| *v)
        || psf_hfd
            .iter()
            .zip(active)
            .any(|(&hfd, &used)| used && (!hfd.is_finite() || hfd <= 0.))
    {
        return Err(SrError::Input(
            "PSF cohorts require enabled rejection, active frames, matching HFDs, positive finite active HFDs and tolerance in 0..=0.25".into(),
        ));
    }
    let mut ordered: Vec<_> = (0..frames.len()).filter(|&i| active[i]).collect();
    ordered.sort_by(|&a, &b| psf_hfd[a].total_cmp(&psf_hfd[b]).then(a.cmp(&b)));
    let mut cohorts: Vec<Vec<usize>> = Vec::new();
    for i in ordered {
        if cohorts
            .last()
            .is_none_or(|group| psf_hfd[i] > psf_hfd[group[0]] * (1. + relative_tolerance))
        {
            cohorts.push(Vec::new());
        }
        cohorts.last_mut().unwrap().push(i);
    }
    if cohorts.len() == 1 {
        return build_maps_with_frame_noise(
            frames,
            warps,
            reference,
            active,
            photometry,
            noise,
            registration_sigma,
            cfg,
            typical,
        );
    }

    let mut selected: Option<RobustnessMaps> = None;
    let mut covered = Vec::<u8>::new();
    for group in cohorts {
        let mut cohort_active = vec![false; frames.len()];
        for &i in &group {
            cohort_active[i] = true;
        }
        let mut current = build_maps_with_frame_noise(
            frames,
            warps,
            reference,
            &cohort_active,
            photometry,
            noise,
            registration_sigma,
            cfg,
            typical,
        )?;
        let count = current.width * current.height;
        let guide_width = current.width;
        let has_native_support = |p: usize| {
            let (rx, ry) = (
                (p % guide_width) as f32 * 2. + 0.5,
                (p / guide_width) as f32 * 2. + 0.5,
            );
            group
                .iter()
                .any(|&i| native_guide_covers(&frames[i], &warps[i], rx, ry))
        };
        if let Some(result) = &mut selected {
            let candidate_luma = current
                .consensus_luma
                .as_ref()
                .expect("enabled mono consensus");
            let result_luma = result
                .consensus_luma
                .as_mut()
                .expect("enabled mono consensus");
            for (p, already_covered) in covered.iter_mut().enumerate() {
                if *already_covered != 0 {
                    for &i in &group {
                        current.maps[i].data[p] = 0;
                    }
                } else if has_native_support(p) {
                    *already_covered = 1;
                    result_luma.data[p] = candidate_luma.data[p];
                }
            }
            for i in group {
                result.maps[i] = std::mem::replace(&mut current.maps[i], Plane::new(0, 0));
            }
        } else {
            covered = (0..count)
                .map(|p| u8::from(has_native_support(p)))
                .collect();
            // Uncovered sites must not retain the finer cohort's luma when
            // a broader cohort subsequently supplies the accepted scene.
            let luma = current
                .consensus_luma
                .as_mut()
                .expect("enabled mono consensus");
            for (p, &has_support) in covered.iter().enumerate() {
                if has_support == 0 {
                    luma.data[p] = 0.;
                }
            }
            selected = Some(current);
        }
    }
    let mut result = selected.expect("at least one active cohort");
    let cut = cfg.reject_below * 255.;
    for (i, map) in result.maps.iter().enumerate() {
        if !map.data.is_empty() {
            let suppressed = map
                .data
                .iter()
                .filter(|&&v| v == 0 || (v as f32) < cut)
                .count();
            result.rejected_fraction[i] = suppressed as f32 / map.data.len() as f32;
        }
    }
    Ok(result)
}

fn native_guide_covers(frame: &RawFrame, warp: &WarpField, rx: f32, ry: f32) -> bool {
    let Some((sx, sy)) = warp.inverse_map(rx, ry) else {
        return false;
    };
    // Exact convention used by RawFrame::structure_guide_rgb and robustness:
    // each guide cell averages a native 2x2 block centered at (2*g + 0.5).
    // Odd trailing detector rows/columns do not form another complete cell.
    let (gw, gh) = (frame.width / 2, frame.height / 2);
    let (tx, ty) = ((sx - 0.5) * 0.5, (sy - 0.5) * 0.5);
    gw > 0
        && gh > 0
        && tx.is_finite()
        && ty.is_finite()
        && tx >= 0.
        && ty >= 0.
        && tx <= (gw - 1) as f32
        && ty <= (gh - 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernelField, MergeInputs, merge::reconstruct_mono_tile};
    use sr_core::{
        CfaPattern, DefectMask, FrameMetadata, GlobalTransform, NoiseSource, ReconstructionConfig,
        Rect, SamplePlane, geometry::RadialChroma,
    };

    struct Fixture {
        frames: Vec<RawFrame>,
        warps: Vec<WarpField>,
        photo: Vec<PhotometricMatch>,
        hfd: Vec<f32>,
    }
    fn fixture(star: bool) -> Fixture {
        fixture_count(star, 4)
    }
    fn fixture_count(star: bool, count_per_rig: usize) -> Fixture {
        let mut out = Fixture {
            frames: vec![],
            warps: vec![],
            photo: vec![],
            hfd: vec![],
        };
        for rig in 0..2 {
            let (size, origin, sigma) = if rig == 0 {
                (96, 0., 1.5f32)
            } else {
                (64, 16., 1.1f32)
            };
            for (dx, dy) in [(0., 0.), (0.25, 0.5), (0.5, 0.25), (0.75, 0.75)]
                .into_iter()
                .cycle()
                .take(count_per_rig)
            {
                let values = (0..size * size)
                    .map(|i| {
                        let (x, y) = (
                            (i % size) as f32 + origin + dx,
                            (i / size) as f32 + origin + dy,
                        );
                        let sky = 0.1 + 0.0001 * x + 0.0002 * y;
                        sky + if star {
                            (-((x - 47.8).powi(2) + (y - 48.2).powi(2)) / (2. * sigma * sigma))
                                .exp()
                                / (2. * std::f32::consts::PI * sigma * sigma)
                        } else {
                            0.
                        }
                    })
                    .collect();
                out.frames.push(RawFrame {
                    width: size,
                    height: size,
                    samples: SamplePlane::from_normalised(size, size, values),
                    cfa: CfaPattern::MONO,
                    defects: DefectMask::none(size, size),
                    noise: NoiseModel::new(0., 1e-6, NoiseSource::Measured),
                    metadata: FrameMetadata::default(),
                });
                out.warps
                    .push(WarpField::global_only(GlobalTransform::translation(
                        origin + dx,
                        origin + dy,
                    )));
                out.photo.push(PhotometricMatch::IDENTITY);
                out.hfd.push(sigma * 2.35482);
            }
        }
        out
    }
    fn gated(
        f: &Fixture,
        hfd: &[f32],
        tolerance: f32,
        cfg: &RobustnessConfig,
    ) -> Result<RobustnessMaps> {
        build_maps_with_psf_cohorts(
            &f.frames,
            &f.warps,
            0,
            &vec![true; f.frames.len()],
            &f.photo,
            &f.frames[0].noise,
            0.5,
            cfg,
            &vec![[0.; 3]; f.frames.len()],
            hfd,
            tolerance,
        )
    }
    fn reconstruct(f: &Fixture, maps: &RobustnessMaps, weights: &[f32]) -> Vec<f32> {
        let cfg = ReconstructionConfig {
            scale: 1.,
            ..Default::default()
        };
        let kernels = KernelField::isotropic(48, 48, 0.16, cfg.kernel.radius, 2);
        let input = MergeInputs {
            frames: &f.frames,
            warps: &f.warps,
            reference: 0,
            photometry: &f.photo,
            noise: f.frames[0].noise,
            robustness: maps,
            kernels: &kernels,
            frame_weight: weights,
            lucky: None,
            chroma: RadialChroma::identity(),
        };
        reconstruct_mono_tile(
            &input,
            &cfg,
            (0., 0.),
            Rect::new(0, 0, 96, 96),
            &vec![[0.1; 3]; f.frames.len()],
        )
        .unwrap()
        .values
    }

    #[test]
    fn fine_stars_match_fine_only_while_coarse_outskirts_remain_covered() {
        let f = fixture(true);
        let cfg = RobustnessConfig::default();
        let maps = gated(&f, &f.hfd, 0.05, &cfg).unwrap();
        let fine = build_maps_with_frame_noise(
            &f.frames,
            &f.warps,
            0,
            &[false, false, false, false, true, true, true, true],
            &f.photo,
            &f.frames[0].noise,
            0.5,
            &cfg,
            &[[0.; 3]; 8],
        )
        .unwrap();
        let image = reconstruct(&f, &maps, &[1.; 8]);
        let reference = reconstruct(&f, &fine, &[0., 0., 0., 0., 1., 1., 1., 1.]);
        for y in 36..60 {
            for x in 36..60 {
                assert!((image[y * 96 + x] - reference[y * 96 + x]).abs() < 1e-7);
            }
        }
        for (x, y) in [(8, 8), (88, 8), (8, 88), (88, 88)] {
            assert!(image[y * 96 + x].is_finite());
        }
        for y in 16..32 {
            for x in 16..32 {
                assert_eq!(
                    maps.consensus_luma.as_ref().unwrap().data[y * 48 + x],
                    fine.consensus_luma.as_ref().unwrap().data[y * 48 + x]
                );
            }
        }
        for i in 0..4 {
            assert_eq!(maps.at(i, 48.5, 48.5), 0.);
        }
        for i in 4..8 {
            assert_eq!(maps.at(i, 48.5, 48.5), 1.);
        }
    }

    #[test]
    fn changing_cohort_at_footprint_edge_does_not_step_a_linear_sky() {
        let f = fixture(false);
        let maps = gated(&f, &f.hfd, 0.05, &RobustnessConfig::default()).unwrap();
        let image = reconstruct(&f, &maps, &[1.; 8]);
        let mut worst = 0f32;
        for y in 8..88 {
            for x in 8..88 {
                let value = image[y * 96 + x];
                assert!(value.is_finite(), "uncovered site {x},{y}");
                worst = worst.max((value - (0.1 + 0.0001 * x as f32 + 0.0002 * y as f32)).abs());
            }
        }
        assert!(worst < 5e-6, "cohort boundary creates sky error {worst}");
    }

    #[test]
    fn one_compatible_cohort_is_exactly_the_existing_noise_api() {
        let f = fixture(true);
        let cfg = RobustnessConfig::default();
        let hfd = [2., 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07];
        let selected = gated(&f, &hfd, 0.05, &cfg).unwrap();
        let existing = build_maps_with_frame_noise(
            &f.frames,
            &f.warps,
            0,
            &[true; 8],
            &f.photo,
            &f.frames[0].noise,
            0.5,
            &cfg,
            &[[0.; 3]; 8],
        )
        .unwrap();
        assert_eq!(selected.maps, existing.maps);
        assert_eq!(selected.rejected_fraction, existing.rejected_fraction);
        assert_eq!(selected.consensus_luma, existing.consensus_luma);
    }

    #[test]
    fn invalid_psf_inputs_fail_and_inactive_dummy_hfd_is_ignored() {
        let f = fixture(false);
        let cfg = RobustnessConfig::default();
        for tolerance in [-0.01, 0.251, f32::NAN, f32::INFINITY] {
            assert!(gated(&f, &f.hfd, tolerance, &cfg).is_err());
        }
        assert!(gated(&f, &f.hfd[..7], 0.05, &cfg).is_err());
        for bad in [0., -1., f32::NAN, f32::INFINITY] {
            let mut hfd = f.hfd.clone();
            hfd[0] = bad;
            assert!(gated(&f, &hfd, 0.05, &cfg).is_err());
        }
        let disabled = RobustnessConfig {
            enabled: false,
            ..Default::default()
        };
        assert!(gated(&f, &f.hfd, 0.05, &disabled).is_err());
        let mut hfd = f.hfd.clone();
        hfd[0] = 0.;
        assert!(
            build_maps_with_psf_cohorts(
                &f.frames,
                &f.warps,
                0,
                &[false, true, true, true, true, true, true, true],
                &f.photo,
                &f.frames[0].noise,
                0.5,
                &cfg,
                &[[0.; 3]; 8],
                &hfd,
                0.05
            )
            .is_ok()
        );
    }

    #[test]
    fn compatibility_is_bounded_by_cohort_minimum_not_chained_neighbors() {
        let f = fixture(false);
        let hfd = [3.5, 3.5, 3.5, 3.5, 2., 2.09, 2.18, 2.27];
        let maps = gated(&f, &hfd, 0.05, &RobustnessConfig::default()).unwrap();
        for i in [4, 5] {
            assert_eq!(maps.at(i, 48.5, 48.5), 1.);
        }
        for i in [0, 1, 2, 3, 6, 7] {
            assert_eq!(maps.at(i, 48.5, 48.5), 0.);
        }
    }

    #[test]
    fn six_fine_frames_do_not_claim_coarse_only_outskirts() {
        let f = fixture_count(false, 6);
        let maps = gated(&f, &f.hfd, 0.05, &RobustnessConfig::default()).unwrap();
        for (x, y) in [(8.5, 8.5), (88.5, 8.5), (8.5, 88.5), (88.5, 88.5)] {
            for i in 6..12 {
                assert_eq!(
                    maps.at(i, x, y),
                    0.,
                    "fine frame {i} has phantom outskirts coverage"
                );
            }
            for i in 0..6 {
                assert_eq!(
                    maps.at(i, x, y),
                    1.,
                    "coarse outskirts incorrectly suppressed"
                );
            }
        }
        let image = reconstruct(&f, &maps, &[1.; 12]);
        for y in 8..88 {
            for x in 8..88 {
                assert!(
                    image[y * 96 + x].is_finite(),
                    "PSF selection left a hole at {x},{y}"
                );
            }
        }
    }

    #[test]
    fn rejected_fine_star_sites_never_admit_broad_interior_fallback() {
        let mut f = fixture_count(true, 2);
        // One member of the fine pair has a compact transient over the star.
        // The pair cannot resolve that disagreement, so some interior guide
        // sites reject both frames even though both detectors cover the sky.
        let frame = &mut f.frames[3];
        let values = (0..64 * 64)
            .map(|p| {
                let (x, y) = ((p % 64) as f32 + 16.25, (p / 64) as f32 + 16.5);
                let sky = 0.1 + 0.0001 * x + 0.0002 * y;
                let star = (-((x - 47.8).powi(2) + (y - 48.2).powi(2)) / (2. * 1.1f32.powi(2)))
                    .exp()
                    / (2. * std::f32::consts::PI * 1.1f32.powi(2));
                sky + star
                    + if (31..33).contains(&(p % 64)) && (31..33).contains(&(p / 64)) {
                        0.6
                    } else {
                        0.
                    }
            })
            .collect();
        frame.samples = SamplePlane::from_normalised(64, 64, values);
        let cfg = RobustnessConfig::default();
        let fine = build_maps_with_frame_noise(
            &f.frames,
            &f.warps,
            0,
            &[false, false, true, true],
            &f.photo,
            &f.frames[0].noise,
            0.5,
            &cfg,
            &[[0.; 3]; 4],
        )
        .unwrap();
        let maps = gated(&f, &f.hfd, 0.05, &cfg).unwrap();
        let mut rejected_sites = 0;
        for gy in 20..28 {
            for gx in 20..28 {
                let p = gy * 48 + gx;
                if fine.maps[2].data[p] == 0 && fine.maps[3].data[p] == 0 {
                    rejected_sites += 1;
                    for i in 0..2 {
                        assert_eq!(
                            maps.maps[i].data[p], 0,
                            "broad frame fills a rejected fine interior site"
                        );
                    }
                    assert_eq!(
                        maps.consensus_luma.as_ref().unwrap().data[p],
                        fine.consensus_luma.as_ref().unwrap().data[p]
                    );
                }
            }
        }
        assert!(
            rejected_sites > 0,
            "fixture must contain rejected fine interior sites"
        );
        let mixed = reconstruct(&f, &maps, &[100., 100., 1., 1.]);
        let reference = reconstruct(&f, &fine, &[0., 0., 1., 1.]);
        for y in 36..60 {
            for x in 36..60 {
                let p = y * 96 + x;
                assert_eq!(
                    mixed[p].is_nan(),
                    reference[p].is_nan(),
                    "changed hole at {x},{y}"
                );
                if mixed[p].is_finite() {
                    assert!(
                        (mixed[p] - reference[p]).abs() < 1e-7,
                        "fine-only mismatch at {x},{y}"
                    );
                }
            }
        }
        for i in 0..2 {
            assert!(maps.at(i, 8.5, 8.5) > 0.99);
        }
    }
}
