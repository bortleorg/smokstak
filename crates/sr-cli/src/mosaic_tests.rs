//! End-to-end bounded mono reconstruction and publication contract tests.
use super::*;

#[test]
fn missing_global_cohort_minimum_cannot_merge_tile_psf_groups() {
    assert_eq!(
        global_psf_cohort_representatives(&[3.0, 3.2, 3.4]),
        vec![3.0, 3.0, 3.4]
    );
    // Missing frames from any of several real-range cohorts cannot change the
    // representatives seen by the tile-local grouping operation.
    let mixed_hfd = [2.778, 3.0, 3.2, 3.4, 3.602, 4.010, 4.3, 4.620];
    let fixed = global_psf_cohort_representatives(&mixed_hfd);
    for subset in 1..(1usize << mixed_hfd.len()) {
        let local: Vec<_> = fixed
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| (subset & (1 << i) != 0).then_some(value))
            .collect();
        assert_eq!(global_psf_cohort_representatives(&local), local);
    }
    // Global groups are [2.0, 2.18] and [2.30]. Without the first frame,
    // regrouping actual local HFDs incorrectly merges 2.18 and 2.30.
    let hfd = [2.30, 2.0, 2.18];
    let representatives = global_psf_cohort_representatives(&hfd);
    assert_eq!(representatives, vec![2.30, 2.0, 2.0]);
    let noise = NoiseModel::new(0.0, 1.0, NoiseSource::Manual);
    let mut fine = empty_frame(32, 32, noise);
    fine.samples = SamplePlane::from_normalised(32, 32, vec![0.1; 32 * 32]);
    let mut coarse = empty_frame(32, 32, noise);
    coarse.samples = SamplePlane::from_normalised(32, 32, vec![0.3; 32 * 32]);
    // This is the actual tile frame collection: dummy reference plus only
    // locally overlapping fine-member and coarse-member exposures.
    let frames = [empty_frame(32, 32, noise), fine, coarse];
    let warps = vec![WarpField::identity(); 3];
    let photo = vec![PhotometricMatch::IDENTITY; 3];
    let cfg = sr_core::RobustnessConfig::default();
    let build = |values: &[f32]| {
        sr_reconstruct::mosaic_quality::build_maps_with_psf_cohorts(
            &frames,
            &warps,
            0,
            &[false, true, true],
            &photo,
            &noise,
            0.5,
            &cfg,
            &[[0.; 3]; 3],
            values,
            PSF_RELATIVE_TOLERANCE,
        )
        .unwrap()
    };
    let regrouped = build(&[0.0, hfd[2], hfd[0]]);
    assert!(
        regrouped.maps[2].data[8 * 16 + 8] > 0,
        "fixture must expose coarse admission under local regrouping"
    );
    let frozen = build(&[0.0, representatives[2], representatives[0]]);
    assert!(frozen.maps[1].data[8 * 16 + 8] > 0);
    assert_eq!(
        frozen.maps[2].data[8 * 16 + 8],
        0,
        "missing sharpest global frame must not admit the next PSF cohort"
    );
}

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "smokstak-mosaic-test-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn scene(x: f64, y: f64) -> f64 {
    let mut value = 12000. + 13. * x + 9. * y + 180. * (x * 0.19).sin() * (y * 0.11).cos();
    for (sx, sy, amplitude) in [(63.5, 63., 8000.), (128., 96., 6500.), (94., 128., 7000.)] {
        value += amplitude * (-((x - sx).powi(2) + (y - sy).powi(2)) / 8.).exp();
    }
    value
}

fn integer_fits(path: &Path, width: usize, height: usize, shift: [f64; 2]) {
    integer_fits_values(path, width, height, |x, y| {
        scene(x as f64 + shift[0], y as f64 + shift[1])
    });
}

fn integer_fits_values(
    path: &Path,
    width: usize,
    height: usize,
    value_at: impl Fn(usize, usize) -> f64,
) {
    let mut bytes = Vec::new();
    for card in [
        "SIMPLE  =                    T".to_owned(),
        "BITPIX  =                   16".to_owned(),
        "NAXIS   =                    2".to_owned(),
        format!("NAXIS1  = {width:>20}"),
        format!("NAXIS2  = {height:>20}"),
        "BZERO   =                32768".to_owned(),
        "BSCALE  =                    1".to_owned(),
        "FILTER  = 'Ha'".to_owned(),
        "ROWORDER= 'TOP-DOWN'".to_owned(),
        "END".to_owned(),
    ] {
        bytes.extend_from_slice(format!("{card:<80}").as_bytes());
    }
    bytes.resize(2880, b' ');
    for y in 0..height {
        for x in 0..width {
            // Force the same unsigned-16 normalization without affecting the
            // tested interior or stars. Saturated samples are excluded by merge.
            let value = if x == 0 && y == 0 {
                65535
            } else {
                value_at(x, y).round() as i32
            };
            bytes.extend_from_slice(&((value - 32768) as i16).to_be_bytes());
        }
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    fs::write(path, bytes).unwrap();
}

fn linear_background(x: f64, y: f64) -> f64 {
    (12000. + 13. * x + 9. * y) / 65535.
}

fn photometry_scene(x: f64, y: f64) -> f64 {
    let mut value = linear_background(x, y);
    for (sx, sy, amplitude) in [(63.5, 63., 8000.), (128., 96., 6500.)] {
        value += amplitude / 65535. * (-((x - sx).powi(2) + (y - sy).powi(2)) / 8.).exp();
    }
    value
}

fn star_measurement(values: &[f32], grid: &Grid, center: (f64, f64)) -> (f64, f64) {
    let (mut flux, mut radial_moment) = (0., 0.);
    let scale = f64::from(grid.scale);
    for y in 0..grid.height {
        for x in 0..grid.width {
            let rx = f64::from(grid.origin[0]) + (x as f64 + 0.5) / scale - 0.5;
            let ry = f64::from(grid.origin[1]) + (y as f64 + 0.5) / scale - 0.5;
            let r2 = (rx - center.0).powi(2) + (ry - center.1).powi(2);
            if r2 <= 64. {
                let v = (f64::from(values[y * grid.width + x]) - linear_background(rx, ry))
                    / scale.powi(2);
                flux += v;
                radial_moment += v * r2;
            }
        }
    }
    (flux, (radial_moment / flux / 2.).sqrt())
}

#[test]
fn gain_offset_and_global_gradient_preserve_sky_and_stars_across_tiles_and_scales() {
    corrected_background_preserves_scene(false, false);
}

#[test]
fn quadratic_overlap_correction_preserves_sky_and_stars_across_tiles_and_scales() {
    corrected_background_preserves_scene(true, false);
}

#[test]
fn spatial_stellar_response_preserves_scene_across_tiles_and_scales() {
    corrected_background_preserves_scene(true, true);
}

#[test]
fn relative_log_gain_bounds_include_interior_extrema_and_reject_extrapolation() {
    let model = RelativeLogGain {
        center: [0.; 2],
        normalization_scale: 1.,
        coefficients: [0., 0., -0.1, 0.03, 0.2],
    };
    let bounds = model.log_bounds([-1., -1., 1., 1.]).unwrap();
    for iy in 0..21 {
        for ix in 0..21 {
            let value = model
                .factor_at(ix as f64 / 10. - 1., iy as f64 / 10. - 1.)
                .ln();
            assert!(value >= bounds[0] - 1e-12 && value <= bounds[1] + 1e-12);
        }
    }
    let fixture = Fixture::new();
    let mut candidate = plan(&fixture);
    candidate.frames[0].relative_log_gain = Some(model);
    assert!(
        validate(&candidate, 64, 128)
            .unwrap_err()
            .to_string()
            .contains("stellar-response")
    );
}

#[test]
fn excessive_or_overflowing_background_correction_is_not_published() {
    let fixture = Fixture::new();
    let mut candidate = plan(&fixture);
    candidate.frames[0].background_quadratic = [0.001, 0., 0.];
    let output = fixture.0.join("excessive-curvature");
    let error = build(&candidate, &output, 64, 128).unwrap_err().to_string();
    assert!(error.contains("interpolation exceeds"), "{error}");
    assert!(!output.exists());
    candidate.frames[0].background_quadratic = [0.; 3];
    candidate.frames[0].background_plane = [f32::MAX, f32::MAX];
    let error = tile_photometry(&candidate.frames[0], (100., 100.), (128, 128))
        .unwrap_err()
        .to_string();
    assert!(error.contains("overflows"), "{error}");
}

#[test]
fn supersampled_tiles_keep_reference_guide_context() {
    let fixture = Fixture::new();
    for scale in [4, 8] {
        let mut candidate = plan(&fixture);
        candidate.grid = Grid {
            origin: [70.25, 50.5],
            width: 24 * scale,
            height: 16 * scale,
            scale: scale as f32,
        };
        let mut results = Vec::new();
        for tile in [64, 128] {
            let output = fixture.0.join(format!("guide-context-{scale}-{tile}"));
            build(&candidate, &output, tile, 128).unwrap();
            results.push(read_output(
                &output.join("image.fits"),
                candidate.grid.width,
                candidate.grid.height,
            ));
        }
        assert!(results[0].iter().any(|v| v.is_finite()));
        for (a, b) in results[0].iter().zip(&results[1]) {
            // Two dithers do not populate every 8x output site. Missing support
            // must stay NaN in the same locations regardless of tile size.
            assert_eq!(a.is_finite(), b.is_finite());
            if a.is_finite() {
                assert!((a - b).abs() < 3e-7, "scale {scale}: {a} vs {b}");
            } else {
                assert!(a.is_nan() && b.is_nan());
            }
        }
    }
}

fn corrected_background_preserves_scene(quadratic: bool, spatial_gain: bool) {
    let fixture = Fixture::new();
    let mut baseline = plan(&fixture);
    baseline.grid.origin = [17.25, 9.5];
    baseline.grid.width = 192;
    baseline.grid.height = 144;
    let mut corrected = baseline.clone();
    for i in 0..2 {
        let (width, height) = [(170, 154), (190, 148)][i];
        let shift = baseline.frames[i].projection.output_center;
        let base = &mut baseline.frames[i];
        base.width = width;
        base.height = height;
        integer_fits_values(&base.path, width, height, |x, y| {
            photometry_scene(x as f64 + shift[0], y as f64 + shift[1]) * 65535.
        });
        base.bytes = fs::metadata(&base.path).unwrap().len();
        base.sha256 = source_hash(&base.path).unwrap();
        let other = &mut corrected.frames[i];
        *other = base.clone();
        other.path = fixture.0.join(format!("photometric-{i}.fits"));
        other.gain = [1.25, 0.8][i];
        if spatial_gain {
            other.relative_log_gain = Some(RelativeLogGain {
                center: [96., 72.],
                normalization_scale: 128.,
                coefficients: [
                    [0.07, -0.04, 0.09, 0.03, -0.06],
                    [-0.05, 0.08, -0.07, 0.02, 0.1],
                ][i],
            });
        }
        other.offset = [0.013, -0.019][i];
        other.background_plane = [[0.00021, -0.00016], [-0.00017, 0.00023]][i];
        if quadratic {
            other.background_quadratic = [[1e-8, -5e-9, 8e-9], [-8e-9, 6e-9, -1e-8]][i];
        }
        other.noise.beta = base.noise.beta / other.gain.powi(2);
        other.sky = (base.sky - other.offset) / other.gain;
        integer_fits_values(&other.path, width, height, |x, y| {
            let (rx, ry) = (x as f64 + shift[0], y as f64 + shift[1]);
            let field = f64::from(other.offset)
                + f64::from(other.background_plane[0]) * rx
                + f64::from(other.background_plane[1]) * ry
                + f64::from(other.background_quadratic[0]) * rx * rx
                + f64::from(other.background_quadratic[1]) * rx * ry
                + f64::from(other.background_quadratic[2]) * ry * ry;
            (photometry_scene(rx, ry) - field) / other.matched_gain(rx, ry) * 65535.
        });
        other.bytes = fs::metadata(&other.path).unwrap().len();
        other.sha256 = source_hash(&other.path).unwrap();
    }
    for scale in [1, 2] {
        for p in [&mut baseline, &mut corrected] {
            p.grid.scale = scale as f32;
            p.grid.width = 192 * scale;
            p.grid.height = 144 * scale;
        }
        let mut images = Vec::new();
        for (label, p) in [("baseline", &baseline), ("corrected", &corrected)] {
            for tile in [64, 128] {
                let out = fixture.0.join(format!("photometry-{label}-{scale}-{tile}"));
                build(p, &out, tile, 128).unwrap();
                images.push(read_output(
                    &out.join("image.fits"),
                    p.grid.width,
                    p.grid.height,
                ));
            }
        }
        let grid = &baseline.grid;
        let mut worst_tile = 0f32;
        let mut worst_correction = 0f32;
        let mut worst_background = 0f64;
        for y in 8 * scale..128 * scale {
            for x in 8 * scale..184 * scale {
                let index = y * grid.width + x;
                let rx = f64::from(grid.origin[0]) + (x as f64 + 0.5) / scale as f64 - 0.5;
                let ry = f64::from(grid.origin[1]) + (y as f64 + 0.5) / scale as f64 - 0.5;
                assert!(images.iter().all(|image| image[index].is_finite()));
                for pair in [(0, 1), (2, 3)] {
                    worst_tile =
                        worst_tile.max((images[pair.0][index] - images[pair.1][index]).abs());
                }
                worst_correction =
                    worst_correction.max((images[0][index] - images[2][index]).abs());
                if [(63.5, 63.), (128., 96.)]
                    .iter()
                    .all(|&(sx, sy)| (rx - sx).hypot(ry - sy) > 12.)
                {
                    worst_background = worst_background
                        .max((f64::from(images[2][index]) - linear_background(rx, ry)).abs());
                }
            }
        }
        println!(
            "photometry scale {scale}: max tile difference {worst_tile}, correction difference {worst_correction}, background error {worst_background}"
        );
        assert!(
            worst_tile <= 1e-5,
            "scale {scale}: tile seam difference {worst_tile}"
        );
        // Two independently rounded uint16 inputs contribute <= ~1.2 ADU
        // after gain correction; allow 2 ADU for floating-point accumulation.
        assert!(
            worst_correction <= 2. / 65535.,
            "scale {scale}: photometry error {worst_correction}"
        );
        assert!(
            worst_background <= 2. / 65535.,
            "scale {scale}: corrected linear background error {worst_background}"
        );
        for center in [(63.5, 63.), (128., 96.)] {
            let (flux, sigma) = star_measurement(&images[0], grid, center);
            let (corrected_flux, corrected_sigma) = star_measurement(&images[2], grid, center);
            assert!(
                (corrected_flux / flux - 1.).abs() < 5e-4,
                "scale {scale}: flux {flux} -> {corrected_flux}"
            );
            assert!(
                (corrected_sigma - sigma).abs() < 0.002,
                "scale {scale}: star sigma {sigma} -> {corrected_sigma}"
            );
        }
    }
}

fn plan(fixture: &Fixture) -> Plan {
    let frames = [[0., 0.], [48.375, 0.25]]
        .into_iter()
        .enumerate()
        .map(|(i, shift)| {
            let path = fixture.0.join(format!("source-{i}.fits"));
            integer_fits(&path, 180, 150, shift);
            FrameSpec {
                label: String::new(),
                group: String::new(),
                bytes: fs::metadata(&path).unwrap().len(),
                sha256: source_hash(&path).unwrap(),
                path,
                width: 180,
                height: 150,
                projection: FrameProjection {
                    center: [0., 0.],
                    normalization_scale: 1.,
                    distortion: [0.; 3],
                    homography: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                    output_center: shift,
                    output_scale: 1.,
                },
                noise: NoiseModel::new(0., 1e-6, NoiseSource::Measured),
                sky: 12000. / 65535.,
                gain: 1.,
                offset: 0.,
                background_plane: [0.; 2],
                background_quadratic: [0.; 3],
                relative_log_gain: None,
                weight: 1.,
                psf_hfd: 2.,
                registration_p50: 0.1,
                registration_p90: 0.2,
                validation_stars: 40,
            }
        })
        .collect();
    Plan {
        version: 1,
        filter: "Ha".into(),
        calibration: "uncalibrated synthetic fixture".into(),
        grid: Grid {
            origin: [0., 0.],
            width: 256,
            height: 192,
            scale: 1.,
        },
        frames,
        notes: vec![],
    }
}

fn read_output(path: &Path, width: usize, height: usize) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    let (_, offset) = sr_raw::fits::read_header(path).unwrap();
    let values: Vec<_> = bytes[offset as usize..offset as usize + width * height * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_be_bytes(*b))
        .collect();
    values
        .chunks_exact(width)
        .rev()
        .flatten()
        .copied()
        .collect()
}

#[test]
fn source_band_cache_changes_reads_not_output() {
    let fixture = Fixture::new();
    let plan = plan(&fixture);
    let cached = fixture.0.join("cached");
    let direct = fixture.0.join("direct");
    build(&plan, &cached, 64, 1024).unwrap();
    super::READ_DIRECTLY.store(true, std::sync::atomic::Ordering::SeqCst);
    let result = build(&plan, &direct, 64, 1024);
    super::READ_DIRECTLY.store(false, std::sync::atomic::Ordering::SeqCst);
    result.unwrap();
    for name in [
        "image.fits",
        "weight.fits",
        "samples.fits",
        "preview-linear.fits",
    ] {
        assert!(
            fs::read(cached.join(name)).unwrap() == fs::read(direct.join(name)).unwrap(),
            "{name} differs"
        );
    }
    let record: serde_json::Value =
        serde_json::from_slice(&fs::read(cached.join("result.json")).unwrap()).unwrap();
    assert!(
        record["source_band_cache"]["hits"].as_u64().unwrap() > 0,
        "{}",
        record["source_band_cache"]
    );
    let record: serde_json::Value =
        serde_json::from_slice(&fs::read(direct.join("result.json")).unwrap()).unwrap();
    assert_eq!(record["source_band_cache"]["misses"], 0);
}

#[test]
fn tiled_controller_preserves_seams_values_weights_counts_and_uncovered_sky() {
    let fixture = Fixture::new();
    let plan = plan(&fixture);
    let small = fixture.0.join("tile64");
    let large = fixture.0.join("tile128");
    build(&plan, &small, 64, 128).unwrap();
    build(&plan, &large, 128, 128).unwrap();
    let mut failures = Vec::new();
    for name in ["image.fits", "weight.fits", "samples.fits"] {
        let a = read_output(&small.join(name), 256, 192);
        let b = read_output(&large.join(name), 256, 192);
        let mut differing = 0;
        let mut worst = (0f32, 0, 0, 0f32, 0f32);
        for y in 8..142 {
            for x in 8..220 {
                let i = y * 256 + x;
                assert!(
                    a[i].is_finite() && b[i].is_finite(),
                    "{name}: missing covered sample {x},{y}"
                );
                let tolerance = if name == "samples.fits" {
                    0.
                } else {
                    1e-5 * a[i].abs().max(1.)
                };
                let error = (a[i] - b[i]).abs();
                if error > tolerance {
                    differing += 1;
                    if error > worst.0 {
                        worst = (error, x, y, a[i], b[i]);
                    }
                }
            }
        }
        if differing > 0 {
            failures.push(format!("{name}: {differing} tile-dependent samples; worst {worst:?} (error,x,y,tile64,tile128)"));
        }
        for (x, y) in [(250, 50), (70, 180), (250, 180)] {
            let i = y * 256 + x;
            if name == "image.fits" {
                assert!(a[i].is_nan() && b[i].is_nan());
            } else {
                assert_eq!(a[i], 0.);
                assert_eq!(b[i], 0.);
            }
        }
    }
    assert!(small.join("result.json").is_file());
    assert!(large.join("result.json").is_file());
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(small.join("result.json")).unwrap()).unwrap();
    assert_eq!(manifest["experimental"], true);
    assert!(
        (manifest["policies"]["psf_hfd_relative_tolerance"]
            .as_f64()
            .unwrap()
            - 0.1)
            .abs()
            < 1e-7
    );
    assert!(
        (manifest["policies"]["detector_edge_feather_fraction"]
            .as_f64()
            .unwrap()
            - 0.1)
            .abs()
            < 1e-7
    );
    assert!(
        manifest["policies"]["rejection"]
            .as_str()
            .unwrap()
            .contains("per-frame photometrically scaled noise")
    );
    assert!(
        manifest["limitations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_str().unwrap().contains("Global stellar HFD"))
    );
    assert!(
        manifest["limitations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_str().unwrap().contains("Uncalibrated detector"))
    );
    let sentinel = fs::read(small.join("image.fits")).unwrap();
    assert!(
        build(&plan, &small, 64, 128)
            .unwrap_err()
            .to_string()
            .contains("already exists")
    );
    assert_eq!(fs::read(small.join("image.fits")).unwrap(), sentinel);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn invalid_registration_and_failed_build_never_publish_completed_output() {
    let fixture = Fixture::new();
    let mut plan = plan(&fixture);
    let output = fixture.0.join("rejected");
    plan.frames[1].registration_p90 = 2.;
    assert!(
        build(&plan, &output, 64, 128)
            .unwrap_err()
            .to_string()
            .contains("registration gate")
    );
    assert!(!output.exists());
    assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 2);
    plan.frames[1].registration_p90 = 0.2;
    // A dimension change passes header/filter preflight but fails preparation,
    // after staging begins. It must retain diagnostics, never publish success.
    plan.frames[1].width += 1;
    assert!(
        build(&plan, &output, 64, 128)
            .unwrap_err()
            .to_string()
            .contains("dimensions changed")
    );
    assert!(!output.exists());
    let partials: Vec<_> = fs::read_dir(&fixture.0)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir())
        .collect();
    assert_eq!(partials.len(), 1);
    assert!(partials[0].join("FAILED.txt").is_file());
    assert!(!partials[0].join("result.json").exists());
}

#[test]
fn malformed_noise_and_registration_fail_before_staging() {
    let fixture = Fixture::new();
    let baseline = plan(&fixture);
    validate(&baseline, 64, 128).unwrap();
    let output = fixture.0.join("invalid-model");
    for (alpha, beta, gain) in [
        (-1., 1e-6, 1.),
        (f32::NAN, 1e-6, 1.),
        (f32::INFINITY, 1e-6, 1.),
        (0., -1., 1.),
        (0., f32::NAN, 1.),
        (0., f32::INFINITY, 1.),
        (0., 0., 1.),
        (1e-6, 0., 1.),
        (f32::MAX, f32::MAX, 1.),
        (0., 1e-6, f32::MAX),
        (0., 1e-6, f32::MIN_POSITIVE),
    ] {
        let mut candidate = baseline.clone();
        candidate.frames[1].noise.alpha = alpha;
        candidate.frames[1].noise.beta = beta;
        candidate.frames[1].gain = gain;
        let error = build(&candidate, &output, 64, 128).unwrap_err().to_string();
        assert!(
            error.contains("noise model"),
            "unexpected rejection: {error}"
        );
        assert!(!output.exists());
    }
    for (p50, p90) in [
        (-0.1, 0.2),
        (0.1, -0.2),
        (0.2, 0.1),
        (f32::NAN, 0.2),
        (0.1, f32::INFINITY),
    ] {
        let mut candidate = baseline.clone();
        candidate.frames[1].registration_p50 = p50;
        candidate.frames[1].registration_p90 = p90;
        assert!(
            build(&candidate, &output, 64, 128)
                .unwrap_err()
                .to_string()
                .contains("registration gate")
        );
        assert!(!output.exists());
    }
    assert_eq!(
        fs::read_dir(&fixture.0).unwrap().count(),
        2,
        "invalid metadata created staging artifacts"
    );
}

#[test]
fn duplicate_canonical_source_aliases_cannot_manufacture_exposure_count() {
    let fixture = Fixture::new();
    let mut candidate = plan(&fixture);
    candidate.frames[1] = candidate.frames[0].clone();
    let output = fixture.0.join("duplicate-output");
    assert!(
        build(&candidate, &output, 64, 128)
            .unwrap_err()
            .to_string()
            .contains("duplicate source")
    );
    fs::create_dir(fixture.0.join("aliases")).unwrap();
    candidate.frames[1].path = fixture.0.join("aliases").join("..").join("source-0.fits");
    assert!(
        build(&candidate, &output, 64, 128)
            .unwrap_err()
            .to_string()
            .contains("duplicate source")
    );
    assert!(!output.exists());
    assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 3);
    let copied = fixture.0.join("copied-exposure.fits");
    fs::copy(&candidate.frames[0].path, &copied).unwrap();
    candidate.frames[1].path = copied;
    assert!(
        build(&candidate, &output, 64, 128)
            .unwrap_err()
            .to_string()
            .contains("duplicate source content")
    );
    assert!(!output.exists());
}

#[test]
fn tile_steps_preserve_the_same_reference_guide_phase() {
    let fixture = Fixture::new();
    let mut candidate = plan(&fixture);
    for (tile, scale) in [(64, 2.), (66, 1.5), (64, 1.), (128, 8.)] {
        candidate.grid.scale = scale;
        validate(&candidate, tile, 128).unwrap();
    }
    for (tile, scale) in [(66, 2.), (64, 1.5), (66, 8.)] {
        candidate.grid.scale = scale;
        let error = validate(&candidate, tile, 128).unwrap_err().to_string();
        assert!(
            error.contains("reference guide lattice"),
            "unexpected error: {error}"
        );
    }
    // This represented scale rounds a f32 quotient onto 66, but its true
    // reference step is not even. Validation must not accept that rounding.
    let scale = 64f32 / 66.;
    // Use a larger tile to keep the example's scale inside the allowed range.
    candidate.grid.scale = scale * 2.;
    assert_eq!(128f32 / candidate.grid.scale, 66.);
    assert_ne!(128f64 / f64::from(candidate.grid.scale), 66.);
    assert!(
        validate(&candidate, 128, 128)
            .unwrap_err()
            .to_string()
            .contains("reference guide lattice")
    );
}

#[test]
fn fractional_scale_preserves_outputs_across_compatible_tile_sizes() {
    let fixture = Fixture::new();
    let mut candidate = plan(&fixture);
    candidate.grid = Grid {
        origin: [-3.25, 2.5],
        width: 384,
        height: 288,
        scale: 1.5,
    };
    let small = fixture.0.join("fractional-66");
    let large = fixture.0.join("fractional-132");
    build(&candidate, &small, 66, 128).unwrap();
    build(&candidate, &large, 132, 128).unwrap();
    let mut failures = Vec::new();
    for name in ["image.fits", "weight.fits", "samples.fits"] {
        let a = read_output(&small.join(name), 384, 288);
        let b = read_output(&large.join(name), 384, 288);
        let (mut finite, mut missing, mut differing) = (0, 0, 0);
        let mut worst = (0f32, 0usize, 0f32, 0f32);
        for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x.is_nan(),
                y.is_nan(),
                "{name}: support differs at {},{}",
                i % 384,
                i / 384
            );
            if x.is_nan() {
                missing += 1;
                continue;
            }
            assert!(
                x.is_finite() && y.is_finite(),
                "{name}: nonfinite weight/value at index {i}"
            );
            finite += 1;
            let tolerance = if name == "samples.fits" {
                0.
            } else {
                1e-5 * x.abs().max(1.)
            };
            let error = (x - y).abs();
            if error > tolerance {
                differing += 1;
                if error > worst.0 {
                    worst = (error, i, x, y);
                }
            }
        }
        if name == "image.fits" {
            assert!(
                finite > 50_000 && missing > 1_000,
                "fixture must include covered and uncovered sky"
            );
        }
        if differing > 0 {
            failures.push(format!(
                "{name}: {differing} differing pixels; worst {worst:?} (error,index,tile66,tile132)"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn oversized_encoded_plan_fails_before_parsing_or_staging() {
    let fixture = Fixture::new();
    let path = fixture.0.join("oversized.json");
    let file = fs::File::create(&path).unwrap();
    // A sparse, invalid JSON file proves rejection happens at the encoded-size
    // boundary, without deserializing or needing any input image paths.
    file.set_len(MAX_PLAN_BYTES as u64 + 1).unwrap();
    drop(file);
    let output = fixture.0.join("unpublished");
    let args = Args {
        command: Command::Build {
            plan_sha256: None,
            plan: path,
            output: output.clone(),
            tile: 64,
            memory_mb: 64,
            experimental: true,
        },
    };
    let error = run(&args).unwrap_err().to_string();
    assert!(
        error.contains("encoded mosaic plan exceeds 2 MiB"),
        "unexpected error: {error}"
    );
    assert!(!output.exists());
    assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
}

#[test]
fn programmatic_plan_metadata_is_bounded_before_source_io() {
    let fixture = Fixture::new();
    let mut baseline = plan(&fixture);
    baseline.frames[0].path = fixture.0.join("does-not-exist.fits");
    let output = fixture.0.join("unpublished");
    let mut cases = Vec::new();
    let mut oversized_string = baseline.clone();
    oversized_string
        .notes
        .push("x".repeat(MAX_PLAN_STRING_BYTES + 1));
    cases.push((oversized_string, "metadata string"));
    let mut too_many_notes = baseline.clone();
    too_many_notes.notes = vec![String::new(); MAX_PLAN_NOTES + 1];
    cases.push((too_many_notes, "too many notes"));
    let mut aggregate = baseline.clone();
    aggregate.notes = vec!["x".repeat(MAX_PLAN_STRING_BYTES); 17];
    cases.push((aggregate, "metadata exceeds 1 MiB"));
    let mut reserved = baseline.clone();
    let mut note = String::with_capacity(MAX_PLAN_BYTES);
    note.push('x');
    reserved.notes.push(note);
    cases.push((reserved, "allocation exceeds 2 MiB"));
    let mut too_many_frames = baseline.clone();
    too_many_frames.frames = vec![baseline.frames[0].clone(); 513];
    cases.push((too_many_frames, "2..512 frames"));
    for (candidate, message) in cases {
        let error = build(&candidate, &output, 64, 64).unwrap_err().to_string();
        assert!(error.contains(message), "unexpected error: {error}");
        assert!(!output.exists());
    }
    assert_eq!(
        fs::read_dir(&fixture.0).unwrap().count(),
        2,
        "invalid metadata created staging files"
    );
}

#[test]
fn bounded_reader_accepts_full_size_plan_metadata_without_source_io() {
    let fixture = Fixture::new();
    let baseline = plan(&fixture);
    for count in [240, 512] {
        let mut candidate = baseline.clone();
        candidate.frames = (0..count)
            .map(|i| {
                let mut frame = baseline.frames[i % 2].clone();
                frame.path = fixture
                    .0
                    .join(format!("unavailable-camera-light-{i:04}.fits"));
                frame
            })
            .collect();
        candidate.notes = (0..count)
            .map(|i| format!("Exposure {i}: geometry independently validated; raw mono Ha."))
            .collect();
        let path = fixture.0.join(format!("full-{count}.json"));
        serde_json::to_writer_pretty(fs::File::create(&path).unwrap(), &candidate).unwrap();
        assert!(fs::metadata(&path).unwrap().len() < MAX_PLAN_BYTES as u64);
        let decoded = read_plan(&path).unwrap();
        assert_eq!(decoded.frames.len(), count);
        assert_eq!(decoded.notes, candidate.notes);
        assert!(plan_memory_bytes(&decoded).unwrap() < MAX_PLAN_BYTES);
    }
}

#[test]
#[ignore = "requires local full-data evaluation plans in reports"]
fn bounded_reader_accepts_real_full_data_plans() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (name, expected_frames) in [
        ("ha-mosaic-full-plan.json", 134),
        ("ha-mixed-full-plan.json", 106),
    ] {
        let path = root.join("reports").join(name);
        let decoded = read_plan(&path).unwrap();
        assert_eq!(decoded.frames.len(), expected_frames, "{name}");
        assert!(plan_memory_bytes(&decoded).unwrap() < MAX_PLAN_BYTES);
    }
}

#[test]
fn display_stretch_excludes_gaps_and_preserves_linear_copy() {
    let linear = Plane::from_vec(
        1002,
        1,
        (0..1002)
            .map(|i| if i < 2 { 0. } else { 0.1 + i as f32 * 0.0001 })
            .collect(),
    );
    let original = linear.data.to_vec();
    let mut display = linear.clone();
    let mut counts = vec![1; 1002];
    counts[..2].fill(0);
    let range = stretch_mosaic_preview(&mut display, &counts);
    assert!(range[0] > 0.1 && range[1] > range[0]);
    assert_eq!(&display.data[..2], &[0., 0.]);
    assert!(
        display
            .data
            .iter()
            .all(|v| v.is_finite() && (0. ..=1.).contains(v))
    );
    assert_eq!(linear.data.as_ref(), original.as_slice());
    assert!(display.data[500] > 0.5);
    counts.fill(0);
    stretch_mosaic_preview(&mut display, &counts);
    assert!(display.data.iter().all(|v| *v == 0.));
}

#[test]
fn changed_reviewed_plan_is_rejected_before_publishing() {
    let fixture = Fixture::new();
    let candidate = plan(&fixture);
    let path = fixture.0.join("reviewed.json");
    let bytes = serde_json::to_vec(&candidate).unwrap();
    fs::write(&path, &bytes).unwrap();
    let (_, digest) = read_review_plan_fingerprinted(&path, 64, 128).unwrap();
    assert_eq!(digest, format!("{:x}", Sha256::digest(&bytes)));
    let mut changed = candidate;
    changed.grid.origin[0] += 1.;
    fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
    let output = fixture.0.join("must-not-exist");
    let args = Args {
        command: Command::Build {
            plan: path,
            output: output.clone(),
            tile: 64,
            memory_mb: 128,
            experimental: true,
            plan_sha256: Some(digest),
        },
    };
    assert!(
        run(&args)
            .unwrap_err()
            .to_string()
            .contains("changed after review")
    );
    assert!(!output.exists());
    assert!(
        !fs::read_dir(&fixture.0)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().contains("partial"))
    );
}
