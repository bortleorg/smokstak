//! The flat-field metric shared by finished-image measurement and project analysis.
use sr_core::plane::Plane;

/// RMS after removing a plane, with three fitted degrees of freedom.
pub fn detrended_tile_sigma(vals: &[f32], tile: usize) -> f32 {
    assert!(tile >= 2 && vals.len() == tile * tile);
    // Remove a linear plane, then take the residual standard deviation.
    let n = vals.len() as f64;
    let c = (tile - 1) as f64 * 0.5;
    let (mut s, mut su, mut sv, mut suu) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for j in 0..tile {
        let dv = j as f64 - c;
        for i in 0..tile {
            let du = i as f64 - c;
            let val = vals[j * tile + i] as f64;
            s += val;
            su += du * val;
            sv += dv * val;
        }
    }
    for i in 0..tile {
        let du = i as f64 - c;
        suu += du * du;
    }
    suu *= tile as f64;
    let c0 = s / n;
    let c1 = if suu > 0.0 { su / suu } else { 0.0 };
    let c2 = if suu > 0.0 { sv / suu } else { 0.0 };
    let mut sse = 0.0f64;
    for j in 0..tile {
        let dv = j as f64 - c;
        for i in 0..tile {
            let du = i as f64 - c;
            let pred = c0 + c1 * du + c2 * dv;
            let d = vals[j * tile + i] as f64 - pred;
            sse += d * d;
        }
    }
    (sse / (n - 3.0).max(1.0)).sqrt() as f32
}

pub fn flat_field_noise(img: &Plane<f32>, tile: usize) -> f32 {
    let tile = tile.max(8);
    let mut sigmas: Vec<f32> = Vec::new();
    let mut y = 0;
    while y + tile <= img.height {
        let mut x = 0;
        while x + tile <= img.width {
            let mut vals = Vec::with_capacity(tile * tile);
            for j in 0..tile {
                for i in 0..tile {
                    vals.push(img.data[(y + j) * img.width + (x + i)]);
                }
            }
            sigmas.push(detrended_tile_sigma(&vals, tile));
            x += tile;
        }
        y += tile;
    }
    lower_decile_noise(&mut sigmas)
}

/// Mean of the lowest decile of tile sigmas: robust to stars and structure in
/// the other tiles. This is not a per-pixel MAD or a claim of pure sensor noise.
pub fn lower_decile_noise(sigmas: &mut [f32]) -> f32 {
    if sigmas.is_empty() {
        return 0.0;
    }
    sigmas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = (sigmas.len() / 10).max(1);
    sigmas[..k].iter().sum::<f32>() / k as f32
}
