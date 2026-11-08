//! Ground-truth quality harness for the frame interpolator.
//!
//! Each scenario is generated analytically: a continuous texture is sampled at
//! pixel centres for t = 0 and t = 1, and the *exact* intermediate frame is the
//! same texture sampled at the half-way motion (translation, line/edge motion,
//! rotation) or at the half-way object position (moving square). The
//! interpolator only ever sees the two real frames, so the difference against
//! the analytic ground truth is a true interpolation error, not a proxy.
//!
//! Metrics: PSNR and a windowed SSIM-like score on the BT.709 luma plane, plus
//! a pure-crossfade PSNR baseline so the value of the motion compensation is
//! visible in the same table. Analytic truth fields are self-checked: the
//! truth_mad column warps prev by the truth flow and compares against next (a
//! correct field leaves only resampling error, ~1e-3), so a mis-signed or
//! mis-centred ground truth cannot silently produce fake EPE numbers.
//! Timings are wall-clock on the current machine.
//!
//! Usage:
//!   cargo run --release -p fluid-interp --example quality
//!   cargo run --release -p fluid-interp --example quality -- --json out/interp-quality.json

use std::fmt::Write as _;
use std::time::Instant;

use fluid_core::{Dims, FlowConfig, Frame, InterpolatorConfig, LumaPlane, PixelFormat};
use fluid_flow::{FlowParams, Vec2, estimate};

/// Deterministic multi-scale blob texture in normalised coordinates.
const BLOBS: [(f32, f32, f32, f32); 22] = [
    (0.13, 0.22, 0.075, 0.55),
    (0.58, 0.15, 0.060, 0.35),
    (0.84, 0.31, 0.080, 0.60),
    (0.31, 0.51, 0.070, 0.45),
    (0.69, 0.63, 0.090, 0.30),
    (0.09, 0.74, 0.065, 0.50),
    (0.46, 0.86, 0.075, 0.40),
    (0.90, 0.82, 0.050, 0.55),
    (0.24, 0.35, 0.040, 0.25),
    (0.55, 0.44, 0.035, 0.20),
    (0.40, 0.10, 0.025, 0.30),
    (0.77, 0.50, 0.022, 0.35),
    (0.18, 0.62, 0.018, 0.30),
    (0.62, 0.72, 0.020, 0.25),
    (0.33, 0.78, 0.016, 0.28),
    (0.87, 0.66, 0.024, 0.32),
    (0.50, 0.28, 0.030, 0.22),
    (0.05, 0.45, 0.028, 0.26),
    (0.71, 0.08, 0.012, 0.30),
    (0.21, 0.91, 0.010, 0.28),
    (0.44, 0.62, 0.008, 0.24),
    (0.93, 0.12, 0.007, 0.26),
];

fn texture(u: f32, v: f32) -> f32 {
    let mut acc = 0.10f32;
    for &(bu, bv, r, amp) in BLOBS.iter() {
        let du = u - bu;
        let dv = v - bv;
        let r2 = du * du + dv * dv;
        let cutoff = r * 3.0;
        if r2 > cutoff * cutoff {
            continue;
        }
        acc += amp * (-r2 / (2.0 * r * r)).exp();
    }
    acc.clamp(0.0, 1.0)
}

fn texture_px(x: f32, y: f32, dims: Dims) -> f32 {
    texture(
        (x + 0.5) / dims.width as f32,
        (y + 0.5) / dims.height as f32,
    )
}

fn grey_frame(dims: Dims, f: impl Fn(f32, f32) -> f32) -> Frame {
    let w = dims.width;
    let h = dims.height;
    let mut data = vec![0u8; dims.area() as usize * 4];
    for y in 0..h {
        for x in 0..w {
            let v = (f(x as f32, y as f32).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let i = ((y * w + x) * 4) as usize;
            data[i] = v;
            data[i + 1] = v;
            data[i + 2] = v;
            data[i + 3] = 255;
        }
    }
    Frame::new(dims, PixelFormat::Rgba8, data).expect("valid frame")
}

fn translate(dims: Dims, dx: f32, dy: f32, t: f32) -> Frame {
    grey_frame(dims, |x, y| texture_px(x - dx * t, y - dy * t, dims))
}

fn rotated(dims: Dims, angle: f32) -> Frame {
    let (w, h) = (dims.width as f32, dims.height as f32);
    let (cx, cy) = (w * 0.5, h * 0.5);
    let (ca, sa) = (angle.cos(), angle.sin());
    grey_frame(dims, |x, y| {
        let dx = x + 0.5 - cx;
        let dy = y + 0.5 - cy;
        let sx = cx + ca * dx + sa * dy;
        let sy = cy - sa * dx + ca * dy;
        texture((sx) / w, (sy) / h)
    })
}

fn thin_line(dims: Dims, dx: f32, t: f32) -> Frame {
    let xc = dims.width as f32 * 0.3 + dx * t;
    grey_frame(dims, |x, y| {
        let bg = texture_px(x, y, dims) * 0.45 + 0.12;
        let cov = (1.5 - (x - xc).abs()).clamp(0.0, 1.0);
        bg + (0.95 - bg) * cov
    })
}

fn high_contrast_edge(dims: Dims, dx: f32, t: f32) -> Frame {
    let xc = dims.width as f32 * 0.45 + dx * t;
    grey_frame(dims, |x, y| {
        let bg = texture_px(x, y, dims);
        let cov = (x - xc + 0.5).clamp(0.0, 1.0);
        0.10 + 0.82 * cov + 0.08 * bg
    })
}

fn moving_square(dims: Dims, dx: f32, dy: f32, t: f32) -> Frame {
    let side = (dims.width as f32 / 13.0).max(24.0);
    let half = side * 0.5;
    let cx = dims.width as f32 * 0.42 + dx * t;
    let cy = dims.height as f32 * 0.45 + dy * t;
    grey_frame(dims, |x, y| {
        let bg = texture_px(x, y, dims);
        let covx = (half + 0.5 - (x - cx).abs()).clamp(0.0, 1.0);
        let covy = (half + 0.5 - (y - cy).abs()).clamp(0.0, 1.0);
        let cov = covx * covy;
        let fg = 0.88 + 0.06 * ((x * 0.013 + y * 0.007).sin() * 0.5 + 0.5);
        bg + (fg - bg) * cov
    })
}

fn scene_cut_pair(dims: Dims) -> (Frame, Frame) {
    let prev = grey_frame(dims, |x, y| texture_px(x, y, dims));
    let next = grey_frame(dims, |x, y| {
        let u = (x + 0.5) / dims.width as f32;
        let v = (y + 0.5) / dims.height as f32;
        (0.9 - texture(u, v)).clamp(0.0, 1.0)
    });
    (prev, next)
}

/// Analytic flow for scenarios where it is known exactly.
#[derive(Clone, Copy)]
enum Truth {
    Constant(Vec2),
    Rotation { angle: f32, cx: f32, cy: f32 },
    Unknown,
}

impl Truth {
    fn at(&self, x: u32, y: u32) -> Vec2 {
        match *self {
            Truth::Constant(v) => v,
            Truth::Rotation { angle, cx, cy } => {
                // rotated() samples the texture at R(-angle) (p + 0.5 - c) + c, so
                // the gather flow satisfying prev(p + f) = next(p) is
                // f = R(-angle) v - v with v = p + 0.5 - c. (An earlier version
                // used R(+angle) and integer pixel positions, i.e. the inverse
                // motion: it inflated the rotation EPE from ~0.5 px to ~8-16 px
                // and made the analytic-flow oracle worse than a plain crossfade.
                // truth_consistency_mad now guards against that class of mistake.)
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let (sa, ca) = (angle.sin(), angle.cos());
                Vec2::new(ca * dx + sa * dy - dx, -sa * dx + ca * dy - dy)
            }
            Truth::Unknown => Vec2::ZERO,
        }
    }
}

/// Flow error against the analytic motion, overall and on trusted pixels only.
fn flow_epe(
    forward: &fluid_flow::FlowField,
    confidence: &[f32],
    truth: Truth,
    prev: &LumaPlane,
    border: u32,
) -> (f64, f64, f64) {
    let dims = forward.dims();
    let (w, h) = (dims.width, dims.height);
    let mut se = 0.0f64;
    let mut n = 0u64;
    let mut se_t = 0.0f64;
    let mut n_t = 0u64;
    let mut se_x = 0.0f64;
    let mut n_x = 0u64;
    if matches!(truth, Truth::Unknown) {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    for y in border..h.saturating_sub(border) {
        for x in border..w.saturating_sub(border) {
            let i = (y * w + x) as usize;
            let d = forward.vectors()[i].sub(truth.at(x, y));
            let e = f64::from(d.len());
            se += e * e;
            n += 1;
            if confidence.get(i).copied().unwrap_or(0.0) >= 0.2 {
                se_t += e * e;
                n_t += 1;
            }
            // Textured pixels carry recoverable motion; flat regions are an
            // aperture ambiguity and are scored separately.
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w - 1);
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            let gx = prev.get(x1, y) - prev.get(x0, y);
            let gy = prev.get(x, y1) - prev.get(x, y0);
            if gx.abs() + gy.abs() >= 0.05 {
                se_x += e * e;
                n_x += 1;
            }
        }
    }
    let epe = (se / n.max(1) as f64).sqrt();
    let epe_t = if n_t == 0 {
        f64::NAN
    } else {
        (se_t / n_t as f64).sqrt()
    };
    let epe_x = if n_x == 0 {
        f64::NAN
    } else {
        (se_x / n_x as f64).sqrt()
    };
    (epe, epe_t, epe_x)
}

/// Benchmark ceiling: motion compensation using the analytic flow.
fn oracle_mc_psnr(
    prev: &LumaPlane,
    next: &LumaPlane,
    mid: &LumaPlane,
    truth: Truth,
    alpha: f32,
) -> f64 {
    if matches!(truth, Truth::Unknown) {
        return f64::NAN;
    }
    let dims = prev.dims;
    let (w, h) = (dims.width, dims.height);
    let mut out = LumaPlane::zeros(dims);
    for y in 0..h {
        for x in 0..w {
            let v = truth.at(x, y);
            let pa = prev.sample_bilinear(x as f32 + alpha * v.x, y as f32 + alpha * v.y);
            let pb = next.sample_bilinear(
                x as f32 - (1.0 - alpha) * v.x,
                y as f32 - (1.0 - alpha) * v.y,
            );
            out.set(x, y, pa + (pb - pa) * alpha);
        }
    }
    psnr_db(&out, mid)
}

/// Sanity check of an analytic truth field: warp prev by the truth flow and
/// compare against next on textured pixels. A correct field leaves only
/// resampling error (~1e-3 on this texture); a mis-signed or mis-centred field
/// leaves a large residual, which is how a wrong ground truth is caught.
fn truth_consistency_mad(prev: &LumaPlane, next: &LumaPlane, truth: Truth, border: u32) -> f64 {
    if matches!(truth, Truth::Unknown) {
        return f64::NAN;
    }
    let dims = prev.dims;
    let (w, h) = (dims.width, dims.height);
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for y in border..h.saturating_sub(border) {
        for x in border..w.saturating_sub(border) {
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w - 1);
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            let gx = prev.get(x1, y) - prev.get(x0, y);
            let gy = prev.get(x, y1) - prev.get(x, y0);
            if gx.abs() + gy.abs() < 0.05 {
                continue;
            }
            let f = truth.at(x, y);
            let warped = prev.sample_bilinear(x as f32 + f.x, y as f32 + f.y);
            sum += f64::from((warped - next.get(x, y)).abs());
            n += 1;
        }
    }
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn psnr_db(a: &LumaPlane, b: &LumaPlane) -> f64 {
    let mut se = 0.0f64;
    for (p, q) in a.data.iter().zip(b.data.iter()) {
        let d = f64::from(*p) - f64::from(*q);
        se += d * d;
    }
    let mse = se / a.data.len().max(1) as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    }
}

/// Windowed SSIM-like score (8x8 windows) on a single luma plane.
fn ssim_like(a: &LumaPlane, b: &LumaPlane) -> f64 {
    const C1: f64 = 0.01 * 0.01;
    const C2: f64 = 0.03 * 0.03;
    let w = a.dims.width as usize;
    let h = a.dims.height as usize;
    let mut total = 0.0f64;
    let mut windows = 0u64;
    let mut by = 0;
    while by < h {
        let mut bx = 0;
        while bx < w {
            let (mut sa, mut sb) = (0.0f64, 0.0f64);
            let (mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64);
            let mut n = 0.0f64;
            for y in by..(by + 8).min(h) {
                for x in bx..(bx + 8).min(w) {
                    let i = y * w + x;
                    let p = f64::from(a.data[i]);
                    let q = f64::from(b.data[i]);
                    sa += p;
                    sb += q;
                    saa += p * p;
                    sbb += q * q;
                    sab += p * q;
                    n += 1.0;
                }
            }
            let ma = sa / n;
            let mb = sb / n;
            let va = saa / n - ma * ma;
            let vb = sbb / n - mb * mb;
            let cov = sab / n - ma * mb;
            total += ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            windows += 1;
            bx += 8;
        }
        by += 8;
    }
    total / windows.max(1) as f64
}

struct Row {
    scenario: &'static str,
    width: u32,
    height: u32,
    psnr: f64,
    ssim: f64,
    crossfade_psnr: f64,
    flow_ms: f64,
    gen_ms: f64,
    frames: usize,
    scene_cut: bool,
    occluded: u64,
    mean_conf: f32,
    mean_flow: f32,
    epe: f64,
    epe_trusted: f64,
    epe_textured: f64,
    oracle_psnr: f64,
    /// Mean |prev(p + f(p)) - next(p)| on textured pixels for the analytic
    /// truth field: a self-check that the ground truth is actually the motion
    /// between the two frames (must be ~0).
    truth_mad: f64,
}

fn fmt_psnr(v: f64) -> String {
    if v.is_nan() {
        "n/a".to_string()
    } else if v.is_finite() {
        format!("{v:.2}")
    } else {
        "inf".to_string()
    }
}

fn fmt_epe(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.2}")
    } else {
        "n/a".to_string()
    }
}

fn main() {
    let mut json_path: Option<String> = None;
    let mut blend: Option<f32> = None;
    let mut occlusion_fill: Option<bool> = None;
    let mut sizes: Vec<Dims> = vec![Dims::new(1280, 720), Dims::new(1920, 1080)];
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" if i + 1 < args.len() => {
                json_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--blend" if i + 1 < args.len() => {
                blend = Some(args[i + 1].parse().expect("--blend needs a number"));
                i += 2;
            }
            "--no-occlusion-fill" => {
                occlusion_fill = Some(false);
                i += 1;
            }
            "--occlusion-fill" => {
                occlusion_fill = Some(true);
                i += 1;
            }
            "--size" if i + 1 < args.len() => {
                let s = &args[i + 1];
                let (w, h) = s.split_once('x').expect("--size needs WIDTHxHEIGHT");
                sizes = vec![Dims::new(
                    w.parse().expect("width"),
                    h.parse().expect("height"),
                )];
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let mut cfg = InterpolatorConfig::default();
    if let Some(b) = blend {
        cfg.blend = b;
    }
    if let Some(o) = occlusion_fill {
        cfg.occlusion_fill = o;
    }
    println!(
        "params: blend {:.2} occlusion_fill {}",
        cfg.blend, cfg.occlusion_fill
    );
    let flow_cfg = FlowConfig::default();
    let params = FlowParams::from(&flow_cfg);
    let mut rows: Vec<Row> = Vec::new();

    for dims in sizes {
        let (w, h) = (dims.width, dims.height);
        let line_dx = 3.0f32;
        let edge_dx = 2.5f32;
        let square_dx = 6.0f32;
        let square_dy = 2.0f32;
        let rotation_angle = 0.5f32.to_radians();
        // Large-angle scenes: translation-only flow cannot follow a rotation of
        // this size (a 5 deg rotation moves corner pixels ~96 px at 720p), so
        // these scenes measure the gap that a rotation-aware path must close.
        let rotation_3deg = 3.0f32.to_radians();
        let rotation_5deg = 5.0f32.to_radians();

        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let cases: Vec<(&'static str, Frame, Frame, Option<Frame>, Truth)> = vec![
            (
                "translation_subpixel",
                translate(dims, 2.5, -1.25, 0.0),
                translate(dims, 2.5, -1.25, 1.0),
                Some(translate(dims, 2.5, -1.25, 0.5)),
                Truth::Constant(Vec2::new(-2.5, 1.25)),
            ),
            (
                "translation_diagonal",
                translate(dims, 6.0, 4.5, 0.0),
                translate(dims, 6.0, 4.5, 1.0),
                Some(translate(dims, 6.0, 4.5, 0.5)),
                Truth::Constant(Vec2::new(-6.0, -4.5)),
            ),
            (
                "thin_line",
                thin_line(dims, line_dx, 0.0),
                thin_line(dims, line_dx, 1.0),
                Some(thin_line(dims, line_dx, 0.5)),
                Truth::Unknown,
            ),
            (
                "high_contrast_edge",
                high_contrast_edge(dims, edge_dx, 0.0),
                high_contrast_edge(dims, edge_dx, 1.0),
                Some(high_contrast_edge(dims, edge_dx, 0.5)),
                Truth::Unknown,
            ),
            (
                "moving_square",
                moving_square(dims, square_dx, square_dy, 0.0),
                moving_square(dims, square_dx, square_dy, 1.0),
                Some(moving_square(dims, square_dx, square_dy, 0.5)),
                Truth::Unknown,
            ),
            (
                "rotation",
                rotated(dims, 0.0),
                rotated(dims, rotation_angle),
                Some(rotated(dims, rotation_angle * 0.5)),
                Truth::Rotation {
                    angle: rotation_angle,
                    cx,
                    cy,
                },
            ),
            (
                "rotation_3deg",
                rotated(dims, 0.0),
                rotated(dims, rotation_3deg),
                Some(rotated(dims, rotation_3deg * 0.5)),
                Truth::Rotation {
                    angle: rotation_3deg,
                    cx,
                    cy,
                },
            ),
            (
                "rotation_5deg",
                rotated(dims, 0.0),
                rotated(dims, rotation_5deg),
                Some(rotated(dims, rotation_5deg * 0.5)),
                Truth::Rotation {
                    angle: rotation_5deg,
                    cx,
                    cy,
                },
            ),
        ];

        for (name, prev, next, truth, truth_flow) in cases {
            let lp = prev.to_luma();
            let ln = next.to_luma();
            let t = Instant::now();
            let flow = match estimate(&lp, &ln, &params) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("{name}: flow failed: {e}");
                    continue;
                }
            };
            let flow_ms = t.elapsed().as_secs_f64() * 1000.0;
            let (epe, epe_trusted, epe_textured) =
                flow_epe(&flow.forward, &flow.confidence, truth_flow, &lp, 8);
            let oracle_psnr = match &truth {
                Some(t) => oracle_mc_psnr(&lp, &ln, &t.to_luma(), truth_flow, 0.5),
                None => f64::NAN,
            };
            let truth_mad = truth_consistency_mad(&lp, &ln, truth_flow, 8);
            if truth_mad.is_finite() && truth_mad > 0.02 {
                eprintln!(
                    "WARNING: analytic truth flow for '{name}' does not match the frames \
                     (truth_mad {truth_mad:.4}); check the Truth formulas."
                );
            }
            let mut it = fluid_interp::Interpolator::new(&cfg);
            let t = Instant::now();
            let frames = match it.generate(&prev, &next, &flow_cfg) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("{name}: generate failed: {e}");
                    continue;
                }
            };
            let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
            let metrics = it.last_metrics();
            let crossfade = it.crossfade(&prev, &next, 0.5);
            let xf_psnr = psnr_db(&crossfade.to_luma(), &truth.as_ref().unwrap().to_luma());

            let (psnr, ssim) = match (&truth, frames.first()) {
                (Some(truth), Some(frame)) => {
                    let lt = truth.to_luma();
                    let lo = frame.to_luma();
                    (psnr_db(&lo, &lt), ssim_like(&lo, &lt))
                }
                _ => (f64::NAN, f64::NAN),
            };

            let row = Row {
                scenario: name,
                width: w,
                height: h,
                psnr,
                ssim,
                crossfade_psnr: xf_psnr,
                flow_ms,
                gen_ms,
                frames: frames.len(),
                scene_cut: metrics.scene_cut,
                occluded: metrics.occluded_px,
                mean_conf: metrics.mean_confidence,
                mean_flow: metrics.mean_flow_px,
                epe,
                epe_trusted,
                epe_textured,
                oracle_psnr,
                truth_mad,
            };
            println!(
                "{:<22} {:>4}x{:<4} psnr {:>7} ssim {:>6.4} xfade {:>7} oracle {:>7} epe {:>6}px epe_tex {:>6}px flow {:>6.2} ms gen {:>6.2} ms frames {} cut {} occl {} conf {:.3} flow {:.2}px",
                row.scenario,
                row.width,
                row.height,
                fmt_psnr(row.psnr),
                row.ssim,
                fmt_psnr(row.crossfade_psnr),
                fmt_psnr(row.oracle_psnr),
                fmt_epe(row.epe),
                fmt_epe(row.epe_textured),
                row.flow_ms,
                row.gen_ms,
                row.frames,
                row.scene_cut,
                row.occluded,
                row.mean_conf,
                row.mean_flow,
            );
            rows.push(row);
        }

        // Scene cut: the interpolator must emit nothing and flag the cut.
        let (prev, next) = scene_cut_pair(dims);
        let mut it = fluid_interp::Interpolator::new(&cfg);
        let t = Instant::now();
        let frames = it.generate(&prev, &next, &flow_cfg).expect("generate");
        let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
        let metrics = it.last_metrics();
        let row = Row {
            scenario: "scene_cut",
            width: w,
            height: h,
            psnr: f64::NAN,
            ssim: f64::NAN,
            crossfade_psnr: f64::NAN,
            flow_ms: 0.0,
            gen_ms,
            frames: frames.len(),
            scene_cut: metrics.scene_cut,
            occluded: metrics.occluded_px,
            mean_conf: metrics.mean_confidence,
            mean_flow: metrics.mean_flow_px,
            epe: f64::NAN,
            epe_trusted: f64::NAN,
            epe_textured: f64::NAN,
            oracle_psnr: f64::NAN,
            truth_mad: f64::NAN,
        };
        println!(
            "{:<22} {:>4}x{:<4} psnr {:>7} ssim {:>6.4} xfade {:>7} oracle {:>7} epe {:>6}px epe_tex {:>6}px flow {:>6.2} ms gen {:>6.2} ms frames {} cut {} occl {} conf {:.3} flow {:.2}px",
            row.scenario,
            row.width,
            row.height,
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            row.flow_ms,
            row.gen_ms,
            row.frames,
            row.scene_cut,
            row.occluded,
            row.mean_conf,
            row.mean_flow,
        );
        rows.push(row);
    }

    println!();
    println!(
        "| scenario | size | PSNR dB | SSIM | crossfade dB | oracle dB | EPE tex px | flow ms | frame ms | frames |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {}x{} | {} | {:.4} | {} | {} | {} | {:.2} | {:.2} | {} |",
            r.scenario,
            r.width,
            r.height,
            fmt_psnr(r.psnr),
            if r.ssim.is_finite() { r.ssim } else { f64::NAN },
            fmt_psnr(r.crossfade_psnr),
            fmt_psnr(r.oracle_psnr),
            fmt_epe(r.epe_textured),
            r.flow_ms,
            r.gen_ms,
            r.frames
        );
    }

    if let Some(path) = json_path {
        let mut out = String::new();
        out.push_str("{\n  \"scenarios\": [\n");
        for (i, r) in rows.iter().enumerate() {
            let psnr = if r.psnr.is_finite() {
                format!("{:.4}", r.psnr)
            } else {
                "null".to_string()
            };
            let ssim = if r.ssim.is_finite() {
                format!("{:.5}", r.ssim)
            } else {
                "null".to_string()
            };
            let xf = if r.crossfade_psnr.is_finite() {
                format!("{:.4}", r.crossfade_psnr)
            } else {
                "null".to_string()
            };
            let comma = if i + 1 == rows.len() { "" } else { "," };
            let epe = if r.epe.is_finite() {
                format!("{:.4}", r.epe)
            } else {
                "null".to_string()
            };
            let epe_t = if r.epe_trusted.is_finite() {
                format!("{:.4}", r.epe_trusted)
            } else {
                "null".to_string()
            };
            let epe_x = if r.epe_textured.is_finite() {
                format!("{:.4}", r.epe_textured)
            } else {
                "null".to_string()
            };
            let oracle = if r.oracle_psnr.is_finite() {
                format!("{:.4}", r.oracle_psnr)
            } else {
                "null".to_string()
            };
            let truth_mad = if r.truth_mad.is_finite() {
                format!("{:.6}", r.truth_mad)
            } else {
                "null".to_string()
            };
            writeln!(
                out,
                "    {{\"scenario\": \"{}\", \"width\": {}, \"height\": {}, \"psnr_db\": {}, \"ssim\": {}, \"crossfade_psnr_db\": {}, \"oracle_psnr_db\": {}, \"epe_px\": {}, \"epe_trusted_px\": {}, \"epe_textured_px\": {}, \"flow_ms\": {:.2}, \"frame_ms\": {:.2}, \"frames\": {}, \"scene_cut\": {}, \"occluded_px\": {}, \"mean_confidence\": {:.4}, \"mean_flow_px\": {:.3}, \"truth_mad\": {}}}{}",
                r.scenario,
                r.width,
                r.height,
                psnr,
                ssim,
                xf,
                oracle,
                epe,
                epe_t,
                epe_x,
                r.flow_ms,
                r.gen_ms,
                r.frames,
                r.scene_cut,
                r.occluded,
                r.mean_conf,
                r.mean_flow,
                truth_mad,
                comma
            )
            .expect("write to string");
        }
        out.push_str("  ]\n}\n");
        if let Some(parent) = std::path::Path::new(&path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
        std::fs::write(&path, out).expect("write json");
        println!();
        println!("json written to {path}");
    }
}
