//! Quality harness: score every upscaling filter against analytic ground truth.
//!
//! The harness renders high-resolution reference frames from pure maths,
//! downscales them to the filter input size with an exact box average, upscales
//! them again and compares the result with the reference. It reports PSNR (mean
//! and worst tile), SSIM, a gradient-energy sharpness proxy, temporal flicker
//! and the per-call time for the full matrix of filters, sharpness levels and
//! scale factors, and can dump the same numbers as JSON.
//!
//! Usage:
//!
//!   cargo run --release -p fluid-upscale --example quality
//!   cargo run --release -p fluid-upscale --example quality -- --json out/upscale-quality.json
//!
//! The worst-tile PSNR is dominated by the chart's Nyquist bands (the one-pixel
//! checkerboard and the one-pixel lines): the box downscale removes that detail
//! before any filter runs, so every spatial filter lands near 7.5 dB there. The
//! per-band lines under each row show where a filter actually wins or loses.
//!
//! See the (analysis-only) `quality` module in the library for the metrics.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use fluid_core::Dims;
use fluid_upscale::quality::{
    BAND_NAMES, MotionCase, MotionMetrics, PSNR_CAP_DB, Scenario, ScenarioMetrics,
    default_motion_cases, default_scenarios, run_motion_case, run_scenario,
};

fn main() {
    let mut json_path: Option<PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => match args.next() {
                Some(path) => json_path = Some(PathBuf::from(path)),
                None => fail("--json needs a path"),
            },
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => fail(&format!("unknown argument {other}")),
        }
    }

    let scenarios = default_scenarios();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let fluid_threads = env::var("FLUID_THREADS").unwrap_or_else(|_| String::from("auto"));

    println!(
        "fluid-upscale quality harness (v{})",
        env!("CARGO_PKG_VERSION")
    );
    println!("workers: {workers} logical (FLUID_THREADS={fluid_threads})");
    println!(
        "reference: analytic six-band chart, PSNR capped at {PSNR_CAP_DB:.0} dB, flicker over three frames"
    );
    println!("chart bands: {}", BAND_NAMES.join(", "));
    println!("{} scenarios", scenarios.len());
    println!();
    println!(
        "{:<11} {:<9} {:>2} {:<9} {:>5} {:>8} {:>8} {:>7} {:>6} {:>8} {:>8}",
        "reference",
        "input",
        "x",
        "filter",
        "sharp",
        "psnr",
        "psnr_min",
        "ssim",
        "grad",
        "flicker",
        "ms"
    );
    println!("{}", "-".repeat(112));

    let mut results: Vec<(Scenario, ScenarioMetrics)> = Vec::with_capacity(scenarios.len());
    for scenario in &scenarios {
        let metrics = run_scenario(scenario);
        println!(
            "{:<11} {:<9} {:>2} {:<9} {:>5.1} {:>8.3} {:>8.3} {:>7.5} {:>6.3} {:>8.3} {:>8.2}",
            format!("{}x{}", scenario.ref_dims.width, scenario.ref_dims.height),
            scenario.low_dims.to_string(),
            scenario.scale_factor(),
            filter_name(scenario.filter),
            scenario.sharpness,
            metrics.psnr_mean,
            metrics.psnr_min,
            metrics.ssim,
            metrics.gradient_ratio,
            metrics.flicker,
            metrics.ms
        );
        let bands: Vec<String> = metrics
            .band_psnr
            .iter()
            .map(|value| format!("{value:6.2}"))
            .collect();
        println!("             bands dB: {}", bands.join(" "));
        results.push((*scenario, metrics));
    }

    println!();
    println!("best filter per case, by mean PSNR:");
    let mut seen: Vec<(Dims, Dims, bool)> = Vec::new();
    for (scenario, _) in &results {
        let key = (scenario.ref_dims, scenario.low_dims, scenario.temporal);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let mut best_name = String::new();
        let mut best_psnr = f64::NEG_INFINITY;
        for (candidate, metrics) in &results {
            if (candidate.ref_dims, candidate.low_dims, candidate.temporal) != key {
                continue;
            }
            if metrics.psnr_mean > best_psnr {
                best_psnr = metrics.psnr_mean;
                best_name = format!(
                    "{} sharp {:.1}",
                    filter_name(candidate.filter),
                    candidate.sharpness
                );
            }
        }
        let kind = if scenario.temporal {
            "temporal"
        } else {
            "chart"
        };
        println!(
            "  {}x{} from {} ({}x, {kind}): {best_name} at {best_psnr:.3} dB",
            scenario.ref_dims.width,
            scenario.ref_dims.height,
            scenario.low_dims,
            scenario.scale_factor()
        );
    }

    let motion_cases = default_motion_cases();
    println!();
    println!("moving-content ghosting, temporal filter (min PSNR over six frames, trail error):");
    println!(
        "{:<11} {:<9} {:>2} {:>5} {:>5} {:>8} {:>9} {:>8} {:>8}",
        "reference", "input", "x", "speed", "sharp", "psnr", "psnr_min", "ghost", "ms"
    );
    println!("{}", "-".repeat(102));
    let mut motion_results: Vec<(MotionCase, MotionMetrics)> =
        Vec::with_capacity(motion_cases.len());
    for case in &motion_cases {
        let metrics = run_motion_case(case);
        println!(
            "{:<11} {:<9} {:>2} {:>5} {:>5.1} {:>8.3} {:>9.3} {:>8.3} {:>8.2}",
            format!("{}x{}", case.ref_dims.width, case.ref_dims.height),
            case.low_dims.to_string(),
            case.scale_factor(),
            case.speed_px,
            case.sharpness,
            metrics.psnr_mean,
            metrics.psnr_min_sequence,
            metrics.ghosting,
            metrics.ms
        );
        motion_results.push((*case, metrics));
    }

    if let Some(path) = json_path {
        let json = build_json(&results, &motion_results, workers, &fluid_threads);
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && let Err(error) = fs::create_dir_all(parent)
        {
            fail(&format!("cannot create {}: {error}", parent.display()));
        }
        match fs::write(&path, json) {
            Ok(()) => println!("{}wrote {}", char::from(10), path.display()),
            Err(error) => fail(&format!("cannot write {}: {error}", path.display())),
        }
    }
}

fn print_help() {
    println!("fluid-upscale quality harness");
    println!();
    println!("Scores every upscaling filter against an analytic ground-truth chart.");
    println!();
    println!("Options:");
    println!("  --json PATH   also write the per-scenario results as JSON");
    println!("  -h, --help    print this help");
    println!();
    println!("Tuning is via the usual FLUID_THREADS variable.");
}

fn fail(message: &str) -> ! {
    eprintln!("quality: {message}");
    std::process::exit(2);
}

fn filter_name(filter: fluid_core::UpscaleFilter) -> &'static str {
    filter.as_str()
}

fn quote(value: &str) -> String {
    let quote_char = char::from(34);
    let backslash = char::from(92);
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote_char);
    for c in value.chars() {
        if c == quote_char || c == backslash {
            out.push(backslash);
        }
        out.push(c);
    }
    out.push(quote_char);
    out
}

/// Compact JSON number: four decimals, non-finite values become `0.0` so the
/// document stays valid JSON, and PSNR is clamped at the documented cap.
fn number(value: f64) -> String {
    if value.is_finite() {
        format!("{:.4}", value.min(PSNR_CAP_DB))
    } else {
        String::from("0.0")
    }
}

fn build_json(
    results: &[(Scenario, ScenarioMetrics)],
    motion_results: &[(MotionCase, MotionMetrics)],
    workers: usize,
    fluid_threads: &str,
) -> String {
    let mut json = String::with_capacity(16 * 1024);
    let _ = writeln!(json, "{{");
    let _ = writeln!(json, "  \"harness\": \"fluid-upscale quality\",");
    let _ = writeln!(json, "  \"version\": \"{}\",", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(
        json,
        "  \"generated_by\": \"crates/fluid-upscale/examples/quality.rs\","
    );
    let _ = writeln!(json, "  \"reference\": \"analytic six-band chart\",");
    let _ = writeln!(json, "  \"psnr_cap_db\": {},", number(PSNR_CAP_DB));
    let _ = writeln!(json, "  \"workers\": {workers},");
    let _ = writeln!(json, "  \"fluid_threads\": {},", quote(fluid_threads));
    let mut bands = String::new();
    for (i, name) in BAND_NAMES.iter().enumerate() {
        if i > 0 {
            bands.push_str(", ");
        }
        let _ = write!(bands, "{}", quote(name));
    }
    let _ = writeln!(json, "  \"bands\": [{bands}],");
    let _ = writeln!(json, "  \"scenarios\": [");
    let total = results.len();
    for (index, (scenario, metrics)) in results.iter().enumerate() {
        let _ = writeln!(json, "    {{");
        let _ = writeln!(
            json,
            "      \"reference\": {{\"width\": {}, \"height\": {}}},",
            scenario.ref_dims.width, scenario.ref_dims.height
        );
        let _ = writeln!(
            json,
            "      \"input\": {{\"width\": {}, \"height\": {}}},",
            scenario.low_dims.width, scenario.low_dims.height
        );
        let _ = writeln!(json, "      \"scale\": {},", scenario.scale_factor());
        let _ = writeln!(
            json,
            "      \"filter\": {},",
            quote(filter_name(scenario.filter))
        );
        let _ = writeln!(
            json,
            "      \"sharpness\": {},",
            number(scenario.sharpness as f64)
        );
        let _ = writeln!(json, "      \"temporal\": {},", scenario.temporal);
        let _ = writeln!(json, "      \"frames\": {},", metrics.frames);
        let _ = writeln!(json, "      \"psnr_mean\": {},", number(metrics.psnr_mean));
        let _ = writeln!(json, "      \"psnr_min\": {},", number(metrics.psnr_min));
        let _ = writeln!(json, "      \"ssim\": {},", number(metrics.ssim));
        let _ = writeln!(
            json,
            "      \"gradient_ratio\": {},",
            number(metrics.gradient_ratio)
        );
        let _ = writeln!(json, "      \"flicker\": {},", number(metrics.flicker));
        let _ = writeln!(json, "      \"ms\": {},", number(metrics.ms));
        let _ = writeln!(json, "      \"checksum\": {},", metrics.checksum);
        let mut band_json = String::new();
        for (band, psnr) in metrics.band_psnr.iter().enumerate() {
            if band > 0 {
                band_json.push_str(", ");
            }
            let name = BAND_NAMES.get(band).copied().unwrap_or("band");
            let _ = write!(
                band_json,
                "{{\"band\": {}, \"psnr\": {}}}",
                quote(name),
                number(*psnr)
            );
        }
        let _ = writeln!(json, "      \"bands\": [{band_json}]");
        let _ = if index + 1 == total {
            writeln!(json, "    }}")
        } else {
            writeln!(json, "    }},")
        };
    }
    let _ = writeln!(json, "  ],");
    let _ = writeln!(json, "  \"motion_cases\": [");
    let motion_total = motion_results.len();
    for (index, (case, metrics)) in motion_results.iter().enumerate() {
        let _ = writeln!(json, "    {{");
        let _ = writeln!(
            json,
            "      \"reference\": {{\"width\": {}, \"height\": {}}},",
            case.ref_dims.width, case.ref_dims.height
        );
        let _ = writeln!(
            json,
            "      \"input\": {{\"width\": {}, \"height\": {}}},",
            case.low_dims.width, case.low_dims.height
        );
        let _ = writeln!(json, "      \"scale\": {},", case.scale_factor());
        let _ = writeln!(json, "      \"filter\": {},", quote("temporal"));
        let _ = writeln!(
            json,
            "      \"sharpness\": {},",
            number(case.sharpness as f64)
        );
        let _ = writeln!(json, "      \"speed_px\": {},", case.speed_px);
        let _ = writeln!(json, "      \"frames\": {},", metrics.frames);
        let _ = writeln!(json, "      \"psnr_mean\": {},", number(metrics.psnr_mean));
        let _ = writeln!(
            json,
            "      \"psnr_min_sequence\": {},",
            number(metrics.psnr_min_sequence)
        );
        let _ = writeln!(json, "      \"ghosting\": {},", number(metrics.ghosting));
        let _ = writeln!(json, "      \"ms\": {},", number(metrics.ms));
        let mut per_frame = String::new();
        for (frame, psnr) in metrics.psnr_per_frame.iter().enumerate() {
            if frame > 0 {
                per_frame.push_str(", ");
            }
            let _ = write!(per_frame, "{}", number(*psnr));
        }
        let _ = writeln!(json, "      \"psnr_per_frame\": [{per_frame}]");
        let _ = if index + 1 == motion_total {
            writeln!(json, "    }}")
        } else {
            writeln!(json, "    }},")
        };
    }
    let _ = writeln!(json, "  ]");
    let _ = writeln!(json, "}}");
    json
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_structurally_sane() {
        let scenario = Scenario {
            ref_dims: Dims::new(64, 36),
            low_dims: Dims::new(32, 18),
            filter: fluid_core::UpscaleFilter::Bilinear,
            sharpness: 0.0,
            temporal: false,
        };
        let results = vec![(scenario, run_scenario(&scenario))];
        let motion_case = MotionCase {
            ref_dims: Dims::new(64, 36),
            low_dims: Dims::new(32, 18),
            speed_px: 6,
            sharpness: 0.0,
        };
        let motion_results = vec![(motion_case, run_motion_case(&motion_case))];
        let json = build_json(&results, &motion_results, 2, "auto");
        let trimmed = json.trim();
        assert!(trimmed.starts_with('{') && trimmed.ends_with('}'));
        assert_eq!(json.matches('{').count(), json.matches('}').count());
        assert!(json.contains("\"psnr_mean\":"));
        assert!(json.contains("\"bands\":"));
        assert!(json.contains("\"motion_cases\":"));
        assert!(json.contains("\"psnr_min_sequence\":"));
        assert!(!json.contains("NaN") && !json.contains("inf"));
    }
}
