//! Implementations of every subcommand.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use fluid_core::{
    CaptureBackendKind, Dims, FluidConfig, FluidError, FrameSink, FrameSource, InterpMode,
    PacingMode, QualityPreset, ScaleMode, UpscaleFilter,
};
use fluid_engine::{Pipeline, PipelineConfig};
use fluid_present::backend_report;
use fluid_video::{
    AviHeader, AviReader, AviWriter, DEFAULT_JPEG_QUALITY, ImageSequence, PngWriter,
    SyntheticSource, Y4mHeader, Y4mReader, Y4mWriter,
};

use crate::cli::{
    self, BenchArgs, Cli, Command, ConfigArgs, DoctorArgs, GuiArgs, RunArgs, VideoArgs,
};

/// Run the subcommand selected on the command line.
pub fn dispatch(cli: &Cli, cancel: &Arc<AtomicBool>) -> Result<(), String> {
    match &cli.command {
        Some(Command::Run(args)) => cmd_run(cli, args, cancel),
        Some(Command::Video(args)) => cmd_video(cli, args, cancel),
        Some(Command::Bench(args)) => cmd_bench(cli, args),
        Some(Command::Devices(args)) => cmd_devices(cli, args),
        Some(Command::Config(args)) => cmd_config(cli, args),
        Some(Command::Doctor(args)) => cmd_doctor(cli, args),
        Some(Command::Gui(args)) => cmd_gui(cli, args),
        Some(Command::Info) => cmd_info(cli),
        None => {
            print_overview();
            Ok(())
        }
    }
}

fn print_overview() {
    println!(
        "{} - frame generation + upscaling for any game, video or screen",
        fluid_core::version_string()
    );
    println!();
    println!("  fluid run     capture the screen, synthesise frames and show the result");
    println!("  fluid video   process a y4m video or PNG sequence offline");
    println!("  fluid bench   measure the pipeline on a synthetic source");
    println!("  fluid devices list displays, backends and windows (--windows)");
    println!("  fluid config  print or validate the configuration");
    println!("  fluid doctor  measure this machine and recommend settings");
    println!("  fluid gui     open the graphical app (beginner friendly)");
    println!("  fluid info    version and build information");
    println!();
    println!("Run 'fluid <command> --help' for the full option list.");
}

fn cmd_run(cli: &Cli, args: &RunArgs, cancel: &Arc<AtomicBool>) -> Result<(), String> {
    let mut cfg = cli::prepare_config(cli, &args.over)?;
    if let Some(backend) = args.capture.as_deref() {
        cfg.capture.backend = backend.parse().map_err(|e: FluidError| e.to_string())?;
    }
    if let Some(monitor) = args.monitor {
        cfg.capture.monitor = monitor;
    }
    if let Some(region) = args.region {
        cfg.capture.region = Some(region);
    }
    if args.window.is_some() || args.window_process.is_some() || args.window_id.is_some() {
        cfg.capture.window = Some(fluid_core::WindowTarget {
            id: args.window_id.unwrap_or(0),
            title: args.window.clone().unwrap_or_default(),
            process: args.window_process.clone().unwrap_or_default(),
        });
    }
    if args.no_cursor {
        cfg.capture.include_cursor = false;
    }
    if let Some(fps) = args.present_fps {
        cfg.latency.target_fps = fps;
    }
    cfg.validate().map_err(|e| e.to_string())?;
    cfg.install_logging();

    let mut source =
        fluid_capture::open(&cfg.capture).map_err(|e| format!("capture failed: {e}"))?;
    let dims = source.dims();
    let out_dims = cfg.upscaler.scale.resolve(dims);
    // Print the plan with the size the source really reports: a hardcoded
    // 1920x1080 preview is wrong on every other display and makes the
    // following pipeline log line look contradictory.
    if !cli.quiet {
        println!("{}", Pipeline::plan(&cfg, dims).map_err(|e| e.to_string())?);
    }
    let mut sink: Box<dyn FrameSink> = if args.no_present {
        Box::new(fluid_present::null_sink(out_dims))
    } else {
        fluid_present::open(&cfg.present, out_dims).map_err(|e| format!("present failed: {e}"))?
    };

    let pcfg = PipelineConfig {
        max_input_frames: args.frames,
        stats_interval: Some(Duration::from_millis(1000)),
        verbose: !cli.quiet,
        cancel: Some(cancel.clone()),
        ..PipelineConfig::default()
    };
    let mut pipeline = Pipeline::with_config(&cfg, dims, pcfg).map_err(|e| e.to_string())?;
    let report = pipeline
        .run(source.as_mut(), sink.as_mut())
        .map_err(|e| format!("pipeline failed: {e}"))?;
    emit_report(&report, args.report.as_deref(), cli.json)?;
    note_cancelled(cancel, report.stats.frames_in);
    Ok(())
}

fn cmd_video(cli: &Cli, args: &VideoArgs, cancel: &Arc<AtomicBool>) -> Result<(), String> {
    let cfg = cli::prepare_config(cli, &args.over)?;
    cfg.install_logging();
    // "-" means the process standard stream, so fluid video can sit in the
    // middle of an ffmpeg pipeline: ffmpeg | fluid video -i - -o - | ffmpeg.
    // When the output is a pipe stdout must stay pure y4m, hence every
    // progress line and the report itself go to stderr in that case.
    let input_stdin = is_stdio(&args.input);
    let output_stdout = is_stdio(&args.output);
    if !cli.quiet {
        eprintln!("input : {}", args.input.display());
        eprintln!("output: {}", args.output.display());
    }

    let mut source: Box<dyn FrameSource> = if input_stdin {
        Box::new(Y4mReader::new_stdin().map_err(|e| format!("cannot read stdin: {e}"))?)
    } else if is_y4m(&args.input) {
        Box::new(
            Y4mReader::open(&args.input)
                .map_err(|e| format!("cannot open {}: {e}", args.input.display()))?,
        )
    } else if is_avi(&args.input) {
        // MJPEG AVI is decoded in-process, so no ffmpeg is required.
        Box::new(
            AviReader::open(&args.input)
                .map_err(|e| format!("cannot open {}: {e}", args.input.display()))?,
        )
    } else {
        let fps = cfg.video.fps();
        Box::new(
            ImageSequence::open_dir(&args.input, fps)
                .map_err(|e| format!("cannot open {}: {e}", args.input.display()))?,
        )
    };
    let dims = source.dims();
    let src_fps = source.fps();
    let fps = if src_fps > 1.0 {
        src_fps
    } else {
        cfg.video.fps()
    };
    // Container frame rate must match the rate the pipeline actually emits.
    // With interpolation disabled the multiplier is ignored (one output frame per
    // input frame), so writing fps * multiplier here would make the file play
    // faster than real time (QA round 5 defect D1).
    let out_multiplier = video_out_multiplier(&cfg);
    let out_fps = fps * out_multiplier as f32;
    let out_dims = cfg.upscaler.scale.resolve(dims);

    let mut sink: Box<dyn FrameSink> = if output_stdout {
        let header = Y4mHeader::new(out_dims, out_fps);
        Box::new(Y4mWriter::new_stdout(header).map_err(|e| format!("cannot write stdout: {e}"))?)
    } else if is_y4m(&args.output) {
        let header = Y4mHeader::new(out_dims, out_fps);
        Box::new(
            Y4mWriter::create(&args.output, header)
                .map_err(|e| format!("cannot create {}: {e}", args.output.display()))?,
        )
    } else if is_avi(&args.output) {
        // Self-contained export: no ffmpeg needed, one MJPEG frame per image.
        let quality = args.jpeg_quality.unwrap_or(DEFAULT_JPEG_QUALITY);
        let header = AviHeader::new(out_dims, out_fps, quality);
        Box::new(
            AviWriter::create(&args.output, header)
                .map_err(|e| format!("cannot create {}: {e}", args.output.display()))?,
        )
    } else {
        Box::new(
            PngWriter::create(&args.output, out_dims)
                .map_err(|e| format!("cannot create {}: {e}", args.output.display()))?,
        )
    };

    if !cli.quiet {
        eprintln!(
            "source {:?} at {:.3} fps -> output {:?} ({}x frames, filter {})",
            dims,
            fps,
            out_dims,
            out_multiplier,
            cfg.upscaler.filter.as_str()
        );
    }

    let pcfg = PipelineConfig {
        max_input_frames: args.frames,
        stats_interval: Some(Duration::from_millis(2000)),
        verbose: !cli.quiet,
        cancel: Some(cancel.clone()),
        ..PipelineConfig::offline()
    };
    let mut pipeline = Pipeline::with_config(&cfg, dims, pcfg).map_err(|e| e.to_string())?;
    let report = pipeline
        .run(source.as_mut(), sink.as_mut())
        .map_err(|e| format!("pipeline failed: {e}"))?;
    if output_stdout {
        // stdout carries the y4m stream: report to the file and/or stderr.
        if let Some(path) = args.report.as_deref() {
            write_text(path, &report.to_json())?;
        }
        if cli.json {
            eprintln!("{}", report.to_json());
        } else {
            eprintln!("{report}");
        }
    } else {
        emit_report(&report, args.report.as_deref(), cli.json)?;
    }
    note_cancelled(cancel, report.stats.frames_in);
    Ok(())
}

fn cmd_bench(cli: &Cli, args: &BenchArgs) -> Result<(), String> {
    let mut cfg = cli::prepare_config(cli, &args.over)?;
    if args.over.is_empty() {
        // A bare bench should exercise the interesting parts of the pipeline.
        cfg.interpolator.mode = InterpMode::Fixed;
        cfg.interpolator.multiplier = 2;
        cfg.upscaler.filter = UpscaleFilter::Bilinear;
        cfg.upscaler.scale.mode = fluid_core::ScaleMode::Factor;
        cfg.upscaler.scale.factor = 1.5;
        cfg.validate().map_err(|e| e.to_string())?;
    }
    cfg.install_logging();
    let dims = args.size;
    if !cli.quiet {
        println!("{}", Pipeline::plan(&cfg, dims).map_err(|e| e.to_string())?);
        println!(
            "benchmark: {} warmup + {} measured frames",
            args.warmup, args.frames
        );
    }

    if args.warmup > 0 {
        run_bench_once(&cfg, dims, args.warmup)?;
    }
    let started = std::time::Instant::now();
    let report = run_bench_once(&cfg, dims, args.frames)?;
    let wall = started.elapsed().as_secs_f64();
    let out_frames = report.stats.frames_out;
    let output_fps = if wall > 0.0 {
        out_frames as f64 / wall
    } else {
        0.0
    };
    let source_fps = if wall > 0.0 {
        report.stats.frames_in as f64 / wall
    } else {
        0.0
    };

    if !cli.quiet && !cli.json {
        println!();
        println!("{report}");
        println!(
            "  throughput  : {:.1} source fps -> {:.1} output fps",
            source_fps, output_fps
        );
        println!(
            "  per-frame budget: {:.2} ms at {:.1} fps output",
            1000.0 / output_fps.max(0.001),
            output_fps
        );
    }
    // The 1 s FpsMeter inside the report is a snapshot taken at the end of the run;
    // for a benchmark the honest number is frames / wall clock. Report both so JSON
    // consumers can tell them apart (QA2 defect D3).
    let json = with_throughput(&report.to_json(), source_fps, output_fps, wall);
    if let Some(path) = args.report.as_deref() {
        write_text(path, &json)?;
    }
    if cli.json {
        println!("{json}");
    }
    Ok(())
}

/// Append measured wall-clock throughput fields to a report JSON object.
fn with_throughput(json: &str, source_fps: f64, output_fps: f64, wall_s: f64) -> String {
    let trimmed = json.trim_end();
    match trimmed.strip_suffix('}') {
        Some(head) => format!(
            "{head},\"throughput_in_fps\":{source_fps:.2},\"throughput_out_fps\":{output_fps:.2},\"wall_s\":{wall_s:.3}}}"
        ),
        None => json.to_string(),
    }
}

fn run_bench_once(
    cfg: &FluidConfig,
    dims: Dims,
    frames: u64,
) -> Result<fluid_engine::PipelineReport, String> {
    let out_dims = cfg.upscaler.scale.resolve(dims);
    let mut source = SyntheticSource::new(dims, 60.0);
    let mut sink = fluid_present::null_sink(out_dims);
    let pcfg = PipelineConfig {
        max_input_frames: Some(frames),
        stats_interval: None,
        verbose: false,
        ..PipelineConfig::offline()
    };
    let mut pipeline = Pipeline::with_config(cfg, dims, pcfg).map_err(|e| e.to_string())?;
    pipeline
        .run(&mut source, &mut sink)
        .map_err(|e| format!("benchmark failed: {e}"))
}

fn cmd_devices(cli: &Cli, args: &crate::cli::DevicesArgs) -> Result<(), String> {
    if cli.json {
        println!("{}", devices_json(args.windows));
        return Ok(());
    }
    if args.windows {
        println!("windows (use --window <title> / --window-process <name> / --window-id <id>):");
        match fluid_capture::enumerate_windows() {
            Ok(windows) if windows.is_empty() => println!("  (none reported)"),
            Ok(windows) => {
                for window in windows {
                    println!("  {window}");
                }
            }
            Err(e) => println!("  unavailable: {e}"),
        }
        println!();
    }
    println!("displays:");
    match fluid_capture::enumerate_monitors() {
        Ok(monitors) if monitors.is_empty() => println!("  (none reported)"),
        Ok(monitors) => {
            for m in monitors {
                println!("  {m}");
            }
        }
        Err(e) => println!("  unavailable: {e}"),
    }
    println!("capture backends:");
    for kind in CaptureBackendKind::ALL {
        let ok = fluid_capture::backend_available(kind);
        println!(
            "  {:<6} {}",
            kind.as_str(),
            if ok { "available" } else { "not available" }
        );
    }
    println!("present backends:");
    match backend_report() {
        list if list.is_empty() => println!("  (none reported)"),
        list => {
            for (name, ok) in list {
                println!(
                    "  {:<20} {}",
                    name,
                    if ok { "available" } else { "not available" }
                );
            }
        }
    }
    println!("gpu backends:");
    match fluid_gpu::probe() {
        Ok(info) => println!(
            "  {:<20} {} ({}, {})",
            fluid_gpu::BACKEND_NAME,
            info.name,
            info.device_type,
            info.backend
        ),
        Err(err) => println!("  {:<20} not available: {err}", fluid_gpu::BACKEND_NAME),
    }
    let cfg = FluidConfig::default();
    println!("config:");
    println!(
        "  default preset    : {}",
        QualityPreset::default().as_str()
    );
    println!("  default pacing    : {}", PacingMode::default().as_str());
    println!(
        "  default filter    : {} (scale {:.2})",
        cfg.upscaler.filter.as_str(),
        cfg.upscaler.scale.factor
    );
    println!(
        "  default interp    : {} (x{}, quality {})",
        cfg.interpolator.mode.as_str(),
        cfg.interpolator.multiplier,
        cfg.interpolator.quality.as_str()
    );
    Ok(())
}

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Machine-readable form of `fluid devices` (QA2 defect D2 used to print an
/// always-empty monitor list). With `include_windows` the window list used by
/// `--window` / `--window-id` is appended so automation can pick a window id.
fn devices_json(include_windows: bool) -> String {
    let mut out = String::from("{\"monitors\":[");
    if let Ok(list) = fluid_capture::enumerate_monitors() {
        for (i, m) in list.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"index\":{},\"name\":\"{}\",\"width\":{},\"height\":{},\"refresh_hz\":{:.2},\"primary\":{},\"left\":{},\"top\":{}}}",
                m.index,
                json_escape(&m.name),
                m.dims.width,
                m.dims.height,
                m.refresh_hz,
                m.primary,
                m.left,
                m.top
            ));
        }
    }
    out.push_str("],\"capture\":{");
    for (i, kind) in CaptureBackendKind::ALL.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "\"{}\":{}",
            kind.as_str(),
            fluid_capture::backend_available(*kind)
        ));
    }
    out.push_str("},\"present\":{");
    for (i, (name, ok)) in backend_report().into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("\"{}\":{}", json_escape(&name), ok));
    }
    out.push_str("},\"gpu\":{");
    match fluid_gpu::probe() {
        Ok(info) => out.push_str(&format!(
            "\"backend\":\"{}\",\"available\":true,\"name\":\"{}\",\"api\":\"{}\",\"device_type\":\"{}\",\"vendor_id\":{}",
            fluid_gpu::BACKEND_NAME,
            json_escape(&info.name),
            json_escape(info.backend),
            json_escape(info.device_type),
            info.vendor_id
        )),
        Err(err) => out.push_str(&format!(
            "\"backend\":\"{}\",\"available\":false,\"error\":\"{}\"",
            fluid_gpu::BACKEND_NAME,
            json_escape(&err.to_string())
        )),
    }
    out.push('}');
    if include_windows {
        out.push_str(",\"windows\":[");
        if let Ok(windows) = fluid_capture::enumerate_windows() {
            for (i, w) in windows.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "{{\"id\":{},\"title\":\"{}\",\"class_name\":\"{}\",\"process_name\":\"{}\",\"width\":{},\"height\":{},\"left\":{},\"top\":{},\"visible\":{},\"minimized\":{}}}",
                    w.id,
                    json_escape(&w.title),
                    json_escape(&w.class_name),
                    json_escape(&w.process_name),
                    w.dims.width,
                    w.dims.height,
                    w.left,
                    w.top,
                    w.visible,
                    w.minimized
                ));
            }
        }
        out.push(']');
    }
    out.push('}');
    out
}

fn cmd_config(cli: &Cli, args: &ConfigArgs) -> Result<(), String> {
    if let Some(path) = args.check.as_deref() {
        let cfg =
            FluidConfig::load(path).map_err(|e| format!("cannot load {}: {e}", path.display()))?;
        cfg.validate().map_err(|e| e.to_string())?;
        if cli.json {
            let escaped = path.display().to_string().replace('\\', "\\\\");
            println!("{{\"path\":\"{escaped}\",\"valid\":true}}");
            return Ok(());
        }
        println!("# {} is valid", path.display());
        let text = cfg.to_toml_string().map_err(|e| e.to_string())?;
        println!("{text}");
        return Ok(());
    }
    if args.effective || !args.over.is_empty() {
        let cfg = cli::prepare_config(cli, &args.over)?;
        let text = cfg.to_toml_string().map_err(|e| e.to_string())?;
        println!("{text}");
        return Ok(());
    }
    if let Some(path) = args.write.as_deref() {
        write_text(path, &FluidConfig::template())?;
        println!("wrote {}", path.display());
        return Ok(());
    }
    print!("{}", FluidConfig::template());
    Ok(())
}

fn cmd_gui(cli: &Cli, args: &GuiArgs) -> Result<(), String> {
    let exe = gui_executable()?;
    if !cli.quiet {
        println!("opening {} ...", exe.display());
    }
    let mut cmd = std::process::Command::new(&exe);
    if let Some(path) = cli.config.as_ref() {
        cmd.arg("--config").arg(path);
    }
    if let Some(secs) = args.smoke {
        if !(secs.is_finite() && secs > 0.0) {
            return Err(format!(
                "--smoke must be a positive number of seconds, got {secs}"
            ));
        }
        cmd.arg("--smoke").arg(format!("{secs}"));
    }
    for extra in &args.extra {
        cmd.arg(extra);
    }
    let status = cmd
        .status()
        .map_err(|e| format!("cannot launch {}: {e}", exe.display()))?;
    if !status.success() {
        return Err(format!("fluid-gui exited with {status}"));
    }
    Ok(())
}

/// Find the GUI binary: FLUID_GUI_EXE, then next to this executable, then PATH.
fn gui_executable() -> Result<std::path::PathBuf, String> {
    if let Some(path) = std::env::var_os("FLUID_GUI_EXE") {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "FLUID_GUI_EXE points at {} which is not a file",
            path.display()
        ));
    }
    let name = if cfg!(windows) {
        "fluid-gui.exe"
    } else {
        "fluid-gui"
    };
    if let Ok(current) = std::env::current_exe()
        && let Some(dir) = current.parent()
    {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Ok(std::path::PathBuf::from(name))
}

fn cmd_info(cli: &Cli) -> Result<(), String> {
    if cli.json {
        println!(
            "{{\"name\":\"fluid\",\"version\":\"{}\",\"target\":\"{}-{}\"}}",
            fluid_core::VERSION,
            std::env::consts::ARCH,
            std::env::consts::OS
        );
        return Ok(());
    }
    println!("{}", fluid_core::version_string());
    println!(
        "  target      : {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("  capture     : {}", std::env::consts::OS);
    println!(
        "  default plan: {}",
        Pipeline::plan(&FluidConfig::default(), Dims::new(1920, 1080)).map_err(|e| e.to_string())?
    );
    println!();
    print_overview();
    Ok(())
}

/// One benchmark profile measured by the doctor command.
struct Probe {
    label: &'static str,
    run_args: &'static str,
    dims: Dims,
    cfg: FluidConfig,
}

/// One measured profile: what it produced and how fast.
struct Measured {
    label: &'static str,
    run_args: &'static str,
    source: Dims,
    output: Dims,
    process_ms: f32,
    output_fps: f64,
}

fn make_probe(
    label: &'static str,
    run_args: &'static str,
    dims: Dims,
    base: &FluidConfig,
    tweak: impl FnOnce(&mut FluidConfig),
) -> Result<Probe, String> {
    let mut cfg = base.clone();
    cfg.present.vsync = false;
    cfg.latency.max_queue = 1;
    tweak(&mut cfg);
    cfg.validate().map_err(|e| e.to_string())?;
    Ok(Probe {
        label,
        run_args,
        dims,
        cfg,
    })
}

fn cmd_doctor(cli: &Cli, args: &DoctorArgs) -> Result<(), String> {
    let frames = args.frames.clamp(4, 2_000);
    let target_fps = if args.target_fps.is_finite() && args.target_fps >= 1.0 {
        args.target_fps
    } else {
        60.0
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let thread_override = std::env::var("FLUID_THREADS").ok();
    let base = cli::prepare_config(cli, &cli::Overrides::default())?;

    let mut probes: Vec<Probe> = Vec::new();
    probes.push(make_probe(
        "720p -> 1440p upscale (bilinear)",
        "--scale 2 --filter bilinear --sharpness 0.3",
        Dims::new(1280, 720),
        &base,
        |cfg| {
            cfg.interpolator.mode = InterpMode::Off;
            cfg.interpolator.multiplier = 1;
            cfg.upscaler.filter = UpscaleFilter::Bilinear;
            cfg.upscaler.scale.mode = ScaleMode::Factor;
            cfg.upscaler.scale.factor = 2.0;
            cfg.upscaler.sharpness = 0.3;
        },
    )?);
    probes.push(make_probe(
        "720p 2x frame generation + 1.5x upscale",
        "--multiplier 2 --scale 1.5 --filter bilinear",
        Dims::new(1280, 720),
        &base,
        |cfg| {
            cfg.interpolator.mode = InterpMode::Fixed;
            cfg.interpolator.multiplier = 2;
            cfg.upscaler.filter = UpscaleFilter::Bilinear;
            cfg.upscaler.scale.mode = ScaleMode::Factor;
            cfg.upscaler.scale.factor = 1.5;
            cfg.upscaler.sharpness = 0.3;
        },
    )?);
    probes.push(make_probe(
        "1080p 2x frame generation (native)",
        "--multiplier 2 --scale 1.0",
        Dims::new(1920, 1080),
        &base,
        |cfg| {
            cfg.interpolator.mode = InterpMode::Fixed;
            cfg.interpolator.multiplier = 2;
            cfg.upscaler.filter = UpscaleFilter::Off;
            cfg.upscaler.scale.mode = ScaleMode::Native;
            cfg.upscaler.sharpness = 0.0;
        },
    )?);
    probes.push(make_probe(
        "1080p -> 4K upscale (bilinear)",
        "--scale 2 --filter bilinear --sharpness 0.3",
        Dims::new(1920, 1080),
        &base,
        |cfg| {
            cfg.interpolator.mode = InterpMode::Off;
            cfg.interpolator.multiplier = 1;
            cfg.upscaler.filter = UpscaleFilter::Bilinear;
            cfg.upscaler.scale.mode = ScaleMode::Factor;
            cfg.upscaler.scale.factor = 2.0;
            cfg.upscaler.sharpness = 0.3;
        },
    )?);

    // Warm up each probe first: the first frames pay for lazy buffer allocation
    // (flow pyramids, upscale scratch, temporal history) which is not steady state.
    let warmup = (frames / 4).clamp(2, 8);
    for probe in &probes {
        run_bench_once(&probe.cfg, probe.dims, warmup)?;
    }

    let mut measured: Vec<Measured> = Vec::with_capacity(probes.len());
    for probe in &probes {
        let started = std::time::Instant::now();
        let report = run_bench_once(&probe.cfg, probe.dims, frames)?;
        let wall = started.elapsed().as_secs_f64().max(1.0e-6);
        measured.push(Measured {
            label: probe.label,
            run_args: probe.run_args,
            source: probe.dims,
            output: probe.cfg.upscaler.scale.resolve(probe.dims),
            process_ms: report.stats.mean_process_ms(),
            output_fps: report.stats.frames_out as f64 / wall,
        });
    }

    if cli.json {
        let gpu = match fluid_gpu::probe() {
            Ok(info) => format!("{info} (wgpu)"),
            Err(err) => format!("unavailable: {err}"),
        };
        let mut out = format!(
            "{{\"threads\":{threads},\"target_fps\":{target_fps:.1},\"gpu\":\"{}\",\"profiles\":[",
            json_escape(&gpu)
        );
        for (i, m) in measured.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"label\":\"{}\",\"args\":\"{}\",\"source\":[{},{}],\"output\":[{},{}],\"output_fps\":{:.2},\"process_ms\":{:.2}}}",
                m.label,
                m.run_args,
                m.source.width,
                m.source.height,
                m.output.width,
                m.output.height,
                m.output_fps,
                m.process_ms
            ));
        }
        out.push_str("]}");
        println!("{out}");
        return Ok(());
    }

    println!("environment:");
    println!("  version     : {}", fluid_core::version_string());
    println!(
        "  build       : {}",
        if cfg!(debug_assertions) {
            "debug (slow - rebuild with cargo build --release)"
        } else {
            "release"
        }
    );
    println!(
        "  cpu threads : {threads}{}",
        match thread_override.as_deref() {
            Some(v) => format!(" (FLUID_THREADS={v})"),
            None => String::new(),
        }
    );
    match fluid_capture::enumerate_monitors() {
        Ok(list) if list.is_empty() => {
            println!("  displays    : none reported (headless or unsupported platform)");
        }
        Ok(list) => {
            println!("  displays    : {}", list.len());
            for monitor in list.iter().take(4) {
                println!("    {monitor}");
            }
        }
        Err(err) => println!("  displays    : unavailable: {err}"),
    }
    let capture: Vec<String> = CaptureBackendKind::ALL
        .iter()
        .map(|kind| {
            format!(
                "{}={}",
                kind.as_str(),
                if fluid_capture::backend_available(*kind) {
                    "yes"
                } else {
                    "no"
                }
            )
        })
        .collect();
    println!("  capture     : {}", capture.join(", "));
    let present: Vec<String> = backend_report()
        .into_iter()
        .map(|(name, ok)| format!("{name}={}", if ok { "yes" } else { "no" }))
        .collect();
    println!("  present     : {}", present.join(", "));
    match fluid_gpu::probe() {
        Ok(info) => println!("  gpu         : {info} (wgpu)"),
        Err(err) => println!("  gpu         : not available: {err}"),
    }

    println!();
    println!("profiles measured with {frames} frames each (null sink, no window):");
    println!(
        "  {:<40} {:>8} {:>10} {:>10}",
        "profile", "out fps", "cpu/frame", "output"
    );
    for m in &measured {
        println!(
            "  {:<40} {:>7.1} {:>9.1}ms {:>10}",
            m.label,
            m.output_fps,
            m.process_ms,
            format!("{}x{}", m.output.width, m.output.height)
        );
    }

    println!();
    let reaching: Vec<&Measured> = measured
        .iter()
        .filter(|m| m.output_fps >= target_fps)
        .collect();
    if let Some(best) = reaching.last() {
        println!(
            "recommendation: \"{}\" reaches {:.1} fps here (target {target_fps:.0} fps).",
            best.label, best.output_fps
        );
        // Capture only the resolution this profile was measured at: a bare run
        // captures the whole screen, which is far slower (QA2 defect D4).
        println!(
            "  fluid run {} --region {}x{}+0+0 --max-queue 1 --no-vsync",
            best.run_args, best.source.width, best.source.height
        );
        println!(
            "  (--region keeps the capture at {}x{}, the size this profile was measured at)",
            best.source.width, best.source.height
        );
    } else {
        let fastest = measured
            .iter()
            .max_by(|a, b| a.output_fps.total_cmp(&b.output_fps));
        if let Some(fastest) = fastest {
            println!(
                "recommendation: this machine cannot reach {target_fps:.0} fps output; the fastest profile is \"{}\" at {:.1} fps.",
                fastest.label, fastest.output_fps
            );
            println!(
                "  fluid run {} --region {}x{}+0+0 --max-queue 1 --no-vsync",
                fastest.run_args, fastest.source.width, fastest.source.height
            );
            println!(
                "  lower --target-fps, pick a smaller --region, try --backend gpu, or reduce --scale."
            );
        }
    }
    println!();
    println!(
        "note: these numbers come from a built-in synthetic source and a null output, so they are\n      a pure processing ceiling. A real screen/window capture is additionally limited by the\n      desktop compositor cadence and the source content, so expect a somewhat lower fps."
    );
    if cfg!(debug_assertions) {
        println!();
        println!(
            "warning: this is a debug build; real-world numbers are several times better in release."
        );
    }
    Ok(())
}

/// Tell the user that a report written after Ctrl+C is a partial one.
fn note_cancelled(cancel: &AtomicBool, frames_in: u64) {
    if crate::ctrl::requested(cancel) {
        eprintln!("fluid: interrupted - report covers {frames_in} input frame(s)");
    }
}

fn emit_report(
    report: &fluid_engine::PipelineReport,
    path: Option<&Path>,
    json: bool,
) -> Result<(), String> {
    if let Some(path) = path {
        write_text(path, &report.to_json())?;
    }
    if json {
        println!("{}", report.to_json());
    } else {
        println!("{report}");
    }
    Ok(())
}

fn write_text(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// True for the "-" pseudo path that selects stdin/stdout.
fn is_stdio(path: &Path) -> bool {
    path == Path::new("-")
}

fn is_y4m(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().eq_ignore_ascii_case("y4m"))
        .unwrap_or(false)
}

/// True for a .avi output path (MJPEG AVI written without ffmpeg).
fn is_avi(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().eq_ignore_ascii_case("avi"))
        .unwrap_or(false)
}

/// Frames emitted per source frame, i.e. the factor applied to the container
/// frame rate. With interpolation off the multiplier is ignored (QA round 5
/// defect D1: `fluid video --interp off` used to write fps * multiplier and the
/// resulting clip played 2x too fast).
fn video_out_multiplier(cfg: &FluidConfig) -> u32 {
    if cfg.interpolator.mode == InterpMode::Off {
        1
    } else {
        u32::from(cfg.interpolator.multiplier.clamp(1, 8))
    }
}
#[cfg(test)]
mod tests {
    use super::{
        devices_json, is_avi, is_stdio, is_y4m, json_escape, video_out_multiplier, with_throughput,
    };
    use fluid_core::{FluidConfig, InterpMode};

    #[test]
    fn throughput_fields_are_appended_to_flat_json() {
        let json = r#"{"frames_in":10,"frames_out":20}"#;
        let out = with_throughput(json, 12.5, 25.0, 0.8);
        assert!(
            out.starts_with(r#"{"frames_in":10,"frames_out":20,"#),
            "{out}"
        );
        assert!(out.contains(r#""throughput_in_fps":12.50"#), "{out}");
        assert!(out.contains(r#""throughput_out_fps":25.00"#), "{out}");
        assert!(out.contains(r#""wall_s":0.800"#), "{out}");
        assert!(out.ends_with('}'), "{out}");
        // The result must still be one JSON object: braces stay balanced.
        assert_eq!(out.matches('{').count(), out.matches('}').count());
    }

    #[test]
    fn throughput_leaves_non_objects_alone() {
        assert_eq!(with_throughput("not json", 1.0, 2.0, 0.1), "not json");
    }

    #[test]
    fn json_escape_handles_quotes_backslashes_and_controls() {
        assert_eq!(json_escape("plain"), "plain");
        assert_eq!(json_escape("DISPLAY1"), "DISPLAY1");
        assert_eq!(json_escape(r#"a"b"#), r#"a\"b"#);
        assert_eq!(json_escape(r"a\b"), r"a\\b");
        assert_eq!(json_escape("a\tb"), r"a\tb");
        assert_eq!(json_escape("a\nb"), r"a\nb");
        assert_eq!(json_escape("\u{1}"), r"\u0001");
        // Non-ASCII must survive untouched so monitor / device names stay readable.
        assert_eq!(json_escape("显示器"), "显示器");
    }

    #[test]
    fn devices_json_optionally_appends_window_list() {
        let plain = devices_json(false);
        assert!(plain.starts_with('{') && plain.ends_with('}'), "{plain}");
        assert!(!plain.contains("\"windows\""), "{plain}");
        assert_eq!(plain.matches('{').count(), plain.matches('}').count());
        let with_windows = devices_json(true);
        assert!(with_windows.contains("\"windows\":["), "{with_windows}");
        assert!(with_windows.ends_with('}'), "{with_windows}");
        assert_eq!(
            with_windows.matches('{').count(),
            with_windows.matches('}').count(),
            "{with_windows}"
        );
    }

    #[test]
    fn video_format_detection_is_case_insensitive_and_extension_driven() {
        fn p(s: &str) -> &std::path::Path {
            std::path::Path::new(s)
        }
        assert!(is_y4m(p("clip.y4m")));
        assert!(is_y4m(p("C:/tmp/CLIP.Y4M")));
        assert!(!is_y4m(p("clip.avi")));
        assert!(!is_y4m(p("clip")));
        assert!(is_avi(p("clip.avi")));
        assert!(is_avi(p("OUT.AVI")));
        assert!(!is_avi(p("clip.y4m")));
        assert!(!is_avi(p("clip.mp4")));
        assert!(is_stdio(p("-")));
        assert!(!is_stdio(p("clip.y4m")));
    }

    #[test]
    fn video_container_fps_follows_the_effective_interpolation_multiplier() {
        let mut cfg = FluidConfig::default();
        // Default config interpolates, so a 30 fps clip becomes 60 fps at 2x.
        cfg.interpolator.mode = InterpMode::Fixed;
        cfg.interpolator.multiplier = 2;
        assert_eq!(video_out_multiplier(&cfg), 2);
        cfg.interpolator.multiplier = 3;
        assert_eq!(video_out_multiplier(&cfg), 3);
        // Interpolation off: one frame per input frame even if multiplier is 4.
        cfg.interpolator.mode = InterpMode::Off;
        cfg.interpolator.multiplier = 4;
        assert_eq!(video_out_multiplier(&cfg), 1);
        // The multiplier is clamped to the range the interpolator supports.
        cfg.interpolator.mode = InterpMode::Adaptive;
        cfg.interpolator.multiplier = 99;
        assert_eq!(video_out_multiplier(&cfg), 8);
        cfg.interpolator.multiplier = 0;
        assert_eq!(video_out_multiplier(&cfg), 1);
    }
}
