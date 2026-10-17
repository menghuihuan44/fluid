#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod hotkeys;
mod pipeline;
mod tray;
mod ui;
mod wake;

use fluid_core::AppConfig;
use std::sync::Arc;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn init_logging() -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let logs_dir = AppConfig::config_path().and_then(|p| p.parent().map(|d| d.join("logs")));
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stdout));
    match logs_dir {
        Some(d) if std::fs::create_dir_all(&d).is_ok() => {
            let appender = tracing_appender::rolling::daily(&d, "fluid.log");
            let (nb, guard) = tracing_appender::non_blocking(appender);
            registry.with(fmt::layer().with_writer(nb)).init();
            Some(guard)
        }
        _ => {
            registry.init();
            None
        }
    }
}

fn main() -> eframe::Result<()> {
    let _log_guard = init_logging();
    tracing::info!("fluid v{} starting", env!("CARGO_PKG_VERSION"));

    unsafe {
        let _ = windows::Win32::UI::HiDpi::SetProcessDpiAwarenessContext(
            windows::Win32::UI::HiDpi::DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        );
    }

    let (cfg, _) = AppConfig::load();
    fluid_capture::start_foreground_watcher();
    let initial_foreground =
        fluid_capture::external_foreground_window().and_then(fluid_capture::window_info);
    let ui_cfg = cfg.clone();
    let hotkeys_enabled = cfg.display.global_hotkeys;
    let shared = Arc::new(fluid_core::SharedState::default());
    let pipe = pipeline::Pipeline::spawn(shared.clone(), cfg);
    tray::set_command_sender(pipe.command_sender());
    tray::spawn();
    hotkeys::spawn(hotkeys_enabled, pipe.command_sender(), shared.clone());

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("fluid")
            .with_inner_size([560.0, 820.0])
            .with_min_inner_size([480.0, 620.0]),
        vsync: false,
        multisampling: 0,
        depth_buffer: 0,
        stencil_buffer: 0,
        ..Default::default()
    };

    let result = eframe::run_native(
        "fluid",
        options,
        Box::new(move |cc| {
            Ok(Box::new(ui::FluidApp::new(
                cc,
                shared,
                pipe,
                ui_cfg,
                initial_foreground,
            )))
        }),
    );

    // Make sure the shell icon and global hotkeys are removed even if the
    // window was closed through the native title bar.
    tray::shutdown();
    hotkeys::shutdown();
    std::thread::sleep(std::time::Duration::from_millis(60));
    result
}
