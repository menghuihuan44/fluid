//! Lightweight liveness monitor for the capture and UI threads.
//!
//! It never touches GPU or window state; it only reads atomic heartbeats and
//! writes a warning when a thread stops advancing.

use fluid_core::{RunState, SharedState};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

pub fn spawn(shared: Arc<SharedState>) {
    let _ = std::thread::Builder::new()
        .name("fluid-watchdog".into())
        .spawn(move || {
            let mut pipeline_misses = 0u32;
            let mut ui_misses = 0u32;
            let mut hud_misses = 0u32;
            let mut last_pipeline = shared.pipeline_heartbeat.load(Ordering::Relaxed);
            let mut last_ui = shared.ui_heartbeat.load(Ordering::Relaxed);
            let mut last_hud = shared.hud_heartbeat.load(Ordering::Relaxed);
            let mut pipeline_reported = false;
            let mut ui_reported = false;
            let mut hud_reported = false;
            loop {
                std::thread::sleep(Duration::from_secs(2));
                let state = shared.state.lock().clone();
                let capturing = matches!(state, RunState::Starting | RunState::Running);

                let pipeline = shared.pipeline_heartbeat.load(Ordering::Relaxed);
                if capturing {
                    if pipeline == last_pipeline {
                        pipeline_misses += 1;
                    } else {
                        pipeline_misses = 0;
                        if pipeline_reported {
                            tracing::info!("watchdog: pipeline heartbeat recovered");
                            pipeline_reported = false;
                        }
                    }
                    if pipeline_misses >= 3 && !pipeline_reported {
                        let stage = shared.pipeline_stage.load(Ordering::Relaxed);
                        tracing::warn!(
                            "watchdog: pipeline stalled ~{}s at stage {} ({}) cleanup_pending={}",
                            pipeline_misses * 2,
                            crate::pipeline::stage_name(stage),
                            stage,
                            fluid_capture::wgc::wgc_cleanup_pending()
                        );
                        pipeline_reported = true;
                    }
                } else {
                    pipeline_misses = 0;
                }
                last_pipeline = pipeline;

                let ui = shared.ui_heartbeat.load(Ordering::Relaxed);
                if ui == last_ui {
                    ui_misses += 1;
                } else {
                    ui_misses = 0;
                    if ui_reported {
                        tracing::info!("watchdog: UI heartbeat recovered");
                        ui_reported = false;
                    }
                }
                if ui_misses >= 4 && !ui_reported {
                    tracing::warn!("watchdog: UI thread heartbeat stalled ~{}s", ui_misses * 2);
                    ui_reported = true;
                }
                last_ui = ui;

                let hud = shared.hud_heartbeat.load(Ordering::Relaxed);
                if hud == last_hud {
                    hud_misses += 1;
                } else {
                    hud_misses = 0;
                    if hud_reported {
                        tracing::info!("watchdog: HUD heartbeat recovered");
                        hud_reported = false;
                    }
                }
                if hud_misses >= 3 && shared.hud_enabled.load(Ordering::Relaxed) && !hud_reported {
                    tracing::warn!(
                        "watchdog: HUD thread heartbeat stalled ~{}s",
                        hud_misses * 2
                    );
                    hud_reported = true;
                }
                last_hud = hud;
            }
        });
}
