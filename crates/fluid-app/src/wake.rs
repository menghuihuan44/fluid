//! Wake the minimized eframe UI when a tray/hotkey event arrives.

use eframe::egui::Context;
use std::sync::{Mutex, OnceLock};

static UI_CONTEXT: OnceLock<Mutex<Option<Context>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<Context>> {
    UI_CONTEXT.get_or_init(|| Mutex::new(None))
}

pub fn set(ctx: &Context) {
    if let Ok(mut slot) = slot().lock() {
        *slot = Some(ctx.clone());
    }
}

pub fn wake() {
    if let Ok(slot) = slot().lock() {
        if let Some(ctx) = slot.as_ref() {
            ctx.request_repaint();
        }
    }
}
