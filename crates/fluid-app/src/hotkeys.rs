//! Global hotkeys on a dedicated Win32 message thread.
//!
//! Ctrl+Alt+Shift+F9: start/stop capture
//! Ctrl+Alt+Shift+F8: open settings and stop capture
//! Ctrl+Alt+Shift+F10: toggle HUD
//!
//! Start/stop and HUD commands are sent directly to the pipeline thread so they
//! keep working while the settings window is minimized.

use crate::pipeline::{CaptureRequest, Command};
use fluid_core::{RunState, SharedState};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, OnceLock};
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    RegisterHotKey, UnregisterHotKey, MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT, VK_F10, VK_F8,
    VK_F9,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetMessageW, PostThreadMessageW, TranslateMessage, MSG, WM_HOTKEY, WM_QUIT,
};

const HOTKEY_START_STOP: i32 = 1;
const HOTKEY_TOGGLE_HUD: i32 = 2;
const HOTKEY_OPEN_SETTINGS: i32 = 3;
const WM_REFRESH_HOTKEYS: u32 = 0x8000 + 1;

static THREAD_ID: AtomicU32 = AtomicU32::new(0);
static ENABLED: AtomicBool = AtomicBool::new(false);
static PLAN: OnceLock<Mutex<Option<HotkeyPlan>>> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
pub enum HotkeyPlan {
    Monitor {
        hmonitor: isize,
    },
    Window {
        hwnd: isize,
    },
    /// Resolve the actual foreground window when the hotkey is pressed.
    Foreground {
        fallback_monitor: isize,
    },
}

fn plan_slot() -> &'static Mutex<Option<HotkeyPlan>> {
    PLAN.get_or_init(|| Mutex::new(None))
}

pub fn set_plan(plan: Option<HotkeyPlan>) {
    if let Ok(mut slot) = plan_slot().lock() {
        *slot = plan;
    }
}

pub fn spawn(enabled: bool, commands: Sender<Command>, shared: Arc<SharedState>) {
    ENABLED.store(enabled, Ordering::SeqCst);
    let _ = std::thread::Builder::new()
        .name("fluid-hotkeys".into())
        .spawn(move || {
            let tid = unsafe { GetCurrentThreadId() };
            THREAD_ID.store(tid, Ordering::SeqCst);
            hotkey_thread(commands, shared);
            THREAD_ID.store(0, Ordering::SeqCst);
        });
}

/// Enable or disable hotkey registration. Safe to call from the UI thread.
pub fn set_enabled(enabled: bool) {
    if ENABLED.swap(enabled, Ordering::SeqCst) != enabled {
        let tid = THREAD_ID.load(Ordering::SeqCst);
        if tid != 0 {
            unsafe {
                let _ = PostThreadMessageW(tid, WM_REFRESH_HOTKEYS, WPARAM(0), LPARAM(0));
            }
        }
    }
}

pub fn shutdown() {
    let tid = THREAD_ID.swap(0, Ordering::SeqCst);
    if tid != 0 {
        unsafe {
            let _ = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0));
        }
    }
}

fn resolve_plan() -> Option<CaptureRequest> {
    let plan = (*plan_slot().lock().ok()?)?;
    match plan {
        HotkeyPlan::Monitor { hmonitor } => Some(CaptureRequest::Monitor { hmonitor }),
        HotkeyPlan::Window { hwnd } => {
            let window = fluid_capture::window::hwnd_from_isize(hwnd);
            fluid_capture::is_window(window).then_some(CaptureRequest::Window { hwnd })
        }
        HotkeyPlan::Foreground { fallback_monitor } => {
            Some(CaptureRequest::Foreground { fallback_monitor })
        }
    }
}

fn toggle_capture(commands: &Sender<Command>, shared: &SharedState) {
    let state = shared.state.lock().clone();
    if matches!(
        state,
        RunState::Running | RunState::Starting | RunState::Paused
    ) {
        let _ = commands.send(Command::Stop);
    } else if let Some(request) = resolve_plan() {
        let _ = commands.send(Command::Start { request });
        shared.minimize_ui_requested.store(true, Ordering::Relaxed);
        crate::wake::wake();
    }
}

fn update_registration() {
    let modifiers = MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_NOREPEAT;
    unsafe {
        if ENABLED.load(Ordering::SeqCst) {
            if let Err(e) = RegisterHotKey(None, HOTKEY_START_STOP, modifiers, VK_F9.0 as u32) {
                tracing::warn!("failed to register Ctrl+Alt+Shift+F9: {e}");
            }
            if let Err(e) = RegisterHotKey(None, HOTKEY_TOGGLE_HUD, modifiers, VK_F10.0 as u32) {
                tracing::warn!("failed to register Ctrl+Alt+Shift+F10: {e}");
            }
            if let Err(e) = RegisterHotKey(None, HOTKEY_OPEN_SETTINGS, modifiers, VK_F8.0 as u32) {
                tracing::warn!("failed to register Ctrl+Alt+Shift+F8: {e}");
            }
        } else {
            let _ = UnregisterHotKey(None, HOTKEY_START_STOP);
            let _ = UnregisterHotKey(None, HOTKEY_TOGGLE_HUD);
            let _ = UnregisterHotKey(None, HOTKEY_OPEN_SETTINGS);
        }
    }
}

fn hotkey_thread(commands: Sender<Command>, shared: Arc<SharedState>) {
    update_registration();
    unsafe {
        let mut msg = MSG::default();
        loop {
            let result = GetMessageW(&mut msg, None, 0, 0);
            if !result.as_bool() {
                break;
            }
            if msg.message == WM_REFRESH_HOTKEYS {
                update_registration();
            } else if msg.message == WM_HOTKEY {
                match msg.wParam.0 as i32 {
                    HOTKEY_START_STOP => {
                        tracing::info!("global hotkey start/stop");
                        toggle_capture(&commands, &shared);
                    }
                    HOTKEY_TOGGLE_HUD => {
                        tracing::info!("global hotkey HUD toggle");
                        let _ = commands.send(Command::ToggleHud);
                    }
                    HOTKEY_OPEN_SETTINGS => {
                        tracing::info!("global hotkey open settings");
                        crate::tray::request_open_settings();
                    }
                    _ => {}
                }
            } else {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        let _ = UnregisterHotKey(None, HOTKEY_START_STOP);
        let _ = UnregisterHotKey(None, HOTKEY_TOGGLE_HUD);
        let _ = UnregisterHotKey(None, HOTKEY_OPEN_SETTINGS);
    }
}
