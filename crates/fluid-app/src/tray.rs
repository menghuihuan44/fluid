//! Tray icon thread with its own Win32 message pump.
//!
//! The TrayIcon object is kept alive inside the thread and dropped when the
//! thread receives WM_QUIT. This removes the shell icon synchronously instead
//! of relying on process termination.

use crate::pipeline::Command;
use fluid_core::{RunState, SharedState};
use font8x8::UnicodeFonts;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, Once, OnceLock};
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
use windows::Win32::Foundation::{BOOL, HWND, LPARAM, WPARAM};
use windows::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, EnumWindows, GetClassNameW, GetMessageW, GetWindowTextW,
    GetWindowThreadProcessId, PostThreadMessageW, SetForegroundWindow, ShowWindowAsync,
    TranslateMessage, MSG, SW_RESTORE, WM_QUIT,
};

pub const ID_OPEN: &str = "fluid.open";
pub const ID_QUIT: &str = "fluid.quit";

static TRAY_THREAD_ID: AtomicU32 = AtomicU32::new(0);
static MENU_EVENTS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
static COMMAND_SENDER: OnceLock<Mutex<Option<Sender<Command>>>> = OnceLock::new();
static SHARED_STATE: OnceLock<Arc<SharedState>> = OnceLock::new();
static OPENING: AtomicBool = AtomicBool::new(false);

pub fn set_command_sender(sender: Sender<Command>) {
    let slot = COMMAND_SENDER.get_or_init(|| Mutex::new(None));
    if let Ok(mut slot) = slot.lock() {
        *slot = Some(sender);
    }
}

pub fn set_shared_state(shared: Arc<SharedState>) {
    let _ = SHARED_STATE.set(shared);
}

fn menu_events() -> &'static Mutex<Vec<String>> {
    MENU_EVENTS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Drain tray menu actions. The handler below also wakes a minimized UI.
pub fn take_events() -> Vec<String> {
    menu_events()
        .lock()
        .map(|mut events| std::mem::take(&mut *events))
        .unwrap_or_default()
}

pub fn spawn() {
    static HANDLER_ONCE: Once = Once::new();
    HANDLER_ONCE.call_once(|| {
        MenuEvent::set_event_handler(Some(|event: MenuEvent| {
            let id = event.id().0.clone();
            match id.as_str() {
                ID_OPEN => request_open_settings(),
                ID_QUIT => {
                    if let Some(slot) = COMMAND_SENDER.get() {
                        if let Ok(slot) = slot.lock() {
                            if let Some(sender) = slot.as_ref() {
                                let _ = sender.send(Command::Stop);
                            }
                        }
                    }
                    if let Ok(mut events) = menu_events().lock() {
                        events.push(ID_QUIT.to_string());
                    }
                    crate::wake::wake();
                    // Prefer the normal UI shutdown path so capture overlays,
                    // HUD resources, config, tray icon and hotkeys are released
                    // in order. Keep a bounded fallback for a wedged UI thread.
                    std::thread::spawn(|| {
                        std::thread::sleep(std::time::Duration::from_millis(1500));
                        shutdown();
                        std::thread::sleep(std::time::Duration::from_millis(150));
                        std::process::exit(0);
                    });
                }
                _ => {
                    if let Ok(mut events) = menu_events().lock() {
                        events.push(id);
                    }
                    crate::wake::wake();
                }
            }
        }));
        TrayIconEvent::set_event_handler(Some(|event: TrayIconEvent| match event {
            TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            }
            | TrayIconEvent::DoubleClick {
                button: MouseButton::Left,
                ..
            } => request_open_settings(),
            _ => {}
        }));
    });
    let _ = std::thread::Builder::new()
        .name("fluid-tray".into())
        .spawn(|| {
            let tid = unsafe { GetCurrentThreadId() };
            TRAY_THREAD_ID.store(tid, Ordering::SeqCst);
            if let Err(e) = tray_thread() {
                tracing::error!("tray initialization failed: {e}");
            }
            TRAY_THREAD_ID.store(0, Ordering::SeqCst);
        });
}

pub fn request_open_settings() {
    if OPENING.swap(true, Ordering::SeqCst) {
        return;
    }
    // Pause the topmost capture overlay while the settings window is open.
    // The persistent capture session stays alive and resumes automatically when
    // the settings window loses focus.
    if let Some(slot) = COMMAND_SENDER.get() {
        if let Ok(slot) = slot.lock() {
            if let Some(sender) = slot.as_ref() {
                let _ = sender.send(Command::SetPaused(true));
            }
        }
    }
    if let Ok(mut events) = menu_events().lock() {
        events.push(ID_OPEN.to_string());
    }
    crate::wake::wake();

    let shared = SHARED_STATE.get().cloned();
    let _ = std::thread::Builder::new()
        .name("fluid-open-settings".into())
        .spawn(move || {
            // Wait until the pipeline has dropped its overlay. If it is stuck,
            // still restore the UI after a bounded delay so the tray cannot
            // become a permanent trap.
            if let Some(shared) = shared {
                for _ in 0..40 {
                    if matches!(
                        *shared.state.lock(),
                        RunState::Paused | RunState::Idle | RunState::Failed
                    ) {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(25));
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(120));
            }
            show_main_window();
            crate::wake::wake();
            OPENING.store(false, Ordering::SeqCst);
        });
}

/// Ask the tray thread to exit and drop its icon. Safe to call from the UI
/// thread; WM_QUIT wakes GetMessageW immediately.
pub fn shutdown() {
    let tid = TRAY_THREAD_ID.swap(0, Ordering::SeqCst);
    if tid != 0 {
        unsafe {
            let _ = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0));
        }
    }
}

fn tray_thread() -> Result<(), Box<dyn std::error::Error>> {
    let icon = tray_icon::Icon::from_rgba(icon_rgba(), 32, 32)?;
    let open = MenuItem::with_id(ID_OPEN, "打开设置", true, None);
    let quit = MenuItem::with_id(ID_QUIT, "完全退出", true, None);
    let menu = Menu::new();
    menu.append(&open)?;
    menu.append(&quit)?;

    let _tray: TrayIcon = TrayIconBuilder::new()
        .with_icon(icon)
        .with_tooltip("fluid — 帧生成 + 画质提升")
        .with_menu(Box::new(menu))
        .with_menu_on_left_click(false)
        .build()?;

    unsafe {
        let mut msg = MSG::default();
        loop {
            let r = GetMessageW(&mut msg, None, 0, 0);
            if !r.as_bool() {
                break;
            }
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    Ok(())
}

struct FindMainWindow {
    pid: u32,
    result: Option<HWND>,
}

unsafe extern "system" fn find_main_window_cb(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let context = &mut *(lparam.0 as *mut FindMainWindow);
    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    if pid != context.pid {
        return true.into();
    }
    let mut title = [0u16; 128];
    let title_len = unsafe { GetWindowTextW(hwnd, &mut title) }.max(0) as usize;
    let title = String::from_utf16_lossy(&title[..title_len.min(title.len())]);
    let mut class = [0u16; 128];
    let class_len = unsafe { GetClassNameW(hwnd, &mut class) }.max(0) as usize;
    let class = String::from_utf16_lossy(&class[..class_len.min(class.len())]);
    if title == "fluid" && class != "FluidOverlayWnd" {
        context.result = Some(hwnd);
        return false.into();
    }
    true.into()
}

/// Restore the settings window directly from the tray thread. This works even
/// when eframe stops processing repaint requests for a minimized window.
fn show_main_window() {
    let mut context = FindMainWindow {
        pid: unsafe { GetCurrentProcessId() },
        result: None,
    };
    unsafe {
        let _ = EnumWindows(
            Some(find_main_window_cb),
            LPARAM(&mut context as *mut FindMainWindow as isize),
        );
        if let Some(hwnd) = context.result {
            let _ = ShowWindowAsync(hwnd, SW_RESTORE);
            let _ = SetForegroundWindow(hwnd);
        }
    }
}

/// Programmatically generated tray icon: blue-cyan gradient disc + white "F".
fn icon_rgba() -> Vec<u8> {
    const S: usize = 32;
    let mut v = vec![0u8; S * S * 4];
    let c = (S as f32 - 1.0) / 2.0;
    for y in 0..S {
        for x in 0..S {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let d = (dx * dx + dy * dy).sqrt();
            if d <= 15.0 {
                let i = (y * S + x) * 4;
                let t = d / 15.0;
                v[i] = (225.0 - 185.0 * t) as u8;
                v[i + 1] = (85.0 + 105.0 * t) as u8;
                v[i + 2] = (25.0 + 15.0 * t) as u8;
                v[i + 3] = 255;
            }
        }
    }
    let glyph = font8x8::BASIC_FONTS.get('F').unwrap_or([0u8; 8]);
    let ox = (32 - 24) / 2;
    let oy = (32 - 24) / 2;
    for (row, &bits) in glyph.iter().enumerate() {
        for col in 0..8 {
            if bits & (1 << col) != 0 {
                for sy in 0..3u32 {
                    for sx in 0..3u32 {
                        let px = ox + col as u32 * 3 + sx;
                        let py = oy + row as u32 * 3 + sy;
                        let inside = ((px as f32 - c) * (px as f32 - c)
                            + (py as f32 - c) * (py as f32 - c))
                            .sqrt()
                            <= 15.0;
                        if px < 32 && py < 32 && inside {
                            let i = ((py * 32 + px) * 4) as usize;
                            v[i] = 255;
                            v[i + 1] = 255;
                            v[i + 2] = 255;
                            v[i + 3] = 255;
                        }
                    }
                }
            }
        }
    }
    v
}
