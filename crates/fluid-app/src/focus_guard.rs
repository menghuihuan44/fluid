//! Safety guard: if the settings window is brought to the foreground while
//! capture is running, stop capture before its topmost overlay can obscure the
//! settings UI.

use crate::pipeline::Command;
use fluid_core::{RunState, SharedState};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::HWND;
use windows::Win32::System::Threading::GetCurrentProcessId;
use windows::Win32::UI::WindowsAndMessaging::{
    GetClassNameW, GetForegroundWindow, GetWindowTextW, GetWindowThreadProcessId, IsIconic,
};

pub fn spawn(shared: Arc<SharedState>, commands: Sender<Command>) {
    let pid = unsafe { GetCurrentProcessId() };
    let _ = std::thread::Builder::new()
        .name("fluid-focus-guard".into())
        .spawn(move || {
            let mut paused_by_guard = false;
            let mut resume_since: Option<Instant> = None;
            loop {
                std::thread::sleep(Duration::from_millis(120));
                let state = shared.state.lock().clone();
                if !matches!(
                    state,
                    RunState::Starting | RunState::Running | RunState::Paused
                ) {
                    paused_by_guard = false;
                    resume_since = None;
                    continue;
                }
                let foreground = unsafe { GetForegroundWindow() };
                let settings_foreground =
                    is_settings_window(foreground, pid) && !unsafe { IsIconic(foreground).as_bool() };
                if settings_foreground {
                    resume_since = None;
                    if !paused_by_guard {
                        tracing::info!(
                            "settings window is foreground; pausing the capture overlay (capture stays alive)"
                        );
                        let _ = commands.send(Command::SetPaused(true));
                        paused_by_guard = true;
                    }
                } else if paused_by_guard {
                    let since = *resume_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= Duration::from_millis(250) {
                        tracing::info!(
                            "settings window lost focus; resuming the capture overlay"
                        );
                        let _ = commands.send(Command::SetPaused(false));
                        paused_by_guard = false;
                        resume_since = None;
                    }
                }
            }
        });
}

fn is_settings_window(hwnd: HWND, pid: u32) -> bool {
    if hwnd.0.is_null() {
        return false;
    }
    let mut window_pid = 0u32;
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut window_pid)) };
    if window_pid != pid {
        return false;
    }
    let mut class = [0u16; 128];
    let len = unsafe { GetClassNameW(hwnd, &mut class) }.max(0) as usize;
    let class = String::from_utf16_lossy(&class[..len.min(class.len())]);
    if class == "FluidOverlayWnd" {
        return false;
    }
    let mut title = [0u16; 128];
    let len = unsafe { GetWindowTextW(hwnd, &mut title) }.max(0) as usize;
    let title = String::from_utf16_lossy(&title[..len.min(title.len())]);
    title == "fluid"
}
