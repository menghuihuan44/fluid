pub mod clock;
pub mod duplication;
pub mod monitor;
pub mod wgc;
pub mod window;

pub use clock::{qpc_freq, qpc_now};
pub use duplication::{CaptureError, Duplication};
pub use monitor::{enumerate_monitors, monitor_from_window, monitor_refresh_hz, MonitorInfo};
pub use wgc::{WgcCapture, WgcFrame};
pub use window::{
    client_rect_on_screen, enumerate_windows, extended_frame_bounds, external_foreground_window,
    find_window_by_ref, foreground_window, hwnd_from_isize, is_iconic, is_window,
    last_external_foreground_window, monitor_of, start_foreground_watcher, uv_rect, window_info,
    window_process_id, window_title, WinEnum,
};
