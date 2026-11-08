//! Graceful interruption: turn Ctrl+C / Ctrl+Break into a cooperative cancel.
//!
//! Windows delivers these as console control events to the whole process group,
//! and with the default handler installed it terminates the process at once
//! (exit status 0xC000013A). That kills a running pipeline before the final
//! report is written. Here we install a tiny handler that only stores a flag;
//! the engine checks that flag once per frame and returns its report normally.
//! A second Ctrl+C falls back to the default handler so a wedged process can
//! always be killed.
//!
//! Non-Windows builds keep the default SIGINT behaviour; the flag simply never
//! gets set there.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Process-wide cancellation flag, shared with the engine and the handler.
static FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Ctrl+C presses seen. The first is handled gracefully, later ones fall back
/// to the OS default so an unresponsive process can still be terminated.
#[cfg(windows)]
static PRESSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Install the console control handler and return the shared cancel flag.
///
/// Safe to call more than once: the flag is process-wide.
pub fn install() -> Arc<AtomicBool> {
    let flag = FLAG
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone();
    install_platform();
    flag
}

/// True once a Ctrl+C / Ctrl+Break asked the process to stop.
pub fn requested(flag: &AtomicBool) -> bool {
    flag.load(Ordering::Relaxed)
}

#[cfg(windows)]
fn install_platform() {
    use windows::Win32::Foundation::{BOOL, FALSE, TRUE};
    use windows::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler};

    /// Console control handler. Windows calls it on a thread of its own (this
    /// is not a Unix signal handler), so touching atomics is all we do.
    ///
    /// SAFETY: the function captures no state; it only reads process-wide
    /// statics that live for the whole process. Returning TRUE marks the event
    /// handled, FALSE lets the default handler terminate the process.
    unsafe extern "system" fn handler(ctrl_type: u32) -> BOOL {
        match ctrl_type {
            CTRL_C_EVENT | CTRL_BREAK_EVENT => {
                if PRESSES.fetch_add(1, Ordering::SeqCst) == 0 {
                    if let Some(flag) = FLAG.get() {
                        flag.store(true, Ordering::SeqCst);
                    }
                    TRUE
                } else {
                    FALSE
                }
            }
            _ => FALSE,
        }
    }

    // SAFETY: SetConsoleCtrlHandler only records the function pointer; the
    // handler matches PHANDLER_ROUTINE and stays valid for the process
    // lifetime because it is a plain function with no captured state.
    if let Err(err) = unsafe { SetConsoleCtrlHandler(Some(handler), TRUE) } {
        eprintln!("fluid: warning: cannot install console control handler: {err}");
    }
}

#[cfg(not(windows))]
fn install_platform() {
    // Nothing to install: SIGINT already terminates the process on these
    // platforms and the CLI has no terminal state to unwind.
}
