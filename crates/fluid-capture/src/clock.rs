use std::sync::OnceLock;

pub fn qpc_now() -> i64 {
    let mut v = 0i64;
    unsafe {
        let _ = windows::Win32::System::Performance::QueryPerformanceCounter(&mut v);
    }
    v
}

pub fn qpc_freq() -> u64 {
    static FREQ: OnceLock<u64> = OnceLock::new();
    *FREQ.get_or_init(|| {
        let mut f = 0i64;
        unsafe {
            let _ = windows::Win32::System::Performance::QueryPerformanceFrequency(&mut f);
        }
        (f.max(1)) as u64
    })
}

/// Convert a WinRT `TimeSpan` expressed in 100 ns system-relative units to QPC
/// ticks. WGC reports frame timestamps this way; using them instead of the
/// polling instant removes up to one output period of interpolation-phase error.
pub fn system_relative_100ns_to_qpc(units: i64) -> i64 {
    if units <= 0 {
        return 0;
    }
    let ticks = units as i128 * qpc_freq() as i128 / 10_000_000;
    ticks.clamp(0, i64::MAX as i128) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hundred_ns_conversion_is_positive_and_monotonic() {
        assert_eq!(system_relative_100ns_to_qpc(0), 0);
        assert!(system_relative_100ns_to_qpc(10_000_000) > 0);
        assert!(
            system_relative_100ns_to_qpc(20_000_000) > system_relative_100ns_to_qpc(10_000_000)
        );
    }
}
