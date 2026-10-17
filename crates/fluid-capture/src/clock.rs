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
