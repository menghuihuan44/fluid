//! 无头自检工具：
//!   fluid-testkit render [秒]  — WARP 渲染管线正确性（CI 可跑）
//!   fluid-testkit capture [秒] — 实机 DDA 采集探针（帧率 + 捕获延迟）

use fluid_capture::clock::{qpc_freq, qpc_now};
use fluid_capture::duplication::Duplication;
use fluid_capture::monitor::enumerate_monitors;
use fluid_core::time::qpc_diff_us;
use std::time::{Duration, Instant};
use windows::Win32::Graphics::Gdi::HMONITOR;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("render") => {
            if let Err(e) = fluid_gfx::selftest::render_identity() {
                eprintln!("render identity selftest: FAIL: {e}");
                std::process::exit(1);
            }
            println!("render identity selftest: PASS");
            if let Err(e) = fluid_gfx::selftest::render_motion_interpolation() {
                eprintln!("render motion/upscale selftest: FAIL: {e}");
                std::process::exit(1);
            }
            println!("render motion/upscale selftest: PASS");
            if let Err(e) = fluid_gfx::selftest::render_hud_window() {
                eprintln!("render independent HUD selftest: FAIL: {e}");
                std::process::exit(1);
            }
            println!("render independent HUD selftest: PASS");
        }
        Some("windows") => {
            for w in fluid_capture::enumerate_windows() {
                println!(
                    "{:x}\t{}x{}\t{}\t{}",
                    w.hwnd,
                    w.rect.2 - w.rect.0,
                    w.rect.3 - w.rect.1,
                    w.class,
                    w.title
                );
            }
        }
        Some("hud") => match fluid_gfx::selftest::render_hud_window() {
            Ok(()) => println!("independent HUD selftest: PASS"),
            Err(e) => {
                eprintln!("independent HUD selftest: FAIL: {e}");
                std::process::exit(1);
            }
        },
        Some("overlay") => match fluid_gfx::selftest::overlay_creation() {
            Ok(()) => println!("overlay selftest: PASS"),
            Err(e) => {
                eprintln!("overlay selftest: FAIL: {e}");
                std::process::exit(1);
            }
        },
        Some("wgc") => {
            if let Err(e) = wgc_probe() {
                eprintln!("WGC probe: FAIL: {e}");
                std::process::exit(1);
            }
        }
        Some("wgc-scan") => {
            if let Err(e) = wgc_scan() {
                eprintln!("WGC scan: FAIL: {e}");
                std::process::exit(1);
            }
        }
        Some("feedback") => {
            if let Err(e) = feedback_probe() {
                eprintln!("capture feedback probe: FAIL: {e}");
                std::process::exit(1);
            }
        }
        Some("capture") => {
            let secs: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            capture_probe(secs);
        }
        _ => {
            eprintln!("用法: fluid-testkit [render|hud|windows|overlay|wgc|wgc-scan|feedback|capture [秒]]");
            std::process::exit(2);
        }
    }
}

fn capture_probe(secs: f64) {
    let monitors = enumerate_monitors();
    if monitors.is_empty() {
        eprintln!("未找到显示器");
        std::process::exit(1);
    }
    for m in &monitors {
        println!(
            "显示器 {} @ {:.0}Hz rect=({},{}),({},{}{})",
            m.name,
            m.refresh_hz,
            m.rect.0,
            m.rect.1,
            m.rect.2,
            m.rect.3,
            if m.hmonitor == monitors[0].hmonitor {
                " [主]"
            } else {
                ""
            }
        );
    }
    let mon = monitors[0].hmonitor;
    let hmon = HMONITOR(mon as *mut core::ffi::c_void);

    let bundle = match fluid_gfx::create_device() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("D3D11 设备创建失败: {e}");
            std::process::exit(1);
        }
    };
    let mut dup = match Duplication::new_for_monitor(&bundle.device, hmon) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("桌面复制初始化失败: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "正在采集 {} ({}x{}) {:.1}s ...",
        monitors[0].name,
        dup.width(),
        dup.height(),
        secs
    );

    let freq = qpc_freq();
    let start = Instant::now();
    let mut frames = 0u64;
    let mut sum_lat_us = 0u64;
    let mut max_lat_us = 0u64;
    let mut stalls = 0u64;
    while start.elapsed().as_secs_f64() < secs {
        match dup.acquire_latest() {
            Ok(Some(f)) => {
                frames += 1;
                let lat = qpc_diff_us(f.qpc, qpc_now(), freq);
                sum_lat_us += lat;
                max_lat_us = max_lat_us.max(lat);
                dup.release_frame();
            }
            Ok(None) => {
                stalls += 1;
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(e) => {
                eprintln!("采集错误，重建: {e}");
                std::thread::sleep(Duration::from_millis(200));
                match Duplication::new_for_monitor(&bundle.device, hmon) {
                    Ok(d) => dup = d,
                    Err(e2) => {
                        eprintln!("重建失败: {e2}");
                        std::process::exit(1);
                    }
                }
            }
        }
    }
    let el = start.elapsed().as_secs_f64();
    println!(
        "结果: {frames} 帧 / {el:.2}s = {:.1} fps | 空轮询 {stalls} 次 | 捕获→读取延迟 avg {:.2}ms max {:.2}ms",
        frames as f64 / el,
        sum_lat_us as f64 / frames.max(1) as f64 / 1000.0,
        max_lat_us as f64 / 1000.0,
    );
    let _ = qpc_freq();
}

fn poll_duplication(dup: &mut Duplication, duration: Duration) -> Result<u32, String> {
    let start = Instant::now();
    let mut updated = 0u32;
    while start.elapsed() < duration {
        match dup.acquire_latest() {
            Ok(Some(frame)) => {
                if frame.desktop_updated {
                    updated += 1;
                }
                dup.release_frame();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(1)),
            Err(e) => return Err(format!("capture error: {e}")),
        }
    }
    Ok(updated)
}

fn feedback_probe() -> Result<(), String> {
    let monitors = enumerate_monitors();
    let monitor = monitors.first().ok_or_else(|| "no monitor".to_string())?;
    let bundle = fluid_gfx::create_device().map_err(|e| format!("device: {e}"))?;
    if bundle.is_warp {
        return Err("feedback probe requires a hardware D3D11 device".into());
    }
    let mut dup = Duplication::new_for_monitor(
        &bundle.device,
        HMONITOR(monitor.hmonitor as *mut core::ffi::c_void),
    )
    .map_err(|e| format!("duplication: {e}"))?;
    let mut overlay = fluid_gfx::OverlayWindow::new(&bundle.device, &bundle.factory2, 640, 360)
        .map_err(|e| format!("overlay: {e}"))?;
    overlay
        .place(&windows::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
        })
        .map_err(|e| format!("overlay place: {e}"))?;
    let rtv = overlay
        .rtv()
        .cloned()
        .ok_or_else(|| "overlay rtv missing".to_string())?;

    overlay.show(false);
    let hidden = poll_duplication(&mut dup, Duration::from_secs(2))?;

    overlay.show(true);
    let start = Instant::now();
    let mut visible_updates = 0u32;
    let mut present_count = 0u32;
    let mut newer_updates = 0u32;
    let mut older_or_equal = 0u32;
    let mut last_present_qpc = 0i64;
    let mut delta_samples: Vec<f64> = Vec::new();
    let mut next_present = Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        if Instant::now() >= next_present {
            unsafe {
                bundle
                    .ctx
                    .ClearRenderTargetView(&rtv, &[0.1, 0.2, 0.3, 1.0]);
            }
            overlay.present(0);
            last_present_qpc = qpc_now();
            present_count += 1;
            next_present += Duration::from_millis(11);
        }
        match dup.acquire_latest() {
            Ok(Some(frame)) => {
                if frame.desktop_updated {
                    visible_updates += 1;
                    if frame.qpc > last_present_qpc {
                        newer_updates += 1;
                    } else {
                        older_or_equal += 1;
                    }
                    if delta_samples.len() < 8 {
                        delta_samples.push(
                            (frame.qpc - last_present_qpc) as f64 / qpc_freq() as f64 * 1000.0,
                        );
                    }
                }
                dup.release_frame();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(1)),
            Err(e) => return Err(format!("capture error: {e}")),
        }
    }
    overlay.show(false);

    println!(
        "feedback probe: hidden_updates={hidden}/2s, visible_updates={visible_updates}/2s, presents={present_count}, newer={newer_updates}, older_or_equal={older_or_equal}"
    );
    println!("signed frame-vs-last-present deltas (ms): {delta_samples:?}");
    Ok(())
}

fn wgc_probe() -> Result<(), String> {
    let windows = fluid_capture::enumerate_windows();
    let target = windows
        .iter()
        .find(|window| window.width() >= 640 && window.height() >= 360)
        .ok_or_else(|| "no target window found".to_string())?;
    println!(
        "WGC target: {} ({}x{})",
        window_label(target),
        target.width(),
        target.height()
    );
    let bundle = fluid_gfx::create_device().map_err(|e| format!("device: {e}"))?;
    if bundle.is_warp {
        return Err("WGC probe requires a hardware D3D11 device".into());
    }
    let mut capture = fluid_capture::WgcCapture::new(
        &bundle.device,
        fluid_capture::window::hwnd_from_isize(target.hwnd),
    )
    .map_err(|e| format!("WGC init: {e}"))?;
    let start = Instant::now();
    let mut frames = 0u32;
    let mut polls = 0u32;
    while start.elapsed() < Duration::from_secs(2) {
        polls += 1;
        match capture.acquire_latest() {
            Ok(Some(_)) => {
                frames += 1;
                capture.release_frame();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(1)),
            Err(e) => return Err(format!("WGC acquire: {e}")),
        }
    }
    println!(
        "WGC probe: frames={frames}/2s polls={polls} size={}x{}",
        capture.width(),
        capture.height()
    );
    Ok(())
}

fn wgc_scan() -> Result<(), String> {
    let windows = fluid_capture::enumerate_windows();
    let bundle = fluid_gfx::create_device().map_err(|e| format!("device: {e}"))?;
    if bundle.is_warp {
        return Err("WGC scan requires a hardware D3D11 device".into());
    }
    let mut ok = 0u32;
    let mut fail = 0u32;
    for window in windows
        .iter()
        .filter(|w| w.width() >= 160 && w.height() >= 120)
    {
        let hwnd = fluid_capture::window::hwnd_from_isize(window.hwnd);
        match fluid_capture::WgcCapture::new(&bundle.device, hwnd) {
            Ok(capture) => {
                ok += 1;
                println!(
                    "OK   {:x}\t{}x{}\t{}\t{}",
                    window.hwnd,
                    window.width(),
                    window.height(),
                    window.class,
                    window.title
                );
                drop(capture);
            }
            Err(e) => {
                fail += 1;
                println!(
                    "FAIL {:x}\t{}x{}\t{}\t{}\t{}",
                    window.hwnd,
                    window.width(),
                    window.height(),
                    window.class,
                    window.title,
                    e
                );
            }
        }
    }
    println!("WGC scan result: ok={ok} fail={fail}");
    Ok(())
}

fn window_label(window: &fluid_capture::WinEnum) -> String {
    if window.title.is_empty() {
        format!("[{}]", window.class)
    } else {
        window.title.clone()
    }
}
