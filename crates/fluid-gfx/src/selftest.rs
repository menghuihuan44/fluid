//! WARP 无头自检：已知渐变纹理走完整渲染管线（帧环 → 采样着色器 → RTV）后
//! 读回中心像素校验，不依赖真实 GPU 与桌面，可在 CI 运行。

use crate::device::create_device_with_driver;
use crate::hud_overlay::HudRenderer;
use crate::renderer::Renderer;
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_WARP;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};

pub fn render_identity() -> Result<(), String> {
    let bundle = create_device_with_driver(None, D3D_DRIVER_TYPE_WARP)
        .map_err(|e| format!("WARP 设备创建失败: {e}"))?;
    let (w, h) = (640u32, 400u32);

    // Source gradient: R = x/w, G = y/h, B = 128; the center is ~128,128,128.
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            data[i] = 128;
            data[i + 1] = (y * 255 / (h - 1)) as u8;
            data[i + 2] = (x * 255 / (w - 1)) as u8;
            data[i + 3] = 255;
        }
    }
    let base = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: 0,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let src_desc = D3D11_TEXTURE2D_DESC {
        BindFlags: (D3D11_BIND_SHADER_RESOURCE.0) as u32,
        ..base
    };
    let srd = D3D11_SUBRESOURCE_DATA {
        pSysMem: data.as_ptr().cast(),
        SysMemPitch: w * 4,
        SysMemSlicePitch: 0,
    };
    let mut src = None;
    unsafe {
        bundle
            .device
            .CreateTexture2D(&src_desc, Some(&srd), Some(&mut src))
    }
    .map_err(|e| format!("源纹理创建失败: {e}"))?;
    let src = src.unwrap();

    let rt_desc = D3D11_TEXTURE2D_DESC {
        BindFlags: (D3D11_BIND_RENDER_TARGET.0) as u32,
        ..base
    };
    let mut rtt = None;
    unsafe {
        bundle
            .device
            .CreateTexture2D(&rt_desc, None, Some(&mut rtt))
    }
    .map_err(|e| format!("RT 纹理创建失败: {e}"))?;
    let rt_tex = rtt.unwrap();
    let mut rtv = None;
    unsafe {
        bundle
            .device
            .CreateRenderTargetView(&rt_tex, None, Some(&mut rtv))
    }
    .map_err(|e| format!("RTV 创建失败: {e}"))?;
    let rtv = rtv.unwrap();

    let mut renderer = Renderer::new(&bundle.device, &bundle.ctx, 2).map_err(|e| e.to_string())?;
    renderer
        .ensure_ring(w, h, DXGI_FORMAT_B8G8R8A8_UNORM)
        .map_err(|e| e.to_string())?;
    renderer.push_frame(&src, 1);
    let sample = renderer
        .latest_snapshot()
        .ok_or_else(|| "frame ring is empty".to_string())?;
    let opts = crate::renderer::RenderOptions {
        scale_mode: 1,
        sharpness: 0.0,
        fg_enabled: false,
        motion_compensation: false,
        preserve_aspect: true,
        fg_strength: 0.0,
        scene_protection: 0.6,
        fg_phase: 1.0,
    };
    renderer
        .render(crate::renderer::RenderRequest {
            target: crate::renderer::RenderTargetInfo {
                rtv: &rtv,
                width: w as f32,
                height: h as f32,
                uv_rect: [0.0, 0.0, 1.0, 1.0],
            },
            previous: &sample,
            current: &sample,
            options: opts,
            hud_text: Some("fluid HUD selftest\ncapture: WGC window\ncapture 60.0 fps | output 120.0 fps\nscale: adaptive | sharpness: 30"),
            hud: crate::renderer::HudOptions::default(),
        })
        .map_err(|e| e.to_string())?;

    // 读回中心像素
    let st_desc = D3D11_TEXTURE2D_DESC {
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: (D3D11_CPU_ACCESS_READ.0) as u32,
        ..base
    };
    let mut st = None;
    unsafe { bundle.device.CreateTexture2D(&st_desc, None, Some(&mut st)) }
        .map_err(|e| format!("Staging 纹理创建失败: {e}"))?;
    let st_tex = st.unwrap();
    unsafe {
        bundle.ctx.CopyResource(&st_tex, &rt_tex);
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        bundle
            .ctx
            .Map(&st_tex, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .map_err(|e| format!("Map 失败: {e}"))?;
        let row = mapped.RowPitch as usize;
        let bytes = std::slice::from_raw_parts(mapped.pData as *const u8, row * h as usize);
        let p = row * (h as usize / 2) + (w as usize / 2) * 4;
        let (b, g, r, a) = (bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]);
        let hud_pixel = row * 30 + 30 * 4;
        let hud_b = bytes[hud_pixel];
        bundle.ctx.Unmap(&st_tex, 0);
        if a != 255 {
            return Err(format!("中心像素 alpha={a}，应为 255"));
        }
        let ok = (r as i32 - 128).abs() <= 2
            && (g as i32 - 128).abs() <= 2
            && (b as i32 - 128).abs() <= 2;
        if !ok {
            return Err(format!(
                "中心像素异常 BGRA=({b},{g},{r},{a})，应为 (128,128,128,255)"
            ));
        }
        if hud_b >= 100 {
            return Err(format!(
                "HUD 未覆盖左上区域：采样蓝色分量={hud_b}，预期 HUD 背景使其明显变小"
            ));
        }
    }
    tracing::info!("渲染自检通过（WARP）");
    Ok(())
}

fn make_source_texture(
    device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
    data: &[u8],
    w: u32,
    h: u32,
) -> Result<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D, String> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_SHADER_RESOURCE.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let srd = D3D11_SUBRESOURCE_DATA {
        pSysMem: data.as_ptr().cast(),
        SysMemPitch: w * 4,
        SysMemSlicePitch: 0,
    };
    let mut tex = None;
    unsafe { device.CreateTexture2D(&desc, Some(&srd), Some(&mut tex)) }
        .map_err(|e| format!("motion test source texture: {e}"))?;
    Ok(tex.unwrap())
}

fn read_back(
    device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
    ctx: &windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
    src: &windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
    w: u32,
    h: u32,
) -> Result<Vec<u8>, String> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: (D3D11_CPU_ACCESS_READ.0) as u32,
        MiscFlags: 0,
    };
    let mut staging = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut staging)) }
        .map_err(|e| format!("motion test staging texture: {e}"))?;
    let staging = staging.unwrap();
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe {
        ctx.CopyResource(&staging, src);
        ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .map_err(|e| format!("motion test map: {e}"))?;
        let row = mapped.RowPitch as usize;
        let bytes = std::slice::from_raw_parts(mapped.pData as *const u8, row * h as usize);
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            out.extend_from_slice(&bytes[y * row..y * row + w as usize * 4]);
        }
        ctx.Unmap(&staging, 0);
        Ok(out)
    }
}

/// Exercise the compute motion estimator, temporal blend pass, adaptive
/// scaling and RCAS pass on WARP. This is the regression test for the
/// frame-generation path: with a checkerboard that moves 2 px per source
/// frame, a phase-0.5 blend must move the content by 1 px (half the motion)
/// rather than showing either endpoint.
pub fn render_motion_interpolation() -> Result<(), String> {
    let bundle = create_device_with_driver(None, D3D_DRIVER_TYPE_WARP)
        .map_err(|e| format!("WARP device creation failed: {e}"))?;
    let (w, h) = (64u32, 64u32);
    // 4-px checker: every 8x8 estimator patch crosses block boundaries, so
    // motion is measurable on the whole grid.
    let block = 4u32;
    let checker_at = |x: u32, y: u32| -> u8 {
        if ((x / block) + (y / block)) & 1 == 0 {
            230
        } else {
            35
        }
    };

    let mut a = vec![0u8; (w * h * 4) as usize];
    let mut b = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let ia = ((y * w + x) * 4) as usize;
            let bx = (x + 2) % w;
            let ib = ((y * w + bx) * 4) as usize;
            let color = checker_at(x, y);
            a[ia] = color / 2;
            a[ia + 1] = color;
            a[ia + 2] = color / 3;
            a[ia + 3] = 255;
            let color_b = checker_at(bx, y);
            b[ib] = color_b / 2;
            b[ib + 1] = color_b;
            b[ib + 2] = color_b / 3;
            b[ib + 3] = 255;
        }
    }
    let tex_a = make_source_texture(&bundle.device, &a, w, h)?;
    let tex_b = make_source_texture(&bundle.device, &b, w, h)?;

    let rt_desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_RENDER_TARGET.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut rt = None;
    unsafe { bundle.device.CreateTexture2D(&rt_desc, None, Some(&mut rt)) }
        .map_err(|e| format!("motion test RT: {e}"))?;
    let rt = rt.unwrap();
    let mut rtv = None;
    unsafe {
        bundle
            .device
            .CreateRenderTargetView(&rt, None, Some(&mut rtv))
    }
    .map_err(|e| format!("motion test RTV: {e}"))?;
    let rtv = rtv.unwrap();

    let mut renderer = Renderer::new(&bundle.device, &bundle.ctx, 3).map_err(|e| e.to_string())?;
    renderer
        .ensure_ring(w, h, DXGI_FORMAT_B8G8R8A8_UNORM)
        .map_err(|e| e.to_string())?;
    renderer.push_frame(&tex_a, 1_000);
    renderer.push_frame(&tex_b, 2_000);
    let (prev, curr, phase) = renderer
        .pair_at_time(1_500)
        .ok_or_else(|| "motion test could not select frame pair".to_string())?;
    if phase < 0.49 || phase > 0.51 {
        return Err(format!("motion test phase={phase}, expected 0.5"));
    }
    let opts = crate::renderer::RenderOptions {
        scale_mode: 2,
        sharpness: 0.0,
        fg_enabled: true,
        motion_compensation: true,
        preserve_aspect: true,
        fg_strength: 0.85,
        scene_protection: 0.6,
        fg_phase: phase,
    };
    renderer
        .render(crate::renderer::RenderRequest {
            target: crate::renderer::RenderTargetInfo {
                rtv: &rtv,
                width: w as f32,
                height: h as f32,
                uv_rect: [0.0, 0.0, 1.0, 1.0],
            },
            previous: &prev,
            current: &curr,
            options: opts,
            hud_text: None,
            hud: crate::renderer::HudOptions::default(),
        })
        .map_err(|e| e.to_string())?;

    let pixels = read_back(&bundle.device, &bundle.ctx, &rt, w, h)?;
    let mut any_color = false;
    for px in pixels.as_chunks::<4>().0 {
        if px[3] != 255 {
            return Err(format!("motion test produced alpha={}", px[3]));
        }
        if px[0] != 0 || px[1] != 0 || px[2] != 0 {
            any_color = true;
        }
    }
    if !any_color {
        return Err("motion test rendered a black frame".into());
    }

    // Input B is A translated right by two pixels, so the estimator must emit
    // a +2 source-pixel vector. At phase 0.5 the correct content at pixel x is
    // A(x-1). Judge the one-column-wide pixels where A(x-1) differs from both
    // the current frame, B(x)=A(x-2), and the unwarped endpoint average.
    let mut verified = 0u32;
    for y in [1u32, 5, 9, 13] {
        for x in (5u32..w).step_by(4) {
            let warped = checker_at(x - 1, y);
            let current = checker_at(x - 2, y);
            let no_motion = 0.5 * (checker_at(x, y) as f32 + current as f32);
            if (warped as f32 - current as f32).abs() < 80.0
                || (warped as f32 - no_motion).abs() < 30.0
            {
                continue;
            }
            let i = ((y * w + x) * 4) as usize;
            let g = pixels[i + 1] as i32;
            let d_warped = (g - warped as i32).abs();
            let d_current = (g - current as i32).abs();
            let d_no_motion = (g as f32 - no_motion).abs() as i32;
            if d_warped >= d_current || d_warped >= d_no_motion {
                return Err(format!(
                    "interpolated pixel ({x},{y}) G={g} does not follow motion \
                     (warped={warped}, current={current}, no_motion={no_motion:.1})"
                ));
            }
            verified += 1;
        }
    }
    if verified < 4 {
        return Err(format!(
            "motion test found only {verified} judgeable pixels; pattern too small"
        ));
    }
    tracing::info!("motion interpolation selftest passed ({verified} pixels verified)");
    Ok(())
}

/// A hard scene cut has no valid motion. The blend must fall back to the real
/// current frame instead of averaging two unrelated images into a ghosted or
/// purple mid-frame.
pub fn render_scene_cut_fallback() -> Result<(), String> {
    let bundle = create_device_with_driver(None, D3D_DRIVER_TYPE_WARP)
        .map_err(|e| format!("WARP device creation failed: {e}"))?;
    let (w, h) = (32u32, 32u32);
    let red = [10u8, 20, 230, 255].repeat((w * h) as usize);
    let blue = [230u8, 20, 10, 255].repeat((w * h) as usize);
    let tex_a = make_source_texture(&bundle.device, &red, w, h)?;
    let tex_b = make_source_texture(&bundle.device, &blue, w, h)?;

    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_RENDER_TARGET.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut rt = None;
    unsafe { bundle.device.CreateTexture2D(&desc, None, Some(&mut rt)) }
        .map_err(|e| format!("scene-cut RT: {e}"))?;
    let rt = rt.unwrap();
    let mut rtv = None;
    unsafe {
        bundle
            .device
            .CreateRenderTargetView(&rt, None, Some(&mut rtv))
    }
    .map_err(|e| format!("scene-cut RTV: {e}"))?;
    let rtv = rtv.unwrap();

    let mut renderer = Renderer::new(&bundle.device, &bundle.ctx, 3).map_err(|e| e.to_string())?;
    renderer
        .ensure_ring(w, h, DXGI_FORMAT_B8G8R8A8_UNORM)
        .map_err(|e| e.to_string())?;
    renderer.push_frame(&tex_a, 1_000);
    renderer.push_frame(&tex_b, 2_000);
    let (prev, curr, phase) = renderer
        .pair_at_time(1_500)
        .ok_or_else(|| "scene-cut test could not select frame pair".to_string())?;
    renderer
        .render(crate::renderer::RenderRequest {
            target: crate::renderer::RenderTargetInfo {
                rtv: &rtv,
                width: w as f32,
                height: h as f32,
                uv_rect: [0.0, 0.0, 1.0, 1.0],
            },
            previous: &prev,
            current: &curr,
            options: crate::renderer::RenderOptions {
                scale_mode: 2,
                sharpness: 0.0,
                fg_enabled: true,
                motion_compensation: true,
                preserve_aspect: true,
                fg_strength: 0.9,
                scene_protection: 0.8,
                fg_phase: phase,
            },
            hud_text: None,
            hud: crate::renderer::HudOptions::default(),
        })
        .map_err(|e| e.to_string())?;

    let pixels = read_back(&bundle.device, &bundle.ctx, &rt, w, h)?;
    let center = ((h / 2 * w + w / 2) * 4) as usize;
    let (b, g, r, a) = (
        pixels[center],
        pixels[center + 1],
        pixels[center + 2],
        pixels[center + 3],
    );
    if a != 255 {
        return Err(format!("scene-cut fallback alpha={a}"));
    }
    // The real current frame is blue. A ghosted blend would move R up and B
    // down toward the red previous frame.
    if r > 40 || b < 200 || g > 50 {
        return Err(format!(
            "scene cut was blended instead of falling back to current: BGRA=({b},{g},{r},{a})"
        ));
    }
    Ok(())
}

/// Exercise the two-pass separable adaptive upscaler and the letterbox path
/// on WARP. A half-red/half-blue source is magnified into a square canvas; the
/// resulting frame must keep both colours on the correct side and put black
/// bars only outside the 4:3 content rectangle.
pub fn render_adaptive_upscale() -> Result<(), String> {
    let bundle = create_device_with_driver(None, D3D_DRIVER_TYPE_WARP)
        .map_err(|e| format!("WARP device creation failed: {e}"))?;
    let (src_w, src_h) = (32u32, 24u32);
    let (out_w, out_h) = (128u32, 128u32);
    let mut src_data = vec![0u8; (src_w * src_h * 4) as usize];
    for y in 0..src_h {
        for x in 0..src_w {
            let i = ((y * src_w + x) * 4) as usize;
            let (b, g, r) = if x < src_w / 2 {
                (10, 20, 220)
            } else {
                (210, 30, 10)
            };
            src_data[i..i + 4].copy_from_slice(&[b, g, r, 255]);
        }
    }
    let src = make_source_texture(&bundle.device, &src_data, src_w, src_h)?;

    let desc = D3D11_TEXTURE2D_DESC {
        Width: out_w,
        Height: out_h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_RENDER_TARGET.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut rt = None;
    unsafe { bundle.device.CreateTexture2D(&desc, None, Some(&mut rt)) }
        .map_err(|e| format!("upscale RT: {e}"))?;
    let rt = rt.unwrap();
    let mut rtv = None;
    unsafe {
        bundle
            .device
            .CreateRenderTargetView(&rt, None, Some(&mut rtv))
    }
    .map_err(|e| format!("upscale RTV: {e}"))?;
    let rtv = rtv.unwrap();

    let mut renderer = Renderer::new(&bundle.device, &bundle.ctx, 3).map_err(|e| e.to_string())?;
    renderer
        .ensure_ring(src_w, src_h, DXGI_FORMAT_B8G8R8A8_UNORM)
        .map_err(|e| e.to_string())?;
    renderer.push_frame(&src, 1);
    let sample = renderer
        .latest_snapshot()
        .ok_or_else(|| "upscale test frame ring is empty".to_string())?;
    renderer
        .render(crate::renderer::RenderRequest {
            target: crate::renderer::RenderTargetInfo {
                rtv: &rtv,
                width: out_w as f32,
                height: out_h as f32,
                uv_rect: [0.0, 0.0, 1.0, 1.0],
            },
            previous: &sample,
            current: &sample,
            options: crate::renderer::RenderOptions {
                scale_mode: 2,
                sharpness: 0.3,
                fg_enabled: false,
                motion_compensation: false,
                preserve_aspect: true,
                fg_strength: 0.0,
                scene_protection: 0.6,
                fg_phase: 1.0,
            },
            hud_text: None,
            hud: crate::renderer::HudOptions::default(),
        })
        .map_err(|e| e.to_string())?;

    let pixels = read_back(&bundle.device, &bundle.ctx, &rt, out_w, out_h)?;
    let pixel = |x: u32, y: u32| -> (u8, u8, u8, u8) {
        let i = ((y * out_w + x) * 4) as usize;
        (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3])
    };
    // The source colour boundary is exactly at x=16 of 32. After the 4x
    // separable magnification it must land at output x=64; x=60/68 are one
    // source pixel away and catch the historical half-/full-texel tap shift.
    for (x, y, want_red) in [
        (32u32, 64u32, true),
        (60u32, 64u32, true),
        (68u32, 64u32, false),
        (96u32, 64u32, false),
    ] {
        let (b, g, r, a) = pixel(x, y);
        if a != 255 {
            return Err(format!("adaptive upscale alpha={a} at ({x},{y})"));
        }
        let ok = if want_red {
            r > 180 && b < 50 && g < 60
        } else {
            b > 170 && r < 50 && g < 70
        };
        if !ok {
            return Err(format!(
                "adaptive upscale wrong colour at ({x},{y}): BGRA=({b},{g},{r},{a})"
            ));
        }
    }
    let (b, g, r, a) = pixel(out_w / 2, 4);
    if a != 255 || b > 4 || g > 4 || r > 4 {
        return Err(format!(
            "adaptive upscale letterbox bar is not black at top: BGRA=({b},{g},{r},{a})"
        ));
    }
    Ok(())
}

/// Verify the independent HUD renderer can rasterize text into an RTV.
pub fn render_hud_window() -> Result<(), String> {
    let bundle = create_device_with_driver(None, D3D_DRIVER_TYPE_WARP)
        .map_err(|e| format!("HUD WARP device: {e}"))?;
    let mut renderer =
        HudRenderer::new(&bundle.device, &bundle.ctx).map_err(|e| format!("HUD renderer: {e}"))?;
    let text = "fluid HUD selftest\ncapture: WGC window\ncapture 60.0 fps | output 120.0 fps\nscale: adaptive | sharpness: 30\nframe gen: on | state: interpolating";
    let bg = [20u8, 80, 160];
    let (w, h) = renderer.update_text(text, 2, bg);
    if w < 128 || h < 64 {
        return Err(format!("HUD size too small: {w}x{h}"));
    }
    let rt_desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_RENDER_TARGET.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut rt = None;
    unsafe { bundle.device.CreateTexture2D(&rt_desc, None, Some(&mut rt)) }
        .map_err(|e| format!("HUD RT creation: {e}"))?;
    let rt = rt.unwrap();
    let mut rtv = None;
    unsafe {
        bundle
            .device
            .CreateRenderTargetView(&rt, None, Some(&mut rtv))
    }
    .map_err(|e| format!("HUD RTV creation: {e}"))?;
    renderer
        .draw(&rtv.unwrap(), w, h)
        .map_err(|e| format!("HUD draw: {e}"))?;
    let pixels = read_back(&bundle.device, &bundle.ctx, &rt, w, h)?;
    let bg_pixel = 4 * (w as usize) + 4;
    let (b, g, r, a) = (
        pixels[bg_pixel * 4],
        pixels[bg_pixel * 4 + 1],
        pixels[bg_pixel * 4 + 2],
        pixels[bg_pixel * 4 + 3],
    );
    if a != 255
        || (b as i32 - bg[2] as i32).abs() > 3
        || (g as i32 - bg[1] as i32).abs() > 3
        || (r as i32 - bg[0] as i32).abs() > 3
    {
        return Err(format!(
            "HUD background color mismatch BGRA=({b},{g},{r},{a}), expected ({},{},{},255)",
            bg[2], bg[1], bg[0]
        ));
    }
    let mut bright = 0usize;
    for px in pixels.as_chunks::<4>().0 {
        if px[3] != 255 {
            return Err(format!("HUD produced alpha={}", px[3]));
        }
        if px[0] > 150 && px[1] > 150 && px[2] > 150 {
            bright += 1;
        }
    }
    if bright < 20 {
        return Err(format!("HUD text was not visible (bright pixels={bright})"));
    }
    Ok(())
}

/// Smoke-test the DirectComposition + waitable-swapchain overlay path.
pub fn overlay_creation() -> Result<(), String> {
    let bundle = crate::create_device().map_err(|e| format!("D3D11 device: {e}"))?;
    let mut overlay = crate::OverlayWindow::new(&bundle.device, &bundle.factory2, 64, 64)
        .map_err(|e| format!("overlay creation: {e}"))?;
    overlay
        .place(&windows::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: 64,
            bottom: 64,
        })
        .map_err(|e| format!("overlay place: {e}"))?;
    overlay.wait_present_slot(50);
    overlay.present(0);
    overlay.show(false);
    Ok(())
}
