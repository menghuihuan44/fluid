//! HLSL 源码内嵌 + 启动时经系统 d3dcompiler_47 编译（Windows 8.1+ 自带）。
//! 编译一次约几毫秒，免去构建期工具链依赖。

use crate::error::GfxError;
use std::sync::OnceLock;
use windows::core::PCSTR;
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};

pub const VS_FULLSCREEN: &str = include_str!("shaders/vs_fullscreen.hlsl");
pub const PS_COMPOSE: &str = include_str!("shaders/ps_compose.hlsl");
pub const PS_RCAS: &str = include_str!("shaders/ps_rcas.hlsl");
pub const PS_HUD: &str = include_str!("shaders/ps_hud.hlsl");
pub const CS_FLOW: &str = include_str!("shaders/cs_flow.hlsl");

#[derive(Clone)]
pub struct ShaderBinaries {
    pub vs: Vec<u8>,
    pub ps_compose: Vec<u8>,
    pub ps_rcas: Vec<u8>,
    pub ps_hud: Vec<u8>,
    pub cs_flow: Vec<u8>,
}

#[derive(Clone)]
pub struct HudShaderBinaries {
    pub vs: Vec<u8>,
    pub ps_hud: Vec<u8>,
}

/// Compile only the two shaders needed by the independent HUD window.
pub fn compile_hud_only() -> Result<HudShaderBinaries, GfxError> {
    static CACHE: OnceLock<Result<HudShaderBinaries, String>> = OnceLock::new();
    let cached = CACHE.get_or_init(|| {
        (|| -> Result<HudShaderBinaries, GfxError> {
            Ok(HudShaderBinaries {
                vs: compile(VS_FULLSCREEN, "vs_fullscreen", "main", "vs_5_0")?,
                ps_hud: compile(PS_HUD, "ps_hud", "main", "ps_5_0")?,
            })
        })()
        .map_err(|e| e.to_string())
    });
    match cached {
        Ok(binaries) => Ok(binaries.clone()),
        Err(message) => Err(GfxError::Other(message.clone())),
    }
}

pub fn compile_all() -> Result<ShaderBinaries, GfxError> {
    static CACHE: OnceLock<Result<ShaderBinaries, String>> = OnceLock::new();
    let cached = CACHE.get_or_init(|| compile_all_uncached().map_err(|e| e.to_string()));
    match cached {
        Ok(binaries) => Ok(binaries.clone()),
        Err(message) => Err(GfxError::Other(message.clone())),
    }
}

fn compile_all_uncached() -> Result<ShaderBinaries, GfxError> {
    Ok(ShaderBinaries {
        vs: compile(VS_FULLSCREEN, "vs_fullscreen", "main", "vs_5_0")?,
        ps_compose: compile(PS_COMPOSE, "ps_compose", "main", "ps_5_0")?,
        ps_rcas: compile(PS_RCAS, "ps_rcas", "main", "ps_5_0")?,
        ps_hud: compile(PS_HUD, "ps_hud", "main", "ps_5_0")?,
        cs_flow: compile(CS_FLOW, "cs_flow", "main", "cs_5_0")?,
    })
}

fn pc(v: &[u8]) -> PCSTR {
    debug_assert!(v.last() == Some(&0));
    PCSTR(v.as_ptr())
}

/// 追加结尾 NUL 的运行时字节串。
fn nul(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(0);
    v
}

fn compile(src: &str, name: &str, entry: &str, target: &str) -> Result<Vec<u8>, GfxError> {
    let entry = nul(entry);
    let target = nul(target);
    let mut blob: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    unsafe {
        D3DCompile(
            src.as_ptr().cast(),
            src.len(),
            pc(b"shader.hlsl\0"),
            None,
            None,
            pc(&entry),
            pc(&target),
            0,
            0,
            &mut blob,
            Some(&mut errors),
        )
        .map_err(|e| {
            let msg = errors
                .as_ref()
                .map(|b| {
                    format!(
                        ": {}",
                        String::from_utf8_lossy(std::slice::from_raw_parts(
                            b.GetBufferPointer().cast::<u8>(),
                            b.GetBufferSize(),
                        ))
                    )
                })
                .unwrap_or_default();
            GfxError::Other(format!("编译 {name} 失败 {e}{msg}"))
        })?;
    }
    let blob = blob.ok_or_else(|| GfxError::Other("D3DCompile 未返回字节码".into()))?;
    Ok(unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    })
}
