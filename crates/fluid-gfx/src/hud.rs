//! 性能 HUD：CPU 光栅化 8x8 点阵文字到 BGRA 纹理（预乘 alpha），每 250ms 上传一次。

use font8x8::UnicodeFonts;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};

const SCALE: u32 = 2;
const GLYPH_W: u32 = 8;
const GLYPH_H: u32 = 8;
const LINE_GAP: u32 = 4;
const MAX_LINES: usize = 2;
const MAX_CHARS: usize = 72;

pub struct HudTexture {
    tex: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    width: u32,
    height: u32,
    buf: Vec<u8>,
    last: String,
}

impl HudTexture {
    pub fn new(device: &ID3D11Device, _ctx: &ID3D11DeviceContext) -> windows::core::Result<Self> {
        let width = 4 + (MAX_CHARS as u32) * GLYPH_W * SCALE + 4;
        let height = 4 + (MAX_LINES as u32) * (GLYPH_H * SCALE) + (MAX_LINES as u32 - 1) * LINE_GAP;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
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
        let mut tex = None;
        unsafe {
            device.CreateTexture2D(&desc, None, Some(&mut tex))?;
        }
        let tex = tex.unwrap();
        let mut srv = None;
        unsafe {
            device.CreateShaderResourceView(&tex, None, Some(&mut srv))?;
        }
        let srv = srv.unwrap();
        Ok(Self {
            tex,
            srv,
            width,
            height,
            buf: vec![0; (width * height * 4) as usize],
            last: String::new(),
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn srv(&self) -> &ID3D11ShaderResourceView {
        &self.srv
    }

    /// 文本变化时重新光栅化并上传。ASCII 之外的字符以 '?' 兜底（点阵字体限制）。
    pub fn update(&mut self, ctx: &ID3D11DeviceContext, text: &str) {
        if text == self.last {
            return;
        }
        self.last = text.to_string();
        self.buf.iter_mut().for_each(|b| *b = 0);
        let w = self.width;
        let h = self.height;
        for (li, line) in text.lines().take(MAX_LINES).enumerate() {
            let y0 = 4 + li as u32 * (GLYPH_H * SCALE + LINE_GAP);
            let mut x = 4u32;
            for ch in line.chars().take(MAX_CHARS) {
                let ch = if ch.is_ascii() { ch } else { '?' };
                let glyph = font8x8::BASIC_FONTS.get(ch).unwrap_or([0u8; 8]);
                for (row, &bits) in glyph.iter().enumerate() {
                    for col in 0..GLYPH_W as usize {
                        if bits & (1 << col) != 0 {
                            for sy in 0..SCALE {
                                for sx in 0..SCALE {
                                    let px = x + col as u32 * SCALE + sx;
                                    let py = y0 + row as u32 * SCALE + sy;
                                    if px < w && py < h {
                                        let i = ((py * w + px) * 4) as usize;
                                        // 预乘 alpha 白色（BGRA 字节序）
                                        self.buf[i] = 255;
                                        self.buf[i + 1] = 255;
                                        self.buf[i + 2] = 255;
                                        self.buf[i + 3] = 255;
                                    }
                                }
                            }
                        }
                    }
                }
                x += GLYPH_W * SCALE;
            }
        }
        unsafe {
            ctx.UpdateSubresource(
                &self.tex,
                0,
                None,
                self.buf.as_ptr().cast(),
                self.width * 4,
                0,
            );
        }
    }
}
