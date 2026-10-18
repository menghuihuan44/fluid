//! CPU-rasterized real-time HUD texture.
//!
//! The texture is only regenerated at a low rate and only when text or style
//! changes. A capacity-based texture keeps reallocations rare while allowing the
//! panel to show many metrics.

use font8x8::UnicodeFonts;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};

const GLYPH_W: u32 = 8;
const GLYPH_H: u32 = 8;
const PADDING: u32 = 6;
const MAX_LINES: usize = 16;
const MAX_CHARS: usize = 120;
const TITLE_COLOR: [u8; 4] = [165, 225, 255, 255];
const TEXT_COLOR: [u8; 4] = [255, 255, 255, 255];
const BORDER_COLOR: [u8; 4] = [58, 78, 104, 180];

pub struct HudTexture {
    device: ID3D11Device,
    tex: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    width: u32,
    height: u32,
    content_width: u32,
    content_height: u32,
    buf: Vec<u8>,
    last_text: String,
    last_font_scale: u8,
    last_background_alpha: u8,
    last_background_color: [u8; 3],
}

impl HudTexture {
    pub fn new(device: &ID3D11Device, _ctx: &ID3D11DeviceContext) -> windows::core::Result<Self> {
        let (tex, srv) = create_texture(device, 128, 64)?;
        Ok(Self {
            device: device.clone(),
            tex,
            srv,
            width: 128,
            height: 64,
            content_width: 0,
            content_height: 0,
            buf: vec![0; 128 * 64 * 4],
            last_text: String::new(),
            last_font_scale: 0,
            last_background_alpha: 255,
            last_background_color: [0, 0, 0],
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn content_size(&self) -> (u32, u32) {
        (self.content_width, self.content_height)
    }

    pub fn srv(&self) -> &ID3D11ShaderResourceView {
        &self.srv
    }

    fn ensure_capacity(&mut self, width: u32, height: u32) -> windows::core::Result<()> {
        if width <= self.width && height <= self.height {
            return Ok(());
        }
        let width = width.div_ceil(64).max(1) * 64;
        let height = height.div_ceil(16).max(1) * 16;
        let (tex, srv) = create_texture(&self.device, width, height)?;
        self.tex = tex;
        self.srv = srv;
        self.width = width;
        self.height = height;
        self.buf = vec![0; (width * height * 4) as usize];
        Ok(())
    }

    /// Returns the used content size in pixels.
    pub fn update(
        &mut self,
        ctx: &ID3D11DeviceContext,
        text: &str,
        font_scale: u8,
        background_alpha: u8,
    ) -> (u32, u32) {
        self.update_colored(ctx, text, font_scale, [0, 0, 0], background_alpha)
    }

    pub fn update_colored(
        &mut self,
        ctx: &ID3D11DeviceContext,
        text: &str,
        font_scale: u8,
        background_color: [u8; 3],
        background_alpha: u8,
    ) -> (u32, u32) {
        let font_scale = font_scale.clamp(1, 3);
        let background_alpha = background_alpha.min(200);
        if text == self.last_text
            && font_scale == self.last_font_scale
            && background_alpha == self.last_background_alpha
            && background_color == self.last_background_color
        {
            return self.content_size();
        }

        let text = text.trim_end_matches('\n');
        let lines: Vec<&str> = text.lines().take(MAX_LINES).collect();
        let line_count = lines.len().max(1) as u32;
        let max_chars = lines
            .iter()
            .map(|line| line.chars().count().min(MAX_CHARS))
            .max()
            .unwrap_or(1)
            .max(1) as u32;
        let glyph_w = GLYPH_W * font_scale as u32;
        let glyph_h = GLYPH_H * font_scale as u32;
        let line_gap = 3 * font_scale as u32;
        // A stable minimum width prevents corner-anchored HUDs from shifting
        // left/right whenever a numeric metric gains or loses a character.
        let width_chars = max_chars.max(52);
        let content_width = PADDING * 2 + width_chars * glyph_w;
        let content_height =
            PADDING * 2 + line_count * glyph_h + line_count.saturating_sub(1) * line_gap;
        if self.ensure_capacity(content_width, content_height).is_err() {
            return self.content_size();
        }

        self.content_width = content_width.min(self.width);
        self.content_height = content_height.min(self.height);
        self.last_text.clear();
        self.last_text.push_str(text);
        self.last_font_scale = font_scale;
        self.last_background_alpha = background_alpha;
        self.last_background_color = background_color;
        self.buf.iter_mut().for_each(|byte| *byte = 0);

        if background_alpha > 0 {
            let alpha = background_alpha as u32;
            let premul_b = (background_color[2] as u32 * alpha / 255) as u8;
            let premul_g = (background_color[1] as u32 * alpha / 255) as u8;
            let premul_r = (background_color[0] as u32 * alpha / 255) as u8;
            for y in 0..self.content_height {
                for x in 0..self.content_width {
                    let index = ((y * self.width + x) * 4) as usize;
                    self.buf[index] = premul_b;
                    self.buf[index + 1] = premul_g;
                    self.buf[index + 2] = premul_r;
                    self.buf[index + 3] = background_alpha;
                }
            }
        }

        // Subtle 2px border makes the independent panel visually distinct from
        // the game while remaining cheap to rasterize.
        for y in 0..self.content_height {
            for x in 0..self.content_width {
                let border =
                    x < 2 || y < 2 || x + 2 >= self.content_width || y + 2 >= self.content_height;
                if border {
                    let index = ((y * self.width + x) * 4) as usize;
                    self.buf[index..index + 4].copy_from_slice(&BORDER_COLOR);
                }
            }
        }

        for (line_index, line) in lines.iter().enumerate() {
            let color = if line_index == 0 {
                TITLE_COLOR
            } else {
                TEXT_COLOR
            };
            let y0 = PADDING + line_index as u32 * (glyph_h + line_gap);
            for (char_index, character) in line.chars().take(MAX_CHARS).enumerate() {
                let character = if character.is_ascii() { character } else { '?' };
                let glyph = font8x8::BASIC_FONTS.get(character).unwrap_or([0u8; 8]);
                let x0 = PADDING + char_index as u32 * glyph_w;
                for (row, &bits) in glyph.iter().enumerate() {
                    for col in 0..GLYPH_W {
                        if bits & (1 << col) == 0 {
                            continue;
                        }
                        for sy in 0..font_scale as u32 {
                            for sx in 0..font_scale as u32 {
                                let px = x0 + col * font_scale as u32 + sx;
                                let py = y0 + row as u32 * font_scale as u32 + sy;
                                if px >= self.content_width || py >= self.content_height {
                                    continue;
                                }
                                let index = ((py * self.width + px) * 4) as usize;
                                // Premultiplied text colors.
                                self.buf[index..index + 4].copy_from_slice(&color);
                            }
                        }
                    }
                }
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
        self.content_size()
    }
}

fn create_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> windows::core::Result<(ID3D11Texture2D, ID3D11ShaderResourceView)> {
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
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut tex))? };
    let tex = tex.unwrap();
    let mut srv = None;
    unsafe { device.CreateShaderResourceView(&tex, None, Some(&mut srv))? };
    Ok((tex, srv.unwrap()))
}
