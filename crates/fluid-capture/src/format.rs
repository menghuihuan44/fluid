//! Desktop pixel-format conversion for screen capture.
//!
//! DXGI Desktop Duplication returns the desktop's native pixel format, which is
//! not necessarily 8-bit BGRA: HDR / 10-bit displays commonly expose
//! "DXGI_FORMAT_R16G16B16A16_FLOAT" (scRGB) or "DXGI_FORMAT_R10G10B10A2_UNORM"
//! instead of "DXGI_FORMAT_B8G8R8A8_UNORM". The capture pipeline works in
//! 8-bit RGBA, so this module maps every supported desktop format to that one
//! output format. The functions are pure (no Windows types, no allocation) so
//! they are unit-tested on the host without HDR hardware.
//!
//! Conversion policy:
//! - BGRA8 / RGBA8: pass through, swapping channels for BGRA, and force alpha
//!   to 255 because desktop/duplication alpha is undefined and the desktop is
//!   visually opaque.
//! - R10G10B10A2_UNORM: already display-referred (sRGB primaries, sRGB
//!   transfer); unpack the 10-bit channels and scale 0..=1023 to 0..=255 with
//!   round-to-nearest. The 2-bit alpha is ignored.
//! - R16G16B16A16_FLOAT: scRGB (linear) with 1.0 = SDR white. Values are
//!   clamped to 0..=1 and encoded with the sRGB OETF for display. There is no
//!   tone mapping, no paper-white scaling and no HDR metadata handling: HDR
//!   content above SDR white is clipped, which is documented behaviour rather
//!   than a silent guess.
//!
//! Rows are converted individually so a mapped desktop texture with a row
//! pitch larger than "width * bytes_per_pixel" can be walked without a copy.

use fluid_core::{FluidError, Result};

/// A desktop pixel format this build can convert to 8-bit RGBA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DesktopFormat {
    /// "DXGI_FORMAT_B8G8R8A8_UNORM": 4 bytes, blue first.
    Bgra8,
    /// "DXGI_FORMAT_R8G8B8A8_UNORM": 4 bytes, red first.
    Rgba8,
    /// "DXGI_FORMAT_R10G10B10A2_UNORM": 4 bytes, 10/10/10/2 bits little endian.
    Rgb10a2,
    /// "DXGI_FORMAT_R16G16B16A16_FLOAT": 4 half-float channels, scRGB linear.
    Rgba16F,
}

impl DesktopFormat {
    /// Bytes one pixel occupies in the source row.
    #[inline]
    pub(crate) fn bytes_per_pixel(self) -> usize {
        match self {
            DesktopFormat::Rgba16F => 8,
            DesktopFormat::Bgra8 | DesktopFormat::Rgba8 | DesktopFormat::Rgb10a2 => 4,
        }
    }

    /// DXGI format name, used in error messages and logs.
    pub(crate) fn name(self) -> &'static str {
        match self {
            DesktopFormat::Bgra8 => "DXGI_FORMAT_B8G8R8A8_UNORM",
            DesktopFormat::Rgba8 => "DXGI_FORMAT_R8G8B8A8_UNORM",
            DesktopFormat::Rgb10a2 => "DXGI_FORMAT_R10G10B10A2_UNORM",
            DesktopFormat::Rgba16F => "DXGI_FORMAT_R16G16B16A16_FLOAT",
        }
    }
}

/// Human readable label for a numeric DXGI format, e.g.
/// "DXGI_FORMAT(115) [R10G]" for an unknown fourcc style format.
pub(crate) fn format_label(value: u32) -> String {
    let bytes = value.to_le_bytes();
    let printable = bytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ');
    if printable && bytes.iter().any(|b| *b != 0) {
        format!("DXGI_FORMAT({value}) [{}]", String::from_utf8_lossy(&bytes))
    } else {
        format!("DXGI_FORMAT({value})")
    }
}

/// Convert one source row to one BGRA-to-RGBA style output row.
///
/// "src" must hold at least "width * fmt.bytes_per_pixel()" readable bytes (a
/// wider row slice is allowed: padding beyond that is ignored). "dst" must hold
/// at least "width * 4" bytes and is fully overwritten for those pixels.
#[inline]
pub(crate) fn convert_row(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    fmt: DesktopFormat,
) -> Result<()> {
    let src_needed = width.saturating_mul(fmt.bytes_per_pixel());
    let dst_needed = width.saturating_mul(4);
    if src.len() < src_needed {
        return Err(FluidError::invalid(format!(
            "dxgi: {} source row is {} bytes, need {src_needed} for {width} pixels",
            fmt.name(),
            src.len()
        )));
    }
    if dst.len() < dst_needed {
        return Err(FluidError::invalid(format!(
            "dxgi: RGBA row buffer is {} bytes, need {dst_needed} for {width} pixels",
            dst.len()
        )));
    }
    let src = &src[..src_needed];
    let dst = &mut dst[..dst_needed];
    match fmt {
        DesktopFormat::Bgra8 => bgra_row_to_rgba(src, dst),
        DesktopFormat::Rgba8 => {
            dst.copy_from_slice(src);
            for pixel in dst.as_chunks_mut::<4>().0 {
                pixel[3] = 255;
            }
        }
        DesktopFormat::Rgb10a2 => rgb10a2_row_to_rgba(src, dst),
        DesktopFormat::Rgba16F => rgba16f_row_to_rgba(src, dst),
    }
    Ok(())
}

/// BGRA -> RGBA row conversion. Alpha is forced opaque: DXGI leaves unused
/// alpha bits at zero, but the desktop should present as opaque.
#[inline]
pub(crate) fn bgra_row_to_rgba(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0.iter_mut())
    {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = 255;
    }
}

/// R10G10B10A2_UNORM -> RGBA row conversion.
#[inline]
fn rgb10a2_row_to_rgba(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0.iter_mut())
    {
        let word = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        d[0] = scale_10_bit(word & 0x03ff);
        d[1] = scale_10_bit((word >> 10) & 0x03ff);
        d[2] = scale_10_bit((word >> 20) & 0x03ff);
        d[3] = 255;
    }
}

/// Round a 10-bit UNORM value to 8 bits: (v * 255 + 511) / 1023.
#[inline]
fn scale_10_bit(value: u32) -> u8 {
    ((value * 255 + 511) / 1023) as u8
}

/// R16G16B16A16_FLOAT (scRGB linear) -> RGBA row conversion.
#[inline]
fn rgba16f_row_to_rgba(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src
        .as_chunks::<8>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0.iter_mut())
    {
        d[0] = sc_rgb_to_u8(half_to_f32(u16::from_le_bytes([s[0], s[1]])));
        d[1] = sc_rgb_to_u8(half_to_f32(u16::from_le_bytes([s[2], s[3]])));
        d[2] = sc_rgb_to_u8(half_to_f32(u16::from_le_bytes([s[4], s[5]])));
        d[3] = 255;
    }
}

/// Convert one linear scRGB channel to an 8-bit sRGB-encoded value.
///
/// Values are clamped to 0..=1 (SDR white), infinities saturate and NaN maps to
/// black; everything above SDR white is clipped, which is the documented
/// no-tone-mapping behaviour of the 8-bit capture path.
#[inline]
fn sc_rgb_to_u8(linear: f32) -> u8 {
    if !linear.is_finite() {
        return if linear.is_sign_positive() { 255 } else { 0 };
    }
    let clamped = linear.clamp(0.0, 1.0);
    let encoded = if clamped <= 0.003_130_8 {
        12.92 * clamped
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    };
    // The OETF is monotonic within 0..=1, so rounding cannot leave the range.
    (encoded * 255.0).round() as u8
}

/// IEEE 754 binary16 -> binary32 conversion without external dependencies.
#[inline]
fn half_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x03ff) as f32;
    let magnitude = match exponent {
        0 => mantissa * 5.960_464_5e-8, // 2^-24: subnormal halves
        31 => {
            if mantissa == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mantissa / 1024.0) * f32::exp2((exponent - 15) as f32),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(r: u32, g: u32, b: u32, a: u32) -> [u8; 4] {
        let packed = (r & 0x03ff) | ((g & 0x03ff) << 10) | ((b & 0x03ff) << 20) | ((a & 0x3) << 30);
        packed.to_le_bytes()
    }

    fn half_bytes(values: [u16; 4]) -> [u8; 8] {
        let mut out = [0u8; 8];
        for (i, v) in values.iter().enumerate() {
            out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn rgb10a2_known_values() {
        let mut dst = [0u8; 4];
        convert_row(&word(0, 0, 0, 1023), &mut dst, 1, DesktopFormat::Rgb10a2).unwrap();
        assert_eq!(dst, [0, 0, 0, 255]);
        convert_row(
            &word(1023, 512, 0, 1023),
            &mut dst,
            1,
            DesktopFormat::Rgb10a2,
        )
        .unwrap();
        assert_eq!(dst, [255, 128, 0, 255]);
        convert_row(&word(1, 1, 1, 0), &mut dst, 1, DesktopFormat::Rgb10a2).unwrap();
        assert_eq!(dst, [0, 0, 0, 255], "one 10-bit step must round to zero");
    }

    #[test]
    fn rgb10a2_covers_every_code() {
        let mut dst = [0u8; 4];
        let mut previous = 0u8;
        for code in 0..=1023u32 {
            convert_row(
                &word(code, code, code, 1023),
                &mut dst,
                1,
                DesktopFormat::Rgb10a2,
            )
            .unwrap();
            assert_eq!(dst[0], ((code * 255 + 511) / 1023) as u8);
            assert!(dst[0] >= previous, "mapping must be monotonic");
            assert_eq!(dst[3], 255, "alpha is always opaque");
            previous = dst[0];
        }
        assert_eq!(previous, 255);
    }

    #[test]
    fn rgba16f_known_values() {
        let mut dst = [0u8; 4];
        // 0.0, 0.5, 1.0 and 2.0 in binary16.
        let row = half_bytes([0x0000, 0x3800, 0x3c00, 0x4000]);
        convert_row(&row, &mut dst, 1, DesktopFormat::Rgba16F).unwrap();
        assert_eq!(
            dst,
            [0, 188, 255, 255],
            "0.5 linear is 188 in sRGB; 2.0 clips"
        );
    }

    #[test]
    fn rgba16f_clamps_and_handles_specials() {
        let mut dst = [0u8; 4];
        convert_row(
            &half_bytes([0xbc00, 0x0001, 0x7c00, 0x7e00]),
            &mut dst,
            1,
            DesktopFormat::Rgba16F,
        )
        .unwrap();
        assert_eq!(
            dst,
            [0, 0, 255, 255],
            "-1 clamps to 0, subnormal to 0, +inf saturates, NaN is black"
        );
        convert_row(
            &half_bytes([0xc000, 0x3400, 0x3c00, 0x3c00]),
            &mut dst,
            1,
            DesktopFormat::Rgba16F,
        )
        .unwrap();
        assert_eq!(dst[0], 0, "-2.0 clamps to 0");
        assert_eq!(dst[1], 137, "0.25 linear is 137 in sRGB");
    }

    #[test]
    fn half_to_f32_matches_reference_values() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn bgra8_swizzles_and_forces_alpha() {
        let mut dst = [0u8; 12];
        convert_row(
            &[10, 20, 30, 0, 1, 2, 3, 128, 4, 5, 6, 255],
            &mut dst,
            3,
            DesktopFormat::Bgra8,
        )
        .unwrap();
        assert_eq!(dst, [30, 20, 10, 255, 3, 2, 1, 255, 6, 5, 4, 255]);
    }

    #[test]
    fn bgra8_regression_matches_previous_path() {
        let src: Vec<u8> = (0..40u8).collect();
        let mut expected = vec![0u8; 40];
        for (s, d) in src
            .as_chunks::<4>()
            .0
            .iter()
            .zip(expected.as_chunks_mut::<4>().0.iter_mut())
        {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = 255;
        }
        let mut actual = vec![0u8; 40];
        convert_row(&src, &mut actual, 10, DesktopFormat::Bgra8).unwrap();
        assert_eq!(
            actual, expected,
            "BGRA8 output must be byte-identical to before"
        );
    }

    #[test]
    fn row_padding_is_ignored() {
        // A desktop row with pitch 32 holds 3 meaningful pixels; the 20 trailing
        // padding bytes must never leak into the destination.
        let mut row = vec![0xeeu8; 32];
        row[..12].copy_from_slice(&[1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]);
        let mut dst = [0u8; 12];
        convert_row(&row, &mut dst, 3, DesktopFormat::Bgra8).unwrap();
        assert_eq!(dst, [3, 2, 1, 255, 6, 5, 4, 255, 9, 8, 7, 255]);
    }

    #[test]
    fn odd_widths_and_multiple_rows() {
        // Two rows of three pixels: red/green/green, then blue/blue/red.
        let row0 = [
            word(1023, 0, 0, 1023),
            word(0, 1023, 0, 1023),
            word(0, 1023, 0, 1023),
        ]
        .concat();
        let row1 = [
            word(0, 0, 1023, 1023),
            word(0, 0, 1023, 1023),
            word(1023, 0, 0, 1023),
        ]
        .concat();
        let mut frame = [0u8; 24];
        convert_row(&row0, &mut frame[..12], 3, DesktopFormat::Rgb10a2).unwrap();
        convert_row(&row1, &mut frame[12..], 3, DesktopFormat::Rgb10a2).unwrap();
        assert_eq!(
            &frame[..12],
            &[255, 0, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255]
        );
        assert_eq!(
            &frame[12..],
            &[0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255]
        );
    }

    #[test]
    fn short_rows_are_rejected() {
        let mut dst = [0u8; 8];
        let err = convert_row(&[0u8; 7], &mut dst, 2, DesktopFormat::Bgra8).unwrap_err();
        assert!(err.to_string().contains("source row is 7 bytes"), "{err}");
        let err = convert_row(&[0u8; 8], &mut dst[..6], 2, DesktopFormat::Bgra8).unwrap_err();
        assert!(
            err.to_string().contains("RGBA row buffer is 6 bytes"),
            "{err}"
        );
        let err = convert_row(&[0u8; 8], &mut dst, 2, DesktopFormat::Rgba16F).unwrap_err();
        assert!(err.to_string().contains("R16G16B16A16_FLOAT"), "{err}");
    }

    #[test]
    fn rgba8_input_stays_rgba_and_opaque() {
        let mut dst = [0u8; 8];
        convert_row(&[1, 2, 3, 4, 5, 6, 7, 8], &mut dst, 2, DesktopFormat::Rgba8).unwrap();
        assert_eq!(dst, [1, 2, 3, 255, 5, 6, 7, 255]);
    }

    #[test]
    fn format_labels_name_known_and_fourcc_values() {
        assert_eq!(
            DesktopFormat::Rgb10a2.name(),
            "DXGI_FORMAT_R10G10B10A2_UNORM"
        );
        assert_eq!(DesktopFormat::Rgba16F.bytes_per_pixel(), 8);
        assert_eq!(DesktopFormat::Bgra8.bytes_per_pixel(), 4);
        let fourcc = 0x3031_3052u32; // "R10" + trailing zero byte in little endian
        let label = format_label(fourcc);
        assert!(label.contains("DXGI_FORMAT("), "{label}");
        assert!(format_label(87).contains("87"));
        assert!(
            !format_label(0).contains('['),
            "unknown zero format has no fourcc"
        );
    }
}
