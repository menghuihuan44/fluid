//! Pure helpers for the offline 视频处理 tab.
//!
//! Container decisions, output file naming, JPEG quality clamping and the
//! progress/ETA math live here so they can be unit tested without a window,
//! a file dialog or a running pipeline.

use std::path::Path;

/// Containers the offline tab can read and write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoFormat {
    /// Lossless YUV4MPEG2 (big files, good for tests).
    Y4m,
    /// Motion-JPEG AVI (small files, near-lossless at high quality).
    Avi,
}

impl VideoFormat {
    pub const ALL: [Self; 2] = [Self::Y4m, Self::Avi];

    pub fn extension(self) -> &'static str {
        match self {
            Self::Y4m => "y4m",
            Self::Avi => "avi",
        }
    }

    /// Beginner-friendly label used in the input metadata line.
    pub fn label(self) -> &'static str {
        match self {
            Self::Y4m => "y4m（无损，体积大）",
            Self::Avi => "avi（MJPEG，体积小）",
        }
    }

    /// Case-insensitive extension match; None for anything else.
    pub fn parse(path: &str) -> Option<Self> {
        match extension_of(path).as_str() {
            "y4m" => Some(Self::Y4m),
            "avi" => Some(Self::Avi),
            _ => None,
        }
    }
}

/// The two extensions every message should name.
pub const SUPPORTED: &str = ".y4m 和 .avi";

/// Lowercase extension of the last path segment, empty when there is none.
pub fn extension_of(path: &str) -> String {
    let trimmed = path.trim();
    let name = trimmed
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(trimmed)
        .to_owned();
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && !ext.is_empty() => ext.to_ascii_lowercase(),
        _ => String::new(),
    }
}

/// Decide the container from the file name, with a friendly error naming both
/// supported formats (the tab shows it inline, it never panics).
pub fn detect_format(path: &str) -> Result<VideoFormat, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(format!("还没有选择文件：支持 {SUPPORTED}。"));
    }
    match VideoFormat::parse(trimmed) {
        Some(format) => Ok(format),
        None => {
            let ext = extension_of(trimmed);
            if ext.is_empty() {
                Err(format!("这个文件没有扩展名：请用 {SUPPORTED} 结尾的路径。"))
            } else {
                Err(format!(
                    "不认识的扩展名 .{ext}：只支持 {SUPPORTED}（输出格式由扩展名决定）。"
                ))
            }
        }
    }
}

/// Suggested output path for an input file: same folder, same container,
/// with a ".fluid" marker before the extension.
pub fn suggest_output(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let path = Path::new(trimmed);
    let ext = extension_of(trimmed);
    let ext = if ext == "y4m" || ext == "avi" {
        ext
    } else {
        "y4m".to_owned()
    };
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("output");
    let name = format!("{stem}.fluid.{ext}");
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            parent.join(&name).to_string_lossy().into_owned()
        }
        _ => name,
    }
}

pub const QUALITY_MIN: u8 = 40;
pub const QUALITY_MAX: u8 = 100;
pub const QUALITY_DEFAULT: u8 = 88;

/// Keep the JPEG quality inside the range the AVI writer accepts.
pub fn clamp_quality(quality: u8) -> u8 {
    quality.clamp(QUALITY_MIN, QUALITY_MAX)
}

/// Slider (f32) to the u8 the AVI writer wants, clamped.
pub fn quality_from_slider(value: f32) -> u8 {
    clamp_quality(value.round().clamp(0.0, 255.0) as u8)
}

/// The JPEG quality only has an effect for the AVI writer.
pub fn quality_applies(format: VideoFormat) -> bool {
    format == VideoFormat::Avi
}

/// Seconds left, from the observed throughput. None while unknown or done.
pub fn eta_seconds(frames_out: u64, total: u64, elapsed_s: f32) -> Option<f32> {
    if frames_out == 0 || total == 0 || frames_out >= total {
        return None;
    }
    if !elapsed_s.is_finite() || elapsed_s <= 0.0 {
        return None;
    }
    let per_frame = elapsed_s as f64 / frames_out as f64;
    let remaining = (total - frames_out) as f64;
    let eta = per_frame * remaining;
    if eta.is_finite() && eta > 0.0 {
        Some(eta as f32)
    } else {
        None
    }
}

/// Human readable remaining time for the progress area.
pub fn format_eta(eta: Option<f32>) -> String {
    match eta {
        None => "剩余时间：计算中…".to_owned(),
        Some(seconds) if seconds < 1.0 => "剩余时间：不到 1 秒".to_owned(),
        Some(seconds) if seconds < 60.0 => format!("剩余时间：约 {:.0} 秒", seconds),
        Some(seconds) if seconds < 3600.0 => {
            let minutes = (seconds / 60.0).floor();
            let rest = seconds - minutes * 60.0;
            format!("剩余时间：约 {:.0} 分 {:.0} 秒", minutes, rest)
        }
        Some(seconds) => format!("剩余时间：约 {:.1} 小时", seconds / 3600.0),
    }
}

/// Note shown after a cancelled run: the partial file is kept on purpose.
pub fn cancelled_note() -> &'static str {
    "已取消：已经写入的输出文件会保留，但内容不完整。可以重新点“开始处理”覆盖它。"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_detection_is_case_insensitive_and_ignores_folders() {
        assert_eq!(detect_format("C:\\videos\\Clip.Y4M"), Ok(VideoFormat::Y4m));
        assert_eq!(detect_format("/tmp/clip.Avi"), Ok(VideoFormat::Avi));
        assert_eq!(
            detect_format("C:\\folder.y4m\\clip"),
            Err("这个文件没有扩展名：请用 .y4m 和 .avi 结尾的路径。".to_owned())
        );
    }

    #[test]
    fn unknown_extensions_name_both_supported_formats() {
        let err = detect_format("clip.mp4").expect_err("mp4 is not supported");
        assert!(err.contains(".y4m") && err.contains(".avi"), "{err}");
        assert!(err.contains("mp4"));
        assert!(
            detect_format("   ")
                .expect_err("empty path")
                .contains("还没有选择")
        );
    }

    #[test]
    fn output_names_keep_the_container_and_mark_the_file() {
        assert_eq!(
            suggest_output("C:\\videos\\clip.y4m"),
            "C:\\videos\\clip.fluid.y4m"
        );
        assert_eq!(
            Path::new(&suggest_output("/data/run.avi")),
            Path::new("/data/run.fluid.avi")
        );
        assert_eq!(suggest_output("clip"), "clip.fluid.y4m");
        assert_eq!(suggest_output("  "), "");
    }

    #[test]
    fn quality_is_clamped_to_the_writer_range() {
        assert_eq!(clamp_quality(0), QUALITY_MIN);
        assert_eq!(clamp_quality(39), QUALITY_MIN);
        assert_eq!(clamp_quality(88), 88);
        assert_eq!(clamp_quality(101), QUALITY_MAX);
        assert_eq!(quality_from_slider(39.4), QUALITY_MIN);
        assert_eq!(quality_from_slider(88.6), 89);
        assert_eq!(quality_from_slider(999.0), QUALITY_MAX);
        assert!(quality_applies(VideoFormat::Avi));
        assert!(!quality_applies(VideoFormat::Y4m));
    }

    #[test]
    fn eta_needs_progress_and_a_known_total() {
        assert_eq!(eta_seconds(0, 100, 5.0), None);
        assert_eq!(eta_seconds(10, 0, 5.0), None);
        assert_eq!(eta_seconds(100, 100, 5.0), None);
        assert_eq!(eta_seconds(10, 100, 0.0), None);
        let eta = eta_seconds(25, 100, 5.0).expect("eta");
        assert!((eta - 15.0).abs() < 1e-3, "{eta}");
    }

    #[test]
    fn eta_text_is_beginner_friendly() {
        assert_eq!(format_eta(None), "剩余时间：计算中…");
        assert_eq!(format_eta(Some(0.4)), "剩余时间：不到 1 秒");
        assert_eq!(format_eta(Some(25.0)), "剩余时间：约 25 秒");
        assert_eq!(format_eta(Some(80.0)), "剩余时间：约 1 分 20 秒");
        assert_eq!(format_eta(Some(7200.0)), "剩余时间：约 2.0 小时");
    }

    #[test]
    fn canceled_runs_tell_the_user_the_file_is_partial() {
        assert!(cancelled_note().contains("保留"));
        assert!(cancelled_note().contains("不完整"));
    }

    #[test]
    fn extension_helper_handles_windows_and_unix_paths() {
        assert_eq!(extension_of("C:\\a\\b.Y4M"), "y4m");
        assert_eq!(extension_of("a/b.avi"), "avi");
        assert_eq!(extension_of("noext"), "");
        assert_eq!(extension_of(".hidden"), "");
        assert_eq!(VideoFormat::Y4m.extension(), "y4m");
        assert_eq!(VideoFormat::Avi.extension(), "avi");
    }
}
