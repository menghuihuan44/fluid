//! Native file dialogs and "show in folder" for the 视频处理 tab.
//!
//! The desktop build uses rfd (native Windows / macOS / Linux dialogs). Android
//! has no system file picker wired up here, so the tab keeps editable path
//! fields and shows a manual-entry hint instead of a dead button.

use std::path::{Path, PathBuf};

use crate::video_format::VideoFormat;

/// Platforms without a native picker show the manual-path hint instead.
pub const DIALOGS_AVAILABLE: bool = cfg!(not(target_os = "android"));

pub const MANUAL_HINT: &str =
    "这个平台没有系统文件选择框，请直接在输入框里输入（或粘贴）文件路径。";

pub fn dialogs_available() -> bool {
    DIALOGS_AVAILABLE
}

/// File name part of a path, for the save dialog's default name.
pub fn file_name_of(path: &str) -> String {
    let trimmed = path.trim();
    trimmed
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(trimmed)
        .to_owned()
}

/// Give a picked output path the extension the chosen container needs.
pub fn with_default_extension(path: PathBuf, format: VideoFormat) -> PathBuf {
    let has_ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| !ext.is_empty());
    if has_ext {
        path
    } else {
        let name = format!(
            "{}.{}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("output"),
            format.extension()
        );
        path.with_file_name(name)
    }
}

#[cfg(not(target_os = "android"))]
pub use desktop::{pick_input_video, pick_output_video, reveal_output};

#[cfg(not(target_os = "android"))]
mod desktop {
    use super::*;

    /// Ask for the video to process. Filters cover both supported containers.
    pub fn pick_input_video() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("选择要处理的视频")
            .add_filter("视频（y4m / avi）", &["y4m", "avi"])
            .add_filter("所有文件", &["*"])
            .pick_file()
    }

    /// Ask where to write the result, defaulting to the suggested file name.
    pub fn pick_output_video(suggested: &str) -> Option<PathBuf> {
        let name = file_name_of(suggested);
        let mut dialog = rfd::FileDialog::new()
            .set_title("保存处理后的视频")
            .add_filter("y4m（无损，体积大）", &["y4m"])
            .add_filter("avi（MJPEG，体积小）", &["avi"]);
        if !name.is_empty() {
            dialog = dialog.set_file_name(&name);
        }
        dialog.save_file()
    }

    /// Reveal the finished file in the system file manager.
    pub fn reveal_output(path: &Path) -> Result<(), String> {
        #[cfg(windows)]
        {
            let arg = format!("/select,{}", path.display());
            std::process::Command::new("explorer")
                .arg(arg)
                .spawn()
                .map(|_| ())
                .map_err(|err| format!("无法打开资源管理器：{err}"))
        }
        #[cfg(not(windows))]
        {
            let folder = path.parent().unwrap_or(path);
            std::process::Command::new("xdg-open")
                .arg(folder)
                .spawn()
                .map(|_| ())
                .map_err(|err| format!("无法打开文件管理器：{err}"))
        }
    }
}

#[cfg(target_os = "android")]
pub use android_impl::{pick_input_video, pick_output_video, reveal_output};

#[cfg(target_os = "android")]
mod android_impl {
    use super::*;

    pub fn pick_input_video() -> Option<PathBuf> {
        None
    }

    pub fn pick_output_video(_suggested: &str) -> Option<PathBuf> {
        None
    }

    pub fn reveal_output(_path: &Path) -> Result<(), String> {
        Err("手机上请用系统的文件管理器打开输出文件夹。".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_work_for_both_separators() {
        assert_eq!(file_name_of("C:\\videos\\clip.y4m"), "clip.y4m");
        assert_eq!(file_name_of("/data/run.avi"), "run.avi");
        assert_eq!(file_name_of("   "), "");
    }

    #[test]
    fn a_picked_output_gets_the_container_extension() {
        assert_eq!(
            with_default_extension(PathBuf::from("C:\\out\\clip"), VideoFormat::Avi),
            PathBuf::from("C:\\out\\clip.avi")
        );
        assert_eq!(
            with_default_extension(PathBuf::from("/out/clip.y4m"), VideoFormat::Y4m),
            PathBuf::from("/out/clip.y4m")
        );
        assert_eq!(
            with_default_extension(PathBuf::from("clip"), VideoFormat::Y4m),
            PathBuf::from("clip.y4m")
        );
    }

    #[test]
    fn manual_hint_is_only_advertised_where_dialogs_are_missing() {
        assert_eq!(dialogs_available(), !cfg!(target_os = "android"));
        assert!(MANUAL_HINT.contains("手动") || MANUAL_HINT.contains("输入"));
    }
}
