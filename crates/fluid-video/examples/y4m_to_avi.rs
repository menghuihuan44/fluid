//! Convert a y4m clip into a self-contained Motion-JPEG AVI file.
//!
//! This is the offline-export path for machines that have no ffmpeg: the AVI
//! writer in fluid-video encodes every frame with the image crate JPEG encoder
//! and wraps it in a standard AVI container, so the result opens in Windows
//! Media Player, VLC and any Media Foundation player. It is deliberately a
//! separate small binary instead of core CLI behaviour, because a real export
//! also needs an audio track and a proper codec, which is what the ffmpeg
//! scripts in scripts/ are for.
//!
//! Usage: y4m_to_avi <input.y4m> <output.avi> [quality 1..=100] [max_frames]
//!
//! Example: cargo run -p fluid-video --example y4m_to_avi -- out/clip.y4m out/clip.avi 88 120
//!
//! Arguments are positional: the first two are required, quality defaults to
//! AviHeader::DEFAULT quality (88), and max_frames (also optional) stops after
//! that many source frames, which is handy for quick smoke tests.

use std::process::ExitCode;

use fluid_core::{FrameSink, FrameSource};
use fluid_video::{AviHeader, AviWriter, DEFAULT_JPEG_QUALITY, Y4mReader};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("y4m_to_avi: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or_else(usage)?;
    let output = args.next().ok_or_else(usage)?;
    let quality = match args.next() {
        Some(text) => text
            .parse::<u8>()
            .map_err(|_| format!("invalid quality {text:?} (expected 1..=100)"))?,
        None => DEFAULT_JPEG_QUALITY,
    };
    let max_frames = match args.next() {
        Some(text) => Some(
            text.parse::<u64>()
                .map_err(|_| format!("invalid frame limit {text:?}"))?,
        ),
        None => None,
    };
    if args.next().is_some() {
        return Err(usage());
    }

    let mut reader = Y4mReader::open(&input).map_err(|e| format!("cannot open {input}: {e}"))?;
    let y4m = reader.header().clone();
    let mut writer = AviWriter::create(&output, AviHeader::from_y4m_header(&y4m, quality))
        .map_err(|e| format!("cannot create {output}: {e}"))?;

    let mut written = 0u64;
    loop {
        if let Some(limit) = max_frames
            && written >= limit
        {
            break;
        }
        match reader
            .next_frame()
            .map_err(|e| format!("read failed: {e}"))?
        {
            Some(frame) => {
                writer
                    .present(&frame)
                    .map_err(|e| format!("write failed: {e}"))?;
                written += 1;
            }
            None => break,
        }
    }
    writer
        .finish()
        .map_err(|e| format!("cannot finish {output}: {e}"))?;
    eprintln!(
        "y4m_to_avi: wrote {written} frame(s) of {} at {:.3} fps to {output} (quality {quality})",
        y4m.dims,
        y4m.fps()
    );
    Ok(())
}

fn usage() -> String {
    "usage: y4m_to_avi <input.y4m> <output.avi> [quality 1..=100] [max_frames]".to_string()
}
