//! Convert a Motion-JPEG AVI file back into a y4m clip.
//!
//! This closes the ffmpeg-free loop: AviWriter produces a self-contained AVI,
//! this binary reads it back and writes a normal y4m stream that every other
//! part of the toolkit (and the y4m test suite) understands. Odd dimensions
//! come out as 4:4:4 because the y4m writer rejects 4:2:0/4:2:2 when a plane
//! would not divide evenly.
//!
//! Usage: avi_to_y4m <input.avi> <output.y4m>

use std::process::ExitCode;

use fluid_core::{FrameSink, FrameSource};
use fluid_video::{AviReader, Y4mFormat, Y4mHeader, Y4mWriter};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("avi_to_y4m: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or_else(usage)?;
    let output = args.next().ok_or_else(usage)?;
    if args.next().is_some() {
        return Err(usage());
    }

    let mut reader = AviReader::open(&input).map_err(|e| format!("cannot open {input}: {e}"))?;
    let dims = reader.dims();
    let fps = reader.fps();
    let mut header = Y4mHeader::new(dims, if fps > 0.0 { fps } else { 30.0 });
    if !dims.width.is_multiple_of(2) || !dims.height.is_multiple_of(2) {
        header.format = Y4mFormat::C444;
    }
    let mut writer =
        Y4mWriter::create(&output, header).map_err(|e| format!("cannot create {output}: {e}"))?;

    let mut written = 0u64;
    while let Some(frame) = reader
        .next_frame()
        .map_err(|e| format!("read failed: {e}"))?
    {
        writer
            .present(&frame)
            .map_err(|e| format!("write failed: {e}"))?;
        written += 1;
    }
    writer
        .finish()
        .map_err(|e| format!("cannot finish {output}: {e}"))?;
    eprintln!(
        "avi_to_y4m: wrote {written} frame(s) of {} at {:.3} fps to {output}",
        dims,
        if fps > 0.0 { fps } else { 30.0 }
    );
    Ok(())
}

fn usage() -> String {
    "usage: avi_to_y4m <input.avi> <output.y4m>".to_string()
}
