//! Generate a deterministic synthetic y4m clip, used by the demo scripts and
//! for eyeballing the frame generator without a capture device.
//!
//! Usage: cargo run --release -p fluid-video --example gen_y4m -- out/clip.y4m [w] [h] [frames] [fps]

use std::process::ExitCode;

use fluid_core::{Dims, FrameSink};
use fluid_video::{Y4mHeader, Y4mWriter, synthetic_pattern};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gen_y4m <out.y4m> [width] [height] [frames] [fps]");
        return ExitCode::FAILURE;
    };
    let parse = |s: Option<String>, default: u32| -> u32 {
        s.and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let width = parse(args.next(), 1280);
    let height = parse(args.next(), 720);
    let frames = parse(args.next(), 90);
    let fps = parse(args.next(), 60);
    let dims = Dims::new(width, height);
    let header = Y4mHeader::new(dims, fps as f32);
    let mut writer = match Y4mWriter::create(&path, header) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("cannot create {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    for i in 0..frames as u64 {
        let t = i as f32 / fps as f32;
        if let Err(e) = writer.present(&synthetic_pattern(dims, t, i)) {
            eprintln!("write failed: {e}");
            return ExitCode::FAILURE;
        }
    }
    if let Err(e) = writer.finish() {
        eprintln!("flush failed: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "wrote {path} ({}x{}, {frames} frames @ {fps} fps)",
        width, height
    );
    ExitCode::SUCCESS
}
