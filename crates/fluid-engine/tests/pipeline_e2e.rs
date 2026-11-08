//! End-to-end tests: real video I/O on both ends of the pipeline.
//!
//! These exercise the same code path the CLI uses, but entirely headless: files
//! in the temp directory, no display, no GPU.

use std::path::PathBuf;

use fluid_core::{Dims, FluidConfig, FrameSink, FrameSource, InterpMode, ScaleMode, UpscaleFilter};
use fluid_engine::{FrameProcessor, Pipeline, PipelineConfig};
use fluid_video::{ImageSequence, PngWriter, Y4mHeader, Y4mReader, Y4mWriter, synthetic_pattern};

fn scratch_dir(tag: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut dir = std::env::temp_dir();
    dir.push(format!("fluid-e2e-{tag}-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn count_y4m_frames(path: &std::path::Path) -> (Y4mHeader, u64) {
    let mut reader = Y4mReader::open(path).unwrap();
    let header = reader.header().clone();
    let mut n = 0u64;
    while reader.next_frame().unwrap().is_some() {
        n += 1;
    }
    (header, n)
}

#[test]
fn y4m_in_y4m_out_with_interpolation_and_upscale() {
    let dir = scratch_dir("y4m");
    let src_path = dir.join("in.y4m");
    let out_path = dir.join("out.y4m");

    let src_dims = Dims::new(64, 48);
    let frames = 12u64;
    let fps = 30.0f32;

    {
        let mut writer = Y4mWriter::create(&src_path, Y4mHeader::new(src_dims, fps)).unwrap();
        for i in 0..frames {
            let t = i as f32 / fps;
            writer.present(&synthetic_pattern(src_dims, t, i)).unwrap();
        }
        writer.finish().unwrap();
    }

    let mut cfg = FluidConfig::default();
    cfg.interpolator.mode = InterpMode::Fixed;
    cfg.interpolator.multiplier = 2;
    cfg.upscaler.filter = UpscaleFilter::Bilinear;
    cfg.upscaler.scale.mode = ScaleMode::Factor;
    cfg.upscaler.scale.factor = 2.0;
    cfg.upscaler.temporal.enabled = false;

    let out_dims = Dims::new(128, 96);
    let mut reader = Y4mReader::open(&src_path).unwrap();
    let mut writer = Y4mWriter::create(&out_path, Y4mHeader::new(out_dims, fps * 2.0)).unwrap();

    let mut pipeline =
        Pipeline::with_config(&cfg, reader.header().dims, PipelineConfig::offline()).unwrap();
    assert_eq!(pipeline.out_dims(), out_dims);
    assert_eq!(pipeline.multiplier(), 2);

    let report = pipeline.run(&mut reader, &mut writer).unwrap();
    writer.finish().unwrap();
    drop(writer);

    assert_eq!(report.stats.frames_in, frames);
    // The first source frame has no predecessor, so it is duplicated instead of
    // interpolated: the clip keeps an exact 2x cadence for its whole length.
    assert_eq!(report.stats.frames_out, frames * 2);
    assert_eq!(report.stats.interpolated, frames - 1);
    assert_eq!(report.multiplier, 2);
    assert!(report.realtime_factor() > 0.0);

    let (header, n) = count_y4m_frames(&out_path);
    assert_eq!(header.dims, out_dims);
    assert_eq!(n, frames * 2);
    // The writer must carry the doubled cadence, otherwise the clip plays at
    // twice the speed.
    assert!(
        (header.fps() - 60.0).abs() < 0.01,
        "fps was {}",
        header.fps()
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn png_sequence_through_pipeline_respects_output_limit() {
    let dir = scratch_dir("png");
    let frames_dir = dir.join("in");
    let src_dims = Dims::new(48, 32);
    let frames = 8u64;

    {
        let mut writer = PngWriter::create(&frames_dir, src_dims).unwrap();
        for i in 0..frames {
            let t = i as f32 / 24.0;
            writer.present(&synthetic_pattern(src_dims, t, i)).unwrap();
        }
    }

    let mut cfg = FluidConfig::default();
    cfg.interpolator.mode = InterpMode::Fixed;
    cfg.interpolator.multiplier = 3;
    cfg.upscaler.filter = UpscaleFilter::Lanczos;
    cfg.upscaler.scale.mode = ScaleMode::Factor;
    cfg.upscaler.scale.factor = 1.5;
    cfg.upscaler.temporal.enabled = false;

    let out_dims = FrameProcessor::new(&cfg, src_dims).unwrap().out_dims();
    assert_eq!(out_dims, Dims::new(72, 48));

    let mut source = ImageSequence::open_dir(&frames_dir, 24.0).unwrap();
    assert_eq!(source.len(), frames as usize);
    let mut sink = fluid_present::null_sink(out_dims);

    let run_cfg = PipelineConfig::offline().with_max_output_frames(5);
    let mut pipeline = Pipeline::with_config(&cfg, src_dims, run_cfg).unwrap();
    let report = pipeline.run(&mut source, &mut sink).unwrap();

    assert_eq!(report.stats.frames_out, 5);
    assert!(
        report.stats.frames_in >= 2,
        "limited run should still consume frames"
    );
    assert_eq!(report.out_dims, out_dims);
    // The first source frame is duplicated (3 frames), then the second source
    // frame contributes 2 interpolated frames before the 5-frame limit hits.
    assert!(
        report.stats.interpolated >= 2,
        "interpolated={}",
        report.stats.interpolated
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn mismatched_sink_dims_fail_fast() {
    let cfg = FluidConfig::default();
    let mut pipeline = Pipeline::new(&cfg, Dims::new(64, 64)).unwrap();
    let mut sink = fluid_present::null_sink(Dims::new(32, 32));
    let mut source = fluid_capture::mock_source(Dims::new(64, 64), 30.0);
    let err = pipeline.run(&mut source, &mut sink).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("sink"), "unexpected error: {text}");
}
