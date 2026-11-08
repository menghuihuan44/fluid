# Fluid Architecture

This document describes how Fluid is put together internally: the crate graph, the
data flow through one frame, the ownership and lifetime rules, and the extension
points that future platform backends plug into.

## Design goals

1. **Universality over integration.** Fluid never asks a game, engine or video
   pipeline for cooperation. Everything it knows comes from the pixels it
   captures. That single decision drives most of the architecture below.
2. **Predictable latency.** Frame generation inevitably adds at least one source
   frame of delay. Fluid makes that cost explicit and gives the operator knobs
   (`--max-queue`, pacing, multiplier) to trade quality for delay instead of hiding it.
3. **Small footprint.** No runtime, no GPU driver install, no background service.
   A single static binary, a TOML file, and the OS capture API.
4. **Safe by construction.** All algorithmic crates are `#![forbid(unsafe_code)]`. Unsafe
   appears only in the platform crates, wrapped in documented FFI shims.
5. **Cross-platform core.** Capture and presentation are traits owned by
   `fluid-core`; everything between them is portable Rust.

## Crate graph

```text
            +--------------+
            |  fluid-core  |  types / config / error / log / stats
            +------+-------+
                   |
   +--------------+--------------+-------------------+------------------+
   |              |              |                   |                  |
fluid-flow   fluid-interp   fluid-upscale     fluid-capture   fluid-present / fluid-video
   |              |              |                   |                  |
   |              |        +-----+------+            |   (platform backends: windows / android / mock)
   |              |        |  fluid-gpu |            |
   |              |        |  wgpu compute            |
   +--------------+--------+------------+------------+------------------+
                                  |
                           +------v-------+
                           | fluid-engine |  FrameProcessor, Pipeline, Pacer, Stats
                           +------+-------+
                                  |
                   +--------------+--------------+
                   |                             |
            +------v-------+              +------v-------+
            |  fluid-cli   |              |  fluid-gui   |
            +--------------+              +--------------+
```

Dependencies only ever point from right to left in the diagram (leaf algorithm
crates depend on `fluid-core` only). `fluid-core` depends on nothing but serde/toml.
`fluid-cli` is the only crate that knows how to talk to a user.

Why this split:

- **fluid-core** owns the vocabulary (`Frame`, `Dims`, `LumaPlane`, `PixelFormat`, `Rect`,
  `FluidConfig`, `FluidError`, `FrameSource`, `FrameSink`, logging, telemetry primitives).
  Changing it ripples everywhere, so it is intentionally small and stable.
- **fluid-flow** is pure maths on luminance planes; it has no idea a GPU or a
  display exists, which makes it trivially unit-testable headless.
- **fluid-interp** and **fluid-upscale** are also pure functions on frames.
- **fluid-capture** / **fluid-present** are the platform boundary. On Windows they
  link `windows` under `[target.'cfg(windows)'.dependencies]`; on Android they compile
  `cfg(target_os = "android")` stubs; everything else gets the mock/null pair so the
  test suite runs on any host.
- **fluid-video** gives deterministic, reproducible I/O (y4m today) so quality and
  performance can be measured without a display.
- **fluid-engine** is the only crate that composes the others: it owns
  `FrameProcessor` (algorithm composition), `Pipeline` (the run loop),
  `Pacer` (timing) and `PipelineStats` (telemetry).

## Frame flow

```text
                 +-----------------------------------------------+
  screen/window  |  FrameSource (DXGI / WGC window / y4m / PNG /   |
  /video         |                synthetic)                       |
  ------------>  +----------------------+------------------------+
                                        | Frame (prev, cur)
                                        v
                 +-----------------------------------------------+
                 | SceneCutDetector  ->  fluid_flow::estimate    |
                 +----------------------+------------------------+
                                        | FlowEstimate (forward, backward, confidence)
                                        v
                 +-----------------------------------------------+
                 | Interpolator: k=1..multiplier-1 intermediates |
                 +----------------------+------------------------+
                                        | Frame list (intermediates..., real frame last)
                                        v
                 +-----------------------------------------------+
                 | Upscaler: resize + temporal blend + sharpen   |
                 +----------------------+------------------------+
                                        | out_dims frames
                                        v
                 +-----------------------------------------------+
                 | FrameSink (Win32 window / y4m / PNG / null)   |
                 +-----------------------------------------------+
```

Per source frame `P`:

1. **Capture.** `FrameSource::next_frame()` returns a `Frame` in the source pixel
   format. The Windows backend copies the DXGI desktop texture (or, for window
   capture, the Windows Graphics Capture frame; DXGI-crop is the fallback) into a
   staging texture, maps it, and hands out a packed buffer. For screen capture it
   returns `Ok(None)` when the desktop did not change within the timeout (a "no new
   frame" event, not an error); the window source instead replays the last frame
   with a fresh timestamp so a minimized window never stalls the pipeline. Both
   report `Err(Cancelled)` when the user closed the session.
2. **Scene analysis.** `FrameProcessor::process()` converts the frame to a
   `LumaPlane` and asks `SceneCutDetector` for a dissimilarity score. Scores above
   `scene_cut_threshold` (default 0.18) mark a hard cut.
3. **Optical flow.** `fluid_flow::estimate(prev_luma, cur_luma, params)` returns a
   `FlowEstimate`: forward and backward dense fields plus a per-pixel confidence
   map derived from the forward-backward consistency error. Matching is pyramid
   block matching (half-resolution on large frames) followed by two damped
   Lucas-Kanade Gauss-Newton passes for sub-pixel accuracy, and a global-rotation
   detector runs on the coarsest pyramid level (2 degree sweep, parabolic refine,
   accepted only above 1.5 degrees when it clearly beats translation) so that
   camera/UI rotation becomes a `GlobalRotation` hint instead of noisy block motion.
4. **Interpolation.** For `k in 1..multiplier`, `Interpolator::interpolate(prev, cur,
   k/multiplier, flow)` warps both neighbours to time `t` and blends them. Pixels whose
   confidence is below `confidence_min` are treated as occluded; they are still
   motion-compensated (both directions sampled and mixed) instead of copying the
   nearer frame, which removes the hard alpha=0.5 seam. Rotation hints use a
   geodesic (rotate-then-translate) gather. After synthesis a gated 5-tap
   unsharp pass restores 2-3 px thin lines/subtitles (gain 4*alpha*(1-alpha),
   clamped to the neighbourhood, skipped for static pixels); `FLUID_NO_SHARP=1`
   disables it. A scene cut yields no intermediates at all.
5. **Upscaling.** Every emitted frame (interpolated *and* real) passes through
   `Upscaler::upscale()`, which resolves the output size from `ScaleConfig`,
   resamples with the configured filter, applies temporal blending when enabled,
   and finishes with a contrast-adaptive sharpen. When the fused GPU backend is
   active (`interp_backend: wgpu-fused`, any GPU adapter class when the output is
   resized; engine-level paired A/B measured 1.059x on an integrated adapter)
   steps 4 and 5 are one GPU submission and the intermediate frame never reaches
   the CPU.
6. **Presentation.** `FrameSink::present()` shows or encodes the frame. The
   pipeline emits interpolated frames first and the real frame last so that, if
   the sink drops a frame, the driver can drop the oldest intermediate rather
   than the newest real frame. On Windows there are two interchangeable present
   backends behind `present.backend`: `dxgi-swapchain` (D3D11 flip-model swap
   chain; default, paces on vsync or presents with tearing) and `win32-window`
   (GDI `StretchDIBits`, the compatibility fallback).

## Invariants

- A `Frame`'s `data.len() == dims.area() * fmt.bytes_per_pixel()` always holds;
  `Frame::new` is the only constructor and validates it.
- Output frames are all `out_dims` and the sink must agree; `Pipeline::run` fails
  fast with `Unsupported` if `sink.dims() != processor.out_dims()`.
- `FrameProcessor` keeps exactly one previous frame; memory is bounded by
  `frame_bytes * max_queue` plus one flow field pair.
- Time is monotonic: pacing uses `std::time::Instant`, frame timestamps are
  i64 nanoseconds since the Unix epoch.
- Errors are values, never panics. `FluidError::Cancelled` is the graceful-stop
  signal; `Pipeline::run` maps it to `Ok(PipelineReport)`.

## Threading model (v0.1)

The *pipeline* is single-threaded: capture -> flow -> synthesis -> upscale ->
present all run on the calling thread, one source frame at a time. That keeps
latency accounting exact, keeps the memory footprint tiny and avoids
queue-induced jitter, which is the whole point of the project. `max_queue`
therefore describes the planned ring buffer depth and `1` already means
"fully synchronous".

Inside the compute stages the *work* is data-parallel: `fluid-flow`,
`fluid-interp` and `fluid-upscale` split their output into fixed row or block
bands and process bands on `std::thread::scope` workers (worker count from
`available_parallelism()`, capped at 16, overridable with the `FLUID_THREADS`
environment variable). Bands are fixed and reduced in band order, so the output
bytes - and the reported metrics - are **bit-identical for any thread count**;
there is a parallel-vs-serial test per stage that asserts exactly that. This is
why the crates stay `#![forbid(unsafe_code)]`: `thread::scope` plus disjoint
mutable slices give the same speed without any `unsafe`.

The pieces are already split so that *pipeline-level* parallelism can be added
later without API changes:

- `FrameSource` and `FrameSink` are `Send`, so a future pipeline can move
  capture and presentation to their own threads behind a ring buffer, and
  overlap stage `n+1` with stage `n` (the ~50 ms 1080p pair cost is currently
  spent serially).


## Static-frame fast path

A source frame whose pixels are byte-identical to the previous one (a static
desktop, a paused game, a menu) skips flow, synthesis and upscale entirely:
`FrameProcessor` reuses the last presented frame for the generated slots and
for the real frame, so a 1080p pipeline that costs ~50 ms per pair drops to a
frame clone (~1 ms). The comparison is an exact byte compare, so it can never
change output pixels; it only stops redundant work. `PipelineStats` counts these
frames in `static_frames` and the report prints them as
`static : N frame(s) reused`.

The report also instruments the two I/O edges, which are the only stages the
engine does not control: `source_wait_mean_ms` (time blocked in
`FrameSource::next_frame`, i.e. waiting for the compositor) and
`present_mean_ms` (time inside `FrameSink::present`, including a vsync wait).
Together with the process mean they make end-to-end latency readable from a
single JSON report instead of guesswork.

## Latency accounting

```text
  stage          measured by             measured cost (i7-1260P, 12C/16T, release)
  ------------   ----------------------    ----------------------------------------
  capture        FrameSource backend      < 1 ms when the desktop does not change
  flow           StageTimings::flow_ns    ~30-34 ms at 1080p, ~16-20 ms at 720p
                                           (pyramid match + 2x LK + rotation detect +
                                            fwd/bwd dense field + confidence; half-res
                                            matching on large frames is already included)
  interpolation  StageTimings::interp_ns  ~18-30 ms per generated 1080p frame
                                           (~10-11 ms at 720p) including the thin-line
                                           post-pass; a 2x pair costs ~73 ms at 1080p
                                           and ~33 ms at 720p on the CPU
  upscale        StageTimings::upscale_ns 1.0 with sharpness 0: ~1 ms (identity, CPU)
                                           720p->1440p bilinear+sharpen: ~13.5 ms cpu / ~11 ms gpu
                                           1080p->4K nearest: ~28 ms cpu / ~20 ms gpu
                                           1080p->4K bilinear: ~37 ms cpu / ~21 ms gpu
                                           1080p->4K adaptive: ~86 ms cpu / ~33 ms gpu
                                           (resample + sharpen + temporal, all parallel;
                                            GPU timings include upload and readback)
  present        FrameSink + Pacer        GDI ~0.5 ms blit; DXGI flip model
                                           present_mean 0.39-0.86 ms at 640x360 and
                                           ~4.5 ms at 1080p with vsync, + vsync wait
```

The numbers above are single-process release measurements; flow, synthesis and
upscale are parallelised with `std::thread::scope` (row/block bands, worker count
from `available_parallelism()`, overridable via the `FLUID_THREADS` env var) and
produce **bit-identical** output for any thread count, so results are reproducible.

Upscaling has a second implementation on the GPU (`fluid-gpu`, wgpu compute).
It is used for every non-identity resize when an adapter exists, and on this GPU
it mirrors the CPU sampling conventions and a parity suite compares the two per
filter. On Windows the wgpu device now comes up on D3D12 first (Vulkan is the
Android and fallback backend; `vendor/gpu-allocator` pins its `windows`
dependency to 0.62 so the D3D12 backend compiles next to our Win32 crates). A 1:1 resize always stays on the CPU: on the GPU it would only
add an upload plus readback (~21 ms at 1080p) around a sharpen pass that the CPU
does faster.

Frame generation has a GPU implementation too: `GpuInterpolator` runs the whole
pyramid matching, LK refinement, occlusion fill, confidence and synthesis chain
as WGSL compute dispatches, so a pair costs two uploads plus a readback instead
of a CPU pass. On the Iris Xe here it wins on small frames and loses on large
ones (720p: 41-44 ms versus 45-52 ms on the CPU; 1080p: 92-101 ms versus
84-92 ms), so `--interp-backend auto` reads the adapter class from wgpu:
discrete and virtual GPUs always take the GPU path, integrated GPUs only up to
1.3 MP of frame area, larger frames stay on the CPU. Either way the report's
`interp_backend` field says which one actually ran (`cpu` / `wgpu-flow`).

Measured end-to-end on the i7-1260P (release, idle machine, 90 frames, see
`scripts/perf-suite.ps1` / `out/perf-suite.md`): 720p 2x interpolation + 1.5x
upscale 34.9 ms per source frame (47.9 fps out) with the fused GPU path versus
71.3 ms (25 fps) CPU-only; 1080p 2x + 1.5x 68.4 ms (24 fps) fused versus 130.1 ms
(13.5 fps) CPU-only; 1080p 2x at native size 87.1 ms (19.3 fps) because the
integrated adapter policy keeps 1:1 frame generation on the CPU; 1080p->4K
nearest/bilinear/adaptive 26.3/27.8/40.2 ms (25/24/18.5 fps) on the GPU versus
28.3/37.0/85.5 ms CPU-only; pass-through 4.2 ms (61 fps). The fused path is the
main GPU win (~1.9x at 1080p with a resize); CPU optical flow still caps 1080p/60
when the output is not resized. Upscaling a 2880x1800 HiDPI desktop is not real
time on this class of CPU - use `--region` or `--window-process` to capture only
the game window.

`PipelineStats` measures each stage with `Instant::now()` spans
(`StageTimings`) and feeds ring buffers
(`LatencyStats`) so `summary()` can report p50/p95/p99. Those numbers are the
input to the `--preset` tuning: presets trade pyramid levels, block size and
search radius (flow cost) against quality.

## Configuration

`FluidConfig` is a plain serde/toml tree with a `validate()` that enforces every
range; the CLI applies overrides on top and re-validates. Every field has a
sensible default, so an empty file is valid, and `config --check` prints the
normalised result. Presets are *suggestions* applied onto a config, never a
separate code path: the algorithm always reads the same fields.

## Extension points

- **New capture backend**: implement `FrameSource` in `fluid-capture`, gate it
  behind `#[cfg(...)]`, add a `CaptureBackendKind` variant in `fluid-core`, and wire it in
  `fluid_capture::open()`.
- **New presentation backend**: same recipe with `FrameSink` in `fluid-present`.
- **New upscale filter**: add a variant to `UpscaleFilter` and handle it inside
  `fluid_upscale::resize`; `filter_supported()` and `filter_cost_hint()` must be
  updated so the CLI can warn about expensive choices.
- **GPU upscaling**: `fluid-upscale` exposes the object-safe trait
  `UpscaleBackend` (backend_name/src_dims/dst_dims/upscale/reset/last_cost_ns).
  `fluid-gpu` implements it with wgpu compute shaders; `FrameProcessor` builds
  either backend through `make_upscaler()` and falls back to the CPU one when
  no adapter exists, so the rest of the engine never cares which one ran.
  A new GPU filter is a WGSL kernel plus a matching tap table; the CPU crate stays
  the reference implementation and the parity tests compare against it.
- **New frame generation backend**: `fluid-engine` owns the object-safe trait
  `FrameGenBackend` (`backend_name`, `multiplier`, `generate(prev, next,
  luma_prev, luma_next) -> Generated`, `reset`, `last_cost_ns`) in
  `crates/fluid-engine/src/framegen.rs`. `CpuFrameGen` wraps `fluid-flow` plus
  `fluid-interp`; `GpuFrameGen` wraps `fluid_gpu::GpuInterpolator` (same chain as
  WGSL compute shaders). `make_framegen(cfg, dims, gpu_ctx)` reads
  `interpolator.backend` (`auto` / `cpu` / `gpu`), applies the adapter-class
  policy in `auto` mode and falls back to the CPU implementation when the GPU
  path is unavailable; the pipeline report prints the backend that actually ran
  in `interp_backend`.

## Testing strategy

- Headless by default: every algorithm test uses `LumaPlane`/`Frame` built in memory
  or the synthetic pattern from `fluid-video`. No test requires a display, a GPU
  or a network.
- Golden-ish checks: flow tests assert that a synthetically translated block is
  recovered within a pixel; interpolation tests assert that a full-blend frame
  equals the average of two identical frames; upscale tests assert
  nearest/bilinear exact values.
- Integration: `fluid-engine` runs `fluid-video::SyntheticSource` into a
  `null_sink` or a y4m/Png writer, then re-reads the result to check frame count,
  dimensions and fps.
- Performance checks live in `fluid bench` (synthetic, reproducible) rather than
  in unit tests, so CI never depends on machine speed.

## Why this is not a cheat

Fluid runs entirely outside the target process: it uses the same OS-level desktop
duplication API as OBS, recording or remote-desktop software (DXGI Desktop
Duplication on Windows), and presents a borderless window. It does not inject
code, hook DirectX/OpenGL, read or write game memory, simulate input, or patch
game files. It only ever sees the composited desktop image, exactly like a
screen recorder, and DRM-protected surfaces are simply black.
