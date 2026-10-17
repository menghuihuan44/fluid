# fluid

A lightweight, fully external frame-generation and image-quality utility for
Windows, implemented in Rust and native D3D11.

fluid does not inject DLLs, hook graphics APIs, or read/write target process
memory. It uses official Windows capture APIs (Windows Graphics Capture for a
single window and Desktop Duplication as a full-screen fallback), composes frames
in its own topmost, click-through overlay, and therefore never modifies a game
or video file itself.

> This is a working development build. The complete path
> "desktop capture -> GPU frame ring -> motion estimation -> interpolation and
> scaling -> low-latency presentation" is implemented. It is not a finished
> replacement for DLSS Frame Generation or a commercial super-resolution model;
> quality and compatibility still need validation on more GPUs, games, and videos.

## Implemented

- **Two external capture backends**:
  - Windows Graphics Capture for the last foreground window or a selected window.
    This is the recommended path: it avoids the DDA overlay feedback loop and only
    produces frames when the captured window changes.
  - DXGI Desktop Duplication for full-monitor capture fallback. Pointer-only
    updates are ignored and unchanged frames are skipped.
- **Improved window discovery**: DWM bounds fallback to GetWindowRect, empty-title
  and large tool windows are included, shell windows are filtered, foreground and
  large windows are sorted first, and the last external foreground window can be
  added directly without searching the list.
- **Recursion-safe overlay**: DirectComposition, FLIP_DISCARD, a waitable
  swapchain with MaximumFrameLatency=1, and WDA_EXCLUDEFROMCAPTURE.
- **GPU-only hot path**: captured textures flow through a 2-slot GPU ring. The CPU
  only updates small constant buffers and statistics; no pixel readback.
- **Motion-compensated interpolation**:
  - A GPU compute pass performs coarse-to-fine block motion estimation at 1/8
    source resolution.
  - It emits motion vectors, residual, and confidence.
  - A pixel shader warps both endpoint frames and blends them at the requested
    display time.
  - Scene cuts, occlusions, and low-confidence motion automatically fall back to
    the real current frame to avoid large ghosting artifacts.
- **Image-quality pass**:
  - Adaptive 4x4 Catmull-Rom reconstruction.
  - Local color-range clamping to suppress edge ringing.
  - RCAS-style contrast-adaptive sharpening with anti-ringing clamping.
- **Output modes**:
  - Match the target window size.
  - Fit the target window to its monitor, for low-resolution borderless windows
    used with external upscaling.
- **Low-latency presentation**: accumulated-frame accounting, refresh pacing,
  SyncInterval=0 mode, and an optional Vsync mode.
- **Observability**: performance HUD, capture/present FPS, GPU timestamp duration,
  effective content latency, dropped frames, and interpolation state. Logs are
  written next to the configuration.
- **Usability**: Chinese settings UI, low-power idle mode, global hotkeys
  (Ctrl+Alt+Shift+F9 start/stop, F10 HUD), direct tray open/quit, optional
  close-to-tray, optional auto-start, and persistent configuration. Starting
  capture minimizes the settings window by default so it costs no UI GPU budget
  while gaming.

## Architecture

```text
target window / desktop
      |
      v
DXGI Desktop Duplication (read-only)
      |  GPU CopyResource
      v
3-slot GPU frame ring
      |
      +-- compute: coarse-to-fine motion estimation at 1/8 resolution
      |
      v
ps_compose: temporal interpolation + bidirectional warp + scene/occlusion
            fallback + adaptive spatial reconstruction
      |
      v
ps_rcas: contrast-adaptive sharpening
      |
      v
HUD overlay -> DirectComposition overlay window
               (click-through, excluded from capture)
```

## Requirements

- Windows 10 1909+ or Windows 11 (Windows Graphics Capture requires 1809+;
  Desktop Duplication fallback requires 1903+)
- A real GPU with D3D11 feature level 11.0 (integrated GPUs are supported)
- The target game should use borderless/windowed mode; exclusive fullscreen
  cannot be captured reliably
- Rust stable for building from source

Desktop Duplication formats BGRA8, 10-bit, and FP16 scRGB are accepted. FP16 capture
uses an approximate SDR tone-mapping path; exact Windows HDR metadata, white-point
handling, and color management remain future work.

## Build

```powershell
cargo build --release -p fluid-app
.\target\release\fluid.exe
```

Quality checks and headless tests:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p fluid-testkit -- render
cargo run -p fluid-testkit -- overlay
cargo run -p fluid-testkit -- wgc
cargo run -p fluid-testkit -- capture 3
```

- `render` runs two WARP GPU regression tests: identity composition/upscale and
  two-frame motion estimation + interpolation + adaptive scaling + sharpening.
- `overlay` smoke-tests DirectComposition and the waitable swapchain.
- `wgc` probes Windows Graphics Capture against a visible target window.
- `capture` probes real Desktop Duplication and needs an interactive desktop.
- `feedback` is a diagnostic that demonstrates why WGC is preferred over
  overlay-over-DDA for the running capture path.

## Usage

1. Start `fluid.exe`. Switch the game to borderless/windowed mode.
2. Choose a capture source:
   - **Recent foreground window (recommended)**: alt-tab to the game once, then
     launch/restore fluid. This uses WGC and needs no window list.
   - **Full monitor**: a compatibility fallback that uses Desktop Duplication.
   - **Specific window**: select from the improved list or use "Use recent
     foreground window".
3. Pick an output mode for window capture:
   - **Match window**: overlay matches the target; best for native-resolution output.
   - **Fill monitor**: fit the target window to its monitor; use for low-resolution
     borderless windows with external upscaling. Aspect-ratio preservation and
     letterboxing are enabled by default.
4. Choose a scaling mode and adjust sharpening.
5. Enable motion-compensated interpolation if desired.
   - It only activates when the captured frame rate is meaningfully lower than the
     display refresh rate (for example 60 FPS source on a 120 Hz display).
   - Interpolation needs two endpoint frames, so it adds roughly half a capture
     interval of latency. That physical cost cannot be fully removed by an external
     interpolator.
6. Click Start or press Ctrl+Alt+Shift+F9. The overlay is click-through, so mouse
   input reaches the game. The settings window minimizes by default to leave the
   foreground CPU/GPU budget to the game. The interpolation multiplier caps GPU
   spending: 2x is conservative, 3-4x favors smoothness.
7. Ctrl+Alt+Shift+F9 stops/starts capture even while the UI is minimized.
   Ctrl+Alt+Shift+F10 toggles the HUD.
8. The X button exits completely by default. Enable close-to-tray if you want the
   icon to remain; tray "open settings" stops capture so the settings window is
   never blocked by the topmost overlay; tray "quit" exits the process directly.

Configuration:

```text
%APPDATA%\fluid-project\fluid\config\config.toml
```

Logs:

```text
%APPDATA%\fluid-project\fluid\config\logs\
```

## Security boundary

- No injection, API hooking, or writing to target process memory.
- No target process handles are opened to read memory; window enumeration uses
  public Win32 window information.
- No modification of game files, video files, drivers, or system graphics settings.
- Capture is a normal system-level desktop-capture operation, similar to recording
  or streaming software.

No third-party software can honestly guarantee that every anti-cheat will accept
it. Some games or platforms may restrict overlays, capture, external helpers, or
particular process combinations. Follow the specific game's rules and terms.

## Known limitations

- Exclusive fullscreen cannot be captured; use borderless/windowed mode.
- Desktop Duplication captures the final desktop composition. If another window
  covers the target, that occlusion is captured too. Keep the target unobstructed.
- HDR/FP16 capture is experimental and uses approximate tone mapping. Disable HDR
  for the most predictable colors. Exact HDR metadata and wide-gamut management are
  future work.
- Fast rotation, particles, camera cuts, and flickering UI may trigger fallback or
  minor artifacts.
- WGC may show the Windows capture border on older Windows builds where
  `SetIsBorderRequired(false)` is unavailable.
- Full-monitor Desktop Duplication can feed overlay presentations back into DDA,
  causing higher capture/render load. Use foreground/window WGC mode whenever
  possible; full-monitor mode remains the compatibility fallback.
- Multi-GPU selection, offline video interpolation, and an installer are still
  roadmap items.
- The spatial scaler is project-native adaptive reconstruction plus RCAS-style
  sharpening. It does not contain AMD FSR1 code; FSR/neural upscaler integration
  points are future work.

## Configuration summary

```toml
[target]
auto_start = false
mode = "ForegroundWindow"   # or "Window" / "Monitor"
monitor = "\\.\DISPLAY1#0"

[display]
present_mode = "LowLatency" # or "Vsync"
show_hud = true
close_to_tray = false
hide_on_start = true
global_hotkeys = true

[scale]
mode = "Adaptive"           # or "Bilinear" / "Passthrough"
output = "MatchWindow"      # or "Monitor"
sharpness = 30              # 0..=100
preserve_aspect = true

[fg]
enabled = true
motion_compensation = true
strength = 85               # 0..=100
scene_protection = 60       # 0..=100
max_multiplier = 2          # cap output FPS at source FPS * this value
```

Old configuration files are migrated automatically with defaults.

## Roadmap

- Configurable hotkeys, per-game profiles, and an option to tune settings while
  capture is running without stopping.
- Temporal motion-field smoothing, better occlusion detection, UI protection, and
  further interpolation quality tuning.
- Explicit HDR metadata/color management, FSR1/neural upscaler integration, and an
  offline video interpolation CLI.
- Objective multi-GPU latency/quality benchmarks and a release packaging pipeline.
