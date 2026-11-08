#![forbid(unsafe_code)]
//! Optical flow estimation, flow field representation and scene-cut detection.
//!
//! The flow estimator is a coarse-to-fine block matcher:
//! build an image pyramid, search a small window per block at the coarsest
//! level, then refine at finer levels with integer window refinement followed by
//! two damped Gauss-Newton/Lucas-Kanade iterations over the whole block patch (a
//! 2x2 gradient-moment sum solved for a sub-pixel displacement, with the
//! gradient planes read through nearest-neighbour taps). Backward flow plus
//! forward/backward consistency produces the per-pixel confidence used by the
//! interpolator to detect occlusions, and both dense fields and the confidence
//! map are written in one fused band sweep. The hot inner loops (block SAD,
//! dense field fill, confidence) use `wide`'s portable `f32x4` types and keep a
//! scalar tail for the leftover pixels.
//!
//! Stages are split into fixed pixel/block row bands and run with
//! `std::thread::scope` on up to `available_parallelism()` workers (capped at 16,
//! or lower through the `FLUID_THREADS` environment variable);
//! tiny frames and single-core machines take the serial path. Band boundaries
//! never depend on the worker count, so the estimate is bit-identical however
//! many threads run it.
//!
//! Large frames (more than 12000 blocks, i.e. 720p and up at the default block
//! size) are matched on a half-resolution pyramid level and the resulting field
//! plus confidence map are bilinearly upsampled; this keeps the block grid and
//! the confidence pass around a quarter of the full-resolution cost at the price
//! of a sub-pixel accuracy loss measured below 0.25 px. Set
//! `FLUID_FULL_RES_FLOW=1` to force the full-resolution path (used by the
//! accuracy regression test).
//!
//! Large-angle rotation (3 to 8 degrees, where the per-block translation that
//! the matcher can represent is no longer an adequate model) gets a second
//! hypothesis: a one-parameter global rotation is fitted on the coarsest
//! pyramid level, guarded by a pure-translation residual so ordinary panning
//! never triggers it. When a rotation wins, the next frame is derotated before
//! matching and the analytic rotation displacement is added back into both
//! dense fields afterwards; `FlowEstimate::rotation` records the angle so the
//! interpolator can follow the exact geodesic instead of the linear field.
//! Set `FLUID_NO_ROTATION=1` to disable the rotation hypothesis.

use fluid_core::{Dims, FlowConfig, FluidError, LumaPlane, Result};
use wide::f32x4;

/// A 2D displacement in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn scaled(self, k: f32) -> Vec2 {
        Vec2::new(self.x * k, self.y * k)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x - other.x, self.y - other.y)
    }
}

/// Dense motion field, one vector per source pixel.
#[derive(Clone, Debug)]
pub struct FlowField {
    dims: Dims,
    vectors: Vec<Vec2>,
}

impl FlowField {
    pub fn new(dims: Dims) -> Self {
        Self {
            dims,
            vectors: vec![Vec2::ZERO; dims.area() as usize],
        }
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn vectors(&self) -> &[Vec2] {
        &self.vectors
    }

    pub fn vectors_mut(&mut self) -> &mut [Vec2] {
        &mut self.vectors
    }

    pub fn at(&self, x: u32, y: u32) -> Vec2 {
        self.vectors[(y as usize) * (self.dims.width as usize) + (x as usize)]
    }

    pub fn set(&mut self, x: u32, y: u32, v: Vec2) {
        let w = self.dims.width as usize;
        self.vectors[(y as usize) * w + (x as usize)] = v;
    }

    /// Bilinear sample clamped to the field bounds.
    pub fn sample(&self, x: f32, y: f32) -> Vec2 {
        if self.is_empty() {
            return Vec2::ZERO;
        }
        let w = self.dims.width as usize;
        let h = self.dims.height as usize;
        let max_x = w as f32 - 1.0;
        let max_y = h as f32 - 1.0;
        let x = if x.is_finite() {
            x.clamp(0.0, max_x)
        } else {
            0.0
        };
        let y = if y.is_finite() {
            y.clamp(0.0, max_y)
        } else {
            0.0
        };
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let a = self.vectors[y0 * w + x0];
        let b = self.vectors[y0 * w + x1];
        let c = self.vectors[y1 * w + x0];
        let d = self.vectors[y1 * w + x1];
        let top = a.add(b.sub(a).scaled(fx));
        let bot = c.add(d.sub(c).scaled(fx));
        top.add(bot.sub(top).scaled(fy))
    }

    pub fn mean_len(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0f32;
        for v in &self.vectors {
            acc += v.len();
        }
        acc / self.vectors.len() as f32
    }

    pub fn max_len(&self) -> f32 {
        let mut best = 0.0f32;
        for v in &self.vectors {
            let l = v.len();
            if l > best {
                best = l;
            }
        }
        best
    }

    pub fn clear(&mut self) {
        self.vectors.iter_mut().for_each(|v| *v = Vec2::ZERO);
    }
}

/// Tuning knobs for the flow estimator (mirrors fluid_core::FlowConfig).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowParams {
    pub pyramid_levels: u32,
    pub block_size: u32,
    pub search_radius: u32,
    pub refine_iters: u32,
    pub confidence_min: f32,
}

impl Default for FlowParams {
    fn default() -> Self {
        Self {
            pyramid_levels: 3,
            block_size: 8,
            search_radius: 8,
            refine_iters: 2,
            confidence_min: 0.2,
        }
    }
}

impl From<&FlowConfig> for FlowParams {
    fn from(c: &FlowConfig) -> Self {
        Self {
            pyramid_levels: c.pyramid_levels,
            block_size: c.block_size,
            search_radius: c.search_radius,
            refine_iters: c.refine_iters,
            confidence_min: c.confidence_min,
        }
    }
}

/// A detected global rotation between two frames.
///
/// Convention: next(p) is approximately prev(c + R(-angle) * (p + 0.5 - c) -
/// 0.5), where c is the rotation centre in full-resolution pixel coordinates.
/// The block matcher derotates the next frame first, so
/// `FlowEstimate::forward` still holds the total prev-to-next displacement while
/// the confidence map is built from the small residual flow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalRotation {
    /// Rotation angle in radians; positive values rotate the next frame around
    /// the centre in the direction of the sampling formula above.
    pub angle: f32,
    /// Rotation centre X in full-resolution pixel coordinates.
    pub center_x: f32,
    /// Rotation centre Y in full-resolution pixel coordinates.
    pub center_y: f32,
}

impl GlobalRotation {
    pub fn new(angle: f32, center_x: f32, center_y: f32) -> Self {
        Self {
            angle,
            center_x,
            center_y,
        }
    }

    /// Sine and cosine of the angle, so per-pixel code avoids libm calls.
    #[inline]
    pub fn cos_sin(self) -> (f32, f32) {
        self.angle.sin_cos()
    }

    /// True when the angle is large enough for the estimator to compensate.
    #[inline]
    pub fn is_significant(self) -> bool {
        self.angle.abs() >= MIN_ROTATION_RAD
    }

    /// Gather displacement of pixel (x, y) under this rotation, i.e. the part
    /// of `forward` explained by the rotation alone.
    #[inline]
    pub fn forward_at(self, x: f32, y: f32) -> Vec2 {
        let (sn, cs) = self.cos_sin();
        let vx = x + 0.5 - self.center_x;
        let vy = y + 0.5 - self.center_y;
        Vec2::new(cs * vx + sn * vy - vx, -sn * vx + cs * vy - vy)
    }

    /// Gather displacement of pixel (x, y) in the next-to-prev direction.
    #[inline]
    pub fn backward_at(self, x: f32, y: f32) -> Vec2 {
        let (sn, cs) = self.cos_sin();
        let vx = x + 0.5 - self.center_x;
        let vy = y + 0.5 - self.center_y;
        Vec2::new(cs * vx - sn * vy - vx, sn * vx + cs * vy - vy)
    }
}

/// Smallest rotation angle (radians) the estimator acts on: 1.5 degrees.
///
/// Below that the coarse-level fit cannot separate the rotation from a small
/// translation plus resampling error, and acting on a misestimated small angle
/// costs more quality than it recovers.
const MIN_ROTATION_RAD: f32 = 0.026_179_744;
/// Largest angle (radians) considered by the rotation sweep: eight degrees.
const MAX_ROTATION_RAD: f32 = 0.139_626_34;
/// Coarse sweep step (radians): two degrees. Nine candidates over the search
/// window; the minimum is refined with a parabolic fit, which recovers the
/// sub-degree angle accuracy a fine second sweep used to provide for free.
const ROTATION_SWEEP_STEP: f32 = 0.034_906_585;
/// A rotation is accepted only if its residual is below this fraction of the
/// best pure-translation residual (translation is the null hypothesis).
const ROTATION_GUARD: f32 = 0.6;
/// Minimum frame area (pixels) for rotation detection; smaller frames keep the
/// cheaper translation-only path.
const MIN_ROTATION_PIXELS: u64 = 640 * 360;
/// Sample stride of the sparse grid used by the rotation search. Four keeps
/// enough textured samples at the coarsest level (about 4.8k at 1080p) while
/// quartering the scoring work of the old stride-two grid.
const ROTATION_SAMPLE_STRIDE: u32 = 4;
/// Sample stride of the cheaper translation guard grid.
const ROTATION_GUARD_STRIDE: u32 = 4;
/// Worker cap for the rotation search. Nine angle candidates do not need a
/// full-core fork/join, and the spawn cost is a large share of the fit on small
/// frames, so eight workers are the sweet spot measured on a 16 thread machine.
const ROTATION_MAX_WORKERS: usize = 8;

/// Forward and backward flow plus per-pixel confidence in `0..=1`.
#[derive(Clone, Debug)]
pub struct FlowEstimate {
    pub forward: FlowField,
    pub backward: FlowField,
    pub confidence: Vec<f32>,
    /// Global rotation the estimator compensated for, if any. The displacement
    /// fields already include the rotation, so this is a hint for consumers
    /// that want the exact geodesic path instead of a linear field blend.
    pub rotation: Option<GlobalRotation>,
}

impl FlowEstimate {
    pub fn new(forward: FlowField, backward: FlowField, confidence: Vec<f32>) -> Self {
        Self {
            forward,
            backward,
            confidence,
            rotation: None,
        }
    }

    /// Attach a detected global rotation to this estimate.
    pub fn with_rotation(mut self, rotation: Option<GlobalRotation>) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn dims(&self) -> Dims {
        self.forward.dims()
    }

    pub fn confidence_at(&self, x: u32, y: u32) -> f32 {
        let w = self.forward.dims().width as usize;
        *self
            .confidence
            .get((y as usize) * w + x as usize)
            .unwrap_or(&0.0)
    }

    /// Low confidence usually means occlusion or a disoccluded region.
    pub fn is_occluded(&self, x: u32, y: u32, min_conf: f32) -> bool {
        self.confidence_at(x, y) < min_conf
    }

    /// Binary occlusion mask (255 = occluded) at full flow resolution.
    pub fn occlusion_mask(&self, min_conf: f32) -> Vec<u8> {
        let n = self.forward.dims().area() as usize;
        let mut mask = vec![0u8; n];
        for (i, m) in mask.iter_mut().enumerate() {
            let c = *self.confidence.get(i).unwrap_or(&0.0);
            if c < min_conf {
                *m = 255;
            }
        }
        mask
    }
}

const TIE_BREAK: f32 = 1.0e-4;
/// Relative weight of the forward/backward consistency term in the confidence map.
const CONF_K: f32 = 0.5;

/// Full-resolution block count above which matching runs on a half-resolution
/// grid (see `matching_target`).
const LARGE_FRAME_BLOCKS: u32 = 12_000;

/// Upper bound on worker threads used by one frame-sized stage.
const MAX_WORKER_THREADS: usize = 16;

/// Worker count requested through the FLUID_THREADS environment variable.
///
/// Mirrors the upscaler convention: a missing, unparsable or zero value means
/// "use every available core".
fn env_thread_limit() -> Option<usize> {
    std::env::var("FLUID_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// Truthy environment switch: 1/true/yes (case-insensitive).
fn switch_truthy(v: &str) -> bool {
    let v = v.trim();
    v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
}

/// Escape hatch for the reduced-resolution fast path: set
/// `FLUID_FULL_RES_FLOW=1` to force full-resolution matching and confidence.
fn force_full_res() -> bool {
    std::env::var("FLUID_FULL_RES_FLOW")
        .ok()
        .map(|v| switch_truthy(&v))
        .unwrap_or(false)
}

/// True when the block grid is large enough for the reduced-resolution fast
/// path (half-resolution matching and confidence), unless the caller forced
/// full resolution through `FLUID_FULL_RES_FLOW`.
fn reduced_resolution(dims: Dims, bs: u32) -> bool {
    if force_full_res() {
        return false;
    }
    let bs = bs.max(1);
    (dims.width / bs) * (dims.height / bs) > LARGE_FRAME_BLOCKS
}

/// Pyramid level at which block matching runs, and therefore the resolution of
/// the confidence grid. Level 1 means half resolution (the dense field is still
/// emitted at full resolution); level 0 means full resolution.
fn target_level(half: bool, levels: usize) -> usize {
    if half {
        (levels.saturating_sub(1)).min(1)
    } else {
        0
    }
}

fn worker_threads(pixels: usize) -> usize {
    if pixels < 65_536 {
        return 1;
    }
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let limit = env_thread_limit().unwrap_or(avail);
    limit.min(avail).clamp(1, MAX_WORKER_THREADS)
}

/// Rows (or blocks) per band: aim for roughly four bands per worker so hybrid
/// CPUs with fast and slow cores stay balanced.
fn band_rows(total_rows: usize, threads: usize) -> usize {
    total_rows.div_ceil(threads.max(1) * 4).max(1)
}

/// Apply `f(band, first_index)` to every fixed-size band of `data`, spreading
/// the bands over `threads` workers through a small local queue.
///
/// Band boundaries depend only on `band_len` and `data.len()`, never on the
/// thread count, and each band writes only its own elements: the output is
/// bit-identical for every worker count, which keeps the estimator
/// deterministic. The queue lets hybrid (performance + efficiency core) CPUs
/// balance the bands dynamically.
fn parallel_bands<T: Send>(
    data: &mut [T],
    band_len: usize,
    threads: usize,
    f: impl Fn(&mut [T], usize) + Sync,
) {
    if data.is_empty() {
        return;
    }
    let band_len = band_len.max(1);
    let mut jobs: Vec<(usize, &mut [T])> = Vec::new();
    let mut rest: &mut [T] = data;
    let mut first = 0usize;
    while !rest.is_empty() {
        let take = band_len.min(rest.len());
        let (band, tail) = rest.split_at_mut(take);
        rest = tail;
        jobs.push((first, band));
        first += take;
    }
    let threads = threads.clamp(1, jobs.len());
    if threads == 1 {
        for (first, band) in jobs {
            f(band, first);
        }
        return;
    }
    let queue = std::sync::Mutex::new(jobs);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let queue = &queue;
            let f = &f;
            scope.spawn(move || {
                loop {
                    let job = queue.lock().expect("band queue poisoned").pop();
                    let Some((first, band)) = job else {
                        break;
                    };
                    f(band, first);
                }
            });
        }
    });
}

/// Row-major wrapper around `parallel_bands`: `f(rows, first_row)`.
fn parallel_row_bands<T: Send>(
    data: &mut [T],
    row_len: usize,
    rows_per_band: usize,
    threads: usize,
    f: impl Fn(&mut [T], usize) + Sync,
) {
    let row_len = row_len.max(1);
    debug_assert_eq!(data.len() % row_len, 0);
    let band_len = rows_per_band.max(1) * row_len;
    parallel_bands(data, band_len, threads, move |band, first| {
        f(band, first / row_len)
    });
}

/// Tri-buffer variant of parallel_row_bands: the same band of three parallel
/// output buffers is handed to a worker, so one stage can write the forward
/// field, the backward field and the confidence map in a single sweep.
fn parallel_tri_bands(
    fwd: &mut [Vec2],
    bwd: &mut [Vec2],
    conf: &mut [f32],
    rows_per_band: usize,
    row_len: usize,
    threads: usize,
    f: impl Fn(&mut [Vec2], &mut [Vec2], &mut [f32], usize) + Sync,
) {
    let row_len = row_len.max(1);
    let band_len = rows_per_band.max(1) * row_len;
    type Job<'a> = (usize, &'a mut [Vec2], &'a mut [Vec2], &'a mut [f32]);
    let mut jobs: Vec<Job<'_>> = Vec::new();
    let mut first = 0usize;
    let mut fi = fwd.chunks_mut(band_len);
    let mut bi = bwd.chunks_mut(band_len);
    let mut ci = conf.chunks_mut(band_len);
    while let (Some(a), Some(b), Some(c)) = (fi.next(), bi.next(), ci.next()) {
        jobs.push((first, a, b, c));
        first += band_len;
    }
    if jobs.is_empty() {
        return;
    }
    let threads = threads.clamp(1, jobs.len());
    if threads == 1 {
        for (first, a, b, c) in jobs {
            f(a, b, c, first / row_len);
        }
        return;
    }
    let queue = std::sync::Mutex::new(jobs);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let queue = &queue;
            let f = &f;
            scope.spawn(move || {
                loop {
                    let job = queue.lock().expect("band queue poisoned").pop();
                    let Some((first, a, b, c)) = job else { break };
                    f(a, b, c, first / row_len);
                }
            });
        }
    });
}

/// 2x box downscale, identical in value to `LumaPlane::downscale2` but spread
/// over rows. Kept in this crate because the core plane has no parallel form.
fn downscale2_par(src: &LumaPlane, threads: usize) -> LumaPlane {
    let w = src.dims.width as usize;
    let h = src.dims.height as usize;
    let dw = (w / 2).max(1);
    let dh = (h / 2).max(1);
    let mut out = LumaPlane::zeros(Dims::new(dw as u32, dh as u32));
    let mut sx0 = vec![0usize; dw];
    let mut sx1 = vec![0usize; dw];
    for x in 0..dw {
        let sx = (x * 2).min(w - 1);
        sx0[x] = sx;
        sx1[x] = (sx + 1).min(w - 1);
    }
    let rows_per_band = band_rows(dh, threads);
    parallel_row_bands(
        &mut out.data,
        dw,
        rows_per_band,
        threads,
        |band, first_y| {
            let band_rows = band.len() / dw;
            for r in 0..band_rows {
                let y = first_y + r;
                let sy = (y * 2).min(h - 1);
                let sy1 = (sy + 1).min(h - 1);
                let row_a = sy * w;
                let row_b = sy1 * w;
                let dst = &mut band[r * dw..(r + 1) * dw];
                for (x, value) in dst.iter_mut().enumerate() {
                    let a = sx0[x];
                    let b = sx1[x];
                    *value = (src.data[row_a + a]
                        + src.data[row_a + b]
                        + src.data[row_b + a]
                        + src.data[row_b + b])
                        * 0.25;
                }
            }
        },
    );
    out
}

/// Build the downscaled pyramid levels of one plane (level 1..levels-1).
fn build_pyramid(full: &LumaPlane, extra_levels: usize, threads: usize) -> Vec<LumaPlane> {
    let mut pyr: Vec<LumaPlane> = Vec::with_capacity(extra_levels);
    for l in 0..extra_levels {
        let src = if l == 0 { full } else { &pyr[l - 1] };
        pyr.push(downscale2_par(src, threads));
    }
    pyr
}

/// Vertex offset of the parabola through (-1, sm), (0, s0), (1, sp), clamped to +-0.5.
#[inline]
fn parabola(sm: f32, s0: f32, sp: f32) -> f32 {
    let denom = sm - 2.0 * s0 + sp;
    if denom > 1.0e-6 {
        ((sm - sp) / (2.0 * denom)).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

/// Vectorised SAD over a block whose displaced copy is fully inside both planes.
///
/// Four samples accumulate into one lane of an `f32x4`, the lane sums are folded
/// in a fixed lane order and the leftover samples (fewer than four, or a strided
/// block) are appended afterwards. Nothing depends on the worker count, so the
/// value is deterministic; it can differ from a strict raster-order scalar sum by
/// a few ULPs because floating point addition is not associative. Candidate
/// *ranking* against the scalar reference is covered by
/// `simd_sad_ranking_matches_sequential`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn sad_block_fast(
    src: &[f32],
    sw: usize,
    dst: &[f32],
    dw: usize,
    x0: usize,
    y0: usize,
    ix: usize,
    iy: usize,
    bw: usize,
    bh: usize,
    step: usize,
) -> f32 {
    let mut acc = f32x4::splat(0.0);
    let mut y = 0usize;
    while y < bh {
        let sbase = (y0 + y) * sw + x0;
        let dbase = (iy + y) * dw + ix;
        let srow = &src[sbase..sbase + bw];
        let drow = &dst[dbase..dbase + bw];
        let mut x = 0usize;
        if step == 1 {
            while x + 4 <= bw {
                let sv = f32x4::from(&srow[x..x + 4]);
                let dv = f32x4::from(&drow[x..x + 4]);
                acc += (sv - dv).abs();
                x += 4;
            }
            if x < bw {
                // A short slice is zero padded; the padded lanes contribute 0.
                let sv = f32x4::from(&srow[x..]);
                let dv = f32x4::from(&drow[x..]);
                acc += (sv - dv).abs();
            }
        } else {
            let mut sampled = [0.0f32; 4];
            let mut other = [0.0f32; 4];
            while x < bw {
                let mut lane = 0usize;
                while lane < 4 && x + lane * step < bw {
                    let i = x + lane * step;
                    sampled[lane] = srow[i];
                    other[lane] = drow[i];
                    lane += 1;
                }
                while lane < 4 {
                    sampled[lane] = 0.0;
                    other[lane] = 0.0;
                    lane += 1;
                }
                acc += (f32x4::from(sampled) - f32x4::from(other)).abs();
                x += 4 * step;
            }
        }
        y += step;
    }
    let lanes = acc.to_array();
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}
/// Sum of absolute differences between the source block at (x0, y0) and the
/// destination frame sampled at the integer offset (dx, dy).
///
/// The number of samples only depends on the source block, so costs for
/// different offsets are directly comparable. Sampling every second pixel
/// (step = 2) accelerates the wide coarse search; the final refinement always
/// uses every pixel.
#[allow(clippy::too_many_arguments)]
#[inline]
fn block_sad(
    src: &LumaPlane,
    dst: &LumaPlane,
    x0: u32,
    y0: u32,
    bs: u32,
    dx: i32,
    dy: i32,
    step: u32,
) -> f32 {
    let sw = src.dims.width as usize;
    let sh = src.dims.height as usize;
    let dw = dst.dims.width as usize;
    let dh = dst.dims.height as usize;
    let x0 = (x0 as usize).min(sw);
    let y0 = (y0 as usize).min(sh);
    let bw = (bs as usize).min(sw - x0);
    let bh = (bs as usize).min(sh - y0);
    let step = step as usize;
    let step = if step == 0 { 1 } else { step };
    let ix = x0 as isize + dx as isize;
    let iy = y0 as isize + dy as isize;
    // Fast path: the displaced block is fully inside both planes, so no column
    // or row clamping is needed and the SIMD kernel can read whole rows.
    if ix >= 0 && iy >= 0 && (ix as usize + bw) <= dw && (iy as usize + bh) <= dh {
        return sad_block_fast(
            &src.data,
            sw,
            &dst.data,
            dw,
            x0,
            y0,
            ix as usize,
            iy as usize,
            bw,
            bh,
            step,
        );
    }
    // Edge path: partial blocks and blocks whose displacement leaves the plane.
    let max_x = dw as isize - 1;
    let max_y = dh as isize - 1;
    let mut sad = 0.0f32;
    let mut y = 0usize;
    while y < bh {
        let py = y0 + y;
        let sy = ((py as isize + dy as isize).clamp(0, max_y)) as usize;
        let sbase = py * sw;
        let dbase = sy * dw;
        let mut x = 0usize;
        while x < bw {
            let sx = ((x0 as isize + x as isize + dx as isize).clamp(0, max_x)) as usize;
            sad += (src.data[sbase + x0 + x] - dst.data[dbase + sx]).abs();
            x += step;
        }
        y += step;
    }
    sad
}

/// Wide integer search on the coarsest level: decimated SAD over the full
/// search radius with a small penalty that prefers shorter displacements.
fn coarse_search(src: &LumaPlane, dst: &LumaPlane, x0: u32, y0: u32, bs: u32, radius: i32) -> Vec2 {
    let step = if radius >= 4 { 2 } else { 1 };
    let mut best = Vec2::ZERO;
    let mut best_cost = block_sad(src, dst, x0, y0, bs, 0, 0, 2);
    let mut dy = -radius;
    while dy <= radius {
        let mut dx = -radius;
        while dx <= radius {
            if dx != 0 || dy != 0 {
                let sad = block_sad(src, dst, x0, y0, bs, dx, dy, 2);
                let cost = sad + TIE_BREAK * (dx * dx + dy * dy) as f32;
                if cost < best_cost - 1.0e-6 {
                    best_cost = cost;
                    best = Vec2::new(dx as f32, dy as f32);
                }
            }
            dx += step;
        }
        dy += step;
    }
    best
}

/// One refinement pass: 3x3 integer search around the current estimate followed
/// by a separable parabolic fit for sub-pixel accuracy.
fn refine_once(
    src: &LumaPlane,
    dst: &LumaPlane,
    x0: u32,
    y0: u32,
    bs: u32,
    start: Vec2,
    patch_step: u32,
) -> Vec2 {
    let base_x = start.x.round() as i32;
    let base_y = start.y.round() as i32;
    let mut best = Vec2::new(base_x as f32, base_y as f32);
    let mut best_cost = f32::INFINITY;
    // The 3x3 SADs are kept: the parabolic fit below usually reads four of them.
    let mut sads = [0.0f32; 9];
    for oy in 0..3 {
        for ox in 0..3 {
            let dx = base_x + ox - 1;
            let dy = base_y + oy - 1;
            let sad = block_sad(src, dst, x0, y0, bs, dx, dy, patch_step);
            sads[(oy * 3 + ox) as usize] = sad;
            let cost = sad + TIE_BREAK * (dx * dx + dy * dy) as f32;
            if cost < best_cost - 1.0e-6 {
                best_cost = cost;
                best = Vec2::new(dx as f32, dy as f32);
            }
        }
    }
    let ix = best.x as i32;
    let iy = best.y as i32;
    let cached = |dx: i32, dy: i32| -> f32 {
        let ox = dx - base_x + 1;
        let oy = dy - base_y + 1;
        if (0..3).contains(&ox) && (0..3).contains(&oy) {
            sads[(oy * 3 + ox) as usize]
        } else {
            block_sad(src, dst, x0, y0, bs, dx, dy, patch_step)
        }
    };
    let s0 = cached(ix, iy);
    let sm = cached(ix - 1, iy);
    let sp = cached(ix + 1, iy);
    let um = cached(ix, iy - 1);
    let up = cached(ix, iy + 1);
    Vec2::new(best.x + parabola(sm, s0, sp), best.y + parabola(um, s0, up))
}

/// Bilinear interpolation of a block grid of offsets at block-centre coordinates.
#[inline]
fn sample_grid(grid: &[Vec2], cols: usize, rows: usize, gx: f32, gy: f32) -> Vec2 {
    let gx = gx.clamp(0.0, (cols - 1) as f32);
    let gy = gy.clamp(0.0, (rows - 1) as f32);
    let x0 = gx.floor() as usize;
    let y0 = gy.floor() as usize;
    let x1 = (x0 + 1).min(cols - 1);
    let y1 = (y0 + 1).min(rows - 1);
    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;
    let a = grid[y0 * cols + x0];
    let b = grid[y0 * cols + x1];
    let c = grid[y1 * cols + x0];
    let d = grid[y1 * cols + x1];
    let top = a.add(b.sub(a).scaled(fx));
    let bot = c.add(d.sub(c).scaled(fx));
    top.add(bot.sub(top).scaled(fy))
}

/// Bilinear sample of a single-channel plane (edge clamped).
#[inline]
fn sample_plane(data: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    if !x.is_finite() || !y.is_finite() {
        return 0.0;
    }
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let a = data[y0 * w + x0];
    let b = data[y0 * w + x1];
    let c = data[y1 * w + x0];
    let d = data[y1 * w + x1];
    let top = a + (b - a) * fx;
    let bot = c + (d - c) * fx;
    top + (bot - top) * fy
}

/// Central-difference gradient planes used by the sub-pixel pass.
fn gradient_planes(p: &LumaPlane) -> (Vec<f32>, Vec<f32>) {
    let w = p.dims.width as usize;
    let h = p.dims.height as usize;
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    for y in 0..h {
        let y_up = y.saturating_sub(1);
        let y_dn = (y + 1).min(h - 1);
        for x in 0..w {
            let x_l = x.saturating_sub(1);
            let x_r = (x + 1).min(w - 1);
            let i = y * w + x;
            gx[i] = 0.5 * (p.data[y * w + x_r] - p.data[y * w + x_l]);
            gy[i] = 0.5 * (p.data[y_dn * w + x] - p.data[y_up * w + x]);
        }
    }
    (gx, gy)
}

/// Damped Gauss-Newton (Lucas-Kanade) iterations per block: fit the
/// displacement that minimises dst(p + o) - src(p) over the whole block patch.
/// Only textured pixels contribute, so flat patches keep the integer estimate.
/// The gradient planes are read with nearest-neighbour taps because they are
/// already smoothed by the 2x downscale that produced them; every pixel of the
/// patch is still sampled, since subsampling the lattice measurably hurt
/// sub-pixel accuracy on thin structures.
#[allow(clippy::too_many_arguments)]
fn lk_refine_block(
    src: &LumaPlane,
    dst: &LumaPlane,
    gx: &[f32],
    gy: &[f32],
    x0: u32,
    y0: u32,
    bs: u32,
    start: Vec2,
    iters: u32,
    limit: f32,
) -> Vec2 {
    let w = dst.dims.width as usize;
    let h = dst.dims.height as usize;
    let sw = src.dims.width;
    let sh = src.dims.height;
    let mut o = start;
    for _ in 0..iters {
        let (mut a11, mut a12, mut a22) = (0.0f32, 0.0f32, 0.0f32);
        let (mut b1, mut b2) = (0.0f32, 0.0f32);
        let mut used = 0u32;
        for py in 0..bs {
            let y = y0 + py;
            if y >= sh {
                break;
            }
            for px in 0..bs {
                let x = x0 + px;
                if x >= sw {
                    break;
                }
                let sx = x as f32 + o.x;
                let sy = y as f32 + o.y;
                let gi = sy.clamp(0.0, (h - 1) as f32) as usize * w
                    + sx.clamp(0.0, (w - 1) as f32) as usize;
                let (gxv, gyv) = (gx[gi], gy[gi]);
                if gxv * gxv + gyv * gyv < 4.0e-4 {
                    continue;
                }
                let dv = sample_plane(&dst.data, w, h, sx, sy);
                let e = src.data[(y * sw + x) as usize] - dv;
                a11 += gxv * gxv;
                a12 += gxv * gyv;
                a22 += gyv * gyv;
                b1 += gxv * e;
                b2 += gyv * e;
                used += 1;
            }
        }
        if used < 4 {
            break;
        }
        let det = a11 * a22 - a12 * a12;
        if !det.is_finite() || det <= 1.0e-9 {
            break;
        }
        let dx = (a22 * b1 - a12 * b2) / det;
        let dy = (a11 * b2 - a12 * b1) / det;
        if !dx.is_finite() || !dy.is_finite() {
            break;
        }
        o.x = (o.x + dx.clamp(-0.5, 0.5)).clamp(-limit, limit);
        o.y = (o.y + dy.clamp(-0.5, 0.5)).clamp(-limit, limit);
        // Early exit: the update is already far below a tenth of a pixel.
        if dx.abs() + dy.abs() < 0.01 {
            break;
        }
    }
    o
}

/// Pyramid accessor: level 0 is the full-resolution plane, level n >= 1 is pyr[n - 1].
#[inline]
fn plane<'a>(full: &'a LumaPlane, pyr: &'a [LumaPlane], level: usize) -> &'a LumaPlane {
    if level == 0 { full } else { &pyr[level - 1] }
}

/// Coarse-to-fine block matching for one direction.
///
/// Returns one offset per block at the target level (in target-level pixels)
/// plus the grid dimensions. An offset is the displacement that maps a source
/// block onto the destination frame, i.e. the content velocity.
#[allow(clippy::too_many_arguments)]
fn match_flow(
    src: &LumaPlane,
    dst: &LumaPlane,
    src_pyr: &[LumaPlane],
    dst_pyr: &[LumaPlane],
    params: &FlowParams,
    levels: usize,
    target: usize,
    threads: usize,
) -> (Vec<Vec2>, u32, u32) {
    let bs = params.block_size.max(1);
    let radius = params.search_radius.min(32) as i32;
    let mut level = levels.saturating_sub(1);
    let mut cols = (plane(src, src_pyr, level).dims.width / bs).max(1);
    let mut rows = (plane(src, src_pyr, level).dims.height / bs).max(1);
    let want_target = target.min(level);
    let mut offsets = vec![Vec2::ZERO; (cols as usize) * (rows as usize)];
    {
        let s = plane(src, src_pyr, level);
        let t = plane(dst, dst_pyr, level);
        let grad = if level == want_target {
            Some(gradient_planes(t))
        } else {
            None
        };
        let row_len = cols as usize;
        let rows_per_band = band_rows(rows as usize, threads);
        parallel_row_bands(
            &mut offsets,
            row_len,
            rows_per_band,
            threads,
            |band, first_by| {
                let band_rows = band.len() / row_len;
                for r in 0..band_rows {
                    let by = (first_by + r) as u32;
                    for bx in 0..cols {
                        let x0 = bx * bs;
                        let y0 = by * bs;
                        let start = coarse_search(s, t, x0, y0, bs, radius);
                        let mut v = refine_once(s, t, x0, y0, bs, start, 1);
                        if let Some((gx, gy)) = &grad {
                            v = lk_refine_block(s, t, gx, gy, x0, y0, bs, v, 2, radius as f32);
                        }
                        band[r * row_len + bx as usize] = v;
                    }
                }
            },
        );
    }
    while level > want_target {
        let old_level = level;
        let old_cols = cols;
        let old_rows = rows;
        let new_level = level - 1;
        let old_w = plane(src, src_pyr, old_level).dims.width as f32;
        let old_h = plane(src, src_pyr, old_level).dims.height as f32;
        let s = plane(src, src_pyr, new_level);
        let t = plane(dst, dst_pyr, new_level);
        let grad = if new_level == want_target {
            Some(gradient_planes(t))
        } else {
            None
        };
        let scale_x = s.dims.width as f32 / old_w.max(1.0);
        let scale_y = s.dims.height as f32 / old_h.max(1.0);
        cols = (s.dims.width / bs).max(1);
        rows = (s.dims.height / bs).max(1);
        let passes = if new_level == want_target {
            params.refine_iters.clamp(1, 4)
        } else {
            1
        };
        let patch_step = if new_level == want_target { 1 } else { 2 };
        let mut next = vec![Vec2::ZERO; (cols as usize) * (rows as usize)];
        let row_len = cols as usize;
        let rows_per_band = band_rows(rows as usize, threads);
        parallel_row_bands(
            &mut next,
            row_len,
            rows_per_band,
            threads,
            |band, first_by| {
                let band_rows = band.len() / row_len;
                for r in 0..band_rows {
                    let by = (first_by + r) as u32;
                    for bx in 0..cols {
                        let gx = (bx as f32 + 0.5) * (old_cols as f32 / cols as f32) - 0.5;
                        let gy = (by as f32 + 0.5) * (old_rows as f32 / rows as f32) - 0.5;
                        let up =
                            sample_grid(&offsets, old_cols as usize, old_rows as usize, gx, gy);
                        let mut v = Vec2::new(up.x * scale_x, up.y * scale_y);
                        for _ in 0..passes {
                            v = refine_once(s, t, bx * bs, by * bs, bs, v, patch_step);
                        }
                        if let Some((gx, gy)) = &grad {
                            v = lk_refine_block(
                                s,
                                t,
                                gx,
                                gy,
                                bx * bs,
                                by * bs,
                                bs,
                                v,
                                2,
                                radius as f32,
                            );
                        }
                        band[r * row_len + bx as usize] = v;
                    }
                }
            },
        );
        offsets = next;
        level = new_level;
    }
    (offsets, cols, rows)
}

/// Two interleaved `Vec2` lanes (four floats) of the vertical pass of `fill_and_confidence`.
#[inline]
fn lerp_interleaved(row_a: &[f32], row_b: &[f32], o: usize, fy: f32x4, scale: f32x4) -> f32x4 {
    let a = f32x4::from(&row_a[o..o + 4]);
    let b = f32x4::from(&row_b[o..o + 4]);
    (a + (b - a) * fy) * scale
}

/// Expand both block grids into dense fields and compute the per-pixel
/// confidence in one sweep over the output rows.
///
/// The dense value at a pixel is a bilinear interpolation between block centres
/// times sign * level_scale, and the confidence combines forward/backward
/// consistency with local texture. Fusing the two fills and the confidence pass
/// keeps the small offset grids hot in cache and avoids three extra sweeps over
/// the multi-megabyte flow fields.
#[allow(clippy::too_many_arguments)]
fn fill_and_confidence(
    offsets_f: &[Vec2],
    offsets_b: &[Vec2],
    cols: u32,
    rows: u32,
    bs: u32,
    level_scale: f32,
    prev: &LumaPlane,
    forward: &mut FlowField,
    backward: &mut FlowField,
    confidence: &mut [f32],
    threads: usize,
) {
    let w = prev.dims.width as usize;
    let h = prev.dims.height as usize;
    if w == 0 || h == 0 || cols == 0 || rows == 0 || offsets_f.is_empty() || offsets_b.is_empty() {
        return;
    }
    if forward.dims().width as usize != w || backward.dims().width as usize != w {
        return;
    }
    if confidence.len() != w * h {
        return;
    }
    let cols = cols as usize;
    let rows = rows as usize;
    let inv = 1.0 / (level_scale * bs as f32);
    let max_c = (cols - 1) as f32;
    let max_r = (rows - 1) as f32;
    // Column taps from full-resolution pixels to block-grid coordinates.
    let mut xi = vec![0usize; w];
    let mut xj = vec![0usize; w];
    let mut xt = vec![0.0f32; w];
    for x in 0..w {
        let g = ((x as f32 + 0.5) * inv - 0.5).clamp(0.0, max_c);
        let i = (g.floor() as usize).min(cols - 1);
        xi[x] = i;
        xj[x] = (i + 1).min(cols - 1);
        xt[x] = (g - i as f32).clamp(0.0, 1.0);
    }
    let data = &prev.data;
    let max_x = w as f32 - 1.0;
    let max_y = h as f32 - 1.0;
    let gather_scale = -level_scale;
    // Confidence is evaluated on every second pixel and bilinearly upsampled;
    // these are the column taps of that upsample.
    let gw = w.div_ceil(2);
    let gh = h.div_ceil(2);
    let last_gc = (gw - 1) as f32;
    let mut ux_i = vec![0usize; w];
    let mut ux_j = vec![0usize; w];
    let mut ux_t = vec![0.0f32; w];
    for x in 0..w {
        let g = (x as f32 * 0.5).clamp(0.0, last_gc);
        let i = (g.floor() as usize).min(gw - 1);
        ux_i[x] = i;
        ux_j[x] = (i + 1).min(gw - 1);
        ux_t[x] = (g - i as f32).clamp(0.0, 1.0);
    }
    let last_gr = (gh - 1) as f32;
    let rows_per_band = band_rows(h, threads);
    parallel_tri_bands(
        forward.vectors_mut(),
        backward.vectors_mut(),
        confidence,
        rows_per_band,
        w,
        threads,
        |fband, bband, cband, first_y| {
            let band_rows = fband.len() / w;
            // Confidence grid rows covered by this band (two pixels per cell).
            let first_gy = (first_y / 2).min(gh - 1);
            // One extra row: the last output row of a band interpolates towards
            // the first grid row of the next band.
            let last_gy = (((first_y + band_rows - 1) / 2) + 1).min(gh - 1);
            let n_gy = last_gy - first_gy + 1;
            let mut cgrid = vec![0.0f32; n_gy * gw];
            let mut top = [0.0f32; 4];
            let mut bot = [0.0f32; 4];
            for k in 0..n_gy {
                let gy = first_gy + k;
                let ypix = (2 * gy).min(h - 1);
                let y_up = ypix.saturating_sub(1);
                let y_dn = (ypix + 1).min(h - 1);
                let fgy = ((ypix as f32 + 0.5) * inv - 0.5).clamp(0.0, max_r);
                let row = &mut cgrid[k * gw..(k + 1) * gw];
                for (gx, slot) in row.iter_mut().enumerate() {
                    let xpix = (2 * gx).min(w - 1);
                    let fgx = ((xpix as f32 + 0.5) * inv - 0.5).clamp(0.0, max_c);
                    let f = sample_grid(offsets_f, cols, rows, fgx, fgy);
                    let fx = f.x * gather_scale;
                    let fy_px = f.y * gather_scale;
                    let px = (xpix as f32 + fx).clamp(0.0, max_x);
                    let py = (ypix as f32 + fy_px).clamp(0.0, max_y);
                    let b = sample_grid(
                        offsets_b,
                        cols,
                        rows,
                        ((px + 0.5) * inv - 0.5).clamp(0.0, max_c),
                        ((py + 0.5) * inv - 0.5).clamp(0.0, max_r),
                    );
                    let ex = fx + b.x * gather_scale;
                    let ey = fy_px + b.y * gather_scale;
                    let consistency = 1.0 / (1.0 + CONF_K * (ex * ex + ey * ey));
                    let xl = xpix.saturating_sub(1);
                    let xr = (xpix + 1).min(w - 1);
                    let g_x = data[ypix * w + xr] - data[ypix * w + xl];
                    let g_y = data[y_dn * w + xpix] - data[y_up * w + xpix];
                    let texture = (0.2 + 3.0 * (g_x.abs() + g_y.abs())).min(1.0);
                    *slot = (consistency * texture).clamp(0.0, 1.0);
                }
            }
            // Interleaved [x, y] scratch for the two neighbouring grid rows of
            // each direction; eight floats (two pixels) load as two f32x4.
            let mut fa = vec![0.0f32; w * 2];
            let mut fb = vec![0.0f32; w * 2];
            let mut ba = vec![0.0f32; w * 2];
            let mut bb = vec![0.0f32; w * 2];
            let mut cached_y0 = usize::MAX;
            for r in 0..band_rows {
                let y = first_y + r;
                let g = ((y as f32 + 0.5) * inv - 0.5).clamp(0.0, max_r);
                let y0 = (g.floor() as usize).min(rows - 1);
                let y1 = (y0 + 1).min(rows - 1);
                if y0 != cached_y0 {
                    cached_y0 = y0;
                    let row_a = y0 * cols;
                    let row_b = y1 * cols;
                    for x in 0..w {
                        let i = xi[x];
                        let j = xj[x];
                        let t = xt[x];
                        let p = offsets_f[row_a + i];
                        let q = offsets_f[row_a + j];
                        let v = p.add(q.sub(p).scaled(t));
                        fa[2 * x] = v.x;
                        fa[2 * x + 1] = v.y;
                        let p = offsets_f[row_b + i];
                        let q = offsets_f[row_b + j];
                        let v = p.add(q.sub(p).scaled(t));
                        fb[2 * x] = v.x;
                        fb[2 * x + 1] = v.y;
                        let p = offsets_b[row_a + i];
                        let q = offsets_b[row_a + j];
                        let v = p.add(q.sub(p).scaled(t));
                        ba[2 * x] = v.x;
                        ba[2 * x + 1] = v.y;
                        let p = offsets_b[row_b + i];
                        let q = offsets_b[row_b + j];
                        let v = p.add(q.sub(p).scaled(t));
                        bb[2 * x] = v.x;
                        bb[2 * x + 1] = v.y;
                    }
                }
                let fy = (g - y0 as f32).clamp(0.0, 1.0);
                let fy4 = f32x4::splat(fy);
                let scale = f32x4::splat(gather_scale);
                let frow = &mut fband[r * w..(r + 1) * w];
                let brow = &mut bband[r * w..(r + 1) * w];
                let mut x = 0usize;
                while x + 4 <= w {
                    let o = 2 * x;
                    let lo = lerp_interleaved(&fa, &fb, o, fy4, scale).to_array();
                    let hi = lerp_interleaved(&fa, &fb, o + 4, fy4, scale).to_array();
                    frow[x] = Vec2::new(lo[0], lo[1]);
                    frow[x + 1] = Vec2::new(lo[2], lo[3]);
                    frow[x + 2] = Vec2::new(hi[0], hi[1]);
                    frow[x + 3] = Vec2::new(hi[2], hi[3]);
                    let lo = lerp_interleaved(&ba, &bb, o, fy4, scale).to_array();
                    let hi = lerp_interleaved(&ba, &bb, o + 4, fy4, scale).to_array();
                    brow[x] = Vec2::new(lo[0], lo[1]);
                    brow[x + 1] = Vec2::new(lo[2], lo[3]);
                    brow[x + 2] = Vec2::new(hi[0], hi[1]);
                    brow[x + 3] = Vec2::new(hi[2], hi[3]);
                    x += 4;
                }
                while x < w {
                    let o = 2 * x;
                    let a = Vec2::new(fa[o], fa[o + 1]);
                    let b = Vec2::new(fb[o], fb[o + 1]);
                    frow[x] = a.add(b.sub(a).scaled(fy)).scaled(gather_scale);
                    let a = Vec2::new(ba[o], ba[o + 1]);
                    let b = Vec2::new(bb[o], bb[o + 1]);
                    brow[x] = a.add(b.sub(a).scaled(fy)).scaled(gather_scale);
                    x += 1;
                }
                // Upsample the band-local confidence grid to this row.
                let gyy = (y as f32 * 0.5).clamp(0.0, last_gr);
                let i0 = (gyy.floor() as usize).clamp(first_gy, last_gy);
                let i1 = (i0 + 1).min(last_gy);
                let ug = (gyy - i0 as f32).clamp(0.0, 1.0);
                let base0 = (i0 - first_gy) * gw;
                let base1 = (i1 - first_gy) * gw;
                let crow = &mut cband[r * w..(r + 1) * w];
                let mut x = 0usize;
                while x + 4 <= w {
                    for (lane, (slot_t, slot_b)) in top.iter_mut().zip(bot.iter_mut()).enumerate() {
                        let xx = x + lane;
                        let a0 = cgrid[base0 + ux_i[xx]];
                        let a1 = cgrid[base0 + ux_j[xx]];
                        *slot_t = a0 + (a1 - a0) * ux_t[xx];
                        let b0 = cgrid[base1 + ux_i[xx]];
                        let b1 = cgrid[base1 + ux_j[xx]];
                        *slot_b = b0 + (b1 - b0) * ux_t[xx];
                    }
                    let tv = f32x4::from(top);
                    let bv = f32x4::from(bot);
                    let t = tv + (bv - tv) * f32x4::splat(ug);
                    crow[x..x + 4].copy_from_slice(&t.to_array());
                    x += 4;
                }
                while x < w {
                    let a0 = cgrid[base0 + ux_i[x]];
                    let a1 = cgrid[base0 + ux_j[x]];
                    let t0 = a0 + (a1 - a0) * ux_t[x];
                    let b0 = cgrid[base1 + ux_i[x]];
                    let b1 = cgrid[base1 + ux_j[x]];
                    let t1 = b0 + (b1 - b0) * ux_t[x];
                    crow[x] = (t0 + (t1 - t0) * ug).clamp(0.0, 1.0);
                    x += 1;
                }
            }
        },
    );
}

/// One sample of the sparse grid used by the global-rotation search.
struct RotationSample {
    x: usize,
    y: usize,
    weight: f32,
}

/// True unless the rotation hypothesis is disabled with FLUID_NO_ROTATION=1.
fn rotation_enabled() -> bool {
    !std::env::var("FLUID_NO_ROTATION")
        .ok()
        .map(|v| switch_truthy(&v))
        .unwrap_or(false)
}

/// Collect texture-weighted samples of one plane for the rotation search.
///
/// Flat samples carry no orientation information and only add noise, so they
/// are dropped; the weight is the local gradient magnitude.
fn collect_rotation_samples(next: &LumaPlane, stride: u32) -> Vec<RotationSample> {
    let w = next.dims.width as usize;
    let h = next.dims.height as usize;
    let stride = stride.max(1) as usize;
    let mut samples = Vec::new();
    if w < 3 || h < 3 {
        return samples;
    }
    let mut y = 1usize;
    while y + 1 < h {
        let mut x = 1usize;
        while x + 1 < w {
            let gx = next.data[y * w + x + 1] - next.data[y * w + x - 1];
            let gy = next.data[(y + 1) * w + x] - next.data[(y - 1) * w + x];
            let weight = ((gx.abs() + gy.abs()) * 10.0).min(1.0);
            if weight > 0.08 {
                samples.push(RotationSample { x, y, weight });
            }
            x += stride;
        }
        y += stride;
    }
    samples
}

/// Weighted residual of a rotation hypothesis, normalised by the total weight.
fn rotation_cost(
    prev: &LumaPlane,
    next: &LumaPlane,
    samples: &[RotationSample],
    plane_scale: f32,
    rotation: GlobalRotation,
) -> f32 {
    let (sn, cs) = rotation.cos_sin();
    let w = next.dims.width as usize;
    let inv = 1.0 / plane_scale;
    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    for s in samples {
        let wx = (s.x as f32 + 0.5) * plane_scale - rotation.center_x;
        let wy = (s.y as f32 + 0.5) * plane_scale - rotation.center_y;
        let ux = (rotation.center_x + cs * wx + sn * wy) * inv - 0.5;
        let uy = (rotation.center_y - sn * wx + cs * wy) * inv - 0.5;
        let a = prev.sample_bilinear(ux, uy);
        let b = next.data[s.y * w + s.x];
        sum += s.weight * (a - b).abs();
        wsum += s.weight;
    }
    if wsum > 0.0 {
        sum / wsum
    } else {
        f32::INFINITY
    }
}

/// Weighted residual of a pure translation hypothesis (in plane pixels).
fn translation_cost(
    prev: &LumaPlane,
    next: &LumaPlane,
    samples: &[RotationSample],
    dx: i32,
    dy: i32,
) -> f32 {
    let w = next.dims.width as usize;
    let h = next.dims.height as usize;
    let max_x = w as i32 - 1;
    let max_y = h as i32 - 1;
    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    for s in samples {
        let sx = (s.x as i32 + dx).clamp(0, max_x) as usize;
        let sy = (s.y as i32 + dy).clamp(0, max_y) as usize;
        let a = prev.data[sy * w + sx];
        let b = next.data[s.y * w + s.x];
        sum += s.weight * (a - b).abs();
        wsum += s.weight;
    }
    if wsum > 0.0 {
        sum / wsum
    } else {
        f32::INFINITY
    }
}

/// Fit a global rotation on the coarsest pyramid level.
///
/// A rotation sweep from minus to plus eight degrees in two degree steps is
/// scored against the texture-weighted residual; the winner is refined with a
/// parabolic fit through the three costs around the minimum, which is free
/// compared with a second scoring pass. A rotation is accepted only when it
/// explains the pair clearly better than the best pure translation over a small
/// offset grid. Translation is the null hypothesis, so ordinary panning keeps
/// the cheap path.
fn detect_global_rotation(
    prev: &LumaPlane,
    next: &LumaPlane,
    plane_scale: f32,
    full: Dims,
    threads: usize,
) -> Option<GlobalRotation> {
    if prev.dims != next.dims || prev.dims.width < 16 || prev.dims.height < 16 {
        return None;
    }
    let threads = threads.clamp(1, ROTATION_MAX_WORKERS);
    let samples = collect_rotation_samples(next, ROTATION_SAMPLE_STRIDE);
    if samples.len() < 64 {
        return None;
    }
    let center_x = full.width as f32 * 0.5;
    let center_y = full.height as f32 * 0.5;
    let guard_samples = collect_rotation_samples(next, ROTATION_GUARD_STRIDE);
    let mut best_translation = f32::INFINITY;
    for dy in -3..=3 {
        for dx in -3..=3 {
            let cost = translation_cost(prev, next, &guard_samples, dx, dy);
            if cost < best_translation {
                best_translation = cost;
            }
        }
    }
    // The angle candidates are independent, so the sweep and the refinement run
    // as parallel bands over a cost vector; the minimum is then reduced in
    // candidate order, which keeps the chosen angle identical for every worker
    // count.
    let mut angles = Vec::with_capacity(16);
    let steps = (MAX_ROTATION_RAD / ROTATION_SWEEP_STEP).round() as i32;
    for k in -steps..=steps {
        angles.push(k as f32 * ROTATION_SWEEP_STEP);
    }
    let (costs, best_idx) = angle_costs(
        prev,
        next,
        &samples,
        plane_scale,
        center_x,
        center_y,
        &angles,
        threads,
    );
    let best = costs[best_idx];
    let mut best_angle = angles[best_idx];
    // Sub-step refinement from the quadratic through the three costs around the
    // minimum. The sweep grid is coarse, so this is what keeps the angle
    // accuracy the old fine second pass provided, at no additional cost.
    if best_idx > 0 && best_idx + 1 < costs.len() {
        let offset = parabola(costs[best_idx - 1], costs[best_idx], costs[best_idx + 1]);
        best_angle += offset * ROTATION_SWEEP_STEP;
    }
    if best_angle.abs() < MIN_ROTATION_RAD {
        return None;
    }
    if !best.is_finite() || best >= ROTATION_GUARD * best_translation {
        return None;
    }
    Some(GlobalRotation::new(best_angle, center_x, center_y))
}

/// Cost of every candidate angle, plus the index of the cheapest one.
///
/// Candidates are scored in parallel bands; the winner comes from a sequential
/// scan in candidate order, so it does not depend on the worker count. The full
/// cost vector allows a parabolic sub-step fit without a second pass.
#[allow(clippy::too_many_arguments)]
fn angle_costs(
    prev: &LumaPlane,
    next: &LumaPlane,
    samples: &[RotationSample],
    plane_scale: f32,
    center_x: f32,
    center_y: f32,
    angles: &[f32],
    threads: usize,
) -> (Vec<f32>, usize) {
    let mut costs = vec![0.0f32; angles.len()];
    let band_len = angles.len().div_ceil(threads.max(1)).max(1);
    parallel_bands(&mut costs, band_len, threads, |band, first| {
        for (k, slot) in band.iter_mut().enumerate() {
            let angle = angles[first + k];
            *slot = rotation_cost(
                prev,
                next,
                samples,
                plane_scale,
                GlobalRotation::new(angle, center_x, center_y),
            );
        }
    });
    let mut best = f32::INFINITY;
    let mut best_idx = 0usize;
    for (k, cost) in costs.iter().enumerate() {
        if *cost < best {
            best = *cost;
            best_idx = k;
        }
    }
    (costs, best_idx)
}

/// Resample the next frame so that the global rotation is removed:
/// next_derotated(p) = next(c + R(angle) * (p + 0.5 - c) - 0.5).
fn derotate_plane(src: &LumaPlane, rotation: GlobalRotation, threads: usize) -> LumaPlane {
    let dims = src.dims;
    let w = dims.width as usize;
    let h = dims.height as usize;
    let mut out = LumaPlane::zeros(dims);
    if w == 0 || h == 0 {
        return out;
    }
    let (sn, cs) = rotation.cos_sin();
    let rows_per_band = band_rows(h, threads);
    parallel_row_bands(&mut out.data, w, rows_per_band, threads, |band, first_y| {
        let rows = band.len() / w;
        for r in 0..rows {
            let y = first_y + r;
            let vy = y as f32 + 0.5 - rotation.center_y;
            let row = &mut band[r * w..(r + 1) * w];
            for (x, slot) in row.iter_mut().enumerate() {
                let vx = x as f32 + 0.5 - rotation.center_x;
                let qx = rotation.center_x + cs * vx - sn * vy - 0.5;
                let qy = rotation.center_y + sn * vx + cs * vy - 0.5;
                *slot = src.sample_bilinear(qx, qy);
            }
        }
    });
    out
}

/// Add the analytic rotation displacement to one residual dense field.
fn add_rotation_to_field(
    field: &mut FlowField,
    rotation: GlobalRotation,
    forward: bool,
    threads: usize,
) {
    let dims = field.dims();
    let w = dims.width as usize;
    let h = dims.height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let (sn, cs) = rotation.cos_sin();
    let rows_per_band = band_rows(h, threads);
    parallel_row_bands(
        field.vectors_mut(),
        w,
        rows_per_band,
        threads,
        |band, first_y| {
            let rows = band.len() / w;
            for r in 0..rows {
                let y = first_y + r;
                let vy = y as f32 + 0.5 - rotation.center_y;
                let row = &mut band[r * w..(r + 1) * w];
                for (x, v) in row.iter_mut().enumerate() {
                    let vx = x as f32 + 0.5 - rotation.center_x;
                    if forward {
                        v.x += cs * vx + sn * vy - vx;
                        v.y += -sn * vx + cs * vy - vy;
                    } else {
                        v.x += cs * vx - sn * vy - vx;
                        v.y += sn * vx + cs * vy - vy;
                    }
                }
            }
        },
    );
}

/// Promote the residual dense fields to total prev-to-next displacements.
fn add_rotation_fields(
    forward: &mut FlowField,
    backward: &mut FlowField,
    rotation: GlobalRotation,
    threads: usize,
) {
    add_rotation_to_field(forward, rotation, true, threads);
    add_rotation_to_field(backward, rotation, false, threads);
}

/// Estimate bidirectional dense flow between two luma frames.
///
/// Forward flow is a gather field: prev(x + forward(x)) is approximately
/// next(x). It is found by block matching prev patches inside next
/// coarse-to-fine; backward flow does the same in the other direction and the
/// pair drives the confidence map.
///
/// Parallel implementation of the public estimator. Exposed for the threaded
/// versus serial equivalence tests: the band layout never depends on the
/// worker count, so the result is bit-identical for every value of `threads`.
fn estimate_with(
    prev: &LumaPlane,
    next: &LumaPlane,
    params: &FlowParams,
    threads: usize,
) -> Result<FlowEstimate> {
    estimate_with_target(prev, next, params, threads, None)
}

/// Like `estimate_with` but with an explicit matching-level override (used by the
/// tuning experiments and their A/B tests; `None` follows the size policy).
fn estimate_with_target(
    prev: &LumaPlane,
    next: &LumaPlane,
    params: &FlowParams,
    threads: usize,
    target_override: Option<usize>,
) -> Result<FlowEstimate> {
    let half = reduced_resolution(prev.dims, params.block_size);
    estimate_impl(prev, next, params, threads, half, target_override)
}

/// Shared implementation. `half` selects the reduced-resolution fast path used
/// for large frames: block matching runs on a coarser pyramid level and the
/// coarse offsets are expanded into the full-resolution dense fields.
fn estimate_impl(
    prev: &LumaPlane,
    next: &LumaPlane,
    params: &FlowParams,
    threads: usize,
    half: bool,
    target_override: Option<usize>,
) -> Result<FlowEstimate> {
    if prev.dims != next.dims {
        return Err(FluidError::invalid(format!(
            "flow input dims mismatch: {} vs {}",
            prev.dims, next.dims
        )));
    }
    if prev.dims.is_empty() {
        return Err(FluidError::invalid("flow input must be non-empty"));
    }
    let bs = params.block_size.max(1);
    let want = params.pyramid_levels.clamp(1, 5) as usize;
    let mut levels = 1usize;
    let mut w = prev.dims.width;
    let mut h = prev.dims.height;
    while levels < want && w > bs && h > bs {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        levels += 1;
    }
    let extra = levels.saturating_sub(1);
    // The two pyramid chains are independent, so they run concurrently.
    let (prev_pyr, next_pyr) = if threads > 1 && extra > 0 {
        let chain_threads = (threads / 2).max(1);
        std::thread::scope(|scope| {
            let a = scope.spawn(|| build_pyramid(prev, extra, chain_threads));
            let b = scope.spawn(|| build_pyramid(next, extra, chain_threads));
            (
                a.join().expect("pyramid worker panicked"),
                b.join().expect("pyramid worker panicked"),
            )
        })
    } else {
        (build_pyramid(prev, extra, 1), build_pyramid(next, extra, 1))
    };
    // Very large frames are matched on a half-resolution grid (the dense field is
    // still full resolution); that keeps large pairs inside the real-time budget.
    let target = target_override
        .map(|t| t.min(levels.saturating_sub(1)))
        .unwrap_or_else(|| target_level(half, levels));
    // A global rotation is a different motion model from per-block
    // translation; fit it on the coarsest level before matching, and if it wins
    // derotate the next frame so the matcher only has to explain the residual.
    let rotation = if extra >= 1 && prev.dims.area() >= MIN_ROTATION_PIXELS && rotation_enabled() {
        let coarse = extra - 1;
        detect_global_rotation(
            &prev_pyr[coarse],
            &next_pyr[coarse],
            (1u32 << extra) as f32,
            prev.dims,
            threads,
        )
    } else {
        None
    };
    // Forward and backward matching are independent; each gets about half the
    // workers and switches between the pyramid levels on its own.
    let per_dir = if threads > 1 { (threads / 2).max(1) } else { 1 };
    let ((offsets_f, cols, rows), offsets_b) = if let Some(rot) = rotation {
        let derotated = derotate_plane(next, rot, threads);
        let derotated_pyr = build_pyramid(&derotated, extra, threads);
        if threads > 1 {
            std::thread::scope(|scope| {
                let a = scope.spawn(|| {
                    match_flow(
                        prev,
                        &derotated,
                        &prev_pyr,
                        &derotated_pyr,
                        params,
                        levels,
                        target,
                        per_dir,
                    )
                });
                let b = scope.spawn(|| {
                    match_flow(
                        &derotated,
                        prev,
                        &derotated_pyr,
                        &prev_pyr,
                        params,
                        levels,
                        target,
                        per_dir,
                    )
                });
                let a = a.join().expect("flow worker panicked");
                let b = b.join().expect("flow worker panicked");
                ((a.0, a.1, a.2), b.0)
            })
        } else {
            let a = match_flow(
                prev,
                &derotated,
                &prev_pyr,
                &derotated_pyr,
                params,
                levels,
                target,
                per_dir,
            );
            let b = match_flow(
                &derotated,
                prev,
                &derotated_pyr,
                &prev_pyr,
                params,
                levels,
                target,
                per_dir,
            );
            ((a.0, a.1, a.2), b.0)
        }
    } else if threads > 1 {
        std::thread::scope(|scope| {
            let a = scope.spawn(|| {
                match_flow(
                    prev, next, &prev_pyr, &next_pyr, params, levels, target, per_dir,
                )
            });
            let b = scope.spawn(|| {
                match_flow(
                    next, prev, &next_pyr, &prev_pyr, params, levels, target, per_dir,
                )
            });
            let a = a.join().expect("flow worker panicked");
            let b = b.join().expect("flow worker panicked");
            ((a.0, a.1, a.2), b.0)
        })
    } else {
        let a = match_flow(
            prev, next, &prev_pyr, &next_pyr, params, levels, target, per_dir,
        );
        let b = match_flow(
            next, prev, &next_pyr, &prev_pyr, params, levels, target, per_dir,
        );
        ((a.0, a.1, a.2), b.0)
    };
    let level_scale = (1u32 << target) as f32;
    let mut forward = FlowField::new(prev.dims);
    let mut backward = FlowField::new(prev.dims);
    let mut confidence = vec![0.0f32; prev.dims.area() as usize];
    // One sweep writes both dense fields and the confidence map.
    fill_and_confidence(
        &offsets_f,
        &offsets_b,
        cols,
        rows,
        bs,
        level_scale,
        prev,
        &mut forward,
        &mut backward,
        &mut confidence,
        threads,
    );
    if let Some(rot) = rotation {
        add_rotation_fields(&mut forward, &mut backward, rot, threads);
    }
    Ok(FlowEstimate {
        forward,
        backward,
        confidence,
        rotation,
    })
}

pub fn estimate(prev: &LumaPlane, next: &LumaPlane, params: &FlowParams) -> Result<FlowEstimate> {
    let threads = worker_threads(prev.dims.area() as usize);
    estimate_with(prev, next, params, threads)
}

/// Mean absolute difference of two luma planes, normalised to 0..=1.
pub fn mean_abs_diff(a: &LumaPlane, b: &LumaPlane) -> f32 {
    if a.dims != b.dims {
        return 1.0;
    }
    if a.data.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for (x, y) in a.data.iter().zip(b.data.iter()) {
        acc += (x - y).abs();
    }
    (acc / a.data.len() as f32).clamp(0.0, 1.0)
}

/// Peak signal-to-noise ratio in dB; +inf for identical frames.
pub fn psnr(a: &LumaPlane, b: &LumaPlane) -> f32 {
    if a.dims != b.dims || a.data.is_empty() {
        return 0.0;
    }
    let mut mse = 0.0f64;
    for (x, y) in a.data.iter().zip(b.data.iter()) {
        let d = f64::from(*x) - f64::from(*y);
        mse += d * d;
    }
    mse /= a.data.len() as f64;
    if mse <= 0.0 {
        f32::INFINITY
    } else {
        (-10.0 * mse.log10()) as f32
    }
}

/// Hysteresis scene-cut detector driven by normalised frame difference.
#[derive(Clone, Debug)]
pub struct SceneCutDetector {
    threshold: f32,
    last_score: f32,
    cuts: u64,
}

impl SceneCutDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            last_score: 0.0,
            cuts: 0,
        }
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn set_threshold(&mut self, t: f32) {
        self.threshold = t.clamp(0.0, 1.0);
    }

    pub fn last_score(&self) -> f32 {
        self.last_score
    }

    pub fn cuts(&self) -> u64 {
        self.cuts
    }

    /// Normalised difference score, independent of resolution.
    pub fn score(&self, prev: &LumaPlane, next: &LumaPlane) -> f32 {
        let threads = worker_threads(prev.dims.area() as usize);
        self.score_with(prev, next, threads)
    }

    /// Parallel implementation of `score`. The row bands are fixed at 32 rows,
    /// so the histograms and the mean do not depend on the worker count.
    fn score_with(&self, prev: &LumaPlane, next: &LumaPlane, threads: usize) -> f32 {
        if prev.dims != next.dims || prev.dims.is_empty() {
            return 1.0;
        }
        // Coarse luma histograms catch exposure/colour jumps while the mean
        // absolute difference catches local structure changes.
        const BINS: usize = 32;
        const BAND_ROWS: usize = 32;
        let w = prev.dims.width as usize;
        let h = prev.dims.height as usize;
        let rows_per_band = BAND_ROWS.min(h).max(1);
        let band_count = h.div_ceil(rows_per_band);
        let mut bands: Vec<([f32; BINS], [f32; BINS], f64)> =
            vec![([0.0; BINS], [0.0; BINS], 0.0); band_count];
        let top = BINS as f32 - 1.0;
        parallel_bands(&mut bands, 1, threads, |chunk, first_band| {
            for (k, slot) in chunk.iter_mut().enumerate() {
                let y0 = (first_band + k) * rows_per_band;
                let y1 = (y0 + rows_per_band).min(h);
                let mut ha = [0.0f32; BINS];
                let mut hb = [0.0f32; BINS];
                let mut mad = 0.0f64;
                for y in y0..y1 {
                    for x in 0..w {
                        let i = y * w + x;
                        let a = prev.data[i];
                        let b = next.data[i];
                        ha[((a.clamp(0.0, 1.0) * top) as usize).min(BINS - 1)] += 1.0;
                        hb[((b.clamp(0.0, 1.0) * top) as usize).min(BINS - 1)] += 1.0;
                        mad += f64::from((a - b).abs());
                    }
                }
                *slot = (ha, hb, mad);
            }
        });
        let mut ha = [0.0f32; BINS];
        let mut hb = [0.0f32; BINS];
        let mut mad = 0.0f64;
        for (band_a, band_b, band_mad) in &bands {
            for i in 0..BINS {
                ha[i] += band_a[i];
                hb[i] += band_b[i];
            }
            mad += band_mad;
        }
        let inv = 1.0 / prev.data.len() as f32;
        let mut tv = 0.0f32;
        for i in 0..BINS {
            tv += (ha[i] - hb[i]).abs() * inv;
        }
        let mad = (mad / prev.data.len() as f64) as f32;
        (0.5 * (tv * 0.5) + 0.5 * mad).clamp(0.0, 1.0)
    }

    /// True when the score crosses the threshold (and bumps the cut counter).
    pub fn is_cut(&mut self, prev: &LumaPlane, next: &LumaPlane) -> bool {
        let s = self.score(prev, next);
        self.last_score = s;
        if s > self.threshold {
            self.cuts += 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(dims: Dims, shift: f32) -> LumaPlane {
        let mut p = LumaPlane::zeros(dims);
        for y in 0..dims.height {
            for x in 0..dims.width {
                p.set(x, y, ((x as f32 + shift) * 0.01).fract() * 0.8 + 0.1);
            }
        }
        p
    }

    #[test]
    fn identical_frames_have_zero_flow() {
        let d = Dims::new(64, 48);
        let a = ramp(d, 0.0);
        let est = estimate(&a, &a, &FlowParams::default()).unwrap();
        assert!(est.forward.vectors().iter().all(|v| v.len() < 0.5));
    }

    #[test]
    fn detects_scene_cut() {
        let d = Dims::new(32, 32);
        let a = LumaPlane::zeros(d);
        let b = {
            let mut p = LumaPlane::zeros(d);
            for i in 0..d.area() as u32 {
                p.data[i as usize] = 1.0;
            }
            p
        };
        let mut det = SceneCutDetector::new(0.18);
        assert!(det.is_cut(&a, &b));
        assert_eq!(det.cuts(), 1);
        assert!(!det.is_cut(&b, &b));
    }

    // Deterministic blob texture: unique across the frame so that block
    // matching is unambiguous even at the coarsest pyramid level.
    const BLOBS: [(f32, f32, f32, f32); 10] = [
        (0.16, 0.20, 0.075, 0.55),
        (0.61, 0.14, 0.060, 0.35),
        (0.85, 0.37, 0.080, 0.60),
        (0.34, 0.54, 0.070, 0.45),
        (0.72, 0.67, 0.090, 0.30),
        (0.11, 0.76, 0.065, 0.50),
        (0.49, 0.87, 0.075, 0.40),
        (0.92, 0.84, 0.050, 0.55),
        (0.27, 0.36, 0.040, 0.25),
        (0.57, 0.44, 0.035, 0.20),
    ];

    /// The blob texture as a continuous function of normalised coordinates.
    fn texture_at(uf: f32, vf: f32) -> f32 {
        let mut val = 0.12f32;
        for &(cx, cy, sigma, amp) in BLOBS.iter() {
            let du = uf - cx;
            let dv = vf - cy;
            let r2 = (du * du + dv * dv) / (2.0 * sigma * sigma);
            if r2 < 9.0 {
                val += amp * (-r2).exp();
            }
        }
        val.clamp(0.0, 1.0)
    }

    fn pattern(dims: Dims, dx: f32, dy: f32) -> LumaPlane {
        let w = dims.width as f32;
        let h = dims.height as f32;
        let mut p = LumaPlane::zeros(dims);
        for y in 0..dims.height {
            let vf = (y as f32 + 0.5 - dy) / h;
            for x in 0..dims.width {
                let uf = (x as f32 + 0.5 - dx) / w;
                p.set(x, y, texture_at(uf, vf));
            }
        }
        p
    }

    fn uniform(dims: Dims, value: f32) -> LumaPlane {
        let mut p = LumaPlane::zeros(dims);
        for v in p.data.iter_mut() {
            *v = value;
        }
        p
    }

    #[test]
    fn translation_is_recovered() {
        let d = Dims::new(96, 64);
        let prev = pattern(d, 0.0, 0.0);
        let next = pattern(d, 2.0, -1.0);
        let est = estimate(&prev, &next, &FlowParams::default()).unwrap();
        let mut n = 0.0f32;
        let (mut fx, mut fy, mut bx, mut by) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for y in 12..52 {
            for x in 12..84 {
                let f = est.forward.at(x, y);
                let b = est.backward.at(x, y);
                fx += f.x;
                fy += f.y;
                bx += b.x;
                by += b.y;
                n += 1.0;
            }
        }
        let (fx, fy, bx, by) = (fx / n, fy / n, bx / n, by / n);
        assert!((fx + 2.0).abs() < 0.5, "forward x mean {fx}");
        assert!((fy - 1.0).abs() < 0.5, "forward y mean {fy}");
        assert!((bx - 2.0).abs() < 0.5, "backward x mean {bx}");
        assert!((by + 1.0).abs() < 0.5, "backward y mean {by}");
    }

    #[test]
    fn tiny_and_odd_frames_are_supported() {
        let cases = [
            Dims::new(1, 1),
            Dims::new(2, 2),
            Dims::new(3, 5),
            Dims::new(8, 8),
        ];
        for d in cases {
            let a = pattern(d, 0.0, 0.0);
            let b = pattern(d, 1.0, 0.0);
            let est = estimate(&a, &b, &FlowParams::default()).unwrap();
            assert_eq!(est.dims(), d);
            assert_eq!(est.confidence.len(), d.area() as usize);
            assert_eq!(est.occlusion_mask(0.5).len(), d.area() as usize);
        }
    }

    #[test]
    fn mismatched_dims_are_rejected() {
        let a = LumaPlane::zeros(Dims::new(16, 16));
        let b = LumaPlane::zeros(Dims::new(16, 8));
        assert!(estimate(&a, &b, &FlowParams::default()).is_err());
        assert_eq!(mean_abs_diff(&a, &b), 1.0);
        assert_eq!(psnr(&a, &b), 0.0);
    }

    #[test]
    fn single_pyramid_level_works() {
        let d = Dims::new(40, 32);
        let prev = pattern(d, 0.0, 0.0);
        let next = pattern(d, 2.0, 0.0);
        let params = FlowParams {
            pyramid_levels: 1,
            block_size: 4,
            ..FlowParams::default()
        };
        let est = estimate(&prev, &next, &params).unwrap();
        assert_eq!(est.dims(), d);
        assert!(
            est.forward
                .vectors()
                .iter()
                .all(|v| v.x.is_finite() && v.y.is_finite())
        );
    }

    #[test]
    fn field_sampling_and_stats() {
        let mut f = FlowField::new(Dims::new(3, 2));
        for y in 0..2 {
            for x in 0..3 {
                f.set(x, y, Vec2::new(x as f32 * 2.0, y as f32 * 2.0));
            }
        }
        let s = f.sample(0.5, 0.0);
        assert!((s.x - 1.0).abs() < 1.0e-6);
        assert!((s.y - 0.0).abs() < 1.0e-6);
        let clamped = f.sample(-10.0, 10.0);
        assert!((clamped.x - 0.0).abs() < 1.0e-6);
        assert!((clamped.y - 2.0).abs() < 1.0e-6);
        let nan = f.sample(f32::NAN, f32::INFINITY);
        assert!(nan.x.is_finite() && nan.y.is_finite());
        let expect_mean = (0.0 + 2.0 + 4.0 + 2.0 + 8.0f32.sqrt() + 20.0f32.sqrt()) / 6.0;
        assert!((f.mean_len() - expect_mean).abs() < 1.0e-4);
        assert!((f.max_len() - 20.0f32.sqrt()).abs() < 1.0e-6);
        f.clear();
        assert_eq!(f.mean_len(), 0.0);
        assert_eq!(f.max_len(), 0.0);

        let empty = FlowField::new(Dims::new(0, 0));
        assert!(empty.is_empty());
        assert_eq!(empty.sample(1.0, 1.0), Vec2::ZERO);
        assert_eq!(empty.mean_len(), 0.0);
        assert_eq!(empty.max_len(), 0.0);
    }

    #[test]
    fn occlusion_mask_marks_low_confidence() {
        let d = Dims::new(4, 1);
        let field = FlowField::new(d);
        let est = FlowEstimate::new(field.clone(), field, vec![1.0, 0.05, 0.5, 0.19]);
        assert_eq!(est.occlusion_mask(0.2), vec![0, 255, 0, 255]);
        assert!(est.is_occluded(1, 0, 0.2));
        assert!(!est.is_occluded(0, 0, 0.2));
        assert_eq!(est.confidence_at(1, 0), 0.05);
        assert_eq!(est.confidence_at(9, 9), 0.0);
        assert_eq!(est.dims(), d);
    }

    #[test]
    fn difference_metrics_are_normalised() {
        let d = Dims::new(16, 16);
        let a = LumaPlane::zeros(d);
        let b = uniform(d, 0.5);
        assert!((mean_abs_diff(&a, &b) - 0.5).abs() < 1.0e-6);
        assert!((psnr(&a, &b) - 6.0206).abs() < 1.0e-3);
        assert!(psnr(&a, &a).is_infinite());
        assert_eq!(mean_abs_diff(&a, &a), 0.0);
    }

    #[test]
    fn scene_cut_detector_thresholds() {
        let d = Dims::new(32, 32);
        let a = LumaPlane::zeros(d);
        let white = uniform(d, 1.0);
        let mut det = SceneCutDetector::new(0.1);
        det.set_threshold(0.9);
        assert!((det.threshold() - 0.9).abs() < 1.0e-6);
        assert_eq!(det.score(&a, &a), 0.0);
        assert!(!det.is_cut(&a, &a));
        assert_eq!(det.cuts(), 0);
        det.set_threshold(0.05);
        assert!(det.is_cut(&a, &white));
        assert_eq!(det.cuts(), 1);
        assert!(det.last_score() > 0.05);
    }

    #[test]
    fn motion_is_not_a_scene_cut() {
        let d = Dims::new(96, 64);
        let prev = pattern(d, 0.0, 0.0);
        let next = pattern(d, 2.0, 0.0);
        let mut det = SceneCutDetector::new(0.18);
        assert!(!det.is_cut(&prev, &next));
        assert_eq!(det.cuts(), 0);
        assert!(det.last_score() < 0.18);
    }

    #[test]
    fn flat_frames_stay_at_confidence_min() {
        let d = Dims::new(32, 24);
        let a = uniform(d, 0.25);
        let b = uniform(d, 0.4);
        let est = estimate(&a, &b, &FlowParams::default()).unwrap();
        let c = est.confidence_at(16, 12);
        assert!(!est.is_occluded(16, 12, 0.2), "flat confidence {c}");
        assert!(c <= 0.2 + 1.0e-6, "flat confidence {c}");
    }

    #[test]
    fn params_from_config() {
        let c = FlowConfig::default();
        let p = FlowParams::from(&c);
        assert_eq!(p.pyramid_levels, c.pyramid_levels);
        assert_eq!(p.block_size, c.block_size);
        assert_eq!(p.search_radius, c.search_radius);
        assert_eq!(p.refine_iters, c.refine_iters);
        assert_eq!(p.confidence_min, c.confidence_min);
    }

    #[test]
    #[ignore = "performance measurement"]
    fn perf_1080p_flow() {
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            let prev = pattern(d, 0.0, 0.0);
            let next = pattern(d, 2.0, -1.0);
            let params = FlowParams::default();
            let t = std::time::Instant::now();
            let serial = estimate_with(&prev, &next, &params, 1).unwrap();
            let serial_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = std::time::Instant::now();
            let est = estimate(&prev, &next, &params).unwrap();
            let par_ms = t.elapsed().as_secs_f64() * 1000.0;
            println!(
                "flow {}x{}: serial {serial_ms:.2} ms, parallel {par_ms:.2} ms ({:.1}x), \
forward mean {:.2}px, backward mean {:.2}px",
                d.width,
                d.height,
                serial_ms / par_ms.max(0.001),
                est.forward.mean_len(),
                est.backward.mean_len()
            );
            assert_eq!(serial.forward.vectors(), est.forward.vectors());
            assert_eq!(serial.backward.vectors(), est.backward.vectors());
            assert_eq!(serial.confidence, est.confidence);
        }
    }

    #[test]
    #[ignore = "performance measurement"]
    fn perf_breakdown() {
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            println!("--- {}x{} ---", d.width, d.height);
            let prev = pattern(d, 0.0, 0.0);
            let next = pattern(d, 2.0, -1.0);
            let t = std::time::Instant::now();
            let p1 = prev.downscale2();
            let n1 = next.downscale2();
            let p2 = p1.downscale2();
            let n2 = n1.downscale2();
            println!("pyr: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
            let params = FlowParams::default();
            let sp = [p1, p2];
            let sn = [n1, n2];
            let threads = worker_threads(d.area() as usize);
            println!("workers: {threads}");
            let t = std::time::Instant::now();
            let half = reduced_resolution(d, params.block_size);
            let match_target = target_level(half, 3);
            let (of, cols, rows) =
                match_flow(&prev, &next, &sp, &sn, &params, 3, match_target, threads);
            println!(
                "match fwd: {:.2} ms ({}x{} blocks)",
                t.elapsed().as_secs_f64() * 1000.0,
                cols,
                rows
            );
            let t = std::time::Instant::now();
            let (ob, _, _) = match_flow(&next, &prev, &sn, &sp, &params, 3, match_target, threads);
            println!("match bwd: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
            let mut fwd = FlowField::new(d);
            let mut bwd = FlowField::new(d);
            let mut conf = vec![0.0f32; d.area() as usize];
            let level_scale = (1u32 << match_target) as f32;
            let t = std::time::Instant::now();
            fill_and_confidence(
                &of,
                &ob,
                cols,
                rows,
                params.block_size,
                level_scale,
                &prev,
                &mut fwd,
                &mut bwd,
                &mut conf,
                threads,
            );
            println!("fill+conf: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
            let t = std::time::Instant::now();
            let e = estimate(&prev, &next, &params).unwrap();
            println!(
                "estimate total: {:.2} ms (fwd mean {:.2})",
                t.elapsed().as_secs_f64() * 1000.0,
                e.forward.mean_len()
            );
        }
    }

    #[test]
    fn parallel_estimate_matches_serial() {
        // Mid-size frame so both the pyramid and the threaded stages engage.
        let d = Dims::new(320, 240);
        let prev = pattern(d, 0.0, 0.0);
        let next = pattern(d, 2.5, 1.5);
        let params = FlowParams::default();
        let serial = estimate_with(&prev, &next, &params, 1).unwrap();
        for tid in [2usize, 4, 8, 16] {
            let par = estimate_with(&prev, &next, &params, tid).unwrap();
            assert_eq!(serial.forward.vectors(), par.forward.vectors(), "tid {tid}");
            assert_eq!(
                serial.backward.vectors(),
                par.backward.vectors(),
                "tid {tid}"
            );
            assert_eq!(serial.confidence, par.confidence, "tid {tid}");
        }
        let par = estimate(&prev, &next, &params).unwrap();
        assert_eq!(serial.forward.vectors(), par.forward.vectors());
        assert_eq!(serial.confidence, par.confidence);
    }

    /// Straightforward sequential reference implementation of the SAD
    /// definition, used to validate the vectorised path and the clamped edge
    /// path.
    #[allow(clippy::too_many_arguments)]
    fn ref_sad(
        src: &LumaPlane,
        dst: &LumaPlane,
        x0: u32,
        y0: u32,
        bs: u32,
        dx: i32,
        dy: i32,
        step: u32,
    ) -> f32 {
        let pw = src.dims.width as i32;
        let ph = src.dims.height as i32;
        let mut sad = 0.0f32;
        let mut y = 0u32;
        while y < bs {
            let py = y0 as i32 + y as i32;
            if py >= ph {
                break;
            }
            let sy = (py + dy).clamp(0, ph - 1) as u32;
            let mut x = 0u32;
            while x < bs {
                let px = x0 as i32 + x as i32;
                if px >= pw {
                    break;
                }
                let sx = (px + dx).clamp(0, pw - 1) as u32;
                sad += (src.get(px as u32, py as u32) - dst.get(sx, sy)).abs();
                x += step;
            }
            y += step;
        }
        sad
    }

    #[test]
    fn block_sad_matches_reference() {
        for (w, h) in [(40u32, 30u32), (37u32, 29u32), (16u32, 16u32)] {
            let d = Dims::new(w, h);
            let mut src = LumaPlane::zeros(d);
            let mut dst = LumaPlane::zeros(d);
            for y in 0..h {
                for x in 0..w {
                    src.set(x, y, ((x * 7 + y * 3) % 17) as f32 / 17.0);
                    dst.set(x, y, ((x * 5 + y * 11 + 2) % 19) as f32 / 19.0);
                }
            }
            let mut bad = 0u32;
            for bx in 0..w.div_ceil(8) {
                for by in 0..h.div_ceil(8) {
                    for dx in -9i32..=9 {
                        for dy in -9i32..=9 {
                            for step in [1u32, 2u32] {
                                let a = block_sad(&src, &dst, bx * 8, by * 8, 8, dx, dy, step);
                                let b = ref_sad(&src, &dst, bx * 8, by * 8, 8, dx, dy, step);
                                if (a - b).abs() > 1.0e-3 {
                                    bad += 1;
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(bad, 0, "block_sad mismatches on {w}x{h}: {bad}");
        }
    }

    #[test]
    fn simd_sad_ranking_matches_sequential() {
        // The vector kernel folds four lane sums in a fixed order, which is not
        // the raster order of the scalar reference, so single SAD values may
        // differ by a few ULPs. The candidate ranking must not change: rerun a
        // full search window with both kernels and compare the winning offsets
        // (length tie-break included) for every block.
        let d = Dims::new(96, 72);
        let prev = pattern(d, 0.0, 0.0);
        let next = pattern(d, 1.7, -1.2);
        let mut noisy = LumaPlane::zeros(d);
        let mut rng = 0x1234_5678u32;
        for y in 0..d.height {
            for x in 0..d.width {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                noisy.set(x, y, f32::from((rng >> 16) as u8) / 255.0);
            }
        }
        let mut differences = 0u32;
        for (src, dst) in [(&prev, &next), (&noisy, &next)] {
            for radius in [2i32, 4, 8] {
                let step = if radius >= 4 { 2u32 } else { 1u32 };
                for by in 0..(d.height / 8) {
                    for bx in 0..(d.width / 8) {
                        let mut best_simd = (0i32, 0i32, f32::INFINITY);
                        let mut best_ref = (0i32, 0i32, f32::INFINITY);
                        for dy in -radius..=radius {
                            for dx in -radius..=radius {
                                let penalty = TIE_BREAK * (dx * dx + dy * dy) as f32;
                                let a =
                                    block_sad(src, dst, bx * 8, by * 8, 8, dx, dy, step) + penalty;
                                if a < best_simd.2 {
                                    best_simd = (dx, dy, a);
                                }
                                let b =
                                    ref_sad(src, dst, bx * 8, by * 8, 8, dx, dy, step) + penalty;
                                if b < best_ref.2 {
                                    best_ref = (dx, dy, b);
                                }
                            }
                        }
                        if best_simd.0 != best_ref.0 || best_simd.1 != best_ref.1 {
                            differences += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            differences, 0,
            "{differences} blocks changed their winning offset"
        );
    }

    /// Ground-truth displacement of the warped pair at a pixel: the content
    /// moves by (dx, dy), so the gather flow that recovers it is (-dx, -dy).
    fn warp_at(dims: Dims, x: u32, y: u32) -> (f32, f32) {
        let uf = (x as f32 + 0.5) / dims.width as f32;
        let vf = (y as f32 + 0.5) / dims.height as f32;
        let tau = std::f32::consts::TAU;
        let dx = 2.0 + 1.2 * (tau * vf).sin();
        let dy = -1.0 + 1.0 * (tau * uf).cos();
        (dx, dy)
    }

    /// Smooth, non-rigid synthetic pair: the second frame is the blob texture
    /// sampled at a sinusoidally displaced position.
    fn warped_pair(dims: Dims) -> (LumaPlane, LumaPlane) {
        let w = dims.width as f32;
        let h = dims.height as f32;
        let mut prev = LumaPlane::zeros(dims);
        let mut next = LumaPlane::zeros(dims);
        for y in 0..dims.height {
            for x in 0..dims.width {
                let uf = (x as f32 + 0.5) / w;
                let vf = (y as f32 + 0.5) / h;
                prev.set(x, y, texture_at(uf, vf));
                let (dx, dy) = warp_at(dims, x, y);
                next.set(x, y, texture_at(uf - dx / w, vf - dy / h));
            }
        }
        (prev, next)
    }

    #[test]
    fn reduced_resolution_policy_is_size_based() {
        assert!(reduced_resolution(Dims::new(1280, 720), 8));
        assert!(reduced_resolution(Dims::new(1920, 1080), 8));
        assert!(!reduced_resolution(Dims::new(640, 480), 8));
        assert!(!reduced_resolution(Dims::new(320, 240), 8));
        assert_eq!(
            target_level(reduced_resolution(Dims::new(1280, 720), 8), 3),
            1
        );
        assert_eq!(
            target_level(reduced_resolution(Dims::new(640, 480), 8), 3),
            0
        );
        // A one-level pyramid has no coarser level to match on.
        assert_eq!(
            target_level(reduced_resolution(Dims::new(1280, 720), 8), 1),
            0
        );
        assert!(switch_truthy("1"));
        assert!(switch_truthy(" TRUE "));
        assert!(switch_truthy("yes"));
        assert!(!switch_truthy("0"));
        assert!(!switch_truthy(""));
        assert!(!switch_truthy("off"));
    }

    #[test]
    fn reduced_resolution_keeps_flow_accuracy() {
        let d = Dims::new(1280, 720);
        let (prev, next) = warped_pair(d);
        let params = FlowParams::default();
        let full = estimate_impl(&prev, &next, &params, 1, false, None).unwrap();
        let half = estimate_impl(&prev, &next, &params, 1, true, None).unwrap();
        let (mut epe_full, mut epe_half, mut n) = (0.0f64, 0.0f64, 0u64);
        let (mut flips, mut conf_delta, mut conf_n) = (0u64, 0.0f64, 0u64);
        let (x1, y1) = (d.width - 8, d.height - 8);
        for y in 8..y1 {
            for x in 8..x1 {
                let i = (y * d.width + x) as usize;
                conf_delta += f64::from((full.confidence[i] - half.confidence[i]).abs());
                conf_n += 1;
                // Flat (untextured) regions have no recoverable motion, so
                // accuracy is only meaningful where the texture ramp gives the
                // pixel real confidence.
                let gx = (prev.get(x + 1, y) - prev.get(x - 1, y)).abs();
                let gy = (prev.get(x, y + 1) - prev.get(x, y - 1)).abs();
                if gx + gy < 0.02 {
                    continue;
                }
                let (dx, dy) = warp_at(d, x, y);
                let (tx, ty) = (-dx, -dy);
                let f = full.forward.at(x, y);
                let h = half.forward.at(x, y);
                epe_full += f64::from(((f.x - tx).powi(2) + (f.y - ty).powi(2)).sqrt());
                epe_half += f64::from(((h.x - tx).powi(2) + (h.y - ty).powi(2)).sqrt());
                if full.is_occluded(x, y, params.confidence_min)
                    != half.is_occluded(x, y, params.confidence_min)
                {
                    flips += 1;
                }
                n += 1;
            }
        }
        let (epe_full, epe_half) = (epe_full / n as f64, epe_half / n as f64);
        let flip_ratio = flips as f64 / n as f64;
        let mean_conf_delta = conf_delta / conf_n as f64;
        println!(
            "half-res quality: epe full {epe_full:.3} half {epe_half:.3} px, \
             occlusion flips {:.2}%, mean conf delta {mean_conf_delta:.4}",
            flip_ratio * 100.0
        );
        // Sub-pixel accuracy must survive the half-resolution grid.
        assert!(epe_full < 1.0, "full-res epe {epe_full:.3} px");
        assert!(
            epe_half <= epe_full * 1.05 + 0.25,
            "epe regression: full {epe_full:.3} px, half {epe_half:.3} px"
        );
        assert!(
            flip_ratio <= 0.05,
            "occlusion decisions changed on {flip_ratio:.1}% of pixels"
        );
        assert!(
            mean_conf_delta <= 0.05,
            "mean confidence delta {mean_conf_delta:.4}"
        );
    }

    #[test]
    fn reduced_resolution_estimate_is_deterministic() {
        // 1024x768 crosses the block-count threshold, so this exercises the
        // half-resolution matching path together with the threaded stages.
        let d = Dims::new(1024, 768);
        let prev = pattern(d, 0.0, 0.0);
        let next = pattern(d, 2.0, -1.0);
        let params = FlowParams::default();
        let serial = estimate_with(&prev, &next, &params, 1).unwrap();
        for tid in [2usize, 4] {
            let par = estimate_with(&prev, &next, &params, tid).unwrap();
            assert_eq!(serial.forward.vectors(), par.forward.vectors(), "tid {tid}");
            assert_eq!(
                serial.backward.vectors(),
                par.backward.vectors(),
                "tid {tid}"
            );
            assert_eq!(serial.confidence, par.confidence, "tid {tid}");
        }
        let default_path = estimate(&prev, &next, &params).unwrap();
        assert_eq!(serial.forward.vectors(), default_path.forward.vectors());
        assert_eq!(serial.confidence, default_path.confidence);
    }

    #[test]
    #[ignore = "performance measurement"]
    fn perf_half_resolution_ab() {
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            let prev = pattern(d, 0.0, 0.0);
            let next = pattern(d, 2.0, -1.0);
            let params = FlowParams::default();
            let threads = worker_threads(d.area() as usize);
            let (mut full_best, mut half_best) = (f64::INFINITY, f64::INFINITY);
            let (mut full_fwd, mut half_fwd) = (0.0f32, 0.0f32);
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let a = estimate_impl(&prev, &next, &params, threads, false, None).unwrap();
                full_best = full_best.min(t.elapsed().as_secs_f64() * 1000.0);
                full_fwd = a.forward.mean_len();
                let t = std::time::Instant::now();
                let b = estimate_impl(&prev, &next, &params, threads, true, None).unwrap();
                half_best = half_best.min(t.elapsed().as_secs_f64() * 1000.0);
                half_fwd = b.forward.mean_len();
            }
            println!(
                "{}x{} threads {threads}: full {full_best:.2} ms (fwd {full_fwd:.2}) half \
                 {half_best:.2} ms (fwd {half_fwd:.2}) speedup {:.2}x",
                d.width,
                d.height,
                full_best / half_best
            );
        }
    }

    /// A/B of the matching pyramid level: level 1 (current policy) versus a
    /// quarter-resolution grid (level 2) and an eighth-resolution one (level 3).
    /// Same-process interleaved best-of-3 so ambient load affects both equally.
    #[test]
    #[ignore]
    fn perf_flow_target_ab() {
        let params = FlowParams::default();
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            let prev = pattern(d, 0.0, 0.0);
            let next = pattern(d, 2.0, -1.0);
            for threads in [1usize, 16] {
                let mut best = [f64::MAX; 3];
                let mut means = [0.0f32; 3];
                for _ in 0..3 {
                    for (k, target) in [1usize, 2, 3].into_iter().enumerate() {
                        let t = std::time::Instant::now();
                        let e = estimate_with_target(&prev, &next, &params, threads, Some(target))
                            .unwrap();
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        if ms < best[k] {
                            best[k] = ms;
                            means[k] = e.forward.mean_len();
                        }
                    }
                }
                println!(
                    "{}x{} threads {threads}: target1 {:.2} ms (fwd {:.2}px) target2 {:.2} ms \
                     (fwd {:.2}px) target3 {:.2} ms (fwd {:.2}px)",
                    d.width, d.height, best[0], means[0], best[1], means[1], best[2], means[2]
                );
            }
        }
    }

    /// Determinism of the coarse-target path: any worker count yields identical
    /// vectors and confidence.
    #[test]
    fn coarse_target_is_deterministic() {
        let d = Dims::new(1280, 720);
        let prev = pattern(d, 0.0, 0.0);
        let next = pattern(d, 2.0, -1.0);
        let params = FlowParams::default();
        let a = estimate_with_target(&prev, &next, &params, 1, Some(2)).unwrap();
        let b = estimate_with_target(&prev, &next, &params, 4, Some(2)).unwrap();
        assert_eq!(a.forward.vectors(), b.forward.vectors());
        assert_eq!(a.backward.vectors(), b.backward.vectors());
        assert_eq!(a.confidence, b.confidence);
    }

    /// Per-stage timings of the real pipeline (same scopes as `estimate_impl`),
    /// plus the cost of empty scoped-thread spawns for reference.
    #[test]
    #[ignore]
    fn perf_stage_breakdown() {
        let params = FlowParams::default();
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            let prev = pattern(d, 0.0, 0.0);
            let next = pattern(d, 2.0, -1.0);
            let threads = worker_threads(d.area() as usize);
            for target in [1usize, 2] {
                let per_dir = (threads / 2).max(1);
                let t = std::time::Instant::now();
                let (pp, np) = std::thread::scope(|s| {
                    let a = s.spawn(|| build_pyramid(&prev, 2, per_dir));
                    let b = s.spawn(|| build_pyramid(&next, 2, per_dir));
                    (a.join().unwrap(), b.join().unwrap())
                });
                let pyr = t.elapsed().as_secs_f64() * 1000.0;
                let t = std::time::Instant::now();
                let (f, b) = std::thread::scope(|s| {
                    let a =
                        s.spawn(|| match_flow(&prev, &next, &pp, &np, &params, 3, target, per_dir));
                    let c =
                        s.spawn(|| match_flow(&next, &prev, &np, &pp, &params, 3, target, per_dir));
                    (a.join().unwrap(), c.join().unwrap())
                });
                let matched = t.elapsed().as_secs_f64() * 1000.0;
                let mut fwd = FlowField::new(d);
                let mut bwd = FlowField::new(d);
                let mut conf = vec![0.0f32; d.area() as usize];
                let level_scale = (1u32 << target) as f32;
                let t = std::time::Instant::now();
                fill_and_confidence(
                    &f.0,
                    &b.0,
                    f.1,
                    f.2,
                    params.block_size,
                    level_scale,
                    &prev,
                    &mut fwd,
                    &mut bwd,
                    &mut conf,
                    threads,
                );
                let fill = t.elapsed().as_secs_f64() * 1000.0;
                let t = std::time::Instant::now();
                let e = estimate_with_target(&prev, &next, &params, threads, Some(target)).unwrap();
                let total = t.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "{}x{} target {target} threads {threads}: pyr {pyr:.2} match {matched:.2} \
                     fill+conf {fill:.2} sum {:.2} total {total:.2} (fwd {:.2}px, {} conf)",
                    d.width,
                    d.height,
                    pyr + matched + fill,
                    e.forward.mean_len(),
                    conf.len()
                );
            }
        }
        let t = std::time::Instant::now();
        for _ in 0..8 {
            std::thread::scope(|s| {
                for _ in 0..8 {
                    s.spawn(|| {});
                }
            });
        }
        println!(
            "spawn overhead: 64 empty scoped threads = {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }

    /// Pair related by an exact global rotation about the frame centre:
    /// next(p) = prev(c + R(-angle) * (p + 0.5 - c) - 0.5).
    fn rotated_pair(dims: Dims, angle: f32) -> (LumaPlane, LumaPlane, GlobalRotation) {
        let prev = pattern(dims, 0.0, 0.0);
        let (cx, cy) = (dims.width as f32 * 0.5, dims.height as f32 * 0.5);
        let (sa, ca) = angle.sin_cos();
        let mut next = LumaPlane::zeros(dims);
        for y in 0..dims.height {
            let vy = y as f32 + 0.5 - cy;
            for x in 0..dims.width {
                let vx = x as f32 + 0.5 - cx;
                let sx = cx + ca * vx + sa * vy - 0.5;
                let sy = cy - sa * vx + ca * vy - 0.5;
                next.set(x, y, prev.sample_bilinear(sx, sy));
            }
        }
        (prev, next, GlobalRotation::new(angle, cx, cy))
    }

    #[test]
    fn global_rotation_is_detected_and_compensated() {
        let d = Dims::new(640, 360);
        let angle = 3.0f32.to_radians();
        let (prev, next, truth) = rotated_pair(d, angle);
        let params = FlowParams::default();
        let est = estimate_with(&prev, &next, &params, 1).unwrap();
        let rot = est.rotation.expect("rotation should be detected");
        assert!(
            (rot.angle - angle).abs() < 0.5f32.to_radians(),
            "detected angle {} rad, truth {angle}",
            rot.angle
        );
        assert!((rot.center_x - truth.center_x).abs() < 1.5);
        assert!((rot.center_y - truth.center_y).abs() < 1.5);
        // The gather field must follow the rotation arc rather than a constant
        // displacement: compare against the analytic rotation displacement.
        let mut err = 0.0f64;
        let mut n = 0u64;
        for y in 16..(d.height - 16) {
            for x in 16..(d.width - 16) {
                let f = est.forward.at(x, y);
                let t = truth.forward_at(x as f32, y as f32);
                err += f64::from(((f.x - t.x).powi(2) + (f.y - t.y).powi(2)).sqrt());
                n += 1;
            }
        }
        let mean = err / n as f64;
        assert!(mean < 0.8, "mean flow error {mean:.3} px");
        // The detected angle and the fields must not depend on the worker count.
        for tid in [2usize, 4] {
            let par = estimate_with(&prev, &next, &params, tid).unwrap();
            assert_eq!(est.forward.vectors(), par.forward.vectors(), "tid {tid}");
            assert_eq!(est.backward.vectors(), par.backward.vectors(), "tid {tid}");
            assert_eq!(est.confidence, par.confidence, "tid {tid}");
            assert_eq!(est.rotation, par.rotation, "tid {tid}");
        }
    }

    #[test]
    fn translation_is_not_taken_for_rotation() {
        let d = Dims::new(640, 360);
        let prev = pattern(d, 0.0, 0.0);
        let next = pattern(d, 2.0, -1.0);
        let est = estimate(&prev, &next, &FlowParams::default()).unwrap();
        assert!(
            est.rotation.is_none(),
            "translation detected as rotation: {:?}",
            est.rotation
        );
    }

    #[test]
    fn small_frames_skip_rotation() {
        // Below MIN_ROTATION_PIXELS the estimator keeps the translation-only
        // path even for a rotation pair.
        let d = Dims::new(320, 240);
        let angle = 3.0f32.to_radians();
        let (prev, next, _) = rotated_pair(d, angle);
        let est = estimate(&prev, &next, &FlowParams::default()).unwrap();
        assert!(est.rotation.is_none());
    }

    /// Stage timings of the rotation path: angle detection, full-resolution
    /// derotation and the total estimate, against a translation pair as the
    /// reference cost.
    #[test]
    #[ignore]
    fn perf_rotation_stages() {
        let params = FlowParams::default();
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            let angle = 3.0f32.to_radians();
            let (prev, next, _) = rotated_pair(d, angle);
            let threads = worker_threads(d.area() as usize);
            let extra = 2usize;
            let (pp, np) = std::thread::scope(|s| {
                let a = s.spawn(|| build_pyramid(&prev, extra, threads / 2));
                let b = s.spawn(|| build_pyramid(&next, extra, threads / 2));
                (a.join().unwrap(), b.join().unwrap())
            });
            let coarse = extra - 1;
            let mut best_detect = f64::INFINITY;
            let mut best_derot = f64::INFINITY;
            let mut best_pyr = f64::INFINITY;
            let mut best_total = f64::INFINITY;
            let mut detected = 0.0f32;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let r = detect_global_rotation(
                    &pp[coarse],
                    &np[coarse],
                    (1u32 << extra) as f32,
                    d,
                    threads,
                );
                best_detect = best_detect.min(t.elapsed().as_secs_f64() * 1000.0);
                let rot = r.expect("rotation should be detected");
                detected = rot.angle.to_degrees();
                let t = std::time::Instant::now();
                let der = derotate_plane(&next, rot, threads);
                best_derot = best_derot.min(t.elapsed().as_secs_f64() * 1000.0);
                let t = std::time::Instant::now();
                let dp = build_pyramid(&der, extra, threads);
                best_pyr = best_pyr.min(t.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(dp.len(), extra);
                let t = std::time::Instant::now();
                let e = estimate_with(&prev, &next, &params, threads).unwrap();
                best_total = best_total.min(t.elapsed().as_secs_f64() * 1000.0);
                assert!(e.rotation.is_some());
            }
            let (tp, tn) = (pattern(d, 0.0, 0.0), pattern(d, 2.0, -1.0));
            let mut best_plain = f64::INFINITY;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let _ = estimate_with(&tp, &tn, &params, threads).unwrap();
                best_plain = best_plain.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            println!(
                "{}x{} threads {threads}: detect {best_detect:.2} ms derotate {best_derot:.2} ms \
                 extra pyramid {best_pyr:.2} ms -> estimate {best_total:.2} ms \
                 (translation pair {best_plain:.2} ms, detected {detected:.2} deg)",
                d.width, d.height
            );
        }
    }

    /// Cost of the global-rotation fit on a plain translation pair (which pays
    /// the overhead without any gain) and on a rotating pair, best of five.
    #[test]
    #[ignore]
    fn perf_rotation_detect() {
        let threads = 8usize;
        let extra = 2usize;
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            let prev = pattern(d, 0.0, 0.0);
            let next = pattern(d, 2.0, -1.0);
            let (pp, np) = std::thread::scope(|s| {
                let a = s.spawn(|| build_pyramid(&prev, extra, threads));
                let b = s.spawn(|| build_pyramid(&next, extra, threads));
                (a.join().unwrap(), b.join().unwrap())
            });
            let (rp, rn, _) = rotated_pair(d, 3.0f32.to_radians());
            let (rpp, rnp) = std::thread::scope(|s| {
                let a = s.spawn(|| build_pyramid(&rp, extra, threads));
                let b = s.spawn(|| build_pyramid(&rn, extra, threads));
                (a.join().unwrap(), b.join().unwrap())
            });
            let coarse = extra - 1;
            let plane_scale = (1u32 << extra) as f32;
            let mut translation = f64::INFINITY;
            let mut rotating = f64::INFINITY;
            let mut rejected = false;
            let mut angle = f32::NAN;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let r = detect_global_rotation(&pp[coarse], &np[coarse], plane_scale, d, threads);
                translation = translation.min(t.elapsed().as_secs_f64() * 1000.0);
                rejected = r.is_none();
                let t = std::time::Instant::now();
                let r = detect_global_rotation(&rpp[coarse], &rnp[coarse], plane_scale, d, threads);
                rotating = rotating.min(t.elapsed().as_secs_f64() * 1000.0);
                angle = r.map(|x| x.angle.to_degrees()).unwrap_or(f32::NAN);
            }
            println!(
                "{}x{} translation pair {translation:.2} ms (rejected {rejected}), \
                 rotation pair {rotating:.2} ms (found {angle:.2} deg)",
                d.width, d.height
            );
        }
    }
}
