//! MJPEG AVI reader: the input side of the ffmpeg-free video path.
//!
//! AVI is a RIFF container: a leading RIFF chunk whose form type is AVI,
//! optionally followed by header lists and stream lists, one movi list that
//! interleaves the per-frame chunks, and an optional idx1 index at the end.
//! An MJPEG AVI stores one JPEG image per video chunk, usually tagged as
//! 00dc; the stream header (strh) gives the frame rate as dwRate divided by
//! dwScale and the stream format header (strf) is a BITMAPINFOHEADER whose
//! biCompression field must be MJPG.
//!
//! This reader is the mirror image of AviWriter in avi.rs and keeps the same
//! no-dependency attitude: it parses the container itself with plain byte
//! reads and decodes each frame with the image crate. The parse is done once
//! in open, which walks the top level chunks, records the absolute position of
//! the movi list data, and either records the idx1 chunk offsets (relative to
//! the movi fourcc, exactly as ffmpeg writes them) or falls back to walking
//! the movi chunks sequentially. Frames are then read on demand: seek to the
//! chunk, read the JPEG payload into a reusable scratch buffer, decode it to
//! Rgb8, and hand out a normal fluid Frame with pts and index filled in.
//!
//! Unknown top level chunks (JUNK, INFO, odml, AVIX, ...), unknown stream
//! headers (audio streams), chunk padding to even boundaries and a missing
//! idx1 are all tolerated; a truncated payload, a chunk that does not start
//! where the index says it does, or a non-MJPEG video stream produce a
//! FluidError that names the failing byte offset (or the unsupported fourcc)
//! instead of panicking. Like every file source in this crate the reader is
//! blocking std::fs IO and one instance per thread.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use fluid_core::{Dims, FluidError, Frame, FrameSource, PixelFormat, Result};

/// Bytes of the outer RIFF header: RIFF, size, AVI form type.
const RIFF_HEADER_LEN: u64 = 12;
/// Size of a chunk header: fourcc plus little endian u32 payload size.
const CHUNK_HEADER_LEN: u64 = 8;
/// BITMAPINFOHEADER fields we need start at this offset inside a strf chunk.
const BITMAPINFOHEADER_MIN_LEN: usize = 20;
/// AVI stream header is 56 bytes, but some writers store a little more.
const STREAM_HEADER_MIN_LEN: usize = 56;
/// avih needs dwMicroSecPerFrame and dwTotalFrames, both inside 40 bytes.
const MAIN_HEADER_MIN_LEN: usize = 40;
/// Fourcc of an MJPEG stream, the only codec this reader accepts.
const MJPG_FOURCC: [u8; 4] = *b"MJPG";

/// A MJPEG AVI file opened for sequential frame reading.
pub struct AviReader {
    inner: BufReader<File>,
    dims: Dims,
    fps_num: u32,
    fps_den: u32,
    /// Total file length captured at open, used to bound every read so a
    /// truncated file becomes a clean InvalidFormat instead of an IO error.
    file_len: u64,
    /// Absolute file offset of the fourcc that starts the movi list payload.
    /// idx1 chunk offsets are relative to this position (ffmpeg convention),
    /// so the first frame chunk is normally at offset 4.
    movi_fourcc_pos: u64,
    /// Absolute offset of the first byte after the movi list payload.
    movi_data_end: u64,
    /// Video chunk offsets from idx1, relative to movi_fourcc_pos. Empty when
    /// the file has no usable index and frames are found by walking chunks.
    index: Vec<u32>,
    /// Frame count from the stream or main header, when the file states it.
    frame_count: Option<u64>,
    /// Number of frames handed out so far; doubles as the next index entry.
    next_index: u64,
    /// Absolute offset of the next chunk header in sequential mode.
    cursor: u64,
    /// Sticky end-of-stream flag so repeated next_frame calls stay cheap.
    finished: bool,
    /// Reusable JPEG payload buffer, so reading a clip does not allocate a
    /// fresh vector for every frame.
    frame_buf: Vec<u8>,
    /// Reusable RIFF chunk header buffer.
    head_buf: [u8; 8],
}

impl AviReader {
    /// Open an AVI file and parse its container headers.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| FluidError::format(format!("cannot open {}: {e}", path.display())))?;
        let file_len = file
            .metadata()
            .map_err(|e| FluidError::format(format!("cannot inspect {}: {e}", path.display())))?
            .len();
        let mut inner = BufReader::with_capacity(64 * 1024, file);

        if file_len < RIFF_HEADER_LEN {
            return Err(FluidError::format(format!(
                "not an AVI: {} is only {file_len} bytes",
                path.display()
            )));
        }
        let mut riff = [0u8; RIFF_HEADER_LEN as usize];
        read_at(&mut inner, file_len, 0, &mut riff)?;
        if &riff[0..4] != b"RIFF" || &riff[8..12] != b"AVI " {
            return Err(FluidError::format(format!(
                "not an AVI: {} does not start with RIFF/AVI magic",
                path.display()
            )));
        }

        let mut hdrl: Option<HdrlInfo> = None;
        let mut index: Vec<u32> = Vec::new();
        let mut movi: Option<MoviInfo> = None;
        let mut pos = RIFF_HEADER_LEN;
        while pos + CHUNK_HEADER_LEN <= file_len {
            let (tag, size) = read_chunk_header(&mut inner, file_len, pos)?;
            let data = pos + CHUNK_HEADER_LEN;
            if &tag == b"LIST" {
                if size < 4 {
                    return Err(FluidError::format(format!(
                        "invalid AVI: LIST chunk at byte offset {pos} is smaller than its list type"
                    )));
                }
                let mut list_type = [0u8; 4];
                read_at(&mut inner, file_len, data, &mut list_type)?;
                let list_data = data + 4;
                let list_end = data + size;
                if &list_type == b"hdrl" && hdrl.is_none() {
                    let end = list_end.min(file_len);
                    hdrl = Some(parse_hdrl(&mut inner, file_len, list_data, end)?);
                } else if &list_type == b"movi" && movi.is_none() {
                    movi = Some(MoviInfo {
                        fourcc_pos: data,
                        data_start: list_data,
                        data_end: if size == 0 {
                            file_len
                        } else {
                            list_end.min(file_len)
                        },
                    });
                }
                // INFO, odml, AVIX and any other list are irrelevant here.
            } else if &tag == b"idx1" && index.is_empty() {
                index = parse_idx1(&mut inner, file_len, data, size)?;
            }
            let next = data + size + (size & 1);
            if next <= pos {
                return Err(FluidError::format(format!(
                    "invalid AVI: chunk at byte offset {pos} has a bogus size {size}"
                )));
            }
            if next > file_len {
                // Streaming writers sometimes stop mid chunk; everything we
                // already parsed is still usable, so tolerate the tail.
                break;
            }
            pos = next;
        }

        let hdrl = hdrl.ok_or_else(|| {
            FluidError::format("invalid AVI: missing the hdrl header list".to_string())
        })?;
        let video = hdrl.video.ok_or_else(|| {
            FluidError::format("invalid AVI: no MJPEG video stream found".to_string())
        })?;
        let movi = movi.ok_or_else(|| {
            FluidError::format("invalid AVI: missing the movi frame list".to_string())
        })?;

        let width = video.width;
        let height = video.height;
        if width <= 0 || height <= 0 {
            return Err(FluidError::format(format!(
                "invalid AVI: stream dimensions are {width}x{height}"
            )));
        }
        let dims = Dims::new(width as u32, height as u32);

        let (fps_num, fps_den) = if video.rate > 0 && video.scale > 0 {
            (video.rate, video.scale)
        } else if let Some(micros) = hdrl.micros.filter(|m| *m > 0) {
            crate::rational_fps(1_000_000.0 / micros as f32)
        } else {
            (0, 1)
        };

        let frame_count = if video.length > 0 {
            Some(u64::from(video.length))
        } else if let Some(total) = hdrl.total_frames.filter(|t| *t > 0) {
            Some(u64::from(total))
        } else if !index.is_empty() {
            Some(index.len() as u64)
        } else {
            None
        };

        Ok(Self {
            inner,
            dims,
            fps_num,
            fps_den,
            file_len,
            movi_fourcc_pos: movi.fourcc_pos,
            movi_data_end: movi.data_end,
            index,
            frame_count,
            next_index: 0,
            cursor: movi.data_start,
            finished: false,
            frame_buf: Vec::new(),
            head_buf: [0u8; 8],
        })
    }

    /// Frame dimensions from the BITMAPINFOHEADER.
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Frame rate from the stream header; 0.0 when the file does not say.
    pub fn fps(&self) -> f32 {
        if self.fps_den == 0 {
            0.0
        } else {
            self.fps_num as f32 / self.fps_den as f32
        }
    }

    /// Frame count when the file states one (stream or main header), else 0.
    /// Use frame_count when the difference between unknown and empty matters.
    pub fn frames(&self) -> u64 {
        self.frame_count.unwrap_or(0)
    }

    /// Exact frame count from the headers or the index, if present.
    pub fn frame_count(&self) -> Option<u64> {
        self.frame_count
    }

    /// Name of the backend, for reports and logs.
    pub fn backend_name(&self) -> &'static str {
        "mjpeg-avi"
    }

    /// Offset of the next video chunk header, or None at end of stream.
    fn next_chunk_pos(&mut self) -> Result<Option<u64>> {
        if self.finished {
            return Ok(None);
        }
        if !self.index.is_empty() {
            let Some(offset) = self.index.get(self.next_index as usize).copied() else {
                self.finished = true;
                return Ok(None);
            };
            if offset < 4 {
                return Err(FluidError::format(format!(
                    "invalid AVI: idx1 entry {} has offset {offset}, which is inside the movi fourcc",
                    self.next_index
                )));
            }
            let pos = self.movi_fourcc_pos + u64::from(offset);
            if pos + CHUNK_HEADER_LEN > self.movi_data_end {
                return Err(FluidError::format(format!(
                    "invalid AVI: idx1 entry {} points past the movi list at byte offset {pos}",
                    self.next_index
                )));
            }
            return Ok(Some(pos));
        }

        loop {
            if self.cursor + CHUNK_HEADER_LEN > self.movi_data_end {
                self.finished = true;
                return Ok(None);
            }
            let (tag, size) = read_chunk_header(&mut self.inner, self.file_len, self.cursor)?;
            if is_video_chunk(&tag) {
                return Ok(Some(self.cursor));
            }
            let next = self.cursor + CHUNK_HEADER_LEN + size + (size & 1);
            if next <= self.cursor || next > self.movi_data_end {
                return Err(FluidError::format(format!(
                    "invalid AVI: non-video chunk at byte offset {} has a size that overflows the movi list",
                    self.cursor
                )));
            }
            self.cursor = next;
        }
    }

    /// Presentation timestamp for frame n, as integer nanoseconds scaled by
    /// the rational frame rate. Integer math keeps 29.97 frame times exact.
    fn pts_ns(&self, n: u64) -> i64 {
        if self.fps_num == 0 || self.fps_den == 0 {
            return 0;
        }
        let ns =
            u128::from(n) * 1_000_000_000u128 * u128::from(self.fps_den) / u128::from(self.fps_num);
        i64::try_from(ns).unwrap_or(i64::MAX)
    }

    /// Decode the video chunk whose header starts at pos.
    fn decode_chunk(&mut self, pos: u64) -> Result<Frame> {
        read_at(&mut self.inner, self.file_len, pos, &mut self.head_buf)?;
        let tag = [
            self.head_buf[0],
            self.head_buf[1],
            self.head_buf[2],
            self.head_buf[3],
        ];
        let size = u32::from_le_bytes([
            self.head_buf[4],
            self.head_buf[5],
            self.head_buf[6],
            self.head_buf[7],
        ]);
        if !is_video_chunk(&tag) {
            return Err(FluidError::format(format!(
                "invalid AVI: expected a video chunk (00dc) at byte offset {pos}, found {:?}",
                fourcc_to_string(&tag)
            )));
        }
        if size == 0 {
            return Err(FluidError::format(format!(
                "invalid AVI: empty JPEG payload at byte offset {pos}"
            )));
        }
        let payload = u64::from(size);
        let available = self.file_len.saturating_sub(pos + CHUNK_HEADER_LEN);
        if payload > available {
            return Err(FluidError::format(format!(
                "invalid AVI: frame at byte offset {pos} claims {payload} bytes but only {available} remain in the file"
            )));
        }

        let len = payload as usize;
        self.frame_buf.resize(len, 0);
        read_at(
            &mut self.inner,
            self.file_len,
            pos + CHUNK_HEADER_LEN,
            &mut self.frame_buf,
        )?;
        let image = image::load_from_memory(&self.frame_buf).map_err(|e| {
            FluidError::format(format!(
                "invalid AVI: cannot decode JPEG frame at byte offset {pos}: {e}"
            ))
        })?;
        let rgb = image.to_rgb8();
        if rgb.width() != self.dims.width || rgb.height() != self.dims.height {
            return Err(FluidError::format(format!(
                "invalid AVI: frame at byte offset {pos} is {}x{} but the header says {}",
                rgb.width(),
                rgb.height(),
                self.dims
            )));
        }
        let frame = Frame::new(self.dims, PixelFormat::Rgb8, rgb.into_raw()).map_err(|e| {
            FluidError::format(format!(
                "invalid AVI: frame at byte offset {pos} does not fit a {}x{} RGB frame: {e}",
                self.dims.width, self.dims.height
            ))
        })?;
        let index = self.next_index;
        let frame = frame.with_pts(self.pts_ns(index)).with_index(index);
        self.next_index += 1;
        if self.index.is_empty() {
            self.cursor = pos + CHUNK_HEADER_LEN + payload + (payload & 1);
        }
        Ok(frame)
    }
}

impl FrameSource for AviReader {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn fps(&self) -> f32 {
        AviReader::fps(self)
    }

    fn backend_name(&self) -> &'static str {
        AviReader::backend_name(self)
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        match self.next_chunk_pos()? {
            Some(pos) => self.decode_chunk(pos).map(Some),
            None => Ok(None),
        }
    }

    fn close(&mut self) {
        // std::fs has no buffered writer side to flush; drop the pointer state
        // so a late next_frame call cannot walk past the end.
        self.finished = true;
    }
}

/// Position bookkeeping for the movi list.
struct MoviInfo {
    fourcc_pos: u64,
    data_start: u64,
    data_end: u64,
}

/// Header information from the hdrl list.
#[derive(Default)]
struct HdrlInfo {
    video: Option<VideoStream>,
    micros: Option<u32>,
    total_frames: Option<u32>,
}

/// The fields of one video stream that the reader needs.
struct VideoStream {
    compression: [u8; 4],
    width: i32,
    height: i32,
    rate: u32,
    scale: u32,
    length: u32,
}

/// Read exactly buf.len() bytes at an absolute file offset.
fn read_at(inner: &mut BufReader<File>, file_len: u64, pos: u64, buf: &mut [u8]) -> Result<()> {
    let end = pos.saturating_add(buf.len() as u64);
    if end > file_len {
        return Err(FluidError::format(format!(
            "invalid AVI: truncated file, {} bytes needed at byte offset {pos} but the file is {file_len} bytes",
            buf.len()
        )));
    }
    inner.seek(SeekFrom::Start(pos)).map_err(|e| {
        FluidError::format(format!(
            "invalid AVI: cannot seek to byte offset {pos}: {e}"
        ))
    })?;
    inner.read_exact(buf).map_err(|e| {
        FluidError::format(format!(
            "invalid AVI: cannot read {} bytes at byte offset {pos}: {e}",
            buf.len()
        ))
    })
}

/// Read an 8-byte chunk header at pos.
fn read_chunk_header(
    inner: &mut BufReader<File>,
    file_len: u64,
    pos: u64,
) -> Result<([u8; 4], u64)> {
    let mut head = [0u8; 8];
    read_at(inner, file_len, pos, &mut head)?;
    let tag = [head[0], head[1], head[2], head[3]];
    let size = u64::from(u32::from_le_bytes([head[4], head[5], head[6], head[7]]));
    Ok((tag, size))
}

/// True for the chunk tags that carry a JPEG frame: 00dc and its numbered
/// variants, plus 00db from older Video for Windows encoders.
fn is_video_chunk(tag: &[u8; 4]) -> bool {
    &tag[2..] == b"dc" || &tag[2..] == b"db"
}

/// Printable fourcc for error messages.
fn fourcc_to_string(tag: &[u8; 4]) -> String {
    String::from_utf8_lossy(tag)
        .trim_end_matches(['\0', ' '])
        .to_string()
}

/// Parse the hdrl list between start and end (absolute offsets).
fn parse_hdrl(
    inner: &mut BufReader<File>,
    file_len: u64,
    start: u64,
    end: u64,
) -> Result<HdrlInfo> {
    let mut info = HdrlInfo::default();
    let mut unsupported: Option<[u8; 4]> = None;
    let mut pos = start;
    while pos + CHUNK_HEADER_LEN <= end {
        let (tag, size) = read_chunk_header(inner, file_len, pos)?;
        let data = pos + CHUNK_HEADER_LEN;
        let data_end = data + size;
        if data_end > end {
            break;
        }
        if &tag == b"avih" && size >= MAIN_HEADER_MIN_LEN as u64 {
            let mut avih = [0u8; MAIN_HEADER_MIN_LEN];
            read_at(inner, file_len, data, &mut avih)?;
            info.micros = u32_at(&avih, 0);
            info.total_frames = u32_at(&avih, 16);
        } else if &tag == b"LIST" && size >= 4 {
            let mut list_type = [0u8; 4];
            read_at(inner, file_len, data, &mut list_type)?;
            if &list_type == b"strl"
                && let Some(stream) = parse_strl(inner, file_len, data + 4, data_end)?
            {
                if stream.compression == MJPG_FOURCC {
                    if info.video.is_none() {
                        info.video = Some(stream);
                    }
                } else if unsupported.is_none() {
                    unsupported = Some(stream.compression);
                }
            }
        }
        let next = data_end + (size & 1);
        if next <= pos {
            return Err(FluidError::format(format!(
                "invalid AVI: hdrl chunk at byte offset {pos} has a bogus size {size}"
            )));
        }
        pos = next;
    }
    if info.video.is_none()
        && let Some(fourcc) = unsupported
    {
        return Err(FluidError::unsupported(format!(
            "AVI video stream uses the unsupported codec {:?}; only MJPEG (MJPG) is supported",
            fourcc_to_string(&fourcc)
        )));
    }
    Ok(info)
}

/// Parse one strl list; returns None for non-video streams (audio, text, ...).
fn parse_strl(
    inner: &mut BufReader<File>,
    file_len: u64,
    start: u64,
    end: u64,
) -> Result<Option<VideoStream>> {
    let mut stream_header: Option<[u8; STREAM_HEADER_MIN_LEN]> = None;
    let mut stream_format: Option<Vec<u8>> = None;
    let mut pos = start;
    while pos + CHUNK_HEADER_LEN <= end {
        let (tag, size) = read_chunk_header(inner, file_len, pos)?;
        let data = pos + CHUNK_HEADER_LEN;
        let data_end = data + size;
        if data_end > end {
            break;
        }
        if &tag == b"strh" && size >= STREAM_HEADER_MIN_LEN as u64 {
            let mut strh = [0u8; STREAM_HEADER_MIN_LEN];
            read_at(inner, file_len, data, &mut strh)?;
            stream_header = Some(strh);
        } else if &tag == b"strf" && size >= BITMAPINFOHEADER_MIN_LEN as u64 {
            let mut strf = vec![0u8; size as usize];
            read_at(inner, file_len, data, &mut strf)?;
            if stream_format.is_none() {
                stream_format = Some(strf);
            }
        }
        let next = data_end + (size & 1);
        if next <= pos {
            return Err(FluidError::format(format!(
                "invalid AVI: strl chunk at byte offset {pos} has a bogus size {size}"
            )));
        }
        pos = next;
    }

    let Some(strh) = stream_header else {
        return Ok(None);
    };
    let Some(strf) = stream_format else {
        return Ok(None);
    };
    let stream_type = [strh[0], strh[1], strh[2], strh[3]];
    if &stream_type != b"vids" {
        return Ok(None);
    }
    let compression = [strf[16], strf[17], strf[18], strf[19]];
    let (Some(width), Some(height)) = (i32_at(&strf, 4), i32_at(&strf, 8)) else {
        return Ok(None);
    };
    let rate = u32_at(&strh, 24).unwrap_or(0);
    let scale = u32_at(&strh, 20).unwrap_or(0);
    let length = u32_at(&strh, 32).unwrap_or(0);
    Ok(Some(VideoStream {
        compression,
        width,
        height: height.unsigned_abs() as i32,
        rate,
        scale,
        length,
    }))
}

/// Parse the idx1 chunk payload into video chunk offsets relative to the movi
/// fourcc. Entries for audio chunks are skipped; a truncated index is read as
/// far as it goes instead of failing the whole file.
fn parse_idx1(
    inner: &mut BufReader<File>,
    file_len: u64,
    data: u64,
    size: u64,
) -> Result<Vec<u32>> {
    let available = size.min(file_len.saturating_sub(data));
    let entries = (available / 16) as usize;
    if entries == 0 {
        return Ok(Vec::new());
    }
    let mut buf = vec![0u8; entries * 16];
    read_at(inner, file_len, data, &mut buf)?;
    let mut offsets = Vec::with_capacity(entries);
    for entry in buf.as_chunks::<16>().0 {
        let tag = [entry[0], entry[1], entry[2], entry[3]];
        if !is_video_chunk(&tag) {
            continue;
        }
        let Some(offset) = u32_at(entry, 8) else {
            continue;
        };
        offsets.push(offset);
    }
    Ok(offsets)
}

/// Bounds checked little endian readers used by the header parsers.
fn u32_at(bytes: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    let slice = bytes.get(off..end)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn i32_at(bytes: &[u8], off: usize) -> Option<i32> {
    Some(u32_at(bytes, off)? as i32)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    use fluid_core::{Dims, FluidError, Frame, FrameSink, FrameSource};

    use crate::avi::{AviHeader, AviWriter};
    use crate::{AviReader, synthetic_pattern};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    /// Temporary directory helper, removed on drop.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let n = NEXT_DIR.fetch_add(1, AtomicOrdering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("fluid-avi-read-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).expect("temp dir");
            Self { path }
        }

        fn join(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_clip(path: &Path, dims: Dims, fps: f32, quality: u8, count: u64) -> Vec<Frame> {
        let frames: Vec<Frame> = (0..count)
            .map(|i| synthetic_pattern(dims, i as f32 / fps, i))
            .collect();
        let mut writer =
            AviWriter::create(path, AviHeader::new(dims, fps, quality)).expect("create");
        for frame in &frames {
            writer.present(frame).expect("present");
        }
        writer.finish().expect("finish");
        frames
    }

    fn mean_abs_diff(a: &Frame, b: &Frame) -> f64 {
        assert_eq!(a.dims, b.dims);
        let sum: u64 = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| u64::from(x.abs_diff(*y)))
            .sum();
        sum as f64 / a.data.len().max(1) as f64
    }

    fn read_bytes(path: &Path) -> Vec<u8> {
        fs::read(path).expect("read")
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).expect("write");
    }

    fn find_last(bytes: &[u8], needle: &[u8; 4]) -> Option<usize> {
        bytes.windows(4).rposition(|w| w == needle)
    }

    #[test]
    fn avi_reader_roundtrips_odd_dimensions() {
        let dir = TempDir::new();
        let path = dir.join("odd.avi");
        let dims = Dims::new(127, 73);
        let frames = write_clip(&path, dims, 29.97, 95, 4);

        let mut reader = AviReader::open(&path).expect("open");
        assert_eq!(reader.dims(), dims);
        assert!((reader.fps() - 29.97).abs() < 0.01, "fps {}", reader.fps());
        assert_eq!(reader.frames(), 4);
        assert_eq!(reader.frame_count(), Some(4));
        assert_eq!(FrameSource::backend_name(&reader), "mjpeg-avi");

        for (i, expected) in frames.iter().enumerate() {
            let got = reader
                .next_frame()
                .expect("read")
                .expect("frame should exist");
            assert_eq!(got.dims, dims);
            assert_eq!(got.index, i as u64);
            let expected_pts = (i as i128 * 1_000_000_000i128 * 1001 / 30000) as i64;
            assert!(
                (got.pts_ns - expected_pts).abs() < 1_000,
                "pts {} vs {}",
                got.pts_ns,
                expected_pts
            );
            let diff = mean_abs_diff(&got, expected);
            assert!(diff <= 6.0, "frame {i} mean abs diff {diff}");
        }
        assert!(reader.next_frame().expect("eof").is_none());
        assert!(reader.next_frame().expect("sticky eof").is_none());
    }

    #[test]
    fn avi_reader_reports_integer_fps_and_empty_clip() {
        let dir = TempDir::new();
        let path = dir.join("sixty.avi");
        let dims = Dims::new(64, 48);
        write_clip(&path, dims, 60.0, 90, 3);
        let reader = AviReader::open(&path).expect("open");
        assert!((reader.fps() - 60.0).abs() < 0.001);
        assert_eq!(reader.frames(), 3);

        let empty = dir.join("empty.avi");
        let _ = write_clip(&empty, dims, 30.0, 90, 0);
        let mut reader = AviReader::open(&empty).expect("open empty");
        assert_eq!(reader.frames(), 0);
        assert_eq!(reader.frame_count(), None);
        assert!(reader.next_frame().expect("eof").is_none());
        assert!(reader.next_frame().expect("sticky eof").is_none());
    }

    #[test]
    fn avi_reader_handles_missing_idx1() {
        let dir = TempDir::new();
        let path = dir.join("no-index.avi");
        let dims = Dims::new(96, 64);
        write_clip(&path, dims, 30.0, 92, 5);
        let bytes = read_bytes(&path);
        let idx = find_last(&bytes, b"idx1").expect("idx1 present");
        write_bytes(&path, &bytes[..idx]);

        let mut reader = AviReader::open(&path).expect("open without index");
        assert_eq!(reader.frames(), 5);
        let mut checksums = Vec::new();
        let mut count = 0u64;
        while let Some(frame) = reader.next_frame().expect("read") {
            checksums.push(frame.checksum());
            count += 1;
        }
        assert_eq!(count, 5);
        assert_ne!(
            checksums[0], checksums[1],
            "frames must not be blank/identical"
        );
        assert!(reader.next_frame().expect("eof").is_none());
    }

    #[test]
    fn avi_reader_skips_unknown_chunks() {
        let dir = TempDir::new();
        let path = dir.join("junk.avi");
        let dims = Dims::new(64, 48);
        write_clip(&path, dims, 30.0, 90, 3);
        let bytes = read_bytes(&path);
        let idx = find_last(&bytes, b"idx1").expect("idx1 present");

        // Insert a JUNK chunk between the movi list and idx1 and grow the RIFF
        // size accordingly, keeping every existing offset valid.
        let mut padded = Vec::with_capacity(bytes.len() + 12);
        padded.extend_from_slice(&bytes[..idx]);
        padded.extend_from_slice(b"JUNK");
        padded.extend_from_slice(&4u32.to_le_bytes());
        padded.extend_from_slice(b"abcd");
        padded.extend_from_slice(&bytes[idx..]);
        let riff_size = u32::from_le_bytes([padded[4], padded[5], padded[6], padded[7]]);
        padded[4..8].copy_from_slice(&(riff_size + 12).to_le_bytes());
        write_bytes(&path, &padded);

        let mut reader = AviReader::open(&path).expect("open with junk chunk");
        let mut count = 0u64;
        while reader.next_frame().expect("read").is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn avi_reader_rejects_truncated_payload() {
        let dir = TempDir::new();
        let path = dir.join("truncated.avi");
        let dims = Dims::new(64, 48);
        write_clip(&path, dims, 30.0, 90, 1);
        let bytes = read_bytes(&path);
        // Header (224 bytes) plus chunk header plus half the JPEG payload.
        let cut = 224 + 8 + (bytes.len() - 224 - 8 - 24) / 2;
        write_bytes(&path, &bytes[..cut]);

        let mut reader = AviReader::open(&path).expect("open truncated");
        match reader.next_frame() {
            Err(FluidError::InvalidFormat(_)) => {}
            other => panic!("expected InvalidFormat, got {:?}", other.err()),
        }
    }

    #[test]
    fn avi_reader_rejects_truncated_header() {
        let dir = TempDir::new();
        let path = dir.join("short.avi");
        let dims = Dims::new(64, 48);
        write_clip(&path, dims, 30.0, 90, 1);
        let bytes = read_bytes(&path);
        write_bytes(&path, &bytes[..100]);
        match AviReader::open(&path) {
            Err(FluidError::InvalidFormat(_)) => {}
            other => panic!("expected InvalidFormat, got {:?}", other.err()),
        }

        write_bytes(&path, &bytes[..8]);
        match AviReader::open(&path) {
            Err(FluidError::InvalidFormat(_)) => {}
            other => panic!(
                "expected InvalidFormat for tiny file, got {:?}",
                other.err()
            ),
        }
    }

    #[test]
    fn avi_reader_rejects_non_mjpeg_video() {
        let dir = TempDir::new();
        let path = dir.join("divx.avi");
        let dims = Dims::new(64, 48);
        write_clip(&path, dims, 30.0, 90, 2);
        let mut bytes = read_bytes(&path);
        // Rewrite both the strh handler and the strf compression fourcc so the
        // stream declaratively is DIVX (the payload is still JPEG, we only
        // check the header path).
        let header_end = 224.min(bytes.len());
        let mut patched = 0;
        for i in 0..header_end.saturating_sub(3) {
            if &bytes[i..i + 4] == b"MJPG" {
                bytes[i..i + 4].copy_from_slice(b"DIVX");
                patched += 1;
            }
        }
        assert!(patched >= 2, "expected strh handler and strf compression");
        write_bytes(&path, &bytes);
        match AviReader::open(&path) {
            Err(FluidError::Unsupported(_)) => {}
            other => panic!("expected Unsupported, got {:?}", other.err()),
        }
        let message = match AviReader::open(&path) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(
            message.contains("DIVX"),
            "message must name the fourcc: {message}"
        );
    }

    #[test]
    fn avi_reader_rejects_bad_index_offsets() {
        let dir = TempDir::new();
        let path = dir.join("bad-index.avi");
        let dims = Dims::new(64, 48);
        write_clip(&path, dims, 30.0, 90, 3);
        let mut bytes = read_bytes(&path);
        let idx = find_last(&bytes, b"idx1").expect("idx1 present");
        // Corrupt the first entry offset to point at the movi fourcc itself.
        bytes[idx + 8 + 8..idx + 8 + 12].copy_from_slice(&0u32.to_le_bytes());
        write_bytes(&path, &bytes);
        let mut reader = AviReader::open(&path).expect("open");
        match reader.next_frame() {
            Err(FluidError::InvalidFormat(_)) => {}
            other => panic!("expected InvalidFormat, got {:?}", other.err()),
        }
    }

    #[test]
    fn avi_reader_rejects_bad_magic() {
        let dir = TempDir::new();
        let path = dir.join("not-avi.avi");
        let dims = Dims::new(64, 48);
        write_clip(&path, dims, 30.0, 90, 1);
        let mut bytes = read_bytes(&path);
        bytes[0..4].copy_from_slice(b"AVI ");
        write_bytes(&path, &bytes);
        match AviReader::open(&path) {
            Err(FluidError::InvalidFormat(_)) => {}
            other => panic!("expected InvalidFormat, got {:?}", other.err()),
        }
    }

    #[test]
    fn avi_reader_decodes_colors_and_every_frame() {
        let dir = TempDir::new();
        let path = dir.join("colors.avi");
        let dims = Dims::new(80, 60);
        let mut writer = AviWriter::create(&path, AviHeader::new(dims, 50.0, 95)).expect("create");
        let mut sent = Vec::new();
        for color in [[230u8, 20, 20], [20, 230, 20], [20, 20, 230]] {
            let data: Vec<u8> = color
                .iter()
                .copied()
                .cycle()
                .take((80 * 60 * 3) as usize)
                .collect();
            let frame = Frame::new(dims, fluid_core::PixelFormat::Rgb8, data).expect("frame");
            writer.present(&frame).expect("present");
            sent.push(frame);
        }
        writer.finish().expect("finish");

        let mut reader = AviReader::open(&path).expect("open");
        for expected in &sent {
            let got = reader.next_frame().expect("read").expect("frame");
            for (channel, want) in expected
                .data
                .as_chunks::<3>()
                .0
                .first()
                .expect("pixel")
                .iter()
                .enumerate()
            {
                let got_channel = got.data[channel];
                assert!(
                    i32::from(got_channel).abs_diff(i32::from(*want)) <= 8,
                    "channel {channel}: {got_channel} vs {want}"
                );
            }
        }
        assert!(reader.next_frame().expect("eof").is_none());
    }
}
