//! Streaming Motion-JPEG AVI writer.
//!
//! Why this exists: the only other way to get a real video file out of Fluid
//! is the ffmpeg pipe documented in docs/VIDEO.md, and many Windows machines
//! have no ffmpeg. A Motion-JPEG AVI needs no codec library at all: every
//! video frame becomes one baseline JPEG, and the container is a plain RIFF
//! index that Windows Media Player, VLC, ffmpeg and any DirectShow/Media
//! Foundation player already understand. The image crate (already a
//! dependency) provides the JPEG encoder, so this module adds no new crates.
//!
//! # File layout
//!
//! The writer produces the classic AVI 1.0 layout, little-endian, every chunk
//! padded to an even size:
//!
//! - RIFF 'AVI ' with a 32-bit size
//! - LIST 'hdrl'
//! - avih: 56 byte main header (frame count, microsecond frame time, size)
//! - LIST 'strl'
//! - strh: stream header (vids / MJPG, scale and rate, frame count)
//! - strf: 40 byte BITMAPINFOHEADER (24 bpp, fourcc MJPG)
//! - LIST 'movi' with one '00dc' chunk per frame holding the JPEG bytes
//! - idx1: one 16 byte entry per frame pointing at its chunk
//!
//! The header is written once at creation with every count/size field left at
//! zero, frame chunks are then streamed with a bounded memory footprint (one
//! reusable JPEG buffer plus one index entry per frame), and [AviWriter::finish]
//! seeks back once to patch the RIFF size, the avih frame count and byte rate,
//! the strh frame count, and the movi list size, before appending idx1. The
//! idx1 offsets are relative to the 'movi' fourcc, so the first chunk sits at
//! offset 4, which is what players expect.
//!
//! # Limits and behaviour
//!
//! AVI 1.0 chunk sizes are 32-bit, so the writer refuses to grow a file past
//! roughly 4 GiB with [fluid_core::FluidError::Unsupported] instead of writing
//! a corrupt container; use y4m plus ffmpeg for longer clips. Dimensions must
//! fit in the 16-bit rcFrame fields (1..=65535 per axis) and the frame rate
//! must be known and positive. [AviWriter] is blocking std::fs IO with a 64 KiB
//! BufWriter, it is Send but not Sync (one writer per thread), it allocates no
//! memory per pixel, and [fluid_core::FrameSink::close] finalises the file just
//! like an explicit [AviWriter::finish].
//!
//! The public API mirrors the other sinks: [fluid_core::FrameSink::present]
//! accepts any [fluid_core::PixelFormat] (non-Rgb8 frames are converted first)
//! and [AviWriter::create] takes a path plus an [AviHeader].

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use image::ExtendedColorType;
use image::codecs::jpeg::JpegEncoder;

use fluid_core::{Dims, FluidError, Frame, FrameSink, PixelFormat, Result};

use crate::Y4mHeader;
use crate::rational_fps;

/// Default JPEG quality used by callers that do not care (88 is visually
/// transparent for game and capture content at a fraction of q100 size).
pub const DEFAULT_JPEG_QUALITY: u8 = 88;

/// AVIF_HASINDEX: the file carries an idx1 chunk.
const AVIF_HASINDEX: u32 = 0x10;
/// AVIIF_KEYFRAME: every Motion-JPEG frame is independently decodable.
const AVIIF_KEYFRAME: u32 = 0x10;

/// RIFF chunk sizes are 32-bit; the RIFF size field is file length minus 8.
const MAX_RIFF_SIZE: u64 = u32::MAX as u64;

/// Bytes of the fixed header written at creation time (through the end of the
/// hdrl list; the movi LIST header follows immediately).
const HEADER_LEN: u64 = 212;
/// Absolute position of the 'movi' fourcc inside the header.
const MOVI_FOURCC_POS: u64 = 220;
/// Absolute position of the first frame chunk ('00dc' header).
const FIRST_CHUNK_POS: u64 = 224;
/// LIST hdrl size field: 'hdrl' fourcc + avih chunk + strl LIST.
const HDRL_SIZE: u32 = 192;
/// LIST strl size field: 'strl' fourcc + strh chunk + strf chunk.
const STRL_SIZE: u32 = 116;
const AVIH_SIZE: u32 = 56;
const STRH_SIZE: u32 = 56;
const STRF_SIZE: u32 = 40;
const FRAME_CHUNK_TAG: &[u8; 4] = b"00dc";

// Offsets of the fields that finish() patches inside HEADER_LEN bytes.
const OFF_RIFF_SIZE: usize = 4;
const OFF_AVIH_MICROS: usize = 32;
const OFF_AVIH_MAX_BYTES: usize = 36;
const OFF_AVIH_FLAGS: usize = 44;
const OFF_AVIH_TOTAL_FRAMES: usize = 48;
const OFF_AVIH_STREAMS: usize = 56;
const OFF_AVIH_SUGGESTED: usize = 60;
const OFF_AVIH_WIDTH: usize = 64;
const OFF_AVIH_HEIGHT: usize = 68;
const OFF_STRH_SCALE: usize = 128;
const OFF_STRH_RATE: usize = 132;
const OFF_STRH_LENGTH: usize = 140;
const OFF_STRH_SUGGESTED: usize = 144;
const OFF_STRH_QUALITY: usize = 148;
const OFF_MOVI_SIZE: usize = 216;

const _: () = assert!(MOVI_FOURCC_POS == HEADER_LEN + 8);
const _: () = assert!(FIRST_CHUNK_POS == MOVI_FOURCC_POS + 4);

// Compile-time guard that the hard-coded offsets really describe one header.
const _: () = assert!(HEADER_LEN as usize == 212 && OFF_MOVI_SIZE + 4 == 220);

/// Description of a Motion-JPEG AVI stream: size, frame rate and JPEG quality.
///
/// The frame rate uses the same rational representation as
/// [crate::Y4mHeader]: fps_num / fps_den, so 60 fps is 60/1 and 29.97 fps is
/// 30000/1001. [AviHeader::new] performs that conversion from a float.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AviHeader {
    /// Frame size in pixels; both axes must be 1..=65535.
    pub dims: Dims,
    /// Frame rate numerator (frames per second * fps_den).
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// JPEG quality, 1 (worst/smallest) ..= 100 (best/largest).
    pub quality: u8,
}

impl AviHeader {
    /// Build a header from a floating point frame rate and a JPEG quality.
    pub fn new(dims: Dims, fps: f32, quality: u8) -> Self {
        let (fps_num, fps_den) = rational_fps(fps);
        Self {
            dims,
            fps_num,
            fps_den,
            quality,
        }
    }

    /// Reuse the frame rate of an already parsed y4m header.
    pub fn from_y4m_header(header: &Y4mHeader, quality: u8) -> Self {
        Self {
            dims: header.dims,
            fps_num: header.fps_num,
            fps_den: header.fps_den,
            quality,
        }
    }

    /// Frame rate in frames per second (0.0 when unknown).
    pub fn fps(&self) -> f32 {
        if self.fps_den == 0 {
            0.0
        } else {
            self.fps_num as f32 / self.fps_den as f32
        }
    }

    /// Validate the fields a container must be able to represent.
    fn validate(&self) -> Result<()> {
        if self.dims.is_empty() {
            return Err(FluidError::invalid(
                "cannot write an AVI with empty dimensions",
            ));
        }
        if self.dims.width > u16::MAX as u32 || self.dims.height > u16::MAX as u32 {
            return Err(FluidError::invalid(format!(
                "AVI rcFrame supports at most 65535x65535 pixels, got {}",
                self.dims
            )));
        }
        if self.fps_num == 0 || self.fps_den == 0 {
            return Err(FluidError::invalid(
                "AVI needs a known, positive frame rate",
            ));
        }
        if !(1..=100).contains(&self.quality) {
            return Err(FluidError::invalid(format!(
                "JPEG quality must be between 1 and 100, got {}",
                self.quality
            )));
        }
        Ok(())
    }

    /// Frame duration in microseconds, as stored in avih.dwMicroSecPerFrame.
    fn micros_per_frame(&self) -> u32 {
        let num = u64::from(self.fps_num).max(1);
        let den = u64::from(self.fps_den);
        ((den * 1_000_000 + num / 2) / num).min(u32::MAX as u64) as u32
    }
}

/// One entry of the idx1 index: where the frame chunk lives and how big it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AviIndex {
    /// Offset of the '00dc' chunk header relative to the 'movi' fourcc.
    offset: u32,
    /// Payload (JPEG) length in bytes, excluding the padding byte.
    len: u32,
}

/// Streaming Motion-JPEG AVI writer.
///
/// Create it with [AviWriter::create], feed frames through
/// [fluid_core::FrameSink::present] (or the inherent present method, which is
/// the same call), then [AviWriter::finish] to patch the headers and append the
/// index. Dropping the writer also finalises the file.
pub struct AviWriter {
    inner: BufWriter<File>,
    header: AviHeader,
    /// The fixed header, kept so finish() can patch counts and sizes.
    header_bytes: Vec<u8>,
    /// Frames accepted so far.
    frames: u64,
    /// Bytes of the movi LIST data written so far, including the 'movi' fourcc
    /// (so the next chunk header sits at this offset relative to that fourcc).
    movi_len: u64,
    index: Vec<AviIndex>,
    /// Reusable JPEG output buffer, so encoding allocates nothing per frame.
    jpeg: Vec<u8>,
    /// Largest JPEG payload seen, reported as the suggested buffer size.
    max_chunk: u32,
    finished: bool,
}

impl AviWriter {
    /// Create an AVI file at path. Fails before touching the filesystem when
    /// the header cannot be represented (empty/oversized dims, zero fps, bad
    /// quality).
    pub fn create(path: impl AsRef<Path>, header: AviHeader) -> Result<Self> {
        header.validate()?;
        let file = File::create(path.as_ref())?;
        let mut inner = BufWriter::with_capacity(1 << 16, file);
        let header_bytes = build_header(&header);
        inner.write_all(&header_bytes)?;
        Ok(Self {
            inner,
            header,
            header_bytes,
            frames: 0,
            movi_len: 4, // the 'movi' fourcc itself
            index: Vec::new(),
            jpeg: Vec::new(),
            max_chunk: 0,
            finished: false,
        })
    }

    /// Number of frames accepted so far (written to disk only after finish).
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// The header this writer was created with.
    pub fn header(&self) -> &AviHeader {
        &self.header
    }

    /// Patch every deferred header field, write the idx1 index and flush.
    ///
    /// Calling finish twice (or after close) is harmless and returns Ok.
    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        let total = predicted_file_len(self.movi_len, self.frames).ok_or_else(too_large)?;
        let riff_size = total - 8;
        if riff_size > MAX_RIFF_SIZE {
            return Err(too_large());
        }
        let frames = u32::try_from(self.frames).map_err(|_| too_large())?;
        let movi_size = u32::try_from(self.movi_len).map_err(|_| too_large())?;
        let max_bytes_per_sec = self.max_bytes_per_sec();

        set_u32(&mut self.header_bytes, OFF_RIFF_SIZE, riff_size as u32);
        set_u32(
            &mut self.header_bytes,
            OFF_AVIH_MAX_BYTES,
            max_bytes_per_sec,
        );
        set_u32(&mut self.header_bytes, OFF_AVIH_TOTAL_FRAMES, frames);
        set_u32(&mut self.header_bytes, OFF_AVIH_SUGGESTED, self.max_chunk);
        set_u32(&mut self.header_bytes, OFF_STRH_LENGTH, frames);
        set_u32(&mut self.header_bytes, OFF_STRH_SUGGESTED, self.max_chunk);
        set_u32(&mut self.header_bytes, OFF_MOVI_SIZE, movi_size);

        self.inner.flush()?;
        {
            let file = self.inner.get_mut();
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&self.header_bytes)?;
            file.seek(SeekFrom::End(0))?;
            // idx1: chunk header, then one entry per frame.
            file.write_all(b"idx1")?;
            file.write_all(&(16 * frames).to_le_bytes())?;
            for entry in &self.index {
                file.write_all(FRAME_CHUNK_TAG)?;
                file.write_all(&AVIIF_KEYFRAME.to_le_bytes())?;
                file.write_all(&entry.offset.to_le_bytes())?;
                file.write_all(&entry.len.to_le_bytes())?;
            }
            file.flush()?;
        }
        self.finished = true;
        Ok(())
    }

    /// Bytes per second the player can expect (avih.dwMaxBytesPerSec).
    fn max_bytes_per_sec(&self) -> u32 {
        let num = u64::from(self.header.fps_num);
        let den = u64::from(self.header.fps_den).max(1);
        let per_second = u64::from(self.max_chunk) * num;
        ((per_second + den / 2) / den).min(u32::MAX as u64) as u32
    }

    /// Encode one frame as baseline JPEG and append it as a '00dc' chunk.
    fn append_frame(&mut self, frame: &Frame) -> Result<()> {
        let converted;
        let rgb: &[u8] = if frame.fmt == PixelFormat::Rgb8 {
            &frame.data
        } else {
            converted = frame.convert(PixelFormat::Rgb8);
            &converted.data
        };
        let dims = self.header.dims;
        self.jpeg.clear();
        {
            let mut encoder = JpegEncoder::new_with_quality(&mut self.jpeg, self.header.quality);
            encoder
                .encode(rgb, dims.width, dims.height, ExtendedColorType::Rgb8)
                .map_err(|e| FluidError::internal(format!("JPEG encoding failed: {e}")))?;
        }
        let len = u64::try_from(self.jpeg.len())
            .map_err(|_| FluidError::internal("JPEG payload too large"))?;
        if len > u32::MAX as u64 {
            return Err(FluidError::internal("JPEG payload exceeds 4 GiB"));
        }
        let pad = len & 1;
        let next_movi = self.movi_len + 8 + len + pad;
        let next_frames = self.frames + 1;
        check_file_growth(next_movi, next_frames)?;

        // The chunk header starts where movi_len currently points, i.e. at
        // offset movi_len relative to the 'movi' fourcc (4 for the first one).
        self.index.push(AviIndex {
            offset: self.movi_len as u32,
            len: len as u32,
        });
        let Self {
            inner,
            jpeg,
            frames,
            movi_len,
            max_chunk,
            ..
        } = self;
        inner.write_all(FRAME_CHUNK_TAG)?;
        inner.write_all(&(len as u32).to_le_bytes())?;
        inner.write_all(jpeg)?;
        if pad == 1 {
            inner.write_all(&[0u8])?;
        }
        *max_chunk = (*max_chunk).max(len as u32);
        *movi_len = next_movi;
        *frames = next_frames;
        Ok(())
    }
}

impl FrameSink for AviWriter {
    fn dims(&self) -> Dims {
        self.header.dims
    }

    fn backend_name(&self) -> &'static str {
        "mjpeg-avi"
    }

    fn present(&mut self, frame: &Frame) -> Result<()> {
        if self.finished {
            return Err(FluidError::invalid(
                "AVI writer is already finished (call present before finish)",
            ));
        }
        if frame.dims != self.header.dims {
            return Err(FluidError::invalid(format!(
                "AVI writer expects {} frames, got {}",
                self.header.dims, frame.dims
            )));
        }
        self.append_frame(frame)
    }

    fn close(&mut self) {
        // The pipeline calls close() when a run ends; finalise so the file is
        // playable even if the caller forgets finish().
        let _ = self.finish();
    }
}

impl Drop for AviWriter {
    fn drop(&mut self) {
        // Best effort: never leave a half-patched header behind.
        if !self.finished {
            let _ = self.finish();
        }
    }
}

/// Total file length for a movi list of movi_len bytes and frames index entries.
#[inline]
fn predicted_file_len(movi_len: u64, frames: u64) -> Option<u64> {
    let chunk_bytes = movi_len.checked_sub(4)?;
    FIRST_CHUNK_POS
        .checked_add(chunk_bytes)?
        .checked_add(8)? // idx1 chunk header
        .checked_add(16u64.checked_mul(frames)?)
}

/// Reject a frame that would push the file past the AVI 1.0 size limit.
fn check_file_growth(movi_len: u64, frames: u64) -> Result<()> {
    match predicted_file_len(movi_len, frames) {
        Some(total) if total - 8 <= MAX_RIFF_SIZE => Ok(()),
        _ => Err(too_large()),
    }
}

fn too_large() -> FluidError {
    FluidError::unsupported(
        "AVI 1.0 files are limited to 4 GiB; shorten the clip or keep using y4m/ffmpeg for longer exports",
    )
}

#[inline]
fn set_u32(buf: &mut [u8], off: usize, value: u32) {
    buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
}

#[inline]
fn set_u16(buf: &mut [u8], off: usize, value: u16) {
    buf[off..off + 2].copy_from_slice(&value.to_le_bytes());
}

/// Build the fixed part of the file: RIFF, hdrl (avih, strl with strh/strf)
/// and the start of the movi LIST. All deferred fields stay zero here.
fn build_header(header: &AviHeader) -> Vec<u8> {
    // Includes the movi LIST header, so the first frame chunk follows directly.
    let mut b = vec![0u8; FIRST_CHUNK_POS as usize];
    b[0..4].copy_from_slice(b"RIFF");
    // Offset 4 (RIFF size) and the frame counts/sizes are patched by finish().
    b[8..12].copy_from_slice(b"AVI ");

    // LIST hdrl.
    b[12..16].copy_from_slice(b"LIST");
    set_u32(&mut b, 16, HDRL_SIZE);
    b[20..24].copy_from_slice(b"hdrl");

    // avih: main AVI header.
    b[24..28].copy_from_slice(b"avih");
    set_u32(&mut b, 28, AVIH_SIZE);
    set_u32(&mut b, OFF_AVIH_MICROS, header.micros_per_frame());
    // 36 (max bytes/sec) and 48 (total frames) and 60 (suggested buffer) are
    // patched at finish.
    set_u32(&mut b, OFF_AVIH_FLAGS, AVIF_HASINDEX);
    set_u32(&mut b, OFF_AVIH_STREAMS, 1);
    set_u32(&mut b, OFF_AVIH_WIDTH, header.dims.width);
    set_u32(&mut b, OFF_AVIH_HEIGHT, header.dims.height);

    // LIST strl.
    b[88..92].copy_from_slice(b"LIST");
    set_u32(&mut b, 92, STRL_SIZE);
    b[96..100].copy_from_slice(b"strl");

    // strh: stream header for the single video stream.
    b[100..104].copy_from_slice(b"strh");
    set_u32(&mut b, 104, STRH_SIZE);
    b[108..112].copy_from_slice(b"vids");
    b[112..116].copy_from_slice(b"MJPG");
    set_u32(&mut b, OFF_STRH_SCALE, header.fps_den);
    set_u32(&mut b, OFF_STRH_RATE, header.fps_num);
    set_u32(&mut b, OFF_STRH_QUALITY, u32::from(header.quality) * 100);
    // rcFrame: 0,0,width,height (16-bit fields at 156..164).
    set_u16(&mut b, 160, header.dims.width as u16);
    set_u16(&mut b, 162, header.dims.height as u16);

    // strf: BITMAPINFOHEADER describing the MJPG stream.
    b[164..168].copy_from_slice(b"strf");
    set_u32(&mut b, 168, STRF_SIZE);
    set_u32(&mut b, 172, STRF_SIZE);
    set_u32(&mut b, 176, header.dims.width);
    set_u32(&mut b, 180, header.dims.height);
    set_u16(&mut b, 184, 1); // biPlanes
    set_u16(&mut b, 186, 24); // biBitCount
    b[188..192].copy_from_slice(b"MJPG"); // biCompression

    // LIST movi (its size is patched at finish).
    b[212..216].copy_from_slice(b"LIST");
    b[220..224].copy_from_slice(b"movi");
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    use crate::{Y4mHeader, synthetic_pattern};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, AtomicOrdering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "fluid-video-avi-{}-{}-{}",
                std::process::id(),
                tag,
                n
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create temp dir");
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

    fn read_u32(bytes: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(bytes[off..off + 4].try_into().expect("u32"))
    }

    fn read_u16(bytes: &[u8], off: usize) -> u16 {
        u16::from_le_bytes(bytes[off..off + 2].try_into().expect("u16"))
    }

    fn tag4(bytes: &[u8], off: usize) -> [u8; 4] {
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&bytes[off..off + 4]);
        tag
    }

    struct ChunkInfo {
        /// Absolute file position of the '00dc' chunk header.
        pos: u64,
        /// Offset stored in idx1 (relative to the 'movi' fourcc).
        offset: u32,
        len: u32,
        jpeg: Vec<u8>,
    }

    struct ParsedAvi {
        file_len: u64,
        riff_size: u32,
        avih_flags: u32,
        avih_micros: u32,
        avih_max_bytes: u32,
        avih_total_frames: u32,
        avih_suggested: u32,
        width: u32,
        height: u32,
        bit_count: u16,
        strh_scale: u32,
        strh_rate: u32,
        strh_length: u32,
        strh_suggested: u32,
        strf_compression: [u8; 4],
        movi_size: u32,
        chunks: Vec<ChunkInfo>,
        /// (offset, len, flags) triples from idx1.
        index: Vec<(u32, u32, u32)>,
    }

    /// Strictly parse our own output, asserting the whole RIFF/AVI structure
    /// and that every index entry really points at its frame chunk.
    fn parse_avi(bytes: &[u8]) -> ParsedAvi {
        assert!(bytes.len() >= 12, "file too short");
        assert_eq!(tag4(bytes, 0), *b"RIFF");
        assert_eq!(tag4(bytes, 8), *b"AVI ");
        let riff_size = read_u32(bytes, 4);
        assert_eq!(
            u64::from(riff_size),
            bytes.len() as u64 - 8,
            "RIFF size must cover the file minus its 8 byte header"
        );

        // LIST hdrl.
        assert_eq!(tag4(bytes, 12), *b"LIST");
        assert_eq!(read_u32(bytes, 16), HDRL_SIZE);
        assert_eq!(tag4(bytes, 20), *b"hdrl");
        let hdrl_end = 20 + HDRL_SIZE as usize;
        assert_eq!(hdrl_end, HEADER_LEN as usize);

        // avih.
        assert_eq!(tag4(bytes, 24), *b"avih");
        assert_eq!(read_u32(bytes, 28), AVIH_SIZE);
        let avih_micros = read_u32(bytes, OFF_AVIH_MICROS);
        let avih_max_bytes = read_u32(bytes, OFF_AVIH_MAX_BYTES);
        let avih_flags = read_u32(bytes, OFF_AVIH_FLAGS);
        let avih_total_frames = read_u32(bytes, OFF_AVIH_TOTAL_FRAMES);
        assert_eq!(read_u32(bytes, OFF_AVIH_STREAMS), 1);
        let avih_suggested = read_u32(bytes, OFF_AVIH_SUGGESTED);
        let width = read_u32(bytes, OFF_AVIH_WIDTH);
        let height = read_u32(bytes, OFF_AVIH_HEIGHT);
        assert_eq!(avih_flags & AVIF_HASINDEX, AVIF_HASINDEX);

        // LIST strl.
        assert_eq!(tag4(bytes, 88), *b"LIST");
        assert_eq!(read_u32(bytes, 92), STRL_SIZE);
        assert_eq!(tag4(bytes, 96), *b"strl");
        assert_eq!(tag4(bytes, 100), *b"strh");
        assert_eq!(read_u32(bytes, 104), STRH_SIZE);
        assert_eq!(tag4(bytes, 108), *b"vids");
        assert_eq!(tag4(bytes, 112), *b"MJPG");
        let strh_scale = read_u32(bytes, OFF_STRH_SCALE);
        let strh_rate = read_u32(bytes, OFF_STRH_RATE);
        let strh_length = read_u32(bytes, OFF_STRH_LENGTH);
        let strh_suggested = read_u32(bytes, OFF_STRH_SUGGESTED);
        assert_eq!(read_u16(bytes, 160), width as u16);
        assert_eq!(read_u16(bytes, 162), height as u16);
        assert_eq!(tag4(bytes, 164), *b"strf");
        assert_eq!(read_u32(bytes, 168), STRF_SIZE);
        assert_eq!(read_u32(bytes, 172), STRF_SIZE);
        assert_eq!(read_u32(bytes, 176), width);
        assert_eq!(read_u32(bytes, 180), height);
        assert_eq!(read_u16(bytes, 184), 1);
        let bit_count = read_u16(bytes, 186);
        let strf_compression = tag4(bytes, 188);

        // LIST movi.
        let movi_pos = HEADER_LEN as usize;
        assert_eq!(tag4(bytes, movi_pos), *b"LIST");
        let movi_size = read_u32(bytes, movi_pos + 4);
        assert_eq!(tag4(bytes, movi_pos + 8), *b"movi");
        let movi_data_start = movi_pos + 12;
        let movi_data_end = movi_pos + 8 + movi_size as usize;

        let mut chunks = Vec::new();
        let mut pos = movi_data_start;
        while pos < movi_data_end {
            assert_eq!(tag4(bytes, pos), *FRAME_CHUNK_TAG);
            let len = read_u32(bytes, pos + 4);
            let payload_end = pos + 8 + len as usize;
            assert!(
                payload_end <= movi_data_end,
                "frame chunk overruns movi list"
            );
            let jpeg = bytes[pos + 8..payload_end].to_vec();
            chunks.push(ChunkInfo {
                pos: pos as u64,
                offset: (pos - MOVI_FOURCC_POS as usize) as u32,
                len,
                jpeg,
            });
            pos = payload_end;
            if len & 1 == 1 {
                assert_eq!(pos, payload_end);
                assert_eq!(bytes[pos], 0, "odd chunk must be padded with zero");
                pos += 1;
            }
            assert_eq!(pos % 2, 0, "chunk headers must stay even aligned");
        }
        assert_eq!(pos, movi_data_end, "movi size must cover every chunk");

        // idx1.
        assert_eq!(tag4(bytes, movi_data_end), *b"idx1");
        let idx_size = read_u32(bytes, movi_data_end + 4);
        let idx_start = movi_data_end + 8;
        assert_eq!(
            idx_size as usize,
            16 * chunks.len(),
            "idx1 must hold one entry per chunk"
        );
        let idx_end = idx_start + idx_size as usize;
        assert_eq!(idx_end, bytes.len(), "idx1 must end the file");
        let mut index = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let off = idx_start + i * 16;
            assert_eq!(tag4(bytes, off), *FRAME_CHUNK_TAG);
            let flags = read_u32(bytes, off + 4);
            let chunk_offset = read_u32(bytes, off + 8);
            let chunk_len = read_u32(bytes, off + 12);
            assert_eq!(flags & AVIIF_KEYFRAME, AVIIF_KEYFRAME);
            assert_eq!(
                chunk_offset, chunk.offset,
                "idx1 entry {i} must point at its chunk header"
            );
            assert_eq!(chunk_len, chunk.len);
            assert_eq!(MOVI_FOURCC_POS + u64::from(chunk_offset), chunk.pos);
            index.push((chunk_offset, chunk_len, flags));
        }

        ParsedAvi {
            file_len: bytes.len() as u64,
            riff_size,
            avih_flags,
            avih_micros,
            avih_max_bytes,
            avih_total_frames,
            avih_suggested,
            width,
            height,
            bit_count,
            strh_scale,
            strh_rate,
            strh_length,
            strh_suggested,
            strf_compression,
            movi_size,
            chunks,
            index,
        }
    }

    fn synthetic_clip(dims: Dims, count: u64) -> Vec<Frame> {
        (0..count)
            .map(|i| synthetic_pattern(dims, i as f32 / 30.0, i))
            .collect()
    }

    fn write_clip(path: &Path, header: AviHeader, frames: &[Frame]) {
        let mut writer = AviWriter::create(path, header).expect("create avi");
        for frame in frames {
            writer.present(frame).expect("present frame");
        }
        writer.finish().expect("finish avi");
    }

    fn solid_frame(dims: Dims, format: PixelFormat, channels: [u8; 4]) -> Frame {
        let mut frame = Frame::zeroed(dims, format);
        let bpp = format.bytes_per_pixel();
        for pixel in frame.data.chunks_exact_mut(bpp) {
            pixel.copy_from_slice(&channels[..bpp]);
        }
        frame
    }

    fn noise_frame(dims: Dims, seed: u32) -> Frame {
        let mut frame = Frame::zeroed(dims, PixelFormat::Rgb8);
        let mut state = seed | 1;
        for byte in &mut frame.data {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *byte = (state >> 8) as u8;
        }
        frame
    }

    #[test]
    fn avi_header_fields_roundtrip() {
        let temp = TempDir::new("header");
        let path = temp.join("clip.avi");
        let dims = Dims::new(64, 48);
        let header = AviHeader::new(dims, 29.97, 82);
        assert_eq!((header.fps_num, header.fps_den), (30000, 1001));
        let frames = synthetic_clip(dims, 3);
        write_clip(&path, header, &frames);

        let parsed = parse_avi(&fs::read(&path).expect("read avi"));
        assert_eq!((parsed.width, parsed.height), (64, 48));
        assert_eq!(parsed.avih_total_frames, 3);
        assert_eq!(parsed.strh_length, 3);
        assert_eq!(parsed.avih_micros, 33_367);
        assert_eq!((parsed.strh_scale, parsed.strh_rate), (1001, 30000));
        assert_eq!(parsed.strf_compression, *b"MJPG");
        assert_eq!(parsed.bit_count, 24);
        assert_eq!(parsed.avih_flags & AVIF_HASINDEX, AVIF_HASINDEX);
        let biggest = parsed.chunks.iter().map(|c| c.len).max().unwrap_or(0);
        assert!(parsed.avih_max_bytes >= biggest);
        assert_eq!(parsed.avih_suggested, biggest);
        assert_eq!(parsed.strh_suggested, biggest);
        assert_eq!(parsed.file_len, fs::metadata(&path).unwrap().len());
    }

    #[test]
    fn avi_index_offsets_and_chunk_sizes_are_consistent() {
        let temp = TempDir::new("index");
        let path = temp.join("clip.avi");
        let dims = Dims::new(48, 32);
        let frames = synthetic_clip(dims, 5);
        write_clip(&path, AviHeader::new(dims, 60.0, 88), &frames);
        let bytes = fs::read(&path).expect("read avi");
        let parsed = parse_avi(&bytes);

        assert_eq!(parsed.chunks.len(), 5);
        assert_eq!(parsed.index.len(), 5);
        // idx1 offsets must be the running sum of padded chunk sizes, starting
        // at 4 (right after the 'movi' fourcc).
        let mut expected = 4u32;
        for (i, chunk) in parsed.chunks.iter().enumerate() {
            assert_eq!(chunk.offset, expected, "chunk {i} offset");
            assert_eq!(parsed.index[i].0, expected);
            assert_eq!(parsed.index[i].1, chunk.len);
            assert_eq!(chunk.pos, MOVI_FOURCC_POS + u64::from(expected));
            expected += 8 + chunk.len + (chunk.len & 1);
        }
        // The movi list size covers the fourcc plus every padded chunk.
        assert_eq!(
            u64::from(parsed.movi_size),
            4 + parsed
                .chunks
                .iter()
                .map(|c| 8 + u64::from(c.len) + u64::from(c.len & 1))
                .sum::<u64>()
        );
        // And the file is exactly header + movi + idx1.
        let expected_len =
            HEADER_LEN + 8 + u64::from(parsed.movi_size) + 8 + 16 * parsed.chunks.len() as u64;
        assert_eq!(parsed.file_len, expected_len);
        assert_eq!(u64::from(parsed.riff_size), expected_len - 8);
    }

    #[test]
    fn avi_jpeg_payloads_decode_to_the_right_dimensions() {
        let temp = TempDir::new("decode");
        let path = temp.join("clip.avi");
        let dims = Dims::new(64, 48);
        let frames = synthetic_clip(dims, 3);
        write_clip(&path, AviHeader::new(dims, 30.0, 90), &frames);
        let parsed = parse_avi(&fs::read(&path).expect("read avi"));
        for (i, chunk) in parsed.chunks.iter().enumerate() {
            let decoded = image::load_from_memory(&chunk.jpeg).expect("decode jpeg payload");
            assert_eq!(decoded.width(), dims.width, "frame {i} width");
            assert_eq!(decoded.height(), dims.height, "frame {i} height");
            let rgb = decoded.to_rgb8();
            let source = &frames[i].data;
            let mut total_error = 0u64;
            for (&a, &b) in rgb.as_raw().iter().zip(source.iter()) {
                total_error += u64::from(a.abs_diff(b));
            }
            let mean_error = total_error as f64 / source.len() as f64;
            assert!(
                mean_error < 8.0,
                "frame {i} mean per-channel error {mean_error:.2} is too high"
            );
        }
    }

    #[test]
    fn avi_handles_odd_dimensions() {
        let temp = TempDir::new("odd");
        let path = temp.join("odd.avi");
        let dims = Dims::new(127, 73);
        let frames = synthetic_clip(dims, 2);
        write_clip(&path, AviHeader::new(dims, 24.0, 92), &frames);
        let parsed = parse_avi(&fs::read(&path).expect("read avi"));
        assert_eq!((parsed.width, parsed.height), (127, 73));
        assert_eq!(parsed.chunks.len(), 2);
        for chunk in &parsed.chunks {
            let decoded = image::load_from_memory(&chunk.jpeg).expect("decode jpeg payload");
            assert_eq!((decoded.width(), decoded.height()), (127, 73));
        }
    }

    #[test]
    fn avi_single_frame_clip_is_valid() {
        let temp = TempDir::new("single");
        let path = temp.join("one.avi");
        let dims = Dims::new(32, 24);
        write_clip(
            &path,
            AviHeader::new(dims, 1.0, 80),
            &synthetic_clip(dims, 1),
        );
        let parsed = parse_avi(&fs::read(&path).expect("read avi"));
        assert_eq!(parsed.avih_total_frames, 1);
        assert_eq!(parsed.strh_length, 1);
        assert_eq!(parsed.chunks.len(), 1);
        assert_eq!(parsed.index.len(), 1);
        assert_eq!(parsed.avih_micros, 1_000_000);
    }

    #[test]
    fn avi_zero_frame_clip_is_valid() {
        let temp = TempDir::new("empty");
        let path = temp.join("empty.avi");
        write_clip(&path, AviHeader::new(Dims::new(32, 24), 30.0, 88), &[]);
        let bytes = fs::read(&path).expect("read avi");
        let parsed = parse_avi(&bytes);
        assert_eq!(parsed.avih_total_frames, 0);
        assert_eq!(parsed.strh_length, 0);
        assert_eq!(parsed.movi_size, 4);
        assert!(parsed.chunks.is_empty());
        assert!(parsed.index.is_empty());
        assert_eq!(bytes.len() as u64, HEADER_LEN + 8 + 4 + 8);
    }

    #[test]
    fn avi_rejects_invalid_headers() {
        let temp = TempDir::new("badheader");
        let path = temp.join("bad.avi");
        let dims = Dims::new(64, 48);
        let cases = [
            AviHeader::new(Dims::new(0, 48), 30.0, 88),
            AviHeader::new(Dims::new(64, 0), 30.0, 88),
            AviHeader::new(Dims::new(70_000, 48), 30.0, 88),
            AviHeader::new(Dims::new(64, 70_000), 30.0, 88),
            AviHeader::new(dims, 0.0, 88),
            AviHeader::new(dims, -30.0, 88),
            AviHeader::new(dims, 30.0, 0),
            AviHeader::new(dims, 30.0, 101),
        ];
        for header in cases {
            let result = AviWriter::create(&path, header);
            assert!(result.is_err(), "header {header:?} should be rejected");
        }
        assert!(!path.exists(), "invalid headers must not create the file");
        let ok = AviHeader::new(dims, 30.0, 100);
        assert!(AviWriter::create(temp.join("ok.avi"), ok).is_ok());
    }

    #[test]
    fn avi_present_after_finish_and_wrong_dims_are_rejected() {
        let temp = TempDir::new("reject");
        let path = temp.join("clip.avi");
        let dims = Dims::new(32, 24);
        let mut writer = AviWriter::create(&path, AviHeader::new(dims, 30.0, 88)).expect("create");
        let frame = synthetic_pattern(dims, 0.0, 0);
        writer.present(&frame).expect("first frame");
        writer.finish().expect("finish");
        assert!(
            writer.present(&frame).is_err(),
            "finish is final for present"
        );
        writer.finish().expect("finish is idempotent");

        let other = synthetic_pattern(Dims::new(16, 16), 0.0, 0);
        let mut writer = AviWriter::create(&path, AviHeader::new(dims, 30.0, 88)).expect("create");
        assert!(writer.present(&other).is_err(), "dims mismatch must fail");
        // The rejected frame must not be counted as written.
        assert_eq!(writer.frames(), 0);
    }

    #[test]
    fn avi_converts_other_pixel_formats() {
        let temp = TempDir::new("formats");
        let path = temp.join("formats.avi");
        let dims = Dims::new(16, 16);
        let frames = [
            solid_frame(dims, PixelFormat::Rgba8, [255, 0, 0, 255]),
            solid_frame(dims, PixelFormat::Bgra8, [255, 0, 0, 255]),
            solid_frame(dims, PixelFormat::Luma8, [128, 128, 128, 128]),
        ];
        write_clip(&path, AviHeader::new(dims, 30.0, 95), &frames);
        let parsed = parse_avi(&fs::read(&path).expect("read avi"));
        assert_eq!(parsed.chunks.len(), 3);

        let red = image::load_from_memory(&parsed.chunks[0].jpeg)
            .expect("decode")
            .to_rgb8();
        let (r, g, b) = red.get_pixel(8, 8).0.into();
        assert!(
            r > 200 && g < 60 && b < 60,
            "RGBA red decoded to {r},{g},{b}"
        );

        let blue = image::load_from_memory(&parsed.chunks[1].jpeg)
            .expect("decode")
            .to_rgb8();
        let (r, g, b) = blue.get_pixel(8, 8).0.into();
        assert!(
            b > 200 && r < 60 && g < 60,
            "BGRA blue decoded to {r},{g},{b}"
        );

        let grey = image::load_from_memory(&parsed.chunks[2].jpeg)
            .expect("decode")
            .to_rgb8();
        let (r, _, _) = grey.get_pixel(8, 8).0.into();
        assert!(r.abs_diff(128) <= 30, "Luma8 128 decoded to brightness {r}");
    }

    #[test]
    fn avi_odd_jpeg_lengths_are_padded() {
        let temp = TempDir::new("padding");
        let path = temp.join("noise.avi");
        let dims = Dims::new(32, 24);
        let frames: Vec<Frame> = (0..8).map(|i| noise_frame(dims, i as u32 + 1)).collect();
        write_clip(&path, AviHeader::new(dims, 30.0, 88), &frames);
        let parsed = parse_avi(&fs::read(&path).expect("read avi"));
        assert_eq!(parsed.chunks.len(), 8);
        assert!(
            parsed.chunks.iter().any(|c| c.len & 1 == 1),
            "test needs at least one odd payload to exercise padding"
        );
        for chunk in &parsed.chunks {
            assert_eq!(chunk.offset & 1, 0, "chunks must be even aligned");
            assert_eq!(chunk.pos, MOVI_FOURCC_POS + u64::from(chunk.offset));
        }
        // Every payload still decodes after the padding byte in between.
        for chunk in &parsed.chunks {
            assert!(
                image::load_from_memory(&chunk.jpeg).is_ok(),
                "padded chunk must decode"
            );
        }
    }

    #[test]
    fn avi_quality_changes_payload_size() {
        let temp = TempDir::new("quality");
        let dims = Dims::new(64, 48);
        let frames = synthetic_clip(dims, 1);
        let low = temp.join("low.avi");
        let high = temp.join("high.avi");
        write_clip(&low, AviHeader::new(dims, 30.0, 20), &frames);
        write_clip(&high, AviHeader::new(dims, 30.0, 95), &frames);
        let low = parse_avi(&fs::read(&low).expect("read avi"));
        let high = parse_avi(&fs::read(&high).expect("read avi"));
        assert!(
            high.chunks[0].len > low.chunks[0].len,
            "q95 payload {} should be larger than q20 payload {}",
            high.chunks[0].len,
            low.chunks[0].len
        );
    }

    #[test]
    fn avi_close_and_drop_finalize_the_file() {
        let temp = TempDir::new("finalize");
        let dims = Dims::new(32, 24);
        let frames = synthetic_clip(dims, 2);

        let closed = temp.join("closed.avi");
        let mut writer =
            AviWriter::create(&closed, AviHeader::new(dims, 30.0, 88)).expect("create");
        for frame in &frames {
            writer.present(frame).expect("present");
        }
        writer.close(); // FrameSink::close must finish the file
        drop(writer);
        let parsed = parse_avi(&fs::read(&closed).expect("read avi"));
        assert_eq!(parsed.avih_total_frames, 2);
        assert_eq!(parsed.chunks.len(), 2);
        assert_eq!(parsed.index.len(), 2);

        let dropped = temp.join("dropped.avi");
        {
            let mut writer =
                AviWriter::create(&dropped, AviHeader::new(dims, 30.0, 88)).expect("create");
            for frame in &frames {
                writer.present(frame).expect("present");
            }
            // No finish: Drop is responsible.
        }
        let parsed = parse_avi(&fs::read(&dropped).expect("read avi"));
        assert_eq!(parsed.avih_total_frames, 2);
        assert_eq!(parsed.chunks.len(), 2);
    }

    #[test]
    fn avi_header_from_y4m_header() {
        let y4m = Y4mHeader::new(Dims::new(640, 360), 59.94);
        let header = AviHeader::from_y4m_header(&y4m, 75);
        assert_eq!(header.dims, Dims::new(640, 360));
        assert_eq!((header.fps_num, header.fps_den), (60000, 1001));
        assert_eq!(header.quality, 75);
        assert!((header.fps() - 59.94).abs() < 0.001);
    }

    #[test]
    fn avi_rejects_files_over_four_gib() {
        assert!(check_file_growth(4, 0).is_ok());
        // A movi list already at the 32-bit limit cannot grow further.
        assert!(check_file_growth(u64::from(u32::MAX), 0).is_err());
        // The index itself also has a 32-bit size limit.
        assert!(check_file_growth(4, u64::from(u32::MAX)).is_err());
    }

    #[test]
    fn avi_second_present_encodes_the_next_frame() {
        // Guards against reusing the JPEG scratch buffer without clearing it.
        let temp = TempDir::new("scratch");
        let path = temp.join("clip.avi");
        let dims = Dims::new(40, 30);
        let frames = synthetic_clip(dims, 2);
        write_clip(&path, AviHeader::new(dims, 30.0, 88), &frames);
        let parsed = parse_avi(&fs::read(&path).expect("read avi"));
        assert_ne!(
            parsed.chunks[0].jpeg, parsed.chunks[1].jpeg,
            "the second frame must be encoded from scratch"
        );
    }
}
