#!/bin/sh
# video-enhance.sh - drive Fluid (frame interpolation + upscaling) over any
# video file by piping through ffmpeg and YUV4MPEG2.
#
# The pipeline is usually:  ffmpeg (decode -> y4m) | fluid video (y4m -> y4m) | ffmpeg (y4m -> encode)
# Fluid never touches audio: the encoder stage copies the audio stream from the
# original input file.  When both the input and the output are .y4m or .avi
# files no ffmpeg is needed at all and Fluid reads/writes the files directly
# (.avi is Fluid's built-in MJPEG writer/reader; it has no audio track).
#
# Examples:
#   ./scripts/video-enhance.sh -i in.mp4 -o out.mp4 -m 2 -s 1.5 -f bilinear
#   ./scripts/video-enhance.sh -i in.mkv -o out.mkv -m 2 -c h265 -q 20 -p slow
#   ./scripts/video-enhance.sh -i in.avi -o out_2x.avi -m 2 -s 1.5 -Q 92   # no ffmpeg needed
#   ./scripts/video-enhance.sh -i frames_in_dir -o out.mp4   # (PNG dirs are not supported by this script)
#
# Requirements: POSIX sh, ffmpeg (only when a container format is involved) and
# a built Fluid binary (cargo build --release).
set -eu

PROG=$(basename "$0")

usage() {
    cat <<EOF
$PROG - enhance a video through Fluid (interpolation + upscaling)

Usage:
  $PROG -i INPUT -o OUTPUT [options]

Options:
  -i, --in FILE         input video (.y4m / .avi need no ffmpeg; mp4/mkv/webm/... use ffmpeg)
  -o, --out FILE        output video (.y4m / .avi are built-in and need no ffmpeg)
  -m, --multiplier N    frame-rate multiplier 1..8 (default 2); output fps = input fps * N
  -s, --scale FACTOR    upscale factor, 0 disables upscaling (default 1.0)
  -f, --filter NAME     off|nearest|bilinear|bicubic|lanczos|adaptive|temporal
  -S, --sharpness X     sharpening amount 0..1 (only passed when given)
  -c, --codec NAME      h264 (default), h265 or vp9; .webm forces vp9
  -q, --crf N           encoder quality 0..51 (default: 18 h264, 22 h265, 32 vp9)
  -p, --preset NAME     encoder preset (default medium)
  -b, --bitrate KBPS    target bitrate mode instead of CRF (default 0 = CRF)
  -Q, --jpeg-quality N  MJPEG quality 1..100 for a .avi output (default 88)
      --fluid PATH      Fluid binary (default: ../target/release/fluid, then PATH)
      --ffmpeg PATH     ffmpeg binary (default: your PATH, then common install dirs)
  -h, --help            show this help

Fluid itself reads/writes YUV4MPEG2 (8-bit 4:2:0/4:2:2/4:4:4/mono) and MJPEG
AVI directly, so .y4m / .avi inputs and outputs work with no ffmpeg at all.
Everything else goes through ffmpeg on stdin/stdout.  Output fps is always
input fps * multiplier: mux the result with the original audio and the original
duration, with N times as many frames.  The built-in AVI path has no audio.
EOF
}

info() { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

is_uint() {
    case "$1" in
        '' | *[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

is_number() {
    case "$1" in
        '' | *[!0-9.]* | *.*.*) return 1 ;;
        *) return 0 ;;
    esac
}

# find_ffmpeg: explicit path, then PATH, then common install locations.
find_ffmpeg() {
    if [ -n "$FFMPEG" ]; then
        [ -x "$FFMPEG" ] || die "ffmpeg not found at '$FFMPEG' (from --ffmpeg/FFMPEG)"
        return 0
    fi
    if command -v ffmpeg >/dev/null 2>&1; then
        FFMPEG=$(command -v ffmpeg)
        return 0
    fi
    for cand in /opt/homebrew/bin/ffmpeg /usr/local/bin/ffmpeg /usr/bin/ffmpeg \
        /opt/local/bin/ffmpeg "${PREFIX:-/nonexistent}/bin/ffmpeg" \
        /data/data/com.termux/files/usr/bin/ffmpeg; do
        if [ -x "$cand" ]; then
            FFMPEG=$cand
            return 0
        fi
    done
    return 1
}

ffmpeg_help() {
    cat >&2 <<EOF
error: ffmpeg was not found on this system.

Fluid decodes and encodes no containers on purpose: ffmpeg decodes the input
container to YUV4MPEG2 and encodes the enhanced stream back.  If both sides are
.y4m or .avi, no ffmpeg is needed at all - the built-in AVI writer/reader is used
instead.  Otherwise install it first:

EOF
    case "$(uname -s 2>/dev/null || echo unknown)" in
        Darwin*)
            cat >&2 <<'EOF'
  macOS:          brew install ffmpeg
  or download a static build from https://evermeet.cx/ffmpeg/
EOF
            ;;
        Linux*)
            if [ -n "${PREFIX:-}" ] && [ -d "${PREFIX:-}/bin" ]; then
                cat >&2 <<'EOF'
  Termux/Android: pkg install ffmpeg
EOF
            fi
            cat >&2 <<'EOF'
  Debian/Ubuntu:  sudo apt install ffmpeg
  Fedora:         sudo dnf install ffmpeg
  Arch:           sudo pacman -S ffmpeg
  Alpine:         sudo apk add ffmpeg
  static build:   https://johnvansickle.com/ffmpeg/
EOF
            ;;
        *)
            cat >&2 <<'EOF'
  Install your platform's ffmpeg package, or download a static build from
  https://ffmpeg.org/download.html and put it on PATH.
EOF
            ;;
    esac
    cat >&2 <<EOF

Then run this script again.  If ffmpeg lives somewhere unusual, pass it with
--ffmpeg /path/to/ffmpeg (or set the FFMPEG environment variable).
EOF
}

# find_fluid: explicit path, then the repo target dirs, then PATH.
find_fluid() {
    if [ -n "$FLUID" ]; then
        [ -x "$FLUID" ] || die "fluid binary not found at '$FLUID' (from --fluid/FLUID_EXE)"
        return 0
    fi
    root=$(CDPATH= cd -- "$SCRIPT_DIR/.." 2>/dev/null && pwd) || root=
    if [ -n "$root" ]; then
        for cand in "$root/target/release/fluid" "$root/target/debug/fluid"; do
            if [ -x "$cand" ]; then
                FLUID=$cand
                return 0
            fi
        done
    fi
    if command -v fluid >/dev/null 2>&1; then
        FLUID=$(command -v fluid)
        return 0
    fi
    return 1
}

# ---- defaults -------------------------------------------------------------
IN=
OUT=
MULTIPLIER=2
SCALE=1.0
FILTER=
SHARPNESS=
CODEC=h264
CRF=0
PRESET=medium
BITRATE=0
JPEG_QUALITY=0
FLUID=${FLUID_EXE:-}
FFMPEG=${FFMPEG:-}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# ---- argument parsing -----------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -i | --in | -o | --out | -m | --multiplier | -s | --scale | -f | --filter | \
        -S | --sharpness | -c | --codec | -q | --crf | -p | --preset | -b | --bitrate | \
        -Q | --jpeg-quality | --fluid | --ffmpeg)
            [ $# -ge 2 ] || die "option $1 needs a value"
            opt=$1
            val=$2
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ $# -gt 0 ]; do
                if [ -z "$IN" ]; then
                    IN=$1
                elif [ -z "$OUT" ]; then
                    OUT=$1
                else
                    die "unexpected argument: $1"
                fi
                shift
            done
            continue
            ;;
        -*)
            die "unknown option: $1 (try --help)"
            ;;
        *)
            if [ -z "$IN" ]; then
                IN=$1
            elif [ -z "$OUT" ]; then
                OUT=$1
            else
                die "unexpected argument: $1"
            fi
            shift
            continue
            ;;
    esac

    case "$opt" in
        -i | --in) IN=$val ;;
        -o | --out) OUT=$val ;;
        -m | --multiplier) MULTIPLIER=$val ;;
        -s | --scale) SCALE=$val ;;
        -f | --filter) FILTER=$val ;;
        -S | --sharpness) SHARPNESS=$val ;;
        -c | --codec) CODEC=$val ;;
        -q | --crf) CRF=$val ;;
        -p | --preset) PRESET=$val ;;
        -b | --bitrate) BITRATE=$val ;;
        -Q | --jpeg-quality) JPEG_QUALITY=$val ;;
        --fluid) FLUID=$val ;;
        --ffmpeg) FFMPEG=$val ;;
    esac
done

# ---- validation -----------------------------------------------------------
[ -n "$IN" ] || die "missing input: use -i INPUT (try --help)"
[ -n "$OUT" ] || die "missing output: use -o OUTPUT (try --help)"
[ -f "$IN" ] || die "input file not found: $IN"

is_uint "$MULTIPLIER" || die "--multiplier must be an integer 1..8, got '$MULTIPLIER'"
[ "$MULTIPLIER" -ge 1 ] && [ "$MULTIPLIER" -le 8 ] ||
    die "--multiplier must be between 1 and 8, got '$MULTIPLIER'"

is_number "$SCALE" || die "--scale must be a number >= 0, got '$SCALE'"
is_uint "$CRF" || die "--crf must be an integer 0..51, got '$CRF'"
[ "$CRF" -le 51 ] || die "--crf must be between 0 and 51, got '$CRF'"
is_uint "$BITRATE" || die "--bitrate must be a whole number of kbps, got '$BITRATE'"
is_uint "$JPEG_QUALITY" || die "--jpeg-quality must be an integer 0..100, got '$JPEG_QUALITY'"
[ "$JPEG_QUALITY" -le 100 ] || die "--jpeg-quality must be between 0 and 100, got '$JPEG_QUALITY'"

if [ -n "$SHARPNESS" ]; then
    is_number "$SHARPNESS" || die "--sharpness must be a number 0..1, got '$SHARPNESS'"
fi

if [ -n "$FILTER" ]; then
    case "$FILTER" in
        off | nearest | bilinear | bicubic | lanczos | adaptive | temporal) ;;
        *) die "unknown --filter '$FILTER' (off|nearest|bilinear|bicubic|lanczos|adaptive|temporal)" ;;
    esac
fi

case "$CODEC" in
    h264 | h265 | vp9) ;;
    *) die "unknown --codec '$CODEC' (h264|h265|vp9)" ;;
esac

case "$PRESET" in
    ultrafast | superfast | veryfast | faster | fast | medium | slow | slower | veryslow | placebo) ;;
    *) die "unknown --preset '$PRESET'" ;;
esac

in_lower=$(lower "$IN")
out_lower=$(lower "$OUT")
# .y4m and .avi are read/written by Fluid itself; everything else needs ffmpeg.
in_native=0
out_native=0
case "$in_lower" in *.y4m | *.avi) in_native=1 ;; esac
case "$out_lower" in *.y4m | *.avi) out_native=1 ;; esac

if [ "$JPEG_QUALITY" -gt 0 ]; then
    case "$out_lower" in
        *.avi) ;;
        *) die "--jpeg-quality only applies to a .avi output" ;;
    esac
fi

case "$out_lower" in
    *.webm)
        if [ "$CODEC" != vp9 ]; then
            warn "webm output: switching the encoder to vp9"
            CODEC=vp9
        fi
        ;;
esac

out_dir=$(dirname -- "$OUT")
[ -d "$out_dir" ] || die "output directory does not exist: $out_dir"

# Fluid and ffmpeg pick the writer/container from the file extension, so the
# temporary output keeps the final extension and only the stem is decorated.
out_base=$(basename -- "$OUT")
case "$out_base" in
    *.*) out_stem=${out_base%.*} ; out_ext=${out_base##*.} ;;
    *) out_stem=$out_base ; out_ext= ;;
esac
if [ -n "$out_ext" ]; then
    TMP_OUT="$out_dir/.$out_stem.fluid-part-$$.$out_ext"
else
    TMP_OUT="$out_dir/.$out_stem.fluid-part-$$"
fi

# ---- resolve tools --------------------------------------------------------
find_fluid || die "fluid binary not found: build it with 'cargo build --release' or pass --fluid /path/to/fluid"

if [ "$in_native" -eq 1 ] && [ "$out_native" -eq 1 ]; then
    NEED_FFMPEG=0
else
    NEED_FFMPEG=1
fi

if [ "$NEED_FFMPEG" -eq 1 ]; then
    if ! find_ffmpeg; then
        NEEDED=
        [ "$in_native" -eq 0 ] && NEEDED="decode '$IN'"
        [ "$out_native" -eq 0 ] && NEEDED="${NEEDED:+$NEEDED and }encode '$OUT'"
        ffmpeg_help
        printf 'ffmpeg is required to %s here; use .y4m or .avi for both sides to avoid it.\n' "$NEEDED" >&2
        exit 1
    fi
fi

# ---- derived options ------------------------------------------------------
FLUID_ARGS="--multiplier $MULTIPLIER --scale $SCALE"
[ -n "$FILTER" ] && FLUID_ARGS="$FLUID_ARGS --filter $FILTER"
[ -n "$SHARPNESS" ] && FLUID_ARGS="$FLUID_ARGS --sharpness $SHARPNESS"
if [ "$JPEG_QUALITY" -gt 0 ]; then
    FLUID_ARGS="$FLUID_ARGS --jpeg-quality $JPEG_QUALITY"
fi

info 'Fluid video enhance'
info "  input    : $IN"
info "  output   : $OUT"
info "  fluid    : $FLUID"
[ "$NEED_FFMPEG" -eq 1 ] && info "  ffmpeg   : $FFMPEG"
if [ "$MULTIPLIER" -eq 1 ]; then
    info "  frames   : x1 (off)"
else
    info "  frames   : x$MULTIPLIER (interpolation)"
fi
info "  scale    : x$SCALE"
if [ "$out_native" -eq 1 ]; then
    case "$out_lower" in
        *.avi)
            q=$JPEG_QUALITY
            [ "$q" -eq 0 ] && q=88
            info "  codec    : MJPEG AVI quality $q (no audio)"
            ;;
        *)
            info '  encode   : none (built-in writer)'
            ;;
    esac
fi

cleanup() {
    rm -rf -- "$TMP_OUT"
}
trap cleanup EXIT HUP INT TERM

if [ "$out_native" -eq 1 ]; then
    if [ "$in_native" -eq 1 ]; then
        info '  pipeline : fluid only (built-in readers/writers, no ffmpeg)'
        info 'running fluid ...'
        "$FLUID" video -i "$IN" -o "$TMP_OUT" $FLUID_ARGS
    else
        info '  pipeline : ffmpeg decode | fluid'
        info 'running ffmpeg | fluid ...'
        "$FFMPEG" -hide_banner -nostdin -loglevel error -i "$IN" -map 0:v:0 \
            -f yuv4mpegpipe -pix_fmt yuv420p - |
            "$FLUID" video -i - -o "$TMP_OUT" $FLUID_ARGS
    fi
    [ -s "$TMP_OUT" ] || die 'pipeline produced no output'
else
    default_crf=18
    [ "$CODEC" = h265 ] && default_crf=22
    [ "$CODEC" = vp9 ] && default_crf=32
    [ "$CRF" -eq 0 ] && CRF=$default_crf

    case "$CODEC" in
        h264)
            CODEC_ARGS="-c:v libx264 -preset $PRESET -crf $CRF"
            ;;
        h265)
            CODEC_ARGS="-c:v libx265 -preset $PRESET -crf $CRF"
            case "$out_lower" in
                *.mp4 | *.m4v | *.mov) CODEC_ARGS="$CODEC_ARGS -tag:v hvc1" ;;
            esac
            ;;
        vp9)
            CODEC_ARGS="-c:v libvpx-vp9 -deadline good -cpu-used 2 -row-mt 1 -b:v 0 -crf $CRF"
            ;;
    esac

    if [ "$BITRATE" -gt 0 ]; then
        case "$CODEC" in
            h264)
                CODEC_ARGS="-c:v libx264 -preset $PRESET -b:v ${BITRATE}k -maxrate ${BITRATE}k -bufsize $((BITRATE * 2))k"
                ;;
            h265)
                CODEC_ARGS="-c:v libx265 -preset $PRESET -b:v ${BITRATE}k -maxrate ${BITRATE}k -bufsize $((BITRATE * 2))k"
                case "$out_lower" in
                    *.mp4 | *.m4v | *.mov) CODEC_ARGS="$CODEC_ARGS -tag:v hvc1" ;;
                esac
                ;;
            vp9)
                CODEC_ARGS="-c:v libvpx-vp9 -deadline good -cpu-used 2 -row-mt 1 -b:v ${BITRATE}k"
                ;;
        esac
    fi

    case "$out_lower" in
        *.webm) AUDIO_ARGS='-c:a libopus -b:a 160k' ;;
        *) AUDIO_ARGS='-c:a copy' ;;
    esac
    MUX_ARGS=
    case "$out_lower" in
        *.mp4 | *.m4v | *.mov) MUX_ARGS='-movflags +faststart' ;;
    esac

    case "$CODEC" in
        h264) info "  encode   : h264 crf $CRF preset $PRESET" ;;
        h265) info "  encode   : h265 crf $CRF preset $PRESET" ;;
        vp9) info "  encode   : vp9 crf $CRF" ;;
    esac
    [ "$BITRATE" -gt 0 ] && info "  bitrate  : ${BITRATE} kbps mode"

    if [ "$in_native" -eq 1 ]; then
        info 'running fluid | ffmpeg ...'
        "$FLUID" video -i "$IN" -o - $FLUID_ARGS |
            "$FFMPEG" -hide_banner -loglevel error -y -f yuv4mpegpipe -i - -i "$IN" \
                -map 0:v:0 -map 1:a? $CODEC_ARGS -pix_fmt yuv420p $AUDIO_ARGS \
                -map_metadata 1 -map_chapters 1 $MUX_ARGS "$TMP_OUT"
    else
        info 'running ffmpeg | fluid | ffmpeg ...'
        "$FFMPEG" -hide_banner -nostdin -loglevel error -i "$IN" -map 0:v:0 \
            -f yuv4mpegpipe -pix_fmt yuv420p - |
            "$FLUID" video -i - -o - $FLUID_ARGS |
            "$FFMPEG" -hide_banner -loglevel error -y -f yuv4mpegpipe -i - -i "$IN" \
                -map 0:v:0 -map 1:a? $CODEC_ARGS -pix_fmt yuv420p $AUDIO_ARGS \
                -map_metadata 1 -map_chapters 1 $MUX_ARGS "$TMP_OUT"
    fi
    [ -s "$TMP_OUT" ] || die 'pipeline produced no output'
fi

mv -f -- "$TMP_OUT" "$OUT"
trap - EXIT HUP INT TERM
info "wrote $OUT ($(wc -c <"$OUT" | tr -d ' ') bytes)"