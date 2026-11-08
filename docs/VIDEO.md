# Fluid 视频增强指南（离线视频模式）

本文档说明如何用 Fluid 对已有视频做**补帧（插帧）**和**超分辨率放大**，
包括手工的 ffmpeg 管道命令、`scripts/video-enhance.ps1` / `scripts/video-enhance.sh`
一键脚本、画质参数选择以及常见问题排查。

> Fluid 内置了 **MJPEG AVI 读写**：`.avi` 可以直接作为 `fluid video` 的输入和输出，
> 完全不需要 ffmpeg（见第 3 节），适合没有 ffmpeg 的 Windows 机器或离线场景。
> 其他容器（mp4/mkv/webm/…）仍然刻意不做内置编解码：Fluid 通过未压缩的
> 8-bit YUV4MPEG2（y4m）管道与 ffmpeg 对接，容器解析、解码、封装、编码全部交给
> ffmpeg，所以能把 Fluid 插到 ffmpeg 能读的任意格式上，而不需要重新编译。

## 1. 为什么内部用 y4m

- y4m 是逐帧的裸 YUV 流：`YUV4MPEG2 W.. H.. F<帧率分子>:<分母> I.. A.. C<色度格式>`，
  每帧前有一个 `FRAME` 标记行，帧内按平面依次排列 Y、Cb、Cr。
- 它不做压缩，所以不会在解码到 Fluid 的路上引入第二代压缩损失；管道式读写，
  不需要把整段视频落盘。
- 代价是体积很大：1280x720、4:2:0、60 fps 的 y4m 每 90 帧约 124 MB。
- Fluid 处理的是 **8-bit BT.709 limited-range RGB/YUV**，读到 y4m 后转成 Rgb8，
  写回时再按 BT.709 转换；一次往返的误差约 ±2 个码值。
- 10-bit y4m（`C420p10` / `C422p10` / `C444p10` / `mono10`）可以直接作为输入：
  读的时候把每个样点右移 2 位（带舍入）截断成 8-bit，再走上面的转换。这只是
  位深截断，**不是** HDR tone mapping，输出仍然是 8-bit。
- 内置的 `.avi` 读写用 MJPEG（每帧一张 JPEG）封装在 RIFF/AVI 容器里，内部同样转成
  Rgb8 处理；它是有损的，但整条链路不需要 ffmpeg。

## 2. 支持的格式

| 位置 | 支持 |
| --- | --- |
| 输入（内置，无需 ffmpeg） | `.y4m`、`.avi`（MJPEG）、PNG 序列目录 |
| 输入（经 ffmpeg） | mp4 / mkv / mov / webm / avi / m4v / ts / flv / …（取决于你的 ffmpeg 构建） |
| 输入像素格式 | ffmpeg 会先转成 `yuv420p`；Fluid 自身支持 y4m 的 C420 / C422 / C444 / mono，以及 10-bit 的 C420p10 / C422p10 / C444p10 / mono10（读入时截断成 8-bit） |
| 输入位深 | 8-bit 与 10-bit y4m 都支持；10-bit 只做 `>>2` 截断，不做 HDR tone mapping；输出永远是 8-bit |
| 输入图片序列 | 目录：`png` / `jpg` / `jpeg` / `bmp`，按自然数字顺序（shot2 < shot10） |
| 输出（内置，无需 ffmpeg） | `.y4m`、`.avi`（MJPEG，`--jpeg-quality` 1..100，默认 88）、PNG 序列目录 |
| 输出（经 ffmpeg） | 任意 ffmpeg 封装格式（mp4/mkv/webm/…） |
| 音频 | Fluid 不接触音频；ffmpeg 路径从原始文件复制或重新编码，内置 AVI 路径没有音轨 |

注意：y4m 的 4:2:0 / 4:2:2 要求宽高为偶数；奇数尺寸只能用 C444 / mono。
奇数尺寸的源请先让 ffmpeg 缩放到偶数尺寸，或直接输出 4:4:4；内置的 AVI 路径没有
偶数尺寸限制。

## 3. 没有 ffmpeg 也能用：内置 AVI

如果机器上没有 ffmpeg（例如不能随意装软件的 Windows 工作机），Fluid 自带 MJPEG
AVI 读写器，可以完全不经过 ffmpeg 完成补帧和放大：

```powershell
.\target\release\fluid.exe video -i clip.avi -o clip_2x.avi --multiplier 2 --scale 1.5 `
    --filter bicubic --sharpness 0.25 --jpeg-quality 92
```

- 输入可以是 `.avi` 或 `.y4m`，输出也可以是 `.avi` 或 `.y4m`，四种组合都不需要 ffmpeg。
- AVI 内部是 **MJPEG**（每帧一张 JPEG）：`--jpeg-quality 1..100` 控制画质（默认 88），
  有损、体积比 H.264/H.265 大很多，但换来零外部依赖。
- AVI 路径**没有音轨**（Fluid 只处理视频帧）；需要保留音频请用 ffmpeg 脚本（第 6 节）。
- 单个 AVI 文件上限约 4 GiB（RIFF 结构限制），超长视频请分段处理。

只用 Fluid 的三条命令示例（提取 → 转换 → 导出）：

```powershell
# 1) 增强 AVI 并直接导出 AVI（最常用的 ffmpeg-free 路径）
.\target\release\fluid.exe video -i clip.avi -o clip_2x.avi --multiplier 2 --scale 1.5 --filter lanczos --jpeg-quality 92

# 2) 导出无损 y4m（体积大，适合作为中间产物或再次处理）
.\target\release\fluid.exe video -i clip_2x.avi -o clip_2x.y4m --multiplier 1 --scale 1.0

# 3) 把 y4m 再导出成 AVI（例如交给另一台没有 ffmpeg 的机器）
.\target\release\fluid.exe video -i clip_2x.y4m -o clip_2x.avi --multiplier 1 --scale 1.0 --jpeg-quality 90
```

- 什么时候仍然应该用 ffmpeg 脚本：需要 H.264/H.265 的小体积输出、需要音轨、需要
  mp4/mkv/webm 等其他容器、需要 HDR tone-map，或者单个文件超过 4 GiB。
- 两个一键脚本也会自动走这条内置路径：输入、输出都是 `.avi`/`.y4m` 时完全不调用
  ffmpeg，并打印 `built-in ... no ffmpeg` 的提示，例如
  `\scripts\video-enhance.ps1 -In clip.avi -Out clip_2x.avi -Multiplier 2 -Scale 1.5`。

## 4. 快速开始

### 4.1 一键脚本（推荐）

Windows (PowerShell 5.1)：

```powershell
.\scripts\video-enhance.ps1 -In  input.mp4 -Out output.mp4 -Multiplier 2 -Scale 1.5 -Filter bilinear
```

Linux / macOS / Termux：

```sh
./scripts/video-enhance.sh -i input.mp4 -o output.mp4 -m 2 -s 1.5 -f bilinear
```

两个脚本行为一致：自动寻找 ffmpeg 与 `target/release/fluid`，解码 → 管道 → 编码，
并把原始音轨复制进输出。当输入、输出都是 `.y4m` 或 `.avi` 时会完全跳过 ffmpeg
（走第 3 节的内置路径）；只有至少一侧是其他容器时才需要 ffmpeg。如果系统里没有
ffmpeg 而又必须解码/编码其他容器，脚本会打印各平台的安装命令并说明是哪一侧需要它，
然后退出，不会留下半成品文件。

常用参数：`-Multiplier/-m`（补帧倍数 1..8）、`-Scale/-s`（放大倍数，
0 表示不放大）、`-Filter/-f`（放大滤波）、`-Sharpness/-S`（锐化 0..1）、
`-Codec/-c`（h264/h265/vp9）、`-Crf/-q`、`-Preset/-p`、`-Bitrate/-b`、
`-JpegQuality/-Q`（仅 `.avi` 输出：内置 MJPEG 画质 1..100，默认 88），
以及 `--fluid` / `--ffmpeg` 手动指定可执行文件。

### 4.2 手工管道命令

bash / sh：

```sh
ffmpeg -hide_banner -nostdin -i input.mp4 -map 0:v:0 -f yuv4mpegpipe -pix_fmt yuv420p - \
  | ./target/release/fluid video -i - -o - --multiplier 2 --scale 1.5 --filter bilinear \
  | ffmpeg -hide_banner -y -f yuv4mpegpipe -i - -i input.mp4 \
      -map 0:v:0 -map 1:a? -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
      -c:a copy -movflags +faststart output.mp4
```



三个管道段的职责固定：**ffmpeg 解码 → Fluid 补帧/放大 → ffmpeg 编码**。
Fluid 的 stdout 只输出 y4m 二进制，日志和统计一律走 stderr，所以可以放心接管道；
PowerShell 里请用 `cmd /c` 或上面的 .ps1 脚本发起管道，不要用 `Write-Output` 中转。

## 5. 帧率语义：输出 fps = 输入 fps × 倍数

- Fluid 读入 y4m 头里的 `F<num>:<den>` 作为输入帧率，写出的 y4m 头是 `F<num×倍数>:<den>`。
- 例如 29.97 fps（30000/1001）输入、倍数 2，输出就是 59.94 fps（60000/1001，
  Fluid 保持有理数帧率，不会累积漂移）。
- 音频没有跟着变：音轨仍是原始时长。因此**必须**按“帧数翻倍、音频不变”的方式封装，
  播放器会把 2 倍帧数的视频按 2 倍 fps 播放，时长与音轨一致。
- 不要在编码后再次改动 `-r` / `-vf fps=`，否则会重复丢帧或重复插帧，造成音画不同步。
- ffmpeg 解码默认输出恒定帧率（CFR）帧序列；VFR 源（手机录像、屏幕录制）会先被
  转成平均帧率的 CFR，时间戳的细微差异不会保留。需要精确保留时间戳的场景请先
  用 ffmpeg 把源转成 CFR。

## 6. 画质参数与推荐组合

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `--multiplier N` | 补帧倍数 1..8，输出帧率 ×N | 24/30 → 2；60 → 2（→120）；慢动作素材 4/8 |
| `--scale X` | 超分倍数，`0` 关闭，`1.0` 表示只做补帧 | 720p→1440p 用 2；1080p→4K 用 2；嫌慢用 1.5 |
| `--target WxH` | 直接指定输出尺寸 | 需要精确 3840x2160 等输出时使用 |
| `--filter F` | 放大滤波 off/nearest/bilinear/bicubic/lanczos/adaptive/temporal | 动画/线条→adaptive；实拍→bicubic 或 lanczos；像素风/UI→nearest |
| `--sharpness 0..1` | 放大后锐化 | 实拍 0.2~0.4；动画 0.3~0.5；录屏 0~0.2 |
| `--interp` | off/fixed/adaptive 补帧策略 | 默认 fixed（SAD 光流+双向插值）；镜头切换多的素材用 adaptive |
| `--preset` | 整体速度/质量档位 | 实时预览 performance；离线成片 quality 或 ultra-quality |

推荐组合：

- **2D 动画 24/30 fps → 60 fps**：`--multiplier 2 --scale 0 --filter adaptive --sharpness 0.4 --preset quality`
  （动画线条干净，adaptive 比 bilinear 更少糊边；不需要放大时把 scale 设 0 最省时间）
- **实拍 30 → 60 fps**：`--multiplier 2 --scale 0 --interp fixed --preset balanced`
  若运动很快，可先 `--frames 120 --report report.json` 试跑，观察 `flow_mean_ms`。
- **实拍 1080p → 4K 60 fps**：`--multiplier 2 --scale 2 --filter bicubic --sharpness 0.25`
  单帧成本较高，见第 9 节；建议离线跑，或改用 `--scale 1.5` 先看效果。
- **屏幕录制 / UI 演示**：`--multiplier 2 --scale 0 --filter nearest --sharpness 0`
  （文字和 UI 边缘用 nearest 最锐利；不要开锐化，否则文字出现振铃）
- **游戏/高动态素材 60 → 120 fps**：`--multiplier 2 --scale 0 --interp fixed --preset ultra-performance`

## 7. 音频处理

Fluid 只处理视频帧，音频必须由 ffmpeg 处理（内置 AVI 路径没有音轨，需要音频时必须
用 ffmpeg 脚本或第 4.2 节的手工管道）。脚本与上面的手工命令都是：

- 编码阶段用**第二个输入** `-i input.mp4` 取原音轨：`-map 0:v:0 -map 1:a?`；
- 视频写 `-map 0:v:0`，音频 `-c:a copy`（无损、最快、不变时长）；
- `.webm` 容器不支持直接复制 AAC/MP3，脚本会自动改成 `-c:a libopus -b:a 160k`；
- `.mp4` 里如果原音轨是 Opus/Vorbis，也要改成 `-c:a aac -b:a 192k`。

如果源文件本来就没有音频，`-map 1:a?` 里的 `?` 会静默跳过，不会报错。

## 8. HDR / 10-bit 素材

Fluid 内部是 8-bit BT.709 limited-range，输出也永远是 8-bit SDR。当前状态：

- **10-bit SDR y4m 可以直接读**（`C420p10` / `C422p10` / `C444p10` / `mono10`）：
  读取时把 10-bit 样点截断成 8-bit（0→0、512→128、1023→255，右移 2 位带舍入）。
  没有 tone mapping，也没有抖动；8-bit 显示器上的观感与直接截断一致。
- **HDR（PQ/HLG）源仍应先 tone-map**：截断会丢掉高光、让颜色发灰/偏暗。此时请
  先用 ffmpeg 转成 SDR 8-bit 再交给 Fluid：

```sh
ffmpeg -hide_banner -i hdr_source.mkv \
  -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" \
  -f yuv4mpegpipe -pix_fmt yuv420p - | ./target/release/fluid video -i - -o - --multiplier 2 | ffmpeg -f yuv4mpegpipe -i - ...
```

输出永远是 8-bit SDR（BT.709 limited-range）。需要 HDR 输出、10-bit 输出或保留
HDR 元数据的场景超出 Fluid 当前的能力范围。

## 9. 性能预期

数据来自 README 的基准（Intel i7-1260P 笔记本、release 构建；实际以你的机器为准，
不要用这些数字承诺别人的实时性能）：

- 光流：720p 约 10 ms/帧对，1080p 约 23 ms；2 倍补帧（光流+插值+合成）约 28–48 ms/帧对。
- 放大：720p→1440p bilinear+锐化约 10 ms；1080p→4K bilinear 约 21 ms、adaptive 约 34 ms；
  `--scale 1.0 --sharpness 0` 时放大基本免费（约 1 ms）。
- 端到端：720p 仅放大到 1440p 14.0 ms/47 fps；720p 2× 补帧 + 1.5× bilinear 40.9 ms/42 fps；
  1080p 2× 补帧 + 1.0× 放大 52.7 ms/31 fps；1080p→4K nearest/bilinear/adaptive 分别为
  22.8/29.9/46.6 ms（27/22/16 fps）；1080p 直通 4.0 ms/63 fps。
- 采集回环（真实 1280x720 区域 2× 补帧）约 34 ms/33 fps；2880x1800 2× 补帧约 147 ms/12 fps（偏慢）。

离线视频模式里 `fluid video` 的吞吐就是上面这些单帧成本的倒数，加上 ffmpeg 解码/编码的
时间与管道 I/O。`--report out.json` 会给出 `process_mean_ms`、`flow_mean_ms`、
`interp_mean_ms`、`upscale_mean_ms` 等分项，可用来定位瓶颈。

## 10. 故障排查

- **`ffmpeg: command not found`**：脚本会打印安装命令；也可以 `--ffmpeg /path/to/ffmpeg`。
- **`fluid: error: cannot read stdin: invalid format`**：上游 ffmpeg 没有输出合法 y4m。
  常见原因：解码失败（检查 `-loglevel error` 的 stderr）、上游提前退出、或把日志写进了 stdout。
- **Windows 管道卡住/死锁**：cmd 的管道缓冲加上两个 ffmpeg 同时读写时可能出现。
  用 `scripts/video-enhance.ps1`（它把整条命令写进临时 .cmd 一次性执行），或先分段：
  `ffmpeg ... -f yuv4mpegpipe out.y4m`，再 `fluid video -i out.y4m -o out2.y4m`，最后编码。
- **`Broken pipe` / `The process tried to write to a nonexistent pipe`**：用 `--frames N` 截断
  测试时上游 ffmpeg 还在写而下游已退出，属正常提示，不影响已写出的文件。
- **stdout 混入日志**：Fluid 的日志在 stderr；重定向时写成 `fluid ... > out.y4m 2> log.txt`，
  不要把 `2>&1` 接到 y4m 输出上。
- **奇数宽高报错**：4:2:0/4:2:2 需要偶数尺寸；用 `--filter` 前先 `scale=trunc(iw/2)*2:trunc(ih/2)*2`，
  或让 Fluid 输出 4:4:4（手工管道里 `-pix_fmt yuv444p` 解码不常见，建议直接缩放到偶数）。
- **音画不同步**：不要对输出再套 `-r`/`fps` 滤镜；确认输出 fps 正好是输入 fps × 倍数，
  且音轨是从原文件复制/重编码而来（`-map 1:a?`）。
- **颜色发灰/过曝**：10-bit SDR y4m 现在可以直接读，但只是截断到 8-bit（不做 tone
  mapping），所以 HDR 源请按第 8 节先 tone-map；或源本身是 full-range，需要
  `-vf scale=in_range=full:out_range=tv` 之类先归一到 limited-range。

- **`unsupported: AVI video stream uses the unsupported codec "XXXX"`**：内置读取器只支持
  MJPEG（MJPG）编码的 AVI。DivX/Xvid/H.264 等其他编码的 AVI 需要先用 ffmpeg 转成
  `-c:v mjpeg` 的 AVI，或直接走 ffmpeg 管道。
- **AVI 路径没有声音**：内置 `.avi` 输出只有视频轨；需要保留音频请用 ffmpeg 脚本
  （第 7 节），或手工把 `fluid video` 输出的 y4m 交给 ffmpeg 加上原音轨。
- **AVI 文件超过 4 GiB 报错**：RIFF 结构的上限；请分段处理，或改用 ffmpeg 脚本输出
  mp4/mkv。
- **`invalid AVI: cannot decode JPEG frame at byte offset ...`**：文件被截断或损坏；
  用 ffmpeg 重新封装/转码后再试。

## 11. 已知限制

- 只处理视频帧，不处理字幕、章节（脚本用 `-map_metadata 1 -map_chapters 1` 保留元数据/章节）。
- 没有硬件编码路径；编码速度取决于 ffmpeg 的 libx264/libx265/libvpx 构建。
- y4m 中间数据是未压缩的，1080p 长片会占用大量管道带宽/临时空间；
  超长视频建议分段处理。
- 内置 AVI 是 MJPEG：有损、没有音轨、单文件 < 4 GiB，体积远大于 H.264/H.265；
  只作为“没有 ffmpeg 也能用”的兜底路径。
