# Fluid 依赖库选型（为什么用这些库，而不是手搓）

Fluid 的原则是：**算法和产品是我们自己的，底座一律用生态里最强最成熟的库**。
凡是需要长期维护、跨平台兼容、硬件驱动适配的部分，都不自己手写；
这样我们能把精力放在插帧、超分、调度、配置、UI 这些真正决定体验的地方。

## 运行时依赖

| 领域 | 依赖 | 版本 | 为什么选它 |
|---|---|---|---|
| GPU 计算 | `wgpu` | 30.0.1 | Rust 生态事实标准的 GPU 抽象层；一套 compute shader 同时跑 Windows 和 Android（都是 Vulkan/GLES）；全部是安全 Rust 接口，不需要写 unsafe 的 D3D11/HLSL。当前 Windows 实际走 Vulkan；D3D12 后端因为本仓库 `Cargo.lock` 里 `windows` 版本与 capture 侧冲突而暂时关闭（对齐版本即可打开，与内核无关） |
| 桌面/移动 UI | `eframe` + `egui` | 0.37 | 纯 Rust 立即模式 GUI，Windows 开箱即用，Android 走 android-game-activity 后端；中文字体运行期加载系统字体，不用打包十几 MB 字形 |
| 窗口/输入 | `winit` | 0.30.13 | eframe 官方使用的版本；Android 上负责把 Activity 和 Surface 生命周期交给 Rust |
| CPU 向量 | `wide` | 1.7.1 | 用安全包装的 SIMD  intrinsics（f32x4 等），在 `#![forbid(unsafe_code)]` 的前提下拿到 1.6-1.8x 热点加速 |
| Windows 系统调用 | `windows` | 0.58 | 微软官方 Rust 绑定：DXGI 桌面复制、DWM、窗口、控制台 Ctrl+C |
| Android 系统调用 | `jni` 0.22 / `ndk-sys` 0.6 | MediaProjection 屏幕捕获（ImageReader/HardwareBuffer）与 ANativeWindow 呈现 |
| 图片编解码 | `image` | 0.25 | PNG/JPEG/BMP 序列读写，只开需要的 feature |
| 配置/命令行 | `serde`、`toml`、`serde_json`、`clap` | 4 | 配置、报告、CLI 都是声明式的，改字段不用写解析代码 |
| 字节搬运 | `bytemuck` | 1 | GPU 缓冲区收发的安全类型转换 |

## 这些部分我们不再自己实现

- **GPU 后端**：早期版本曾计划手写 Direct3D 11 + 运行期编译 HLSL，实现到一半果断换成 wgpu。理由：手写 D3D11 只能覆盖 Windows，Android 还要再写一遍 Vulkan；而 wgpu 一份代码两端通吃，后续加 Metal/WebGPU 也不用重写内核。旧方案已删除。
- **UI 框架**：没有自己用 win32 画控件。eframe/egui 自带布局、滚动、下拉、滑块、多视口（全屏输出窗口）、DPI 缩放、触屏输入，桌面和手机共用同一套页面代码。
- **图像编解码**：没有手写 PNG 压缩，用 `image`。
- **屏幕捕获/呈现的系统调用**：没有手写汇编或 FFI 声明，全部走 `windows` / `jni` / `ndk-sys` 的官方绑定。
- **SIMD**：没有写 `core::arch` 的 unsafe intrinsics，用 `wide` 的安全类型。
- **GPU 内核调度**：没有自己管理 D3D/Vulkan 的队列、同步、描述符堆，交给 wgpu 的 `Device`/`Queue`/`ComputePipeline`；我们只写 WGSL 内核和缓冲区布局。放大/锐化/时域混合与插帧（金字塔匹配、LK 精化、遮挡填充、置信度、合成）都是这套调度上的 shader，共 20+ 个 WGSL 文件。
- **窗口捕获**：没有自己 hook 游戏渲染，用系统自带的 Windows Graphics Capture（`windows` crate 绑定）拿窗口内容，和 OBS 同类 API。

> 唯一保留的“手搓”是 y4m 读写（视频管线里最简单的一种裸流格式，`ffmpeg` 交接用）。真正的容器/编码交给 ffmpeg（`scripts/video-enhance.*`），不自己实现。

## 我们自己保留的核心（不可替代的部分）

- 光流估计（金字塔块匹配 + Lucas-Kanade 亚像素精修 + 前后向一致性置信度）
- 运动补偿插帧（遮挡填充、场景切换检测、时域混合）
- 超分内核（nearest/bilinear/bicubic/Lanczos-3/Adaptive 细节恢复、RCAS 风格锐化、时域 EMA）及其与 GPU 严格对齐的采样约定
- 流水线调度（捕获→处理→呈现、节流、延迟统计、取消、后端回退）
- 产品层（预设、配置持久化、体检推荐、视频处理、中文界面）

## 体积/性能权衡

引入 wgpu + eframe 让二进制从约 3 MB 涨到约 11 MB（release，已 strip），编译时间变长，
但换来的是：手机端和桌面端共用全部算法与界面、GPU 加速成为默认路径、后续换渲染 API 不需要重写内核。
对一个 DLSS/FSR 类软件来说，这笔交换是值得的——用户感知不到 8 MB，但感知得到 30 fps 和 120 fps 的差别。
