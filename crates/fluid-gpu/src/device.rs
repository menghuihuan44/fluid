//! wgpu instance, adapter selection and device creation.
//!
//! The backend is built on wgpu so the same compute kernels run on Direct3D 12,
//! Vulkan and OpenGL ES: the Windows desktop, phones and any other wgpu target
//! share one code path. Everything in this module is safe Rust; the wgpu API
//! validates shaders, bind groups and dispatches for us.
//!
//! Device creation never panics: a missing driver, an adapter without a
//! compute-capable backend or a rejected device returns
//! [`FluidError::Unsupported`]/[`FluidError::Platform`], which lets the engine
//! fall back to the CPU upscaler on machines without a usable GPU.

use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use fluid_core::{FluidError, Result};
use wgpu::{Adapter, AdapterInfo, Backend, DeviceType, Instance, InstanceDescriptor};

use crate::GpuInfo;

/// Last uncaptured wgpu error, shared between the device handler and the caller.
///
/// wgpu's default handler panics, which would abort a game session. Installing a
/// handler that records the message instead lets [crate::GpuUpscaler::upscale]
/// turn a dispatch-time validation error into a normal [FluidError].
pub(crate) type ErrorSlot = Arc<Mutex<Option<String>>>;

/// Take the first uncaptured wgpu error recorded by the device handler.
pub(crate) fn take_error(slot: &ErrorSlot) -> Option<String> {
    slot.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
}

/// Minimal executor for the few wgpu futures used at startup.
///
/// They complete through the waker on the calling thread, so a park/unpark loop
/// is enough and no async runtime dependency is needed. Everything after device
/// creation is synchronous (submit + poll), which keeps frame latency honest.
pub(crate) fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    struct ThreadWaker(std::thread::Thread);

    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut fut = Box::pin(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// A live instance; adapters borrow from it through their own references.
fn instance() -> Instance {
    Instance::new(InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..InstanceDescriptor::new_without_display_handle()
    })
}

/// Ranking key: prefer real GPUs, and on the same device prefer Direct3D 12
/// over Vulkan over GL.
///
/// D3D12 is the primary Windows desktop path (it is the backend that ships with
/// every current Windows GPU driver); Vulkan stays the shared desktop/Android
/// path and the automatic fallback when a D3D12 adapter cannot create a device.
/// DXGI enumeration only produces D3D12 adapters on Windows, so this ordering is
/// platform-neutral.
fn rank(info: &AdapterInfo) -> (u8, u8) {
    let device = match info.device_type {
        DeviceType::DiscreteGpu => 0,
        DeviceType::IntegratedGpu => 1,
        DeviceType::VirtualGpu => 2,
        DeviceType::Other => 3,
        DeviceType::Cpu => 4,
    };
    let backend = match info.backend {
        Backend::Dx12 => 0,
        Backend::Vulkan => 1,
        Backend::Gl => 2,
        _ => 3,
    };
    (device, backend)
}

/// All adapters visible to the process, best candidate first.
fn adapters(instance: &Instance) -> Vec<Adapter> {
    let mut found = block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    found.sort_by_key(|adapter| rank(&adapter.get_info()));
    found
}

/// Stable adapter class name used by the engine's backend policy.
fn device_type_str(kind: DeviceType) -> &'static str {
    match kind {
        DeviceType::DiscreteGpu => "discrete",
        DeviceType::IntegratedGpu => "integrated",
        DeviceType::VirtualGpu => "virtual",
        DeviceType::Cpu => "cpu",
        DeviceType::Other => "other",
    }
}

/// Build the public [`GpuInfo`] for one adapter.
fn info_of(adapter: &Adapter) -> GpuInfo {
    let info = adapter.get_info();
    let limits = adapter.limits();
    GpuInfo {
        device_type: device_type_str(info.device_type),
        name: info.name,
        backend: info.backend.to_str(),
        vendor_id: info.vendor,
        max_feature_level: format!(
            "wgpu limits: buffer {} MiB, storage binding {} MiB, workgroup {}x{}x{}",
            limits.max_buffer_size / (1024 * 1024),
            limits.max_storage_buffer_binding_size / (1024 * 1024),
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        ),
    }
}

/// Find the best adapter without creating a device.
///
/// Useful for capability reporting and for the engine to decide whether to ask
/// for the GPU backend at all, without paying for device creation.
pub(crate) fn probe_info() -> Result<GpuInfo> {
    let instance = instance();
    let found = adapters(&instance);
    match found.first() {
        Some(adapter) => Ok(info_of(adapter)),
        None => Err(FluidError::unsupported(
            "gpu: no wgpu adapter found (install a Vulkan, D3D12 or GLES driver)",
        )),
    }
}

/// Create a device and queue on the best adapter that accepts one.
///
/// Every candidate adapter is tried in rank order, so a broken Vulkan driver
/// falls back to Direct3D 12 and finally to the software rasteriser instead of
/// failing outright. Returns the device, its queue, the adapter description and
/// the slot where uncaptured validation errors are recorded.
pub(crate) fn create_device() -> Result<(wgpu::Device, wgpu::Queue, GpuInfo, ErrorSlot)> {
    let instance = instance();
    let found = adapters(&instance);
    if found.is_empty() {
        return Err(FluidError::unsupported(
            "gpu: no wgpu adapter found (install a Vulkan, D3D12 or GLES driver)",
        ));
    }
    let mut last_error = String::from("no adapter accepted a device");
    for adapter in &found {
        let info = info_of(adapter);
        let desc = wgpu::DeviceDescriptor {
            label: Some("fluid-gpu"),
            ..Default::default()
        };
        match block_on(adapter.request_device(&desc)) {
            Ok((device, queue)) => {
                let errors: ErrorSlot = Arc::new(Mutex::new(None));
                let slot = Arc::clone(&errors);
                device.on_uncaptured_error(Arc::new(move |error| {
                    let mut slot = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    if slot.is_none() {
                        *slot = Some(error.to_string());
                    }
                }));
                return Ok((device, queue, info, errors));
            }
            Err(error) => {
                last_error = format!("{} via {}: {error}", info.name, info.backend);
            }
        }
    }
    Err(FluidError::unsupported(format!(
        "gpu: no usable wgpu compute device ({last_error})"
    )))
}
