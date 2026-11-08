//! Error type shared by every Fluid crate.

use std::fmt;

/// Result alias used across Fluid.
pub type Result<T> = std::result::Result<T, FluidError>;

/// Every failure surfaced by Fluid.
#[derive(Debug)]
pub enum FluidError {
    /// Caller passed something invalid (bad dimensions, bad range, ...).
    InvalidArgument(String),
    /// Filesystem / stream failure.
    Io(std::io::Error),
    /// A frame/stream was not in the expected format.
    InvalidFormat(String),
    /// Requested feature is not compiled in or not supported on this platform.
    Unsupported(String),
    /// Operating-system level failure (DXGI, D3D11, windowing, ...).
    Platform(String),
    /// Something that should never happen did happen.
    Internal(String),
    /// Configuration file problem.
    Config(String),
    /// Operation was cancelled by the user or by shutdown.
    Cancelled,
}

impl FluidError {
    pub fn invalid(msg: impl Into<String>) -> Self {
        FluidError::InvalidArgument(msg.into())
    }
    pub fn format(msg: impl Into<String>) -> Self {
        FluidError::InvalidFormat(msg.into())
    }
    pub fn unsupported(msg: impl Into<String>) -> Self {
        FluidError::Unsupported(msg.into())
    }
    pub fn platform(msg: impl Into<String>) -> Self {
        FluidError::Platform(msg.into())
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        FluidError::Internal(msg.into())
    }
    pub fn config(msg: impl Into<String>) -> Self {
        FluidError::Config(msg.into())
    }
    pub fn cancelled() -> Self {
        FluidError::Cancelled
    }
}

impl fmt::Display for FluidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FluidError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            FluidError::Io(e) => write!(f, "io error: {e}"),
            FluidError::InvalidFormat(m) => write!(f, "invalid format: {m}"),
            FluidError::Unsupported(m) => write!(f, "unsupported: {m}"),
            FluidError::Platform(m) => write!(f, "platform error: {m}"),
            FluidError::Internal(m) => write!(f, "internal error: {m}"),
            FluidError::Config(m) => write!(f, "config error: {m}"),
            FluidError::Cancelled => write!(f, "operation cancelled"),
        }
    }
}

impl std::error::Error for FluidError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FluidError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for FluidError {
    fn from(v: std::io::Error) -> Self {
        FluidError::Io(v)
    }
}

impl From<String> for FluidError {
    fn from(v: String) -> Self {
        FluidError::InvalidArgument(v)
    }
}

impl From<&str> for FluidError {
    fn from(v: &str) -> Self {
        FluidError::InvalidArgument(v.to_string())
    }
}
