#[derive(Debug, thiserror::Error)]
pub enum GfxError {
    #[error("Windows API 错误: {0}")]
    Windows(#[from] windows::core::Error),
    #[error("{0}")]
    Other(String),
}
