/// 两个 QPC 计数之间的微秒数。
pub fn qpc_diff_us(start: i64, end: i64, freq: u64) -> u64 {
    if freq == 0 || end <= start {
        return 0;
    }
    let diff = end as i128 - start as i128;
    if diff <= 0 {
        return 0;
    }
    let us = diff * 1_000_000 / freq as i128;
    us.clamp(0, u64::MAX as i128) as u64
}
