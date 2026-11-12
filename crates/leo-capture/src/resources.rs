//! Cooperative limits for native background work. These are admission/pause
//! thresholds, not OS allocation limits: an in-flight geometry operation can
//! exceed the memory threshold. Never discard retained simulation data.
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

pub const GIB: u64 = 1024 * 1024 * 1024;
static WORKERS: AtomicUsize = AtomicUsize::new(0);
static MEMORY: AtomicU64 = AtomicU64::new(0);
static DUTY: AtomicUsize = AtomicUsize::new(75);

pub fn cpu_count() -> usize {
    static CPUS: OnceLock<usize> = OnceLock::new();
    *CPUS.get_or_init(|| std::thread::available_parallelism().map_or(1, usize::from))
}
pub fn worker_limit() -> usize {
    let configured = WORKERS.load(Ordering::Relaxed);
    if configured == 0 {
        cpu_count().saturating_sub(2).clamp(1, 8)
    } else {
        configured.min(cpu_count()).max(1)
    }
}
pub fn set_worker_limit(workers: usize) {
    WORKERS.store(workers.clamp(1, cpu_count()), Ordering::Relaxed);
}
pub fn memory_limit() -> u64 {
    let configured = MEMORY.load(Ordering::Relaxed);
    if configured != 0 {
        return configured;
    }
    16 * GIB
}
pub fn set_memory_limit(bytes: u64) {
    MEMORY.store(bytes.max(GIB / 2), Ordering::Relaxed);
}
pub fn duty_percent() -> usize {
    DUTY.load(Ordering::Relaxed)
}
pub fn set_duty_percent(percent: usize) {
    DUTY.store(percent.clamp(10, 100), Ordering::Relaxed);
}

/// Process footprint on macOS (includes compressed memory), RSS + swap on Linux.
/// Unknown platforms expose no pretend memory measurement.
pub fn process_memory() -> Option<u64> {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut info: libc::rusage_info_v2 = std::mem::zeroed();
        let result = libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V2,
            (&mut info as *mut libc::rusage_info_v2).cast(),
        );
        return (result == 0).then_some(info.ri_phys_footprint);
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let field = |key: &str| {
            status
                .lines()
                .find(|l| l.starts_with(key))?
                .split_whitespace()
                .nth(1)?
                .parse::<u64>()
                .ok()
        };
        return Some((field("VmRSS:")? + field("VmSwap:").unwrap_or(0)) * 1024);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    None
}
pub fn physical_memory() -> Option<u64> {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut bytes: u64 = 0;
        let mut len = std::mem::size_of_val(&bytes);
        let result = libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        );
        return (result == 0 && bytes > 0).then_some(bytes);
    }
    #[cfg(target_os = "linux")]
    {
        return std::fs::read_to_string("/proc/meminfo")
            .ok()?
            .lines()
            .find(|l| l.starts_with("MemTotal:"))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()
            .map(|n| n * 1024);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    None
}
pub fn memory_error(used: Option<u64>, limit: u64) -> Option<String> {
    used.filter(|&n| n >= limit).map(|n| format!(
        "Client memory is {:.2} GiB, at the {:.2} GiB limit. Free memory or raise the limit before continuing.",
        n as f64 / GIB as f64, limit as f64 / GIB as f64))
}
pub fn check_memory() -> Result<(), String> {
    memory_error(process_memory(), memory_limit()).map_or(Ok(()), Err)
}

/// Keep calculation threads below interactive work on macOS. This changes only
/// the calling thread, never the UI thread or the process-wide priority.
pub fn background_thread() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
        }
        let _ = pthread_set_qos_class_self_np(0x11, 0); // QOS_CLASS_UTILITY
    }
}

/// Shared permits prevent simultaneous AOI/job/atlas pools from multiplying
/// CPU use. Pool build callbacks must not recursively start another pool.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct Permit;
#[cfg(not(target_arch = "wasm32"))]
static ACTIVE: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
#[cfg(not(target_arch = "wasm32"))]
static AVAILABLE: std::sync::Condvar = std::sync::Condvar::new();
#[cfg(not(target_arch = "wasm32"))]
impl Permit {
    pub(crate) fn acquire() -> Self {
        let mut active = ACTIVE.lock().unwrap();
        while *active >= worker_limit() {
            active = AVAILABLE
                .wait_timeout(active, std::time::Duration::from_millis(100))
                .unwrap()
                .0;
        }
        *active += 1;
        Self
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl Drop for Permit {
    fn drop(&mut self) {
        *ACTIVE.lock().unwrap() -= 1;
        AVAILABLE.notify_all();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_threshold_and_unknown_measurements() {
        assert!(memory_error(None, GIB).is_none());
        assert!(memory_error(Some(GIB - 1), GIB).is_none());
        assert!(memory_error(Some(GIB), GIB).unwrap().contains("limit"));
        assert!((1..=cpu_count()).contains(&worker_limit()));
    }
}
