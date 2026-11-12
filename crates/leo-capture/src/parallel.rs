//! Bounded native generation workers with ordered results; sequential on wasm.

/// Available logical CPUs, limited to the amount of work, with a one-worker fallback.
pub fn worker_count(items: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    let available = crate::resources::worker_limit();
    #[cfg(target_arch = "wasm32")]
    let available = 1;
    available.min(items.max(1))
}

/// Run independent items concurrently, reporting completed items from the caller
/// thread. Queueing is bounded, output retains input order, and failures never
/// return a partial result. Small batches amortize locks while balancing large AOIs.
pub fn map<T: Send, R: Send>(
    items: Vec<T>,
    workers: usize,
    build: impl Fn(T) -> Result<R, String> + Sync,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<R>, String> {
    try_map(items, workers, build, |n, total| {
        progress(n, total);
        Ok(())
    })
}

/// Like `map`, but progress can cancel, including while waiting for a slow item.
pub fn try_map<T: Send, R: Send>(
    items: Vec<T>,
    workers: usize,
    build: impl Fn(T) -> Result<R, String> + Sync,
    progress: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<Vec<R>, String> {
    map_impl(items, workers, build, progress, true)
}

/// A single atomic simulation tick already admitted by its caller. Complete it
/// before checking the memory threshold again so it remains checkpointable.
pub(crate) fn in_flight<T: Send, R: Send>(
    items: Vec<T>,
    workers: usize,
    build: impl Fn(T) -> Result<R, String> + Sync,
) -> Result<Vec<R>, String> {
    map_impl(items, workers, build, |_, _| Ok(()), false)
}
fn map_impl<T: Send, R: Send>(
    items: Vec<T>,
    workers: usize,
    build: impl Fn(T) -> Result<R, String> + Sync,
    mut progress: impl FnMut(usize, usize) -> Result<(), String>,
    check_memory: bool,
) -> Result<Vec<R>, String> {
    let memory = || {
        if check_memory {
            crate::resources::check_memory()
        } else {
            Ok(())
        }
    };
    let total = items.len();
    memory()?;
    progress(0, total)?;
    #[cfg(not(target_arch = "wasm32"))]
    let mut last_memory = std::time::Instant::now();
    #[cfg(target_arch = "wasm32")]
    let workers = {
        let _ = workers;
        1
    };
    let workers = workers.max(1).min(total.max(1));
    if workers == 1 {
        #[cfg(not(target_arch = "wasm32"))]
        let _permit = crate::resources::Permit::acquire();
        return items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                let result = build(item)?;
                #[cfg(not(target_arch = "wasm32"))]
                if i + 1 == total || last_memory.elapsed().as_millis() >= 100 {
                    memory()?;
                    last_memory = std::time::Instant::now();
                }
                progress(i + 1, total)?;
                Ok(result)
            })
            .collect();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Mutex,
        };
        let queue = Mutex::new(items.into_iter().enumerate());
        let cancelled = AtomicBool::new(false);
        let batch_size = (total / (workers * 8)).clamp(1, 64);
        let (tx, rx) = mpsc::sync_channel(workers * 2);
        let mut results: Vec<Option<R>> = std::iter::repeat_with(|| None).take(total).collect();
        std::thread::scope(|scope| {
            let mut error = None;
            let mut handles = Vec::new();
            for index in 0..workers {
                let (tx, queue, cancelled, build) = (tx.clone(), &queue, &cancelled, &build);
                let spawn = std::thread::Builder::new()
                    .name(format!("generation-{index}"))
                    .spawn_scoped(scope, move || {
                        let outcome =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                crate::resources::background_thread();
                                while !cancelled.load(Ordering::Relaxed) {
                                    let _permit = crate::resources::Permit::acquire();
                                    if cancelled.load(Ordering::Relaxed) {
                                        break;
                                    }
                                    let batch: Vec<_> =
                                        queue.lock().unwrap().by_ref().take(batch_size).collect();
                                    if batch.is_empty() {
                                        break;
                                    }
                                    let result = (|| {
                                        let mut output = Vec::with_capacity(batch.len());
                                        for (i, item) in batch {
                                            if cancelled.load(Ordering::Relaxed) {
                                                break;
                                            }
                                            output.push((i, build(item)?));
                                        }
                                        Ok::<_, String>(output)
                                    })();
                                    if result.is_err() {
                                        cancelled.store(true, Ordering::Relaxed);
                                    }
                                    if tx.send(result).is_err() {
                                        break;
                                    }
                                }
                            }));
                        if let Err(payload) = outcome {
                            cancelled.store(true, Ordering::Relaxed);
                            let message = payload
                                .downcast_ref::<String>()
                                .cloned()
                                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                                .unwrap_or_else(|| "panic without text".into());
                            let _ =
                                tx.send(Err(format!("A calculation worker panicked: {message}")));
                        }
                    });
                match spawn {
                    Ok(handle) => handles.push(handle),
                    Err(e) => {
                        cancelled.store(true, Ordering::Relaxed);
                        error = Some(format!("Could not start generation worker: {e}"));
                        break;
                    }
                }
            }
            drop(tx);
            let mut completed = 0;
            loop {
                let result = match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(result) => Some(result),
                    Err(mpsc::RecvTimeoutError::Timeout) => None,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
                if error.is_none() {
                    let check = if last_memory.elapsed().as_millis() >= 100 {
                        last_memory = std::time::Instant::now();
                        memory()
                    } else {
                        Ok(())
                    };
                    if let Err(e) = check.and_then(|_| {
                        if result.is_none() {
                            progress(completed, total)
                        } else {
                            Ok(())
                        }
                    }) {
                        error = Some(e);
                        cancelled.store(true, Ordering::Relaxed);
                    }
                }
                let Some(result) = result else { continue };
                match result {
                    Ok(batch) if error.is_none() => {
                        completed += batch.len();
                        for (index, result) in batch {
                            results[index] = Some(result);
                        }
                        if let Err(e) = progress(completed, total) {
                            error = Some(e);
                            cancelled.store(true, Ordering::Relaxed);
                        }
                    }
                    Err(e) => {
                        error.get_or_insert(e);
                    }
                    _ => (),
                }
            }
            for handle in handles {
                if handle.join().is_err() {
                    error.get_or_insert("Generation worker stopped unexpectedly.".into());
                }
            }
            if let Some(error) = error {
                return Err(error);
            }
            memory()?;
            results
                .into_iter()
                .map(|r| r.ok_or_else(|| "Generation ended before all items completed.".into()))
                .collect()
        })
    }
    #[cfg(target_arch = "wasm32")]
    unreachable!()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    fn workers_overlap_but_results_and_progress_remain_ordered() {
        let count = crate::resources::worker_limit().min(4);
        let barrier = std::sync::Barrier::new(count);
        let threads = std::sync::Mutex::new(std::collections::HashSet::new());
        let mut completed = Vec::new();
        let output = map(
            (0..count).collect(),
            count,
            |i| {
                threads.lock().unwrap().insert(std::thread::current().id());
                barrier.wait();
                Ok(i * 2)
            },
            |n, total| {
                assert_eq!(total, count);
                completed.push(n);
            },
        )
        .unwrap();
        assert_eq!(output, (0..count).map(|i| i * 2).collect::<Vec<_>>());
        assert_eq!(threads.lock().unwrap().len(), count);
        assert!(completed.windows(2).all(|w| w[0] <= w[1]));
        completed.dedup();
        assert_eq!(completed, (0..=count).collect::<Vec<_>>());
    }
    #[test]
    fn simultaneous_pools_share_the_cpu_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let limit = crate::resources::worker_limit();
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let (active, peak) = (&active, &peak);
                scope.spawn(move || {
                    map(
                        (0..32).collect(),
                        limit + 2,
                        |i| {
                            let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(n, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(2));
                            active.fetch_sub(1, Ordering::SeqCst);
                            Ok(i)
                        },
                        |_, _| {},
                    )
                    .unwrap();
                });
            }
        });
        assert!(peak.load(Ordering::SeqCst) <= limit);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn cancellation_stops_queued_work_and_joins_in_flight_items() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let built = AtomicUsize::new(0);
        let result = try_map(
            (0..1000).collect(),
            4,
            |i| {
                std::thread::sleep(std::time::Duration::from_millis(1));
                built.fetch_add(1, Ordering::Relaxed);
                Ok(i)
            },
            |n, _| {
                if n >= 8 {
                    Err("cancelled by caller".into())
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(result.unwrap_err(), "cancelled by caller");
        let count = built.load(Ordering::Relaxed);
        assert!(count < 1000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert_eq!(built.load(Ordering::Relaxed), count);
    }
    #[test]
    fn errors_and_panics_join_workers_without_publishing_partial_results() {
        assert!(map(
            (0..100).collect(),
            4,
            |i| if i == 2 {
                Err("bad geometry".into())
            } else {
                Ok(i)
            },
            |_, _| {}
        )
        .is_err());
        assert!(map(
            (0..100).collect(),
            4,
            |i| {
                assert_ne!(i, 2, "worker failure");
                Ok(i)
            },
            |_, _| {}
        )
        .is_err());
        assert_eq!(map(vec![1, 2], 0, Ok, |_, _| {}).unwrap(), [1, 2]);
        assert!(map::<u32, u32>(vec![], 8, Ok, |_, _| {})
            .unwrap()
            .is_empty());
        assert_eq!(worker_count(1), 1);
        assert_eq!(worker_count(usize::MAX), crate::resources::worker_limit());
    }
}
