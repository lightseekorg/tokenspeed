// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! The L3 executor (`ts-l3`) for Python callers outside an engine rank:
//! tests and tools that drive the storage tier against a data plane's Host
//! pool. An engine rank's host runs its own.

use std::{sync::Mutex, time::Duration};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyList,
};
use ts_l3::{Executor, Operation, Settings, StorageKey};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Arc, Barrier};

    use super::*;
    use ts_l3::{
        monotonic_clock, ExecutorConfig, GroupLayout, HostBuffer, HostLayout, IoResult,
        MemoryStore, Namespace, Range, Store, StoreError,
    };

    struct WaitingStore {
        memory: MemoryStore,
        entered: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }

    impl Store for WaitingStore {
        fn failures_retire(&self) -> bool {
            true
        }

        fn batch_exists(&self, keys: &[String]) -> Result<Vec<bool>, StoreError> {
            self.entered.send(()).unwrap();
            self.release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            self.memory.batch_exists(keys)
        }

        fn batch_get_into(
            &self,
            keys: &[String],
            host: HostBuffer,
            ranges: &[Range],
        ) -> Result<IoResult, StoreError> {
            self.memory.batch_get_into(keys, host, ranges)
        }

        fn batch_put_from(
            &self,
            keys: &[String],
            host: HostBuffer,
            ranges: &[Range],
        ) -> Result<IoResult, StoreError> {
            self.memory.batch_put_from(keys, host, ranges)
        }

        fn remove_by_prefix(&self, prefix: &str) -> bool {
            self.memory.remove_by_prefix(prefix)
        }
        fn close(&self) {
            self.memory.close();
        }
    }

    fn executor(store: Box<dyn Store>, pool: &mut [u8]) -> Executor {
        Executor::start(
            store,
            HostBuffer {
                base: pool.as_mut_ptr() as usize,
                len: pool.len(),
            },
            HostLayout {
                groups: vec![GroupLayout {
                    packing: 1,
                    block_bytes: 1,
                    shard_count: 1,
                }],
                lcm_block_bytes: 1,
                num_lcm_blocks: pool.len(),
                owner_rank: 0,
                tp_rank: 0,
            },
            Namespace {
                deployment_id: "python-close-test".to_owned(),
                has_draft: false,
            },
            "v1",
            ExecutorConfig {
                max_operations: 4,
                max_keys_per_operation: 4,
                query_workers: 1,
                read_workers: 1,
                write_workers: 1,
                failure_threshold: 1,
                recovery_interval: Duration::from_secs(1),
            },
            monotonic_clock(),
        )
    }

    #[test]
    fn simultaneous_python_closes_cannot_report_a_pending_close_as_success() {
        Python::initialize();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut pool = vec![0_u8; 4];
        let running = executor(
            Box::new(WaitingStore {
                memory: MemoryStore::new(),
                entered: entered_tx,
                release: Mutex::new(release_rx),
            }),
            &mut pool,
        );
        running
            .submit(&[Operation::Query {
                op_id: 1,
                keys: vec![StorageKey {
                    group: 0,
                    content_hash: "waiting".to_owned(),
                    page_offset: 0,
                }],
            }])
            .unwrap();
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let binding = Arc::new(Python::attach(|py| {
            Py::new(
                py,
                L3Executor {
                    executor: Mutex::new(Some(running)),
                },
            )
            .unwrap()
        }));
        let start = Arc::new(Barrier::new(3));
        let closers: Vec<_> = (0..2)
            .map(|_| {
                let (binding, start) = (Arc::clone(&binding), Arc::clone(&start));
                std::thread::spawn(move || {
                    start.wait();
                    Python::attach(|py| binding.bind(py).call_method1("close", (0.5,)).is_err())
                })
            })
            .collect();
        start.wait();
        let results: Vec<bool> = closers
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        let readable = Python::attach(|py| binding.bind(py).call_method0("snapshot").is_ok());
        release_tx.send(()).unwrap();
        Python::attach(|py| {
            binding.bind(py).call_method1("close", (2.0,)).unwrap();
        });
        assert_eq!(results, [true, true]);
        assert!(
            readable,
            "failed closes must keep the diagnostics available"
        );
    }

    #[test]
    fn invalid_python_close_timeouts_leave_the_executor_usable() {
        Python::initialize();
        let mut pool = vec![0_u8; 4];
        let running = executor(Box::new(MemoryStore::new()), &mut pool);
        Python::attach(|py| {
            let binding = Py::new(
                py,
                L3Executor {
                    executor: Mutex::new(Some(running)),
                },
            )
            .unwrap();
            for timeout in [-1.0, f64::NAN, f64::INFINITY] {
                let error = binding
                    .bind(py)
                    .call_method1("close", (timeout,))
                    .unwrap_err();
                assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(binding.bind(py).call_method0("snapshot").is_ok());
            }
            binding.bind(py).call_method1("close", (2.0,)).unwrap();
        });
    }
}

/// One rank's L3 executor, started from an ``L3Settings``.
#[pyclass(module = "tokenspeed_engine._native")]
pub struct L3Executor {
    executor: Mutex<Option<Executor>>,
}

impl L3Executor {
    fn with<R>(&self, run: impl FnOnce(&Executor) -> PyResult<R>) -> PyResult<R> {
        let executor = self.executor.lock().expect("L3 executor lock");
        run(executor
            .as_ref()
            .ok_or_else(|| runtime_error("the L3 executor is closed"))?)
    }
}

/// A scheduler storage operation (``Cache.Storage*Op``) as the executor
/// takes it.
fn operation(op: &Bound<'_, PyAny>) -> PyResult<Operation> {
    let op_id: u64 = op.getattr("op_id")?.extract()?;
    let kind: String = op.getattr("kind")?.getattr("name")?.extract()?;
    let keys = || -> PyResult<Vec<StorageKey>> {
        let keys: Vec<(u32, String, i32)> = op.getattr("keys")?.extract()?;
        Ok(keys
            .into_iter()
            .map(|(group, content_hash, page_offset)| StorageKey {
                group,
                content_hash,
                page_offset,
            })
            .collect())
    };
    Ok(match kind.as_str() {
        "Query" => Operation::Query {
            op_id,
            keys: keys()?,
        },
        "Get" => Operation::Get {
            op_id,
            keys: keys()?,
            host_blocks: op.getattr("host_blocks")?.extract()?,
        },
        "Put" => Operation::Put {
            op_id,
            keys: keys()?,
            host_blocks: op.getattr("host_blocks")?.extract()?,
        },
        "Cancel" => Operation::Cancel { op_id },
        other => return Err(runtime_error(format!("unknown storage operation {other}"))),
    })
}

#[pymethods]
impl L3Executor {
    /// Connect the store and start the workers.
    #[new]
    fn new(settings: &Bound<'_, PyAny>) -> PyResult<Self> {
        let settings = Settings::from_python(settings)?;
        let executor = settings.start().map_err(runtime_error)?;
        Ok(Self {
            executor: Mutex::new(Some(executor)),
        })
    }

    /// Start scheduler storage operations (``Cache.Storage*Op``), in order;
    /// none waits for I/O.
    fn submit(&self, operations: &Bound<'_, PyList>) -> PyResult<()> {
        let operations = operations
            .iter()
            .map(|op| operation(&op))
            .collect::<PyResult<Vec<_>>>()?;
        self.with(|executor| executor.submit(&operations).map_err(runtime_error))
    }

    /// The storage clock in microseconds, the operations finished since the
    /// last poll ``(op_id, success, retired, object_failures)``, and the
    /// queue pressure ``(operations, bytes, oldest operation's age in
    /// microseconds)``.
    #[allow(clippy::type_complexity)]
    fn poll(&self) -> PyResult<(i64, Vec<(u64, Vec<bool>, bool, Vec<bool>)>, (i64, i64, i64))> {
        self.with(|executor| {
            let poll = executor.poll();
            Ok((
                poll.now_us,
                poll.completions
                    .into_iter()
                    .map(|done| (done.op_id, done.success, done.retired, done.object_failures))
                    .collect(),
                (
                    poll.pressure.queued_operations,
                    poll.pressure.queued_bytes,
                    poll.pressure.oldest_operation_us,
                ),
            ))
        })
    }

    /// Per operation kind ``(kind, events, success_keys, failed_keys,
    /// success_bytes, queue_seconds, backend_seconds, operations, bytes)``,
    /// the backend's health state and epoch, and whether new I/O is off
    /// (unproven retirement).
    #[allow(clippy::type_complexity)]
    fn snapshot(
        &self,
    ) -> PyResult<(
        Vec<(
            String,
            Vec<(String, f64)>,
            f64,
            f64,
            f64,
            f64,
            f64,
            Vec<(String, f64)>,
            Vec<(String, f64)>,
        )>,
        String,
        f64,
        bool,
    )> {
        self.with(|executor| {
            let snapshot = executor.snapshot();
            Ok((
                snapshot
                    .operations
                    .into_iter()
                    .map(|row| {
                        (
                            row.kind,
                            row.events,
                            row.success_keys,
                            row.failed_keys,
                            row.success_bytes,
                            row.queue_seconds,
                            row.backend_seconds,
                            row.operations,
                            row.bytes,
                        )
                    })
                    .collect(),
                snapshot.health_state,
                snapshot.health_epoch,
                snapshot.unsafe_io,
            ))
        })
    }

    /// The namespace prefix objects are named under now.
    fn key_prefix(&self) -> PyResult<String> {
        self.with(|executor| Ok(executor.key_prefix()))
    }

    /// Nothing in flight, and no unproven retirement.
    fn is_idle(&self) -> PyResult<bool> {
        self.with(|executor| Ok(executor.is_idle()))
    }

    /// Switch the namespace to ``weight_version``; the executor must be idle.
    fn set_weight_version(&self, weight_version: &str) -> PyResult<()> {
        self.with(|executor| {
            executor
                .set_weight_version(weight_version)
                .map_err(runtime_error)
        })
    }

    /// Delete the namespace's objects once idle: whether they are gone.
    fn delete_namespace(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.with(|executor| Ok(executor.delete_namespace())))
    }

    /// Stop the workers within ``timeout_s`` and release the store; raises
    /// when native I/O could not prove its retirement (the store and the
    /// Host pool's registration then stay for the process). A failed close
    /// keeps the executor: its snapshot and counters stay readable for
    /// diagnosis, the quarantine gauges included, and a later call
    /// retries. A successful close is final; later calls do nothing.
    fn close(&self, py: Python<'_>, timeout_s: f64) -> PyResult<()> {
        let timeout = Duration::try_from_secs_f64(timeout_s)
            .map_err(|_| PyValueError::new_err("timeout_s must be finite and nonnegative"))?;
        py.detach(|| {
            // Serialize closes without holding the GIL. A pending close
            // must not look like a successful close to another caller.
            let mut slot = self.executor.lock().expect("L3 executor lock");
            if let Some(executor) = slot.as_mut() {
                executor.close(timeout).map_err(runtime_error)?;
                // Only successful retirement makes the handle terminal.
                *slot = None;
            }
            Ok(())
        })
    }
}
