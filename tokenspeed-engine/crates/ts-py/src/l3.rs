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

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList};
use ts_l3::{Executor, Operation, Settings, StorageKey};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
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
    /// Host pool's registration then stay for the process).
    fn close(&self, py: Python<'_>, timeout_s: f64) -> PyResult<()> {
        let executor = self.executor.lock().expect("L3 executor lock").take();
        let Some(mut executor) = executor else {
            return Ok(());
        };
        py.detach(|| executor.close(Duration::from_secs_f64(timeout_s)))
            .map_err(runtime_error)
    }
}
