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

//! `tokenspeed_engine._native`: the Python frontend's connection to
//! TokenSpeed engines, over [`ts_client`].
//!
//! The Python frontend (`AsyncLLM`, under the `Engine` API and the gRPC
//! servicer) tokenizes, detokenizes and matches stop strings; this module
//! submits its tokenized requests to the engines and hands back their
//! outputs. The client's sockets and streams run on the Tokio runtime
//! `pyo3-async-runtimes` owns, so a Python thread waiting on them does not
//! hold the GIL.
//!
//! Every request's outputs land in one queue, which the frontend drains a
//! batch at a time ([`Frontend::next_outputs`]): one Python wakeup per step
//! of the engines, not one per request.
//!
//! Requests and control operations cross as their wire frames
//! (`zmq_wire`'s encoding of the `io_struct` messages), so the Python
//! structs stay the frontend's representation; control replies come back the
//! same way.

mod bootstrap;
mod l3;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use pyo3::{
    create_exception,
    exceptions::PyException,
    prelude::*,
    types::{PyBytes, PyDict, PyList},
    IntoPyObjectExt,
};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use ts_client::{
    ClientConfig, ClientListener, EngineClient, LoadBalance, OutputChunk, Routing, StreamError,
};
use ts_wire::{EngineOutput, EngineRequest, Value};

create_exception!(
    tokenspeed_engine._native,
    EngineError,
    PyException,
    "The engines are unreachable or refused a submission."
);

fn engine_error(error: impl std::fmt::Display) -> PyErr {
    EngineError::new_err(error.to_string())
}

fn runtime() -> &'static tokio::runtime::Runtime {
    pyo3_async_runtimes::tokio::get_runtime()
}

/// The frontend's sockets, bound, before the engines connect.
#[pyclass(frozen, module = "tokenspeed_engine._native")]
struct FrontendListener {
    listener: Mutex<Option<ClientListener>>,
    handshake_address: String,
}

#[pymethods]
impl FrontendListener {
    /// Bind the handshake, input and output sockets.
    ///
    /// Args:
    ///     handshake_address: The handshake ROUTER the engines dial; a
    ///         `tcp://<host>:0` address binds a free port (read it back from
    ///         `handshake_address`).
    ///     input_address: The input ROUTER, bound like the handshake.
    ///     output_address: The output PULL, bound like the handshake.
    ///     first_engine_index: The first engine's `--zmq-engine-index`.
    ///     num_engines: Engines to wait for, one per attention-DP rank.
    ///     routing: `"round_robin"`, `"least_in_flight"` or
    ///         `"minimum_cache_usage"` for aggregated serving, or
    ///         `"bootstrap_room"` for disaggregated serving (a pinned
    ///         decode request goes to its `data_parallel_rank`).
    ///     handshake_timeout_s: How long `accept` waits for every engine.
    ///     send_timeout_s: How long one message may wait for an engine's
    ///         socket.
    #[staticmethod]
    #[pyo3(signature = (
        *,
        handshake_address,
        input_address,
        output_address,
        first_engine_index,
        num_engines,
        routing,
        handshake_timeout_s,
        send_timeout_s,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn bind(
        py: Python<'_>,
        handshake_address: String,
        input_address: String,
        output_address: String,
        first_engine_index: u16,
        num_engines: u16,
        routing: &str,
        handshake_timeout_s: f64,
        send_timeout_s: f64,
    ) -> PyResult<Self> {
        let routing = match routing {
            "round_robin" => Routing::Aggregated(LoadBalance::RoundRobin),
            "least_in_flight" => Routing::Aggregated(LoadBalance::LeastInFlight),
            "minimum_cache_usage" => Routing::Aggregated(LoadBalance::MinimumCacheUsage),
            "bootstrap_room" => Routing::BootstrapRoom,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown routing {other:?}"
                )))
            }
        };
        let config = ClientConfig {
            handshake_address,
            input_address,
            output_address,
            first_engine_index,
            num_engines,
            routing,
            handshake_timeout: Duration::from_secs_f64(handshake_timeout_s),
            send_timeout: Duration::from_secs_f64(send_timeout_s),
        };
        let listener = py
            .detach(|| runtime().block_on(ClientListener::bind(config)))
            .map_err(engine_error)?;
        Ok(Self {
            handshake_address: listener.handshake_address().to_owned(),
            listener: Mutex::new(Some(listener)),
        })
    }

    /// The handshake address the engines must dial, as bound.
    #[getter]
    fn handshake_address(&self) -> &str {
        &self.handshake_address
    }

    /// Start waiting for every engine to register; `wait` on the result
    /// returns the connected frontend. The wait runs in the background, so
    /// a launcher can start the engines and watch them meanwhile.
    fn accept(&self) -> PyResult<PendingFrontend> {
        let listener = self
            .listener
            .lock()
            .expect("listener lock")
            .take()
            .ok_or_else(|| engine_error("the listener was already accepted"))?;
        Ok(PendingFrontend {
            accept: Mutex::new(Some(runtime().spawn(listener.accept()))),
        })
    }
}

type Accept = tokio::task::JoinHandle<ts_client::Result<EngineClient>>;

/// A frontend waiting for its engines.
#[pyclass(frozen, module = "tokenspeed_engine._native")]
struct PendingFrontend {
    accept: Mutex<Option<Accept>>,
}

#[pymethods]
impl PendingFrontend {
    /// Block until every engine registered, without holding the GIL; raise
    /// `EngineError` if the handshake failed or timed out.
    fn wait(&self, py: Python<'_>) -> PyResult<Frontend> {
        let accept = self
            .accept
            .lock()
            .expect("accept lock")
            .take()
            .ok_or_else(|| engine_error("the frontend was already waited for"))?;
        let client = py
            .detach(|| runtime().block_on(accept))
            .map_err(engine_error)?
            .map_err(engine_error)?;
        let (outputs, receiver) = mpsc::unbounded_channel();
        Ok(Frontend {
            client: Arc::new(client),
            outputs,
            receiver: Arc::new(AsyncMutex::new(receiver)),
        })
    }
}

/// One output of one request: its chunk, or why its stream ended early.
type Delivery = (String, Result<OutputChunk, StreamError>);

/// The connected engines.
#[pyclass(frozen, module = "tokenspeed_engine._native")]
struct Frontend {
    client: Arc<EngineClient>,
    /// Every request's outputs, in arrival order per request.
    outputs: mpsc::UnboundedSender<Delivery>,
    receiver: Arc<AsyncMutex<mpsc::UnboundedReceiver<Delivery>>>,
}

#[pymethods]
impl Frontend {
    /// The engines, in index order: each one's index and the configuration
    /// it registered with (`max_model_len`, `data_parallel_rank`, ...).
    fn engines<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let engines = PyList::empty(py);
        for engine in self.client.engines() {
            let info = PyDict::new(py);
            info.set_item("index", engine.index)?;
            info.set_item("max_model_len", engine.ready.max_model_len)?;
            info.set_item("data_parallel_rank", engine.ready.data_parallel_rank)?;
            info.set_item("data_parallel_size", engine.ready.data_parallel_size)?;
            engines.append(info)?;
        }
        Ok(engines)
    }

    /// Each engine's latest reported load, in index order:
    /// `(num_running, num_waiting, kv_active_pages, kv_total_pages)`.
    fn loads(&self) -> Vec<(i64, i64, i64, i64)> {
        self.client
            .loads()
            .into_iter()
            .map(|load| {
                (
                    load.num_running,
                    load.num_waiting,
                    load.kv_active_pages,
                    load.kv_total_pages,
                )
            })
            .collect()
    }

    /// This frontend's unfinished requests per engine, in index order.
    fn in_flight(&self) -> Vec<usize> {
        self.client.in_flight()
    }

    /// Submit a generate request, given as its wire frames (the payload
    /// `zmq_wire` encodes, without the type frame). Its outputs arrive
    /// through `next_outputs`; awaiting this returns once the engine has
    /// the request.
    fn generate<'py>(&self, py: Python<'py>, frames: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyAny>> {
        let request = match EngineRequest::from_frames(&request_frames(
            ts_wire::frames::REQ_TYPE_ADD,
            frames,
        ))
        .map_err(engine_error)?
        {
            EngineRequest::Add(request) => *request,
            _ => unreachable!("the type frame names a generate request"),
        };
        let client = self.client.clone();
        let outputs = self.outputs.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut stream = client.generate(request).await.map_err(engine_error)?;
            let rid = stream.rid().to_owned();
            tokio::spawn(async move {
                while let Some(item) = stream.next().await {
                    if outputs.send((rid.clone(), item)).is_err() {
                        // The frontend is gone; dropping the stream aborts it.
                        return;
                    }
                }
            });
            Ok(())
        })
    }

    /// Wait for outputs and return every one that arrived, oldest first: a
    /// `(rid, chunk)` per output, where the chunk is a dict, or an
    /// `EngineError` for a stream the engines ended without a finished
    /// output.
    fn next_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = self.receiver.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut receiver = receiver.lock().await;
            let Some(first) = receiver.recv().await else {
                return Err(engine_error("the frontend is closed"));
            };
            let mut batch = vec![first];
            while let Ok(next) = receiver.try_recv() {
                batch.push(next);
            }
            Ok(Deliveries(batch))
        })
    }

    /// Ask the engine to abort a request; its outputs then end with an
    /// abort. Resolves to False if the request is not in flight.
    fn abort<'py>(&self, py: Python<'py>, rid: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.client.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.abort(&rid).await.map_err(engine_error)
        })
    }

    /// Run a control operation, given as its wire frames (`zmq_wire`'s
    /// control encoding, without the type frame), on every engine; resolves
    /// to each engine's reply frames, in arrival order.
    fn control<'py>(&self, py: Python<'py>, frames: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyAny>> {
        let request = match EngineRequest::from_frames(&request_frames(
            ts_wire::frames::REQ_TYPE_CONTROL,
            frames,
        ))
        .map_err(engine_error)?
        {
            EngineRequest::Control(request) => *request,
            _ => unreachable!("the type frame names a control operation"),
        };
        let client = self.client.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let replies = client.control(request).await.map_err(engine_error)?;
            Ok(replies
                .into_iter()
                .map(|reply| {
                    EngineOutput::Control(Box::new(reply))
                        .to_frames()
                        .into_iter()
                        .map(|frame| frame.to_vec())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>())
        })
    }
}

fn request_frames(type_frame: u8, frames: Vec<Vec<u8>>) -> Vec<Bytes> {
    std::iter::once(Bytes::copy_from_slice(&[type_frame]))
        .chain(frames.into_iter().map(Bytes::from))
        .collect()
}

/// A batch of [`Delivery`]s, as Python receives it.
struct Deliveries(Vec<Delivery>);

impl<'py> IntoPyObject<'py> for Deliveries {
    type Target = PyList;
    type Output = Bound<'py, PyList>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let list = PyList::empty(py);
        for (rid, item) in self.0 {
            let output = match item {
                Ok(chunk) => chunk_dict(py, chunk)?.into_any(),
                Err(error) => EngineError::new_err(error.to_string())
                    .into_value(py)
                    .into_bound_py_any(py)?,
            };
            list.append((rid, output))?;
        }
        Ok(list)
    }
}

/// An output chunk as a dict with the fields of [`OutputChunk`].
fn chunk_dict<'py>(py: Python<'py>, chunk: OutputChunk) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("output_ids", chunk.output_ids)?;
    dict.set_item("finish_reason", chunk.finish_reason)?;
    dict.set_item(
        "finish_details",
        chunk
            .finish_details
            .map(|details| {
                let map = PyDict::new(py);
                for (key, value) in details.0 {
                    map.set_item(key, value_to_py(py, value)?)?;
                }
                PyResult::Ok(map)
            })
            .transpose()?,
    )?;
    dict.set_item("prompt_tokens", chunk.prompt_tokens)?;
    dict.set_item("completion_tokens", chunk.completion_tokens)?;
    dict.set_item("cached_tokens", chunk.cached_tokens)?;
    dict.set_item("output_token_logprobs", chunk.output_token_logprobs)?;
    dict.set_item("spec_verify_ct", chunk.spec_verify_ct)?;
    dict.set_item("accept_draft_tokens", chunk.accept_draft_tokens)?;
    dict.set_item("input_token_logprobs", chunk.input_token_logprobs)?;
    dict.set_item("input_top_logprobs", chunk.input_top_logprobs)?;
    dict.set_item("output_top_logprobs", chunk.output_top_logprobs)?;
    Ok(dict)
}

fn value_to_py(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    Ok(match value {
        Value::Nil => py.None(),
        Value::Bool(value) => value.into_py_any(py)?,
        Value::Int(value) => match value {
            ts_wire::msgpack::Integer::Signed(value) => value.into_py_any(py)?,
            ts_wire::msgpack::Integer::Unsigned(value) => value.into_py_any(py)?,
        },
        Value::Float(value) => value.into_py_any(py)?,
        Value::Str(value) => value.into_py_any(py)?,
        Value::Bin(value) => PyBytes::new(py, &value).into_py_any(py)?,
        Value::Array(values) => values
            .into_iter()
            .map(|value| value_to_py(py, value))
            .collect::<PyResult<Vec<_>>>()?
            .into_py_any(py)?,
        Value::Map(entries) => {
            let map = PyDict::new(py);
            for (key, value) in entries {
                map.set_item(value_to_py(py, key)?, value_to_py(py, value)?)?;
            }
            map.into_py_any(py)?
        }
        Value::Ext(_, value) => PyBytes::new(py, &value).into_py_any(py)?,
    })
}

/// The Python module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<bootstrap::BootstrapServer>()?;
    module.add_class::<l3::L3Executor>()?;
    module.add_class::<FrontendListener>()?;
    module.add_class::<PendingFrontend>()?;
    module.add_class::<Frontend>()?;
    module.add("EngineError", module.py().get_type::<EngineError>())?;
    Ok(())
}
