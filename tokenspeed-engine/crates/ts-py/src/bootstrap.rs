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

//! The bootstrap server of a disaggregated deployment ([`ts_pd::bootstrap`]),
//! for the launcher.

use std::sync::Mutex;

use pyo3::{exceptions::PyRuntimeError, prelude::*};
use ts_pd::{BootstrapServer as Server, Registry};

/// The HTTP rendezvous of a disaggregated deployment's source ranks (a
/// Prefill node's, or an EPD encode node's) and the ranks that read from
/// them, serving on its own threads until closed.
#[pyclass(frozen, module = "tokenspeed_engine._native")]
pub struct BootstrapServer {
    server: Mutex<Server>,
    port: u16,
}

#[pymethods]
impl BootstrapServer {
    /// Bind `port` on every interface and serve.
    ///
    /// Args:
    ///     port: The port; 0 picks a free one (read it back from `port`).
    ///     cache_layouts: The source ranks also register their node's cache
    ///         transfer layout and per-stage field placement, which must
    ///         agree, and the parallel-info route serves them (a PD Prefill
    ///         node); else only their addresses (an EPD encode node).
    ///
    /// Raises:
    ///     RuntimeError: The port could not be bound.
    #[new]
    #[pyo3(signature = (port, *, cache_layouts))]
    fn new(port: u16, cache_layouts: bool) -> PyResult<Self> {
        let registry = if cache_layouts {
            Registry::CacheLayouts
        } else {
            Registry::Addresses
        };
        let server = Server::start(port, registry).map_err(|error| {
            PyRuntimeError::new_err(format!(
                "Bootstrap server failed to bind on port {port}: {error}"
            ))
        })?;
        Ok(Self {
            port: server.port(),
            server: Mutex::new(server),
        })
    }

    /// The bound port.
    #[getter]
    fn port(&self) -> u16 {
        self.port
    }

    /// Stop serving and release the port; idempotent. Waits for the
    /// server's accept loop without the GIL.
    fn close(&self, py: Python<'_>) {
        py.detach(|| self.server.lock().expect("bootstrap server lock").close());
    }
}
