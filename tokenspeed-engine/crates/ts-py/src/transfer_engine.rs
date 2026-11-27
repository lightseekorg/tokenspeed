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

//! The Mooncake transfer engine (`ts-mooncake`) for a Python process that
//! only hosts memory other engines read and write: a shared L2 pool's
//! provider (`tokenspeed.runtime.cache.shared_l2.provider`). An engine
//! rank's host runs its own.

use std::path::PathBuf;

use pyo3::{exceptions::PyRuntimeError, prelude::*};
use ts_mooncake::{Engine, Library};

/// A started transfer engine. It stops when the object is dropped.
#[pyclass(frozen, module = "tokenspeed_engine._native")]
pub struct TransferEngine {
    engine: Engine,
}

#[pymethods]
impl TransferEngine {
    /// Load the Mooncake library and start an engine. Waits without the GIL.
    ///
    /// Args:
    ///     library: Path of `libtokenspeed_transfer_engine.so`.
    ///     hostname: The address other engines reach this one at.
    ///     protocol: The engine's transport.
    ///     device_filter: The devices to use, comma-separated; empty for all.
    ///
    /// Raises:
    ///     RuntimeError: The library did not load or the engine did not
    ///         start.
    #[new]
    #[pyo3(signature = (*, library, hostname, protocol, device_filter))]
    fn new(
        py: Python<'_>,
        library: PathBuf,
        hostname: &str,
        protocol: &str,
        device_filter: &str,
    ) -> PyResult<Self> {
        py.detach(|| Engine::start(Library::load(&library)?, hostname, protocol, device_filter))
            .map(|engine| Self { engine })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// The id other engines address this engine's memory by.
    #[getter]
    fn session_id(&self) -> &str {
        self.engine.session_id()
    }

    /// Register `length` bytes at `address` for other engines to read and
    /// write. The caller keeps the memory mapped until it is unregistered
    /// or the engine stops. Waits without the GIL.
    ///
    /// Raises:
    ///     RuntimeError: The registration failed.
    fn register(&self, py: Python<'_>, address: usize, length: usize) -> PyResult<()> {
        py.detach(|| self.engine.register(address, length))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Undo `register` of the region at `address`. Waits without the GIL.
    ///
    /// Raises:
    ///     RuntimeError: The deregistration failed.
    fn unregister(&self, py: Python<'_>, address: usize) -> PyResult<()> {
        py.detach(|| self.engine.unregister(address))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}
