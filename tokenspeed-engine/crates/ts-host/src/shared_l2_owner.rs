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

//! A shared L2 pool's owner in the launching Python interpreter: ts-control's
//! serving loop (`shared_l2::owner`) over a pool that Python builds, as a
//! rank's loop runs over a scheduler that Python built.

use std::{
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use ts_control::shared_l2::owner::{Owner, OwnerSocket};
use ts_scheduler::SharedL2Pool;

use crate::{data_plane, logging, watch_signals};

/// Serve a shared L2 pool's owner at `address` on the calling (main) thread
/// until SIGTERM or SIGINT, with the GIL released.
///
/// The owner waits for `num_providers` providers and drops a client silent
/// for `client_timeout_s` seconds. `make_pool(geometry, num_lcm_blocks)`
/// returns the scheduler's `SharedL2Pool` for the first client's geometry,
/// which it is handed as msgpack bytes, over the LCM blocks of every
/// provider together; it is the one call that takes the GIL again.
#[pyfunction]
pub(crate) fn run_shared_l2_owner(
    py: Python<'_>,
    address: &str,
    num_providers: usize,
    client_timeout_s: f64,
    make_pool: Py<PyAny>,
) -> PyResult<()> {
    let client_timeout = Duration::try_from_secs_f64(client_timeout_s)
        .map_err(|error| PyValueError::new_err(format!("client_timeout_s: {error}")))?;
    let mut owner = Owner::new(
        num_providers,
        client_timeout,
        move |geometry, num_lcm_blocks| {
            Python::attach(|py| {
                make_pool
                    .bind(py)
                    .call1((PyBytes::new(py, geometry), num_lcm_blocks))
                    .and_then(|pool| SharedL2Pool::from_python(&pool))
                    .map_err(|error| {
                        tracing::error!(
                            "the shared L2 pool was not built:\n{}",
                            data_plane::describe(py, &error)
                        );
                        error.to_string()
                    })
            })
        },
    );
    let served = py.detach(|| {
        logging::init(false);
        let stop = Arc::new(AtomicBool::new(false));
        watch_signals(stop.clone()).map_err(|error| error.to_string())?;
        let mut socket = OwnerSocket::bind(address)
            .map_err(|error| format!("the shared L2 owner cannot bind {address}: {error}"))?;
        tracing::info!(
            address = socket.address(),
            num_providers,
            ?client_timeout,
            "shared L2 owner serving"
        );
        owner.serve(&mut socket, &stop);
        Ok(())
    });
    // The owner's pool is a Python object: it goes with the GIL held.
    drop(owner);
    served.map_err(PyRuntimeError::new_err::<String>)
}
