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

//! Embedding transport for an external encoder; no model or scheduler.

use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::{path::PathBuf, sync::Mutex, time::Duration};
use ts_pd::{
    epd::sender::{Sender, SenderConfig},
    EmbeddingChunk, EmbeddingSender as _, PeerConfig,
};

/// The caller owns registered memory and keeps each submitted chunk unchanged
/// until its room appears in `concluded`, including after a call to `fail`.
#[pyclass(frozen, module = "tokenspeed_engine._native")]
pub struct EmbeddingSender {
    sender: Mutex<Sender>,
}

#[pymethods]
impl EmbeddingSender {
    #[new]
    #[pyo3(signature = (*, library, local_ip, ib_device, regions, engine_rank, bootstrap_url, tp_size, timeout_s, transfer_workers))]
    fn new(
        py: Python<'_>,
        library: String,
        local_ip: String,
        ib_device: String,
        regions: Vec<(usize, usize)>,
        engine_rank: i64,
        bootstrap_url: String,
        tp_size: i64,
        timeout_s: u64,
        transfer_workers: usize,
    ) -> PyResult<Self> {
        let config = SenderConfig {
            peer: PeerConfig {
                library: PathBuf::from(library),
                local_ip,
                ib_device,
                regions,
                engine_rank,
                metrics: None,
            },
            bootstrap_url,
            tp_size,
            bootstrap_timeout: Duration::from_secs(timeout_s),
            transfer_workers,
        };
        let sender = py
            .detach(|| Sender::start(config))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            sender: Mutex::new(sender),
        })
    }

    fn begin(&self, room: u64) -> bool {
        self.sender
            .lock()
            .expect("embedding sender lock")
            .begin(room)
    }

    #[pyo3(signature = (*, room, source, rows, hidden, dtype, nbytes, deepstack_source, deepstack_width, deepstack_nbytes))]
    fn send(
        &self,
        room: u64,
        source: u64,
        rows: i64,
        hidden: i64,
        dtype: String,
        nbytes: u64,
        deepstack_source: u64,
        deepstack_width: i64,
        deepstack_nbytes: u64,
    ) {
        self.sender
            .lock()
            .expect("embedding sender lock")
            .send(EmbeddingChunk {
                room,
                source,
                rows,
                hidden,
                dtype,
                nbytes,
                deepstack_source,
                deepstack_width,
                deepstack_nbytes,
            });
    }

    fn fail(&self, room: u64, reason: &str) {
        self.sender
            .lock()
            .expect("embedding sender lock")
            .fail(room, reason);
    }

    /// Completed or failed rooms; their source buffers may now be reused.
    fn concluded(&self) -> Vec<u64> {
        self.sender
            .lock()
            .expect("embedding sender lock")
            .concluded()
    }
}
