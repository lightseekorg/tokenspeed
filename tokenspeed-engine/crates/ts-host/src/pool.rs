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

//! A shared L2 pool on the host: the rank's client of the pool's owner
//! (ts-control's `shared_l2::client`) over the rank's transfer engine, and
//! the shared L2 handle, through which the rank's Python reaches both.
//!
//! The rank joins the pool at device build (`cache/shared_l2/rank.py`). It
//! starts the transfer engine through the shared L2 handle, with the Device
//! cache registered, says its hello to the owner over its own Python
//! connection, and then starts the client through the shared L2 handle,
//! under the id the hello chose. The loop then takes the client; the data
//! plane's forward thread hands each plan's load to the shared L2 handle,
//! its cache poll reads the loads that finished from the handle's loader
//! (`PythonDataPlane::poll_cache_events`), and the rank's shutdown closes
//! it.

use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use parking_lot::Mutex;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use ts_control::{
    peer::read::{EngineReader, PageReader},
    shared_l2::{
        client::{self, ClientSettings, EngineWriter, PageWriter, RankClient, RankThreads},
        geometry::{PageField, PageGroup, PoolGeometry, Provider},
        load::{Loads, SignalWords},
        PoolClient,
    },
};

/// An idle rank pings the owner at this pace: the owner drops a client it
/// has not heard for its client timeout.
const KEEPALIVE: Duration = Duration::from_secs(5);

/// How long a close waits for the client's threads.
const CLOSE_WAIT: Duration = Duration::from_secs(10);

/// The rank's signal words, the start and ready words of its loads:
/// `count` 32-bit words at `address`, host memory the device maps (the
/// region of `HostSignals` in `cache/shared_l2/executor.py`). The rank
/// keeps them allocated until the client's threads have ended.
pub(crate) struct HostSignals {
    address: usize,
    count: usize,
}

impl HostSignals {
    #[allow(unsafe_code)]
    fn word(&self, word: usize) -> &AtomicU32 {
        assert!(
            word < self.count,
            "signal word {word} of {} words",
            self.count
        );
        // SAFETY: the word lies within the `count` words at `address`, which
        // outlive the threads that reach it, and is read and written as a
        // whole by them, by the forward thread's streams and by the device.
        unsafe { AtomicU32::from_ptr((self.address as *mut u32).add(word)) }
    }
}

impl SignalWords for HostSignals {
    fn num_words(&self) -> usize {
        self.count
    }

    fn get(&self, word: usize) -> u32 {
        self.word(word).load(Ordering::Relaxed)
    }

    fn set(&self, word: usize, value: u32) {
        self.word(word).store(value, Ordering::Relaxed);
    }
}

/// Start the rank's transfer engine and register `regions`, the Device
/// cache's buffers, as a peer's engine starts (`peer::start`).
fn start_engine(
    library: &Path,
    hostname: &str,
    protocol: &str,
    device_filter: &str,
    regions: &[(usize, usize)],
) -> Result<Arc<ts_mooncake::Engine>, String> {
    let library = ts_mooncake::Library::load(library).map_err(|error| error.to_string())?;
    if !library.reads() {
        return Err(format!(
            "{} has no READ entry points: a shared L2 pool needs a tokenspeed-mooncake \
             build with ts_mooncake_read",
            library.path().display()
        ));
    }
    let engine = ts_mooncake::Engine::start(library, hostname, protocol, device_filter)
        .map_err(|error| error.to_string())?;
    for &(address, length) in regions {
        engine
            .register(address, length)
            .map_err(|error| error.to_string())?;
    }
    Ok(Arc::new(engine))
}

/// What the rank's client starts from, and where its signal words are
/// (`SharedL2ClientSettings`).
pub(crate) fn client_settings(
    settings: &Bound<'_, PyAny>,
) -> PyResult<(ClientSettings, HostSignals)> {
    let invalid = |message: String| PyRuntimeError::new_err(message);
    let workers = |name: &str| -> PyResult<NonZeroUsize> {
        NonZeroUsize::new(settings.getattr(name)?.extract()?)
            .ok_or_else(|| invalid(format!("a shared L2 rank needs at least one of {name}")))
    };
    let providers = settings
        .getattr("providers")?
        .try_iter()?
        .map(|provider| {
            let provider = provider?;
            Ok(Provider {
                session: provider.getattr("session")?.extract()?,
                base: provider.getattr("base")?.extract()?,
                lcm_blocks: provider.getattr("lcm_blocks")?.extract()?,
                start: provider.getattr("start")?.extract()?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let groups = settings
        .getattr("groups")?
        .try_iter()?
        .map(|group| {
            let group = group?;
            let fields = group
                .getattr("fields")?
                .try_iter()?
                .map(|field| {
                    let field = field?;
                    Ok(PageField {
                        device_address: field.getattr("device_address")?.extract()?,
                        device_stride: field.getattr("device_stride")?.extract()?,
                        bytes: field.getattr("bytes")?.extract()?,
                        page_offset: field.getattr("page_offset")?.extract()?,
                        layer: field.getattr("layer")?.extract()?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PageGroup {
                packing: group.getattr("packing")?.extract()?,
                page_bytes: group.getattr("page_bytes")?.extract()?,
                fields,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let geometry = PoolGeometry {
        lcm_bytes: settings.getattr("lcm_bytes")?.extract()?,
        providers,
        groups,
        num_layers: settings.getattr("num_layers")?.extract()?,
    };
    Ok((
        ClientSettings {
            owner_address: settings.getattr("owner_address")?.extract()?,
            client_id: settings.getattr("client_id")?.extract()?,
            request_timeout: Duration::from_millis(
                settings.getattr("request_timeout_ms")?.extract()?,
            ),
            keepalive: KEEPALIVE,
            load_workers: workers("load_workers")?,
            write_workers: workers("write_workers")?,
            geometry,
        },
        HostSignals {
            address: settings.getattr("signal_words_address")?.extract()?,
            count: settings.getattr("num_signal_words")?.extract()?,
        },
    ))
}

/// The shared L2 handle: the rank's shared L2 pool as its Python reaches
/// it. The host makes one per rank, and a rank with `--shared-l2-config`
/// starts it.
#[pyclass(frozen, name = "SharedL2Handle")]
pub struct Handle {
    /// The transfer engine, from `start_engine` until the client takes it.
    engine: Mutex<Option<Arc<ts_mooncake::Engine>>>,
    /// The client for the loop, from `start_client` until the loop takes it.
    client: Mutex<Option<RankClient>>,
    threads: OnceLock<RankThreads>,
}

impl Handle {
    /// A shared L2 handle whose rank has not joined a pool.
    pub fn new() -> Self {
        Self {
            engine: Mutex::new(None),
            client: Mutex::new(None),
            threads: OnceLock::new(),
        }
    }

    /// The rank's client of the pool's owner, for the loop; `None` when the
    /// rank joined no pool.
    pub fn take_client(&self) -> Option<Box<dyn PoolClient + Send>> {
        self.client
            .lock()
            .take()
            .map(|client| Box::new(client) as Box<dyn PoolClient + Send>)
    }

    /// Start the client with what it reads through, writes through and
    /// signals a load's layers through.
    pub(crate) fn start_client_with(
        &self,
        settings: ClientSettings,
        reader: Arc<dyn PageReader>,
        writer: Arc<dyn PageWriter>,
        signal_words: Arc<dyn SignalWords>,
    ) -> Result<(), String> {
        let (client, threads) =
            client::start(settings, reader, writer, signal_words).map_err(|error| error.0)?;
        if self.threads.set(threads).is_err() {
            return Err("the rank's shared L2 client started twice".to_owned());
        }
        *self.client.lock() = Some(client);
        Ok(())
    }

    /// The rank's loads, once its client started; `None` for a rank that
    /// joined no pool.
    pub(crate) fn loads(&self) -> Option<&Loads> {
        self.threads.get().map(RankThreads::loads)
    }
}

#[pymethods]
impl Handle {
    /// Start the rank's transfer engine and register its Device cache:
    /// before the rank's hello, as registering takes time. Blocks without
    /// the GIL.
    ///
    /// Args:
    ///     library: Path of `libtokenspeed_transfer_engine.so`.
    ///     hostname: The address other engines reach this one at.
    ///     protocol: The engine's transport.
    ///     device_filter: The devices to use, comma-separated; empty for all.
    ///     regions: `(address, bytes)` of the Device cache's buffers.
    ///
    /// Raises:
    ///     RuntimeError: The library did not load or has no READ entry
    ///         points, the engine did not start, a registration failed, or
    ///         the engine was started before.
    #[pyo3(signature = (*, library, hostname, protocol, device_filter, regions))]
    fn start_engine(
        &self,
        py: Python<'_>,
        library: PathBuf,
        hostname: &str,
        protocol: &str,
        device_filter: &str,
        regions: Vec<(usize, usize)>,
    ) -> PyResult<()> {
        let engine = py
            .detach(|| start_engine(&library, hostname, protocol, device_filter, &regions))
            .map_err(PyRuntimeError::new_err)?;
        if self.engine.lock().replace(engine).is_some() {
            return Err(PyRuntimeError::new_err(
                "the rank's shared L2 transfer engine started twice",
            ));
        }
        Ok(())
    }

    /// Start the rank's client of the pool's owner, from the
    /// `SharedL2ClientSettings` of the pool the rank just joined: its
    /// connections, its write and notice threads, its keepalive and its
    /// loads. Blocks without the GIL.
    fn start_client(&self, py: Python<'_>, settings: &Bound<'_, PyAny>) -> PyResult<()> {
        let (settings, signal_words) = client_settings(settings)?;
        let engine = self.engine.lock().take().ok_or_else(|| {
            PyRuntimeError::new_err("the rank's shared L2 transfer engine has not started")
        })?;
        // The client holds the engine from here, past its threads: the
        // close that sees them all end stops it.
        py.detach(|| {
            self.start_client_with(
                settings,
                Arc::new(EngineReader(engine.clone())),
                Arc::new(EngineWriter(engine)),
                Arc::new(signal_words),
            )
        })
        .map_err(PyRuntimeError::new_err)
    }

    /// Hand a plan's load over: `rows` are `(group, proxy Host block,
    /// Device block)`, `start_word` is the load's start word, which its
    /// layers' ready words follow, and `seq` the value that marks the load
    /// in them. The forward thread runs this, inside the plan's execution;
    /// it never waits.
    #[pyo3(signature = (*, op_ids, rows, start_word, seq))]
    fn submit_load(
        &self,
        op_ids: Vec<u32>,
        rows: Vec<(u32, i32, i32)>,
        start_word: usize,
        seq: u32,
    ) -> PyResult<()> {
        self.loads()
            .ok_or_else(|| PyRuntimeError::new_err("the rank's shared L2 client has not started"))?
            .submit(op_ids, &rows, start_word, seq)
            .map_err(|error| PyRuntimeError::new_err(error.0))
    }

    /// Stop the client's threads once what is queued ran, wait for them no
    /// longer than the close wait, and then stop the transfer engine, which
    /// deregisters the cache, in this call. Returns whether every thread
    /// ended: one that still waits for the owner or for a copy keeps the
    /// engine, the registered cache and the signal words in use, and then
    /// the engine is left running for the process's exit to take. Blocks
    /// without the GIL.
    fn close(&self, py: Python<'_>) -> bool {
        match self.threads.get() {
            Some(threads) => py.detach(|| threads.close(CLOSE_WAIT)),
            // No client started: no thread to stop.
            None => true,
        }
    }
}
