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

//! Everything an executor starts from, as the data plane resolves it at
//! startup (`cache/l3/config.py`'s `L3Settings`): the limits, the namespace,
//! the Host pool and the store.

use std::path::PathBuf;

use ts_mooncake::store::StoreSetup;

use crate::{
    executor::{monotonic_clock, Executor, ExecutorConfig, Namespace},
    layout::HostLayout,
    mooncake::MooncakeStore,
    store::{process_memory_store, HostBuffer, Store},
};

/// The store an executor runs against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreSettings {
    /// The process's in-memory store (`--kvstore-storage-backend memory`),
    /// shared by its executors.
    Memory,
    /// The Mooncake store, through `library`.
    Mooncake {
        /// `libmooncake_store.so`.
        library: PathBuf,
        /// How the client connects.
        setup: StoreSetup,
    },
}

/// One rank's L3 tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// The limits and the workers.
    pub config: ExecutorConfig,
    /// What objects are named in.
    pub namespace: Namespace,
    /// The weight version the namespace starts at.
    pub weight_version: String,
    /// The Host pool's bytes.
    pub host: HostBuffer,
    /// Its geometry and this rank's place among the owners.
    pub layout: HostLayout,
    /// The store.
    pub store: StoreSettings,
}

impl Settings {
    /// Connect the store and start the workers.
    pub fn start(&self) -> Result<Executor, ts_mooncake::Error> {
        let store: Box<dyn Store> = match &self.store {
            StoreSettings::Memory => Box::new(process_memory_store()),
            StoreSettings::Mooncake { library, setup } => {
                Box::new(MooncakeStore::connect(library, setup, self.host)?)
            }
        };
        Ok(Executor::start(
            store,
            self.host,
            self.layout.clone(),
            self.namespace.clone(),
            &self.weight_version,
            self.config,
            monotonic_clock(),
        ))
    }
}

#[cfg(feature = "python")]
impl Settings {
    /// From the data plane's `L3Settings`.
    pub fn from_python(settings: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use std::time::Duration;

        use pyo3::prelude::*;

        let runtime = settings.getattr("runtime")?;
        let scheduler = runtime.getattr("scheduler")?;
        let config = ExecutorConfig {
            max_operations: scheduler.getattr("max_operations")?.extract()?,
            max_keys_per_operation: scheduler.getattr("max_keys_per_operation")?.extract()?,
            query_workers: runtime.getattr("query_workers")?.extract()?,
            read_workers: runtime.getattr("read_workers")?.extract()?,
            write_workers: runtime.getattr("write_workers")?.extract()?,
            failure_threshold: runtime.getattr("failure_threshold")?.extract()?,
            recovery_interval: Duration::from_millis(
                runtime.getattr("recovery_interval_ms")?.extract()?,
            ),
        };
        let groups: Vec<(usize, usize, usize)> = settings.getattr("groups")?.extract()?;
        let layout = HostLayout {
            groups: groups
                .into_iter()
                .map(
                    |(packing, block_bytes, shard_count)| crate::layout::GroupLayout {
                        packing,
                        block_bytes,
                        shard_count,
                    },
                )
                .collect(),
            lcm_block_bytes: settings.getattr("lcm_block_bytes")?.extract()?,
            num_lcm_blocks: settings.getattr("num_lcm_blocks")?.extract()?,
            owner_rank: settings.getattr("owner_rank")?.extract()?,
            tp_rank: settings.getattr("tp_rank")?.extract()?,
        };
        let host = HostBuffer {
            base: settings.getattr("host_buffer")?.extract()?,
            len: settings.getattr("host_buffer_bytes")?.extract()?,
        };
        if host.len != layout.bytes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "the {}-byte Host pool does not match its layout ({} bytes)",
                host.len,
                layout.bytes()
            )));
        }
        let backend: String = settings.getattr("backend")?.extract()?;
        let store = match backend.as_str() {
            "memory" => StoreSettings::Memory,
            "mooncake" => {
                let mooncake = settings.getattr("mooncake")?;
                StoreSettings::Mooncake {
                    library: PathBuf::from(mooncake.getattr("library")?.extract::<String>()?),
                    setup: StoreSetup {
                        local_hostname: mooncake.getattr("local_hostname")?.extract()?,
                        metadata_server: mooncake.getattr("metadata_server")?.extract()?,
                        global_segment_size: mooncake.getattr("global_segment_size")?.extract()?,
                        local_buffer_size: mooncake.getattr("local_buffer_size")?.extract()?,
                        protocol: mooncake.getattr("protocol")?.extract()?,
                        device_name: mooncake.getattr("device_name")?.extract()?,
                        master_server_address: mooncake
                            .getattr("master_server_address")?
                            .extract()?,
                        tenant_id: mooncake.getattr("tenant_id")?.extract()?,
                    },
                }
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported L3 store {other:?}"
                )))
            }
        };
        Ok(Self {
            config,
            namespace: Namespace {
                deployment_id: settings.getattr("deployment_id")?.extract()?,
                has_draft: settings.getattr("has_draft")?.extract()?,
            },
            weight_version: settings.getattr("weight_version")?.extract()?,
            host,
            layout,
            store,
        })
    }
}
