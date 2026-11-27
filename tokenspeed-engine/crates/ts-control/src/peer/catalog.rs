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

//! A bounded catalog of immutable source layouts, independent of live Decode.
//! Only model-generation layouts are advertised; no request or prefix inventory.
use super::{storage::Error, stored_layout::StoredLayout};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use ts_l3::MetadataHandle;

const LAYOUT_SLOTS: usize = 8;
const REFRESH: Duration = Duration::from_secs(10);

pub(super) fn keys(semantic: &[u8; 32], weight: &str) -> Vec<String> {
    let mut hash = Sha256::new();
    hash.update(b"tokenspeed-l3-layout-v1");
    hash.update(semantic);
    hash.update(weight.as_bytes());
    let namespace = format!("{:x}", hash.finalize());
    (0..LAYOUT_SLOTS)
        .map(|slot| format!("tsl3-layout-{namespace}-{slot}"))
        .collect()
}

/// Create-only slots let different physical layouts coexist without a mutable
/// central index. Racing publishers read back the winner before using a slot.
pub(super) fn publish(
    io: &MetadataHandle,
    keys: &[String],
    source: &StoredLayout,
) -> Result<(), Error> {
    let bytes = serde_json::to_vec(source).map_err(|e| Error(e.to_string()))?;
    for key in keys {
        let existing = io.get(key).map_err(|e| Error(e.to_string()))?;
        if existing.as_deref() == Some(bytes.as_slice()) {
            return Ok(());
        }
        if existing.is_none() {
            io.put(key, &bytes).map_err(|e| Error(e.to_string()))?;
            if io.get(key).map_err(|e| Error(e.to_string()))?.as_deref() == Some(bytes.as_slice()) {
                return Ok(());
            }
        }
    }
    Err(Error(
        "persistent layout catalog is full; private-layout PUT remains available".into(),
    ))
}
pub(super) fn discover(
    io: &MetadataHandle,
    keys: &[String],
    semantic: &[u8; 32],
    weight: &str,
) -> Result<Vec<Arc<StoredLayout>>, Error> {
    let mut layouts = Vec::new();
    for key in keys {
        let Some(bytes) = io.get(key).map_err(|e| Error(e.to_string()))? else {
            continue;
        };
        let Ok(source) = serde_json::from_slice::<StoredLayout>(&bytes) else {
            continue;
        };
        if source.semantic == *semantic
            && source.weight_version == weight
            && source.parsed().is_ok()
        {
            layouts.push(Arc::new(source));
        }
    }
    Ok(layouts)
}
struct Worker {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
    layouts: Arc<RwLock<Vec<Arc<StoredLayout>>>>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.thread.thread().unpark();
        // A blocked SDK call owns its MetadataHandle. Executor shutdown drains
        // its active guard or quarantines the connection; never free its buffer.
    }
}
/// Background publisher or discovery cache for bounded source layouts.
#[derive(Clone)]
pub struct Catalog(Arc<Worker>);
impl Catalog {
    /// Publish a Decode layout, or discover layouts for a direct Prefill reader.
    pub fn start(
        io: MetadataHandle,
        semantic: [u8; 32],
        weight: String,
        source: Option<StoredLayout>,
    ) -> Result<Self, Error> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let layouts = Arc::new(RwLock::new(Vec::new()));
        let published = layouts.clone();
        let keys = keys(&semantic, &weight);
        let thread = thread::Builder::new()
            .name("ts-l3-layouts".into())
            .spawn(move || {
                while !stopping.load(Ordering::Acquire) {
                    if let Some(source) = &source {
                        if let Err(error) = publish(&io, &keys, source) {
                            tracing::warn!(%error,"L3 layout advertisement unavailable");
                        }
                    } else {
                        match discover(&io, &keys, &semantic, &weight) {
                            Ok(value) => {
                                *published.write().unwrap_or_else(|e| e.into_inner()) = value
                            }
                            Err(error) => {
                                tracing::warn!(%error,"L3 source layout discovery unavailable")
                            }
                        }
                    }
                    if !stopping.load(Ordering::Acquire) {
                        thread::park_timeout(REFRESH);
                    }
                }
            })
            .map_err(|e| Error(e.to_string()))?;
        Ok(Self(Arc::new(Worker {
            stop,
            thread,
            layouts,
        })))
    }
    /// Scheduler-side planning reads a snapshot and never waits for store I/O.
    pub fn snapshot(&self) -> Vec<Arc<StoredLayout>> {
        self.0
            .layouts
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}
