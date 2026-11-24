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

//! Bounded native SSD metadata and prepare/release RPCs.
//!
//! Calls block and belong on storage/read workers, never the control thread.
//! A prepared lease owns registered memory at the storage provider. Only a
//! `ReadingLease` exposes addresses: dropping it without proof of retirement
//! leaves cleanup to the provider's GC, rather than freeing a DMA source early.

use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use std::{
    ffi::{c_char, c_int, CStr, CString},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

/// A protocol, transport, or bounded-input error.
#[derive(Debug, thiserror::Error)]
#[error("offload RPC: {0}")]
pub struct Error(pub String);

/// A complete SSD replica the master reports. This is not a source pin.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Replica {
    /// Provider RPC endpoint for preparation and release.
    pub endpoint: String,
    /// Exact object size at the SSD replica.
    pub bytes: u64,
    /// Exact SSD incarnation; [0, 0] denotes the legacy unversioned layout.
    pub generation: [u64; 2],
}

/// Memory prepared at one storage owner under its native buffer lease.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Ready {
    /// Opaque local handle bound to the original provider connection.
    pub batch_id: u64,
    /// Registered object addresses, in preparation key order.
    pub addresses: Vec<u64>,
    /// Transfer engine session registering the addresses.
    pub session: String,
    /// Provider buffer lifetime advertised by preparation.
    pub ttl_ms: u64,
}

type Call = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> *mut c_char;
type Free = unsafe extern "C" fn(*mut c_char);

/// The separately loaded, SDK-compatible RPC bridge. No device resources.
pub struct Client {
    call: Call,
    free: Free,
    _library: libloading::Library,
}

impl Client {
    /// Load `native/mooncake-offload`'s explicitly selected library.
    pub fn load(path: &Path) -> Result<Arc<Self>, Error> {
        // SAFETY: the bridge initializes only client-side RPC statics. Symbols
        // implement the C signatures in native/mooncake-offload/client.cpp.
        unsafe {
            let library = libloading::Library::new(path).map_err(|e| Error(e.to_string()))?;
            let abi = library
                .get::<unsafe extern "C" fn() -> c_int>(b"ts_mooncake_offload_abi")
                .map_err(|e| Error(e.to_string()))?;
            if abi() != 4 {
                return Err(Error("unsupported client ABI".into()));
            }
            Ok(Arc::new(Self {
                call: *library
                    .get(b"ts_mooncake_offload_call")
                    .map_err(|e| Error(e.to_string()))?,
                free: *library
                    .get(b"ts_mooncake_offload_free")
                    .map_err(|e| Error(e.to_string()))?,
                _library: library,
            }))
        }
    }

    fn invoke<T: DeserializeOwned>(&self, endpoint: &str, request: Value) -> Result<T, Error> {
        let (host, port) = if endpoint.starts_with("etcd://") && request["mode"] == "query" {
            (endpoint, "")
        } else {
            let (host, port) = endpoint
                .rsplit_once(':')
                .ok_or_else(|| Error("invalid endpoint".into()))?;
            if host.is_empty() || port.parse::<u16>().ok().is_none_or(|p| p == 0) {
                return Err(Error("invalid endpoint".into()));
            }
            (host, port)
        };
        let host = CString::new(host.trim_start_matches('[').trim_end_matches(']'))
            .map_err(|e| Error(e.to_string()))?;
        let port = CString::new(port).map_err(|e| Error(e.to_string()))?;
        let request = CString::new(request.to_string()).map_err(|e| Error(e.to_string()))?;
        // SAFETY: every input outlives the synchronous call. The bridge returns
        // a NUL-terminated allocation freed once through its own allocator.
        let response = unsafe { (self.call)(host.as_ptr(), port.as_ptr(), request.as_ptr()) };
        if response.is_null() {
            return Err(Error("bridge allocation failed".into()));
        }
        struct Response<'a> {
            ptr: *mut c_char,
            client: &'a Client,
        }
        impl Drop for Response<'_> {
            fn drop(&mut self) {
                unsafe { (self.client.free)(self.ptr) };
            }
        }
        let response = Response {
            ptr: response,
            client: self,
        };
        let value: Value =
            serde_json::from_slice(unsafe { CStr::from_ptr(response.ptr) }.to_bytes())
                .map_err(|e| Error(e.to_string()))?;
        if let Some(error) = value.get("error").or_else(|| value.get("remote_error")) {
            return Err(Error(error.to_string()));
        }
        serde_json::from_value(value).map_err(|e| Error(e.to_string()))
    }

    /// Only complete SSD locations, in input key order. A missing key or one
    /// without an SSD replica has an empty row; other master errors propagate.
    pub fn query(
        &self,
        endpoint: &str,
        tenant: &str,
        keys: &[String],
        timeout: Duration,
    ) -> Result<Vec<Vec<Replica>>, Error> {
        validate_keys(tenant, keys)?;
        #[derive(Deserialize)]
        struct Reply {
            objects: Vec<Value>,
        }
        let reply: Reply = self.invoke(
            endpoint,
            json!({"mode":"query", "tenant":tenant,
            "keys":keys, "timeout_ms":timeout_ms(timeout)?}),
        )?;
        if reply.objects.len() != keys.len() {
            return Err(Error("query column mismatch".into()));
        }
        reply
            .objects
            .into_iter()
            .map(|row| {
                if row.get("error").and_then(Value::as_i64) == Some(-704) {
                    return Ok(Vec::new());
                }
                if let Some(error) = row.get("error") {
                    return Err(Error(error.to_string()));
                }
                let replicas: Vec<Replica> = serde_json::from_value(
                    row.get("ssd")
                        .cloned()
                        .ok_or_else(|| Error("missing replica list".into()))?,
                )
                .map_err(|e| Error(e.to_string()))?;
                if replicas
                    .iter()
                    .any(|r| r.endpoint.is_empty() || r.bytes == 0)
                {
                    return Err(Error("malformed SSD descriptor".into()));
                }
                Ok(replicas)
            })
            .collect()
    }

    /// Prepare packed objects without supplying or allocating any L1/L2
    /// destination. The provider enforces its own bounded staging arena.
    pub fn prepare(
        self: &Arc<Self>,
        endpoint: &str,
        tenant: &str,
        keys: &[String],
        sizes: &[u64],
        generations: &[[u64; 2]],
        timeout: Duration,
    ) -> Result<PreparedLease, Error> {
        validate_keys(tenant, keys)?;
        if sizes.len() != keys.len()
            || generations.len() != keys.len()
            || sizes.iter().any(|n| *n == 0)
            || sizes
                .iter()
                .try_fold(0u64, |a, b| a.checked_add(*b))
                .is_none_or(|n| n > (1 << 31))
        {
            return Err(Error("invalid prepare sizes".into()));
        }
        let began = Instant::now();
        let ready: Ready = self.invoke(
            endpoint,
            json!({"mode":"prepare", "tenant":tenant,
            "keys":keys, "sizes":sizes, "generations":generations,
            "timeout_ms":timeout_ms(timeout)?}),
        )?;
        // The native lease starts at or after dispatch, so this earlier local
        // deadline is conservative and needs no synchronized provider clock.
        let expires = began
            .checked_add(Duration::from_millis(ready.ttl_ms))
            .ok_or_else(|| Error("lease deadline overflow".into()))?;
        let lease = PreparedLease {
            client: self.clone(),
            endpoint: endpoint.into(),
            ready,
            expires,
            release_on_drop: true,
            timeout,
        };
        if lease.ready.addresses.len() != keys.len()
            || lease.ready.session.is_empty()
            || lease.ready.ttl_ms == 0
            || lease
                .ready
                .addresses
                .iter()
                .zip(sizes)
                .any(|(a, n)| *a == 0 || a.checked_add(*n).is_none())
        {
            return Err(Error("malformed prepared source".into()));
        }
        Ok(lease)
    }

    fn release(&self, endpoint: &str, id: u64, timeout: Duration) -> Result<bool, Error> {
        #[derive(Deserialize)]
        struct Reply {
            released: bool,
        }
        let reply: Reply = self.invoke(
            endpoint,
            json!({"mode":"release", "batch_id":id,
            "timeout_ms":timeout_ms(timeout)?}),
        )?;
        Ok(reply.released)
    }
}

fn timeout_ms(timeout: Duration) -> Result<u64, Error> {
    let millis =
        u64::try_from(timeout.as_millis()).map_err(|_| Error("RPC timeout overflow".into()))?;
    if !(1..=10000).contains(&millis) {
        return Err(Error("RPC timeout outside 1..=10000 ms".into()));
    }
    Ok(millis)
}

fn validate_keys(tenant: &str, keys: &[String]) -> Result<(), Error> {
    if tenant.is_empty()
        || tenant.len() > 4096
        || tenant.contains('\0')
        || keys.is_empty()
        || keys.len() > 1024
        || keys
            .iter()
            .any(|key| key.is_empty() || key.len() > 4096 || key.contains('\0'))
    {
        return Err(Error("invalid bounded object query".into()));
    }
    Ok(())
}

/// Prepared source ownership before admission or after confirmed retirement.
/// Drop releases unused sources; it must run on an I/O worker.
pub struct PreparedLease {
    client: Arc<Client>,
    endpoint: String,
    ready: Ready,
    expires: Instant,
    release_on_drop: bool,
    timeout: Duration,
}

impl PreparedLease {
    /// Called on the read worker after ordinary admission and the Device fence.
    /// The remaining source lifetime must cover the whole native read budget.
    pub fn begin_read(mut self, budget: Duration) -> Result<ReadingLease, Error> {
        if budget.is_zero()
            || Instant::now()
                .checked_add(budget)
                .is_none_or(|end| end >= self.expires)
        {
            return Err(Error("insufficient source lease lifetime".into()));
        }
        self.release_on_drop = false;
        Ok(ReadingLease(self))
    }

    /// Confirm that the source remained leased through a completed read.
    /// False or an RPC error forbids publishing the destination prefix.
    pub fn release(mut self) -> Result<bool, Error> {
        self.release_on_drop = false;
        self.client
            .release(&self.endpoint, self.ready.batch_id, self.timeout)
    }
}

impl Drop for PreparedLease {
    fn drop(&mut self) {
        if self.release_on_drop {
            // Cancellation before DMA, or after explicit retirement. Failure
            // leaves the provider's native TTL cleanup in charge.
            let _ = self
                .client
                .release(&self.endpoint, self.ready.batch_id, self.timeout);
        }
    }
}

/// A submitted read owns its source lease. A cancelled/failed read without
/// retirement proof deliberately sends no early release. Its caller must also
/// quarantine destination pages until the transport proves writes have stopped.
pub struct ReadingLease(PreparedLease);

impl ReadingLease {
    /// The registered sources held for this read.
    pub fn source(&self) -> &Ready {
        &self.0.ready
    }

    /// The read backend has proved every native access ended, not merely that
    /// a request deadline elapsed or a cancellation was requested.
    pub fn retired(mut self) -> PreparedLease {
        self.0.release_on_drop = true;
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires TS_OFFLOAD_TEST_FIXTURE naming an isolated SSD object and native bridge"]
    fn native_query_and_source_lease_lifecycle() {
        #[derive(Deserialize)]
        struct Fixture {
            library: String,
            master: String,
            tenant: String,
            key: String,
            bytes: u64,
        }
        let path = std::env::var("TS_OFFLOAD_TEST_FIXTURE").expect("explicit test fixture");
        let fixture: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let client = Client::load(Path::new(&fixture.library)).unwrap();
        let timeout = Duration::from_secs(2);
        let rows = client
            .query(
                &fixture.master,
                &fixture.tenant,
                &[
                    fixture.key.clone(),
                    format!("{}-missing-lease-test", fixture.key),
                ],
                timeout,
            )
            .unwrap();
        assert!(rows[1].is_empty());
        let owner = rows[0]
            .iter()
            .find(|r| r.bytes == fixture.bytes)
            .expect("SSD replica");
        let prepare = || {
            client
                .prepare(
                    &owner.endpoint,
                    &fixture.tenant,
                    std::slice::from_ref(&fixture.key),
                    &[fixture.bytes],
                    &[owner.generation],
                    timeout,
                )
                .unwrap()
        };

        let unused = prepare();
        let id = unused.ready.batch_id;
        drop(unused);
        assert!(!client.release(&owner.endpoint, id, timeout).unwrap());

        let lease = prepare();
        let id = lease.ready.batch_id;
        assert!(lease.begin_read(Duration::from_secs(3600)).is_err());
        assert!(!client.release(&owner.endpoint, id, timeout).unwrap());

        let reading = prepare().begin_read(Duration::from_secs(1)).unwrap();
        assert_eq!(reading.source().addresses.len(), 1);
        let id = reading.source().batch_id;
        // This contract test submits no actual DMA. Dropping a guard in its
        // submitted state must nevertheless not release a possible DMA source.
        drop(reading);
        assert!(client.release(&owner.endpoint, id, timeout).unwrap());

        let reading = prepare().begin_read(Duration::from_secs(1)).unwrap();
        let id = reading.source().batch_id;
        assert!(reading.retired().release().unwrap());
        assert!(!client.release(&owner.endpoint, id, timeout).unwrap());
    }

    #[test]
    fn inputs_and_timeouts_are_bounded_before_native_calls() {
        assert!(validate_keys("default", &["object".into()]).is_ok());
        assert!(validate_keys("default", &["a\0b".into()]).is_err());
        assert!(validate_keys("default", &vec!["key".into(); 1025]).is_err());
        assert!(validate_keys("", &["object".into()]).is_err());
        assert!(timeout_ms(Duration::ZERO).is_err());
        assert!(timeout_ms(Duration::from_secs(11)).is_err());
        assert_eq!(timeout_ms(Duration::from_millis(200)).unwrap(), 200);
    }
}
