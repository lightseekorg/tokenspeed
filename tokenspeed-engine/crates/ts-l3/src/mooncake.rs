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

//! The Mooncake store as an L3 store: zero-copy gets and puts against the
//! registered Host pool, as SGLang's HiCache and vLLM's connector use it.

use std::{path::Path, sync::Arc};

use ts_mooncake::store::{regex_escape, Store as Client, StoreLibrary, StoreSetup};

use crate::{
    layout::Range,
    store::{HostBuffer, IoResult, Store, StoreError},
};

/// `OBJECT_NOT_FOUND`: a metadata miss before any transfer started.
const OBJECT_NOT_FOUND: i64 = -704;
/// `OBJECT_ALREADY_EXISTS`: a create-only rejection before any transfer.
const OBJECT_ALREADY_EXISTS: i32 = -705;
/// `NO_AVAILABLE_HANDLE`: the master's allocation rejection, before any
/// source transfer.
const NO_AVAILABLE_HANDLE: i32 = -200;

/// A store client with the Host pool registered.
pub struct MooncakeStore {
    client: Client,
    host: HostBuffer,
}

impl MooncakeStore {
    /// Load `library`, connect as `setup` says and register the Host pool.
    pub fn connect(
        library: &Path,
        setup: &StoreSetup,
        host: HostBuffer,
    ) -> Result<Self, ts_mooncake::Error> {
        let library = StoreLibrary::load(library)?;
        let client = Client::connect(Arc::clone(&library), setup)?;
        if host.len > 0 {
            client.register_buffer(host.base, host.len)?;
        }
        tracing::info!(
            protocol = %setup.protocol,
            master = %setup.master_server_address,
            segment = setup.global_segment_size,
            "Mooncake Store L3 ready"
        );
        Ok(Self { client, host })
    }

    fn addresses(host: HostBuffer, ranges: &[Range]) -> (Vec<usize>, Vec<usize>) {
        ranges
            .iter()
            .map(|&range| (host.address(range), range.size))
            .unzip()
    }
}

fn failed(error: ts_mooncake::Error) -> StoreError {
    StoreError(error.to_string())
}

/// A zero-copy GET's outcome from its per-key results: the bytes read or
/// an error code. A miss precedes any transfer, so it retires as an object
/// failure; a short read retires too, but the object is invalid. Another
/// error may leave writes into the pool in flight.
fn get_outcome(results: &[i64], sizes: &[usize]) -> IoResult {
    let full = |result: i64, size: usize| u64::try_from(result).is_ok_and(|n| n == size as u64);
    IoResult {
        success: results
            .iter()
            .zip(sizes)
            .map(|(&result, &size)| full(result, size))
            .collect(),
        retired: results
            .iter()
            .all(|&result| result >= 0 || result == OBJECT_NOT_FOUND),
        object_failures: results
            .iter()
            .zip(sizes)
            .map(|(&result, &size)| {
                result == OBJECT_NOT_FOUND || (result >= 0 && !full(result, size))
            })
            .collect(),
    }
}

/// A PUT's outcome: the objects that already `exist` succeed without a
/// write, and each `missing` key takes its write's result. An existing
/// object (another writer's) is a success; an allocation rejection retires
/// as a failure the scheduler offers again; other errors do not prove that
/// the source bytes are no longer read.
fn put_outcome(exists: Vec<bool>, missing: &[usize], results: &[i32]) -> IoResult {
    let mut success = exists;
    for (&index, &result) in missing.iter().zip(results) {
        success[index] = result == 0 || result == OBJECT_ALREADY_EXISTS;
    }
    IoResult {
        object_failures: vec![false; success.len()],
        success,
        retired: results
            .iter()
            .all(|&result| [0, OBJECT_ALREADY_EXISTS, NO_AVAILABLE_HANDLE].contains(&result)),
    }
}

/// The objects of the namespace `prefix`, as the store's regular expression.
fn namespace_pattern(prefix: &str) -> String {
    format!("^{}.*", regex_escape(prefix))
}

impl Store for MooncakeStore {
    fn failures_retire(&self) -> bool {
        // A failed SDK call does not prove that RDMA stopped using the Host
        // bytes.
        false
    }

    fn batch_exists(&self, keys: &[String]) -> Result<Vec<bool>, StoreError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let results = self.client.batch_is_exist(keys).map_err(failed)?;
        if results.iter().any(|&flag| flag != 0 && flag != 1) {
            return Err(StoreError(format!(
                "Mooncake existence query failed: {results:?}"
            )));
        }
        Ok(results.into_iter().map(|flag| flag == 1).collect())
    }

    fn batch_get_into(
        &self,
        keys: &[String],
        host: HostBuffer,
        ranges: &[Range],
    ) -> Result<IoResult, StoreError> {
        if keys.len() != ranges.len() {
            return Err(StoreError("ragged L3 get".to_owned()));
        }
        if keys.is_empty() {
            return Ok(IoResult {
                success: Vec::new(),
                retired: true,
                object_failures: Vec::new(),
            });
        }
        let (addrs, sizes) = Self::addresses(host, ranges);
        let results = self
            .client
            .batch_get_into(keys, &addrs, &sizes)
            .map_err(failed)?;
        Ok(get_outcome(&results, &sizes))
    }

    fn batch_put_from(
        &self,
        keys: &[String],
        host: HostBuffer,
        ranges: &[Range],
    ) -> Result<IoResult, StoreError> {
        if keys.len() != ranges.len() {
            return Err(StoreError("ragged L3 put".to_owned()));
        }
        // A metadata failure precedes any access to the source bytes: no
        // write starts, and nothing is left reading them.
        let exists = match self.batch_exists(keys) {
            Ok(exists) => exists,
            Err(error) => {
                tracing::error!(%error, "Mooncake existence query before PUT failed");
                return Ok(IoResult {
                    success: vec![false; keys.len()],
                    retired: true,
                    object_failures: vec![false; keys.len()],
                });
            }
        };
        let missing: Vec<usize> = (0..keys.len()).filter(|&i| !exists[i]).collect();
        if missing.is_empty() {
            return Ok(put_outcome(exists, &missing, &[]));
        }
        let put_keys: Vec<String> = missing.iter().map(|&i| keys[i].clone()).collect();
        let put_ranges: Vec<Range> = missing.iter().map(|&i| ranges[i]).collect();
        let (addrs, sizes) = Self::addresses(host, &put_ranges);
        let results = self
            .client
            .batch_put_from(&put_keys, &addrs, &sizes)
            .map_err(failed)?;
        Ok(put_outcome(exists, &missing, &results))
    }

    fn remove_by_prefix(&self, prefix: &str) -> bool {
        match self
            .client
            .remove_by_regex(&namespace_pattern(prefix), true)
        {
            // The count removed; a negative value is an error.
            Ok(removed) if removed >= 0 => true,
            Ok(status) => {
                tracing::error!(status, prefix, "clearing the Mooncake L3 namespace failed");
                false
            }
            Err(error) => {
                tracing::error!(%error, prefix, "clearing the Mooncake L3 namespace failed");
                false
            }
        }
    }

    fn close(&self) {
        if self.host.len > 0 {
            if let Err(error) = self.client.unregister_buffer(self.host.base) {
                tracing::warn!(%error, "deregistering the Host pool from the store failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_objects_are_not_rewritten_and_rejections_retire() {
        // `existing` is stored; `saved` writes; `pressure` meets a full
        // store, a failure the scheduler offers again.
        let outcome = put_outcome(vec![true, false, false], &[1, 2], &[0, NO_AVAILABLE_HANDLE]);
        assert_eq!(outcome.success, [true, true, false]);
        assert!(outcome.retired);
        assert_eq!(outcome.object_failures, [false, false, false]);
        // Another writer's concurrent create is a success.
        let outcome = put_outcome(vec![false], &[0], &[OBJECT_ALREADY_EXISTS]);
        assert_eq!(outcome.success, [true]);
        assert!(outcome.retired);
    }

    #[test]
    fn a_failed_put_does_not_retire() {
        for error in [-1, -800] {
            let outcome = put_outcome(vec![true, false, false], &[1, 2], &[0, error]);
            assert_eq!(outcome.success, [true, true, false]);
            assert!(!outcome.retired);
        }
    }

    #[test]
    fn a_get_miss_retires_as_an_object_failure() {
        // The object was evicted after the existence query.
        let outcome = get_outcome(&[OBJECT_NOT_FOUND, 8], &[8, 8]);
        assert_eq!(outcome.success, [false, true]);
        assert!(outcome.retired);
        assert_eq!(outcome.object_failures, [true, false]);
        // An unknown error may still write into the pool.
        let outcome = get_outcome(&[OBJECT_NOT_FOUND, -800], &[8, 8]);
        assert!(!outcome.retired);
        assert_eq!(outcome.object_failures, [true, false]);
    }

    #[test]
    fn a_short_get_is_an_invalid_object() {
        let outcome = get_outcome(&[4, 8], &[8, 8]);
        assert_eq!(outcome.success, [false, true]);
        assert!(outcome.retired);
        assert_eq!(outcome.object_failures, [true, false]);
        let outcome = get_outcome(&[0, -1], &[8, 8]);
        assert_eq!(outcome.success, [false, false]);
        assert!(!outcome.retired);
        assert_eq!(outcome.object_failures, [true, false]);
    }

    #[test]
    fn a_namespace_clear_matches_only_its_prefix() {
        assert_eq!(namespace_pattern("model.v1_"), r"^model\.v1_.*");
        assert_eq!(namespace_pattern("tsl3-ab12"), r"^tsl3\-ab12.*");
    }
}
