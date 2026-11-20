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

use std::{
    collections::HashSet,
    path::Path,
    sync::{Arc, Mutex},
};

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
/// Ownership rejection at End after the audited default Host transfers finish.
const ILLEGAL_CLIENT: i32 = -601;

/// A store client with the Host pool registered.
pub struct MooncakeStore {
    client: Client,
    host: HostBuffer,
    in_flight_puts: InFlightPuts,
}

/// One Store client is shared by several workers. Keep its concurrent writes
/// to the same complete object key from reusing that client's write ownership.
#[derive(Default)]
struct InFlightPuts {
    keys: Mutex<HashSet<String>>,
}

struct PutClaim<'a> {
    registry: &'a InFlightPuts,
    entries: Vec<(usize, String)>,
    release_on_drop: bool,
}

impl InFlightPuts {
    fn claim<'a>(&'a self, keys: &[String], candidates: &[usize]) -> PutClaim<'a> {
        let mut claimed = self.keys.lock().expect("Mooncake in-flight PUT lock");
        let entries = candidates
            .iter()
            .filter_map(|&index| {
                claimed
                    .insert(keys[index].clone())
                    .then(|| (index, keys[index].clone()))
            })
            .collect();
        PutClaim {
            registry: self,
            entries,
            release_on_drop: true,
        }
    }
}

impl PutClaim<'_> {
    fn native_started(&mut self) {
        self.release_on_drop = false;
    }

    fn native_finished(&mut self, retired: bool) {
        // The default Host result contract below is the only retirement
        // evidence. A lost/timed-out End may still act on this client's key;
        // waiting longer cannot make it safe to release that key's claim.
        self.release_on_drop = retired;
    }
}

impl Drop for PutClaim<'_> {
    fn drop(&mut self) {
        if !self.release_on_drop {
            // The executor retains this batch's Host references and operation
            // credits too, so these keys remain bounded by the same admission
            // budgets. Do not permit another write while old I/O is unknown.
            return;
        }
        // A guard can drop during unwinding; another panic would abort.
        let mut claimed = self.registry.keys.lock().unwrap_or_else(|p| p.into_inner());
        for (_, key) in &self.entries {
            claimed.remove(key);
        }
    }
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
        Ok(Self {
            client,
            host,
            in_flight_puts: InFlightPuts::default(),
        })
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

/// Keep the SDK's accepted/existing PUT acknowledgment as a historical hint,
/// not a promise that another writer's object is readable yet. Actual restore
/// still needs full-length GET success independently of that acknowledgment.
/// The audited default Host C ABI uses the reliable memory policy: End follows
/// successful allocated transfers, and a failed transfer is not hidden by a
/// later Revoke rejection. ILLEGAL_CLIENT therefore retires sources without
/// proving publication. This does not generalize to flexible memory/NoF or
/// custom backend policies; see docs/design/async-l3-cache.md for source links.
fn put_outcome(exists: Vec<bool>, missing: &[usize], results: &[i32]) -> IoResult {
    let mut success = exists;
    for (&index, &result) in missing.iter().zip(results) {
        success[index] = result == 0 || result == OBJECT_ALREADY_EXISTS;
    }
    IoResult {
        object_failures: vec![false; success.len()],
        success,
        retired: results.iter().all(|&result| {
            [
                0,
                OBJECT_ALREADY_EXISTS,
                NO_AVAILABLE_HANDLE,
                ILLEGAL_CLIENT,
            ]
            .contains(&result)
        }),
    }
}

/// The objects of the namespace `prefix`, as the store's regular expression.
fn namespace_pattern(prefix: &str) -> String {
    format!("^{}.*", regex_escape(prefix))
}

impl Store for MooncakeStore {
    fn failures_retire(&self) -> bool {
        // A top-level SDK error supplies no per-key status vector and does not
        // prove that native access to the Host bytes stopped.
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
        let candidates: Vec<usize> = (0..keys.len()).filter(|&i| !exists[i]).collect();
        let mut claim = self.in_flight_puts.claim(keys, &candidates);
        let missing: Vec<usize> = claim.entries.iter().map(|(index, _)| *index).collect();
        if missing.is_empty() {
            return Ok(put_outcome(exists, &missing, &[]));
        }
        let put_keys: Vec<String> = missing.iter().map(|&i| keys[i].clone()).collect();
        let put_ranges: Vec<Range> = missing.iter().map(|&i| ranges[i]).collect();
        let (addrs, sizes) = Self::addresses(host, &put_ranges);
        claim.native_started();
        let results = self
            .client
            .batch_put_from(&put_keys, &addrs, &sizes)
            .map_err(failed)?;
        let result = put_outcome(exists, &missing, &results);
        claim.native_finished(result.retired);
        Ok(result)
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
    use std::sync::Barrier;

    const RPC_FAIL: i32 = -900;
    const RPC_TIMEOUT: i32 = -901;

    #[test]
    fn put_claims_filter_only_overlapping_objects_and_release_before_submission() {
        let registry = InFlightPuts::default();
        let keys = vec![
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "a".to_owned(),
        ];
        let first = registry.claim(&keys, &[0, 1, 3]);
        assert_eq!(first.entries, [(0, "a".to_owned()), (1, "b".to_owned())]);
        let second = registry.claim(&keys, &[1, 2]);
        assert_eq!(second.entries, [(2, "c".to_owned())]);
        drop(first); // No native call occurred, including an early local error.
        let next = registry.claim(&keys, &[0, 1, 2]);
        assert_eq!(next.entries, [(0, "a".to_owned()), (1, "b".to_owned())]);
        drop(next);
        drop(second);
        assert!(registry.keys.lock().unwrap().is_empty());
    }

    #[test]
    fn default_host_put_results_release_only_safely_retired_claims() {
        for (result, retired) in [
            (0, true),
            (OBJECT_ALREADY_EXISTS, true),
            (NO_AVAILABLE_HANDLE, true),
            (ILLEGAL_CLIENT, true),
            (-800, false),
            (-1, false),
        ] {
            let registry = InFlightPuts::default();
            let keys = vec!["a".to_owned()];
            let mut claim = registry.claim(&keys, &[0]);
            claim.native_started();
            let outcome = put_outcome(vec![false], &[0], &[result]);
            assert_eq!(outcome.retired, retired);
            claim.native_finished(outcome.retired);
            drop(claim);
            let next = registry.claim(&keys, &[0]);
            assert_eq!(next.entries.is_empty(), !retired);
        }
    }

    #[test]
    fn an_unknown_end_rpc_retains_both_the_put_claim_and_source_ownership() {
        for result in [RPC_FAIL, RPC_TIMEOUT] {
            let registry = InFlightPuts::default();
            let keys = vec!["a".to_owned()];
            let mut claim = registry.claim(&keys, &[0]);
            claim.native_started();
            let outcome = put_outcome(vec![false], &[0], &[result]);
            assert_eq!(outcome.success, [false]);
            assert!(!outcome.retired);
            claim.native_finished(outcome.retired);
            drop(claim);
            assert!(registry.claim(&keys, &[0]).entries.is_empty());
            let next_namespace = vec!["v2/a".to_owned()];
            assert_eq!(registry.claim(&next_namespace, &[0]).entries.len(), 1);
        }
    }

    #[test]
    fn a_native_call_without_a_result_does_not_release_its_claim() {
        let registry = InFlightPuts::default();
        let keys = vec!["a".to_owned()];
        {
            let mut claim = registry.claim(&keys, &[0]);
            claim.native_started();
            // A failed SDK call returns through `?` without a status vector.
        }
        assert!(registry.claim(&keys, &[0]).entries.is_empty());
    }

    #[test]
    fn concurrent_workers_claim_one_shared_key_without_serializing_other_keys() {
        let registry = Arc::new(InFlightPuts::default());
        let barrier = Arc::new(Barrier::new(4));
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let registry = Arc::clone(&registry);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let keys = vec!["shared".to_owned(), format!("private-{worker}")];
                    barrier.wait();
                    let claim = registry.claim(&keys, &[0, 1]);
                    let shared = claim.entries.iter().any(|(index, _)| *index == 0);
                    assert!(claim.entries.iter().any(|(index, _)| *index == 1));
                    barrier.wait(); // Every claim stays live until all workers saw it.
                    shared
                })
            })
            .collect();
        let shared_writers = workers
            .into_iter()
            .map(|worker| usize::from(worker.join().unwrap()))
            .sum::<usize>();
        assert_eq!(shared_writers, 1);
        assert!(registry.keys.lock().unwrap().is_empty());
    }

    #[test]
    fn existing_objects_are_not_rewritten_and_rejections_retire() {
        // `existing` is stored; `saved` writes; `pressure` meets a full
        // store, a failure the scheduler offers again.
        let outcome = put_outcome(vec![true, true, false], &[1, 2], &[0, NO_AVAILABLE_HANDLE]);
        assert_eq!(outcome.success, [true, true, false]);
        assert!(outcome.retired);
        assert_eq!(outcome.object_failures, [false, false, false]);
        // Another writer's accepted create keeps the SDK's best-effort ACK.
        let outcome = put_outcome(vec![false], &[0], &[OBJECT_ALREADY_EXISTS]);
        assert_eq!(outcome.success, [true]);
        assert!(outcome.retired);
    }

    #[test]
    fn a_failed_put_does_not_retire() {
        for error in [-1, -800] {
            let outcome = put_outcome(vec![true, true, false], &[1, 2], &[0, error]);
            assert_eq!(outcome.success, [true, true, false]);
            assert!(!outcome.retired);
        }
    }

    #[test]
    fn zero_and_duplicate_results_keep_the_existing_sdk_ack_semantics() {
        for result in [0, OBJECT_ALREADY_EXISTS] {
            let outcome = put_outcome(vec![false], &[0], &[result]);
            assert_eq!(outcome.success, [true]);
            assert!(outcome.retired);
            assert_eq!(outcome.object_failures, [false]);
        }
    }

    #[test]
    fn a_put_ack_does_not_retire_another_keys_unknown_io() {
        for unknown in [-800, RPC_FAIL, RPC_TIMEOUT, -1] {
            let outcome = put_outcome(vec![false, false], &[0, 1], &[0, unknown]);
            assert_eq!(outcome.success, [true, false]);
            assert!(!outcome.retired);
            assert_eq!(outcome.object_failures, [false, false]);
        }
    }

    #[test]
    fn a_put_ack_does_not_override_a_later_get_miss_or_short_read() {
        for status in [0, OBJECT_ALREADY_EXISTS] {
            let ack = put_outcome(vec![false], &[0], &[status]);
            assert_eq!(ack.success, [true]);
            assert!(ack.retired);
            for result in [OBJECT_NOT_FOUND, 4] {
                let read = get_outcome(&[result], &[8]);
                assert_eq!(read.success, [false]);
                assert!(read.retired);
                assert_eq!(read.object_failures, [true]);
            }
        }
    }

    #[test]
    fn an_owner_rejection_retires_default_host_sources_without_publication() {
        let outcome = put_outcome(vec![true, true], &[0, 1], &[0, ILLEGAL_CLIENT]);
        assert_eq!(outcome.success, [true, false]);
        assert!(outcome.retired);
        assert_eq!(outcome.object_failures, [false, false]);
    }

    #[test]
    fn an_owner_rejection_does_not_hide_another_keys_unknown_transfer() {
        for unknown in [-800, RPC_FAIL, RPC_TIMEOUT, -1] {
            for results in [[ILLEGAL_CLIENT, unknown], [unknown, ILLEGAL_CLIENT]] {
                let registry = InFlightPuts::default();
                let keys = vec!["a".to_owned(), "b".to_owned()];
                let mut claim = registry.claim(&keys, &[0, 1]);
                claim.native_started();
                let outcome = put_outcome(vec![false, false], &[0, 1], &results);
                assert_eq!(outcome.success, [false, false]);
                assert!(!outcome.retired);
                assert_eq!(outcome.object_failures, [false, false]);
                claim.native_finished(outcome.retired);
                drop(claim);
                assert!(registry.claim(&keys, &[0, 1]).entries.is_empty());
            }
        }
    }

    #[test]
    fn default_host_statuses_keep_sdk_acks_separate_from_source_retirement() {
        let outcome = put_outcome(
            vec![false; 4],
            &[0, 1, 2, 3],
            &[
                ILLEGAL_CLIENT,
                0,
                OBJECT_ALREADY_EXISTS,
                NO_AVAILABLE_HANDLE,
            ],
        );
        assert_eq!(outcome.success, [false, true, true, false]);
        assert!(outcome.retired);
        assert_eq!(outcome.object_failures, [false; 4]);
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
