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

//! The byte stores packed Host cache blocks go to, and the in-memory one.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use crate::layout::Range;

/// The Host pool's bytes, as the stores read and write them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostBuffer {
    /// The pool's address.
    pub base: usize,
    /// Its size.
    pub len: usize,
}

impl HostBuffer {
    /// The address of `range`, which must lie inside the pool.
    pub fn address(&self, range: Range) -> usize {
        assert!(
            range
                .offset
                .checked_add(range.size)
                .is_some_and(|end| end <= self.len),
            "{range:?} is outside the {}-byte Host pool",
            self.len
        );
        self.base + range.offset
    }
}

/// What one batch did, per key: whether it succeeded, whether the store
/// confirmed the object missing or invalid, and whether every native access
/// to the Host bytes has ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoResult {
    /// Per key: a full read, or a published (or already present) object.
    pub success: Vec<bool>,
    /// Native retirement: nothing still touches the batch's Host bytes.
    pub retired: bool,
    /// Per key: the object is missing or invalid; never a transport or
    /// policy failure.
    pub object_failures: Vec<bool>,
}

/// A call that failed as a whole.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct StoreError(pub String);

/// A byte store for packed Host cache blocks. The executor's workers call
/// it from several threads at once.
pub trait Store: Send + Sync {
    /// Whether a failed call proves its Host access ended. Valid results
    /// carry their own `retired`.
    fn failures_retire(&self) -> bool;

    /// Per key: the object exists.
    fn batch_exists(&self, keys: &[String]) -> Result<Vec<bool>, StoreError>;

    /// Read objects into Host ranges.
    fn batch_get_into(
        &self,
        keys: &[String],
        host: HostBuffer,
        ranges: &[Range],
    ) -> Result<IoResult, StoreError>;

    /// Write Host ranges as objects; an object already present succeeds.
    fn batch_put_from(
        &self,
        keys: &[String],
        host: HostBuffer,
        ranges: &[Range],
    ) -> Result<IoResult, StoreError>;

    /// Remove every object whose key starts with `prefix`: true when they
    /// are gone. False must not be followed by an irreversible cache clear.
    fn remove_by_prefix(&self, prefix: &str) -> bool;

    /// Release the client's resources once nothing touches the Host pool;
    /// the objects stay in the store.
    fn close(&self);
}

impl<S: Store + ?Sized> Store for Arc<S> {
    fn failures_retire(&self) -> bool {
        (**self).failures_retire()
    }

    fn batch_exists(&self, keys: &[String]) -> Result<Vec<bool>, StoreError> {
        (**self).batch_exists(keys)
    }

    fn batch_get_into(
        &self,
        keys: &[String],
        host: HostBuffer,
        ranges: &[Range],
    ) -> Result<IoResult, StoreError> {
        (**self).batch_get_into(keys, host, ranges)
    }

    fn batch_put_from(
        &self,
        keys: &[String],
        host: HostBuffer,
        ranges: &[Range],
    ) -> Result<IoResult, StoreError> {
        (**self).batch_put_from(keys, host, ranges)
    }

    fn remove_by_prefix(&self, prefix: &str) -> bool {
        (**self).remove_by_prefix(prefix)
    }

    fn close(&self) {
        (**self).close();
    }
}

/// An in-process store: for tests, and as a reference with no Mooncake.
#[derive(Debug, Default)]
pub struct MemoryStore {
    objects: Mutex<HashMap<String, Vec<u8>>>,
}

/// The in-memory store every `memory` executor of the process shares: like
/// a remote store, it keeps objects after their writer closes.
pub fn process_memory_store() -> Arc<MemoryStore> {
    static STORE: OnceLock<Arc<MemoryStore>> = OnceLock::new();
    Arc::clone(STORE.get_or_init(Arc::default))
}

impl MemoryStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// The keys stored, sorted.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .objects
            .lock()
            .expect("memory store lock")
            .keys()
            .cloned()
            .collect();
        keys.sort();
        keys
    }

    /// Drop the objects whose keys `forget` selects.
    pub fn forget(&self, forget: impl Fn(&str) -> bool) {
        self.objects
            .lock()
            .expect("memory store lock")
            .retain(|key, _| !forget(key));
    }
}

impl Store for MemoryStore {
    fn failures_retire(&self) -> bool {
        true
    }

    fn batch_exists(&self, keys: &[String]) -> Result<Vec<bool>, StoreError> {
        let objects = self.objects.lock().expect("memory store lock");
        Ok(keys.iter().map(|key| objects.contains_key(key)).collect())
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
        let objects = self.objects.lock().expect("memory store lock");
        let success: Vec<bool> = keys
            .iter()
            .zip(ranges)
            .map(|(key, &range)| match objects.get(key) {
                Some(payload) if payload.len() == range.size => {
                    // SAFETY: the range lies inside the live Host pool (the
                    // address check), which the scheduler holds for this
                    // operation until its retirement, so nothing else writes
                    // it meanwhile.
                    #[allow(unsafe_code)]
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            payload.as_ptr(),
                            host.address(range) as *mut u8,
                            range.size,
                        );
                    }
                    true
                }
                _ => false,
            })
            .collect();
        let object_failures = success.iter().map(|ok| !ok).collect();
        Ok(IoResult {
            success,
            retired: true,
            object_failures,
        })
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
        let mut objects = self.objects.lock().expect("memory store lock");
        for (key, &range) in keys.iter().zip(ranges) {
            if objects.contains_key(key) {
                continue;
            }
            let mut payload = vec![0_u8; range.size];
            // SAFETY: as in `batch_get_into`; the source blocks are pinned
            // until the operation retires.
            #[allow(unsafe_code)]
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host.address(range) as *const u8,
                    payload.as_mut_ptr(),
                    range.size,
                );
            }
            objects.insert(key.clone(), payload);
        }
        Ok(IoResult {
            success: vec![true; keys.len()],
            retired: true,
            object_failures: vec![false; keys.len()],
        })
    }

    fn remove_by_prefix(&self, prefix: &str) -> bool {
        self.forget(|key| key.starts_with(prefix));
        true
    }

    fn close(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_round_trip_through_the_host_pool() {
        let mut pool = vec![0_u8; 16];
        pool[0..4].copy_from_slice(b"abcd");
        let host = HostBuffer {
            base: pool.as_mut_ptr() as usize,
            len: pool.len(),
        };
        let store = MemoryStore::new();
        let keys = vec!["k1".to_owned(), "k2".to_owned()];
        let first = Range { offset: 0, size: 4 };
        let put = store.batch_put_from(&keys[..1], host, &[first]).unwrap();
        assert_eq!(put.success, [true]);
        assert_eq!(store.batch_exists(&keys).unwrap(), [true, false]);

        let target = Range { offset: 8, size: 4 };
        let short = Range {
            offset: 12,
            size: 3,
        };
        let got = store
            .batch_get_into(&["k1".to_owned(), "k1".to_owned()], host, &[target, short])
            .unwrap();
        // A length mismatch is an invalid object, not a full read.
        assert_eq!(got.success, [true, false]);
        assert_eq!(got.object_failures, [false, true]);
        assert!(got.retired);
        assert_eq!(&pool[8..12], b"abcd");

        assert!(store.remove_by_prefix("k"));
        assert!(store.keys().is_empty());
    }
}
