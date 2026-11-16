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

//! Ownership-based counters for batches waiting on, queued to, or processed by
//! the publisher. Encoded replay-buffer retention is outside these counters.
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

/// A publisher's current backlog and cumulative queue cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct KvEventStats {
    /// Batches owned by a blocked caller, the channel, or the worker.
    pub pending_batches: usize,
    /// Logical snapshot bytes charged through completion, excluding replay buffers.
    pub pending_bytes: usize,
    /// Maximum outstanding batch count, including a blocked caller and worker.
    pub peak_pending_batches: usize,
    /// Maximum outstanding logical payload bytes.
    pub peak_pending_bytes: usize,
    /// Batches the worker finished encoding/sending, including send errors.
    pub completed_batches: u64,
    /// Cumulative blocking-send call time, including successful fast enqueues.
    pub enqueue_ns: u64,
}

#[derive(Default)]
struct Counters {
    batches: AtomicUsize,
    bytes: AtomicUsize,
    peak_batches: AtomicUsize,
    peak_bytes: AtomicUsize,
    completed: AtomicU64,
    enqueue_ns: AtomicU64,
}

/// A read-only observation handle that does not keep the publisher alive.
#[derive(Clone, Default)]
pub struct KvEventProgress(Arc<Counters>);
impl KvEventProgress {
    /// Read diagnostic counters; concurrent fields are sampled independently.
    pub fn snapshot(&self) -> KvEventStats {
        KvEventStats {
            pending_batches: self.0.batches.load(Ordering::Acquire),
            pending_bytes: self.0.bytes.load(Ordering::Relaxed),
            peak_pending_batches: self.0.peak_batches.load(Ordering::Relaxed),
            peak_pending_bytes: self.0.peak_bytes.load(Ordering::Relaxed),
            completed_batches: self.0.completed.load(Ordering::Relaxed),
            enqueue_ns: self.0.enqueue_ns.load(Ordering::Relaxed),
        }
    }
    pub(super) fn enqueued(&self, duration: Duration) {
        self.0.enqueue_ns.fetch_add(
            duration.as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }
    pub(super) fn track<T>(&self, payload: T, bytes: usize) -> PendingBatch<T> {
        let batches = self.0.batches.fetch_add(1, Ordering::Relaxed) + 1;
        let total_bytes = self.0.bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.0.peak_batches.fetch_max(batches, Ordering::Relaxed);
        self.0.peak_bytes.fetch_max(total_bytes, Ordering::Relaxed);
        PendingBatch {
            payload,
            lease: BatchLease {
                counters: self.0.clone(),
                bytes,
            },
        }
    }
}

pub(super) struct PendingBatch<T> {
    pub payload: T,
    pub lease: BatchLease,
}
pub(super) struct BatchLease {
    counters: Arc<Counters>,
    bytes: usize,
}
impl BatchLease {
    pub fn complete(self) {
        self.counters.completed.fetch_add(1, Ordering::Relaxed);
    }
}
impl Drop for BatchLease {
    fn drop(&mut self) {
        self.counters.bytes.fetch_sub(self.bytes, Ordering::Relaxed);
        self.counters.batches.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use crate::kv_events::stats::{KvEventProgress, PendingBatch};
    #[test]
    fn dropped_and_completed_batches_release_their_own_counts() {
        let progress = KvEventProgress::default();
        let a = progress.track(vec![1u8; 5], 5);
        let b = progress.track(vec![2u8; 7], 7);
        assert_eq!(progress.snapshot().pending_batches, 2);
        assert_eq!(progress.snapshot().pending_bytes, 12);
        drop(a);
        assert_eq!(progress.snapshot().pending_bytes, 7);
        let PendingBatch { payload, lease } = b;
        assert_eq!(payload, vec![2; 7]);
        lease.complete();
        let end = progress.snapshot();
        assert_eq!(end.pending_batches, 0);
        assert_eq!(end.pending_bytes, 0);
        assert_eq!(end.completed_batches, 1);
        assert_eq!(end.peak_pending_bytes, 12);
    }
    #[test]
    fn a_rejected_send_releases_its_backlog() {
        let progress = KvEventProgress::default();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let error = tx
            .blocking_send(progress.track(vec![1u8; 3], 3))
            .unwrap_err();
        drop(error);
        assert_eq!(progress.snapshot().pending_batches, 0);
        assert_eq!(progress.snapshot().completed_batches, 0);
    }
}
