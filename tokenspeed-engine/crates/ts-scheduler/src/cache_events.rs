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

//! Bounded metadata delivery from native scheduler ownership callbacks.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use crate::CacheObservation;

/// Maximum request identity copied or formatted by an observation consumer.
pub const MAX_CACHE_EVENT_REQUEST_BYTES: usize = 256;

/// Lost observations invalidate evidence; they never change scheduling.
#[derive(Clone)]
pub struct CacheObservationLosses(Arc<AtomicU64>);

impl CacheObservationLosses {
    /// Drain the number of rejected or panicking deliveries since the last poll.
    pub fn take(&self) -> u64 {
        self.0.swap(0, Ordering::Relaxed)
    }
}

/// An owned observer. The native scheduler retains it through slot teardown.
///
/// The handler must be nonblocking and cannot enter Python or issue device work.
/// The request identity is borrowed only for this call. Delivery failure is
/// counted rather than unwinding through a native resource destructor.
pub struct CacheEventObserver {
    handler: Box<dyn Fn(&CacheObservation, &str) + Send + Sync>,
    losses: CacheObservationLosses,
}

impl CacheEventObserver {
    /// Build an observer and its explicit delivery-loss counter.
    pub fn new(
        handler: impl Fn(&CacheObservation, &str) + Send + Sync + 'static,
    ) -> (Self, CacheObservationLosses) {
        let losses = CacheObservationLosses(Arc::new(AtomicU64::new(0)));
        (
            Self {
                handler: Box::new(handler),
                losses: losses.clone(),
            },
            losses,
        )
    }

    pub(crate) fn record(&self, value: CacheObservation, request_id: &[u8]) {
        if request_id.is_empty() || request_id.len() > MAX_CACHE_EVENT_REQUEST_BYTES {
            self.losses.0.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let Ok(request_id) = std::str::from_utf8(request_id) else {
            self.losses.0.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if catch_unwind(AssertUnwindSafe(|| (self.handler)(&value, request_id))).is_err() {
            self.losses.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CacheObservationKind;
    use std::sync::atomic::AtomicUsize;

    fn observation() -> CacheObservation {
        CacheObservation {
            kind: CacheObservationKind::SlotReleased,
            slot: 1,
            slot_generation: 2,
            admission_id: 3,
            slot_previous_admission_id: 0,
            op_id: 0,
            op_generation: 0,
            previous_slot: -1,
            previous_slot_generation: 0,
            restore_sequence: 0,
            committed_tokens: 0,
            computed_tokens: 0,
            prompt_tokens: 0,
        }
    }

    #[test]
    fn invalid_or_overlong_identity_is_loss_not_truncated_evidence() {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let (observer, losses) = CacheEventObserver::new(move |_, _| {
            seen.fetch_add(1, Ordering::Relaxed);
        });
        observer.record(observation(), &[255]);
        observer.record(observation(), b"");
        observer.record(
            observation(),
            &vec![b'a'; MAX_CACHE_EVENT_REQUEST_BYTES + 1],
        );
        observer.record(observation(), b"valid");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(losses.take(), 3);
        assert_eq!(losses.take(), 0);
    }

    #[test]
    fn delivery_panic_cannot_escape_a_native_release_callback() {
        let (observer, losses) = CacheEventObserver::new(|_, _| panic!("failed diagnostic sink"));
        observer.record(observation(), b"request");
        assert_eq!(losses.take(), 1);
    }
}
