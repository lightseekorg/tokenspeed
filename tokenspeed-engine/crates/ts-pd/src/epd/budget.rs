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

//! Whole-request tensor-payload reservations for reconstructed embeddings on one rank.
//! Allocator rounding, fragmentation and model temporaries require separate headroom.
//! A reservation outlives receiving and returns only after data-plane retirement.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crate::control::EmbeddingItem;

pub(super) struct Budget {
    limit: u64,
    free: Arc<AtomicU64>,
}

pub(super) struct Reservation {
    bytes: u64,
    free: Arc<AtomicU64>,
}

impl Budget {
    pub(super) fn new(limit: u64) -> Self {
        Self { limit, free: Arc::new(AtomicU64::new(limit)) }
    }

    pub(super) fn available(&self) -> u64 {
        self.free.load(Ordering::Relaxed)
    }

    pub(super) fn reserve(&self, bytes: u64) -> Option<Reservation> {
        self.free.fetch_update(Ordering::Relaxed, Ordering::Relaxed,
            |free| free.checked_sub(bytes)).ok()?;
        Some(Reservation { bytes, free: self.free.clone() })
    }

    pub(super) fn request_bytes(
        &self, items: &[EmbeddingItem], hidden: i64, element_size: i64, deepstack: i64,
    ) -> Result<u64, String> {
        let bytes = (|| {
            let width = u64::try_from(hidden).ok().filter(|&n| n > 0)?;
            let element = u64::try_from(element_size).ok().filter(|&n| n > 0)?;
            let levels = u64::try_from(deepstack).ok()?.checked_add(1)?;
            let rows = items.iter().try_fold(0u64, |total, item| {
                total.checked_add(u64::try_from(item.rows).ok()?)
            })?;
            rows.checked_mul(width)?.checked_mul(element)?.checked_mul(levels)
                .filter(|&n| n <= i64::MAX as u64)
        })().ok_or_else(|| "EPD embedding shape is invalid or its byte count overflows".to_owned())?;
        if bytes > self.limit {
            return Err(format!(
                "EPD request requires {bytes} bytes of reconstructed embeddings per rank, \
                 exceeding the {}-byte TOKENSPEED_EPD_EMBEDDING_BUDGET_MB budget",
                self.limit
            ));
        }
        Ok(bytes)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.free.fetch_add(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_atomic_and_return_on_drop() {
        let budget = Budget::new(100);
        let first = budget.reserve(70).unwrap();
        assert!(budget.reserve(31).is_none());
        assert_eq!(budget.available(), 30);
        let second = budget.reserve(30).unwrap();
        drop(first);
        assert_eq!(budget.available(), 70);
        drop(second);
        assert_eq!(budget.available(), 100);
    }

    #[test]
    fn complete_rows_and_deepstack_are_charged_without_tp_division() {
        let budget = Budget::new(512);
        let mut item = EmbeddingItem {
            index: 0, host: String::new(), port: 0, room: 1, rows: 8,
        };
        assert_eq!(budget.request_bytes(&[item.clone()], 8, 2, 3), Ok(512));
        item.rows = 9;
        assert!(budget.request_bytes(&[item.clone()], 8, 2, 3).is_err());
        item.rows = i64::MAX;
        assert!(budget.request_bytes(&[item.clone()], i64::MAX, 8, 3).is_err());
        item.rows = -1;
        assert!(budget.request_bytes(&[item], 8, 2, 0).is_err());
    }
}
