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

//! Fixed slots of a lifetime-registered buffer, leased one per transfer.
//!
//! A slot is reused only once nothing may still write it: a landed slot
//! returns when its owner says the rows were copied out, a failed one after
//! a quarantine longer than any write still in flight (under one lifetime
//! registration a late write would land silently in the next tenant's rows).

use std::{collections::VecDeque, time::Instant};

/// The slots of one buffer.
#[derive(Debug)]
pub struct SlotPool {
    slot_bytes: u64,
    free: Vec<usize>,
    /// Failed slots and when they may return, in that order.
    quarantine: VecDeque<(Instant, usize)>,
}

/// A transfer larger than a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Oversized {
    /// The bytes asked for.
    pub bytes: u64,
    /// A slot's bytes.
    pub slot_bytes: u64,
}

impl SlotPool {
    /// `slots` slots of `slot_bytes` each, all free.
    pub fn new(slots: usize, slot_bytes: u64) -> Self {
        Self {
            slot_bytes,
            // Popped from the back: the lowest slots go first.
            free: (0..slots).rev().collect(),
            quarantine: VecDeque::new(),
        }
    }

    /// A slot's bytes.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// A free slot for `bytes`, `None` while every slot is taken, or the
    /// refusal of a transfer no slot can hold.
    pub fn lease(&mut self, bytes: u64, now: Instant) -> Result<Option<usize>, Oversized> {
        if bytes > self.slot_bytes {
            return Err(Oversized {
                bytes,
                slot_bytes: self.slot_bytes,
            });
        }
        while self
            .quarantine
            .front()
            .is_some_and(|&(until, _)| until <= now)
        {
            let (_, slot) = self.quarantine.pop_front().expect("checked");
            self.free.push(slot);
        }
        Ok(self.free.pop())
    }

    /// `slot`'s rows were copied out: it takes a new transfer.
    pub fn release(&mut self, slot: usize) {
        self.free.push(slot);
    }

    /// `slot`'s transfer failed: it returns at `until`, once no write to it
    /// can still be in flight.
    pub fn quarantine(&mut self, slot: usize, until: Instant) {
        self.quarantine.push_back((until, slot));
    }

    /// Free slots now.
    pub fn free(&self) -> usize {
        self.free.len()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn slots_are_leased_released_and_quarantined() {
        let now = Instant::now();
        let mut pool = SlotPool::new(2, 100);
        assert_eq!(
            pool.lease(101, now),
            Err(Oversized {
                bytes: 101,
                slot_bytes: 100
            })
        );
        assert_eq!(pool.lease(100, now), Ok(Some(0)));
        assert_eq!(pool.lease(1, now), Ok(Some(1)));
        // Exhausted: the transfer waits for a slot.
        assert_eq!(pool.lease(1, now), Ok(None));
        pool.release(1);
        assert_eq!(pool.lease(1, now), Ok(Some(1)));
        // A failed slot comes back only after its quarantine.
        pool.quarantine(0, now + Duration::from_secs(10));
        assert_eq!(pool.lease(1, now + Duration::from_secs(9)), Ok(None));
        assert_eq!(pool.lease(1, now + Duration::from_secs(10)), Ok(Some(0)));
    }
}
