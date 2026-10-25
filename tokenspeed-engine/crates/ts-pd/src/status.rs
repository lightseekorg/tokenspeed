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

//! A room's transfer state, as the PD and EPD peers' status messages carry
//! it.

use std::collections::HashMap;

/// A transfer's state, in the order it advances. The values are the wire
/// encoding of the P/D status messages and of the replica's MIN agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransferPoll {
    /// The transfer failed; sticky.
    Failed = 0,
    /// The peers are connecting.
    Bootstrapping = 1,
    /// The peers are connected.
    Bootstrapped = 2,
    /// Decode: the manifest went out; waiting for the KV.
    WaitingForInput = 3,
    /// Unused: a transfer is not observed mid-flight.
    Transferring = 4,
    /// The KV arrived (or left, on Prefill).
    Success = 5,
}

impl TransferPoll {
    /// The state a wire value names.
    pub fn from_wire(value: i64) -> Option<Self> {
        Some(match value {
            0 => Self::Failed,
            1 => Self::Bootstrapping,
            2 => Self::Bootstrapped,
            3 => Self::WaitingForInput,
            4 => Self::Transferring,
            5 => Self::Success,
            _ => return None,
        })
    }
}

/// Room states and failure reasons.
#[derive(Debug, Default)]
pub struct RoomTable {
    states: HashMap<u64, TransferPoll>,
    failures: HashMap<u64, String>,
}

impl RoomTable {
    /// A room's state, if the room is known.
    pub fn get(&self, room: u64) -> Option<TransferPoll> {
        self.states.get(&room).copied()
    }

    /// Advance a room: states only move forward, and `Failed` is sticky, so
    /// a late success cannot resurrect a broken transfer.
    pub fn update(&mut self, room: u64, status: TransferPoll) {
        let entry = self.states.entry(room).or_insert(status);
        *entry = if *entry == TransferPoll::Failed || status == TransferPoll::Failed {
            TransferPoll::Failed
        } else {
            (*entry).max(status)
        };
    }

    /// Record why a room failed; a later reason replaces an earlier one.
    pub fn record_failure(&mut self, room: u64, reason: impl Into<String>) {
        self.failures.insert(room, reason.into());
    }

    /// Whether a failure reason is on record.
    pub fn has_failure(&self, room: u64) -> bool {
        self.failures.contains_key(&room)
    }

    /// The failure reason on record.
    pub fn failure(&self, room: u64) -> Option<&str> {
        self.failures.get(&room).map(String::as_str)
    }

    /// Forget a room.
    pub fn remove(&mut self, room: u64) {
        self.states.remove(&room);
        self.failures.remove(&room);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn states_advance_and_failure_sticks() {
        let mut table = RoomTable::default();
        table.update(1, TransferPoll::Bootstrapped);
        table.update(1, TransferPoll::Bootstrapping);
        assert_eq!(table.get(1), Some(TransferPoll::Bootstrapped));
        table.update(1, TransferPoll::Failed);
        table.update(1, TransferPoll::Success);
        assert_eq!(table.get(1), Some(TransferPoll::Failed));
        assert_eq!(table.get(2), None);
        assert_eq!(TransferPoll::from_wire(5), Some(TransferPoll::Success));
        assert_eq!(TransferPoll::from_wire(9), None);
    }
}
