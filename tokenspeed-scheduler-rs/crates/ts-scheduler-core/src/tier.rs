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

//! Tier-transfer operations and the transfer manager.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/tier/transfer.{h,cpp}` and
//! `transfer_manager.{h,cpp}`. The manager owns the mechanics and asynchronous
//! lifetime of Device<->Host cache transfers; scheduling policy stays in the
//! scheduler.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::cache_block_ref::{CacheBlockLocation, CacheBlockRef};
use crate::cache_coordinator::CacheCoordinator;
use crate::cache_types::{BlockTransfer, CacheKey};

/// One device<->host page pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheTransfer {
    pub group_id: u32,
    pub source_page: i32,
    pub destination_page: i32,
}

/// DEVICE -> HOST store operation.
#[derive(Debug, Clone, Default)]
pub struct WriteBackOperation {
    pub op_id: u32,
    pub transfers: Vec<CacheTransfer>,
}

/// HOST -> DEVICE load operation.
#[derive(Debug, Clone, Default)]
pub struct LoadBackOperation {
    pub op_id: u32,
    pub transfers: Vec<CacheTransfer>,
}

/// Batched write-back view exposed to Python (deduplicates transfers per op).
#[derive(Debug, Clone, Default)]
pub struct WriteBackBatch {
    pub op_ids: Vec<u32>,
    pub group_ids: Vec<Vec<u32>>,
    pub src_pages: Vec<Vec<i32>>,
    pub dst_pages: Vec<Vec<i32>>,
}

impl WriteBackBatch {
    pub fn new(ops: &[WriteBackOperation]) -> Self {
        let mut batch = WriteBackBatch::default();
        let mut seen = HashSet::new();
        for op in ops {
            let mut operation_groups = Vec::new();
            let mut operation_sources = Vec::new();
            let mut operation_destinations = Vec::new();
            for transfer in &op.transfers {
                if seen.insert(*transfer) {
                    operation_groups.push(transfer.group_id);
                    operation_sources.push(transfer.source_page);
                    operation_destinations.push(transfer.destination_page);
                }
            }
            batch.op_ids.push(op.op_id);
            batch.group_ids.push(operation_groups);
            batch.src_pages.push(operation_sources);
            batch.dst_pages.push(operation_destinations);
        }
        batch
    }
}

/// Batched load-back view exposed to Python (deduplicates transfers per op).
#[derive(Debug, Clone, Default)]
pub struct LoadBackBatch {
    pub op_ids: Vec<u32>,
    pub group_ids: Vec<Vec<u32>>,
    pub src_pages: Vec<Vec<i32>>,
    pub dst_pages: Vec<Vec<i32>>,
}

impl LoadBackBatch {
    pub fn new(ops: &[LoadBackOperation]) -> Self {
        let mut batch = LoadBackBatch::default();
        let mut seen = HashSet::new();
        for op in ops {
            let mut operation_groups = Vec::new();
            let mut operation_sources = Vec::new();
            let mut operation_destinations = Vec::new();
            for transfer in &op.transfers {
                if seen.insert(*transfer) {
                    operation_groups.push(transfer.group_id);
                    operation_sources.push(transfer.source_page);
                    operation_destinations.push(transfer.destination_page);
                }
            }
            batch.op_ids.push(op.op_id);
            batch.group_ids.push(operation_groups);
            batch.src_pages.push(operation_sources);
            batch.dst_pages.push(operation_destinations);
        }
        batch
    }
}

/// A cache operation inside an execution plan.
#[derive(Debug, Clone)]
pub enum CacheOperation {
    LoadBack(LoadBackBatch),
    WriteBack(WriteBackBatch),
}

/// One in-flight device->host store: both tiers are pinned until the runtime
/// acknowledges the copy.
#[derive(Debug)]
struct StoreTicket {
    key: CacheKey,
    device_block_ref: CacheBlockRef,
    host_block_ref: CacheBlockRef,
}

/// Owns the mechanics and asynchronous lifetime of transfers between Device and
/// Host cache tiers. Holds the coordinator through `Rc<RefCell<..>>` because it
/// mutates the coordinator while the scheduler owns it too.
pub struct TierTransferManager {
    coordinator: Rc<RefCell<CacheCoordinator>>,
    write_backs: HashMap<u32, Vec<StoreTicket>>,
    store_keys: HashSet<CacheKey>,
    load_backs: HashMap<u32, Vec<BlockTransfer>>,
    next_op_id: u32,
}

impl TierTransferManager {
    pub fn new(coordinator: Rc<RefCell<CacheCoordinator>>) -> Self {
        Self {
            coordinator,
            write_backs: HashMap::new(),
            store_keys: HashSet::new(),
            load_backs: HashMap::new(),
            next_op_id: 0,
        }
    }

    pub fn has_stores_in_flight(&self) -> bool {
        !self.write_backs.is_empty()
    }

    pub fn has_load_backs_in_flight(&self) -> bool {
        !self.load_backs.is_empty()
    }

    pub fn has_any_in_flight(&self) -> bool {
        !self.write_backs.is_empty() || !self.load_backs.is_empty()
    }

    /// Locations that become reclaimable once in-flight Store ACKs arrive.
    pub fn device_locations_released_on_store_ack(&self) -> Vec<(u32, CacheBlockLocation)> {
        let mut locations = Vec::new();
        for stores in self.write_backs.values() {
            for ticket in stores {
                locations.push((
                    ticket.key.group_id,
                    ticket
                        .device_block_ref
                        .location()
                        .expect("store ticket pins a device block"),
                ));
            }
        }
        locations
    }

    /// Drain the coordinator's pending stores into one write-back op.
    pub fn start_pending_stores(&mut self) -> Option<WriteBackOperation> {
        let mut transfers = Vec::new();
        let mut tickets = Vec::new();
        let mut batch_keys = HashSet::new();
        let mut coordinator = self.coordinator.borrow_mut();
        for candidate in coordinator.take_pending_stores() {
            if coordinator.contains_host_cached_block(&candidate.key)
                || self.store_keys.contains(&candidate.key)
                || !batch_keys.insert(candidate.key.clone())
            {
                continue;
            }
            let device_block_ref = coordinator.acquire_device_cached_block(&candidate.key);
            if device_block_ref.is_null() {
                continue;
            }
            let host_block_ref = coordinator.acquire_host_block(candidate.key.group_id);
            if host_block_ref.is_null() {
                continue;
            }
            let device_location = device_block_ref
                .location()
                .expect("device block has a location");
            let host_location = host_block_ref
                .location()
                .expect("host block has a location");
            let manager = coordinator.allocator(candidate.key.group_id as usize);
            transfers.push(CacheTransfer {
                group_id: candidate.key.group_id,
                source_page: manager.resolve_cache_block_id(device_location),
                destination_page: manager.resolve_cache_block_id(host_location),
            });
            tickets.push(StoreTicket {
                key: candidate.key,
                device_block_ref,
                host_block_ref,
            });
        }
        drop(coordinator);
        if transfers.is_empty() {
            return None;
        }
        let op_id = self.next_op_id();
        for ticket in &tickets {
            self.store_keys.insert(ticket.key.clone());
        }
        assert!(
            self.write_backs.insert(op_id, tickets).is_none(),
            "duplicate store op id"
        );
        Some(WriteBackOperation { op_id, transfers })
    }

    /// Start a prefix load from pinned host blocks.
    pub fn start_prefix_load(&mut self, block_transfers: Vec<BlockTransfer>) -> LoadBackOperation {
        assert!(
            !block_transfers.is_empty(),
            "prefix load requires at least one block transfer"
        );
        let coordinator = self.coordinator.borrow();
        for pair in &block_transfers {
            assert!(
                coordinator.is_host_cached_block(
                    pair.source
                        .location()
                        .expect("pinned source has a location")
                ),
                "pinned Host block lost its cache entry before load emission"
            );
        }
        drop(coordinator);
        self.start_load_back(block_transfers)
    }

    fn start_load_back(&mut self, block_transfers: Vec<BlockTransfer>) -> LoadBackOperation {
        let transfers = self.resolve_transfers(&block_transfers);
        let op_id = self.next_op_id();
        assert!(
            self.load_backs.insert(op_id, block_transfers).is_none(),
            "duplicate loadback op id"
        );
        LoadBackOperation { op_id, transfers }
    }

    pub fn complete_write_back(&mut self, op_id: u32) {
        // The runtime emits this ACK only after the asynchronous copy completes.
        let Some(stores) = self.write_backs.remove(&op_id) else {
            return;
        };
        for ticket in &stores {
            self.store_keys.remove(&ticket.key);
        }
        let mut coordinator = self.coordinator.borrow_mut();
        for mut ticket in stores {
            coordinator.cache_host_block(&mut ticket.host_block_ref, &ticket.key);
        }
    }

    pub fn complete_load_back(&mut self, op_id: u32) {
        self.load_backs.remove(&op_id);
    }

    fn next_op_id(&mut self) -> u32 {
        let id = self.next_op_id;
        self.next_op_id += 1;
        id
    }

    fn resolve_transfers(&self, block_transfers: &[BlockTransfer]) -> Vec<CacheTransfer> {
        let coordinator = self.coordinator.borrow();
        let mut transfers = Vec::with_capacity(block_transfers.len());
        for block_transfer in block_transfers {
            assert!(
                !block_transfer.source.is_null() && !block_transfer.destination.is_null(),
                "cache transfer requires pinned source and destination blocks"
            );
            let manager = coordinator.allocator(block_transfer.group_id as usize);
            transfers.push(CacheTransfer {
                group_id: block_transfer.group_id,
                source_page: manager.resolve_cache_block_id(
                    block_transfer
                        .source
                        .location()
                        .expect("source has a location"),
                ),
                destination_page: manager.resolve_cache_block_id(
                    block_transfer
                        .destination
                        .location()
                        .expect("destination has a location"),
                ),
            });
        }
        transfers
    }
}
