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

//! An encode rank's sender and prefill ranks' receivers in one process,
//! rendezvousing through a bootstrap server, the rows crossing over RDMA.
//!
//! Needs an RDMA device and `TS_MOONCAKE_LIBRARY` naming
//! `libtokenspeed_transfer_engine.so`; without the variable the tests return
//! at once. The send slot and the receive pools are host buffers.

use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use super::{
    receiver::{PoolRegion, Receiver, ReceiverConfig},
    sender::{Sender, SenderConfig},
    shard_rows,
};
use crate::{
    bootstrap::{BootstrapServer, Registry},
    control::{EmbeddingChunk, EmbeddingItem, EmbeddingReceiver, EmbeddingSender, LandedRequest},
    PeerConfig, Transport,
};

const HIDDEN: i64 = 8;
const ELEMENT: i64 = 4;
const SLOT_BYTES: usize = 4096;

fn library() -> Option<PathBuf> {
    std::env::var_os("TS_MOONCAKE_LIBRARY").map(PathBuf::from)
}

fn peer(library: PathBuf, regions: Vec<(usize, usize)>, engine_rank: i64) -> PeerConfig {
    PeerConfig {
        library,
        local_ip: "127.0.0.1".to_owned(),
        protocol: "rdma".to_owned(),
        ib_device: String::new(),
        regions,
        engine_rank,
        metrics: None,
    }
}

struct Node {
    server: BootstrapServer,
    sender: Sender,
    /// The send slot the rows are read from.
    ring: Vec<u8>,
}

fn encode_node(library: PathBuf, bootstrap_timeout: Duration) -> Node {
    let server = BootstrapServer::start(0, Registry::Addresses).unwrap();
    let ring: Vec<u8> = (0..SLOT_BYTES).map(|i| (i % 251) as u8 + 1).collect();
    let sender = Sender::start(SenderConfig {
        peer: peer(library, vec![(ring.as_ptr() as usize, ring.len())], 0),
        bootstrap_url: format!("http://127.0.0.1:{}/route", server.port()),
        tp_size: 1,
        bootstrap_timeout,
        transfer_workers: 2,
    })
    .unwrap();
    Node {
        server,
        sender,
        ring,
    }
}

/// A prefill rank's receiver over its own transfer engine, with one-slot
/// pools.
fn receiver(library: PathBuf, pool: &[u8], tp_size: i64, tp_rank: i64, shard: bool) -> Receiver {
    receiver_with_budget(library, pool, tp_size, tp_rank, shard, 1 << 20)
}

fn receiver_with_budget(
    library: PathBuf, pool: &[u8], tp_size: i64, tp_rank: i64, shard: bool,
    embedding_budget_bytes: u64,
) -> Receiver {
    let (incoming, _messages) = mpsc::channel();
    let transport: Arc<Transport> =
        Transport::start(&peer(library, Vec::new(), 10 + tp_rank), incoming).unwrap();
    Receiver::start(
        ReceiverConfig {
            tp_size,
            tp_rank,
            hidden: HIDDEN,
            num_deepstack: 0,
            dtype: "torch.float32".to_owned(),
            element_size: ELEMENT,
            shard,
            pool: Some(PoolRegion {
                base: pool.as_ptr() as u64,
                slot_bytes: pool.len() as u64,
                slots: 1,
            }),
            deepstack_pool: None,
            embedding_budget_bytes,
            timeout: Duration::from_secs(30),
        },
        transport,
    )
    .unwrap()
}

fn item(port: u16, room: u64, rows: i64) -> EmbeddingItem {
    EmbeddingItem {
        index: 0,
        host: "127.0.0.1".to_owned(),
        port,
        room,
        rows,
    }
}

fn chunk(ring: &[u8], room: u64, rows: i64) -> EmbeddingChunk {
    EmbeddingChunk {
        room,
        source: ring.as_ptr() as u64,
        rows,
        hidden: HIDDEN,
        dtype: "torch.float32".to_owned(),
        nbytes: (rows * HIDDEN * ELEMENT) as u64,
        deepstack_source: 0,
        deepstack_width: 0,
        deepstack_nbytes: 0,
    }
}

/// Poll `receivers` (each pre-allocates only when polled) until each one's
/// staged request settles; their codes and, for those that landed, where.
fn settle_all(receivers: &mut [Receiver]) -> Vec<(i64, Vec<LandedRequest>)> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut settled: Vec<Option<(i64, Vec<LandedRequest>)>> = vec![None; receivers.len()];
    while settled.iter().any(Option::is_none) {
        for (receiver, outcome) in receivers.iter_mut().zip(&mut settled) {
            if outcome.is_some() {
                continue;
            }
            receiver.admit_requests(&receiver.admission_plan());
            let capacity = receiver.lease_capacity();
            receiver.lease_items(capacity);
            let codes = receiver.poll();
            if codes[0] != 1 {
                *outcome = Some((codes[0], receiver.settle(&codes)));
            }
        }
        assert!(Instant::now() < deadline, "a receive never settled");
        thread::sleep(Duration::from_millis(5));
    }
    settled.into_iter().map(Option::unwrap).collect()
}

fn settle(receiver: &mut Receiver) -> (i64, Vec<LandedRequest>) {
    settle_all(std::slice::from_mut(receiver)).pop().unwrap()
}

fn concluded(sender: &mut Sender, room: u64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !sender.concluded().contains(&room) {
        assert!(Instant::now() < deadline, "room {room} never concluded");
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn an_item_reaches_its_prefill_rank() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let mut node = encode_node(library.clone(), Duration::from_secs(30));
    let pool = vec![0u8; SLOT_BYTES];
    let mut prefill = receiver(library, &pool, 1, 0, false);
    let rows = 5;
    // The receiver may ask before the encode request arrives, or after.
    prefill.stage("r1", &[item(node.server.port(), 42, rows)]);
    assert!(node.sender.begin(42));
    // Another request cannot take the room.
    assert!(!node.sender.begin(42));
    node.sender.send(chunk(&node.ring, 42, rows));

    let (code, landed) = settle(&mut prefill);
    assert_eq!(code, 2);
    let item = &landed[0].items[0];
    assert_eq!(
        (item.slot, item.row_start, item.row_count),
        (Some(0), 0, rows)
    );
    let bytes = (rows * HIDDEN * ELEMENT) as usize;
    assert_eq!(pool[..bytes], node.ring[..bytes]);
    assert!(pool[bytes..].iter().all(|&byte| byte == 0));
    concluded(&mut node.sender, 42);
    // The data plane copied the rows out: the slot takes the next item.
    assert!(prefill.awaiting_release());
    prefill.release(&[0]);
    prefill.retire_pending(&["r1".to_owned()]);
    prefill.retire_requests(&["r1".to_owned()]);
    assert!(!prefill.awaiting_release());
}

#[test]
fn sharded_prefill_ranks_each_receive_their_rows() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let mut node = encode_node(library.clone(), Duration::from_secs(30));
    let pools = [vec![0u8; SLOT_BYTES], vec![0u8; SLOT_BYTES]];
    // One encode rank serves both: it writes once both asked.
    let mut ranks: Vec<Receiver> = (0..2)
        .map(|rank| receiver(library.clone(), &pools[rank], 2, rank as i64, true))
        .collect();
    let rows = 7;
    assert!(node.sender.begin(9));
    node.sender.send(chunk(&node.ring, 9, rows));
    for rank in &mut ranks {
        rank.stage("r1", &[item(node.server.port(), 9, rows)]);
    }
    let row_bytes = (HIDDEN * ELEMENT) as usize;
    for (rank, (code, landed)) in settle_all(&mut ranks).into_iter().enumerate() {
        assert_eq!(code, 2);
        let (start, count) = shard_rows(rows, rank as i64, 2);
        let item = &landed[0].items[0];
        assert_eq!(
            (item.row_start, item.row_count, item.sharded),
            (start, count, true)
        );
        // The shard lands packed at the slot's start.
        let (start, count) = (start as usize * row_bytes, count as usize * row_bytes);
        assert_eq!(pools[rank][..count], node.ring[start..start + count]);
    }
    concluded(&mut node.sender, 9);
}

#[test]
fn a_failed_room_fails_its_prefill_ranks() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let mut node = encode_node(library.clone(), Duration::from_secs(30));
    let pool = vec![0u8; SLOT_BYTES];
    let mut prefill = receiver(library.clone(), &pool, 1, 0, false);
    assert!(node.sender.begin(5));
    prefill.stage("r1", &[item(node.server.port(), 5, 3)]);
    // Say the vision tower failed on it.
    thread::sleep(Duration::from_millis(200));
    node.sender.fail(5, "the vision tower failed");
    assert_eq!(settle(&mut prefill).0, 0);
    concluded(&mut node.sender, 5);
    // A prefill rank that asks only now learns it at once.
    let mut late = receiver(library, &pool, 1, 0, false);
    late.stage("r1", &[item(node.server.port(), 5, 3)]);
    assert_eq!(settle(&mut late).0, 0);
}

#[test]
fn an_item_whose_fanout_never_completes_fails() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let mut node = encode_node(library.clone(), Duration::from_secs(1));
    let pool = vec![0u8; SLOT_BYTES];
    // Rank 0 of a two-rank prefill node asks; rank 1 never does.
    let mut prefill = receiver(library, &pool, 2, 0, true);
    assert!(node.sender.begin(11));
    prefill.stage("r1", &[item(node.server.port(), 11, 4)]);
    node.sender.send(chunk(&node.ring, 11, 4));
    assert_eq!(settle(&mut prefill).0, 0);
    concluded(&mut node.sender, 11);
}

/// Exercise the real sender/receiver and socket handshake using a CPU copy
/// transport. Every request has more images than the one-slot receive pool.
#[test]
fn incremental_images_reuse_slots_across_requests_and_shards() {
    for tp in [1, 2, 3] {
        let native = crate::tests_support::NativeLibrary::new();
        let mut node = encode_node(native.path().to_owned(), Duration::from_secs(30));
        let pools: Vec<_> = (0..tp).map(|_| vec![0u8; SLOT_BYTES]).collect();
        let mut ranks: Vec<_> = (0..tp)
            .map(|rank| {
                receiver(
                    native.path().to_owned(),
                    &pools[rank],
                    tp as i64,
                    rank as i64,
                    true,
                )
            })
            .collect();
        // Two concurrent requests; the second includes empty TP shards.
        let rows = [7, 1];
        for (request, &rows) in rows.iter().enumerate() {
            let items: Vec<_> = (0..17)
                .map(|index| {
                    let room = 100 + (request * 17 + index) as u64;
                    assert!(node.sender.begin(room));
                    node.sender.send(chunk(&node.ring, room, rows));
                    let mut input = item(node.server.port(), room, rows);
                    input.index = index;
                    input
                })
                .collect();
            for rank in &mut ranks {
                rank.stage(&format!("images-{request}"), &items);
            }
        }
        let mut seen: Vec<Vec<(String, usize)>> = vec![Vec::new(); tp];
        // Delay each rank's copy completion differently, changing the slow
        // rank over time. Free-slot counts must not change logical item order.
        let mut pending: Vec<Vec<(usize, usize)>> = vec![Vec::new(); tp];
        let mut round = 0usize;
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            round += 1;
            for (rank_index, rank) in ranks.iter_mut().enumerate() {
                pending[rank_index].retain(|&(at, slot)| {
                    if at <= round {
                        rank.release(&[slot]);
                        false
                    } else {
                        true
                    }
                });
            }
            let local_admissions: Vec<_> = ranks.iter().map(EmbeddingReceiver::admission_plan).collect();
            let admissions: Vec<_> = (0..local_admissions[0].len())
                .map(|i| local_admissions.iter().map(|v| v[i]).min().unwrap()).collect();
            for rank in &mut ranks { rank.admit_requests(&admissions); }
            let capacity = ranks.iter_mut()
                .map(EmbeddingReceiver::lease_capacity).min().unwrap();
            for rank in &mut ranks {
                rank.lease_items(capacity);
            }
            let local: Vec<_> = ranks.iter_mut().map(EmbeddingReceiver::poll).collect();
            if local[0].is_empty() {
                if pending.iter().all(Vec::is_empty) {
                    break;
                }
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(2));
                continue;
            }
            assert!(
                Instant::now() < deadline,
                "pool did not recycle for oversized requests"
            );
            let codes: Vec<_> = (0..local[0].len())
                .map(|i| local.iter().map(|v| v[i]).min().unwrap())
                .collect();
            assert!(!codes.contains(&0), "receive failed");
            let local_prefixes: Vec<_> = ranks
                .iter()
                .map(EmbeddingReceiver::landed_prefixes)
                .collect();
            let prefixes: Vec<_> = (0..codes.len())
                .map(|i| local_prefixes.iter().map(|v| v[i]).min().unwrap())
                .collect();
            for (rank_index, rank) in ranks.iter_mut().enumerate() {
                let landed = rank.publish_prefixes(&prefixes);
                for request in landed {
                    for item in request.items {
                        let start = (item.row_start * HIDDEN * ELEMENT) as usize;
                        let count = (item.row_count * HIDDEN * ELEMENT) as usize;
                        assert_eq!(pools[rank_index][..count], node.ring[start..start + count]);
                        seen[rank_index].push((request.rid.clone(), item.index));
                        if let Some(slot) = item.slot {
                            assert!(rank.awaiting_release());
                            let delay = 1 + (rank_index + seen[rank_index].len()) % 5;
                            pending[rank_index].push((round + delay, slot));
                        }
                    }
                }
                assert!(rank.settle(&codes).iter().all(|r| r.items.is_empty()));
            }
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(seen[0].len(), 34);
        // With one receive slot and two waiting requests, rotating the first
        // lease prevents either request from monopolizing every released slot.
        assert_ne!(seen[0][0].0, seen[0][1].0);
        assert_ne!(seen[0][2].0, seen[0][3].0);
        for rank in 0..tp {
            assert_eq!(seen[rank], seen[0], "rebuild collective order");
            ranks[rank].retire_pending(&["images-0".to_owned(), "images-1".to_owned()]);
            ranks[rank].retire_requests(&["images-0".to_owned(), "images-1".to_owned()]);
            assert!(!ranks[rank].awaiting_release());
        }
    }
}

#[test]
fn failed_partial_request_does_not_reclaim_a_published_slot() {
    let native = crate::tests_support::NativeLibrary::new();
    let mut node = encode_node(native.path().to_owned(), Duration::from_secs(30));
    let pool = vec![0u8; SLOT_BYTES];
    let mut rank = receiver(native.path().to_owned(), &pool, 1, 0, false);
    let mut second = item(node.server.port(), 202, 5);
    second.index = 1;
    rank.stage("cancelled", &[item(node.server.port(), 201, 5), second]);
    assert!(node.sender.begin(201));
    node.sender.send(chunk(&node.ring, 201, 5));
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        rank.admit_requests(&rank.admission_plan());
        let capacity = rank.lease_capacity();
        rank.lease_items(capacity);
        assert_eq!(rank.poll(), [1]);
        if rank.landed_prefixes() == [1] {
            break;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(2));
    }
    let landed = rank.publish_prefixes(&[1]);
    assert_eq!(landed[0].items[0].slot, Some(0));
    // The unfinished request fails before the device copy finishes.
    assert!(rank.settle(&[0]).is_empty());
    assert!(rank.awaiting_release());
    rank.release(&[0]);
    rank.retire_pending(&["cancelled".to_owned()]);
    rank.retire_requests(&["cancelled".to_owned()]);
    assert!(!rank.awaiting_release());
    // Failure must not quarantine the published slot after its release.
    rank.stage("next", &[item(node.server.port(), 203, 5)]);
    assert!(node.sender.begin(203));
    node.sender.send(chunk(&node.ring, 203, 5));
    let (code, landed) = settle(&mut rank);
    assert_eq!(code, 2);
    assert_eq!(landed[0].items[0].slot, Some(0));
    rank.release(&[0]);
}


#[test]
fn whole_request_budget_waits_for_device_retirement_on_every_rank() {
    let native = crate::tests_support::NativeLibrary::new();
    let mut node = encode_node(native.path().to_owned(), Duration::from_secs(30));
    let pools = [vec![0u8; SLOT_BYTES], vec![0u8; SLOT_BYTES]];
    let mut ranks: Vec<_> = (0..2).map(|rank| {
        receiver_with_budget(native.path().to_owned(), &pools[rank], 2, rank as i64,
            true, 224)
    }).collect();
    for (rid, room) in [("first", 301), ("second", 302)] {
        assert!(node.sender.begin(room));
        node.sender.send(chunk(&node.ring, room, 7));
        for rank in &mut ranks {
            rank.stage(rid, &[item(node.server.port(), room, 7)]);
        }
    }
    for rank in &mut ranks {
        // Full 7 x 8 x 4 bytes per request, even though TP2 receives fewer rows.
        assert_eq!(rank.admission_plan(), [2, 1]);
        rank.admit_requests(&[2, 1]);
        let capacity = rank.lease_capacity();
        rank.lease_items(capacity);
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let codes: Vec<_> = ranks.iter_mut().map(EmbeddingReceiver::poll).collect();
        assert!(codes.iter().all(|codes| !codes.contains(&0)));
        if codes.iter().all(|codes| codes == &[2, 1]) { break; }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(2));
    }
    for rank in &mut ranks {
        let landed = rank.publish_prefixes(&[1, 0]);
        assert_eq!(landed.len(), 1);
        rank.settle(&[2, 1]);
        rank.release(&[0]);
        assert_eq!(rank.lease_capacity(), 1);
        assert_eq!(rank.admission_plan(), [1], "slot release must not return embedding budget");
    }
    ranks[0].retire_pending(&["first".to_owned()]);
    ranks[0].retire_requests(&["first".to_owned()]);
    assert_eq!(ranks[0].admission_plan(), [2]);
    assert_eq!(ranks[1].admission_plan(), [1]);
    for rank in &mut ranks { rank.admit_requests(&[1]); }
    assert_eq!(ranks[0].poll(), [1], "cannot receive before replica admission");
    ranks[1].retire_pending(&["first".to_owned()]);
    ranks[1].retire_requests(&["first".to_owned()]);
    for rank in &mut ranks { rank.admit_requests(&[2]); }
    for (code, landed) in settle_all(&mut ranks) {
        assert_eq!(code, 2);
        assert_eq!(landed[0].rid, "second");
    }
    for rank in &mut ranks {
        rank.release(&[0]);
        rank.retire_pending(&["second".to_owned()]);
        rank.retire_requests(&["second".to_owned()]);
        assert!(!rank.awaiting_release());
    }
}

#[test]
fn an_over_budget_request_is_rejected_before_leasing() {
    let native = crate::tests_support::NativeLibrary::new();
    let node = encode_node(native.path().to_owned(), Duration::from_secs(30));
    let pool = vec![0u8; SLOT_BYTES];
    let mut rank = receiver_with_budget(native.path().to_owned(), &pool, 1, 0, false, 32);
    rank.stage("large", &[item(node.server.port(), 401, 7)]);
    assert_eq!(rank.admission_plan(), [0]);
    rank.admit_requests(&[0]);
    rank.lease_items(1);
    assert_eq!(rank.poll(), [0]);
    assert_eq!(rank.lease_capacity(), 1);
    assert!(rank.failure_reason("large").unwrap().contains("requires 224 bytes"));
    rank.settle(&[0]);
    // Rejected requests still retire their data-plane record.
    rank.retire_pending(&["large".to_owned()]);
    rank.retire_requests(&["large".to_owned()]);
    assert!(!rank.awaiting_release());
}

#[test]
fn cancellation_releases_unleased_encode_rooms_before_bootstrap_timeout() {
    let native = crate::tests_support::NativeLibrary::new();
    let library = native.path().to_owned();
    let mut node = encode_node(library.clone(), Duration::from_secs(120));
    let pool = vec![0u8; SLOT_BYTES];
    let mut prefill = receiver(library, &pool, 1, 0, false);
    // More items than the receive pool. None has published a destination.
    let items: Vec<_> = (100..117).map(|room| item(node.server.port(), room, 5)).collect();
    for request in &items {
        assert!(node.sender.begin(request.room));
        node.sender.send(chunk(&node.ring, request.room, 5));
    }
    prefill.stage("cancel", &items);
    prefill.admit_requests(&[0]);
    assert_eq!(prefill.poll(), [0]);
    prefill.settle(&[0]);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut done = std::collections::HashSet::new();
    while done.len() < items.len() {
        done.extend(node.sender.concluded());
        assert!(Instant::now() < deadline, "cancelled items still hold send slots: {done:?}");
        thread::sleep(Duration::from_millis(5));
    }
    for request in &items {
        assert!(!node.sender.begin(request.room), "late encode revived a cancelled room");
    }
    prefill.retire_requests(&["cancel".to_owned()]);
    // A fresh request immediately reuses the receive pool and sender.
    prefill.stage("healthy", &[item(node.server.port(), 200, 5)]);
    assert!(node.sender.begin(200));
    node.sender.send(chunk(&node.ring, 200, 5));
    assert_eq!(settle(&mut prefill).0, 2);
    concluded(&mut node.sender, 200);
}

#[test]
fn cancellation_does_not_return_a_send_slot_while_dma_reads_it() {
    let native = crate::tests_support::NativeLibrary::new();
    let library = native.path().to_owned();
    let mut node = encode_node(library.clone(), Duration::from_secs(120));
    let pool = vec![0u8; SLOT_BYTES];
    let mut prefill = receiver(library, &pool, 1, 0, false);
    native.block_writes();
    prefill.stage("writing", &[item(node.server.port(), 301, 5)]);
    prefill.admit_requests(&[2]);
    prefill.lease_items(1);
    assert!(node.sender.begin(301));
    node.sender.send(chunk(&node.ring, 301, 5));
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        prefill.poll();
        if native.wait_write_entered(Duration::from_millis(5)) { break; }
        assert!(Instant::now() < deadline, "DMA never started");
    }
    prefill.admit_requests(&[0]);
    prefill.settle(&[0]);
    let observe_until = Instant::now() + Duration::from_millis(200);
    while Instant::now() < observe_until {
        assert!(node.sender.concluded().is_empty(), "DMA source slot retired too early");
        thread::sleep(Duration::from_millis(5));
    }
    native.release_writes();
    concluded(&mut node.sender, 301);
    assert_eq!(&pool[..160], &node.ring[..160]);
}
