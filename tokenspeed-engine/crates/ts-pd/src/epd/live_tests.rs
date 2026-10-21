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
