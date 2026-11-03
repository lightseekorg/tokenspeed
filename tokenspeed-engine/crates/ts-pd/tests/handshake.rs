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

//! A Prefill peer and a Decode peer in one process run the whole handshake
//! through the bootstrap server, and the KV crosses over RDMA.
//!
//! Needs an RDMA device and `TS_MOONCAKE_LIBRARY` naming
//! `libtokenspeed_transfer_engine.so`; without the variable the tests return
//! at once. Each side's arena is one 4 KiB buffer holding a one-group cache
//! of two fields in two planes; the tests stand in for the forward thread,
//! staging the plans' remote streams and handing them over.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use serde_json::{json, Value};
use ts_collective::{Group, Listener, Mesh};
use ts_pd::{
    layout::{contract::Contract, manifest::BlockTable},
    BatchRows, BootstrapInfo, BootstrapServer, DecodeConfig, DecodePeer, Layerwise, PdEvent,
    PeerConfig, PrefillConfig, PrefillLayout, PrefillPeer, Registry, RemotePrefill, StatusMessage,
    TransferPeer, TransferPoll,
};

const BYTES: usize = 4096;

/// Where each plane's pages 1-3 sit in an arena: page 0 is the null page,
/// which no transfer touches.
const PAGES: [std::ops::Range<usize>; 2] = [512..2048, 2560..4096];

fn library() -> Option<PathBuf> {
    std::env::var_os("TS_MOONCAKE_LIBRARY").map(PathBuf::from)
}

/// The test cache's contract: one 16-token-block group of three pages and
/// the null page, a 512-byte field per plane; with the fields' landing
/// addresses for an arena at `base`.
fn layout(base: Option<u64>) -> String {
    let addresses = base.map_or(
        Value::Null,
        |base| json!({"layer.0.kv": base, "layer.1.kv": base + 2048}),
    );
    json!({
        "plan": {
            "prefix_granularity": 16,
            "lcm_block_bytes": 1024,
            "num_lcm_blocks": 3,
            "groups": [{"group_id": "history", "cache_blocks_per_lcm_block": 1, "page_count": 4}],
            "planes": [
                {"plane_id": "p0", "bytes_per_lcm_block": 512, "arena_offset_bytes": 0},
                {"plane_id": "p1", "bytes_per_lcm_block": 512, "arena_offset_bytes": 2048}
            ],
            "fields": [
                {"group_id": "history", "field_id": "layer.0.kv", "plane_id": "p0",
                 "shape": [256], "dtype": "bfloat16", "field_offset_bytes": 0,
                 "page_stride_bytes": 512},
                {"group_id": "history", "field_id": "layer.1.kv", "plane_id": "p1",
                 "shape": [256], "dtype": "bfloat16", "field_offset_bytes": 0,
                 "page_stride_bytes": 512}
            ],
            "resident_block_bytes": null
        },
        "group_specs": [{"group_id": "history", "attn_kind": "full_history", "rows_per_page": 16,
                         "entry_stride_tokens": 1, "sliding_window_tokens": null,
                         "replayable": false, "transfer_policy": "full_suffix", "shard_count": 1}],
        "transfer_schema": {"fields": []},
        "field_addresses": addresses
    })
    .to_string()
}

const FIELDS: [&str; 2] = ["layer.0.kv", "layer.1.kv"];

/// The prompt every test request carries: three blocks.
const PROMPT: i32 = 48;

fn tables() -> ts_pd::layout::manifest::BlockTables {
    BTreeMap::from([(
        "history".to_owned(),
        BlockTable::from_rows(&[vec![1, 2, 3]], 3),
    )])
}

/// A Decode plan's remote prefill: `rid`'s three pages admitted at slot 3.
fn admission(rid: &str) -> BatchRows {
    BatchRows {
        request_ids: vec![rid.to_owned()],
        request_pool_indices: vec![3],
        input_lengths: vec![PROMPT],
        prefill_lengths: vec![PROMPT],
        extend_prefix_lens: vec![0],
        state_prefix_cached: vec![false],
        num_extends: 1,
        tables: tables(),
        ..BatchRows::default()
    }
}

/// A Prefill plan's remote decode: `rid`'s prompt done, `token` its first
/// output.
fn remote_decode(rid: &str, token: i32, candidates: Vec<i32>) -> BatchRows {
    BatchRows {
        request_ids: vec![rid.to_owned()],
        request_pool_indices: vec![0],
        input_lengths: vec![1],
        prefill_lengths: vec![PROMPT],
        decode_input_ids: vec![token],
        spec_candidate_ids: vec![candidates],
        tables: tables(),
        ..BatchRows::default()
    }
}

/// A Prefill plan's forward: `rid`'s whole prompt in one chunk.
fn prompt_forward(rid: &str) -> BatchRows {
    BatchRows {
        request_ids: vec![rid.to_owned()],
        request_pool_indices: vec![0],
        input_lengths: vec![PROMPT],
        prefill_lengths: vec![PROMPT],
        extend_prefix_lens: vec![0],
        state_prefix_cached: vec![false],
        num_extends: 1,
        tables: tables(),
        ..BatchRows::default()
    }
}

/// Push `frames` to a peer's control socket, as a foreign peer would.
fn push(port: u16, frames: Vec<Bytes>) {
    use zeromq::{Socket, SocketSend};
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut socket = zeromq::PushSocket::new();
        socket
            .connect(&format!("tcp://127.0.0.1:{port}"))
            .await
            .unwrap();
        socket
            .send(zeromq::ZmqMessage::try_from(frames).unwrap())
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    });
}

fn single_rank() -> (Mesh, Group) {
    let listener = Listener::bind("127.0.0.1").unwrap();
    let address = format!("127.0.0.1:{}", listener.port().unwrap());
    let mesh = Mesh::connect(0, listener, &[address], Duration::from_secs(5)).unwrap();
    (mesh, Group::new(0, vec![0], 0).unwrap())
}

fn peer_config(library: PathBuf, region: &[u8]) -> PeerConfig {
    PeerConfig {
        library,
        local_ip: "127.0.0.1".to_owned(),
        protocol: "rdma".to_owned(),
        ib_device: String::new(),
        regions: vec![(region.as_ptr() as usize, region.len())],
        engine_rank: 0,
        metrics: None,
    }
}

struct Pair {
    prefill: PrefillPeer,
    prefill_outbox: ts_pd::PrefillOutbox,
    decode: DecodePeer,
    decode_outbox: ts_pd::DecodeOutbox,
    bootstrap: BootstrapInfo,
    _server: BootstrapServer,
}

fn pair(library: PathBuf, source: &[u8], destination: &[u8], bootstrap_timeout: Duration) -> Pair {
    pair_with(library, source, destination, bootstrap_timeout, None)
}

fn pair_with(
    library: PathBuf,
    source: &[u8],
    destination: &[u8],
    bootstrap_timeout: Duration,
    step_address: Option<usize>,
) -> Pair {
    let server = BootstrapServer::start(0, Registry::CacheLayouts).unwrap();
    let port = server.port();
    let source_base = source.as_ptr() as u64;
    let parse = |wire: String| Contract::from_wire(wire.as_bytes()).unwrap();
    let (prefill, prefill_outbox) = PrefillPeer::start(PrefillConfig {
        peer: peer_config(library.clone(), source),
        layout: PrefillLayout {
            local: parse(layout(Some(source_base))),
            wire: parse(layout(None)),
            arena_base: source_base,
            tp_size: 1,
            tp_rank: 0,
            pp_rank: 0,
            fields_by_stage: vec![FIELDS.map(str::to_owned).to_vec()],
        },
        status_rank: 0,
        bootstrap_url: format!("http://127.0.0.1:{port}/route"),
        bootstrap_payload: json!({
            "role": "Prefill",
            "world_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "rank_ip": "127.0.0.1",
            "engine_rank": 0,
            "cache_layout": layout(None),
            "cache_fields_by_stage": [FIELDS],
        }),
        bootstrap_timeout,
        failed_session_ttl: Duration::from_secs(30),
        transfer_workers: 2,
        layerwise: step_address.map(|step_address| Layerwise {
            step_address,
            fields_by_step: FIELDS.map(|field| vec![field.to_owned()]).to_vec(),
            interval: 1,
        }),
    })
    .unwrap();
    let destination_base = destination.as_ptr() as u64;
    let (decode, decode_outbox) = DecodePeer::start(DecodeConfig {
        peer: peer_config(library, destination),
        arena_base: destination_base,
        layout: layout(Some(destination_base)).into_bytes(),
        tp_size: 1,
        tp_rank: 0,
        waiting_timeout: Duration::from_secs(60),
        heartbeat_interval: Duration::from_secs(2),
        heartbeat_max_failures: 2,
    })
    .unwrap();
    Pair {
        prefill,
        prefill_outbox,
        decode,
        decode_outbox,
        bootstrap: BootstrapInfo {
            host: "127.0.0.1".to_owned(),
            port,
            room: 4242,
        },
        _server: server,
    }
}

impl Pair {
    /// The forward thread hands over a Decode plan admitting `rid`.
    fn admit(&self, rid: &str) {
        self.decode_outbox
            .stage_remote_prefill(admission(rid))
            .unwrap();
        self.decode_outbox.hand_over_remote_prefill().unwrap();
    }

    /// The forward thread hands over a Prefill plan's remote decode of
    /// `rid`.
    fn send(&self, rid: &str, token: i32, candidates: Vec<i32>) -> Result<(), String> {
        self.prefill_outbox
            .stage_remote_decode(remote_decode(rid, token, candidates))
            .unwrap();
        self.prefill_outbox.hand_over_remote_decode()
    }
}

/// The transferred pages of `destination` hold `source`'s bytes; the null
/// pages are untouched.
fn landed(destination: &[u8], source: &[u8]) -> bool {
    PAGES
        .iter()
        .all(|pages| destination[pages.clone()] == source[pages.clone()])
        && destination[..512].iter().all(|&byte| byte == 0)
        && destination[2048..2560].iter().all(|&byte| byte == 0)
}

/// Poll until `peer` reports an event.
fn next_event(peer: &mut dyn TransferPeer, mesh: &mut Mesh, group: &Group) -> PdEvent {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let mut events = peer.poll(mesh, group).unwrap();
        if let Some(event) = events.pop() {
            assert!(events.is_empty());
            return event;
        }
        assert!(Instant::now() < deadline, "no transfer event");
        thread::sleep(Duration::from_millis(5));
    }
}

/// An arena's bytes: a pattern, page by page.
fn pattern() -> Vec<u8> {
    (0..BYTES).map(|i| (i % 251) as u8 + 1).collect()
}

#[test]
fn a_request_bootstraps_transfers_and_lands() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(library, &source, &destination, Duration::from_secs(60));
    let (mut mesh, group) = single_rank();
    let rid = "req-0";

    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    // Decode connects and registers its session; the room is bootstrapped
    // on its side before Prefill hears of a manifest.
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    // The scheduler admits the pages; the manifest goes out.
    pair.admit(rid);
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    // The prompt's last chunk commits, then its remote decode runs.
    pair.prefill.record_usage(rid, 128, Some(-0.5));
    pair.send(rid, 777, vec![1, 2, 3]).unwrap();
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Succeeded(rid.to_owned())
    );
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::RemotePrefillDone(RemotePrefill {
            rid: rid.to_owned(),
            bootstrap_token: 777,
            cached_tokens: 128,
            bootstrap_logprob: Some(-0.5),
            spec_candidate_ids: Some(vec![1, 2, 3]),
            request_pool_index: 3,
        })
    );
    assert!(landed(&destination, &source));
    // Both sides forgot the request.
    assert!(pair.prefill.poll(&mut mesh, &group).unwrap().is_empty());
    assert!(pair.decode.poll(&mut mesh, &group).unwrap().is_empty());
}

/// Run one request through `pair`, from registration to landing; a sleep
/// is refused on both sides while it is in flight.
fn transfer_one(pair: &mut Pair, rid: &str, room: u64, mesh: &mut Mesh, group: &Group) {
    let bootstrap = BootstrapInfo {
        room,
        ..pair.bootstrap.clone()
    };
    pair.prefill.register(rid, &bootstrap).unwrap();
    pair.decode.register(rid, &bootstrap).unwrap();
    let refused = pair.prefill_outbox.release_registration().unwrap_err();
    assert!(refused.contains("KV sends are in flight"), "{refused}");
    let refused = pair.decode_outbox.release_registration().unwrap_err();
    assert!(refused.contains("KV receives are in flight"), "{refused}");
    assert_eq!(
        next_event(&mut pair.decode, mesh, group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.admit(rid);
    assert_eq!(
        next_event(&mut pair.prefill, mesh, group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.send(rid, 777, Vec::new()).unwrap();
    assert_eq!(
        next_event(&mut pair.prefill, mesh, group),
        PdEvent::Succeeded(rid.to_owned())
    );
    assert!(matches!(
        next_event(&mut pair.decode, mesh, group),
        PdEvent::RemotePrefillDone(_)
    ));
}

#[test]
fn a_sleeping_pair_deregisters_and_wakes_to_transfer_again() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let mut destination = vec![0u8; BYTES];
    let mut pair = pair(library, &source, &destination, Duration::from_secs(60));
    let (mut mesh, group) = single_rank();
    transfer_one(&mut pair, "before", 4242, &mut mesh, &group);
    assert!(landed(&destination, &source));

    let session = pair.decode.session_id();
    pair.prefill_outbox.release_registration().unwrap();
    pair.decode_outbox.release_registration().unwrap();
    pair.prefill_outbox.restore_registration().unwrap();
    pair.decode_outbox.restore_registration().unwrap();
    // The Decode rank wakes under a fresh session, and the next request
    // registers it with the Prefill rank again.
    assert_ne!(pair.decode.session_id(), session);
    destination.iter_mut().for_each(|byte| *byte = 0);
    transfer_one(&mut pair, "after", 4244, &mut mesh, &group);
    assert!(landed(&destination, &source));
}

#[test]
fn layerwise_chunks_follow_the_producer_steps() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    // The counter the device would update; the peer polls it. Each field is
    // one producer step, and one transfer interval.
    let steps = Box::new(std::sync::atomic::AtomicI64::new(0));
    let mut pair = pair_with(
        library,
        &source,
        &destination,
        Duration::from_secs(60),
        Some(steps.as_ptr() as usize),
    );
    let (mut mesh, group) = single_rank();
    let rid = "req-lw";
    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.admit(rid);
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    // The prompt's only chunk arms ahead of its forward: its intervals are
    // complete once the counter passes steps 0 and 1.
    assert!(pair.prefill_outbox.streams_layerwise());
    pair.prefill_outbox
        .arm_layerwise(&prompt_forward(rid))
        .unwrap();
    thread::sleep(Duration::from_millis(50));
    assert!(
        destination.iter().all(|&byte| byte == 0),
        "nothing before its step"
    );
    steps.store(1, std::sync::atomic::Ordering::Release);
    let first = PAGES[0].clone();
    let deadline = Instant::now() + Duration::from_secs(10);
    while std::hint::black_box(destination.as_slice())[first.clone()] != source[first.clone()] {
        assert!(Instant::now() < deadline, "the first field never landed");
        thread::sleep(Duration::from_millis(5));
    }
    assert!(destination[PAGES[1].clone()].iter().all(|&byte| byte == 0));
    steps.store(2, std::sync::atomic::Ordering::Release);
    // The room reports done only once its remote decode brings the payload.
    thread::sleep(Duration::from_millis(50));
    assert!(pair.prefill.poll(&mut mesh, &group).unwrap().is_empty());
    pair.prefill.record_usage(rid, 0, None);
    pair.send(rid, 99, Vec::new()).unwrap();
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Succeeded(rid.to_owned())
    );
    match next_event(&mut pair.decode, &mut mesh, &group) {
        PdEvent::RemotePrefillDone(done) => assert_eq!(done.bootstrap_token, 99),
        other => panic!("expected the landing, got {other:?}"),
    }
    assert!(landed(&destination, &source));
}

#[test]
fn a_refused_preallocation_fails_both_sides() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(library, &source, &destination, Duration::from_secs(60));
    let (mut mesh, group) = single_rank();
    let rid = "req-1";

    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    // The registered Decode session pre-allocates a manifest naming a block
    // outside its own layout.
    let manifest = br#"{"groups":[{"group_id":"history","block_ids":[1,2,9]}],"prefix_len":0,"prompt_len":48,"state_prefix_cached":false}"#;
    push(
        pair.prefill.port(),
        vec![
            Bytes::from("4242"),
            Bytes::from(pair.decode.session_id()),
            Bytes::from_static(manifest),
        ],
    );
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    // A late remote decode of the failed room sends nothing.
    pair.send(rid, 5, Vec::new()).unwrap();
    thread::sleep(Duration::from_millis(50));
    assert!(destination.iter().all(|&byte| byte == 0));
}

#[test]
fn a_room_no_decode_rank_reaches_times_out() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(library, &source, &destination, Duration::from_millis(200));
    let (mut mesh, group) = single_rank();

    pair.prefill.register("lonely", &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Failed("lonely".to_owned())
    );
    // A room still bootstrapping has no destinations to send to: a remote
    // decode of it breaks an invariant.
    pair.prefill
        .register(
            "pending",
            &BootstrapInfo {
                room: 9,
                ..pair.bootstrap.clone()
            },
        )
        .unwrap();
    let error = pair.send("pending", 5, Vec::new()).unwrap_err();
    assert!(
        error.contains("destination metadata is unavailable"),
        "{error}"
    );
}

#[test]
fn rooms_no_request_claimed_neither_hold_a_sleep_nor_linger() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(library, &source, &destination, Duration::from_millis(500));
    let (mut mesh, group) = single_rank();
    // The Prefill rank failed a request before it registered there (an EPD
    // embedding timeout).
    pair.prefill.abort_room("epd-0", 77);
    // A Decode rank pre-allocates for a request the Prefill rank never
    // registers (it refused it at admission).
    let rid = "refused";
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.admit(rid);
    thread::sleep(Duration::from_millis(100));
    // Neither room sends anything, so neither holds a sleep.
    pair.prefill_outbox.release_registration().unwrap();
    pair.prefill_outbox.restore_registration().unwrap();
    // Unclaimed past the bootstrap timeout, the pre-allocated room is
    // dropped and its Decode rank told.
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
}

#[test]
fn an_admission_after_its_request_ended_is_not_sent() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(library, &source, &destination, Duration::from_secs(60));
    let (mut mesh, group) = single_rank();
    // No bootstrap server listens there: the room's route lookup fails.
    let unreachable = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let lost = BootstrapInfo {
        port: unreachable,
        room: 9,
        ..pair.bootstrap.clone()
    };
    pair.decode.register("lost", &lost).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed("lost".to_owned())
    );
    // The admission the loop planned before the failure reaches the outbox
    // only after it: the room stays forgotten, and a sleep proceeds.
    pair.admit("lost");
    pair.decode_outbox.release_registration().unwrap();
    assert!(pair.decode.poll(&mut mesh, &group).unwrap().is_empty());
}

#[test]
fn a_landing_without_its_first_token_fails_the_request() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(library, &source, &destination, Duration::from_secs(60));
    let (mut mesh, group) = single_rank();
    let rid = "tokenless";
    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.admit(rid);
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    // This rank refuses to send a prompt without its first token...
    let error = pair.send(rid, -1, Vec::new()).unwrap_err();
    assert!(error.contains("bootstrap token is unavailable"), "{error}");
    // ...but an older or foreign Prefill peer reports the KV sent with none.
    push(
        pair.decode.port(),
        StatusMessage::bare(4242, TransferPoll::Success, 0).to_frames(),
    );
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
}

#[test]
fn a_conflicting_registration_rejects_its_session() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(library, &source, &destination, Duration::from_secs(60));
    let (mut mesh, group) = single_rank();
    let rid = "req-2";
    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.admit(rid);
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    // The same session registers again with its arena elsewhere: the
    // Prefill rank no longer trusts it, and fails the rooms it serves.
    let moved = destination.as_ptr() as u64 + 0x10000;
    push(
        pair.prefill.port(),
        vec![
            Bytes::from_static(b"None"),
            Bytes::from("127.0.0.1"),
            Bytes::from(pair.decode.port().to_string()),
            Bytes::from(pair.decode.session_id()),
            Bytes::copy_from_slice(&moved.to_ne_bytes()),
            Bytes::from(layout(Some(moved))),
            Bytes::from("1"),
            Bytes::from("0"),
        ],
    );
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
}
