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

use bytes::Bytes;
use std::{
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};
use ts_pd::{BootstrapInfo, PdEvent, RemotePrefill, StatusMessage, TransferPeer, TransferPoll};

mod support;
use support::*;

fn library() -> Option<PathBuf> {
    std::env::var_os("TS_MOONCAKE_LIBRARY").map(PathBuf::from)
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
    let cache_id = pair.decode.cache_id();
    pair.prefill_outbox.release_registration().unwrap();
    pair.decode_outbox.release_registration().unwrap();
    pair.prefill_outbox.restore_registration().unwrap();
    pair.decode_outbox.restore_registration().unwrap();
    // The Decode rank wakes under a fresh session, and the next request
    // registers it with the Prefill rank again.
    assert_ne!(pair.decode.session_id(), session);
    assert_ne!(pair.decode.cache_id(), cache_id);
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
            Bytes::from(pair.decode.cache_id()),
            Bytes::from_static(manifest),
        ],
    );
    expect_cancelled(&mut pair.prefill, rid, &mut mesh, &group);
    expect_cancelled(&mut pair.decode, rid, &mut mesh, &group);
    // A late remote decode of the failed room sends nothing.
    pair.send(rid, 5, Vec::new()).unwrap();
    thread::sleep(Duration::from_millis(50));
    assert!(destination.iter().all(|&byte| byte == 0));
}

#[test]
fn failure_retires_queued_layerwise_reads_before_releasing_the_room() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let steps = Box::new(std::sync::atomic::AtomicI64::new(0));
    let mut pair = pair_with(
        library,
        &source,
        &destination,
        Duration::from_secs(60),
        Some(steps.as_ptr() as usize),
    );
    let (mut mesh, group) = single_rank();
    let rid = "cancel-queued";
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
    for _ in 0..64 {
        pair.prefill_outbox
            .arm_layerwise(&prompt_forward(rid))
            .unwrap();
    }
    // The first job waits for its producer; the rest are still queued.
    pair.prefill.abort_room(rid, pair.bootstrap.room);
    expect_cancelled(&mut pair.prefill, rid, &mut mesh, &group);
    assert!(!pair.prefill.has_in_flight());
    steps.store(1000, std::sync::atomic::Ordering::Release);
    thread::sleep(Duration::from_millis(100));
    assert!(
        destination.iter().all(|&byte| byte == 0),
        "retired work must never resume reads after the producer advances"
    );
    assert!(pair.prefill.poll(&mut mesh, &group).unwrap().is_empty());
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
    expect_cancelled(&mut pair.prefill, "lonely", &mut mesh, &group);
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
    expect_cancelled(&mut pair.decode, rid, &mut mesh, &group);
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
    expect_cancelled(&mut pair.decode, "lost", &mut mesh, &group);
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
    expect_cancelled(&mut pair.decode, rid, &mut mesh, &group);
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
            Bytes::from(pair.decode.cache_id()),
        ],
    );
    expect_cancelled(&mut pair.prefill, rid, &mut mesh, &group);
    expect_cancelled(&mut pair.decode, rid, &mut mesh, &group);
}
