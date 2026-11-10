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

//! CPU integration tests for CachePD cache incarnations and address reuse.
//!
//! The real peers, bootstrap service and control sockets run unchanged. Only
//! the unavailable native RDMA library is replaced by checked CPU copies.

use std::{
    collections::HashSet,
    sync::atomic::{AtomicI64, Ordering},
    thread,
    time::Duration,
};

use bytes::Bytes;
use ts_pd::{BootstrapInfo, PdEvent, TransferPeer};

#[path = "../src/tests_support.rs"]
mod native;
mod support;
use support::*;

#[test]
fn repeated_wakes_transfer_with_reused_addresses_and_reject_old_cache_messages() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let mut destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(60),
    );
    let (mut mesh, group) = single_rank();
    transfer_one(&mut pair, "initial", 4242, &mut mesh, &group);
    assert!(landed(&destination, &source));

    let first_cache = pair.decode.cache_id();
    let mut caches = HashSet::from([first_cache.clone()]);
    let mut addresses = HashSet::from([pair.decode.session_id()]);
    let mut reused = false;
    // The native fixture has four reusable addresses. Five successive
    // cache incarnations must revisit one while Prefill remains alive.
    for round in 1..=4 {
        pair.decode_outbox.release_registration().unwrap();
        pair.decode_outbox.restore_registration().unwrap();
        assert!(
            caches.insert(pair.decode.cache_id()),
            "a wake must change cache identity"
        );
        reused |= !addresses.insert(pair.decode.session_id());
        destination.fill(0);
        transfer_one(
            &mut pair,
            &format!("wake-{round}"),
            4242 + round,
            &mut mesh,
            &group,
        );
        assert!(
            landed(&destination, &source),
            "KV did not land after wake {round}"
        );
    }
    assert!(
        reused,
        "the test must actually reuse a historical transfer address"
    );

    // A delayed request naming the first, retired cache must never write to
    // the current incarnation, even though its transport address was reused.
    destination.fill(0);
    let stale = BootstrapInfo {
        room: 5000,
        ..pair.bootstrap.clone()
    };
    pair.prefill.register("stale", &stale).unwrap();
    let manifest = br#"{"groups":[{"group_id":"history","block_ids":[1,2,3]}],"prefix_len":0,"prompt_len":48,"state_prefix_cached":false}"#;
    push(
        pair.prefill.port(),
        vec![
            Bytes::from("5000"),
            Bytes::from(first_cache),
            Bytes::from_static(manifest),
        ],
    );
    expect_cancelled(&mut pair.prefill, "stale", &mut mesh, &group);
    assert!(destination.iter().all(|&byte| byte == 0));

    transfer_one(&mut pair, "current", 5001, &mut mesh, &group);
    assert!(
        landed(&destination, &source),
        "a late old-cache message must not poison the current cache"
    );
}

#[test]
fn replacing_a_cache_retires_queued_layerwise_reads_before_releasing_the_room() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let steps = Box::new(AtomicI64::new(0));
    let mut pair = pair_with(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(60),
        Some(steps.as_ptr() as usize),
    );
    let (mut mesh, group) = single_rank();
    let rid = "retired-queued";
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

    // A replacement at the same endpoint retires both the job waiting for
    // its producer and the jobs behind it, before the request's pages leave.
    let base = destination.as_ptr() as u64;
    push(
        pair.prefill.port(),
        vec![
            Bytes::from_static(b"None"),
            Bytes::from("127.0.0.1"),
            Bytes::from(pair.decode.port().to_string()),
            Bytes::from(pair.decode.session_id()),
            Bytes::copy_from_slice(&base.to_ne_bytes()),
            Bytes::from(layout(Some(base))),
            Bytes::from("1"),
            Bytes::from("0"),
            Bytes::from("replacement-cache"),
        ],
    );
    expect_cancelled(&mut pair.prefill, rid, &mut mesh, &group);
    assert!(!pair.prefill.has_in_flight());
    steps.store(1000, Ordering::Release);
    thread::sleep(Duration::from_millis(100));
    assert!(
        destination.iter().all(|&byte| byte == 0),
        "retired cache work must not resume writes after its producer advances"
    );
    assert!(pair.prefill.poll(&mut mesh, &group).unwrap().is_empty());
}
