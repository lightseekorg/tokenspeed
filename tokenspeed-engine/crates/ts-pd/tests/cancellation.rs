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

//! Safe cancellation through the real peers, sockets and bootstrap service.

use bytes::Bytes;
use std::{
    sync::atomic::{AtomicI64, Ordering},
    thread,
    time::{Duration, Instant},
};
use ts_pd::{PdEvent, TransferPeer};

#[path = "../src/tests_support.rs"]
mod native;
mod support;
use support::*;

fn prepare(
    pair: &mut Pair,
    rid: &str,
    mesh: &mut ts_collective::Mesh,
    group: &ts_collective::Group,
) {
    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, mesh, group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.admit(rid);
    assert_eq!(
        next_event(&mut pair.prefill, mesh, group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
}

#[test]
fn cancellation_before_connection_does_not_wait_for_a_prefill_request() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
    );
    let (mut mesh, group) = single_rank();
    pair.decode.register("cancelled", &pair.bootstrap).unwrap();
    pair.decode.cancel("cancelled");
    pair.decode.cancel("cancelled");
    pair.decode.cancel("unknown");
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed("cancelled".to_owned())
    );
    // A plan already handed to the forward thread cannot publish a late manifest.
    pair.admit("cancelled");
    assert!(!pair.decode.has_in_flight());
    pair.decode_outbox.release_registration().unwrap();
    assert!(destination.iter().all(|&byte| byte == 0));
}

#[test]
fn an_unclaimed_room_cancels_without_waiting_for_ttl_and_cannot_be_revived() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
    );
    let (mut mesh, group) = single_rank();
    let rid = "unclaimed";
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.admit(rid);
    pair.decode.cancel(rid);
    pair.decode.cancel(rid);
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    assert!(!pair.decode.has_in_flight());
    // A delayed manifest and claim cannot restart the cancelled room.
    let manifest = br#"{"groups":[{"group_id":"history","block_ids":[1,2,3]}],"prefix_len":0,"prompt_len":48,"state_prefix_cached":false}"#;
    push(
        pair.prefill.port(),
        vec![
            Bytes::from(pair.bootstrap.room.to_string()),
            Bytes::from(pair.decode.cache_id()),
            Bytes::from_static(manifest),
        ],
    );
    pair.prefill
        .register("late-claim", &pair.bootstrap)
        .unwrap();
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Failed("late-claim".to_owned())
    );
    assert!(destination.iter().all(|&byte| byte == 0));
}

#[test]
fn cancellation_retires_queued_layerwise_work_before_releasing_the_room() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let steps = Box::new(AtomicI64::new(0));
    let mut pair = pair_with(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
        Some(steps.as_ptr() as usize),
    );
    let (mut mesh, group) = single_rank();
    let rid = "queued";
    prepare(&mut pair, rid, &mut mesh, &group);
    for _ in 0..64 {
        pair.prefill_outbox
            .arm_layerwise(&prompt_forward(rid))
            .unwrap();
    }
    pair.decode.cancel(rid);
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    steps.store(1000, Ordering::Release);
    thread::sleep(Duration::from_millis(100));
    assert!(
        destination.iter().all(|&byte| byte == 0),
        "cancelled queued work resumed writes"
    );
    assert!(!pair.decode.has_in_flight() && !pair.prefill.has_in_flight());
}

#[test]
fn cancellation_cannot_release_pages_while_a_native_write_is_executing() {
    blocked_write_retirement(true);
}

#[test]
fn a_prefill_failure_also_requires_remote_write_retirement_before_decode_releases() {
    blocked_write_retirement(false);
}

fn blocked_write_retirement(cancel_decode: bool) {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let mut destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
    );
    let (mut mesh, group) = single_rank();
    let rid = "writing";
    prepare(&mut pair, rid, &mut mesh, &group);
    library.block_writes();
    pair.send(rid, 7, Vec::new()).unwrap();
    assert!(library.wait_write_entered(Duration::from_secs(5)));
    if cancel_decode {
        pair.decode.cancel(rid);
        pair.decode.cancel(rid);
    }
    // Without a local abort, Decode must treat the ordinary failure report
    // as cancellation intent, not proof that its pages are free to reuse.
    pair.prefill.cancel(rid);
    let until = Instant::now() + Duration::from_millis(150);
    while Instant::now() < until {
        assert!(pair.decode.poll(&mut mesh, &group).unwrap().is_empty());
        assert!(pair.prefill.poll(&mut mesh, &group).unwrap().is_empty());
        thread::sleep(Duration::from_millis(5));
    }
    assert!(pair.decode_outbox.release_registration().is_err());
    assert!(pair.prefill_outbox.release_registration().is_err());
    library.release_writes();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    destination.fill(0);
    thread::sleep(Duration::from_millis(100));
    assert!(
        destination.iter().all(|&byte| byte == 0),
        "a retired write touched recycled pages"
    );
    transfer_one(&mut pair, "next", 4243, &mut mesh, &group);
    assert!(landed(&destination, &source));
}
