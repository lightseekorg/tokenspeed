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

//! PD failure isolation and retirement through real peers and control sockets.
use std::{
    net::TcpListener,
    thread,
    time::{Duration, Instant},
};
use ts_pd::{BootstrapInfo, PdEvent, TransferPeer};
#[path = "../src/tests_support.rs"]
mod native;
mod support;
use support::*;

#[test]
fn expired_prefill_rejects_late_decode_manifest_promptly() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_millis(800),
    );
    let (mut mesh, group) = single_rank();
    let rid = "expired-before-d-admission";
    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.into())
    );
    expect_cancelled(&mut pair.prefill, rid, &mut mesh, &group);
    assert!(!pair.prefill.has_in_flight());
    let unseen = pair.decode.poll(&mut mesh, &group).unwrap();
    println!("case1: P expired; D events before late admission: {unseen:?}");
    let started = Instant::now();
    pair.admit(rid);
    expect_cancelled(&mut pair.decode, rid, &mut mesh, &group);
    let waited = started.elapsed();
    println!(
        "case1: D retained admitted room for {:.3}s after P had already expired",
        waited.as_secs_f64()
    );
    assert!(
        waited < Duration::from_millis(300),
        "expired P room was forgotten: late D waits another TTL ({waited:?})"
    );
}

#[test]
fn explicit_prefill_cancel_rejects_late_decode_manifest_promptly() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(60),
    );
    let (mut mesh, group) = single_rank();
    let rid = "explicitly-cancelled-before-admission";
    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.into())
    );
    pair.prefill.cancel(rid);
    expect_cancelled(&mut pair.prefill, rid, &mut mesh, &group);
    let started = Instant::now();
    pair.admit(rid);
    expect_cancelled(&mut pair.decode, rid, &mut mesh, &group);
    println!(
        "control: explicit P cancellation retired late D in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    assert!(started.elapsed() < Duration::from_millis(300));
}

#[test]
fn slow_bootstrap_does_not_delay_healthy_peer_registration() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(60),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (accepted, accepted_rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        accepted.send(()).unwrap();
        thread::sleep(Duration::from_secs(6));
    });
    let bad = BootstrapInfo {
        port,
        room: 999,
        ..pair.bootstrap.clone()
    };
    let (mut mesh, group) = single_rank();
    pair.decode.register("bad-bootstrap", &bad).unwrap();
    accepted_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let started = Instant::now();
    pair.decode
        .register("healthy-bootstrap", &pair.bootstrap)
        .unwrap();
    loop {
        let events = pair.decode.poll(&mut mesh, &group).unwrap();
        if events.contains(&PdEvent::Bootstrapped("healthy-bootstrap".into())) {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(10));
        thread::sleep(Duration::from_millis(5));
    }
    let waited = started.elapsed();
    println!(
        "bootstrap: healthy peer registration waited {:.3}s behind a silent HTTP peer",
        waited.as_secs_f64()
    );
    pair.decode.cancel("healthy-bootstrap");
    pair.decode.cancel("bad-bootstrap");
    let mut cancelled = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(2);
    while pair.decode.has_in_flight() {
        cancelled.extend(pair.decode.poll(&mut mesh, &group).unwrap());
        assert!(
            Instant::now() < deadline,
            "cancelled lookups did not retire"
        );
        thread::sleep(Duration::from_millis(5));
    }
    for rid in ["healthy-bootstrap", "bad-bootstrap"] {
        assert!(cancelled.contains(&PdEvent::Failed(rid.into())));
        assert!(cancelled.contains(&PdEvent::Stopped(rid.into())));
    }
    assert!(
        waited < Duration::from_secs(1),
        "healthy route lookup blocked behind another endpoint ({waited:?})"
    );
}

fn extra_decode(
    library: &native::NativeLibrary,
    destination: &[u8],
    waiting_timeout: Duration,
) -> (ts_pd::DecodePeer, ts_pd::DecodeOutbox) {
    ts_pd::DecodePeer::start(ts_pd::DecodeConfig {
        peer: peer_config(library.path().to_path_buf(), destination),
        arena_base: destination.as_ptr() as u64,
        layout: layout(Some(destination.as_ptr() as u64)).into_bytes(),
        tp_size: 1,
        tp_rank: 0,
        waiting_timeout,
        heartbeat_interval: Duration::from_secs(2),
        heartbeat_max_failures: 2,
    })
    .unwrap()
}

#[test]
fn decode_queue_wait_is_bounded_by_waiting_timeout() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(60),
    );
    let extra_destination = vec![0u8; BYTES];
    let (mut decode, _outbox) =
        extra_decode(&library, &extra_destination, Duration::from_millis(200));
    let (mut mesh, group) = single_rank();
    let rid = "never-admitted";
    decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.into())
    );
    let started = Instant::now();
    let mut events = Vec::new();
    while started.elapsed() < Duration::from_millis(600) {
        events.extend(decode.poll(&mut mesh, &group).unwrap());
        thread::sleep(Duration::from_millis(5));
    }
    println!("pre-admission: configured waiting_timeout=200ms, polled {:.3}s; events={events:?}; in_flight={}", started.elapsed().as_secs_f64(), decode.has_in_flight());
    if decode.has_in_flight() {
        decode.cancel(rid);
        expect_cancelled(&mut decode, rid, &mut mesh, &group);
    }
    assert!(
        events.contains(&PdEvent::Failed(rid.into())),
        "waiting_timeout does not cover the pre-admission queue"
    );
}

#[test]
fn blocked_write_does_not_block_fresh_decode_registration() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(60),
    );
    let (mut mesh, group) = single_rank();
    let rid = "blocked-writer";
    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.into())
    );
    pair.admit(rid);
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.into())
    );
    library.block_writes();
    pair.send(rid, 7, vec![]).unwrap();
    assert!(library.wait_write_entered(Duration::from_secs(2)));

    let extra_destination = vec![0u8; BYTES];
    let (mut extra, extra_outbox) =
        extra_decode(&library, &extra_destination, Duration::from_secs(60));
    let bootstrap = BootstrapInfo {
        room: 4243,
        ..pair.bootstrap.clone()
    };
    let fresh = "fresh-destination";
    pair.prefill.register(fresh, &bootstrap).unwrap();
    extra.register(fresh, &bootstrap).unwrap();
    assert_eq!(
        next_event(&mut extra, &mut mesh, &group),
        PdEvent::Bootstrapped(fresh.into())
    );
    extra_outbox.stage_remote_prefill(admission(fresh)).unwrap();
    extra_outbox.hand_over_remote_prefill().unwrap();
    let started = Instant::now();
    let mut before = Vec::new();
    while started.elapsed() < Duration::from_millis(400) {
        before.extend(pair.prefill.poll(&mut mesh, &group).unwrap());
        thread::sleep(Duration::from_millis(5));
    }
    library.release_writes();
    let released = Instant::now();
    let mut after = Vec::new();
    while !before.contains(&PdEvent::Bootstrapped(fresh.into()))
        && !after.contains(&PdEvent::Bootstrapped(fresh.into()))
        && released.elapsed() < Duration::from_secs(2)
    {
        after.extend(pair.prefill.poll(&mut mesh, &group).unwrap());
        thread::sleep(Duration::from_millis(5));
    }
    println!("registration gate: during blocked native write (400ms): {before:?}; after release ({:.3}s): {after:?}", released.elapsed().as_secs_f64());
    extra.cancel(fresh);
    expect_cancelled(&mut extra, fresh, &mut mesh, &group);
    assert!(
        before.contains(&PdEvent::Bootstrapped(fresh.into())),
        "a write to D1 blocked P control handling of fresh D2 registration"
    );
}

#[test]
fn paired_cancel_before_admission_allows_a_later_healthy_request() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_millis(800),
    );
    let (mut mesh, group) = single_rank();
    let old = "upstream-cancel-after-p-timeout";
    pair.prefill.register(old, &pair.bootstrap).unwrap();
    pair.decode.register(old, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(old.into())
    );
    expect_cancelled(&mut pair.prefill, old, &mut mesh, &group);
    // The action current Dynamo takes when it consumes P's terminal error.
    pair.decode.cancel(old);
    expect_cancelled(&mut pair.decode, old, &mut mesh, &group);
    pair.admit(old);
    assert!(!pair.decode.has_in_flight());
    transfer_one(&mut pair, "healthy-followup", 9898, &mut mesh, &group);
    assert!(landed(&destination, &source));
    assert!(!pair.prefill.has_in_flight());
    assert!(!pair.decode.has_in_flight());
    println!("recovery: paired D cancel before admission suppressed the old manifest; same peers transferred a new room successfully");
}

#[test]
fn healthy_restarted_prefill_serves_a_surviving_decode() {
    let fixture = native::NativeLibrary::new();
    let library = std::env::var_os("TS_MOONCAKE_LIBRARY")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| fixture.path().to_owned());
    let source = pattern();
    let mut old_dest = vec![0; BYTES];
    let fresh_dest = vec![0; BYTES];
    let ttl = Duration::from_millis(1200);
    let (mut mesh, group) = single_rank();
    let mut old = pair(library.clone(), &source, &old_dest, ttl);
    transfer_one(&mut old, "before-restart", 101, &mut mesh, &group);
    assert!(landed(&old_dest, &source));
    let bootstrap = old.bootstrap.port;
    let old_port = old.prefill.port();
    let old_decode = old.decode;
    let old_outbox = old.decode_outbox;
    drop(old.prefill);
    drop(old.prefill_outbox);
    drop(old._server);
    let mut new = pair_on_port(
        library.clone(),
        &source,
        &fresh_dest,
        ttl,
        None,
        bootstrap,
        "early",
    );
    let new_port = new.prefill.port();
    assert_ne!(old_port, new_port);
    let url = format!("http://127.0.0.1:{bootstrap}");
    assert_eq!(
        ureq::get(&format!("{url}/health")).call().unwrap().status(),
        200
    );
    let rank: serde_json::Value =
        ureq::get(&format!("{url}/route?engine_rank=0&target_dp_group=0"))
            .call()
            .unwrap()
            .into_json()
            .unwrap();
    assert_eq!(rank["rank_port"].as_u64(), Some(u64::from(new_port)));

    // An independent D transfers through the restarted P at this same address.
    transfer_one(&mut new, "fresh-decode", 102, &mut mesh, &group);
    assert!(landed(&fresh_dest, &source));
    new.decode = old_decode;
    new.decode_outbox = old_outbox;
    old_dest.fill(0);
    let rid = "surviving-decode";
    let info = BootstrapInfo {
        room: 103,
        ..new.bootstrap.clone()
    };
    new.prefill.register(rid, &info).unwrap();
    new.decode.register(rid, &info).unwrap();
    assert_eq!(
        next_event(&mut new.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.into())
    );
    new.admit(rid);
    assert_eq!(
        next_event(&mut new.prefill, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.into())
    );
    new.send(rid, 777, vec![]).unwrap();
    assert_eq!(
        next_event(&mut new.prefill, &mut mesh, &group),
        PdEvent::Succeeded(rid.into())
    );
    assert!(matches!(
        next_event(&mut new.decode, &mut mesh, &group),
        PdEvent::RemotePrefillDone(_)
    ));
    assert!(landed(&old_dest, &source));
}
