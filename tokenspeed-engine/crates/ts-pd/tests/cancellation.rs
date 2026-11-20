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
use ts_pd::{BootstrapInfo, PdEvent, TransferPeer};

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
    expect_cancelled(&mut pair.decode, "cancelled", &mut mesh, &group);
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
    expect_cancelled(&mut pair.decode, rid, &mut mesh, &group);
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
    expect_cancelled(&mut pair.prefill, "late-claim", &mut mesh, &group);
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
    expect_cancelled(&mut pair.decode, rid, &mut mesh, &group);
    expect_cancelled(&mut pair.prefill, rid, &mut mesh, &group);
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
    let started = Instant::now();
    let deadline = started + Duration::from_secs(5);
    let mut decode_events = Vec::new();
    let mut prefill_events = Vec::new();
    let expected_failure = [PdEvent::Failed(rid.to_owned())];
    loop {
        decode_events.extend(pair.decode.poll(&mut mesh, &group).unwrap());
        prefill_events.extend(pair.prefill.poll(&mut mesh, &group).unwrap());
        assert!(
            expected_failure.starts_with(&decode_events),
            "stopped before write retirement: {decode_events:?}"
        );
        assert!(
            expected_failure.starts_with(&prefill_events),
            "stopped before write retirement: {prefill_events:?}"
        );
        assert!(pair.decode.has_in_flight() && pair.prefill.has_in_flight());
        if decode_events == expected_failure
            && prefill_events == expected_failure
            && started.elapsed() >= Duration::from_millis(150)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "failure was not reported while the write was blocked"
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(decode_events, expected_failure);
    assert_eq!(prefill_events, expected_failure);
    assert!(pair.decode_outbox.release_registration().is_err());
    assert!(pair.prefill_outbox.release_registration().is_err());
    library.release_writes();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Stopped(rid.to_owned())
    );
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Stopped(rid.to_owned())
    );
    assert!(!pair.decode.has_in_flight() && !pair.prefill.has_in_flight());
    destination.fill(0);
    thread::sleep(Duration::from_millis(100));
    assert!(
        destination.iter().all(|&byte| byte == 0),
        "a retired write touched recycled pages"
    );
    transfer_one(&mut pair, "next", 4243, &mut mesh, &group);
    assert!(landed(&destination, &source));
}

#[test]
fn a_cancellation_unacknowledged_past_the_receive_budget_is_retired_by_revoking_the_transport() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let mut destination = vec![0u8; BYTES];
    let budget = Duration::from_secs(3);
    let mut pair = pair_with_waiting_timeout(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
        budget,
    );
    let (mut mesh, group) = single_rank();
    let rid = "writing";
    prepare(&mut pair, rid, &mut mesh, &group);
    library.block_writes();
    pair.send(rid, 7, Vec::new()).unwrap();
    assert!(library.wait_write_entered(Duration::from_secs(5)));
    // The write is executing, so Prefill withholds its acknowledgment, as
    // a dead Prefill rank would.
    pair.decode.cancel(rid);
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    let failed = Instant::now();
    // Another request is connected, within its own budget, but not admitted
    // yet when the transport is replaced.
    thread::sleep(budget / 2);
    let queued = BootstrapInfo {
        room: 4343,
        ..pair.bootstrap.clone()
    };
    pair.decode.register("queued", &queued).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped("queued".to_owned())
    );
    let cache_id = pair.decode.cache_id();
    thread::sleep((failed + budget).saturating_duration_since(Instant::now()));
    // Past the budget, a poll starts replacing the transfer engine.
    assert!(pair.decode.poll(&mut mesh, &group).unwrap().is_empty());
    let deadline = Instant::now() + Duration::from_secs(10);
    while pair.decode.cache_id() == cache_id {
        assert!(Instant::now() < deadline, "the transport was not replaced");
        thread::sleep(Duration::from_millis(1));
    }
    let fresh = pair.decode.cache_id();
    // The queued request failed with the replaced transport; admitting it
    // now must not publish a manifest no Prefill rank could acknowledge
    // (which would take another revocation to retire).
    pair.admit("queued");
    let expected = [
        PdEvent::Stopped(rid.to_owned()),
        PdEvent::Failed("queued".to_owned()),
        PdEvent::Stopped("queued".to_owned()),
    ];
    let mut events = Vec::new();
    while !expected.iter().all(|event| events.contains(event)) {
        events.extend(pair.decode.poll(&mut mesh, &group).unwrap());
        assert!(Instant::now() < deadline, "not retired: {events:?}");
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(events.len(), expected.len(), "{events:?}");
    assert_eq!(
        pair.decode.cache_id(),
        fresh,
        "the transport was replaced again"
    );
    assert!(!pair.decode.has_in_flight());
    // A sleep and wake replace the engine again while the revoked write is
    // still blocked: its old session must not resolve to the new engine.
    pair.decode_outbox.release_registration().unwrap();
    pair.decode_outbox.restore_registration().unwrap();
    // The pages hold other data now; the writer, released, cannot reach them.
    destination.fill(0xa5);
    library.release_writes();
    expect_cancelled(&mut pair.prefill, rid, &mut mesh, &group);
    thread::sleep(Duration::from_millis(100));
    assert!(
        destination.iter().all(|&byte| byte == 0xa5),
        "a revoked write reached recycled pages"
    );
    // The same Prefill rank serves the next request through the fresh engine.
    destination.fill(0);
    transfer_one(&mut pair, "next", 4444, &mut mesh, &group);
    assert!(landed(&destination, &source));
}

#[test]
fn a_revocation_whose_deregistration_fails_keeps_waiting_for_acknowledgments() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let budget = Duration::from_secs(2);
    let mut pair = pair_with_waiting_timeout(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
        budget,
    );
    let (mut mesh, group) = single_rank();
    let rid = "writing";
    prepare(&mut pair, rid, &mut mesh, &group);
    library.block_writes();
    pair.send(rid, 7, Vec::new()).unwrap();
    assert!(library.wait_write_entered(Duration::from_secs(5)));
    pair.decode.cancel(rid);
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    let cache_id = pair.decode.cache_id();
    library.fail_unregistrations();
    thread::sleep(budget);
    assert!(pair.decode.poll(&mut mesh, &group).unwrap().is_empty());
    let deadline = Instant::now() + Duration::from_secs(10);
    while pair.decode.cache_id() == cache_id {
        assert!(Instant::now() < deadline, "the transport was not replaced");
        thread::sleep(Duration::from_millis(1));
    }
    // The replaced engine kept the cache registered, so its writer is not
    // fenced: the receive keeps waiting, and starts no other revocation.
    let fresh = pair.decode.cache_id();
    let waited = Instant::now() + 2 * budget;
    while Instant::now() < waited {
        assert!(pair.decode.poll(&mut mesh, &group).unwrap().is_empty());
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(pair.decode.cache_id(), fresh);
    assert!(pair.decode.has_in_flight());
    // The write lands in still-pinned pages, then Prefill acknowledges.
    library.release_writes();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Stopped(rid.to_owned())
    );
    // The cache stays registered with the replaced engine: never released.
    let refused = pair.decode_outbox.release_registration().unwrap_err();
    assert!(refused.contains("kept it registered"), "{refused}");
}

fn transfer_counter(directory: &std::path::Path, name: &str) -> f64 {
    let result = std::process::Command::new("python3")
        .env("PROMETHEUS_MULTIPROC_DIR", directory)
        .arg("-c")
        .arg("import sys; from prometheus_client import CollectorRegistry, multiprocess; r=CollectorRegistry(); multiprocess.MultiProcessCollector(r); print(sum(s.value for m in r.collect() for s in m.samples if s.name == sys.argv[1]))")
        .arg(name)
        .output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

fn room_end_metrics(name: &str, cancel_decode: Option<bool>) {
    let directory =
        std::env::temp_dir().join(format!("ts-pd-metrics-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let exporter = ts_metrics::Exporter::new(&ts_metrics::MetricsConfig {
        prometheus_dir: Some(directory.clone()),
        relay_endpoint: None,
    })
    .unwrap();
    let metrics = std::sync::Arc::new(ts_metrics::KvTransferMetrics::new(&exporter, "m", ""));
    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0u8; BYTES];
    let mut pair = pair_on_port_with_metrics(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
        None,
        0,
        "early",
        Some(metrics),
    );
    let (mut mesh, group) = single_rank();
    prepare(&mut pair, "r", &mut mesh, &group);
    match cancel_decode {
        Some(true) => pair.decode.cancel("r"),
        Some(false) => pair.prefill.cancel("r"),
        None => {
            pair.prefill.abort_room("r", pair.bootstrap.room);
            pair.prefill.abort_room("r", pair.bootstrap.room);
        }
    }
    expect_cancelled(&mut pair.prefill, "r", &mut mesh, &group);
    expect_cancelled(&mut pair.decode, "r", &mut mesh, &group);
    assert_eq!(
        transfer_counter(&directory, "tokenspeed:kv_transfer_failures_total"),
        if cancel_decode.is_some() { 0.0 } else { 1.0 }
    );
}

#[test]
fn explicit_prefill_cancel_does_not_increment_transfer_failures() {
    room_end_metrics("prefill-cancel", Some(false));
}

#[test]
fn remote_decode_cancel_does_not_increment_transfer_failures() {
    room_end_metrics("decode-cancel", Some(true));
}

#[test]
fn a_failed_room_increments_transfer_failures_once() {
    room_end_metrics("failure", None);
}

fn replica_end_metrics(cause: &str) {
    let library = native::NativeLibrary::new();
    let listeners: Vec<_> = (0..2)
        .map(|_| ts_collective::Listener::bind("127.0.0.1").unwrap())
        .collect();
    let addresses: Vec<_> = listeners
        .iter()
        .map(|l| format!("127.0.0.1:{}", l.port().unwrap()))
        .collect();
    let results = thread::scope(|scope| {
        let workers: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addresses = &addresses;
                let library = library.path().to_path_buf();
                scope.spawn(move || {
                    let directory = std::env::temp_dir().join(format!(
                        "ts-pd-replica-{cause}-{rank}-{}",
                        std::process::id()
                    ));
                    std::fs::create_dir_all(&directory).unwrap();
                    let exporter = ts_metrics::Exporter::new(&ts_metrics::MetricsConfig {
                        prometheus_dir: Some(directory.clone()),
                        relay_endpoint: None,
                    })
                    .unwrap();
                    let metrics =
                        std::sync::Arc::new(ts_metrics::KvTransferMetrics::new(&exporter, "m", ""));
                    let source = pattern();
                    let destination = vec![0u8; BYTES];
                    let timeout = if rank == 0 && cause == "timeout" {
                        Duration::ZERO
                    } else {
                        Duration::from_secs(120)
                    };
                    let mut pair = pair_on_port_with_metrics(
                        library,
                        &source,
                        &destination,
                        timeout,
                        None,
                        0,
                        "early",
                        Some(metrics),
                    );
                    pair.prefill.register("r", &pair.bootstrap).unwrap();
                    // Only one rank has observed a terminal event when the real
                    // collective runs; the other still waits for its manifest.
                    if rank == 0 {
                        match cause {
                            "cancel" => pair.prefill.cancel("r"),
                            "failure" => pair.prefill.abort_room("r", pair.bootstrap.room),
                            "timeout" => {}
                            _ => unreachable!(),
                        }
                    }
                    let mut mesh = ts_collective::Mesh::connect(
                        rank,
                        listener,
                        addresses,
                        Duration::from_secs(5),
                    )
                    .unwrap();
                    let replica = ts_collective::Group::new(0, vec![0, 1], rank).unwrap();
                    let events = pair.prefill.poll(&mut mesh, &replica).unwrap();
                    assert_eq!(
                        events,
                        [PdEvent::Failed("r".into()), PdEvent::Stopped("r".into())]
                    );
                    assert!(pair.prefill.poll(&mut mesh, &replica).unwrap().is_empty());
                    (
                        transfer_counter(&directory, "tokenspeed:kv_transfer_failures_total"),
                        transfer_counter(&directory, "tokenspeed:kv_transfer_timeouts_total"),
                    )
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    let expected = match cause {
        "cancel" => vec![(0.0, 0.0), (0.0, 0.0)],
        "timeout" => vec![(0.0, 1.0), (0.0, 0.0)],
        "failure" => vec![(1.0, 0.0), (0.0, 0.0)],
        _ => unreachable!(),
    };
    assert_eq!(results, expected);
}

#[test]
fn replica_cancel_propagation_does_not_invent_a_transfer_failure() {
    replica_end_metrics("cancel");
}

#[test]
fn replica_timeout_propagation_keeps_timeout_separate_from_failure() {
    replica_end_metrics("timeout");
}

#[test]
fn replica_failure_propagation_counts_only_the_rank_that_detected_it() {
    replica_end_metrics("failure");
}
