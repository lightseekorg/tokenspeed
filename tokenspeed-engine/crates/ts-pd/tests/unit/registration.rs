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

//! Cache identity retirement through the real registration and request handlers.

use super::*;
use crate::tests_support::NativeLibrary;

fn layout(base: Option<u64>) -> String {
    let mut value: serde_json::Value =
        serde_json::from_str(&crate::layout::contract::tests::wire()).unwrap();
    if let Some(base) = base {
        value["field_addresses"] = serde_json::json!({
            "layer.0.kv": base, "layer.1.state": base + 16,
        });
    }
    value.to_string()
}

fn shared(native: &NativeLibrary) -> Shared {
    let (incoming, _received) = mpsc::channel();
    let transport = Transport::start(
        &PeerConfig {
            library: native.path().to_path_buf(),
            local_ip: "127.0.0.1".to_owned(),
            protocol: "rdma".to_owned(),
            ib_device: String::new(),
            regions: vec![],
            engine_rank: 0,
            metrics: None,
        },
        incoming,
    )
    .unwrap();
    Shared {
        transport,
        layout: PrefillLayout {
            local: Contract::from_wire(layout(Some(0x1000)).as_bytes()).unwrap(),
            wire: Contract::from_wire(layout(None).as_bytes()).unwrap(),
            arena_base: 0x1000,
            tp_size: 1,
            tp_rank: 0,
            pp_rank: 0,
            fields_by_stage: vec![vec!["layer.0.kv".to_owned(), "layer.1.state".to_owned()]],
        },
        state: Mutex::new(State::default()),
        status_rank: 0,
        failed_session_ttl: Duration::from_secs(30),
        metrics: None,
        layerwise: None,
        deferred: false,
        interval_fields: vec![],
        metadata_ready: Condvar::new(),
    }
}

fn registration(cache: &str, session: &str, port: u16) -> Vec<Bytes> {
    wire::registration_frames(
        "127.0.0.1",
        port,
        session,
        cache,
        0x2000,
        layout(Some(0x2000)).as_bytes(),
        1,
        0,
    )
}

fn register(peer: &Shared, cache: &str, session: &str, port: u16) {
    let frames = registration(cache, session, port);
    peer.plan_registration(&frames)
        .expect("individually valid registration");
    peer.handle_registration(&frames);
}

fn preallocate(peer: &Shared, cache: &str, room: u64) -> Result<(), String> {
    let manifest = serde_json::json!({"groups": [
        {"group_id": "history", "block_ids": [1]},
        {"group_id": "state", "block_ids": [2]}],
        "prefix_len": 0, "prompt_len": 2, "state_prefix_cached": false})
    .to_string();
    peer.accept_preallocation(
        room,
        cache,
        &wire::preallocation_frames(room, cache, Some(manifest.as_bytes())),
    )
}

#[test]
fn identical_registration_remains_idempotent() {
    let native = NativeLibrary::new();
    let peer = shared(&native);
    register(&peer, "cache-a", "127.0.0.1:41001", 42001);
    register(&peer, "cache-a", "127.0.0.1:41001", 42001);
    assert_eq!(preallocate(&peer, "cache-a", 1), Ok(()));
}

#[test]
fn a_rebuilt_decode_can_reuse_an_idle_transport_address() {
    let native = NativeLibrary::new();
    let peer = shared(&native);
    register(&peer, "cache-a", "127.0.0.1:41001", 42001);
    assert_eq!(preallocate(&peer, "cache-a", 1), Ok(()));
    // The request ended; its old cache registration remains until replacement.
    peer.state().clear(1);
    register(&peer, "cache-b", "127.0.0.1:41001", 42002);
    assert_eq!(preallocate(&peer, "cache-b", 2), Ok(()));
    assert_eq!(peer.state().registrations.len(), 1);
    assert!(preallocate(&peer, "cache-a", 3)
        .unwrap_err()
        .contains("unregistered"));
    // Neither a delayed old registration nor pre-allocation can evict cache-b.
    register(&peer, "cache-a", "127.0.0.1:41001", 42001);
    assert_eq!(preallocate(&peer, "cache-b", 4), Ok(()));
    assert_eq!(peer.state().registrations.len(), 1);
}

#[test]
fn waking_at_a_new_address_retires_the_old_cache_at_the_same_control_socket() {
    let native = NativeLibrary::new();
    let peer = shared(&native);
    register(&peer, "cache-a", "127.0.0.1:41001", 42001);
    assert_eq!(preallocate(&peer, "cache-a", 1), Ok(()));
    let planned = peer
        .plan_registration(&registration("cache-a", "127.0.0.1:41001", 42001))
        .unwrap();
    register(&peer, "cache-b", "127.0.0.1:41002", 42001);
    assert_eq!(peer.state().rooms.get(1), Some(TransferPoll::Failed));
    assert_eq!(peer.state().registrations.len(), 1);
    assert_eq!(preallocate(&peer, "cache-b", 2), Ok(()));
    // A worker that planned its copies before retirement must not execute them.
    assert!(peer
        .write_copies(1, &planned, &[])
        .unwrap_err()
        .contains("retired"));
}

#[test]
fn conflicting_metadata_for_the_same_cache_identity_is_still_rejected() {
    let native = NativeLibrary::new();
    let peer = shared(&native);
    register(&peer, "cache-a", "127.0.0.1:41001", 42001);
    assert_eq!(preallocate(&peer, "cache-a", 1), Ok(()));
    register(&peer, "cache-a", "127.0.0.1:41001", 42002);
    assert!(peer
        .state()
        .rooms
        .failure(1)
        .unwrap()
        .contains("conflicting registration"));
    assert!(preallocate(&peer, "cache-a", 2)
        .unwrap_err()
        .contains("unregistered"));
}

#[test]
fn cancellation_survives_the_unclaimed_ttl_until_its_cache_is_retired() {
    let native = NativeLibrary::new();
    let peer = shared(&native);
    register(&peer, "cache-a", "127.0.0.1:41001", 42001);
    assert_eq!(preallocate(&peer, "cache-a", 1), Ok(()));
    peer.handle_cancellation(&wire::cancellation_frames(1, "cache-a"));
    peer.drop_unclaimed(Duration::ZERO);
    assert_eq!(peer.state().rooms.get(1), None);
    assert_eq!(preallocate(&peer, "cache-a", 1), Ok(()));
    assert!(!peer.state().destinations.contains_key(&1));
    assert!(peer.state().cancelled.contains_key(&1));
    register(&peer, "cache-b", "127.0.0.1:41001", 42001);
    assert!(peer.state().cancelled.is_empty());
    assert!(preallocate(&peer, "cache-a", 1).is_err());
    assert_eq!(preallocate(&peer, "cache-b", 2), Ok(()));
}

#[test]
fn a_cancel_before_any_manifest_binds_to_a_late_cache_identity() {
    let native = NativeLibrary::new();
    let peer = shared(&native);
    peer.state().cancel_room(1, None);
    peer.drop_unclaimed(Duration::ZERO);
    register(&peer, "cache-a", "127.0.0.1:41001", 42001);
    assert_eq!(preallocate(&peer, "cache-a", 1), Ok(()));
    assert!(!peer.state().destinations.contains_key(&1));
    register(&peer, "cache-b", "127.0.0.1:41001", 42001);
    assert!(peer.state().cancelled.is_empty());
}

#[test]
fn a_cancelled_room_stays_closed_until_every_decode_cache_retires() {
    let native = NativeLibrary::new();
    let peer = shared(&native);
    let register_rank = |cache, session, port, rank| {
        let mut frames = registration(cache, session, port);
        frames[6] = Bytes::from("2");
        frames[7] = Bytes::from(format!("{rank}"));
        peer.plan_registration(&frames)
            .expect("valid TP registration");
        peer.handle_registration(&frames);
    };
    register_rank("cache-a", "127.0.0.1:41001", 42001, 0);
    register_rank("cache-b", "127.0.0.1:41002", 42002, 1);
    assert_eq!(preallocate(&peer, "cache-a", 1), Ok(()));
    assert_eq!(preallocate(&peer, "cache-b", 1), Ok(()));
    peer.handle_cancellation(&wire::cancellation_frames(1, "cache-a"));
    peer.drop_unclaimed(Duration::ZERO);
    register_rank("next-a", "127.0.0.1:41001", 42001, 0);
    assert!(peer.state().cancelled.contains_key(&1));
    assert_eq!(preallocate(&peer, "cache-b", 1), Ok(()));
    assert!(!peer.state().destinations.contains_key(&1));
    register_rank("next-b", "127.0.0.1:41002", 42002, 1);
    assert!(peer.state().cancelled.is_empty());
    assert!(preallocate(&peer, "cache-a", 1).is_err());
    assert!(preallocate(&peer, "cache-b", 1).is_err());
}

#[test]
fn replacement_waits_for_its_writer_without_blocking_other_registrations() {
    let native = NativeLibrary::new();
    let peer = Arc::new(shared(&native));
    let source = Arc::new(vec![73u8; 32]);
    let destination = Arc::new(vec![0u8; 32]);
    let receiver =
        ts_mooncake::Engine::start(peer.transport.library.clone(), "127.0.0.1", "rdma", "")
            .unwrap();
    peer.transport
        .engine()
        .register(source.as_ptr() as usize, source.len())
        .unwrap();
    receiver
        .register(destination.as_ptr() as usize, destination.len())
        .unwrap();
    let session = receiver.session_id();
    register(&peer, "old-cache", session, 42001);
    preallocate(&peer, "old-cache", 1).unwrap();
    let old = peer
        .plan_registration(&registration("old-cache", session, 42001))
        .unwrap();
    native.block_writes();
    let writer = {
        let peer = peer.clone();
        let source = source.clone();
        let destination = destination.clone();
        let old = old.clone();
        thread::spawn(move || {
            peer.write_copies(
                1,
                &old,
                &[Copy::Spans {
                    sources: vec![source.as_ptr() as u64],
                    destinations: vec![destination.as_ptr() as u64],
                    lengths: vec![32],
                }],
            )
        })
    };
    assert!(native.wait_write_entered(Duration::from_secs(2)));
    let (incoming, received) = mpsc::channel();
    let replies = crate::net::Sockets::start("127.0.0.1", incoming).unwrap();
    register(&peer, "new-cache", session, replies.port());
    // The replacement cannot publish while the old write is inside native I/O.
    assert!(preallocate(&peer, "new-cache", 2).is_err());
    preallocate(&peer, "old-cache", 3).unwrap();
    register(&peer, "independent-cache", "127.0.0.1:41003", 42003);
    preallocate(&peer, "independent-cache", 4).unwrap();
    // Cancellation of a deferred cache is processed before activation. Its
    // queued manifest must not revive the cancelled room after replacement.
    peer.handle_cancellation(&wire::cancellation_frames(2, "new-cache"));
    let ack = received.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(
        wire::cancellation_ack(&ack).unwrap(),
        (2, "new-cache".to_owned(), 0)
    );
    let manifest = wire::preallocation_frames(2, "new-cache", Some(b"{}"));
    peer.handle_preallocation(&manifest);
    native.release_writes();
    writer.join().unwrap().unwrap();
    assert_eq!(*destination, *source);
    peer.flush_registrations();
    assert!(peer
        .write_copies(1, &old, &[])
        .unwrap_err()
        .contains("retired"));
    assert!(preallocate(&peer, "old-cache", 5).is_err());
    preallocate(&peer, "new-cache", 6).unwrap();
    assert!(!peer.state().destinations.contains_key(&2));
    preallocate(&peer, "independent-cache", 7).unwrap();
}

/// A Prefill rank under deferred decode allocation, driven as the event loop
/// and the control socket drive it; and a Decode rank's control socket with
/// what arrives there.
fn deferred_allocation_peer(
    native: &NativeLibrary,
    bootstrap_timeout: Duration,
) -> (PrefillPeer, PrefillOutbox, Mesh, Group) {
    let mut shared = shared(native);
    shared.deferred = true;
    // What it counts lands in the native library's directory (`timeouts`).
    let exporter = ts_metrics::Exporter::new(&ts_metrics::MetricsConfig {
        prometheus_dir: native.path().parent().map(std::path::Path::to_path_buf),
        relay_endpoint: None,
    })
    .unwrap();
    shared.metrics = Some(Arc::new(ts_metrics::KvTransferMetrics::new(
        &exporter, "m", "",
    )));
    let shared = Arc::new(shared);
    let listener = ts_collective::Listener::bind("127.0.0.1").unwrap();
    let address = format!("127.0.0.1:{}", listener.port().unwrap());
    let mesh = Mesh::connect(0, listener, &[address], Duration::from_secs(2)).unwrap();
    (
        PrefillPeer {
            shared: shared.clone(),
            stop: Arc::new(AtomicBool::new(false)),
            bootstrap_timeout,
            order: Vec::new(),
            senders: HashMap::new(),
            stall_logged_at: None,
            drain_logged_at: None,
            timeouts_logged: HashSet::new(),
        },
        PrefillOutbox {
            shared,
            queues: Arc::new(Vec::new()),
            staged: Staged::default(),
            armed_steps: Arc::new(Mutex::new(0)),
        },
        mesh,
        Group::new(0, vec![0], 0).unwrap(),
    )
}

fn decode_socket() -> (crate::net::Sockets, mpsc::Receiver<Vec<Bytes>>) {
    let (incoming, received) = mpsc::channel();
    (
        crate::net::Sockets::start("127.0.0.1", incoming).unwrap(),
        received,
    )
}

/// The next status a Decode rank's socket receives.
fn status(received: &mpsc::Receiver<Vec<Bytes>>) -> (u64, TransferPoll) {
    let frames = received
        .recv_timeout(Duration::from_secs(3))
        .expect("a status message");
    let message = StatusMessage::from_frames(&frames).unwrap();
    (message.room, message.status)
}

/// The answer to an announcement of `room` by `cache`. An identical answer
/// still on its way out stands for the new one, so the announcement is
/// repeated, as a heartbeat repeats it, until an answer arrives.
fn announce_until_answered(
    peer: &PrefillPeer,
    received: &mpsc::Receiver<Vec<Bytes>>,
    room: u64,
    cache: &str,
) -> (u64, TransferPoll) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        peer.shared.handle(&wire::announce_frames(room, cache));
        if let Ok(frames) = received.recv_timeout(Duration::from_millis(50)) {
            let message = StatusMessage::from_frames(&frames).unwrap();
            return (message.room, message.status);
        }
        assert!(Instant::now() < deadline, "the announcement got no answer");
    }
}

/// The transfer timeouts the metrics of `deferred_allocation_peer` counted, by
/// stage: the series of the Prometheus multiprocess file they write (a used
/// length, then per series its key's length, the key padded to 8 bytes, its
/// value and a timestamp).
fn timeouts(native: &NativeLibrary) -> Vec<(String, f64)> {
    let file = native
        .path()
        .with_file_name(format!("counter_{}.db", std::process::id()));
    let Ok(bytes) = std::fs::read(file) else {
        return Vec::new();
    };
    let used = i32::from_ne_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut stages = Vec::new();
    let mut at = 8;
    while at < used {
        let length = i32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let key: serde_json::Value =
            serde_json::from_slice(&bytes[at + 4..at + 4 + length]).unwrap();
        at += 4 + length + (8 - (length + 4) % 8);
        if let Some(stage) = key[2]["stage"].as_str() {
            let count = f64::from_ne_bytes(bytes[at..at + 8].try_into().unwrap());
            stages.push((stage.to_owned(), count));
        }
        at += 16;
    }
    stages.sort_by(|a, b| a.0.cmp(&b.0));
    stages
}

fn room(room: u64) -> BootstrapInfo {
    BootstrapInfo {
        host: "127.0.0.1".to_owned(),
        port: 1,
        room,
    }
}

#[test]
fn a_request_under_deferred_allocation_is_admitted_by_its_announcement_and_granted_by_its_manifest()
{
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) =
        deferred_allocation_peer(&native, Duration::from_secs(60));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());

    // Not announced: the request waits.
    peer.register("r", &room(7)).unwrap();
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    peer.shared.handle(&wire::announce_frames(7, "cache-a"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("r".to_owned())]
    );
    // The announcement is answered only once the prompt holds its pages,
    // and a manifest before that is refused: the room fails on both sides.
    assert!(received.recv_timeout(Duration::from_millis(100)).is_err());
    assert!(preallocate(&peer.shared, "cache-a", 7)
        .unwrap_err()
        .contains("does not hold its pages"));

    // The plan carrying the prompt's last chunk goes to the device; a
    // request this rank does not track is skipped.
    outbox.mark_pages_held(&["r".to_owned(), "unknown".to_owned()]);
    assert_eq!(status(&received), (7, TransferPoll::Bootstrapped));
    // Told once: the mark again sends nothing.
    outbox.mark_pages_held(&["r".to_owned()]);
    assert!(received.recv_timeout(Duration::from_millis(100)).is_err());
    // Every announcement is answered: a repeated one, again.
    assert_eq!(
        announce_until_answered(&peer, &received, 7, "cache-a"),
        (7, TransferPoll::Bootstrapped)
    );

    assert_eq!(preallocate(&peer.shared, "cache-a", 7), Ok(()));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::DestinationReady("r".to_owned())]
    );
}

#[test]
fn an_announcement_before_its_request_waits_for_it_and_expires_unclaimed() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        deferred_allocation_peer(&native, Duration::from_secs(60));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    peer.shared.handle(&wire::announce_frames(7, "cache-a"));
    peer.shared.handle(&wire::announce_frames(8, "cache-a"));
    // Room 7's request arrives: it is admitted at once.
    peer.register("r", &room(7)).unwrap();
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("r".to_owned())]
    );
    // Room 8's never does: its Decode rank is told, and the room stays
    // failed for a later announcement or manifest.
    peer.shared.drop_unclaimed(Duration::ZERO);
    assert_eq!(status(&received), (8, TransferPoll::Failed));
    assert_eq!(
        announce_until_answered(&peer, &received, 8, "cache-a"),
        (8, TransferPoll::Failed)
    );
    let state = peer.shared.state();
    assert!(!state.announced.contains_key(&8));
    assert!(state.announced.contains_key(&7));
}

#[test]
fn a_manifest_for_a_room_that_was_not_announced_fails_the_request() {
    // A Decode rank that does not know the order (an older build, a stale
    // route) pre-allocates at once.
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        deferred_allocation_peer(&native, Duration::from_secs(60));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    peer.register("r", &room(7)).unwrap();
    let manifest = serde_json::json!({"groups": [
        {"group_id": "history", "block_ids": [1]},
        {"group_id": "state", "block_ids": [2]}],
        "prefix_len": 0, "prompt_len": 2, "state_prefix_cached": false})
    .to_string();
    peer.shared.handle(&wire::preallocation_frames(
        7,
        "cache-a",
        Some(manifest.as_bytes()),
    ));
    assert_eq!(status(&received), (7, TransferPoll::Failed));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("r".to_owned()),
            PdEvent::Stopped("r".to_owned())
        ]
    );
}

#[test]
fn a_request_holding_its_pages_times_out_from_then_and_its_announcer_is_told() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) =
        deferred_allocation_peer(&native, Duration::from_millis(150));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    peer.register("r", &room(7)).unwrap();
    peer.shared.handle(&wire::announce_frames(7, "cache-a"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("r".to_owned())]
    );
    // While the prompt computes nothing times out here.
    thread::sleep(Duration::from_millis(300));
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    // From the moment it holds its pages its Decode node has the bootstrap
    // timeout to allocate, announced again or not.
    outbox.mark_pages_held(&["r".to_owned()]);
    assert_eq!(status(&received), (7, TransferPoll::Bootstrapped));
    thread::sleep(Duration::from_millis(300));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("r".to_owned()),
            PdEvent::Stopped("r".to_owned())
        ]
    );
    assert_eq!(status(&received), (7, TransferPoll::Failed));
    assert_eq!(timeouts(&native), [("bootstrapping".to_owned(), 1.0)]);
}

#[test]
fn an_early_rank_answers_an_announcement_at_once() {
    let native = NativeLibrary::new();
    let peer = shared(&native);
    let (decode, received) = decode_socket();
    register(&peer, "cache-a", "127.0.0.1:41001", decode.port());
    peer.handle(&wire::announce_frames(7, "cache-a"));
    assert_eq!(status(&received), (7, TransferPoll::Bootstrapped));
    assert!(peer.state().announced.is_empty());
    assert_eq!(preallocate(&peer, "cache-a", 7), Ok(()));
}

#[test]
fn an_announcement_of_a_room_that_ended_binds_its_tombstone_to_the_cache() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, _mesh, _group) =
        deferred_allocation_peer(&native, Duration::from_secs(60));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    // The request ends here before its Decode rank announced it: nothing
    // says yet which cache the room's tombstone may retire with.
    peer.register("r", &room(7)).unwrap();
    peer.cancel("r");
    assert!(peer.shared.state().cancelled[&7].is_empty());
    peer.shared.handle(&wire::announce_frames(7, "cache-a"));
    assert_eq!(status(&received), (7, TransferPoll::Failed));
    assert!(peer.shared.state().cancelled[&7].contains("cache-a"));
    // The cache is replaced at the same control socket: the tombstone goes.
    register(&peer.shared, "cache-b", "127.0.0.1:41002", decode.port());
    assert!(!peer.shared.state().cancelled.contains_key(&7));
}

#[test]
fn a_request_its_decode_node_never_announces_times_out_from_its_registration() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        deferred_allocation_peer(&native, Duration::from_millis(150));
    register(&peer.shared, "cache-a", "127.0.0.1:41001", 42001);
    peer.register("r", &room(7)).unwrap();
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    thread::sleep(Duration::from_millis(300));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("r".to_owned()),
            PdEvent::Stopped("r".to_owned())
        ]
    );
    assert_eq!(
        timeouts(&native),
        [("waiting_for_announcement".to_owned(), 1.0)]
    );
}

#[test]
fn a_room_is_admitted_once_every_rank_of_its_decode_node_announced_it() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        deferred_allocation_peer(&native, Duration::from_secs(60));
    // One Decode node of two attention-TP ranks: one destination.
    for (cache, session, port, rank) in [
        ("d-0", "127.0.0.1:41001", 42001, 0),
        ("d-1", "127.0.0.1:41002", 42002, 1),
    ] {
        let mut frames = registration(cache, session, port);
        frames[6] = Bytes::from("2");
        frames[7] = Bytes::from(format!("{rank}"));
        peer.shared
            .plan_registration(&frames)
            .expect("valid TP registration");
        peer.shared.handle_registration(&frames);
    }
    for (rid, number) in [("a", 1), ("b", 2), ("c", 3), ("d", 4)] {
        peer.register(rid, &room(number)).unwrap();
        peer.shared.handle(&wire::announce_frames(number, "d-0"));
        // The fourth room's second announcement is still on its way.
        if number != 4 {
            peer.shared.handle(&wire::announce_frames(number, "d-1"));
        }
    }
    // Every request the node announced is admitted at once, whatever the
    // node has not allocated yet.
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Bootstrapped("a".to_owned()),
            PdEvent::Bootstrapped("b".to_owned()),
            PdEvent::Bootstrapped("c".to_owned())
        ]
    );
    peer.shared.handle(&wire::announce_frames(4, "d-1"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("d".to_owned())]
    );
}

#[test]
fn a_request_waiting_for_its_announcement_is_reported_after_the_stall_interval() {
    use std::{io::Read, os::unix::net::UnixStream};
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        deferred_allocation_peer(&native, Duration::from_secs(60));
    let (decode, _received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    for (rid, number) in [("computing", 1), ("unannounced", 2)] {
        peer.register(rid, &room(number)).unwrap();
    }
    peer.shared.handle(&wire::announce_frames(1, "cache-a"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("computing".to_owned())]
    );
    // Both registered a stall interval ago. `computing` was admitted and
    // waits for nothing while it computes; `unannounced` is reported.
    for sender in peer.senders.values_mut() {
        sender.started -= BOOTSTRAP_STALL;
    }
    let (writer, mut reader) = UnixStream::pair().unwrap();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(move || writer.try_clone().unwrap())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    });
    let mut log = String::new();
    reader.read_to_string(&mut log).unwrap();
    assert!(
        log.contains("WARN")
            && log.contains("waiting for their decode instance's announcement")
            && log.contains("unannounced(room=2"),
        "{log}"
    );
    assert!(!log.contains("computing(room=1"), "{log}");
}
