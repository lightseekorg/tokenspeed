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
        independent: false,
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

/// A Prefill rank that allocates independently, driven as the event loop
/// and the control socket drive it; and a Decode rank's control socket with
/// what arrives there.
fn independent_peer(
    native: &NativeLibrary,
    bootstrap_timeout: Duration,
) -> (PrefillPeer, PrefillOutbox, Mesh, Group) {
    let mut shared = shared(native);
    shared.independent = true;
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

/// The transfer timeouts the metrics of `independent_peer` counted, by
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
fn an_independent_request_is_admitted_by_its_announcement_and_granted_by_its_manifest() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) = independent_peer(&native, Duration::from_secs(60));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());

    // Not announced: the request waits, whatever the window.
    peer.register("r", &room(7)).unwrap();
    peer.occupy("r", 4, 64);
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
    assert!(peer.shared.state().charged.is_empty());
}

#[test]
fn an_announcement_before_its_request_waits_for_it_and_expires_unclaimed() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) = independent_peer(&native, Duration::from_secs(60));
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
    let (mut peer, _outbox, mut mesh, group) = independent_peer(&native, Duration::from_secs(60));
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
fn a_destination_that_does_not_allocate_holds_back_only_its_own_requests() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) = independent_peer(&native, Duration::from_secs(60));
    let (slow, _slow_received) = decode_socket();
    let (healthy, _healthy_received) = decode_socket();
    register(&peer.shared, "slow", "127.0.0.1:41001", slow.port());
    register(&peer.shared, "healthy", "127.0.0.1:41002", healthy.port());
    // Two destinations share a pool of 64 blocks: 32 each, one 20-block
    // prompt outstanding at a time.
    for (rid, number, cache) in [
        ("slow-1", 1, "slow"),
        ("slow-2", 2, "slow"),
        ("slow-3", 3, "slow"),
        ("healthy-1", 4, "healthy"),
    ] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 20, 64);
        peer.shared.handle(&wire::announce_frames(number, cache));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Bootstrapped("slow-1".to_owned()),
            PdEvent::Bootstrapped("healthy-1".to_owned())
        ]
    );
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    // The slow destination allocates its first request: the next one in
    // arrival order takes the room.
    outbox.mark_pages_held(&["slow-1".to_owned()]);
    assert_eq!(preallocate(&peer.shared, "slow", 1), Ok(()));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::DestinationReady("slow-1".to_owned()),
            PdEvent::Bootstrapped("slow-2".to_owned())
        ]
    );
    // A request that ends gives its room back too.
    peer.cancel("slow-2");
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("slow-2".to_owned()),
            PdEvent::Stopped("slow-2".to_owned()),
            PdEvent::Bootstrapped("slow-3".to_owned())
        ]
    );
}

#[test]
fn a_request_holding_its_pages_times_out_from_then_and_its_announcer_is_told() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) = independent_peer(&native, Duration::from_millis(150));
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
fn registrations_without_a_request_take_no_share_of_the_pool() {
    // One live Decode rank, and the registrations of seven Decode processes
    // that are gone: nothing here replaces them, and they have no request.
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) = independent_peer(&native, Duration::from_secs(60));
    register(&peer.shared, "live", "127.0.0.1:41001", 42001);
    for gone in 0..7 {
        register(
            &peer.shared,
            &format!("gone-{gone}"),
            &format!("127.0.0.1:4110{gone}"),
            42100 + gone,
        );
    }
    // The live one has the whole 64-block pool: three 20-block prompts.
    for (rid, number) in [("a", 1), ("b", 2), ("c", 3), ("d", 4)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 20, 64);
        peer.shared.handle(&wire::announce_frames(number, "live"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Bootstrapped("a".to_owned()),
            PdEvent::Bootstrapped("b".to_owned()),
            PdEvent::Bootstrapped("c".to_owned())
        ]
    );
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
}

#[test]
fn an_announcement_of_a_room_that_ended_binds_its_tombstone_to_the_cache() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, _mesh, _group) = independent_peer(&native, Duration::from_secs(60));
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
        independent_peer(&native, Duration::from_millis(150));
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
fn a_request_behind_one_that_still_computes_does_not_time_out() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) = independent_peer(&native, Duration::from_millis(150));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    // One destination and a pool of 64 blocks: one 40-block prompt at a time.
    for (rid, number) in [("first", 1), ("second", 2)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 40, 64);
        peer.shared
            .handle(&wire::announce_frames(number, "cache-a"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("first".to_owned())]
    );
    // `first` computes for longer than the timeout. Its destination owes
    // nothing before it holds its pages, so `second` only waits.
    thread::sleep(Duration::from_millis(300));
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    outbox.mark_pages_held(&["first".to_owned()]);
    assert_eq!(status(&received), (1, TransferPoll::Bootstrapped));
    assert_eq!(preallocate(&peer.shared, "cache-a", 1), Ok(()));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::DestinationReady("first".to_owned()),
            PdEvent::Bootstrapped("second".to_owned())
        ]
    );
}

#[test]
fn a_destination_that_stops_allocating_fails_the_requests_it_holds_back_uncomputed() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) = independent_peer(&native, Duration::from_millis(800));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    for (rid, number) in [("first", 1), ("second", 2)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 40, 64);
        peer.shared
            .handle(&wire::announce_frames(number, "cache-a"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("first".to_owned())]
    );
    // `first` computes for half a timeout, then holds its pages; its Decode
    // node never allocates it. A request that arrives now waits behind it
    // like `second`.
    thread::sleep(Duration::from_millis(400));
    outbox.mark_pages_held(&["first".to_owned()]);
    assert_eq!(status(&received), (1, TransferPoll::Bootstrapped));
    peer.register("third", &room(3)).unwrap();
    peer.occupy("third", 40, 64);
    peer.shared.handle(&wire::announce_frames(3, "cache-a"));
    // A timeout after `second` registered, its destination has owed an
    // allocation for half of one: nothing fails yet.
    thread::sleep(Duration::from_millis(400));
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    // A timeout after `first` held its pages it fails, and with it the
    // requests the window held back, none of which was admitted.
    thread::sleep(Duration::from_millis(400));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("first".to_owned()),
            PdEvent::Stopped("first".to_owned()),
            PdEvent::Failed("second".to_owned()),
            PdEvent::Stopped("second".to_owned()),
            PdEvent::Failed("third".to_owned()),
            PdEvent::Stopped("third".to_owned()),
        ]
    );
    assert_eq!(
        timeouts(&native),
        [
            ("bootstrapping".to_owned(), 1.0),
            ("waiting_for_window".to_owned(), 2.0)
        ]
    );
    let mut told: Vec<_> = (0..3).map(|_| status(&received)).collect();
    told.sort_by_key(|(room, _)| *room);
    assert_eq!(
        told,
        [
            (1, TransferPoll::Failed),
            (2, TransferPoll::Failed),
            (3, TransferPoll::Failed)
        ]
    );
    // The window is empty again: the next request for this destination is
    // admitted.
    peer.register("fourth", &room(4)).unwrap();
    peer.occupy("fourth", 40, 64);
    peer.shared.handle(&wire::announce_frames(4, "cache-a"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("fourth".to_owned())]
    );
}

#[test]
fn a_destination_that_allocates_slowly_fails_none_of_its_requests() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) = independent_peer(&native, Duration::from_millis(600));
    let (decode, received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    for (rid, number) in [("a", 1), ("b", 2), ("c", 3), ("d", 4)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 40, 64);
        peer.shared
            .handle(&wire::announce_frames(number, "cache-a"));
    }
    // One 40-block prompt at a time fits the 64-block pool. Each holds its
    // pages at once and is allocated 0.6 timeouts later, so the last one
    // waits 1.8 timeouts for its turn, never one whole timeout for a
    // single allocation.
    let mut expected = vec![PdEvent::Bootstrapped("a".to_owned())];
    for (rid, number, next) in [
        ("a", 1, Some("b")),
        ("b", 2, Some("c")),
        ("c", 3, Some("d")),
        ("d", 4, None),
    ] {
        assert_eq!(peer.poll(&mut mesh, &group).unwrap(), expected);
        outbox.mark_pages_held(&[rid.to_owned()]);
        assert_eq!(status(&received), (number, TransferPoll::Bootstrapped));
        thread::sleep(Duration::from_millis(180));
        assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
        thread::sleep(Duration::from_millis(180));
        assert_eq!(preallocate(&peer.shared, "cache-a", number), Ok(()));
        expected = vec![PdEvent::DestinationReady(rid.to_owned())];
        expected.extend(next.map(|next| PdEvent::Bootstrapped(next.to_owned())));
    }
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), expected);
}

#[test]
fn a_destination_keeps_its_share_for_a_timeout_after_its_last_request() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        independent_peer(&native, Duration::from_millis(500));
    let (slow, _slow_received) = decode_socket();
    let (healthy, _healthy_received) = decode_socket();
    register(&peer.shared, "slow", "127.0.0.1:41001", slow.port());
    register(&peer.shared, "healthy", "127.0.0.1:41002", healthy.port());
    // The healthy destination's only request ends.
    peer.register("healthy-0", &room(9)).unwrap();
    peer.occupy("healthy-0", 20, 64);
    peer.shared.handle(&wire::announce_frames(9, "healthy"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("healthy-0".to_owned())]
    );
    peer.cancel("healthy-0");
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("healthy-0".to_owned()),
            PdEvent::Stopped("healthy-0".to_owned())
        ]
    );
    // The slow destination, alone for the moment, is still held to its half
    // of the 64-block pool: one 20-block prompt outstanding.
    for (rid, number) in [("slow-1", 1), ("slow-2", 2), ("slow-3", 3)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 20, 64);
        peer.shared.handle(&wire::announce_frames(number, "slow"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("slow-1".to_owned())]
    );
    // So the healthy destination's next request fits beside it.
    peer.register("healthy-1", &room(4)).unwrap();
    peer.occupy("healthy-1", 30, 64);
    peer.shared.handle(&wire::announce_frames(4, "healthy"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("healthy-1".to_owned())]
    );
    let charged = |peer: &PrefillPeer| -> i64 {
        let state = peer.shared.state();
        state.charged.values().map(|(_, blocks)| blocks).sum()
    };
    assert_eq!(charged(&peer), 50);
    // The healthy destination has no request for a timeout: its share
    // lapses and the slow one has the whole pool.
    peer.cancel("healthy-1");
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("healthy-1".to_owned()),
            PdEvent::Stopped("healthy-1".to_owned())
        ]
    );
    thread::sleep(Duration::from_millis(600));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Bootstrapped("slow-2".to_owned()),
            PdEvent::Bootstrapped("slow-3".to_owned())
        ]
    );
    assert_eq!(charged(&peer), 60);
}

#[test]
fn a_room_part_of_its_decode_node_announced_takes_no_share_of_the_pool() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) = independent_peer(&native, Duration::from_secs(60));
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
        peer.occupy(rid, 20, 64);
        peer.shared.handle(&wire::announce_frames(number, "d-0"));
        // The fourth room's second announcement is still on its way.
        if number != 4 {
            peer.shared.handle(&wire::announce_frames(number, "d-1"));
        }
    }
    // The node has the whole 64-block pool: three 20-block prompts.
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Bootstrapped("a".to_owned()),
            PdEvent::Bootstrapped("b".to_owned()),
            PdEvent::Bootstrapped("c".to_owned())
        ]
    );
    peer.shared.handle(&wire::announce_frames(4, "d-1"));
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
}

#[test]
fn a_destination_that_allocates_its_other_requests_fails_only_the_one_it_owes() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) = independent_peer(&native, Duration::from_millis(800));
    let (decode, _received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    // A pool of 64 blocks and 30-block prompts: two outstanding at a time.
    for (rid, number) in [("big", 1), ("s1", 2)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 30, 64);
        peer.shared
            .handle(&wire::announce_frames(number, "cache-a"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Bootstrapped("big".to_owned()),
            PdEvent::Bootstrapped("s1".to_owned())
        ]
    );
    // Both hold their pages. The Decode node allocates `s1` at once and
    // never `big`; half a timeout later it allocates `s2` as promptly.
    outbox.mark_pages_held(&["big".to_owned(), "s1".to_owned()]);
    assert_eq!(preallocate(&peer.shared, "cache-a", 2), Ok(()));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::DestinationReady("s1".to_owned())]
    );
    thread::sleep(Duration::from_millis(500));
    peer.register("s2", &room(3)).unwrap();
    peer.occupy("s2", 30, 64);
    peer.shared.handle(&wire::announce_frames(3, "cache-a"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("s2".to_owned())]
    );
    outbox.mark_pages_held(&["s2".to_owned()]);
    assert_eq!(preallocate(&peer.shared, "cache-a", 3), Ok(()));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::DestinationReady("s2".to_owned())]
    );
    // `s3` takes the room `s2` left and computes; `late` waits behind it.
    for (rid, number) in [("s3", 4), ("late", 5)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 30, 64);
        peer.shared
            .handle(&wire::announce_frames(number, "cache-a"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("s3".to_owned())]
    );
    // A timeout after `big` held its pages it fails, alone: its destination
    // allocated another request half a timeout ago.
    thread::sleep(Duration::from_millis(400));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("big".to_owned()),
            PdEvent::Stopped("big".to_owned())
        ]
    );
    assert_eq!(timeouts(&native), [("bootstrapping".to_owned(), 1.0)]);
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("late".to_owned())]
    );
}

#[test]
fn a_decode_rank_that_restarted_leaves_no_share_behind() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) = independent_peer(&native, Duration::from_secs(60));
    let (decode, _received) = decode_socket();
    // The rank registers three caches in turn at its control socket, each
    // retiring the one before, and each had a request admitted.
    for (generation, cache) in ["gen-0", "gen-1", "gen-2"].into_iter().enumerate() {
        register(
            &peer.shared,
            cache,
            &format!("127.0.0.1:4100{generation}"),
            decode.port(),
        );
        let rid = format!("warm-{generation}");
        let number = 10 + generation as u64;
        peer.register(&rid, &room(number)).unwrap();
        peer.occupy(&rid, 20, 64);
        peer.shared.handle(&wire::announce_frames(number, cache));
        assert!(peer
            .poll(&mut mesh, &group)
            .unwrap()
            .contains(&PdEvent::Bootstrapped(rid)));
    }
    peer.cancel("warm-2");
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("warm-2".to_owned()),
            PdEvent::Stopped("warm-2".to_owned())
        ]
    );
    // Only the live cache is a destination: it has the whole 64-block pool,
    // three 20-block prompts.
    for (rid, number) in [("a", 1), ("b", 2), ("c", 3)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 20, 64);
        peer.shared.handle(&wire::announce_frames(number, "gen-2"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Bootstrapped("a".to_owned()),
            PdEvent::Bootstrapped("b".to_owned()),
            PdEvent::Bootstrapped("c".to_owned())
        ]
    );
}

#[test]
fn a_request_behind_prompts_this_rank_still_computes_is_reported_after_the_stall_interval() {
    use std::{io::Read, os::unix::net::UnixStream};
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) = independent_peer(&native, Duration::from_secs(60));
    let (decode, _received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    for (rid, number) in [("ahead", 1), ("behind", 2)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 40, 64);
        peer.shared
            .handle(&wire::announce_frames(number, "cache-a"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("ahead".to_owned())]
    );
    // Both registered a stall interval ago; `ahead` still computes, so
    // nothing times `behind` out. It is reported, as nothing that points at
    // a fault; `ahead` is not.
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
        log.contains("INFO")
            && log.contains("behind prompts this rank is still computing")
            && log.contains("behind(room=2"),
        "{log}"
    );
    assert!(
        !log.contains("WARN") && !log.contains("ahead(room=1"),
        "{log}"
    );
}

#[test]
fn a_destination_that_allocated_long_ago_is_timed_from_the_moment_it_owes() {
    let native = NativeLibrary::new();
    let (mut peer, outbox, mut mesh, group) = independent_peer(&native, Duration::from_millis(800));
    let (decode, _received) = decode_socket();
    register(&peer.shared, "cache-a", "127.0.0.1:41001", decode.port());
    // The destination allocates a request, then owes nothing for most of a
    // timeout.
    peer.register("first", &room(1)).unwrap();
    peer.occupy("first", 40, 64);
    peer.shared.handle(&wire::announce_frames(1, "cache-a"));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("first".to_owned())]
    );
    outbox.mark_pages_held(&["first".to_owned()]);
    assert_eq!(preallocate(&peer.shared, "cache-a", 1), Ok(()));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::DestinationReady("first".to_owned())]
    );
    thread::sleep(Duration::from_millis(600));
    // One 40-block prompt at a time fits the 64-block pool: `owed` holds
    // its pages and `behind` waits for the window.
    for (rid, number) in [("owed", 2), ("behind", 3)] {
        peer.register(rid, &room(number)).unwrap();
        peer.occupy(rid, 40, 64);
        peer.shared
            .handle(&wire::announce_frames(number, "cache-a"));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [PdEvent::Bootstrapped("owed".to_owned())]
    );
    outbox.mark_pages_held(&["owed".to_owned()]);
    // Half a timeout after `owed` held its pages nothing fails, although
    // the destination's last allocation is more than a timeout old.
    thread::sleep(Duration::from_millis(400));
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    // A timeout after it held them both fail.
    thread::sleep(Duration::from_millis(500));
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        [
            PdEvent::Failed("owed".to_owned()),
            PdEvent::Stopped("owed".to_owned()),
            PdEvent::Failed("behind".to_owned()),
            PdEvent::Stopped("behind".to_owned())
        ]
    );
    assert_eq!(
        timeouts(&native),
        [
            ("bootstrapping".to_owned(), 1.0),
            ("waiting_for_window".to_owned(), 1.0)
        ]
    );
}
