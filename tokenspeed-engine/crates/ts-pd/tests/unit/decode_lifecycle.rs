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

//! Connect worker retirement through real peers, bootstrap HTTP and control sockets.

use super::*;
use crate::{net::Sockets, tests_support::NativeLibrary, BootstrapServer, PeerConfig, Registry};
use ts_collective::Listener;

fn setup(
    native: &NativeLibrary,
    heartbeat_interval: Duration,
    waiting_timeout: Duration,
) -> (DecodePeer, DecodeOutbox, Mesh, Group) {
    let (peer, outbox) = DecodePeer::start(DecodeConfig {
        peer: PeerConfig {
            library: native.path().to_path_buf(),
            local_ip: "127.0.0.1".to_owned(),
            protocol: "rdma".to_owned(),
            ib_device: String::new(),
            regions: Vec::new(),
            engine_rank: 0,
            metrics: None,
        },
        arena_base: 0x1000,
        layout: crate::layout::contract::tests::wire().into_bytes(),
        tp_size: 1,
        tp_rank: 0,
        waiting_timeout,
        heartbeat_interval,
        heartbeat_max_failures: 2,
    })
    .unwrap();
    let listener = Listener::bind("127.0.0.1").unwrap();
    let address = format!("127.0.0.1:{}", listener.port().unwrap());
    let mesh = Mesh::connect(0, listener, &[address], Duration::from_secs(2)).unwrap();
    (peer, outbox, mesh, Group::new(0, vec![0], 0).unwrap())
}

fn bootstrap(port: u16, room: u64) -> BootstrapInfo {
    BootstrapInfo {
        host: "127.0.0.1".to_owned(),
        port,
        room,
    }
}

fn until(mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !ready() {
        assert!(Instant::now() < deadline, "connect worker did not settle");
        thread::sleep(Duration::from_millis(5));
    }
}

fn event(peer: &mut DecodePeer, mesh: &mut Mesh, group: &Group, expected: PdEvent) {
    until(|| peer.poll(mesh, group).unwrap().contains(&expected));
}

/// A Prefill control socket that answers each registration, at the endpoint
/// it names, with the messages `answer` builds from its id and cache
/// identity; everything it receives is passed on to the returned receiver.
fn answering_prefill(
    answer: impl Fn(u64, &str) -> Vec<Vec<bytes::Bytes>> + Send + 'static,
) -> (Arc<Sockets>, mpsc::Receiver<Vec<bytes::Bytes>>) {
    let (incoming, arriving) = mpsc::channel::<Vec<bytes::Bytes>>();
    let (forward, received) = mpsc::channel();
    let sockets = Arc::new(Sockets::start("127.0.0.1", incoming).unwrap());
    // Weak: the thread ends once the test drops the socket.
    let replies = Arc::downgrade(&sockets);
    thread::spawn(move || {
        while let Ok(frames) = arriving.recv() {
            let Some(replies) = replies.upgrade() else {
                return;
            };
            if let (true, Ok(Some(id)), Ok((endpoint, port, cache_id))) = (
                wire::is_registration(&frames),
                wire::registration_id(&frames),
                wire::registration_route(&frames),
            ) {
                for reply in answer(id, &cache_id) {
                    let _ = replies.send(&endpoint, port, reply);
                }
            }
            let _ = forward.send(frames);
        }
    });
    (sockets, received)
}

/// A bootstrap server whose single Prefill rank listens at `port`, with its
/// decode allocation policy if it states one.
fn bootstrap_to(port: u16, policy: Option<&str>) -> BootstrapServer {
    let server = BootstrapServer::start(0, Registry::CacheLayouts).unwrap();
    let mut registration = serde_json::json!({
        "role": "Prefill", "world_size": 1, "dp_size": 1, "pp_size": 1,
        "rank_ip": "127.0.0.1", "rank_port": port, "engine_rank": 0,
        "cache_layout": crate::layout::contract::tests::wire(),
        "cache_fields_by_stage": [["layer.0.kv", "layer.1.state"]]
    });
    if let Some(policy) = policy {
        registration["decode_allocation_policy"] = policy.into();
    }
    http::put_json(
        &format!("http://127.0.0.1:{}/route", server.port()),
        &registration,
        Duration::from_secs(2),
    )
    .unwrap();
    server
}

/// Acknowledge a registration as Prefill rank 0.
fn acknowledge(id: u64, cache_id: &str) -> Vec<Vec<bytes::Bytes>> {
    vec![wire::registration_ack_frames(id, cache_id, 0)]
}

/// Poll until `rid` stopped, within `wait`; returns every event seen.
fn until_stopped(
    peer: &mut DecodePeer,
    mesh: &mut Mesh,
    group: &Group,
    rid: &str,
    wait: Duration,
) -> Vec<PdEvent> {
    let deadline = Instant::now() + wait;
    let mut events = Vec::new();
    while !events.contains(&PdEvent::Stopped(rid.to_owned())) {
        assert!(Instant::now() < deadline, "{rid} did not stop: {events:?}");
        events.extend(peer.poll(mesh, group).unwrap());
        thread::sleep(Duration::from_millis(5));
    }
    events
}

#[test]
fn lost_routes_release_workers_on_idle_polls_and_recreate_them_on_demand() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    // Each worker owns a Shared reference; returning to this count proves
    // that it exited, rather than merely losing the map's sender.
    let owners = Arc::strong_count(&peer.shared);
    let (sink, _received) = answering_prefill(acknowledge);
    for endpoint in 0..4 {
        let server = bootstrap_to(sink.port(), None);
        for attempt in 0..2 {
            let rid = format!("request-{endpoint}-{attempt}");
            let info = bootstrap(server.port(), endpoint * 2 + attempt);
            peer.register(&rid, &info).unwrap();
            event(
                &mut peer,
                &mut mesh,
                &group,
                PdEvent::Bootstrapped(rid.clone()),
            );
            peer.cancel(&rid);
            event(&mut peer, &mut mesh, &group, PdEvent::Stopped(rid));
            assert!(!peer.has_in_flight());
            // Exercise the same route invalidation used by the heartbeat,
            // including loss when no request remains to drive the poll loop.
            peer.shared
                .lose_node(&join_host_port(&info.host, info.port));
            assert!(peer.poll(&mut mesh, &group).unwrap().is_empty());
            assert!(peer.connect.is_empty());
            until(|| Arc::strong_count(&peer.shared) == owners);
        }
    }
}

#[test]
fn pending_first_lookup_keeps_one_worker_and_cancellation_reaps_it() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let owners = Arc::strong_count(&peer.shared);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    peer.register("first", &bootstrap(port, 1)).unwrap();
    let mut held = None;
    until(|| match listener.accept() {
        Ok((stream, _)) => {
            held = Some(stream);
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => false,
        Err(error) => panic!("accept: {error}"),
    });
    assert!(peer.shared.state().routes.is_empty());
    assert!(peer.poll(&mut mesh, &group).unwrap().is_empty());
    peer.register("second", &bootstrap(port, 2)).unwrap();
    assert_eq!(Arc::strong_count(&peer.shared), owners + 1);
    assert_eq!(peer.connect.len(), 1);
    for rid in ["first", "second"] {
        peer.cancel(rid);
    }
    let events = peer.poll(&mut mesh, &group).unwrap();
    for rid in ["first", "second"] {
        assert!(events.contains(&PdEvent::Stopped(rid.to_owned())));
    }
    assert!(peer.connect.is_empty());
    // End the in-flight HTTP call. Its queued cancelled request must not
    // start another lookup before the worker exits.
    drop(held);
    until(|| Arc::strong_count(&peer.shared) == owners);
    assert!(
        matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
    );
}

#[test]
fn failed_first_lookups_do_not_leave_workers_behind() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let owners = Arc::strong_count(&peer.shared);
    for room in 0..4 {
        // An empty registry cannot supply a route, so each real HTTP lookup fails.
        let server = BootstrapServer::start(0, Registry::CacheLayouts).unwrap();
        let rid = format!("failed-{room}");
        peer.register(&rid, &bootstrap(server.port(), room))
            .unwrap();
        event(&mut peer, &mut mesh, &group, PdEvent::Stopped(rid));
        assert!(peer.shared.state().routes.is_empty());
        assert!(peer.connect.is_empty());
        until(|| Arc::strong_count(&peer.shared) == owners);
    }
}

/// A Prefill node's control socket, acknowledging every registration, with
/// what a Decode rank sends it, and the bootstrap server its rank registered
/// at, with its decode allocation policy if it states one.
fn prefill_node(
    policy: Option<&str>,
) -> (
    Arc<Sockets>,
    mpsc::Receiver<Vec<bytes::Bytes>>,
    BootstrapServer,
) {
    let (socket, received) = answering_prefill(acknowledge);
    let server = bootstrap_to(socket.port(), policy);
    (socket, received, server)
}

#[test]
fn an_announced_room_is_admitted_by_its_notice_and_repeats_its_announcement_until_then() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_millis(100), Duration::from_secs(60));
    let (prefill, received, server) = prefill_node(Some("deferred"));
    peer.register("r", &bootstrap(server.port(), 5)).unwrap();
    // Connected, the room is announced, again with every heartbeat, and the
    // scheduler does not see the request.
    let mut announcements = 0;
    while announcements < 3 {
        let frames = received
            .recv_timeout(Duration::from_secs(3))
            .expect("an announcement");
        if wire::is_announce(&frames) {
            assert_eq!(wire::announce_route(&frames).unwrap(), (5, peer.cache_id()));
            announcements += 1;
        }
        assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    }
    // The Prefill rank holds every page of the prompt.
    prefill
        .send(
            "127.0.0.1",
            peer.port(),
            StatusMessage::bare(5, TransferPoll::Bootstrapped, 0).to_frames(),
        )
        .unwrap();
    event(
        &mut peer,
        &mut mesh,
        &group,
        PdEvent::Bootstrapped("r".to_owned()),
    );
    // Answered: the announcements stop.
    thread::sleep(Duration::from_millis(150));
    while received.try_recv().is_ok() {}
    thread::sleep(Duration::from_millis(400));
    assert!(received.try_recv().is_err());

    // A Prefill node that states no policy allocates early: its room is
    // admitted once connected and never announced.
    let (_early, early_received, early_server) = prefill_node(None);
    peer.register("early", &bootstrap(early_server.port(), 6))
        .unwrap();
    event(
        &mut peer,
        &mut mesh,
        &group,
        PdEvent::Bootstrapped("early".to_owned()),
    );
    thread::sleep(Duration::from_millis(300));
    assert!(early_received
        .try_iter()
        .all(|frames: Vec<bytes::Bytes>| !wire::is_announce(&frames)));
}

#[test]
fn an_announced_room_times_out_from_registration_and_again_from_its_notice() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_millis(600));
    let (prefill, _received, server) = prefill_node(Some("deferred"));
    // Never noticed: its Prefill node is gone, lost the request, or never
    // held its pages.
    let registered = Instant::now();
    peer.register("lost", &bootstrap(server.port(), 6)).unwrap();
    event(
        &mut peer,
        &mut mesh,
        &group,
        PdEvent::Failed("lost".to_owned()),
    );
    assert!(registered.elapsed() >= Duration::from_millis(550));

    // Noticed late, and never allocated here: the wait restarted at the
    // notice.
    peer.register("noticed", &bootstrap(server.port(), 7))
        .unwrap();
    thread::sleep(Duration::from_millis(350));
    prefill
        .send(
            "127.0.0.1",
            peer.port(),
            StatusMessage::bare(7, TransferPoll::Bootstrapped, 0).to_frames(),
        )
        .unwrap();
    event(
        &mut peer,
        &mut mesh,
        &group,
        PdEvent::Bootstrapped("noticed".to_owned()),
    );
    let noticed = Instant::now();
    event(
        &mut peer,
        &mut mesh,
        &group,
        PdEvent::Failed("noticed".to_owned()),
    );
    assert!(noticed.elapsed() >= Duration::from_millis(550));
}

#[test]
fn every_noticed_request_is_shown_to_the_scheduler() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let (_prefill, _received, server) = prefill_node(Some("deferred"));
    // Twelve requests arrive and connect; the scheduler sees none of them
    // before its notice.
    let mut shown = Vec::new();
    for room in 0..12 {
        peer.register(&format!("r{room}"), &bootstrap(server.port(), room))
            .unwrap();
    }
    until(|| {
        shown.extend(peer.poll(&mut mesh, &group).unwrap());
        let state = peer.shared.state();
        (0..12).all(|room| state.rooms.get(room) == Some(TransferPoll::Bootstrapped))
    });
    assert_eq!(shown, []);
    // Their notices arrive: the scheduler sees every one, whatever it has
    // allocated, and chooses among them.
    for room in 0..12 {
        peer.shared
            .state()
            .apply(StatusMessage::bare(room, TransferPoll::Bootstrapped, 0));
    }
    assert_eq!(
        peer.poll(&mut mesh, &group).unwrap(),
        (0..12)
            .map(|room| PdEvent::Bootstrapped(format!("r{room}")))
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_heartbeat_repeats_a_bounded_number_of_announcements_and_covers_every_room_in_turn() {
    let native = NativeLibrary::new();
    let (peer, _outbox, _mesh, _group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let (incoming, received) = mpsc::channel();
    let prefill = Sockets::start("127.0.0.1", incoming).unwrap();
    // 300 connected rooms wait for the notice of one Prefill rank.
    {
        let mut state = peer.shared.state();
        let targets = Arc::new(vec![Target {
            rank: 0,
            host: "127.0.0.1".to_owned(),
            port: prefill.port(),
            is_dummy: false,
        }]);
        for room in 0..300 {
            state.tracked.insert(room);
            state.announced.insert(room);
            state.expected.insert(room, BTreeSet::from([0]));
            state.targets.insert(room, targets.clone());
        }
    }
    let announced = |count: usize| -> BTreeSet<u64> {
        (0..count)
            .map(|_| {
                let frames: Vec<bytes::Bytes> = received
                    .recv_timeout(Duration::from_secs(3))
                    .expect("an announcement");
                wire::announce_route(&frames).unwrap().0
            })
            .collect()
    };
    // One beat reaches as many rooms as the burst allows and says where it
    // stopped.
    assert_eq!(peer.shared.announce_waiting(0), 256);
    assert_eq!(announced(256), (0..256).collect());
    assert!(received.recv_timeout(Duration::from_millis(200)).is_err());
    // The next beat starts there and wraps around.
    assert_eq!(peer.shared.announce_waiting(256), 212);
    assert_eq!(announced(256), (256..300).chain(0..212).collect());
    assert!(received.recv_timeout(Duration::from_millis(200)).is_err());
    // A room whose notice arrived is announced no more.
    for room in 0..299 {
        peer.shared
            .state()
            .apply(StatusMessage::bare(room, TransferPoll::Bootstrapped, 0));
    }
    assert_eq!(peer.shared.announce_waiting(212), 0);
    assert_eq!(announced(1), BTreeSet::from([299]));
}

#[test]
fn replies_for_another_cache_id_or_rank_do_not_settle_a_registration() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let (prefill, _received) = answering_prefill(|id, cache_id| {
        vec![
            wire::registration_ack_frames(id, "another-cache", 0),
            wire::registration_ack_frames(id + 1, cache_id, 0),
            wire::registration_ack_frames(id, cache_id, 1),
            wire::registration_reject_frames(id, "another-cache", 0, "stale"),
            wire::registration_reject_frames(id, cache_id, 1, "not a target"),
        ]
    });
    let server = bootstrap_to(prefill.port(), None);
    let started = Instant::now();
    peer.register("stray", &bootstrap(server.port(), 1))
        .unwrap();
    let events = until_stopped(
        &mut peer,
        &mut mesh,
        &group,
        "stray",
        REGISTRATION_TIMEOUT + Duration::from_secs(5),
    );
    // Only the deadline ended it: no stray reply settled it either way.
    assert!(started.elapsed() >= REGISTRATION_TIMEOUT);
    assert!(!events.contains(&PdEvent::Bootstrapped("stray".to_owned())));
    assert!(peer.shared.state().connections.is_empty());
}

#[test]
fn a_rejected_registration_fails_its_room_at_once() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let (prefill, _received) = answering_prefill(|id, cache_id| {
        vec![wire::registration_reject_frames(
            id,
            cache_id,
            0,
            "incompatible layout",
        )]
    });
    let server = bootstrap_to(prefill.port(), None);
    peer.register("rejected", &bootstrap(server.port(), 1))
        .unwrap();
    let events = until_stopped(
        &mut peer,
        &mut mesh,
        &group,
        "rejected",
        REGISTRATION_TIMEOUT / 2,
    );
    assert!(!events.contains(&PdEvent::Bootstrapped("rejected".to_owned())));
    assert!(peer.shared.state().connections.is_empty());
}

#[test]
fn a_full_control_queue_delays_a_registration_instead_of_failing_its_room() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let (prefill, _received) = answering_prefill(acknowledge);
    // A relay in front of the Prefill socket leaves its first connection
    // silent, so the queue behind it fills until the connect deadline drops
    // that connection with its backlog; later connections are relayed.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let relay = listener.local_addr().unwrap().port();
    let target = prefill.port();
    thread::spawn(move || {
        let mut silent = Vec::new();
        for (index, client) in listener.incoming().enumerate() {
            let Ok(client) = client else { return };
            if index == 0 {
                silent.push(client);
                continue;
            }
            let server = std::net::TcpStream::connect(("127.0.0.1", target)).unwrap();
            for (mut from, mut to) in [
                (client.try_clone().unwrap(), server.try_clone().unwrap()),
                (server, client),
            ] {
                thread::spawn(move || {
                    let _ = std::io::copy(&mut from, &mut to);
                    let _ = to.shutdown(std::net::Shutdown::Write);
                });
            }
        }
    });
    let mut queued = 0;
    while peer
        .shared
        .transport
        .sockets
        .send("127.0.0.1", relay, vec![crate::net::ascii(queued)])
        .is_ok()
    {
        queued += 1;
        assert!(queued <= 4096, "the control queue never filled");
    }
    let server = bootstrap_to(relay, None);
    peer.register("queued", &bootstrap(server.port(), 1))
        .unwrap();
    let deadline = Instant::now() + REGISTRATION_TIMEOUT;
    let mut events = Vec::new();
    while !events.contains(&PdEvent::Bootstrapped("queued".to_owned())) {
        assert!(Instant::now() < deadline, "not connected: {events:?}");
        events.extend(peer.poll(&mut mesh, &group).unwrap());
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(events, [PdEvent::Bootstrapped("queued".to_owned())]);
}

#[test]
fn announcements_after_a_revocation_name_the_cache_their_room_registered() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, _mesh, _group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let (_prefill, received, server) = prefill_node(Some("deferred"));
    let registered = peer.cache_id();
    peer.register("r", &bootstrap(server.port(), 5)).unwrap();
    let announced = |received: &mpsc::Receiver<Vec<bytes::Bytes>>| loop {
        let frames = received
            .recv_timeout(Duration::from_secs(3))
            .expect("an announcement");
        if wire::is_announce(&frames) {
            break wire::announce_route(&frames).unwrap();
        }
    };
    assert_eq!(announced(&received), (5, registered.clone()));
    // A revocation replaces the transfer engine and the cache identity while
    // the room still waits for its notice.
    peer.shared.state().revoking = true;
    let mut revocation = Revocation {
        shared: peer.shared.clone(),
        retry_at: Instant::now(),
        ended: false,
    };
    peer.shared.revoke(&mut revocation);
    drop(revocation);
    assert_ne!(peer.cache_id(), registered);
    while received.try_recv().is_ok() {}
    // Prefill knows only the identity the room's route registered.
    peer.shared.announce_waiting(0);
    assert_eq!(announced(&received), (5, registered));
}

#[test]
fn revocation_outcomes_are_logged_as_events() {
    use std::{io::Read, os::unix::net::UnixStream};

    let native = NativeLibrary::new();
    let cache = [0u8; 64];
    let (peer, outbox) = DecodePeer::start(DecodeConfig {
        peer: PeerConfig {
            library: native.path().to_path_buf(),
            local_ip: "127.0.0.1".to_owned(),
            protocol: "rdma".to_owned(),
            ib_device: String::new(),
            regions: vec![(cache.as_ptr() as usize, cache.len())],
            engine_rank: 3,
            metrics: None,
        },
        arena_base: cache.as_ptr() as u64,
        layout: crate::layout::contract::tests::wire().into_bytes(),
        tp_size: 1,
        tp_rank: 0,
        waiting_timeout: Duration::from_secs(60),
        heartbeat_interval: Duration::from_secs(60),
        heartbeat_max_failures: 2,
    })
    .unwrap();
    // Driven on this thread, where the subscriber below listens.
    let revoke = |peer: &DecodePeer| {
        peer.shared.state().revoking = true;
        let mut revocation = Revocation {
            shared: peer.shared.clone(),
            retry_at: Instant::now(),
            ended: false,
        };
        peer.shared.revoke(&mut revocation);
    };
    let (writer, mut reader) = UnixStream::pair().unwrap();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(move || writer.try_clone().unwrap())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        revoke(&peer);
        native.fail_unregistrations();
        revoke(&peer);
        let refused = outbox.release_registration().unwrap_err();
        assert!(refused.contains("kept it registered"), "{refused}");
    });
    let mut logs = String::new();
    reader.read_to_string(&mut logs).unwrap();
    let line = |event: &str| {
        logs.lines()
            .find(|line| line.contains(event))
            .unwrap_or_else(|| panic!("{event} missing from:\n{logs}"))
            .to_owned()
    };
    assert!(line("event=ts_decode_revocation_fenced").contains("rank=3"));
    let unfenced = line("event=ts_decode_revocation_unfenced");
    assert!(
        unfenced.contains("ERROR") && unfenced.contains("failed to deregister 1"),
        "{unfenced}"
    );
    assert!(line("event=ts_decode_release_refused_unfenced").contains("ERROR"));
}

#[test]
fn a_revocation_marks_its_rooms_as_it_ends() {
    let native = NativeLibrary::new();
    let (peer, _outbox, _mesh, _group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    // A room whose manifest named the engine about to be replaced.
    {
        let mut state = peer.shared.state();
        let cache_id = state.cache_id.clone();
        state.tracked.insert(7);
        state.attached_caches.insert(7, cache_id);
        state.admissions.insert(7, 0);
        state.revoking = true;
    }
    let mut revocation = Revocation {
        shared: peer.shared.clone(),
        retry_at: Instant::now(),
        ended: false,
    };
    peer.shared.revoke(&mut revocation);
    {
        let state = peer.shared.state();
        assert!(state.revoked.contains(&7) && !state.revoking);
    }
    // The room retires, through acknowledgments, before the guard drops:
    // nothing may mark it again.
    peer.shared.state().clear(7, "prefill:80");
    drop(revocation);
    assert!(peer.shared.state().revoked.is_empty());
}

#[test]
fn a_registration_interrupted_by_a_revocation_connects_under_the_fresh_cache() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    // Prefill acknowledges only a cache other than the one registering now,
    // so the room's first registration waits.
    let replaced = peer.cache_id();
    let pending = replaced.clone();
    let (prefill, _received) = answering_prefill(move |id, cache_id| {
        if cache_id == pending {
            Vec::new()
        } else {
            acknowledge(id, cache_id)
        }
    });
    let server = bootstrap_to(prefill.port(), None);
    peer.register("r", &bootstrap(server.port(), 1)).unwrap();
    until(|| !peer.shared.state().registering.is_empty());
    // A revocation replaces the transfer engine and cache identity meanwhile.
    peer.shared.state().revoking = true;
    let mut revocation = Revocation {
        shared: peer.shared.clone(),
        retry_at: Instant::now(),
        ended: false,
    };
    peer.shared.revoke(&mut revocation);
    drop(revocation);
    assert_ne!(peer.cache_id(), replaced);
    // The room gives up the replaced cache's registration and connects again
    // under the fresh one, instead of waiting until its budget runs out.
    let deadline = Instant::now() + REGISTRATION_TIMEOUT / 2;
    let mut events = Vec::new();
    while !events.contains(&PdEvent::Bootstrapped("r".to_owned())) {
        assert!(Instant::now() < deadline, "not connected: {events:?}");
        events.extend(peer.poll(&mut mesh, &group).unwrap());
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(events, [PdEvent::Bootstrapped("r".to_owned())]);
}
