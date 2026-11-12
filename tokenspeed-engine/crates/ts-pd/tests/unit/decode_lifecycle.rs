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

#[test]
fn lost_routes_release_workers_on_idle_polls_and_recreate_them_on_demand() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    // Each worker owns a Shared reference; returning to this count proves
    // that it exited, rather than merely losing the map's sender.
    let owners = Arc::strong_count(&peer.shared);
    let (incoming, _received) = mpsc::channel();
    let sink = Sockets::start("127.0.0.1", incoming).unwrap();
    for endpoint in 0..4 {
        let server = BootstrapServer::start(0, Registry::CacheLayouts).unwrap();
        http::put_json(
            &format!("http://127.0.0.1:{}/route", server.port()),
            &serde_json::json!({
                "role": "Prefill", "world_size": 1, "dp_size": 1, "pp_size": 1,
                "rank_ip": "127.0.0.1", "rank_port": sink.port(), "engine_rank": 0,
                "cache_layout": crate::layout::contract::tests::wire(),
                "cache_fields_by_stage": [["layer.0.kv", "layer.1.state"]]
            }),
            Duration::from_secs(2),
        )
        .unwrap();
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

/// A Prefill node's control socket with what a Decode rank sends it, and the
/// bootstrap server its rank registered at, with its decode allocation
/// policy if it states one.
fn prefill_node(
    policy: Option<&str>,
) -> (Sockets, mpsc::Receiver<Vec<bytes::Bytes>>, BootstrapServer) {
    let (incoming, received) = mpsc::channel();
    let socket = Sockets::start("127.0.0.1", incoming).unwrap();
    let server = BootstrapServer::start(0, Registry::CacheLayouts).unwrap();
    let mut registration = serde_json::json!({
        "role": "Prefill", "world_size": 1, "dp_size": 1, "pp_size": 1,
        "rank_ip": "127.0.0.1", "rank_port": socket.port(), "engine_rank": 0,
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
    (socket, received, server)
}

#[test]
fn an_announced_room_is_admitted_by_its_notice_and_repeats_its_announcement_until_then() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_millis(100), Duration::from_secs(60));
    let (prefill, received, server) = prefill_node(Some("independent"));
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
    let (prefill, _received, server) = prefill_node(Some("independent"));
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
fn a_noticed_request_is_passed_a_bounded_number_of_times() {
    let native = NativeLibrary::new();
    let (mut peer, _outbox, mut mesh, group) =
        setup(&native, Duration::from_secs(60), Duration::from_secs(60));
    let (_prefill, _received, server) = prefill_node(Some("independent"));
    // Requests r<first>..r<last> arrive, connect, and are noticed in order;
    // returns what the scheduler is shown.
    let arrive = |peer: &mut DecodePeer, mesh: &mut Mesh, rooms: std::ops::Range<u64>| {
        let mut shown = Vec::new();
        for room in rooms.clone() {
            peer.register(&format!("r{room}"), &bootstrap(server.port(), room))
                .unwrap();
        }
        until(|| {
            shown.extend(peer.poll(mesh, &group).unwrap());
            let state = peer.shared.state();
            rooms
                .clone()
                .all(|room| state.rooms.get(room) == Some(TransferPoll::Bootstrapped))
        });
        for room in rooms {
            peer.shared
                .state()
                .apply(StatusMessage::bare(room, TransferPoll::Bootstrapped, 0));
            shown.extend(peer.poll(mesh, &group).unwrap());
        }
        shown
    };
    let shown = |rooms: &[u64]| -> Vec<PdEvent> {
        rooms
            .iter()
            .map(|room| PdEvent::Bootstrapped(format!("r{room}")))
            .collect()
    };
    // The scheduler sees the eight oldest.
    assert_eq!(
        arrive(&mut peer, &mut mesh, 0..12),
        shown(&[0, 1, 2, 3, 4, 5, 6, 7])
    );
    // It allocates others than the oldest (their manifests go out): each
    // time one more is shown.
    for (allocated, next) in [(1, 8), (2, 9), (3, 10), (4, 11)] {
        peer.shared
            .state()
            .rooms
            .update(allocated, TransferPoll::WaitingForInput);
        assert_eq!(peer.poll(&mut mesh, &group).unwrap(), shown(&[next]));
    }
    assert_eq!(arrive(&mut peer, &mut mesh, 12..16), []);
    for (allocated, next) in [(5, 12), (6, 13), (7, 14)] {
        peer.shared
            .state()
            .rooms
            .update(allocated, TransferPoll::WaitingForInput);
        assert_eq!(peer.poll(&mut mesh, &group).unwrap(), shown(&[next]));
    }
    // r0 was passed eight times now: nothing new is shown until it is
    // allocated, though the requests already shown may still go first.
    for allocated in [8, 9] {
        peer.shared
            .state()
            .rooms
            .update(allocated, TransferPoll::WaitingForInput);
        assert_eq!(peer.poll(&mut mesh, &group).unwrap(), []);
    }
    peer.shared
        .state()
        .rooms
        .update(0, TransferPoll::WaitingForInput);
    assert_eq!(peer.poll(&mut mesh, &group).unwrap(), shown(&[15]));
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
