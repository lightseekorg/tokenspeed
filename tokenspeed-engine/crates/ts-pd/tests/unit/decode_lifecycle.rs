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

fn setup(native: &NativeLibrary) -> (DecodePeer, DecodeOutbox, Mesh, Group) {
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
        waiting_timeout: Duration::from_secs(60),
        heartbeat_interval: Duration::from_secs(60),
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
    let (mut peer, _outbox, mut mesh, group) = setup(&native);
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
    let (mut peer, _outbox, mut mesh, group) = setup(&native);
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
    let (mut peer, _outbox, mut mesh, group) = setup(&native);
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
