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
//! against a stand-in bootstrap server, and the KV crosses over RDMA.
//!
//! Needs an RDMA device and `TS_MOONCAKE_LIBRARY` naming
//! `libtokenspeed_transfer_engine.so`; without the variable the tests return
//! at once. The planners stand in for the data plane: one region per side,
//! one span per transfer.

use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use ts_collective::{Group, Listener, Mesh};
use ts_control::pd::{BootstrapInfo, PdEvent, RemotePrefill, TransferPeer};
use ts_pd::{
    Copy, DecodeConfig, DecodePeer, DecodePlanner, PeerConfig, PrefillConfig, PrefillPeer,
    PrefillPlanner, PrefillRoute, Registration,
};

const BYTES: usize = 4096;

fn library() -> Option<PathBuf> {
    std::env::var_os("TS_MOONCAKE_LIBRARY").map(PathBuf::from)
}

/// A bootstrap server: Prefill ranks register, Decode ranks look them up.
fn bootstrap_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let ranks: Arc<Mutex<Option<serde_json::Value>>> = Arc::default();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut length = 0;
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                if header == "\r\n" || header.is_empty() {
                    break;
                }
                if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let target = request_line.split_whitespace().nth(1).unwrap_or_default();
            let reply = if request_line.starts_with("PUT /route") {
                *ranks.lock().unwrap() = Some(serde_json::from_slice(&body).unwrap());
                "OK".to_owned()
            } else if target.starts_with("/route?engine_rank=-1") {
                r#"{"prefill_tp_size":1,"prefill_dp_size":1,"prefill_pp_size":1}"#.to_owned()
            } else if target.starts_with("/route?") {
                let registered = ranks.lock().unwrap().clone().unwrap();
                serde_json::json!({
                    "rank_ip": registered["rank_ip"],
                    "rank_port": registered["rank_port"],
                })
                .to_string()
            } else {
                "OK".to_owned()
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{reply}",
                reply.len()
            );
        }
    });
    port
}

/// One span per transfer: the whole source region to the session's region.
struct Planner {
    source: u64,
    destinations: Mutex<HashMap<String, u64>>,
    refuse_preallocation: bool,
}

impl PrefillPlanner for Planner {
    fn prepare_registration(&self, frames: &[Bytes]) -> Result<Registration, String> {
        let session_id = String::from_utf8(frames[3].to_vec()).unwrap();
        let base = u64::from_ne_bytes(frames[4].as_ref().try_into().unwrap());
        self.destinations
            .lock()
            .unwrap()
            .insert(session_id.clone(), base);
        Ok(Registration {
            endpoint: String::from_utf8(frames[1].to_vec()).unwrap(),
            port: std::str::from_utf8(&frames[2]).unwrap().parse().unwrap(),
            session_id,
            decode_tp_size: 1,
            decode_tp_rank: 0,
            is_dummy: false,
            expected_decode_ranks: vec![0],
        })
    }

    fn reject_registration(&self, _session_id: &str) {}

    fn accept_preallocation(&self, frames: &[Bytes]) -> Result<bool, String> {
        if self.refuse_preallocation {
            return Err("the manifest disagrees with the registration".to_owned());
        }
        Ok(frames[2].is_empty())
    }

    fn plan_send(&self, _room: u64, session_id: &str) -> Result<Vec<Copy>, String> {
        let destination = self.destinations.lock().unwrap()[session_id];
        Ok(vec![Copy::Spans {
            sources: vec![self.source],
            destinations: vec![destination],
            lengths: vec![BYTES as u64],
        }])
    }

    /// Two intervals, each the source's half.
    fn plan_layerwise(
        &self,
        _chunk_id: u64,
        sessions: &[String],
    ) -> Result<Vec<Vec<Vec<Copy>>>, String> {
        let half = (BYTES / 2) as u64;
        Ok(sessions
            .iter()
            .map(|session| {
                let destination = self.destinations.lock().unwrap()[session];
                (0..2)
                    .map(|interval| {
                        vec![Copy::Spans {
                            sources: vec![self.source + interval * half],
                            destinations: vec![destination + interval * half],
                            lengths: vec![half],
                        }]
                    })
                    .collect()
            })
            .collect())
    }

    fn drop_room(&self, _room: u64) {}
}

struct Route;

impl DecodePlanner for Route {
    fn plan_route(&self, prefill_info: &str) -> Result<PrefillRoute, String> {
        assert!(prefill_info.contains("prefill_tp_size"));
        Ok(PrefillRoute {
            target_tp_ranks: vec![0],
            dummy_tp_ranks: vec![],
            expected_prefill_ranks: vec![0],
            prefill_dp_size: 1,
        })
    }
}

fn single_rank() -> (Mesh, Group) {
    let listener = Listener::bind("127.0.0.1").unwrap();
    let address = format!("127.0.0.1:{}", listener.port().unwrap());
    let mesh = Mesh::connect(0, listener, &[address], Duration::from_secs(5)).unwrap();
    (mesh, Group::new(0, vec![0], 0).unwrap())
}

fn peer_config(library: PathBuf, region: &[u8]) -> PeerConfig {
    PeerConfig {
        library,
        local_ip: "127.0.0.1".to_owned(),
        ib_device: String::new(),
        regions: vec![(region.as_ptr() as usize, region.len())],
        engine_rank: 0,
        metrics: None,
    }
}

struct Pair {
    prefill: PrefillPeer,
    prefill_outbox: ts_pd::PrefillOutbox,
    decode: DecodePeer,
    decode_outbox: ts_pd::DecodeOutbox,
    bootstrap: BootstrapInfo,
}

fn pair(
    library: PathBuf,
    source: &[u8],
    destination: &[u8],
    refuse_preallocation: bool,
    bootstrap_timeout: Duration,
) -> Pair {
    pair_with(
        library,
        source,
        destination,
        refuse_preallocation,
        bootstrap_timeout,
        None,
    )
}

fn pair_with(
    library: PathBuf,
    source: &[u8],
    destination: &[u8],
    refuse_preallocation: bool,
    bootstrap_timeout: Duration,
    layerwise_step_address: Option<usize>,
) -> Pair {
    let port = bootstrap_server();
    let planner = Arc::new(Planner {
        source: source.as_ptr() as u64,
        destinations: Mutex::default(),
        refuse_preallocation,
    });
    let (prefill, prefill_outbox) = PrefillPeer::start(
        PrefillConfig {
            peer: peer_config(library.clone(), source),
            status_rank: 0,
            bootstrap_url: format!("http://127.0.0.1:{port}/route"),
            bootstrap_payload: serde_json::json!({
                "role": "Prefill",
                "world_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "rank_ip": "127.0.0.1",
                "engine_rank": 0,
            }),
            bootstrap_timeout,
            failed_session_ttl: Duration::from_secs(30),
            transfer_workers: 2,
            layerwise_step_address,
        },
        planner,
    )
    .unwrap();
    let (decode, decode_outbox) = DecodePeer::start(
        DecodeConfig {
            peer: peer_config(library, destination),
            arena_base: destination.as_ptr() as u64,
            layout: b"layout".to_vec(),
            tp_size: 1,
            tp_rank: 0,
            waiting_timeout: Duration::from_secs(60),
            heartbeat_interval: Duration::from_secs(2),
            heartbeat_max_failures: 2,
        },
        Arc::new(Route),
    )
    .unwrap();
    Pair {
        prefill,
        prefill_outbox,
        decode,
        decode_outbox,
        bootstrap: BootstrapInfo {
            host: "127.0.0.1".to_owned(),
            port,
            room: 4242,
        },
    }
}

/// Poll until `peer` reports an event.
fn next_event(peer: &mut dyn TransferPeer, mesh: &mut Mesh, group: &Group) -> PdEvent {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let mut events = peer.poll(mesh, group).unwrap();
        if let Some(event) = events.pop() {
            assert!(events.is_empty());
            return event;
        }
        assert!(Instant::now() < deadline, "no transfer event");
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_request_bootstraps_transfers_and_lands() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source: Vec<u8> = (0..BYTES).map(|i| (i % 251) as u8).collect();
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library,
        &source,
        &destination,
        false,
        Duration::from_secs(60),
    );
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
    pair.decode_outbox.receive(rid, 4242, 3, b"manifest");
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    // The prompt's last chunk commits, then its remote decode runs.
    pair.prefill.record_usage(rid, 128, Some(-0.5));
    pair.prefill_outbox
        .send(rid, 4242, 777, Some(vec![1, 2, 3]));
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
    assert_eq!(destination, source);
    // Both sides forgot the request.
    assert!(pair.prefill.poll(&mut mesh, &group).unwrap().is_empty());
    assert!(pair.decode.poll(&mut mesh, &group).unwrap().is_empty());
}

/// Run one request through `pair`, from registration to landing; a sleep
/// is refused on both sides while it is in flight.
fn transfer_one(pair: &mut Pair, rid: &str, room: u64, mesh: &mut Mesh, group: &Group) {
    let bootstrap = BootstrapInfo {
        room,
        ..pair.bootstrap.clone()
    };
    pair.prefill.register(rid, &bootstrap).unwrap();
    pair.decode.register(rid, &bootstrap).unwrap();
    let refused = pair.prefill_outbox.release_registration().unwrap_err();
    assert!(refused.contains("KV sends are in flight"), "{refused}");
    let refused = pair.decode_outbox.release_registration().unwrap_err();
    assert!(refused.contains("KV receives are in flight"), "{refused}");
    assert_eq!(
        next_event(&mut pair.decode, mesh, group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.decode_outbox.receive(rid, room, 3, b"manifest");
    assert_eq!(
        next_event(&mut pair.prefill, mesh, group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.prefill_outbox.send(rid, room, 777, None);
    assert_eq!(
        next_event(&mut pair.prefill, mesh, group),
        PdEvent::Succeeded(rid.to_owned())
    );
    assert!(matches!(
        next_event(&mut pair.decode, mesh, group),
        PdEvent::RemotePrefillDone(_)
    ));
}

#[test]
fn a_sleeping_pair_deregisters_and_wakes_to_transfer_again() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source: Vec<u8> = (0..BYTES).map(|i| (i % 251) as u8).collect();
    let mut destination = vec![0u8; BYTES];
    let mut pair = pair(
        library,
        &source,
        &destination,
        false,
        Duration::from_secs(60),
    );
    let (mut mesh, group) = single_rank();
    transfer_one(&mut pair, "before", 4242, &mut mesh, &group);
    assert_eq!(destination, source);

    let session = pair.decode.session_id();
    pair.prefill_outbox.release_registration().unwrap();
    pair.decode_outbox.release_registration().unwrap();
    pair.prefill_outbox.restore_registration().unwrap();
    pair.decode_outbox.restore_registration().unwrap();
    // The Decode rank wakes under a fresh session, and the next request
    // registers it with the Prefill rank again.
    assert_ne!(pair.decode.session_id(), session);
    destination.iter_mut().for_each(|byte| *byte = 0);
    transfer_one(&mut pair, "after", 4244, &mut mesh, &group);
    assert_eq!(destination, source);
}

#[test]
fn layerwise_chunks_follow_the_producer_steps() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source: Vec<u8> = (0..BYTES).map(|i| (i % 251) as u8).collect();
    let destination = vec![0u8; BYTES];
    // The counter the device would update; the peer polls it.
    let steps = Box::new(std::sync::atomic::AtomicI64::new(10));
    let mut pair = pair_with(
        library,
        &source,
        &destination,
        false,
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
    pair.decode_outbox.receive(rid, 4242, 3, b"manifest");
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    // The final chunk arms with intervals complete at steps 11 and 12: the
    // counter passes a step once that step's KV landed.
    pair.prefill_outbox
        .send_layerwise(rid, 4242, 0, vec![11, 12], true);
    let half = BYTES / 2;
    thread::sleep(Duration::from_millis(50));
    assert!(
        destination.iter().all(|&byte| byte == 0),
        "nothing before its step"
    );
    steps.store(12, std::sync::atomic::Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(10);
    while std::hint::black_box(destination.as_slice())[..half] != source[..half] {
        assert!(Instant::now() < deadline, "the first interval never landed");
        thread::sleep(Duration::from_millis(5));
    }
    assert!(destination[half..].iter().all(|&byte| byte == 0));
    steps.store(13, std::sync::atomic::Ordering::Release);
    // The room reports done only once its remote decode brings the payload.
    thread::sleep(Duration::from_millis(50));
    assert!(pair.prefill.poll(&mut mesh, &group).unwrap().is_empty());
    pair.prefill.record_usage(rid, 0, None);
    pair.prefill_outbox.send(rid, 4242, 99, None);
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Succeeded(rid.to_owned())
    );
    match next_event(&mut pair.decode, &mut mesh, &group) {
        PdEvent::RemotePrefillDone(done) => assert_eq!(done.bootstrap_token, 99),
        other => panic!("expected the landing, got {other:?}"),
    }
    assert_eq!(destination, source);
}

#[test]
fn a_refused_preallocation_fails_both_sides() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = vec![1u8; BYTES];
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library,
        &source,
        &destination,
        true,
        Duration::from_secs(60),
    );
    let (mut mesh, group) = single_rank();
    let rid = "req-1";

    pair.prefill.register(rid, &pair.bootstrap).unwrap();
    pair.decode.register(rid, &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Bootstrapped(rid.to_owned())
    );
    pair.decode_outbox.receive(rid, 4242, 0, b"manifest");
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    assert_eq!(
        next_event(&mut pair.decode, &mut mesh, &group),
        PdEvent::Failed(rid.to_owned())
    );
    // A late send for the failed room is dropped.
    pair.prefill_outbox.send(rid, 4242, 5, None);
    assert!(pair.prefill_outbox.send_unplanned(rid, 4242).is_ok());
    assert!(destination.iter().all(|&byte| byte == 0));
}

#[test]
fn a_room_no_decode_rank_reaches_times_out() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let source = vec![1u8; BYTES];
    let destination = vec![0u8; BYTES];
    let mut pair = pair(
        library,
        &source,
        &destination,
        false,
        Duration::from_millis(200),
    );
    let (mut mesh, group) = single_rank();

    pair.prefill.register("lonely", &pair.bootstrap).unwrap();
    assert_eq!(
        next_event(&mut pair.prefill, &mut mesh, &group),
        PdEvent::Failed("lonely".to_owned())
    );
    // A room still bootstrapping has no destinations to send to.
    pair.prefill
        .register(
            "pending",
            &BootstrapInfo {
                room: 9,
                ..pair.bootstrap.clone()
            },
        )
        .unwrap();
    assert!(pair.prefill_outbox.send_unplanned("pending", 9).is_err());
}
