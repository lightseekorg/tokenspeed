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
        registration_gate: RwLock::new(()),
        status_rank: 0,
        failed_session_ttl: Duration::from_secs(30),
        metrics: None,
        layerwise: None,
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
