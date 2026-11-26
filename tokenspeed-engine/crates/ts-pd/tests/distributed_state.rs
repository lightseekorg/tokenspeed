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

//! Exercise state fan-out through both live peers and a CPU transfer library.
use serde_json::{json, Value};
use std::{path::Path, time::Duration};
use ts_pd::{
    layout::{contract::Contract, manifest::BlockTable},
    BootstrapInfo, BootstrapServer, DecodeConfig, DecodePeer, PdEvent, PrefillConfig,
    PrefillLayout, PrefillPeer, Registry, TransferPeer,
};

#[path = "../src/tests_support.rs"]
mod native;
mod support;
use support::*;

// Keep one ordinary field on the scheduler owner and split the second field's rows.
fn hybrid(base: Option<u64>) -> Vec<u8> {
    let mut wire: Value = serde_json::from_str(&layout(base)).unwrap();
    wire["plan"]["groups"].as_array_mut().unwrap().push(json!({
        "group_id":"state", "cache_blocks_per_lcm_block":1, "page_count":4
    }));
    wire["plan"]["fields"][1]["group_id"] = json!("state");
    wire["group_specs"].as_array_mut().unwrap().push(json!({
        "group_id":"state","attn_kind":"state","rows_per_page":null,"entry_stride_tokens":null,
        "sliding_window_tokens":null,"replayable":false,"transfer_policy":"latest_snapshot","shard_count":1
    }));
    serde_json::to_vec(&wire).unwrap()
}

fn receiver(
    lib: &Path,
    rank: i64,
    arena: &[u8],
    state: &[u8],
) -> (DecodePeer, ts_pd::DecodeOutbox) {
    let mut peer = peer_config(lib.to_owned(), arena);
    peer.engine_rank = rank;
    peer.regions.push((state.as_ptr() as usize, state.len()));
    let base = arena.as_ptr() as u64;
    DecodePeer::start(DecodeConfig {
        peer,
        arena_base: base,
        layout: hybrid(Some(base)),
        tp_size: 1,
        tp_rank: 0,
        waiting_timeout: Duration::from_secs(10),
        heartbeat_interval: Duration::from_secs(5),
        heartbeat_max_failures: 2,
    })
    .unwrap()
}

#[test]
fn state_rows_land_on_both_members_before_the_owner_finishes() {
    let lib = native::NativeLibrary::new();
    let src = pattern();
    let ordinary = [vec![0; BYTES], vec![0; BYTES]];
    let physical = [vec![0; 2048], vec![0; 2048]];
    let server = BootstrapServer::start(0, Registry::CacheLayouts).unwrap();
    let port = server.port();
    let base = src.as_ptr() as u64;
    let fields = FIELDS.map(str::to_owned).to_vec();
    let (mut p, send) = PrefillPeer::start(PrefillConfig {
        peer:peer_config(lib.path().to_owned(), &src),
        layout:PrefillLayout {
            local:Contract::from_wire(&hybrid(Some(base))).unwrap(),
            wire:Contract::from_wire(&hybrid(None)).unwrap(), arena_base:base,
            tp_size:1,tp_rank:0,pp_rank:0,fields_by_stage:vec![fields.clone()],
        },
        status_rank:0, bootstrap_url:format!("http://127.0.0.1:{port}/route"),
        bootstrap_payload:json!({"role":"Prefill","world_size":1,"dp_size":1,"pp_size":1,
            "rank_ip":"127.0.0.1","engine_rank":0,"cache_layout":String::from_utf8(hybrid(None)).unwrap(),
            "cache_fields_by_stage":[fields],"decode_allocation_policy":"early"}),
        bootstrap_timeout:Duration::from_secs(10),failed_session_ttl:Duration::from_secs(10),
        transfer_workers:1,layerwise:None,
    }).unwrap();
    let mut ranks = [
        receiver(lib.path(), 3, &ordinary[0], &physical[0]),
        receiver(lib.path(), 7, &ordinary[1], &physical[1]),
    ];
    let members: Vec<_>=ranks.iter().enumerate().map(|(i,(d,_))|json!({
        "rank":([3, 7][i]),"session":d.session_id(),"cache":d.cache_id(),
        "fields":{(FIELDS[1]):{"base":physical[i].as_ptr() as usize,"page_stride":256,"pages_per_owner":3,"rows_per_page":2}}
    })).collect();
    for (d, out) in &ranks {
        let mut stale = members.clone();
        for member in &mut stale {
            member["cache"] = json!("stale");
        }
        assert!(d
            .install_state_shards(serde_json::to_vec(&stale).unwrap())
            .is_err());
        d.install_state_shards(serde_json::to_vec(&members).unwrap())
            .unwrap();
        assert!(out.release_registration().is_err());
        assert!(out.restore_registration().is_err());
    }
    let (mut mesh, group) = single_rank();
    for (round, owner) in [1usize, 0, 0, 1].into_iter().enumerate() {
        let rid = format!("request-{round}");
        let bootstrap = BootstrapInfo {
            host: "127.0.0.1".into(),
            port,
            room: 100 + round as u64,
        };
        if round == 1 {
            lib.block_writes_to(&ranks[1].0.session_id());
        }
        let (d, receive) = &mut ranks[owner];
        p.register(&rid, &bootstrap).unwrap();
        d.register(&rid, &bootstrap).unwrap();
        assert_eq!(
            next_event(d, &mut mesh, &group),
            PdEvent::Bootstrapped(rid.clone())
        );
        let mut admission = admission(&rid);
        admission.tables.insert(
            "state".into(),
            BlockTable::from_rows(&[vec![0, if round == 2 { 3 } else { 2 }]], 2),
        );
        receive.stage_remote_prefill(admission).unwrap();
        receive.hand_over_remote_prefill().unwrap();
        assert_eq!(
            next_event(&mut p, &mut mesh, &group),
            PdEvent::Bootstrapped(rid.clone())
        );
        let mut ready = remote_decode(&rid, 123, vec![]);
        let source_page = if round == 3 { 3 } else { 1 };
        ready.tables.insert(
            "state".into(),
            BlockTable::from_rows(&[vec![0, source_page]], 2),
        );
        send.stage_remote_decode(ready).unwrap();
        send.hand_over_remote_decode().unwrap();
        if round == 2 {
            let expected = [PdEvent::Failed(rid.clone()), PdEvent::Stopped(rid)];
            let (mut sender_events, mut owner_events) = (Vec::new(), Vec::new());
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while sender_events.len() < 2 || owner_events.len() < 2 {
                sender_events.extend(p.poll(&mut mesh, &group).unwrap());
                owner_events.extend(d.poll(&mut mesh, &group).unwrap());
                assert!(
                    std::time::Instant::now() < deadline,
                    "{sender_events:?}, {owner_events:?}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(sender_events, expected);
            assert_eq!(owner_events, expected);
            continue;
        }
        if round == 1 {
            assert!(lib.wait_write_entered(Duration::from_secs(5)));
            assert!(d.poll(&mut mesh, &group).unwrap().is_empty());
            // The owner has received its ordinary field. The last sibling is still writing.
            assert_eq!(&ordinary[owner][512..2048], &src[512..2048]);
            assert!(physical[1][512..768].iter().all(|&b| b == 0));
            lib.release_writes();
        }
        assert_eq!(
            next_event(&mut p, &mut mesh, &group),
            PdEvent::Succeeded(rid)
        );
        assert!(matches!(
            next_event(d, &mut mesh, &group),
            PdEvent::RemotePrefillDone(_)
        ));
        assert_eq!(&ordinary[owner][512..2048], &src[512..2048]);
        assert!(ordinary[owner][2048..].iter().all(|&b| b == 0));
        let start = (owner * 3 + 2) * 256;
        for (shard, buf) in physical.iter().enumerate() {
            for row in 0..2 {
                let source = 2048 + 512 * source_page as usize + row * 256 + shard * 128;
                assert_eq!(
                    &buf[start + row * 128..start + (row + 1) * 128],
                    &src[source..source + 128]
                );
            }
            if round == 0 {
                assert!(buf[..768].iter().all(|&b| b == 0));
            }
        }
    }
}
