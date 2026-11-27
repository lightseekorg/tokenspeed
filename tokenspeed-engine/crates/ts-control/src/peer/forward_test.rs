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

//! Real demux, eight native schedulers and a separate eight-member TCP mesh.
//! The holder controls when the valid manifest goes on the wire. Device work
//! is a sleep so the test runs on a CPU-only builder.
use super::super::{
    geometry::{DeviceField, RankGeometry},
    offers::{manifest, prefix},
    progress::Progress,
    registered::Member,
    start_endpoint,
    v1::Offer,
    PeerListener, PeerProgress, PeerSettings, Timeouts,
};
use super::*;
use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};
use std::{ffi::CString, thread, time::Duration};
use ts_collective::{Group, Listener, Mesh};
use ts_scheduler::{PeerPageView, PeerTier, RequestSpec};

fn scheduler(name: &str, budget: Duration) -> Arc<Mutex<Scheduler>> {
    std::env::set_var("PYTHONHOME", env!("TS_CONTROL_TEST_PYTHONHOME"));
    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        globals
            .set_item("ts", ts_scheduler::python_module(py).unwrap())
            .unwrap();
        globals.set_item("name", name).unwrap();
        globals
            .set_item("timeout_us", budget.as_micros() as u64)
            .unwrap();
        let code = r#"
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.P
cfg.prefix_granularity = 4
cfg.max_scheduled_tokens = 256
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = 64
cfg.num_snapshot_pages = 1
cfg.disable_l2_cache = True
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full", block_granularity=4, total_pages=64, cache_blocks_per_lcm_block=8,
    attn_kind=ts.AttnKind.Full, transfer_policy=ts.CacheTransferPolicy.FullSuffix, shard_count=8)]
cfg.peer_config = ts.PeerConfig(name=name, incarnation=1, max_leases=8, soft_pin_us=1000000,
    hard_pin_us=1000000, lookup_timeout_us=timeout_us, min_fetch_tokens=4,
    landing_tier=ts.PeerLandingTier.Device, max_host_landing_blocks=0, host_ready_timeout_us=0,
    versioned_shards=True)
result = ts.Scheduler(cfg)
"#;
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Arc::new(Mutex::new(
            Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap(),
        ))
    })
}

fn layout() -> Arc<Replica> {
    let wire = serde_json::json!({
        "plan":{"prefix_granularity":4,"lcm_block_bytes":64,"num_lcm_blocks":63,
          "groups":[{"group_id":"full","cache_blocks_per_lcm_block":1,"page_count":64}],
          "planes":[{"plane_id":"kv","bytes_per_lcm_block":64,"arena_offset_bytes":0}],
          "fields":[{"group_id":"full","field_id":"kv","plane_id":"kv","shape":[4,16],"dtype":"uint8","field_offset_bytes":0,"page_stride_bytes":64}],
          "resident_block_bytes":null},
        "group_specs":[{"group_id":"full","attn_kind":"full_history","rows_per_page":4,"entry_stride_tokens":1,
            "sliding_window_tokens":null,"replayable":false,"transfer_policy":"full_suffix","shard_count":8}],
        "transfer_schema":{"fields":[]},"field_addresses":null});
    let members = (0..8)
        .map(|rank| {
            let geometry = RankGeometry {
                session: format!("rank-{rank}"),
                device: vec![DeviceField {
                    group_id: 0,
                    field_id: "kv".into(),
                    address: 4096,
                    page_stride: 64,
                    bytes: 64,
                }],
                host: None,
                push: None,
            };
            Member {
                geometry,
                bounds: vec![64],
                regions: vec![(4096, 4096)],
            }
            .encode(rank)
        })
        .collect::<Vec<_>>();
    Arc::new(Replica::register([7; 32], wire.to_string().as_bytes(), &members).unwrap())
}

#[derive(Debug, Clone, Copy)]
enum Delivery {
    DuringForward,
    DelayedAgreement,
    Late,
}

fn exercise(delivery: Delivery, budget: Duration, start: impl FnOnce(), finish: impl FnOnce()) {
    let layout = layout();
    let mut listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    listener.set_registered_layout(layout.clone());
    let name = format!("127.0.0.1:{}", listener.port());
    let ch = listener.versioned_channel();
    let (tx, rx) = mpsc::channel();
    let holder = super::super::net::PeerNet::start("127.0.0.1", 0, tx).unwrap();
    let source = Generation {
        name: format!("127.0.0.1:{}", holder.port()),
        epoch: 1,
        incarnation: 1,
    };
    let caches = (0..8).map(|_| scheduler(&name, budget)).collect::<Vec<_>>();
    let origin = Instant::now();
    let _endpoint = start_endpoint(
        &PeerSettings {
            name: name.clone(),
            incarnation: 1,
            deployment_id: "deployment".into(),
            weight_version: "1".into(),
            timeouts: Timeouts {
                answer_us: budget.as_micros() as i64,
                quarantine_us: 10000,
                push_us: 2000000,
            },
            record_lifetime: Duration::from_secs(5),
        },
        PeerProgress::Coordinated,
        layout.members.iter().map(|m| m.geometry.clone()).collect(),
        listener,
        caches[0].clone(),
    )
    .unwrap();
    let sockets = (0..8)
        .map(|_| Listener::bind("127.0.0.1").unwrap())
        .collect::<Vec<_>>();
    let addresses = sockets
        .iter()
        .map(|s| format!("127.0.0.1:{}", s.port().unwrap()))
        .collect::<Vec<_>>();
    let meshes = thread::scope(|scope| {
        sockets
            .into_iter()
            .enumerate()
            .map(|(rank, s)| {
                let addresses = &addresses;
                scope.spawn(move || {
                    Mesh::connect(rank, s, addresses, Duration::from_secs(5)).unwrap()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut channel = Some(ch);
    let mut controls = Vec::new();
    let mut workers = Vec::new();
    let mut queries = Vec::new();
    for (i, mesh) in meshes.into_iter().enumerate() {
        let (mut c, _) = Control::new(
            origin,
            layout.clone(),
            i,
            Generation {
                name: name.clone(),
                incarnation: 1,
                epoch: 1,
            },
            channel.take(),
            Arc::default(),
            0,
        );
        c.known
            .insert(source.name.clone(), (source.clone(), layout.clone()));
        let control = Arc::new(Mutex::new(c));
        workers.push(
            Progress::start(
                Some((mesh, Group::new(0, (0..8).collect(), i).unwrap())),
                control.clone(),
                caches[i].clone(),
                None,
                origin,
            )
            .unwrap(),
        );
        controls.push(control);
        let mut scheduler = caches[i].lock();
        assert!(scheduler.enable_async_device_peer(false));
        scheduler
            .submit_requests(&[RequestSpec {
                admission_id: 0,
                request_id: "follow".into(),
                session_id: String::new(),
                cache_group_id: String::new(),
                shared_prefix_tokens: vec![],
                peer_holder: source.name.clone(),
                tokens: (1..=133).collect(),
                max_new_tokens: 1,
                max_cached_prefix_tokens: i32::MAX,
            }])
            .unwrap();
        scheduler
            .advance(&[ts_scheduler::ForwardEvent {
                kind: ts_scheduler::ForwardEventKind::PdBootstrapped,
                request_id: "follow".into(),
                tokens: vec![],
                spec_candidate_ids: vec![],
                reserve_num_tokens: 0,
                bootstrap_token: -1,
            }])
            .unwrap();
        scheduler.next_execution_plan(true, false);
        let q = scheduler.take_peer_lookups();
        assert_eq!(q.len(), 1);
        queries.push(q);
    }
    let idle = Instant::now();
    let idle_guards = caches.iter().map(|s| s.lock()).collect::<Vec<_>>();
    let idle_us = idle.elapsed().as_micros();
    drop(idle_guards);
    let issued = Instant::now();
    for (c, q) in controls.iter().zip(queries) {
        c.lock().dispatch(vec![], q, 0);
    }
    let windows = workers
        .iter()
        .enumerate()
        .map(|(rank, w)| {
            w.open(
                1,
                super::super::CacheWait::Execute,
                if rank == 7 && matches!(delivery, Delivery::DelayedAgreement) {
                    1
                } else {
                    0
                },
            )
        })
        .collect::<Vec<_>>();
    start();
    let frames = rx.recv_timeout(Duration::from_secs(3)).unwrap();
    let Some(Message::Lookup(k, _, _, _, prompt)) = Message::decode(&frames[1], false) else {
        panic!("expected lookup")
    };
    let pages = (0..32)
        .map(|slot| PeerPageView {
            group_id: 0,
            slot,
            tier: PeerTier::Device,
            block_id: slot + 1,
        })
        .collect::<Vec<_>>();
    let reply = Message::Offered(
        k,
        Some((
            Offer {
                id: 1,
                end: 128,
                manifest: manifest(&pages),
            },
            prefix(&prompt.tokens[..128]),
            pages,
        )),
    );
    if matches!(delivery, Delivery::Late) {
        thread::sleep(budget + Duration::from_millis(10));
    }
    holder.versioned(&name, &reply).unwrap();
    let sent = issued.elapsed();
    finish();
    // No foreground agreement or scheduler tick until the forward completes.
    assert!(issued.elapsed() > budget);
    // Foreground ranks reacquire their cache locks before closing the window.
    let resume = Instant::now();
    let mut schedulers = caches.iter().map(|s| s.lock()).collect::<Vec<_>>();
    let resume_us = resume.elapsed().as_micros();
    drop(windows);
    eprintln!("lock_sample case={delivery:?} idle_us={idle_us} resume_us={resume_us}");
    let reports = controls
        .iter()
        .map(|c| c.lock().report(vec![]))
        .collect::<Vec<_>>();
    let now = origin.elapsed().as_micros() as i64;
    for (rank, ((c, w), s)) in controls
        .iter()
        .zip(&workers)
        .zip(&mut schedulers)
        .enumerate()
    {
        let mut round = PeerRound {
            now_us: now,
            ..Default::default()
        };
        w.drain(&mut round).unwrap();
        assert_eq!(
            round.events.len(),
            usize::from(!matches!(delivery, Delivery::DelayedAgreement)),
            "background rank {rank}, {delivery:?}"
        );
        c.lock()
            .advance(Control::unpack(&reports).unwrap().1, s, None, &mut round)
            .unwrap();
        assert_eq!(round.events.len(), 1);
        let hit = !matches!(delivery, Delivery::Late);
        assert_eq!(round.events[0].num_tokens, if hit { 128 } else { 0 });
        eprintln!("case={delivery:?} rank={rank} timeout_ms={} holder_send_ms={:.2} scheduler_delivery_ms={:.2} tokens={}",budget.as_millis(), sent.as_secs_f64()*1000.0,issued.elapsed().as_secs_f64()*1000.0,round.events[0].num_tokens);
        s.advance_peer(now, &round.events).unwrap();
        let plan = s.next_execution_plan(true, false);
        assert_eq!(
            plan.peer_fetches().len(),
            usize::from(hit),
            "scheduler rank {rank}, {delivery:?}"
        );
    }
    drop(schedulers);
}
#[test]
fn eight_schedulers_keep_timely_sharded_answers_across_a_long_forward() {
    for delivery in [
        Delivery::DuringForward,
        Delivery::DelayedAgreement,
        Delivery::Late,
    ] {
        exercise(
            delivery,
            Duration::from_millis(300),
            || {},
            || thread::sleep(Duration::from_millis(400)),
        );
    }
}

#[test]
#[ignore = "requires eight visible Ascend NPUs, torch_npu and the MLP runtime environment"]
fn eight_npu_forwards_outlast_the_sharded_lookup_deadline() {
    use pyo3::types::PyModule;
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
    std::env::set_var("PYTHONHOME", env!("TS_CONTROL_TEST_PYTHONHOME"));
    Python::initialize();
    let module = Python::attach(|py| {
        PyModule::from_code(
            py,
            &CString::new(include_str!("../../tests/peer_forward_npu.py")).unwrap(),
            c"peer_forward_npu.py",
            c"peer_forward_npu",
        )
        .unwrap()
        .unbind()
    });
    for delivery in [
        Delivery::DuringForward,
        Delivery::DelayedAgreement,
        Delivery::Late,
    ] {
        Python::attach(|py| {
            module.call_method0(py, "arm").unwrap();
        });
        exercise(
            delivery,
            Duration::from_millis(50),
            || {
                Python::attach(|py| {
                    module.call_method0(py, "begin").unwrap();
                })
            },
            || {
                Python::attach(|py| {
                    let timings: Vec<f64> = module
                        .call_method0(py, "finish")
                        .unwrap()
                        .extract(py)
                        .unwrap();
                    eprintln!("case={delivery:?} device_forward_ms={timings:?}");
                    assert_eq!(timings.len(), 8);
                    assert!(timings.iter().all(|ms| *ms > 50.0));
                })
            },
        );
    }
}
