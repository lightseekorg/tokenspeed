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

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use ts_control::peer::registered::{Member, Replica as RegisteredReplica};

const LOCAL_PAGES: u64 = 33;
const GRAIN: i32 = 8 * PAGE;
const STRIDE: u64 = 8192;
const HOLDER: u64 = 8192;
const TARGET: u64 = 100_000;
const SOURCE: u64 = 200_000;

struct Node {
    peer: Peer,
    scheduler: Arc<Mutex<Scheduler>>,
}
struct Group {
    layout: Arc<RegisteredReplica>,
    wire: Vec<u8>,
    name: String,
    nodes: Vec<Node>,
    events: Vec<ts_scheduler::PeerEventView>,
}

fn group_scheduler(name: &str, width: usize, storage: bool) -> Scheduler {
    interpreter();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.{role}
if cfg.role == ts.SchedulerConfig.Role.P:
    cfg.decode_allocation_policy = ts.SchedulerConfig.DecodeAllocationPolicy.Deferred
cfg.prefix_granularity = {GRAIN}
cfg.max_scheduled_tokens = 512
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = {physical}
cfg.num_snapshot_pages = 1
cfg.disable_l2_cache = True
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity={PAGE}, cache_blocks_per_lcm_block=8, total_pages={pages}, shard_count={width}, attn_kind=ts.AttnKind.Full, transfer_policy=ts.CacheTransferPolicy.FullSuffix)]
cfg.peer_config = ts.PeerConfig(name="{name}", incarnation=1, max_leases=64, soft_pin_us=10_000_000, hard_pin_us=10_000_000, lookup_timeout_us=2_000_000, min_fetch_tokens={PAGE}, landing_tier=ts.PeerLandingTier.Device, max_host_landing_blocks=0, host_ready_timeout_us=0, versioned_shards={versioned}, storage_min_fetch_tokens={minimum}, storage_max_tokens={maximum}, storage_lookup_timeout_us={timeout})
result = ts.Scheduler(cfg)
"#,
            role = if width == 8 { "P" } else { "Fused" },
            physical = LOCAL_PAGES,
            pages = 1 + (LOCAL_PAGES - 1) * width as u64,
            versioned = if width == 8 { "True" } else { "False" },
            minimum = if storage { GRAIN } else { 0 },
            maximum = if storage { 2 * GRAIN } else { 0 },
            timeout = if storage { 2_000_000 } else { 0 }
        );
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

impl Group {
    fn new(width: usize, base: u64, storage: bool, reader: Arc<dyn PageReader>) -> Self {
        let mut listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let name = format!("127.0.0.1:{}", listener.port());
        let geometries: Vec<_> = (0..width)
            .map(|rank| {
                let mut geometry = geometry_with_host(base + rank as u64 * STRIDE, 0);
                geometry.session = format!("{name}/{rank}");
                geometry
            })
            .collect();
        let wire = serde_json::json!({
            "plan":{"prefix_granularity":GRAIN,"lcm_block_bytes":PAGE_BYTES,"num_lcm_blocks":LOCAL_PAGES-1,
                "groups":[{"group_id":"full_attention","cache_blocks_per_lcm_block":8,"page_count":LOCAL_PAGES}],
                "planes":[{"plane_id":"kv","bytes_per_lcm_block":PAGE_BYTES,"arena_offset_bytes":0}],
                "fields":[{"group_id":"full_attention","field_id":"layer.0.kv","plane_id":"kv","shape":[PAGE,PAGE_BYTES/PAGE as u64],"dtype":"uint8","field_offset_bytes":0,"page_stride_bytes":PAGE_BYTES}],"resident_block_bytes":null},
            "group_specs":[{"group_id":"full_attention","attn_kind":"full_history","rows_per_page":PAGE,"entry_stride_tokens":1,"sliding_window_tokens":null,"replayable":false,"transfer_policy":"full_suffix","shard_count":width}],
            "transfer_schema":{"fields":[]},"field_addresses":null
        });
        let members: Vec<_> = geometries
            .iter()
            .enumerate()
            .map(|(rank, g)| {
                Member {
                    geometry: g.clone(),
                    bounds: vec![(1 + (LOCAL_PAGES - 1) * width as u64) as i64],
                    regions: vec![(g.device[0].address, LOCAL_PAGES * PAGE_BYTES)],
                }
                .encode(rank)
            })
            .collect();
        let layout = Arc::new(
            RegisteredReplica::register([7; 32], wire.to_string().as_bytes(), &members).unwrap(),
        );
        listener.set_registered_layout(layout.clone());
        let mut channel = Some(listener.versioned_channel());
        let mut listener = Some(listener);
        let nodes = (0..width)
            .map(|rank| {
                let scheduler = Arc::new(Mutex::new(group_scheduler(&name, width, storage)));
                let mut settings = settings(&name, "v1").peer;
                settings.incarnation = 1;
                let mut peer = Peer::start(
                    settings,
                    PeerProgress::Coordinated,
                    rank,
                    geometries.clone(),
                    listener.take(),
                    scheduler.clone(),
                    PeerTransfers {
                        reader: reader.clone(),
                        pusher: Arc::new(NoPushes),
                    },
                )
                .unwrap();
                peer.enable_versioned(layout.clone(), name.clone(), 1, channel.take(), 0);
                Node { peer, scheduler }
            })
            .collect();
        Self {
            layout,
            wire: wire.to_string().into_bytes(),
            name,
            nodes,
            events: Vec::new(),
        }
    }
    fn submit(&self, rid: &str, tokens: Vec<i32>, holder: &str) {
        for node in &self.nodes {
            node.scheduler
                .lock()
                .submit_requests(&[spec(rid, tokens.clone(), holder)])
                .unwrap();
        }
    }
    fn round(&mut self, signal: bool) -> (Vec<(String, i32)>, Vec<PeerFetchView>) {
        let reports: Vec<_> = self.nodes.iter_mut().map(|n| n.peer.report()).collect();
        let mut result = (Vec::new(), Vec::new());
        for (rank, node) in self.nodes.iter_mut().enumerate() {
            let mut scheduler = node.scheduler.lock();
            let round = node
                .peer
                .agree_with_scheduler(&reports, &mut scheduler)
                .unwrap();
            scheduler.advance_peer(round.now_us, &round.events).unwrap();
            if rank == 0 {
                self.events.extend(round.events);
            }
            let plan = scheduler.next_execution_plan(true, false);
            let fetches = plan.peer_fetches();
            let lookups = scheduler.take_peer_lookups();
            drop(scheduler);
            node.peer
                .dispatch(fetches.clone(), lookups, round.now_us)
                .unwrap();
            if signal && !fetches.is_empty() {
                node.peer.fence().ready();
            }
            if rank == 0 {
                result.1 = fetches;
                result.0 = plan
                    .forward_batches()
                    .into_iter()
                    .flat_map(|b| {
                        b.request_ids
                            .into_iter()
                            .zip(b.extend_prefix_lens)
                            .take(b.num_extends as usize)
                            .collect::<Vec<_>>()
                    })
                    .collect();
            }
        }
        result
    }
    fn finish(&self, rid: &str) {
        for node in &self.nodes {
            node.scheduler
                .lock()
                .advance(&[
                    event(ForwardEventKind::ExtendResult, rid, vec![7]),
                    event(ForwardEventKind::Finish, rid, vec![]),
                ])
                .unwrap();
        }
    }
    fn fence(&self) {
        for node in &self.nodes {
            node.peer.fence().ready();
        }
    }
}

struct GatedRead {
    arena: Arena,
    started: Arc<AtomicUsize>,
    hot_done: Arc<AtomicUsize>,
    release: Arc<AtomicBool>,
    fail_storage: bool,
}
impl PageReader for GatedRead {
    fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
        self.arena.read(session, spans)?;
        if !spans.is_empty() {
            self.hot_done.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }
    fn read_outcome(&self, session: &str, spans: &Spans) -> ts_control::peer::read::ReadOutcome {
        use ts_control::peer::read::ReadOutcome;
        if session == "storage-provider" && !spans.is_empty() {
            self.started.fetch_add(1, Ordering::SeqCst);
            let until = Instant::now() + Duration::from_secs(5);
            while !self.release.load(Ordering::SeqCst) && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(1));
            }
            if self.fail_storage {
                return ReadOutcome::FailedRetired;
            }
        }
        self.arena.read_outcome(session, spans)
    }
}

fn check_landing(hot: bool, missing: bool, fail_storage: bool, cross_layout: bool) {
    let arena = Arena::new(300_000);
    let started = Arc::new(AtomicUsize::new(0));
    let hot_done = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(AtomicBool::new(false));
    let mut holder = Group::new(1, HOLDER, false, Arc::new(arena.clone()));
    let tokens: Vec<_> = (1..=2 * GRAIN + PAGE).collect();
    for block in 1..LOCAL_PAGES {
        arena.write(
            HOLDER + block * PAGE_BYTES,
            &vec![block as u8; PAGE_BYTES as usize],
        );
    }
    if hot {
        holder.submit("seed", tokens[..GRAIN as usize].to_vec(), "");
        assert_eq!(holder.round(true).0, vec![("seed".into(), 0)]);
        holder.finish("seed");
        holder.round(true);
    }
    let mut reader = Group::new(
        8,
        TARGET,
        true,
        Arc::new(GatedRead {
            arena: arena.clone(),
            started: started.clone(),
            hot_done: hot_done.clone(),
            release: release.clone(),
            fail_storage,
        }),
    );
    let layouts: Vec<_> = (0..8)
        .map(|rank| HostLayout {
            groups: vec![GroupLayout {
                packing: 8,
                block_bytes: PAGE_BYTES as usize,
                shard_count: 8,
            }],
            lcm_block_bytes: PAGE_BYTES as usize,
            num_lcm_blocks: (LOCAL_PAGES - 1) as usize,
            owner_rank: rank,
            tp_rank: rank,
        })
        .collect();
    let candidate = PrefixPlan::from_scheduler(
        &reader.nodes[0].scheduler.lock(),
        &tokens,
        2 * GRAIN,
        "test-namespace",
        &layouts,
        256,
    )
    .unwrap()
    .unwrap();
    let candidate = if cross_layout {
        let source = ts_control::peer::stored_layout::StoredLayout::from_decode(
            &holder.layout,
            &holder.wire,
            &ts_l3::Namespace {
                deployment_id: "decode-only-namespace".into(),
                has_draft: false,
            },
            "v1",
        )
        .unwrap();
        let metadata = reader.nodes[0]
            .scheduler
            .lock()
            .storage_prefix_plan(&tokens, 2 * GRAIN, 256)
            .unwrap();
        source
            .plan(&metadata, &reader.layout, "v1", 256)
            .unwrap()
            .unwrap()
    } else {
        candidate
    };
    let mut objects = HashMap::new();
    for object in &candidate.objects {
        if missing || (hot && object.slot < 8) {
            continue;
        }
        let address = SOURCE + object.slot as u64 * PAGE_BYTES;
        arena.write(address, &vec![100 + object.slot as u8; PAGE_BYTES as usize]);
        objects.insert(object.key.clone(), (address, PAGE_BYTES));
    }
    let live = Arc::new(Mutex::new(0));
    let queries = Arc::new(Mutex::new(0));
    let prepared = Arc::new(Mutex::new(Vec::new()));
    let budget = SourceBudget::new(16 * PAGE_BYTES).unwrap();
    let provider = Provider {
        objects,
        live: live.clone(),
        queries: queries.clone(),
        prepared_keys: prepared.clone(),
    };
    let local = reader.nodes[0].scheduler.clone();
    let service = storage_servicer::Service::start(
        storage_servicer::Settings {
            stamp: Stamp {
                name: format!("l3@{}", reader.name),
                incarnation: 1,
                identity: "deployment/v1".into(),
            },
            weight_version: "v1".into(),
            members: 8,
            limits: Limits {
                max_query_keys: 256,
                max_candidates: 1,
                max_prepare_bytes: 16 * PAGE_BYTES,
                max_source_batch_bytes: 16 * PAGE_BYTES,
                max_source_batches: 1,
            },
            max_records: 8,
            min_extra_tokens: GRAIN,
            record_lifetime: Duration::from_secs(10),
            read_budget: Duration::from_secs(1),
        },
        Arc::new(move |_, _| Ok(vec![candidate.clone()])),
        Arc::new(move |p| {
            local
                .lock()
                .local_prefix_tokens(&p.tokens, p.max_tokens)
                .map_err(|e| Error(e.to_string()))
        }),
        provider,
        budget.clone(),
    )
    .unwrap();
    reader.nodes[0].peer.attach_storage_source(service);
    reader.submit("request", tokens, &holder.name);
    let until = Instant::now() + Duration::from_secs(5);
    let mut fetches = Vec::new();
    while fetches.is_empty() && Instant::now() < until {
        holder.round(true);
        let (rows, parts) = reader.round(false);
        assert!(rows.is_empty(), "must perform lookup before compute");
        fetches.extend(parts);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!fetches.is_empty());
    for _ in 0..10 {
        holder.round(true);
        assert!(reader.round(false).0.is_empty());
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        started.load(Ordering::SeqCst),
        0,
        "L3 may not write before the plan fence"
    );
    reader.fence();
    if !missing {
        while started.load(Ordering::SeqCst) == 0 && Instant::now() < until {
            holder.round(true);
            assert!(reader.round(false).0.is_empty());
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            started.load(Ordering::SeqCst) > 0,
            "registered reader must reach the L3 backend"
        );
        if hot {
            while hot_done.load(Ordering::SeqCst) < 8 && Instant::now() < until {
                holder.round(true);
                assert!(reader.round(false).0.is_empty());
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(
                hot_done.load(Ordering::SeqCst),
                8,
                "hot reads must finish while L3 remains blocked"
            );
            assert_eq!(
                *live.lock(),
                1,
                "blocked L3 must retain its prepared source"
            );
        }
    }
    release.store(true, Ordering::SeqCst);
    let mut rows = Vec::new();
    while rows.is_empty() && Instant::now() < until {
        holder.round(true);
        rows = reader.round(true).0;
        std::thread::sleep(Duration::from_millis(1));
    }
    let expected = if fail_storage {
        0
    } else if missing {
        GRAIN
    } else {
        2 * GRAIN
    };
    assert_eq!(rows, vec![("request".into(), expected)]);
    assert_eq!(*queries.lock(), 1);
    assert_eq!(fetches.len(), if hot && !missing { 2 } else { 1 });
    if !fail_storage {
        for fetch in &fetches {
            for page in &fetch.destinations {
                let rank = (page.block_id as u64 - 1) % 8;
                let local = 1 + (page.block_id as u64 - 1) / 8;
                let address = TARGET + rank * STRIDE + local * PAGE_BYTES;
                let actual = arena.read_bytes(address, PAGE_BYTES as usize);
                if fetch.holder.starts_with("l3@") {
                    assert_eq!(actual, vec![100 + page.slot as u8; PAGE_BYTES as usize]);
                } else {
                    assert_ne!(actual, vec![0; PAGE_BYTES as usize]);
                }
            }
        }
    }
    assert_eq!(*live.lock(), 0);
    assert_eq!(budget.usage(), (0, 0));
    let done: Vec<_> = reader
        .events
        .iter()
        .filter(|e| e.kind == ts_scheduler::PeerEventKind::FetchDone && e.request_id == "request")
        .collect();
    assert_eq!(
        done.len(),
        1,
        "one request must have exactly one completion"
    );
    for node in &reader.nodes {
        node.scheduler
            .lock()
            .advance(&[
                event(ForwardEventKind::ExtendResult, "request", vec![7]),
                event(ForwardEventKind::Abort, "request", vec![]),
            ])
            .unwrap();
    }
    reader.round(true);
    holder.round(true);
    for node in &reader.nodes {
        assert_eq!(node.peer.num_landings(), 0);
    }
}

#[test]
fn versioned_reader_restores_l3_without_a_hot_hit() {
    check_landing(false, false, false, false);
}
#[test]
fn versioned_reader_keeps_hot_prefix_after_l3_miss() {
    check_landing(true, true, false, false);
}
#[test]
fn versioned_reader_composes_peer_and_l3_in_parallel() {
    check_landing(true, false, false, false);
}
#[test]
fn versioned_reader_recomputes_after_retired_l3_failure() {
    check_landing(true, false, true, false);
}

#[test]
fn versioned_reader_restores_decode_namespace_without_hot_data() {
    check_landing(false, false, false, true);
}
#[test]
fn versioned_reader_combines_hot_peer_with_decode_storage_namespace() {
    check_landing(true, false, false, true);
}
