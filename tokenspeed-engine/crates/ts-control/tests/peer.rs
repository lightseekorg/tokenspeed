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

//! Two replicas sharing a prefix through their peer endpoints on this host:
//! real schedulers, real sockets, and memory both replicas address through
//! one arena (`docs/rfcs/0004-peer-prefix-sharing.md`).

use std::{
    ffi::CString,
    sync::{mpsc, Arc, Once},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};
use ts_control::peer::{geometry::Spans, read::PageReader};
use ts_control::peer::{
    geometry::{DeviceField, HostGroup, HostPool, RankGeometry},
    landing::Timeouts,
    net::{Inbound, PeerNet},
    push::{PagePusher, PushItem},
    read::Arena,
    wire::{Reply, Request},
    Peer, PeerListener, PeerSettings,
};
use ts_scheduler::{
    ForwardEvent, ForwardEventKind, PeerFetchView, PeerTier, RequestSpec, Scheduler,
};

const PAGE: i32 = 16;
const PAGE_BYTES: u64 = 64;
const PAGES: u64 = 64;
const HOLDER_BASE: u64 = 0;
const READER_BASE: u64 = 100_000;
/// A Host-landing reader's Host pool: one page per block.
const READER_HOST_BASE: u64 = 150_000;

fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        std::env::set_var("PYTHONHOME", env!("TS_CONTROL_TEST_PYTHONHOME"));
        Python::initialize();
        Python::attach(|py| {
            let module = ts_scheduler::python_module(py).unwrap();
            py.import("sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item("tokenspeed_scheduler.tokenspeed_scheduler_ext", module)
                .unwrap();
            let package = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../tokenspeed-scheduler/python"
            );
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1("insert", (0, package))
                .unwrap();
        });
    });
}

/// Where a replica lands peer prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Landing {
    Device,
    /// Into a Host pool of `PAGES` one-page blocks, before admission.
    Host,
}

/// A replica's scheduler in `role` (`Fused` or `P`), landing as `landing`.
fn scheduler(
    name: &str,
    incarnation: u64,
    role: &str,
    landing: Landing,
    timeout_us: i64,
    pages: u64,
    soft_pin_us: i64,
) -> Scheduler {
    interpreter();
    let host = match landing {
        Landing::Device => "cfg.disable_l2_cache = True".to_owned(),
        Landing::Host => format!(
            "cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = {pages}\n\
             cfg.host_group_packing = [1]\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack"
        ),
    };
    let tier = match landing {
        Landing::Device => {
            "landing_tier=ts.PeerLandingTier.Device, max_host_landing_blocks=0, host_ready_timeout_us=0"
                .to_owned()
        }
        Landing::Host => format!(
            "landing_tier=ts.PeerLandingTier.Host, max_host_landing_blocks={}, \
             host_ready_timeout_us=1_000_000",
            pages - 1
        ),
    };
    Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.{role}
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = max(512, {pages} * {PAGE})
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = {pages}
cfg.num_snapshot_pages = 1
{host}
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity={PAGE},
                                        total_pages={pages}, attn_kind=ts.AttnKind.Full,
                                        transfer_policy=ts.CacheTransferPolicy.FullSuffix)]
cfg.peer_config = ts.PeerConfig(name="{name}", incarnation={incarnation}, max_leases=8,
                                soft_pin_us={soft_pin_us}, hard_pin_us=1_000_000,
                                lookup_timeout_us={timeout_us}, min_fetch_tokens={PAGE},
                                {tier})
result = ts.Scheduler(cfg)
"#
        );
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

fn geometry(base: u64) -> RankGeometry {
    RankGeometry {
        session: "local".into(),
        device: vec![DeviceField {
            group_id: 0,
            field_id: "layer.0.kv".into(),
            address: base,
            page_stride: PAGE_BYTES,
            bytes: PAGE_BYTES,
        }],
        host: None,
        push: None,
    }
}

/// A Host-landing reader's geometry: its Device fields and a Host pool of
/// one-page blocks at `host_base`.
fn geometry_with_host(base: u64, host_base: u64) -> RankGeometry {
    RankGeometry {
        host: Some(HostPool {
            address: host_base,
            groups: vec![HostGroup {
                parent_bytes: PAGE_BYTES,
                packing: 1,
                block_bytes: PAGE_BYTES,
                field_offsets: vec![("layer.0.kv".into(), 0)],
            }],
        }),
        ..geometry(base)
    }
}

/// A replica whose holders never push: none of them has a Disk tier.
struct NoPushes;

impl PagePusher for NoPushes {
    fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
        Err("no holder pushes".into())
    }
}

fn settings(name: &str, weight_version: &str) -> PeerSettings {
    PeerSettings {
        autonomous_host: false,
        device_admission: None,
        grant_first: false,
        checkpoint_pin: false,
        name: name.to_owned(),
        incarnation: 1,
        deployment_id: "deployment".into(),
        weight_version: weight_version.to_owned(),
        timeouts: Timeouts {
            answer_us: 1_000_000,
            quarantine_us: 10_000,
            push_us: 1_000_000,
        },
        record_lifetime: Duration::from_secs(5),
    }
}

/// One single-member replica.
struct Replica {
    name: String,
    scheduler: Arc<Mutex<Scheduler>>,
    peer: Peer,
    decisions: Vec<(String, &'static str)>,
    landed: Vec<(String, i32, i32)>,
}

impl Replica {
    fn start(weight_version: &str, base: u64, arena: &Arena) -> Self {
        Self::start_as(weight_version, base, arena, "Fused")
    }

    fn start_as(weight_version: &str, base: u64, arena: &Arena, role: &str) -> Self {
        let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let name = format!("127.0.0.1:{}", listener.port());
        Self::with_listener(settings(&name, weight_version), listener, base, arena, role)
    }

    fn with_listener(
        settings: PeerSettings,
        listener: PeerListener,
        base: u64,
        arena: &Arena,
        role: &str,
    ) -> Self {
        Self::with_landing(
            settings,
            listener,
            geometry(base),
            arena,
            role,
            Landing::Device,
        )
    }

    /// A Fused replica landing into Host, its pool at `host_base`.
    fn start_host_landing(weight_version: &str, base: u64, host_base: u64, arena: &Arena) -> Self {
        let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let name = format!("127.0.0.1:{}", listener.port());
        Self::with_landing(
            settings(&name, weight_version),
            listener,
            geometry_with_host(base, host_base),
            arena,
            "Fused",
            Landing::Host,
        )
    }

    fn with_landing(
        settings: PeerSettings,
        listener: PeerListener,
        geometry: RankGeometry,
        arena: &Arena,
        role: &str,
        landing: Landing,
    ) -> Self {
        Self::with_reader(
            settings,
            listener,
            geometry,
            Arc::new(arena.clone()),
            role,
            landing,
        )
    }

    fn with_reader(
        settings: PeerSettings,
        listener: PeerListener,
        geometry: RankGeometry,
        reader: Arc<dyn PageReader>,
        role: &str,
        landing: Landing,
    ) -> Self {
        let pages = if settings.grant_first { 512 } else { PAGES };
        Self::with_reader_pages(
            settings, listener, geometry, reader, role, landing, pages, 1_000_000,
        )
    }

    fn with_reader_pages(
        settings: PeerSettings,
        listener: PeerListener,
        geometry: RankGeometry,
        reader: Arc<dyn PageReader>,
        role: &str,
        landing: Landing,
        pages: u64,
        soft_pin_us: i64,
    ) -> Self {
        let scheduler = Arc::new(Mutex::new(scheduler(
            &settings.name,
            1,
            role,
            landing,
            settings.timeouts.answer_us,
            pages,
            soft_pin_us,
        )));
        let name = settings.name.clone();
        let peer = Peer::start(
            settings,
            0,
            vec![geometry],
            Some(listener),
            scheduler.clone(),
            reader,
            Arc::new(NoPushes),
        )
        .unwrap();
        Self {
            name,
            scheduler,
            peer,
            decisions: Vec::new(),
            landed: Vec::new(),
        }
    }

    fn enable_logical_metrics(&mut self) {
        let exporter = ts_metrics::Exporter::new(&Default::default()).unwrap();
        let metrics = ts_metrics::PeerMetrics::new(
            &exporter,
            &ts_metrics::EngineLabels {
                model_name: "test".into(),
                app_key: String::new(),
                dp_rank: 0,
            },
        );
        self.peer.set_metrics(Some(Arc::new(metrics)), None);
    }

    /// One round as the control loop runs it: the agreement at the head,
    /// then the plan, its landings and lookups dispatched, and its device
    /// work reported done at once (its reads start with the next
    /// agreement). Returns the plan's prefill rows, as (request, prefix
    /// extended), and its landings.
    fn round(&mut self) -> (Vec<(String, i32)>, Vec<PeerFetchView>) {
        let (rows, fetches, _) = self.round_with_device();
        (rows, fetches)
    }

    /// `round`, and whether its plan went to the data plane.
    fn round_with_device(&mut self) -> (Vec<(String, i32)>, Vec<PeerFetchView>, bool) {
        let report = self.peer.report();
        let round = self.peer.agree(&[report]).unwrap();
        self.decisions.extend(round.decisions);
        self.landed.extend(round.landed);
        let mut scheduler = self.scheduler.lock();
        scheduler.advance_peer(round.now_us, &round.events).unwrap();
        let plan = scheduler.next_execution_plan(true, false);
        let fetches = plan.peer_fetches();
        let lookups = scheduler.take_peer_lookups();
        drop(scheduler);
        self.peer.dispatch(fetches.clone(), lookups, round.now_us);
        // The data plane reports the fence of every plan it runs, as the
        // control loop hands it the plans that need the device.
        let needs_device = plan.needs_device();
        if needs_device && !fetches.is_empty() {
            self.peer.fence().ready();
        }
        let rows = plan
            .forward_batches()
            .into_iter()
            .flat_map(|batch| {
                let extends = batch.num_extends;
                batch
                    .request_ids
                    .into_iter()
                    .zip(batch.extend_prefix_lens)
                    .take(extends)
                    .collect::<Vec<_>>()
            })
            .collect();
        (rows, fetches, needs_device)
    }

    fn advance(&self, events: &[ForwardEvent]) {
        self.scheduler.lock().advance(events).unwrap();
    }
}

fn event(kind: ForwardEventKind, request_id: &str, tokens: Vec<i32>) -> ForwardEvent {
    ForwardEvent {
        kind,
        request_id: request_id.to_owned(),
        tokens,
        spec_candidate_ids: Vec::new(),
        reserve_num_tokens: 0,
        bootstrap_token: -1,
    }
}

fn spec(request_id: &str, tokens: Vec<i32>, holder: &str) -> RequestSpec {
    RequestSpec {
        admission_id: 0,
        request_id: request_id.to_owned(),
        session_id: String::new(),
        cache_group_id: String::new(),
        shared_prefix_tokens: Vec::new(),
        peer_holder: holder.to_owned(),
        peer_expected_tokens: 0,
        tokens,
        max_new_tokens: 4,
        max_cached_prefix_tokens: i32::MAX,
    }
}

/// The holder computes `tokens` and finishes, publishing them; each of its
/// pages holds its block id as a byte pattern.
fn holder_computes(holder: &mut Replica, tokens: &[i32], arena: &Arena) {
    for block in 1..PAGES {
        arena.write(
            HOLDER_BASE + block * PAGE_BYTES,
            &[block as u8; PAGE_BYTES as usize],
        );
    }
    holder
        .scheduler
        .lock()
        .submit_requests(&[spec("h", tokens.to_vec(), "")])
        .unwrap();
    let (rows, _) = holder.round();
    assert_eq!(rows, vec![("h".to_owned(), 0)]);
    holder.advance(&[
        event(ForwardEventKind::ExtendResult, "h", vec![7]),
        event(ForwardEventKind::Finish, "h", Vec::new()),
    ]);
    holder.round();
}

/// Runs both replicas' rounds until the reader prefills `request`; returns
/// the prefix it extends and the landings the reader's plans carried.
fn rounds_until_prefill(
    reader: &mut Replica,
    holder: Option<&mut Replica>,
    request: &str,
) -> (i32, Vec<PeerFetchView>) {
    let mut holder = holder;
    let mut landings = Vec::new();
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        if let Some(holder) = holder.as_deref_mut() {
            holder.round();
        }
        let (rows, fetches) = reader.round();
        landings.extend(fetches);
        if let Some((_, prefix)) = rows.into_iter().find(|(rid, _)| rid == request) {
            return (prefix, landings);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("the reader never prefilled {request}");
}

#[test]
fn a_listener_keeps_its_port_reserved_through_peer_startup() {
    let arena = Arena::new(200_000);
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let port = listener.port();
    let name = format!("127.0.0.1:{port}");
    assert_eq!(
        std::net::TcpListener::bind(&name).unwrap_err().kind(),
        std::io::ErrorKind::AddrInUse
    );
    let replica = Replica::with_listener(
        settings(&name, "v1"),
        listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );
    assert_eq!(replica.peer.port(), Some(port));
    assert_eq!(
        std::net::TcpListener::bind(&name).unwrap_err().kind(),
        std::io::ErrorKind::AddrInUse
    );
}

#[test]
fn two_dp_replicas_read_through_one_public_port() {
    for public_reads in [true, false] {
        read_through_node_port(public_reads);
    }
}

fn node_listeners() -> [PeerListener; 2] {
    use ts_collective::{Listener, Mesh};

    let mesh_listeners = [(); 2].map(|_| Listener::bind("127.0.0.1").unwrap());
    let addresses = mesh_listeners
        .each_ref()
        .map(|socket| format!("127.0.0.1:{}", socket.port().unwrap()));
    std::thread::scope(|scope| {
        let addresses = &addresses;
        let spawn = |rank, socket| {
            scope.spawn(move || {
                let mut mesh =
                    Mesh::connect(rank, socket, addresses, Duration::from_secs(5)).unwrap();
                PeerListener::bind_node(&mut mesh, 4, &[(0, 16), (1, 31)], "127.0.0.1", 0).unwrap()
            })
        };
        let [a, b] = mesh_listeners;
        [spawn(0, a), spawn(1, b)].map(|worker| worker.join().unwrap())
    })
}

#[test]
fn sibling_receives_requests_while_public_scheduler_is_locked() {
    let arena = Arena::new(200_000);
    let [public, private] = node_listeners();
    let address = format!("127.0.0.1:{}", public.port());
    let replica = |rank, listener, base| {
        Replica::with_listener(
            settings(&format!("{address}/{rank}"), "v1"),
            listener,
            base,
            &arena,
            "Fused",
        )
    };
    let owner = replica(16, public, READER_BASE);
    let sibling = replica(31, private, HOLDER_BASE);
    let (incoming, replies) = mpsc::channel();
    let client = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let layout = Request::Layout {
        reply_to: format!("127.0.0.1:{}", client.port()),
    };
    let receive_holder = || {
        let frames = replies.recv_timeout(Duration::from_secs(5)).unwrap();
        let Inbound::Reply(Reply::Layout { stamp, .. }) = Inbound::decode(&frames).unwrap() else {
            panic!("expected a layout reply");
        };
        stamp.name
    };
    // Both messages enter the same public socket in FIFO order.
    let lock = owner.scheduler.lock();
    client.request(&owner.name, &layout).unwrap();
    client.request(&sibling.name, &layout).unwrap();
    assert_eq!(receive_holder(), sibling.name);
    drop(lock);
    assert_eq!(receive_holder(), owner.name);
}

fn read_through_node_port(public_reads: bool) {
    let arena = Arena::new(200_000);
    let [public, private] = node_listeners();
    let address = format!("127.0.0.1:{}", public.port());
    let (reader_listener, reader_rank, holder_listener, holder_rank) = if public_reads {
        (public, 16, private, 31)
    } else {
        (private, 31, public, 16)
    };
    let mut reader = Replica::with_listener(
        settings(&format!("{address}/{reader_rank}"), "v1"),
        reader_listener,
        READER_BASE,
        &arena,
        "Fused",
    );
    let mut holder = Replica::with_listener(
        settings(&format!("{address}/{holder_rank}"), "v1"),
        holder_listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    let mut longer = tokens;
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!(landings.len(), 1);
    assert!(prefix >= 5 * PAGE);
    assert_eq!(landings[0].end_tokens, prefix);
    for destination in &landings[0].destinations {
        let page = arena.read_bytes(
            READER_BASE + destination.block_id as u64 * PAGE_BYTES,
            PAGE_BYTES as usize,
        );
        assert!(page[0] != 0 && page.iter().all(|&byte| byte == page[0]));
    }
}

#[test]
fn independent_device_keeps_timely_answers_and_waits_for_plan_fence() {
    independent_device_choice(false);
}

#[test]
fn independent_device_recomputes_a_short_extension() {
    independent_device_choice(true);
}

fn independent_device_choice(adaptive: bool) {
    let arena = Arena::new(200_000);
    let replica = |base| {
        let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let mut cfg = settings(&format!("127.0.0.1:{}", listener.port()), "v1");
        cfg.device_admission = Some(adaptive);
        cfg.timeouts.answer_us = 50_000;
        Replica::with_listener(cfg, listener, base, &arena, "Fused")
    };
    let mut holder = replica(HOLDER_BASE);
    let mut reader = replica(READER_BASE);
    let mut tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    tokens.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", tokens, &holder.name)])
        .unwrap();
    reader.round();
    std::thread::sleep(Duration::from_millis(160));
    // This round comes well after the 50 ms lookup budget.
    let report = reader.peer.report();
    let round = reader.peer.agree(&[report]).unwrap();
    assert!(round.events.iter().any(|event| event.num_tokens > 0));
    let mut sched = reader.scheduler.lock();
    sched.advance_peer(round.now_us, &round.events).unwrap();
    let plan = sched.next_execution_plan(true, false);
    let fetches = plan.peer_fetches();
    if adaptive {
        assert!(fetches.is_empty());
        assert_eq!(plan.forward_batches()[0].extend_prefix_lens[0], 0);
        return;
    }
    assert_eq!(fetches.len(), 1);
    let page = &fetches[0].destinations[0];
    let addr = READER_BASE + page.block_id as u64 * PAGE_BYTES;
    drop(sched);
    reader.peer.dispatch(fetches, vec![], round.now_us);
    std::thread::sleep(Duration::from_millis(120));
    assert_eq!(arena.read_bytes(addr, 1), vec![0]);
    reader.peer.fence().ready();
    let (prefix, _) = rounds_until_prefill(&mut reader, None, "r");
    assert!(prefix >= 5 * PAGE);
    assert_ne!(arena.read_bytes(addr, 1), vec![0]);
}

#[test]
fn a_reader_lands_the_holders_prefix_byte_for_byte_and_computes_the_rest() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    reader.enable_logical_metrics();
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");

    assert_eq!(landings.len(), 1, "one landing");
    let landing = &landings[0];
    assert_eq!(landing.begin_tokens, 0);
    assert_eq!(
        prefix, landing.end_tokens,
        "it computes on from the landed boundary"
    );
    assert!(landing.end_tokens >= 5 * PAGE);
    // Every destination holds one of the holder's pages, byte for byte.
    for destination in &landing.destinations {
        let page = arena.read_bytes(
            READER_BASE + destination.block_id as u64 * PAGE_BYTES,
            PAGE_BYTES as usize,
        );
        assert!(
            page[0] != 0 && page.iter().all(|&byte| byte == page[0]),
            "slot {}",
            destination.slot
        );
    }
    // The holder applied the release at its next head.
    holder.round();
    let stats = holder.scheduler.lock().peer_stats();
    assert_eq!((stats.leases, stats.pinned_blocks), (0, 0));
    assert_eq!(
        reader.scheduler.lock().peer_stats().landed_tokens,
        landing.end_tokens as u64
    );
    assert_eq!(reader.landed, vec![("r".into(), 0, prefix)]);
    assert!(reader.decisions.contains(&("r".into(), "admission")));
}

#[test]
fn a_reader_landing_into_host_admits_the_prefix_as_a_host_hit() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start_host_landing("v1", READER_BASE, READER_HOST_BASE, &arena);
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");

    assert_eq!(landings.len(), 1, "one landing");
    let landing = &landings[0];
    assert!(landing.end_tokens >= 5 * PAGE);
    assert_eq!(
        prefix, landing.end_tokens,
        "admitted with the landed prefix"
    );
    // Every destination is a Host block holding one of the holder's pages.
    for destination in &landing.destinations {
        assert_eq!(destination.tier, PeerTier::Host);
        let page = arena.read_bytes(
            READER_HOST_BASE + (destination.block_id as u64 - 1) * PAGE_BYTES,
            PAGE_BYTES as usize,
        );
        assert!(
            page[0] != 0 && page.iter().all(|&byte| byte == page[0]),
            "slot {}",
            destination.slot
        );
    }
    holder.round();
    let stats = holder.scheduler.lock().peer_stats();
    assert_eq!((stats.leases, stats.pinned_blocks), (0, 0));
    assert_eq!(
        reader.scheduler.lock().peer_stats().landed_tokens,
        landing.end_tokens as u64
    );
}

/// A prefill replica reads a prefix too, and its landing's plan goes to the
/// data plane although it prepays no decode room and so zeroes, copies and
/// computes nothing else: the reads start from the fence the data plane
/// reports for that plan.
#[test]
fn a_prefill_replica_lands_a_prefix_from_a_plan_with_no_other_device_work() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start_as("v1", READER_BASE, &arena, "P");
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    reader.advance(&[event(ForwardEventKind::PdBootstrapped, "r", Vec::new())]);
    let start = Instant::now();
    let mut landing_plans = Vec::new();
    while start.elapsed() < Duration::from_secs(10) {
        holder.round();
        let (rows, fetches, needs_device) = reader.round_with_device();
        if !fetches.is_empty() {
            landing_plans.push(needs_device);
        }
        if let Some((_, prefix)) = rows.into_iter().find(|(rid, _)| rid == "r") {
            assert_eq!(
                landing_plans,
                [true],
                "one landing, from a plan the data plane runs"
            );
            assert!(
                prefix >= 5 * PAGE,
                "it computes on from the landed boundary"
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("the prefill replica never prefilled r; landing plans {landing_plans:?}");
}

#[test]
fn a_holder_that_never_answers_leaves_the_reader_to_compute() {
    let arena = Arena::new(200_000);
    // Hold a real endpoint without a servicer, so no other test can answer it.
    let silent = PeerListener::bind("127.0.0.1", 0).unwrap();
    let silent_name = format!("127.0.0.1:{}", silent.port());
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    reader.enable_logical_metrics();
    let tokens: Vec<i32> = (1..=4 * PAGE).collect();
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", tokens, &silent_name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, None, "r");
    assert_eq!((prefix, landings.len()), (0, 0));
    assert!(reader.decisions.contains(&("r".into(), "lookup_timeout")));
}

#[test]
fn a_lost_layout_query_does_not_disable_later_landings() {
    let arena = Arena::new(200_000);
    let (incoming, received) = mpsc::channel();
    let proxy = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let holder_name = format!("127.0.0.1:{}", proxy.port());
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_endpoint = format!("127.0.0.1:{}", listener.port());
    let mut holder = Replica::with_listener(
        settings(&holder_name, "v1"),
        listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    let prefixes: Vec<Vec<i32>> = [1, 1001]
        .into_iter()
        .map(|first| (first..first + 6 * PAGE).collect())
        .collect();
    for tokens in &prefixes {
        holder_computes(&mut holder, tokens, &arena);
    }
    let mut dropped_layout = false;
    for (index, tokens) in prefixes.into_iter().enumerate() {
        let request = format!("r{index}");
        let mut longer = tokens;
        longer.extend(2001..2001 + PAGE);
        reader
            .scheduler
            .lock()
            .submit_requests(&[spec(&request, longer, &holder_name)])
            .unwrap();
        let start = Instant::now();
        let (prefix, landings) = loop {
            assert!(start.elapsed() < Duration::from_secs(10));
            for frames in received.try_iter() {
                let Inbound::Request(message) = Inbound::decode(&frames).unwrap() else {
                    panic!("the proxy only receives requests");
                };
                // Drop one real message, then restore the network path.
                if matches!(message, Request::Layout { .. }) && !dropped_layout {
                    dropped_layout = true;
                } else {
                    proxy.request(&holder_endpoint, &message).unwrap();
                }
            }
            holder.round();
            let (rows, _) = reader.round();
            if let Some((_, prefix)) = rows.into_iter().find(|(id, _)| id == &request) {
                break (prefix, reader.scheduler.lock().peer_stats().landed_tokens);
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        if index == 0 {
            assert!(dropped_layout);
            assert_eq!((prefix, landings), (0, 0));
        } else {
            assert!(prefix >= 5 * PAGE, "the recovered holder must be reusable");
            assert_eq!(landings, prefix as u64);
        }
        reader.advance(&[
            event(ForwardEventKind::ExtendResult, &request, vec![7]),
            event(ForwardEventKind::Finish, &request, Vec::new()),
        ]);
        reader.round();
    }
}

#[test]
fn an_empty_holder_returns_a_lookup_miss() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    reader.enable_logical_metrics();
    let tokens = (1..=4 * PAGE).collect();
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", tokens, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!((prefix, landings.len()), (0, 0));
    assert_eq!(reader.decisions, vec![("r".into(), "lookup_miss")]);
}

#[test]
fn a_holder_of_another_namespace_is_never_read() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v2", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    reader.enable_logical_metrics();
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!((prefix, landings.len()), (0, 0));
    assert!(reader.decisions.contains(&("r".into(), "incompatible")));
}

#[test]
fn a_reader_that_moved_to_the_holders_weights_reads_them() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v2", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    reader.peer.set_weight_version("v2");

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!(landings.len(), 1, "one landing");
    assert_eq!(prefix, landings[0].end_tokens);
}

#[test]
fn a_holder_that_moved_to_new_weights_is_no_longer_read() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    holder.peer.set_weight_version("v2");

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!((prefix, landings.len()), (0, 0));
}

struct CheckedRead {
    arena: Arena,
    holder: Arc<Mutex<Scheduler>>,
    retired: mpsc::Sender<(Instant, usize)>,
    before_retirement: Option<mpsc::Sender<mpsc::SyncSender<()>>>,
}

impl PageReader for CheckedRead {
    fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
        assert!(self.holder.lock().peer_stats().pinned_blocks > 0);
        self.arena.read(session, spans)?;
        for ((src, dst), len) in spans
            .sources
            .iter()
            .zip(&spans.destinations)
            .zip(&spans.lengths)
        {
            assert!(*dst >= READER_HOST_BASE);
            assert_eq!(
                self.arena.read_bytes(*src, *len as usize),
                self.arena.read_bytes(*dst, *len as usize)
            );
        }
        if let Some(wake) = &self.before_retirement {
            let (ready, wait) = mpsc::sync_channel(0);
            wake.send(ready).unwrap();
            wait.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        self.retired
            .send((Instant::now(), spans.lengths.iter().sum::<u64>() as usize))
            .unwrap();
        Ok(())
    }
}

// Real control protocol, C++ plan, Host allocator and loopback endpoints.
// The four-second completion stand-in replaces only B's device execution.
#[test]
fn host_read_and_release_retire_before_the_next_scheduler_round() {
    host_during_forward(true, Duration::ZERO, false, false);
}

#[test]
fn round_driven_host_reader_misses_the_same_forward_window() {
    host_during_forward(false, Duration::ZERO, false, false);
}

#[test]
fn slow_host_reply_falls_back_without_posting_a_read() {
    host_during_forward(true, Duration::from_millis(130), false, false);
}

#[test]
fn contended_pin_does_not_delay_another_completed_host_read() {
    host_during_forward(true, Duration::ZERO, true, false);
}

#[test]
fn timely_lookup_survives_reader_cache_lock_beyond_the_lookup_budget() {
    host_during_forward(true, Duration::ZERO, false, true);
}

fn host_during_forward(autonomous: bool, holder_delay: Duration, contend: bool, hold_reader: bool) {
    let arena = Arena::new(270_000);
    let bound = || PeerListener::bind("127.0.0.1", 0).unwrap();
    let address = |port| format!("127.0.0.1:{port}");
    let config = |name: &str| {
        let mut cfg = settings(name, "v1");
        cfg.autonomous_host = autonomous;
        cfg.timeouts.answer_us = 50_000;
        cfg
    };
    let listener = bound();
    let mut holder = Replica::with_landing(
        config(&address(listener.port())),
        listener,
        geometry_with_host(HOLDER_BASE, 215_000),
        &arena,
        "Fused",
        Landing::Host,
    );
    let pages = (PAGES as i32 - 8).min(314);
    let tokens: Vec<i32> = (1..=pages * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    let (read_tx, read_rx) = mpsc::channel();
    let (before, lock_at_read) = mpsc::channel::<mpsc::SyncSender<()>>();
    let observed = Arc::new(CheckedRead {
        arena: arena.clone(),
        holder: holder.scheduler.clone(),
        retired: read_tx,
        before_retirement: contend.then_some(before),
    });
    let (wire_tx, wire_rx) = mpsc::channel();
    let proxy = PeerNet::start("127.0.0.1", 0, wire_tx).unwrap();
    let listener = bound();
    let actual = address(listener.port());
    let mut reader = Replica::with_reader(
        config(&address(proxy.port())),
        listener,
        geometry_with_host(READER_BASE, READER_HOST_BASE),
        observed,
        "P",
        Landing::Host,
    );
    let mut longer = tokens.clone();
    longer.extend(200_000..200_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[
            spec("A", longer, &holder.name),
            spec("B", vec![2_000_000; PAGE as usize], ""),
        ])
        .unwrap();
    reader.advance(&[
        event(ForwardEventKind::PdBootstrapped, "A", vec![]),
        event(ForwardEventKind::PdBootstrapped, "B", vec![]),
    ]);
    let (ack_tx, ack_rx) = mpsc::channel();
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        if contend {
            let cache = holder.scheduler.clone();
            let endpoint = holder.name.clone();
            let (tx, rx) = mpsc::channel();
            let x = PeerNet::start("127.0.0.1", 0, tx).unwrap();
            let lease = ts_scheduler::PeerLeaseView {
                reader: address(x.port()),
                id: 9,
            };
            // Ask a real scheduler for X's lookup: use its page granularity
            // and hash chain, including in long-prefix replays.
            let fixture_listener = bound();
            let fixture = Replica::with_landing(
                config(&address(fixture_listener.port())),
                fixture_listener,
                geometry_with_host(READER_BASE, READER_HOST_BASE),
                &arena,
                "P",
                Landing::Host,
            );
            let mut x_tokens = tokens.clone();
            x_tokens.push(500_000);
            let mut scheduler = fixture.scheduler.lock();
            scheduler
                .submit_requests(&[spec("X", x_tokens, &endpoint)])
                .unwrap();
            scheduler
                .advance(&[event(ForwardEventKind::PdBootstrapped, "X", vec![])])
                .unwrap();
            let _plan = scheduler.next_execution_plan(true, false);
            let lookup = scheduler.take_peer_lookups().pop().expect("X lookup");
            drop(scheduler);
            x.request(
                &endpoint,
                &Request::Lookup {
                    lease: lease.clone(),
                    prompt: ts_control::peer::wire::Prompt {
                        prefix_hashes: lookup.prefix_hashes,
                        tokens: lookup.tokens,
                        max_tokens: lookup.max_tokens,
                    },
                },
            )
            .unwrap();
            assert!(
                matches!(Inbound::decode(&rx.recv_timeout(Duration::from_secs(2)).unwrap()),
                Ok(Inbound::Reply(Reply::Answer{answer,..})) if answer.num_tokens == tokens.len() as i32)
            );
            scope.spawn(move || {
                let ready = lock_at_read.recv_timeout(Duration::from_secs(2)).unwrap();
                let guard = cache.lock();
                x.request(
                    &endpoint,
                    &Request::Pin {
                        lease: lease.clone(),
                    },
                )
                .unwrap();
                // FIFO for one destination: the layout reply proves PIN(X)
                // reached ingress before we let A's read worker retire.
                x.request(
                    &endpoint,
                    &Request::Layout {
                        reply_to: lease.reader.clone(),
                    },
                )
                .unwrap();
                assert!(matches!(
                    Inbound::decode(&rx.recv_timeout(Duration::from_secs(2)).unwrap()),
                    Ok(Inbound::Reply(Reply::Layout { .. }))
                ));
                ready.send(()).unwrap();
                std::thread::sleep(Duration::from_secs(4));
                assert_eq!(guard.peer_stats().pinned_blocks, 0);
                assert!(
                    rx.try_recv().is_err(),
                    "X cannot be authorized while the cache is locked"
                );
                x.request(
                    &endpoint,
                    &Request::Release {
                        lease,
                        demote: false,
                    },
                )
                .unwrap();
                drop(guard);
            });
        }
        let stop_ref = &stop;
        let endpoint = holder.name.clone();
        let holder_cache = holder.scheduler.clone();
        let reader_cache = reader.scheduler.clone();
        scope.spawn(move || {
            while !stop_ref.load(std::sync::atomic::Ordering::Acquire) {
                let Ok(frames) = wire_rx.recv_timeout(Duration::from_millis(10)) else { continue };
                let Inbound::Reply(reply) = Inbound::decode(&frames).unwrap() else {panic!("expected reply")};
                let release = matches!(&reply, Reply::Answer{answer,..} if answer.kind == ts_scheduler::PeerAnswerKind::Release && answer.held);
                if matches!(&reply, Reply::Answer{answer,..} if answer.kind == ts_scheduler::PeerAnswerKind::Lookup) {
                    std::thread::sleep(holder_delay);
                }
                let lookup = matches!(&reply, Reply::Answer{answer,..} if answer.kind == ts_scheduler::PeerAnswerKind::Lookup);
                let guard = (hold_reader && lookup).then(|| reader_cache.lock());
                proxy.reply(&actual,&reply).unwrap();
                if guard.is_some() {
                    std::thread::sleep(Duration::from_millis(150));
                }
                drop(guard);
                if release {
                    let at = Instant::now();
                    if contend {
                        let Reply::Answer {answer, ..} = &reply else {unreachable!()};
                        proxy.request(&endpoint, &Request::Release {
                            lease: answer.lease.clone(), demote: true
                        }).unwrap();
                        let repeated = wire_rx.recv_timeout(Duration::from_millis(50)).unwrap();
                        assert!(matches!(Inbound::decode(&repeated), Ok(Inbound::Reply(duplicate)) if duplicate == reply));
                    }
                    let stats = holder_cache.lock().peer_stats();
                    ack_tx.send((at,stats.pinned_blocks,stats.leases)).unwrap();
                }
            }
        });
        struct Stop<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::Release);
            }
        }
        let _stop = Stop(&stop);
        let start = Instant::now();
        let (batch, _, device) = reader.round_with_device();
        assert_eq!(batch, vec![("B".to_owned(), 0)]);
        assert!(device);
        std::thread::sleep(Duration::from_secs(4));
        let forward_end = Instant::now();
        let read = read_rx.try_recv().ok();
        let released = if contend {
            ack_rx.recv_timeout(Duration::from_secs(1)).ok()
        } else {
            ack_rx.try_recv().ok()
        };
        assert_eq!(
            reader.scheduler.lock().peer_stats().landed_tokens,
            0,
            "not yet published by the reader"
        );
        reader.advance(&[event(ForwardEventKind::ExtendResult, "B", vec![7])]);
        let (next, _) = reader.round();
        let prefix = next
            .iter()
            .find(|(r, _)| r == "A")
            .expect("A in the very next plan")
            .1;
        let expect = autonomous && holder_delay.is_zero();
        assert_eq!(prefix, if expect { tokens.len() as i32 } else { 0 });
        assert_eq!(
            reader.scheduler.lock().peer_stats().landed_tokens,
            prefix as u64
        );
        if autonomous {
            assert_eq!(reader.peer.num_landings(), 0);
        }
        if expect {
            let (retired, bytes) = read.unwrap();
            let (ack, pins, leases) = released.unwrap();
            assert!(start <= retired && retired <= ack && ack < forward_end);
            if contend {
                assert!(ack.duration_since(retired) < Duration::from_millis(50));
            } else {
                assert_eq!((pins, leases), (0, 0));
            }
            assert!(bytes > 0);
            println!(
                "HOST_WINDOW read_us={} release_us={} forward_us={} bytes={} next_prefix={prefix}",
                (retired - start).as_micros(),
                (ack - start).as_micros(),
                (forward_end - start).as_micros(),
                bytes
            );
        } else {
            assert!(read.is_none() && released.is_none());
            println!(
                "HOST_CONTROL async={autonomous} delay_ms={} forward_us={} next_prefix={prefix}",
                holder_delay.as_millis(),
                (forward_end - start).as_micros()
            );
        }
    });
}

#[test]
fn an_aborted_host_read_keeps_its_pages_until_retirement_without_publishing() {
    struct HeldRead {
        arena: Arena,
        entered: mpsc::Sender<()>,
        resume: Mutex<mpsc::Receiver<()>>,
    }
    impl PageReader for HeldRead {
        fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
            self.entered.send(()).unwrap();
            self.resume
                .lock()
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            self.arena.read(session, spans)
        }
    }
    let arena = Arena::new(270_000);
    let start = |base, host, reader: Arc<dyn PageReader>| {
        let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let mut config = settings(&format!("127.0.0.1:{}", listener.port()), "v1");
        config.autonomous_host = true;
        Replica::with_reader(
            config,
            listener,
            geometry_with_host(base, host),
            reader,
            "Fused",
            Landing::Host,
        )
    };
    let mut holder = start(HOLDER_BASE, 215_000, Arc::new(arena.clone()));
    let (entered, reading) = mpsc::channel();
    let (resume, gate) = mpsc::channel();
    let mut reader = start(
        READER_BASE,
        READER_HOST_BASE,
        Arc::new(HeldRead {
            arena: arena.clone(),
            entered,
            resume: Mutex::new(gate),
        }),
    );
    let mut prompt: Vec<_> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &prompt, &arena);
    prompt.extend(100_000..100_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("cancel", prompt, &holder.name)])
        .unwrap();
    reader.round();
    reading
        .recv_timeout(Duration::from_secs(2))
        .expect("READ submitted");
    reader.advance(&[event(ForwardEventKind::Abort, "cancel", vec![])]);
    assert_eq!(reader.scheduler.lock().peer_stats().landing_requests, 1);
    assert!(holder.scheduler.lock().peer_stats().pinned_blocks > 0);
    resume.send(()).unwrap();
    let start = Instant::now();
    loop {
        assert!(start.elapsed() < Duration::from_secs(3));
        let (rows, _) = reader.round();
        assert!(!rows.iter().any(|(r, _)| r == "cancel"));
        if reader.scheduler.lock().peer_stats().landing_requests == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(reader.scheduler.lock().peer_stats().landed_tokens, 0);
    let stats = holder.scheduler.lock().peer_stats();
    assert_eq!((stats.leases, stats.pinned_blocks), (0, 0));
}

fn grant_replica(base: u64, memory: Arc<dyn PageReader>) -> Replica {
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let mut cfg = settings(&format!("127.0.0.1:{}", listener.port()), "v1");
    cfg.device_admission = Some(true);
    cfg.grant_first = true;
    cfg.timeouts.answer_us = 150_000;
    Replica::with_reader(
        cfg,
        listener,
        geometry(base),
        memory,
        "Fused",
        Landing::Device,
    )
}

// Stop before the data-plane fence. Ownership of the ordinary HBM pages is
// committed; no handler at the holder should yet know about this request.
fn grant_without_fence(
    reader: &mut Replica,
    holder: &Replica,
    id: &str,
    prefix: &[i32],
) -> PeerFetchView {
    let mut tokens = prefix.to_vec();
    tokens.extend(50000..50000 + PAGE);
    let mut request = spec(id, tokens, &holder.name);
    request.peer_expected_tokens = prefix.len() as i32;
    let mut scheduler = reader.scheduler.lock();
    scheduler.submit_requests(&[request]).unwrap();
    let plan = scheduler.next_execution_plan(true, false);
    let fetches = plan.peer_fetches();
    let lookups = scheduler.take_peer_lookups();
    assert_eq!(fetches.len(), 1);
    assert_eq!(lookups.len(), 1);
    assert!(plan
        .forward_batches()
        .iter()
        .all(|batch| !batch.request_ids.iter().any(|rid| rid == id)));
    let fetch = fetches[0].clone();
    assert_eq!(fetch.end_tokens, prefix.len() as i32);
    drop(scheduler);
    reader.peer.dispatch(fetches, lookups, 0);
    fetch
}

#[test]
fn reciprocal_device_grants_complete_without_another_scheduler_round() {
    let arena = Arena::new(200_000);
    let mut a = grant_replica(HOLDER_BASE, Arc::new(arena.clone()));
    let mut b = grant_replica(READER_BASE, Arc::new(arena.clone()));
    let left: Vec<i32> = (1..=2048).collect();
    let right: Vec<i32> = (9001..=11048).collect();
    holder_computes(&mut a, &left, &arena);
    holder_computes(&mut b, &right, &arena);
    for i in 1..512 {
        arena.write(HOLDER_BASE + i * PAGE_BYTES, &[17; PAGE_BYTES as usize]);
        arena.write(READER_BASE + i * PAGE_BYTES, &[29; PAGE_BYTES as usize]);
    }
    let af = grant_without_fence(&mut a, &b, "a", &right);
    let bf = grant_without_fence(&mut b, &a, "b", &left);
    std::thread::sleep(Duration::from_millis(210));
    assert_eq!(a.scheduler.lock().peer_stats().leases, 0);
    assert_eq!(b.scheduler.lock().peer_stats().leases, 0);
    a.peer.fence().ready();
    b.peer.fence().ready();
    // Neither scheduler takes a round during the simulated long forward.
    std::thread::sleep(Duration::from_secs(4));
    for (replica, fetch, base, byte) in
        [(&mut a, af, HOLDER_BASE, 29), (&mut b, bf, READER_BASE, 17)]
    {
        let addr = base + fetch.destinations[0].block_id as u64 * PAGE_BYTES;
        assert_eq!(arena.read_bytes(addr, 1), vec![byte]);
        assert_eq!(replica.scheduler.lock().peer_stats().pinned_blocks, 0);
        assert_eq!(replica.round().0, vec![(fetch.request_id, 2048)]);
    }
}

#[test]
fn device_grant_abort_retires_only_after_its_fence_or_dma() {
    struct GatedRead(Arena, mpsc::Sender<()>, Mutex<mpsc::Receiver<()>>);
    impl PageReader for GatedRead {
        fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
            self.1.send(()).unwrap();
            self.2.lock().recv_timeout(Duration::from_secs(4)).unwrap();
            self.0.read(session, spans)
        }
    }
    for before_fence in [true, false] {
        let arena = Arena::new(200_000);
        let (posted, reading) = mpsc::channel();
        let (resume, gate) = mpsc::channel();
        let mut holder = grant_replica(HOLDER_BASE, Arc::new(arena.clone()));
        let mut reader = grant_replica(
            READER_BASE,
            Arc::new(GatedRead(arena.clone(), posted, Mutex::new(gate))),
        );
        let prefix: Vec<i32> = (1..=2048).collect();
        holder_computes(&mut holder, &prefix, &arena);
        grant_without_fence(&mut reader, &holder, "abort", &prefix);
        if !before_fence {
            reader.peer.fence().ready();
            reading
                .recv_timeout(Duration::from_secs(2))
                .expect("DMA posted");
        }
        reader.advance(&[event(ForwardEventKind::Abort, "abort", vec![])]);
        reader.peer.cancel_grant("abort");
        let report = reader.peer.report();
        assert!(reader.peer.agree(&[report]).unwrap().events.is_empty());
        if before_fence {
            reader.peer.fence().ready();
        } else {
            assert!(holder.scheduler.lock().peer_stats().pinned_blocks > 0);
            resume.send(()).unwrap();
        }
        let until = Instant::now() + Duration::from_secs(3);
        loop {
            assert!(Instant::now() < until, "cancelled grant did not retire");
            let report = reader.peer.report();
            let round = reader.peer.agree(&[report]).unwrap();
            let retired = round
                .events
                .iter()
                .any(|e| e.kind == ts_scheduler::PeerEventKind::FetchDone);
            reader
                .scheduler
                .lock()
                .advance_peer(round.now_us, &round.events)
                .unwrap();
            if retired {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(reader.round().0.is_empty());
        if before_fence {
            assert!(reading.try_recv().is_err());
        }
        assert_eq!(holder.scheduler.lock().peer_stats().pinned_blocks, 0);
    }
}

#[test]
fn device_grant_returns_to_local_when_holder_cannot_answer() {
    let arena = Arena::new(200_000);
    let holder = grant_replica(HOLDER_BASE, Arc::new(arena.clone()));
    let mut reader = grant_replica(READER_BASE, Arc::new(arena));
    let prefix: Vec<i32> = (1..=2048).collect();
    grant_without_fence(&mut reader, &holder, "timeout", &prefix);
    let holder_lock = holder.scheduler.lock();
    reader.peer.fence().ready();
    let until = Instant::now() + Duration::from_secs(2);
    loop {
        assert!(
            Instant::now() < until,
            "reader waited for the holder's scheduler"
        );
        let (rows, fetches) = reader.round();
        assert!(fetches.is_empty(), "reissued an expired grant");
        if !rows.is_empty() {
            assert_eq!(rows, vec![("timeout".to_owned(), 0)]);
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(reader.scheduler.lock().peer_stats().landing_requests, 0);
    assert!(reader
        .decisions
        .contains(&("timeout".to_owned(), "lookup_timeout")));
    drop(holder_lock);
}

#[test]
fn shorter_holder_prefix_cannot_write_a_longer_device_grant() {
    struct UnexpectedRead;
    impl PageReader for UnexpectedRead {
        fn read(&self, _: &str, _: &Spans) -> Result<(), String> {
            panic!("partial prefix started a DMA into an incompatible grant");
        }
    }
    let arena = Arena::new(200_000);
    let mut holder = grant_replica(HOLDER_BASE, Arc::new(arena.clone()));
    let mut reader = grant_replica(READER_BASE, Arc::new(UnexpectedRead));
    let prefix: Vec<i32> = (1..=2048).collect();
    holder_computes(&mut holder, &prefix[..1024], &arena);
    grant_without_fence(&mut reader, &holder, "partial", &prefix);
    reader.peer.fence().ready();
    let (used, subsequent) = rounds_until_prefill(&mut reader, None, "partial");
    assert_eq!(used, 0);
    assert!(subsequent.is_empty());
    assert!(reader
        .decisions
        .contains(&("partial".to_owned(), "boundary")));
    assert_eq!(reader.scheduler.lock().peer_stats().landing_requests, 0);
    assert_eq!(holder.scheduler.lock().peer_stats().pinned_blocks, 0);
}

fn delayed_checkpoint_pin(repin: bool, evict: bool) {
    let arena = Arena::new(200_000);
    let replica = |base| {
        let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let mut cfg = settings(&format!("127.0.0.1:{}", listener.port()), "v1");
        cfg.device_admission = Some(true);
        cfg.checkpoint_pin = repin;
        cfg.timeouts.answer_us = 50_000;
        // Also expire the Rust holder's lookup metadata before the PIN.
        cfg.record_lifetime = Duration::from_secs(if repin { 2 } else { 60 });
        Replica::with_reader_pages(
            cfg,
            listener,
            geometry(base),
            Arc::new(arena.clone()),
            "Fused",
            Landing::Device,
            512,
            2_000_000,
        )
    };
    let mut holder = replica(HOLDER_BASE);
    let mut reader = replica(READER_BASE);
    let mut tokens: Vec<i32> = (1..=2048).collect();
    holder_computes(&mut holder, &tokens, &arena);
    for i in 1..512 {
        arena.write(HOLDER_BASE + i * PAGE_BYTES, &[29; PAGE_BYTES as usize]);
    }
    tokens.extend(50000..50000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[
            spec("read", tokens, &holder.name),
            spec("forward", vec![98; 64], ""),
        ])
        .unwrap();
    assert_eq!(reader.round().0, vec![("forward".into(), 0)]);
    std::thread::sleep(Duration::from_secs(4));
    // Retire the soft lease explicitly before the late grant.
    holder.round();
    assert_eq!(holder.scheduler.lock().peer_stats().leases, 0);
    if evict {
        assert!(holder.scheduler.lock().clear_cache());
    }
    reader.advance(&[
        event(ForwardEventKind::ExtendResult, "forward", vec![7]),
        event(ForwardEventKind::Finish, "forward", vec![]),
    ]);
    let (prefix, fetches) = rounds_until_prefill(&mut reader, None, "read");
    assert_eq!(
        fetches.len(),
        1,
        "timely lookup must survive the long forward"
    );
    let fetch = &fetches[0];
    let addr = READER_BASE + fetch.destinations[0].block_id as u64 * PAGE_BYTES;
    if repin && !evict {
        assert_eq!(prefix, 2048);
        assert_eq!(arena.read_bytes(addr, 1), vec![29]);
    } else {
        assert_eq!(prefix, 0);
        assert_eq!(arena.read_bytes(addr, 1), vec![0]);
    }
    assert_eq!(holder.scheduler.lock().peer_stats().pinned_blocks, 0);
    println!(
        "delayed PIN repin={repin} evict={evict} forward_ms=4000 fetches={} consumed={prefix}",
        fetches.len()
    );
}
#[test]
fn expired_checkpoint_pin_keeps_an_unchanged_prefix() {
    delayed_checkpoint_pin(true, false);
}
#[test]
fn expired_checkpoint_pin_rejects_an_evicted_prefix() {
    delayed_checkpoint_pin(true, true);
}
#[test]
fn original_adaptive_loses_the_expired_pin() {
    delayed_checkpoint_pin(false, false);
}
