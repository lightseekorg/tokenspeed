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
fn scheduler(name: &str, incarnation: u64, role: &str, landing: Landing) -> Scheduler {
    interpreter();
    let host = match landing {
        Landing::Device => "cfg.disable_l2_cache = True".to_owned(),
        Landing::Host => format!(
            "cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = {PAGES}\n\
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
            PAGES - 1
        ),
    };
    Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.{role}
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = 512
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = {PAGES}
cfg.num_snapshot_pages = 1
{host}
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity={PAGE},
                                        total_pages={PAGES}, attn_kind=ts.AttnKind.Full,
                                        transfer_policy=ts.CacheTransferPolicy.FullSuffix)]
cfg.peer_config = ts.PeerConfig(name="{name}", incarnation={incarnation}, max_leases=8,
                                soft_pin_us=1_000_000, hard_pin_us=1_000_000,
                                lookup_timeout_us=1_000_000, min_fetch_tokens={PAGE},
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
        device_regions: Vec::new(),
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
            region_lengths: Vec::new(),
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
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_INCARNATION: AtomicU64 = AtomicU64::new(1);
    PeerSettings {
        name: name.to_owned(),
        incarnation: NEXT_INCARNATION.fetch_add(1, Ordering::Relaxed),
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
        let scheduler = Arc::new(Mutex::new(scheduler(
            &settings.name,
            settings.incarnation,
            role,
            landing,
        )));
        let name = settings.name.clone();
        let peer = Peer::start(
            settings,
            0,
            vec![geometry],
            Some(listener),
            scheduler.clone(),
            Arc::new(arena.clone()),
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
    let client_addr = format!("127.0.0.1:{}", client.port());
    let layout = Request::Layout {
        reply_to: client_addr.clone(),
    };
    let receive_layout = || {
        let frames = replies.recv_timeout(Duration::from_secs(5)).unwrap();
        let Inbound::Reply(Reply::Layout { stamp, .. }) = Inbound::decode(&frames).unwrap() else {
            panic!("expected a layout reply");
        };
        stamp.name
    };
    let receive_pin = || {
        let frames = replies.recv_timeout(Duration::from_secs(5)).unwrap();
        let Inbound::Reply(Reply::Answer { stamp, .. }) = Inbound::decode(&frames).unwrap() else {
            panic!("expected a pin answer reply");
        };
        stamp.name
    };
    // 1. `Request::Layout` on `owner` answers immediately WITHOUT locking `owner.scheduler`!
    let lock = owner.scheduler.lock();
    client.request(&owner.name, &layout).unwrap();
    assert_eq!(receive_layout(), owner.name);

    // 2. Both `Request::Pin` messages enter the same public socket in FIFO order;
    //    `owner`'s servicer waits on `owner.scheduler.lock()`, while `ts-peer-router`
    //    forwards `sibling`'s request without head-of-line blocking!
    let pin = Request::Pin {
        lease: ts_scheduler::PeerLeaseView {
            reader: client_addr,
            id: 77,
        },
    };
    client.request(&owner.name, &pin).unwrap();
    client.request(&sibling.name, &pin).unwrap();
    assert_eq!(receive_pin(), sibling.name);
    drop(lock);
    assert_eq!(receive_pin(), owner.name);
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
fn a_reader_cannot_pin_device_pages_during_replay_retirement() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);

    let mut longer = tokens;
    longer.extend(1000..1000 + PAGE);
    holder.peer.block_device_pins(true);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("blocked", longer.clone(), &holder.name)])
        .unwrap();
    let (prefix, fetches) = rounds_until_prefill(&mut reader, Some(&mut holder), "blocked");
    // A reader reserves the landing before it asks the holder to pin.
    assert_eq!(fetches.len(), 1);
    assert_eq!(prefix, 0);
    assert_eq!(reader.scheduler.lock().peer_stats().landed_tokens, 0);
    for destination in &fetches[0].destinations {
        let page = arena.read_bytes(
            READER_BASE + destination.block_id as u64 * PAGE_BYTES,
            PAGE_BYTES as usize,
        );
        assert!(page.iter().all(|&byte| byte == 0));
    }

    reader.advance(&[
        event(ForwardEventKind::ExtendResult, "blocked", vec![7]),
        event(ForwardEventKind::Abort, "blocked", Vec::new()),
    ]);
    holder.peer.block_device_pins(false);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("ready", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "ready");
    assert_eq!(landings.len(), 1);
    assert!(prefix >= 5 * PAGE);
    assert_eq!(prefix, landings[0].end_tokens);
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

#[test]
fn servicer_demux_avoids_hol_blocking_and_dp0_aliases_and_late_layout_succeed() {
    use ts_control::peer::{normalize_peer_holder, wire::Prompt};
    use ts_scheduler::{PeerAnswerKind, PeerLeaseView};

    assert_eq!(normalize_peer_holder("10.0.0.1:7100/0"), "10.0.0.1:7100");
    assert_eq!(normalize_peer_holder("10.0.0.1:7100/00"), "10.0.0.1:7100");
    assert_eq!(normalize_peer_holder("10.0.0.1:7100/+0"), "10.0.0.1:7100");
    assert_eq!(normalize_peer_holder("[::1]:7100/00"), "[::1]:7100");
    assert_eq!(normalize_peer_holder("10.0.0.1:7100/1"), "10.0.0.1:7100/1");

    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let prefix: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &prefix, &arena);

    // 1. Non-canonical "/0" hint normalizes and arrived Reply::Layout past answer_us advances to Pin cleanly.
    let rl = PeerListener::bind("127.0.0.1", 0).unwrap();
    let rname = format!("127.0.0.1:{}", rl.port());
    let mut rs = settings(&rname, "v1");
    rs.timeouts.answer_us = 40_000;
    let mut reader = Replica::with_listener(rs, rl, READER_BASE, &arena, "Fused");
    let mut prompt = prefix.clone();
    prompt.extend(95_000..95_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec(
            "slash0-late-layout",
            prompt,
            &format!("{}/0", holder.name),
        )])
        .unwrap();
    let start = Instant::now();
    loop {
        assert!(start.elapsed() < Duration::from_secs(5));
        holder.round();
        if !reader.round().1.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    std::thread::sleep(Duration::from_millis(80)); // > answer_us, Reply::Layout already arrived
    let (hit, _) = rounds_until_prefill(&mut reader, Some(&mut holder), "slash0-late-layout");
    assert!(hit >= 5 * PAGE);

    // 2. ts-peer-demux delivers Layout + Inbound::Reply while scheduler.lock() is held behind queued Lookups.
    let seed_r: Vec<i32> = (500..500 + 4 * PAGE).collect();
    holder_computes(&mut reader, &seed_r, &arena);
    let (ext_tx, ext_rx) = mpsc::channel();
    let ext = PeerNet::start("127.0.0.1", 0, ext_tx).unwrap();
    let ext_addr = format!("127.0.0.1:{}", ext.port());
    let lookup_prompt = {
        let mut s = scheduler("scratch", 99, "Fused", Landing::Device);
        let mut t = seed_r;
        t.extend(88_000..88_000 + PAGE);
        s.submit_requests(&[spec("s", t, "target")]).unwrap();
        let _ = s.next_execution_plan(true, false);
        let l = s.take_peer_lookups().pop().unwrap();
        Prompt {
            prefix_hashes: l.prefix_hashes,
            tokens: l.tokens,
            max_tokens: l.max_tokens,
        }
    };
    let guard = reader.scheduler.lock();
    for i in 0..3u64 {
        ext.request(
            &reader.name,
            &Request::Lookup {
                lease: PeerLeaseView {
                    reader: ext_addr.clone(),
                    id: 500 + i,
                },
                prompt: lookup_prompt.clone(),
            },
        )
        .unwrap();
    }
    std::thread::sleep(Duration::from_millis(15));
    ext.request(&reader.name, &Request::Layout { reply_to: ext_addr })
        .unwrap();
    let layout_frame = ext_rx
        .recv_timeout(Duration::from_millis(300))
        .expect("Layout was HOL-blocked behind scheduler.lock()");
    assert!(matches!(
        Inbound::decode(&layout_frame).unwrap(),
        Inbound::Reply(Reply::Layout { .. })
    ));
    drop(guard);
    for _ in 0..3 {
        let f = ext_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            matches!(Inbound::decode(&f).unwrap(), Inbound::Reply(Reply::Answer { answer, .. }) if answer.kind == PeerAnswerKind::Lookup && answer.num_tokens == 4 * PAGE)
        );
    }
}
