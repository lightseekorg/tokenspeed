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
    Peer, PeerListener, PeerProgress, PeerSettings, PeerTransfers,
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
    scheduler_with_storage(
        name,
        incarnation,
        role,
        landing,
        timeout_us,
        pages,
        soft_pin_us,
        false,
    )
}

fn scheduler_with_storage(
    name: &str,
    incarnation: u64,
    role: &str,
    landing: Landing,
    timeout_us: i64,
    pages: u64,
    soft_pin_us: i64,
    storage: bool,
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
            "landing_tier=ts.PeerLandingTier.Device, max_host_landing_blocks=0, host_ready_timeout_us=0, versioned_shards=False"
                .to_owned()
        }
        Landing::Host => format!(
            "landing_tier=ts.PeerLandingTier.Host, max_host_landing_blocks={}, \
             host_ready_timeout_us=1_000_000, versioned_shards=False",
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
                                {tier}, storage_min_fetch_tokens={storage_min}, storage_max_tokens={storage_max},
                                storage_lookup_timeout_us={storage_timeout})
result = ts.Scheduler(cfg)
"#,
            storage_min = if storage { PAGE } else { 0 },
            storage_max = if storage { 8 * PAGE } else { 0 },
            storage_timeout = if storage { 1_000_000 } else { 0 },
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

#[derive(Clone, Copy)]
enum TestDriver {
    Coordinated,
    Automatic,
    Eager,
}

struct FixtureSettings {
    pages: u64,
    soft_pin_us: i64,
    driver: TestDriver,
    peer: PeerSettings,
}

fn settings(name: &str, weight_version: &str) -> FixtureSettings {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_INCARNATION: AtomicU64 = AtomicU64::new(1);
    FixtureSettings {
        pages: PAGES,
        soft_pin_us: 1_000_000,
        driver: TestDriver::Coordinated,
        peer: PeerSettings {
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
        },
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
        settings: FixtureSettings,
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
        settings: FixtureSettings,
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
        settings: FixtureSettings,
        listener: PeerListener,
        geometry: RankGeometry,
        reader: Arc<dyn PageReader>,
        role: &str,
        landing: Landing,
    ) -> Self {
        let scheduler = Arc::new(Mutex::new(scheduler(
            &settings.peer.name,
            settings.peer.incarnation,
            role,
            landing,
            settings.peer.timeouts.answer_us,
            settings.pages,
            settings.soft_pin_us,
        )));
        let name = settings.peer.name.clone();
        let progress = match settings.driver {
            TestDriver::Coordinated => PeerProgress::Coordinated,
            TestDriver::Automatic | TestDriver::Eager => {
                let mut scheduler = scheduler.lock();
                let mode = PeerProgress::for_replica(1, &mut scheduler).unwrap();
                if matches!(settings.driver, TestDriver::Eager) {
                    assert_eq!(mode, PeerProgress::Device);
                    assert!(scheduler.enable_async_device_peer(false));
                }
                mode
            }
        };
        let peer = Peer::start(
            settings.peer,
            progress,
            0,
            vec![geometry],
            Some(listener),
            scheduler.clone(),
            PeerTransfers {
                reader,
                pusher: Arc::new(NoPushes),
            },
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
        self.peer
            .dispatch(fetches.clone(), lookups, round.now_us)
            .unwrap();
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
fn coordinated_members_use_the_leaders_clock_despite_skewed_reports() {
    let arena = Arena::new(200_000);
    let mut listener = Some(PeerListener::bind("127.0.0.1", 0).unwrap());
    let name = format!("127.0.0.1:{}", listener.as_ref().unwrap().port());
    let settings = settings(&name, "v1");
    let sched = Arc::new(Mutex::new(scheduler(
        &name,
        settings.peer.incarnation,
        "Fused",
        Landing::Device,
        settings.peer.timeouts.answer_us,
        settings.pages,
        settings.soft_pin_us,
    )));
    let geometries = vec![geometry(HOLDER_BASE); 8];
    let mut members: Vec<_> = (0..8)
        .map(|i| {
            Peer::start(
                settings.peer.clone(),
                PeerProgress::Coordinated,
                i,
                geometries.clone(),
                listener.take(),
                sched.clone(),
                PeerTransfers {
                    reader: Arc::new(arena.clone()),
                    pusher: Arc::new(NoPushes),
                },
            )
            .unwrap()
        })
        .collect();
    let mut reports: Vec<_> = members.iter_mut().map(Peer::report).collect();
    // The first i64 in the coordinated report is the member's elapsed microseconds.
    // Place the local clocks on opposite sides of the two-second soft deadline.
    for (i, report) in reports.iter_mut().enumerate() {
        report[..8].copy_from_slice(&(10_000_000_i64 * i as i64).to_le_bytes());
    }
    for leader_now in [1_999_999_i64, 2_000_000] {
        reports[0][..8].copy_from_slice(&leader_now.to_le_bytes());
        for member in &mut members {
            assert_eq!(member.agree(&reports).unwrap().now_us, leader_now);
        }
    }
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
        cfg.driver = if adaptive {
            TestDriver::Automatic
        } else {
            TestDriver::Eager
        };
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
    // Wait for the protocol result without giving the scheduler another round.
    let deadline = Instant::now() + Duration::from_secs(5);
    let round = loop {
        let report = reader.peer.report();
        let round = reader.peer.agree(&[report]).unwrap();
        if !round.events.is_empty() {
            break round;
        }
        assert!(Instant::now() < deadline, "lookup did not complete");
        std::thread::yield_now();
    };
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
    reader.peer.dispatch(fetches, vec![], round.now_us).unwrap();
    // Give a premature write time to surface while the device fence stays closed.
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

#[test]
fn prepared_storage_objects_land_through_ordinary_admission_without_host_destinations() {
    use ts_control::peer::{
        geometry::{HostGroup, HostPool},
        prepared::{read_spans, PreparedObject},
        read::PageReader,
        servicer::empty_event,
    };
    use ts_scheduler::{PeerEventKind, PeerEventView};

    let mut scheduler = scheduler(
        "reader",
        1,
        "Fused",
        Landing::Device,
        1_000_000,
        PAGES,
        1_000_000,
    );
    let tokens: Vec<i32> = (1..=6 * PAGE + 1).collect();
    let before = scheduler.admission_snapshot(PAGE);
    let dependency = scheduler.storage_prefix_plan(&tokens, 4 * PAGE, 4).unwrap();
    assert_eq!(dependency.max_tokens, 4 * PAGE);
    let arena = Arena::new(200_000);
    let objects: Vec<_> = dependency
        .objects
        .iter()
        .map(|object| {
            let address = 1024 + object.slot as u64 * PAGE_BYTES;
            arena.write(address, &vec![object.slot as u8 + 1; PAGE_BYTES as usize]);
            PreparedObject {
                group_id: object.key.group_id,
                slot: object.slot,
                session: "prepared-storage".into(),
                address,
                bytes: PAGE_BYTES,
            }
        })
        .collect();
    assert_eq!(
        scheduler.admission_snapshot(PAGE).available_blocks,
        before.available_blocks
    );
    scheduler
        .submit_requests(&[spec("restore", tokens, "storage-owner")])
        .unwrap();
    scheduler.advance_peer(0, &[]).unwrap();
    let looking = scheduler.next_execution_plan(true, false);
    assert!(looking
        .forward_batches()
        .iter()
        .all(|batch| batch.request_ids.is_empty()));
    assert!(looking.peer_fetches().is_empty());
    assert_eq!(
        scheduler.admission_snapshot(PAGE).available_blocks,
        before.available_blocks
    );
    let lookup = scheduler.take_peer_lookups().pop().unwrap();
    scheduler
        .advance_peer(
            1,
            &[PeerEventView {
                kind: PeerEventKind::LookupAnswered,
                lease: lookup.lease,
                num_tokens: dependency.max_tokens,
                ..empty_event()
            }],
        )
        .unwrap();
    let admitted = scheduler.next_execution_plan(true, false);
    assert!(admitted
        .forward_batches()
        .iter()
        .all(|batch| batch.request_ids.is_empty()));
    let fetches = admitted.peer_fetches();
    assert_eq!(fetches.len(), 1);
    let fetch = &fetches[0];
    assert_eq!(fetch.end_tokens, dependency.max_tokens);
    assert!(fetch
        .destinations
        .iter()
        .all(|page| page.tier == ts_scheduler::PeerTier::Device));
    assert_eq!(
        scheduler.admission_snapshot(PAGE).host_free_blocks,
        before.host_free_blocks
    );
    let mut layout = geometry(READER_BASE);
    // The schema describes existing packed objects, not an allocated L2 pool.
    layout.host = Some(HostPool {
        address: 0,
        groups: vec![HostGroup {
            parent_bytes: PAGE_BYTES,
            packing: 1,
            block_bytes: PAGE_BYTES,
            field_offsets: vec![("layer.0.kv".into(), 0)],
        }],
    });
    let reads = read_spans(&objects, &layout, &fetch.destinations).unwrap();
    for (session, spans) in reads {
        arena.read(&session, &spans).unwrap();
    }
    for page in &fetch.destinations {
        assert_eq!(
            arena.read_bytes(
                READER_BASE + page.block_id as u64 * PAGE_BYTES,
                PAGE_BYTES as usize
            ),
            vec![page.slot as u8 + 1; PAGE_BYTES as usize]
        );
    }
    // Stand in for the I/O owner's successful retirement/release agreement.
    // The native lease test verifies that agreement against a real provider.
    scheduler
        .advance_peer(
            2,
            &[PeerEventView {
                kind: PeerEventKind::FetchDone,
                request_id: "restore".into(),
                num_tokens: dependency.max_tokens,
                ..empty_event()
            }],
        )
        .unwrap();
    let compute = scheduler.next_execution_plan(true, false).forward_batches();
    assert_eq!(compute.len(), 1);
    assert_eq!(compute[0].extend_prefix_lens, [dependency.max_tokens]);
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

// Only the network storage provider is substituted; the planner, source broker,
// sockets, scheduler admission, fences, scatter reads and completion agreement
// below are the runtime implementations.
mod storage_restore {
    mod versioned {
        include!("peer/versioned_storage.rs");
    }
    use super::*;
    use std::collections::HashMap;
    use ts_control::peer::{storage::*, storage_servicer, wire::Stamp};
    use ts_l3::layout::{GroupLayout, HostLayout};
    use ts_mooncake::offload::{Ready, Replica as SsdReplica};

    #[derive(Clone)]
    struct Provider {
        objects: HashMap<String, (u64, u64)>,
        live: Arc<Mutex<usize>>,
        queries: Arc<Mutex<usize>>,
        prepared_keys: Arc<Mutex<Vec<String>>>,
    }
    struct Source {
        ready: Ready,
        live: Arc<Mutex<usize>>,
        release: bool,
    }
    impl Drop for Source {
        fn drop(&mut self) {
            if self.release {
                *self.live.lock() -= 1;
            }
        }
    }
    impl PreparedSource for Source {
        type Reading = Source;
        fn begin_read(mut self, _: Duration) -> Result<Self, Error> {
            self.release = false;
            Ok(self)
        }
    }
    impl ReadingSource for Source {
        fn source(&self) -> &Ready {
            &self.ready
        }
        fn release_retired(mut self) -> Result<bool, Error> {
            self.release = true;
            Ok(true)
        }
    }
    impl Backend for Provider {
        type Prepared = Source;
        fn query(&self, keys: &[String]) -> Result<Vec<Vec<SsdReplica>>, Error> {
            *self.queries.lock() += 1;
            Ok(keys
                .iter()
                .map(|k| {
                    self.objects
                        .get(k)
                        .map(|(_, bytes)| {
                            vec![SsdReplica {
                                endpoint: "storage-provider".into(),
                                bytes: *bytes,
                                generation: [0, 0],
                            }]
                        })
                        .unwrap_or_default()
                })
                .collect())
        }
        fn prepare(
            &self,
            _: &str,
            keys: &[String],
            _: &[u64],
            generations: &[[u64; 2]],
        ) -> Result<Source, Error> {
            assert_eq!(generations, vec![[0, 0]; keys.len()]);
            self.prepared_keys.lock().extend_from_slice(keys);
            *self.live.lock() += 1;
            Ok(Source {
                ready: Ready {
                    batch_id: 1,
                    addresses: keys.iter().map(|k| self.objects[k].0).collect(),
                    session: "storage-provider".into(),
                    ttl_ms: 60_000,
                },
                live: self.live.clone(),
                release: true,
            })
        }
    }

    #[test]
    fn hot_prefix_and_storage_suffix_read_together_and_publish_only_after_both_finish() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use ts_control::peer::{
            geometry::Spans,
            read::{PageReader, ReadOutcome},
        };

        struct ControlledReader {
            arena: Arena,
            started: Arc<AtomicUsize>,
            release: Arc<AtomicBool>,
            fail_storage: bool,
            fail_hot: bool,
        }
        impl PageReader for ControlledReader {
            fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
                self.arena.read(session, spans)
            }
            fn read_outcome(&self, session: &str, spans: &Spans) -> ReadOutcome {
                if session == "storage-provider" {
                    self.started.fetch_or(2, Ordering::SeqCst);
                    let until = Instant::now() + Duration::from_secs(5);
                    while !self.release.load(Ordering::SeqCst) && Instant::now() < until {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    if self.fail_storage {
                        return ReadOutcome::FailedRetired;
                    }
                }
                if session != "storage-provider" {
                    self.started.fetch_or(1, Ordering::SeqCst);
                    if self.fail_hot {
                        return ReadOutcome::FailedRetired;
                    }
                }
                self.arena.read(session, spans).unwrap();
                ReadOutcome::Complete
            }
        }

        for (background, fail_storage, fail_hot) in [
            (false, false, false),
            (false, true, false),
            (false, false, true),
            (true, false, false),
            (true, true, false),
            (true, false, true),
        ] {
            let arena = Arena::new(200_000);
            let holder_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
            let holder_name = format!("127.0.0.1:{}", holder_listener.port());
            let mut holder_settings = settings(&holder_name, "v1");
            if background {
                holder_settings.driver = TestDriver::Eager;
            }
            let mut holder = super::Replica::with_listener(
                holder_settings,
                holder_listener,
                HOLDER_BASE,
                &arena,
                "Fused",
            );
            let tokens: Vec<_> = (1..=9 * PAGE).collect();
            holder_computes(&mut holder, &tokens[..2 * PAGE as usize], &arena);
            let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
            let name = format!("127.0.0.1:{}", listener.port());
            let scheduler = Arc::new(Mutex::new(scheduler_with_storage(
                &name,
                1,
                "Fused",
                Landing::Device,
                1_000_000,
                PAGES,
                1_000_000,
                true,
            )));
            let started = Arc::new(AtomicUsize::new(0));
            let release = Arc::new(AtomicBool::new(false));
            let progress = if background {
                let mut scheduler = scheduler.lock();
                let progress = PeerProgress::for_replica(1, &mut scheduler).unwrap();
                assert!(scheduler.enable_async_device_peer(false));
                progress
            } else {
                PeerProgress::Coordinated
            };
            let peer = Peer::start(
                settings(&name, "v1").peer,
                progress,
                0,
                vec![geometry_with_host(READER_BASE, 0)],
                Some(listener),
                scheduler.clone(),
                PeerTransfers {
                    reader: Arc::new(ControlledReader {
                        arena: arena.clone(),
                        started: started.clone(),
                        release: release.clone(),
                        fail_storage,
                        fail_hot,
                    }),
                    pusher: Arc::new(NoPushes),
                },
            )
            .unwrap();
            let mut reader = super::Replica {
                name: name.clone(),
                scheduler,
                peer,
                decisions: Vec::new(),
                landed: Vec::new(),
            };
            reader.enable_logical_metrics();
            let layout = HostLayout {
                groups: vec![GroupLayout {
                    packing: 1,
                    block_bytes: PAGE_BYTES as usize,
                    shard_count: 1,
                }],
                lcm_block_bytes: PAGE_BYTES as usize,
                num_lcm_blocks: 64,
                owner_rank: 0,
                tp_rank: 0,
            };
            let candidate = PrefixPlan::from_scheduler(
                &reader.scheduler.lock(),
                &tokens,
                8 * PAGE,
                "test-namespace",
                &[layout],
                64,
            )
            .unwrap()
            .unwrap();
            // L3 has only the six suffix pages. The first two are available
            // solely from the hot peer; a full-prefix storage query would miss.
            let mut objects = HashMap::new();
            let mut expected_keys = Vec::new();
            for object in candidate.objects.iter().filter(|object| object.slot >= 2) {
                let address = 10_000 + object.slot as u64 * PAGE_BYTES;
                arena.write(address, &vec![object.slot as u8 + 100; PAGE_BYTES as usize]);
                objects.insert(object.key.clone(), (address, PAGE_BYTES));
                expected_keys.push(object.key.clone());
            }
            let live = Arc::new(Mutex::new(0));
            let queries = Arc::new(Mutex::new(0));
            let prepared_keys = Arc::new(Mutex::new(Vec::new()));
            let backend = Provider {
                objects,
                live: live.clone(),
                queries: queries.clone(),
                prepared_keys: prepared_keys.clone(),
            };
            let budget = SourceBudget::new(6 * PAGE_BYTES).unwrap();
            let local_scheduler = reader.scheduler.clone();
            let service = storage_servicer::Service::start(
                storage_servicer::Settings {
                    stamp: Stamp {
                        name: format!("l3@{name}"),
                        incarnation: 42,
                        identity: "deployment/v1".into(),
                    },
                    weight_version: "v1".into(),
                    members: 1,
                    limits: Limits {
                        max_query_keys: 64,
                        max_candidates: 1,
                        max_prepare_bytes: 6 * PAGE_BYTES,
                        max_source_batch_bytes: 6 * PAGE_BYTES,
                        max_source_batches: 1,
                    },
                    max_records: 8,
                    min_extra_tokens: PAGE,
                    record_lifetime: Duration::from_secs(10),
                    read_budget: Duration::from_secs(1),
                },
                Arc::new(move |_, _| Ok(vec![candidate.clone()])),
                Arc::new(move |prompt| {
                    local_scheduler
                        .lock()
                        .local_prefix_tokens(&prompt.tokens, prompt.max_tokens)
                        .map_err(|error| Error(error.to_string()))
                }),
                backend,
                budget.clone(),
            )
            .unwrap();
            reader.peer.attach_storage_source(service);
            reader
                .scheduler
                .lock()
                .submit_requests(&[spec("mixed", tokens, &holder.name)])
                .unwrap();
            let mut fetches = Vec::new();
            let until = Instant::now() + Duration::from_secs(3);
            let mut early_rows = Vec::new();
            while Instant::now() < until && started.load(Ordering::SeqCst) != 3 {
                holder.round();
                let (rows, parts) = reader.round();
                early_rows.extend(rows);
                fetches.extend(parts);
                std::thread::sleep(Duration::from_millis(2));
            }
            // The hot read can retire while the suffix is still in flight.
            for _ in 0..4 {
                holder.round();
                early_rows.extend(reader.round().0);
            }
            let observed = started.load(Ordering::SeqCst);
            let published_early = !reader.landed.is_empty();
            release.store(true, Ordering::SeqCst);
            assert_eq!(observed, 3, "both sources must read concurrently");
            assert!(
                early_rows.is_empty(),
                "a partial prefix must not enter forward"
            );
            assert!(!published_early);
            assert_eq!(fetches.len(), 2);
            assert_eq!(fetches[0].destinations.len(), 2);
            assert_eq!(fetches[1].destinations.len(), 6);
            let (prefix, extra) = rounds_until_prefill(&mut reader, Some(&mut holder), "mixed");
            assert!(extra.is_empty());
            let failed = fail_storage || fail_hot;
            assert_eq!(prefix, if failed { 0 } else { 8 * PAGE });
            if !failed {
                assert_eq!(reader.landed, [("mixed".into(), 0, 8 * PAGE)]);
                for destination in &fetches[1].destinations {
                    assert_eq!(
                        arena.read_bytes(
                            READER_BASE + destination.block_id as u64 * PAGE_BYTES,
                            PAGE_BYTES as usize
                        ),
                        vec![destination.slot as u8 + 100; PAGE_BYTES as usize]
                    );
                }
            }
            expected_keys.sort();
            let mut prepared = prepared_keys.lock().clone();
            prepared.sort();
            assert_eq!(
                prepared, expected_keys,
                "only suffix data may be staged from SSD"
            );
            assert_eq!(*queries.lock(), 1);
            assert_eq!(*live.lock(), 0);
            assert_eq!(budget.usage(), (0, 0));
            assert_eq!(reader.peer.num_landings(), 0);
            holder.round();
            assert_eq!(holder.scheduler.lock().peer_stats().pinned_blocks, 0);
        }
    }

    #[test]
    fn storage_broker_restores_through_the_real_admission_and_read_agreement() {
        for missing in [false, true] {
            let arena = Arena::new(200_000);
            let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
            let name = format!("127.0.0.1:{}", listener.port());
            let mut reader = super::Replica::with_landing(
                settings(&name, "v1"),
                listener,
                geometry_with_host(READER_BASE, 0),
                &arena,
                "Fused",
                Landing::Device,
            );
            reader.enable_logical_metrics();
            let tokens: Vec<_> = (1..=5 * PAGE).collect();
            let layout = HostLayout {
                groups: vec![GroupLayout {
                    packing: 1,
                    block_bytes: PAGE_BYTES as usize,
                    shard_count: 1,
                }],
                lcm_block_bytes: PAGE_BYTES as usize,
                num_lcm_blocks: 64,
                owner_rank: 0,
                tp_rank: 0,
            };
            let candidate = PrefixPlan::from_scheduler(
                &reader.scheduler.lock(),
                &tokens,
                4 * PAGE,
                "test-namespace",
                &[layout],
                64,
            )
            .unwrap()
            .unwrap();
            let mut objects = HashMap::new();
            for object in &candidate.objects {
                let address = 10_000 + object.slot as u64 * PAGE_BYTES;
                arena.write(address, &vec![object.slot as u8 + 1; PAGE_BYTES as usize]);
                objects.insert(object.key.clone(), (address, PAGE_BYTES));
            }
            if missing {
                objects.remove(&candidate.objects[0].key);
            }
            let live = Arc::new(Mutex::new(0));
            let queries = Arc::new(Mutex::new(0));
            let backend = Provider {
                objects,
                live: live.clone(),
                queries: queries.clone(),
                prepared_keys: Arc::new(Mutex::new(Vec::new())),
            };
            let name = format!("l3@{}", reader.name);
            let expected = tokens.clone();
            let local_scheduler = reader.scheduler.clone();
            let source_budget = SourceBudget::new(4096).unwrap();
            let service = storage_servicer::Service::start(
                storage_servicer::Settings {
                    stamp: Stamp {
                        name: name.clone(),
                        incarnation: 42,
                        identity: "deployment/v1".into(),
                    },
                    weight_version: "v1".into(),
                    members: 1,
                    limits: Limits {
                        max_query_keys: 64,
                        max_candidates: 1,
                        max_prepare_bytes: 4096,
                        max_source_batch_bytes: 4096,
                        max_source_batches: 4,
                    },
                    max_records: 8,
                    min_extra_tokens: PAGE,
                    record_lifetime: Duration::from_secs(5),
                    read_budget: Duration::from_secs(1),
                },
                Arc::new(move |prompt, _weight_version| {
                    assert_eq!(prompt.tokens, expected[..prompt.tokens.len()]);
                    Ok(vec![candidate.clone()])
                }),
                Arc::new(move |prompt| {
                    local_scheduler
                        .lock()
                        .local_prefix_tokens(&prompt.tokens, prompt.max_tokens)
                        .map_err(|error| ts_control::peer::storage::Error(error.to_string()))
                }),
                backend,
                source_budget.clone(),
            )
            .unwrap();
            let before = reader.scheduler.lock().admission_snapshot(PAGE);
            let probe = ts_control::peer::wire::Request::Lookup {
                lease: ts_scheduler::PeerLeaseView {
                    reader: reader.name.clone(),
                    id: 999,
                },
                prompt: ts_control::peer::wire::Prompt {
                    tokens: tokens.clone(),
                    prefix_hashes: Vec::new(),
                    max_tokens: 4 * PAGE,
                },
            };
            service.request(probe).unwrap();
            let until = Instant::now() + Duration::from_secs(5);
            let reply = loop {
                if let Some(reply) = service.poll().next() {
                    break reply;
                }
                assert!(Instant::now() < until);
                std::thread::sleep(Duration::from_millis(2));
            };
            match reply {
                ts_control::peer::wire::Reply::Answer { answer, .. } => {
                    assert_eq!(answer.num_tokens, if missing { 0 } else { 4 * PAGE });
                }
                _ => panic!("expected metadata lookup answer"),
            }
            assert_eq!(*live.lock(), 0, "lookup must not prepare SSD buffers");
            assert_eq!(source_budget.usage(), (0, 0));
            assert_eq!(
                reader
                    .scheduler
                    .lock()
                    .admission_snapshot(PAGE)
                    .available_blocks,
                before.available_blocks,
                "lookup must not reserve L1"
            );
            reader.peer.attach_storage_source(service);
            reader
                .scheduler
                .lock()
                .submit_requests(&[spec("storage-r", tokens, &name)])
                .unwrap();
            let first = reader.round();
            assert!(first.0.is_empty());
            assert_eq!(
                reader
                    .scheduler
                    .lock()
                    .admission_snapshot(PAGE)
                    .available_blocks,
                before.available_blocks,
                "lookup and preparation cannot allocate L1"
            );
            let (prefix, landings) = rounds_until_prefill(&mut reader, None, "storage-r");
            if missing {
                assert_eq!(prefix, 0);
                assert!(landings.is_empty());
            } else {
                assert_eq!(prefix, 4 * PAGE);
                assert_eq!(landings.len(), 1);
                for destination in &landings[0].destinations {
                    assert!(destination.tier == ts_scheduler::PeerTier::Device);
                    assert_eq!(
                        arena.read_bytes(
                            READER_BASE + destination.block_id as u64 * PAGE_BYTES,
                            PAGE_BYTES as usize
                        ),
                        vec![destination.slot as u8 + 1; PAGE_BYTES as usize]
                    );
                }
                assert_eq!(
                    reader.scheduler.lock().peer_stats().landed_tokens,
                    (4 * PAGE) as u64
                );
            }
            assert_eq!(
                reader
                    .scheduler
                    .lock()
                    .admission_snapshot(PAGE)
                    .host_free_blocks,
                before.host_free_blocks
            );
            assert_eq!(
                *live.lock(),
                0,
                "source release acknowledged before forward"
            );
            assert_eq!(source_budget.usage(), (0, 0));
            // Once the reader has computed/published the complete prefix, the
            // same routing probe must stop before touching the storage backend.
            reader.advance(&[
                event(ForwardEventKind::ExtendResult, "storage-r", vec![7]),
                event(ForwardEventKind::Finish, "storage-r", Vec::new()),
            ]);
            reader.round();
            let queries_before = *queries.lock();
            reader
                .scheduler
                .lock()
                .submit_requests(&[spec("storage-hot", (1..=4 * PAGE + 1).collect(), &name)])
                .unwrap();
            // A fully local prefix is admitted without a storage lookup.
            reader.round();
            assert_eq!(*queries.lock(), queries_before);
        }
    }
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
    rs.peer.timeouts.answer_us = 40_000;
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
        let mut s = scheduler(
            "scratch",
            99,
            "Fused",
            Landing::Device,
            1_000_000,
            PAGES,
            1_000_000,
        );
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

#[test]
fn an_abandoned_pin_reports_the_lease_without_recycling_live_pages() {
    let arena = Arena::new(270_000);
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let mut cfg = settings(&format!("127.0.0.1:{}", listener.port()), "v1");
    cfg.driver = TestDriver::Automatic;
    cfg.peer.record_lifetime = Duration::from_millis(500);
    let mut holder = Replica::with_landing(
        cfg,
        listener,
        geometry_with_host(HOLDER_BASE, 215_000),
        &arena,
        "Fused",
        Landing::Host,
    );
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    let fixture = Replica::start("v1", READER_BASE, &arena);
    let mut prompt = tokens;
    prompt.push(500_000);
    let mut cache = fixture.scheduler.lock();
    cache
        .submit_requests(&[spec("source", prompt, &holder.name)])
        .unwrap();
    let _plan = cache.next_execution_plan(true, false);
    let lookup = cache.take_peer_lookups().pop().unwrap();
    drop(cache);
    let (tx, rx) = mpsc::channel();
    let reader = PeerNet::start("127.0.0.1", 0, tx).unwrap();
    let lease = ts_scheduler::PeerLeaseView {
        reader: format!("127.0.0.1:{}", reader.port()),
        id: 41,
    };
    let receive = || {
        let frames = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let Inbound::Reply(Reply::Answer { answer, .. }) = Inbound::decode(&frames).unwrap() else {
            panic!("expected answer")
        };
        answer
    };
    reader
        .request(
            &holder.name,
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
    assert!(receive().num_tokens > 0);
    reader
        .request(
            &holder.name,
            &Request::Pin {
                lease: lease.clone(),
            },
        )
        .unwrap();
    assert!(receive().held);
    // The live reader may spend longer than one lifetime in DMA.
    for _ in 0..4 {
        std::thread::sleep(Duration::from_millis(150));
        reader
            .request(
                &holder.name,
                &Request::Reading {
                    lease: lease.clone(),
                },
            )
            .unwrap();
    }
    holder.peer.agree(&[]).unwrap();
    assert!(holder.scheduler.lock().peer_stats().pinned_blocks > 0);
    drop(reader);
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        match holder.peer.agree(&[]) {
            Err(ts_control::peer::PeerError::UnreleasedPin(id)) => {
                assert_eq!(id, lease);
                break;
            }
            Ok(_) => std::thread::sleep(Duration::from_millis(5)),
            Err(error) => panic!("unexpected error {error}"),
        }
        assert!(Instant::now() < until, "holder never surfaced the orphan");
    }
    assert!(holder.scheduler.lock().peer_stats().pinned_blocks > 0);
    holder.peer.end_lookup("unused", "cancelled");
    assert!(matches!(
        holder.peer.agree(&[]),
        Err(ts_control::peer::PeerError::WorkerExited)
    ));
}

fn host_during_forward(autonomous: bool, holder_delay: Duration, contend: bool, hold_reader: bool) {
    let arena = Arena::new(270_000);
    let bound = || PeerListener::bind("127.0.0.1", 0).unwrap();
    let address = |port| format!("127.0.0.1:{port}");
    // Only the late-reply control needs the production 50 ms deadline.
    let lookup_budget = if holder_delay.is_zero() {
        Duration::from_secs(1)
    } else {
        Duration::from_millis(50)
    };
    let config = |name: &str| {
        let mut cfg = settings(name, "v1");
        cfg.driver = if autonomous {
            TestDriver::Automatic
        } else {
            TestDriver::Coordinated
        };
        cfg.peer.timeouts.answer_us = lookup_budget.as_micros() as i64;
        if hold_reader {
            // The holder's soft lease must survive the intentional wait on the reader lock.
            cfg.soft_pin_us = 4_000_000;
        }
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
    let (ack_while_locked, observed_ack) = mpsc::channel();
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
            // Empty rounds can hand the cache to a parked holder without
            // otherwise unlocking it. LOOKUP must participate in that handoff.
            let mut round = cache.lock();
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
            let deadline = Instant::now() + Duration::from_secs(2);
            let reply = loop {
                parking_lot::MutexGuard::bump(&mut round);
                if let Ok(reply) = rx.try_recv() {
                    break reply;
                }
                assert!(Instant::now() < deadline, "holder missed the lock handoff");
                std::thread::yield_now();
            };
            drop(round);
            assert!(matches!(Inbound::decode(&reply),
                Ok(Inbound::Reply(Reply::Answer{answer,..})) if answer.num_tokens == tokens.len() as i32));
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
                observed_ack
                    .recv_timeout(Duration::from_secs(3))
                    .expect("release ACK before unlocking");
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
                let mut guard = (hold_reader && lookup).then(|| reader_cache.lock());
                proxy.reply(&actual,&reply).unwrap();
                if let Some(round) = guard.as_mut() {
                    // Keep the lock through the budget even if the wire reply was immediate.
                    std::thread::sleep(lookup_budget + Duration::from_millis(10));
                    let until = Instant::now() + Duration::from_secs(2);
                    while round.peer_stats().landing_requests == 0 {
                        parking_lot::MutexGuard::bump(round);
                        assert!(Instant::now() < until, "reader missed the scheduler handoff");
                        std::thread::yield_now();
                    }
                }
                drop(guard);
                if release {
                    let at = Instant::now();
                    if contend {
                        let Reply::Answer {answer, ..} = &reply else {unreachable!()};
                        proxy.request(&endpoint, &Request::Release {
                            lease: answer.lease.clone(), demote: true
                        }).unwrap();
                        let repeated = wire_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                        assert!(matches!(Inbound::decode(&repeated), Ok(Inbound::Reply(duplicate)) if duplicate == reply));
                        ack_while_locked.send(()).unwrap();
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
            if !contend {
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
    held_host_read(false);
}

#[test]
fn shutdown_drains_submitted_dma_and_discards_the_next_host_reservation() {
    held_host_read(true);
}

fn held_host_read(shutdown: bool) {
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
        config.driver = TestDriver::Automatic;
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
        .submit_requests(&[spec("cancel", prompt.clone(), &holder.name)])
        .unwrap();
    reader.round();
    reading
        .recv_timeout(Duration::from_secs(2))
        .expect("READ submitted");
    if shutdown {
        let mut cache = reader.scheduler.lock();
        cache
            .submit_requests(&[spec("queued", prompt, &holder.name)])
            .unwrap();
        let _plan = cache.next_execution_plan(true, false);
        let lookups = cache.take_peer_lookups();
        assert_eq!(lookups.len(), 1);
        reader.peer.dispatch(vec![], lookups, 0).unwrap();
        let until = Instant::now() + Duration::from_secs(3);
        loop {
            if reader
                .peer
                .agree(&[])
                .unwrap()
                .events
                .iter()
                .any(|e| e.num_tokens > 0)
            {
                break;
            }
            assert!(Instant::now() < until, "second reservation never queued");
            std::thread::yield_now();
        }
        drop(reader.peer);
        assert!(holder.scheduler.lock().peer_stats().pinned_blocks > 0);
        resume.send(()).unwrap();
        // Disconnection proves that the read pool exited without picking up B.
        assert_eq!(
            reading.recv_timeout(Duration::from_secs(3)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        );
        drop(cache);
        assert_eq!(holder.scheduler.lock().peer_stats().pinned_blocks, 0);
        return;
    }
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

#[derive(Clone, Copy, Debug)]
enum CheckpointState {
    Ready,
    Evicted,
    Replaying,
    Retired,
}

fn delayed_checkpoint_pin(state: CheckpointState) {
    let arena = Arena::new(200_000);
    let replica = |base| {
        let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let mut cfg = settings(&format!("127.0.0.1:{}", listener.port()), "v1");
        cfg.driver = TestDriver::Automatic;
        // This test exercises checkpoint expiry; allow the normal loopback lookup budget.
        // Also expire the Rust holder's lookup metadata before the PIN.
        cfg.peer.record_lifetime = Duration::from_secs(2);
        cfg.pages = 512;
        cfg.soft_pin_us = 2_000_000;
        Replica::with_landing(
            cfg,
            listener,
            geometry(base),
            &arena,
            "Fused",
            Landing::Device,
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
    if matches!(state, CheckpointState::Replaying | CheckpointState::Retired) {
        holder.peer.block_device_pins(true);
    }
    std::thread::sleep(Duration::from_secs(4));
    // Retire the soft lease explicitly before the late grant.
    holder.round();
    assert_eq!(holder.scheduler.lock().peer_stats().leases, 0);
    match state {
        CheckpointState::Ready | CheckpointState::Replaying => {}
        CheckpointState::Evicted => assert!(holder.scheduler.lock().clear_cache()),
        CheckpointState::Retired => holder.peer.block_device_pins(false),
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
    if matches!(state, CheckpointState::Ready | CheckpointState::Retired) {
        assert_eq!(prefix, 2048);
        assert_eq!(arena.read_bytes(addr, 1), vec![29]);
    } else {
        assert_eq!(prefix, 0);
        assert_eq!(arena.read_bytes(addr, 1), vec![0]);
    }
    assert_eq!(holder.scheduler.lock().peer_stats().pinned_blocks, 0);
    println!(
        "delayed PIN state={state:?} forward_ms=4000 fetches={} consumed={prefix}",
        fetches.len()
    );
}
#[test]
fn expired_checkpoint_pin_keeps_an_unchanged_prefix() {
    delayed_checkpoint_pin(CheckpointState::Ready);
}
#[test]
fn expired_checkpoint_pin_rejects_an_evicted_prefix() {
    delayed_checkpoint_pin(CheckpointState::Evicted);
}

#[test]
fn expired_checkpoint_pin_rejects_device_pages_during_replay() {
    delayed_checkpoint_pin(CheckpointState::Replaying);
}

#[test]
fn expired_checkpoint_pin_accepts_device_pages_after_replay_retires() {
    delayed_checkpoint_pin(CheckpointState::Retired);
}

// Exercise the lookup-answer bridge as well as the scheduler's feedback key.
#[test]
fn slow_disk_push_does_not_penalize_host_only_reads() {
    use ts_control::peer::servicer::empty_event;
    use ts_scheduler::{PeerEventKind, PeerEventView, PeerPageView};

    let mut reader = scheduler(
        "127.0.0.1:3100",
        1,
        "Fused",
        Landing::Device,
        1_000_000,
        512,
        2_000_000,
    );
    assert!(reader.enable_async_device_peer(true));
    let mut now = 1;
    for (i, (id, tiers, expect_read)) in [
        ("disk", vec![PeerTier::Host, PeerTier::Disk], true),
        ("same-path", vec![PeerTier::Host, PeerTier::Disk], false),
        ("host", vec![PeerTier::Host], true),
    ]
    .into_iter()
    .enumerate()
    {
        reader.advance_peer(now, &[]).unwrap();
        let offset = 10_000 * i as i32;
        reader
            .submit_requests(&[spec(
                id,
                (offset..offset + 4096 + PAGE).collect(),
                "127.0.0.1:3101",
            )])
            .unwrap();
        reader.next_execution_plan(true, false);
        let lookup = reader.take_peer_lookups().pop().unwrap();
        let pages = tiers
            .into_iter()
            .enumerate()
            .map(|(i, tier)| PeerPageView {
                group_id: 0,
                slot: i as i32,
                block_id: 1,
                tier,
            })
            .collect();
        reader
            .advance_peer(
                now,
                &[PeerEventView {
                    kind: PeerEventKind::LookupAnswered,
                    lease: lookup.lease,
                    num_tokens: 4096,
                    pages,
                    ..empty_event()
                }],
            )
            .unwrap();
        let plan = reader.next_execution_plan(true, false);
        assert_eq!(!plan.peer_fetches().is_empty(), expect_read, "{id}");
        if expect_read {
            let start = now;
            now += 4_000_000;
            reader
                .advance_peer(
                    now,
                    &[PeerEventView {
                        kind: PeerEventKind::FetchDone,
                        request_id: id.into(),
                        num_tokens: 4096,
                        dispatch_us: start,
                        writable_us: start,
                        terminal_us: now,
                        ..empty_event()
                    }],
                )
                .unwrap();
            assert!(!reader
                .next_execution_plan(true, false)
                .forward_batches()
                .is_empty());
        }
        reader
            .advance(&[
                event(ForwardEventKind::ExtendResult, id, vec![7]),
                event(ForwardEventKind::Finish, id, vec![]),
            ])
            .unwrap();
        reader.next_execution_plan(true, false);
        now += 1;
    }
}

#[test]
fn topology_selects_the_protocol_owner() {
    for (landing, expected) in [
        (Landing::Host, PeerProgress::Host),
        (Landing::Device, PeerProgress::Device),
    ] {
        let mut scheduler = scheduler("test", 1, "Fused", landing, 50_000, PAGES, 2_000_000);
        assert!(matches!(
            PeerProgress::for_replica(0, &mut scheduler),
            Err(ts_control::peer::PeerError::Topology)
        ));
        assert_eq!(
            PeerProgress::for_replica(2, &mut scheduler).unwrap(),
            PeerProgress::Coordinated
        );
        assert_eq!(
            PeerProgress::for_replica(1, &mut scheduler).unwrap(),
            expected
        );
    }
}

#[test]
fn a_device_reader_rejects_the_previous_device_namespace() {
    let arena = Arena::new(200_000);
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let mut old = settings(&format!("127.0.0.1:{}", listener.port()), "v1");
    // The coordinated fixture advertises this literal deployment/weight stamp.
    old.peer.deployment_id = "async-device-v1:deployment".into();
    let mut holder = Replica::with_listener(old, listener, HOLDER_BASE, &arena, "Fused");
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let mut current = settings(&format!("127.0.0.1:{}", listener.port()), "v1");
    current.driver = TestDriver::Automatic;
    let mut reader = Replica::with_listener(current, listener, READER_BASE, &arena, "Fused");
    reader.enable_logical_metrics();
    let mut tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    tokens.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", tokens, &holder.name)])
        .unwrap();
    let (prefix, fetches) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert!(reader.decisions.contains(&("r".into(), "incompatible")));
    assert_eq!((prefix, fetches.len()), (0, 0));
}
