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
    scheduler_with_storage(name, incarnation, role, landing, false)
}

fn scheduler_with_storage(
    name: &str,
    incarnation: u64,
    role: &str,
    landing: Landing,
    storage: bool,
) -> Scheduler {
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

fn settings(name: &str, weight_version: &str) -> PeerSettings {
    PeerSettings {
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
        let scheduler = Arc::new(Mutex::new(scheduler(&settings.name, 1, role, landing)));
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

    let mut scheduler = scheduler("reader", 1, "Fused", Landing::Device);
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

        for (fail_storage, fail_hot) in [(false, false), (true, false), (false, true)] {
            let arena = Arena::new(200_000);
            let mut holder = super::Replica::start("v1", HOLDER_BASE, &arena);
            let tokens: Vec<_> = (1..=9 * PAGE).collect();
            holder_computes(&mut holder, &tokens[..2 * PAGE as usize], &arena);
            let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
            let name = format!("127.0.0.1:{}", listener.port());
            let scheduler = Arc::new(Mutex::new(scheduler_with_storage(
                &name,
                1,
                "Fused",
                Landing::Device,
                true,
            )));
            let started = Arc::new(AtomicUsize::new(0));
            let release = Arc::new(AtomicBool::new(false));
            let peer = Peer::start(
                settings(&name, "v1"),
                0,
                vec![geometry_with_host(READER_BASE, 0)],
                Some(listener),
                scheduler.clone(),
                Arc::new(ControlledReader {
                    arena: arena.clone(),
                    started: started.clone(),
                    release: release.clone(),
                    fail_storage,
                    fail_hot,
                }),
                Arc::new(NoPushes),
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
