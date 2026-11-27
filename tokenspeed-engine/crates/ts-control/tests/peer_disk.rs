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

//! A holder whose prefix only its Disk tier still holds pushes it into a
//! reader's landing (`docs/rfcs/0006-disk-tier.md`, Peer sharing): real
//! schedulers, a real Disk file and staging ring, real sockets, and the
//! reader's pages in a test arena the holder's writes copy into. The file
//! lives under `TS_DISK_TEST_DIR`, or the temporary directory, which must
//! support `O_DIRECT` (not tmpfs).

use std::{
    collections::HashMap,
    ffi::CString,
    path::PathBuf,
    sync::{mpsc, Arc, Once},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};
use ts_control::{
    disk::{DiskSettings, DiskTier},
    peer::{
        geometry::{DeviceField, HostGroup, HostPool, RankGeometry, Spans},
        landing::Timeouts,
        net::PeerNet,
        push::{PageWriter, PushEndpoint, PushSource},
        read::Arena,
        registered::{Member, Replica as RegisteredReplica},
        v1::{Generation, Key, Message},
        wire::Prompt,
        Peer, PeerListener, PeerProgress, PeerSettings, PeerTransfers,
    },
};
use ts_disk::{Backend, ExecutorConfig, Memory};
use ts_l3::{GroupLayout, HostBuffer, HostLayout};
use ts_scheduler::{
    CacheEvent, CacheEventKind, CacheOpKind, DiskOpKind, DiskOpView, ForwardEvent,
    ForwardEventKind, PeerFetchView, RequestSpec, Scheduler,
};

const PAGE: i32 = 4;
const BLOCK_BYTES: usize = 4096;
const DEVICE_PAGES: usize = 9;
const HOST_BLOCKS: usize = 8;
const DISK_BLOCKS: usize = 32;
/// The reader's Device pages in the arena.
const READER_BASE: u64 = 0;

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

/// A fused scheduler with peer sharing: the holder's with eight usable
/// Device blocks over a write-back Host tier and a Disk tier, the reader's
/// with a Device cache alone.
fn scheduler(name: &str, incarnation: u64, holder: bool) -> Scheduler {
    interpreter();
    let tiers = if holder {
        format!(
            "cfg.num_device_pages = {DEVICE_PAGES}\n\
             cfg.num_snapshot_pages = {DEVICE_PAGES}\n\
             cfg.max_retracted_requests = 8\n\
             cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = {host_pages}\n\
             cfg.host_group_packing = [1]\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack\n\
             cfg.num_disk_pages = {disk_pages}\n\
             cfg.disk_config = ts.DiskConfig(max_operations=4, max_blocks_per_operation=64,\n\
                                             max_host_blocks=4, max_write_host_blocks=2,\n\
                                             max_queued_cleanings=64, min_prefetch_tokens=1,\n\
                                             wait_timeout_us=10_000_000, ready_timeout_us=10_000_000,\n\
                                             min_clean_tokens=0, require_reuse_for_cleaning=False)\n\
             device_pages = {DEVICE_PAGES}",
            host_pages = HOST_BLOCKS + 1,
            disk_pages = DISK_BLOCKS + 1,
        )
    } else {
        "cfg.num_device_pages = 64\n\
         cfg.num_snapshot_pages = 1\n\
         cfg.disable_l2_cache = True\n\
         device_pages = 64"
            .to_owned()
    };
    Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.Fused
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = 64
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
{tiers}
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full", block_granularity={PAGE},
                                        total_pages=device_pages, attn_kind=ts.AttnKind.Full,
                                        transfer_policy=ts.CacheTransferPolicy.FullSuffix)]
cfg.peer_config = ts.PeerConfig(name="{name}", incarnation={incarnation}, max_leases=8,
                                soft_pin_us=1_000_000, hard_pin_us=1_000_000,
                                lookup_timeout_us=1_000_000, min_fetch_tokens={PAGE},
                                landing_tier=ts.PeerLandingTier.Device, max_host_landing_blocks=0,
                                host_ready_timeout_us=0, versioned_shards=False)
result = ts.Scheduler(cfg)
"#
        );
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

fn layout(num_lcm_blocks: usize) -> HostLayout {
    HostLayout {
        groups: vec![GroupLayout {
            packing: 1,
            block_bytes: BLOCK_BYTES,
            shard_count: 1,
        }],
        lcm_block_bytes: BLOCK_BYTES,
        num_lcm_blocks,
        owner_rank: 0,
        tp_rank: 0,
    }
}

/// A rank's geometry: one field per page, its Device pages at `base` in the
/// arena, and a Host pool (and so Disk blocks) of one-page blocks.
fn geometry(base: u64, push: &str) -> RankGeometry {
    RankGeometry {
        session: "local".into(),
        device: vec![DeviceField {
            group_id: 0,
            field_id: "layer.0.kv".into(),
            address: base,
            page_stride: BLOCK_BYTES as u64,
            bytes: BLOCK_BYTES as u64,
        }],
        host: Some(HostPool {
            address: 0,
            groups: vec![HostGroup {
                parent_bytes: BLOCK_BYTES as u64,
                packing: 1,
                block_bytes: BLOCK_BYTES as u64,
                field_offsets: vec![("layer.0.kv".into(), 0)],
            }],
        }),
        push: Some(push.to_owned()),
    }
}

/// The holder's transfer engine in the test: its writes copy out of the
/// staging ring, real memory, into the arena the reader's pages live in.
struct ArenaWriter {
    ring: Arc<Memory>,
    arena: Arena,
}

impl PageWriter for ArenaWriter {
    fn write(&self, _session: &str, spans: &Spans) -> Result<(), String> {
        for ((&source, &destination), &length) in spans
            .sources
            .iter()
            .zip(&spans.destinations)
            .zip(&spans.lengths)
        {
            let mut bytes = vec![0; length as usize];
            self.ring
                .read_at(source as usize - self.ring.address(), &mut bytes);
            self.arena.write(destination, &bytes);
        }
        Ok(())
    }
}

struct ScratchFile(PathBuf);

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn settings(name: &str, incarnation: u64) -> PeerSettings {
    PeerSettings {
        name: name.to_owned(),
        incarnation,
        deployment_id: "deployment".into(),
        weight_version: "v1".into(),
        timeouts: Timeouts {
            answer_us: 1_000_000,
            quarantine_us: 10_000,
            push_us: 2_000_000,
        },
        record_lifetime: Duration::from_secs(5),
    }
}

/// One single-member replica, its Disk tier if any, and the data plane's part
/// played in memory: Device copies finish at once, and each page the Host
/// receives is a byte pattern of its own, which a cleaning write takes to
/// Disk.
struct Replica {
    name: String,
    scheduler: Arc<Mutex<Scheduler>>,
    peer: Peer,
    disk: Option<(DiskTier, Memory, ScratchFile)>,
    acks: Vec<CacheEvent>,
    in_flight: HashMap<u64, DiskOpView>,
    /// The bytes each cleaned Disk block holds.
    on_disk: HashMap<i32, Vec<u8>>,
    next_pattern: u8,
}

impl Replica {
    fn start(holder: bool, arena: &Arena) -> Self {
        let incarnation = if holder { 1 } else { 2 };
        let mut listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let name = format!("127.0.0.1:{}", listener.port());
        let push = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", incarnation).unwrap();
        let disk = holder.then(|| {
            let dir = std::env::var_os("TS_DISK_TEST_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            let file = ScratchFile(dir.join(format!("ts-peer-disk-{}", std::process::id())));
            let host = Memory::map(HOST_BLOCKS * BLOCK_BYTES).unwrap();
            let tier = DiskTier::start(
                &DiskSettings {
                    file: file.0.clone(),
                    file_bytes: (DISK_BLOCKS * BLOCK_BYTES) as u64,
                    executor: ExecutorConfig {
                        backend: Backend::Psync,
                        queue_depth: 4,
                        write_queue_depth: 2,
                        max_io_bytes: 1 << 20,
                        write_bytes_per_second: 1 << 30,
                    },
                    host: HostBuffer {
                        base: host.address(),
                        len: host.size(),
                    },
                    host_layout: layout(HOST_BLOCKS),
                    disk_layout: layout(DISK_BLOCKS),
                    max_operations: 4,
                    push_ring_bytes: 2 * BLOCK_BYTES,
                },
                None,
            )
            .unwrap();
            let (executor, disk_layout, ring, slot_bytes) = tier.push_source().unwrap();
            push.serve(PushSource {
                executor,
                disk_layout,
                writer: Arc::new(ArenaWriter {
                    ring: ring.clone(),
                    arena: arena.clone(),
                }),
                ring,
                slot_bytes,
            })
            .unwrap();
            (tier, host, file)
        });
        let scheduler = Arc::new(Mutex::new(scheduler(&name, incarnation, holder)));
        // The holder's Device pages are never read: its prefix is on Disk.
        let base = if holder { 1 << 30 } else { READER_BASE };
        let geometry = geometry(base, push.name());
        let registered = holder.then(|| {
            let wire = serde_json::json!({
                "plan": {"prefix_granularity": PAGE, "lcm_block_bytes": BLOCK_BYTES,
                    "num_lcm_blocks": DEVICE_PAGES - 1,
                    "groups": [{"group_id":"full","cache_blocks_per_lcm_block":1,"page_count":DEVICE_PAGES}],
                    "planes": [{"plane_id":"kv","bytes_per_lcm_block":BLOCK_BYTES,"arena_offset_bytes":0}],
                    "fields": [{"group_id":"full","field_id":"layer.0.kv","plane_id":"kv",
                        "shape":[PAGE, BLOCK_BYTES / PAGE as usize],"dtype":"uint8","field_offset_bytes":0,
                        "page_stride_bytes":BLOCK_BYTES}], "resident_block_bytes":null},
                "group_specs":[{"group_id":"full","attn_kind":"full_history","rows_per_page":PAGE,
                    "entry_stride_tokens":1,"sliding_window_tokens":null,"replayable":false,
                    "transfer_policy":"full_suffix","shard_count":1}],
                "transfer_schema":{"fields":[]},"field_addresses":null
            });
            let member = Member {
                geometry: geometry.clone(),
                bounds: vec![(HOST_BLOCKS + 1) as i64],
                regions: vec![(base, (DEVICE_PAGES * BLOCK_BYTES) as u64)],
            };
            let layout = Arc::new(RegisteredReplica::register([7; 32], wire.to_string().as_bytes(), &[member.encode(0)]).unwrap());
            listener.set_registered_layout(layout.clone());
            (layout, listener.versioned_channel())
        });
        let mut peer = Peer::start(
            settings(&name, incarnation),
            PeerProgress::Coordinated,
            0,
            vec![geometry],
            Some(listener),
            scheduler.clone(),
            PeerTransfers {
                reader: Arc::new(arena.clone()),
                pusher: Arc::new(push),
            },
        )
        .unwrap();
        if let Some((layout, channel)) = registered {
            peer.enable_versioned(layout, name.clone(), incarnation, Some(channel), 0);
        }
        Self {
            name,
            scheduler,
            peer,
            disk,
            acks: Vec::new(),
            in_flight: HashMap::new(),
            on_disk: HashMap::new(),
            next_pattern: 1,
        }
    }

    /// One control round: the Disk tier's and the peers' agreements at the
    /// head, then the plan and its work. Returns the plan's prefill rows and
    /// its landings.
    fn round(&mut self) -> (Vec<(String, i32)>, Vec<PeerFetchView>) {
        let mut scheduler = self.scheduler.lock();
        if let Some((disk, host, _)) = &mut self.disk {
            let storage = disk.agree(&[disk.report()], true, true).unwrap();
            for done in &storage.disk_done {
                assert!(done.success, "Disk operation {} failed", done.op_id);
                let op = self.in_flight.remove(&done.op_id).expect("a started op");
                if op.kind == DiskOpKind::Write {
                    // The Host copy may be dropped now; the Disk holds it.
                    for &block in &op.host_blocks {
                        host.write_at((block as usize - 1) * BLOCK_BYTES, &[0; BLOCK_BYTES]);
                    }
                }
            }
            let acks = std::mem::take(&mut self.acks);
            scheduler.advance_head(&storage, &acks).unwrap();
        }
        let report = self.peer.report();
        let round = self
            .peer
            .agree_with_scheduler(&[report], &mut scheduler)
            .unwrap();
        scheduler.advance_peer(round.now_us, &round.events).unwrap();
        let plan = scheduler.next_execution_plan(true, false);
        let fetches = plan.peer_fetches();
        let lookups = scheduler.take_peer_lookups();
        drop(scheduler);
        if let Some((disk, host, _)) = &mut self.disk {
            for op in plan.disk_ops() {
                if op.kind == DiskOpKind::Write {
                    for row in 0..op.group_ids.len() {
                        let page = vec![self.next_pattern; BLOCK_BYTES];
                        self.next_pattern = self.next_pattern.wrapping_add(1).max(1);
                        host.write_at((op.host_blocks[row] as usize - 1) * BLOCK_BYTES, &page);
                        self.on_disk.insert(op.disk_blocks[row], page);
                    }
                }
                self.in_flight.insert(op.op_id, op);
            }
            disk.submit(&plan).unwrap();
            self.acks = plan
                .cache_ops()
                .into_iter()
                .map(|op| CacheEvent {
                    kind: match op.kind {
                        CacheOpKind::WriteBack => CacheEventKind::WriteBackDone,
                        CacheOpKind::LoadBack => CacheEventKind::LoadBackDone,
                        CacheOpKind::SnapshotStore => CacheEventKind::SnapshotDone,
                        CacheOpKind::SnapshotRestore => CacheEventKind::RestoreDone,
                        _ => unreachable!(),
                    },
                    op_id: op.op_id,
                })
                .collect();
        }
        self.peer.dispatch(fetches.clone(), lookups, round.now_us);
        if plan.needs_device() && !fetches.is_empty() {
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
        (rows, fetches)
    }

    fn submit(&self, request: &str, tokens: Vec<i32>, holder: &str) {
        self.scheduler
            .lock()
            .submit_requests(&[RequestSpec {
                admission_id: 0,
                request_id: request.to_owned(),
                session_id: String::new(),
                cache_group_id: String::new(),
                shared_prefix_tokens: Vec::new(),
                peer_holder: holder.to_owned(),
                tokens,
                max_new_tokens: 1,
                max_cached_prefix_tokens: i32::MAX,
            }])
            .unwrap();
    }

    fn finish(&self, request: &str) {
        let event = |kind, tokens| ForwardEvent {
            kind,
            request_id: request.to_owned(),
            tokens,
            spec_candidate_ids: Vec::new(),
            reserve_num_tokens: 0,
            bootstrap_token: -1,
        };
        self.scheduler
            .lock()
            .advance(&[
                event(ForwardEventKind::ExtendResult, vec![900]),
                event(ForwardEventKind::Finish, Vec::new()),
            ])
            .unwrap();
    }

    /// Admits, prefills and finishes a local request.
    fn compute(&mut self, request: &str, tokens: Vec<i32>) {
        self.submit(request, tokens, "");
        let start = Instant::now();
        while !self.round().0.iter().any(|(rid, _)| rid == request) {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "{request} never prefilled"
            );
        }
        self.finish(request);
    }

    /// Runs rounds until every queued cleaning is written.
    fn settle(&mut self) {
        let start = Instant::now();
        while !self.in_flight.is_empty()
            || !self.acks.is_empty()
            || self.scheduler.lock().disk_stats().queued_cleanings > 0
        {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "cleaning stalled"
            );
            self.round();
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Exercise the sharded reader's wire messages against the same real holder as the legacy push.
fn v1_exchange(
    holder: &mut Replica,
    reader: &PeerNet,
    replies: &mpsc::Receiver<Vec<bytes::Bytes>>,
    request: Message,
) -> Message {
    reader.versioned(&holder.name, &request).unwrap();
    let start = Instant::now();
    loop {
        holder.round();
        if let Ok(frames) = replies.recv_timeout(Duration::from_millis(2)) {
            assert_eq!(&frames[0][..], b"r");
            return Message::decode(&frames[1], true).expect("a versioned reply");
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "versioned lookup timed out"
        );
    }
}

#[test]
fn a_holder_pushes_a_prefix_only_its_disk_holds_into_the_readers_landing() {
    let arena = Arena::new(64 * BLOCK_BYTES);
    let mut holder = Replica::start(true, &arena);
    let mut reader = Replica::start(false, &arena);
    let seed: Vec<i32> = (1..=8).collect();
    holder.compute("seed", seed.clone());
    let (tx, replies) = mpsc::channel();
    let client = PeerNet::start("127.0.0.1", 0, tx).unwrap();
    let reply_to = format!("127.0.0.1:{}", client.port());
    let Message::LayoutReply(_, generation, semantic, digest, _) = v1_exchange(
        &mut holder,
        &client,
        &replies,
        Message::LayoutReq(1, reply_to.clone()),
    ) else {
        panic!("holder did not return its layout")
    };
    // Ask a real scheduler to construct the canonical lookup hashes for the seeded prefix.
    let probe = Replica::start(false, &arena);
    probe.submit("probe", (1..=9).collect(), &holder.name);
    let mut scheduler = probe.scheduler.lock();
    let _ = scheduler.next_execution_plan(true, false);
    let lookup = scheduler.take_peer_lookups().pop().expect("reader lookup");
    drop(scheduler);
    let prompt = Prompt {
        tokens: lookup.tokens,
        max_tokens: lookup.max_tokens,
        prefix_hashes: lookup.prefix_hashes,
    };
    let query = |lease| {
        Message::Lookup(
            Key {
                reader: Generation {
                    name: reply_to.clone(),
                    incarnation: 3,
                    epoch: 1,
                },
                holder: generation.clone(),
                lease,
            },
            semantic,
            [2; 32],
            digest,
            prompt.clone(),
        )
    };
    assert!(
        matches!(v1_exchange(&mut holder,&client,&replies,query(1)), Message::Offered(_,Some((offer,_,_))) if offer.end == 8)
    );
    // Each 23-token prompt evicts what came before it from the holder's
    // Device into its Host, which drops its oldest entries, already on Disk.
    for (index, base) in [100, 200, 300].into_iter().enumerate() {
        holder.compute(&format!("pressure-{index}"), (base..base + 23).collect());
        holder.settle();
    }

    // The identical lookup now has only Disk pages. A new lease must miss before PIN or DMA.
    assert!(matches!(
        v1_exchange(&mut holder, &client, &replies, query(2)),
        Message::Offered(_, None)
    ));
    let stats = holder.scheduler.lock().peer_stats();
    assert_eq!((stats.leases, stats.pinned_blocks), (0, 0));

    let mut hit = seed;
    hit.push(9);
    reader.submit("r", hit, &holder.name);
    let start = Instant::now();
    let mut landings = Vec::new();
    let prefix = loop {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the reader never prefilled"
        );
        holder.round();
        let (rows, fetches) = reader.round();
        landings.extend(fetches);
        if let Some((_, prefix)) = rows.into_iter().find(|(rid, _)| rid == "r") {
            break prefix;
        }
        std::thread::sleep(Duration::from_millis(2));
    };

    assert_eq!(landings.len(), 1, "one landing");
    let landing = &landings[0];
    assert_eq!((landing.begin_tokens, landing.end_tokens), (0, 8));
    assert_eq!(prefix, 8, "the pushed prefix resumes the prompt");
    // Each destination holds a page the holder cleaned to Disk, byte for byte.
    let mut landed: Vec<Vec<u8>> = landing
        .destinations
        .iter()
        .map(|destination| {
            arena.read_bytes(
                READER_BASE + destination.block_id as u64 * BLOCK_BYTES as u64,
                BLOCK_BYTES,
            )
        })
        .collect();
    landed.dedup();
    assert_eq!(landed.len(), 2, "two distinct pages");
    for page in &landed {
        assert!(
            holder.on_disk.values().any(|cleaned| cleaned == page),
            "a landed page is not one the holder cleaned"
        );
    }
    holder.round();
    let stats = holder.scheduler.lock().peer_stats();
    assert_eq!((stats.leases, stats.pinned_blocks), (0, 0));
}
